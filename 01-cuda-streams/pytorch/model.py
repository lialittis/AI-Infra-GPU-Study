"""Minimal Llama-style decoder with a static KV cache, random weights, pure PyTorch.

The cache has a fixed shape [layers, batch, kv_heads, max_len, head_dim] and decode positions are
GPU tensors, so a decode step has static shapes and no host sync, which makes it CUDA-graph capturable.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Config:
    vocab: int = 32000
    dim: int = 2048
    layers: int = 16
    heads: int = 16
    kv_heads: int = 4
    ffn: int = 5632
    max_len: int = 1024

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads


def rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Rotary embedding. x: [B, H, T, D], pos: [T] (int64, on device)."""
    d = x.shape[-1]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = pos.float()[:, None] * inv_freq[None, :]  # [T, D/2]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.attn_norm = nn.RMSNorm(c.dim)
        self.qkv = nn.Linear(c.dim, (c.heads + 2 * c.kv_heads) * c.head_dim, bias=False)
        self.o = nn.Linear(c.dim, c.dim, bias=False)
        self.mlp_norm = nn.RMSNorm(c.dim)
        self.gate_up = nn.Linear(c.dim, 2 * c.ffn, bias=False)
        self.down = nn.Linear(c.ffn, c.dim, bias=False)

    def forward(self, x, pos, k_cache, v_cache, mask):
        c = self.c
        B, T, _ = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).split(
            [c.heads * c.head_dim, c.kv_heads * c.head_dim, c.kv_heads * c.head_dim], dim=-1)
        q = rope(q.view(B, T, c.heads, c.head_dim).transpose(1, 2), pos)
        k = rope(k.view(B, T, c.kv_heads, c.head_dim).transpose(1, 2), pos)
        v = v.view(B, T, c.kv_heads, c.head_dim).transpose(1, 2)
        k_cache.index_copy_(2, pos, k)
        v_cache.index_copy_(2, pos, v)
        # Attend over the whole static cache; `mask` hides future / unwritten positions.
        a = F.scaled_dot_product_attention(q, k_cache, v_cache, attn_mask=mask, enable_gqa=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, c.dim))
        gate, up = self.gate_up(self.mlp_norm(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)


class Decoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab, c.dim)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.norm = nn.RMSNorm(c.dim)
        self.lm_head = nn.Linear(c.dim, c.vocab, bias=False)

    def new_cache(self, batch: int, device, dtype):
        c = self.c
        shape = (c.layers, batch, c.kv_heads, c.max_len, c.head_dim)
        return torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)

    def forward(self, tokens, pos, k_cache, v_cache):
        """tokens: [B, T], pos: [T] absolute positions. Returns logits of the last position, [B, vocab]."""
        # mask[t, j] = key j is visible to query t (j <= pos[t]); shape [T, max_len] broadcasts over B, H.
        mask = torch.arange(self.c.max_len, device=tokens.device)[None, :] <= pos[:, None]
        x = self.embed(tokens)
        for i, blk in enumerate(self.blocks):
            x = blk(x, pos, k_cache[i], v_cache[i], mask)
        return self.lm_head(self.norm(x[:, -1]))
