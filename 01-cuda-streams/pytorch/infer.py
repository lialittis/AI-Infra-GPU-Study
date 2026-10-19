"""Step B: greedy generation with a small decoder in four styles, annotated with NVTX ranges.

Modes:
  eager    textbook loop: every decode step ends with `.cpu()` on the sampled token (host sync)
  async        sampled tokens are copied D2H on a side stream into pinned memory; the host reads step
               i's token while the GPU already runs step i+1 (the "async output / overlap scheduling"
               idea). But the position is still built per step with torch.tensor(..., device="cuda"):
               a pageable H2D copy that blocks the host, so the CPU can never run ahead
  async-fixed  same, with the position kept on the GPU and incremented in place (no host sync at all)
  graph-sync   one decode step captured into a CUDA graph by hand, replayed per token, synchronous copies
  graph        same graph, with async copies
  compile      torch.compile(mode="reduce-overhead"), i.e. Inductor kernels + automatic CUDA graphs
               (+ async copies, device-side position)

--host-work-ms simulates per-token CPU work of a real server (scheduling, detokenization, stop checks)
done after each token reaches the host; async copies exist to overlap exactly this with GPU work.

Run under nsys and analyze with nsys_streams.py, or standalone for timings.
"""

import argparse
import time

import torch
import torch.cuda.nvtx as nvtx

from model import Config, Decoder


def sample(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1, keepdim=True)  # [B, 1]


def prefill(model, prompt, kc, vc):
    pos = torch.arange(prompt.shape[1], device=prompt.device)
    return sample(model(prompt, pos, kc, vc))


def host_work(seconds: float):
    """Busy-wait to stand in for the CPU work a server does per token."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def gen_sync(model, prompt, kc, vc, steps, step_fn=None, work=0.0):
    out = []
    with nvtx.range("prefill"):
        tok = prefill(model, prompt, kc, vc)
        out.append(tok.cpu())
    for i in range(steps):
        with nvtx.range("decode"):
            if step_fn is None:
                pos = torch.tensor([prompt.shape[1] + i], device=prompt.device)
                tok = sample(model(tok, pos, kc, vc))
            else:
                tok = step_fn(tok, prompt.shape[1] + i)
            out.append(tok.cpu())  # blocks the host until the GPU has finished this step
            host_work(work)
    return out


class AsyncOutput:
    """Copy each step's token to pinned host memory on a dedicated stream; read it one step later."""

    def __init__(self, batch: int, depth: int = 2):
        self.stream = torch.cuda.Stream()
        self.bufs = [torch.empty(batch, 1, dtype=torch.long, pin_memory=True) for _ in range(depth)]
        self.events = [torch.cuda.Event() for _ in range(depth)]
        self.n = 0

    def push(self, tok: torch.Tensor):
        slot = self.n % len(self.bufs)
        self.stream.wait_stream(torch.cuda.current_stream())  # copy only after `tok` is computed
        with torch.cuda.stream(self.stream):
            self.bufs[slot].copy_(tok, non_blocking=True)
            self.events[slot].record()
        tok.record_stream(self.stream)
        self.n += 1

    def pop(self, k: int) -> torch.Tensor:
        """Token of step k; waits only for that copy, not for the compute stream."""
        slot = k % len(self.bufs)
        self.events[slot].synchronize()
        return self.bufs[slot].clone()


def gen_async(model, prompt, kc, vc, steps, step_fn=None, device_pos=False, work=0.0):
    io = AsyncOutput(prompt.shape[0])
    out = []
    with nvtx.range("prefill"):
        tok = prefill(model, prompt, kc, vc)
        io.push(tok)
        dpos = torch.full((1,), prompt.shape[1], device=prompt.device)
    for i in range(steps):
        with nvtx.range("decode"):
            if step_fn is None:
                if device_pos:
                    pos = dpos.clone()  # the model must not see the in-place increment below
                    dpos += 1
                else:
                    pos = torch.tensor([prompt.shape[1] + i], device=prompt.device)  # pageable H2D: host sync
                tok = sample(model(tok, pos, kc, vc))
            else:
                tok = step_fn(tok, prompt.shape[1] + i)
            io.push(tok)
            out.append(io.pop(i))  # previous step's token: the GPU is already working on this step
            host_work(work)
    out.append(io.pop(steps))
    return out


def make_graph_step(model, batch, kc, vc, device):
    """Capture one decode step. Inputs/outputs live in static buffers that replay reads/writes."""
    s_tok = torch.zeros(batch, 1, dtype=torch.long, device=device)
    s_pos = torch.zeros(1, dtype=torch.long, device=device)
    # Warm up on a side stream (required before capture so lazy init / workspace allocation is done).
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            sample(model(s_tok, s_pos, kc, vc))
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        s_out = sample(model(s_tok, s_pos, kc, vc))

    def step(tok, position):
        s_tok.copy_(tok)
        s_pos.fill_(position)
        graph.replay()
        return s_out.clone()  # s_out is overwritten by the next replay

    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eager", "async", "async-fixed", "graph-sync", "graph", "compile"], default="eager")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=2, help="untimed generations before the measured one")
    ap.add_argument("--host-work-ms", type=float, default=0.0, help="simulated CPU work per generated token")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev, dtype = torch.device("cuda"), torch.bfloat16
    cfg = Config(layers=args.layers, max_len=args.prompt + args.steps + 1)
    model = Decoder(cfg).to(dev, dtype).eval()
    prompt = torch.randint(0, cfg.vocab, (args.batch, args.prompt), device=dev)
    kc, vc = model.new_cache(args.batch, dev, dtype)
    nparams = sum(p.numel() for p in model.parameters())

    run_model = model
    if args.mode == "compile":
        # Without this, Inductor sees the KV cache as a mutated *input* with no fixed address and
        # silently skips CUDA graphs ("skipping cudagraphs due to mutated inputs").
        torch._dynamo.mark_static_address(kc)
        torch._dynamo.mark_static_address(vc)
        run_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    work = args.host_work_ms / 1e3

    def generate():
        if args.mode == "eager":
            return gen_sync(run_model, prompt, kc, vc, args.steps, work=work)
        if args.mode == "graph-sync":
            return gen_sync(model, prompt, kc, vc, args.steps, step_fn=graph_step, work=work)
        if args.mode == "graph":
            return gen_async(model, prompt, kc, vc, args.steps, step_fn=graph_step, work=work)
        return gen_async(run_model, prompt, kc, vc, args.steps, device_pos=args.mode != "async", work=work)

    with torch.inference_mode():
        graph_step = make_graph_step(model, args.batch, kc, vc, dev) if args.mode.startswith("graph") else None
        for _ in range(args.warmup):
            generate()
        torch.cuda.synchronize()
        # The analyzer only looks at GPU work launched inside this range (warmup/compile excluded).
        with nvtx.range("measured"):
            t0 = time.perf_counter()
            toks = generate()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0

    seq = torch.cat(toks, dim=1)
    print(f"mode={args.mode} params={nparams / 1e9:.2f}B batch={args.batch} prompt={args.prompt} "
          f"host_work={args.host_work_ms}ms "
          f"steps={args.steps}: {dt * 1e3:.1f} ms total, {dt * 1e3 / (args.steps + 1):.2f} ms/token "
          f"(first tokens {seq[0, :6].tolist()})")


if __name__ == "__main__":
    main()
