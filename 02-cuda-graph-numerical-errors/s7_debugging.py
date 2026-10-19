"""Page section "Debugging Numerical Errors": compare per-layer outputs of eager mode and graph replay to find the
first layer where they diverge. The page's snippet leaves the model out; here it is four Linear+tanh blocks, and the
bug variant plants the page's "different precision during capture" anti-pattern in block 2."""

import os

import torch

from harness import Outcome, case, log


class Block(torch.nn.Module):
    def __init__(self, buggy: bool):
        super().__init__()
        self.lin = torch.nn.Linear(256, 256)
        self.buggy = buggy

    def forward(self, x):
        x = torch.tanh(self.lin(x))
        if self.buggy and torch.cuda.is_current_stream_capturing():
            x = x.half().float()  # ❌ different precision during capture
        return x


def first_divergent_layer(buggy_layer):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required by deterministic cuBLAS
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    layers = torch.nn.ModuleList(Block(i == buggy_layer) for i in range(4)).cuda()
    inp = torch.randn(32, 256, device="cuda")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    eager_intermediates = []
    x = inp.clone()
    for layer in layers:
        x = layer(x)
        eager_intermediates.append(x.clone())

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    graph_intermediates = []
    g = torch.cuda.CUDAGraph()
    static_input = inp.clone()
    with torch.cuda.graph(g):
        x = static_input
        for layer in layers:
            x = layer(x)
            graph_intermediates.append(x.clone())

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    static_input.copy_(inp)
    g.replay()
    torch.cuda.synchronize()

    threshold = 1e-5
    for i, (eager, graph) in enumerate(zip(eager_intermediates, graph_intermediates)):
        diff = (eager - graph).abs().max().item()
        log(f"Layer {i}: max diff = {diff:.2e}")
        if diff > threshold:
            log(f"Issue first appears at layer {i}")
            return i
    log("no divergence")
    return None


@case("layer-diff/bug", "bug")
def _():
    return Outcome(first_divergent_layer(buggy_layer=2), None)


@case("layer-diff/clean", "fix")
def _():
    return Outcome(first_divergent_layer(buggy_layer=None), None)
