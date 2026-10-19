"""Page section "Missing Operations in Captured Graph": only work issued to the capture stream (or streams forked
from it) is recorded, and graph replay bypasses submodule forward() calls, so their hooks never run."""

import torch

from harness import Outcome, case, log


def library_stream_run(library_call):
    """The page's loop. library_call(y) returns y * 2 issued however the 'library' chooses."""
    x = torch.ones(1, device="cuda", dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()
    out = []
    for i in range(3):  # warmup
        x.fill_(i)
        y = library_call(x + 1)
        out.append(y.item())
        log(f"warmup {i=}, {y.item()=}")
    x.fill_(3)
    with torch.cuda.graph(graph):
        y = x + 1  # captured on the capture stream
        y = library_call(y)
    for i in range(4, 7):
        x.fill_(i)
        graph.replay()
        out.append(y.item())
        log(f"replay {i=}, {y.item()=}")
    return out


EXPECT_LIB = [2, 4, 6, 10, 12, 14]  # (i + 1) * 2


@case("library-stream/bug", "bug")
def _():
    def library_call(x):
        stream = torch.cuda.default_stream()  # ❌ assumes work runs on the default stream
        with torch.cuda.stream(stream):
            return x * 2

    return Outcome(library_stream_run(library_call), EXPECT_LIB, page=[2, 4, 6, 0, 0, 0])


@case("library-stream/fix-current-stream", "fix")
def _():
    def library_call(x):
        current = torch.cuda.current_stream()  # ✅ PyTorch's current stream is the capture stream
        with torch.cuda.stream(current):
            x = x * 2
        return x

    return Outcome(library_stream_run(library_call), EXPECT_LIB)


@case("library-stream/fix-pass-stream", "fix")
def _():
    def library_call_a(x, stream):  # option A: use the provided stream directly
        with torch.cuda.stream(stream):
            return x * 2

    return Outcome(library_stream_run(lambda y: library_call_a(y, torch.cuda.current_stream())), EXPECT_LIB)


@case("library-stream/fix-internal-stream", "fix")
def _():
    def library_call_b(x, stream):  # option B: fork an internal stream from the provided one and join back
        internal_stream = torch.cuda.Stream()
        internal_stream.wait_stream(stream)
        with torch.cuda.stream(internal_stream):
            result = x * 2
        stream.wait_stream(internal_stream)
        return result

    return Outcome(library_stream_run(lambda y: library_call_b(y, torch.cuda.current_stream())), EXPECT_LIB)


def graphed_sequential():
    model = torch.nn.Sequential(
        torch.nn.Embedding(4096, 4096, device="cuda"),
        torch.nn.Linear(4096, 4096, device="cuda"),
        torch.nn.Linear(4096, 4096, device="cuda"),
    ).cuda()
    inp = torch.randint(high=128, size=(256,), device="cuda")
    return torch.cuda.make_graphed_callables(model, (inp,)), inp


@case("module-hooks/bug", "bug")
def _():
    model, inp = graphed_sequential()
    num_registered, num_called = 0, [0]

    def hook(module, input, output):
        num_called[0] += 1

    for module in model.modules():  # Sequential + 3 submodules
        num_registered += 1
        module.register_forward_hook(hook)  # registered AFTER graphing, as required
    model(inp)
    log(f"Registered: {num_registered}, Called: {num_called[0]}")
    return Outcome(num_called[0], num_registered, page=1)  # eager would call all four


@case("module-hooks/workaround-top-level", "fix")
def _():
    model, inp = graphed_sequential()
    seen = []

    def hook(module, input, output):
        for name, _ in module.named_parameters(recurse=True):  # ✅ every parameter in the tree
            seen.append(name)

    model.register_forward_hook(hook)
    model(inp)
    log(f"top-level hook saw {len(seen)} parameters: {seen}")
    return Outcome(len(seen), 5)  # embedding weight + 2 x (linear weight, bias)
