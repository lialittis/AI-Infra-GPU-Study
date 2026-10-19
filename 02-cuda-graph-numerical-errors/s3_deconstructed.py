"""Page section "Deconstructed Input Tensors": a graph keeps raw addresses, not references. If a CUDA input tensor
or the host tensor behind a captured H2D copy is freed, replay reads whatever now lives at that address."""

import torch

from harness import Outcome, case, log, page_loop


@case("freed-cuda-input/bug", "bug")
def _():
    def capture_graph():
        static_input = torch.ones(4, device="cuda")  # ❌ local: freed when the function returns
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            output = static_input * 2
        return g, output, static_input.data_ptr()

    g, output, freed_ptr = capture_graph()
    other_tensor = torch.randn(4, device="cuda")  # the caching allocator hands out the freed block again
    log(f"freed static_input was at {freed_ptr:#x}, other_tensor is at {other_tensor.data_ptr():#x}")
    g.replay()
    log(f"{output.tolist()=}")
    return Outcome(output.tolist(), [2.0] * 4)


@case("freed-cuda-input/fix", "fix")
def _():
    def capture_graph():
        static_input = torch.ones(4, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            output = static_input * 2
        return g, static_input, output  # ✅ return static_input to keep it alive

    g, static_input, output = capture_graph()
    other_tensor = torch.randn(4, device="cuda")
    log(f"static_input at {static_input.data_ptr():#x}, other_tensor at {other_tensor.data_ptr():#x}")
    static_input.fill_(3)
    g.replay()
    log(f"{output.tolist()=}")
    return Outcome(output.tolist(), [6.0] * 4, page=[6.0] * 4)


# The page's CPU-tensor example also passes a Python int, so once captured the graph should keep producing the
# capture-time value 3 (the dynamic-scalar issue, which the page sets aside). Correct replay therefore gives 3.
EXPECT_CPU = [0, 1, 2, 3, 3, 3]


@case("freed-cpu-input/bug", "bug")
def _():
    def func(i):
        t_cpu = torch.tensor(i)  # CPU tensor
        t_gpu = t_cpu.to(device="cuda", non_blocking=True)  # H2D copy from t_cpu's host address is captured
        return t_gpu  # ❌ t_cpu loses its last reference here

    return Outcome(page_loop(func), EXPECT_CPU)


@case("freed-cpu-input/fix-keep-alive", "fix")
def _():
    cpu_tensors = []

    def func(i):
        t_cpu = torch.tensor(i)
        cpu_tensors.append(t_cpu)  # ✅ keep the reference alive for the graph's lifetime
        return t_cpu.to(device="cuda", non_blocking=True)

    return Outcome(page_loop(func), EXPECT_CPU)


@case("freed-cpu-input/fix-gpu-tensor", "fix")
def _():
    def func(i):
        return torch.tensor(i, device="cuda")  # the page's "create directly on the GPU"

    return Outcome(page_loop(func), EXPECT_CPU)


# Recent PyTorch refuses pageable CPU<->CUDA copies during capture, so the three cases above fail loudly at capture
# (torch.tensor(i, device="cuda") is itself a pageable H2D copy). The silent bug the page describes still exists
# with pinned memory, which is what a real H2D input pipeline uses:


def pinned_loop(func):
    """page_loop, but before every replay something else allocates a pinned tensor of the same size, fills it and
    drops it, the host-side analogue of other_tensor in freed-cuda-input/bug."""
    def clobber(i):
        if i > 3:
            t = torch.full((), 1000 + i, dtype=torch.int64).pin_memory()
            log(f"new pinned tensor at {t.data_ptr():#x}")

    return page_loop(func, before=clobber)


@case("freed-cpu-input/bug-pinned", "bug")
def _():
    def func(i):
        t_cpu = torch.tensor(i).pin_memory()
        if torch.cuda.is_current_stream_capturing():
            log(f"captured H2D copy reads host address {t_cpu.data_ptr():#x}")
        return t_cpu.to(device="cuda", non_blocking=True)  # ❌ t_cpu freed after return

    # In eager mode the host caching allocator hands a freed pinned block out again once its copy has completed;
    # a block whose copy was captured is withheld, so this stays correct on recent PyTorch.
    return Outcome(pinned_loop(func), EXPECT_CPU)


@case("freed-cpu-input/fix-pinned-keep-alive", "fix")
def _():
    cpu_tensors = []

    def func(i):
        t_cpu = torch.tensor(i).pin_memory()
        cpu_tensors.append(t_cpu)  # ✅
        return t_cpu.to(device="cuda", non_blocking=True)

    return Outcome(pinned_loop(func), EXPECT_CPU)


@case("freed-cpu-input/fix-full", "fix")
def _():
    def func(i):
        return torch.full((), i, device="cuda", dtype=torch.int64)  # ✅ a fill kernel, no host memory involved

    return Outcome(page_loop(func), EXPECT_CPU)
