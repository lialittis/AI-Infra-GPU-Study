"""Page section "Dynamic Behavior Failures": control flow, capture-aware code, dynamic input tensors (incl. the
autocast cache), dynamic scalars (incl. the hidden ones: learning rate, step counter, GradScaler, RNG) and dynamic
shapes. A graph replays the GPU work recorded at capture: fixed branches, fixed addresses, fixed kernel arguments,
fixed launch shapes."""

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from harness import Outcome, case, log, page_loop

BUILD = Path(__file__).resolve().parent / "build"

# ---------------------------------------------------------------- dynamic control flow

EXPECT_FLOW = [2, 0, 2, 0, 2, 0]  # eager: even idx -> (1 + 1) * 1, odd idx -> (1 - 1) * 1


def flow_func(idx, x, y):
    if idx % 2 == 0:  # ❌ Python branch on a runtime value
        z = x + y
    else:
        z = x - y
    return z * x


def ones_int():
    return torch.ones(1, device="cuda", dtype=torch.int32)


@case("control-flow/bug", "bug")
def _():
    x, y = ones_int(), ones_int()
    # idx=3 is odd: the subtraction branch is captured and replayed for every later idx
    return Outcome(page_loop(lambda i: flow_func(i, x, y)), EXPECT_FLOW, page=[2, 0, 2, 0, 0, 0])


@case("control-flow/fix-where", "fix")
def _():
    def func(idx_tensor, x, y):
        condition = idx_tensor % 2 == 0  # GPU tensor, evaluated on every replay
        return torch.where(condition, x + y, x - y) * x  # both branches run, the GPU selects

    x, y = ones_int(), ones_int()
    idx_tensor = torch.zeros(1, device="cuda", dtype=torch.int32)
    out = page_loop(lambda i: func(idx_tensor, x, y), before=lambda i: idx_tensor.fill_(i))
    return Outcome(out, EXPECT_FLOW, page=EXPECT_FLOW)


CONDITIONAL_OP_CUDA = r"""
#include <c10/cuda/CUDAStream.h>

__global__ void conditional_op_kernel(int* output, const int* x, const int* y, const int* idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (idx[0] % 2 == 0) {
            output[i] = (x[i] + y[i]) * x[i];  // Addition path
        } else {
            output[i] = (x[i] - y[i]) * x[i];  // Subtraction path
        }
    }
}

torch::Tensor conditional_op(torch::Tensor x, torch::Tensor y, torch::Tensor idx) {
    auto output = torch::empty_like(x);
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    // Use PyTorch's current stream for CUDA graph capture compatibility
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    conditional_op_kernel<<<blocks, threads, 0, stream>>>(
        output.data_ptr<int>(), x.data_ptr<int>(), y.data_ptr<int>(), idx.data_ptr<int>(), n);
    return output;
}
"""


@case("control-flow/fix-kernel", "fix")
def _():
    from torch.utils.cpp_extension import load_inline

    (BUILD / "conditional_op").mkdir(parents=True, exist_ok=True)
    mod = load_inline(name="conditional_op", cpp_sources="torch::Tensor conditional_op(torch::Tensor x, "
                      "torch::Tensor y, torch::Tensor idx);", cuda_sources=CONDITIONAL_OP_CUDA,
                      functions=["conditional_op"], build_directory=str(BUILD / "conditional_op"))
    x, y = ones_int(), ones_int()
    idx_tensor = torch.zeros(1, device="cuda", dtype=torch.int32)
    out = page_loop(lambda i: mod.conditional_op(x, y, idx_tensor), before=lambda i: idx_tensor.fill_(i))
    return Outcome(out, EXPECT_FLOW, page=EXPECT_FLOW)


@case("control-flow/fix-multi-graph", "fix")
def _():
    x, y = ones_int(), ones_int()
    g_add, g_sub = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_add):
        z_add = (x + y) * x
    with torch.cuda.graph(g_sub):
        z_sub = (x - y) * x
    out = []
    for idx in range(6):  # the branch is chosen on the CPU, outside any graph
        if idx % 2 == 0:
            g_add.replay()
            z = z_add
        else:
            g_sub.replay()
            z = z_sub
        out.append(z.item())
        log(f"{idx=}, {z.item()=}")
    return Outcome(out, EXPECT_FLOW)


@case("control-flow/fix-partial", "fix")
def _():
    x, y = ones_int(), ones_int()
    z_static = torch.empty_like(x)  # boundary between the eager (dynamic) part and the graphed (static) part
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_static = z_static * x
    out = []
    for idx in range(6):
        z_static.copy_(x + y if idx % 2 == 0 else x - y)  # control flow stays eager
        graph.replay()
        out.append(out_static.item())
        log(f"{idx=}, {out_static.item()=}")
    return Outcome(out, EXPECT_FLOW)


# ---------------------------------------------------------------- capture-aware code

def capture_aware_run(func):
    x = torch.zeros(1, device="cuda", dtype=torch.int32)
    y = torch.zeros(1, device="cuda", dtype=torch.int32)
    return page_loop(lambda i: func(x, y), before=lambda i: y.fill_(i))


@case("capture-aware/bug", "bug")
def _():
    def func(x, y):
        z = x + y
        if torch.cuda.is_current_stream_capturing():
            z *= -1  # ❌ Different logic during capture!
        return z

    return Outcome(capture_aware_run(func), [0, 1, 2, 3, 4, 5], page=[0, 1, 2, -3, -4, -5])


@case("capture-aware/fix", "fix")
def _():
    def func(x, y):
        z = x + y
        if torch.cuda.is_current_stream_capturing():
            z = z.contiguous()  # ✅ changes the implementation, not the computation
        return z

    return Outcome(capture_aware_run(func), [0, 1, 2, 3, 4, 5])


@case("capture-aware/fix-sleep", "fix")
def _():
    def func(x, y):
        z = x + y
        if torch.cuda.is_current_stream_capturing():
            torch.cuda._sleep(1000)  # ✅ graph-safe sleep alternative
        else:
            time.sleep(0.001)
        return z

    return Outcome(capture_aware_run(func), [0, 1, 2, 3, 4, 5])


# ---------------------------------------------------------------- dynamic input tensors

@case("dynamic-input/bug", "bug")
def _():
    graph = torch.cuda.CUDAGraph()
    base_lr = torch.zeros(1, device="cuda", dtype=torch.int32)
    outs = []
    for i in range(2):  # warmup
        lr = base_lr + i  # ❌ Creates NEW tensor each iteration
        out = 1 * lr
        outs.append(out.item())
        log(f"{i=}, {out.item()=}")
    with torch.cuda.graph(graph):  # captures the address of lr, which currently holds base_lr + 1
        out = 1 * lr
    for i in range(2, 6):
        lr = base_lr + i  # new tensor at a new address: the graph never sees it
        graph.replay()
        outs.append(out.item())
        log(f"{i=}, {out.item()=}")
    return Outcome(outs, [0, 1, 2, 3, 4, 5], page=[0, 1, 1, 1, 1, 1])


@case("dynamic-input/fix", "fix")
def _():
    graph = torch.cuda.CUDAGraph()
    base_lr = torch.zeros(1, device="cuda", dtype=torch.int32)
    lr = torch.zeros(1, device="cuda", dtype=torch.int32)  # allocate static tensor ONCE
    outs = []
    for i in range(2):
        lr.copy_(base_lr + i)  # ✅ update values, same address
        out = 1 * lr
        outs.append(out.item())
        log(f"{i=}, {out.item()=}")
    with torch.cuda.graph(graph):
        out = 1 * lr
    for i in range(2, 6):
        lr.copy_(base_lr + i)
        graph.replay()
        outs.append(out.item())
        log(f"{i=}, {out.item()=}")
    return Outcome(outs, [0, 1, 2, 3, 4, 5])


def autocast_run(variant):
    """Capture y = x @ w under autocast, then change w. Returns y after a replay with w = 1 and with w = 2."""
    w = torch.ones(1, 1, device="cuda", dtype=torch.float32, requires_grad=True)
    x = torch.ones(1, 1, device="cuda", dtype=torch.float32)
    g = torch.cuda.CUDAGraph()
    if variant == "bug":
        with torch.amp.autocast("cuda"):
            _ = x @ w  # w cast to FP16 and cached
            with torch.cuda.graph(g):
                y = x @ w  # the graph reads the cached FP16 copy of w
        # exiting autocast clears the cache; the cached copy's memory may be recycled
    elif variant == "no-cache":
        with torch.amp.autocast("cuda", cache_enabled=False):
            _ = x @ w
            with torch.cuda.graph(g):
                y = x @ w  # the cast is a captured kernel
    else:  # "inside"
        # The page's snippet has no warmup; then the process's first matmul (cuBLAS handle creation) happens
        # inside the capture and the capture fails with CUBLAS_STATUS_NOT_INITIALIZED.
        with torch.amp.autocast("cuda"):
            _ = x @ w
        with torch.cuda.graph(g):
            with torch.amp.autocast("cuda"):
                y = x @ w  # the cast is captured in the graph
    out = []
    g.replay()
    log(f"{w.item()=}, {y.item()=}")
    out.append(y.item())
    with torch.no_grad():
        w.fill_(2)
    g.replay()
    log(f"{w.item()=}, {y.item()=}")
    out.append(y.item())
    return out


@case("autocast-cache/bug", "bug")
def _():
    return Outcome(autocast_run("bug"), [1.0, 2.0], page=[1.0, 1.0])


@case("autocast-cache/fix-no-cache", "fix")
def _():
    return Outcome(autocast_run("no-cache"), [1.0, 2.0])


@case("autocast-cache/fix-inside", "fix")
def _():
    return Outcome(autocast_run("inside"), [1.0, 2.0])


# ---------------------------------------------------------------- dynamic scalars

EXPECT_POW = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]  # 2 ** i


@case("dynamic-scalar/bug", "bug")
def _():
    x = torch.tensor(2, device="cuda", dtype=torch.int32)
    # exponent is a Python int: a kernel argument, frozen at its capture-time value 3
    return Outcome(page_loop(lambda i: torch.float_power(x, i)), EXPECT_POW, page=[1.0, 2.0, 4.0, 8.0, 8.0, 8.0])


@case("dynamic-scalar/fix", "fix")
def _():
    x = torch.tensor(2, device="cuda", dtype=torch.int32)
    exponent_tensor = torch.zeros(1, device="cuda", dtype=torch.int32)  # the kernel reads it from memory
    out = page_loop(lambda i: torch.float_power(x, exponent_tensor), before=lambda i: exponent_tensor.fill_(i))
    return Outcome(out, EXPECT_POW)


LRS = [1.0, 0.5, 0.25, 0.125]  # a learning-rate schedule
EXPECT_SGD = [-1.0, -1.5, -1.75, -1.875]  # w after each step of w -= lr * grad, grad = 1


@case("hidden-scalar/lr-bug", "bug")
def _():
    w, grad = torch.zeros(1, device="cuda"), torch.ones(1, device="cuda")
    lr = LRS[0]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w.sub_(grad * lr)  # ❌ lr is a Python float
    out = []
    for step in range(len(LRS)):
        lr = LRS[step]  # different each step, but the graph keeps LRS[0]
        graph.replay()
        out.append(w.item())
        log(f"{step=}, {lr=}, {w.item()=}")
    return Outcome(out, EXPECT_SGD)


@case("hidden-scalar/lr-fix", "fix")
def _():
    w, grad = torch.zeros(1, device="cuda"), torch.ones(1, device="cuda")
    lr_tensor = torch.zeros(1, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w.sub_(grad * lr_tensor)  # ✅
    out = []
    for step in range(len(LRS)):
        lr_tensor.fill_(LRS[step])
        graph.replay()
        out.append(w.item())
        log(f"{step=}, lr={LRS[step]}, {w.item()=}")
    return Outcome(out, EXPECT_SGD)


@case("hidden-scalar/step-counter-bug", "bug")
def _():
    inp = torch.ones(1, device="cuda")
    model = lambda x, step: x * step  # noqa: E731  a model that uses the global step
    return Outcome(page_loop(lambda global_step: model(inp, step=global_step)), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


@case("hidden-scalar/step-counter-fix", "fix")
def _():
    inp = torch.ones(1, device="cuda")
    model = lambda x, step: x * step  # noqa: E731
    step_tensor = torch.zeros(1, device="cuda", dtype=torch.int64)
    out = page_loop(lambda i: model(inp, step=step_tensor), before=lambda global_step: step_tensor.fill_(global_step))
    return Outcome(out, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


def scaler_setup():
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 1, device="cuda")
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    # growth_interval=1 doubles the loss scale after every step, so a frozen scale would show up at once
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 4, growth_interval=1)
    gen = torch.Generator(device="cuda").manual_seed(1)
    data = [(torch.randn(16, 8, device="cuda", generator=gen), torch.randn(16, 1, device="cuda", generator=gen))
            for _ in range(5)]
    return model, opt, scaler, data


def mse(model, x, t):
    return ((model(x) - t) ** 2).mean()


def scaler_eager_reference():
    model, opt, scaler, data = scaler_setup()
    for x, t in data:
        opt.zero_grad()
        scaler.scale(mse(model, x, t)).backward()
        scaler.step(opt)
        scaler.update()
    return model.weight.flatten().tolist() + model.bias.tolist() + [scaler.get_scale()]


def scaler_graphed(step_inside: bool):
    """Whole fwd+bwd capture with a GradScaler; step/update are inside the graph only if step_inside."""
    model, opt, scaler, data = scaler_setup()
    static_x, static_t = data[0][0].clone(), data[0][1].clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):  # warmup on a side stream: grads only, no optimizer step
        for _ in range(3):
            scaler.scale(mse(model, static_x, static_t)).backward()
    torch.cuda.current_stream().wait_stream(s)
    opt.zero_grad(set_to_none=True)  # backward in the capture allocates static .grad tensors
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        scaler.scale(mse(model, static_x, static_t)).backward()
        if step_inside:  # ❌
            scaler.step(opt)
            scaler.update()
    for x, t in data:
        static_x.copy_(x)
        static_t.copy_(t)
        g.replay()
        if not step_inside:  # ✅
            scaler.step(opt)
            scaler.update()
        log(f"scale={scaler.get_scale()}, weight[0]={model.weight[0, 0].item():.6f}")
    return model.weight.flatten().tolist() + model.bias.tolist() + [scaler.get_scale()]


@case("hidden-scalar/grad-scaler-bug", "bug")
def _():
    return Outcome(scaler_graphed(step_inside=True), scaler_eager_reference())


@case("hidden-scalar/grad-scaler-fix", "fix")
def _():
    return Outcome(scaler_graphed(step_inside=False), scaler_eager_reference())


def rng_distinct_draws(draw, register=None, replays=5):
    """Capture one random draw, replay it several times, count how many different results come out."""
    draw()  # warmup
    g = torch.cuda.CUDAGraph()
    if register is not None:
        g.register_generator_state(register)
    with torch.cuda.graph(g):
        out = draw()
    seen = set()
    for r in range(replays):
        g.replay()
        seen.add(tuple(out.tolist()))
        log(f"replay {r}: first values {[round(v, 4) for v in out[:4].tolist()]}")
    return len(seen)


@case("rng/default-auto", "fix")
def _():
    inp = torch.ones(64, device="cuda")
    return Outcome(rng_distinct_draws(lambda: F.dropout(inp, p=0.5, training=True)), 5)


@case("rng/default-registered", "fix")
def _():
    inp = torch.ones(64, device="cuda")
    draws = rng_distinct_draws(lambda: F.dropout(inp, p=0.5, training=True), register=torch.cuda.default_generators[0])
    return Outcome(draws, 5)


@case("rng/custom-unregistered", "bug")
def _():
    gen = torch.Generator(device="cuda").manual_seed(0)
    return Outcome(rng_distinct_draws(lambda: torch.rand(64, device="cuda", generator=gen)), 5)


@case("rng/custom-registered", "fix")
def _():
    gen = torch.Generator(device="cuda").manual_seed(0)
    return Outcome(rng_distinct_draws(lambda: torch.rand(64, device="cuda", generator=gen), register=gen), 5)


# ---------------------------------------------------------------- dynamic shapes

@case("dynamic-shape/bug", "bug")
def _():
    graph = torch.cuda.CUDAGraph()
    x = torch.ones(4, device="cuda", dtype=torch.int32)  # capture with shape (4,)
    with torch.cuda.graph(graph):
        y = x * 2
    x.fill_(3)
    graph.replay()
    log(f"same shape: {y.tolist()=}")
    x_large = torch.ones(8, device="cuda", dtype=torch.int32) * 5
    x.copy_(x_large[:4])  # can only update the captured tensor's 4 elements
    graph.replay()
    log(f"larger input: {y.tolist()=}")
    return Outcome(y.tolist(), [10] * 8, page=[10] * 4)


PAD_DATA = [3, 8, 5]  # sizes of the variable-length inputs; input k of size n is arange(n) + k


def padded_inputs():
    return [torch.arange(n, device="cuda", dtype=torch.float32) + k for k, n in enumerate(PAD_DATA)]


def expected_padded():
    return [((torch.arange(n, dtype=torch.float32) + k) * 2).tolist() for k, n in enumerate(PAD_DATA)]


@case("dynamic-shape/fix-pad", "fix")
def _():
    model = lambda t: t * 2  # noqa: E731
    max_size = 1024
    static_input = torch.zeros(max_size, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_output = model(static_input)
    results = []
    for data in padded_inputs():
        actual_size = data.shape[0]
        static_input[:actual_size].copy_(data)
        static_input[actual_size:].zero_()  # zero padding
        g.replay()
        results.append(static_output[:actual_size].clone().tolist())  # trim to actual size
        log(f"size {actual_size}: {results[-1]}")
    return Outcome(results, expected_padded())


@case("dynamic-shape/fix-bucket", "fix")
def _():
    model = lambda t: t * 2  # noqa: E731
    buckets = [4, 8, 16]
    graphs = {}
    for size in buckets:
        static_input = torch.zeros(size, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_output = model(static_input)
        graphs[size] = {"graph": g, "input": static_input, "output": static_output}

    def forward(data):
        size = data.shape[0]
        bucket = min([s for s in buckets if s >= size], default=max(buckets))
        graphs[bucket]["input"][:size].copy_(data)
        graphs[bucket]["graph"].replay()
        log(f"size {size} -> bucket {bucket}")
        return graphs[bucket]["output"][:size].clone()

    return Outcome([forward(d).tolist() for d in padded_inputs()], expected_padded())
