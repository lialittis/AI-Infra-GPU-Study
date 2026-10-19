"""Page section "Memory Pool Sharing Issues": graphs captured into one pool may be handed the same addresses, which
is only safe if they replay sequentially, in capture order."""

import os

import torch

from harness import Outcome, case, log

# ---------------------------------------------------------------- shared pool corruption


def pool_funcs():
    def func1(x1):
        t1 = x1 * 3  # intermediate tensor
        y1 = t1 + 5
        if torch.cuda.is_current_stream_capturing():
            log(f"{x1.data_ptr()=:#x}, {t1.data_ptr()=:#x}, {y1.data_ptr()=:#x}")
        return y1

    def func2(x2):
        y2 = x2 ** 2  # output tensor
        if torch.cuda.is_current_stream_capturing():
            log(f"{x2.data_ptr()=:#x}, {y2.data_ptr()=:#x}")
        return y2

    return func1, func2


def pool_setup(shared: bool):
    func1, func2 = pool_funcs()
    x1 = torch.tensor([1.0], device="cuda")
    x2 = torch.tensor([2.0], device="cuda")
    graph1, graph2 = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    y1, y2 = func1(x1), func2(x2)  # warmup
    log(f"Correct: {y1.item()=}, {y2.item()=}")
    with torch.cuda.graph(graph1):
        y1 = func1(x1)
    with torch.cuda.graph(graph2, pool=graph1.pool() if shared else None):
        y2 = func2(x2)
    # The inputs must outlive this function (see s3_deconstructed): without x1 in the return value, the clone in
    # fix-read-immediately is handed x1's freed block and graph1 then reads 4.0 instead of 1.0.
    return graph1, graph2, y1, y2, (x1, x2)


@case("shared-pool/bug", "bug")
def _():
    graph1, graph2, y1, y2, inputs = pool_setup(shared=True)
    graph2.replay()  # y2 = 4
    graph1.replay()  # ❌ graph1's intermediate t1 lives where y2 lives
    log(f"Actual: {y1.item()=}, {y2.item()=}")
    return Outcome([y1.item(), y2.item()], [8.0, 4.0], page=[8.0, 3.0])


@case("shared-pool/fix-read-immediately", "fix")
def _():
    graph1, graph2, y1, y2, inputs = pool_setup(shared=True)
    graph2.replay()
    result2 = y2.clone()  # ✅ save before another graph replays
    graph1.replay()
    result1 = y1.clone()
    log(f"{result1.item()=}, {result2.item()=}")
    return Outcome([result1.item(), result2.item()], [8.0, 4.0])


@case("shared-pool/fix-separate-pools", "fix")
def _():
    graph1, graph2, y1, y2, inputs = pool_setup(shared=False)  # ✅ each graph gets its own pool
    graph2.replay()
    graph1.replay()
    log(f"{y1.item()=}, {y2.item()=}")
    return Outcome([y1.item(), y2.item()], [8.0, 4.0])


@case("shared-pool/fix-capture-order", "fix")
def _():
    graph1, graph2, y1, y2, inputs = pool_setup(shared=True)
    graph1.replay()  # ✅ same order as capture
    graph2.replay()
    log(f"{y1.item()=}, {y2.item()=}")
    return Outcome([y1.item(), y2.item()], [8.0, 4.0])


# ---------------------------------------------------------------- replay order mismatch

func = lambda x: x ** 2 + 2 * x + 1  # noqa: E731   d/dx = 2x + 2
EXPECT_GRADS = [4.0, 6.0, 8.0]


def leaf_inputs():
    return [torch.tensor([v], device="cuda", requires_grad=True) for v in (1.0, 2.0, 3.0)]


def report_grads(xs):
    grads = [x.grad.item() for x in xs]
    log(f"Actual grad: {grads}, correct grad: {EXPECT_GRADS}")
    return grads


@case("replay-order/bug", "bug")
def _():
    xs = leaf_inputs()
    # one make_graphed_callables call shares one pool across all six graphs, captured in the order
    # f0_fprop, f0_bprop, f1_fprop, f1_bprop, f2_fprop, f2_bprop
    graphed = torch.cuda.make_graphed_callables((func, func, func), tuple((x,) for x in xs))
    y0 = graphed[0](xs[0])  # replay order: f0_fprop, f1_fprop, f0_bprop, f2_fprop, f1_bprop, f2_bprop  ❌
    y1 = graphed[1](xs[1])
    y0.backward()
    y2 = graphed[2](xs[2])
    y1.backward()
    y2.backward()
    return Outcome(report_grads(xs), EXPECT_GRADS, page=[6.0, 3.0, 8.0])


@case("replay-order/fix-capture-order", "fix")
def _():
    xs = leaf_inputs()
    graphed = torch.cuda.make_graphed_callables((func, func, func), tuple((x,) for x in xs))
    for g, x in zip(graphed, xs):  # the page's fix: f0_fprop, f0_bprop, f1_fprop, ... "as captured"
        g(x).backward()
    return Outcome(report_grads(xs), EXPECT_GRADS)


@case("replay-order/fix-documented-order", "fix")
def _():
    xs = leaf_inputs()
    # torch/cuda/graphs.py actually captures fwd 0, fwd 1, fwd 2, then bwd 2, bwd 1, bwd 0; replay that order
    graphed = torch.cuda.make_graphed_callables((func, func, func), tuple((x,) for x in xs))
    ys = [g(x) for g, x in zip(graphed, xs)]
    for y in reversed(ys):
        y.backward()
    return Outcome(report_grads(xs), EXPECT_GRADS)


@case("replay-order/fix-separate-pools", "fix")
def _():
    xs = leaf_inputs()
    graphed = [torch.cuda.make_graphed_callables(func, (x,)) for x in xs]  # ✅ one pool per call
    y0 = graphed[0](xs[0])  # the same interleaved order as the bug
    y1 = graphed[1](xs[1])
    y0.backward()
    y2 = graphed[2](xs[2])
    y1.backward()
    y2.backward()
    return Outcome(report_grads(xs), EXPECT_GRADS)


@case("replay-order/fix-capture-in-replay-order", "fix")
def _():
    xs = leaf_inputs()
    s = torch.cuda.Stream()  # warm up on a side stream, as make_graphed_callables does
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for x in xs:
            torch.autograd.grad(func(x), x, torch.ones_like(x))
    torch.cuda.current_stream().wait_stream(s)

    # ✅ capture the six graphs, sharing one pool, in exactly the order they will be replayed
    order = [("fwd", 0), ("fwd", 1), ("bwd", 0), ("fwd", 2), ("bwd", 1), ("bwd", 2)]
    grad_out = torch.ones(1, device="cuda")
    graphs, ys, grads, pool = [], {}, {}, None
    for kind, i in order:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool):
            if kind == "fwd":
                ys[i] = func(xs[i])
            else:
                (grads[i],) = torch.autograd.grad(ys[i], xs[i], grad_out)
        pool = g.pool()
        graphs.append(g)
    for g in graphs:
        g.replay()
    out = [grads[i].item() for i in range(3)]
    log(f"Actual grad: {out}, correct grad: {EXPECT_GRADS}")
    return Outcome(out, EXPECT_GRADS)


# ---------------------------------------------------------------- parallel replay

# The page uses 1024**3 elements (4 GiB per tensor); kernels must run long enough to overlap.
RACE_ELEMS = int(os.environ.get("RACE_ELEMS", 256 * 1024 ** 2))
RACE_ITERS = 100
EXPECT_RACE = [4.0, 10.0, 18.0]  # f_k(k) = k^2 + 2k + k


def race_setup(shared: bool):
    x = [torch.ones(RACE_ELEMS, device="cuda") * k for k in (1, 2, 3)]
    fs = [lambda x: x ** 2 + 2 * x + 1, lambda x: x ** 2 + 2 * x + 2, lambda x: x ** 2 + 2 * x + 3]
    for _ in range(3):  # warmup
        ys = [f(xi) for f, xi in zip(fs, x)]
    log(f"Correct: {[y[0].item() for y in ys]}")
    del ys
    graphs, ys, pool = [], [], None
    for f, xi in zip(fs, x):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool):
            ys.append(f(xi))
        if shared:
            pool = g.pool()
        graphs.append(g)
    return graphs, ys


def race_report(ys):
    torch.cuda.synchronize()
    log(f"Actual: {[y[0].item() for y in ys]}")
    wrong = [int((y != e).sum().item()) for y, e in zip(ys, EXPECT_RACE)]
    log(f"wrong elements per output (of {RACE_ELEMS}): {wrong}")
    return wrong


def parallel_replay(graphs):
    streams = [torch.cuda.Stream() for _ in graphs]
    current = torch.cuda.current_stream()
    for s in streams:
        s.wait_stream(current)
    for _ in range(RACE_ITERS):
        for g, s in zip(graphs, streams):
            with torch.cuda.stream(s):
                g.replay()
    for s in streams:
        current.wait_stream(s)


@case("parallel-replay/bug", "bug")
def _():
    graphs, ys = race_setup(shared=True)
    parallel_replay(graphs)  # ❌ graphs sharing a pool run concurrently
    return Outcome(race_report(ys), [0, 0, 0])


@case("parallel-replay/fix-separate-pools", "fix")
def _():
    graphs, ys = race_setup(shared=False)
    parallel_replay(graphs)  # ✅ concurrent, but no shared addresses
    return Outcome(race_report(ys), [0, 0, 0])


@case("parallel-replay/fix-serialize", "fix")
def _():
    graphs, ys = race_setup(shared=True)
    streams = [torch.cuda.Stream() for _ in graphs]
    prev = torch.cuda.current_stream()
    for _ in range(RACE_ITERS):
        for g, s in zip(graphs, streams):  # ✅ each replay waits for the previous one
            s.wait_stream(prev)
            with torch.cuda.stream(s):
                g.replay()
            prev = s
    torch.cuda.current_stream().wait_stream(prev)
    return Outcome(race_report(ys), [0, 0, 0])
