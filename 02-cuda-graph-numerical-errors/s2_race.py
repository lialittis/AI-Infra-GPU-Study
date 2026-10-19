"""Page section "Race Conditions": the CPU rewrites a pinned host buffer while an asynchronous H2D copy that reads it
is still queued behind GPU work. graph.replay() is always asynchronous, so a graph makes this easy to hit."""

import torch

from harness import Outcome, case, log

EXPECT = [0.0, 1.0, 4.0, 9.0]  # x ** 2 for x = 0..3


def cycles_per_ms():
    """Calibrate torch.cuda._sleep (the page uses torch.testing._internal.common_utils.get_cycles_per_ms)."""
    torch.cuda._sleep(1000)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    cycles = 10_000_000
    start.record()
    torch.cuda._sleep(cycles)
    end.record()
    end.synchronize()
    return cycles / start.elapsed_time(end)


def make_func():
    sleep_cycles = int(50 * cycles_per_ms())

    def func(x):
        torch.cuda._sleep(sleep_cycles)  # simulate 50 ms of GPU work queued before the copy
        x_cuda = x.to(device="cuda", non_blocking=True)
        return x_cuda ** 2

    return func


def pinned():
    return torch.empty(1, device="cpu", pin_memory=True)


def report(outputs):
    torch.cuda.synchronize()
    values = [o.item() for o in outputs]
    for i, v in enumerate(values):
        log(f"{i=}, expect {i ** 2}, got {v}")
    return values


@case("pinned/bug-eager", "bug")
def _():
    func, x = make_func(), pinned()
    stream = torch.cuda.Stream()
    outputs = []
    with torch.cuda.stream(stream):
        for i in range(4):
            x.fill_(i)  # ❌ CPU writes while the GPU may still be about to read
            y = func(x)
            outputs.append(y.clone())
    return Outcome(report(outputs), EXPECT, page=[0.0, 9.0, 9.0, 9.0])


def captured(func, x):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = func(x)
    return graph, y


@case("pinned/bug-graph", "bug")
def _():
    func, x = make_func(), pinned()
    graph, y = captured(func, x)
    outputs = []
    for i in range(4):
        x.fill_(i)  # ❌
        graph.replay()  # asynchronous: returns immediately
        outputs.append(y.clone())
    return Outcome(report(outputs), EXPECT, page=[9.0, 9.0, 9.0, 9.0])


@case("pinned/fix-synchronize", "fix")
def _():
    func, x = make_func(), pinned()
    graph, y = captured(func, x)
    outputs = []
    for i in range(4):
        torch.cuda.synchronize()  # ✅ wait for the GPU to finish reading
        x.fill_(i)
        graph.replay()
        outputs.append(y.clone())
    return Outcome(report(outputs), EXPECT)


@case("pinned/fix-event", "fix")
def _():
    func, x = make_func(), pinned()
    graph, y = captured(func, x)
    outputs = []
    event = torch.cuda.Event()
    for i in range(4):
        event.synchronize()  # ✅ wait for the previous replay only
        x.fill_(i)
        graph.replay()
        outputs.append(y.clone())
        event.record()
    return Outcome(report(outputs), EXPECT)


@case("pinned/fix-cuda-input", "fix")
def _():
    sleep_cycles = int(50 * cycles_per_ms())
    x_cpu = pinned()
    x_cuda = torch.empty(1, device="cuda")  # static graph input
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.cuda._sleep(sleep_cycles)
        y = x_cuda ** 2
    outputs = []
    for i in range(4):
        x_cpu.fill_(i)
        x_cuda.copy_(x_cpu)  # ✅ enqueued on the stream, so ordered with the replays
        graph.replay()
        outputs.append(y.clone())
    return Outcome(report(outputs), EXPECT)
