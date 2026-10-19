"""Page sections "Output Tensor Reuse" and "CPU Code Eliminated During Graph Replay": a graph's outputs are the same
memory on every replay, and Python code inside the captured region runs only once, during capture."""

import torch

from harness import Outcome, case, log


def output_reuse(clone: bool):
    x = torch.empty(1, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = x ** 2 + 1
    ys = []
    for i in range(3):
        x.fill_(i)
        graph.replay()
        ys.append(y.clone() if clone else y)  # without clone(): three references to the same tensor
    out = torch.sum(torch.cat(ys))
    log(f"{ys[0].item()=}, {ys[-1].item()=}, {out.item()=}")
    return [ys[0].item(), ys[-1].item(), out.item()]


@case("output-reuse/bug", "bug")
def _():
    return Outcome(output_reuse(clone=False), [1.0, 5.0, 8.0], page=[5.0, 5.0, 15.0])


@case("output-reuse/fix", "fix")
def _():
    return Outcome(output_reuse(clone=True), [1.0, 5.0, 8.0], page=[1.0, 5.0, 8.0])


EXPECT_STATE = [1, -1, 1, -1, 1, -1]  # eager: state counts 1..6, even state negates y


def eager_func(state_value, x):
    if state_value % 2 == 0:  # depends on state updated by the graphed function
        x *= -1
    return x


def state_loop(graphed, read_state):
    """The page's loop: graphed(...) is warmed up, captured at i=3 and replayed; eager code then reads the state."""
    x = torch.ones(1, device="cuda", dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()
    out = []
    for i in range(6):
        if i < 3:
            y = graphed(x)
        elif i == 3:
            with torch.cuda.graph(graph):
                y = graphed(x)
            graph.replay()
        else:
            graph.replay()
        y = eager_func(read_state(), y)
        out.append(y.item())
        log(f"{i=}, state={read_state()}, {y.item()=}")
    return out


@case("host-state/bug", "bug")
def _():
    state = [0]

    def graphed_func(x):
        y = x * 1
        state[0] = state[0] + 1  # ❌ CPU mutation: runs during capture, skipped on replay
        return y

    return Outcome(state_loop(graphed_func, lambda: state[0]), EXPECT_STATE, page=[1, -1, 1, -1, -1, -1])


@case("host-state/fix-outside", "fix")
def _():
    state = [0]

    def graphed_func(x):
        return x * 1

    x = torch.ones(1, device="cuda", dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()
    out = []
    for i in range(6):
        if i < 3:
            y = graphed_func(x)
        elif i == 3:
            with torch.cuda.graph(graph):
                y = graphed_func(x)
            graph.replay()
        else:
            graph.replay()
        state[0] += 1  # ✅ the update runs in eager code, once per iteration
        y = eager_func(state[0], y)
        out.append(y.item())
        log(f"{i=}, {state=}, {y.item()=}")
    return Outcome(out, EXPECT_STATE)


@case("host-state/fix-gpu-tensor", "fix")
def _():
    state_tensor = torch.zeros(1, device="cuda", dtype=torch.int64)

    def graphed_func(x):
        y = x * 1
        state_tensor.add_(1)  # ✅ GPU op: captured and replayed
        return y

    return Outcome(state_loop(graphed_func, lambda: state_tensor.item()), EXPECT_STATE)
