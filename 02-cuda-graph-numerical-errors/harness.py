"""Case registry shared by the section modules s*_*.py; run.py executes the cases.

A case reproduces one example from NVIDIA's "CUDA Graph Troubleshooting: Numerical Errors" page. It returns an
Outcome: what the code produced, what correct (eager-equivalent) code would produce, and, when the page prints one,
the output the page shows. A "bug" case is code the page says gives wrong results; a "fix" case is the page's fix.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class Case:
    name: str
    kind: str  # "bug" or "fix"
    section: str
    fn: Callable[[], "Outcome"]


@dataclass
class Outcome:
    observed: Any
    expected: Any
    page: Any = None


CASES: dict[str, Case] = {}


def case(name: str, kind: str):
    assert kind in ("bug", "fix")

    def register(fn):
        CASES[name] = Case(name, kind, fn.__module__, fn)
        return fn

    return register


def log(*args):
    print(*args, flush=True)


def same(a, b, rel=1e-5, abs_tol=1e-6) -> bool:
    """Structural equality with a float tolerance."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, rel, abs_tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)
    return a == b


def page_loop(step, before=None, n=6, capture_at=3):
    """The loop most of the page's examples use: run step(i) eagerly for i < capture_at, capture it at
    i == capture_at (and replay once, since capture does not execute anything), then only replay the graph.
    step(i) returns the output tensor; before(i), if given, runs eagerly every iteration (e.g. to update inputs).
    Returns z.item() per iteration."""
    graph = torch.cuda.CUDAGraph()
    outs = []
    for i in range(n):
        if before is not None:
            before(i)
        if i < capture_at:  # warmup
            z = step(i)
        elif i == capture_at:  # capture
            with torch.cuda.graph(graph):
                z = step(i)
            graph.replay()
        else:  # replay
            graph.replay()
        outs.append(z.item())
        log(f"{i=}, {z.item()=}")
    return outs
