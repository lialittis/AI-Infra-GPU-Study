# CUDA graph numerical errors: reproducing NVIDIA's troubleshooting page

Every example on NVIDIA's [CUDA Graph Troubleshooting: Numerical Errors](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/numerical-errors.html)
page, buggy and fixed, as runnable cases checked against the correct (eager-equivalent) result. Where the page shows
only a fragment (most fixes, the learning-rate / step-counter / GradScaler / RNG notes, the debugging recipe) it is
completed into a runnable loop.

For a visual walkthrough of every example (code side by side, measured values per iteration, address and
timeline diagrams, and the mechanism behind each), open [explainer.html](explainer.html) in a browser.

## Summary

- **One rule explains every failure:** a CUDA graph is a recorded list of GPU operations, each with its kernel
  arguments, launch shape and raw memory addresses fixed. `replay()` re-issues exactly that list,
  asynchronously, and nothing else. The CPU-side Python does not run, and nothing is re-resolved.
- **59 of 66 cases behave as the page says** on torch 2.14.1 / CUDA 13.0 / GB10. Every bug for which the page
  prints deterministic output reproduces with identical numbers, except the freed-CPU-tensor example (see point 2
  below).
- **Seven cases differ, and each difference has an identified cause** (see [Where this run differs from the page](#where-this-run-differs-from-the-page)):
  - The page's fix for replay-order mismatch is wrong. `make_graphed_callables` captures all forwards, then the
    backwards in reverse; it does not alternate forward and backward per callable.
  - Recent PyTorch turns three of the page's silent bugs into either a loud capture error or a non-issue:
    - Pageable host copies are refused during capture.
    - Pinned blocks used by a capture are never recycled.
    - Custom RNG generators are registered automatically.
  - Putting the GradScaler step inside the graph fails capture outright, instead of silently using a stale scale.
- **One case went wrong because of a bug in this harness**, not the page. The harness itself freed a graph input
  (see [A bug the harness made itself](#a-bug-the-harness-made-itself)).

## Running

```bash
./run.sh                    # all cases; log saved to results/<timestamp>.txt (needs ../.venv; see the top-level README)
../.venv/bin/python run.py --list
../.venv/bin/python run.py pinned shared-pool   # only cases whose names contain these strings
```

- **One process per case.** Each case runs in its own process, so a corrupted allocator or CUDA context can't
  affect the next case.
- **What a case reports:**
  - *observed*: what the code produced
  - *expected*: what correct code gives
  - *page*: the output the page prints, when it prints one
- **Verdicts:**
  - Bug cases: `REPRODUCED` (observed differs from expected) or `NOT REPRODUCED`.
  - Fix cases: `CORRECT` or `WRONG`.
  - Either kind: `ERROR` if the code raised.
- **Code layout:** one module per page section, `s1_dynamic.py` … `s7_debugging.py`.
  - The page's ❌ / ✅ comments are kept at the lines they refer to.
  - `harness.page_loop` is the page's recurring loop: eager warmup for i < 3, capture at i = 3, replay after that.
- **Extra requirements:**
  - `control-flow/fix-kernel` compiles a CUDA extension with `load_inline`, so it needs `nvcc` on PATH and `ninja`
    (in `requirements.txt`).
  - `parallel-replay/*` needs about 10 GB of device memory. `RACE_ELEMS` sets the tensor size (the page uses
    1024³ elements, 4 GiB per tensor).

## A systematic view

The page lists seven categories. Underneath, they come down to four ways the replayed operation list can differ
from what eager code would do:

| What the graph froze | How it breaks | Page sections | Fix pattern |
|---|---|---|---|
| **Decisions made on the CPU** | Python `if`, Python scalars as kernel arguments, launch shapes, Python-side state, `is_current_stream_capturing()` branches. Only the capture-time choice exists | Dynamic control flow, capture-aware code, dynamic scalars, dynamic shapes, host state | Move the decision to the GPU (a tensor read by the kernel, `torch.where`, a branching kernel). Or keep it outside the graph (several graphs, partial capture, pad/bucket shapes, update state in eager code) |
| **Addresses** | The graph reads and writes fixed pointers. A reassigned Python variable, a freed input, a list of references to one output buffer, or a pool block shared with another graph all change *what lives at* the pointer without the graph knowing | Dynamic input tensors, autocast cache, freed inputs, output reuse, shared pool, replay order | Allocate inputs once and update them in place (`copy_` / `fill_`). Keep inputs alive. `clone()` outputs. Share a pool only with replay order = capture order |
| **Which work was recorded** | Only work on the capture stream, or on streams forked from it with events, is recorded. Work issued elsewhere ran once during capture and is missing from replays. Submodule `forward` calls, and their hooks, don't happen at all | Library stream awareness, module hooks | Issue work on `torch.cuda.current_stream()`, or fork and join an internal stream with `wait_stream`. Put hook logic outside the graph |
| **Timing** | `replay()` only enqueues work. CPU writes to pinned memory race the copies already queued, and graphs replayed on several streams race each other on shared pool memory | Pinned memory race, parallel replay | Wait on an event (or synchronize) before writing to host memory, or copy into a static device tensor. Don't replay pool-sharing graphs concurrently |

The first three rows are "a graph is a recording", and the last row is "a replay is asynchronous". The last row
also ties back to [01-cuda-streams](../01-cuda-streams/README.md):
- Concurrent replay is just multi-stream execution, with the same rule as before: overlap is only safe when the two
  sides share no memory.
- The library-stream bug is the capture-time version of "work on the wrong stream". Step B/C's stream analysis
  (`nsys_streams.py`) is exactly the tool the page recommends for finding it.

## Results

Measured on torch 2.14.1+cu130, CUDA 13.0, NVIDIA GB10 (`results/20261019-145927.txt`). "Page" means the page
printed output for that case and our output is identical.

**Dynamic behavior** (`s1_dynamic.py`)

| Case | Result | Observed |
|---|---|---|
| control-flow/bug | reproduced, same as page | 2,0,2,0,**0**,0: the odd (subtract) branch captured at i=3 is replayed for i=4 |
| control-flow/fix-where, fix-kernel, fix-multi-graph, fix-partial | correct | 2,0,2,0,2,0 (`load_inline` builds and runs on sm_121) |
| capture-aware/bug | reproduced, same as page | 0,1,2,−3,−4,−5 |
| capture-aware/fix, fix-sleep | correct | 0..5 |
| dynamic-input/bug | reproduced, same as page | 0,1,1,1,1,1 |
| autocast-cache/bug | reproduced, same as page | y stays 1.0 after w := 2 (the graph reads the stale cached FP16 copy) |
| autocast-cache/fix-no-cache, fix-inside | correct | 1.0 then 2.0 (fix-inside needs a warmup matmul first, see below) |
| dynamic-scalar/bug | reproduced, same as page | 1,2,4,8,**8,8** |
| hidden-scalar/lr-bug | reproduced | w = −1,−2,−3,−4 instead of −1,−1.5,−1.75,−1.875: lr stays at its first value |
| hidden-scalar/step-counter-bug | reproduced | the step stays at its capture-time value |
| hidden-scalar/*-fix | correct | the value is held in a device tensor, updated with `fill_` |
| hidden-scalar/grad-scaler-bug | **error** | capture fails: `scaler.step()` synchronizes (`cuStreamSynchronize` not permitted while capturing) |
| hidden-scalar/grad-scaler-fix | correct | fwd + `scale(loss).backward()` captured, `step` / `update` in eager code. Weights and loss scale (doubling every step) match an eager run exactly |
| rng/default-auto, default-registered, custom-registered | correct | 5 replays → 5 different dropout masks / random draws |
| rng/custom-unregistered | **not reproduced** | an unregistered custom generator also gives 5 different draws |
| dynamic-shape/bug | reproduced, same as page | output keeps 4 elements |
| dynamic-shape/fix-pad, fix-bucket | correct | sizes 3, 8, 5 handled by padding to 1024, or by buckets 4/8/16 |

**Race conditions** (`s2_race.py`)

| Case | Result | Observed |
|---|---|---|
| pinned/bug-eager | reproduced, same as page | 0,9,9,9: the CPU loop runs ahead of the 50 ms GPU sleep and overwrites the pinned buffer before each queued copy reads it |
| pinned/bug-graph | reproduced, same as page | 9,9,9,9 |
| pinned/fix-synchronize, fix-event, fix-cuda-input | correct | 0,1,4,9 |

**Deconstructed inputs** (`s3_deconstructed.py`)

| Case | Result | Observed |
|---|---|---|
| freed-cuda-input/bug | reproduced | `other_tensor` lands at the freed input's address (0x32ee00000), so the replay doubles random numbers |
| freed-cuda-input/fix | correct, same as page | 6,6,6,6 |
| freed-cpu-input/bug, fix-keep-alive, fix-gpu-tensor | **error** | `Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned` |
| freed-cpu-input/bug-pinned (ours) | **not reproduced** | the captured pinned block is never handed out again, so the replay stays correct |
| freed-cpu-input/fix-pinned-keep-alive, fix-full (ours) | correct | 0,1,2,3,3,3 |

**Output reuse and host state** (`s4_outputs_state.py`)

| Case | Result | Observed |
|---|---|---|
| output-reuse/bug | reproduced, same as page | ys[0]=5, ys[−1]=5, sum=15 |
| output-reuse/fix | correct, same as page | 1, 5, 8 |
| host-state/bug | reproduced, same as page | state frozen at 4 → y = 1,−1,1,−1,**−1**,−1 |
| host-state/fix-outside, fix-gpu-tensor | correct | 1,−1,1,−1,1,−1 |

**Missing operations** (`s5_missing_ops.py`)

| Case | Result | Observed |
|---|---|---|
| library-stream/bug | reproduced, same as page | replays give 0,0,0. The `x * 2` issued on the default stream during capture ran once, eagerly, on not-yet-written pool memory, and is not in the graph. No capture error is raised |
| library-stream/fix-current-stream, fix-pass-stream, fix-internal-stream | correct | 10,12,14. An internal stream forked from and joined back to the capture stream with `wait_stream` is recorded too |
| module-hooks/bug | reproduced, same as page | 4 hooks registered, 1 called |
| module-hooks/workaround-top-level | correct | the top-level hook sees all 5 parameters via `parameters(recurse=True)` |

**Memory pool sharing** (`s6_memory_pool.py`)

| Case | Result | Observed |
|---|---|---|
| shared-pool/bug | reproduced, same as page | y2 = 3.0: t1 and y2 have the same address |
| shared-pool/fix-read-immediately, fix-separate-pools, fix-capture-order | correct | 8, 4 |
| replay-order/bug | reproduced, same as page | grads 6, 3, 8 instead of 4, 6, 8 |
| replay-order/fix-capture-order (the page's fix) | **wrong** | still 6, 3, 8 |
| replay-order/fix-documented-order (ours) | correct | forwards f0, f1, f2, then backwards f2, f1, f0 |
| replay-order/fix-separate-pools, fix-capture-in-replay-order | correct | 4, 6, 8 |
| parallel-replay/bug | reproduced | y1 correct, but every element of y2 and y3 wrong (7 and 11 instead of 10 and 18). The page shows 13, 12, 19; the values depend on timing |
| parallel-replay/fix-separate-pools, fix-serialize | correct | 0 wrong elements |

**Debugging recipe** (`s7_debugging.py`): the page's layer-by-layer comparison, run on four Linear+tanh blocks.
Block 2 carries the page's "different precision during capture" anti-pattern.
- With that bug, the first divergence is found at layer 2 (max diff 2.3e−4).
- Without it, every layer matches exactly (diff 0), because eager mode and the replay run the same kernels.

## Where this run differs from the page

1. **The replay-order fix is wrong for current PyTorch.**
   - The page says `make_graphed_callables((f0, f1, f2), ...)` captures f0 fwd, f0 bwd, f1 fwd, f1 bwd, …
   - `torch/cuda/graphs.py` instead captures "fwd 1, fwd 2, ... fwd N, then bwd N, bwd N-1, ... bwd 1". Its
     docstring says the tuple order must be the order the callables run in the live workload.
   - So the page's "fixed" loop (`y0.backward()` right after `f0`) is itself a mismatch, and gives the same wrong
     gradients as the bug.
   - Correct order: all forwards in tuple order, then backwards in reverse (`fix-documented-order`). The page's other
     two fixes (separate pools, manual capture in replay order) work.
2. **Pageable host copies are rejected during capture.**
   - torch 2.14 raises `Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is
     pinned`.
   - The page's freed-CPU-tensor example therefore fails loudly instead of returning garbage.
   - So do both of its fixes. "Create directly on the GPU" uses `torch.tensor(i, device="cuda")`, which is itself a
     pageable H2D copy. A graph-safe way to make that tensor is `torch.full((), i, device="cuda")`, a fill kernel.
3. **Pinned blocks used by a capture aren't recycled.**
   - We rebuilt the page's freed-CPU-tensor bug with pinned memory. Allocating more pinned tensors after capture
     never reuses the captured block (0x32ee00000; the new tensors land at 0x32ee00200).
   - In eager mode, the same allocator does hand a freed pinned block out again once its copy has completed. A
     direct probe confirmed this.
   - So PyTorch's host caching allocator protects host memory used by a capture. (We did not trace which change
     added this protection.)
   - The *race* version of the bug (`pinned/bug-*`: the CPU writes to a live pinned input) still reproduces.
4. **Unregistered custom generators work.**
   - The generator keeps per-capture RNG state, keyed by capture id and created on first use (see
     `ATen/cuda/CUDAGeneratorImpl.h`).
   - A custom `torch.Generator` used during capture therefore advances on every replay without
     `register_generator_state`. Registering it explicitly does no harm.
5. **GradScaler step/update inside the graph fails loudly.**
   - `scaler.step()` reads `found_inf` on the host, a stream synchronize that is illegal during capture.
   - So the page's ❌ variant is a capture failure, not a silently stale loss scale. The ✅ variant reproduces
     exactly.
   - The loss scale itself is a device tensor that `update()` modifies in place, so a graph that multiplies by it
     sees every new value.
6. **Snippet gap: the autocast "capture inside the graph" fix needs a warmup.**
   - As printed, it has no warmup. In a fresh process the first matmul creates the cuBLAS handle inside the capture,
     and capture fails with `CUBLAS_STATUS_NOT_INITIALIZED`.
   - With one eager matmul first (the page's general "warm up before capture" rule), the fix works.

## A bug the harness made itself

- **What went wrong.** The first version of `shared-pool/fix-read-immediately` returned 17 instead of 8.
  `pool_setup()` built `x1` locally and returned only the graphs and outputs, so `x1` was freed. The `y2.clone()`
  in that case was then handed x1's block, and `graph1` read 4.0 as its input: 4·3 + 5 = 17.
- **Why it's instructive.** This is exactly the page's freed-CUDA-input failure, created by accident in code
  written to demonstrate a different failure. The other shared-pool cases happened to allocate nothing in between,
  so they looked correct.
- **The fix.** Keep the inputs alive: `pool_setup` now returns them.

## Takeaways

- **Before capturing a region, list everything that crosses its boundary:**
  - every tensor read from outside (it must be static and alive)
  - every Python value that becomes a kernel argument or shape (frozen)
  - every Python side effect (it happens once)
  - every library launch (it must use the current stream)
- **Treat graph outputs as buffers that the next replay overwrites.** This applies to every graph that shares the
  pool, not only the one that produced them.
- **Compare against eager mode, element by element, on more than one iteration.** Nearly all of these bugs are
  correct at the capture iteration and wrong only afterwards.
- **Many of the page's silent failures now fail loudly** on recent PyTorch: pageable copies, GradScaler syncs, and
  missing cuBLAS warmup. The ones that remain silent are the address and timing bugs, which no runtime check can
  catch:
  - stale inputs
  - output aliasing
  - pool aliasing and replay order
  - concurrent replay
  - pinned races
  - work on the wrong stream
