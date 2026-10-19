# 01 · CUDA Streams

**Questions:** How do CUDA streams work, and how many streams does a real AI inference process use?

Plan:
- **Step A** (this directory): CUDA C microbenchmarks to understand the mechanism.
- **Step B**: PyTorch eager under `nsys`.
- **Step C**: a real serving engine (vLLM / SGLang) under `nsys --cuda-graph-trace=node`.

## 1. Mental model

**Programming model.** A stream is an **in-order queue of GPU work** (kernels, memcpys, memsets, event records).
- Work in the same stream runs in order.
- Work in different streams *may* overlap. Nothing guarantees it.
- Cross-stream ordering is expressed with events: `cudaEventRecord` followed by `cudaStreamWaitEvent`.

**What happens underneath:**

```
host thread ──launch──▶ command buffer ──▶ hardware work queue (connection) ──▶ work distributor ──▶ SMs
             (async,                         CUDA_DEVICE_MAX_CONNECTIONS              │
              returns ~µs)                   (default 8, max 32)                       └─▶ copy engine(s) for memcpy
```

1. **Launching is asynchronous.** The CPU only writes a command; the GPU consumes it later.
2. **Software streams are multiplexed onto a few hardware queues** ("connections", Hyper-Q). Streams sharing a queue can block each other. These are **false dependencies**.
3. **The work distributor schedules thread blocks** from runnable kernels onto SMs. Two kernels truly overlap only when **both fit at the same time** (threads, registers, shared memory, block slots).
4. **Memcpys run on DMA copy engines**, separate from the SMs. That's why copy and compute can overlap.
5. **The legacy default stream (stream 0) acts as a global barrier** for all *blocking* streams.

Core lesson: **streams give the GPU *permission* to overlap. Free resources decide whether it actually does.**

## 2. Method

All kernels are `spin_kernel` (see `common.cuh`): each block busy-waits for an exact time on `%globaltimer`. Each kernel records its first-block start and last-block end on the GPU clock, so overlap is measured on the device itself, not inferred from host timing:

- `concurrency = Σ kernel durations / wall time` (1.0 = serial)
- `peak` = the most kernels running at the same instant

Run everything with `./run_all.sh`. Raw output is in [`results/`](results/).

## 3. Results on GB10 (DGX Spark)

### 00 · Device facts that matter

| Property | Value | Why it matters |
|---|---|---|
| SMs | 48 | Total "room" that concurrent kernels share |
| Max threads / SM | 1536 | A 1024-thread block leaves no room for another 1024-thread block |
| Max blocks / SM | 24 | |
| `asyncEngineCount` | **1** | **Only one copy engine**: H2D and D2H can't run at the same time (H100 has 2+) |
| `integrated` | 1 | CPU and GPU share the same physical memory |
| `pageableMemoryAccess` (via host page tables) | 1 | The GPU can directly read plain `malloc` memory |
| Stream priorities | 0 (low) … −5 (high) | 6 levels |

### 01 · Overlap depends on how much of the GPU each kernel fills

8 kernels of 1 ms each, 1024-thread blocks (capacity = 48 blocks):

| blocks/kernel | % of GPU | 1 stream (µs) | 8 streams (µs) | concurrency |
|---|---|---|---|---|
| 1 | 2% | 8014 | **1023** | 7.8 |
| 6 | 12.5% | 8015 | **1015** | 7.9 |
| 12 | 25% | 8014 | 2007 | 4.0 |
| 24 | 50% | 8016 | 4004 | 2.0 |
| 48 | 100% | 8016 | **8002** | 1.0 |
| 96 | 200% | 16016 | 16003 | 1.0 |

**Measured concurrency ≈ min(#streams, capacity / blocks), exactly as predicted.** Once a kernel fills the GPU, extra streams buy nothing. Big GEMMs in LLM prefill are in that regime.

### 02 · False dependencies (32 streams × chain of 4 dependent 1 ms kernels; ideal 4 ms)

| `CUDA_DEVICE_MAX_CONNECTIONS` | depth-first issue | breadth-first issue |
|---|---|---|
| 1 | **97.3 ms** | 4.07 ms |
| 2 | 49.2 ms | 4.07 ms |
| 4 | 25.1 ms | 4.08 ms |
| 8 (default) | **13.1 ms** | 4.08 ms |
| 16 | 7.2 ms | 4.08 ms |
| 32 | 4.3 ms | 4.07 ms |
| 32, with 64 streams | 7.3 ms | 4.15 ms |

The numbers can be explained exactly. Each hardware queue is FIFO, and a kernel waiting on its predecessor blocks everything behind it in that queue. With 1 queue, the only overlap is at stream boundaries: the last kernel of stream *i* runs alongside the first kernel of stream *i+1*. That gives 32 × 3 ms + 1 ms = **97 ms**. With *C* queues, the wall time is roughly 97/*C*.

**Takeaways:**
- **Issue order matters as much as the number of streams.** Breadth-first issue never puts a dependent kernel in front of an independent one.
- **Default connections = 8.** A program using more than 8 streams (as some frameworks do) can silently serialize.
- **Megatron-LM *sets* `CUDA_DEVICE_MAX_CONNECTIONS=1` on purpose.** It forces communication kernels to be scheduled in issue order relative to compute. Here, connections are used as a scheduling tool rather than avoided.

### 03 · The legacy default stream is a hidden barrier (4 streams × 1 ms; ideal 1 ms)

| Variant | wall | concurrency |
|---|---|---|
| Blocking streams, nothing on stream 0 | 1007 µs | 3.97 |
| Blocking streams + tiny kernels on **legacy** stream 0 in between | **4049 µs** | **0.99** |
| **Non-blocking** streams + legacy stream 0 | 1013 µs | 3.95 |
| Blocking streams + **per-thread** default stream | 1015 µs | 3.94 |

A single 10 µs kernel on stream 0 serialized everything. Library code (or a stray `cudaMemcpy`/`cudaMemset` without a stream) can do this to you. PyTorch avoids the problem by never using the legacy stream for its own work.

### 04 · Copy/compute overlap (256 MB, kernel time matched to copy time)

| Case | copy | overlap |
|---|---|---|
| H2D pinned, kernel launched first | 4.4 ms (61 GB/s) | 0.96 |
| H2D pageable, kernel launched first | 4.4 ms (61 GB/s) | 0.99 |
| H2D pageable, **copy launched first** | 4.4 ms | **0.00**: host thread blocked inside `cudaMemcpyAsync`, so the kernel was launched too late |
| D2H pinned / pageable, kernel first | 4.4 ms | 0.99–1.00 |
| **H2D ‖ D2H at once** (pinned) | | **0.03**: only one copy engine |

GB10-specific observations:
- Pageable copies run as fast as pinned ones (~61 GB/s). This is plausible on an integrated GPU that uses the host page tables. On a discrete PCIe GPU, pageable copies are much slower.
- **Copy engines are slow compared with memory bandwidth** (~61 GB/s vs ~273 GB/s LPDDR5x). Because memory is unified, "copying" is often unnecessary: the GPU can read host memory directly. *Follow-up: measure zero-copy kernel reads of `malloc` memory.*
- Pageable `cudaMemcpyAsync` is **not async from the host's point of view**. Launch order decides whether you get any overlap.

### 05 · Priorities (big kernel = 20 waves × 200 µs blocks; small kernel launched ~500 µs later)

| Small kernel's stream | small starts after big starts | finishes before big ends |
|---|---|---|
| Same (low) priority | **4006 µs**: waits until every block of the big kernel is dispatched | −40 µs |
| **High** priority | **601 µs**: ~500 µs launch delay + ≤ 1 block boundary | 3412 µs |

Priorities don't preempt running blocks. They make the scheduler pick the high-priority kernel's blocks **at the next block boundary**. This is the mechanism for latency-critical side work, such as communication kernels that must not wait behind a long GEMM.

## 4. What this predicts for real inference

1. **Prefill.** Large GEMMs fill all SMs, so streams can't add much. Expect one dominant compute stream.
2. **Decode.** Kernels are small (batch is small), so there *is* room for overlap. But it's a strict layer-by-layer dependency chain, so there's nothing independent to overlap with. The real bottleneck is CPU launch overhead, which is solved by **CUDA Graphs**, not streams.
3. **Where side streams *are* worth it:**
   - Overlapping copies (D2H of sampled tokens, KV offload). On GB10 these compete for the single copy engine.
   - Multi-GPU communication.
   - Independent branches (MoE shared expert vs. routed experts).
   - High-priority streams for latency-critical work.
4. **Count to collect in Steps B and C:** streams created vs. streams with activity vs. share of GPU time per stream.

## 5. Step B: PyTorch generation loop under nsys

Code is in [`pytorch/`](pytorch/):
- `model.py`: Llama-style decoder, 0.85B params, bf16, random weights, static KV cache.
- `infer.py`: greedy generation in six styles, annotated with NVTX ranges.
- `nsys_streams.py`: analyzes an nsys report.
- `run.sh`: profiles every mode.

Setup is `scripts/setup_venv.sh`. Raw output is in `results/stepB-*.txt`. Every run uses batch 1, a 512-token prompt and 32 decode steps.

| Mode | What it does |
|---|---|
| `eager` | Textbook loop: `tok.cpu()` after each step (host sync) |
| `async` | Token D2H copy on a side stream into pinned memory; host reads step *i*'s token while the GPU runs *i+1* |
| `async-fixed` | `async` without the hidden per-step sync (see below) |
| `graph-sync` | Decode step captured into a CUDA graph by hand; synchronous token copy |
| `graph` | Same graph + async token copy |
| `compile` | `torch.compile(mode="reduce-overhead")`: Inductor kernels + automatic CUDA graphs |

### "How many streams?" has three answers

| Mode | Streams **created** | Streams that **received work** (whole run) | Streams carrying **GPU time** (measured) |
|---|---|---|---|
| eager | 7 (none by PyTorch) | 1 | **1** (100%) |
| async / async-fixed | 135 | 4 | **1** (100%) + 8-byte copies on 1 side stream |
| graph / graph-sync | 136 | 5 / 2 | **1** (100%) |
| compile | 137 | 5 | **1** (100%) + side copy stream |

1. **Created: 7 baseline, +128 the moment you touch `torch.cuda.Stream()`.**
   - The 7 baseline streams exist before PyTorch creates any. They include the null stream and one stream at priority −5. No runtime create call is traced for them, so they come from the driver or libraries.
   - Asking for *one* side stream makes PyTorch lazily build its **whole stream pool**: 4 priority levels (0, −1, −2, −3) × 32 streams = 128 `cudaStreamCreateWithPriority` calls. Graph capture and `torch.compile` each take one more stream.
2. **Received work: 1–5.** The extra active streams in graph/compile modes are only used during warmup and capture. PyTorch requires warming up on a side stream, and capture itself runs on a side stream.
3. **GPU time: always exactly 1 stream.** In every mode, 100% of kernel time is on one stream, which is **the legacy null stream** (id 7). PyTorch's default "current stream" *is* stream 0. Graph replays are also launched there. The only work on another stream is the 8-byte token D2H copy.

This matches the Step A prediction: a single-GPU decoder is one long dependency chain, so there is nothing for a second compute stream to do.

### Where the time goes: launch overhead, not streams

| Mode | ms/token | kernels/step | Host ms/step | GPU kernel ms/step | GPU busy |
|---|---|---|---|---|---|
| eager | 12.62 | 950 | 12.45 | 10.12 | 82% |
| async | 12.50 | 950 | 12.49 | 10.11 | 82% |
| async-fixed | 12.47 | 951 | 12.24 | 10.19 | 83% |
| graph-sync | 11.06* | 951 (950 in graph) | 10.82 | 10.09 | 93% |
| graph | 10.43 | 951 (950 in graph) | 10.18 | 10.05 | 98% |
| compile | 10.22 | **269** (266 in graph) | 10.20 | 9.91 | 98% |

\* graph-sync is 10.33 ms/token without nsys. Tracing makes each sync more expensive.

- **Eager is launch-bound.** Issuing 950 kernels takes the CPU ~12.4 ms (~13 µs per op of Python + dispatcher on this ARM CPU). The GPU needs only ~10.1 ms, so the GPU sits idle 18% of the time.
- **Async copies alone don't help** (eager 12.62 → async-fixed 12.47). The bottleneck is CPU launch throughput, not sync bubbles.
- **CUDA graphs fix it** (one launch per step, 98% busy). `torch.compile` additionally fuses 950 → 269 kernels.
- At ~10 ms per step for 1.7 GB of bf16 weights, decode streams weights at ~170 GB/s, about 60% of the GB10's ~273 GB/s. It's memory-bound, as expected for batch 1.

### When the side stream *does* pay off: hiding host work

A real server does CPU work for every token (scheduling, detokenization, stop checks). `--host-work-ms` simulates that work with a busy-wait after each token arrives on the host:

| Host work per token | graph-sync (ms/token) | graph + async copy (ms/token) |
|---|---|---|
| 0 ms | 10.33 | 10.40 |
| 3 ms | 13.17 | **10.24** |
| 6 ms | 16.24 | **10.40** |

With synchronous copies, time per token = GPU time + host work. With the copy on a side stream (signalled by an event, read one step late), host work overlaps the next step completely, as long as it's shorter than a step. **This is the reason inference engines keep a dedicated output-copy stream** ("async scheduling" / "overlap scheduler").

### Pitfalls found along the way

- **Hidden sync from `torch.tensor(..., device="cuda")`.** It does a *pageable* H2D copy, which blocks the host (Step A, experiment 04). The `async` mode had one in every step. The analyzer's memcpy breakdown (`Pageable->Device`) exposed it. Keep loop state such as positions on the GPU and update it in place.
- **`torch.compile(mode="reduce-overhead")` can silently skip CUDA graphs.** It prints `skipping cudagraphs due to mutated inputs` when a mutated input (the KV cache) has no fixed address. Fix: `torch._dynamo.mark_static_address(cache)`. Without the fix, the "compile" numbers would have been compile-without-graphs.
- **SDPA with a sliced bool mask** adds one small D2D copy per layer (to make the mask contiguous). It's harmless here, but it's an example of copies you don't expect.
- **Environment:** PyTorch crashed on import on this machine. The cause was a `cpuinfo` heap overflow from the container's partial CPU set; see [`tools/cpuinfo-fix`](../tools/cpuinfo-fix/README.md).

## 6. Next

- [ ] Step C: vLLM / SGLang on a small (real) model. Do they also put all compute on one stream? Which side streams do they add (output copy, KV offload, sampling)? What do batch size and prefill/decode mixing change?
- [ ] Step A follow-up: zero-copy reads of `malloc`/pinned memory from a kernel on GB10, compared with copy-then-compute.
- [ ] Optional: multi-stream inside one forward pass (e.g. overlapping an independent branch) to see when a second *compute* stream helps.
