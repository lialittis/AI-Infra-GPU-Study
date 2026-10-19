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

## 5. Next

- [ ] Step A follow-up: zero-copy reads of `malloc`/pinned memory from a kernel on GB10, compared with copy-then-compute.
- [ ] Step A follow-up: view one run in `nsys` to learn how streams look on the timeline.
- [ ] Step B: PyTorch eager small transformer: number of streams, where `non_blocking=True` copies land.
- [ ] Step C: vLLM / SGLang on a small model: streams per phase, eager vs. CUDA Graph.
