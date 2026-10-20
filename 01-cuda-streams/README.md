# 01 · CUDA Streams

**Questions:** How do CUDA streams work, and how many streams does a real AI inference process use?

Plan:
- **Step A** (this directory): CUDA C microbenchmarks to understand the mechanism.
- **Step B** ([`pytorch/`](pytorch/)): a PyTorch generation loop under `nsys`.
- **Step C** ([`vllm/`](vllm/)): vLLM serving a real model under `nsys`.
- **Step D** ([`vllm/spec.py`](vllm/spec.py)): vLLM speculative decoding, a feature expected to add streams.
- **Step E** ([`vllm/offload.py`](vllm/offload.py)): vLLM KV-cache CPU offload, the other one.

## Summary

Measured on a single GB10. Details and numbers are in sections 3, 5, 6, 7 and 8.

**How streams work (Step A):**
- A stream is an in-order queue. Streams give the GPU *permission* to overlap work; free SMs decide whether it actually does (concurrency ≈ min(#streams, capacity / blocks per kernel)).
- Hidden serializers:
  - more streams than `CUDA_DEVICE_MAX_CONNECTIONS` hardware queues (default 8), combined with depth-first issue order
  - the legacy default stream (stream 0), which acts as a global barrier for blocking streams
  - pageable `cudaMemcpyAsync`, which blocks the host
- Priorities act at thread-block boundaries; running blocks are never preempted.
- GB10 has only one copy engine, so H2D and D2H copies can't overlap.

**How many streams real inference uses (Steps B and C):**

| Count | PyTorch loop | vLLM |
|---|---|---|
| Streams created | 7 → 135 (the first `torch.cuda.Stream()` builds PyTorch's 128-stream pool) | 135 eager, **3130** with graphs (+1 per instantiated CUDA graph) |
| Streams that received any work | 1–5 (extras only during warmup / graph capture) | 2–5 |
| Streams carrying compute | **1**: the legacy null stream | **1**: the legacy null stream |

- **One stream for all compute, plus one for output copies** (D2H of sampled tokens). On a single GPU, the forward pass is one dependency chain, so a second compute stream has nothing to run.
- **Performance comes from two other mechanisms:**
  - **CUDA graphs** remove CPU launch overhead. Step B's eager loop was launch-bound (~13 µs × 950 kernels per step on this CPU).
  - **Async scheduling** overlaps per-token CPU work with GPU work through that one copy stream. In vLLM, async scheduling is worth 13–15% and graphs about 5%.
- **Speculative decoding (Step D) adds no compute stream either.** The draft model and the verify pass run one after the other on the null stream. The only new active stream copies draft tokens to the host, and only when structured output needs them.
- **KV offload (Step E) adds transfer streams, not compute streams.** Its copies overlap decode compute (46–94% of their busy time), stores cost 1–5%, and reloading an evicted 2k-token prefix is ~10× faster than recomputing it.
- Multiple *compute* streams appear only for multi-GPU communication and micro-batch overlap, which a single GPU can't exercise.

What graphs and async copies cost in correctness, and how each failure ties back to these results, is covered in
[02-cuda-graph-numerical-errors: Relation to 01](../02-cuda-graph-numerical-errors/README.md#relation-to-01-cuda-streams).

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
- `../nsys_streams.py`: analyzes an nsys report (shared with Step C).
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

## 6. Step C: vLLM on a real model

Code is in [`vllm/`](vllm/):
- `bench.py`: offline `LLM.generate` with fixed-shape workloads, each inside an NVTX `measured:<name>` range.
- `run.sh`: profiles each config with nsys.

Setup is `scripts/setup_vllm_venv.sh`, which creates a separate venv because vLLM pins its own torch. Raw output is in `results/stepC-*.txt`.

Setup details:
- **Software:** vLLM 0.32.0 (Model Runner V2, V1 engine), torch 2.13.0+cu130.
- **Model:** **Qwen2.5-1.5B-Instruct**, bf16, run with `gpu_memory_utilization=0.3`. This is a shared machine, and on unified memory that fraction is of system RAM.
- **Inputs:** prompts are random token IDs, EOS is ignored, and prefix caching is off, so every run has identical shapes.

**Does vLLM support GB10 (sm_121)?** This was checked before installing, by inspecting the aarch64 wheel with `cuobjdump`:

| vLLM library | GPU code in the wheel | On GB10 |
|---|---|---|
| `_C_stable_libtorch`, `_moe_C_stable_libtorch` (core / MoE kernels) | sm_80 … sm_120 binaries + PTX | ✅ sm_120 binaries run on sm_121 (same major version) |
| `vllm_flash_attn/_vllm_fa2_C` | sm_80 binary + sm_80 PTX | ✅ via driver PTX JIT (selected backend: `FLASH_ATTN`, FA2) |
| `vllm_flash_attn/_vllm_fa3_C` | sm_90a | ❌ |
| FlashMLA, DeepSelect, QuTLASS, DeepGEMM | sm_90a / sm_100 | ❌ only needed for MLA (DeepSeek-style) / FP4 models |

The aarch64 wheel pins torch 2.13.0, whose PyPI aarch64 build is CUDA 13.0, matching the 580 driver. It goes in its own venv so it doesn't clash with Step B's torch 2.14.1. Engine startup takes about 2 minutes: model load, `torch.compile` (~7 s, cached afterwards), and two rounds of CUDA graph capture (~40 s each).

| Config | Meaning |
|---|---|
| `default` | vLLM defaults: `torch.compile` + CUDA graphs (full graphs for decode, piecewise for mixed batches) + async scheduling |
| `no-async` | `async_scheduling=False`: the engine waits for each step's output before scheduling the next |
| `eager` | `enforce_eager=True`: no compile, no graphs (async scheduling still on) |

| Workload | Requests × prompt → output |
|---|---|
| `decode-b1` | 1 × 512 → 128 |
| `decode-b32` | 32 × 512 → 128 |
| `prefill-b32` | 32 × 1024 → 1 |

### Stream counts

| | default / no-async | eager |
|---|---|---|
| Streams **created** (`EngineCore` process) | **3130** | 135 |
| … of which: PyTorch's pool (128 traced `cudaStreamCreateWithPriority` calls) | 128 | 128 |
| … of which: driver-internal, **one per instantiated CUDA graph** (2995 `cudaGraphInstantiate` calls) | 2995 | 0 |
| Streams that **received work** (whole run) | 5 | 2 |
| Streams carrying **GPU time** (every workload) | **1** (100%) | **1** (100%) |

- **All compute runs on the legacy null stream (id 7), for every config and workload,** including batch 32 and the 32k-token prefill. Graph replays are launched there too. vLLM, like our Step B harness, uses PyTorch's default stream.
- **The only other stream doing work during inference is stream 13, vLLM's `output_copy_stream`.** It carries the D2H copies of sampled token IDs and their counts: 8 B + 4 B per step at batch 1, and 256 B + 128 B at batch 32 (32 × int64 and 32 × int32). In Model Runner V2 this stream is created unconditionally, so it's used even with `async_scheduling=False`.
- **The 3 extra active streams in graph configs** are only used during warmup and graph capture.
- **Why ~3000 streams are "created":** vLLM captures 2995 CUDA graphs (piecewise pieces × capture batch sizes, plus full decode graphs). Each instantiated graph shows up as one extra context stream in nsys, with no create call traced; the count matches exactly: 3033 − 38 baseline = 2995. These streams never receive work we can see. "Number of streams created" is therefore a meaningless metric for graph-heavy programs.
- **No compute side stream exists in the code path.** Reading the V2 runner's source (`vllm/v1/worker/gpu/`), the only other streams are for structured-output copies, speculative decoding copies, and the comm/capture streams for micro-batch overlap (DBO, multi-GPU). None is active in this setup.

### Performance by config

| Workload | Metric | default | no-async | eager |
|---|---|---|---|---|
| decode-b1 | ms / step (wall) | **16.24** | 18.74 | 17.08 |
| | GPU busy | 99% | 87% | 95% |
| | kernels / step (in graph) | 321 (307) | 323 (308) | 323 (0) |
| decode-b32 | ms / step (wall) | **22.17** | 25.10 | 23.07 |
| | GPU busy | 98% | 87% | 96% |
| prefill-b32 | prompt tok/s | 23.6k | 23.5k | 23.4k |
| | kernels in graph | 0 | 0 | 0 |

- **Async scheduling matters more than CUDA graphs here.** Turning it off costs 15% at batch 1 and 13% at batch 32: the GPU idles 13% while the CPU schedules the next step. That is the Step B "hide host work" experiment, in a real engine.
- **CUDA graphs only gain about 5% in vLLM**, much less than in Step B (12.5 → 10.4 ms). vLLM's eager decode issues only about 320 kernels per step (our toy model issued 950). With async scheduling the CPU can run ahead, so launch cost is mostly hidden: eager still keeps the GPU 95% busy.
- **Batching:** 32 sequences cost 1.37× the time of one (22.2 vs 16.2 ms/step), giving 23× more throughput. At batch 1, decode streams ~3.1 GB of weights in 16.1 ms, about 190 GB/s or roughly 70% of GB10 peak.
- **Prefill does not use CUDA graphs.** A 32 × 1024-token batch is split into chunks of up to 8192 tokens (5–7 engine steps), which is far above the largest captured graph size, so it runs piecewise-compiled kernels launched eagerly. The steps are GPU-bound (~200–270 ms each), so neither graphs nor async scheduling matter.

### Answer to the original question

> How many streams does a real AI inference process use?

On a single GPU: **one stream for all compute, plus one for output copies.** Several hundred to several thousand more may be *created*: PyTorch's 128-stream pool and one per CUDA graph. A few more are touched during startup, but none carry inference work. Real engines get performance from **CUDA graphs** (launch overhead) and **async scheduling** (overlapping CPU work with the GPU via that one copy stream), not from concurrent compute streams. Multiple compute streams only become relevant with multi-GPU communication (DBO's `comm_stream`), KV-cache offload, or speculative decoding. None of these can be tested on this single-GPU machine except offload and speculative decoding.

### Pitfalls found along the way

- **nsys and vLLM:**
  - `EngineCore` is *forked* without `exec`. Without `--trace-fork-before-exec=true`, nsys silently records only the front-end process (an 88 KB report with no CUDA data).
  - vLLM kills `EngineCore` at shutdown, which loses trace data nsys hasn't flushed. In our first run, the last ~40% of the measured workloads were missing. The fix is `profiler_config=ProfilerConfig(profiler="cuda")` plus `llm.start_profile()` / `stop_profile()`: the worker calls `cudaProfilerStop()`, which flushes the buffers.
- **Read the code that actually runs.** vLLM 0.32 uses Model Runner V2 (`v1/worker/gpu/model_runner.py`), not the older `gpu_model_runner.py`. The startup log line (`[model_runner.py:...]`) tells you which one. V2 has no per-step NVTX scopes; the scheduler's `schedule: update_after_schedule` range marks steps instead.
- **FlashAttention on sm_121.** vLLM's FA2 library contains only sm_80 binaries *plus sm_80 PTX*, so the driver JIT-compiles it for GB10 on first use (cached in `~/.nv/ComputeCache`). FA3 (sm_90a) is unusable, and so are FlashMLA and DeepGEMM (sm_90a/sm_100).
- **Unified memory:** `gpu_memory_utilization` is a fraction of *system* RAM on GB10. The default 0.9 would claim ~110 GB of a shared machine.

## 7. Step D: speculative decoding in vLLM

**Question:** Speculative decoding runs a second "model" (a draft) next to the target. Does it add compute
streams, or only copies? And what does it cost and change?

Code is in [`vllm/`](vllm/):
- `spec.py`: configs × workloads, each workload in an NVTX `measured:<name>` range.
- `compare_tokens.py`: token-by-token comparison against a reference config (shared with Step E).
- `run_spec.sh`: runs every config under nsys, then `../nsys_streams.py`, then the comparison.

Raw output is in `results/stepD-*.txt`. Same engine and model as Step C (vLLM 0.32.0, Qwen2.5-1.5B-Instruct, bf16).
Async scheduling stays on in every config: vLLM enables it for `ngram_gpu` and `draft_model`, and turns it off for
CPU `ngram`.

| Config | Drafting |
|---|---|
| `baseline` | none |
| `ngram-k4` | prompt lookup on the GPU (`ngram_gpu`, 5-gram match), 4 draft tokens, no extra model |
| `draft-k2` / `k4` / `k8` | Qwen2.5-0.5B-Instruct as draft model, 2 / 4 / 8 draft tokens |

The workloads use real text through the chat template, because acceptance depends on content:

| Workload | Requests | Task | Expected acceptance |
|---|---|---|---|
| `repeat-b1`, `repeat-b32` | 1, 32 | copy a 3-paragraph passage verbatim | high: the output is in the prompt |
| `story-b1`, `story-b32` | 1, 32 | write a short story | low |
| `json-b1` | 1 | the story prompt under a JSON schema | — (structured output is the only case that copies draft tokens to the host) |

### Streams: still one compute stream

| | baseline | ngram-k4 | draft-k2 | draft-k4 | draft-k8 |
|---|---|---|---|---|---|
| Streams created | 3130 | 3494 | 3673 | 3494 | 3429 |
| … `cudaGraphInstantiate` calls | 2995 | 3359 | 3538 | 3359 | 3294 |
| Streams that received work | 5 | 6 | 6 | 6 | 6 |
| Streams carrying GPU time | **1** (null) | **1** | **1** | **1** | **1** |

- **The draft and the verify pass run back to back on the null stream (id 7).** Every config and workload puts
  99.9–100% of GPU time there. The V2 runner's speculators (`v1/worker/gpu/spec_decode/`) never switch streams.
- **The one new active stream is id 17, vLLM's `DraftTokensHandler.copy_stream`.** It does one D2H copy of the
  draft tokens per step, and only in `json-b1`: the scheduler needs drafts on the host only to check them against
  the grammar. Stream 21, the structured-output copy stream (`v1/worker/gpu/structured_outputs.py`), is active in
  `json-b1` for the baseline too.
- **More streams are "created" only because more graphs are captured** (one driver stream per instantiated graph,
  as in Step C).

### Performance

Batch 1 (ms per output token; acceptance rate and tokens per verify step in parentheses):

| Config | repeat-b1 | story-b1 | json-b1 |
|---|---|---|---|
| baseline | 16.35 | 16.34 | 16.48 |
| ngram-k4 | **3.39** (88%, 4.54) | 14.39 (0%) | 14.41 (0%) |
| draft-k2 | 9.85 (98%, 2.97) | **12.76** (61%, 2.22) | **12.28** (67%) |
| draft-k4 | 9.04 (99%, 4.94) | 14.51 (48%, 2.93) | 13.70 (55%) |
| draft-k8 | 8.65 (97%, 8.75) | 19.63 (32%, 3.59) | 18.78 (37%) |

Batch 32 (output tok/s):

| Config | repeat-b32 | story-b32 |
|---|---|---|
| baseline | 1727 | 2024 |
| ngram-k4 | **2055** (68%) | 1558 (0%) |
| draft-k2 | 1881 (97%) | 1860 (63%) |
| draft-k4 | 1710 (94%) | 1486 (48%) |
| draft-k8 | 1390 (88%) | 1029 (30%) |

What explains the numbers:
- **Prompt lookup is nearly free, so it wins whenever the output repeats the prompt.** On `repeat-b1` it is 4.8×
  faster than the baseline, against 1.8× for the draft model with the same acceptance.
- **The draft model runs eagerly.** `standalone_ar/speculator.py` says "CUDA graph not yet supported for standalone
  AR speculator":
  - Kernels per step rise from 324 (baseline) to 838 / 1364 / 2388 at k = 2 / 4 / 8.
  - The fraction of kernels inside graphs falls from 95% to 33% / 20% / 11%, and the GPU is idle 4–7% of the time.
  - This is Step B's launch overhead coming back.
- **The draft is also too big for this target.** Each draft pass costs about 5.8 ms, roughly 40% of a target step
  (37.7 ms per k=4 step vs 14.5 ms for ngram). Qwen2.5-0.5B has about a third of the target's weights, and eager
  launches add the rest. Raising k only pays off while nearly every draft is accepted: k=8 is the fastest config
  on `repeat-b1` and the slowest on `story-b1`.
- **ngram never skips verification.** When no 5-gram matches, `ngram_gpu` fills the draft slots with the last
  sampled token, so every step verifies k+1 tokens. That keeps shapes fixed for CUDA graphs: it is
  [02's pad-to-a-static-shape fix](../02-cuda-graph-numerical-errors/README.md#a-systematic-view). The cost shows
  at batch 32, where `story-b32` verifies 160 tokens per step for nothing (−23%).
- **On GB10, verifying 5 tokens is cheaper than decoding 1.** On `story-b1`, ngram accepts nothing yet is 12% faster
  per step than the baseline (14.2 vs 16.1 ms of GPU time). The traces show cuBLAS choosing a different GEMM
  kernel by row count: the M=1 kernel takes 15.6 ms per step for the 112 linear layers (28 layers × 4), the M=5
  kernel 13.6 ms. That is about 190 vs 220 GB/s of weight streaming. Plain batch-1 decode on this machine is
  slower than it needs to be.
- **Batching and speculation compete for the same headroom.** Both use spare compute while weights stream from
  memory. At batch 32 only high acceptance with small k still gains (ngram +19%, draft-k2 +9% on `repeat-b32`).

### Correctness: greedy output changes only at near-ties

With temperature 0, speculative decoding is meant to produce exactly the target model's text. `compare_tokens.py`
checks this the way 02 does: token by token against a reference, which is the baseline run again with
`logprobs=2`. At each first mismatch it reads the reference's top-2 logprob gap.

| Comparison | Identical requests | Top-2 gap at first mismatch |
|---|---|---|
| baseline b1 vs the same prompt inside b32 (no speculation) | repeat 1/1, story 0/1 | 0.25 |
| baseline measured pass vs reference (same config, logprobs off vs on) | all, except story-b32 28/32 | 0, 0.125 |
| ngram-k4 / draft-k2 / k4 / k8 | repeat-b1 1/1, repeat-b32 29–31/32, story-b1 0/1, story-b32 0–1/32, json-b1 0/1 | always 0, 0.125 or 0.25 |

- **Every mismatch is a near-tie at the resolution of bf16 logits.** Logits around 16–32 are spaced 0.125 apart in
  bf16, so a gap of 0 or 0.125 means the top two tokens were equal or one step apart. No mismatch had a larger
  gap, and nearly always the other run picked the reference's runner-up.
- **The cause is the verify pass's shape, not speculation itself.** All four speculative configs diverge on
  `story-b1` at the same position (76) with the same token. That includes `ngram-k4`, which accepted no draft, so
  its text came entirely from the target. What they share is running the target on k+1 tokens per step, which
  uses different GEMM kernels and so different rounding.
- **The baseline does this to itself as well.** Batch 1 vs batch 32 flips a tie, and so does the same config at
  batch 32 with logprobs on vs off. "Lossless" speculative decoding is exact only up to the same batch-invariance limits as plain
  decoding.
- **Structured output stays valid.** `json-b1` diverges from the reference at a tie, but every config's output
  parses as JSON.

### Answer

On one GPU, speculative decoding adds **no compute concurrency**. The draft and the target take turns on the same
stream. The only new stream copies draft tokens to the host, and only with structured output. Its value comes
from doing more work per weight read, as batching does. On GB10 with vLLM 0.32:
- Prompt lookup is the clear win when outputs repeat the input.
- A draft model is limited by running without CUDA graphs and by its size relative to the target.

## 8. Step E: KV-cache CPU offload in vLLM

**Question:** KV offload moves cache blocks between GPU and host memory on side streams. Do those transfers
overlap with compute? What do they cost when nothing is reused, and what do they save when an evicted prefix
comes back?

Code is in [`vllm/`](vllm/): `offload.py` (configs × workloads), `run_offload.sh` (nsys + `../nsys_streams.py`
+ `compare_tokens.py`). Raw output is in `results/stepE-*.txt`. Same engine and model as Steps C/D, with
**prefix caching on**.

**What the two offload paths do** (vLLM 0.32 source, confirmed in the traces):

| | `native` (`OffloadingConnector`, default) | `simple` (`SimpleCPUOffloadConnector`, `VLLM_USE_SIMPLE_KV_OFFLOAD=1`) |
|---|---|---|
| Enabled by | `kv_offloading_size=<GiB>` | the same + env var |
| Streams | one stream per transfer, from PyTorch's pool, default priority; each waits on compute and on the previous transfer | one load and one store stream, **lowest** priority, gated by a compute-done event |
| Stores (GPU→CPU) | `cuMemcpyBatchAsync` (copy engine), every full block written through as it is computed | `cuMemcpyBatchAsync` |
| Loads (CPU→GPU) | Triton `_swap_blocks_kernel` reading pinned host memory (used for pages < 28 KiB), so loads run **on the SMs** | `cuMemcpyBatchAsync` (copy engine) |

| Config | GPU KV cache | Offload |
|---|---|---|
| `prefix` | 0.5 GiB (~18.7k tokens at 28 KiB/token) | none: an evicted prefix is recomputed |
| `prefix-big` | 8 GiB: nothing is evicted | none |
| `offload-native` / `offload-simple` | 0.5 GiB | 8 GiB of pinned host memory |

| Workload | What runs |
|---|---|
| `reuse-prefill` | fill with 32 distinct 2048-token documents (3.5× the small cache); measured: documents 0–15 again with a new 32-token suffix, 1 output token |
| `reuse-decode` | the same with a fresh document set and 64 output tokens |
| `mixed` | 16 requests decode alone for 32 tokens, then 16 refilled documents arrive; the decode requests' token gaps show the stall |
| `store-decode` / `store-prefill` | Step C's `decode-b32` / `prefill-b32` with unique prompts: no reuse, only store traffic |

Prompts are random token ids. A one-time small fill-and-reload cycle runs first, so the Triton kernel's
compilation stays out of the measurements. vLLM's own counters confirm every reuse run actually hit the CPU cache:
32,768 external prefix-cache hits and 939.5 MB loaded = 32,768 tokens × 28 KiB.

### Streams: transfer streams that overlap compute, still one compute stream

- **Compute stays on the null stream in every config.** Streams created don't change (1649): native's transfer
  streams come from PyTorch's existing 128-stream pool. Native's run had 33 active streams (up to 24 non-null
  streams in one workload, one per transfer); simple's had 2 more than `prefix`.
- **Transfers do overlap compute when there is compute to overlap**:

| Workload | native: side-stream busy / overlapped with null-stream kernels | simple |
|---|---|---|
| `store-decode` | 47.7 ms / **94%** | 28.1 ms / 46% |
| `store-prefill` | 75.0 ms / 79% | 21.7 ms / 20% |
| `mixed` (loads during decode) | 47.2 ms / **87%** | 153.0 ms / 71% |
| `reuse-prefill` (loads before the prefill that needs them) | 15.1 ms / 6% | 55.8 ms / 0% |

  In `reuse-prefill` nothing else is running: each request's prefill waits for its own load, so the transfer
  can't overlap anything.
- **Loads on the SMs beat loads on the copy engine on GB10.** Native's Triton kernels moved the 939.5 MB in 15.1 ms
  of busy time (~62 GB/s); simple's DMA took 55.3 ms (17 GB/s). Stores, both by `cuMemcpyBatchAsync`, ran at
  10–12.5 GB/s (native) and 30–43 GB/s (simple) in the store-heavy workloads (`store-*`, `mixed`). All are batches of many small block copies, well below the
  61 GB/s Step A measured for one 256 MB copy.
- **Stream priority has nothing to act on in `simple`.** CUDA documents priorities for scheduling kernel blocks
  (Step A 05); simple's low-priority streams carry only copies. Native's loads *are* kernels sharing the SMs, yet
  they didn't slow decode visibly (below).

### Performance

| Workload | prefix (recompute) | offload-native | offload-simple | prefix-big (GPU hit) |
|---|---|---|---|---|
| `reuse-prefill` | 1475 ms | **152 ms** | **156 ms** | 65 ms |
| `reuse-decode` | 3590 ms | **2435 ms** | **2358 ms** | 1263 ms* |
| `mixed`: decode token gap after arrival, median / max | 18.0 / **348.7 ms** | 18.4 / 25.2 ms | 18.2 / 35.1 ms | 20.5 / 39.2 ms |
| `mixed`: total | 6079 ms | 4975 ms | 4778 ms | 4664 ms |
| `store-decode` (ms / step) | 30.32 | 30.67 (+1.2%) | 30.59 (+0.9%) | 22.81* |
| `store-prefill` | 1405 ms | 1476 ms (+5.1%) | 1452 ms (+3.3%) | 1400 ms |

\* Not comparable: `reuse-decode` (16 × 2144 tokens) and `store-decode` (32 × 640) need more than the small
cache's 18.7k tokens, so the three small-cache configs can't run all requests at once. Compare those rows
against `prefix`. Native's `store-decode` run even *loaded* 40 MB with no reuse in the workload, most likely
preempted requests resuming from their CPU copies.

- **Reloading beats recomputing by ~10×.** Re-prefilling 16 × 2048 evicted tokens takes 1475 ms; loading them
  takes 152–156 ms.
- **The remaining gap to a GPU hit is scheduling, not bandwidth.** The load kernels were busy for only 15 ms,
  yet offload took ~90 ms longer than a GPU hit, spread over 15 (native) or 36 (simple) engine steps instead of 4,
  with the GPU idle 13–23% of the window. A request whose blocks are loading waits until the load completes,
  then is scheduled in a later step.
- **Offload protects running decodes.** Recomputing the arriving documents stalls the other requests: one decode
  token took 348.7 ms because a prefill chunk ran inside its step. With offload, the worst gap was 25–35 ms,
  no worse than a GPU hit.
- **Write-through stores cost 1–5%** when nothing is ever reused.
- **Offload restores evicted prefixes, not capacity.** Running requests still need their KV on the GPU, so
  `reuse-decode` stays far behind `prefix-big`.

### Correctness: loaded KV is exact

| Workload | prefix / offload-native / offload-simple vs reference (`prefix-big` with logprobs) |
|---|---|
| `reuse-prefill` | 16/16 identical, all configs |
| `mixed` | 32/32 identical, all configs |
| `reuse-decode` | 14–15/16: only requests 2 and 8 differ, at position 0, top-2 gap **0.0000** |

- **The only differences are exact ties.** Requests 2 and 8 start with an exact tie, and even `prefix-big`'s own
  measured pass flips them against its logprob pass. Once on the other branch, a second tie at position 3 of
  request 8 also differs between configs. Everything else is bit-identical across all four configs.
- **No timing race showed up.** A load racing the zeroing of its destination block, or a store reading KV before
  compute wrote it, would give wrong output. That is [02's async-timing failure](../02-cuda-graph-numerical-errors/README.md#a-systematic-view)
  at full scale. vLLM orders both with `wait_stream` / events, and no output differed outside ties.

### Answer

KV offload adds **transfer streams, not compute concurrency**. Unlike Step D's tiny draft-token copies, these
transfers are large and do overlap decode compute (46–94% of their busy time). The win is not throughput but avoided recomputation:
~10× faster reuse of evicted prefixes and no prefill stall for running requests, for 1–5% of store overhead.

GB10 caveats:
- "CPU" and "GPU" KV live in the same LPDDR5x, so offload adds no real memory capacity here. The mechanism is
  measured faithfully; the capacity benefit would differ on a discrete GPU.
- Over PCIe, native's SM-based loads would read host memory across the bus; the ranking of SM vs DMA loads
  could differ.

## 9. Next

- [ ] Step A follow-up: zero-copy reads of `malloc`/pinned memory from a kernel on GB10, compared with copy-then-compute.
- [x] Speculative decoding (Step D): only a draft-token copy stream, no compute concurrency.
- [x] KV-cache CPU offload (Step E): transfer streams that overlap compute, no compute concurrency.
- [ ] Step E follow-ups: why stores via `cuMemcpyBatchAsync` reach only 10–43 GB/s; offload on a discrete GPU, where it adds real capacity.
- [ ] Step D follow-ups: a draft model ~10× smaller than the target (or EAGLE heads) once graphed; the M=1 vs M=5 GEMM kernel choice on GB10 in isolation.
- [ ] Optional: multi-stream inside one forward pass (e.g. overlapping an independent branch) to see when a second *compute* stream helps.
- [ ] Multi-GPU (needs different hardware): NCCL communication streams, DBO `comm_stream`, `CUDA_DEVICE_MAX_CONNECTIONS=1` as used by Megatron.
