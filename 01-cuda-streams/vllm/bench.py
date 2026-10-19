"""Step C: offline vLLM generation with fixed-shape workloads, each wrapped in an NVTX "measured:<name>" range.

Configs:
  default   torch.compile + CUDA graphs + async scheduling (vLLM's defaults on one GPU)
  no-async  same, async scheduling off (the engine waits for each step's output before scheduling the next)
  eager     enforce_eager: no torch.compile, no CUDA graphs (async scheduling stays on)

Prompts are random token ids, generation ignores EOS and prefix caching is off, so every run has exactly the
same shapes. The model runs in vLLM's EngineCore child process (forked, so nsys needs
--trace-fork-before-exec=true); analyze the report with ../nsys_streams.py. VLLM_NVTX_SCOPES_FOR_PROFILING=1
adds per-step NVTX ranges from the scheduler.

The engine is shut down by killing EngineCore, which loses CUDA trace data nsys has not flushed yet. vLLM's
"cuda" profiler makes the worker call cudaProfilerStop() on stop_profile(), which flushes it.
"""

import argparse
import random
import time

import torch.cuda.nvtx as nvtx
from vllm import LLM, SamplingParams
from vllm.config import ProfilerConfig
from vllm.inputs import TokensPrompt

# name: (requests, prompt tokens, output tokens)
WORKLOADS = {
    "decode-b1": (1, 512, 128),
    "decode-b32": (32, 512, 128),
    "prefill-b32": (32, 1024, 1),
}

CONFIGS = {
    "default": {},
    "no-async": {"async_scheduling": False},
    "eager": {"enforce_eager": True},
}


def make_prompts(n: int, length: int, seed: int) -> list[TokensPrompt]:
    rng = random.Random(seed)
    return [TokensPrompt(prompt_token_ids=[rng.randrange(100, 20000) for _ in range(length)]) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--config", choices=list(CONFIGS), default="default")
    ap.add_argument("--workloads", nargs="+", choices=list(WORKLOADS), default=list(WORKLOADS))
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.3,
                    help="fraction of device memory vLLM may claim (on unified-memory GPUs this is system RAM)")
    args = ap.parse_args()

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=2048,
              enable_prefix_caching=False, seed=0, profiler_config=ProfilerConfig(profiler="cuda"),
              **CONFIGS[args.config])
    llm.start_profile()

    for name in args.workloads:
        n, prompt_len, out_len = WORKLOADS[name]
        params = SamplingParams(temperature=0, max_tokens=out_len, ignore_eos=True)
        llm.generate(make_prompts(n, prompt_len, seed=1), params, use_tqdm=False)  # warm up this shape
        prompts = make_prompts(n, prompt_len, seed=2)
        with nvtx.range(f"measured:{name}"):
            t0 = time.perf_counter()
            outs = llm.generate(prompts, params, use_tqdm=False)
            dt = time.perf_counter() - t0
        out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"RESULT config={args.config} workload={name} requests={n} prompt={prompt_len} output={out_len}: "
              f"{dt * 1e3:.1f} ms, {out_tokens / dt:.0f} output tok/s, {n * prompt_len / dt:.0f} prompt tok/s, "
              f"{dt * 1e3 / out_len:.2f} ms per output step", flush=True)
    llm.stop_profile()  # worker calls cudaProfilerStop(): flushes nsys' CUDA trace buffers


if __name__ == "__main__":
    main()
