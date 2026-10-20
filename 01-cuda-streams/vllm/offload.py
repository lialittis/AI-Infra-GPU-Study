"""Step E: KV-cache CPU offload in vLLM, offline, with each workload inside an NVTX "measured:<name>" range.

Configs (prefix caching on in all of them; Step C had it off):
  prefix          GPU KV cache fixed at SMALL_KV bytes, no offload: an evicted prefix is recomputed
  prefix-big      GPU KV cache of BIG_KV bytes, nothing is evicted: the upper bound for prefix hits
  offload-native  SMALL_KV + kv_offloading_size (vLLM's OffloadingConnector)
  offload-simple  the same with VLLM_USE_SIMPLE_KV_OFFLOAD=1 (SimpleCPUOffloadConnector)

Workloads, in this order, because they share one engine's caches:
  reuse-prefill  fill: 32 distinct 2048-token documents (3.5x the SMALL_KV cache), so the first ones are evicted
                 from the GPU (and stored to CPU when offloading); measured: documents 0-15 again, each with a new
                 32-token suffix, max_tokens=1 (prefill only)
  reuse-decode   the same with a fresh document set and 64 output tokens; outputs are compared across configs
  mixed          16 decode requests (512 -> 256) run alone for 32 tokens, then 16 refilled documents arrive. The
                 per-token gaps of the decode requests show how much the arrivals stall them
  store-decode / store-prefill   Step C's decode-b32 / prefill-b32 with unique prompts: no reuse, so offloading
                 can only add store traffic. Last, so they don't push the documents out of BIG_KV

Prompts are random token ids (exact lengths; content doesn't matter for caching). Generated token ids go to
build/offload-tokens-<config>.json for compare_tokens.py; prefix-big also runs reuse-decode once more with
logprobs=2 as the reference. Profiling and flushing work as in bench.py (Step C).
"""

import argparse
import json
import os
import random
import statistics
import time

KV_BYTES_PER_TOKEN = 28 * 2 * 2 * 128 * 2  # Qwen2.5-1.5B: layers x (K, V) x KV heads x head dim x bf16 = 28 KiB
SMALL_KV = 512 * 1024 ** 2  # ~18.7k tokens
BIG_KV = 8 * 1024 ** 3
OFFLOAD_GIB = 8

CONFIGS = {
    "prefix": ({"kv_cache_memory_bytes": SMALL_KV}, {}),
    "prefix-big": ({"kv_cache_memory_bytes": BIG_KV}, {}),
    "offload-native": ({"kv_cache_memory_bytes": SMALL_KV, "kv_offloading_size": OFFLOAD_GIB}, {}),
    "offload-simple": ({"kv_cache_memory_bytes": SMALL_KV, "kv_offloading_size": OFFLOAD_GIB},
                       {"VLLM_USE_SIMPLE_KV_OFFLOAD": "1"}),
}
WORKLOADS = ["reuse-prefill", "reuse-decode", "mixed", "store-decode", "store-prefill"]
DOCS, DOC_LEN, SUFFIX_LEN, REUSED = 32, 2048, 32, 16


def tokens(rng, n):
    return [rng.randrange(100, 20000) for _ in range(n)]


def documents(set_id):
    rng = random.Random(1000 + set_id)
    return [tokens(rng, DOC_LEN) for _ in range(DOCS)]


def offload_counters(llm):
    """Sum of every prefix-cache / offload counter (and histogram sums), by name."""
    totals = {}
    for m in llm.get_metrics():
        if "prefix_cache" in m.name or "offload" in m.name:
            v = getattr(m, "value", None)
            if v is None:
                v = getattr(m, "sum", None)
            if isinstance(v, (int, float)):
                totals[m.name] = totals.get(m.name, 0) + v
    return totals


def counter_delta(before, after):
    out = []
    for k in sorted(after):
        d = after[k] - before.get(k, 0)
        if d:
            short = k.removeprefix("vllm:")
            out.append(f"{short}={d:.4g}" if isinstance(d, float) and d != int(d) else f"{short}={int(d)}")
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--config", choices=list(CONFIGS), default="prefix")
    ap.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=WORKLOADS)
    ap.add_argument("--out-dir", default="build")
    args = ap.parse_args()

    engine_kwargs, env = CONFIGS[args.config]
    os.environ.update(env)  # before vLLM reads its env (EngineCore is forked and inherits it)
    import torch.cuda.nvtx as nvtx
    from vllm import LLM, SamplingParams
    from vllm.config import ProfilerConfig
    from vllm.inputs import TokensPrompt

    llm = LLM(model=args.model, max_model_len=4096, enable_prefix_caching=True, seed=0,
              profiler_config=ProfilerConfig(profiler="cuda"), disable_log_stats=False, **engine_kwargs)
    vc = llm.llm_engine.vllm_config
    print(f"CONFIG {args.config}: kv_cache_memory_bytes={vc.cache_config.kv_cache_memory_bytes} "
          f"kv_connector={vc.kv_transfer_config.kv_connector if vc.kv_transfer_config else None} "
          f"async_scheduling={vc.scheduler_config.async_scheduling}", flush=True)
    llm.start_profile()

    def generate(prompt_ids, max_tokens, logprobs=None):
        params = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True, logprobs=logprobs)
        return llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_ids], params, use_tqdm=False)

    def fill_and_resend(set_id):
        """Fill the cache with one document set; return the reused prompts (documents 0..REUSED-1 + suffix)."""
        docs = documents(set_id)
        generate(docs, 1)
        rng = random.Random(2000 + set_id)
        return [d + tokens(rng, SUFFIX_LEN) for d in docs[:REUSED]]

    def measured(name, fn):
        before = offload_counters(llm)
        with nvtx.range(f"measured:{name}"):
            t0 = time.perf_counter()
            result = fn()
            dt = time.perf_counter() - t0
        return result, dt, counter_delta(before, offload_counters(llm))

    # One small fill-and-reuse cycle first, so one-time costs (e.g. compiling the Triton load kernel) stay out of
    # the measured runs.
    generate(fill_and_resend(set_id=99)[:2], 1)

    saved = {}
    for set_id, name in enumerate(args.workloads):
        if name in ("reuse-prefill", "reuse-decode"):
            prompts = fill_and_resend(set_id)
            out_len = 1 if name == "reuse-prefill" else 64
            outs, dt, counters = measured(name, lambda: generate(prompts, out_len))
            print(f"RESULT config={args.config} workload={name} requests={len(prompts)} "
                  f"prompt={DOC_LEN + SUFFIX_LEN} output={out_len}: {dt * 1e3:.1f} ms | {counters}", flush=True)
            entry = {"tokens": [list(o.outputs[0].token_ids) for o in outs]}
            if name == "reuse-decode" and args.config == "prefix-big":  # the reference, outside the measured range
                ref = generate(prompts, out_len, logprobs=2)
                entry["ref_tokens"] = [list(o.outputs[0].token_ids) for o in ref]
                entry["ref_top2"] = [[sorted(([t, lp.logprob] for t, lp in step.items()), key=lambda x: -x[1])[:2]
                                      for step in o.outputs[0].logprobs] for o in ref]
            saved[name] = entry

        elif name == "mixed":
            reused = fill_and_resend(set_id)
            rng = random.Random(3000)
            decode_prompts = [tokens(rng, 512) for _ in range(16)]
            (gaps_before, gaps_after, outs), dt, counters = measured(
                name, lambda: run_mixed(llm, decode_prompts, reused, SamplingParams, TokensPrompt))
            after_window = gaps_after[:16 * 10]  # the first ~10 tokens of each decode request after arrival
            print(f"RESULT config={args.config} workload=mixed: {dt * 1e3:.1f} ms; decode token gap "
                  f"before arrival median {statistics.median(gaps_before):.1f} ms, "
                  f"after arrival median {statistics.median(after_window):.1f} / max {max(after_window):.1f} ms"
                  f" | {counters}", flush=True)
            saved[name] = {"tokens": outs}

        else:  # store-decode / store-prefill: unique prompts, nothing to reuse
            n, prompt_len, out_len = (32, 512, 128) if name == "store-decode" else (32, 1024, 1)
            rng = random.Random(4000 + set_id)
            generate([tokens(rng, prompt_len) for _ in range(n)], out_len)  # warm up this shape
            prompts = [tokens(rng, prompt_len) for _ in range(n)]
            outs, dt, counters = measured(name, lambda: generate(prompts, out_len))
            print(f"RESULT config={args.config} workload={name} requests={n} prompt={prompt_len} "
                  f"output={out_len}: {dt * 1e3:.1f} ms, {dt * 1e3 / out_len:.2f} ms per output step | {counters}",
                  flush=True)
    llm.stop_profile()  # worker calls cudaProfilerStop(): flushes nsys' CUDA trace buffers

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"offload-tokens-{args.config}.json"), "w") as f:
        json.dump(saved, f)


def run_mixed(llm, decode_prompts, reused, SamplingParams, TokensPrompt):
    """Drive the engine step by step: decode requests first, the reused documents after 32 decode tokens.
    Returns the decode requests' token gaps (ms) before and after the arrival, and all outputs' token ids."""
    engine = llm.llm_engine
    decode_params = SamplingParams(temperature=0, max_tokens=256, ignore_eos=True)
    doc_params = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
    for i, p in enumerate(decode_prompts):
        engine.add_request(f"decode-{i}", TokensPrompt(prompt_token_ids=p), decode_params)
    seen, last_t, arrived = {}, {}, False
    gaps_before, gaps_after, final = [], [], {}
    while engine.has_unfinished_requests():
        outputs = engine.step()
        now = time.perf_counter()
        for o in outputs:
            n = len(o.outputs[0].token_ids)
            if o.request_id.startswith("decode-") and n > seen.get(o.request_id, 0):
                if o.request_id in last_t and seen[o.request_id] >= 2:  # skip the prefill -> first token gap
                    (gaps_after if arrived else gaps_before).append((now - last_t[o.request_id]) * 1e3)
                seen[o.request_id], last_t[o.request_id] = n, now
            if o.finished:
                final[o.request_id] = list(o.outputs[0].token_ids)
        if not arrived and seen and min(seen.values()) >= 32:
            for i, p in enumerate(reused):
                engine.add_request(f"doc-{i}", TokensPrompt(prompt_token_ids=p), doc_params)
            arrived = True
    order = [f"decode-{i}" for i in range(len(decode_prompts))] + [f"doc-{i}" for i in range(len(reused))]
    return gaps_before, gaps_after, [final[r] for r in order]


if __name__ == "__main__":
    main()
