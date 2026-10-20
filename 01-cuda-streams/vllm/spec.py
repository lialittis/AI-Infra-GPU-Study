"""Step D: speculative decoding in vLLM, offline, with each workload inside an NVTX "measured:<name>" range.

Configs:
  baseline   vLLM defaults (torch.compile + CUDA graphs + async scheduling), no speculation
  ngram-k4   prompt-lookup drafting on the GPU (method "ngram_gpu"), 4 draft tokens per step, no draft model
  draft-k2/4/8  a small draft model from the same family (default Qwen2.5-0.5B-Instruct), 2/4/8 draft tokens

Workloads use real text through the chat template, because acceptance depends on the content:
  repeat-*   copy a passage verbatim: the output is in the prompt, so prompt lookup and the draft model both hit
  story-*    write a short story: little overlap with the prompt
  json-b1    the story prompt under a JSON schema; structured output is the only case where vLLM's
             DraftTokensHandler copies draft tokens back to the host on its own stream

The b1 prompt is the first b32 prompt, so b1 vs b32 shows how much greedy output depends on batch shape alone.
Generated token ids go to build/spec-tokens-<config>.json for compare_tokens.py. The baseline also runs every workload
once more with logprobs=2, outside the measured range. That pass is the reference: it has the top-2 logprobs at
every position of its own output, which tells near-ties from real errors where another run diverges from it.

Profiling and flushing work as in bench.py (Step C).
"""

import argparse
import itertools
import json
import os
import time

import torch.cuda.nvtx as nvtx
from vllm import LLM, SamplingParams
from vllm.config import ProfilerConfig
from vllm.sampling_params import StructuredOutputsParams

# Original filler text, so the harness needs no dataset download.
PARAGRAPHS = [
    "A graphics processor runs thousands of threads at once, but it is fed by a single host thread that writes "
    "commands into a queue. When each command does very little work, the host becomes the bottleneck and the "
    "device waits. Recording the commands once and replaying them later removes most of that cost.",
    "Large language models generate text one token at a time. Every new token requires reading all of the model "
    "weights from memory, so at small batch sizes the speed is set by memory bandwidth rather than by arithmetic. "
    "Doing more useful work per weight read is the main way to go faster.",
    "Speculative decoding guesses several future tokens cheaply and then checks all of the guesses with one pass "
    "of the large model. Correct guesses are kept, the first wrong guess is replaced by the large model's own "
    "choice, and the rest are thrown away. With greedy sampling the final text should not change.",
    "A key value cache stores the attention keys and values of every token seen so far. It grows with the length "
    "of the conversation and the number of requests, and it often uses more memory than the weights. Moving rarely "
    "used blocks to host memory frees room on the device.",
    "Copy engines move data between host and device memory while the processors keep computing. A transfer only "
    "overlaps with computation when it is queued on a different stream and the host does not block while issuing "
    "it. Small transfers mostly pay for latency, not for bandwidth.",
    "A scheduler decides which requests run in the next step. It must account for memory, for fairness between "
    "users, and for requests that arrive while others are halfway done. When the scheduler runs on the host while "
    "the device executes the previous step, its cost is hidden.",
]
TOPICS = [
    "a lighthouse keeper who finds a map", "a robot learning to paint", "two rivals stuck in an elevator",
    "a village where it never stops snowing", "a detective who is afraid of the dark", "a cat that runs a bakery",
    "the last train of the night", "a scientist who shrinks herself", "a pirate who hates the sea",
    "a library that rearranges itself", "a child who can talk to clocks", "an astronaut's first day home",
    "a garden that grows overnight", "a chess match against a ghost", "a musician who lost her voice",
    "a forgotten city under the sand", "a dragon afraid of heights", "a postman delivering letters to the moon",
    "a mountain climber and a stranger", "a mirror that shows tomorrow", "a farmer and a talking crow",
    "a submarine crew hearing a song", "a painter whose pictures move", "a king who swaps places with a cook",
    "a storm that carries away a house", "a twin who switches lives", "a tea shop open only at midnight",
    "an inventor and a broken compass", "a whale that follows a ship", "a soldier walking home after the war",
    "a girl who collects shadows", "an old man teaching a fox to read",
]
JSON_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "characters": {"type": "array", "items": {"type": "string"}},
                   "story": {"type": "string"}},
    "required": ["title", "characters", "story"],
}

DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONFIGS = {
    "baseline": {},
    "ngram-k4": {"method": "ngram_gpu", "num_speculative_tokens": 4},
    "draft-k2": {"method": "draft_model", "model": DRAFT_MODEL, "num_speculative_tokens": 2},
    "draft-k4": {"method": "draft_model", "model": DRAFT_MODEL, "num_speculative_tokens": 4},
    "draft-k8": {"method": "draft_model", "model": DRAFT_MODEL, "num_speculative_tokens": 8},
}
# name: (kind, requests)
WORKLOADS = {
    "repeat-b1": ("repeat", 1),
    "repeat-b32": ("repeat", 32),
    "story-b1": ("story", 1),
    "story-b32": ("story", 32),
    "json-b1": ("json", 1),
}
MAX_TOKENS = {"repeat": 256, "story": 256, "json": 512}  # json: room to close the object


def passages(offset: int) -> list[str]:
    """Distinct three-paragraph passages; offset 0 for measured runs, another offset for warmup."""
    combos = list(itertools.permutations(range(len(PARAGRAPHS)), 3))
    return ["\n\n".join(PARAGRAPHS[i] for i in combos[(offset + 7 * k) % len(combos)]) for k in range(32)]


def user_messages(kind: str, n: int, warmup: bool) -> list[str]:
    if kind == "repeat":
        return [f"Repeat the following text exactly, with no other words:\n\n{p}" for p in passages(50 if warmup else 0)[:n]]
    topics = TOPICS[::-1] if warmup else TOPICS
    suffix = " Answer as JSON." if kind == "json" else ""
    return [f"Write a short story about {t}.{suffix}" for t in topics[:n]]


def spec_counters(llm) -> dict[str, float]:
    totals = {}
    for m in llm.get_metrics():
        if m.name.startswith("vllm:spec_decode_num_") and hasattr(m, "value"):
            totals[m.name] = totals.get(m.name, 0) + m.value
    return totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--config", choices=list(CONFIGS), default="baseline")
    ap.add_argument("--workloads", nargs="+", choices=list(WORKLOADS), default=list(WORKLOADS))
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.3,
                    help="fraction of device memory vLLM may claim (on unified-memory GPUs this is system RAM)")
    ap.add_argument("--out-dir", default="build")
    args = ap.parse_args()

    spec = CONFIGS[args.config]
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=2048,
              enable_prefix_caching=False, seed=0, profiler_config=ProfilerConfig(profiler="cuda"),
              disable_log_stats=False, speculative_config=spec or None)
    sched = llm.llm_engine.vllm_config.scheduler_config
    print(f"CONFIG {args.config}: speculative_config={llm.llm_engine.vllm_config.speculative_config} "
          f"async_scheduling={sched.async_scheduling}", flush=True)
    tok = llm.get_tokenizer()
    llm.start_profile()

    saved = {}
    for name in args.workloads:
        kind, n = WORKLOADS[name]
        params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS[kind])
        if kind == "json":
            params.structured_outputs = StructuredOutputsParams(json=JSON_SCHEMA)

        def prompts(warmup):
            return [tok.apply_chat_template([{"role": "user", "content": m}], tokenize=False,
                                            add_generation_prompt=True) for m in user_messages(kind, n, warmup)]

        llm.generate(prompts(warmup=True), params, use_tqdm=False)  # warm up this shape
        measured = prompts(warmup=False)
        before = spec_counters(llm)
        with nvtx.range(f"measured:{name}"):
            t0 = time.perf_counter()
            outs = llm.generate(measured, params, use_tqdm=False)
            dt = time.perf_counter() - t0
        after = spec_counters(llm)
        delta = {k.removeprefix("vllm:spec_decode_num_"): after[k] - before.get(k, 0) for k in after}

        out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        prompt_tokens = sum(len(o.prompt_token_ids) for o in outs)
        line = (f"RESULT config={args.config} workload={name} requests={n} prompt_tokens={prompt_tokens} "
                f"output_tokens={out_tokens}: {dt * 1e3:.1f} ms, {out_tokens / dt:.0f} output tok/s")
        if n == 1:
            line += f", {dt * 1e3 / out_tokens:.2f} ms per output token"
        if delta.get("draft_tokens"):
            line += (f", drafted {delta['draft_tokens']:.0f} accepted {delta['accepted_tokens']:.0f} "
                     f"({100 * delta['accepted_tokens'] / delta['draft_tokens']:.0f}%), "
                     f"{1 + delta['accepted_tokens'] / delta['drafts']:.2f} tokens per verify step")
        elif spec:
            line += ", no drafts proposed"
        print(line, flush=True)

        entry = {"tokens": [list(o.outputs[0].token_ids) for o in outs],
                 "text": [o.outputs[0].text for o in outs],
                 "finish": [o.outputs[0].finish_reason for o in outs]}
        if not spec:  # the reference pass, outside the measured range
            ref = llm.generate(measured, SamplingParams(temperature=0, max_tokens=MAX_TOKENS[kind], logprobs=2,
                                                        structured_outputs=params.structured_outputs),
                               use_tqdm=False)
            entry["ref_tokens"] = [list(o.outputs[0].token_ids) for o in ref]
            # per position: [[token_id, logprob], ...] for the two most likely tokens
            entry["ref_top2"] = [[sorted(([t, lp.logprob] for t, lp in step.items()), key=lambda x: -x[1])[:2]
                                  for step in o.outputs[0].logprobs] for o in ref]
        saved[name] = entry
    llm.stop_profile()  # worker calls cudaProfilerStop(): flushes nsys' CUDA trace buffers

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"spec-tokens-{args.config}.json"), "w") as f:
        json.dump(saved, f)


if __name__ == "__main__":
    main()
