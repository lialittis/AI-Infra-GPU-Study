"""Compare greedy outputs of each config against a reference config, token by token (Steps D and E).

With temperature 0, a feature that should not change the math (speculative decoding, KV offload) should not change
the text. In bf16 it can still differ: different kernel shapes round differently, as does a different batch size
with no feature at all. The reference is the reference config's logprobs=2 pass where the workload has one,
otherwise its measured pass. For each mismatch this prints the position and, where available, the reference's
top-2 logprob gap there: a gap at the resolution of bf16 logits (0, 0.125, 0.25) is a near-tie flipped by
rounding; a large gap would be a real error.

Usage: python compare_tokens.py <build_dir> <prefix> <reference_config>
  e.g. compare_tokens.py build spec baseline       (reads build/spec-tokens-<config>.json from spec.py)
       compare_tokens.py build offload prefix-big  (reads build/offload-tokens-<config>.json from offload.py)
"""

import glob
import json
import os
import sys
from collections import Counter


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def describe(i, ref, other, top2):
    """One mismatch: position, tokens on each side, and the reference top-2 gap at that position."""
    if i >= len(ref) or i >= len(other):
        return f"pos {i}: one side stopped early (lengths {len(ref)} vs {len(other)})"
    msg = f"pos {i}: reference token {ref[i]}, other {other[i]}"
    if top2 is not None and i < len(top2) and len(top2[i]) == 2:
        (t1, lp1), (t2, lp2) = top2[i]
        msg += f"; reference top-2 logprob gap {lp1 - lp2:.4f}"
        msg += " (other picked the runner-up)" if other[i] == t2 else ""
    return msg


def compare(label, refs, top2s, others, show=4):
    same, gaps, lines = 0, [], []
    for r, (a, b) in enumerate(zip(refs, others)):
        i = first_diff(a, b)
        if i is None:
            same += 1
            continue
        top2 = top2s[r] if top2s else None
        lines.append(f"    request {r}: {describe(i, a, b, top2)}")
        if top2 is not None and i < len(a) and i < len(b) and len(top2[i]) == 2:
            gaps.append(round(top2s[r][i][0][1] - top2s[r][i][1][1], 4))
    summary = f"  {label}: {same}/{len(refs)} identical"
    if gaps:
        summary += f"; top-2 gap at first mismatch: {dict(sorted(Counter(gaps).items()))}"
    print(summary)
    for line in lines[:show]:
        print(line)
    if len(lines) > show:
        print(f"    ... {len(lines) - show} more")


def check_json(entry):
    finished = [t for t, f in zip(entry["text"], entry["finish"]) if f == "stop"]
    valid = 0
    for text in finished:
        try:
            json.loads(text)
            valid += 1
        except ValueError:
            pass
    print(f"    JSON: {valid}/{len(finished)} finished outputs parse; "
          f"{len(entry['text']) - len(finished)} hit max_tokens")


def main(build, prefix, reference):
    head = f"{prefix}-tokens-"
    files = {os.path.basename(p)[len(head):-len(".json")]: p for p in glob.glob(os.path.join(build, head + "*.json"))}
    if reference not in files:
        sys.exit(f"no {head}{reference}.json in {build}")
    data = {c: json.load(open(p)) for c, p in files.items()}
    base = data[reference]

    # Step D: the same prompt alone and inside a batch of 32, both from the reference pass
    controls = [(k, base[f"{k}-b1"], base[f"{k}-b32"]) for k in ("repeat", "story")
                if f"{k}-b1" in base and f"{k}-b32" in base]
    if controls:
        print(f"== controls ({reference}, batch shape only)")
        for kind, b1, b32 in controls:
            compare(f"{kind}: reference b1 vs the same prompt in reference b32",
                    b1["ref_tokens"], b1["ref_top2"], b32["ref_tokens"][:1])

    for config in [reference] + sorted(c for c in data if c != reference):
        print(f"\n== {config}{' measured pass (no logprobs)' if config == reference else ''} vs reference")
        for name, entry in data[config].items():
            if name in base:
                if config == reference and "ref_tokens" not in entry:
                    continue  # this pass is the reference itself
                ref = base[name]
                compare(name, ref.get("ref_tokens", ref["tokens"]), ref.get("ref_top2"), entry["tokens"])
            if name.startswith("json"):
                check_json(entry)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
