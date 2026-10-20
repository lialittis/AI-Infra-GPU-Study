"""Compare greedy outputs of each Step D run against the baseline reference pass, token by token.

With temperature 0, speculative decoding should not change the text. In bf16 it can still differ, because the
verify pass runs the target model on k+1 tokens per request (different kernel shapes, different rounding), and so
does a different batch size without any speculation. The reference is the baseline's logprobs=2 pass. For each
mismatch this prints the position and the reference's top-2 logprob gap there: a gap at the resolution of bf16
logits (0, 0.125, 0.25) is a near-tie flipped by rounding; a large gap would be a real error.

Usage: python spec_compare.py [build_dir]   (reads spec-tokens-<config>.json written by spec.py)
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
    if i < len(top2) and len(top2[i]) == 2:
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
        lines.append(f"    request {r}: {describe(i, a, b, top2s[r])}")
        if i < len(a) and i < len(b) and len(top2s[r][i]) == 2:
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


def main(build):
    files = {os.path.basename(p)[len("spec-tokens-"):-len(".json")]: p
             for p in glob.glob(os.path.join(build, "spec-tokens-*.json"))}
    if "baseline" not in files:
        sys.exit(f"no spec-tokens-baseline.json in {build}")
    data = {c: json.load(open(p)) for c, p in files.items()}
    base = data["baseline"]

    print("== controls (no speculation)")
    for kind in ("repeat", "story"):
        b1, b32 = base.get(f"{kind}-b1"), base.get(f"{kind}-b32")
        if b1 and b32:
            compare(f"{kind}: reference b1 vs the same prompt in reference b32",
                    b1["ref_tokens"], b1["ref_top2"], b32["ref_tokens"][:1])

    for config in ["baseline"] + sorted(c for c in data if c != "baseline"):
        label = "baseline measured pass (no logprobs)" if config == "baseline" else config
        print(f"\n== {label} vs reference")
        for name, entry in data[config].items():
            if name in base:
                compare(name, base[name]["ref_tokens"], base[name]["ref_top2"], entry["tokens"])
            if name.startswith("json"):
                check_json(entry)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build")
