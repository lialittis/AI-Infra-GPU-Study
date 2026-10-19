"""Run the reproduction cases, each in a fresh process so a corrupted CUDA context cannot leak into the next one.

  python run.py                 run every case
  python run.py pinned pool     run cases whose name contains any of the given substrings
  python run.py --list          list the cases
  python run.py --one NAME      (internal) run one case in this process and print its result as JSON
"""

import argparse
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

from harness import CASES, same

HERE = Path(__file__).resolve().parent
RESULT_TAG = "@@RESULT "


def load_cases():
    for path in sorted(HERE.glob("s[0-9]_*.py")):
        importlib.import_module(path.stem)


def run_in_process(name):
    c = CASES[name]
    try:
        out = c.fn()
        res = {"status": "ok", "observed": out.observed, "expected": out.expected, "page": out.page,
               "correct": same(out.observed, out.expected),
               "page_match": None if out.page is None else same(out.observed, out.page)}
    except Exception as e:  # an exception is an outcome too: some "silent" bugs fail loudly instead
        msg = str(e).strip().splitlines()[0] if str(e).strip() else ""
        res = {"status": "error", "error": f"{type(e).__name__}: {msg}"[:300]}
    print(RESULT_TAG + json.dumps(res, default=str), flush=True)


def verdict(kind, res):
    if res["status"] == "crash":
        return "CRASH"
    if res["status"] == "error":
        return "ERROR"
    if kind == "bug":
        return "NOT REPRODUCED" if res["correct"] else "REPRODUCED"
    return "CORRECT" if res["correct"] else "WRONG"


def run_child(name, timeout):
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, str(HERE / "run.py"), "--one", name], capture_output=True, text=True,
                           timeout=timeout, cwd=HERE)
        stdout, stderr, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        stdout, stderr, rc = e.stdout or "", f"timeout after {timeout}s", None
        stdout = stdout.decode() if isinstance(stdout, bytes) else stdout
    res = None
    for line in stdout.splitlines():
        if line.startswith(RESULT_TAG):
            res = json.loads(line[len(RESULT_TAG):])
        else:
            print("    " + line)
    if res is None:
        tail = [l for l in (stderr or "").strip().splitlines() if l.strip()][-3:]
        res = {"status": "crash", "error": f"exit code {rc}: " + " | ".join(tail)}
    return res, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="*")
    ap.add_argument("--one")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    load_cases()
    if args.one:
        return run_in_process(args.one)
    names = [n for n in CASES if not args.patterns or any(p in n for p in args.patterns)]
    if args.list:
        for n in names:
            print(f"{CASES[n].kind:4} {n}")
        return

    rows = []
    for n in names:
        c = CASES[n]
        print(f"=== {n} ({c.kind}, {c.section}) ===", flush=True)
        res, dt = run_child(n, args.timeout)
        v = verdict(c.kind, res)
        if res["status"] in ("error", "crash"):
            print(f"  -> {v}: {res['error']}")
        else:
            print(f"  -> {v}: observed {res['observed']}, expected {res['expected']}"
                  + ("" if res["page"] is None else f", page shows {res['page']}"))
        print(flush=True)
        page = {None: "-", True: "same", False: "differs"}[res.get("page_match")]
        rows.append((n, c.kind, v, page, dt))

    w = max(len(r[0]) for r in rows)
    print(f"{'case':{w}}  kind  {'verdict':14}  vs page  time")
    for n, kind, v, page, dt in rows:
        print(f"{n:{w}}  {kind:4}  {v:14}  {page:7}  {dt:4.1f}s")
    expected = {"bug": "REPRODUCED", "fix": "CORRECT"}
    surprises = [r for r in rows if r[2] != expected[r[1]]]
    print(f"\n{len(rows)} cases, {len(rows) - len(surprises)} as the page describes, "
          f"{len(surprises)} different: {', '.join(r[0] for r in surprises) or 'none'}")


if __name__ == "__main__":
    main()
