"""Summarize CUDA stream usage from an Nsight Systems report (single- or multi-process).

Answers three different questions that "how many streams" can mean:
  1. created  - streams each CUDA context knows about (TARGET_INFO_CUDA_STREAM) and the API calls that made them
  2. active   - streams that received any GPU work (kernels, copies, memsets, event records) during the run
  3. busy     - how GPU time inside each NVTX "measured*" range splits across streams

GPU work is attributed to a window / phase through the runtime API call that launched it (correlationId,
which is only unique per process), because the GPU executes work later than the host enqueues it.
Optional NVTX ranges refine the report:
  "prefill" / "decode"           phases (Step B harness)
  "decode" / "schedule: update_after_schedule"   one per model step (Step B / vLLM with
                                                  VLLM_NVTX_SCOPES_FOR_PROFILING=1)

Usage: python nsys_streams.py report.nsys-rep|report.sqlite
"""

import bisect
import os
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict

PHASES = ["prefill", "decode"]
STEP_RANGES = ["decode", "schedule: update_after_schedule"]


def pid_of(global_id: int) -> int:
    """nsys global ids pack hw/vm id, pid and tid; the pid is bits 24..47."""
    return (global_id >> 24) & 0xFFFFFF


def open_db(path: str) -> sqlite3.Connection:
    if path.endswith(".nsys-rep"):
        out = path[: -len(".nsys-rep")] + ".sqlite"
        if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(path):
            subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "-o", out, path],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        path = out
    db = sqlite3.connect(path)
    if not has_table(db, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        sys.exit(f"{path}: no CUDA API trace. Was the process doing GPU work traced? "
                 "(forked children need nsys --trace-fork-before-exec=true)")
    db.execute("create index if not exists idx_runtime_corr on CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)")
    return db


def has_table(db, name):
    return db.execute("select 1 from sqlite_master where type='table' and name=?", (name,)).fetchone() is not None


def nvtx_ranges(db, pattern):
    """(start, end, name) of closed NVTX ranges whose name matches a SQL LIKE pattern."""
    return sorted(db.execute(
        "select n.start, n.end, coalesce(n.text, s.value) as name from NVTX_EVENTS n "
        "left join StringIds s on s.id = n.textId where n.end is not null and name like ?", (pattern,)).fetchall())


class Phases:
    """Find which named NVTX range (if any) contains a host timestamp."""

    def __init__(self, ranges):
        self.items = sorted(ranges)
        self.starts = [s for s, _, _ in self.items]

    def of(self, t):
        i = bisect.bisect_right(self.starts, t) - 1
        if i >= 0 and self.items[i][0] <= t <= self.items[i][1]:
            return self.items[i][2]
        return None


def gpu_activities(db):
    """(kind, pid, streamId, start, end, api_start, in_graph) for kernels, memcpys and memsets."""
    acts = []
    sources = [("kernel", "CUPTI_ACTIVITY_KIND_KERNEL", "a.graphNodeId is not null"),
               ("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY", "a.graphNodeId is not null"),
               ("memset", "CUPTI_ACTIVITY_KIND_MEMSET", "0")]
    for kind, table, graph_expr in sources:
        if not has_table(db, table):
            continue
        for gpid, *row in db.execute(
                f"select a.globalPid, a.streamId, a.start, a.end, r.start, {graph_expr} from {table} a "
                f"join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId = a.correlationId "
                f"and (r.globalTid >> 24) = (a.globalPid >> 24)"):
            acts.append((kind, pid_of(gpid), *row))
    return acts


def memcpy_details(db, w_start, w_end):
    """Counter of (pid, streamId, direction, src->dst memory kind, bytes) for copies launched in the window."""
    if not has_table(db, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        return Counter()
    op = dict(db.execute("select id, label from ENUM_CUDA_MEMCPY_OPER"))
    mem = dict(db.execute("select id, label from ENUM_CUDA_MEM_KIND"))
    rows = db.execute(
        "select a.globalPid, a.streamId, a.copyKind, a.srcKind, a.dstKind, a.bytes from CUPTI_ACTIVITY_KIND_MEMCPY a "
        "join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId = a.correlationId "
        "and (r.globalTid >> 24) = (a.globalPid >> 24) where r.start between ? and ?", (w_start, w_end))
    return Counter((pid_of(g), sid, op.get(k, k), f"{mem.get(s, s)}->{mem.get(d, d)}", b)
                   for g, sid, k, s, d, b in rows)


def union_ns(intervals):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s if cur_e is not None else 0)


def report_created(db, acts):
    """Per process with a CUDA context: streams created, by priority, creation calls, streams with any work."""
    names = dict(db.execute("select pid, name from PROCESSES")) if has_table(db, "PROCESSES") else {}
    null_streams = dict(db.execute("select processId, nullStreamId from TARGET_INFO_CUDA_CONTEXT_INFO"))
    prio = {}
    created = defaultdict(list)
    for pid, sid, p in db.execute("select processId, streamId, priority from TARGET_INFO_CUDA_STREAM"):
        p = p - (1 << 32) if p >= 1 << 31 else p  # stored as uint32; CUDA priorities are small negative ints
        prio[pid, sid] = p
        created[pid].append(p)
    creates = defaultdict(Counter)
    for gtid, name in db.execute(
            "select r.globalTid, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id = r.nameId "
            "where s.value like '%StreamCreate%' or s.value like '%GraphInstantiate%'"):
        creates[pid_of(gtid)][name] += 1
    active = defaultdict(set)
    for a in acts:
        active[a[1]].add(a[2])
    if has_table(db, "CUPTI_ACTIVITY_KIND_CUDA_EVENT"):
        for gpid, sid in db.execute("select globalPid, streamId from CUPTI_ACTIVITY_KIND_CUDA_EVENT"):
            active[pid_of(gpid)].add(sid)
    for pid in sorted(created):
        print(f"[process {pid} {names.get(pid, '')}] null stream id = {null_streams.get(pid)}")
        print(f"  [created] {len(created[pid])} streams, by priority: {dict(sorted(Counter(created[pid]).items()))}; "
              f"stream-create / graph-instantiate calls: {dict(creates[pid]) or 'none traced'}")
        print(f"  [active]  {len(active[pid])} streams received work during the whole run: {sorted(active[pid])}")
    return prio, null_streams


def report_window(db, name, w_start, w_end, acts, prio, null_streams):
    inside = [a for a in acts if w_start <= a[5] <= w_end]
    wall = w_end - w_start
    busy = union_ns([(a[3], a[4]) for a in inside])
    step_ranges = [r for n in STEP_RANGES for r in nvtx_ranges(db, n) if w_start <= r[0] <= w_end]
    steps = len(step_ranges)
    multi = len({a[1] for a in inside}) > 1
    print(f"\n[{name}] wall {wall / 1e6:.1f} ms, GPU busy {busy / 1e6:.1f} ms ({100 * busy / wall:.0f}%), "
          f"{len(inside)} GPU ops, {steps} model steps")

    phase_ranges = [r for p in PHASES for r in nvtx_ranges(db, p) if w_start <= r[0] <= w_end]
    phases = Phases(phase_ranges)
    groups = defaultdict(list)
    for a in inside:
        groups[phases.of(a[5]) if phase_ranges else "all"].append(a)

    for phase in PHASES + ["all"]:
        rows = groups.get(phase)
        if not rows:
            continue
        per = defaultdict(Counter)
        for kind, pid, sid, s, e, _, in_graph in rows:
            c = per[pid, sid]
            c[kind] += 1
            c["graph"] += bool(in_graph) and kind == "kernel"
            c["ns"] += e - s
        total_ns = sum(c["ns"] for c in per.values()) or 1
        kernels = sum(c["kernel"] for c in per.values())
        n_steps = len([r for r in phase_ranges if r[2] == "decode"]) if phase == "decode" else \
            (1 if phase == "prefill" else steps)
        line = f"\n  phase={phase}: {len(per)} stream(s) with work"
        if n_steps:
            line += f", {kernels / n_steps:.0f} kernels/step, GPU kernel time {total_ns / n_steps / 1e6:.2f} ms/step"
        print(line)
        if phase in ("decode", "all") and step_ranges:
            host = sum(e - s for s, e, _ in step_ranges) / len(step_ranges)
            wall_step = f"wall {wall / steps / 1e6:.2f} ms/step; " if phase == "all" else ""
            print(f"  {wall_step}host time inside NVTX '{step_ranges[0][2]}' {host / 1e6:.2f} ms/step")
        pid_col = f"{'pid':>8} " if multi else ""
        print(f"  {pid_col}{'stream':>8} {'prio':>5} {'kernels':>8} {'in-graph':>9} {'memcpy':>7} {'memset':>7} "
              f"{'GPU ms':>9} {'share':>6}")
        for (pid, sid), c in sorted(per.items(), key=lambda kv: -kv[1]["ns"]):
            tag = " (null)" if sid == null_streams.get(pid) else ""
            pid_val = f"{pid:>8} " if multi else ""
            print(f"  {pid_val}{sid:>8} {prio.get((pid, sid), '?'):>5} {c['kernel']:>8} {c['graph']:>9} "
                  f"{c['memcpy']:>7} {c['memset']:>7} {c['ns'] / 1e6:>9.2f} {100 * c['ns'] / total_ns:>5.1f}%{tag}")

    copies = memcpy_details(db, w_start, w_end)
    if copies:
        print("\n  memcpys (pid, stream, direction, memory kinds, bytes): count")
        for (pid, sid, kind, mem, nbytes), n in sorted(copies.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {pid:>8} {sid:>6} {kind:>18} {mem:>20} {nbytes:>10} B: {n}")


def main(path):
    db = open_db(path)
    acts = gpu_activities(db)
    prio, null_streams = report_created(db, acts)
    windows = nvtx_ranges(db, "measured%")
    if not windows:
        print("no NVTX 'measured*' range found")
    for s, e, name in windows:
        report_window(db, name, s, e, acts, prio, null_streams)


if __name__ == "__main__":
    main(sys.argv[1])
