"""Summarize CUDA stream usage from an Nsight Systems report.

Answers three different questions that "how many streams" can mean:
  1. created  - streams the CUDA context knows about (TARGET_INFO_CUDA_STREAM) and the API calls that made them
  2. active   - streams that received any GPU work (kernels, copies, memsets, event records)
  3. busy     - how GPU time inside the NVTX "measured" range splits across streams, per phase

GPU work is attributed to a phase (NVTX "prefill" / "decode" range) through the runtime API call that
launched it (correlationId), because the GPU executes it later than the host enqueued it.

Usage: python nsys_streams.py report.nsys-rep|report.sqlite
"""

import bisect
import os
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict


def open_db(path: str) -> sqlite3.Connection:
    if path.endswith(".nsys-rep"):
        out = path[: -len(".nsys-rep")] + ".sqlite"
        if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(path):
            subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "-o", out, path],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        path = out
    return sqlite3.connect(path)


def has_table(db, name):
    return db.execute("select 1 from sqlite_master where type='table' and name=?", (name,)).fetchone() is not None


def nvtx_ranges(db, name):
    return sorted(db.execute(
        "select n.start, n.end from NVTX_EVENTS n left join StringIds s on s.id = n.textId "
        "where coalesce(n.text, s.value) = ? and n.end is not null", (name,)).fetchall())


class Phases:
    """Find which named NVTX range (if any) contains a host timestamp."""

    def __init__(self, db, names):
        self.items = sorted((s, e, n) for n in names for s, e in nvtx_ranges(db, n))
        self.starts = [s for s, _, _ in self.items]

    def of(self, t):
        i = bisect.bisect_right(self.starts, t) - 1
        if i >= 0 and self.items[i][0] <= t <= self.items[i][1]:
            return self.items[i][2]
        return None


def gpu_activities(db):
    """(kind, streamId, start, end, api_start, in_graph) for kernels, memcpys and memsets."""
    acts = []
    sources = [("kernel", "CUPTI_ACTIVITY_KIND_KERNEL", "a.graphNodeId is not null"),
               ("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY", "a.graphNodeId is not null"),
               ("memset", "CUPTI_ACTIVITY_KIND_MEMSET", "0")]
    for kind, table, graph_expr in sources:
        if not has_table(db, table):
            continue
        for row in db.execute(
                f"select a.streamId, a.start, a.end, r.start, {graph_expr} from {table} a "
                f"join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId = a.correlationId"):
            acts.append((kind, *row))
    return acts


def memcpy_details(db, m_start, m_end):
    """Counter of (streamId, direction, src->dst memory kind, bytes) for copies launched in [m_start, m_end]."""
    if not has_table(db, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        return Counter()
    op = dict(db.execute("select id, label from ENUM_CUDA_MEMCPY_OPER"))
    mem = dict(db.execute("select id, label from ENUM_CUDA_MEM_KIND"))
    rows = db.execute(
        "select a.streamId, a.copyKind, a.srcKind, a.dstKind, a.bytes from CUPTI_ACTIVITY_KIND_MEMCPY a "
        "join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId = a.correlationId where r.start between ? and ?",
        (m_start, m_end))
    return Counter((sid, op.get(k, k), f"{mem.get(s, s)}->{mem.get(d, d)}", b) for sid, k, s, d, b in rows)


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


def main(path):
    db = open_db(path)
    strings = dict(db.execute("select id, value from StringIds"))

    # 1. Streams created.
    null_stream = db.execute("select nullStreamId from TARGET_INFO_CUDA_CONTEXT_INFO").fetchone()[0]
    # priority is stored as uint32; CUDA priorities are small negative ints (lower = higher priority).
    streams = [(sid, p - (1 << 32) if p >= 1 << 31 else p, flag)
               for sid, p, flag in db.execute("select streamId, priority, flag from TARGET_INFO_CUDA_STREAM")]
    prio = {sid: p for sid, p, _ in streams}
    print(f"[created] {len(streams)} streams known to the context (legacy null stream id = {null_stream})")
    print("  by priority:", dict(sorted(Counter(p for _, p, _ in streams).items())),
          " by flag:", dict(sorted(Counter(f for _, _, f in streams).items())))
    creates = Counter(strings[n] for (n,) in db.execute(
        "select nameId from CUPTI_ACTIVITY_KIND_RUNTIME where nameId in "
        "(select id from StringIds where value like '%StreamCreate%')"))
    print("  creation API calls:", dict(creates) or "none traced")

    # 2. Streams with any work over the whole run (incl. warmup / compilation).
    acts = gpu_activities(db)
    events = Counter(s for (s,) in db.execute("select streamId from CUPTI_ACTIVITY_KIND_CUDA_EVENT")) \
        if has_table(db, "CUPTI_ACTIVITY_KIND_CUDA_EVENT") else Counter()
    active_all = {a[1] for a in acts} | set(events)
    print(f"[active] {len(active_all)} streams received work during the whole run: {sorted(active_all)}")

    # 3. The measured generation only.
    measured = nvtx_ranges(db, "measured")
    if not measured:
        print("no NVTX 'measured' range found")
        return
    m_start, m_end = measured[-1]
    phases = Phases(db, ["prefill", "decode"])
    inside = [a for a in acts if m_start <= a[4] <= m_end]
    decode_ranges = [r for r in nvtx_ranges(db, "decode") if m_start <= r[0] <= m_end]
    n_decode = len(decode_ranges)

    busy = union_ns([(a[2], a[3]) for a in inside])
    wall = m_end - m_start
    print(f"[measured] wall {wall / 1e6:.1f} ms, GPU busy {busy / 1e6:.1f} ms ({100 * busy / wall:.0f}%), "
          f"{len(inside)} GPU ops, {n_decode} decode steps")

    for phase in ["prefill", "decode"]:
        rows = [a for a in inside if phases.of(a[4]) == phase]
        if not rows:
            continue
        per = defaultdict(lambda: Counter())
        for kind, sid, s, e, _, in_graph in rows:
            per[sid][kind] += 1
            per[sid]["graph"] += bool(in_graph) and kind == "kernel"
            per[sid]["ns"] += e - s
        total_ns = sum(c["ns"] for c in per.values())
        steps = n_decode if phase == "decode" else 1
        print(f"\n  phase={phase}: {len(per)} stream(s) with work, "
              f"{sum(c['kernel'] for c in per.values()) / steps:.0f} kernels per {'step' if steps > 1 else 'pass'}")
        if phase == "decode":
            host = sum(e - s for s, e in decode_ranges) / n_decode
            print(f"  per step: host (NVTX) {host / 1e6:.2f} ms, GPU kernel time {total_ns / n_decode / 1e6:.2f} ms")
        print(f"  {'stream':>8} {'prio':>5} {'kernels':>8} {'in-graph':>9} {'memcpy':>7} {'memset':>7} "
              f"{'GPU ms':>8} {'share':>6}")
        for sid, c in sorted(per.items(), key=lambda kv: -kv[1]["ns"]):
            tag = " (null)" if sid == null_stream else ""
            print(f"  {sid:>8} {prio.get(sid, '?'):>5} {c['kernel']:>8} {c['graph']:>9} {c['memcpy']:>7} "
                  f"{c['memset']:>7} {c['ns'] / 1e6:>8.2f} {100 * c['ns'] / total_ns:>5.1f}%{tag}")

    copies = memcpy_details(db, m_start, m_end)
    if copies:
        print("\n  memcpys in measured range (stream, direction, memory kinds, bytes): count")
        for (sid, kind, mem, nbytes), n in sorted(copies.items(), key=lambda kv: -kv[1]):
            print(f"  {sid:>8} {kind:>18} {mem:>20} {nbytes:>10} B: {n}")


if __name__ == "__main__":
    main(sys.argv[1])
