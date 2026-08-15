"""
trace_reader.py — CUPTI SQLite trace connection/reading (census stage 1)
========================================================================

Usage:
  1) python3 trace_reader.py <trace.sqlite>
     → runs schema introspection only (prints tables/columns/row counts/samples)
  2) Compare the output with MAPPING below and fix table/column names to match
     your schema
  3) python3 trace_reader.py <trace.sqlite> --run
     → runs the reports:
        [R1] settled-step detection  (per-step cudaMalloc count — find where the noise stops)
        [R2] collective census       (per-step AG/RS kernel counts — check against predicted 8/4)
        [R3] thread audit            (thread distribution of launch APIs — NVTX thread-local check)
        [INTERP] bare-launch block attribution (backward, autograd thread)
        [B0DIFF] b1 vs b0 bare kernel-name diff — resolve the −2 anomaly
        [BLEED] boundary bleed A/B
        [INTERP-F] main-thread bare-launch forward attribution

Design notes:
  * MAPPING is my guess. Your tracer schema is custom, so the first task is to
    check it against the introspection output and fix it. All queries go through
    MAPPING so that MAPPING is the only thing you ever need to edit.
  * Step boundaries: the [start, end] of NVTX 'step_N' ranges are used as
    wall-clock windows.
  * Membership is decided by the launch (API) timestamp, not the kernel's GPU
    execution timestamp — immune to async drift.
  * Aggregate with DISTINCT kernel ids only — runtime-path kernels have duplicate
    api rows for the same correlation, which once inflated JOIN aggregates;
    this prevents that bug from coming back.
  * The definition of "bare" is relative to each thread's baseline NVTX stack:
      - autograd thread: baseline = empty stack → bare = (active_nvtx_id = 0)
      - main thread:     baseline = [step_N]    → bare = (active_nvtx_id = step_N's rid)
    The same value 0 means different things depending on the thread
    (discovered 2026-08-05).
  * The steps tuple has 4 elements (idx, s, e, rid) — rid is the nvtx_id of the
    step range. Reports that don't need rid ignore it with `*_`.
  * The DB is opened read-only (URI mode=ro) — safe against training-side writes
    even with WAL.
"""

import argparse
import sqlite3
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# MAPPING — ★ compare with the introspection output and edit only this ★
# ---------------------------------------------------------------------------
MAPPING = {
    # GPU kernel activity records (reflects the actual schema)
    "kernels": {
        "table": "kernel_records",
        "id": "id",
        "name": "name",              # mangled (_ZN...) — nccl kernels look like ncclDevKernel_*
        "start": "start_ns",
        "end": "end_ns",
        "stream": "stream_id",
        "corr": "correlation_id",
    },
    # API records — note: RUNTIME/DRIVER duplicate rows exist for the same correlation
    "api": {
        "table": "api_records",
        "id": "id",
        "name": "func_name",
        "start": "enter_ns",
        "end": "exit_ns",
        "thread": "thread_id",
        "corr": "correlation_id",
        "domain": "domain",          # 'RUNTIME' | 'DRIVER'
        "nvtx": "active_nvtx_id",
    },
    # NVTX range records
    "nvtx": {
        "table": "nvtx_ranges",
        "rid": "nvtx_id",            # the key api.active_nvtx_id refers to (confirmed 2026-08-05)
        "name": "name",
        "start": "push_ns",
        "end": "pop_ns",
        "thread": "thread_id",
    },
    # MEMCPY activity
    "memcpy": {
        "table": "memcpy_records",
        "start": "start_ns",
        "end": "end_ns",
        "kind": "copy_kind",         # 'HtoD' | 'DtoH' | ...
        "corr": "correlation_id",
    },
}

STEP_RANGE_PREFIX = "step_"          # census_run.py's range_push(f"step_{i}")
AG_PATTERN = "%AllGather%"           # ncclDevKernel_AllGather...
RS_PATTERN = "%ReduceScatter%"
MALLOC_PATTERN = "%cudaMalloc%"
LAUNCH_PATTERN = "%aunchKernel%"     # cudaLaunchKernel / cuLaunchKernel / cuLaunchKernelEx
FREE_PATTERN   = "%cudaFree%"


def connect_ro(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# 0. Introspection — no query can be trusted without knowing the schema
# ---------------------------------------------------------------------------
def introspect(conn: sqlite3.Connection):

    print("=" * 72)
    print(" SCHEMA INTROSPECTION")
    print("=" * 72)

    tables = [r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]

    if not tables:
        print("  (no tables — check the path, and whether only a WAL file exists)")
        return

    for t in tables:

        cols = conn.execute(f"PRAGMA table_info({t})").fetchall()

        n = conn.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"]

        col_desc = ", ".join(f"{c['name']}:{c['type']}" for c in cols)

        print(f"\n[{t}]  rows={n}")
        print(f"  columns: {col_desc}")

        for row in conn.execute(f"SELECT * FROM {t} LIMIT 3"):
            vals = ", ".join(f"{k}={str(row[k])[:40]}" for k in row.keys())
            print(f"    sample: {vals}")

    print("\n→ Compare the output above with MAPPING, fix table/column names, then rerun with --run.")


# ---------------------------------------------------------------------------
# Load step windows — NVTX 'step_N' ranges as wall-clock windows
#   Returns: (idx, push_ns, pop_ns, rid) 4-tuples — rid is the step range's nvtx_id
#   (needed for main-thread bare detection: active_nvtx_id = rid is the
#    "outside any anchor" signature)
# ---------------------------------------------------------------------------
def load_step_windows(conn):

    m = MAPPING["nvtx"]

    q = (f"SELECT {m['rid']} AS rid, {m['name']} AS name, "
         f"{m['start']} AS s, {m['end']} AS e "
         f"FROM {m['table']} WHERE {m['name']} LIKE ? ORDER BY s")

    steps = []

    for r in conn.execute(q, (STEP_RANGE_PREFIX + "%",)):
        try:
            idx = int(str(r["name"]).split("_")[-1])
        except ValueError:
            continue
        steps.append((idx, r["s"], r["e"], r["rid"]))   # 4-tuple

    return steps


# ---------------------------------------------------------------------------
# R0. NCCL kernel-name survey — required before trusting the AG/RS LIKE patterns
# ---------------------------------------------------------------------------
def report_nccl_names(conn):
    k = MAPPING["kernels"]
    print("\n[R0] NCCL kernel-name survey (whole trace)")
    q = (f"SELECT {k['name']} AS n, COUNT(*) c FROM {k['table']} "
         f"WHERE {k['name']} LIKE '%nccl%' GROUP BY n ORDER BY c DESC")
    rows = conn.execute(q).fetchall()
    if not rows:
        print("  0 nccl kernels — check name patterns / tracing coverage!")
        return False
    ok = False
    for r in rows:
        print(f"  {r['c']:>6} × {r['n']}")
        if "AllGather" in r["n"] or "ReduceScatter" in r["n"]:
            ok = True
    if not ok:
        print("  ⚠ No AllGather/ReduceScatter in names (generic unified kernel suspected)")
        print("    → R2's name-based census is invalid — plan B needed.")
    return ok


# ---------------------------------------------------------------------------
# R1. Settled-step detection — per-step cudaMalloc count (silence = settled)
# ---------------------------------------------------------------------------
def report_settled(conn, steps):
    a = MAPPING["api"]
    print("\n[R1] Per-step cudaMalloc / cudaFree counts (both 0 = settled)")
    print(f"{'step':>6} {'cudaMalloc':>12} {'cudaFree':>10}")
    q = (f"SELECT COUNT(*) c FROM {a['table']} "
         f"WHERE {a['name']} LIKE ? AND {a['start']} BETWEEN ? AND ?")
    for idx, s, e, *_ in steps:
        m = conn.execute(q, (MALLOC_PATTERN, s, e)).fetchone()["c"]
        f = conn.execute(q, (FREE_PATTERN,   s, e)).fetchone()["c"]
        print(f"{idx:>6} {m:>12} {f:>10}")
    print("  ※ Early steps are expected to be large due to compile warmup.")


# ---------------------------------------------------------------------------
# R2. Collective census — launch-time membership, DISTINCT-kernel aggregation
# ---------------------------------------------------------------------------
def census_one_step(conn, s, e):

    k, a = MAPPING["kernels"], MAPPING["api"]

    q = (f"SELECT COUNT(DISTINCT k.{k['id']}) c "
         f"FROM {k['table']} k JOIN {a['table']} a "
         f"  ON k.{k['corr']} = a.{a['corr']} "
         f"WHERE k.{k['name']} LIKE ? AND a.{a['start']} BETWEEN ? AND ?")
    ag = conn.execute(q, (AG_PATTERN, s, e)).fetchone()["c"]
    rs = conn.execute(q, (RS_PATTERN, s, e)).fetchone()["c"]
    return ag, rs


def report_census(conn, steps, pick=None):
    print("\n[R2] Collective census (launch-time membership, DISTINCT kernels)")
    print(f"{'step':>6} {'AG':>6} {'RS':>6}   expected: AG=8, RS=4 (4 blocks + paramless root)")
    targets = steps if pick is None else [t for t in steps if t[0] in pick]
    counts = {}
    for idx, s, e, *_ in targets:
        ag, rs = census_one_step(conn, s, e)
        counts[idx] = (ag, rs)
        print(f"{idx:>6} {ag:>6} {rs:>6}")
    settled = {i: c for i, c in counts.items() if i >= 10}
    if len(set(settled.values())) > 1:
        print("  ⚠ Count mismatch across settled steps — truncation/loss/dynamic behavior suspected")
    return counts


# ---------------------------------------------------------------------------
# R3. Thread audit — thread distribution of launch APIs (NVTX thread-local check)
# ---------------------------------------------------------------------------
def report_threads(conn, steps, pick_idx):
    a = MAPPING["api"]
    win = [t for t in steps if t[0] == pick_idx]
    if not win:
        print(f"\n[R3] Could not find window for step {pick_idx}")
        return
    _, s, e, *_ = win[0]
    print(f"\n[R3] Launch-API thread distribution for step {pick_idx}")
    q = (f"SELECT {a['thread']} AS tid, COUNT(DISTINCT {a['corr']}) c, "
         f"       MIN({a['start']}) mn, MAX({a['start']}) mx "
         f"FROM {a['table']} "
         f"WHERE {a['name']} LIKE ? AND {a['start']} BETWEEN ? AND ? "
         f"GROUP BY {a['thread']} ORDER BY mn")
    rows = conn.execute(q, (LAUNCH_PATTERN, s, e)).fetchall()
    for r in rows:
        span_ms = (r["mx"] - r["mn"]) / 1e6
        print(f"  thread {r['tid']}: launches={r['c']}, "
              f"active span {span_ms:.2f} ms (relative start in window {(r['mn']-s)/1e6:.2f} ms)")
    if len(rows) >= 2:
        print("  → 2+ threads = forward (main) and backward (autograd) launch separately.")


# ---------------------------------------------------------------------------
# INTERP shared helper — computes one step's bucket boundaries and the autograd
#   thread tid. The decision logic lives in one place so that report_interp and
#   report_b0diff are guaranteed to use the same boundaries / same filters.
#   Returns: (ag3, rs_dict, tid) — None if a boundary anchor is missing or
#   there are no launches
# ---------------------------------------------------------------------------
def interp_boundaries(conn, s, e):

    a = MAPPING["api"]
    n = MAPPING["nvtx"]

    q_ag3 = (f"SELECT MAX({n['start']}) v FROM {n['table']} "
             f"WHERE {n['name']} = 'AG_unshard (3)' AND {n['start']} BETWEEN ? AND ?")

    q_rs  = (f"SELECT {n['end']} v FROM {n['table']} "
             f"WHERE {n['name']} = ? AND {n['start']} BETWEEN ? AND ?")

    q_tid = (f"SELECT {a['thread']} t FROM {a['table']} "
             f"WHERE {a['name']} LIKE ? AND {a['start']} BETWEEN ? AND ? "
             f"GROUP BY {a['thread']} ORDER BY MIN({a['start']}) DESC LIMIT 1")

    ag3 = conn.execute(q_ag3, (s, e)).fetchone()["v"]

    rs = {}
    for blk in (3, 2, 1, 0):
        r = conn.execute(q_rs, (f"RS_post_backward ({blk})", s, e)).fetchone()
        rs[blk] = r["v"] if r else None

    if ag3 is None or any(rs[b] is None for b in rs):
        return None

    row = conn.execute(q_tid, (LAUNCH_PATTERN, s, e)).fetchone()
    if row is None:
        return None

    return ag3, rs, row["t"]


# ---------------------------------------------------------------------------
# [INTERP] bare-launch block attribution — INTERP-v1 (backward, autograd thread)
#   bare = active_nvtx_id = 0  (valid because the autograd thread's baseline
#   stack is empty)
# ---------------------------------------------------------------------------
def report_interp(conn, steps):

    a = MAPPING["api"]

    print("\n[INTERP] bare-launch block attribution (INTERP-v1, autograd thread)")
    print(f"{'step':>6} {'head':>6} {'b3':>5} {'b2':>5} {'b1':>5} {'b0':>5} {'post':>5} {'total':>6}")

    for idx, s, e, *_ in steps:

        bounds = interp_boundaries(conn, s, e)
        if bounds is None:
            print(f"{idx:>6}  (boundary anchor missing or no launches — skipped)")
            continue

        ag3, rs, tid = bounds

        q_bucket = (
            f"SELECT CASE"
            f"  WHEN {a['start']} < ? THEN '0head'"
            f"  WHEN {a['start']} < ? THEN '1b3'"
            f"  WHEN {a['start']} < ? THEN '2b2'"
            f"  WHEN {a['start']} < ? THEN '3b1'"
            f"  WHEN {a['start']} < ? THEN '4b0'"
            f"  ELSE '5post' END AS k, "
            f"COUNT(DISTINCT {a['corr']}) c "
            f"FROM {a['table']} "
            f"WHERE {a['name']} LIKE ? AND {a['nvtx']} = 0 "
            f"  AND {a['thread']} = ? AND {a['start']} BETWEEN ? AND ? "
            f"GROUP BY k")

        counts = {r["k"]: r["c"] for r in conn.execute(
            q_bucket, (ag3, rs[3], rs[2], rs[1], rs[0], LAUNCH_PATTERN, tid, s, e))}

        h  = counts.get("0head", 0)
        b3 = counts.get("1b3", 0)
        b2 = counts.get("2b2", 0)
        b1 = counts.get("3b1", 0)
        b0 = counts.get("4b0", 0)
        po = counts.get("5post", 0)

        tot = h + b3 + b2 + b1 + b0 + po
        flag = "  ⚠post≠0" if po else ""
        print(f"{idx:>6} {h:>6} {b3:>5} {b2:>5} {b1:>5} {b0:>5} {po:>5} {tot:>6}{flag}")

    print("  ※ Verdict criteria: on settled steps b3≈b2≈b1≈b0 (identical-structure "
          "blocks), post=0, total≈bare grand total")


# ---------------------------------------------------------------------------
# [B0DIFF] b1 vs b0 bare kernel-name diff — resolve the b0 −2 anomaly
#   Verdict (confirmed 2026-08-05): deficit = exactly one layer_norm_backward
#   pair (reduction + pointwise); cause = block 0 input requires_grad=False →
#   the leading layer_norm's input-grad path is elided at compile time, plus
#   block-0-specialized fused variants (_16↔_17 count-neutral swap).
# ---------------------------------------------------------------------------
def report_b0diff(conn, steps, pick_idx):

    k, a = MAPPING["kernels"], MAPPING["api"]

    win = [t for t in steps if t[0] == pick_idx]
    if not win:
        print(f"\n[B0DIFF] Could not find window for step {pick_idx}")
        return
    _, s, e, *_ = win[0]

    bounds = interp_boundaries(conn, s, e)
    if bounds is None:
        print(f"\n[B0DIFF] step {pick_idx}: boundary anchor missing — skipped")
        return

    ag3, rs, tid = bounds

    # b2: [rs[3], rs[2]), b1: [rs[2], rs[1]), b0: [rs[1], rs[0])
    q_names = (
        f"SELECT k.{k['name']} AS n, COUNT(DISTINCT k.{k['id']}) c "
        f"FROM {a['table']} a JOIN {k['table']} k "
        f"  ON a.{a['corr']} = k.{k['corr']} "
        f"WHERE a.{a['name']} LIKE ? AND a.{a['nvtx']} = 0 "
        f"  AND a.{a['thread']} = ? "
        f"  AND a.{a['start']} >= ? AND a.{a['start']} < ? "
        f"GROUP BY n")

    def census(lo, hi):
        return {r["n"]: r["c"] for r in conn.execute(
            q_names, (LAUNCH_PATTERN, tid, lo, hi))}

    c_b2 = census(rs[3], rs[2])
    c_b1 = census(rs[2], rs[1])
    c_b0 = census(rs[1], rs[0])

    print(f"\n[B0DIFF] step {pick_idx}: b1 vs b0 bare kernel-name census "
          f"(same boundaries as INTERP, DISTINCT kernels)")
    print(f"  b1 total={sum(c_b1.values())}, b0 total={sum(c_b0.values())} "
          f"(expected 42 vs 40)")

    all_names = sorted(set(c_b1) | set(c_b0))
    diff_names = []
    for n in all_names:
        v1, v0 = c_b1.get(n, 0), c_b0.get(n, 0)
        marker = ""
        if v1 != v0:
            marker = "   <-- DIFF"
            diff_names.append((n, v1, v0))
        print(f"  {n[:120]:120s} b1={v1:3d} b0={v0:3d}{marker}")

    # --- control group b2 vs b1 (expected: 0 DIFF entries) ---
    ctrl_diff = [(n, c_b2.get(n, 0), c_b1.get(n, 0))
                 for n in sorted(set(c_b2) | set(c_b1))
                 if c_b2.get(n, 0) != c_b1.get(n, 0)]
    print(f"  [control] b2 vs b1: {len(ctrl_diff)} DIFF entries (expected 0), "
          f"b2 total={sum(c_b2.values())}")
    for n, v2, v1 in ctrl_diff:
        print(f"    {n[:120]:120s} b2={v2:3d} b1={v1:3d}")

    # --- verdict ---
    print("  " + "-" * 60)
    deficit = sum(v1 - v0 for _, v1, v0 in diff_names)
    print(f"  {len(diff_names)} DIFF entries, total difference = {deficit}, "
          f"control DIFF = {len(ctrl_diff)} entries")
    if deficit == 2 and not ctrl_diff:
        print("  → Verdict: structural deficit confirmed — middle blocks fully "
              "identical, only b0 is −2 (compile-time elision due to input "
              "requires_grad=False).")
    else:
        print("  → Verdict withheld: DIFF in control group or total ≠ 2 — "
              "re-examine boundaries.")


# ---------------------------------------------------------------------------
# [BLEED] boundary bleed A/B experiment — how the attribution rule affects
# per-step duration sums
# ---------------------------------------------------------------------------
def report_bleed(conn, steps):

    k, a = MAPPING["kernels"], MAPPING["api"]
    print("\n[BLEED] boundary bleed A/B (per-step kernel duration sum, ms)")

    print(f"{'step':>6} {'A(kernel-time)':>15} {'B(launch-attr)':>15} {'A-B':>10}")

    qa = (f"SELECT COALESCE(SUM({k['end']} - {k['start']}), 0) s "
          f"FROM {k['table']} WHERE {k['start']} BETWEEN ? AND ?")

    qb = (f"SELECT COALESCE(SUM(dur), 0) s FROM ("
          f"  SELECT k.{k['id']} kid, MIN(k.{k['end']} - k.{k['start']}) AS dur "
          f"  FROM {k['table']} k JOIN {a['table']} a "
          f"    ON k.{k['corr']} = a.{a['corr']} "
          f"  WHERE a.{a['start']} BETWEEN ? AND ? "
          f"  GROUP BY kid)")

    tot_b = 0
    for idx, s, e, *_ in steps:

        A = conn.execute(qa, (s, e)).fetchone()["s"] / 1e6
        B = conn.execute(qb, (s, e)).fetchone()["s"] / 1e6
        tot_b += B
        print(f"{idx:>6} {A:>15.3f} {B:>15.3f} {A-B:>10.3f}")

    allk = conn.execute(
        f"SELECT COALESCE(SUM({k['end']} - {k['start']}),0) s FROM {k['table']}"
    ).fetchone()["s"] / 1e6

    print(f"  (note: B summed over all steps {tot_b:.1f} ms vs whole-trace kernel "
          f"duration sum {allk:.1f} ms — difference = kernels launched outside "
          f"windows: warmup/barrier etc.)")


def report_gapdrain(conn, steps):
    """
    [GAPDRAIN] Duration sum of kernels that STARTED on the GPU inside a gap
    (pop(i)~push(i+1)).
    Verdict criteria: sum_ms ≈ that step's |A−B|
      - settled (gaps after 2..16): ≈ 0.052 constant
      - 1→2: ≈ 20.4 (post-compile drain)
      - 17→18, 18→19, 19→(end): ≈ 2.6
    The comparison target is sum_ms (BLEED's A is a simple start-in-window sum).
    merged/overlap are side info for observing stream overlap inside the gap.
    """
    k = MAPPING["kernels"]

    print("\n[GAPDRAIN] gap-interval kernel durations (membership by GPU start)")
    print(f"{'gap':>10} {'n_kern':>7} {'sum_ms':>9} {'merged_ms':>10} {'overlap_ms':>10}")

    q = (f"SELECT {k['start']} AS s, {k['end']} AS e "
         f"FROM {k['table']} "
         f"WHERE {k['start']} >= ? AND {k['start']} < ? "
         f"ORDER BY s")

    for (i, s1, e1, *_), (j, s2, e2, *_) in zip(steps, steps[1:]):
        gap_s, gap_e = e1, s2          # pop(i) ~ push(i+1)

        rows = conn.execute(q, (gap_s, gap_e)).fetchall()

        n = len(rows)
        simple = sum(r["e"] - r["s"] for r in rows)

        # interval merge — for observing stream overlap inside the gap
        merged = 0
        cur_s = cur_e = None
        for r in rows:
            if cur_e is None or r["s"] > cur_e:
                if cur_e is not None:
                    merged += cur_e - cur_s
                cur_s, cur_e = r["s"], r["e"]
            else:
                cur_e = max(cur_e, r["e"])
        if cur_e is not None:
            merged += cur_e - cur_s

        print(f"{i}->{j:>4} {n:>7} {simple/1e6:>9.3f} "
              f"{merged/1e6:>10.3f} {(simple-merged)/1e6:>10.3f}")

    print("  ※ Verdict criteria: if sum_ms matches [BLEED] |A−B| gap-by-gap 1:1, "
          "the evaporation model is confirmed.")


def report_gapdrain_names(conn, steps, pick_gap=(17, 18)):
    k, a = MAPPING["kernels"], MAPPING["api"]
    i, j = pick_gap
    e1 = [t[2] for t in steps if t[0] == i][0]   # pop(i)
    s2 = [t[1] for t in steps if t[0] == j][0]   # push(j)
    q = (f"SELECT k.{k['name']} AS n, k.{k['start']} AS s, "
         f"       k.{k['end']} - k.{k['start']} AS dur, "
         f"       a.{a['thread']} AS tid, a.{a['start']} AS launch_t "
         f"FROM {k['table']} k JOIN {a['table']} a "
         f"  ON k.{k['corr']} = a.{a['corr']} "
         f"WHERE k.{k['start']} >= ? AND k.{k['start']} < ? "
         f"GROUP BY k.{k['id']} ORDER BY s")
    print(f"\n[GAPDRAIN-NAMES] gap {i}->{j}")
    for r in conn.execute(q, (e1, s2)):
        print(f"  dur={r['dur']/1e6:.3f}ms launch@{(r['launch_t']-e1)/1e6:+.3f}ms(vs pop) "
              f"tid={r['tid']} {r['n'][:100]}")
# ---------------------------------------------------------------------------
# [INTERP-F] forward block attribution of main-thread bare launches — INTERP-F v0
#   Boundaries: MIN push of AG_unshard(k) = the forward one (bwd/prefetch come
#               later in time)
#               AG(3).MAX = backward start (same value as INTERP-v1's ag3)
#   bare = active_nvtx_id = the step range's rid
#          (the main thread's baseline stack is [step_N], so 0 never occurs —
#           filtering on 0 yields all-zero. The 2026-08-05 trap.)
#   Buckets: pre / f0 / f1 / f2 / f3h(=b3 fwd+head/loss) / bwdwin(expected 0) / opt
#   Verdict criteria: f0≈f1≈f2 (block symmetry), bwdwin=0, total=77 constant
#          across steps (77 = main total 85 − 8 forward-anchor-attributed)
# ---------------------------------------------------------------------------
def report_interpf(conn, steps):

    a = MAPPING["api"]
    n = MAPPING["nvtx"]

    print("\n[INTERP-F] main-thread bare-launch forward attribution")
    print(f"{'step':>6} {'pre':>5} {'f0':>5} {'f1':>5} {'f2':>5} {'f3h':>5} "
          f"{'bwdwin':>7} {'opt':>5} {'total':>6}")

    q_ag_min = (f"SELECT MIN({n['start']}) v FROM {n['table']} "
                f"WHERE {n['name']} = ? AND {n['start']} BETWEEN ? AND ?")
    q_ag_max = (f"SELECT MAX({n['start']}) v FROM {n['table']} "
                f"WHERE {n['name']} = ? AND {n['start']} BETWEEN ? AND ?")
    q_rs0    = (f"SELECT {n['end']} v FROM {n['table']} "
                f"WHERE {n['name']} = 'RS_post_backward (0)' AND {n['start']} BETWEEN ? AND ?")
    # main thread = the thread that started launching first inside the window
    # (opposite of INTERP's DESC)
    q_tid = (f"SELECT {a['thread']} t FROM {a['table']} "
             f"WHERE {a['name']} LIKE ? AND {a['start']} BETWEEN ? AND ? "
             f"GROUP BY {a['thread']} ORDER BY MIN({a['start']}) ASC LIMIT 1")

    for idx, s, e, rid in steps:

        ag_min = {}
        for blk in (0, 1, 2, 3):
            r = conn.execute(q_ag_min, (f"AG_unshard ({blk})", s, e)).fetchone()
            ag_min[blk] = r["v"] if r else None
        ag3_max = conn.execute(q_ag_max, ("AG_unshard (3)", s, e)).fetchone()["v"]
        rs0 = conn.execute(q_rs0, (s, e)).fetchone()
        rs0 = rs0["v"] if rs0 else None

        if any(ag_min[b] is None for b in ag_min) or ag3_max is None or rs0 is None:
            print(f"{idx:>6}  (boundary anchor missing — skipped)")
            continue

        row = conn.execute(q_tid, (LAUNCH_PATTERN, s, e)).fetchone()
        if row is None:
            print(f"{idx:>6}  (no launches)")
            continue
        tid = row["t"]

        q_bucket = (
            f"SELECT CASE"
            f"  WHEN {a['start']} < ? THEN '0pre'"
            f"  WHEN {a['start']} < ? THEN '1f0'"
            f"  WHEN {a['start']} < ? THEN '2f1'"
            f"  WHEN {a['start']} < ? THEN '3f2'"
            f"  WHEN {a['start']} < ? THEN '4f3h'"
            f"  WHEN {a['start']} < ? THEN '5bwdwin'"
            f"  ELSE '6opt' END AS k, "
            f"COUNT(DISTINCT {a['corr']}) c "
            f"FROM {a['table']} "
            f"WHERE {a['name']} LIKE ? AND {a['nvtx']} = ? "
            f"  AND {a['thread']} = ? AND {a['start']} BETWEEN ? AND ? "
            f"GROUP BY k")

        counts = {r["k"]: r["c"] for r in conn.execute(
            q_bucket, (ag_min[0], ag_min[1], ag_min[2], ag_min[3],
                       ag3_max, rs0, LAUNCH_PATTERN, rid, tid, s, e))}

        pre = counts.get("0pre", 0)
        f0  = counts.get("1f0", 0)
        f1  = counts.get("2f1", 0)
        f2  = counts.get("3f2", 0)
        f3h = counts.get("4f3h", 0)
        bw  = counts.get("5bwdwin", 0)
        op  = counts.get("6opt", 0)

        tot = pre + f0 + f1 + f2 + f3h + bw + op
        flag = "  ⚠bwdwin≠0" if bw else ""
        print(f"{idx:>6} {pre:>5} {f0:>5} {f1:>5} {f2:>5} {f3h:>5} "
              f"{bw:>7} {op:>5} {tot:>6}{flag}")

    print("  ※ Verdict criteria: on settled steps f0≈f1≈f2 (block symmetry), "
          "f3h > f2 (head's share), bwdwin=0, total=77 constant across steps")


def report_rs_durations(conn, steps):
    """
    [RSDUR] Durations of the 4 ReduceScatter kernels per step (ms).
    Membership = launch time (kernels executing outside the window still belong
    to their own step — needed to catch RS(0) in 17~18).
    Output order = GPU execution order = b3, b2, b1, b0 (sequentiality:
    verified Aug 3).
    Verdict criteria:
      - Last column (RS0) small on settled steps but spiking only at 17~18 →
        duration increase = NCCL peer-wait = R3 (skew) candidate
      - RS0 flat at ~2.5ms on all steps → duration innocent → by elimination,
        start delay = queue backlog (R2 family)
      - All four columns spiking → a cause covering the whole step
        (clock/contention)
    """
    k, a = MAPPING["kernels"], MAPPING["api"]

    print("\n[RSDUR] per-step RS kernel durations (ms, execution order = b3 b2 b1 b0)")

    q = (f"SELECT k.{k['start']} AS s, "
         f"       MIN(k.{k['end']} - k.{k['start']}) AS dur "
         f"FROM {k['table']} k JOIN {a['table']} a "
         f"  ON k.{k['corr']} = a.{a['corr']} "
         f"WHERE k.{k['name']} LIKE ? AND a.{a['start']} BETWEEN ? AND ? "
         f"GROUP BY k.{k['id']} ORDER BY s")

    for idx, s, e, *_ in steps:
        durs = [r["dur"] / 1e6 for r in conn.execute(q, (RS_PATTERN, s, e))]
        line = "  ".join(f"{d:7.3f}" for d in durs)
        flag = "" if len(durs) == 4 else f"  ⚠ {len(durs)} RS kernels (expected 4)"
        print(f"  step {idx:>2}: {line}{flag}")


def _merge(intervals):
    """[(s,e), ...] → total length with overlaps folded. Input need not be sorted."""
    tot = 0
    cur_s = cur_e = None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def report_segd(conn, steps):
    """
    [SEGD] inter-RS segment decomposition (sum over 3 segments, ms).
    comp = merged non-nccl kernels, nccl = merged nccl kernels (backward AG
    prefetch; includes waiting),
    union = merge of everything, idle = span − union (true idle),
    ovlp = comp + nccl − union (compute-communication concurrency),
    qlag = median(kernel.start − launch.enter), comp kernels only.
    Verdict criteria: comp↑ = workload / idle↑ & qlag small = launch-side /
    idle↑ & qlag large = queue delay.
    """
    k, a = MAPPING["kernels"], MAPPING["api"]

    q_rs = (f"SELECT k.{k['start']} AS s, MIN(k.{k['end']}) AS e "
            f"FROM {k['table']} k JOIN {a['table']} a "
            f"  ON k.{k['corr']} = a.{a['corr']} "
            f"WHERE k.{k['name']} LIKE ? AND a.{a['start']} BETWEEN ? AND ? "
            f"GROUP BY k.{k['id']} ORDER BY s")

    q_seg = (f"SELECT k.{k['start']} AS s, k.{k['end']} AS e, "
             f"       MIN(a.{a['start']}) AS le, "
             f"       CASE WHEN k.{k['name']} LIKE '%nccl%' THEN 1 ELSE 0 END AS isn "
             f"FROM {k['table']} k JOIN {a['table']} a "
             f"  ON k.{k['corr']} = a.{a['corr']} "
             f"WHERE k.{k['start']} >= ? AND k.{k['start']} < ? "
             f"GROUP BY k.{k['id']}")

    print("\n[SEGD] inter-RS segment decomposition (sum over 3 segments, ms)")
    print(f"{'step':>6} {'span':>8} {'comp':>7} {'nccl':>7} {'idle':>8} "
          f"{'ovlp':>6} {'qlag':>7} {'n_k':>5}")

    for idx, s, e, *_ in steps:
        rs = conn.execute(q_rs, (RS_PATTERN, s, e)).fetchall()
        if len(rs) != 4:
            print(f"{idx:>6}  ({len(rs)} RS kernels — skipped)")
            continue

        span = 0
        comp_iv, nccl_iv, lags = [], [], []
        for i in range(3):                      # b2, b1, b0
            lo, hi = rs[i]["e"], rs[i + 1]["s"]
            span += hi - lo
            for r in conn.execute(q_seg, (lo, hi)):
                iv = (r["s"], r["e"])
                if r["isn"]:
                    nccl_iv.append(iv)
                else:
                    comp_iv.append(iv)
                    lags.append(r["s"] - r["le"])   # qlag from comp only

        comp  = _merge(comp_iv)
        nccl  = _merge(nccl_iv)
        union = _merge(comp_iv + nccl_iv)
        idle  = span - union
        ovlp  = comp + nccl - union
        nk    = len(comp_iv) + len(nccl_iv)

        lags.sort()
        qlag = lags[len(lags) // 2] if lags else 0
        print(f"{idx:>6} {span/1e6:>8.3f} {comp/1e6:>7.3f} {nccl/1e6:>7.3f} "
              f"{idle/1e6:>8.3f} {ovlp/1e6:>6.3f} {qlag/1e6:>7.3f} {nk:>5}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db", help="path to a per-rank SQLite trace")
    ap.add_argument("--run", action="store_true", help="run the reports")
    ap.add_argument("--step", type=int, default=12, help="step to use for R3/B0DIFF (default 12)")
    args = ap.parse_args()

    conn = connect_ro(args.db)
    if not args.run:
        introspect(conn)
        return

    steps = load_step_windows(conn)
    if not steps:
        print("No NVTX step ranges found — check MAPPING['nvtx'] names/columns.")
        sys.exit(1)
    print(f"Found {len(steps)} step windows: {steps[0][0]} ~ {steps[-1][0]}")

    names_ok = report_nccl_names(conn)
    report_settled(conn, steps)
    if names_ok:
        report_census(conn, steps)
    else:
        print("\n[R2] skipped — AG/RS name-pattern check failed in R0")
    report_threads(conn, steps, args.step)

    report_interp(conn, steps)

    report_b0diff(conn, steps, args.step)

    report_bleed(conn, steps)

    report_interpf(conn, steps)

    report_gapdrain(conn, steps)

    report_gapdrain_names(conn, steps)

    report_rs_durations(conn, steps)

    report_segd(conn, steps)


if __name__ == "__main__":
    main()