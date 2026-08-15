
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from trace_reader import MAPPING, connect_ro, load_step_windows

def cat(name):

    if "ReduceScatter" in name: return "RS"
    if "AllGather" in name:     return "AG"
    if "nccl" in name:          return "nccl_etc"
    if "multi_tensor" in name:  return "opt"

    return "compute"

COLORS = {"RS": "#d62728", "AG": "#ff7f0e", "nccl_etc": "#8c564b",
          "opt": "#9467bd", "compute": "#2ca02c", "memcpy": "#7f7f7f"}

def fetch(conn, s, e, pad):

    k = MAPPING["kernels"]
    m = MAPPING["memcpy"]

    rows = []

    q = (f"SELECT {k['name']} n, {k['start']} s, {k['end']} e, {k['stream']} st "
         f"FROM {k['table']} WHERE {k['start']} < ? AND {k['end']} > ?")

    for r in conn.execute(q, (e + pad, s - pad)):
        rows.append((r["st"], r["s"], r["e"], cat(r["n"])))

    qm = (f"SELECT {m['start']} s, {m['end']} e FROM {m['table']} "
          f"WHERE {m['start']} < ? AND {m['end']} > ?")

    for r in conn.execute(qm, (e + pad, s - pad)):
        rows.append(("memcpy", r["s"], r["e"], "memcpy"))

    return rows

def draw(ax, rows, t0, pop, title):

    lanes = sorted({r[0] for r in rows}, key=str)
    
    ymap = {st: i for i, st in enumerate(lanes)}

    for st, s, e, c in rows:
        ax.broken_barh([((s - t0) / 1e6, max((e - s) / 1e6, 0.02))],
                       (ymap[st] - 0.35, 0.7), color=COLORS[c], lw=0)

    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.axvline((pop - t0) / 1e6, color="k", ls="--", lw=0.8)
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels([f"stream {st}" for st in lanes], fontsize=8)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="x", lw=0.3, alpha=0.5)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db0"); ap.add_argument("db1")
    ap.add_argument("--step", type=int, default=12)
    ap.add_argument("--pad_ms", type=float, default=3.0)
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    for ax, db, label in ((axes[0], args.db0, "rank 0"), (axes[1], args.db1, "rank 1")):
        conn = conn = connect_ro(db)
        steps = {t[0]: t for t in load_step_windows(conn)}
        _, s, e, *_ = steps[args.step]
        pad = int(args.pad_ms * 1e6)
        draw(ax, fetch(conn, s, e, pad), s, e, f"{label} — step {args.step} (t=0: push, dot line: pop)")

    axes[1].set_xlabel("ms")

    fig.legend(handles=[Patch(color=v, label=k) for k, v in COLORS.items()],
               loc="upper right", ncol=6, fontsize=8)
    fig.tight_layout()

    out = f"step_{args.step}_timeline.png"
    fig.savefig(out, dpi=160)
    print(f"saved: {out}")

if __name__ == "__main__":
    main()