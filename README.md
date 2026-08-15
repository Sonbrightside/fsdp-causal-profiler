# GapScope

**Step-time gap profiler for distributed training — computes a feasible lower bound and classifies the gap into recoverable vs. hard-constrained stalls.**

Most GPU profilers tell you *where* time goes. GapScope answers a different question: **how much of your step time could you actually get back?** It computes a feasible step-time lower bound (`T_lb`) from measured kernel work and a rules-based dependency model, attributes the gap (`observed − T_lb`) to concrete mechanisms, and issues a *recoverability verdict* for each one.

## Why

- **The compiler can't see your collectives.** With PyTorch's default FSDP2 + `torch.compile` configuration, communication ops are invisible to graph-level analysis (`skip_fsdp_hooks=True`). Runtime measurement is the only reliable source of truth — so GapScope measures at runtime, via a lightweight custom CUPTI tracer instead of heavyweight tools like Nsight Systems.
- **"Idle time" is not one thing.** A gap caused by late kernel launches is recoverable; a gap forced by the dependency structure is not. Treating them the same wastes optimization effort. GapScope separates them.

## Gap taxonomy

| Class | Meaning | Recoverable? |
|-------|---------|--------------|
| R1 | Late launch (CPU launch-pacing starvation) | Yes |
| R2 | Queue serialization | Yes |
| R3 | Cross-rank skew (one rank waits for a laggard) | Yes |
| H1 | Structural orphan (dependency-forced idle) | No |
| H2 | Budget insufficient (not enough parallel work) | No |

Classification is validated with controlled fault-injection experiments (injection → confusion matrix).

## How it works

```
training run (FSDP2 + torch.compile)
        │
        ▼
┌─────────────────────┐
│  CUPTI tracer (C++) │  injection library, per-rank SQLite output
│  + NVTX anchors     │  runtime/driver API + kernel/memcpy activity
└─────────┬───────────┘  + correlation-ID attribution
          ▼
┌─────────────────────┐
│  analysis (Python)  │  per-step census, block-level attribution,
│  trace_reader.py    │  as-executed overlap analysis,
└─────────┬───────────┘  cross-rank skew, launch-pacing reports
          │
          │      ┌──────────────────────────────┐
          │      │  dependency model            │
          ├──────│  (rules-based, from FSDP2's  │
          │      │  algorithmic structure —     │
          │      │  validated against traces,   │
          │      │  NOT inferred from them)     │
          ▼      └──────────────────────────────┘
┌─────────────────────┐
│  T_lb + gap verdict │  measured per-node work × dependency model
│                     │  → feasible lower bound (+ binding-bound
│                     │  analysis: which chain pins T_lb)
│                     │  → observed − T_lb
│                     │  → R1/R2/R3 vs H1/H2 classification
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Gantt timelines    │  per-step, per-stream visual inspection
│  plot_step.py       │  (from GapScope's own trace DB)
└─────────────────────┘
```

Key design choices:

- **Correlation-based attribution** (not time-overlap): every GPU activity record is joined to its launching API call and the NVTX context active *at launch time*, per thread.
- **Anchor + interpolation**: FSDP2 hook points (`unshard` / `post_backward`) are NVTX-annotated via a minimal monkey-patch (`nvtx_cov.py`); the remaining bare launches are block-attributed by interpolating between collective anchors — validated to single-kernel accuracy.
- **Rules-based dependency model, trace-validated**: the dependency structure used for `T_lb` is authored from FSDP2's algorithmic definition (unshard → compute → post_backward → reduce-scatter), never inferred from observed execution order — inferring dependencies from a run and then judging that run with them would be circular. Traces are used only to *validate* the model (anchor-order checks caught real behavior the rules alone would miss, e.g. backward prefetch).
- **Clock-alignment for multi-rank analysis**: per-rank (CUPTI, host-monotonic) anchor pairs plus a ping-pong host-offset exchange, so cross-rank timelines compose without relying on collective-simultaneity assumptions.

## Scope

GapScope classifies the gap against a feasible bound; it does not compute as-executed critical paths (see dPRO, CASITA for that line of work). Cross-rank analysis relies on collective all-arrive semantics, so straggler attribution stays O(N) in rank count. As-executed critical-path composition and large-scale (100s of ranks) heatmap-level reporting are future work.

## Repository layout

```
gapscope/
├── tracer/          # C++ CUPTI injection tracer (builds libcupti_trace.so)
├── analysis/        # trace_reader.py (reports), plot_step.py (Gantt timelines)
├── instrumentation/ # nvtx_cov.py (FSDP2 NVTX anchors), census_run.py (workload driver)
├── experiments/     # sbatch scripts and run configurations
└── docs/            # design notes and experiment log
```

## Quick start

> ⚠️ Work in progress — interfaces may change.

```bash
# 1. Build the tracer
cd tracer && make          # requires CUDA toolkit (nvcc) with CUPTI

# 2. Run a workload under the tracer
export NVTX_INJECTION64_PATH=$PWD/tracer/libcupti_trace.so   # CUPTI NVTX capture
LD_PRELOAD=$PWD/tracer/libcupti_trace.so \
  torchrun --nproc_per_node=2 instrumentation/census_run.py

# On a Slurm cluster, use the template instead:
#   edit CONFIG in experiments/submit_fsdp.sbatch, then: sbatch experiments/submit_fsdp.sbatch

# 3. Analyze the per-rank trace databases
python analysis/trace_reader.py trace_rank0.db
python analysis/plot_step.py trace_rank0.db --step 12
```

## Requirements

- PyTorch ≥ 2.9 (FSDP2 / `fully_shard`), CUDA 12.x with CUPTI
- Python 3.10+, `matplotlib` for timeline plots
- Linux; tested with Slurm (`sbatch`) on multi-GPU nodes

## Status

Active development toward a paper submission. Current state: tracer and attribution pipeline validated end-to-end on a multi-block FSDP2 workload (per-step collective census, block-level launch attribution, cross-rank skew measurement); `T_lb` assembly and injection-based classifier validation in progress.

## License

TBD (MIT or Apache-2.0 planned).
