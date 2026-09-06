# Running the experiments

```bash
python run_experiments.py --smoke        # toy scale, ~2 min, checks the pipeline
python run_experiments.py --phases bench # phase 1 only: cheap, run this first
python run_experiments.py                # the full protocol
```

Locally the interpreter is `.venv/Scripts/python.exe`; in Colab it is plain `python`.

## Flags

| flag | default | what it does |
|---|---|---|
| `--smoke` | off | toy scale for every phase: `n_embd=64`, 2 layers, block 64, batches 8/16, ranks 4/8, 50 iters, 5 bench reps, tensorized arms trained **eager**. Validates the pipeline end to end in minutes. |
| `--phases [bench train]` | `bench train` | which phases may **compute**. The memory and rank phases always run and only re-read the cache. Passing `--phases` with no values re-plots everything from the cache without touching the GPU. |
| `--force` | off | recompute cells that are already cached instead of loading them. |
| `--out DIR` | `results` | root for `runs/` (cache), `tables/` (csv), `plots/` (png). |
| `--quiet` | off | silence inductor/dynamo compile chatter. |
| `--progress` | off | per-iteration tqdm bars. Off by default because colab `!python` is not a tty and tqdm then writes one line per update, burying the results. |
| `--init scaled\|normal` | `scaled` | `scaled`: cores sized so the *contracted* matrix has std 0.02. `normal`: literal std=1.0, which diverges — there to demonstrate why the scaling exists. |

### Phase 1 — benchmark

| flag | default | what it does |
|---|---|---|
| `--batches 32 64 128` | `32 64 128` (smoke `8 16`) | batch sizes on the x axis of the epoch-time and peak-memory plots. |
| `--compile-modes eager compile cudagraph` | all three | which of the three modes to measure. `compile` is inductor fusion, `cudagraph` is fusion + CUDA Graphs (a no-op on xpu). |
| `--bench-rank N` | `32` | `max_rank` of the tensorized model being benchmarked. |
| `--bench-reps N` | `30` (smoke `5`) | timed forward/backward repetitions per cell, after 5 warmup steps (2 in smoke). |

### Phase 2 — training

| flag | default | what it does |
|---|---|---|
| `--ranks 4 8 16 32` | `4 8 16 32` (smoke `4 8`) | the rank ladder. One `uniform-r<R>` arm and one adaptive arm per `lr_rank` for each entry. |
| `--lr-ranks 3e-3 1e-2` | `3e-3 1e-2` | learning rates for the rank parameters; each value becomes its own adaptive family, i.e. its own Pareto curve. Larger = more aggressive pruning. |
| `--gamma G` | `0.1` (`comera.GAMMA`) | weight of the rank loss in `comera_loss`. Held fixed across arms so `lr_rank` is the only pruning-aggressiveness axis. |
| `--arms NAME ...` | all | subset of arm names, e.g. `--arms dense uniform-r16 adaptive-r16-lr0.01`. |
| `--train-compile-mode eager\|compile\|cudagraph` | `compile` (smoke `eager`) | how the tensorized arms are compiled. **dense always trains eager.** `cudagraph` is not the default: CUDA Graphs replay a captured graph whose parameters AdamW mutates outside it, which inductor skips silently — phase 1 is where the graph win is measured. |
| `--iters N` | `1500` (smoke `50`) | training iterations per arm. |
| `--log-interval N` | `500` (smoke `25`) | iterations between printed progress lines. Losses are still recorded every 100 iterations for the plots. |

### Phase 5 — core diagnostics

Sampled during training, so these flags only affect arms that are actually (re)computed.

| flag | default | what it does |
|---|---|---|
| `--core-diag-interval N` | `100` (smoke `10`) | iterations between core snapshots; `0` disables. Deliberately **not** part of the cache key — retuning the diagnostics would otherwise invalidate hours of cached training. A cached arm therefore keeps whatever diagnostics it was trained with; `--force` resamples it. |
| `--core-diag-role c_attn\|attn_proj\|c_fc\|mlp_proj` | `c_fc` | which role the per-arm figure draws. Every probed role is written to `core_stats.csv` regardless; only the plot is narrowed, because a panel with 4 roles x 2d cores is unreadable. |

What is probed: every TT role in the **first, middle and last** block (`{0, n_layer//2, n_layer-1}`),
every core of those layers. Recorded per core: `mean`, `std`, `absmax`, `norm`, the gradient norm of
that step, and — in adaptive arms — the mean/min of the rank parameter gating its trailing bond plus
how many of its entries are still above the threshold (`rank_alive`). The whole snapshot is one
`stack` + one `.tolist()`, i.e. a single device synchronize, so it does not leak into the step-time
medians.

## Outputs

`results/runs/*.json` is the cache — one file per benchmark cell and per training arm, keyed by a hash
of the cell and the `TrainConfig`, so an interrupted Colab session resumes and a smoke run can never
be mistaken for a full one.

| file | contents |
|---|---|
| `tables/bench.csv` | per configuration and batch: fwd/bwd step time, projected minutes per epoch, peak memory, compile diagnostics |
| `tables/train.csv` | per arm: effective params, compression, val loss/ppl, step time, memory |
| `tables/ranks.csv` | one row per arm x TT layer x bond: initial and final rank |
| `tables/rank_summary.csv` | per adaptive arm: `pruned_frac`, mean/min/max rank, dead bonds |
| `tables/core_stats.csv` | one row per arm x snapshot x TT layer x core: mean/std/absmax/norm, gradient norm, rank-parameter state |
| `tables/memory.csv` | static footprint next to the measured peaks |
| `plots/1_epoch_{forward,backward}.png`, `2_epoch_total.png` | the CoMERA bar chart |
| `plots/3_pareto_params.png` | val loss vs surviving parameters, one curve per family |
| `plots/4_loss_curves.png`, `4b_pruning_over_training.png` | training dynamics |
| `plots/5_rank_pruned_frac.png`, `5b_rank_hist.png`, `5c_rank_heatmap.png`, `5d_rank_by_role.png` | final rank configuration |
| `plots/6_cores_<arm>.png` | one figure per arm: core std, `|max|`, gradient norm and rank-parameter mean through training, one row per probed block, one line per core (log axes wherever the quantity stays positive) |
| `plots/7_mem_static.png`, `8_mem_peak_{forward,backward}.png` | memory footprint |

## Reading the warnings

The summary block prints `!` lines that mean a number should not be trusted:

- `dynamo compiled 0 frames` — compilation fell back to eager, most likely dynamo's
  `cache_size_limit`. The cell is not measuring what it claims.
- `inductor skipped CUDA Graphs Nx` — the cudagraph cell is really a fusion cell. Expected on xpu,
  which has no CUDA-graph backend.
- `nothing pruned at all` — under Adam a rank parameter decays by at most `lr_rank` per step, so from
  its init of 1.0 it needs more than `1/lr_rank` steps to cross the 1e-2 threshold. Raise `--iters`,
  `--lr-ranks` or `--gamma`; until then the adaptive Pareto curves are just the uniform ones.

`reduce-overhead` (`cudagraph`) peak memory is not comparable with the other modes — CUDA Graphs
allocate from a private pool that `max_memory_allocated` does not account for the same way — so those
rows are marked `memory_comparable=False` and dropped from the memory plots.
