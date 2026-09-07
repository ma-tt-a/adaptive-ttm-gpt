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
| `--track off\|tensorboard\|wandb` | `off` | live tracking of losses, perplexity and core diagnostics — see below. |
| `--progress` | off | per-iteration tqdm bars. Off by default because colab `!python` is not a tty and tqdm then writes one line per update, burying the results. |
| `--init scaled\|normal` | `scaled` | `scaled`: cores sized so the *contracted* matrix has std 0.02. `normal`: literal std=1.0, which diverges — there to demonstrate why the scaling exists. |

### Phase 1 — benchmark

| flag | default | what it does |
|---|---|---|
| `--batches 1 8 32 64 128` | `1 8 32 64 128` (smoke `8 16`) | batch sizes on the x axis of the epoch-time, step-time and peak-memory plots. The small batches are the interesting ones for compile: by batch 128 the matmuls amortise the launch overhead on their own and every mode converges. |
| `--compile-modes eager compile cudagraph` | all three | which of the three modes to measure. `compile` is inductor fusion, `cudagraph` is fusion + CUDA Graphs (a no-op on xpu). |
| `--bench-rank N` | `32` | `max_rank` of the tensorized model being benchmarked. |
| `--bench-reps N` | `30` (smoke `5`) | timed forward/backward repetitions per cell, after 5 warmup steps (2 in smoke). |

The memory pass of a bench cell runs a **full training step with AdamW**, kept separate from the
timing pass (which stays forward/backward only — that is the CoMERA figure). Two states per parameter
is exactly the part of the tensorized saving that forward/backward alone cannot show, while the extra
intermediate `TTMatVec` saves (`X` *and* `T_1`) is charged either way. Measured at `n_embd=256`,
6 layers, eager, peak of a full step:

| batch | dense | tensorized | ratio |
|---|---|---|---|
| 1 | 96 MB | 26 MB | 0.27x |
| 8 | 191 MB | 144 MB | 0.76x |
| 128 | 2162 MB | 2163 MB | 1.00x |

Parameters plus AdamW states are 57.5 MB dense against 6.8 MB tensorized *at every batch size* — what
changes is the activation memory piled on top, which scales with tokens per step and is identical in
both models. The tensorized memory win is a parameter-side win, so it is visible exactly when the
model width is comparable to the tokens per step (roughly `3*C/BT` of the activation cost with Adam),
and invisible at large batch. `BENCH_PROTOCOL` guards this: bumping it recomputes bench cells whose
memory numbers were measured under an older protocol instead of mixing them with new ones.

### Phase 2 — training

| flag | default | what it does |
|---|---|---|
| `--ranks 4 8 16 32` | `4 8 16 32` (smoke `4 8`) | the rank ladder. One `uniform-r<R>` arm and one adaptive arm per `lr_rank` for each entry. |
| `--lr-ranks 3e-3 1e-2` | `3e-3 1e-2` | learning rates for the rank parameters; each value becomes its own adaptive family, i.e. its own Pareto curve. Larger = more aggressive pruning. |
| `--gamma G` | `0.1` (`comera.GAMMA`) | weight of the rank loss in `comera_loss`. Held fixed across arms so `lr_rank` is the only pruning-aggressiveness axis. |
| `--arms NAME ...` | all | subset of arm names, e.g. `--arms dense uniform-r16 adaptive-r16-lr0.01`. |
| `--train-compile-mode eager\|compile\|cudagraph` | `compile` (smoke `eager`) | how the tensorized arms are compiled. **dense always trains eager.** `cudagraph` is not the default: CUDA Graphs replay a captured graph whose parameters AdamW mutates outside it, which inductor skips silently — phase 1 is where the graph win is measured. |
| `--iters N` | `1500` (smoke `50`) | training iterations per arm — i.e. optimizer steps, each of which now consumes `--grad-accum` micro-batches. |
| `--micro-batch N` | `32` (smoke `8`) | sequences per forward. This is what activation memory scales with. |
| `--grad-accum N` | `1` | forward/backward passes accumulated before each optimizer step. |

The batch the optimizer actually steps on is `--micro-batch x --grad-accum`, so a batch that does not
fit in memory is reached by raising the accumulation rather than the micro-batch: `--micro-batch 16
--grad-accum 8` trains on 128 sequences while only ever holding 16 sequences' worth of activations.
The header line prints the arithmetic (`batch: 16 micro x 8 accum = 128`).

Two details of the implementation worth knowing:

- The per-micro-batch loss is divided by `--grad-accum`, so the accumulated gradient is the mean the
  full batch would have produced, not its sum.
- CoMERA's rank loss is a property of the weights, not of the data, so it is added on the **last**
  micro-step only. That keeps it undivided and pays for `comera.rank_loss` once per optimizer step
  instead of once per forward — it costs ~21 ms against a ~110 ms step, which is not something to
  multiply by the accumulation factor.

`step_time_s` therefore measures a whole optimizer step, accumulation included; at fixed effective
batch, halving the micro-batch roughly doubles it. Measured under `--smoke`: `8 x 1` and `4 x 2` end
at val 3.90 and 3.88 (same batch, different sampling), with peak memory 7.5 MB against 4.9 MB.

`--grad-accum 1` is bit-identical to the pre-accumulation loop and is left out of the cache key, so
existing cached arms stay valid; any other value is part of the key.
| `--log-interval N` | `500` (smoke `25`) | iterations between printed progress lines. Losses are still recorded every 100 iterations for the plots. |

### Phase 5 — core diagnostics

Sampled during training, so these flags only affect arms that are actually (re)computed.

| flag | default | what it does |
|---|---|---|
| `--core-diag-interval N` | `100` (smoke `10`) | iterations between core snapshots; `0` disables. Deliberately **not** part of the cache key — retuning the diagnostics would otherwise invalidate hours of cached training. A cached arm therefore keeps whatever diagnostics it was trained with; `--force` resamples it. |
| `--core-diag-role c_attn\|attn_proj\|c_fc\|mlp_proj` | `c_fc` | which role the per-arm figure draws. Every probed role is written to `core_stats.csv` regardless; only the plot is narrowed, because a panel with 4 roles x 2d cores is unreadable. |

What is probed: every TT role in the **first, middle and last** block (`{0, n_layer//2, n_layer-1}`),
every core of those layers. Two tables come out of each snapshot:

- **per core** (`core_stats.csv`) — `mean`, `std`, `absmax`, `norm`, the gradient norm of that step,
  and, in adaptive arms, the mean/min of the rank parameter gating its trailing bond plus how many of
  its entries are still above the threshold (`rank_alive`).
- **per core group** (`core_dist.csv`) — percentiles (p1/p25/p50/p75/p99) of the concatenated entries
  of the cores left of the middle bond, the two centre cores, and the cores right of it. Percentiles
  of the concatenation, not an average of per-core percentiles, which would not be a percentile of
  anything. Cores are read unmasked; the mask is reported separately as `rank_mean`.

The whole snapshot is one `cat` + one `.tolist()`, i.e. a single device synchronize, so it does not
leak into the step-time medians.

## Tracking a run live

Off unless `--track` is given. `--track tensorboard` is the default choice; `--track wandb` uses the
same interface against Weights & Biases.

```bash
python run_experiments.py --track tensorboard
```

then, in another shell (or a Colab cell, see below):

```bash
tensorboard --logdir results/tb
```

Three kinds of run per invocation, each its own subdirectory under `results/tb/<group>/`, so
TensorBoard overlays them and the run selector on the left doubles as the arm filter:

| run | how many | what it carries |
|---|---|---|
| `train-<arm>` | one per training arm | live curves: `train/loss`, `val/loss`, **`train/ppl`, `val/ppl`**, `params/effective`, `rank_loss`, `step_ms`, `lr_mult` every `eval_interval`, plus the core diagnostics (`core/grad_norm`, `core/rank_mean`, `core/<group>/p*`) every `core_diag_interval`. Ends with `final/*` scalars, the generated text sample under TEXT, and an HPARAMS row pairing the arm's config with its final metrics |
| `bench-bench` | one per invocation | the phase-1 sweep as a markdown table under TEXT, plus `epoch_total_min/<cell>`, `step_ms/<cell>`, `peak_step_mb/<cell>` scalars |
| `report-report` | one per invocation | every plot under IMAGES and the csv list under TEXT, once the phases are done |

Perplexity is logged next to the loss because it is the number the arms are judged on; both are
clamped at `exp(min(loss, 20))`, so a diverged arm cannot stretch the chart's y-range to infinity.
Everything is written against `iter`, so arms logged at different cadences line up on one x axis, and
the writer flushes every 30 s -- the point is watching a run while it happens.

**Colab.** TensorBoard runs inside the notebook, no account and no network setup:

```bash
%load_ext tensorboard
```

```bash
%tensorboard --logdir results/tb
```

Start it *before* the training cell and it refreshes on its own while `!python run_experiments.py
--track tensorboard` runs.

**A remote box (A100).** Two machines, so mind which shell each command belongs in.

On the **server**, in the SSH session where the training is running (a second session, or a `tmux`
window, since it keeps running):

```bash
tensorboard --logdir results/tb --port 6006
```

On your **own machine**, in a second terminal, forward the port and leave it running:

```bash
ssh -N -L 6006:localhost:6006 user@host
```

`user@host` is the login the provider gave you -- `ubuntu@203.0.113.42`, `root@ssh5.vast.ai -p 41022`
(rented boxes usually listen on a non-standard port, hence the `-p`). Then open
<http://localhost:6006> locally. No `--bind_all`: the tunnel reaches the server's own localhost, and
binding TensorBoard to every interface would publish the dashboard to the internet instead.

Some providers (RunPod, vast.ai) can expose a port through their web console, which replaces the
tunnel entirely -- then TensorBoard does need `--bind_all` to be reachable from outside.

| flag | default | what it does |
|---|---|---|
| `--track off\|tensorboard\|wandb` | `off` | which backend to log to |
| `--wandb-group NAME` | timestamp | names the `results/tb/<group>` directory, or the wandb group |
| `--wandb-project NAME` | `adaptive-ttm-gpt` | wandb only |
| `--wandb-entity NAME` | your default entity | wandb only |
| `--wandb-mode online\|offline\|disabled` | `online` | wandb only; `offline` writes to `<out>/wandb` for a later `wandb sync`, and is also the automatic fallback when no API key is available, since an interactive login would hang a colab `!python` job |

**Nothing here can kill a run.** `tracking.Run` is a working object whether or not the backend is
installed or reachable; every call swallows its exception after reporting the first one. A missing
package prints one line and the harness continues.

**Cached cells do not log.** Runs are created where the compute happens, so an arm loaded from
`results/runs/*.json` produces no tracking run. Re-run it with `--force` to put it on the dashboard.

## Outputs

`results/runs/*.json` is the cache — one file per benchmark cell and per training arm, keyed by a hash
of the cell and the `TrainConfig`, so an interrupted Colab session resumes and a smoke run can never
be mistaken for a full one.

| file | contents |
|---|---|
| `tables/bench.csv` | per configuration and batch: fwd/bwd step time, projected minutes per epoch, peak memory (fwd / fwd+bwd / fwd+bwd+AdamW step), optimizer-state size, compile diagnostics |
| `tables/train.csv` | per arm: effective params, compression, val loss/ppl, step time, memory |
| `tables/ranks.csv` | one row per arm x TT layer x bond: initial and final rank |
| `tables/rank_summary.csv` | per adaptive arm: `pruned_frac`, mean/min/max rank, dead bonds |
| `tables/core_stats.csv` | one row per arm x snapshot x TT layer x core: mean/std/absmax/norm, gradient norm, rank-parameter state |
| `tables/core_dist.csv` | one row per arm x snapshot x TT layer x core group: percentiles of the core entries |
| `tables/memory.csv` | static footprint next to the measured peaks |
| `plots/1_epoch_{forward,backward}.png`, `2_epoch_total.png` | the CoMERA bar chart |
| `plots/2b_step_time.png` | ms per step (fwd + bwd) — the view the small batches exist for, since an epoch at batch 1 is 128x more steps than at batch 128 |
| `plots/3_pareto_params.png` | val loss vs surviving parameters, one curve per family |
| `plots/4_loss_curves.png`, `4b_pruning_over_training.png` | training dynamics |
| `plots/5_rank_pruned_frac.png`, `5c_rank_heatmap.png`, `5d_rank_by_role.png` | final rank configuration |
| `plots/5e_rank_kept_frac.png` | same layer x bond grid as `5c`, but `kept_frac` instead of the absolute rank. Read this one to judge pruning: the chain ends never start at `max_rank` (`get_uniform_rank` clips them by the mode products), so on the absolute map every arm looks pruned at the edges when nothing was pruned there. |
| `plots/6_cores_<arm>.png` | one figure per arm, one row per probed block: three percentile-band panels for the core values (left half of the chain, the two centre cores, right half), then the per-core gradient norm and rank-parameter mean (log axes wherever the quantity stays positive) |
| `plots/7_mem_static.png`, `8_mem_peak_{forward,backward}.png`, `8b_mem_peak_step.png` | memory footprint. `8b` is the one to read: it includes the AdamW states, i.e. the half of the tensorized saving a fwd+bwd measurement never sees |

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
