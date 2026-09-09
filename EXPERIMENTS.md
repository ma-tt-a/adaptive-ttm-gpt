# Running the experiments

## General flags

| flag | default | what it does |
|---|---|---|
| `--smoke` | off | toy scale for every phase: `n_embd=64`, 2 layers, block 64, batches 8/16, ranks 4/8, 50 iters, 5 bench reps, tensorized arms trained **eager**. Validates the pipeline end to end in minutes. |
| `--phases [bench train]` | `bench train` | which phases may **compute**. Memory, rank and core phases only re-read the cache and always run. Passing `--phases` with no values re-plots everything without touching the GPU. |
| `--force` | off | recompute cached cells instead of loading them. |
| `--out DIR` | `results` | root for `runs/` (cache), `tables/` (csv), `plots/` (png). |
| `--quiet` | off | silence inductor/dynamo compile chatter. |
| `--progress` | off | per-iteration tqdm bars. Off by default: colab `!python` is not a tty, so tqdm writes one line per update and buries the results. |
| `--matmul-precision tf32\|fp32` | `tf32` | `tf32` runs float32 matmuls on the tensor cores, `fp32` keeps full float32 arithmetic. Part of the cache key, so switching recomputes instead of mixing. cuda-only; xpu and cpu ignore it. |
| `--init scaled\|normal` | `scaled` | `scaled` sizes the cores so the *contracted* matrix has std 0.02. `normal` is a literal std=1.0 and diverges — it exists to demonstrate why the scaling does. |

Live tracking is off unless `--track` is given; its flags are in [Tracking](#tracking).

## Model and data

| flag | default | what it does |
|---|---|---|
| `--dataset shakespeare\|fineweb-edu` | `shakespeare` | which corpus. `shakespeare` is char level (65 tokens, 1.0M train tokens, held in memory). `fineweb-edu` is [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) tokenized with the gpt2 BPE (50257 tokens) into `data/fineweb_edu_<subset>/` and read back as a memmap. |
| `--data-tokens N` | `100_000_000` | fineweb-edu only: how many **training** tokens the corpus must hold. Only the parquet row groups actually needed are downloaded, so this is the knob that decides the wait and the disk cost (2 bytes/token: 1e9 = 2 GB, 1e10 = 20 GB). |
| `--data-subset` | `sample-10BT` | fineweb-edu only: which slice of the repo the tokens come from — `sample-10BT`, `sample-100BT`, `sample-350BT`, `full` (~1.3T). A ceiling on what is available, not a download size; each subset gets its own directory. |
| `--model smoke\|base\|gpt2-small` | `base` | named `n_layer` / `n_head` / `n_embd` / `block_size` quadruple. `gpt2-small` is the published 124M configuration: 12 layers, 12 heads, 768 wide, 1024 context. The flags below still override it. |
| `--block-size N` | `128` (`gpt2-small` 1024) | maximum sequence length, i.e. the context trained on and the `T` in every tokens-per-step number. |
| `--lr LR` | `5e-5` (`comera.LR_ORIGIN`) | learning rate of everything that is not a TT core: embeddings, norms, biases, and every dense layer of the dense baseline. |
| `--lr-tensor LR` | `1e-4` (`comera.LR_TENSOR`) | learning rate of the TT cores. CoMERA trains them faster than the rest; rank parameters get their own rate from `--lr-ranks`. |
| `--warmup N` | `100` (smoke `5`) | linear warmup iterations before the cosine decay to `min_lr_frac` of the peak. |
| `--train-tokens N` | — | train for this many tokens instead of a fixed iteration count: `max_iters = ceil(N / (micro x accum x block))`. Mutually exclusive with `--iters`; only `max_iters` reaches the cache key, so the two address the same cached run. |

### Preparing the corpus

Optional — the first training run would download it anyway, into the same directory:

```bash
python data.py --dataset fineweb-edu --tokens 500000000
python data.py --dataset fineweb-edu --tokens 20000000000 --subset sample-100BT
```

The corpus is 100M-token shards plus a `meta.json` recording the completed shards and the
`(parquet file, row group)` to read next, rewritten after every shard. Two consequences:

- **Resumable.** An interruption drops the partial shard and restarts from the checkpointed row
  group — one HTTP range request, no re-download of what is already tokenized. Hence `pyarrow` over
  `HfFileSystem` rather than `datasets`' streaming iterator, which can only resume by re-reading
  from the beginning.
- **Extensible.** `--data-tokens` is a floor: asking for 10B in a directory holding 1B appends
  shards, asking for less reads a prefix. The validation split is written once, before any training
  shard, and never grows, so val losses stay comparable across budgets.

### The token budget

Phase 2 prints the budget before the first arm — the number that decides whether the run is worth
launching, and the one thing the loss curve cannot tell you afterwards:

```
budget: 131,072 tokens/step (8 micro x 16 accum x 1024 ctx) x 15,000 iters = 1.97B tokens per arm
        fineweb-edu train split 1.99B tokens -> 0.99 epochs; warmup 700 iters (91.75M tokens, 4.7% of the run)
```

More than ~1.5 passes over the corpus prints a warning: at this size repetition is a choice, and
`--data-tokens` is cheaper than an extra epoch. The single-epoch case is `--train-tokens` equal to
`--data-tokens`:

```bash
python run_experiments.py --phases train --dataset fineweb-edu --data-tokens 2000000000 \
    --model gpt2-small --micro-batch 8 --grad-accum 16 --train-tokens 2000000000 \
    --warmup 700 --lr 6e-4 --lr-tensor 6e-4 --arms dense
```

## Wall-clock and memory benchmark

| flag | default | what it does |
|---|---|---|
| `--batches N ...` | `1 8 32 64 128` (smoke `8 16`) | batch sizes on the x axis of the epoch-time, step-time and peak-memory plots. The small ones are where compile is legible: by batch 128 the matmuls amortise the launch overhead and every mode converges. |
| `--compile-modes eager compile cudagraph` | all three | `compile` is inductor fusion, `cudagraph` is fusion + CUDA Graphs (a no-op on xpu). |
| `--bench-rank N` | `32` | `max_rank` of the benchmarked tensorized model. |
| `--bench-reps N` | `30` (smoke `5`) | timed forward/backward repetitions per cell, after 5 warmup steps (2 in smoke). |
| `--kinds dense tensorized` | both | which models phase 1 may **compute**; the other is still read from the cache, so the plots keep both series. One kind per process is the only route to a memory number nothing else is holding — dynamo caches and CUDA-Graph pools do not survive an exit, whatever they survive inside one run. |

### Memory scenarios

| column | scenario | resident when the peak is taken |
|---|---|---|
| `peak_infer_mb` | inference forward | weights + transient activations. Under `model.eval()` + `no_grad`, **before the optimizer exists** — once AdamW's states are allocated there is no way to take a forward peak without them. Deliberately not a prefix of the two below. |
| `peak_bwd_mb` | forward + backward | weights + AdamW states + gradients + saved activations |
| `peak_step_mb` | the full training step | the above + AdamW's `_foreach_` temporaries. The honest training comparison. |

## Training

| flag | default | what it does |
|---|---|---|
| `--ranks 4 8 16 32` | `4 8 16 32` (smoke `4 8`) | the rank ladder: one `uniform-r<R>` arm plus one adaptive arm per `lr_rank` for each entry. |
| `--lr-ranks 3e-3 1e-2` | `3e-3 1e-2` | learning rates of the rank parameters. Each value is its own adaptive family, i.e. its own Pareto curve; larger prunes more aggressively. |
| `--gamma G` | `0.1` (`comera.GAMMA`) | weight of the rank loss in `comera_loss`. Fixed across arms so `lr_rank` is the only pruning-aggressiveness axis. |
| `--arms NAME ...` | all | subset of arm names, e.g. `--arms dense uniform-r16 adaptive-r16-lr0.01`. |
| `--train-compile-mode eager\|compile\|cudagraph` | `compile` (smoke `eager`) | how the tensorized arms are compiled; **dense always trains eager**. `cudagraph` is not the default: CUDA Graphs replay a captured graph whose parameters AdamW mutates outside it, which inductor skips silently. Phase 1 is where the graph win is measured. |
| `--iters N` | `1500` (smoke `50`) | optimizer steps per arm, each consuming `--grad-accum` micro-batches. |
| `--micro-batch N` | `32` (smoke `8`) | sequences per forward. This is what activation memory scales with. |
| `--grad-accum N` | `1` | forward/backward passes accumulated before each optimizer step. |
| `--log-interval N` | `500` (smoke `25`) | iterations between printed progress lines. Losses are recorded every 100 iterations for the plots regardless. |

### Gradient accumulation

The batch the optimizer steps on is `--micro-batch x --grad-accum`, so a batch that does not fit is
reached by raising the accumulation, not the micro-batch: `--micro-batch 16 --grad-accum 8` trains
on 128 sequences while holding 16 sequences' worth of activations. The header prints the arithmetic
(`batch: 16 micro x 8 accum = 128`). Two implementation details:

- The per-micro loss is divided by `--grad-accum`, so the accumulated gradient is the mean the full
  batch would have produced, not its sum.
- CoMERA's rank loss is added on the **last**
  micro-step only. That keeps it undivided and calls `comera.rank_loss` once per optimizer step
  rather than once per forward.

`step_time_s` therefore covers a whole step, accumulation included.

## Core diagnostics

| flag | default | what it does |
|---|---|---|
| `--core-diag-interval N` | `100` (smoke `10`) | iterations between core snapshots; `0` disables. Deliberately **not** in the cache key, since retuning the diagnostics would otherwise invalidate hours of cached training. A cached arm keeps whatever diagnostics it was trained with; `--force` resamples it. |
| `--core-diag-role c_attn\|attn_proj\|c_fc\|mlp_proj` | `c_fc` | which role the per-arm figure draws. Every probed role reaches `core_stats.csv` regardless; only the plot is narrowed, because a panel with 4 roles x 2d cores is unreadable. |

## Tracking

Off unless `--track` is given. `tensorboard` is the default choice; `wandb` uses the same interface.

| flag | default | what it does |
|---|---|---|
| `--track off\|tensorboard\|wandb` | `off` | which backend to log to |
| `--wandb-group NAME` | timestamp | names the `results/tb/<group>` directory, or the wandb group |
| `--wandb-project NAME` | `adaptive-ttm-gpt` | wandb only |
| `--wandb-entity NAME` | your default entity | wandb only |
| `--wandb-mode online\|offline\|disabled` | `online` | wandb only. `offline` writes to `<out>/wandb` for a later `wandb sync`, and is the automatic fallback when no API key is available — an interactive login would hang a colab `!python` job |

```bash
python run_experiments.py --track tensorboard
tensorboard --logdir results/tb          # in another shell
```

Three kinds of run per invocation, each its own subdirectory under `results/tb/<group>/`, so
TensorBoard overlays them and the run selector doubles as an arm filter:

| run | how many | what it carries |
|---|---|---|
| `train-<arm>` | one per training arm | `train/loss`, `val/loss`, **`train/ppl`, `val/ppl`**, `params/effective`, `rank_loss`, `step_ms`, `lr_mult` every `eval_interval`, plus core diagnostics (`core/grad_norm`, `core/rank_mean`, `core/<group>/p*`) every `core_diag_interval`. Ends with `final/*` scalars, the generated sample under TEXT, and an HPARAMS row pairing the config with the final metrics |
| `bench-bench` | one per invocation | the phase-1 sweep as a markdown table under TEXT, plus `epoch_total_min/<cell>`, `step_ms/<cell>`, `peak_step_mb/<cell>` |
| `report-report` | one per invocation | every plot under IMAGES and the csv list under TEXT, once the phases are done |


### TensorBoard

TensorBoard runs inside the notebook. Start it *before* the training cell and it refreshes on its own while `!python run_experiments.py --track tensorboard`
runs:

```bash
%load_ext tensorboard
%tensorboard --logdir results/tb
```

## Outputs

`results/runs/*.json` is the cache: one file per benchmark cell and per training arm, keyed by a
hash of the cell and the `TrainConfig`.

| file | contents |
|---|---|
| `tables/bench.csv` | per configuration and batch: fwd/bwd step time, projected minutes per epoch, the three peaks, optimizer-state size, compile diagnostics |
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
| `plots/5e_rank_kept_frac.png` | the `5c` layer x bond grid in `kept_frac` instead of absolute rank. **Read this one to judge pruning:** the chain ends never start at `max_rank` (`get_uniform_rank` clips them by the mode products), so on the absolute map every arm looks pruned at the edges where nothing was pruned |
| `plots/6_cores_<arm>.png` | one figure per arm, one row per probed block: three percentile-band panels for the core values (left half, the two centre cores, right half), then the per-core gradient norm and rank-parameter mean (log axes wherever the quantity stays positive) |
| `plots/7_mem_static.png`, `8_mem_peak_{inference,backward}.png`, `8b_mem_peak_step.png` | memory footprint. `8b` is the training number, AdamW states included — the half of the tensorized saving a fwd+bwd measurement never sees. `8_mem_peak_inference.png` is the deployment number: no states, no gradients |

## Warnings

The summary block prints `!` lines that mean a number should not be trusted:

- `dynamo compiled 0 frames` — compilation fell back to eager, most likely on dynamo's
  `cache_size_limit`. The cell is not measuring what it claims.
- `inductor skipped CUDA Graphs Nx` — the cudagraph cell is really a fusion cell. Expected on xpu,
  which has no CUDA-graph backend.
- `nothing pruned at all`