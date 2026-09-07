"""
fp32 vs tf32 vs bf16, for the dense and the tensorized GPT.

run_experiments.py measures one speed axis, torch.compile. This is the other
one. For a tensorized model it is not the routine "turn on bf16" question: a
TTLinear forward is a chain of einsums whose intermediates (A_d, B_d, T_1) are
contracted against each other, so reduced precision has more places to
accumulate error than a single dense matmul has -- while also being where a TT
layer could win back part of the per-step penalty it pays against dense. So
every cell reports speed *and* the error of its own logits, and the two are
plotted side by side.

The grid is {dense, tensorized} x {fp32, tf32, bf16} x {eager, compile,
cudagraph} over a batch ladder. Cells are cached to results/runs/prec_*.json
and skipped on re-run, exactly like phase 1.

    python bench_precision.py --smoke                 # toy scale, minutes
    python bench_precision.py --batches 1 8 32 128    # the real grid
    python bench_precision.py --plots-only            # re-plot the cache

The three precisions are three different mechanisms and must not be conflated:

  fp32  float32 storage, float32 matmuls ("highest", allow_tf32 off)
  tf32  float32 storage, TF32 tensor-core matmuls ("high", allow_tf32 on).
        cuda + sm_80 only; on xpu/cpu the cell is recorded as skipped rather
        than silently producing an fp32 number under a tf32 label
  bf16  fp32 master weights under autocast(bfloat16). No GradScaler: bf16 has
        fp32's exponent range. Forward and loss are inside the autocast block,
        backward outside it, per torch's AMP contract

Reading the output: a speedup is never worth reading without the error next to
it. 9b (speedup) and 9d (logit relative error) are the pair.
"""
import argparse
import contextlib
import math
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch as t
from tqdm.auto import tqdm

import comera
import run_experiments as rx
from data import CharDataset
from run_experiments import (Arm, StepTimer, TrainConfig, build_model,
                             cached_or_compute, compile_model,
                             compiled_frames, cudagraph_skips, graph_breaks,
                             hash_payload, median, peak_memory, reset_memory,
                             MODE_FROM_TAG, MODE_TAG, MODE_LABEL)
from utils import get_device

PRECISIONS = ["fp32", "tf32", "bf16"]
KINDS = ["dense", "tensorized"]
BATCHES = [1, 8, 32, 64, 128]
REPS = 30
WARMUP = 5
MAX_RANK = rx.BENCH_MAX_RANK

# bumped whenever a cell measures something different under the same flags, so
# stale cells recompute instead of being silently mixed in with fresh ones
PRECISION_PROTOCOL = 1

# one colour family per precision, one shade per compile mode, so a
# configuration keeps its colour across every plot here
PRECISION_COLORS = {"fp32": ["#08306b", "#2171b5", "#6baed6"],
                    "tf32": ["#7f2704", "#e6550d", "#fdae6b"],
                    "bf16": ["#00441b", "#238b45", "#a1d99b"]}


# ============================================================================
# the precisions themselves
# ============================================================================

def precision_available(name: str, device: t.device) -> Optional[str]:
    """
    None if the precision can run here, else why it cannot
    """
    if name == "fp32":
        return None
    if name == "tf32":
        if device.type != "cuda":
            return f"tf32 needs cuda, device is {device.type}"
        if t.cuda.get_device_capability()[0] < 8:
            return "tf32 needs sm_80 (Ampere) or newer"
        return None
    if name == "bf16":
        if device.type == "cuda" and not t.cuda.is_bf16_supported():
            return "bf16 unsupported by this cuda device"
        return None
    raise ValueError(f"unknown precision {name}")


def cell_available(precision: str, mode: Optional[str], device: t.device,
                   allow_xpu_bf16_compile: bool = False) -> Optional[str]:
    """
    None if the (precision, compile mode) pair can run here, else why not.

    The pair matters, not just the precision: on xpu the inductor backend emits
    a kernel using SPV_KHR_bfloat16, which the Level Zero driver rejects, and
    it does so by aborting the process -- not by raising, so cached_or_compute
    cannot contain it and one cell takes the whole grid down with it. Quarantine
    it here; --allow-xpu-bf16-compile is the escape hatch for a fixed driver.
    """
    why = precision_available(precision, device)
    if why is not None:
        return why
    if (precision == "bf16" and mode is not None and device.type == "xpu"
            and not allow_xpu_bf16_compile):
        return "bf16 + inductor aborts the xpu driver (SPV_KHR_bfloat16)"
    return None


@contextlib.contextmanager
def precision_ctx(name: str, device: t.device):
    """
    The global float32 flags for one cell, restored on the way out.

    Restoring matters: these are process-global, and a leaked allow_tf32 would
    relabel every later cell rather than fail.
    """
    old_matmul = t.get_float32_matmul_precision()
    old_cuda = t.backends.cuda.matmul.allow_tf32
    old_cudnn = t.backends.cudnn.allow_tf32
    tf32 = (name == "tf32")
    try:
        t.set_float32_matmul_precision("high" if tf32 else "highest")
        t.backends.cuda.matmul.allow_tf32 = tf32
        t.backends.cudnn.allow_tf32 = tf32
        yield
    finally:
        t.set_float32_matmul_precision(old_matmul)
        t.backends.cuda.matmul.allow_tf32 = old_cuda
        t.backends.cudnn.allow_tf32 = old_cudnn


def autocast_ctx(name: str, device: t.device):
    """
    The forward-side half of a precision: autocast for bf16, nothing otherwise
    """
    if name != "bf16":
        return contextlib.nullcontext()
    return t.autocast(device_type=device.type, dtype=t.bfloat16)


# ============================================================================
# one cell
# ============================================================================

# fp32/eager logits per (kind, batch), the yardstick for every other cell of
# the same shape. Computed here rather than read out of the fp32 cache record
# so that a cell is measurable on its own, cache hit or not.
_REFERENCE: Dict[Tuple[str, int], Tuple[t.Tensor, float]] = {}


def reference(kind: str, batch: int, cfg: TrainConfig, ds: CharDataset,
              device: t.device, max_rank: int) -> Tuple[t.Tensor, float]:
    """
    fp32 eager logits and loss on the accuracy batch
    """
    key = (kind, batch)
    if key not in _REFERENCE:
        model = build_model(arm_of(kind, max_rank), cfg, ds, device)
        X, Y = accuracy_batch(batch, cfg, ds, device)
        with precision_ctx("fp32", device), t.no_grad():
            logits, loss = model(X, Y)
        _REFERENCE[key] = (logits.float(), float(loss))
        del model
        reset_memory(device)
    return _REFERENCE[key]


def arm_of(kind: str, max_rank: int) -> Arm:
    return Arm(kind, tensorized=(kind == "tensorized"), max_rank=max_rank)


def accuracy_batch(batch: int, cfg: TrainConfig, ds: CharDataset,
                   device: t.device):
    """
    The one batch every accuracy number is computed on. Its own generator seed,
    so it does not move when the timing batch does.
    """
    gen = t.Generator().manual_seed(cfg.seed + 991)
    return ds.get_batch("val", batch, cfg.block_size, device, generator=gen)


def precision_cell(kind: str, precision: str, mode: Optional[str], batch: int,
                   cfg: TrainConfig, ds: CharDataset, device: t.device,
                   max_rank: int, reps: int, warmup: int,
                   allow_xpu_bf16_compile: bool = False) -> dict:
    """
    One (kind, precision, compile mode, batch) cell.

    Deliberately the same protocol as run_experiments.bench_cell -- separate
    timing and memory passes, forward and backward timed as two StepTimer
    blocks -- so the two tables are comparable, plus an accuracy pass.
    """
    rec = {
        "kind": kind,
        "precision": precision,
        "compile_mode": MODE_TAG[mode],
        "label": f"{kind}, {precision}, {MODE_LABEL[mode]}",
        "batch": batch,
        "block_size": cfg.block_size,
        "max_rank": max_rank if kind == "tensorized" else None,
        "skipped": cell_available(precision, mode, device,
                                  allow_xpu_bf16_compile),
        "config_hash": want_hash(kind, precision, mode, batch, max_rank, cfg,
                                 reps, warmup),
    }
    if rec["skipped"] is not None:
        return rec

    model = build_model(arm_of(kind, max_rank), cfg, ds, device)
    raw_model = model
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    gen = t.Generator().manual_seed(cfg.seed)
    X, Y = ds.get_batch("train", batch, cfg.block_size, device, generator=gen)

    with precision_ctx(precision, device):
        model, compile_time = compile_model(model, mode)
        cast = autocast_ctx(precision, device)

        first_step_s = None
        for i in range(max(warmup, 1)):
            # host timing on purpose: this measures the JIT, which runs on cpu
            t0 = time.perf_counter()
            with cast:
                _, loss = model(X, Y)
            loss.backward()          # backward outside the autocast block
            raw_model.zero_grad(set_to_none=True)
            if i == 0:
                first_step_s = time.perf_counter() - t0

        # timing pass. Forward and backward as two separate blocks, so
        # StepTimer's synchronize lands between them: that serialises the
        # halves and inflates each slightly against one fused measurement,
        # which is the price of reporting them separately at all.
        fwd, bwd = [], []
        for _ in range(reps):
            timer = StepTimer(device)
            with timer, cast:
                _, loss = model(X, Y)
            fwd.extend(timer.times)
            timer = StepTimer(device)
            with timer:
                loss.backward()
            bwd.extend(timer.times)
            raw_model.zero_grad(set_to_none=True)

        # accuracy pass, on the untrained weights: build_model seeds from
        # cfg.seed, so this model and the fp32 reference are bit-identical
        # before the first matmul and the whole difference is the arithmetic.
        # It runs *before* the memory pass, which takes real optimizer steps
        # and would otherwise move the weights out from under the comparison.
        Xa, Ya = accuracy_batch(batch, cfg, ds, device)
        with t.no_grad(), cast:
            logits, loss_a = model(Xa, Ya)

        # memory pass, kept apart because reset_peak_memory_stats and
        # empty_cache perturb the numbers above. The optimizer is part of it:
        # AdamW's two states per parameter are half of what a tensorized model
        # saves, and fwd+bwd alone never sees them.
        opt = comera.make_optimizer(raw_model)
        with cast:
            _, loss = model(X, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()  # Adam allocates its states lazily on the first step

        peak_fwd = peak_bwd = peak_step = 0
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            reset_memory(device)
            with cast:
                _, loss = model(X, Y)
            peak_fwd = max(peak_fwd, peak_memory(device))
            loss.backward()
            peak_bwd = max(peak_bwd, peak_memory(device))
            opt.step()
            peak_step = max(peak_step, peak_memory(device))
        opt_state_bytes = sum(v.numel() * v.element_size()
                              for st in opt.state.values()
                              for v in st.values() if t.is_tensor(v))
        opt.zero_grad(set_to_none=True)

    ref_logits, ref_loss = reference(kind, batch, cfg, ds, device, max_rank)
    err = (logits.float() - ref_logits).norm().item()
    rec["logit_rel_err"] = err / ref_logits.norm().item()
    rec["logit_max_abs_err"] = (logits.float() -
                                ref_logits).abs().max().item()
    rec["loss"] = float(loss_a)
    rec["loss_ref"] = ref_loss
    rec["loss_abs_err"] = abs(float(loss_a) - ref_loss)

    steps_per_epoch = max(1, len(ds.splits["train"]) //
                          (batch * cfg.block_size))
    fwd_s, bwd_s = median(fwd), median(bwd)
    rec.update({
        "fwd_s": fwd_s,
        "bwd_s": bwd_s,
        "step_s": fwd_s + bwd_s,
        "step_ms": (fwd_s + bwd_s) * 1e3,
        "tokens_per_s": batch * cfg.block_size / (fwd_s + bwd_s),
        "steps_per_epoch": steps_per_epoch,
        "epoch_total_min": (fwd_s + bwd_s) * steps_per_epoch / 60,
        "peak_fwd_mb": peak_fwd / 1e6,
        "peak_bwd_mb": peak_bwd / 1e6,
        "peak_step_mb": peak_step / 1e6,
        "param_mb": param_bytes / 1e6,
        "opt_state_mb": opt_state_bytes / 1e6,
        "first_step_s": first_step_s,
        "compile_time_s": compile_time,
        "graph_breaks": graph_breaks() if mode is not None else 0,
        "compiled_frames": compiled_frames() if mode is not None else 0,
        "cudagraph_skips": (cudagraph_skips()
                            if mode == "reduce-overhead" else 0),
        # CUDA Graphs allocate from a private pool that max_memory_allocated
        # does not account for the same way, so these rows' memory is not
        # comparable with the others and the memory plot drops them
        "memory_comparable": mode != "reduce-overhead",
    })
    del model, raw_model, opt
    reset_memory(device)
    return rec


def want_hash(kind, precision, mode, batch, max_rank, cfg, reps, warmup) -> str:
    return hash_payload(kind, precision, MODE_TAG[mode], batch, max_rank,
                        asdict(cfg), reps, warmup, PRECISION_PROTOCOL)


# ============================================================================
# the grid
# ============================================================================

def run_precision(cfg: TrainConfig, ds: CharDataset, device: t.device,
                  kinds: List[str], precisions: List[str],
                  modes: List[Optional[str]], batches: List[int],
                  max_rank: int, reps: int, warmup: int, force: bool,
                  allowed: bool,
                  allow_xpu_bf16_compile: bool = False) -> List[dict]:
    records = []
    cells = [(k, p, m, b) for k in kinds for p in precisions
             for m in modes for b in batches]
    for kind, prec, mode, batch in tqdm(cells, desc="precision",
                                        disable=rx.BAR_DISABLE):
        tag = f"{kind}_{prec}_{MODE_TAG[mode]}_b{batch}"
        want = want_hash(kind, prec, mode, batch, max_rank, cfg, reps, warmup)
        path = os.path.join(rx.RUNS_DIR, f"prec_{tag}_{want}.json")
        rec = cached_or_compute(
            path, want,
            lambda k=kind, p=prec, m=mode, b=batch: precision_cell(
                k, p, m, b, cfg, ds, device, max_rank, reps, warmup,
                allow_xpu_bf16_compile),
            force, tag, allowed)
        if rec is None:
            continue
        records.append(rec)
        if rec.get("skipped"):
            tqdm.write(f"  {tag}: skipped -- {rec['skipped']}")
        elif "wall_clock_s" in rec:
            warn = "  !cudagraph skipped" if rec["cudagraph_skips"] else ""
            tqdm.write(f"  {tag}: {rec['step_ms']:.2f} ms/step  "
                       f"peak {rec['peak_step_mb']:.0f} MB  "
                       f"rel err {rec['logit_rel_err']:.2e}{warn}")
    return records


def speedup_field(records: List[dict]) -> List[dict]:
    """
    Every cell against the fp32 eager cell of the same kind and batch
    """
    for r in records:
        base = next((b for b in records
                     if b["kind"] == r["kind"] and b["batch"] == r["batch"]
                     and b["precision"] == "fp32"
                     and b["compile_mode"] == "eager"
                     and b.get("step_s")), None)
        r["speedup_vs_fp32_eager"] = (base["step_s"] / r["step_s"]
                                      if base and r.get("step_s")
                                      else float("nan"))
    return records


COLUMNS = ["kind", "precision", "compile_mode", "batch", "block_size",
           "max_rank", "skipped", "fwd_s", "bwd_s", "step_s", "step_ms",
           "speedup_vs_fp32_eager", "tokens_per_s", "steps_per_epoch",
           "epoch_total_min", "peak_fwd_mb", "peak_bwd_mb", "peak_step_mb",
           "param_mb", "opt_state_mb", "logit_rel_err", "logit_max_abs_err",
           "loss", "loss_ref", "loss_abs_err", "first_step_s",
           "compile_time_s", "graph_breaks", "compiled_frames",
           "cudagraph_skips", "memory_comparable", "wall_clock_s"]


# ============================================================================
# plots
# ============================================================================

def series(records: List[dict], precisions: List[str],
           modes: List[Optional[str]], kind: str, batches: List[int],
           field: str, comparable_only: bool = False):
    """
    (labels, colours, values[j][i]) for rx.grouped_bars, one series per
    (precision, compile mode)
    """
    labels, colors, values = [], [], []
    for prec in precisions:
        for j, mode in enumerate(modes):
            if comparable_only and mode == "reduce-overhead":
                continue
            cells = [next((r for r in records
                           if r["kind"] == kind and r["precision"] == prec
                           and r["compile_mode"] == MODE_TAG[mode]
                           and r["batch"] == b), None) for b in batches]
            row = [c.get(field) if c else None for c in cells]
            if all(v is None for v in row):
                continue
            labels.append(f"{prec}, {MODE_TAG[mode]}")
            colors.append(PRECISION_COLORS[prec][j % 3])
            values.append(row)
    return labels, colors, values


def panels(records, kinds, precisions, modes, batches, field, title, ylabel,
           name, fmt="{:.1f}", comparable_only=False, log=True):
    plt = rx._plt()
    kinds = [k for k in kinds if any(r["kind"] == k for r in records)]
    if not kinds:
        return
    fig, axes = plt.subplots(1, len(kinds), figsize=(7.5 * len(kinds), 4.6),
                             squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        labels, colors, values = series(records, precisions, modes, kind,
                                        batches, field, comparable_only)
        rx.grouped_bars(ax, batches, labels, values, fmt=fmt, colors=colors)
        if log:
            rx.maybe_log_y(ax, values)
        ax.set_title(kind)
        ax.set_xlabel("batch size")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, ncols=2)
    fig.suptitle(title)
    rx.save(fig, name)


def plot_all(records: List[dict], kinds: List[str], precisions: List[str],
             modes: List[Optional[str]], batches: List[int]):
    live = [r for r in records if not r.get("skipped")]
    if not live:
        print("  nothing to plot")
        return
    precisions = [p for p in precisions
                  if any(r["precision"] == p for r in live)]
    panels(live, kinds, precisions, modes, batches, "step_ms",
           "Time per step (fwd + bwd)", "ms / step",
           "9a_precision_step_time.png", fmt="{:.1f}")
    panels(live, kinds, precisions, modes, batches, "speedup_vs_fp32_eager",
           "Speedup against fp32 eager, same kind and batch",
           "x faster", "9b_precision_speedup.png", fmt="{:.2f}", log=False)
    panels(live, kinds, precisions, modes, batches, "peak_step_mb",
           "Peak memory, fwd + bwd + AdamW step "
           "(CUDA-Graph rows dropped: private pool, not comparable)",
           "MB", "9c_precision_memory.png", fmt="{:.0f}",
           comparable_only=True)
    panels(live, kinds, precisions, modes, batches, "logit_rel_err",
           "Relative logit error against the fp32 eager model",
           "||dY|| / ||Y||", "9d_precision_error.png", fmt="{:.1e}")
    print(f"  plots -> {rx.PLOTS_DIR}")


def summarize(records: List[dict], batches: List[int]):
    live = [r for r in records if not r.get("skipped")]
    skipped = [r for r in records if r.get("skipped")]
    if skipped:
        reasons = sorted({r["skipped"] for r in skipped})
        print(f"  {len(skipped)} cells skipped: " + "; ".join(reasons))
    blind = [r for r in live
             if r["compile_mode"] != "eager" and r["compiled_frames"] == 0]
    if blind:
        print(f"  ! {len(blind)} compiled cells compiled no frames -- dynamo "
              f"fell back to eager")
    flat = [r for r in live
            if r["precision"] != "fp32" and r.get("logit_rel_err") == 0.0]
    if flat:
        print(f"  ! {len(flat)} non-fp32 cells have zero logit error -- the "
              f"precision never engaged")
    if not live:
        return
    print()
    rows = sorted(live, key=lambda r: (r["kind"], r["batch"],
                                       PRECISIONS.index(r["precision"]),
                                       list(MODE_TAG.values())
                                       .index(r["compile_mode"])))
    cols = ["kind", "precision", "compile_mode", "batch", "step_ms",
            "speedup_vs_fp32_eager", "peak_step_mb", "logit_rel_err"]
    print(rx.markdown_table(rows, cols,
                            fmt={"step_ms": ".2f",
                                 "speedup_vs_fp32_eager": ".2f",
                                 "peak_step_mb": ".1f",
                                 "logit_rel_err": ".2e"}))


# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="toy scale, to validate the grid end to end")
    ap.add_argument("--plots-only", action="store_true",
                    help="re-plot the cache without touching the GPU")
    ap.add_argument("--force", action="store_true",
                    help="recompute cells that are already cached")
    ap.add_argument("--kinds", nargs="*", default=KINDS, choices=KINDS)
    ap.add_argument("--precisions", nargs="*", default=PRECISIONS,
                    choices=PRECISIONS)
    ap.add_argument("--compile-modes", nargs="*",
                    default=["eager", "compile", "cudagraph"],
                    choices=["eager", "compile", "cudagraph"])
    ap.add_argument("--batches", nargs="*", type=int, default=None,
                    help=f"default {' '.join(map(str, BATCHES))}")
    ap.add_argument("--rank", type=int, default=MAX_RANK,
                    help="max_rank of the tensorized model")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--allow-xpu-bf16-compile", action="store_true",
                    help="run the bf16 x compiled cells on xpu. They are "
                         "skipped by default because the driver aborts the "
                         "whole process on the kernel inductor emits")
    ap.add_argument("--progress", action="store_true",
                    help="show the tqdm bar (off by default)")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress inductor/dynamo compile chatter")
    ap.add_argument("--out", default=rx.RESULTS)
    args = ap.parse_args()
    # a driver-level abort takes the process down without unwinding, so buffered
    # progress lines would be lost with it
    sys.stdout.reconfigure(line_buffering=True)

    rx.RESULTS = args.out
    rx.RUNS_DIR = os.path.join(args.out, "runs")
    rx.TABLES_DIR = os.path.join(args.out, "tables")
    rx.PLOTS_DIR = os.path.join(args.out, "plots")
    os.makedirs(rx.RUNS_DIR, exist_ok=True)

    cfg = TrainConfig()
    batches, reps, warmup = BATCHES, REPS, WARMUP
    if args.smoke:
        cfg = cfg.smoke()
        batches, reps, warmup = [8, 16], 5, 2
    if args.batches:
        batches = args.batches
    if args.reps is not None:
        reps = args.reps
    if args.warmup is not None:
        warmup = args.warmup
    if args.progress:
        rx.BAR_DISABLE = False
    if args.quiet:
        import logging
        for name in ("torch._inductor", "torch._dynamo", "torch._functorch"):
            logging.getLogger(name).setLevel(logging.ERROR)

    device = get_device()
    t.manual_seed(cfg.seed)
    ds = CharDataset()
    modes = [MODE_FROM_TAG[m] for m in args.compile_modes]
    max_rank = min(args.rank, 8) if args.smoke else args.rank

    n = len(args.kinds) * len(args.precisions) * len(modes) * len(batches)
    print(f"device={device}  n_layer={cfg.n_layer} n_embd={cfg.n_embd} "
          f"block={cfg.block_size} rank={max_rank}")
    for p in args.precisions:
        whys = {cell_available(p, m, device, args.allow_xpu_bf16_compile)
                for m in modes}
        ok = sorted(w for w in whys if w is not None)
        print(f"  {p}: {'ok' if not ok else '; '.join(ok)}")
    print(f"{n} cells = {len(args.kinds)} kinds x {len(args.precisions)} "
          f"precisions x {len(modes)} modes x {len(batches)} batches\n")

    records = run_precision(cfg, ds, device, args.kinds, args.precisions,
                            modes, batches, max_rank, reps, warmup,
                            args.force, not args.plots_only,
                            args.allow_xpu_bf16_compile)
    records = speedup_field(records)

    path = os.path.join(rx.TABLES_DIR, "precision.csv")
    rx.write_csv(path, records, COLUMNS)
    print(f"\n  table -> {path}")
    plot_all(records, args.kinds, args.precisions, modes, batches)
    summarize(records, batches)


if __name__ == "__main__":
    main()
