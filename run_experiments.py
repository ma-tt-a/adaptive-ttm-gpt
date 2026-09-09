"""
Dense vs tensorized GPT on tiny-shakespeare, in CoMERA's comparison protocol.

Five phases. The first two compute and cache to results/runs/*.json, so an
interrupted colab session resumes where it stopped (--force recomputes):

  1 bench   6 configurations -- {dense, tensorized} x {eager, compiled w/o
            CUDAGraph, compiled w/ CUDAGraph} -- timed across batch sizes,
            forward and backward separately, reported as projected minutes per
            epoch. This is the figure from the CoMERA paper.
  2 train   only what is worth training: dense eager, plus compiled tensorized
            arms over a rank ladder and two rank-pruning aggressivenesses.
            Pareto curves of val loss against surviving parameters.
  3 memory  static parameter footprint of the trained models, and peak training
            memory from phase 1 (per batch, forward and backward).
  4 ranks   what the adaptive scheme actually did: final rank distributions and
            the pruned fraction as a function of lr_rank and starting max_rank.
  5 cores   how the TT cores themselves behave during training: mean / std /
            |max| / gradient norm of every core of the probed layers (every
            role in the first, middle and last block), sampled every
            core_diag_interval iterations.

    python run_experiments.py --smoke          # toy scale, minutes
    python run_experiments.py --phases bench   # cheap, run this first
    python run_experiments.py                  # everything

Phases 3, 4 and 5 compute nothing of their own; they re-plot the phase-1/2 cache,
so they are free to re-run after a plotting change and they work even when the
compute phases are deselected.

Why the split: eager and compiled are numerically the same model, so training
every arm in every compile mode buys nothing but hours. Speed and memory are
measured once, across batch sizes; quality is measured once, compiled only.
"""
import argparse
import csv
import gc
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch as t
import torch._dynamo
from tqdm.auto import tqdm

import comera
import data
import tracking
from data import DATASETS, get_dataset, human_tokens
from gpt import GPT, GPTConfig, MODEL_PRESETS, TT_SHAPES
from tensorized_layers import TTLinear
from utils import get_device, device_module, get_uniform_rank

# tqdm disable flag: True = off (the default), False = on via --progress.
# Off unconditionally rather than by tty detection: under `!python` colab pipes
# stdout so every bar update becomes its own line, and inside a notebook cell
# tqdm.auto resolves to the widget version, which renders either way.
BAR_DISABLE = True

# None = eager; the strings are torch.compile modes. MODE_TAG is what goes into
# filenames and code, MODE_LABEL is what a reader of a plot sees.
MODE_TAG = {None: "eager", "default": "compile",
            "reduce-overhead": "cudagraph"}
MODE_LABEL = {None: "eager",
              "default": "compiled w/o CUDAGraph",
              "reduce-overhead": "compiled w/ CUDAGraph"}
MODE_FROM_TAG = {v: k for k, v in MODE_TAG.items()}

RESULTS = "results"
RUNS_DIR = os.path.join(RESULTS, "runs")
TABLES_DIR = os.path.join(RESULTS, "tables")
PLOTS_DIR = os.path.join(RESULTS, "plots")


def printable(text: str) -> str:
    """
    Drop what stdout cannot encode.

    A gpt2 sample is arbitrary unicode and a Windows console is cp1252, which
    raises on it -- at the very last print of a multi-hour run.
    """
    enc = sys.stdout.encoding or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


def label_of(kind: str, mode: Optional[str]) -> str:
    return f"{kind}, {MODE_LABEL[mode]}"


# ============================================================================
# configuration
# ============================================================================

# phase 1
BENCH_KINDS = ["dense", "tensorized"]
BENCH_MODES = [None, "default", "reduce-overhead"]
# batch 1 and 8 are where compile is actually interesting: at batch 128 the
# matmuls are large enough to amortise the launch overhead on their own, so
# every mode converges and the plot says nothing
BENCH_BATCHES = [1, 8, 32, 64, 128]
BENCH_MAX_RANK = 32
BENCH_WARMUP = 5     # steps paid before timing, so the JIT and the CUDA-Graph
BENCH_REPS = 30      # capture are out of the way
# bumped whenever a bench cell measures something different under the same
# flags, so stale cells recompute instead of being silently mixed in with new
# ones. 2: the memory pass runs a full AdamW step, states included.
# 3: cells are isolated from each other (release_cell), which every recorded
# peak before it was measured on top of whatever the previous cells left alive.
# 4: peaks are deltas over the baseline plus the exact resident sizes, since
# release_cell does not in fact free the cudagraph cells' captured graphs.
# 5: three named scenarios measured absolutely -- peak_fwd, which silently
# carried the AdamW states, is now peak_infer and is taken under no_grad
# before the optimizer exists
BENCH_PROTOCOL = 5

# phase 2
UNIFORM_RANKS = [4, 8, 16, 32]
ADAPTIVE_RANKS = [4, 8, 16, 32]
# the pruning-aggressiveness axis. gamma is held at comera.GAMMA so the two
# adaptive families differ in exactly one thing.
ADAPTIVE_LRS = [3e-3, 1e-2]

# colours, so a configuration keeps its colour across every plot
KIND_COLORS = {"dense": ["#d62728", "#ff7f0e", "#8c564b"],
               "tensorized": ["#2ca02c", "#1f77b4", "#17becf"]}


@dataclass
class Arm:
    name: str
    tensorized: bool = False
    adaptive: bool = False
    max_rank: int = BENCH_MAX_RANK
    gamma: float = 0.0
    lr_rank: float = comera.LR_RANK

    @property
    def kind(self) -> str:
        if not self.tensorized:
            return "dense"
        return "adaptive" if self.adaptive else "uniform"

    @property
    def family(self) -> str:
        """
        Which Pareto curve this arm belongs to
        """
        if not self.tensorized:
            return "dense"
        if not self.adaptive:
            return "uniform"
        return f"adaptive lr={self.lr_rank:g}"


def build_arms(uniform_ranks: List[int], adaptive_ranks: List[int],
               adaptive_lrs: List[float], gamma: float) -> List[Arm]:
    arms = [Arm("dense")]
    arms += [Arm(f"uniform-r{r}", tensorized=True, max_rank=r)
             for r in uniform_ranks]
    arms += [Arm(f"adaptive-r{r}-lr{lr:g}", tensorized=True, adaptive=True,
                 max_rank=r, gamma=gamma, lr_rank=lr)
             for lr in adaptive_lrs for r in adaptive_ranks]
    return arms


@dataclass
class TrainConfig:
    max_iters: int = 1500
    # batch_size is the *micro*-batch: what one forward sees, and what the
    # activation memory scales with. The batch the optimizer actually steps on
    # is batch_size * grad_accum
    batch_size: int = 32
    grad_accum: int = 1
    block_size: int = 128
    eval_interval: int = 100   # how often losses/ranks are recorded for plots
    log_interval: int = 500    # how often a progress line is printed
    eval_iters: int = 20
    warmup_iters: int = 100
    min_lr_frac: float = 0.1
    grad_clip: float = 1.0
    warmup_timing: int = 20  # iters excluded from the wall-clock median
    seed: int = 42
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    init_std: float = 2e-2
    sample_tokens: int = 200
    core_diag_interval: int = 100  # iterations between core snapshots; 0 = off
    # tf32 by default: on an Ampere-or-newer cuda device this is the tensor-core
    # path for float32 matmuls, and it is what the target hardware would
    # actually train in. It is part of the config, not a global flag, so it goes
    # into the cache key -- a cell measured in fp32 must not be read back as a
    # tf32 result. No effect on xpu/cpu, which have no TF32 path.
    matmul_precision: str = "tf32"   # tf32 | fp32
    # which corpus, and -- for the streamed ones -- how much of it is
    # tokenized to disk. tiny-shakespeare ignores data_tokens
    dataset: str = "shakespeare"
    data_tokens: int = data.FINEWEB_TOKENS
    data_subset: str = data.FINEWEB_SUBSET
    # the two learning rates comera.param_groups splits the model into; the
    # third (rank params) is per-arm, since it is the pruning axis
    lr: float = comera.LR_ORIGIN      # everything that is not a TT core
    lr_tensor: float = comera.LR_TENSOR   # the TT cores

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum

    @property
    def tokens_per_step(self) -> int:
        return self.effective_batch * self.block_size

    def smoke(self) -> "TrainConfig":
        return TrainConfig(
            max_iters=50, batch_size=8, block_size=64, eval_interval=25,
            log_interval=25,
            eval_iters=5, warmup_iters=5, warmup_timing=5,
            n_layer=2, n_head=4, n_embd=64, sample_tokens=64,
            core_diag_interval=10,
            init_std=self.init_std, seed=self.seed,
            matmul_precision=self.matmul_precision,
            dataset=self.dataset, data_tokens=self.data_tokens,
            data_subset=self.data_subset,
            lr=self.lr, lr_tensor=self.lr_tensor,
        )


# ============================================================================
# timing / memory helpers
# ============================================================================

class StepTimer:
    """
    Wall-clock of one block, on device events where they exist.

    cuda.Event / xpu.Event, never perf_counter, for anything that runs on the
    accelerator: the host returns from a kernel launch long before the kernel
    finishes. perf_counter is the cpu fallback only.
    """

    def __init__(self, device: t.device):
        self.device = device
        self.mod = device_module(device)
        self.times: List[float] = []

    def __enter__(self):
        if self.mod is not None:
            self.start_ev = self.mod.Event(enable_timing=True)
            self.end_ev = self.mod.Event(enable_timing=True)
            self.start_ev.record()
        else:
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.mod is not None:
            self.end_ev.record()
            self.mod.synchronize()
            self.times.append(self.start_ev.elapsed_time(self.end_ev) / 1000)
        else:
            self.times.append(time.perf_counter() - self.t0)
        return False


def set_matmul_precision(name: str):
    """
    Whether float32 matmuls may run on the TF32 tensor cores.

    Three switches for one decision: set_float32_matmul_precision covers what
    goes through torch's own dispatch, and the two backend flags cover cuBLAS
    and cuDNN, which read their own globals.
    """
    tf32 = (name == "tf32")
    t.set_float32_matmul_precision("high" if tf32 else "highest")
    t.backends.cuda.matmul.allow_tf32 = tf32
    t.backends.cudnn.allow_tf32 = tf32


def median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def reset_memory(device: t.device):
    mod = device_module(device)
    if mod is not None:
        mod.reset_peak_memory_stats()
        mod.empty_cache()


def release_cell(device: t.device):
    """
    Drop everything the previous cell left alive, before the next one measures.

    reset_peak_memory_stats sets the peak to whatever is *currently* allocated,
    so a cell inherits every tensor still reachable when it starts. That is not
    hypothetical: phase 1 runs kind-major (all 15 dense cells before the first
    tensorized one), dynamo's caches keep each compiled module alive, and the
    reduce-overhead cells keep a CUDA-Graph private pool, so every tensorized
    peak was reported ~190 MB above the truth -- a batch-independent offset,
    which is what made it read as a constant factor rather than as garbage.
    Measured at n_embd=256, batch 1: tensorized fell 229 -> 39 MB against
    dense's 113.

    _dynamo.reset() is the one that matters and the one bench_cell never called
    on the eager path; gc.collect() catches the reference cycles an autograd
    graph leaves behind, which refcounting alone does not.

    It is not sufficient on its own, and measured proof that it is not: after
    it, a tensorized cell preceded by three cudagraph ones still reported a
    136 MB intercept against dense's 78, where its parameters and states are
    2.7 MB. Inductor's captured graphs and the cuBLAS workspaces of the
    capture streams survive the reset. So the peaks are *also* measured as a
    delta over the baseline this leaves behind (see measured_peak), which is
    what makes the number independent of who else is holding memory; this
    function only keeps that baseline small enough that a cell does not fail
    to allocate.
    """
    t._dynamo.reset()
    gc.collect()
    reset_memory(device)


def peak_memory(device: t.device) -> int:
    mod = device_module(device)
    return int(mod.max_memory_allocated()) if mod is not None else 0


def allocated_memory(device: t.device) -> int:
    """
    What is live right now -- the baseline a measured peak is charged against
    """
    mod = device_module(device)
    return int(mod.memory_allocated()) if mod is not None else 0


def graph_breaks() -> int:
    try:
        from torch._dynamo.utils import counters
        return int(sum(counters["graph_break"].values()))
    except Exception:
        return -1


def reset_graph_breaks():
    try:
        from torch._dynamo.utils import counters
        counters["graph_break"].clear()
        counters["inductor"].clear()
        counters["stats"].clear()
    except Exception:
        pass


def compiled_frames() -> int:
    """
    How many frames dynamo actually compiled since the last reset.

    This, not the first-step time, is the honest tell for "compilation never
    ran": inductor caches its artifacts on disk, so the second cell to compile
    an identical model has a first step of well under a second while being
    genuinely compiled.
    """
    try:
        from torch._dynamo.utils import counters
        return int(counters["stats"].get("unique_graphs", 0))
    except Exception:
        return -1


def cudagraph_skips() -> int:
    """
    How many times inductor declined to use CUDA Graphs.

    It falls back silently -- mutated inputs, dynamic shapes, cpu scalars --
    and a skipped reduce-overhead run is indistinguishable from a graphed one
    except that it is slow. Same failure shape as the dynamo cache limit.
    """
    try:
        from torch._dynamo.utils import counters
        return int(sum(v for k, v in counters["inductor"].items()
                       if "cudagraph" in k and "skip" in k))
    except Exception:
        return -1


def hash_payload(*parts) -> str:
    """
    Fingerprint of everything that changes what a measurement means.

    Without this a --smoke run and a full run share a cache file, and the full
    run silently reports the 2-layer smoke numbers as its results.
    """
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


# fields that change what a run *records* but not what it measures. They are
# kept out of the fingerprint so retuning the diagnostics does not invalidate
# hours of cached training runs -- the price is that a cached record keeps
# whatever diagnostics it was computed with (--force to resample).
HASH_IGNORED = ("core_diag_interval",)

# defaults of fields added after the cache already existed. A run that leaves
# one alone measures exactly what the old code measured, so the field is
# dropped from the fingerprint and every cached record stays readable; any
# other value is a different experiment and gets its own key.
HASH_LEGACY = {"dataset": "shakespeare",
               "data_tokens": data.FINEWEB_TOKENS,
               "data_subset": data.FINEWEB_SUBSET,
               "lr": comera.LR_ORIGIN,
               "lr_tensor": comera.LR_TENSOR}


def cfg_fingerprint(cfg: TrainConfig) -> dict:
    return {k: v for k, v in asdict(cfg).items()
            if k not in HASH_LEGACY or HASH_LEGACY[k] != v}


def config_hash(arm: Arm, cfg: TrainConfig) -> str:
    cfg_d = {k: v for k, v in cfg_fingerprint(cfg).items()
             if k not in HASH_IGNORED}
    # at grad_accum 1 the loop is the pre-accumulation one, step for step, so
    # the field is dropped from the fingerprint and the existing cache stays
    # valid. Any other value changes the optimizer's batch and must not
    # collide with it
    if cfg_d.get("grad_accum") == 1:
        cfg_d.pop("grad_accum")
    return hash_payload(asdict(arm), cfg_d)


def cached_or_compute(path: str, want: str, compute, force: bool, tag: str,
                      allowed: bool) -> Optional[dict]:
    """
    Read a cached record, or compute it when its phase is selected.

    allowed=False is how phases 3/4 read the phase-1/2 results without ever
    triggering compute -- and how a --kinds run reads back the kind it is not
    computing. force only applies to a cell that is allowed to compute: a
    forced recomputation of one selection must not blank out everything
    outside it, which is what dropping the cached record would do.
    """
    if os.path.exists(path) and not (force and allowed):
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            rec = None
        if rec is not None and rec.get("config_hash") == want:
            tqdm.write(f"  {tag}: cached")
            return rec
        if rec is not None and allowed:
            tqdm.write(f"  {tag}: cache is from another config, recomputing")
    if not allowed:
        return None
    try:
        t0 = time.perf_counter()
        rec = compute()
        rec["wall_clock_s"] = time.perf_counter() - t0
    except Exception as e:
        tqdm.write(f"  {tag}: FAILED -- {type(e).__name__}: {e}")
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    return rec


# ============================================================================
# model / training primitives
# ============================================================================

def build_model(arm: Arm, cfg: TrainConfig, ds: data.Dataset,
                device: t.device) -> GPT:
    t.manual_seed(cfg.seed)
    gpt_cfg = GPTConfig(
        block_size=cfg.block_size,
        vocab_size=ds.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        init_std=cfg.init_std,
        tensorized=arm.tensorized,
        adaptive=arm.adaptive,
        max_rank=arm.max_rank,
    )
    return GPT(gpt_cfg).to(device)


def compile_model(model: GPT, mode: Optional[str]) -> Tuple[object, float]:
    """
    torch.compile plus the dynamo reset every arm needs.

    Every arm recompiles the same GPT.forward code object with a different
    module config. Without a reset the later arms blow past dynamo's
    cache_size_limit and silently fall back to eager -- which reads as
    "compile did not help" rather than "compile never ran". The tell is a
    first-step time of ~0.1s instead of tens of seconds.
    fullgraph is left off: TTMatVec is a custom autograd.Function.
    """
    reset_graph_breaks()
    if mode is None:
        return model, 0.0
    t._dynamo.reset()
    t0 = time.perf_counter()
    compiled = t.compile(model, mode=mode)
    # the wrapper only; the JIT cost is lazy and lands on the first step
    return compiled, time.perf_counter() - t0


def lr_multiplier(it: int, cfg: TrainConfig) -> float:
    if it < cfg.warmup_iters:
        return (it + 1) / cfg.warmup_iters
    progress = (it - cfg.warmup_iters) / \
        max(1, cfg.max_iters - cfg.warmup_iters)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return cfg.min_lr_frac + (1 - cfg.min_lr_frac) * cosine


@t.no_grad()
def estimate_loss(model, ds: data.Dataset, cfg: TrainConfig,
                  device: t.device) -> Dict[str, float]:
    model.eval()
    out = {}
    gen = t.Generator().manual_seed(cfg.seed + 1234)  # same eval batches everywhere
    for split in ("train", "val"):
        losses = []
        for _ in range(cfg.eval_iters):
            X, Y = ds.get_batch(split, cfg.batch_size, cfg.block_size,
                                device, generator=gen)
            _, loss = model(X, Y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


# ============================================================================
# phase 1 -- speed and memory across batch sizes, forward vs backward
# ============================================================================

def bench_cell(kind: str, mode: Optional[str], batch: int, cfg: TrainConfig,
               ds: data.Dataset, device: t.device, max_rank: int,
               reps: int, warmup: int) -> dict:
    """
    One (configuration, batch size) cell of the CoMERA figure
    """
    # before anything is allocated: this cell must not inherit the previous
    # one's compiled modules or CUDA-Graph pool, which reset_peak_memory_stats
    # would fold into its baseline
    release_cell(device)

    arm = Arm(kind, tensorized=(kind == "tensorized"), max_rank=max_rank)
    model = build_model(arm, cfg, ds, device)
    raw_model = model
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    nominal_params = sum(p.numel() for p in model.parameters())

    model, compile_time = compile_model(model, mode)

    gen = t.Generator().manual_seed(cfg.seed)
    X, Y = ds.get_batch("train", batch, cfg.block_size, device, generator=gen)

    first_step_s = None
    for i in range(max(warmup, 1)):
        # host timing on purpose: this measures the JIT, which runs on the cpu
        t0 = time.perf_counter()
        _, loss = model(X, Y)
        loss.backward()
        raw_model.zero_grad(set_to_none=True)
        if i == 0:
            first_step_s = time.perf_counter() - t0

    # timing pass. Forward and backward are timed as two separate blocks, so
    # StepTimer's synchronize lands between them; that serialises the two
    # halves and inflates each slightly against one fused measurement, which is
    # the price of being able to report them separately at all.
    fwd, bwd = [], []
    for _ in range(reps):
        timer = StepTimer(device)
        with timer:
            _, loss = model(X, Y)
        fwd.extend(timer.times)
        timer = StepTimer(device)
        with timer:
            loss.backward()
        bwd.extend(timer.times)
        raw_model.zero_grad(set_to_none=True)

    # memory pass, kept apart from the timing pass because
    # reset_peak_memory_stats/empty_cache perturb the numbers above.
    #
    # Three scenarios, not three points on one curve. Each peak is absolute,
    # and what makes it mean what its name says is what is alive at its
    # reset_memory: reset_peak_memory_stats starts the peak at whatever is
    # currently allocated, so the resident half of each answer is included by
    # construction and the ordering of the two passes is the measurement.
    #
    #   peak_infer  weights + transient activations
    #   peak_bwd    weights + AdamW states + gradients + saved activations
    #   peak_step   the above + AdamW's _foreach_ temporaries
    #
    # the timing loop's last graph is still reachable through loss; a cycle,
    # so refcounting alone does not collect it. _dynamo.reset() is deliberately
    # not called here -- model may be the compiled wrapper this cell measures
    del loss
    gc.collect()
    reset_memory(device)

    # inference, and it has to run first: after make_optimizer and one step
    # the AdamW states exist, and there is no way to take a forward peak
    # without them. no_grad is the point of the scenario -- nothing is kept
    # for a backward that is not coming
    peak_infer = 0
    model.eval()
    with t.no_grad():
        # a compiled model builds a second graph for the no_grad path, so the
        # first call here is a compile, not a forward
        model(X, Y)
        for _ in range(2):
            reset_memory(device)
            model(X, Y)
            peak_infer = max(peak_infer, peak_memory(device))
    model.train()

    # training. The optimizer is part of it on purpose: AdamW keeps two states
    # per parameter, so fwd+bwd alone charges the tensorized model for the
    # extra intermediate TTMatVec saves (X *and* T_1) while crediting it for
    # only half of what it saves on the parameter side -- the comparison it
    # loses is not the one training actually pays. Same three-way split as
    # phase 2, so the two phases measure the same optimizer.
    opt = comera.make_optimizer(raw_model)
    # Adam allocates exp_avg/exp_avg_sq lazily on the first step; that
    # allocation belongs to the steady state, not to the measured peak
    _, loss = model(X, Y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    peak_bwd = peak_step = 0
    for _ in range(2):
        # gradients are freed before the reset, so they are charged to the
        # backward that allocates them rather than being resident already
        opt.zero_grad(set_to_none=True)
        reset_memory(device)
        _, loss = model(X, Y)
        loss.backward()
        peak_bwd = max(peak_bwd, peak_memory(device))
        opt.step()
        peak_step = max(peak_step, peak_memory(device))   # + the update
    opt_state_bytes = sum(v.numel() * v.element_size()
                          for st in opt.state.values() for v in st.values()
                          if t.is_tensor(v))
    opt.zero_grad(set_to_none=True)

    steps_per_epoch = max(1, len(ds.splits["train"]) //
                          (batch * cfg.block_size))
    fwd_s, bwd_s = median(fwd), median(bwd)
    return {
        "kind": kind,
        "compile_mode": MODE_TAG[mode],
        "label": label_of(kind, mode),
        "batch": batch,
        "block_size": cfg.block_size,
        "max_rank": max_rank if kind == "tensorized" else None,
        "config_hash": hash_payload(kind, MODE_TAG[mode], batch, max_rank,
                                    cfg_fingerprint(cfg), reps, warmup,
                                    BENCH_PROTOCOL),
        "fwd_s": fwd_s,
        "bwd_s": bwd_s,
        "step_s": fwd_s + bwd_s,
        "steps_per_epoch": steps_per_epoch,
        "epoch_fwd_min": fwd_s * steps_per_epoch / 60,
        "epoch_bwd_min": bwd_s * steps_per_epoch / 60,
        "epoch_total_min": (fwd_s + bwd_s) * steps_per_epoch / 60,
        "peak_infer_bytes": peak_infer,
        "peak_bwd_bytes": peak_bwd,
        "peak_step_bytes": peak_step,
        "peak_infer_mb": peak_infer / 1e6,
        "peak_bwd_mb": peak_bwd / 1e6,
        "peak_step_mb": peak_step / 1e6,
        "param_bytes": param_bytes,
        "param_mb": param_bytes / 1e6,
        # what AdamW itself holds: two states per parameter, i.e. the half of
        # the tensorized saving that a fwd+bwd-only measurement never sees
        "opt_state_bytes": opt_state_bytes,
        "opt_state_mb": opt_state_bytes / 1e6,
        "nominal_params": nominal_params,
        "first_step_s": first_step_s,
        "compile_time_s": compile_time,
        "graph_breaks": graph_breaks() if mode is not None else 0,
        "compiled_frames": compiled_frames() if mode is not None else 0,
        "cudagraph_skips": (cudagraph_skips()
                            if mode == "reduce-overhead" else 0),
        # CUDA Graphs allocate from a private pool that max_memory_allocated
        # does not account for the same way, so this row's memory is not
        # comparable with the others. Measured: a cudagraph run reported 2.2 MB
        # against 20.9 MB for the identical fusion-only run.
        "memory_comparable": mode != "reduce-overhead",
    }


def run_bench(cfg: TrainConfig, ds: data.Dataset, device: t.device,
              batches: List[int], modes: List[Optional[str]], max_rank: int,
              reps: int, warmup: int, force: bool,
              allowed: bool, kinds: List[str]) -> List[dict]:
    records = []
    # one run for the whole sweep: a bench cell is a handful of numbers, not a
    # time series, so the useful artefact is the table plus flat summaries
    run = tracking.Run("bench", "bench", tags=["bench"],
                       config={**asdict(cfg), "phase": "bench",
                               "device": str(device), "batches": batches,
                               "bench_rank": max_rank, "reps": reps,
                               "warmup": warmup, "kinds": kinds,
                               "modes": [MODE_TAG[m] for m in modes]})
    # mode-major: the reduce-overhead cells keep a CUDA-Graph private pool that
    # _dynamo.reset() does not hand back, so they run last and there is nothing
    # left for a later cell to inherit. Kind-major put them in the middle,
    # which is how every tensorized peak came to be measured on top of them
    cells = [(k, m, b) for m in modes for k in kinds for b in batches]
    # every kind is still read back, so a dense-only invocation re-plots the
    # tensorized series from the cache instead of dropping it off the chart
    cells += [(k, m, b) for m in modes for k in BENCH_KINDS
              if k not in kinds for b in batches]
    for kind, mode, batch in tqdm(cells, desc="bench", disable=BAR_DISABLE):
        tag = f"{kind}_{MODE_TAG[mode]}_b{batch}"
        want = hash_payload(kind, MODE_TAG[mode], batch, max_rank,
                            cfg_fingerprint(cfg), reps, warmup,
                            BENCH_PROTOCOL)
        path = os.path.join(RUNS_DIR, f"bench_{tag}_{want}.json")
        rec = cached_or_compute(
            path, want,
            lambda k=kind, m=mode, b=batch: bench_cell(
                k, m, b, cfg, ds, device, max_rank, reps, warmup),
            force, tag, allowed and kind in kinds)
        if rec is None:
            continue
        records.append(rec)
        cell = f"{rec['kind']}/{rec['compile_mode']}/b{batch}"
        run.summary({f"epoch_total_min/{cell}": rec["epoch_total_min"],
                     f"step_ms/{cell}": rec["step_s"] * 1e3,
                     f"peak_step_mb/{cell}": rec.get("peak_step_mb")})
        if "wall_clock_s" in rec:
            tqdm.write(f"  {tag}: fwd {rec['fwd_s']*1e3:.2f} ms  "
                       f"bwd {rec['bwd_s']*1e3:.2f} ms  -> epoch "
                       f"{rec['epoch_total_min']:.1f} min  "
                       f"peak {rec['peak_bwd_mb']:.0f} MB")
    run.table("bench", BENCH_COLUMNS, records)
    run.finish()
    return records


# ============================================================================
# phase 2 -- training
# ============================================================================

# ============================================================================
# phase 5 -- core diagnostics
# ============================================================================

# which roles are probed, and which one the per-arm figure draws. Everything
# recorded lands in core_stats.csv; the plot shows one role because a panel
# with 4 roles x 2d cores is unreadable.
CORE_DIAG_PLOT_ROLE = "c_fc"
# every block is recorded; the figure draws at most this many of them, evenly
# spaced with the two ends always in, because a 12-block model would otherwise
# be a 30-inch-tall png. Depth detail is in core_stats.csv
CORE_DIAG_PLOT_BLOCKS = 4
# line panels, one line per core. std and absmax used to live here too; a
# summary statistic of a roughly symmetric distribution says little that the
# percentile bands below do not say better
CORE_DIAG_FIELDS = [("grad_norm", "grad norm", True),
                    ("rank_grad_norm", "rank param grad norm", True),
                    ("rank_mean", "rank param mean", False)]

# band panels: the value distribution of a group of cores, as percentiles of
# the concatenated entries. Split around the middle bond, because the two
# halves of a TT chain carry different modes -- input on the left, output on
# the right -- and there is no reason for them to drift together
CORE_DIAG_QUANTILES = (0.01, 0.25, 0.5, 0.75, 0.99)
PCT_LABEL = "p%d"
CORE_DIST_GROUPS = [("left", "cores left of centre"),
                    ("centre", "the two centre cores"),
                    ("right", "cores right of centre")]


def core_groups(cores: List[t.Tensor]):
    """
    (name, cores) per half of the chain, plus the pair around the middle bond
    -- the widest bond, and the one the adaptive scheme has the most to remove
    from.
    """
    d = len(cores) // 2
    return [("left", cores[:d]),
            ("centre", cores[d - 1:d + 1]),
            ("right", cores[d:])]


def diag_layers(model: GPT, cfg: TrainConfig) -> List[Tuple[str, TTLinear]]:
    """
    The TT layers sampled during training: every role of every block. The
    snapshot is still one synchronize whatever the count, and depth is exactly
    the axis a per-block drift lives on, so the whole model is probed rather
    than a first/middle/last sample of it.
    """
    return [(name, m) for name, m in model.named_modules()
            if isinstance(m, TTLinear)]


@t.no_grad()
def core_snapshot(layers: List[Tuple[str, TTLinear]],
                  it: int) -> Tuple[List[dict], List[dict]]:
    """
    Two views of the same moment: one row per core (summary statistics, the
    gradient, and the rank parameter gating its trailing bond together with
    its own gradient) and one row per core group (percentiles of the
    concatenated entries).

    The rank gradient is the one that decides whether the adaptive scheme
    prunes at all: it carries the task term and the `gamma * rank_loss` term
    together, and if it never outweighs Adam's step cap the mask entries do
    not reach `threshold` inside the iteration budget.

    The percentiles are taken over the concatenation rather than averaged
    across per-core percentiles, which would not be a percentile of anything.

    Everything is computed on device and read back through a single flat cat
    + .tolist(), so a snapshot costs one synchronize rather than one per
    number -- otherwise sampling every 100 iterations would show up in the
    step-time medians. The cores are read unmasked; the mask is a separate
    quantity, reported as rank_mean.
    """
    stats, meta = [], []
    dist, dist_meta = [], []
    for name, m in layers:
        rank_params = list(m.rank_params) if m.rank_params is not None else []
        thr = m.cfg.threshold
        cores = list(m.cores)
        for n, G in enumerate(cores):
            nan = t.full((), float("nan"), device=G.device)
            zero = t.zeros((), device=G.device)
            g = G.grad
            r = rank_params[n] if n < len(rank_params) else None
            rg = r.grad if r is not None else None
            stats.append(t.stack([
                G.mean(), G.std(), G.abs().max(), G.norm(),
                g.norm() if g is not None else nan,
                r.mean() if r is not None else nan,
                r.min() if r is not None else nan,
                (r > thr).sum().float() if r is not None else zero,
                rg.norm() if rg is not None else nan,
                rg.abs().max() if rg is not None else nan,
            ]))
            meta.append((name, n))

        q = t.tensor(CORE_DIAG_QUANTILES, device=cores[0].device)
        for gname, group in core_groups(cores):
            x = t.cat([G.reshape(-1) for G in group]).float()
            dist.append(t.quantile(x, q))
            dist_meta.append((name, gname))
    if not stats:
        return [], []

    # one read-back for both tables
    nstat, nq = len(stats), len(CORE_DIAG_QUANTILES)
    nfield = stats[0].numel()
    flat = t.cat([t.stack(stats).reshape(-1),
                  t.stack(dist).reshape(-1)]).cpu().tolist()
    core_vals, dist_vals = flat[:nstat * nfield], flat[nstat * nfield:]

    rows = []
    for i, (name, n) in enumerate(meta):
        v = core_vals[i * nfield:(i + 1) * nfield]
        block, role = parse_layer_name(name)
        rows.append({"iter": it, "layer": name, "block": block, "role": role,
                     "core": n, "mean": v[0], "std": v[1], "absmax": v[2],
                     "norm": v[3], "grad_norm": v[4], "rank_mean": v[5],
                     "rank_min": v[6], "rank_alive": v[7],
                     "rank_grad_norm": v[8], "rank_grad_absmax": v[9]})

    drows = []
    labels = [PCT_LABEL % int(q * 100) for q in CORE_DIAG_QUANTILES]
    for i, (name, gname) in enumerate(dist_meta):
        v = dist_vals[i * nq:(i + 1) * nq]
        block, role = parse_layer_name(name)
        row = {"iter": it, "layer": name, "block": block, "role": role,
               "group": gname}
        row.update(zip(labels, v))
        drows.append(row)
    return rows, drows


def core_track_metrics(core_snap: List[dict],
                       dist_snap: List[dict]) -> Dict[str, float]:
    """
    A snapshot reduced to ~16 series for the live dashboard.

    Averaged over the probed layers on purpose: logging every core of every
    probed layer separately is 500 series, which is a table, not a chart. The
    per-core detail is in core_stats.csv / core_dist.csv.
    """
    out: Dict[str, float] = {}
    for field, key in [("grad_norm", "core/grad_norm"),
                       ("rank_grad_norm", "core/rank_grad_norm"),
                       ("rank_grad_absmax", "core/rank_grad_absmax"),
                       ("rank_mean", "core/rank_mean")]:
        vals = [r[field] for r in core_snap
                if math.isfinite(r.get(field, float("nan")))]
        if vals:
            out[key] = sum(vals) / len(vals)
    for group, _ in CORE_DIST_GROUPS:
        mine = [r for r in dist_snap if r["group"] == group]
        if not mine:
            continue
        for q in CORE_DIAG_QUANTILES:
            key = PCT_LABEL % int(q * 100)
            out[f"core/{group}/{key}"] = sum(r[key] for r in mine) / len(mine)
    return out


def core_rows(rec: dict, key: str = "core_diag") -> List[dict]:
    """
    A trained arm's snapshots, tagged with the arm they came from
    """
    return [dict(r, arm=rec["arm"], family=rec["family"])
            for r in rec.get(key, [])]


def train_arm(arm: Arm, mode: Optional[str], cfg: TrainConfig,
              ds: data.Dataset, device: t.device) -> dict:
    """
    Train one arm and collect every measurement the plots need
    """
    release_cell(device)   # this arm's peak is its own, not the last arm's

    model = build_model(arm, cfg, ds, device)
    raw_model = model

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    nominal_params = sum(p.numel() for p in model.parameters())
    init_ranks = comera.rank_report(raw_model)

    model, compile_time = compile_model(model, mode)

    opt = comera.make_optimizer(raw_model, lr_origin=cfg.lr,
                                lr_tensor=cfg.lr_tensor,
                                lr_rank=arm.lr_rank)
    base_lrs = [g["lr"] for g in opt.param_groups]

    gen = t.Generator().manual_seed(cfg.seed)  # identical batch stream per arm
    reset_memory(device)
    # same baseline trick as bench_cell: the arm's peak is charged against what
    # was live when it started, and its own parameters are added back from
    # their exact size. AdamW's states are allocated lazily inside the loop, so
    # they land in the transient and must not be added a second time
    mem_base = allocated_memory(device)

    run = tracking.Run("train", arm.name, tags=[arm.kind, MODE_TAG[mode]],
                       config={**asdict(arm), **asdict(cfg),
                               "phase": "train",
                               "compile_mode": MODE_TAG[mode],
                               "device": str(device),
                               "nominal_params": nominal_params,
                               "param_mb": param_bytes / 1e6,
                               "effective_batch": cfg.effective_batch,
                               "tokens_per_step": cfg.tokens_per_step})

    history = {"iter": [], "train": [], "val": [], "eff_params": [],
               "eff_size": [], "rank_loss": []}
    probed = diag_layers(raw_model, cfg) if cfg.core_diag_interval else []
    core_diag: List[dict] = []
    core_dist: List[dict] = []
    step_times: List[float] = []
    first_step_time = None

    model.train()
    bar = tqdm(range(cfg.max_iters), desc=f"{arm.name}/{MODE_TAG[mode]}",
               leave=False, disable=BAR_DISABLE)
    last_logged = 0  # iteration-based spacing, so the cadence is exactly
    # log_interval regardless of how eval_interval divides into it
    for it in bar:
        mult = lr_multiplier(it, cfg)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * mult

        timer = StepTimer(device)
        with timer:
            opt.zero_grad(set_to_none=True)
            for micro in range(cfg.grad_accum):
                X, Y = ds.get_batch("train", cfg.batch_size, cfg.block_size,
                                    device, generator=gen)
                _, model_loss = model(X, Y)
                # mean over the micro-batches, so the gradient is the one the
                # full batch would have produced
                loss = model_loss / cfg.grad_accum
                # the rank loss is a property of the weights, not of the data:
                # adding it on the last micro-step only keeps it undivided and
                # pays for comera.rank_loss once per optimizer step instead of
                # once per forward
                if micro == cfg.grad_accum - 1:
                    loss = comera.comera_loss(loss, raw_model, arm.gamma)
                loss.backward()
            if cfg.grad_clip > 0:
                t.nn.utils.clip_grad_norm_(
                    raw_model.parameters(), cfg.grad_clip)
            opt.step()
        # after opt.step(), so the gradients of this step are still live
        if probed and (it % cfg.core_diag_interval == 0
                       or it == cfg.max_iters - 1):
            snap, dsnap = core_snapshot(probed, it)
            core_diag.extend(snap)
            core_dist.extend(dsnap)
            run.log(core_track_metrics(snap, dsnap), it)

        if it == 0:
            first_step_time = timer.times[0]
        # never let the warmup exclusion swallow every sample on a short run
        if it >= min(cfg.warmup_timing, cfg.max_iters // 2):
            step_times.extend(timer.times)

        if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
            losses = estimate_loss(model, ds, cfg, device)
            rl = float(comera.rank_loss(raw_model)) if arm.adaptive else 0.0
            history["iter"].append(it)
            history["train"].append(losses["train"])
            history["val"].append(losses["val"])
            history["eff_params"].append(comera.effective_params(raw_model))
            history["eff_size"].append(comera.model_size(raw_model))
            history["rank_loss"].append(rl)
            # perplexity alongside the loss: it is the number the run is
            # actually judged on, and exp() of a diverged loss overflows the
            # chart, so it is clamped the same way the final metric is
            run.log({"train/loss": losses["train"],
                     "val/loss": losses["val"],
                     "train/ppl": math.exp(min(losses["train"], 20)),
                     "val/ppl": math.exp(min(losses["val"], 20)),
                     "params/effective": history["eff_params"][-1],
                     "params/tt_size": history["eff_size"][-1],
                     "rank_loss": rl,
                     "lr_mult": mult,
                     "step_ms": (median(step_times) * 1e3
                                 if step_times else float("nan"))}, it)
            bar.set_postfix(val=f"{losses['val']:.3f}",
                            eff=f"{history['eff_params'][-1]/1e3:.0f}k")
            first = (it == 0 and cfg.max_iters > 1)
            due = (it - last_logged) >= cfg.log_interval
            if not first and (due or it == cfg.max_iters - 1):
                last_logged = it
                ms = median(step_times) * 1e3 if step_times else float("nan")
                tqdm.write(
                    f"    {arm.name}/{MODE_TAG[mode]} it {it+1}/{cfg.max_iters}"
                    f"  train {losses['train']:.4f}  val {losses['val']:.4f}"
                    f"  eff {history['eff_params'][-1]/1e3:.0f}k"
                    f"  {ms:.1f} ms/it")
    bar.close()

    mem = peak_memory(device) - mem_base + param_bytes
    ranks = comera.rank_report(raw_model)

    try:
        ctx = t.zeros((1, 1), dtype=t.long, device=device)
        # top_k is dropped on xpu with a real tokenizer: torch.topk over a row
        # wider than ~2k *aborts the process* on this driver rather than
        # raising, so the except below cannot contain it and one sample would
        # take the whole run down at the last line of an arm. cuda is fine
        top_k = None if (device.type == "xpu" and ds.vocab_size > 2048) else 40
        sample = ds.decode(raw_model.generate(ctx, cfg.sample_tokens,
                                              temperature=0.8,
                                              top_k=top_k)[0])
    except Exception as e:
        sample = f"<generation failed: {e}>"

    final_val = history["val"][-1]
    eff_params = history["eff_params"][-1]
    run.summary({"final/val_loss": final_val,
                 "final/val_ppl": math.exp(min(final_val, 20)),
                 "final/train_loss": history["train"][-1],
                 "final/best_val_loss": min(history["val"]),
                 "final/effective_params": eff_params,
                 "final/step_ms": median(step_times) * 1e3,
                 "final/peak_memory_mb": mem / 1e6,
                 "final/compile_time_s": compile_time,
                 "final/diverged": not math.isfinite(final_val),
                 "sample": sample})
    run.finish()
    return {
        "arm": arm.name,
        "kind": arm.kind,
        "family": arm.family,
        "config_hash": config_hash(arm, cfg),
        "compiled": mode is not None,
        "compile_mode": MODE_TAG[mode],
        "max_rank": arm.max_rank if arm.tensorized else None,
        "gamma": arm.gamma,
        "lr_rank": arm.lr_rank if arm.adaptive else None,
        "micro_batch": cfg.batch_size,
        "grad_accum": cfg.grad_accum,
        "effective_batch": cfg.effective_batch,
        "nominal_params": nominal_params,
        "param_bytes": param_bytes,
        "param_mb": param_bytes / 1e6,
        "effective_params": eff_params,
        # 4 bytes/param: what the pruned model would actually cost to store
        "effective_param_mb": eff_params * 4 / 1e6,
        "tt_nominal_size": comera.nominal_size(raw_model),
        "tt_effective_size": comera.model_size(raw_model),
        "final_train_loss": history["train"][-1],
        "final_val_loss": final_val,
        "final_val_ppl": math.exp(min(final_val, 20)),
        "best_val_loss": min(history["val"]),
        "step_time_s": median(step_times),
        "first_step_s": first_step_time,
        "total_train_s": sum(step_times),
        "peak_memory_bytes": mem,
        "peak_memory_mb": mem / 1e6,
        "compile_time_s": compile_time,
        "graph_breaks": graph_breaks() if mode is not None else 0,
        "compiled_frames": compiled_frames() if mode is not None else 0,
        "cudagraph_skips": (cudagraph_skips()
                            if mode == "reduce-overhead" else 0),
        "memory_comparable": mode != "reduce-overhead",
        "diverged": not math.isfinite(final_val),
        "history": history,
        "core_diag": core_diag,
        "core_dist": core_dist,
        "init_ranks": init_ranks,
        "final_ranks": ranks,
        "sample": sample,
    }


def token_budget(cfg: TrainConfig, ds: data.Dataset,
                 n_arms: int = 1) -> dict:
    """
    How much data one arm actually sees, in tokens and in passes over the corpus
    """
    train_tokens = len(ds.splits["train"])
    per_arm = cfg.tokens_per_step * cfg.max_iters
    return {"tokens_per_step": cfg.tokens_per_step,
            "tokens_per_arm": per_arm,
            "tokens_total": per_arm * n_arms,
            "train_tokens": train_tokens,
            "epochs": per_arm / max(1, train_tokens),
            "warmup_tokens": cfg.tokens_per_step * cfg.warmup_iters}


def print_budget(cfg: TrainConfig, ds: data.Dataset, n_arms: int) -> None:
    """
    The line to read before committing a GPU-day: tokens, epochs over the
    corpus, and how much of the run is warmup. Printed before the first arm
    starts, since none of it is recoverable from the loss curve afterwards
    """
    b = token_budget(cfg, ds, n_arms)
    print(f"budget: {b['tokens_per_step']:,d} tokens/step "
          f"({cfg.batch_size} micro x {cfg.grad_accum} accum x "
          f"{cfg.block_size} ctx) x {cfg.max_iters:,d} iters = "
          f"{human_tokens(b['tokens_per_arm'])} tokens per arm")
    print(f"        {ds.name} train split {human_tokens(b['train_tokens'])} "
          f"tokens -> {b['epochs']:.2f} epochs; warmup {cfg.warmup_iters} "
          f"iters ({human_tokens(b['warmup_tokens'])} tokens, "
          f"{cfg.warmup_iters / max(1, cfg.max_iters):.1%} of the run)")
    if n_arms > 1:
        print(f"        {n_arms} arms -> "
              f"{human_tokens(b['tokens_total'])} tokens in total")
    if b["epochs"] > 1.5 and ds.name != "shakespeare":
        print(f"        ! {b['epochs']:.1f} passes over the corpus -- raise "
              f"--data-tokens to keep the run single-epoch")


def run_train(arms: List[Arm], cfg: TrainConfig, ds: data.Dataset,
              device: t.device, train_mode: Optional[str], force: bool,
              allowed: bool) -> List[dict]:
    records = []
    t_start = time.perf_counter()
    for arm in tqdm(arms, desc="arms", disable=BAR_DISABLE):
        # dense is trained eager: it has no launch-overhead problem to solve,
        # and compiling it only adds a JIT bill to the baseline
        mode = None if not arm.tensorized else train_mode
        tag = f"{arm.name}_{MODE_TAG[mode]}"
        want = config_hash(arm, cfg)
        path = os.path.join(RUNS_DIR, f"train_{tag}_{want}.json")
        rec = cached_or_compute(
            path, want,
            lambda a=arm, m=mode: train_arm(a, m, cfg, ds, device),
            force, tag, allowed)
        if rec is None:
            continue
        records.append(rec)
        if "wall_clock_s" in rec:
            tqdm.write(f"  {tag}: val {rec['final_val_loss']:.4f}  "
                       f"{rec['step_time_s']*1e3:.2f} ms/it  "
                       f"{rec['effective_params']:,d} params  "
                       f"[{rec['wall_clock_s']/60:.1f} min, "
                       f"{(time.perf_counter()-t_start)/60:.1f} min total]")
    return records


# ============================================================================
# phase 4 -- rank bookkeeping
# ============================================================================

ROLE_OF = {("attn", "c_attn"): "c_attn", ("attn", "c_proj"): "attn_proj",
           ("mlp", "c_fc"): "c_fc", ("mlp", "c_proj"): "mlp_proj"}
# forward order through a block, so plots read top-to-bottom like the model
ROLES = ["c_attn", "attn_proj", "c_fc", "mlp_proj"]


def parse_layer_name(name: str) -> Tuple[int, str]:
    """
    'transformer.h.3.mlp.c_proj' -> (3, 'mlp_proj')
    """
    m = re.search(r"h\.(\d+)\.(attn|mlp)\.(c_attn|c_proj|c_fc)", name)
    if not m:
        return -1, name
    return int(m.group(1)), ROLE_OF[(m.group(2), m.group(3))]


def rank_rows(rec: dict, cfg: TrainConfig) -> List[dict]:
    """
    One row per (arm, TT layer, bond), with the starting rank alongside.

    The starting ranks are recomputed with get_uniform_rank rather than assumed
    to be max_rank everywhere: the chain ends are clipped by the mode products,
    so a rank-32 layer does not start at 32 on every bond.
    """
    rows = []
    shapes = TT_SHAPES.get(cfg.n_embd, {})
    for name, final in sorted(rec.get("final_ranks", {}).items()):
        block, role = parse_layer_name(name)
        init = rec.get("init_ranks", {}).get(name)
        if init is None and role in shapes:
            in_shape, out_shape = shapes[role]
            init = list(get_uniform_rank(in_shape, out_shape,
                                         rec["max_rank"])[1:-1])
        init = init or [rec["max_rank"]] * len(final)
        for bond, (r0, r1) in enumerate(zip(init, final)):
            rows.append({
                "arm": rec["arm"],
                "family": rec["family"],
                "max_rank": rec["max_rank"],
                "lr_rank": rec["lr_rank"],
                "layer": name,
                "block": block,
                "role": role,
                "bond": bond,
                "rank_init": r0,
                "rank_final": r1,
                "kept_frac": r1 / r0 if r0 else float("nan"),
            })
    return rows


def rank_summary(rec: dict, rows: List[dict]) -> dict:
    mine = [r for r in rows if r["arm"] == rec["arm"]]
    finals = [r["rank_final"] for r in mine]
    init_total = sum(r["rank_init"] for r in mine)
    final_total = sum(finals)
    return {
        "arm": rec["arm"],
        "family": rec["family"],
        "max_rank": rec["max_rank"],
        "lr_rank": rec["lr_rank"],
        "gamma": rec["gamma"],
        "bonds": len(mine),
        "rank_init_total": init_total,
        "rank_final_total": final_total,
        "pruned_frac": 1 - final_total / init_total if init_total else 0.0,
        "rank_mean": final_total / len(finals) if finals else 0.0,
        "rank_min": min(finals) if finals else 0,
        "rank_max": max(finals) if finals else 0,
        "dead_bonds": sum(1 for r in finals if r == 0),
        "effective_params": rec["effective_params"],
        "effective_param_mb": rec["effective_param_mb"],
        "final_val_loss": rec["final_val_loss"],
    }


# ============================================================================
# tables
# ============================================================================

def write_csv(path: str, rows: List[dict], columns: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def markdown_table(rows: List[dict], columns: List[str],
                   fmt: Optional[Dict[str, str]] = None) -> str:
    fmt = fmt or {}

    def cell(r, c):
        v = r.get(c)
        if v is None:
            return "-"
        if isinstance(v, float) and not math.isfinite(v):
            return "nan"
        return format(v, fmt[c]) if c in fmt and isinstance(v, (int, float)) \
            else str(v)

    widths = [max(len(c), *(len(cell(r, c)) for r in rows)) if rows else len(c)
              for c in columns]
    head = "| " + " | ".join(c.ljust(w)
                             for c, w in zip(columns, widths)) + " |"
    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    body = "\n".join(
        "| " + " | ".join(cell(r, c).ljust(w)
                          for c, w in zip(columns, widths)) + " |"
        for r in rows)
    return "\n".join([head, sep, body])


BENCH_COLUMNS = ["label", "kind", "compile_mode", "batch", "fwd_s", "bwd_s",
                 "epoch_fwd_min", "epoch_bwd_min", "epoch_total_min",
                 "peak_infer_mb", "peak_bwd_mb", "peak_step_mb", "param_mb",
                 "opt_state_mb", "memory_comparable",
                 "first_step_s", "compiled_frames", "graph_breaks",
                 "cudagraph_skips"]
BENCH_FMT = {"fwd_s": ".5f", "bwd_s": ".5f", "epoch_fwd_min": ".2f",
             "epoch_bwd_min": ".2f", "epoch_total_min": ".2f",
             "peak_infer_mb": ".1f", "peak_bwd_mb": ".1f",
             "peak_step_mb": ".1f", "param_mb": ".2f", "opt_state_mb": ".2f",
             "first_step_s": ".2f"}

TRAIN_COLUMNS = ["arm", "family", "compile_mode", "max_rank", "lr_rank",
                 "nominal_params", "effective_params", "compression",
                 "final_val_loss", "final_val_ppl", "step_time_s",
                 "param_mb", "effective_param_mb", "peak_memory_mb",
                 "compiled_frames", "graph_breaks"]
TRAIN_FMT = {"nominal_params": ",d", "effective_params": ",d",
             "compression": ".2f", "final_val_loss": ".4f",
             "final_val_ppl": ".2f", "step_time_s": ".5f", "param_mb": ".2f",
             "effective_param_mb": ".2f", "peak_memory_mb": ".1f",
             "lr_rank": ".0e"}

RANK_COLUMNS = ["arm", "family", "max_rank", "lr_rank", "layer", "block",
                "role", "bond", "rank_init", "rank_final", "kept_frac"]
RANK_SUMMARY_COLUMNS = ["arm", "family", "max_rank", "lr_rank", "gamma",
                        "bonds", "rank_init_total", "rank_final_total",
                        "pruned_frac", "rank_mean", "rank_min", "rank_max",
                        "dead_bonds", "effective_params", "effective_param_mb",
                        "final_val_loss"]
RANK_SUMMARY_FMT = {"pruned_frac": ".3f", "rank_mean": ".2f", "lr_rank": ".0e",
                    "effective_params": ",d", "effective_param_mb": ".2f",
                    "final_val_loss": ".4f"}

CORE_COLUMNS = ["arm", "family", "iter", "layer", "block", "role", "core",
                "mean", "std", "absmax", "norm", "grad_norm", "rank_mean",
                "rank_min", "rank_alive", "rank_grad_norm", "rank_grad_absmax"]
CORE_FMT = {"mean": ".3e", "std": ".3e", "absmax": ".3e", "norm": ".3e",
            "grad_norm": ".3e", "rank_mean": ".4f", "rank_min": ".4f",
            "rank_alive": ".0f", "rank_grad_norm": ".3e",
            "rank_grad_absmax": ".3e"}

CORE_DIST_COLUMNS = (["arm", "family", "iter", "layer", "block", "role",
                      "group"]
                     + [PCT_LABEL % int(q * 100) for q in CORE_DIAG_QUANTILES])

MEMORY_COLUMNS = ["scope", "name", "batch", "param_mb", "effective_param_mb",
                  "opt_state_mb", "peak_infer_mb", "peak_bwd_mb",
                  "peak_step_mb",
                  "memory_comparable"]
MEMORY_FMT = {"param_mb": ".2f", "effective_param_mb": ".2f",
              "opt_state_mb": ".2f", "peak_infer_mb": ".1f",
              "peak_bwd_mb": ".1f", "peak_step_mb": ".1f"}


def decorate_train(records: List[dict]) -> List[dict]:
    """
    Compression, measured against the dense arm
    """
    base = next((r for r in records if r["kind"] == "dense"), None)
    base_params = base["effective_params"] if base else None
    for r in records:
        r["compression"] = (base_params / r["effective_params"]
                            if base_params and r["effective_params"]
                            else float("nan"))
    return records


# ============================================================================
# plots
# ============================================================================

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(PLOTS_DIR, exist_ok=True)
    return plt


def save(fig, name: str):
    path = os.path.join(PLOTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    import matplotlib.pyplot as plt
    plt.close(fig)


def grouped_bars(ax, group_labels, series_labels, values, fmt="{:.1f}",
                 colors=None, label_size=7):
    """
    The CoMERA bar chart: groups along x, one labelled bar per series.

    values[j][i] is series j in group i; nan/None draws nothing.
    """
    n = max(len(series_labels), 1)
    width = 0.8 / n
    xs = list(range(len(group_labels)))
    for j, name in enumerate(series_labels):
        off = (j - (n - 1) / 2) * width
        raw = values[j]
        vals = [v if v is not None and math.isfinite(v) else 0.0 for v in raw]
        bars = ax.bar([x + off for x in xs], vals, width, label=name,
                      color=None if colors is None else colors[j])
        ax.bar_label(bars, labels=[fmt.format(v) if v else "" for v in vals],
                     fontsize=label_size, padding=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(g) for g in group_labels])
    ax.grid(alpha=0.3, axis="y")
    ax.set_axisbelow(True)


def maybe_log_y(ax, values, span: float = 30.0):
    """
    Log axis as soon as the spread would flatten the small bars.

    Adding batch 1 to the sweep stretches the epoch-time range ~100x, and on a
    linear axis every large-batch bar then sits on the baseline.
    """
    xs = [v for row in values for v in row
          if v is not None and math.isfinite(v) and v > 0]
    if xs and max(xs) / min(xs) > span:
        ax.set_yscale("log")


def bench_series(records: List[dict], batches: List[int], field: str,
                 comparable_only: bool = False):
    """
    (series labels, colours, values[j][i]) for grouped_bars
    """
    labels, colors, values = [], [], []
    for kind in BENCH_KINDS:
        for j, mode in enumerate(BENCH_MODES):
            if comparable_only and mode == "reduce-overhead":
                continue
            cells = [next((r for r in records
                           if r["kind"] == kind
                           and r["compile_mode"] == MODE_TAG[mode]
                           and r["batch"] == b), None) for b in batches]
            if not any(cells):
                continue
            labels.append(label_of(kind, mode))
            colors.append(KIND_COLORS[kind][j])
            values.append([c[field] if c else float("nan") for c in cells])
    return labels, colors, values


def plot_bench(records: List[dict], batches: List[int]):
    plt = _plt()
    for field, name, title in (
            ("epoch_fwd_min", "1_epoch_forward.png", "forward"),
            ("epoch_bwd_min", "1_epoch_backward.png", "backward"),
            ("epoch_total_min", "2_epoch_total.png", "forward + backward")):
        labels, colors, values = bench_series(records, batches, field)
        if not labels:
            continue
        fig, ax = plt.subplots(figsize=(11, 5))
        grouped_bars(ax, batches, labels, values, fmt="{:.1f}", colors=colors)
        maybe_log_y(ax, values)
        ax.set(xlabel="batch size", ylabel="time (min)",
               title=f"projected time for one epoch -- {title}")
        ax.legend(fontsize=8, ncol=2)
        save(fig, name)

    # per-step time, which is the view the small batches exist for: an epoch at
    # batch 1 is 128x more steps than at batch 128, so the epoch chart answers
    # "which batch size finishes an epoch first", not "what does compile do to
    # one step"
    labels, colors, values = bench_series(records, batches, "step_s")
    if labels:
        ms = [[v * 1e3 for v in row] for row in values]
        fig, ax = plt.subplots(figsize=(11, 5))
        grouped_bars(ax, batches, labels, ms, fmt="{:.1f}", colors=colors)
        maybe_log_y(ax, ms)
        ax.set(xlabel="batch size", ylabel="ms per step",
               title="time for one step (forward + backward)")
        ax.legend(fontsize=8, ncol=2)
        save(fig, "2b_step_time.png")
    print(f"plots -> {PLOTS_DIR}/")


def plot_memory(bench_records: List[dict], train_records: List[dict],
                batches: List[int]):
    plt = _plt()

    # peak training memory, per batch. reduce-overhead is left out: CUDA Graphs
    # allocate from a private pool max_memory_allocated does not see the same
    # way, so those bars would not be on the same scale as the rest.
    for field, name, title in (
            ("peak_infer_mb", "8_mem_peak_inference.png",
             "inference forward -- weights + activations, no_grad"),
            ("peak_bwd_mb", "8_mem_peak_backward.png",
             "training step -- forward + backward"),
            ("peak_step_mb", "8b_mem_peak_step.png",
             "training step -- forward + backward + AdamW update")):
        labels, colors, values = bench_series(bench_records, batches, field,
                                              comparable_only=True)
        if not labels:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        grouped_bars(ax, batches, labels, values, fmt="{:.0f}", colors=colors)
        ax.set(xlabel="batch size", ylabel="peak allocated (MB)",
               title=f"peak memory -- {title}\n"
               "CUDAGraph rows excluded: private pool, not comparable")
        ax.legend(fontsize=8)
        save(fig, name)

    # static footprint of the trained models
    if train_records:
        recs = sorted(train_records, key=lambda r: r["effective_params"])
        names = [r["arm"] for r in recs]
        fig, ax = plt.subplots(figsize=(11, 5))
        grouped_bars(ax, names, ["allocated parameters",
                                 "effective (post-pruning)"],
                     [[r["param_mb"] for r in recs],
                      [r["effective_param_mb"] for r in recs]],
                     fmt="{:.2f}", colors=["#bbbbbb", "#1f77b4"])
        ax.set(ylabel="MB", title="static parameter footprint")
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        for lab in ax.get_xticklabels():
            lab.set_ha("right")
        ax.legend(fontsize=8)
        save(fig, "7_mem_static.png")


def plot_train(records: List[dict]):
    plt = _plt()
    dense = next((r for r in records if r["kind"] == "dense"), None)

    # pareto: loss against surviving parameters, one line per family
    fig, ax = plt.subplots(figsize=(9, 6))
    families = sorted({r["family"] for r in records if r["kind"] != "dense"},
                      key=lambda f: (f != "uniform", f))
    markers = {"uniform": "o"}
    for i, fam in enumerate(families):
        pts = sorted((r for r in records if r["family"] == fam),
                     key=lambda r: r["effective_params"])
        ax.plot([r["effective_params"] for r in pts],
                [r["final_val_loss"] for r in pts],
                marker=markers.get(fam, "s"), ms=7, color=f"C{i}", label=fam)
        for r in pts:
            ax.annotate(f"r{r['max_rank']}",
                        (r["effective_params"], r["final_val_loss"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    if dense:
        ax.plot([dense["effective_params"]], [dense["final_val_loss"]],
                marker="*", ms=16, ls="none", color="k", label="dense")
        ax.axhline(dense["final_val_loss"], color="k", ls=":", lw=1)
    ax.set(xscale="log", xlabel="effective (non-pruned) parameters",
           ylabel="final val loss",
           title="accuracy vs size -- down-left is better")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "3_pareto_params.png")

    # loss curves
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, r in enumerate(sorted(records, key=lambda r: r["arm"])):
        h = r["history"]
        ax.plot(h["iter"], h["val"], color=f"C{i % 10}", label=r["arm"])
        ax.plot(h["iter"], h["train"], color=f"C{i % 10}", alpha=0.3, ls="--")
    ax.set(xlabel="iteration", ylabel="loss",
           title="tiny-shakespeare: val (solid) / train (dashed)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    save(fig, "4_loss_curves.png")

    # pruning over training
    adaptive = [r for r in records if r["kind"] == "adaptive"]
    if adaptive:
        fig, ax = plt.subplots(figsize=(9, 5))
        for r in sorted(adaptive, key=lambda r: (r["lr_rank"], r["max_rank"])):
            h = r["history"]
            start = h["eff_params"][0]
            ax.plot(h["iter"], [p / start for p in h["eff_params"]],
                    label=r["arm"],
                    ls="-" if r["lr_rank"] == min(a["lr_rank"]
                                                  for a in adaptive) else "--")
        ax.set(xlabel="iteration", ylabel="effective params / initial",
               title="pruning over training (solid = milder lr_rank)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        save(fig, "4b_pruning_over_training.png")


def rank_heatmap(plt, rows: List[dict], arms: List[str], field: str,
                 vmax: float, cmap: str, title: str, fname: str):
    """
    layer x bond grid of `field`, one panel per arm
    """
    ncol = min(len(arms), 4)
    nrow = (len(arms) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow),
                             squeeze=False)
    for k, arm in enumerate(arms):
        ax = axes[k // ncol][k % ncol]
        mine = [r for r in rows if r["arm"] == arm]
        layers = sorted({(r["block"], r["role"], r["layer"]) for r in mine},
                        key=lambda l: (l[0], ROLES.index(l[1])
                                       if l[1] in ROLES else 99))
        nbond = max((r["bond"] for r in mine), default=0) + 1
        grid = [[float("nan")] * nbond for _ in layers]
        index = {lay[2]: i for i, lay in enumerate(layers)}
        for r in mine:
            grid[index[r["layer"]]][r["bond"]] = r[field]
        im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"b{b}.{role}" for b, role, _ in layers],
                           fontsize=6)
        ax.set_xticks(range(nbond))
        ax.set_xlabel("bond", fontsize=8)
        ax.set_title(arm, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.04)
    for k in range(len(arms), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(title, fontsize=11)
    save(fig, fname)


def plot_ranks(summaries: List[dict], rows: List[dict]):
    """
    The four views of the final rank configuration
    """
    if not summaries:
        return
    plt = _plt()
    lrs = sorted({s["lr_rank"] for s in summaries})
    ranks = sorted({s["max_rank"] for s in summaries})

    def cell(lr, mr, field, default=float("nan")):
        s = next((s for s in summaries
                  if s["lr_rank"] == lr and s["max_rank"] == mr), None)
        return s[field] if s else default

    # 1 -- pruned fraction against starting rank, one bar per lr_rank
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped_bars(ax, [f"max_rank {r}" for r in ranks],
                 [f"lr_rank {lr:g}" for lr in lrs],
                 [[cell(lr, mr, "pruned_frac") for mr in ranks] for lr in lrs],
                 fmt="{:.2f}")
    # headroom for the bar labels, which bar_label draws *above* the bar
    ax.set(ylabel="pruned fraction of total rank", ylim=(0, 1.12),
           title="how much rank the adaptive scheme removes")
    ax.legend(fontsize=8, ncol=len(lrs), loc="upper center",
              bbox_to_anchor=(0.5, -0.08))
    save(fig, "5_rank_pruned_frac.png")

    # 2 -- where the rank actually sits: layer x bond, one panel per arm.
    # Two views of the same grid, and the second is the honest one: the chain
    # ends never start at max_rank, since get_uniform_rank clips them by the
    # mode products (bond 0 of a (4,8,8) input is 4 whatever max_rank is), so
    # on the absolute map every arm looks pruned at the edges when nothing was
    # pruned there at all. kept_frac divides that structure out.
    arms = sorted({r["arm"] for r in rows})
    rank_max = max((r["rank_final"] for r in rows), default=1) or 1
    for field, vmax, cmap, title, fname in (
            ("rank_final", rank_max, "viridis",
             "surviving rank per layer and bond", "5c_rank_heatmap.png"),
            ("kept_frac", 1.0, "magma",
             "surviving rank / initial -- what the scheme actually removed",
             "5e_rank_kept_frac.png")):
        rank_heatmap(plt, rows, arms, field, vmax, cmap, title, fname)

    # 3 -- does pruning concentrate in a particular role?
    roles = ROLES
    fig, ax = plt.subplots(figsize=(11, 5.6))
    values = []
    for arm in arms:
        col = []
        for role in roles:
            xs = [r["kept_frac"] for r in rows
                  if r["arm"] == arm and r["role"] == role]
            col.append(sum(xs) / len(xs) if xs else float("nan"))
        values.append(col)
    grouped_bars(ax, roles, arms, values, fmt="{:.2f}")
    # a bar at 1.00 plus its label needs room above it, and with 8 arms the
    # legend does not fit inside the axes at all -- put it under them
    ax.set(ylabel="mean surviving rank / initial", ylim=(0, 1.12),
           title="which roles keep their rank")
    ax.legend(fontsize=7, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.08))
    save(fig, "5d_rank_by_role.png")


def core_bands(ax, rows: List[dict], legend: bool = False):
    """
    Percentiles of one core group against training iteration.

    Two nested bands (p1-p99, p25-p75) around the median: the outer one is
    where a blow-up shows first, the inner one is where a collapse towards
    zero shows first, and a single std curve conflates the two.
    """
    pts = sorted(rows, key=lambda r: r["iter"])
    if not pts:
        return
    xs = [r["iter"] for r in pts]
    lo, q1, med, q3, hi = (
        [[r[PCT_LABEL % int(q * 100)] for r in pts]
         for q in CORE_DIAG_QUANTILES])
    ax.fill_between(xs, lo, hi, color="#1f77b4", alpha=0.18,
                    label="p1-p99" if legend else None)
    ax.fill_between(xs, q1, q3, color="#1f77b4", alpha=0.40,
                    label="p25-p75" if legend else None)
    ax.plot(xs, med, color="#08306b", lw=1.4,
            label="median" if legend else None)
    ax.axhline(0.0, color="#999999", lw=0.6, ls=":")
    if legend:
        ax.legend(fontsize=6)


def plot_blocks(blocks: List[int],
                limit: int = CORE_DIAG_PLOT_BLOCKS) -> List[int]:
    """
    At most `limit` of the recorded blocks, evenly spaced and keeping the first
    and the last -- the ends are where a depth drift shows first.
    """
    if len(blocks) <= limit:
        return blocks
    idx = {round(i * (len(blocks) - 1) / (limit - 1)) for i in range(limit)}
    return [blocks[i] for i in sorted(idx)]


def plot_core_diag(rows: List[dict], dist_rows: List[dict]):
    """
    How the cores move during training: one figure per arm, one row per plotted
    block (a `plot_blocks` sample of the recorded ones), and for the
    CORE_DIAG_PLOT_ROLE layer three value-distribution
    panels (left half of the chain, the two centre cores, right half) next to
    the per-core gradient norm and rank parameter.

    Log axes wherever the quantity stays positive -- gradient norms span
    orders of magnitude along the chain, and a linear axis hides exactly the
    collapse or blow-up this plot exists to catch. The band panels stay
    linear: core entries are signed and roughly symmetric about zero.
    """
    if not rows:
        return
    plt = _plt()
    for arm in sorted({r["arm"] for r in rows}):
        mine = [r for r in rows
                if r["arm"] == arm and r["role"] == CORE_DIAG_PLOT_ROLE]
        if not mine:
            continue
        blocks = plot_blocks(sorted({r["block"] for r in mine}))
        mine = [r for r in mine if r["block"] in blocks]
        ncores = max(r["core"] for r in mine) + 1
        dmine = [r for r in dist_rows
                 if r["arm"] == arm and r["role"] == CORE_DIAG_PLOT_ROLE]
        panels = CORE_DIST_GROUPS + CORE_DIAG_FIELDS
        cmap = plt.get_cmap("viridis")
        fig, axes = plt.subplots(
            len(blocks), len(panels),
            figsize=(3.2 * len(panels), 2.6 * len(blocks)),
            squeeze=False, sharex=True)
        for i, b in enumerate(blocks):
            for j, panel in enumerate(panels):
                ax = axes[i][j]
                if len(panel) == 2:      # band panel: a group's percentiles
                    group, title = panel
                    core_bands(ax, [r for r in dmine
                                    if r["block"] == b
                                    and r["group"] == group],
                               legend=(i, j) == (0, 0))
                else:                    # line panel: one line per core
                    field, title, logy = panel
                    positive = True
                    for c in range(ncores):
                        # .get: an arm cached before a field existed keeps the
                        # diagnostics it was trained with, and simply has no
                        # line in that panel
                        pts = sorted((r["iter"], r.get(field, float("nan")))
                                     for r in mine
                                     if r["block"] == b and r["core"] == c)
                        ys = [y for _, y in pts if math.isfinite(y)]
                        if not ys:
                            continue
                        positive = positive and all(y > 0 for y in ys)
                        ax.plot([x for x, _ in pts], [y for _, y in pts],
                                color=cmap(c / max(ncores - 1, 1)), lw=1.2,
                                label=f"core {c}" if i == 0 else None)
                    if logy and positive and ax.lines:
                        ax.set_yscale("log")
                    if i == 0 and ax.lines:
                        ax.legend(fontsize=6, ncol=2)
                ax.grid(alpha=0.3)
                if i == 0:
                    ax.set_title(title, fontsize=9)
                if j == 0:
                    ax.set_ylabel(f"block {b}", fontsize=9)
                if i == len(blocks) - 1:
                    ax.set_xlabel("iteration", fontsize=8)
        fig.suptitle(f"{arm} -- {CORE_DIAG_PLOT_ROLE} cores through training",
                     fontsize=11)
        save(fig, f"6_cores_{arm}.png")


# ============================================================================
# summary
# ============================================================================

def summarize(bench: List[dict], train: List[dict], summaries: List[dict],
              cores: List[dict], batches: List[int]):
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    if bench:
        print("\nprojected minutes per epoch (fwd + bwd), by batch size")
        for kind in BENCH_KINDS:
            for mode in BENCH_MODES:
                cells = [next((r for r in bench
                               if r["kind"] == kind
                               and r["compile_mode"] == MODE_TAG[mode]
                               and r["batch"] == b), None) for b in batches]
                if not any(cells):
                    continue
                nums = "  ".join(f"b{b}: {c['epoch_total_min']:6.1f}"
                                 if c else f"b{b}: {'-':>6s}"
                                 for b, c in zip(batches, cells))
                print(f"      {label_of(kind, mode):<38s} {nums}")
        # the two silent-fallback tells this harness has been bitten by
        for r in bench:
            if r["compile_mode"] != "eager" and r.get("compiled_frames") == 0:
                print(f"      ! {r['label']} b{r['batch']}: dynamo compiled 0 "
                      f"frames -- it fell back to eager (cache_size_limit?)")
            if r.get("cudagraph_skips", 0) > 0:
                print(f"      ! {r['label']} b{r['batch']}: inductor skipped "
                      f"CUDA Graphs {r['cudagraph_skips']}x -- fusion only")
        for b in batches:
            d = next((r for r in bench if r["kind"] == "dense"
                      and r["compile_mode"] == "eager" and r["batch"] == b), None)
            g = next((r for r in bench if r["kind"] == "tensorized"
                      and r["compile_mode"] == "cudagraph"
                      and r["batch"] == b), None)
            e = next((r for r in bench if r["kind"] == "tensorized"
                      and r["compile_mode"] == "eager" and r["batch"] == b), None)
            if e and g:
                print(f"      batch {b}: compile+graphs give tensorized "
                      f"{e['epoch_total_min']/g['epoch_total_min']:.2f}x"
                      + (f", still {g['epoch_total_min']/d['epoch_total_min']:.2f}x "
                         f"dense-eager" if d else ""))

        # where the parameter saving actually shows up: activations scale with
        # tokens per step, parameters + AdamW states do not, so the tensorized
        # win only surfaces at the small-batch end
        print("\npeak memory of a full step (fwd + bwd + AdamW), eager rows")
        for b in batches:
            d = next((r for r in bench if r["kind"] == "dense"
                      and r["compile_mode"] == "eager" and r["batch"] == b), None)
            e = next((r for r in bench if r["kind"] == "tensorized"
                      and r["compile_mode"] == "eager" and r["batch"] == b), None)
            if not (d and e and d.get("peak_step_mb") and e.get("peak_step_mb")):
                continue
            print(f"      batch {b:>4d}: dense {d['peak_step_mb']:7.1f} MB "
                  f"(params+states {d['param_mb']+d.get('opt_state_mb', 0):6.1f})"
                  f"   tensorized {e['peak_step_mb']:7.1f} MB "
                  f"(params+states {e['param_mb']+e.get('opt_state_mb', 0):6.1f})"
                  f"   {e['peak_step_mb']/d['peak_step_mb']:.2f}x")

    if train:
        print("\nquality vs size")
        dense = next((r for r in train if r["kind"] == "dense"), None)
        for r in sorted(train, key=lambda r: r["effective_params"]):
            line = (f"      {r['arm']:<24s} {r['effective_params']:>9,d} params  "
                    f"val {r['final_val_loss']:.4f}  ppl {r['final_val_ppl']:7.2f}")
            if dense:
                line += f"  {r['compression']:5.2f}x vs dense"
            print(line)
        tens = [r for r in train if r["kind"] != "dense"]
        if dense and tens:
            better = [r for r in tens
                      if r["final_val_loss"] <= dense["final_val_loss"]]
            print(f"      {len(better)}/{len(tens)} tensorized arms match or "
                  f"beat dense at {dense['final_val_loss']:.4f}")
        unif = [r for r in train if r["kind"] == "uniform"]
        adap = [r for r in train if r["kind"] == "adaptive"]
        doms = [(a, u) for a in adap for u in unif
                if a["effective_params"] < u["effective_params"]
                and a["final_val_loss"] < u["final_val_loss"]]
        for a, u in sorted(doms, key=lambda p: p[0]["effective_params"]):
            print(f"      {a['arm']} dominates {u['arm']}: "
                  f"{a['effective_params']:,d} @ {a['final_val_loss']:.4f} vs "
                  f"{u['effective_params']:,d} @ {u['final_val_loss']:.4f}")

    if summaries:
        print("\npruning by lr_rank and starting rank")
        for s in sorted(summaries, key=lambda s: (s["lr_rank"], s["max_rank"])):
            print(f"      lr {s['lr_rank']:.0e}  max_rank {s['max_rank']:>3d}"
                  f"  pruned {s['pruned_frac']*100:5.1f}%"
                  f"  mean rank {s['rank_mean']:5.2f}"
                  f"  dead bonds {s['dead_bonds']:>3d}/{s['bonds']}")
        if all(s["pruned_frac"] == 0 for s in summaries):
            print("      ! nothing pruned at all -- with this iteration budget "
                  "a rank parameter cannot cross the threshold; raise lr_rank "
                  "or gamma (see the note in comera.py)")

    if cores:
        def avg(xs):
            xs = [x for x in xs if math.isfinite(x)]
            return sum(xs) / len(xs) if xs else float("nan")

        print("\ncores: first -> last snapshot, averaged over probed layers")
        for arm in sorted({r["arm"] for r in cores}):
            mine = [r for r in cores if r["arm"] == arm]
            it0 = min(r["iter"] for r in mine)
            it1 = max(r["iter"] for r in mine)
            s0 = avg([r["std"] for r in mine if r["iter"] == it0])
            s1 = avg([r["std"] for r in mine if r["iter"] == it1])
            gn = avg([r["grad_norm"] for r in mine if r["iter"] == it1])
            mx = max((r["absmax"] for r in mine if r["iter"] == it1),
                     default=float("nan"))
            bad = sum(1 for r in mine
                      if not math.isfinite(r["std"])
                      or not math.isfinite(r["grad_norm"]))
            print(f"      {arm:<24s} std {s0:.2e} -> {s1:.2e} "
                  f"(x{s1/s0:5.2f})  |max| {mx:.2e}  |grad| {gn:.2e}"
                  + (f"  ! {bad} non-finite" if bad else ""))
    print("=" * 78 + "\n")


# ============================================================================
# main
# ============================================================================

def main():
    global RESULTS, RUNS_DIR, TABLES_DIR, PLOTS_DIR, BAR_DISABLE
    global CORE_DIAG_PLOT_ROLE
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="toy end-to-end run to validate the pipeline")
    ap.add_argument("--phases", nargs="*", default=["bench", "train"],
                    choices=["bench", "train"],
                    help="which phases may compute. The memory and rank "
                         "phases always run and only re-plot the cache")
    ap.add_argument("--force", action="store_true",
                    help="recompute cells that are already cached")
    ap.add_argument("--matmul-precision", choices=["tf32", "fp32"],
                    default=None,
                    help="tf32 (the default) runs float32 matmuls on the "
                         "tensor cores; fp32 keeps full float32 arithmetic. "
                         "Part of the cache key, so switching recomputes "
                         "instead of mixing the two. cuda-only; xpu and cpu "
                         "have no TF32 path and ignore it")
    ap.add_argument("--init", choices=["scaled", "normal"], default="scaled",
                    help="scaled: cores sized so the contracted matrix has "
                         "std 0.02; normal: literal standard normal")
    ap.add_argument("--batches", nargs="*", type=int, default=None,
                    help="batch sizes for the benchmark "
                         "(default 1 8 32 64 128)")
    ap.add_argument("--bench-rank", type=int, default=BENCH_MAX_RANK,
                    help="max_rank of the tensorized model in the benchmark")
    ap.add_argument("--bench-reps", type=int, default=None)
    ap.add_argument("--compile-modes", nargs="*",
                    default=["eager", "compile", "cudagraph"],
                    choices=["eager", "compile", "cudagraph"],
                    help="which of the three benchmark modes to run")
    ap.add_argument("--kinds", nargs="+", default=list(BENCH_KINDS),
                    choices=list(BENCH_KINDS),
                    help="which models phase 1 may *compute*; the other one "
                         "is still read from the cache, so the plots keep "
                         "both series. One kind per process is the way to a "
                         "memory number nothing else in the process is "
                         "holding: dynamo caches and CUDA-Graph pools do not "
                         "survive an exit, whatever they survive inside one")
    ap.add_argument("--train-compile-mode",
                    choices=["eager", "compile", "cudagraph"],
                    default=None,
                    help="compile mode for the tensorized training arms. "
                         "'compile' (inductor fusion) is the default: CUDA "
                         "Graphs replay a captured graph whose parameters the "
                         "optimizer mutates outside it, which is exactly what "
                         "inductor skips silently. dense always trains eager. "
                         "Default: 'compile', or 'eager' under --smoke, where "
                         "the JIT bill dwarfs the 50 iterations it would speed "
                         "up -- phase 1 is where compile is measured anyway")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="subset of training arm names")
    ap.add_argument("--ranks", nargs="*", type=int, default=None,
                    help="rank ladder for the training arms")
    ap.add_argument("--lr-ranks", nargs="*", type=float, default=None,
                    help="rank learning rates, one adaptive family each")
    ap.add_argument("--gamma", type=float, default=comera.GAMMA,
                    help="rank-loss weight for the adaptive arms")
    ap.add_argument("--dataset", default=None, choices=list(DATASETS),
                    help="corpus to train on: shakespeare (char level, 65 "
                         "tokens) or fineweb-edu (gpt2 tokens, streamed and "
                         "tokenized into data/ on first use)")
    ap.add_argument("--data-tokens", type=int, default=None,
                    help="fineweb-edu: gpt2 tokens tokenized to disk "
                         f"(default {data.FINEWEB_TOKENS:,d}). Only that much "
                         "of the stream is ever downloaded")
    ap.add_argument("--data-subset", default=None,
                    choices=list(data.FINEWEB_SUBSETS),
                    help="fineweb-edu: which slice of the repo the tokens are "
                         "drawn from -- sample-10BT (the default) caps at 10B "
                         "tokens, then sample-100BT / sample-350BT / full. "
                         "Each subset is tokenized into its own directory, and "
                         "only the row groups actually needed are downloaded, "
                         "so a larger subset costs nothing until it is used")
    ap.add_argument("--train-tokens", type=int, default=None,
                    help="train for this many tokens instead of a fixed "
                         "iteration count: max_iters = ceil(N / tokens per "
                         "step). The inverse of --iters, which it replaces; "
                         "the two cannot both be given")
    ap.add_argument("--model", default=None, choices=list(MODEL_PRESETS),
                    help="named model size -- n_layer / n_head / n_embd / "
                         "block_size. gpt2-small is the published 124M "
                         "configuration (12 layers, 12 heads, 768 wide, 1024 "
                         "context); the individual flags still override it")
    ap.add_argument("--block-size", type=int, default=None,
                    help="maximum sequence length, i.e. the context trained "
                         "on (default 128; 1024 under --model gpt2-small)")
    ap.add_argument("--lr", type=float, default=None,
                    help="learning rate of everything that is not a TT core: "
                         "embeddings, norms, biases, dense layers "
                         f"(default {comera.LR_ORIGIN:g})")
    ap.add_argument("--lr-tensor", type=float, default=None,
                    help=f"learning rate of the TT cores (default "
                         f"{comera.LR_TENSOR:g}). The rank parameters have "
                         "their own, --lr-ranks")
    ap.add_argument("--warmup", type=int, default=None,
                    help="linear warmup iterations before the cosine decay "
                         "(default 100, 5 under --smoke)")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="sequences per forward (default 32, smoke 8). This "
                         "is what activation memory scales with")
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="forward/backward passes accumulated before each "
                         "optimizer step (default 1). The batch the optimizer "
                         "steps on is --micro-batch x --grad-accum, so a "
                         "batch too large to fit is reached by raising this "
                         "rather than the micro-batch")
    ap.add_argument("--log-interval", type=int, default=None,
                    help="iterations between progress lines (default 500)")
    ap.add_argument("--core-diag-interval", type=int, default=None,
                    help="iterations between core snapshots (default 100, "
                         "10 under --smoke; 0 disables). Not part of the cache "
                         "key, so cached arms keep the diagnostics they were "
                         "trained with -- use --force to resample them")
    ap.add_argument("--core-diag-role", default=CORE_DIAG_PLOT_ROLE,
                    choices=ROLES,
                    help="which TT role the per-arm core figure draws. Every "
                         "probed role is written to core_stats.csv regardless")
    ap.add_argument("--track", default="off",
                    choices=list(tracking.BACKENDS),
                    help="live experiment tracking: one run per training arm "
                         "(loss and perplexity curves), one for the benchmark "
                         "sweep, one for the final plots and tables. "
                         "tensorboard writes to <out>/tb/<group>/")
    ap.add_argument("--wandb-project", default=tracking.PROJECT)
    ap.add_argument("--wandb-entity", default=None,
                    help="wandb only: team/user the runs belong to")
    ap.add_argument("--wandb-mode", default="online",
                    choices=["online", "offline", "disabled"],
                    help="wandb only: offline writes to <out>/wandb for a "
                         "later `wandb sync`; that is also the fallback "
                         "when no API key is available, since an interactive "
                         "login prompt would hang a colab `!python` job")
    ap.add_argument("--wandb-group", default=None,
                    help="name tying this invocation's runs together: a "
                         "wandb group, or the <out>/tb subdirectory "
                         "(default: a timestamp)")
    ap.add_argument("--progress", action="store_true",
                    help="show per-iteration tqdm bars (off by default)")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress inductor/dynamo compile chatter")
    ap.add_argument("--out", default=RESULTS)
    args = ap.parse_args()

    RESULTS = args.out
    RUNS_DIR = os.path.join(RESULTS, "runs")
    TABLES_DIR = os.path.join(RESULTS, "tables")
    PLOTS_DIR = os.path.join(RESULTS, "plots")
    os.makedirs(RUNS_DIR, exist_ok=True)

    cfg = TrainConfig()
    batches = BENCH_BATCHES
    ranks, lrs = UNIFORM_RANKS, ADAPTIVE_LRS
    reps, warmup = BENCH_REPS, BENCH_WARMUP
    train_mode = args.train_compile_mode or "compile"
    if args.smoke:
        train_mode = args.train_compile_mode or "eager"
        cfg = cfg.smoke()
        batches = [8, 16]
        ranks = [4, 8]
        reps, warmup = 5, 2
    if args.matmul_precision is not None:
        cfg.matmul_precision = args.matmul_precision
    if args.init == "normal":
        # what "just use standard normal" actually means; expect divergence
        cfg.init_std = 1.0
    if args.model is not None:
        for k, v in MODEL_PRESETS[args.model].items():
            setattr(cfg, k, v)
    if args.dataset is not None:
        cfg.dataset = args.dataset
    if args.data_tokens is not None:
        cfg.data_tokens = args.data_tokens
    if args.data_subset is not None:
        cfg.data_subset = args.data_subset
    if args.block_size is not None:
        cfg.block_size = args.block_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.lr_tensor is not None:
        cfg.lr_tensor = args.lr_tensor
    if args.warmup is not None:
        cfg.warmup_iters = args.warmup
    if args.iters is not None:
        cfg.max_iters = args.iters
    if args.log_interval is not None:
        cfg.log_interval = args.log_interval
    if args.micro_batch is not None:
        cfg.batch_size = args.micro_batch
    if args.grad_accum is not None:
        assert args.grad_accum >= 1, "--grad-accum must be at least 1"
        cfg.grad_accum = args.grad_accum
    if args.train_tokens is not None:
        assert args.iters is None,             "--iters and --train-tokens both set the iteration count"
        # applied after the batch and block flags, since it is defined in terms
        # of them. max_iters is what lands in the cache key either way, so a
        # token budget and the equivalent --iters address the same cached run
        cfg.max_iters = max(1, math.ceil(args.train_tokens
                                         / cfg.tokens_per_step))
    if args.core_diag_interval is not None:
        cfg.core_diag_interval = args.core_diag_interval
    CORE_DIAG_PLOT_ROLE = args.core_diag_role
    if args.batches:
        batches = args.batches
    if args.ranks:
        ranks = args.ranks
    if args.lr_ranks:
        lrs = args.lr_ranks
    if args.bench_reps is not None:
        reps = args.bench_reps
    if args.progress:
        BAR_DISABLE = False
    if args.quiet:
        import logging
        for name in ("torch._inductor", "torch._dynamo", "torch._functorch"):
            logging.getLogger(name).setLevel(logging.ERROR)

    track = tracking.configure(
        backend=args.track, dir=RESULTS,
        group=args.wandb_group or tracking.session_group(),
        project=args.wandb_project, entity=args.wandb_entity,
        mode=args.wandb_mode)

    device = get_device()
    set_matmul_precision(cfg.matmul_precision)
    t.manual_seed(cfg.seed)
    ds = get_dataset(cfg.dataset, cfg.data_tokens, cfg.data_subset)

    modes = [MODE_FROM_TAG[m] for m in args.compile_modes]
    bench_rank = min(args.bench_rank, max(ranks)) if args.smoke \
        else args.bench_rank
    arms = build_arms(ranks, ranks, lrs, args.gamma)
    if args.arms:
        arms = [a for a in arms if a.name in args.arms]

    if track.backend == "tensorboard":
        print(f"tensorboard: {tracking.log_dir()}\n"
              f"  tensorboard --logdir {os.path.join(RESULTS, tracking.TB_SUBDIR)}")
    elif track.backend == "wandb":
        print(f"wandb: project={track.project} group={track.group} "
              f"mode={track.mode}")
    print(f"batch: {cfg.batch_size} micro x {cfg.grad_accum} accum = "
          f"{cfg.effective_batch} ({cfg.tokens_per_step:,d} tokens/step)")
    print(f"device={device}  matmul={cfg.matmul_precision}"
          f"{'' if device.type == 'cuda' else ' (no TF32 path here)'}  "
          f"init_std={cfg.init_std}  "
          f"n_layer={cfg.n_layer} n_head={cfg.n_head} n_embd={cfg.n_embd} "
          f"block={cfg.block_size} iters={cfg.max_iters}")
    corpus = ds.name + (f"/{cfg.data_subset}" if cfg.dataset != "shakespeare"
                        else "")
    print(f"data={corpus} vocab={ds.vocab_size}  "
          f"lr={cfg.lr:g} (cores {cfg.lr_tensor:g})  "
          f"warmup={cfg.warmup_iters}")
    print(f"phase 1: {len(args.kinds)*len(modes)} configs x {len(batches)} "
          f"batches (rank {bench_rank})"
          + ("" if len(args.kinds) == len(BENCH_KINDS)
             else f" -- computing {'/'.join(args.kinds)} only, the rest "
                  "read from the cache"))
    print(f"phase 2: {len(arms)} arms, tensorized compiled with "
          f"'{args.train_compile_mode}'\n")

    print("phase 1 -- benchmark")
    bench = run_bench(cfg, ds, device, batches, modes, bench_rank, reps,
                      warmup, args.force, "bench" in args.phases, args.kinds)

    print("\nphase 2 -- training")
    if "train" in args.phases:
        print_budget(cfg, ds, len(arms))
    train = run_train(arms, cfg, ds, device, MODE_FROM_TAG[train_mode],
                      args.force, "train" in args.phases)

    if not bench and not train:
        print("no results")
        return

    if bench:
        bench.sort(key=lambda r: (r["kind"] != "dense", r["batch"],
                                  list(MODE_TAG.values()).index(r["compile_mode"])))
        write_csv(os.path.join(TABLES_DIR, "bench.csv"), bench, BENCH_COLUMNS)
        print("\n### phase 1 -- time and memory per configuration\n")
        print(markdown_table(bench, BENCH_COLUMNS, BENCH_FMT))

    rank_all: List[dict] = []
    summaries: List[dict] = []
    core_all: List[dict] = []
    core_dist_all: List[dict] = []
    if train:
        train = decorate_train(train)
        train.sort(key=lambda r: (r["kind"] != "dense", r["family"],
                                  r["max_rank"] or 0))
        write_csv(os.path.join(TABLES_DIR, "train.csv"), train, TRAIN_COLUMNS)
        print("\n### phase 2 -- quality per arm\n")
        print(markdown_table(train, TRAIN_COLUMNS, TRAIN_FMT))

        for r in train:
            core_all.extend(core_rows(r))
            core_dist_all.extend(core_rows(r, "core_dist"))
        if core_all:
            write_csv(os.path.join(TABLES_DIR, "core_stats.csv"), core_all,
                      CORE_COLUMNS)
        if core_dist_all:
            write_csv(os.path.join(TABLES_DIR, "core_dist.csv"), core_dist_all,
                      CORE_DIST_COLUMNS)

        for r in train:
            if r["kind"] != "adaptive":
                continue
            rows = rank_rows(r, cfg)
            rank_all.extend(rows)
            summaries.append(rank_summary(r, rows))
        if rank_all:
            write_csv(os.path.join(TABLES_DIR, "ranks.csv"), rank_all,
                      RANK_COLUMNS)
            write_csv(os.path.join(TABLES_DIR, "rank_summary.csv"), summaries,
                      RANK_SUMMARY_COLUMNS)
            print("\n### phase 4 -- final rank configuration\n")
            print(markdown_table(summaries, RANK_SUMMARY_COLUMNS,
                                 RANK_SUMMARY_FMT))

    # phase 3 table: static footprint next to the measured peaks
    mem_rows = [{"scope": "static", "name": r["arm"], "batch": cfg.batch_size,
                 "param_mb": r["param_mb"],
                 "effective_param_mb": r["effective_param_mb"],
                 "peak_bwd_mb": r["peak_memory_mb"],
                 "memory_comparable": r["memory_comparable"]}
                for r in train]
    mem_rows += [{"scope": "bench", "name": r["label"], "batch": r["batch"],
                  "param_mb": r["param_mb"],
                  "opt_state_mb": r.get("opt_state_mb"),
                  "peak_infer_mb": r["peak_infer_mb"],
                  "peak_bwd_mb": r["peak_bwd_mb"],
                  "peak_step_mb": r.get("peak_step_mb"),
                  "memory_comparable": r["memory_comparable"]}
                 for r in bench]
    if mem_rows:
        write_csv(os.path.join(TABLES_DIR, "memory.csv"), mem_rows,
                  MEMORY_COLUMNS)
    print(f"\ntables -> {TABLES_DIR}/")

    try:
        if bench:
            plot_bench(bench, batches)
        if train:
            plot_train(train)
        plot_memory(bench, train, batches)
        plot_ranks(summaries, rank_all)
        plot_core_diag(core_all, core_dist_all)
    except Exception as e:
        print(f"plotting failed: {type(e).__name__}: {e}")

    if track.backend != "off":
        with tracking.Run("report", "report", tags=["report"],
                          config={"phase": "report"}) as rep:
            rep.images(sorted(glob.glob(os.path.join(PLOTS_DIR, "*.png"))))
            rep.artifact("tables", "results",
                         sorted(glob.glob(os.path.join(TABLES_DIR, "*.csv"))))
            for r in train:
                rep.summary({f"val_ppl/{r['arm']}": r["final_val_ppl"],
                             f"params/{r['arm']}": r["effective_params"]})
            if rep.url:
                print(f"report run: {rep.url}")

    summarize(bench, train, summaries, core_all, batches)

    if train:
        best = min(train, key=lambda r: r["final_val_loss"])
        print(f"sample from {best['arm']}:\n{'-'*40}\n{printable(best['sample'])}\n"
              f"{'-'*40}")


if __name__ == "__main__":
    main()
