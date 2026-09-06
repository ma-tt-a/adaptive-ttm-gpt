"""
Experiment tracking for the harness: TensorBoard by default, wandb optional.

Optional and non-fatal by construction: a multi-hour training run must not die
because a logging call failed or a machine has no network. Every entry point
degrades to a no-op, and `Run` is a working object even when the backend is
missing entirely, so callers never branch on whether tracking is on.
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

BACKENDS = ("off", "tensorboard", "wandb")
PROJECT = "adaptive-ttm-gpt"
TB_SUBDIR = "tb"
FLUSH_SECS = 30   # the point of this is watching a run live, not after it


@dataclass
class TrackConfig:
    backend: str = "off"
    dir: str = "results"    # <dir>/tb/<group>/<run> or <dir>/wandb
    group: str = ""         # one invocation of the harness
    project: str = PROJECT  # wandb only
    entity: Optional[str] = None
    mode: str = "online"    # wandb only: online | offline | disabled


_CFG = TrackConfig()
_MUTED = False              # logging errors are reported once, then swallowed


def session_group() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _import(backend: str):
    """
    The backend module, or None with one line saying why not.
    """
    try:
        if backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            return SummaryWriter
        import wandb
        return wandb
    except Exception as e:
        pkg = "tensorboard" if backend == "tensorboard" else "wandb"
        print(f"  {pkg} not importable ({type(e).__name__}: {e}) "
              f"-- tracking off. pip install {pkg}")
        return None


def _has_wandb_key(wandb) -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        return bool(wandb.api.api_key)
    except Exception:
        return False


def configure(backend: str = "off", dir: str = "results", group: str = "",
              project: str = PROJECT, entity: Optional[str] = None,
              mode: str = "online") -> TrackConfig:
    """
    Resolve the tracking settings once, before any run is created.
    """
    global _CFG
    _CFG = TrackConfig(backend=backend, dir=dir, group=group or session_group(),
                       project=project, entity=entity, mode=mode)
    if _CFG.backend == "off":
        return _CFG

    mod = _import(_CFG.backend)
    if mod is None:
        _CFG.backend = "off"
        return _CFG

    if _CFG.backend == "tensorboard":
        os.makedirs(log_dir(), exist_ok=True)
    else:
        # on a server the key comes from WANDB_API_KEY; in colab `!python` is
        # not a tty, so an interactive login prompt would block the job
        # forever. A missing key downgrades to offline -- the data is still on
        # disk for a later `wandb sync` -- instead of hanging
        if _CFG.mode == "online" and not _has_wandb_key(mod):
            print(f"  no WANDB_API_KEY and no cached login -- logging "
                  f"offline. Set the key (or run `wandb login`), then push "
                  f"with `wandb sync {os.path.join(_CFG.dir, 'wandb')}`")
            _CFG.mode = "offline"
        os.makedirs(_CFG.dir, exist_ok=True)
    return _CFG


def config() -> TrackConfig:
    return _CFG


def log_dir() -> str:
    """
    Where TensorBoard should be pointed. One directory per invocation, so
    `--logdir results/tb` overlays every session and `--logdir results/tb/<group>`
    shows just one.
    """
    return os.path.join(_CFG.dir, TB_SUBDIR, _CFG.group)


def _scalar(v: Any) -> Optional[float]:
    """
    Numbers TensorBoard will accept: real, finite, not a bool-as-metric.
    """
    if isinstance(v, bool):
        return float(v)
    if not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) else None


def _hparam(v: Any) -> Any:
    """
    add_hparams only takes scalars and strings; everything else is stringified
    rather than dropped, since the config is what tells two arms apart.
    """
    if isinstance(v, (int, float, bool, str)):
        return v
    return "None" if v is None else str(v)


class _TensorBoardRun:
    def __init__(self, writer_cls, job_type: str, name: str,
                 config: Dict[str, Any], tags: List[str]):
        self.dir = os.path.join(log_dir(), f"{job_type}-{name}")
        self.writer = writer_cls(log_dir=self.dir, flush_secs=FLUSH_SECS)
        self.config = {k: _hparam(v) for k, v in config.items()}
        if tags:
            self.config["tags"] = ",".join(tags)
        self.last_step = 0
        # the config as text, because the HParams tab only fills in at finish
        self.writer.add_text(
            "config", "\n".join(f"- **{k}**: {v}"
                                for k, v in sorted(self.config.items())), 0)

    def log(self, data: Dict[str, Any], it: Optional[int]):
        step = self.last_step if it is None else it
        self.last_step = max(self.last_step, step)
        for k, v in data.items():
            x = _scalar(v)
            if x is not None:
                self.writer.add_scalar(k, x, global_step=step)

    def summary(self, data: Dict[str, Any]):
        # scalars go in at the last step so they show up on the curves too;
        # strings (the text sample) become their own text panel
        for k, v in data.items():
            x = _scalar(v)
            if x is not None:
                self.writer.add_scalar(k, x, global_step=self.last_step)
            elif isinstance(v, str):
                self.writer.add_text(k, v, self.last_step)
        metrics = {k: _scalar(v) for k, v in data.items()
                   if _scalar(v) is not None}
        if metrics and self.config:
            # the HParams tab: one row per run, config against final metrics.
            # run_name keeps it in this run's directory instead of spawning a
            # sibling one per call
            self.writer.add_hparams(self.config, metrics, run_name=".")

    def table(self, key: str, columns: List[str], rows: List[dict]):
        head = "| " + " | ".join(columns) + " |"
        sep = "|" + "|".join("---" for _ in columns) + "|"
        body = [
            "| " + " | ".join(
                f"{r.get(c):.4g}" if isinstance(r.get(c), float)
                else str(r.get(c)) for c in columns) + " |"
            for r in rows]
        self.writer.add_text(key, "\n".join([head, sep] + body),
                             self.last_step)

    def images(self, paths: List[str], prefix: str):
        import matplotlib.image as mpimg
        for p in paths:
            img = mpimg.imread(p)          # HxWx4 float in [0, 1]
            name = os.path.splitext(os.path.basename(p))[0]
            self.writer.add_image(f"{prefix}/{name}", img, self.last_step,
                                  dataformats="HWC")

    def artifact(self, name: str, kind: str, paths: List[str]):
        # TensorBoard has no artifact store; the csvs are already in
        # results/tables, so this only records what was produced
        if paths:
            self.writer.add_text(
                name, "\n".join(f"- `{p}`" for p in paths), self.last_step)

    def finish(self):
        self.writer.close()

    @property
    def url(self) -> Optional[str]:
        return None


class _WandbRun:
    def __init__(self, wandb, job_type: str, name: str,
                 config: Dict[str, Any], tags: List[str]):
        self.wandb = wandb
        kwargs = dict(
            project=_CFG.project, entity=_CFG.entity, group=_CFG.group,
            job_type=job_type, name=f"{name}-{_CFG.group}", config=config,
            tags=tags, dir=_CFG.dir,
            # mode is passed here, not through WANDB_MODE: an explicit
            # Settings object outranks the environment, and once a session
            # exists wandb ignores env changes and warns about it
            mode=_CFG.mode, settings=wandb.Settings(quiet=True))
        # arms run one after another in the same process, and an arm that
        # raised leaves its run open; finish_previous closes it. The boolean
        # spelling is the pre-0.19 one, kept as a fallback
        try:
            self.run = wandb.init(reinit="finish_previous", **kwargs)
        except (TypeError, ValueError):
            self.run = wandb.init(reinit=True, **kwargs)
        # every series is plotted against the training iteration rather than
        # wandb's own monotonic step, so arms logged at different cadences
        # line up
        self.run.define_metric("iter")
        self.run.define_metric("*", step_metric="iter")

    def log(self, data: Dict[str, Any], it: Optional[int]):
        payload = dict(data)
        if it is not None:
            payload["iter"] = it
        self.run.log(payload)

    def summary(self, data: Dict[str, Any]):
        self.run.summary.update(data)

    def table(self, key: str, columns: List[str], rows: List[dict]):
        tbl = self.wandb.Table(columns=list(columns))
        for r in rows:
            tbl.add_data(*[r.get(c) for c in columns])
        self.run.log({key: tbl})

    def images(self, paths: List[str], prefix: str):
        self.run.log({
            f"{prefix}/{os.path.splitext(os.path.basename(p))[0]}":
            self.wandb.Image(p) for p in paths})

    def artifact(self, name: str, kind: str, paths: List[str]):
        art = self.wandb.Artifact(name, type=kind)
        for p in paths:
            art.add_file(p)
        self.run.log_artifact(art)

    def finish(self):
        self.run.finish()

    @property
    def url(self) -> Optional[str]:
        try:
            return self.run.url
        except Exception:
            return None


class Run:
    """
    One tracked run, or a no-op with the same interface.

    Every method swallows its exception after reporting the first one: the
    harness is the thing being measured, and losing it to a logging failure
    would be the worse outcome.
    """

    def __init__(self, job_type: str, name: str,
                 config: Optional[Dict[str, Any]] = None,
                 tags: Optional[List[str]] = None):
        self.name = name
        self.impl = None
        if _CFG.backend == "off":
            return
        mod = _import(_CFG.backend)
        if mod is None:
            return
        try:
            cls = (_TensorBoardRun if _CFG.backend == "tensorboard"
                   else _WandbRun)
            self.impl = cls(mod, job_type, name, config or {}, tags or [])
        except Exception as e:
            print(f"  tracking init failed for {name} "
                  f"({type(e).__name__}: {e}) -- off for this run")
            self.impl = None

    def _guard(self, what: str, fn):
        global _MUTED
        if self.impl is None:
            return
        try:
            fn()
        except Exception as e:
            if not _MUTED:
                _MUTED = True
                print(f"  tracking {what} failed ({type(e).__name__}: {e}) "
                      f"-- further logging errors are silenced")

    @property
    def url(self) -> Optional[str]:
        return self.impl.url if self.impl is not None else None

    def log(self, data: Dict[str, Any], it: Optional[int] = None):
        self._guard("log", lambda: self.impl.log(data, it))

    def summary(self, data: Dict[str, Any]):
        self._guard("summary", lambda: self.impl.summary(data))

    def table(self, key: str, columns: List[str], rows: List[dict]):
        self._guard("table", lambda: self.impl.table(key, columns, rows))

    def images(self, paths: List[str], prefix: str = "plots"):
        if paths:
            self._guard("images", lambda: self.impl.images(paths, prefix))

    def artifact(self, name: str, kind: str, paths: List[str]):
        if paths:
            self._guard("artifact",
                        lambda: self.impl.artifact(name, kind, paths))

    def finish(self):
        self._guard("finish", lambda: self.impl.finish())
        self.impl = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finish()
        return False
