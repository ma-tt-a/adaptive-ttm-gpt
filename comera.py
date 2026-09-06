"""
Rank-adaptive optimization, ported from ziyangjoy/CoMERA (MNLI_trainer.py, utils.py, run_MNLI.sh).

Only their early stage is implemented.
"""

import weakref
from typing import Dict, List
import torch as t
import torch.nn.functional as F
from tensorized_layers import TTLinear


LR_TENSOR = 1e-4
LR_ORIGIN = 5e-5
LR_RANK = 1e-2
GAMMA = 1e-1

_RANK_CACHE: "weakref.WeakKeyDictionary[t.nn.Module, list]" = (
    weakref.WeakKeyDictionary())


def _rank_groups(model) -> list:
    """
    Rank parameters grouped by threshold, collected once per model.
    """
    if model not in _RANK_CACHE:
        groups: Dict[float, list] = {}
        for m in model.modules():
            if isinstance(m, TTLinear) and m.rank_params is not None:
                groups.setdefault(m.cfg.threshold, []).extend(m.rank_params)
        _RANK_CACHE[model] = list(groups.items())
    return _RANK_CACHE[model]


def rank_loss(model) -> t.Tensor:
    """
    sum(threshold(x, tol, 0)) / #{x > tol} over every rank parameter.
    """
    groups = _rank_groups(model)
    if not groups:
        return t.zeros((), device=next(model.parameters()).device)

    loss, count = 0.0, 0
    for tol, params in groups:
        x = t.cat([p.reshape(-1) for p in params])
        count = count + t.sum(x > tol)
        loss = loss + t.sum(F.threshold(x, tol, 0))
    return loss / count.clamp(min=1)


def comera_loss(model_loss: t.Tensor, model, gamma: float = GAMMA) -> t.Tensor:
    """
    Early-stage CoMERA objective: task loss + gamma * rank loss
    """
    if gamma == 0.0:
        return model_loss
    return model_loss + gamma * rank_loss(model)


def model_size(model) -> int:
    """
    # core parameters implied by the surviving ranks
    """
    return sum(m.effective_size() for m in model.modules()
               if isinstance(m, TTLinear))


def nominal_size(model) -> int:
    """
    # core parameters at full rank, i.e. what is actually allocated
    """
    return sum(p.numel() for m in model.modules()
               if isinstance(m, TTLinear) for p in m.cores)


def effective_params(model) -> int:
    """
    Total parameter count charging TT layers only for their surviving ranks
    """
    total = sum(p.numel() for p in model.parameters())
    return total - nominal_size(model) + model_size(model)


def rank_report(model) -> Dict[str, List[int]]:
    """
    Surviving TT-ranks per TT layer, keyed by module name
    """
    return {name: m.effective_rank() for name, m in model.named_modules()
            if isinstance(m, TTLinear)}


def param_groups(model, lr_tensor: float = LR_TENSOR,
                 lr_origin: float = LR_ORIGIN, lr_rank: float = LR_RANK):
    """
    Their three-way split: TT cores, rank parameters, everything else.
    """
    rank_ids = set()
    core_ids = set()
    for m in model.modules():
        if not isinstance(m, TTLinear):
            continue
        core_ids.update(id(p) for p in m.cores)
        if m.rank_params is not None:
            rank_ids.update(id(p) for p in m.rank_params)

    par_tensor, par_rank, par_origin = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in rank_ids:
            par_rank.append(p)
        elif id(p) in core_ids:
            par_tensor.append(p)
        else:
            par_origin.append(p)

    groups = [
        {"params": par_tensor, "lr": lr_tensor, "weight_decay": 0.0},
        {"params": par_origin, "lr": lr_origin, "weight_decay": 0.0},
    ]
    if par_rank:
        groups.append({"params": par_rank, "lr": lr_rank, "weight_decay": 0.0})
    return [g for g in groups if g["params"]]


def make_optimizer(model, lr_scale: float = 1.0, **kwargs):
    groups = param_groups(model, **kwargs)
    for g in groups:
        g["lr"] *= lr_scale
    return t.optim.AdamW(groups)
