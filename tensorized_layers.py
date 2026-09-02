from dataclasses import dataclass
from utils import TTMatVec, build_cores_gaus
from typing import List
import torch as t
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TTLinearConfig:
    in_shape: t.Size  # I_1, ..., I_d
    out_shape: t.Size  # J_1, ..., J_d
    rank: t.Size  # 1, R_1, ..., R_2d-1, 1
    threshold: float = 1e-2
    adaptive: bool = True
    bias: bool = True

    def __post_init__(self):
        self.N = len(self.in_shape) + len(self.out_shape)
        assert len(self.rank) == self.N + \
            1, f"TT-rank: expected={self.N + 1}, given={len(self.rank)}"
        assert self.rank[0] == self.rank[-1] == 1, f"It is not TT format: R_0 != 1 or R_-1 != 1"


class TTLinear(nn.Module):
    def __init__(self, cfg: TTLinearConfig):
        super().__init__()
        self.cfg = cfg
        self.I, self.J = cfg.in_shape.numel(), cfg.out_shape.numel()
        self.cores = nn.ParameterList(self._build_cores())
        self.rank_params = nn.ParameterList(
            self._build_rank()
        ) if cfg.adaptive else None
        self.bias = nn.Parameter(t.zeros(self.J)) if cfg.bias else None

    def _build_cores(self) -> List[nn.Parameter]:
        rank = self.cfg.rank
        shape = self.cfg.in_shape + self.cfg.out_shape
        # res = [nn.Parameter(core) for core in build_cores_gaus(shape, rank)]
        res = [nn.Parameter(core)
               for core in build_cores_gaus(shape, rank)]
        return res

    def _build_rank(self) -> List[nn.Parameter]:
        rank = self.cfg.rank
        rank_params = [nn.Parameter(t.ones(rank[n]))
                       for n in range(1, self.cfg.N)]
        return rank_params

    def get_rank_mask(self) -> List[t.Tensor]:
        assert self.rank_params is not None, "The mask is only available in adaptive mode"
        mask = []
        threshold = self.cfg.threshold
        for x in self.rank_params:
            y = F.threshold(x, threshold, 0)
            mask.append(y)
        return mask

    def get_cores(self, masked: bool = True) -> List[t.Tensor]:
        if not self.cfg.adaptive or not masked:
            return list(self.cores)
        res = []
        mask = self.get_rank_mask()
        for n in range(len(mask)):
            D = mask[n][None, None, :]
            G = self.cores[n]
            res.append(G * D)
        res.append(self.cores[-1])
        return res

    def forward(self, X):
        out = TTMatVec.apply(X, *self.get_cores())
        return out + self.bias if self.cfg.bias else out
