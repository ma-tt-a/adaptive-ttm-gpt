from dataclasses import dataclass
from utils import TTMatVec, build_cores_gauss, get_xavier_std, get_uniform_rank
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
    init_std: float = 2e-2  # std of the contracted matrix, not of the cores

    @classmethod
    def from_max_rank(cls, in_shape: t.Size, out_shape: t.Size,
                      max_rank: int, **kwargs):
        rank = get_uniform_rank(in_shape, out_shape, max_rank)
        return cls(in_shape=in_shape, out_shape=out_shape, rank=rank, **kwargs)

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
        std = get_xavier_std(rank, self.cfg.init_std)
        res = [nn.Parameter(core)
               for core in build_cores_gauss(shape, rank, std)]
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

    def effective_rank(self) -> List[int]:
        """
        Surviving TT-ranks, i.e. rank entries above the threshold
        """
        if not self.cfg.adaptive:
            return list(self.cfg.rank[1:-1])
        threshold = self.cfg.threshold
        return [int(t.sum(x > threshold)) for x in self.rank_params]

    def effective_size(self) -> int:
        """
        # core parameters implied by the surviving ranks
        """
        R = [1] + self.effective_rank() + [1]
        if any(r == 0 for r in R):
            return 0
        return sum(G.shape[1] * R[n] * R[n + 1]
                   for n, G in enumerate(self.cores))

    def forward(self, X):
        # TTMatVec is a matrix product: fold any leading dims into the batch
        sh = X.shape
        Y = TTMatVec.apply(X.reshape(-1, sh[-1]), *self.get_cores())
        out = Y.reshape(sh[:-1] + (self.J,))
        return out + self.bias if self.cfg.bias else out
