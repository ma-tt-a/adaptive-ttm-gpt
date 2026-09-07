"""
nanoGPT-style model with dense and TT-tensorized modes.

Mirrors https://github.com/karpathy/nanoGPT/blob/master/model.py; the only
structural change is that the four linears inside each block are built by
_linear, which returns either nn.Linear or TTLinear.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from tensorized_layers import TTLinear, TTLinearConfig

# per-role TT factorizations for n_embd = 256; d = 3, so 6 cores
TT_SHAPES_256 = {
    "c_attn": (t.Size([4, 8, 8]), t.Size([8, 8, 12])),    # 256 -> 768
    "attn_proj": (t.Size([4, 8, 8]), t.Size([4, 8, 8])),  # 256 -> 256
    "c_fc": (t.Size([4, 8, 8]), t.Size([8, 8, 16])),      # 256 -> 1024
    "mlp_proj": (t.Size([8, 8, 16]), t.Size([4, 8, 8])),  # 1024 -> 256
}

# smoke-test size, n_embd = 64
TT_SHAPES_64 = {
    "c_attn": (t.Size([4, 4, 4]), t.Size([4, 4, 12])),    # 64 -> 192
    "attn_proj": (t.Size([4, 4, 4]), t.Size([4, 4, 4])),  # 64 -> 64
    "c_fc": (t.Size([4, 4, 4]), t.Size([4, 4, 16])),      # 64 -> 256
    "mlp_proj": (t.Size([4, 4, 16]), t.Size([4, 4, 4])),  # 256 -> 64
}

# gpt2-small width, n_embd = 768; d = 3, so 6 cores
TT_SHAPES_768 = {
    "c_attn": (t.Size([8, 8, 12]), t.Size([12, 12, 16])),   # 768 -> 2304
    "attn_proj": (t.Size([8, 8, 12]), t.Size([8, 8, 12])),  # 768 -> 768
    "c_fc": (t.Size([8, 8, 12]), t.Size([12, 16, 16])),     # 768 -> 3072
    "mlp_proj": (t.Size([12, 16, 16]), t.Size([8, 8, 12])),  # 3072 -> 768
}

TT_SHAPES = {768: TT_SHAPES_768, 256: TT_SHAPES_256, 64: TT_SHAPES_64}

# named model sizes. The values are exactly the fields TrainConfig carries, so
# a preset is applied by setattr and nothing else knows about it. "base" is the
# harness default spelled out, so selecting it changes no cache key.
MODEL_PRESETS = {
    "smoke": dict(n_layer=2, n_head=4, n_embd=64, block_size=64),
    "base": dict(n_layer=6, n_head=8, n_embd=256, block_size=128),
    # radford et al. 2019, 124M parameters: 12 layers, 12 heads, 768 wide,
    # 1024 context. With the gpt2 tokenizer (vocab 50257) this is gpt2-small
    # as published -- the TT arms replace the 4 linears per block, nothing else
    "gpt2-small": dict(n_layer=12, n_head=12, n_embd=768, block_size=1024),
}


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = True
    init_std: float = 2e-2
    # tensorization
    tensorized: bool = False
    max_rank: int = 30
    adaptive: bool = False
    threshold: float = 1e-2
    tt_shapes: Optional[Dict[str, Tuple[t.Size, t.Size]]] = None

    def __post_init__(self):
        if self.tensorized and self.tt_shapes is None:
            assert self.n_embd in TT_SHAPES, \
                f"no TT factorization for n_embd={self.n_embd}"
            self.tt_shapes = TT_SHAPES[self.n_embd]


def _linear(cfg: GPTConfig, in_f: int, out_f: int, role: str) -> nn.Module:
    """
    TTLinear when the model is tensorized, otherwise a plain nn.Linear
    """
    if not cfg.tensorized:
        layer = nn.Linear(in_f, out_f, bias=cfg.bias)
        nn.init.normal_(layer.weight, mean=0.0, std=cfg.init_std)
        if cfg.bias:
            nn.init.zeros_(layer.bias)
        return layer

    in_shape, out_shape = cfg.tt_shapes[role]
    assert in_shape.numel() == in_f and out_shape.numel() == out_f, \
        f"{role}: TT shapes {tuple(in_shape)}->{tuple(out_shape)} " \
        f"do not match {in_f}->{out_f}"
    tt_cfg = TTLinearConfig.from_max_rank(
        in_shape, out_shape, cfg.max_rank,
        adaptive=cfg.adaptive,
        bias=cfg.bias,
        threshold=cfg.threshold,
        init_std=cfg.init_std,
    )
    return TTLinear(tt_cfg)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.c_attn = _linear(cfg, cfg.n_embd, 3 * cfg.n_embd, "c_attn")
        self.c_proj = _linear(cfg, cfg.n_embd, cfg.n_embd, "attn_proj")
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = _linear(cfg, cfg.n_embd, 4 * cfg.n_embd, "c_fc")
        self.gelu = nn.GELU()
        self.c_proj = _linear(cfg, 4 * cfg.n_embd, cfg.n_embd, "mlp_proj")
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying, as in nanoGPT; embeddings stay dense in every mode
        self.transformer.wte.weight = self.lm_head.weight

        nn.init.normal_(self.transformer.wte.weight,
                        mean=0.0, std=cfg.init_std)
        nn.init.normal_(self.transformer.wpe.weight,
                        mean=0.0, std=cfg.init_std)

    def tt_layers(self) -> List[TTLinear]:
        return [m for m in self.modules() if isinstance(m, TTLinear)]

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, \
            f"sequence length {T} > block size {self.cfg.block_size}"
        pos = t.arange(T, dtype=t.long, device=idx.device)
        x = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @t.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = t.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = t.cat([idx, t.multinomial(probs, num_samples=1)], dim=1)
        return idx
