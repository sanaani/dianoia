"""The intent head: frozen SONAR content vector (1024-d) in, nested 256-d intent vector out.

Matryoshka prefixes: the first 64 dimensions are trained to hold the core (every keep variant,
modifiers dropped included, is a positive there); 128 and 256 are trained with the same positives
except the modifier-dropped one, so modifiers can only live in the later blocks. Every move variant of
the same source is a hard negative at every prefix; the rest of the batch supplies ordinary negatives.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

PREFIXES = (64, 128, 256)
CORE_ONLY = {"drop_modifiers"}


class IntentHead(nn.Module):
    def __init__(self, d_in: int = 1024, d_hidden: int = 1024, d_out: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def prefix(z: torch.Tensor, d: int) -> torch.Tensor:
    return F.normalize(z[..., :d], dim=-1)


def loss(anchor: torch.Tensor, pos: dict[str, torch.Tensor], pos_mask: dict[str, torch.Tensor],
         neg: torch.Tensor, neg_mask: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    """InfoNCE at every prefix.

    anchor (B, D); pos[kind] (B, D) with pos_mask[kind] (B,) marking which rows exist;
    neg (B, M, D) hard negatives with neg_mask (B, M). Other rows' anchors and positives are
    in-batch negatives for each anchor.
    """
    total, count = anchor.new_zeros(()), 0
    b = anchor.shape[0]
    eye = torch.eye(b, dtype=torch.bool, device=anchor.device)
    for d in PREFIXES:
        a = prefix(anchor, d)
        h = prefix(neg, d)
        hard = torch.einsum("bd,bmd->bm", a, h) / tau
        hard = hard.masked_fill(~neg_mask, float("-inf"))
        batch_other = (a @ a.T / tau).masked_fill(eye, float("-inf"))
        for kind, p in pos.items():
            if d > PREFIXES[0] and kind in CORE_ONLY:
                continue
            m = pos_mask[kind]
            if not m.any():
                continue
            pp = prefix(p, d)
            s_pos = (a * pp).sum(-1, keepdim=True) / tau
            cross = (a @ pp.T / tau).masked_fill(eye, float("-inf"))
            logits = torch.cat([s_pos, hard, batch_other, cross], dim=1)
            nll = -(s_pos.squeeze(1) - torch.logsumexp(logits, dim=1))
            total = total + nll[m].mean()
            count += 1
    return total / max(count, 1)
