"""Shared DiT building blocks for the Stage-3 flow backbones.

Extracted from the earlier one-step-FM implementation so the CAP-Flow core
(:mod:`models.cap_flow`) can be reproduced without that legacy module.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core_modules.attention import FeedForward, MultiHeadSelfAttention


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    if t.ndim != 1:
        t = t.reshape(-1)
    half = dim // 2
    if half == 0:
        return t[:, None]
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / max(half - 1, 1)
    )
    angles = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class DiTBlock(nn.Module):
    """AdaLN-Zero transformer block (DiT-style), time/condition modulated."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.1, ff_mult: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = MultiHeadSelfAttention(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = FeedForward(dim, ff_mult, dropout)
        self.adaln = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        s1, sh1, g1, s2, sh2, g2 = self.adaln(cond).chunk(6, dim=-1)
        h = self.norm1(x)
        h = h * (1.0 + s1[:, None]) + sh1[:, None]
        h = self.attn(h)
        x = x + g1[:, None] * h
        h = self.norm2(x)
        h = h * (1.0 + s2[:, None]) + sh2[:, None]
        h = self.ffn(h)
        x = x + g2[:, None] * h
        return x
