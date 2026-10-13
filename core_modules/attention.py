"""Transformer building blocks shared by the Stage-1 and Stage-3 encoders."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(b, n, self.heads, -1).transpose(1, 2)
        k = k.reshape(b, n, self.heads, -1).transpose(1, 2)
        v = v.reshape(b, n, self.heads, -1).transpose(1, 2)
        sim = (q @ k.transpose(-1, -2)) * self.scale
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (multi-head self-attention + GEGLU FFN)."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 64, dropout: float = 0.0, ff_mult: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ff_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x
