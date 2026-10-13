"""Network building blocks for scRAG-Flow."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLUBlock(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, dropout: float) -> None:
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_hidden * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.fc(x).chunk(2, dim=-1)
        return self.dropout(value * torch.sigmoid(gate))


class GRNBlock(nn.Module):
    """Lightweight gated residual block for tabular-style data."""

    def __init__(self, dim_in: int, dim_hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.glu = GLUBlock(dim_in, dim_hidden, dropout)
        self.proj = nn.Linear(dim_hidden, dim_in)
        self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.glu(h))
        h = self.proj(h)
        return x + h


class GRNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [GRNBlock(hidden_dim, hidden_dim, dropout=dropout) for _ in range(depth)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.output_norm(h)
        return self.output_proj(h)


class ProjectionHead(nn.Module):
    def __init__(self, dim: int, proj_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ZINBDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.pi_head = nn.Linear(hidden_dim, output_dim)
        self.theta = nn.Parameter(torch.zeros(output_dim))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(z)
        mu = F.softplus(self.mu_head(h)) + 1e-4
        theta = F.softplus(self.theta) + 1e-4
        pi = self.pi_head(h)
        return mu, theta, pi


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    scale = math.log(10000) / max(half_dim - 1, 1)
    freqs = torch.exp(torch.arange(half_dim, device=t.device) * -scale)
    angles = t.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class AdaLNZeroBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.cond = nn.Linear(cond_dim, hidden_dim * 3)
        nn.init.zeros_(self.cond.weight)
        nn.init.zeros_(self.cond.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.cond(cond).chunk(3, dim=-1)
        h = self.norm(x)
        h = h * (1 + scale) + shift
        h = self.mlp(h)
        return x + torch.tanh(gate) * h


class AdaLNZeroMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        time_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [AdaLNZeroBlock(hidden_dim, hidden_dim, dropout=dropout) for _ in range(depth)]
        )
        self.output = nn.Linear(hidden_dim, input_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        t_emb = self.time_proj(timestep_embedding(t, self.time_dim))
        cond_emb = self.cond_proj(cond)
        joint = t_emb + cond_emb
        for block in self.blocks:
            h = block(h, joint)
        return self.output(h)
