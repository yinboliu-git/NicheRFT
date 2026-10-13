"""Retrieval-centered conditional flow matching for NicheRFT.

A fitting-set-calibrated Gaussian source is constructed around the retrieved
protein prior. A DiT-style time-conditioned vector field is then trained with
conditional flow matching and evaluated by multi-step ODE integration.

This module implements flow matching, not diffusion; DiT is used only as the
vector-field backbone.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dit_blocks import DiTBlock, timestep_embedding


@dataclass
class CapFlowLossOutput:
    loss: torch.Tensor
    fm_loss: torch.Tensor
    mean_logvar: torch.Tensor


class PriorAdapter(nn.Module):
    """Fixed, fitting-set-calibrated Gaussian source distribution.

    The retrieval prior is de-shrunk around a *global fitting-set retrieval mean*
    and assigned a per-protein residual variance estimated from fitting cells.
    These quantities are registered as buffers rather than learned parameters,
    so the source distribution remains fixed while the conditional vector field
    is optimized. This keeps source calibration separate from flow matching and
    makes predictions independent of minibatch composition.
    """

    def __init__(self, protein_dim: int, rna_dim: int = 64, hidden: int = 128, n_near: int = 4, direct_hidden: int = 128) -> None:
        super().__init__()
        self.n_near = n_near
        self.register_buffer("de_shrink", torch.ones(protein_dim))
        self.register_buffer("retrieval_mean", torch.zeros(protein_dim))
        self.register_buffer("logvar_bias", torch.zeros(protein_dim))

    def set_logvar_bias(self, logvar_bias) -> None:
        with torch.no_grad():
            t = torch.as_tensor(logvar_bias, dtype=self.logvar_bias.dtype, device=self.logvar_bias.device)
            if t.shape != self.logvar_bias.shape:
                raise ValueError(f"logvar_bias shape {tuple(t.shape)} != {tuple(self.logvar_bias.shape)}")
            self.logvar_bias.copy_(t)

    def set_de_shrink(self, de_shrink) -> None:
        with torch.no_grad():
            t = torch.as_tensor(de_shrink, dtype=self.de_shrink.dtype, device=self.de_shrink.device)
            if t.shape != self.de_shrink.shape:
                raise ValueError(f"de_shrink shape {tuple(t.shape)} != {tuple(self.de_shrink.shape)}")
            self.de_shrink.copy_(t)

    def set_retrieval_mean(self, retrieval_mean) -> None:
        with torch.no_grad():
            t = torch.as_tensor(retrieval_mean, dtype=self.retrieval_mean.dtype, device=self.retrieval_mean.device)
            if t.shape != self.retrieval_mean.shape:
                raise ValueError(f"retrieval_mean shape {tuple(t.shape)} != {tuple(self.retrieval_mean.shape)}")
            self.retrieval_mean.copy_(t)

    def forward(self, x0: torch.Tensor, query_rna: torch.Tensor, spatial_z: torch.Tensor, similarity: torch.Tensor):
        del query_rna, spatial_z, similarity  # source statistics are fixed from fitting data
        mean = self.retrieval_mean.unsqueeze(0).to(dtype=x0.dtype, device=x0.device)
        mu = mean + (x0 - mean) * self.de_shrink.unsqueeze(0).to(dtype=x0.dtype, device=x0.device)
        logvar = self.logvar_bias.unsqueeze(0).to(dtype=x0.dtype, device=x0.device).expand_as(x0)
        return mu, logvar

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor, eps: torch.Tensor | None = None, sigma_clip: float = 1e-3) -> torch.Tensor:
        sigma = torch.exp(0.5 * logvar).clamp(min=sigma_clip)
        if eps is None:
            eps = torch.randn_like(mu)
        return mu + eps * sigma


class CapFlowVelocity(nn.Module):
    """DiT velocity field over the token sequence
    ``[protein(x_t) | query | reference | retrieved x k' | spatial x L]``."""

    def __init__(
        self,
        protein_dim: int,
        rna_dim: int = 64,
        hidden_dim: int = 512,
        depth: int = 4,
        n_heads: int = 8,
        time_dim: int = 64,
        dropout: float = 0.1,
        n_near: int = 4,
        k_retrieved: int = 8,
        rank: int | None = None,
        per_protein_head: bool = False,
    ) -> None:
        super().__init__()
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        self.rna_proj = nn.Linear(rna_dim, hidden_dim)
        self.displacement_embed = nn.Linear(2, hidden_dim)

        self.token_protein = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.token_prior = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.token_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.token_reference = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.token_retrieved = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.token_near = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.token_far = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.sim_proj = nn.Linear(1, hidden_dim)

        dim_head = hidden_dim // n_heads
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, n_heads, dim_head, dropout) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.per_protein_head = per_protein_head
        self.velocity_factors = None
        if per_protein_head:
            # cTP-net style: shared backbone -> per-protein independent nonlinear heads
            self.velocity_branches = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
                for _ in range(protein_dim)
            ])
            for b in self.velocity_branches:
                nn.init.zeros_(b[-1].weight)
                nn.init.zeros_(b[-1].bias)
        elif rank is not None and rank < protein_dim:
            self.velocity_factors = nn.Linear(hidden_dim, rank)
            self.velocity_head = nn.Linear(rank, protein_dim)
            nn.init.zeros_(self.velocity_head.weight)
            nn.init.zeros_(self.velocity_head.bias)
        else:
            self.velocity_head = nn.Linear(hidden_dim, protein_dim)
            nn.init.zeros_(self.velocity_head.weight)
            nn.init.zeros_(self.velocity_head.bias)

        self.n_near = n_near
        self.time_dim = time_dim

    def build_tokens(self, x_t, query_rna, reference_rna, retrieved_z, spatial_z, displacement, prior=None):
        protein_tok = self.protein_proj(x_t).unsqueeze(1) + self.token_protein
        tokens = [protein_tok]
        if prior is not None:
            prior_tok = self.protein_proj(prior).unsqueeze(1) + self.token_prior
            tokens.append(prior_tok)
        query_tok = self.rna_proj(query_rna).unsqueeze(1) + self.token_query
        reference_tok = self.rna_proj(reference_rna).unsqueeze(1) + self.token_reference
        retrieved_tok = self.rna_proj(retrieved_z) + self.token_retrieved
        spatial_tok = self.rna_proj(spatial_z) + self.displacement_embed(displacement)
        L = spatial_z.shape[1]
        n_near = min(self.n_near, L)
        roles = torch.cat(
            [self.token_near.repeat(1, n_near, 1), self.token_far.repeat(1, L - n_near, 1)],
            dim=1,
        )
        spatial_tok = spatial_tok + roles
        tokens.extend([query_tok, reference_tok, retrieved_tok, spatial_tok])
        return torch.cat(tokens, dim=1)

    def forward(self, x_t, t, query_rna, reference_rna, retrieved_z, spatial_z, displacement, similarity=None, prior=None):
        seq = self.build_tokens(x_t, query_rna, reference_rna, retrieved_z, spatial_z, displacement, prior)
        t = t.reshape(-1).to(x_t.dtype)
        cond = self.time_embed(timestep_embedding(t, self.time_dim))
        if similarity is not None:
            sim = similarity.reshape(-1, 1) if similarity.ndim == 1 else similarity
            cond = cond + self.sim_proj(sim)
        for block in self.blocks:
            seq = block(seq, cond)
        h = self.final_norm(seq[:, 0])
        if self.per_protein_head:
            return torch.cat([b(h) for b in self.velocity_branches], dim=1)
        if self.velocity_factors is not None:
            h = F.relu(self.velocity_factors(h))
        return self.velocity_head(h)


class CapFlow(nn.Module):
    """Combined prior adapter + velocity field."""

    def __init__(
        self,
        protein_dim: int,
        rna_dim: int = 64,
        hidden_dim: int = 512,
        depth: int = 4,
        n_heads: int = 8,
        time_dim: int = 64,
        dropout: float = 0.1,
        n_near: int = 4,
        k_retrieved: int = 8,
        prior_hidden: int = 128,
        rank: int | None = None,
        per_protein_head: bool = False,
    ) -> None:
        super().__init__()
        self.prior_adapter = PriorAdapter(protein_dim, rna_dim, prior_hidden, n_near)
        self.velocity = CapFlowVelocity(
            protein_dim, rna_dim, hidden_dim, depth, n_heads, time_dim, dropout, n_near, k_retrieved, rank, per_protein_head
        )

    def source(self, x0, query_rna, spatial_z, similarity, eps=None, sigma_clip=1e-3):
        mu, logvar = self.prior_adapter(x0, query_rna, spatial_z, similarity)
        z = self.prior_adapter.sample(mu, logvar, eps=eps, sigma_clip=sigma_clip)
        return z, mu, logvar

    def forward(self, x_t, t, query_rna, reference_rna, retrieved_z, spatial_z, displacement, similarity=None, prior=None):
        return self.velocity(x_t, t, query_rna, reference_rna, retrieved_z, spatial_z, displacement, similarity, prior)


def cap_flow_matching_loss(
    model: CapFlow,
    x0: torch.Tensor,
    x1: torch.Tensor,
    query_rna: torch.Tensor,
    reference_rna: torch.Tensor,
    retrieved_z: torch.Tensor,
    spatial_z: torch.Tensor,
    displacement: torch.Tensor,
    similarity: torch.Tensor | None = None,
    sigma_clip: float = 1e-3,
    mask_prob: float = 0.0,
    cond_drop_prob: float = 0.0,
    flow_path: str = "straight",
) -> CapFlowLossOutput:
    if flow_path == "noise":
        # Ablation "no retrieval, Gaussian source": drop the retrieval prior token AND
        # every retrieval context (neighbor latents / reference RNA / similarity), and
        # use the well-posed straight source--target coupling from pure Gaussian noise to data. The
        # flow is conditioned only on the query's own RNA and spatial niche.
        z = torch.randn_like(x1)
        t = torch.rand(z.shape[0], device=z.device, dtype=z.dtype)
        x_t = (1.0 - t[:, None]) * z + t[:, None] * x1
        target_v = x1 - z
        zero_ref = torch.zeros_like(reference_rna)
        zero_retz = torch.zeros_like(retrieved_z)
        zero_sim = torch.zeros_like(similarity) if similarity is not None else None
        v_pred = model.forward(x_t, t, query_rna, zero_ref, zero_retz, spatial_z, displacement, zero_sim, prior=None)
        fm_loss = F.mse_loss(v_pred, target_v)

        return CapFlowLossOutput(
            loss=fm_loss, fm_loss=fm_loss,
            mean_logvar=torch.tensor(0.0, device=x1.device),
        )

    mu, logvar = model.prior_adapter(x0, query_rna, spatial_z, similarity)
    z = model.prior_adapter.sample(mu.detach(), logvar.detach(), eps=None, sigma_clip=sigma_clip)  # fixed stochastic source for FM coupling

    # Optional source masking ablation: zero a random subset of the source's
    # protein values; the model must impute them from RNA + the unmasked proteins.
    z_masked = z
    if mask_prob > 0:
        mask = torch.rand_like(z) < mask_prob
        z_masked = torch.where(mask, torch.zeros_like(z), z)

    t = torch.rand(z.shape[0], device=z.device, dtype=z.dtype)
    if flow_path == "trig":
        # trigonometric curved path: x_t = cos(pi t/2) z + sin(pi t/2) x1.
        # The velocity is time-dependent, so multi-step Euler integration is needed
        # changing the prescribed probability path; one step is not generally exact.
        ang = (torch.pi / 2.0) * t
        ct = torch.cos(ang)[:, None]
        st = torch.sin(ang)[:, None]
        x_t = ct * z_masked + st * x1
        target_v = (torch.pi / 2.0) * (ct * x1 - st * z_masked)
    else:  # straight conditional coupling
        target_v = x1 - z_masked
        x_t = (1.0 - t[:, None]) * z_masked + t[:, None] * x1

    # classifier-free-guidance conditioning dropout: drop the retrieval context
    # (reference RNA / retrieved neighbors / similarity) so the model also learns
    # the unconditional velocity field.
    ref_fm, retz_fm, sim_fm = reference_rna, retrieved_z, similarity
    if cond_drop_prob > 0:
        keep = (~(torch.rand(z.shape[0], 1, device=z.device) < cond_drop_prob)).float()
        ref_fm = reference_rna * keep
        retz_fm = retrieved_z * keep.unsqueeze(-1)
        sim_fm = similarity * keep if similarity is not None else None

    v_pred = model.forward(x_t, t, query_rna, ref_fm, retz_fm, spatial_z, displacement, sim_fm)
    fm_loss = F.mse_loss(v_pred, target_v)

    # Pure conditional flow matching: there is no endpoint-regression shortcut
    # and no direct metric-optimization term in the training objective.
    return CapFlowLossOutput(loss=fm_loss, fm_loss=fm_loss, mean_logvar=logvar.mean())


@torch.no_grad()
def integrate_cap_flow(
    model: CapFlow,
    x0: torch.Tensor,
    query_rna: torch.Tensor,
    reference_rna: torch.Tensor,
    retrieved_z: torch.Tensor,
    spatial_z: torch.Tensor,
    displacement: torch.Tensor,
    similarity: torch.Tensor | None = None,
    nfe: int = 8,
    stochastic: bool = False,
    sigma_clip: float = 1e-3,
    cfg_scale: float = 1.0,
    flow_path: str = "straight",
) -> torch.Tensor:
    if not isinstance(nfe, int) or nfe < 1:
        raise ValueError("nfe must be a positive integer")
    if flow_path not in {"straight", "trig", "noise"}:
        raise ValueError("Unknown flow path")
    if flow_path == "noise":
        # Gaussian source N(0,I), straight conditional path, no retrieval context: integrate
        # from pure Gaussian noise to data conditioned only on own RNA + spatial niche.
        x = torch.randn_like(x0)
        zero_ref = torch.zeros_like(reference_rna)
        zero_retz = torch.zeros_like(retrieved_z)
        zero_sim = torch.zeros_like(similarity) if similarity is not None else None
        dt = 1.0 / nfe
        for step in range(nfe):
            t0 = torch.full((x.shape[0],), step * dt, device=x.device, dtype=x.dtype)
            v = model.forward(x, t0, query_rna, zero_ref, zero_retz, spatial_z, displacement, zero_sim, prior=None)
            x = x + dt * v
        return x

    mu, logvar = model.prior_adapter(x0, query_rna, spatial_z, similarity)
    eps = torch.randn_like(mu) if stochastic else torch.zeros_like(mu)
    x = model.prior_adapter.sample(mu, logvar, eps=eps, sigma_clip=sigma_clip)
    dt = 1.0 / nfe
    for step in range(nfe):
        t0 = torch.full((x.shape[0],), step * dt, device=x.device, dtype=x.dtype)
        if cfg_scale != 1.0:
            v_cond = model.forward(x, t0, query_rna, reference_rna, retrieved_z, spatial_z, displacement, similarity)
            v_uncond = model.forward(
                x, t0, query_rna,
                torch.zeros_like(reference_rna), torch.zeros_like(retrieved_z),
                spatial_z, displacement,
                torch.zeros_like(similarity) if similarity is not None else None,
            )
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = model.forward(x, t0, query_rna, reference_rna, retrieved_z, spatial_z, displacement, similarity)
        x = x + dt * v
    return x


def matched_residual_prediction(model, x0, query_rna, reference_rna, retrieved_z, spatial_z, displacement, similarity=None):
    """Same DiT and calibration as Full; one direct update, no Gaussian draw or ODE."""
    mu, _ = model.prior_adapter(x0, query_rna, spatial_z, similarity)
    t = torch.zeros(len(mu), device=mu.device, dtype=mu.dtype)
    residual = model(mu, t, query_rna, reference_rna, retrieved_z, spatial_z, displacement, similarity)
    return mu + residual


def matched_residual_loss(model, x0, target, query_rna, reference_rna, retrieved_z, spatial_z, displacement, similarity=None):
    prediction = matched_residual_prediction(model, x0, query_rna, reference_rna, retrieved_z, spatial_z, displacement, similarity)
    return F.mse_loss(prediction, target)
