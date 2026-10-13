"""Stage-1 contrastive autoencoder (GRN + split-invariant niche fusion)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from core_modules.losses import LearnableInfoNCELoss, ZINBLoss
from core_modules.networks import GRNEncoder, ProjectionHead, ZINBDecoder


@dataclass
class Stage1LossWeights:
    contrastive: float = 1.0
    rna_recon: float = 0.5
    protein_recon: float = 0.5


class SpatialGRNEncoder(nn.Module):
    """GRN encoder with an additive split-invariant niche branch.

    The center RNA is encoded by an unchanged GRN; a separate gated MLP encodes
    the niche context (niche RNA + transcriptomic delta + displacement summary)
    and is added to the center latent. No attention is used, and the niche
    signal (neighbor RNA + relative displacement) is translation-invariant, so
    it does not leak the train/test spatial split.
    """

    def __init__(
        self,
        rna_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        depth: int = 2,
        dropout: float = 0.1,
        spatial_hidden: int = 128,
        use_niche: bool = True,
    ) -> None:
        super().__init__()
        self.use_niche = use_niche
        self.center_encoder = GRNEncoder(rna_dim, hidden_dim, latent_dim, depth, dropout)
        self.spatial_mlp = nn.Sequential(
            nn.Linear(2 * rna_dim + 5, spatial_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spatial_hidden, latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        center: torch.Tensor,
        niche: torch.Tensor,
        disp_ctx: torch.Tensor,
    ) -> torch.Tensor:
        # center: [N, G]; niche: [N, G]; disp_ctx: [N, 5]
        z_center = self.center_encoder(center)
        if not self.use_niche:
            return self.norm(z_center)
        delta = center - niche
        spatial_input = torch.cat([niche, delta, disp_ctx], dim=-1)
        z_niche = self.spatial_mlp(spatial_input)
        return self.norm(z_center + z_niche)


class ContrastiveAutoencoder(nn.Module):
    def __init__(
        self,
        rna_dim: int,
        protein_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        proj_dim: int = 32,
        depth: int = 2,
        dropout: float = 0.1,
        spatial_hidden: int = 128,
        use_niche: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.rna_encoder = SpatialGRNEncoder(
            rna_dim, hidden_dim, latent_dim, depth, dropout, spatial_hidden, use_niche
        )
        self.protein_encoder = GRNEncoder(protein_dim, hidden_dim, latent_dim, depth, dropout)
        self.rna_proj = ProjectionHead(latent_dim, proj_dim)
        self.protein_proj = ProjectionHead(latent_dim, proj_dim)
        self.rna_decoder = ZINBDecoder(latent_dim, hidden_dim, rna_dim)
        self.protein_decoder = ZINBDecoder(latent_dim, hidden_dim, protein_dim)
        self.contrastive_loss = LearnableInfoNCELoss()
        self.zinb_loss = ZINBLoss(ridge_lambda=1e-4)

    def encode_rna(
        self, center: torch.Tensor, niche: torch.Tensor, disp_ctx: torch.Tensor
    ) -> torch.Tensor:
        return self.rna_encoder(center, niche, disp_ctx)

    def encode_protein(self, x: torch.Tensor) -> torch.Tensor:
        return self.protein_encoder(x)

    def forward(
        self,
        center_rna: torch.Tensor,
        niche_rna: torch.Tensor,
        disp_ctx: torch.Tensor,
        protein: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z_rna = self.encode_rna(center_rna, niche_rna, disp_ctx)
        z_protein = self.encode_protein(protein)
        p_rna = self.rna_proj(z_rna)
        p_protein = self.protein_proj(z_protein)
        rna_mu, rna_theta, rna_pi = self.rna_decoder(z_rna)
        protein_mu, protein_theta, protein_pi = self.protein_decoder(z_protein)
        return {
            "z_rna": z_rna,
            "z_protein": z_protein,
            "p_rna": p_rna,
            "p_protein": p_protein,
            "rna_mu": rna_mu,
            "rna_theta": rna_theta,
            "rna_pi": rna_pi,
            "protein_mu": protein_mu,
            "protein_theta": protein_theta,
            "protein_pi": protein_pi,
        }

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        weights: Stage1LossWeights,
    ) -> dict[str, torch.Tensor]:
        out = self.forward(
            batch["rna_norm"], batch["niche_rna"], batch["disp_ctx"], batch["protein_norm"]
        )
        loss_contrastive = self.contrastive_loss(out["p_rna"], out["p_protein"])
        loss_rna = self.zinb_loss(
            batch["rna_raw"], out["rna_mu"], out["rna_theta"], out["rna_pi"]
        )
        loss_protein = self.zinb_loss(
            batch["protein_raw"], out["protein_mu"], out["protein_theta"], out["protein_pi"]
        )
        loss = (
            weights.contrastive * loss_contrastive
            + weights.rna_recon * loss_rna
            + weights.protein_recon * loss_protein
        )
        return {
            "loss": loss,
            "contrastive": loss_contrastive,
            "rna_recon": loss_rna,
            "protein_recon": loss_protein,
        }
