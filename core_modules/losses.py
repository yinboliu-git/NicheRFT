"""Loss functions for contrastive alignment and count reconstruction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableInfoNCELoss(nn.Module):
    """Symmetric InfoNCE with a learnable temperature."""

    def __init__(self, init_temperature: float = 0.07) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp_min(1e-4)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)
        logits = z_a @ z_b.t() / self.temperature
        labels = torch.arange(z_a.size(0), device=z_a.device)
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_ab + loss_ba)


def log_nb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    if theta.ndimension() == 1:
        theta = theta.unsqueeze(0)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    return (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )


def log_zinb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    if theta.ndimension() == 1:
        theta = theta.unsqueeze(0)

    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    return ((x < eps).float() * case_zero) + ((x > eps).float() * case_non_zero)


class ZINBLoss(nn.Module):
    """ZINB negative log-likelihood."""

    def __init__(self, ridge_lambda: float = 0.0) -> None:
        super().__init__()
        self.ridge_lambda = ridge_lambda

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        theta: torch.Tensor,
        pi: torch.Tensor,
    ) -> torch.Tensor:
        zinb = -log_zinb_positive(x=x, mu=mu, theta=theta, pi=pi).mean()
        if self.ridge_lambda > 0:
            zinb = zinb + self.ridge_lambda * torch.square(pi).mean()
        return zinb


class HuberLoss(nn.Module):
    """Huber loss for robust regression."""

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(pred, target, delta=self.delta)


class CosineSimilarityLoss(nn.Module):
    """1 - mean cosine similarity between predicted and target proteins."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sim = F.cosine_similarity(pred, target, dim=-1).mean()
        return 1.0 - sim


class PearsonCorrelationLoss(nn.Module):
    """1 - mean Pearson correlation per protein across the batch."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_c = pred - pred.mean(dim=0, keepdim=True)
        target_c = target - target.mean(dim=0, keepdim=True)
        pred_std = pred_c.std(dim=0, unbiased=False)
        target_std = target_c.std(dim=0, unbiased=False)
        r = (pred_c * target_c).mean(dim=0) / (pred_std * target_std + 1e-8)
        return 1.0 - r.mean()


class CompositeLoss(nn.Module):
    """Weighted combination of MSE + Cosine + Pearson losses."""

    def __init__(self, w_mse: float = 1.0, w_cos: float = 0.1, w_pearson: float = 0.1) -> None:
        super().__init__()
        self.w_mse = w_mse
        self.w_cos = w_cos
        self.w_pearson = w_pearson

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.w_mse * F.mse_loss(pred, target)
        if self.w_cos > 0:
            loss = loss + self.w_cos * CosineSimilarityLoss()(pred, target)
        if self.w_pearson > 0:
            loss = loss + self.w_pearson * PearsonCorrelationLoss()(pred, target)
        return loss
