"""Normal and structure losses for the V421-V600 formal route.

These losses are deliberately small and dependency-light so they can be reused
by a Modal trainer without pulling in the old residual composer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NormalStructureLossWeights:
    normal_cosine: float = 1.0
    normal_nonzero: float = 0.25
    local_smoothness_margin: float = 0.5


def normalize_vectors(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def normal_cosine_loss(pred_normal: torch.Tensor, target_normal: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    pred = normalize_vectors(pred_normal)
    target = normalize_vectors(target_normal)
    loss = 1.0 - (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    if valid_mask is not None:
        mask = valid_mask.to(loss.dtype)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)
    return loss.mean()


def normal_nonzero_penalty(pred_normal: torch.Tensor, valid_mask: torch.Tensor | None = None, threshold: float = 1e-4) -> torch.Tensor:
    mag = pred_normal.norm(dim=-1)
    penalty = F.relu(threshold - mag)
    if valid_mask is not None:
        mask = valid_mask.to(penalty.dtype)
        return (penalty * mask).sum() / mask.sum().clamp_min(1.0)
    return penalty.mean()


def local_surface_continuity(points: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Finite-difference continuity for points shaped (..., H, W, 3)."""
    dx = points[..., :, 1:, :] - points[..., :, :-1, :]
    dy = points[..., 1:, :, :] - points[..., :-1, :, :]
    lx = dx.norm(dim=-1)
    ly = dy.norm(dim=-1)
    if valid_mask is None:
        return 0.5 * (lx.mean() + ly.mean())
    mx = (valid_mask[..., :, 1:] & valid_mask[..., :, :-1]).to(lx.dtype)
    my = (valid_mask[..., 1:, :] & valid_mask[..., :-1, :]).to(ly.dtype)
    return 0.5 * ((lx * mx).sum() / mx.sum().clamp_min(1.0) + (ly * my).sum() / my.sum().clamp_min(1.0))


def anti_smoothing_margin_loss(true_score: torch.Tensor, smoothing_score: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Penalize smoothing controls that meet or exceed the true semantic score."""
    return F.relu(smoothing_score + margin - true_score).mean()


def combined_normal_structure_loss(
    pred_points: torch.Tensor,
    pred_normal: torch.Tensor,
    target_normal: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    weights: NormalStructureLossWeights | None = None,
) -> dict[str, torch.Tensor]:
    weights = weights or NormalStructureLossWeights()
    normal_loss = normal_cosine_loss(pred_normal, target_normal, valid_mask)
    nonzero_loss = normal_nonzero_penalty(pred_normal, valid_mask)
    continuity = local_surface_continuity(pred_points, valid_mask)
    total = weights.normal_cosine * normal_loss + weights.normal_nonzero * nonzero_loss + continuity
    return {
        "loss": total,
        "normal_cosine_loss": normal_loss,
        "normal_nonzero_penalty": nonzero_loss,
        "local_surface_continuity": continuity,
    }
