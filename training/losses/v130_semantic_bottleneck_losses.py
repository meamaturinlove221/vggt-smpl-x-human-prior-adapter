"""Losses for the V130 semantic-bottleneck route."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SemanticBottleneckLossWeights:
    point: float = 1.0
    depth: float = 0.2
    normal: float = 0.25
    occupancy: float = 0.1
    reliability: float = 0.05
    canonical_aux: float = 0.25
    body_part_aux: float = 0.2
    vertex_aux: float = 0.05
    skinning_aux: float = 0.1
    contrastive_aux: float = 0.15
    causal_margin: float = 0.25
    support_residual_penalty: float = 0.1


def _optional_zero(device: torch.device) -> torch.Tensor:
    return torch.zeros((), device=device)


def geometry_losses(
    model_out: dict[str, Any],
    target: dict[str, torch.Tensor],
    weights: SemanticBottleneckLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or SemanticBottleneckLossWeights()
    outputs = model_out["outputs"]
    device = outputs["delta_point"].device
    terms: dict[str, torch.Tensor] = {}
    terms["point"] = F.smooth_l1_loss(outputs["delta_point"], target["delta_point"])
    if "depth" in target and "depth" in outputs:
        terms["depth"] = F.smooth_l1_loss(outputs["depth"], target["depth"])
    else:
        terms["depth"] = _optional_zero(device)
    if "normal" in target:
        pred_normal = F.normalize(outputs["delta_normal"], dim=-1, eps=1e-6)
        true_normal = F.normalize(target["normal"], dim=-1, eps=1e-6)
        terms["normal"] = (1.0 - (pred_normal * true_normal).sum(dim=-1)).mean()
    else:
        terms["normal"] = _optional_zero(device)
    if "occupancy" in target:
        terms["occupancy"] = F.binary_cross_entropy(outputs["occupancy"].clamp(1e-5, 1 - 1e-5), target["occupancy"])
    else:
        terms["occupancy"] = _optional_zero(device)
    if "reliability" in target:
        terms["reliability"] = F.binary_cross_entropy(
            outputs["reliability"].clamp(1e-5, 1 - 1e-5),
            target["reliability"],
        )
    else:
        terms["reliability"] = _optional_zero(device)

    total = (
        weights.point * terms["point"]
        + weights.depth * terms["depth"]
        + weights.normal * terms["normal"]
        + weights.occupancy * terms["occupancy"]
        + weights.reliability * terms["reliability"]
    )
    metrics = {f"loss_{k}": float(v.detach().cpu()) for k, v in terms.items()}
    metrics["loss_geometry_total"] = float(total.detach().cpu())
    return total, metrics


def semantic_auxiliary_losses(
    model_out: dict[str, Any],
    aux_target: dict[str, torch.Tensor],
    weights: SemanticBottleneckLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or SemanticBottleneckLossWeights()
    aux = model_out["semantic"]["aux"]
    device = aux["canonical_xyz"].device
    terms: dict[str, torch.Tensor] = {}
    terms["canonical_aux"] = F.smooth_l1_loss(aux["canonical_xyz"], aux_target["canonical_xyz"])
    if "body_part" in aux_target:
        terms["body_part_aux"] = F.cross_entropy(aux["body_part_logits"], aux_target["body_part"].long())
    else:
        terms["body_part_aux"] = _optional_zero(device)
    if "nearest_vertex_bin" in aux_target:
        terms["vertex_aux"] = F.cross_entropy(aux["nearest_vertex_logits"], aux_target["nearest_vertex_bin"].long())
    else:
        terms["vertex_aux"] = _optional_zero(device)
    if "skinning_weights" in aux_target:
        terms["skinning_aux"] = F.smooth_l1_loss(aux["skinning_weights"], aux_target["skinning_weights"])
    else:
        terms["skinning_aux"] = _optional_zero(device)
    if "is_true_semantic" in aux_target:
        terms["contrastive_aux"] = F.binary_cross_entropy_with_logits(
            aux["true_vs_shuffled_logit"].squeeze(-1),
            aux_target["is_true_semantic"].float(),
        )
    else:
        terms["contrastive_aux"] = _optional_zero(device)

    total = (
        weights.canonical_aux * terms["canonical_aux"]
        + weights.body_part_aux * terms["body_part_aux"]
        + weights.vertex_aux * terms["vertex_aux"]
        + weights.skinning_aux * terms["skinning_aux"]
        + weights.contrastive_aux * terms["contrastive_aux"]
    )
    metrics = {f"loss_{k}": float(v.detach().cpu()) for k, v in terms.items()}
    metrics["loss_aux_total"] = float(total.detach().cpu())
    return total, metrics


def causal_margin_loss(
    true_score: torch.Tensor,
    counterfactual_score: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    """Require true semantic to outperform a same-support counterfactual."""

    return F.relu(margin - (true_score - counterfactual_score)).mean()


def support_direct_residual_penalty(model_out: dict[str, Any]) -> torch.Tensor:
    """Penalize geometry magnitude when semantic gate is near zero."""

    gate = model_out["fusion"]["semantic_gate"].mean(dim=-1, keepdim=True)
    delta = model_out["outputs"]["delta_point"]
    return ((1.0 - gate.detach()) * delta.square().mean(dim=-1, keepdim=True)).mean()


def combined_semantic_bottleneck_loss(
    model_out: dict[str, Any],
    target: dict[str, torch.Tensor],
    aux_target: dict[str, torch.Tensor],
    *,
    weights: SemanticBottleneckLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or SemanticBottleneckLossWeights()
    geom, geom_metrics = geometry_losses(model_out, target, weights)
    aux, aux_metrics = semantic_auxiliary_losses(model_out, aux_target, weights)
    support_penalty = support_direct_residual_penalty(model_out)
    total = geom + aux + weights.support_residual_penalty * support_penalty
    metrics = {**geom_metrics, **aux_metrics}
    metrics["loss_support_direct_residual_penalty"] = float(support_penalty.detach().cpu())
    metrics["loss_total"] = float(total.detach().cpu())
    return total, metrics
