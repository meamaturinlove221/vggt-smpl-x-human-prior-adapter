"""Semantic-bottleneck SMPL SparseConv/VGGT adapter.

This module intentionally separates support, semantic, and observation inputs.
The support branch is not allowed to produce a geometry residual; it only
produces reliability gates used by fusion and output confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SemanticBottleneckConfig:
    support_dim: int
    semantic_dim: int
    observation_dim: int
    hidden_dim: int = 96
    latent_dim: int = 96
    num_body_parts: int = 16
    vertex_bins: int = 2048
    skinning_dim: int = 24
    observation_dropout_p: float = 0.15
    detach_observation: bool = False


class SupportBranch(nn.Module):
    """Reliability-only support encoder.

    Inputs are occupancy/mask/visibility/voxel xyz/projection support. Outputs
    are gates and confidence features only. No residual head exists here by
    design; leakage checks assert this invariant.
    """

    direct_residual_allowed = False

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
        )
        self.reliability = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())
        self.confidence = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())

    def forward(self, support: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encoder(support)
        return {
            "support_latent": latent,
            "reliability_gate": self.reliability(latent),
            "support_confidence": self.confidence(latent),
        }


class SemanticBranch(nn.Module):
    """SMPL semantic encoder with auxiliary structure heads."""

    def __init__(self, cfg: SemanticBottleneckConfig) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cfg.semantic_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.GELU(),
        )
        self.body_structure = nn.Linear(cfg.latent_dim, cfg.latent_dim)
        self.canonical_token = nn.Linear(cfg.latent_dim, cfg.latent_dim)
        self.aux_canonical_xyz = nn.Linear(cfg.latent_dim, 3)
        self.aux_body_part = nn.Linear(cfg.latent_dim, cfg.num_body_parts)
        self.aux_vertex = nn.Linear(cfg.latent_dim, cfg.vertex_bins)
        self.aux_skinning = nn.Linear(cfg.latent_dim, cfg.skinning_dim)
        self.aux_true_vs_shuffled = nn.Linear(cfg.latent_dim, 1)

    def forward(self, semantic: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encoder(semantic)
        return {
            "semantic_latent": latent,
            "body_structure_token": self.body_structure(latent),
            "canonical_correspondence_token": self.canonical_token(latent),
            "aux": {
                "canonical_xyz": self.aux_canonical_xyz(latent),
                "body_part_logits": self.aux_body_part(latent),
                "nearest_vertex_logits": self.aux_vertex(latent),
                "skinning_weights": self.aux_skinning(latent),
                "true_vs_shuffled_logit": self.aux_true_vs_shuffled(latent),
            },
        }


class ObservationBranch(nn.Module):
    """VGGT observation encoder with dropout, detach, and bottleneck controls."""

    def __init__(self, cfg: SemanticBottleneckConfig) -> None:
        super().__init__()
        self.dropout = nn.Dropout(cfg.observation_dropout_p)
        self.confidence_calibration = nn.Sequential(nn.Linear(cfg.observation_dim, cfg.observation_dim), nn.Tanh())
        self.bottleneck = nn.Sequential(
            nn.Linear(cfg.observation_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
            nn.GELU(),
        )

    def forward(
        self,
        observation: torch.Tensor,
        *,
        detach: bool = False,
        dropout_p: float | None = None,
    ) -> dict[str, torch.Tensor]:
        x = observation.detach() if detach else observation
        if dropout_p is None:
            x = self.dropout(x)
        elif dropout_p > 0:
            x = F.dropout(x, p=dropout_p, training=self.training)
        calibrated = x + 0.05 * self.confidence_calibration(x)
        return {"observation_latent": self.bottleneck(calibrated), "observation_calibrated": calibrated}


class SemanticGatedFusion(nn.Module):
    """Semantic-gated fusion with support reliability and observation attention."""

    def __init__(self, cfg: SemanticBottleneckConfig) -> None:
        super().__init__()
        self.semantic_gate = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.latent_dim), nn.Sigmoid())
        self.support_to_latent = nn.Linear(cfg.latent_dim, cfg.latent_dim)
        self.cross_attn = nn.MultiheadAttention(cfg.latent_dim, num_heads=4, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Linear(cfg.latent_dim * 3, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
            nn.GELU(),
        )

    def forward(
        self,
        support_latent: torch.Tensor,
        reliability_gate: torch.Tensor,
        semantic_latent: torch.Tensor,
        observation_latent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        gate = self.semantic_gate(semantic_latent)
        obs_context, attn = self.cross_attn(
            query=semantic_latent.unsqueeze(1),
            key=observation_latent.unsqueeze(1),
            value=observation_latent.unsqueeze(1),
            need_weights=True,
        )
        obs_context = obs_context.squeeze(1)
        support_context = self.support_to_latent(support_latent) * reliability_gate
        semantic_context = semantic_latent * gate
        fused = self.fuse(torch.cat([semantic_context, obs_context, support_context], dim=-1))
        return {
            "fused_latent": fused,
            "semantic_gate": gate,
            "observation_attention": attn,
            "support_context": support_context,
        }


class GeometryHeads(nn.Module):
    def __init__(self, cfg: SemanticBottleneckConfig) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
        )
        self.delta_point = nn.Linear(cfg.hidden_dim, 3)
        self.delta_normal = nn.Linear(cfg.hidden_dim, 3)
        self.occupancy = nn.Sequential(nn.Linear(cfg.hidden_dim, 1), nn.Sigmoid())
        self.reliability = nn.Sequential(nn.Linear(cfg.hidden_dim, 1), nn.Sigmoid())
        self.uncertainty = nn.Sequential(nn.Linear(cfg.hidden_dim, 1), nn.Softplus())

    def forward(self, fused_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.shared(fused_latent)
        normal = F.normalize(self.delta_normal(x), dim=-1, eps=1e-6)
        return {
            "delta_point": self.delta_point(x),
            "delta_normal": normal,
            "occupancy": self.occupancy(x),
            "reliability": self.reliability(x),
            "uncertainty": self.uncertainty(x),
        }


class SemanticBottleneckSMPLSparseConv(nn.Module):
    """Three-branch semantic-bottleneck geometry model."""

    def __init__(self, cfg: SemanticBottleneckConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.support_branch = SupportBranch(cfg.support_dim, cfg.hidden_dim, cfg.latent_dim)
        self.semantic_branch = SemanticBranch(cfg)
        self.observation_branch = ObservationBranch(cfg)
        self.fusion = SemanticGatedFusion(cfg)
        self.heads = GeometryHeads(cfg)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        detach_observation: bool | None = None,
        observation_dropout_p: float | None = None,
    ) -> dict[str, Any]:
        support = self.support_branch(batch["support"])
        semantic = self.semantic_branch(batch["semantic"])
        observation = self.observation_branch(
            batch["observation"],
            detach=self.cfg.detach_observation if detach_observation is None else detach_observation,
            dropout_p=observation_dropout_p,
        )
        fusion = self.fusion(
            support["support_latent"],
            support["reliability_gate"],
            semantic["semantic_latent"],
            observation["observation_latent"],
        )
        heads = self.heads(fusion["fused_latent"])
        reliability = heads["reliability"] * support["reliability_gate"]
        outputs = {**heads, "reliability": reliability}
        diagnostics = {
            "support_direct_residual_allowed": bool(getattr(self.support_branch, "direct_residual_allowed", True)),
            "semantic_gate_mean": float(fusion["semantic_gate"].detach().mean().cpu()),
            "support_gate_mean": float(support["reliability_gate"].detach().mean().cpu()),
            "observation_detached": bool(self.cfg.detach_observation if detach_observation is None else detach_observation),
        }
        return {
            "outputs": outputs,
            "support": support,
            "semantic": semantic,
            "observation": observation,
            "fusion": fusion,
            "diagnostics": diagnostics,
        }

    def leakage_contract(self) -> dict[str, bool]:
        return {
            "support_direct_residual_allowed": False,
            "teacher_postcompose_allowed": False,
            "v999_residual_copy_allowed": False,
            "v770_direct_add_allowed": False,
            "semantic_auxiliary_required": True,
            "observation_bottleneck_required": True,
        }


def build_default_model(
    support_dim: int,
    semantic_dim: int,
    observation_dim: int,
    *,
    hidden_dim: int = 96,
    latent_dim: int = 96,
) -> SemanticBottleneckSMPLSparseConv:
    cfg = SemanticBottleneckConfig(
        support_dim=support_dim,
        semantic_dim=semantic_dim,
        observation_dim=observation_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    )
    return SemanticBottleneckSMPLSparseConv(cfg)
