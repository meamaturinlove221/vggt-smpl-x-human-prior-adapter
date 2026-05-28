from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.v135_anti_billboard_topology_volume_student import FORBIDDEN_INFERENCE_KEYS


PART_PAIR_EXCLUSION_PAIRS = [
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (1, 6),
    (1, 7),
    (2, 4),
    (2, 5),
    (2, 6),
    (2, 7),
    (3, 4),
    (3, 5),
    (3, 6),
    (3, 7),
    (4, 7),
    (5, 6),
]

VALID_CONTACT_PAIRS = [
    (0, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (4, 6),
    (5, 7),
]


@dataclass(frozen=True)
class PartPairExclusionConfig:
    anchor_feature_dim: int = 32
    smpl_feature_dim: int = 64
    token_dim: int = 64
    hidden_dim: int = 256
    part_count: int = 8
    cross_section_bins: int = 8
    max_residual: float = 0.045
    max_shell_offset: float = 0.050
    max_part_center_offset: float = 0.075
    max_rgb_delta: float = 0.025


def _normalize(v: Tensor) -> Tensor:
    return F.normalize(v, dim=-1, eps=1e-6)


def _orthonormal_frame(raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    normal = _normalize(raw[..., 0:3])
    tangent_raw = raw[..., 3:6]
    tangent = tangent_raw - (tangent_raw * normal).sum(dim=-1, keepdim=True) * normal
    tangent = _normalize(tangent)
    binormal = _normalize(torch.cross(normal, tangent, dim=-1))
    return normal, tangent, binormal


class PartPairExclusionTopologyStudent(nn.Module):
    """Topology-volume student with explicit semantic part-pair heads.

    Unlike the previous global multishell decoder, this model exposes per-part
    occupancy gates and pair-exclusion/contact logits before point decoding.
    Teacher/Kinect inputs are rejected at inference.
    """

    def __init__(self, cfg: PartPairExclusionConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PartPairExclusionConfig()
        in_dim = (
            3
            + 3
            + 1
            + 1
            + 1
            + self.cfg.anchor_feature_dim
            + self.cfg.smpl_feature_dim
            + self.cfg.token_dim
        )
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, self.cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.GELU(),
        )
        self.frame_head = nn.Linear(self.cfg.hidden_dim, 9)
        self.residual_head = nn.Linear(self.cfg.hidden_dim, 3)
        self.shell_head = nn.Linear(self.cfg.hidden_dim, 4)
        self.cross_section_head = nn.Linear(self.cfg.hidden_dim, self.cfg.cross_section_bins)
        self.part_gate_head = nn.Linear(self.cfg.hidden_dim, self.cfg.part_count)
        self.part_center_head = nn.Linear(self.cfg.hidden_dim, self.cfg.part_count * 3)
        self.pair_exclusion_head = nn.Linear(self.cfg.hidden_dim, len(PART_PAIR_EXCLUSION_PAIRS))
        self.valid_contact_head = nn.Linear(self.cfg.hidden_dim, len(VALID_CONTACT_PAIRS))
        self.occupancy_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.visibility_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.rgb_delta_head = nn.Linear(self.cfg.hidden_dim, 3)

    @staticmethod
    def reject_forbidden(batch: Mapping[str, Any]) -> None:
        present = sorted(FORBIDDEN_INFERENCE_KEYS.intersection(batch.keys()))
        if present:
            raise ValueError(f"Forbidden inference keys present: {present}")

    @staticmethod
    def _first(batch: Mapping[str, Tensor], *keys: str) -> Tensor:
        for key in keys:
            if key in batch:
                return batch[key]
        raise KeyError(f"missing any of required keys: {keys}")

    @staticmethod
    def _optional(batch: Mapping[str, Tensor], key: str, shape: tuple[int, ...], device: torch.device) -> Tensor:
        if key in batch:
            return batch[key]
        return torch.zeros(*shape, device=device)

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        self.reject_forbidden(batch)
        anchor_xyz = self._first(batch, "anchor_xyz", "surfel_xyz").float()
        anchor_rgb = self._first(batch, "anchor_rgb", "surfel_rgb").float().clamp(0.0, 1.0)
        confidence = self._first(batch, "confidence", "surfel_confidence").float()
        weak_region = self._first(batch, "weak_region", "weak").float()
        billboard_region = self._first(batch, "billboard_region", "flat_region", "weak_region").float()
        bsz, points, _ = anchor_xyz.shape
        if confidence.ndim == 2:
            confidence = confidence.unsqueeze(-1)
        if weak_region.ndim == 2:
            weak_region = weak_region.unsqueeze(-1)
        if billboard_region.ndim == 2:
            billboard_region = billboard_region.unsqueeze(-1)
        anchor_features = self._optional(batch, "anchor_features", (bsz, points, self.cfg.anchor_feature_dim), anchor_xyz.device).float()
        smpl_features = self._optional(batch, "smpl_features", (bsz, points, self.cfg.smpl_feature_dim), anchor_xyz.device).float()
        token_context = self._first(batch, "vggt_token_context", "token_context").float()
        if token_context.ndim == 2:
            token_context = token_context.unsqueeze(1).expand(-1, points, -1)
        elif token_context.shape[1] != points:
            token_context = F.interpolate(
                token_context.transpose(1, 2),
                size=points,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        x = torch.cat(
            [
                anchor_xyz,
                anchor_rgb,
                confidence,
                weak_region,
                billboard_region,
                anchor_features,
                smpl_features,
                token_context,
            ],
            dim=-1,
        )
        h = self.trunk(x)
        normal, tangent, binormal = _orthonormal_frame(self.frame_head(h))
        gate = torch.maximum(weak_region, billboard_region).clamp(0.0, 1.0)
        local = torch.tanh(self.residual_head(h)) * self.cfg.max_residual
        residual = (tangent * local[..., 0:1] + binormal * local[..., 1:2] + normal * local[..., 2:3]) * gate
        shell_offsets = torch.sigmoid(self.shell_head(h)) * self.cfg.max_shell_offset
        part_gate_logits = self.part_gate_head(h)
        part_gate = torch.sigmoid(part_gate_logits)
        part_centers_local = torch.tanh(self.part_center_head(h)).view(bsz, points, self.cfg.part_count, 3) * self.cfg.max_part_center_offset
        part_center_offsets = (
            tangent.unsqueeze(-2) * part_centers_local[..., 0:1]
            + binormal.unsqueeze(-2) * part_centers_local[..., 1:2]
            + normal.unsqueeze(-2) * part_centers_local[..., 2:3]
        ) * gate.unsqueeze(-2)
        if "body_part_id" in batch:
            body = batch["body_part_id"].long().clamp(0, self.cfg.part_count - 1)
            selected_part_offset = part_center_offsets.gather(
                2,
                body[..., None, None].expand(-1, -1, 1, 3),
            ).squeeze(2)
        else:
            selected_part_offset = (part_center_offsets * part_gate.unsqueeze(-1)).sum(dim=2) / part_gate.sum(dim=2, keepdim=True).clamp_min(1e-6)
        graph_anchor = anchor_xyz + selected_part_offset
        student_xyz = graph_anchor + residual
        front_shell = graph_anchor + normal * shell_offsets[..., 0:1]
        back_shell = graph_anchor - normal * shell_offsets[..., 1:2]
        left_shell = graph_anchor + tangent * shell_offsets[..., 2:3]
        right_shell = graph_anchor - tangent * shell_offsets[..., 3:4]
        cross_section_logits = self.cross_section_head(h)
        pair_exclusion_logits = self.pair_exclusion_head(h)
        valid_contact_logits = self.valid_contact_head(h)
        rgb_delta = torch.tanh(self.rgb_delta_head(h)) * self.cfg.max_rgb_delta * gate
        student_rgb = (anchor_rgb + rgb_delta).clamp(0.0, 1.0)
        return {
            "student_xyz": student_xyz,
            "student_points": student_xyz,
            "student_rgb": student_rgb,
            "rgb": student_rgb,
            "residual_xyz": residual,
            "normal": normal,
            "tangent": tangent,
            "binormal": binormal,
            "shell_offsets": shell_offsets,
            "front_shell": front_shell,
            "back_shell": back_shell,
            "left_shell": left_shell,
            "right_shell": right_shell,
            "cross_section_logits": cross_section_logits,
            "cross_section_occupancy": torch.sigmoid(cross_section_logits),
            "part_gate_logits": part_gate_logits,
            "part_gate": part_gate,
            "part_center_offsets": part_center_offsets,
            "selected_part_offset": selected_part_offset,
            "pair_exclusion_logits": pair_exclusion_logits,
            "pair_exclusion": torch.sigmoid(pair_exclusion_logits),
            "valid_contact_logits": valid_contact_logits,
            "valid_contact": torch.sigmoid(valid_contact_logits),
            "part_continuity_logits": part_gate_logits,
            "occupancy": torch.sigmoid(self.occupancy_head(h)),
            "visibility": torch.sigmoid(self.visibility_head(h)),
            "rgb_delta": rgb_delta,
            "model_owned_student_output": torch.tensor(True, device=anchor_xyz.device),
            "no_teacher_points_inference": torch.tensor(True, device=anchor_xyz.device),
            "no_raw_kinect_depth_inference": torch.tensor(True, device=anchor_xyz.device),
        }


def smoke_test() -> dict[str, Any]:
    cfg = PartPairExclusionConfig()
    model = PartPairExclusionTopologyStudent(cfg)
    gen = torch.Generator()
    gen.manual_seed(182)
    body = torch.randint(0, cfg.part_count, (1, 128), generator=gen)
    smpl = torch.zeros(1, 128, cfg.smpl_feature_dim)
    smpl[..., : cfg.part_count] = F.one_hot(body, num_classes=cfg.part_count).float()
    batch = {
        "anchor_xyz": torch.randn(1, 128, 3, generator=gen) * 0.1,
        "anchor_rgb": torch.rand(1, 128, 3, generator=gen),
        "confidence": torch.rand(1, 128, generator=gen),
        "weak_region": torch.rand(1, 128, generator=gen),
        "billboard_region": torch.rand(1, 128, generator=gen),
        "anchor_features": torch.randn(1, 128, cfg.anchor_feature_dim, generator=gen) * 0.1,
        "smpl_features": smpl,
        "vggt_token_context": torch.randn(1, cfg.token_dim, generator=gen) * 0.1,
        "body_part_id": body,
    }
    out = model(batch)
    loss = (
        out["student_xyz"].square().mean()
        + out["part_gate"].mean()
        + out["pair_exclusion"].mean()
        + out["valid_contact"].mean()
    )
    loss.backward()
    grad_norm = sum(float(p.grad.detach().abs().sum()) for p in model.parameters() if p.grad is not None)
    bad = dict(batch)
    bad["teacher_points"] = torch.zeros_like(batch["anchor_xyz"])
    forbidden_rejected = False
    try:
        model(bad)
    except ValueError:
        forbidden_rejected = True
    return {
        "student_xyz_shape": list(out["student_xyz"].shape),
        "part_gate_shape": list(out["part_gate"].shape),
        "pair_exclusion_shape": list(out["pair_exclusion"].shape),
        "valid_contact_shape": list(out["valid_contact"].shape),
        "grad_norm_positive": grad_norm > 0,
        "forbidden_teacher_points_rejected": forbidden_rejected,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(smoke_test(), indent=2))
