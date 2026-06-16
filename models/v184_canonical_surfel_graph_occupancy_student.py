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


@dataclass(frozen=True)
class CanonicalSurfelGraphOccupancyConfig:
    surfel_feature_dim: int = 64
    vggt_feature_dim: int = 64
    token_dim: int = 64
    hidden_dim: int = 256
    part_count: int = 8
    max_residual: float = 0.040
    max_thickness: float = 0.055
    max_rgb_delta: float = 0.025


def _normalize(v: Tensor) -> Tensor:
    return F.normalize(v, dim=-1, eps=1e-6)


class CanonicalSurfelGraphOccupancyStudent(nn.Module):
    """Canonical SMPL-X surfel/graph occupancy student.

    This is the next architecture after point-anchor shell decoders failed.
    It decodes on canonical surfel support and exposes occupancy, visibility,
    residual, thickness, normal, RGB, part graph, and pair-exclusion heads.
    """

    def __init__(self, cfg: CanonicalSurfelGraphOccupancyConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CanonicalSurfelGraphOccupancyConfig()
        in_dim = (
            3
            + 3
            + 3
            + 3
            + 3
            + 1
            + self.cfg.surfel_feature_dim
            + self.cfg.vggt_feature_dim
            + self.cfg.token_dim
            + self.cfg.part_count
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
        self.occupancy_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.visibility_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.residual_local_head = nn.Linear(self.cfg.hidden_dim, 3)
        self.thickness_head = nn.Linear(self.cfg.hidden_dim, 2)
        self.normal_delta_head = nn.Linear(self.cfg.hidden_dim, 3)
        self.rgb_delta_head = nn.Linear(self.cfg.hidden_dim, 3)
        self.part_graph_head = nn.Linear(self.cfg.hidden_dim, self.cfg.part_count)
        self.exclusion_head = nn.Linear(self.cfg.hidden_dim, self.cfg.part_count)

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
        surfel_xyz = self._first(batch, "surfel_xyz", "canonical_surfel_xyz").float()
        surfel_rgb = self._first(batch, "surfel_rgb", "anchor_rgb").float().clamp(0.0, 1.0)
        normal = _normalize(self._first(batch, "surfel_normal", "normal").float())
        tangent = _normalize(self._first(batch, "surfel_tangent", "tangent").float())
        binormal = _normalize(torch.cross(normal, tangent, dim=-1))
        confidence = self._first(batch, "vggt_confidence", "confidence").float()
        if confidence.ndim == 2:
            confidence = confidence.unsqueeze(-1)
        bsz, points, _ = surfel_xyz.shape
        surfel_features = self._optional(batch, "surfel_features", (bsz, points, self.cfg.surfel_feature_dim), surfel_xyz.device).float()
        vggt_features = self._optional(batch, "vggt_features", (bsz, points, self.cfg.vggt_feature_dim), surfel_xyz.device).float()
        token_context = self._first(batch, "vggt_token_context", "token_context").float()
        if token_context.ndim == 2:
            token_context = token_context.unsqueeze(1).expand(-1, points, -1)
        elif token_context.shape[1] != points:
            token_context = F.interpolate(token_context.transpose(1, 2), size=points, mode="linear", align_corners=False).transpose(1, 2)
        if "body_part_id" in batch:
            body = batch["body_part_id"].long().clamp(0, self.cfg.part_count - 1)
            part_one_hot = F.one_hot(body, num_classes=self.cfg.part_count).float()
        else:
            part_one_hot = self._optional(batch, "body_part_one_hot", (bsz, points, self.cfg.part_count), surfel_xyz.device).float()
        x = torch.cat(
            [
                surfel_xyz,
                surfel_rgb,
                normal,
                tangent,
                binormal,
                confidence,
                surfel_features,
                vggt_features,
                token_context,
                part_one_hot,
            ],
            dim=-1,
        )
        h = self.trunk(x)
        occupancy_logits = self.occupancy_head(h)
        occupancy = torch.sigmoid(occupancy_logits)
        visibility_logits = self.visibility_head(h)
        visibility = torch.sigmoid(visibility_logits)
        local = torch.tanh(self.residual_local_head(h)) * self.cfg.max_residual
        residual = tangent * local[..., 0:1] + binormal * local[..., 1:2] + normal * local[..., 2:3]
        thickness = torch.sigmoid(self.thickness_head(h)) * self.cfg.max_thickness
        normal_delta = torch.tanh(self.normal_delta_head(h)) * 0.22
        refined_normal = _normalize(normal + normal_delta)
        front_shell = surfel_xyz + residual + refined_normal * thickness[..., 0:1]
        back_shell = surfel_xyz + residual - refined_normal * thickness[..., 1:2]
        student_xyz = 0.5 * (front_shell + back_shell)
        rgb_delta = torch.tanh(self.rgb_delta_head(h)) * self.cfg.max_rgb_delta * visibility
        student_rgb = (surfel_rgb + rgb_delta).clamp(0.0, 1.0)
        return {
            "student_xyz": student_xyz,
            "student_points": student_xyz,
            "student_rgb": student_rgb,
            "rgb": student_rgb,
            "occupancy_logits": occupancy_logits,
            "occupancy": occupancy,
            "visibility_logits": visibility_logits,
            "visibility": visibility,
            "residual_xyz": residual,
            "front_shell": front_shell,
            "back_shell": back_shell,
            "thickness": thickness,
            "normal": refined_normal,
            "part_graph_logits": self.part_graph_head(h),
            "part_exclusion_logits": self.exclusion_head(h),
            "model_owned_student_output": torch.tensor(True, device=surfel_xyz.device),
            "no_teacher_points_inference": torch.tensor(True, device=surfel_xyz.device),
            "no_raw_kinect_depth_inference": torch.tensor(True, device=surfel_xyz.device),
        }


def smoke_test() -> dict[str, Any]:
    cfg = CanonicalSurfelGraphOccupancyConfig()
    model = CanonicalSurfelGraphOccupancyStudent(cfg)
    gen = torch.Generator()
    gen.manual_seed(184)
    body = torch.randint(0, cfg.part_count, (1, 256), generator=gen)
    batch = {
        "surfel_xyz": torch.randn(1, 256, 3, generator=gen) * 0.15,
        "surfel_rgb": torch.rand(1, 256, 3, generator=gen),
        "surfel_normal": F.normalize(torch.randn(1, 256, 3, generator=gen), dim=-1),
        "surfel_tangent": F.normalize(torch.randn(1, 256, 3, generator=gen), dim=-1),
        "vggt_confidence": torch.rand(1, 256, generator=gen),
        "surfel_features": torch.randn(1, 256, cfg.surfel_feature_dim, generator=gen) * 0.1,
        "vggt_features": torch.randn(1, 256, cfg.vggt_feature_dim, generator=gen) * 0.1,
        "vggt_token_context": torch.randn(1, cfg.token_dim, generator=gen) * 0.1,
        "body_part_id": body,
    }
    out = model(batch)
    loss = (
        out["student_xyz"].square().mean()
        + out["occupancy"].mean()
        + out["visibility"].mean()
        + out["part_graph_logits"].square().mean()
    )
    loss.backward()
    grad_norm = sum(float(p.grad.detach().abs().sum()) for p in model.parameters() if p.grad is not None)
    bad = dict(batch)
    bad["teacher_points"] = torch.zeros_like(batch["surfel_xyz"])
    forbidden_rejected = False
    try:
        model(bad)
    except ValueError:
        forbidden_rejected = True
    return {
        "student_xyz_shape": list(out["student_xyz"].shape),
        "occupancy_shape": list(out["occupancy"].shape),
        "front_shell_shape": list(out["front_shell"].shape),
        "back_shell_shape": list(out["back_shell"].shape),
        "part_graph_logits_shape": list(out["part_graph_logits"].shape),
        "part_exclusion_logits_shape": list(out["part_exclusion_logits"].shape),
        "grad_norm_positive": grad_norm > 0,
        "forbidden_teacher_points_rejected": forbidden_rejected,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(smoke_test(), indent=2))
