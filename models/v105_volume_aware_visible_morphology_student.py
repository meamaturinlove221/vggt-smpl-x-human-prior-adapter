from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


FORBIDDEN_INFERENCE_KEYS = {
    "raw_kinect_depth",
    "kinect_depth",
    "teacher_points",
    "teacher_xyz",
    "v591_points",
    "v591_teacher",
    "dense_teacher",
}


@dataclass(frozen=True)
class VolumeAwareVisibleMorphologyConfig:
    anchor_feature_dim: int = 32
    smpl_feature_dim: int = 48
    token_dim: int = 64
    hidden_dim: int = 224
    source_label_count: int = 8
    max_residual: float = 0.040
    max_rgb_delta: float = 0.050
    max_shell_offset: float = 0.035


class VolumeAwareVisibleMorphologyStudent(nn.Module):
    """Volume-aware VGGT/SMPL-X residual student.

    The model keeps VGGT high-confidence baseline anchors as the primary layer,
    uses SMPL-X features as topology/part conditioning, and predicts weak-region
    gated residuals plus front/back/side shell offsets. It rejects dense teacher
    inputs at inference by construction.
    """

    def __init__(self, cfg: VolumeAwareVisibleMorphologyConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or VolumeAwareVisibleMorphologyConfig()
        in_dim = (
            3
            + 3
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
        self.residual_head = nn.Linear(self.cfg.hidden_dim, 3)
        self.thickness_head = nn.Linear(self.cfg.hidden_dim, 3)
        self.occupancy_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.visibility_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.normal_head = nn.Linear(self.cfg.hidden_dim, 3)
        self.rgb_delta_head = nn.Linear(self.cfg.hidden_dim, 3)
        self.source_head = nn.Linear(self.cfg.hidden_dim, self.cfg.source_label_count)

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
        bsz, points, _ = anchor_xyz.shape
        if confidence.ndim == 2:
            confidence = confidence.unsqueeze(-1)
        if weak_region.ndim == 2:
            weak_region = weak_region.unsqueeze(-1)
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
                anchor_features,
                smpl_features,
                token_context,
            ],
            dim=-1,
        )
        h = self.trunk(x)
        residual = torch.tanh(self.residual_head(h)) * self.cfg.max_residual
        residual = residual * weak_region.clamp(0.0, 1.0)
        thickness_field = torch.sigmoid(self.thickness_head(h)) * self.cfg.max_shell_offset
        normal = F.normalize(self.normal_head(h), dim=-1, eps=1e-6)
        front_shell = anchor_xyz + normal * thickness_field[..., 0:1]
        back_shell = anchor_xyz - normal * thickness_field[..., 1:2]
        side_shell = anchor_xyz + torch.roll(normal, shifts=1, dims=-1) * thickness_field[..., 2:3]
        student_xyz = anchor_xyz + residual
        rgb_delta = torch.tanh(self.rgb_delta_head(h)) * self.cfg.max_rgb_delta
        rgb_delta = rgb_delta * weak_region.clamp(0.0, 1.0)
        student_rgb = (anchor_rgb + rgb_delta).clamp(0.0, 1.0)
        occupancy = torch.sigmoid(self.occupancy_head(h))
        visibility = torch.sigmoid(self.visibility_head(h))
        source_logits = self.source_head(h)
        return {
            "student_xyz": student_xyz,
            "student_points": student_xyz,
            "student_rgb": student_rgb,
            "rgb": student_rgb,
            "residual_xyz": residual,
            "thickness_field": thickness_field,
            "front_shell": front_shell,
            "back_shell": back_shell,
            "side_shell": side_shell,
            "normal": normal,
            "occupancy": occupancy,
            "visibility": visibility,
            "rgb_delta": rgb_delta,
            "source_logits": source_logits,
            "source_label": source_logits.argmax(dim=-1),
            "model_owned_student_output": torch.tensor(True, device=anchor_xyz.device),
            "no_teacher_points_inference": torch.tensor(True, device=anchor_xyz.device),
            "no_raw_kinect_depth_inference": torch.tensor(True, device=anchor_xyz.device),
        }


def make_smoke_batch(
    *,
    batch_size: int = 1,
    points: int = 128,
    cfg: VolumeAwareVisibleMorphologyConfig | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    cfg = cfg or VolumeAwareVisibleMorphologyConfig()
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(105)
    return {
        "anchor_xyz": torch.randn(batch_size, points, 3, generator=gen, device=device) * 0.2,
        "anchor_rgb": torch.rand(batch_size, points, 3, generator=gen, device=device),
        "confidence": torch.rand(batch_size, points, generator=gen, device=device),
        "weak_region": torch.rand(batch_size, points, generator=gen, device=device),
        "anchor_features": torch.randn(batch_size, points, cfg.anchor_feature_dim, generator=gen, device=device) * 0.1,
        "smpl_features": torch.randn(batch_size, points, cfg.smpl_feature_dim, generator=gen, device=device) * 0.1,
        "vggt_token_context": torch.randn(batch_size, cfg.token_dim, generator=gen, device=device) * 0.1,
    }


def smoke_test() -> dict[str, Any]:
    cfg = VolumeAwareVisibleMorphologyConfig()
    model = VolumeAwareVisibleMorphologyStudent(cfg)
    batch = make_smoke_batch(cfg=cfg)
    out = model(batch)
    loss = out["student_points"].square().mean() + out["thickness_field"].mean() + out["occupancy"].mean()
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
        "thickness_field_shape": list(out["thickness_field"].shape),
        "front_shell_shape": list(out["front_shell"].shape),
        "back_shell_shape": list(out["back_shell"].shape),
        "side_shell_shape": list(out["side_shell"].shape),
        "occupancy_shape": list(out["occupancy"].shape),
        "visibility_shape": list(out["visibility"].shape),
        "grad_norm_positive": grad_norm > 0,
        "forbidden_key_rejection": forbidden_rejected,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(smoke_test(), indent=2))
