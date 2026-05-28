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
class AntiBillboardTopologyVolumeConfig:
    anchor_feature_dim: int = 32
    smpl_feature_dim: int = 64
    token_dim: int = 64
    hidden_dim: int = 256
    cross_section_bins: int = 8
    part_count: int = 8
    source_label_count: int = 10
    max_residual: float = 0.045
    max_shell_offset: float = 0.050
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


class AntiBillboardTopologyVolumeStudent(nn.Module):
    """Model-owned anti-billboard topology-volume student.

    This module is intentionally representation-first: it keeps VGGT baseline
    anchors, applies residuals only through weak/billboard gates, predicts
    front/back/side shell separation, and exposes cross-section occupancy plus
    part-continuity logits. It rejects teacher/Kinect inputs at inference.
    """

    def __init__(self, cfg: AntiBillboardTopologyVolumeConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or AntiBillboardTopologyVolumeConfig()
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
        self.residual_local_head = nn.Linear(self.cfg.hidden_dim, 3)
        self.shell_offset_head = nn.Linear(self.cfg.hidden_dim, 4)
        self.cross_section_head = nn.Linear(self.cfg.hidden_dim, self.cfg.cross_section_bins)
        self.occupancy_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.visibility_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.part_continuity_head = nn.Linear(self.cfg.hidden_dim, self.cfg.part_count)
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
        local_residual = torch.tanh(self.residual_local_head(h)) * self.cfg.max_residual
        residual = (
            tangent * local_residual[..., 0:1]
            + binormal * local_residual[..., 1:2]
            + normal * local_residual[..., 2:3]
        ) * gate
        shell_offsets = torch.sigmoid(self.shell_offset_head(h)) * self.cfg.max_shell_offset
        front_shell = anchor_xyz + normal * shell_offsets[..., 0:1]
        back_shell = anchor_xyz - normal * shell_offsets[..., 1:2]
        left_shell = anchor_xyz + tangent * shell_offsets[..., 2:3]
        right_shell = anchor_xyz - tangent * shell_offsets[..., 3:4]
        side_shell = 0.5 * (left_shell + right_shell)
        student_xyz = anchor_xyz + residual
        cross_section_logits = self.cross_section_head(h)
        cross_section_occupancy = torch.sigmoid(cross_section_logits)
        occupancy = torch.sigmoid(self.occupancy_head(h))
        visibility = torch.sigmoid(self.visibility_head(h))
        rgb_delta = torch.tanh(self.rgb_delta_head(h)) * self.cfg.max_rgb_delta * gate
        student_rgb = (anchor_rgb + rgb_delta).clamp(0.0, 1.0)
        part_continuity_logits = self.part_continuity_head(h)
        source_logits = self.source_head(h)
        return {
            "student_xyz": student_xyz,
            "student_points": student_xyz,
            "student_rgb": student_rgb,
            "rgb": student_rgb,
            "residual_xyz": residual,
            "local_residual": local_residual,
            "normal": normal,
            "tangent": tangent,
            "binormal": binormal,
            "shell_offsets": shell_offsets,
            "front_shell": front_shell,
            "back_shell": back_shell,
            "left_shell": left_shell,
            "right_shell": right_shell,
            "side_shell": side_shell,
            "cross_section_logits": cross_section_logits,
            "cross_section_occupancy": cross_section_occupancy,
            "occupancy": occupancy,
            "visibility": visibility,
            "part_continuity_logits": part_continuity_logits,
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
    points: int = 256,
    cfg: AntiBillboardTopologyVolumeConfig | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    cfg = cfg or AntiBillboardTopologyVolumeConfig()
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(135)
    body = torch.randint(0, cfg.part_count, (batch_size, points), generator=gen, device=device)
    smpl = torch.zeros(batch_size, points, cfg.smpl_feature_dim, device=device)
    smpl[..., : cfg.part_count] = F.one_hot(body, num_classes=cfg.part_count).float()
    smpl[..., cfg.part_count : cfg.part_count + 3] = torch.randn(batch_size, points, 3, generator=gen, device=device) * 0.1
    return {
        "anchor_xyz": torch.randn(batch_size, points, 3, generator=gen, device=device) * 0.2,
        "anchor_rgb": torch.rand(batch_size, points, 3, generator=gen, device=device),
        "confidence": torch.rand(batch_size, points, generator=gen, device=device),
        "weak_region": torch.rand(batch_size, points, generator=gen, device=device),
        "billboard_region": torch.rand(batch_size, points, generator=gen, device=device),
        "anchor_features": torch.randn(batch_size, points, cfg.anchor_feature_dim, generator=gen, device=device) * 0.1,
        "smpl_features": smpl,
        "vggt_token_context": torch.randn(batch_size, cfg.token_dim, generator=gen, device=device) * 0.1,
        "body_part_id": body,
    }


def smoke_test() -> dict[str, Any]:
    cfg = AntiBillboardTopologyVolumeConfig()
    model = AntiBillboardTopologyVolumeStudent(cfg)
    batch = make_smoke_batch(cfg=cfg)
    out = model(batch)
    loss = (
        out["student_points"].square().mean()
        + out["cross_section_occupancy"].mean()
        + out["shell_offsets"].mean()
        + out["part_continuity_logits"].square().mean()
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
        "front_shell_shape": list(out["front_shell"].shape),
        "back_shell_shape": list(out["back_shell"].shape),
        "left_shell_shape": list(out["left_shell"].shape),
        "right_shell_shape": list(out["right_shell"].shape),
        "cross_section_occupancy_shape": list(out["cross_section_occupancy"].shape),
        "part_continuity_logits_shape": list(out["part_continuity_logits"].shape),
        "grad_norm_positive": grad_norm > 0,
        "forbidden_key_rejection": forbidden_rejected,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(smoke_test(), indent=2))
