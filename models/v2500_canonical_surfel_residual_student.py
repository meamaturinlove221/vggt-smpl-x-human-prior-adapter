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
class CanonicalSurfelResidualConfig:
    surfel_feature_dim: int = 32
    smpl_feature_dim: int = 32
    token_dim: int = 64
    hidden_dim: int = 192
    source_label_count: int = 6
    max_residual: float = 0.045
    max_rgb_delta: float = 0.12


class CanonicalSurfelResidualStudent(nn.Module):
    """Canonical SMPL-X surfel residual student.

    SMPL surfels are anchors and conditioning features. They are not directly
    emitted as the final visible body layer; the network predicts residual,
    occupancy, visibility, normals, and RGB deltas that downstream code applies
    to preserved VGGT baseline points / visible residual anchors.
    """

    def __init__(self, cfg: CanonicalSurfelResidualConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CanonicalSurfelResidualConfig()
        in_dim = self.cfg.surfel_feature_dim + self.cfg.smpl_feature_dim + self.cfg.token_dim + 3 + 3 + 1
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
        anchor_xyz = self._first(batch, "surfel_xyz", "anchor_xyz").float()
        anchor_rgb = self._first(batch, "surfel_rgb", "anchor_rgb").float().clamp(0.0, 1.0)
        confidence = self._first(batch, "surfel_confidence", "confidence").float()
        bsz, points, _ = anchor_xyz.shape
        surfel_features = self._optional(batch, "surfel_features", (bsz, points, self.cfg.surfel_feature_dim), anchor_xyz.device).float()
        smpl_features = self._optional(batch, "smpl_features", (bsz, points, self.cfg.smpl_feature_dim), anchor_xyz.device).float()
        token_context = self._first(batch, "vggt_token_context", "token_context").float()
        if confidence.ndim == 2:
            confidence = confidence.unsqueeze(-1)
        if token_context.ndim == 2:
            token_context = token_context.unsqueeze(1).expand(-1, anchor_xyz.shape[1], -1)
        elif token_context.shape[1] != anchor_xyz.shape[1]:
            token_context = F.interpolate(token_context.transpose(1, 2), size=anchor_xyz.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        x = torch.cat([anchor_xyz, anchor_rgb, confidence, surfel_features, smpl_features, token_context], dim=-1)
        h = self.trunk(x)
        residual = torch.tanh(self.residual_head(h)) * self.cfg.max_residual
        rgb_delta = torch.tanh(self.rgb_delta_head(h)) * self.cfg.max_rgb_delta
        normal = F.normalize(self.normal_head(h), dim=-1, eps=1e-6)
        source_logits = self.source_head(h)
        occupancy = torch.sigmoid(self.occupancy_head(h))
        visibility = torch.sigmoid(self.visibility_head(h))
        student_xyz = anchor_xyz + residual
        student_rgb = (anchor_rgb + rgb_delta).clamp(0.0, 1.0)
        return {
            "student_xyz": student_xyz,
            "student_points": student_xyz,
            "residual_xyz": residual,
            "occupancy": occupancy,
            "visibility": visibility,
            "normal": normal,
            "student_rgb": student_rgb,
            "rgb": student_rgb,
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
    cfg: CanonicalSurfelResidualConfig | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    cfg = cfg or CanonicalSurfelResidualConfig()
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(2500)
    return {
        "surfel_xyz": torch.randn(batch_size, points, 3, generator=gen, device=device) * 0.2,
        "surfel_rgb": torch.rand(batch_size, points, 3, generator=gen, device=device),
        "surfel_confidence": torch.rand(batch_size, points, generator=gen, device=device),
        "surfel_features": torch.randn(batch_size, points, cfg.surfel_feature_dim, generator=gen, device=device) * 0.1,
        "smpl_features": torch.randn(batch_size, points, cfg.smpl_feature_dim, generator=gen, device=device) * 0.1,
        "vggt_token_context": torch.randn(batch_size, cfg.token_dim, generator=gen, device=device) * 0.1,
    }


def smoke_test() -> dict[str, Any]:
    cfg = CanonicalSurfelResidualConfig()
    model = CanonicalSurfelResidualStudent(cfg)
    batch = make_smoke_batch(cfg=cfg)
    out = model(batch)
    loss = out["student_points"].square().mean() + out["occupancy"].mean() + out["visibility"].mean()
    loss.backward()
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += float(param.grad.detach().abs().sum())
    return {
        "student_xyz_shape": list(out["student_xyz"].shape),
        "residual_xyz_shape": list(out["residual_xyz"].shape),
        "occupancy_shape": list(out["occupancy"].shape),
        "visibility_shape": list(out["visibility"].shape),
        "normal_shape": list(out["normal"].shape),
        "rgb_delta_shape": list(out["rgb_delta"].shape),
        "source_logits_shape": list(out["source_logits"].shape),
        "grad_norm_positive": grad_norm > 0,
        "forbidden_key_rejection": _forbidden_key_rejection(model, batch),
    }


def _forbidden_key_rejection(model: CanonicalSurfelResidualStudent, batch: dict[str, Tensor]) -> bool:
    bad = dict(batch)
    bad["teacher_points"] = torch.zeros_like(batch["surfel_xyz"])
    try:
        model(bad)
    except ValueError:
        return True
    return False


if __name__ == "__main__":
    import json

    print(json.dumps(smoke_test(), indent=2))
