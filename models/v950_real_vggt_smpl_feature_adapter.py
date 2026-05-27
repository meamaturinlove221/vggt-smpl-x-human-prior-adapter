from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
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
    "synthetic_scene_tokens",
    "tiny_v330_scene_tokens",
}

SOURCE_LABELS = {
    0: "baseline_preserved",
    1: "smpl_feature_completed",
    2: "vggt_detail_grafted",
    3: "residual_refined",
    4: "environment",
    5: "auxiliary_control",
}


@dataclass(frozen=True)
class V950AdapterConfig:
    smpl_feature_channels: int = 21
    vggt_token_dim: int = 64
    hidden_dim: int = 160
    patch_size: int = 14
    source_label_count: int = 6
    max_point_offset: float = 0.055
    max_rgb_delta: float = 0.16
    num_heads: int = 4


class SMPLFeatureEncoder(nn.Module):
    """Encode scene/world SMPL-X feature maps into VGGT-compatible prior tokens."""

    def __init__(self, cfg: V950AdapterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patchifier = nn.Sequential(
            nn.Conv2d(cfg.smpl_feature_channels, cfg.hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, cfg.hidden_dim),
            nn.GELU(),
            nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, cfg.hidden_dim),
            nn.GELU(),
            nn.Conv2d(cfg.hidden_dim, cfg.vggt_token_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
        )
        self.point_proj = nn.Sequential(
            nn.Linear(3 + 3 + 1 + 8, cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.vggt_token_dim),
        )
        self.norm = nn.LayerNorm(cfg.vggt_token_dim)

    def forward(self, feature_images: Tensor, point_features: Tensor | None = None) -> dict[str, Tensor]:
        if feature_images.ndim != 5:
            raise ValueError(f"feature_images must be [B,S,C,H,W], got {tuple(feature_images.shape)}")
        bsz, views, channels, height, width = feature_images.shape
        if channels != self.cfg.smpl_feature_channels:
            raise ValueError(f"expected {self.cfg.smpl_feature_channels} feature channels, got {channels}")
        x = feature_images.reshape(bsz * views, channels, height, width).float()
        tokens = self.patchifier(x).flatten(2).transpose(1, 2)
        tokens = self.norm(tokens).view(bsz, views, tokens.shape[1], tokens.shape[2])
        point_context = None
        if point_features is not None:
            if point_features.ndim != 3:
                raise ValueError(f"point_features must be [B,N,C], got {tuple(point_features.shape)}")
            point_context = self.point_proj(point_features.float())
        return {"smpl_prior_tokens": tokens, "point_context": point_context}


class RealVGGTTokenBinder(nn.Module):
    """Bind SMPL feature tokens into real VGGT/Aggregator tokens."""

    def __init__(self, cfg: V950AdapterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_norm = nn.LayerNorm(cfg.vggt_token_dim)
        self.prior_norm = nn.LayerNorm(cfg.vggt_token_dim)
        self.cross_attn = nn.MultiheadAttention(cfg.vggt_token_dim, cfg.num_heads, batch_first=True)
        self.gate = nn.Sequential(nn.LayerNorm(cfg.vggt_token_dim), nn.Linear(cfg.vggt_token_dim, 1))
        self.delta_proj = nn.Linear(cfg.vggt_token_dim, cfg.vggt_token_dim)
        self.out_norm = nn.LayerNorm(cfg.vggt_token_dim)
        self.effect_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, real_vggt_tokens: Tensor, smpl_prior_tokens: Tensor) -> dict[str, Tensor]:
        if real_vggt_tokens.ndim != 4:
            raise ValueError(f"real_vggt_tokens must be [B,S,N,C], got {tuple(real_vggt_tokens.shape)}")
        if smpl_prior_tokens.ndim != 4:
            raise ValueError(f"smpl_prior_tokens must be [B,S,Np,C], got {tuple(smpl_prior_tokens.shape)}")
        bsz, views, token_count, channels = real_vggt_tokens.shape
        if channels != self.cfg.vggt_token_dim or smpl_prior_tokens.shape[-1] != channels:
            raise ValueError("VGGT token dim and SMPL prior token dim must match config")
        q = self.token_norm(real_vggt_tokens.reshape(bsz * views, token_count, channels))
        k = self.prior_norm(smpl_prior_tokens.reshape(bsz * views, smpl_prior_tokens.shape[2], channels))
        attended, attn = self.cross_attn(q, k, k, need_weights=True)
        delta = self.delta_proj(attended)
        gate = torch.sigmoid(self.gate(delta))
        bound = self.out_norm(q + gate * delta * self.effect_scale.to(dtype=q.dtype, device=q.device))
        return {
            "bound_tokens": bound.view(bsz, views, token_count, channels),
            "binding_delta": (bound - q).view(bsz, views, token_count, channels),
            "binding_gate": gate.view(bsz, views, token_count, 1),
            "attention": attn.view(bsz, views, token_count, smpl_prior_tokens.shape[2]),
        }


class DetailPreservingDecoder(nn.Module):
    """Decode bound real VGGT context onto model-owned scene-space human surfels."""

    def __init__(self, cfg: V950AdapterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = 3 + 3 + 1 + 8 + cfg.vggt_token_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.point_offset = nn.Linear(cfg.hidden_dim, 3)
        self.rgb_delta = nn.Linear(cfg.hidden_dim, 3)
        self.occupancy = nn.Linear(cfg.hidden_dim, 1)
        self.source = nn.Linear(cfg.hidden_dim, cfg.source_label_count)

    def forward(
        self,
        *,
        world_points: Tensor,
        rgb: Tensor,
        confidence: Tensor,
        skinning_weights: Tensor,
        bound_tokens: Tensor,
        smpl_point_context: Tensor | None,
    ) -> dict[str, Tensor]:
        if world_points.ndim != 3:
            raise ValueError(f"world_points must be [B,N,3], got {tuple(world_points.shape)}")
        bsz, points, _ = world_points.shape
        token_summary = bound_tokens.mean(dim=(1, 2), keepdim=False).unsqueeze(1).expand(-1, points, -1)
        if smpl_point_context is None:
            smpl_point_context = token_summary.new_zeros(bsz, points, self.cfg.vggt_token_dim)
        elif smpl_point_context.shape[1] != points:
            smpl_point_context = F.interpolate(
                smpl_point_context.transpose(1, 2),
                size=points,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        if skinning_weights.shape[-1] < 8:
            skinning_weights = F.pad(skinning_weights.float(), (0, 8 - skinning_weights.shape[-1]))
        skinning_weights = skinning_weights[..., :8].float()
        confidence = confidence.float().unsqueeze(-1) if confidence.ndim == 2 else confidence.float()
        x = torch.cat(
            [
                world_points.float(),
                rgb.float().clamp(0.0, 1.0),
                confidence,
                skinning_weights,
                token_summary,
                smpl_point_context.float(),
            ],
            dim=-1,
        )
        hidden = self.net(x)
        offset = torch.tanh(self.point_offset(hidden)) * self.cfg.max_point_offset
        rgb_out = (rgb.float().clamp(0.0, 1.0) + torch.tanh(self.rgb_delta(hidden)) * self.cfg.max_rgb_delta).clamp(0.0, 1.0)
        source_logits = self.source(hidden)
        return {
            "student_points": world_points.float() + offset,
            "rgb": rgb_out,
            "occupancy": torch.sigmoid(self.occupancy(hidden).squeeze(-1)),
            "source_logits": source_logits,
            "source_label": source_logits.argmax(dim=-1),
            "offset": offset,
        }


class RealVGGT_SMPLFeatureDetailAdapter(nn.Module):
    """Model-owned student that binds SMPL-X features into real VGGT tokens."""

    def __init__(self, cfg: V950AdapterConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or V950AdapterConfig()
        self.smpl_encoder = SMPLFeatureEncoder(self.cfg)
        self.token_binder = RealVGGTTokenBinder(self.cfg)
        self.decoder = DetailPreservingDecoder(self.cfg)

    @staticmethod
    def _reject_forbidden(batch: Mapping[str, Any]) -> None:
        present = sorted(FORBIDDEN_INFERENCE_KEYS.intersection(batch.keys()))
        if present:
            raise ValueError(f"V950 inference forbids these keys: {present}")

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        self._reject_forbidden(batch)
        real_tokens = batch["real_vggt_tokens"].float()
        smpl_feature_images = batch["smpl_feature_images"].float()
        point_features = batch.get("smpl_point_features")
        encoded = self.smpl_encoder(smpl_feature_images, point_features)
        bound = self.token_binder(real_tokens, encoded["smpl_prior_tokens"])
        decoded = self.decoder(
            world_points=batch["world_points"].float(),
            rgb=batch["rgb"].float(),
            confidence=batch["confidence"].float(),
            skinning_weights=batch["skinning_weights"].float(),
            bound_tokens=bound["bound_tokens"],
            smpl_point_context=encoded["point_context"],
        )
        return {
            **decoded,
            **bound,
            "smpl_prior_tokens": encoded["smpl_prior_tokens"],
            "binding_delta_norm": bound["binding_delta"].norm(dim=-1).mean(),
            "token_gradient_expected": torch.tensor(True, device=real_tokens.device),
            "model_owned_student_output": torch.tensor(True, device=real_tokens.device),
            "no_teacher_points_inference": torch.tensor(True, device=real_tokens.device),
            "no_raw_kinect_depth_inference": torch.tensor(True, device=real_tokens.device),
            "source_label_policy": torch.arange(self.cfg.source_label_count, device=real_tokens.device),
        }


def make_v950_batch_from_npz(token_npz: Mapping[str, Any], feature_npz: Mapping[str, Any], *, max_points: int = 4096) -> dict[str, Tensor]:
    real_tokens = np.asarray(token_npz["real_aggregator_tokens_with_smpl_prior_last"], dtype=np.float32)
    feature_image = np.asarray(feature_npz["smpl_feature_image"], dtype=np.float32)
    world = np.asarray(feature_npz["world_points"], dtype=np.float32)
    rgb = np.asarray(feature_npz["rgb"], dtype=np.float32) / 255.0
    confidence = np.asarray(feature_npz["confidence"], dtype=np.float32)
    skinning = np.asarray(feature_npz["skinning_weights"], dtype=np.float32)
    if len(world) > max_points:
        idx = np.linspace(0, len(world) - 1, max_points, dtype=np.int64)
        world = world[idx]
        rgb = rgb[idx]
        confidence = confidence[idx]
        skinning = skinning[idx]
    point_features = np.concatenate([world, rgb, confidence[:, None], skinning[:, :8]], axis=1).astype(np.float32)
    return {
        "real_vggt_tokens": torch.from_numpy(real_tokens).float(),
        "smpl_feature_images": torch.from_numpy(feature_image[None, None]).float().repeat(1, real_tokens.shape[1], 1, 1, 1),
        "smpl_point_features": torch.from_numpy(point_features[None]).float(),
        "world_points": torch.from_numpy(world[None]).float(),
        "rgb": torch.from_numpy(rgb[None]).float(),
        "confidence": torch.from_numpy(confidence[None]).float(),
        "skinning_weights": torch.from_numpy(skinning[None]).float(),
    }
