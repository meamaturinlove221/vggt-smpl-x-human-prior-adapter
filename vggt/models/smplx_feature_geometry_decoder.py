from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GeometryDecoderShape:
    batch: int
    points: int
    channels: int


class SMPLXFeatureGeometryDecoder(nn.Module):
    """Decode small geometry residuals from SMPL-X or fused point features."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        max_delta_point: float = 1.0e-3,
        max_delta_normal: float = 0.10,
        sdf_scale: float = 1.0e-2,
        reliability_bias_init: float = -2.0,
        use_xyz: bool = True,
    ) -> None:
        super().__init__()
        if int(feature_dim) <= 0:
            raise ValueError("feature_dim must be positive")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_delta_point = float(max_delta_point)
        self.max_delta_normal = float(max_delta_normal)
        self.sdf_scale = float(sdf_scale)
        self.use_xyz = bool(use_xyz)
        in_dim = self.feature_dim + (3 if self.use_xyz else 0)

        layers: list[nn.Module] = [nn.LayerNorm(in_dim), nn.Linear(in_dim, self.hidden_dim), nn.GELU()]
        for _ in range(max(0, int(num_layers) - 1)):
            layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU()])
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(self.hidden_dim, 9)
        self.reliability_bias = nn.Parameter(torch.tensor(float(reliability_bias_init)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _validate_features(self, features: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if not torch.is_tensor(features):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim not in (3, 4):
            raise ValueError(f"features must have shape [B, N, C] or [B, V, N, C], got {tuple(features.shape)}")
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"expected feature_dim={self.feature_dim}, got {features.shape[-1]}")
        if not torch.is_floating_point(features):
            raise TypeError("features must be a floating point tensor")
        prefix_shape = tuple(features.shape[:-1])
        flat = features.reshape(-1, features.shape[-2], self.feature_dim) if features.ndim == 4 else features
        return flat, prefix_shape

    def _prepare_xyz(self, canonical_xyz: torch.Tensor | None, prefix_shape: tuple[int, ...], features: torch.Tensor) -> torch.Tensor | None:
        if not self.use_xyz:
            return None
        if canonical_xyz is None:
            return features.new_zeros(*prefix_shape, 3)
        if not torch.is_tensor(canonical_xyz):
            raise TypeError("canonical_xyz must be a torch.Tensor")
        if canonical_xyz.shape != (*prefix_shape, 3):
            raise ValueError(f"canonical_xyz must have shape {(*prefix_shape, 3)}, got {tuple(canonical_xyz.shape)}")
        if not torch.is_floating_point(canonical_xyz):
            raise TypeError("canonical_xyz must be a floating point tensor")
        return canonical_xyz.to(device=features.device, dtype=features.dtype)

    def forward(
        self,
        features: torch.Tensor,
        canonical_xyz: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        *,
        return_logits: bool = False,
    ) -> dict[str, torch.Tensor]:
        flat_features, prefix_shape = self._validate_features(features)
        xyz = self._prepare_xyz(canonical_xyz, prefix_shape, features)
        if xyz is not None:
            flat_xyz = xyz.reshape(flat_features.shape[0], flat_features.shape[1], 3)
            decoder_in = torch.cat([flat_features, flat_xyz], dim=-1)
        else:
            decoder_in = flat_features

        raw = self.head(self.backbone(decoder_in))
        reliability = torch.sigmoid(raw[..., 8:9] + self.reliability_bias.to(device=raw.device, dtype=raw.dtype))
        delta_point = torch.tanh(raw[..., 0:3]) * self.max_delta_point * reliability
        delta_normal = torch.tanh(raw[..., 3:6]) * self.max_delta_normal * reliability
        occupancy_logits = raw[..., 6:7]
        sdf = torch.tanh(raw[..., 7:8]) * self.sdf_scale

        if mask is not None:
            if mask.shape == prefix_shape:
                mask_tensor = mask[..., None]
            elif mask.shape == (*prefix_shape, 1):
                mask_tensor = mask
            else:
                raise ValueError(f"mask must have shape {prefix_shape} or {(*prefix_shape, 1)}, got {tuple(mask.shape)}")
            mask_tensor = mask_tensor.to(device=raw.device, dtype=raw.dtype)
            flat_mask = mask_tensor.reshape(flat_features.shape[0], flat_features.shape[1], 1)
            delta_point = delta_point * flat_mask
            delta_normal = delta_normal * flat_mask
            reliability = reliability * flat_mask

        def restore(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(*prefix_shape, tensor.shape[-1])

        out = {
            "delta_point": restore(delta_point),
            "delta_normal": restore(delta_normal),
            "occupancy": restore(torch.sigmoid(occupancy_logits)),
            "sdf": restore(sdf),
            "reliability": restore(reliability),
        }
        if return_logits:
            out["occupancy_logits"] = restore(occupancy_logits)
        return out

    def apply_residuals(
        self,
        base_points: torch.Tensor,
        base_normals: torch.Tensor | None,
        features: torch.Tensor,
        canonical_xyz: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if base_points.shape[:-1] != features.shape[:-1] or base_points.shape[-1] != 3:
            raise ValueError("base_points must have shape matching features prefix plus 3 channels")
        residuals = self(features, canonical_xyz=canonical_xyz, mask=mask)
        points = base_points + residuals["delta_point"].to(device=base_points.device, dtype=base_points.dtype)
        out = {**residuals, "points": points}
        if base_normals is not None:
            if base_normals.shape != base_points.shape:
                raise ValueError("base_normals must match base_points shape")
            normals = base_normals + residuals["delta_normal"].to(device=base_normals.device, dtype=base_normals.dtype)
            out["normals"] = F.normalize(normals, dim=-1, eps=1e-6)
        return out

    def output_shape(self, features: torch.Tensor) -> GeometryDecoderShape:
        if features.ndim not in (3, 4):
            raise ValueError("features must have shape [B, N, C] or [B, V, N, C]")
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"expected feature_dim={self.feature_dim}, got {features.shape[-1]}")
        points = int(features.shape[-2])
        batch = int(torch.tensor(features.shape[:-2]).prod().item()) if len(features.shape[:-2]) > 0 else 1
        return GeometryDecoderShape(batch=batch, points=points, channels=9)
