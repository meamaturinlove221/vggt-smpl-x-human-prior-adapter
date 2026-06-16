from __future__ import annotations

import torch
from torch import nn


class HighResCropGeometryBranch(nn.Module):
    """Feature-conditioned local geometry branch for high-resolution human crops.

    The branch is deliberately identity-initialized. It only changes dense VGGT
    outputs when the caller provides crop features and source-pixel indices, and
    when training has moved the final projection away from zero.

    This module owns *where* local residuals may be applied:
    ``crop_indices`` are integer ``(view, y, x)`` coordinates in the original
    dense output grid. The branch never updates pixels outside those indices.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        max_delta_point: float = 1.0e-3,
        max_delta_depth: float = 1.0e-3,
        max_delta_normal: float = 0.10,
        gate_bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.max_delta_point = float(max_delta_point)
        self.max_delta_depth = float(max_delta_depth)
        self.max_delta_normal = float(max_delta_normal)
        self.backbone = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(hidden_dim, 8)
        self.gate_bias = nn.Parameter(torch.tensor(float(gate_bias_init)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.7)
                nn.init.zeros_(module.bias)
        # Strict identity at construction: delta, gate residual and uncertainty
        # outputs are all zero before optimization.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def predict_residuals(self, crop_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if crop_features.ndim != 3:
            raise ValueError("crop_features must have shape [B, N, C]")
        if crop_features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {crop_features.shape[-1]}")
        h = self.backbone(crop_features)
        raw = self.delta_head(h)
        gate = torch.sigmoid(raw[..., 7:8] + self.gate_bias)
        return {
            "delta_point": torch.tanh(raw[..., 0:3]) * self.max_delta_point * gate,
            "delta_depth": torch.tanh(raw[..., 3:4]) * self.max_delta_depth * gate,
            "delta_normal": torch.tanh(raw[..., 4:7]) * self.max_delta_normal * gate,
            "apply_gate": gate,
            "uncertainty": torch.sigmoid(-raw[..., 7:8]),
        }

    @staticmethod
    def _scatter_add_dense(base: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Scatter-add values into ``base`` at ``(view, y, x)`` indices.

        Args:
            base: Tensor with shape [B, S, H, W, C].
            indices: Tensor with shape [B, N, 3] containing view/y/x.
            values: Tensor with shape [B, N, C].
        """
        if base.ndim != 5:
            raise ValueError("base must have shape [B, S, H, W, C]")
        if indices.ndim != 3 or indices.shape[-1] != 3:
            raise ValueError("indices must have shape [B, N, 3]")
        if values.ndim != 3:
            raise ValueError("values must have shape [B, N, C]")
        bsz, seq, height, width, channels = base.shape
        if values.shape[0] != bsz or values.shape[-1] != channels:
            raise ValueError("values batch/channel dimensions must match base")
        out = base.clone()
        idx = indices.long()
        valid = (
            (idx[..., 0] >= 0)
            & (idx[..., 0] < seq)
            & (idx[..., 1] >= 0)
            & (idx[..., 1] < height)
            & (idx[..., 2] >= 0)
            & (idx[..., 2] < width)
        )
        for b in range(bsz):
            if valid[b].any():
                ii = idx[b, valid[b]]
                out[b, ii[:, 0], ii[:, 1], ii[:, 2]] = out[b, ii[:, 0], ii[:, 1], ii[:, 2]] + values[b, valid[b]]
        return out

    def forward(
        self,
        world_points: torch.Tensor,
        depth: torch.Tensor | None,
        normal: torch.Tensor | None,
        crop_features: torch.Tensor,
        crop_indices: torch.Tensor,
        crop_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if world_points.ndim != 5 or world_points.shape[-1] != 3:
            raise ValueError("world_points must have shape [B, S, H, W, 3]")
        residuals = self.predict_residuals(crop_features)
        weights = 1.0
        if crop_weights is not None:
            if crop_weights.ndim == 2:
                crop_weights = crop_weights[..., None]
            weights = crop_weights.to(dtype=world_points.dtype, device=world_points.device)
        delta_point = residuals["delta_point"].to(dtype=world_points.dtype, device=world_points.device) * weights
        corrected_points = self._scatter_add_dense(world_points, crop_indices.to(world_points.device), delta_point)
        out = {
            "world_points": corrected_points,
            "crop_delta_point": delta_point,
            "crop_apply_gate": residuals["apply_gate"],
            "crop_uncertainty": residuals["uncertainty"],
        }
        if depth is not None:
            squeeze_last = False
            depth_base = depth
            if depth_base.ndim == 5 and depth_base.shape[-1] == 1:
                depth_base = depth_base
            elif depth_base.ndim == 4:
                depth_base = depth_base[..., None]
                squeeze_last = True
            else:
                raise ValueError("depth must have shape [B, S, H, W] or [B, S, H, W, 1]")
            delta_depth = residuals["delta_depth"].to(dtype=depth_base.dtype, device=depth_base.device) * weights
            corrected_depth = self._scatter_add_dense(depth_base, crop_indices.to(depth_base.device), delta_depth)
            out["depth"] = corrected_depth[..., 0] if squeeze_last else corrected_depth
            out["crop_delta_depth"] = delta_depth
        if normal is not None:
            delta_normal = residuals["delta_normal"].to(dtype=normal.dtype, device=normal.device) * weights
            corrected_normal = self._scatter_add_dense(normal, crop_indices.to(normal.device), delta_normal)
            corrected_normal = torch.nn.functional.normalize(corrected_normal, dim=-1, eps=1e-6)
            out["normal"] = corrected_normal
            out["crop_delta_normal"] = delta_normal
        return out
