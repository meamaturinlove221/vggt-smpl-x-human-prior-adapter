from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


TriPlaneReduce = Literal["sum", "mean", "concat"]


@dataclass(frozen=True)
class TriPlaneTextureShape:
    batch: int
    points: int
    channels: int


def _canonical_bounds_tensor(bounds: tuple[float, float] | tuple[tuple[float, float], ...]) -> torch.Tensor:
    tensor = torch.as_tensor(bounds, dtype=torch.float32)
    if tensor.shape == (2,):
        tensor = tensor.view(1, 2).expand(3, 2).clone()
    if tensor.shape != (3, 2):
        raise ValueError("bounds must have shape [2] or [3, 2]")
    if not torch.all(tensor[:, 1] > tensor[:, 0]):
        raise ValueError("bounds max values must be greater than min values")
    return tensor


def _check_xyz(canonical_xyz: torch.Tensor, *, check_finite: bool) -> None:
    if not torch.is_tensor(canonical_xyz):
        raise TypeError("canonical_xyz must be a torch.Tensor")
    if canonical_xyz.ndim != 3 or canonical_xyz.shape[-1] != 3:
        raise ValueError(f"canonical_xyz must have shape [B, N, 3], got {tuple(canonical_xyz.shape)}")
    if not torch.is_floating_point(canonical_xyz):
        raise TypeError("canonical_xyz must be a floating point tensor")
    if check_finite and not bool(torch.isfinite(canonical_xyz).all().item()):
        raise ValueError("canonical_xyz contains non-finite values")


class DeterministicCanonicalFourier(nn.Module):
    """Parameter-free canonical xyz encoding used beside learnable tri-planes."""

    def __init__(self, num_bands: int = 4, include_xyz: bool = True) -> None:
        super().__init__()
        self.num_bands = int(max(0, num_bands))
        self.include_xyz = bool(include_xyz)

    @property
    def out_dim(self) -> int:
        base = 3 if self.include_xyz else 0
        return base + 6 * self.num_bands

    def forward(self, xyz_normalized: torch.Tensor) -> torch.Tensor:
        if xyz_normalized.ndim != 3 or xyz_normalized.shape[-1] != 3:
            raise ValueError("xyz_normalized must have shape [B, N, 3]")
        parts: list[torch.Tensor] = []
        if self.include_xyz:
            parts.append(xyz_normalized)
        if self.num_bands > 0:
            freq = torch.arange(
                self.num_bands,
                device=xyz_normalized.device,
                dtype=xyz_normalized.dtype,
            )
            freq = (2.0**freq) * math.pi
            angles = xyz_normalized[..., None, :] * freq.view(1, 1, -1, 1)
            parts.extend([torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)])
        if not parts:
            return xyz_normalized.new_zeros(*xyz_normalized.shape[:2], 0)
        return torch.cat(parts, dim=-1)


class SMPLXTriPlaneNeuralTexture(nn.Module):
    """Learnable tri-plane neural texture sampler for canonical SMPL-X xyz.

    The module is intentionally small and deterministic: given the same planes
    and coordinates it only uses ``grid_sample`` and fixed coordinate encodings.
    Learnable planes are zero/small initialized so the texture is safe to attach
    as a residual feature source before training.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        plane_resolution: int = 64,
        bounds: tuple[float, float] | tuple[tuple[float, float], ...] = (-1.0, 1.0),
        reduce: TriPlaneReduce = "mean",
        init_std: float = 0.0,
        padding_mode: Literal["zeros", "border", "reflection"] = "zeros",
        align_corners: bool = True,
        deterministic_bands: int = 4,
        include_xyz: bool = True,
        check_finite: bool = True,
    ) -> None:
        super().__init__()
        if int(feature_dim) <= 0:
            raise ValueError("feature_dim must be positive")
        if int(plane_resolution) <= 1:
            raise ValueError("plane_resolution must be greater than 1")
        if reduce not in {"sum", "mean", "concat"}:
            raise ValueError("reduce must be one of: 'sum', 'mean', 'concat'")
        if padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError("padding_mode must be 'zeros', 'border', or 'reflection'")

        self.feature_dim = int(feature_dim)
        self.plane_resolution = int(plane_resolution)
        self.reduce = reduce
        self.padding_mode = padding_mode
        self.align_corners = bool(align_corners)
        self.check_finite = bool(check_finite)

        self.register_buffer("bounds", _canonical_bounds_tensor(bounds), persistent=False)
        self.planes = nn.Parameter(torch.empty(3, self.feature_dim, self.plane_resolution, self.plane_resolution))
        self.deterministic = DeterministicCanonicalFourier(
            num_bands=deterministic_bands,
            include_xyz=include_xyz,
        )
        self.reset_parameters(init_std=init_std)

    @property
    def output_dim(self) -> int:
        if self.reduce == "concat":
            return self.feature_dim * 3
        return self.feature_dim

    @property
    def deterministic_dim(self) -> int:
        return self.deterministic.out_dim

    def reset_parameters(self, init_std: float = 0.0) -> None:
        if float(init_std) == 0.0:
            nn.init.zeros_(self.planes)
        else:
            nn.init.normal_(self.planes, mean=0.0, std=float(init_std))

    def normalize_xyz(self, canonical_xyz: torch.Tensor) -> torch.Tensor:
        _check_xyz(canonical_xyz, check_finite=self.check_finite)
        bounds = self.bounds.to(device=canonical_xyz.device, dtype=canonical_xyz.dtype)
        mins = bounds[:, 0].view(1, 1, 3)
        spans = (bounds[:, 1] - bounds[:, 0]).view(1, 1, 3).clamp_min(torch.finfo(canonical_xyz.dtype).eps)
        return ((canonical_xyz - mins) / spans) * 2.0 - 1.0

    def _coerce_planes(self, planes: torch.Tensor | None, batch: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        source = self.planes if planes is None else planes
        if source.ndim == 4:
            if source.shape[0] != 3 or source.shape[1] != self.feature_dim:
                raise ValueError(
                    "planes with shape [3, C, H, W] must match "
                    f"[3, {self.feature_dim}, H, W], got {tuple(source.shape)}"
                )
            source = source.unsqueeze(0).expand(batch, -1, -1, -1, -1)
        elif source.ndim == 5:
            if source.shape[0] != batch or source.shape[1] != 3 or source.shape[2] != self.feature_dim:
                raise ValueError(
                    "batched planes must have shape [B, 3, C, H, W] with matching B/C, "
                    f"got {tuple(source.shape)}"
                )
        else:
            raise ValueError("planes must have shape [3, C, H, W] or [B, 3, C, H, W]")
        if source.shape[-1] <= 1 or source.shape[-2] <= 1:
            raise ValueError("plane spatial dimensions must be greater than 1")
        return source.to(device=device, dtype=dtype)

    def sample_planes(
        self,
        canonical_xyz: torch.Tensor,
        planes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz_norm = self.normalize_xyz(canonical_xyz)
        batch, points, _ = xyz_norm.shape
        plane_bank = self._coerce_planes(planes, batch, canonical_xyz.dtype, canonical_xyz.device)
        coords = (
            xyz_norm[..., (0, 1)],
            xyz_norm[..., (0, 2)],
            xyz_norm[..., (1, 2)],
        )
        sampled = []
        for plane_idx, grid_coords in enumerate(coords):
            plane = plane_bank[:, plane_idx]
            grid = grid_coords.view(batch, points, 1, 2)
            value = F.grid_sample(
                plane,
                grid,
                mode="bilinear",
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
            sampled.append(value.squeeze(-1).transpose(1, 2).contiguous())
        stacked = torch.stack(sampled, dim=2)
        if self.reduce == "sum":
            features = stacked.sum(dim=2)
        elif self.reduce == "mean":
            features = stacked.mean(dim=2)
        else:
            features = stacked.flatten(2)
        return features, stacked, xyz_norm

    def forward(
        self,
        canonical_xyz: torch.Tensor,
        planes: torch.Tensor | None = None,
        *,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        features, per_plane_features, xyz_normalized = self.sample_planes(canonical_xyz, planes=planes)
        deterministic_features = self.deterministic(xyz_normalized)
        if not return_dict:
            return features
        return {
            "features": features,
            "per_plane_features": per_plane_features,
            "xyz_normalized": xyz_normalized,
            "deterministic_features": deterministic_features,
        }

    def output_shape(self, canonical_xyz: torch.Tensor) -> TriPlaneTextureShape:
        _check_xyz(canonical_xyz, check_finite=False)
        return TriPlaneTextureShape(canonical_xyz.shape[0], canonical_xyz.shape[1], self.output_dim)
