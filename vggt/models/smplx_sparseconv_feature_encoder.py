from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


AggregationMode = Literal["mean", "sum"]


@dataclass(frozen=True)
class SparseVoxelShape:
    voxels: int
    channels: int
    batch_size: int
    grid_size: tuple[int, int, int]


def _bounds_tensor(bounds: tuple[float, float] | tuple[tuple[float, float], ...]) -> torch.Tensor:
    tensor = torch.as_tensor(bounds, dtype=torch.float32)
    if tensor.shape == (2,):
        tensor = tensor.view(1, 2).expand(3, 2).clone()
    if tensor.shape != (3, 2):
        raise ValueError("bounds must have shape [2] or [3, 2]")
    if not torch.all(tensor[:, 1] > tensor[:, 0]):
        raise ValueError("bounds max values must be greater than min values")
    return tensor


def _grid_tuple(grid_size: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(grid_size, int):
        grid = (grid_size, grid_size, grid_size)
    else:
        grid = tuple(int(v) for v in grid_size)
    if len(grid) != 3 or any(v <= 0 for v in grid):
        raise ValueError("grid_size must be a positive int or a length-3 tuple")
    return grid


def sparse_backend_available() -> str | None:
    if importlib.util.find_spec("spconv") is not None:
        return "spconv"
    if importlib.util.find_spec("MinkowskiEngine") is not None:
        return "minkowski"
    return None


def _validate_points_and_features(
    canonical_xyz: torch.Tensor,
    point_features: torch.Tensor,
    expected_dim: int | None = None,
) -> tuple[int, int, int]:
    if not torch.is_tensor(canonical_xyz) or not torch.is_tensor(point_features):
        raise TypeError("canonical_xyz and point_features must be torch.Tensor instances")
    if canonical_xyz.ndim != 3 or canonical_xyz.shape[-1] != 3:
        raise ValueError(f"canonical_xyz must have shape [B, N, 3], got {tuple(canonical_xyz.shape)}")
    if point_features.ndim != 3:
        raise ValueError(f"point_features must have shape [B, N, C], got {tuple(point_features.shape)}")
    if canonical_xyz.shape[:2] != point_features.shape[:2]:
        raise ValueError(
            "canonical_xyz and point_features batch/point dimensions must match, "
            f"got {tuple(canonical_xyz.shape[:2])} vs {tuple(point_features.shape[:2])}"
        )
    if not torch.is_floating_point(canonical_xyz) or not torch.is_floating_point(point_features):
        raise TypeError("canonical_xyz and point_features must be floating point tensors")
    if expected_dim is not None and point_features.shape[-1] != expected_dim:
        raise ValueError(f"expected point feature dim {expected_dim}, got {point_features.shape[-1]}")
    return canonical_xyz.shape[0], canonical_xyz.shape[1], point_features.shape[-1]


def _linear_ids(coords: torch.Tensor, grid_size: tuple[int, int, int]) -> torch.Tensor:
    gx, gy, gz = grid_size
    volume = gx * gy * gz
    return coords[:, 0] * volume + coords[:, 1] * (gy * gz) + coords[:, 2] * gz + coords[:, 3]


class SMPLXSparseVoxelFeatureBuilder(nn.Module):
    """Build NeuralBody-style sparse voxels from canonical xyz point features."""

    def __init__(
        self,
        bounds: tuple[float, float] | tuple[tuple[float, float], ...] = (-1.0, 1.0),
        grid_size: int | tuple[int, int, int] = 64,
        aggregation: AggregationMode = "mean",
        check_finite: bool = True,
    ) -> None:
        super().__init__()
        if aggregation not in {"mean", "sum"}:
            raise ValueError("aggregation must be 'mean' or 'sum'")
        self.grid_size = _grid_tuple(grid_size)
        self.aggregation = aggregation
        self.check_finite = bool(check_finite)
        self.register_buffer("bounds", _bounds_tensor(bounds), persistent=False)

    def _valid_mask(self, canonical_xyz: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        bounds = self.bounds.to(device=canonical_xyz.device, dtype=canonical_xyz.dtype)
        mins = bounds[:, 0].view(1, 1, 3)
        spans = (bounds[:, 1] - bounds[:, 0]).view(1, 1, 3).clamp_min(torch.finfo(canonical_xyz.dtype).eps)
        rel = (canonical_xyz - mins) / spans
        valid = (rel >= 0.0).all(dim=-1) & (rel <= 1.0).all(dim=-1)
        if self.check_finite:
            valid = valid & torch.isfinite(canonical_xyz).all(dim=-1)
        if mask is not None:
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask[..., 0]
            if mask.ndim != 2 or mask.shape != canonical_xyz.shape[:2]:
                raise ValueError(f"mask must have shape [B, N] or [B, N, 1], got {tuple(mask.shape)}")
            valid = valid & mask.to(device=canonical_xyz.device, dtype=torch.bool)
        return rel, valid

    def quantize(self, canonical_xyz: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if canonical_xyz.ndim != 3 or canonical_xyz.shape[-1] != 3:
            raise ValueError(f"canonical_xyz must have shape [B, N, 3], got {tuple(canonical_xyz.shape)}")
        rel, valid = self._valid_mask(canonical_xyz, mask)
        grid = torch.tensor(self.grid_size, device=canonical_xyz.device, dtype=canonical_xyz.dtype).view(1, 1, 3)
        coords = torch.floor(rel * grid).clamp_min(0)
        max_coord = torch.tensor(
            [self.grid_size[0] - 1, self.grid_size[1] - 1, self.grid_size[2] - 1],
            device=canonical_xyz.device,
            dtype=coords.dtype,
        ).view(1, 1, 3)
        coords = torch.minimum(coords, max_coord).long()
        return coords, valid

    def forward(
        self,
        canonical_xyz: torch.Tensor,
        point_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | tuple[int, int, int] | int]:
        batch, points, channels = _validate_points_and_features(canonical_xyz, point_features)
        coords_xyz, valid = self.quantize(canonical_xyz, mask=mask)
        if self.check_finite:
            valid = valid & torch.isfinite(point_features).all(dim=-1)

        flat_valid = valid.reshape(-1)
        point_to_voxel = torch.full((batch * points,), -1, device=canonical_xyz.device, dtype=torch.long)
        if not bool(flat_valid.any().item()):
            return {
                "voxel_coords": torch.empty(0, 4, device=canonical_xyz.device, dtype=torch.long),
                "voxel_features": point_features.new_zeros(0, channels),
                "voxel_counts": point_features.new_zeros(0, 1),
                "point_to_voxel": point_to_voxel.view(batch, points),
                "valid_mask": valid,
                "grid_size": self.grid_size,
                "batch_size": batch,
            }

        batch_ids = torch.arange(batch, device=canonical_xyz.device).view(batch, 1).expand(batch, points)
        flat_coords = coords_xyz.reshape(batch * points, 3)[flat_valid]
        flat_batches = batch_ids.reshape(-1)[flat_valid].long()
        coord4 = torch.cat([flat_batches[:, None], flat_coords], dim=1)
        unique_ids, inverse = torch.unique(_linear_ids(coord4, self.grid_size), sorted=True, return_inverse=True)

        gx, gy, gz = self.grid_size
        volume = gx * gy * gz
        ub = unique_ids // volume
        rem = unique_ids % volume
        ux = rem // (gy * gz)
        rem = rem % (gy * gz)
        uy = rem // gz
        uz = rem % gz
        voxel_coords = torch.stack([ub, ux, uy, uz], dim=1).long()

        valid_features = point_features.reshape(batch * points, channels)[flat_valid]
        voxel_features = point_features.new_zeros(voxel_coords.shape[0], channels)
        voxel_features.index_add_(0, inverse, valid_features)
        counts = torch.bincount(inverse, minlength=voxel_coords.shape[0]).to(dtype=point_features.dtype).view(-1, 1)
        if self.aggregation == "mean":
            voxel_features = voxel_features / counts.clamp_min(1.0)

        point_to_voxel[flat_valid] = inverse
        return {
            "voxel_coords": voxel_coords,
            "voxel_features": voxel_features,
            "voxel_counts": counts,
            "point_to_voxel": point_to_voxel.view(batch, points),
            "valid_mask": valid,
            "grid_size": self.grid_size,
            "batch_size": batch,
        }


class TorchSparseConvBlock(nn.Module):
    """Tiny pure PyTorch sparse neighbor block used as the default fallback."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.neighbor_norm = nn.LayerNorm(dim)
        self.self_proj = nn.Linear(dim, dim)
        self.neighbor_proj = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.self_proj.weight, gain=0.5)
        nn.init.zeros_(self.self_proj.bias)
        nn.init.xavier_uniform_(self.neighbor_proj.weight, gain=0.5)
        nn.init.zeros_(self.neighbor_proj.bias)

    @staticmethod
    def neighbor_mean(coords: torch.Tensor, features: torch.Tensor, grid_size: tuple[int, int, int]) -> torch.Tensor:
        if coords.numel() == 0:
            return features
        ids = _linear_ids(coords, grid_size)
        sorted_ids, order = torch.sort(ids)
        sorted_features = features[order]
        shifts = torch.tensor(
            [
                [0, 1, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, -1],
            ],
            device=coords.device,
            dtype=coords.dtype,
        )
        accum = features.new_zeros(features.shape)
        counts = features.new_zeros(features.shape[0], 1)
        gx, gy, gz = grid_size
        for shift in shifts:
            neighbor_coords = coords + shift.view(1, 4)
            valid = (
                (neighbor_coords[:, 1] >= 0)
                & (neighbor_coords[:, 1] < gx)
                & (neighbor_coords[:, 2] >= 0)
                & (neighbor_coords[:, 2] < gy)
                & (neighbor_coords[:, 3] >= 0)
                & (neighbor_coords[:, 3] < gz)
            )
            if not bool(valid.any().item()):
                continue
            target_index = valid.nonzero(as_tuple=False).flatten()
            neighbor_ids = _linear_ids(neighbor_coords[target_index], grid_size)
            pos = torch.searchsorted(sorted_ids, neighbor_ids)
            found = (pos < sorted_ids.numel()) & (sorted_ids[pos.clamp_max(sorted_ids.numel() - 1)] == neighbor_ids)
            if not bool(found.any().item()):
                continue
            target_index = target_index[found]
            pos = pos[found]
            accum[target_index] = accum[target_index] + sorted_features[pos]
            counts[target_index] = counts[target_index] + 1.0
        return accum / counts.clamp_min(1.0)

    def forward(self, coords: torch.Tensor, features: torch.Tensor, grid_size: tuple[int, int, int]) -> torch.Tensor:
        neighbor = self.neighbor_mean(coords, features, grid_size)
        update = self.self_proj(self.self_norm(features)) + self.neighbor_proj(self.neighbor_norm(neighbor))
        return features + self.act(update)


class SMPLXSparseConvFeatureEncoder(nn.Module):
    """Dependency-safe sparse voxel encoder with a pure PyTorch sparse fallback."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        bounds: tuple[float, float] | tuple[tuple[float, float], ...] = (-1.0, 1.0),
        grid_size: int | tuple[int, int, int] = 64,
        aggregation: AggregationMode = "mean",
        backend: Literal["auto", "torch", "spconv", "minkowski"] = "auto",
    ) -> None:
        super().__init__()
        if int(in_dim) <= 0 or int(out_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("in_dim, out_dim, and hidden_dim must be positive")
        if backend not in {"auto", "torch", "spconv", "minkowski"}:
            raise ValueError("backend must be auto, torch, spconv, or minkowski")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.requested_backend = backend
        self.available_sparse_backend = sparse_backend_available()
        self.active_backend = "torch"
        self.builder = SMPLXSparseVoxelFeatureBuilder(
            bounds=bounds,
            grid_size=grid_size,
            aggregation=aggregation,
        )
        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([TorchSparseConvBlock(self.hidden_dim) for _ in range(max(0, int(num_layers)))])
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.out_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        linear = self.input_proj[1]
        if isinstance(linear, nn.Linear):
            nn.init.xavier_uniform_(linear.weight, gain=0.5)
            nn.init.zeros_(linear.bias)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.5)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def gather_point_features(
        encoded_voxel_features: torch.Tensor,
        point_to_voxel: torch.Tensor,
    ) -> torch.Tensor:
        if point_to_voxel.ndim != 2:
            raise ValueError("point_to_voxel must have shape [B, N]")
        if encoded_voxel_features.ndim != 2:
            raise ValueError("encoded_voxel_features must have shape [M, C]")
        batch, points = point_to_voxel.shape
        channels = encoded_voxel_features.shape[-1]
        flat = point_to_voxel.reshape(-1)
        out = encoded_voxel_features.new_zeros(batch * points, channels)
        valid = flat >= 0
        if bool(valid.any().item()):
            out[valid] = encoded_voxel_features[flat[valid]]
        return out.view(batch, points, channels)

    def encode_voxels(
        self,
        voxel_coords: torch.Tensor,
        voxel_features: torch.Tensor,
        grid_size: tuple[int, int, int],
    ) -> torch.Tensor:
        if voxel_coords.ndim != 2 or voxel_coords.shape[-1] != 4:
            raise ValueError("voxel_coords must have shape [M, 4]")
        if voxel_features.ndim != 2 or voxel_features.shape[-1] != self.in_dim:
            raise ValueError(f"voxel_features must have shape [M, {self.in_dim}]")
        if voxel_coords.shape[0] != voxel_features.shape[0]:
            raise ValueError("voxel_coords and voxel_features must have the same voxel count")
        if voxel_features.shape[0] == 0:
            return voxel_features.new_zeros(0, self.out_dim)
        h = self.input_proj(voxel_features)
        for block in self.blocks:
            h = block(voxel_coords, h, grid_size)
        return self.output_proj(self.output_norm(h))

    def forward(
        self,
        canonical_xyz: torch.Tensor,
        point_features: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        return_point_features: bool = True,
    ) -> dict[str, torch.Tensor | tuple[int, int, int] | int | str | None]:
        _validate_points_and_features(canonical_xyz, point_features, expected_dim=self.in_dim)
        sparse = self.builder(canonical_xyz, point_features, mask=mask)
        encoded_voxels = self.encode_voxels(
            sparse["voxel_coords"],
            sparse["voxel_features"],
            sparse["grid_size"],
        )
        out: dict[str, torch.Tensor | tuple[int, int, int] | int | str | None] = {
            **sparse,
            "encoded_voxel_features": encoded_voxels,
            "active_backend": self.active_backend,
            "available_sparse_backend": self.available_sparse_backend,
        }
        if return_point_features:
            out["encoded_point_features"] = self.gather_point_features(encoded_voxels, sparse["point_to_voxel"])
        return out

    def output_shape(self, canonical_xyz: torch.Tensor, point_features: torch.Tensor) -> SparseVoxelShape:
        sparse = self.builder(canonical_xyz, point_features)
        return SparseVoxelShape(
            voxels=int(sparse["voxel_features"].shape[0]),
            channels=self.out_dim,
            batch_size=int(sparse["batch_size"]),
            grid_size=sparse["grid_size"],
        )
