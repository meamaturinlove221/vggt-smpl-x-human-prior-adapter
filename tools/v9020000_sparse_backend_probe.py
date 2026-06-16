from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vggt.models.smplx_sparseconv_feature_encoder import (  # noqa: E402
    SMPLXSparseConvFeatureEncoder,
    SMPLXSparseVoxelFeatureBuilder,
    sparse_backend_available,
)


MAIN = Path(r"D:\vggt\vggt-main")
DEFAULT_OUT = MAIN / "local_report_auxiliary" / "V600_quality_rebuild" / "output" / "V9020000_sparse_backend_probe"
STATUS_REAL = "REAL_SPARSE_BACKEND"
STATUS_TORCH = "TORCH_FALLBACK_ONLY"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jable(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def module_probe(name: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "module": name,
        "find_spec": False,
        "import_ok": False,
        "version": None,
        "error": None,
    }
    try:
        spec = importlib.util.find_spec(name)
        info["find_spec"] = spec is not None
        if spec is None:
            return info
        module = importlib.import_module(name)
        info["import_ok"] = True
        info["version"] = str(getattr(module, "__version__", "unknown"))
        return info
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info


def backend_inventory() -> dict[str, Any]:
    probes = {
        "spconv": module_probe("spconv"),
        "spconv.pytorch": module_probe("spconv.pytorch"),
        "MinkowskiEngine": module_probe("MinkowskiEngine"),
    }
    usable = []
    if probes["spconv.pytorch"]["import_ok"]:
        usable.append("spconv")
    if probes["MinkowskiEngine"]["import_ok"]:
        usable.append("minkowski")
    return {
        "created_utc": now_utc(),
        "model_helper_sparse_backend_available": sparse_backend_available(),
        "usable_external_backends": usable,
        "selected_external_backend": usable[0] if usable else None,
        "torch": {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "python": sys.version,
        "platform": platform.platform(),
        "probes": probes,
    }


def build_smplx_style_points(points: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    counts = {
        "body": max(16, int(points * 0.56)),
        "head": max(8, int(points * 0.14)),
        "left_hand": max(8, int(points * 0.15)),
    }
    counts["right_hand"] = max(8, int(points) - sum(counts.values()))
    parts: list[torch.Tensor] = []
    part_ids: list[torch.Tensor] = []

    body_n = counts["body"]
    y = torch.rand(body_n, generator=generator) * 1.45 - 0.82
    shoulder = 0.30 - 0.10 * torch.sigmoid((y - 0.35) * 9.0)
    waist = 0.18 + 0.05 * torch.cos((y + 0.15) * math.pi)
    radius_x = torch.maximum(shoulder, waist).clamp(0.12, 0.36)
    radius_z = (0.15 + 0.04 * torch.cos((y - 0.05) * math.pi)).clamp(0.08, 0.21)
    theta = torch.rand(body_n, generator=generator) * (2.0 * math.pi)
    r = torch.sqrt(torch.rand(body_n, generator=generator))
    body = torch.stack(
        [
            torch.cos(theta) * r * radius_x,
            y,
            torch.sin(theta) * r * radius_z,
        ],
        dim=-1,
    )
    body = body + 0.012 * torch.randn(body.shape, generator=generator)
    parts.append(body)
    part_ids.append(torch.zeros(body_n, dtype=torch.long))

    def ellipsoid(n: int, center: tuple[float, float, float], scale: tuple[float, float, float], part: int) -> None:
        direction = torch.randn(n, 3, generator=generator)
        direction = F.normalize(direction, dim=-1)
        radius = torch.rand(n, 1, generator=generator).pow(1.0 / 3.0)
        pts = torch.tensor(center).view(1, 3) + direction * radius * torch.tensor(scale).view(1, 3)
        pts = pts + 0.008 * torch.randn(pts.shape, generator=generator)
        parts.append(pts)
        part_ids.append(torch.full((n,), int(part), dtype=torch.long))

    ellipsoid(counts["head"], (0.0, 0.86, 0.02), (0.18, 0.22, 0.17), 1)
    ellipsoid(counts["left_hand"], (-0.58, 0.10, 0.03), (0.105, 0.145, 0.08), 2)
    ellipsoid(counts["right_hand"], (0.58, 0.10, 0.03), (0.105, 0.145, 0.08), 3)

    xyz = torch.cat(parts, dim=0).clamp(-1.15, 1.15)
    ids = torch.cat(part_ids, dim=0)
    order = torch.randperm(xyz.shape[0], generator=generator)
    xyz = xyz[order]
    ids = ids[order]
    mask = torch.ones(xyz.shape[0], dtype=torch.bool)
    part_onehot = F.one_hot(ids, num_classes=4).float()
    x, y, z = xyz.unbind(dim=-1)
    radial = torch.sqrt((x / 0.7).pow(2) + (z / 0.35).pow(2)).unsqueeze(-1)
    features = torch.cat(
        [
            xyz,
            xyz.square(),
            torch.sin(math.pi * xyz),
            torch.cos(math.pi * xyz),
            radial,
            part_onehot,
            (y.unsqueeze(-1) + 1.2) / 2.4,
            (z.unsqueeze(-1) + 1.2) / 2.4,
        ],
        dim=-1,
    ).float()
    hand_sign = (part_onehot[:, 3:4] - part_onehot[:, 2:3])
    head = part_onehot[:, 1:2]
    hands = part_onehot[:, 2:4].sum(dim=-1, keepdim=True)
    target = torch.cat(
        [
            0.08 * torch.sin(3.0 * x).unsqueeze(-1) + 0.025 * hand_sign,
            0.06 * torch.cos(2.2 * y).unsqueeze(-1) + 0.030 * head,
            0.055 * torch.sin(4.0 * z + 0.6 * x).unsqueeze(-1) + 0.015 * hands,
            0.10 * torch.exp(-5.0 * radial) + 0.020 * head,
        ],
        dim=-1,
    ).float()
    return xyz.unsqueeze(0), features.unsqueeze(0), mask.unsqueeze(0), target.unsqueeze(0)


def coord_linear_ids(coords: torch.Tensor, grid_size: tuple[int, int, int]) -> torch.Tensor:
    gx, gy, gz = grid_size
    volume = gx * gy * gz
    coords = coords.long()
    return coords[:, 0] * volume + coords[:, 1] * (gy * gz) + coords[:, 2] * gz + coords[:, 3]


def aggregate_targets(point_to_voxel: torch.Tensor, point_targets: torch.Tensor, voxel_count: int) -> torch.Tensor:
    flat_map = point_to_voxel.reshape(-1)
    flat_targets = point_targets.reshape(-1, point_targets.shape[-1])
    valid = flat_map >= 0
    out = flat_targets.new_zeros(voxel_count, flat_targets.shape[-1])
    counts = flat_targets.new_zeros(voxel_count, 1)
    if bool(valid.any().item()):
        out.index_add_(0, flat_map[valid], flat_targets[valid])
        counts.index_add_(0, flat_map[valid], torch.ones_like(flat_targets[valid, :1]))
    return out / counts.clamp_min(1.0)


def gather_point_prediction(voxel_prediction: torch.Tensor, point_to_voxel: torch.Tensor) -> torch.Tensor:
    flat = point_to_voxel.reshape(-1)
    out = voxel_prediction.new_zeros(flat.shape[0], voxel_prediction.shape[-1])
    valid = flat >= 0
    if bool(valid.any().item()):
        out[valid] = voxel_prediction[flat[valid]]
    return out.view(*point_to_voxel.shape, voxel_prediction.shape[-1])


def align_by_coords(
    pred_features: torch.Tensor,
    pred_coords: torch.Tensor,
    ref_coords: torch.Tensor,
    grid_size: tuple[int, int, int],
) -> tuple[torch.Tensor, int]:
    if pred_coords.shape == ref_coords.shape and bool(torch.equal(pred_coords.long(), ref_coords.long())):
        return pred_features, 0
    pred_ids = coord_linear_ids(pred_coords.long(), grid_size)
    ref_ids = coord_linear_ids(ref_coords.long(), grid_size)
    sorted_ids, order = torch.sort(pred_ids)
    pos = torch.searchsorted(sorted_ids, ref_ids)
    in_range = pos < sorted_ids.numel()
    safe_pos = pos.clamp_max(max(sorted_ids.numel() - 1, 0))
    found = in_range & (sorted_ids[safe_pos] == ref_ids)
    aligned = pred_features.new_zeros(ref_coords.shape[0], pred_features.shape[-1])
    if bool(found.any().item()):
        aligned[found] = pred_features[order[safe_pos[found]]]
    return aligned, int((~found).sum().item())


def grad_and_param_stats(model: nn.Module, initial_params: torch.Tensor) -> dict[str, Any]:
    grad_nonzero_tensors = 0
    grad_total_tensors = 0
    grad_abs_sum = 0.0
    grad_max = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_total_tensors += 1
        grad_abs = param.grad.detach().abs()
        value = float(grad_abs.sum().item())
        grad_abs_sum += value
        grad_max = max(grad_max, float(grad_abs.max().item()) if grad_abs.numel() else 0.0)
        if value > 0.0:
            grad_nonzero_tensors += 1
    current = torch.cat([p.detach().flatten().cpu() for p in model.parameters() if p.requires_grad])
    delta = current - initial_params
    return {
        "grad_nonzero": bool(grad_abs_sum > 0.0),
        "grad_nonzero_tensors": grad_nonzero_tensors,
        "grad_total_tensors": grad_total_tensors,
        "grad_abs_sum": grad_abs_sum,
        "grad_abs_max": grad_max,
        "parameter_delta_l2": float(torch.linalg.vector_norm(delta).item()),
        "parameter_delta_max_abs": float(delta.abs().max().item()) if delta.numel() else 0.0,
    }


def param_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten().cpu() for p in model.parameters() if p.requires_grad])


def coverage_stats(sparse: dict[str, Any], point_mask: torch.Tensor, part_ids: np.ndarray | None = None) -> dict[str, Any]:
    coords = sparse["voxel_coords"].detach().cpu()
    counts = sparse["voxel_counts"].detach().cpu().numpy().reshape(-1)
    grid_size = tuple(int(v) for v in sparse["grid_size"])
    batch_size = int(sparse["batch_size"])
    valid_mask = sparse["valid_mask"].detach().cpu().numpy().astype(bool)
    point_to_voxel = sparse["point_to_voxel"].detach().cpu().numpy()
    volume = int(np.prod(grid_size) * max(batch_size, 1))
    out: dict[str, Any] = {
        "point_count": int(point_mask.numel()),
        "valid_point_count": int(valid_mask.sum()),
        "valid_point_ratio": float(valid_mask.mean()),
        "occupied_voxel_count": int(coords.shape[0]),
        "grid_size": list(grid_size),
        "batch_size": batch_size,
        "grid_capacity": volume,
        "occupied_voxel_ratio": float(coords.shape[0] / max(volume, 1)),
        "voxel_count_min": float(counts.min()) if counts.size else 0.0,
        "voxel_count_mean": float(counts.mean()) if counts.size else 0.0,
        "voxel_count_median": float(np.median(counts)) if counts.size else 0.0,
        "voxel_count_max": float(counts.max()) if counts.size else 0.0,
    }
    if part_ids is not None:
        names = ["body", "head", "left_hand", "right_hand"]
        flat_map = point_to_voxel.reshape(-1)
        flat_valid = flat_map >= 0
        by_part = {}
        for part, name in enumerate(names):
            sel = (part_ids.reshape(-1) == part) & flat_valid
            by_part[name] = {
                "points": int(sel.sum()),
                "voxels": int(np.unique(flat_map[sel]).shape[0]) if sel.any() else 0,
            }
        out["part_coverage"] = by_part
    return out


class TorchFallbackRoute(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, grid_size: int) -> None:
        super().__init__()
        self.encoder = SMPLXSparseConvFeatureEncoder(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            bounds=(-1.2, 1.2),
            grid_size=grid_size,
            backend="torch",
        )

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, Any]:
        return self.encoder(xyz, features, mask=mask, return_point_features=False)


def build_spconv_route(in_dim: int, out_dim: int, hidden_dim: int, grid_size: int) -> nn.Module:
    spconv = importlib.import_module("spconv.pytorch")

    class SpconvRoute(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.grid = (int(grid_size), int(grid_size), int(grid_size))
            self.input_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.ReLU())
            self.net = spconv.SparseSequential(
                spconv.SubMConv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False, indice_key="v902_subm1"),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                spconv.SubMConv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False, indice_key="v902_subm2"),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
            self.output = nn.Linear(hidden_dim, out_dim)

        def forward(self, coords: torch.Tensor, features: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, int]:
            h = self.input_proj(features)
            # spconv uses [batch, z, y, x] coordinates for a 3D spatial shape.
            spconv_coords = torch.stack([coords[:, 0], coords[:, 3], coords[:, 2], coords[:, 1]], dim=1).int()
            tensor = spconv.SparseConvTensor(
                features=h,
                indices=spconv_coords,
                spatial_shape=[self.grid[2], self.grid[1], self.grid[0]],
                batch_size=int(batch_size),
            )
            encoded = self.net(tensor)
            out_coords = torch.stack(
                [encoded.indices[:, 0], encoded.indices[:, 3], encoded.indices[:, 2], encoded.indices[:, 1]],
                dim=1,
            ).long()
            pred = self.output(encoded.features)
            return align_by_coords(pred, out_coords, coords, self.grid)

    return SpconvRoute()


def build_minkowski_route(in_dim: int, out_dim: int, hidden_dim: int, grid_size: int) -> nn.Module:
    me = importlib.import_module("MinkowskiEngine")

    class MinkowskiRoute(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.grid = (int(grid_size), int(grid_size), int(grid_size))
            self.input = me.MinkowskiLinear(in_dim, hidden_dim)
            self.conv1 = me.MinkowskiConvolution(hidden_dim, hidden_dim, kernel_size=3, dimension=3)
            self.bn1 = me.MinkowskiBatchNorm(hidden_dim)
            self.relu1 = me.MinkowskiReLU(inplace=True)
            self.conv2 = me.MinkowskiConvolution(hidden_dim, hidden_dim, kernel_size=3, dimension=3)
            self.bn2 = me.MinkowskiBatchNorm(hidden_dim)
            self.relu2 = me.MinkowskiReLU(inplace=True)
            self.output = me.MinkowskiLinear(hidden_dim, out_dim)

        def forward(self, coords: torch.Tensor, features: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, int]:
            del batch_size
            tensor = me.SparseTensor(features=features, coordinates=coords.int(), device=features.device)
            encoded = self.output(self.relu2(self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(self.input(tensor))))))))
            return align_by_coords(encoded.F, encoded.C.long().to(coords.device), coords, self.grid)

    return MinkowskiRoute()


def try_external_route(
    backend: str,
    voxel_coords: torch.Tensor,
    voxel_features: torch.Tensor,
    voxel_targets: torch.Tensor,
    batch_size: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], nn.Module | None, torch.Tensor | None]:
    build: Callable[[int, int, int, int], nn.Module]
    if backend == "spconv":
        build = build_spconv_route
    elif backend == "minkowski":
        build = build_minkowski_route
    else:
        return {"ok": False, "backend": backend, "error": "unsupported backend"}, None, None

    device = choose_device(args.device, prefer_cuda=True)
    try:
        model = build(voxel_features.shape[-1], voxel_targets.shape[-1], args.hidden_dim, args.grid_size).to(device)
        coords = voxel_coords.to(device)
        features = voxel_features.to(device)
        targets = voxel_targets.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        initial = param_vector(model)
        losses: list[float] = []
        missing_coords = 0
        last_stats: dict[str, Any] = {}
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            pred, missing_coords = model(coords, features, batch_size)
            loss = F.mse_loss(pred, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            last_stats = grad_and_param_stats(model, initial)
        with torch.no_grad():
            pred, missing_coords = model(coords, features, batch_size)
        route = {
            "ok": True,
            "backend": backend,
            "device": str(device),
            "losses": losses,
            "loss_start": losses[0],
            "loss_end": losses[-1],
            "missing_output_coord_count": int(missing_coords),
            **last_stats,
        }
        return route, model, pred.detach().cpu()
    except Exception as exc:
        return {
            "ok": False,
            "backend": backend,
            "device": str(device),
            "error": f"{type(exc).__name__}: {exc}",
        }, None, None


def choose_device(requested: str, prefer_cuda: bool = False) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_torch_fallback(
    xyz: torch.Tensor,
    features: torch.Tensor,
    mask: torch.Tensor,
    point_targets: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, torch.Tensor]:
    device = choose_device(args.device, prefer_cuda=False)
    model = TorchFallbackRoute(features.shape[-1], point_targets.shape[-1], args.hidden_dim, args.grid_size).to(device)
    xyz = xyz.to(device)
    features = features.to(device)
    mask = mask.to(device)
    point_targets = point_targets.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    initial = param_vector(model)
    losses: list[float] = []
    last_stats: dict[str, Any] = {}
    sparse: dict[str, Any] = {}
    prediction = point_targets.new_zeros(point_targets.shape)

    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        sparse = model(xyz, features, mask)
        voxel_targets = aggregate_targets(sparse["point_to_voxel"], point_targets, sparse["encoded_voxel_features"].shape[0])
        loss = F.mse_loss(sparse["encoded_voxel_features"], voxel_targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        last_stats = grad_and_param_stats(model, initial)

    with torch.no_grad():
        sparse = model(xyz, features, mask)
        prediction = gather_point_prediction(sparse["encoded_voxel_features"], sparse["point_to_voxel"])
    route = {
        "ok": True,
        "backend": "torch_fallback",
        "device": str(device),
        "losses": losses,
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "active_backend": sparse.get("active_backend"),
        "available_sparse_backend": sparse.get("available_sparse_backend"),
        **last_stats,
    }
    sparse_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sparse.items()}
    return route, sparse_cpu, sparse_cpu["encoded_voxel_features"], prediction.detach().cpu()


def make_board(
    path: Path,
    xyz: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    point_to_voxel: np.ndarray,
    losses: list[float],
    status: str,
    backend: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 960, 720
    image = Image.new("RGB", (width, height), (248, 248, 246))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((24, 16), f"V902 sparse backend probe: {status} ({backend})", fill=(20, 20, 20), font=font)

    panels = [
        (24, 54, 456, 338, "canonical XY coverage"),
        (504, 54, 936, 338, "optimizer loss"),
        (24, 388, 456, 686, "target vs prediction ch0"),
        (504, 388, 936, 686, "absolute error in XY"),
    ]
    for x0, y0, x1, y1, title in panels:
        draw.rectangle((x0, y0, x1, y1), outline=(70, 70, 70), width=1)
        draw.text((x0 + 10, y0 + 8), title, fill=(30, 30, 30), font=font)

    flat_xyz = xyz.reshape(-1, 3)
    flat_target = target.reshape(-1, target.shape[-1])
    flat_pred = pred.reshape(-1, pred.shape[-1])
    valid = point_to_voxel.reshape(-1) >= 0
    sample = np.linspace(0, flat_xyz.shape[0] - 1, min(flat_xyz.shape[0], 2500)).astype(np.int64)

    def xy_to_panel(points: np.ndarray, panel: tuple[int, int, int, int, str]) -> tuple[np.ndarray, np.ndarray]:
        x0, y0, x1, y1, _ = panel
        px = x0 + 20 + (points[:, 0] + 1.2) / 2.4 * (x1 - x0 - 40)
        py = y1 - 20 - (points[:, 1] + 1.2) / 2.4 * (y1 - y0 - 50)
        return px, py

    px, py = xy_to_panel(flat_xyz[sample], panels[0])
    for i, idx in enumerate(sample):
        color = (40, 130, 190) if valid[idx] else (210, 60, 60)
        draw.point((float(px[i]), float(py[i])), fill=color)

    x0, y0, x1, y1, _ = panels[1]
    if losses:
        lo, hi = min(losses), max(losses)
        span = max(hi - lo, 1e-12)
        pts = []
        for i, loss in enumerate(losses):
            x = x0 + 24 + i / max(len(losses) - 1, 1) * (x1 - x0 - 48)
            y = y1 - 26 - (loss - lo) / span * (y1 - y0 - 68)
            pts.append((x, y))
        if len(pts) > 1:
            draw.line(pts, fill=(55, 100, 190), width=3)
        for p in pts:
            draw.ellipse((p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2), fill=(25, 80, 170))
        draw.text((x0 + 16, y1 - 42), f"start={losses[0]:.6f}  end={losses[-1]:.6f}", fill=(30, 30, 30), font=font)

    x0, y0, x1, y1, _ = panels[2]
    tv = flat_target[sample, 0]
    pv = flat_pred[sample, 0]
    lo = float(min(tv.min(), pv.min()))
    hi = float(max(tv.max(), pv.max()))
    span = max(hi - lo, 1e-9)
    for t, p in zip(tv, pv):
        sx = x0 + 28 + (t - lo) / span * (x1 - x0 - 56)
        sy = y1 - 28 - (p - lo) / span * (y1 - y0 - 68)
        draw.point((float(sx), float(sy)), fill=(35, 130, 90))
    draw.line((x0 + 28, y1 - 28, x1 - 28, y0 + 40), fill=(90, 90, 90), width=1)

    err = np.mean(np.abs(flat_pred - flat_target), axis=-1)
    denom = max(float(np.percentile(err, 98)), 1e-9)
    px, py = xy_to_panel(flat_xyz[sample], panels[3])
    for i, idx in enumerate(sample):
        heat = float(np.clip(err[idx] / denom, 0.0, 1.0))
        color = (int(240 * heat), int(70 + 120 * (1.0 - heat)), int(210 * (1.0 - heat)))
        draw.point((float(px[i]), float(py[i])), fill=color)
    image.save(path)


def git_info() -> dict[str, Any]:
    import subprocess

    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    return {
        "branch": run(["git", "branch", "--show-current"]),
        "commit": run(["git", "rev-parse", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))

    inventory = backend_inventory()
    xyz, features, mask, point_targets = build_smplx_style_points(args.points, args.seed)
    part_ids = torch.argmax(features[..., 13:17], dim=-1).detach().cpu().numpy()
    builder = SMPLXSparseVoxelFeatureBuilder(bounds=(-1.2, 1.2), grid_size=args.grid_size)
    with torch.no_grad():
        sparse_seed = builder(xyz, features, mask=mask)
        voxel_targets_seed = aggregate_targets(
            sparse_seed["point_to_voxel"],
            point_targets,
            sparse_seed["voxel_features"].shape[0],
        )
    coverage = coverage_stats(sparse_seed, mask, part_ids=part_ids)

    external_attempts: list[dict[str, Any]] = []
    final_status = STATUS_TORCH
    route: dict[str, Any]
    latent: torch.Tensor
    point_prediction: torch.Tensor
    sparse_final: dict[str, Any]
    backend_used = "torch_fallback"

    preferred = args.backend
    selected_external = inventory["selected_external_backend"]
    external_order: list[str] = []
    if preferred in {"spconv", "minkowski"}:
        external_order = [preferred]
    elif preferred == "auto" and selected_external:
        external_order = [selected_external] + [b for b in ("spconv", "minkowski") if b != selected_external and b in inventory["usable_external_backends"]]

    for backend in external_order:
        attempt, _model, voxel_prediction = try_external_route(
            backend,
            sparse_seed["voxel_coords"],
            sparse_seed["voxel_features"],
            voxel_targets_seed,
            int(sparse_seed["batch_size"]),
            args,
        )
        external_attempts.append(attempt)
        if attempt.get("ok") and voxel_prediction is not None:
            final_status = STATUS_REAL
            backend_used = backend
            route = attempt
            latent = voxel_prediction.detach().cpu()
            point_prediction = gather_point_prediction(latent, sparse_seed["point_to_voxel"]).detach().cpu()
            sparse_final = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sparse_seed.items()}
            break
    else:
        route, sparse_final, latent, point_prediction = run_torch_fallback(xyz, features, mask, point_targets, args)

    losses = [float(v) for v in route.get("losses", [])]
    target_np = point_targets.detach().cpu().numpy()
    pred_np = point_prediction.detach().cpu().numpy()
    xyz_np = xyz.detach().cpu().numpy()
    point_to_voxel_np = sparse_final["point_to_voxel"].detach().cpu().numpy()
    voxel_target_final = aggregate_targets(sparse_final["point_to_voxel"], point_targets.cpu(), latent.shape[0])
    mean_abs_error = float(np.mean(np.abs(pred_np - target_np)))
    rmse = float(np.sqrt(np.mean((pred_np - target_np) ** 2)))

    latent_path = out_dir / "latent_field.npz"
    predictions_path = out_dir / "predictions.npz"
    board_path = out_dir / "board.png"
    eval_path = out_dir / "eval.json"
    config_path = out_dir / "config.json"

    save_npz(
        latent_path,
        voxel_coords=sparse_final["voxel_coords"].detach().cpu().numpy().astype(np.int32),
        voxel_features=sparse_final["voxel_features"].detach().cpu().numpy().astype(np.float32),
        voxel_counts=sparse_final["voxel_counts"].detach().cpu().numpy().astype(np.float32),
        voxel_targets=voxel_target_final.detach().cpu().numpy().astype(np.float32),
        latent=latent.detach().cpu().numpy().astype(np.float32),
        grid_size=np.asarray(sparse_final["grid_size"], dtype=np.int32),
        backend=np.asarray([backend_used]),
        final_status=np.asarray([final_status]),
    )
    save_npz(
        predictions_path,
        canonical_xyz=xyz_np.astype(np.float32),
        point_features=features.detach().cpu().numpy().astype(np.float32),
        target=target_np.astype(np.float32),
        prediction=pred_np.astype(np.float32),
        valid_mask=sparse_final["valid_mask"].detach().cpu().numpy().astype(np.bool_),
        point_to_voxel=point_to_voxel_np.astype(np.int32),
        backend=np.asarray([backend_used]),
        final_status=np.asarray([final_status]),
    )
    make_board(board_path, xyz_np, target_np, pred_np, point_to_voxel_np, losses, final_status, backend_used)

    eval_payload = {
        "created_utc": now_utc(),
        "status": final_status,
        "backend_used": backend_used,
        "backend_inventory": inventory,
        "external_attempts": external_attempts,
        "grad_nonzero": bool(route.get("grad_nonzero", False)),
        "loss_start": float(route.get("loss_start", 0.0)),
        "loss_end": float(route.get("loss_end", 0.0)),
        "loss_decreased": bool(route.get("loss_end", 0.0) < route.get("loss_start", 0.0)),
        "parameter_delta_l2": float(route.get("parameter_delta_l2", 0.0)),
        "parameter_delta_max_abs": float(route.get("parameter_delta_max_abs", 0.0)),
        "grad_stats": {
            "grad_nonzero_tensors": route.get("grad_nonzero_tensors", 0),
            "grad_total_tensors": route.get("grad_total_tensors", 0),
            "grad_abs_sum": route.get("grad_abs_sum", 0.0),
            "grad_abs_max": route.get("grad_abs_max", 0.0),
        },
        "coverage_stats": coverage,
        "prediction_metrics": {
            "mean_abs_error": mean_abs_error,
            "rmse": rmse,
            "target_abs_mean": float(np.mean(np.abs(target_np))),
            "prediction_abs_mean": float(np.mean(np.abs(pred_np))),
        },
        "artifacts": {
            "latent_field": latent_path,
            "predictions": predictions_path,
            "board": board_path,
            "config": config_path,
            "eval": eval_path,
        },
        "honest_scope": (
            "REAL_SPARSE_BACKEND means an importable spconv or MinkowskiEngine route ran optimizer steps. "
            "TORCH_FALLBACK_ONLY means the existing SMPLXSparseConvFeatureEncoder PyTorch sparse-neighbor route "
            "trained on sparse voxel features, with no external SparseConv3D backend claim."
        ),
        "runtime_seconds": float(time.time() - started),
        "git": git_info(),
    }
    config_payload = {
        "created_utc": eval_payload["created_utc"],
        "script": Path(__file__).resolve(),
        "out_dir": out_dir.resolve(),
        "args": vars(args),
        "final_status": final_status,
        "backend_used": backend_used,
        "route": route,
        "data": {
            "source": "deterministic NeuralBody-style SMPL-X canonical sparse voxel probe",
            "points": int(args.points),
            "feature_dim": int(features.shape[-1]),
            "target_dim": int(point_targets.shape[-1]),
            "bounds": [-1.2, 1.2],
            "grid_size": int(args.grid_size),
        },
    }
    write_json(eval_path, eval_payload)
    write_json(config_path, config_payload)
    write_json(out_dir / "backend_inventory.json", inventory)
    write_json(out_dir / "final_status.json", {"status": final_status, "backend_used": backend_used, "eval": eval_path})
    return eval_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V902 real sparse backend probe with honest PyTorch fallback training for SMPL-X sparse voxel features."
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--backend", choices=["auto", "spconv", "minkowski", "torch"], default="auto")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string.")
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=9020000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.points <= 0:
        raise ValueError("--points must be positive")
    if args.grid_size <= 1:
        raise ValueError("--grid-size must be greater than 1")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    result = run(args)
    print(json.dumps({"status": result["status"], "backend_used": result["backend_used"], "eval": str(result["artifacts"]["eval"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
