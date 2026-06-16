from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vggt.models.smplx_feature_token_adapter import SMPLXFeatureTokenAdapter
from vggt.models.smplx_triplane_neural_texture import SMPLXTriPlaneNeuralTexture


MAIN = Path(r"D:\vggt\vggt-main")
LOCAL = MAIN / "local_report_auxiliary" / "V600_quality_rebuild"
V810_OUT = LOCAL / "output" / "V8100000_V9000000_smplx_feature_encoding"
DEFAULT_OUT = LOCAL / "output" / "V9010000_triplane_adapter_training"
FEATURE_MAPS = V810_OUT / "V8200000_smplx_feature_raster" / "feature_maps.npz"
V770_BASE = (
    LOCAL
    / "output"
    / "V701000_V900000_production_live_highres"
    / "V770000_production_composition_NOT_CANDIDATE"
    / "predictions.npz"
)
TARGET_CANDIDATES = [
    V810_OUT / "V8700000_candidates" / "V870_C7_hybrid_sparse_token" / "predictions.npz",
    V810_OUT / "V8700000_candidates" / "V870_C5_humanram_token" / "predictions.npz",
    V810_OUT / "V8700000_candidates" / "V870_C3_triplane_T0" / "predictions.npz",
    V810_OUT / "V8700000_candidates" / "V870_C1_canonical_xyz" / "predictions.npz",
]


def now() -> str:
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
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: jable(row.get(k, "")) for k in keys})


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def load_pred(path: Path) -> dict[str, np.ndarray]:
    z = load_npz(path)
    points = z.get("world_points", z.get("points"))
    if points is None:
        raise KeyError(f"{path} has no world_points/points")
    depth = z.get("depth", points[..., 2])
    conf = z.get("world_points_conf", z.get("confidence", np.ones(points.shape[:-1], dtype=np.float32)))
    normal = z.get("normal", np.zeros_like(points, dtype=np.float32))
    normal_conf = z.get("normal_conf", np.ones(points.shape[:-1], dtype=np.float32))
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    if normal_conf.ndim == 4 and normal_conf.shape[-1] == 1:
        normal_conf = normal_conf[..., 0]
    return {
        "points": points.astype(np.float32),
        "depth": depth.astype(np.float32),
        "confidence": conf.astype(np.float32),
        "normal": normal.astype(np.float32),
        "normal_conf": normal_conf.astype(np.float32),
    }


def save_pred(path: Path, pred: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        world_points=pred["points"].astype(np.float32),
        points=pred["points"].astype(np.float32),
        depth=pred["depth"].astype(np.float32),
        world_points_conf=pred["confidence"].astype(np.float32),
        confidence=pred["confidence"].astype(np.float32),
        normal=pred["normal"].astype(np.float32),
        normal_conf=pred["normal_conf"].astype(np.float32),
    )


def select_target_candidate(explicit_path: str | None) -> tuple[Path | None, str]:
    if explicit_path:
        path = Path(explicit_path)
        if path.exists():
            return path, "explicit_candidate_array"
        raise FileNotFoundError(f"target candidate does not exist: {path}")
    for path in TARGET_CANDIDATES:
        if path.exists():
            return path, "existing_v810_v900_candidate_array"
    return None, "canonical_feature_proxy_target"


def resize_feature_maps(feature_maps: np.ndarray, train_size: int) -> torch.Tensor:
    x = torch.from_numpy(np.nan_to_num(feature_maps.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
    y = F.interpolate(x, size=(train_size, train_size), mode="bilinear", align_corners=False)
    return y


def resize_points(points: np.ndarray, train_size: int) -> torch.Tensor:
    x = torch.from_numpy(np.nan_to_num(points.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
    x = x.permute(0, 3, 1, 2)
    y = F.interpolate(x, size=(train_size, train_size), mode="bilinear", align_corners=False)
    return y


def resize_mask(mask: np.ndarray, train_size: int) -> torch.Tensor:
    x = torch.from_numpy(mask.astype(np.float32))[:, None]
    y = F.interpolate(x, size=(train_size, train_size), mode="nearest")
    return y[:, 0]


def normalize_channelwise(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=(0, 2, 3), keepdim=True)
    std = features.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-5)
    return (features - mean) / std, mean[:, :, 0, 0], std[:, :, 0, 0]


def canonical_bounds(canonical: torch.Tensor, mask: torch.Tensor) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    coords = canonical.permute(0, 2, 3, 1)[mask > 0.5]
    if coords.numel() == 0:
        coords = canonical.permute(0, 2, 3, 1).reshape(-1, 3)
    bounds = []
    for axis in range(3):
        vals = coords[:, axis]
        lo = float(torch.quantile(vals, 0.001).item())
        hi = float(torch.quantile(vals, 0.999).item())
        pad = max((hi - lo) * 0.05, 1e-3)
        bounds.append((lo - pad, hi + pad))
    return tuple(bounds)  # type: ignore[return-value]


class TriplaneGatedTokenProxy(nn.Module):
    def __init__(
        self,
        *,
        views: int,
        feature_channels: int,
        height: int,
        width: int,
        patch_size: int,
        token_dim: int,
        hidden_dim: int,
        triplane_feature_dim: int,
        triplane_resolution: int,
        triplane_image_channels: int,
        bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ) -> None:
        super().__init__()
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("height and width must be divisible by patch_size")
        self.views = int(views)
        self.height = int(height)
        self.width = int(width)
        self.patch_size = int(patch_size)
        self.patch_h = self.height // self.patch_size
        self.patch_w = self.width // self.patch_size
        self.patch_count = self.patch_h * self.patch_w
        self.triplane_image_channels = int(triplane_image_channels)

        self.triplane = SMPLXTriPlaneNeuralTexture(
            feature_dim=triplane_feature_dim,
            plane_resolution=triplane_resolution,
            bounds=bounds,
            reduce="concat",
            init_std=0.01,
            deterministic_bands=3,
            include_xyz=True,
            padding_mode="border",
        )
        tri_in = self.triplane.output_dim + self.triplane.deterministic_dim
        self.triplane_to_image = nn.Sequential(
            nn.Linear(tri_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.triplane_image_channels),
        )
        self.adapter = SMPLXFeatureTokenAdapter(
            in_chans=feature_channels + self.triplane_image_channels,
            c_vggt=token_dim,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            mode="add",
            prefix_tokens=2,
            num_heads=4,
            gate_init=0.02,
        )
        self.base_tokens = nn.Parameter(torch.zeros(1, self.views, self.patch_count, token_dim))
        self.pred_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, normalized_features: torch.Tensor, canonical_xyz: torch.Tensor) -> dict[str, torch.Tensor]:
        if normalized_features.ndim != 5:
            raise ValueError("normalized_features must have shape [B, V, C, H, W]")
        batch, views, _, height, width = normalized_features.shape
        if views != self.views or height != self.height or width != self.width:
            raise ValueError("feature shape does not match model construction shape")

        xyz = canonical_xyz.permute(0, 1, 3, 4, 2).reshape(batch * views, height * width, 3)
        tri = self.triplane(xyz, return_dict=True)
        tri_features = torch.cat([tri["features"], tri["deterministic_features"]], dim=-1)
        tri_image = self.triplane_to_image(tri_features)
        tri_image = tri_image.transpose(1, 2).reshape(
            batch,
            views,
            self.triplane_image_channels,
            height,
            width,
        )
        feature_image = torch.cat([normalized_features, tri_image], dim=2)
        base_tokens = self.base_tokens.expand(batch, -1, -1, -1)
        adapted = self.adapter(
            feature_image,
            base_tokens,
            mode="add",
            patch_start_idx=0,
            return_dict=True,
        )
        patch_residual = self.pred_head(adapted["tokens"])
        residual_grid = patch_residual.reshape(batch, views, self.patch_h, self.patch_w, 3).permute(0, 1, 4, 2, 3)
        residual_image = F.interpolate(
            residual_grid.reshape(batch * views, 3, self.patch_h, self.patch_w),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, views, 3, height, width)
        return {
            "residual_image": residual_image,
            "patch_residual": patch_residual,
            "smplx_patch_tokens": adapted["smplx_patch_tokens"],
            "adapted_tokens": adapted["tokens"],
            "triplane_image": tri_image,
        }


def patch_average(residual: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch, views, channels, height, width = residual.shape
    pooled = F.avg_pool2d(residual.reshape(batch * views, channels, height, width), kernel_size=patch_size, stride=patch_size)
    return pooled.reshape(batch, views, channels, pooled.shape[-2], pooled.shape[-1]).permute(0, 1, 3, 4, 2).reshape(batch, views, -1, channels)


def collect_grad_stats(model: nn.Module) -> dict[str, Any]:
    rows = []
    total_sq = 0.0
    nonzero_tensors = 0
    for name, param in model.named_parameters():
        if param.grad is None:
            rows.append({"name": name, "has_grad": False, "nonzero": False, "grad_norm": 0.0, "grad_max": 0.0})
            continue
        grad = param.grad.detach()
        grad_norm = float(torch.linalg.vector_norm(grad).item())
        grad_max = float(grad.abs().max().item()) if grad.numel() else 0.0
        nonzero = bool(torch.count_nonzero(grad).item() > 0)
        nonzero_tensors += int(nonzero)
        total_sq += grad_norm * grad_norm
        rows.append({"name": name, "has_grad": True, "nonzero": nonzero, "grad_norm": grad_norm, "grad_max": grad_max})
    return {
        "total_grad_norm": math.sqrt(total_sq),
        "nonzero_parameter_tensors": nonzero_tensors,
        "parameter_tensors": len(rows),
        "by_name": rows,
    }


def snapshot_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters() if param.requires_grad}


def parameter_delta(model: nn.Module, before: dict[str, torch.Tensor]) -> dict[str, Any]:
    total_sq = 0.0
    max_abs = 0.0
    changed = 0
    groups: dict[str, float] = {}
    for name, param in model.named_parameters():
        if name not in before:
            continue
        delta = (param.detach().cpu() - before[name]).float()
        norm = float(torch.linalg.vector_norm(delta).item())
        max_abs = max(max_abs, float(delta.abs().max().item()) if delta.numel() else 0.0)
        changed += int(norm > 0.0)
        total_sq += norm * norm
        group = name.split(".", 1)[0]
        groups[group] = groups.get(group, 0.0) + norm * norm
    return {
        "total_l2": math.sqrt(total_sq),
        "max_abs": max_abs,
        "changed_parameter_tensors": changed,
        "parameter_tensors": len(before),
        "group_l2": {k: math.sqrt(v) for k, v in sorted(groups.items())},
    }


def tensor_to_numpy_residual(residual: torch.Tensor) -> np.ndarray:
    arr = residual.detach().cpu().numpy()[0]
    return np.moveaxis(arr, 1, -1).astype(np.float32)


def upsample_residual(residual_lr: np.ndarray, full_hw: tuple[int, int]) -> np.ndarray:
    x = torch.from_numpy(residual_lr).permute(0, 3, 1, 2)
    y = F.interpolate(x, size=full_hw, mode="bilinear", align_corners=False)
    return y.permute(0, 2, 3, 1).numpy().astype(np.float32)


def mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = (a - b).astype(np.float64)
    if mask is not None:
        diff = diff[mask]
    if diff.size == 0:
        return 0.0
    return float(np.mean(diff * diff))


def mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = np.abs(a - b).astype(np.float64)
    if mask is not None:
        diff = diff[mask]
    if diff.size == 0:
        return 0.0
    return float(np.mean(diff))


def robust_mag(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4 and arr.shape[-1] == 3:
        arr = np.linalg.norm(arr, axis=-1)
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if finite.any():
        hi = float(np.percentile(arr[finite], 99.0))
        if hi > 1e-12:
            arr = arr / hi
    return np.clip(arr, 0.0, 1.0)


def heat_rgb(arr: np.ndarray) -> np.ndarray:
    x = robust_mag(arr)
    return np.stack([x, np.sqrt(x), 1.0 - x], axis=-1)


def write_board(path: Path, target_residual: np.ndarray, pred_residual: np.ndarray, mask: np.ndarray) -> None:
    target = heat_rgb(target_residual).mean(axis=0)
    pred = heat_rgb(pred_residual).mean(axis=0)
    err = heat_rgb(np.abs(pred_residual - target_residual)).mean(axis=0)
    m = np.repeat(mask.astype(np.float32).mean(axis=0)[..., None], 3, axis=-1)
    tiles = [target, pred, err, m]
    labels = ["target residual", "trained prediction", "absolute error", "train mask"]
    tile_w, tile_h = target.shape[1], target.shape[0]
    label_h = 24
    canvas = Image.new("RGB", (tile_w * 2, (tile_h + label_h) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (tile, label) in enumerate(zip(tiles, labels)):
        x = (idx % 2) * tile_w
        y = (idx // 2) * (tile_h + label_h)
        draw.text((x + 6, y + 4), label, fill=(0, 0, 0))
        image = Image.fromarray((np.clip(tile, 0, 1) * 255).astype(np.uint8))
        canvas.paste(image, (x, y + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_loss_curve(path: Path, history: list[dict[str, Any]]) -> None:
    width, height = 900, 420
    margin_l, margin_r, margin_t, margin_b = 70, 20, 28, 55
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    draw.rectangle((margin_l, margin_t, margin_l + plot_w, margin_t + plot_h), outline=(40, 40, 40))
    if not history:
        draw.text((margin_l + 20, margin_t + 20), "no history", fill=(0, 0, 0))
        canvas.save(path)
        return

    keys = [("loss", (20, 80, 180)), ("pixel_loss", (200, 60, 50)), ("patch_loss", (40, 140, 70))]
    values = [max(float(row[k]), 1e-12) for row in history for k, _ in keys if k in row]
    y_min = math.log10(max(min(values), 1e-12))
    y_max = math.log10(max(values))
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0
    steps = [float(row["step"]) for row in history]
    x_min, x_max = min(steps), max(steps)
    if x_max <= x_min:
        x_max = x_min + 1.0

    def xy(step: float, value: float) -> tuple[int, int]:
        x = margin_l + int((step - x_min) / (x_max - x_min) * plot_w)
        y = margin_t + plot_h - int((math.log10(max(float(value), 1e-12)) - y_min) / (y_max - y_min) * plot_h)
        return x, y

    for key, color in keys:
        pts = [xy(float(row["step"]), float(row[key])) for row in history if key in row]
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=3)
        for p in pts[:: max(1, len(pts) // 16)]:
            draw.ellipse((p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2), fill=color)

    draw.text((margin_l, height - 35), "step", fill=(0, 0, 0))
    draw.text((8, margin_t + 4), "log10 loss", fill=(0, 0, 0))
    lx = margin_l + 20
    for label, color in keys:
        draw.line((lx, 14, lx + 24, 14), fill=color, width=3)
        draw.text((lx + 30, 7), label, fill=(0, 0, 0))
        lx += 160
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def git_info() -> dict[str, str]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as exc:  # pragma: no cover - diagnostic only
            return f"ERROR: {exc}"

    return {
        "branch": run(["branch", "--show-current"]),
        "head": run(["rev-parse", "--short", "HEAD"]),
        "status_short": run(["status", "--short"]),
    }


def choose_device(force_cpu: bool) -> tuple[torch.device, str]:
    if force_cpu:
        return torch.device("cpu"), "forced_cpu"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cuda_available = torch.cuda.is_available()
        cuda_warnings = [str(w.message).splitlines()[0] for w in caught]
    except Exception as exc:
        return torch.device("cpu"), f"cuda_availability_check_failed_fallback_cpu: {type(exc).__name__}: {exc}"
    if not cuda_available:
        return torch.device("cpu"), "cuda_not_available"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            device = torch.device("cuda")
            probe = torch.ones(1, device=device)
            probe = probe * 2.0
            torch.cuda.synchronize()
        cuda_warnings.extend(str(w.message).splitlines()[0] for w in caught)
        if float(probe.detach().cpu().item()) == 2.0:
            note = "cuda_probe_passed"
            if cuda_warnings:
                note += "; cuda_warnings=" + " | ".join(cuda_warnings[:3])
            return device, note
    except Exception as exc:
        message = str(exc).splitlines()[0]
        note = f"cuda_probe_failed_fallback_cpu: {type(exc).__name__}: {message}"
        if cuda_warnings:
            note += "; cuda_warnings=" + " | ".join(cuda_warnings[:3])
        return torch.device("cpu"), note
    return torch.device("cpu"), "cuda_probe_failed_fallback_cpu"


def build_proxy_target(canonical_full: np.ndarray, mask_full: np.ndarray) -> np.ndarray:
    smooth = np.tanh(canonical_full.astype(np.float32))
    direction = np.stack(
        [
            0.004 * smooth[:, :, :, 0],
            -0.003 * smooth[:, :, :, 1],
            0.005 * np.sin(np.pi * smooth[:, :, :, 2]),
        ],
        axis=-1,
    )
    return direction * mask_full[..., None].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a lightweight SMPL-X tri-plane plus gated token adapter proxy.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--feature-maps", default=str(FEATURE_MAPS))
    parser.add_argument("--base-predictions", default=str(V770_BASE))
    parser.add_argument("--target-candidate", default=None)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--train-size", type=int, default=112)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--triplane-feature-dim", type=int, default=8)
    parser.add_argument("--triplane-resolution", type=int, default=32)
    parser.add_argument("--triplane-image-channels", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=9010000)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    started = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    device, device_note = choose_device(args.cpu)

    feature_path = Path(args.feature_maps)
    if not feature_path.exists():
        raise FileNotFoundError(f"missing feature maps: {feature_path}")
    raw_features = load_npz(feature_path)
    feature_maps = raw_features["feature_maps"].astype(np.float32)
    channel_names = [str(x) for x in raw_features.get("channel_names", np.asarray([f"ch_{i}" for i in range(feature_maps.shape[1])]))]
    if feature_maps.ndim != 4 or feature_maps.shape[1] < 3:
        raise ValueError(f"feature_maps must have shape [V, C, H, W], got {feature_maps.shape}")
    if args.train_size % args.patch_size != 0:
        raise ValueError("train-size must be divisible by patch-size")

    base_path = Path(args.base_predictions)
    base_pred = load_pred(base_path) if base_path.exists() else None
    target_path, target_kind = select_target_candidate(args.target_candidate)
    target_pred = load_pred(target_path) if target_path is not None else None

    canonical_full = np.moveaxis(feature_maps[:, :3], 1, -1).astype(np.float32)
    visibility_idx = channel_names.index("smplx_visibility") if "smplx_visibility" in channel_names else min(10, feature_maps.shape[1] - 1)
    mask_full = feature_maps[:, visibility_idx] > 0.5
    full_hw = (int(feature_maps.shape[2]), int(feature_maps.shape[3]))

    if base_pred is not None and target_pred is not None:
        target_residual_full = target_pred["points"] - base_pred["points"]
        target_mode = "candidate_residual_supervision"
        target_source = str(target_path)
    else:
        target_residual_full = build_proxy_target(canonical_full, mask_full)
        target_mode = "canonical_feature_proxy_supervision"
        target_source = "generated_from_existing_canonical_feature_maps"
        if base_pred is None:
            base_pred = {
                "points": canonical_full.copy(),
                "depth": canonical_full[..., 2].copy(),
                "confidence": mask_full.astype(np.float32),
                "normal": np.zeros_like(canonical_full, dtype=np.float32),
                "normal_conf": mask_full.astype(np.float32),
            }

    features_lr = resize_feature_maps(feature_maps, args.train_size)
    normalized_lr, feature_mean, feature_std = normalize_channelwise(features_lr)
    canonical_lr = features_lr[:, :3]
    mask_lr = resize_mask(mask_full, args.train_size)
    target_residual_lr = resize_points(target_residual_full, args.train_size)
    residual_mag_lr = torch.linalg.vector_norm(target_residual_lr, dim=1)
    changed_lr = (residual_mag_lr > 1e-7).float()
    train_weight = torch.clamp(mask_lr.float() + changed_lr * 2.0, min=0.1, max=3.0)

    model = TriplaneGatedTokenProxy(
        views=feature_maps.shape[0],
        feature_channels=feature_maps.shape[1],
        height=args.train_size,
        width=args.train_size,
        patch_size=args.patch_size,
        token_dim=args.token_dim,
        hidden_dim=args.hidden_dim,
        triplane_feature_dim=args.triplane_feature_dim,
        triplane_resolution=args.triplane_resolution,
        triplane_image_channels=args.triplane_image_channels,
        bounds=canonical_bounds(canonical_lr, mask_lr),
    ).to(device)

    feature_batch = normalized_lr.unsqueeze(0).to(device)
    canonical_batch = canonical_lr.unsqueeze(0).to(device)
    target_batch = target_residual_lr.unsqueeze(0).to(device)
    weight_batch = train_weight.unsqueeze(0).unsqueeze(2).to(device)
    target_patch = patch_average(target_batch, args.patch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    initial_params = snapshot_parameters(model)
    history: list[dict[str, Any]] = []
    first_grad_stats: dict[str, Any] | None = None
    last_grad_stats: dict[str, Any] | None = None

    for step in range(int(args.steps)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(feature_batch, canonical_batch)
        diff = (out["residual_image"] - target_batch) * weight_batch
        pixel_loss = (diff * diff).mean()
        patch_loss = F.mse_loss(out["patch_residual"], target_patch)
        reg_loss = 1e-5 * (out["triplane_image"] * out["triplane_image"]).mean()
        loss = pixel_loss + 0.35 * patch_loss + reg_loss
        loss.backward()
        grad_stats = collect_grad_stats(model)
        if first_grad_stats is None:
            first_grad_stats = grad_stats
        last_grad_stats = grad_stats
        optimizer.step()
        history.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "pixel_loss": float(pixel_loss.detach().cpu().item()),
                "patch_loss": float(patch_loss.detach().cpu().item()),
                "reg_loss": float(reg_loss.detach().cpu().item()),
                "grad_norm": float(grad_stats["total_grad_norm"]),
                "nonzero_grad_tensors": int(grad_stats["nonzero_parameter_tensors"]),
            }
        )

    model.eval()
    with torch.no_grad():
        final_out = model(feature_batch, canonical_batch)
    pred_residual_lr = tensor_to_numpy_residual(final_out["residual_image"])
    pred_residual_full = upsample_residual(pred_residual_lr, full_hw)
    pred_residual_full = pred_residual_full * mask_full[..., None].astype(np.float32)

    pred_points = (base_pred["points"] + pred_residual_full).astype(np.float32)
    pred = {
        "points": pred_points,
        "depth": pred_points[..., 2].astype(np.float32),
        "confidence": base_pred["confidence"].astype(np.float32),
        "normal": base_pred["normal"].astype(np.float32),
        "normal_conf": base_pred["normal_conf"].astype(np.float32),
    }

    final_delta = parameter_delta(model, initial_params)
    loss_start = float(history[0]["loss"]) if history else 0.0
    loss_end = float(history[-1]["loss"]) if history else 0.0
    target_mask = mask_full | (np.linalg.norm(target_residual_full, axis=-1) > 1e-7)
    eval_payload = {
        "created_utc": now(),
        "status": "V9010000_TRAINED_LIGHTWEIGHT_PROXY_NOT_PROMOTED",
        "real_optimizer_steps": int(args.steps),
        "device": str(device),
        "device_note": device_note,
        "used_cuda": bool(device.type == "cuda"),
        "target_mode": target_mode,
        "target_source": target_source,
        "input_feature_maps": str(feature_path),
        "base_predictions": str(base_path),
        "loss_start": loss_start,
        "loss_end": loss_end,
        "loss_improved": bool(loss_end < loss_start),
        "nonzero_grad_check": {
            "passed": bool(first_grad_stats and first_grad_stats["nonzero_parameter_tensors"] > 0),
            "first_backward": first_grad_stats,
            "last_backward": last_grad_stats,
        },
        "parameter_delta": final_delta,
        "prediction_metrics": {
            "train_lr_mse": mse(pred_residual_lr, target_residual_lr.permute(0, 2, 3, 1).numpy(), mask_lr.numpy() > 0.5),
            "full_mse_vs_target_residual": mse(pred_residual_full, target_residual_full, target_mask),
            "full_mae_vs_target_residual": mae(pred_residual_full, target_residual_full, target_mask),
            "predicted_residual_l2_mean": float(np.linalg.norm(pred_residual_full, axis=-1).mean()),
            "target_residual_l2_mean": float(np.linalg.norm(target_residual_full, axis=-1).mean()),
            "changed_pixels": int((np.linalg.norm(pred_residual_full, axis=-1) > 1e-7).sum()),
            "masked_training_pixels": int(mask_full.sum()),
        },
        "honest_limitations": [
            "This trains a lightweight tri-plane plus gated token adapter proxy, not a full HumanRAM renderer.",
            "Supervision is from existing candidate arrays when available, not new human ground-truth captures.",
            "The script does not promote, register, or replace any active candidate.",
            "Normals and confidence are copied from the base prediction because this proxy only trains point residuals.",
        ],
        "git": git_info(),
        "runtime_seconds": float(time.time() - started),
    }

    config_payload = {
        "created_utc": now(),
        "script": str(Path(__file__).resolve()),
        "args": vars(args),
        "feature_shape_full": list(feature_maps.shape),
        "channel_names": channel_names,
        "feature_mean": feature_mean.numpy().reshape(-1).tolist(),
        "feature_std": feature_std.numpy().reshape(-1).tolist(),
        "model": {
            "triplane": {
                "feature_dim": args.triplane_feature_dim,
                "resolution": args.triplane_resolution,
                "image_channels": args.triplane_image_channels,
                "bounds": model.triplane.bounds.detach().cpu().numpy().tolist(),
                "output_dim": model.triplane.output_dim,
                "deterministic_dim": model.triplane.deterministic_dim,
            },
            "token_adapter": {
                "c_vggt": args.token_dim,
                "patch_size": args.patch_size,
                "mode": "add",
                "patch_count": model.patch_count,
                "gate": "SMPLXFeatureTokenAdapter.add_gamma",
            },
        },
        "data": {
            "target_kind": target_kind,
            "target_mode": target_mode,
            "target_source": target_source,
            "base_predictions": str(base_path),
            "output_predictions_are_candidate": False,
            "promotion": False,
        },
    }

    write_csv(out_dir / "loss_curve.csv", history)
    write_loss_curve(out_dir / "loss_curve.png", history)
    save_pred(out_dir / "predictions.npz", pred)
    np.savez_compressed(
        out_dir / "training_arrays.npz",
        target_residual_lr=target_residual_lr.numpy().astype(np.float32),
        predicted_residual_lr=pred_residual_lr.astype(np.float32),
        target_residual_full=target_residual_full.astype(np.float32),
        predicted_residual_full=pred_residual_full.astype(np.float32),
        train_mask_lr=mask_lr.numpy().astype(np.float32),
    )
    write_board(out_dir / "board.png", target_residual_full, pred_residual_full, target_mask)
    write_json(out_dir / "eval.json", eval_payload)
    write_json(out_dir / "config.json", config_payload)

    print(json.dumps(jable({"status": eval_payload["status"], "output_dir": out_dir, "loss_start": loss_start, "loss_end": loss_end}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
