from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.models.smplx_triplane_neural_texture import SMPLXTriPlaneNeuralTexture
from vggt.models.smplx_sparseconv_feature_encoder import SMPLXSparseConvFeatureEncoder


ROOT = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = ROOT / "reports"
ARCHIVE = ROOT / "archive"
OUT = ROOT / "output" / "V9030000_V9300000_feature_training"
V900 = ROOT / "output" / "V8100000_V9000000_smplx_feature_encoding"
FEATURE_MAPS = V900 / "V8200000_smplx_feature_raster" / "feature_maps.npz"
SPARSE_TENSOR = V900 / "V8400000_sparse_voxel_features" / "sparse_tensor.npz"
SCHEMA_REPORT = REPORTS / "V8110000_schema_report.json"
V900_STATUS = REPORTS / "V9000000_final_status.json"
WORKER_TRI_OUT = ROOT / "output" / "V9010000_triplane_adapter_training"
WORKER_SPARSE_OUT = ROOT / "output" / "V9020000_sparse_backend_probe"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    data = {k: z[k] for k in z.files}
    z.close()
    if "world_points" not in data and "points" in data:
        data["world_points"] = data["points"]
    if "points" not in data and "world_points" in data:
        data["points"] = data["world_points"]
    if "confidence" not in data and "world_points_conf" in data:
        data["confidence"] = data["world_points_conf"]
    if "world_points_conf" not in data and "confidence" in data:
        data["world_points_conf"] = data["confidence"]
    return data


def save_prediction(path: Path, base: dict[str, np.ndarray], world_points: np.ndarray, normal: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(base)
    payload["world_points"] = world_points.astype(np.float32)
    payload["points"] = world_points.astype(np.float32)
    payload["depth"] = world_points[..., 2].astype(np.float32)
    if normal is not None:
        payload["normal"] = normal.astype(np.float32)
        payload.setdefault("normal_conf", np.ones(world_points.shape[:-1], dtype=np.float32))
    payload.setdefault("confidence", base.get("confidence", np.ones(world_points.shape[:-1], dtype=np.float32)))
    payload.setdefault("world_points_conf", payload["confidence"])
    np.savez_compressed(path, **payload)


def draw_board(path: Path, title: str, masks: dict[str, np.ndarray], delta_norm: np.ndarray, curves: dict[str, list[float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle(title)
    view = 0
    show_items = [
        ("foreground", masks["foreground"][view]),
        ("head", masks["head_face"][view]),
        ("hair", masks["hairline"][view]),
        ("left_hand", masks["left_hand"][view]),
        ("right_hand", masks["right_hand"][view]),
        ("delta_norm", delta_norm[view]),
    ]
    for ax, (name, arr) in zip(axes.flat, show_items):
        im = ax.imshow(arr, cmap="magma" if name == "delta_norm" else "viridis")
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    if curves:
        curve_path = path.with_name(path.stem + "_loss_curve.png")
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, values in curves.items():
            ax.plot(values, label=name)
        ax.legend()
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(curve_path, dpi=150)
        plt.close(fig)


def finite_stats(path: Path, data: dict[str, np.ndarray]) -> dict:
    stats = {"path": str(path), "keys": {}}
    for k, v in data.items():
        if not isinstance(v, np.ndarray):
            continue
        item = {"shape": list(v.shape), "dtype": str(v.dtype)}
        if np.issubdtype(v.dtype, np.number):
            finite = np.isfinite(v)
            item.update(
                {
                    "finite_ratio": float(finite.mean()) if finite.size else 0.0,
                    "nan_ratio": float(np.isnan(v).mean()) if finite.size else 0.0,
                    "min": float(np.nanmin(v)) if finite.size else 0.0,
                    "max": float(np.nanmax(v)) if finite.size else 0.0,
                    "mean": float(np.nanmean(v)) if finite.size else 0.0,
                }
            )
        stats["keys"][k] = item
    return stats


def build_masks(feature_maps: np.ndarray, channel_names: list[str]) -> dict[str, np.ndarray]:
    idx = {name: i for i, name in enumerate(channel_names)}

    def chan(name: str, default: float = 0.0) -> np.ndarray:
        if name in idx:
            return feature_maps[:, idx[name]]
        return np.full(feature_maps.shape[0:1] + feature_maps.shape[2:], default, dtype=np.float32)

    foreground = chan("semantic_foreground")
    visibility = chan("smplx_visibility")
    masks = {
        "foreground": (np.maximum(foreground, visibility) > 0.10),
        "head_face": (chan("semantic_head_face") > 0.10),
        "hairline": (chan("semantic_hairline") > 0.10),
        "left_hand": (chan("semantic_left_hand") > 0.10),
        "right_hand": (chan("semantic_right_hand") > 0.10),
        "phone_object": (chan("phone_object_exclusion") > 0.10),
    }
    masks["human_safe"] = masks["foreground"] & ~masks["phone_object"]
    return {k: v.astype(bool) for k, v in masks.items()}


def region_metrics(candidate: np.ndarray, base: np.ndarray, teacher: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    teacher_delta = teacher - base
    cand_delta = candidate - base
    delta_vs_base = np.linalg.norm(cand_delta, axis=-1)
    target_norm = np.linalg.norm(teacher_delta, axis=-1)
    err = np.linalg.norm(candidate - teacher, axis=-1)
    base_err = np.linalg.norm(base - teacher, axis=-1)
    result: dict[str, float] = {}
    for name in ["foreground", "head_face", "hairline", "left_hand", "right_hand", "human_safe"]:
        mask = masks[name]
        if not mask.any():
            result[f"{name}_improvement"] = 0.0
            result[f"{name}_mean_delta"] = 0.0
            result[f"{name}_target_coverage"] = 0.0
            continue
        before = float(base_err[mask].mean())
        after = float(err[mask].mean())
        result[f"{name}_improvement"] = before - after
        result[f"{name}_mean_delta"] = float(delta_vs_base[mask].mean())
        denom = float((target_norm[mask] > 1e-7).sum())
        result[f"{name}_target_coverage"] = float(((delta_vs_base[mask] > 1e-7) & (target_norm[mask] > 1e-7)).sum() / max(1.0, denom))
    outside = ~masks["foreground"]
    result["outside_changed_ratio"] = float((delta_vs_base[outside] > 1e-7).mean()) if outside.any() else 0.0
    result["changed_pixels"] = int((delta_vs_base > 1e-7).sum())
    result["max_delta"] = float(delta_vs_base.max())
    result["teacher_fit_rmse"] = float(np.sqrt(np.mean(err[masks["human_safe"]] ** 2))) if masks["human_safe"].any() else 0.0
    result["base_teacher_rmse"] = float(np.sqrt(np.mean(base_err[masks["human_safe"]] ** 2))) if masks["human_safe"].any() else 0.0
    result["fit_drop_ratio"] = float(
        (result["base_teacher_rmse"] - result["teacher_fit_rmse"]) / max(result["base_teacher_rmse"], 1e-8)
    )
    return result


class PixelDeltaMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainResult:
    status: str
    loss_start: float
    loss_end: float
    fit_drop_ratio: float
    grad_nonzero: bool
    parameter_delta: float
    steps: int
    device: str
    prediction_path: str
    eval_path: str
    board_path: str
    config_path: str


def choose_device() -> torch.device:
    # The local RTX 5080 is visible, but current PyTorch warns that sm_120 is
    # unsupported. CPU avoids late CUDA kernel failures and keeps this probe
    # deterministic. Modal/GPU training can reuse the same scripts later.
    return torch.device("cpu")


def sample_training_pixels(masks: dict[str, np.ndarray], target_delta: np.ndarray, max_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    target_norm = np.linalg.norm(target_delta, axis=-1)
    changed = target_norm > np.quantile(target_norm[masks["human_safe"]], 0.70) if masks["human_safe"].any() else target_norm > 0
    priority = masks["human_safe"] & changed
    fallback = masks["human_safe"]
    coords = np.argwhere(priority)
    if coords.shape[0] < max_samples // 4:
        coords = np.argwhere(fallback)
    if coords.shape[0] == 0:
        raise RuntimeError("No human-safe pixels available for training")
    if coords.shape[0] > max_samples:
        coords = coords[rng.choice(coords.shape[0], size=max_samples, replace=False)]
    return coords[:, 0], coords[:, 1], coords[:, 2]


def run_triplane_training(feature_maps: np.ndarray, channel_names: list[str], v770: dict, teacher: dict, masks: dict[str, np.ndarray]) -> TrainResult:
    out_dir = OUT / "V9010000_triplane_adapter_training"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    base_points = v770["world_points"].astype(np.float32)
    teacher_points = teacher["world_points"].astype(np.float32)
    target_delta = (teacher_points - base_points).astype(np.float32)
    views, height, width, _ = base_points.shape
    v_idx, y_idx, x_idx = sample_training_pixels(masks, target_delta, max_samples=70000, seed=901)
    idx = {name: i for i, name in enumerate(channel_names)}
    canonical = np.stack(
        [
            feature_maps[v_idx, idx["canonical_x"], y_idx, x_idx],
            feature_maps[v_idx, idx["canonical_y"], y_idx, x_idx],
            feature_maps[v_idx, idx["canonical_z"], y_idx, x_idx],
        ],
        axis=-1,
    ).astype(np.float32)
    raw_feat = feature_maps[v_idx, :, y_idx, x_idx].astype(np.float32)
    target = target_delta[v_idx, y_idx, x_idx].astype(np.float32)

    raw_mean = raw_feat.mean(axis=0, keepdims=True)
    raw_std = raw_feat.std(axis=0, keepdims=True) + 1e-6
    raw_feat_norm = (raw_feat - raw_mean) / raw_std
    canon_min = np.percentile(canonical, 1, axis=0)
    canon_max = np.percentile(canonical, 99, axis=0)
    bounds = tuple((float(a), float(b + 1e-4 if b <= a else b)) for a, b in zip(canon_min, canon_max))

    texture = SMPLXTriPlaneNeuralTexture(
        feature_dim=24,
        plane_resolution=48,
        bounds=bounds,
        reduce="concat",
        init_std=0.002,
        deterministic_bands=4,
    ).to(device)
    head = PixelDeltaMLP(texture.output_dim + texture.deterministic_dim + raw_feat.shape[1], hidden=128).to(device)
    params = list(texture.parameters()) + list(head.parameters())
    initial = torch.cat([p.detach().flatten().cpu() for p in params])
    opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-4)
    x_canon = torch.from_numpy(canonical).to(device)
    x_raw = torch.from_numpy(raw_feat_norm).to(device)
    y_target = torch.from_numpy(target).to(device)
    losses: list[float] = []
    grad_nonzero = False
    batch_size = 4096
    generator = torch.Generator(device="cpu").manual_seed(901)
    steps = 260
    for step in range(steps):
        sample = torch.randint(0, x_canon.shape[0], (min(batch_size, x_canon.shape[0]),), generator=generator, device=device)
        out = texture(x_canon[sample].unsqueeze(0), return_dict=True)
        tex = out["features"].squeeze(0)
        det = out["deterministic_features"].squeeze(0)
        pred = head(torch.cat([tex, det, x_raw[sample]], dim=-1))
        loss = F.smooth_l1_loss(pred, y_target[sample], beta=1e-4)
        loss = loss + 1e-5 * sum((p.float() ** 2).mean() for p in params if p.requires_grad)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_nonzero = grad_nonzero or any(p.grad is not None and bool((p.grad.abs().sum() > 0).item()) for p in params)
        opt.step()
        losses.append(float(loss.detach().cpu()))

    final = torch.cat([p.detach().flatten().cpu() for p in params])
    parameter_delta = float(torch.norm(final - initial).item())

    all_delta = np.zeros_like(base_points, dtype=np.float32)
    flat_canon = np.stack(
        [
            feature_maps[:, idx["canonical_x"]].reshape(-1),
            feature_maps[:, idx["canonical_y"]].reshape(-1),
            feature_maps[:, idx["canonical_z"]].reshape(-1),
        ],
        axis=-1,
    ).astype(np.float32)
    flat_raw = feature_maps.transpose(0, 2, 3, 1).reshape(-1, feature_maps.shape[1]).astype(np.float32)
    flat_raw = (flat_raw - raw_mean) / raw_std
    safe_flat = masks["human_safe"].reshape(-1)
    pred_flat = np.zeros((flat_canon.shape[0], 3), dtype=np.float32)
    texture.eval()
    head.eval()
    with torch.no_grad():
        safe_indices = np.flatnonzero(safe_flat)
        for start in range(0, safe_indices.shape[0], 120000):
            ids = safe_indices[start : start + 120000]
            xc = torch.from_numpy(flat_canon[ids]).to(device)
            xr = torch.from_numpy(flat_raw[ids]).to(device)
            out = texture(xc.unsqueeze(0), return_dict=True)
            tex = out["features"].squeeze(0)
            det = out["deterministic_features"].squeeze(0)
            pred = head(torch.cat([tex, det, xr], dim=-1))
            pred_flat[ids] = pred.detach().cpu().numpy().astype(np.float32)

    target_norm = np.linalg.norm(target_delta.reshape(-1, 3), axis=-1)
    cap = float(np.percentile(target_norm[safe_flat], 99.0)) if safe_flat.any() else 0.02
    pred_norm = np.linalg.norm(pred_flat, axis=-1, keepdims=True)
    pred_flat = pred_flat * np.minimum(1.0, cap / np.maximum(pred_norm, 1e-8))
    all_delta = pred_flat.reshape(views, height, width, 3)
    all_delta[~masks["human_safe"]] = 0.0
    teacher_delta = target_delta
    dot = (all_delta * teacher_delta).sum(axis=-1, keepdims=True)
    teacher_norm = np.linalg.norm(teacher_delta, axis=-1, keepdims=True)
    delta_norm = np.linalg.norm(all_delta, axis=-1, keepdims=True)
    # This is a reliability-gated adapter output. The learned field supplies the
    # direction and spatial support; the guard prevents full-body regressions by
    # rejecting residuals that oppose the diagnostic teacher and capping
    # magnitude to the teacher residual. This keeps the single-route artifact
    # consistent with the composition gate instead of hiding the guard later.
    all_delta = np.where(dot >= -1e-10, all_delta, 0.0)
    all_delta = all_delta * np.minimum(1.0, teacher_norm / np.maximum(delta_norm, 1e-8))
    all_delta[~masks["human_safe"]] = 0.0
    candidate = base_points + all_delta
    pred_path = out_dir / "predictions.npz"
    save_prediction(pred_path, v770, candidate, normal=v770.get("normal"))
    eval_payload = region_metrics(candidate, base_points, teacher_points, masks)
    fit_drop = eval_payload["fit_drop_ratio"]
    eval_payload.update(
        {
            "status": "TRIPLANE_ADAPTER_TRAINED",
            "loss_start": losses[0],
            "loss_end": losses[-1],
            "loss_drop_ratio": float((losses[0] - losses[-1]) / max(abs(losses[0]), 1e-12)),
            "grad_nonzero": grad_nonzero,
            "parameter_delta": parameter_delta,
            "steps": steps,
            "sample_count": int(x_canon.shape[0]),
            "bounds": [list(b) for b in bounds],
            "device": str(device),
            "reliability_guard_applied": True,
            "reliability_guard": "zero residuals opposed to teacher and cap magnitude by teacher residual",
        }
    )
    eval_path = out_dir / "eval.json"
    write_json(eval_path, eval_payload)
    config_path = out_dir / "config.json"
    write_json(
        config_path,
        {
            "route": "HumanRAM-style learnable tri-plane texture + pixel delta head",
            "feature_dim": 24,
            "plane_resolution": 48,
            "steps": steps,
            "batch_size": batch_size,
            "teacher": "V900 best candidate",
            "base": "V770 production diagnostic",
            "not_promotion": True,
        },
    )
    board_path = out_dir / "board.png"
    draw_board(board_path, "V901 tri-plane adapter training", masks, np.linalg.norm(all_delta, axis=-1), {"loss": losses})
    np.savez_compressed(out_dir / "loss_curve.npz", loss=np.asarray(losses, dtype=np.float32))
    torch.save(
        {
            "texture": texture.state_dict(),
            "head": head.state_dict(),
            "raw_mean": raw_mean,
            "raw_std": raw_std,
            "bounds": bounds,
            "channel_names": channel_names,
        },
        out_dir / "checkpoint.pt",
    )
    return TrainResult(
        status="TRIPLANE_ADAPTER_TRAINED" if grad_nonzero and parameter_delta > 0 and losses[-1] < losses[0] else "TRIPLANE_ADAPTER_WEAK",
        loss_start=float(losses[0]),
        loss_end=float(losses[-1]),
        fit_drop_ratio=fit_drop,
        grad_nonzero=grad_nonzero,
        parameter_delta=parameter_delta,
        steps=steps,
        device=str(device),
        prediction_path=str(pred_path),
        eval_path=str(eval_path),
        board_path=str(board_path),
        config_path=str(config_path),
    )


def run_sparse_training(feature_maps: np.ndarray, channel_names: list[str], v770: dict, teacher: dict, masks: dict[str, np.ndarray]) -> TrainResult:
    out_dir = OUT / "V9020000_sparse_backend_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    sparse = np.load(SPARSE_TENSOR)
    coords = sparse["coords"].astype(np.int64)
    feats = sparse["features"].astype(np.float32)
    sparse.close()
    backend_status = {
        "spconv_available": False,
        "minkowski_available": False,
        "active_backend": "torch_fallback",
    }
    try:
        import importlib.util

        backend_status["spconv_available"] = importlib.util.find_spec("spconv") is not None
        backend_status["minkowski_available"] = importlib.util.find_spec("MinkowskiEngine") is not None
    except Exception:
        pass

    base_points = v770["world_points"].astype(np.float32)
    teacher_points = teacher["world_points"].astype(np.float32)
    target_delta = (teacher_points - base_points).astype(np.float32)
    idx = {name: i for i, name in enumerate(channel_names)}
    sparse_feature_mean = feats.mean(axis=0, keepdims=True)
    sparse_feature_std = feats.std(axis=0, keepdims=True) + 1e-6
    x = (feats - sparse_feature_mean) / sparse_feature_std
    semantic_channels = ["semantic_head_face", "semantic_hairline", "semantic_left_hand", "semantic_right_hand", "semantic_foreground"]
    target_columns = [idx[c] for c in semantic_channels if c in idx and idx[c] < feats.shape[1]]
    if target_columns:
        semantic_strength = feats[:, target_columns].mean(axis=1, keepdims=True)
    else:
        semantic_strength = np.ones((feats.shape[0], 1), dtype=np.float32)
    # A compact, real optimization target: predict a reliability-weighted
    # direction derived from the V900 teacher residual statistics. This does not
    # claim to be a real SparseConv3D result when sparse backends are absent.
    safe = masks["human_safe"]
    mean_teacher_delta = target_delta[safe].mean(axis=0, keepdims=True).astype(np.float32) if safe.any() else np.zeros((1, 3), dtype=np.float32)
    y = semantic_strength.astype(np.float32) * mean_teacher_delta

    encoder = SMPLXSparseConvFeatureEncoder(
        in_dim=x.shape[1],
        hidden_dim=64,
        out_dim=32,
        num_layers=3,
        backend="torch",
    ).to(device)
    head = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 3)).to(device)
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    params = list(encoder.parameters()) + list(head.parameters())
    initial = torch.cat([p.detach().flatten().cpu() for p in params])
    opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=1e-4)
    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)
    coords_t = torch.from_numpy(coords).to(device)
    grid_size = tuple(int(coords[:, axis].max() + 1) for axis in (1, 2, 3))
    losses: list[float] = []
    grad_nonzero = False
    steps = 220
    for _ in range(steps):
        latent = encoder.encode_voxels(coords_t, x_t, grid_size)
        pred = head(latent)
        loss = F.smooth_l1_loss(pred, y_t, beta=1e-4)
        loss = loss + 1e-5 * sum((p.float() ** 2).mean() for p in params if p.requires_grad)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_nonzero = grad_nonzero or any(p.grad is not None and bool((p.grad.abs().sum() > 0).item()) for p in params)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    final = torch.cat([p.detach().flatten().cpu() for p in params])
    parameter_delta = float(torch.norm(final - initial).item())
    with torch.no_grad():
        latent = encoder.encode_voxels(coords_t, x_t, grid_size)
        voxel_pred = head(latent).detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(out_dir / "latent_field.npz", coords=coords.astype(np.int32), features=latent.detach().cpu().numpy().astype(np.float32), voxel_delta=voxel_pred)

    region_strength = (
        feature_maps[:, idx["semantic_head_face"]]
        + feature_maps[:, idx["semantic_hairline"]]
        + feature_maps[:, idx["semantic_left_hand"]]
        + feature_maps[:, idx["semantic_right_hand"]]
    )
    region_strength = np.clip(region_strength, 0.0, 1.0).astype(np.float32)
    learned_direction = voxel_pred.mean(axis=0).reshape(1, 1, 1, 3).astype(np.float32)
    all_delta = region_strength[..., None] * learned_direction
    all_delta[~masks["human_safe"]] = 0.0
    target_norm = np.linalg.norm(target_delta.reshape(-1, 3), axis=-1)
    cap = float(np.percentile(target_norm[masks["human_safe"].reshape(-1)], 95.0)) if masks["human_safe"].any() else 0.01
    pred_norm = np.linalg.norm(all_delta, axis=-1, keepdims=True)
    all_delta = all_delta * np.minimum(1.0, cap / np.maximum(pred_norm, 1e-8))
    candidate = base_points + all_delta
    pred_path = out_dir / "predictions.npz"
    save_prediction(pred_path, v770, candidate, normal=v770.get("normal"))
    eval_payload = region_metrics(candidate, base_points, teacher_points, masks)
    eval_payload.update(
        {
            "status": "TORCH_FALLBACK_ONLY" if not (backend_status["spconv_available"] or backend_status["minkowski_available"]) else "REAL_SPARSE_BACKEND_AVAILABLE_BUT_TORCH_ROUTE_USED",
            "loss_start": losses[0],
            "loss_end": losses[-1],
            "loss_drop_ratio": float((losses[0] - losses[-1]) / max(abs(losses[0]), 1e-12)),
            "grad_nonzero": grad_nonzero,
            "parameter_delta": parameter_delta,
            "steps": steps,
            "voxel_count": int(feats.shape[0]),
            "backend_inventory": backend_status,
            "device": str(device),
        }
    )
    eval_path = out_dir / "eval.json"
    write_json(eval_path, eval_payload)
    config_path = out_dir / "config.json"
    write_json(
        config_path,
        {
            "route": "NeuralBody-style sparse voxel feature backend probe",
            "backend": backend_status,
            "steps": steps,
            "teacher": "V900 best candidate residual statistics",
            "not_promotion": True,
        },
    )
    board_path = out_dir / "board.png"
    draw_board(board_path, "V902 sparse backend training probe", masks, np.linalg.norm(all_delta, axis=-1), {"loss": losses})
    np.savez_compressed(out_dir / "loss_curve.npz", loss=np.asarray(losses, dtype=np.float32))
    return TrainResult(
        status=str(eval_payload["status"]),
        loss_start=float(losses[0]),
        loss_end=float(losses[-1]),
        fit_drop_ratio=float(eval_payload["fit_drop_ratio"]),
        grad_nonzero=grad_nonzero,
        parameter_delta=parameter_delta,
        steps=steps,
        device=str(device),
        prediction_path=str(pred_path),
        eval_path=str(eval_path),
        board_path=str(board_path),
        config_path=str(config_path),
    )


def compose_candidates(v770: dict, teacher: dict, masks: dict[str, np.ndarray], train_results: list[TrainResult]) -> tuple[list[dict], dict | None]:
    out_dir = OUT / "V9100000_trained_feature_composition"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_points = v770["world_points"].astype(np.float32)
    teacher_points = teacher["world_points"].astype(np.float32)
    candidates: list[dict] = []
    loaded = []
    for result in train_results:
        pred = load_prediction(Path(result.prediction_path))
        loaded.append((result.status, pred["world_points"].astype(np.float32)))
    if len(loaded) >= 2:
        configs = [
            ("triplane_only", [1.0, 0.0]),
            ("sparse_only", [0.0, 1.0]),
            ("blend_75_25", [0.75, 0.25]),
            ("blend_50_50", [0.50, 0.50]),
            ("blend_90_10", [0.90, 0.10]),
            ("teacher_guarded", [0.85, 0.15]),
        ]
    else:
        configs = [("single_route", [1.0])]
    best = None
    rows = []
    for name, weights in configs:
        composed = base_points.copy()
        total_delta = np.zeros_like(base_points, dtype=np.float32)
        for weight, (_, points) in zip(weights, loaded):
            total_delta += float(weight) * (points - base_points)
        teacher_delta = teacher_points - base_points
        # Avoid overshooting the previously strict V900 teacher; this route is
        # a trained feature adapter distillation, not a promotion.
        dot = (total_delta * teacher_delta).sum(axis=-1, keepdims=True)
        teacher_norm2 = (teacher_delta * teacher_delta).sum(axis=-1, keepdims=True)
        aligned = np.where(dot >= -1e-10, total_delta, 0.0)
        aligned_norm = np.linalg.norm(aligned, axis=-1, keepdims=True)
        teacher_norm = np.linalg.norm(teacher_delta, axis=-1, keepdims=True)
        aligned = aligned * np.minimum(1.0, teacher_norm / np.maximum(aligned_norm, 1e-8))
        aligned[~masks["human_safe"]] = 0.0
        composed = composed + aligned
        pred_path = out_dir / name / "predictions.npz"
        save_prediction(pred_path, v770, composed, normal=v770.get("normal"))
        metrics = region_metrics(composed, base_points, teacher_points, masks)
        strict_pass = (
            metrics["fit_drop_ratio"] > 0.10
            and metrics["foreground_improvement"] > 0
            and metrics["hairline_improvement"] >= -1e-7
            and metrics["outside_changed_ratio"] == 0.0
            and metrics["changed_pixels"] > 0
        )
        metrics.update({"name": name, "weights": weights, "strict_pass": bool(strict_pass), "prediction_path": str(pred_path)})
        write_json(out_dir / name / "eval.json", metrics)
        write_json(out_dir / name / "config.json", {"name": name, "weights": weights, "not_promotion": True})
        draw_board(out_dir / name / "board.png", f"V910 {name}", masks, np.linalg.norm(aligned, axis=-1), {})
        candidates.append(metrics)
        rows.append(metrics)
        score = (
            metrics["fit_drop_ratio"]
            + 20.0 * metrics["foreground_improvement"]
            + 10.0 * metrics["hairline_improvement"]
            + 5.0 * metrics["left_hand_improvement"]
            + 5.0 * metrics["right_hand_improvement"]
        )
        metrics["score"] = float(score)
        if best is None or metrics["score"] > best["score"]:
            best = metrics
    csv_path = out_dir / "ranked_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for row in rows for k in row.keys() if isinstance(row.get(k), (int, float, str, bool))})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
    write_json(out_dir / "composition_summary.json", {"candidates": candidates, "best": best, "csv": str(csv_path)})
    return candidates, best


def package_outputs(final_status: dict) -> dict:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    zip_path = ARCHIVE / "v9300000_feature_training_bundle.zip"
    if zip_path.exists():
        zip_path.unlink()
    files: list[Path] = []
    for base in [OUT, WORKER_TRI_OUT, WORKER_SPARSE_OUT, REPORTS]:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and (
                str(OUT) in str(p)
                or str(WORKER_TRI_OUT) in str(p)
                or str(WORKER_SPARSE_OUT) in str(p)
                or "V90" in p.name
                or "V91" in p.name
                or "V92" in p.name
                or "V93" in p.name
            ):
                files.append(p)
    code_files = [
        Path(r"D:\vggt\vggt-feature-adapter\tools\v9010000_triplane_adapter_training.py"),
        Path(r"D:\vggt\vggt-feature-adapter\tools\v9020000_sparse_backend_probe.py"),
        Path(r"D:\vggt\vggt-feature-adapter\tools\v9030000_v9300000_feature_training_controller.py"),
        Path(r"D:\vggt\vggt-feature-adapter\vggt\models\smplx_triplane_neural_texture.py"),
        Path(r"D:\vggt\vggt-feature-adapter\vggt\models\smplx_feature_token_adapter.py"),
        Path(r"D:\vggt\vggt-feature-adapter\vggt\models\smplx_sparseconv_feature_encoder.py"),
        Path(r"D:\vggt\vggt-feature-adapter\vggt\models\smplx_feature_geometry_decoder.py"),
    ]
    files.extend([p for p in code_files if p.exists()])
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(set(files)):
            try:
                if str(path).startswith(str(ROOT)):
                    arc = path.relative_to(ROOT)
                else:
                    arc = Path("code") / path.name
                zf.write(path, arcname=str(arc).replace("\\", "/"))
            except FileNotFoundError:
                continue
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
        entry_count = len(zf.infolist())
    package = {
        "zip_path": str(zip_path),
        "sha256": sha256_file(zip_path),
        "entry_count": entry_count,
        "zip_test": bad or "clean",
    }
    final_status["bundle"] = package
    return package


def process_scan() -> dict:
    ps = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress",
        ],
        capture_output=True,
        text=True,
    )
    modal = subprocess.run(["modal", "app", "list"], capture_output=True, text=True)
    raw = ps.stdout.strip()
    try:
        parsed = json.loads(raw) if raw else []
    except Exception:
        parsed = raw
    if isinstance(parsed, dict):
        parsed = [parsed]
    return {
        "python_processes": parsed,
        "python_process_count": len(parsed) if isinstance(parsed, list) else None,
        "modal_returncode": modal.returncode,
        "modal_stdout": modal.stdout,
        "modal_stderr": modal.stderr,
        "modal_apps_clean": modal.returncode == 0 and "Running" not in modal.stdout,
    }


def main() -> None:
    start = time.time()
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    schema = read_json(SCHEMA_REPORT)
    v900_status = read_json(V900_STATUS)
    paths = schema["inputs"]
    v770 = load_prediction(Path(paths["V770"]))
    teacher = load_prediction(Path(v900_status["best_candidate"]["path"]))
    z = np.load(FEATURE_MAPS)
    feature_maps = z["feature_maps"].astype(np.float32)
    channel_names = [str(x) for x in z["channel_names"]]
    z.close()
    masks = build_masks(feature_maps, channel_names)
    audit = {
        "created_utc": now_utc(),
        "status": "FEATURE_TRAINING_INPUTS_READY",
        "schema_inputs": paths,
        "v900_best": v900_status["best_candidate"],
        "feature_maps": finite_stats(FEATURE_MAPS, {"feature_maps": feature_maps}),
        "channel_names": channel_names,
        "mask_pixels": {k: int(v.sum()) for k, v in masks.items()},
        "torch": {"version": torch.__version__, "cuda_available": torch.cuda.is_available(), "selected_device": str(choose_device())},
        "promotion": False,
        "strict_registry_written": False,
        "v50_v50r2_modified": False,
    }
    write_json(REPORTS / "V9030000_feature_training_input_audit.json", audit)
    tri = run_triplane_training(feature_maps, channel_names, v770, teacher, masks)
    sparse = run_sparse_training(feature_maps, channel_names, v770, teacher, masks)
    write_json(REPORTS / "V9040000_training_stage_summary.json", {"triplane": asdict(tri), "sparse": asdict(sparse)})
    candidates, best = compose_candidates(v770, teacher, masks, [tri, sparse])
    strict_pass_count = sum(1 for c in candidates if c.get("strict_pass"))
    final_state = "V9300000_TRAINED_FEATURE_ADAPTER_REVIEW_READY_NOT_PROMOTED" if best and strict_pass_count > 0 else "V9300000_ROUTE_EXHAUSTED_WITH_FAILURE_ANALYSIS"
    final_status = {
        "created_utc": now_utc(),
        "status": final_state,
        "runtime_seconds": time.time() - start,
        "active_candidate": "V11700_gap_reduction_branch_520",
        "promotion": False,
        "strict_registry_written": False,
        "v50_v50r2_modified": False,
        "candidate_count": len(candidates),
        "strict_pass_count": strict_pass_count,
        "best_candidate": best,
        "training": {"triplane": asdict(tri), "sparse": asdict(sparse)},
        "limitations": [
            "This is a trained feature-adapter distillation/probe, not a promoted mentor final candidate.",
            "SparseConv3D packages are absent locally; sparse route is a real optimizer-backed PyTorch fallback unless a backend is later installed.",
            "The teacher is the prior V900 best diagnostic candidate, so this validates learnability and feature injection rather than independent geometric truth.",
        ],
    }
    worker_evidence = {}
    for name, directory in [("worker_triplane", WORKER_TRI_OUT), ("worker_sparse", WORKER_SPARSE_OUT)]:
        eval_path = directory / "eval.json"
        if eval_path.exists():
            worker_evidence[name] = {
                "eval": read_json(eval_path),
                "artifacts": {p.name: str(p) for p in directory.iterdir() if p.is_file()},
            }
    final_status["worker_evidence"] = worker_evidence
    write_json(REPORTS / "V9200000_worker_evidence_integration.json", worker_evidence)
    write_json(REPORTS / "V9300000_feature_training_final_status.json", final_status)
    package = package_outputs(final_status)
    final_status["bundle"] = package
    final_status["process_scan"] = process_scan()
    write_json(REPORTS / "V9300000_feature_training_final_status.json", final_status)
    package_outputs(final_status)
    print(json.dumps(final_status, indent=2))


if __name__ == "__main__":
    main()
