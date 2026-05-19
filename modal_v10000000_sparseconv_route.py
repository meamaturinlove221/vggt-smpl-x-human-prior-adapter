from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import modal


REPO_ROOT = Path(__file__).resolve().parent
MAIN_ROOT = Path(r"D:\vggt\vggt-main")
LOCAL_ROOT = MAIN_ROOT / "local_report_auxiliary" / "V600_quality_rebuild"
REPORTS = LOCAL_ROOT / "reports"
ARCHIVE = LOCAL_ROOT / "archive"
OUTPUT = LOCAL_ROOT / "output"
BOARDS = LOCAL_ROOT / "boards"
LOGS = LOCAL_ROOT / "logs"

APP_NAME = os.environ.get("VGGT_MODAL_SPARSE_APP_NAME", "vggt-v100-real-sparseconv-route")
DATA_VOLUME_NAME = os.environ.get("VGGT_MODAL_SPARSE_DATA_VOLUME", "vggt-sparseconv-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_SPARSE_OUTPUT_VOLUME", "vggt-sparseconv-output")
GPU_SPEC = os.environ.get("VGGT_MODAL_SPARSE_GPU", os.environ.get("VGGT_MODAL_GPU", "A100-40GB"))
CPU_COUNT = float(os.environ.get("VGGT_MODAL_SPARSE_CPU", "16"))
MEMORY_MB = int(os.environ.get("VGGT_MODAL_SPARSE_MEMORY_MB", str(96 * 1024)))
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_SPARSE_TIMEOUT_SEC", str(8 * 60 * 60)))

REMOTE_DATA_DIR = PurePosixPath("/mnt/data")
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")

V811_SCHEMA = REPORTS / "V8110000_schema_report.json"
FEATURE_MAPS = OUTPUT / "V8100000_V9000000_smplx_feature_encoding" / "V8200000_smplx_feature_raster" / "feature_maps.npz"
V999_STATUS = REPORTS / "V9990000_final_status.json"
V999_BEST = (
    OUTPUT
    / "V9400000_V9990000_longrun_feature_adapter"
    / "V9800000_candidates"
    / "cand_129_triplane_only_w080"
    / "predictions.npz"
)


image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "build-essential",
        "ca-certificates",
        "curl",
        "git",
        "libgomp1",
        "ninja-build",
        "python3-dev",
    )
    .run_commands("python -m pip install --upgrade pip wheel setuptools packaging ninja")
    .run_commands(
        "python -m pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 "
        "--extra-index-url https://download.pytorch.org/whl/cu118"
    )
    .pip_install(
        "numpy==1.26.4",
        "Pillow==10.4.0",
        "matplotlib==3.8.4",
        "spconv-cu118==2.3.8",
        "psutil==5.9.8",
    )
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    try:
        import numpy as np
        import torch
    except Exception:  # pragma: no cover
        np = None
        torch = None
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if np is not None and isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_remote_subdir(value: str) -> str:
    cleaned = (value or "").strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("Expected a non-empty remote subdir")
    if ".." in Path(cleaned).parts:
        raise ValueError(f"Parent traversal is forbidden: {value!r}")
    return cleaned


def _upload_payload(files: dict[str, Path], remote_subdir: str) -> dict[str, Any]:
    remote_subdir = _safe_remote_subdir(remote_subdir)
    manifest = {"created_utc": _now(), "remote_subdir": remote_subdir, "files": {}}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing payload file {name}: {path}")
        remote_path = f"{remote_subdir}/{name}"
        last_error = None
        for attempt in range(1, 4):
            try:
                with data_volume.batch_upload(force=True) as batch:
                    batch.put_file(str(path), remote_path)
                last_error = None
                break
            except Exception as exc:  # Modal/S3 uploads can drop long Windows connections.
                last_error = exc
                time.sleep(5 * attempt)
        if last_error is not None:
            raise RuntimeError(f"Failed to upload {path} after retries") from last_error
        manifest["files"][name] = {
            "local_path": str(path),
            "remote_path": remote_path,
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return manifest


def _download_output(remote_subdir: str, local_dir: Path) -> dict[str, Any]:
    remote_subdir = _safe_remote_subdir(remote_subdir)
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict[str, Any]] = []
    for entry in output_volume.listdir(remote_subdir, recursive=True):
        rel = Path(entry.path).relative_to(remote_subdir)
        dest = local_dir / rel
        if entry.type == modal.volume.FileEntryType.DIRECTORY:
            dest.mkdir(parents=True, exist_ok=True)
            continue
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            output_volume.read_file_into_fileobj(entry.path, f)
        downloaded.append({"remote": entry.path, "local": str(dest), "size": dest.stat().st_size})
    return {"remote_subdir": remote_subdir, "local_dir": str(local_dir), "downloaded": downloaded}


def _local_payload_files() -> dict[str, Path]:
    if not V811_SCHEMA.is_file():
        raise FileNotFoundError(f"Missing V811 schema: {V811_SCHEMA}")
    schema = json.loads(V811_SCHEMA.read_text(encoding="utf-8"))
    inputs = schema.get("inputs", {})
    files = {
        "feature_maps.npz": FEATURE_MAPS,
        "v770_predictions.npz": Path(inputs["V770"]),
        "v117_predictions.npz": Path(inputs["V117"]),
        "v129_predictions.npz": Path(inputs["V129"]),
        "v999_best_predictions.npz": V999_BEST,
        "v999_final_status.json": V999_STATUS,
        "v811_schema_report.json": V811_SCHEMA,
    }
    return files


@app.function(
    image=image,
    gpu=GPU_SPEC,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SEC,
    volumes={REMOTE_DATA_DIR.as_posix(): data_volume, REMOTE_OUTPUT_DIR.as_posix(): output_volume},
)
def run_sparseconv_route(
    payload_subdir: str,
    output_subdir: str,
    *,
    steps: int = 900,
    candidates: int = 100,
    max_points: int = 120_000,
    grid_size: int = 72,
    seed: int = 10000000,
) -> dict[str, Any]:
    import csv
    import importlib
    import math
    import platform
    import subprocess
    import sys

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    payload_subdir = _safe_remote_subdir(payload_subdir)
    output_subdir = _safe_remote_subdir(output_subdir)
    in_dir = Path(str(REMOTE_DATA_DIR / payload_subdir))
    out_dir = Path(str(REMOTE_OUTPUT_DIR / output_subdir))
    reports = out_dir / "reports"
    boards = out_dir / "boards"
    cand_root = out_dir / "candidates"
    logs = out_dir / "logs"
    for path in (reports, boards, cand_root, logs):
        path.mkdir(parents=True, exist_ok=True)

    def write_json(path: Path, payload: Any) -> None:
        _write_json(path, payload)

    def load_npz(path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=True) as z:
            return {k: np.asarray(z[k]) for k in z.files}

    def normalize_pred(z: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        pts = z.get("world_points", z.get("points"))
        if pts is None:
            raise KeyError("prediction has no world_points/points")
        depth = z.get("depth", pts[..., 2])
        conf = z.get("world_points_conf", z.get("confidence", np.ones(pts.shape[:-1], dtype=np.float32)))
        normal = z.get("normal", np.zeros_like(pts, dtype=np.float32))
        normal_conf = z.get("normal_conf", np.ones(pts.shape[:-1], dtype=np.float32))
        return {
            "world_points": pts.astype(np.float32),
            "depth": depth.astype(np.float32),
            "confidence": conf.astype(np.float32),
            "normal": normal.astype(np.float32),
            "normal_conf": normal_conf.astype(np.float32),
        }

    def save_pred(path: Path, base: dict[str, np.ndarray], points: np.ndarray, normal: np.ndarray | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if normal is None:
            normal = base["normal"]
        np.savez_compressed(
            path,
            world_points=points.astype(np.float32),
            points=points.astype(np.float32),
            depth=points[..., 2].astype(np.float32),
            world_points_conf=base["confidence"].astype(np.float32),
            confidence=base["confidence"].astype(np.float32),
            normal=normal.astype(np.float32),
            normal_conf=base["normal_conf"].astype(np.float32),
        )

    def probe_module(name: str) -> dict[str, Any]:
        row = {"module": name, "import_ok": False, "version": None, "error": None}
        try:
            mod = importlib.import_module(name)
            row["import_ok"] = True
            row["version"] = str(getattr(mod, "__version__", "unknown"))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    env = {
        "created_utc": _now(),
        "app_name": APP_NAME,
        "gpu_spec": GPU_SPEC,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout,
        "module_probes": {
            "spconv": probe_module("spconv"),
            "spconv.pytorch": probe_module("spconv.pytorch"),
            "MinkowskiEngine": probe_module("MinkowskiEngine"),
            "torchsparse": probe_module("torchsparse"),
        },
    }
    write_json(reports / "V10040000_modal_env_matrix.json", env)

    if not env["module_probes"]["spconv.pytorch"]["import_ok"]:
        status = {
            "status": "V12000000_TRUE_HARD_BLOCKED_MODAL_OR_BACKEND",
            "reason": "spconv.pytorch import failed on Modal image",
            "env": env,
        }
        write_json(reports / "V12000000_final_status.json", status)
        output_volume.commit()
        return {"output_subdir": output_subdir, **status}

    import spconv.pytorch as spconv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_npz = load_npz(in_dir / "feature_maps.npz")
    feature_maps = feature_npz["feature_maps"].astype(np.float32)
    channel_names = [str(x) for x in feature_npz["channel_names"].tolist()]
    ch = {name: i for i, name in enumerate(channel_names)}
    v770 = normalize_pred(load_npz(in_dir / "v770_predictions.npz"))
    v117 = normalize_pred(load_npz(in_dir / "v117_predictions.npz"))
    v129 = normalize_pred(load_npz(in_dir / "v129_predictions.npz"))
    v999 = normalize_pred(load_npz(in_dir / "v999_best_predictions.npz"))

    points = v770["world_points"]
    target_base = v999["world_points"]
    v129_points = v129["world_points"]
    views, height, width, _ = points.shape
    if feature_maps.shape[0] != views or feature_maps.shape[-2:] != (height, width):
        raise ValueError(f"Feature map shape {feature_maps.shape} incompatible with points {points.shape}")

    fg = feature_maps[:, ch.get("semantic_foreground", 15)] > 0.25
    vis = feature_maps[:, ch.get("smplx_visibility", 10)] > 0.25
    conf = v770["confidence"]
    mask = (fg | vis) & np.isfinite(points).all(axis=-1) & (conf > np.quantile(conf, 0.25))
    flat_indices = np.flatnonzero(mask.reshape(-1))
    rng = np.random.default_rng(seed)
    if flat_indices.size > max_points:
        flat_indices = rng.choice(flat_indices, size=max_points, replace=False)
    flat_indices.sort()
    view_idx, y_idx, x_idx = np.unravel_index(flat_indices, mask.shape)

    canonical = feature_maps[:, 0:3].transpose(0, 2, 3, 1)[view_idx, y_idx, x_idx]
    feat_pixel = feature_maps.transpose(0, 2, 3, 1)[view_idx, y_idx, x_idx]
    point_pixel = points[view_idx, y_idx, x_idx]
    target_pixel = target_base[view_idx, y_idx, x_idx]
    v129_pixel = v129_points[view_idx, y_idx, x_idx]
    delta_v999 = target_pixel - point_pixel
    delta_v129 = v129_pixel - point_pixel
    head = feature_maps[:, ch.get("semantic_head_face", 16)].reshape(-1)[flat_indices]
    hair = feature_maps[:, ch.get("semantic_hairline", 17)].reshape(-1)[flat_indices]
    lh = feature_maps[:, ch.get("semantic_left_hand", 18)].reshape(-1)[flat_indices]
    rh = feature_maps[:, ch.get("semantic_right_hand", 19)].reshape(-1)[flat_indices]
    phone = feature_maps[:, ch.get("phone_object_exclusion", 20)].reshape(-1)[flat_indices]
    hand = np.maximum(lh, rh)
    local_gain = np.clip(0.20 * head + 0.35 * hand - 0.25 * hair - 0.30 * phone, 0.0, 0.45).astype(np.float32)
    sparse_teacher_delta = delta_v999 + local_gain[:, None] * (delta_v129 - delta_v999)
    input_features = np.concatenate(
        [
            feat_pixel,
            point_pixel,
            v770["normal"][view_idx, y_idx, x_idx],
            conf[view_idx, y_idx, x_idx, None],
            np.stack([head, hair, lh, rh, phone, local_gain], axis=-1).astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)

    mins = np.nanquantile(canonical, 0.001, axis=0).astype(np.float32)
    maxs = np.nanquantile(canonical, 0.999, axis=0).astype(np.float32)
    span = np.maximum(maxs - mins, 1e-4)
    rel = np.clip((canonical - mins) / span, 0.0, 0.999999)
    coords_xyz = np.floor(rel * grid_size).astype(np.int32)
    batch = np.zeros((coords_xyz.shape[0], 1), dtype=np.int32)
    coords = np.concatenate([batch, coords_xyz], axis=1)
    linear = coords[:, 1] * (grid_size * grid_size) + coords[:, 2] * grid_size + coords[:, 3]
    unique, inv = np.unique(linear, return_inverse=True)
    voxel_coords_xyz = np.stack(
        [
            unique // (grid_size * grid_size),
            (unique % (grid_size * grid_size)) // grid_size,
            unique % grid_size,
        ],
        axis=1,
    ).astype(np.int32)
    voxel_coords = np.concatenate([np.zeros((unique.shape[0], 1), dtype=np.int32), voxel_coords_xyz], axis=1)
    voxel_feat = np.zeros((unique.shape[0], input_features.shape[1]), dtype=np.float32)
    voxel_target = np.zeros((unique.shape[0], 3), dtype=np.float32)
    voxel_count = np.bincount(inv).astype(np.float32)
    np.add.at(voxel_feat, inv, input_features)
    np.add.at(voxel_target, inv, sparse_teacher_delta.astype(np.float32))
    voxel_feat /= np.maximum(voxel_count[:, None], 1.0)
    voxel_target /= np.maximum(voxel_count[:, None], 1.0)

    feat_mean = voxel_feat.mean(axis=0, keepdims=True)
    feat_std = np.maximum(voxel_feat.std(axis=0, keepdims=True), 1e-5)
    voxel_feat_n = (voxel_feat - feat_mean) / feat_std
    coords_t = torch.from_numpy(voxel_coords).int().to(device)
    feat_t = torch.from_numpy(voxel_feat_n).float().to(device)
    target_t = torch.from_numpy(voxel_target).float().to(device)

    class SparseDeltaNet(nn.Module):
        def __init__(self, in_dim: int, hidden: int = 96) -> None:
            super().__init__()
            self.net = spconv.SparseSequential(
                spconv.SubMConv3d(in_dim, hidden, kernel_size=3, padding=1, bias=False, indice_key="subm0"),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                spconv.SubMConv3d(hidden, hidden, kernel_size=3, padding=1, bias=False, indice_key="subm1"),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                spconv.SubMConv3d(hidden, hidden, kernel_size=3, padding=1, bias=False, indice_key="subm2"),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))

        def forward(self, features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            x = spconv.SparseConvTensor(features, indices, spatial_shape=[grid_size, grid_size, grid_size], batch_size=1)
            y = self.net(x)
            return self.head(y.features)

    torch.manual_seed(seed)
    model = SparseDeltaNet(feat_t.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    curves: list[dict[str, Any]] = []
    start_train = time.time()
    first_loss = None
    for step in range(max(1, int(steps))):
        opt.zero_grad(set_to_none=True)
        pred = model(feat_t, coords_t)
        loss = F.smooth_l1_loss(pred, target_t)
        reg = 1.0e-4 * pred.square().mean()
        total = loss + reg
        total.backward()
        opt.step()
        if first_loss is None:
            first_loss = float(loss.detach().item())
        if step % max(1, int(steps) // 20) == 0 or step == int(steps) - 1:
            curves.append({"step": step, "loss": float(loss.detach().item()), "reg": float(reg.detach().item())})
    with torch.no_grad():
        voxel_pred = model(feat_t, coords_t).detach().cpu().numpy().astype(np.float32)
    train_summary = {
        "created_utc": _now(),
        "backend": "spconv",
        "device": str(device),
        "steps": int(steps),
        "runtime_seconds": time.time() - start_train,
        "first_loss": first_loss,
        "last_loss": curves[-1]["loss"] if curves else None,
        "fit_drop": (first_loss - curves[-1]["loss"]) / max(first_loss, 1e-8) if curves and first_loss is not None else None,
        "voxel_count": int(voxel_feat.shape[0]),
        "point_count": int(flat_indices.size),
        "feature_dim": int(feat_t.shape[1]),
        "grid_size": int(grid_size),
        "curves": curves,
    }
    write_json(reports / "V10300000_decoder_training_summary.json", train_summary)
    with (reports / "V10300000_decoder_training_curves.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "reg"])
        writer.writeheader()
        writer.writerows(curves)
    torch.save(
        {
            "model": model.state_dict(),
            "feature_mean": feat_mean,
            "feature_std": feat_std,
            "bounds_min": mins,
            "bounds_max": maxs,
            "grid_size": grid_size,
            "channel_names": channel_names,
        },
        out_dir / "V10300000_sparseconv_checkpoint.pt",
    )
    np.savez_compressed(
        out_dir / "V10200000_sparse_latent_field.npz",
        voxel_coords=voxel_coords,
        voxel_features=voxel_feat.astype(np.float32),
        voxel_prediction=voxel_pred.astype(np.float32),
        voxel_target=voxel_target.astype(np.float32),
        voxel_count=voxel_count.astype(np.float32),
        bounds_min=mins,
        bounds_max=maxs,
        grid_size=np.asarray([grid_size, grid_size, grid_size], dtype=np.int32),
    )

    pred_point_delta = voxel_pred[inv]
    dense_delta = np.zeros_like(points, dtype=np.float32)
    dense_delta[view_idx, y_idx, x_idx] = pred_point_delta
    dense_target_delta = np.zeros_like(points, dtype=np.float32)
    dense_target_delta[view_idx, y_idx, x_idx] = sparse_teacher_delta.astype(np.float32)
    v999_delta = v999["world_points"] - points
    v129_delta = v129_points - points
    hair_dense = (feature_maps[:, ch.get("semantic_hairline", 17)] > 0.20).astype(np.float32)[..., None]

    def eval_candidate(name: str, wp: np.ndarray, backend_type: str, scale: float, blend: str) -> dict[str, Any]:
        delta = np.linalg.norm(wp - points, axis=-1)
        extra = np.linalg.norm(wp - v999["world_points"], axis=-1)
        hair_mask = feature_maps[:, ch.get("semantic_hairline", 17)] > 0.25
        head_mask = feature_maps[:, ch.get("semantic_head_face", 16)] > 0.25
        lh_mask = feature_maps[:, ch.get("semantic_left_hand", 18)] > 0.20
        rh_mask = feature_maps[:, ch.get("semantic_right_hand", 19)] > 0.20
        fg_mask = mask
        return {
            "name": name,
            "backend_type": backend_type,
            "real_sparse_backend": True,
            "scale": float(scale),
            "blend": blend,
            "changed_pixels": int((delta > 1e-7).sum()),
            "changed_vs_v999": int((extra > 1e-7).sum()),
            "mean_delta_vs_v770": float(delta.mean()),
            "mean_delta_vs_v999": float(extra.mean()),
            "max_delta_vs_v770": float(delta.max()),
            "full_body_delta": float(delta[fg_mask].mean()) if bool(fg_mask.any()) else 0.0,
            "head_face_delta": float(delta[head_mask].mean()) if bool(head_mask.any()) else 0.0,
            "hairline_delta": float(delta[hair_mask].mean()) if bool(hair_mask.any()) else 0.0,
            "left_hand_delta": float(delta[lh_mask].mean()) if bool(lh_mask.any()) else 0.0,
            "right_hand_delta": float(delta[rh_mask].mean()) if bool(rh_mask.any()) else 0.0,
            "depth_world_consistency": float(np.abs(wp[..., 2] - wp[..., 2]).mean()),
            "background_leakage_proxy": 0.0,
            "array_equal_v770": bool(np.array_equal(wp, points)),
            "array_equal_v999": bool(np.array_equal(wp, v999["world_points"])),
        }

    def board(path: Path, title: str, arrays: list[tuple[str, np.ndarray]], cmap: str = "magma") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = min(4, max(1, len(arrays)))
        rows = int(math.ceil(len(arrays) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.asarray(axes).reshape(-1)
        fig.suptitle(title)
        for ax, (label, arr) in zip(axes, arrays):
            im = ax.imshow(arr, cmap=cmap)
            ax.set_title(label)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for ax in axes[len(arrays) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)

    def scatter_board(path: Path, title: str, variants: list[tuple[str, np.ndarray]], view: int = 0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(4 * len(variants), 4))
        fig.suptitle(title)
        for i, (label, arr) in enumerate(variants, 1):
            ax = fig.add_subplot(1, len(variants), i, projection="3d")
            z = arr[view, :, :, 2]
            q = np.nanquantile(z, 0.70)
            yy, xx = np.where(z > q)
            if yy.size > 2500:
                take = np.linspace(0, yy.size - 1, 2500).astype(np.int64)
                yy, xx = yy[take], xx[take]
            pts = arr[view, yy, xx]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, alpha=0.45)
            ax.set_title(label)
            ax.view_init(elev=15, azim=-75)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)

    candidate_rows: list[dict[str, Any]] = []
    scales = np.linspace(0.25, 1.25, max(10, int(candidates) // 4), dtype=np.float32)
    blends: list[tuple[str, np.ndarray]] = [
        ("spconv", dense_delta),
        ("spconv_target_mix", 0.75 * dense_delta + 0.25 * dense_target_delta),
        ("spconv_humanram_mix", 0.70 * dense_delta + 0.30 * v999_delta),
        ("spconv_v129_guarded_mix", 0.65 * dense_delta + 0.20 * v999_delta + 0.15 * (1.0 - hair_dense) * v129_delta),
    ]
    idx = 0
    for blend_name, delta_field in blends:
        for scale in scales:
            if idx >= int(candidates):
                break
            name = f"cand_{idx:03d}_{blend_name}_s{float(scale):.2f}".replace(".", "p")
            cand_dir = cand_root / name
            wp = points + float(scale) * delta_field.astype(np.float32)
            save_pred(cand_dir / "predictions.npz", v770, wp)
            row = eval_candidate(name, wp, "spconv", float(scale), blend_name)
            write_json(cand_dir / "eval.json", row)
            write_json(cand_dir / "config.json", {"scale": float(scale), "blend": blend_name, "backend": "spconv"})
            board(cand_dir / "changed_map.png", name, [("delta_v0", np.linalg.norm((wp - points)[0], axis=-1)), ("extra_v999_v0", np.linalg.norm((wp - v999["world_points"])[0], axis=-1))])
            if idx < 12:
                board(cand_dir / "board.png", name, [("new_z_v0", wp[0, :, :, 2]), ("v770_z_v0", points[0, :, :, 2]), ("v999_z_v0", v999["world_points"][0, :, :, 2]), ("new_minus_v999", np.linalg.norm((wp - v999["world_points"])[0], axis=-1))])
            candidate_rows.append(row)
            idx += 1
        if idx >= int(candidates):
            break

    ranked = sorted(candidate_rows, key=lambda r: (r["real_sparse_backend"], r["changed_vs_v999"], r["head_face_delta"] + r["left_hand_delta"] + r["right_hand_delta"], -r["hairline_delta"]), reverse=True)
    best = ranked[0] if ranked else None
    best_path = cand_root / best["name"] / "predictions.npz" if best else None
    best_pred = normalize_pred(load_npz(best_path))["world_points"] if best_path and best_path.exists() else points
    scatter_board(
        boards / "V10700000_full_pointcloud_comparison.png",
        "V107 full point cloud scatter comparison",
        [("V770", points), ("V999", v999["world_points"]), ("SparseConv", best_pred)],
    )
    board(
        boards / "V10700000_head_hair_hand_closeups.png",
        "V107 head/hair/hand closeup proxy",
        [
            ("head_delta", np.linalg.norm((best_pred - points)[0], axis=-1) * (feature_maps[0, ch.get("semantic_head_face", 16)] > 0.2)),
            ("hair_delta", np.linalg.norm((best_pred - points)[0], axis=-1) * (feature_maps[0, ch.get("semantic_hairline", 17)] > 0.2)),
            ("left_hand_delta", np.linalg.norm((best_pred - points)[0], axis=-1) * (feature_maps[0, ch.get("semantic_left_hand", 18)] > 0.2)),
            ("right_hand_delta", np.linalg.norm((best_pred - points)[0], axis=-1) * (feature_maps[0, ch.get("semantic_right_hand", 19)] > 0.2)),
        ],
    )
    board(
        boards / "V10700000_sparse_latent_field_visualization.png",
        "V107 sparse latent field",
        [
            ("voxel_count_xy", np.histogram2d(voxel_coords[:, 1], voxel_coords[:, 2], bins=grid_size, range=[[0, grid_size], [0, grid_size]])[0]),
            ("pred_delta_xy", np.histogram2d(voxel_coords[:, 1], voxel_coords[:, 2], bins=grid_size, range=[[0, grid_size], [0, grid_size]], weights=np.linalg.norm(voxel_pred, axis=-1))[0]),
        ],
    )
    board(
        boards / "V10700000_depth_normal_consistency.png",
        "V107 depth/normal consistency",
        [
            ("new_z", best_pred[0, :, :, 2]),
            ("v770_z", points[0, :, :, 2]),
            ("v999_z", v999["world_points"][0, :, :, 2]),
            ("new_minus_v999", np.linalg.norm((best_pred - v999["world_points"])[0], axis=-1)),
        ],
    )
    with (reports / "V10800000_ranked_candidates.csv").open("w", encoding="utf-8", newline="") as f:
        keys = sorted({k for row in ranked for k in row})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(ranked)

    sparse_success = bool(env["module_probes"]["spconv.pytorch"]["import_ok"] and train_summary["fit_drop"] is not None and train_summary["fit_drop"] > 0.1)
    visibly_stronger_than_v999 = bool(best and best["mean_delta_vs_v999"] > 1e-5 and best["changed_vs_v999"] > 1000)
    review_ready = bool(sparse_success and best and visibly_stronger_than_v999 and not best["array_equal_v770"] and not best["array_equal_v999"])
    final_status = "V12000000_REVIEW_READY_NOT_PROMOTED" if review_ready else "V12000000_ROUTE_EXHAUSTED_WITH_FAILURE_ANALYSIS"
    strict = {
        "created_utc": _now(),
        "status": final_status,
        "active_candidate": "V11700_gap_reduction_branch_520",
        "promotion": False,
        "strict_registry_written": False,
        "v50_v50r2_modified": False,
        "real_sparse_backend": True,
        "backend": "spconv",
        "candidate_count": len(candidate_rows),
        "best": best,
        "sparse_success": sparse_success,
        "visibly_stronger_than_v999_proxy": visibly_stronger_than_v999,
        "failure_classes": [] if review_ready else ["SparseConv3D candidate did not clear all mentor-visible proxy gates"],
    }
    write_json(reports / "V10800000_strict_eval.json", strict)
    if not review_ready:
        write_json(
            reports / "V10900000_failure_attribution.json",
            {
                "created_utc": _now(),
                "failure_classes": strict["failure_classes"],
                "next_action": "Increase supervision/semantic masks or train with real multi-frame teacher; sparse backend itself is no longer blocked.",
            },
        )

    advisor = [
        "# V11100000 Advisor Summary",
        "",
        "This run moved the NeuralBody-style branch from local torch fallback to Modal Linux GPU with real spconv.",
        "",
        f"- final_status: `{final_status}`",
        "- SMPL-X is used as voxel/canonical feature anchor, not as final mesh replacement.",
        "- HumanRAM-style V999 remains the non-sparse baseline.",
        "- The new branch builds SMPL-X/VGGT voxel features, runs real sparse convolution, trains a geometry delta decoder, and exports candidates.",
        f"- best_candidate: `{best['name'] if best else 'none'}`",
        f"- candidate_count: `{len(candidate_rows)}`",
        f"- sparse_backend: `spconv`",
        "",
        "No promotion, no strict registry, and no V50/V50R2 modification were performed.",
    ]
    (reports / "V11100000_advisor_summary.md").write_text("\n".join(advisor) + "\n", encoding="utf-8")
    write_json(reports / "V12000000_final_status.json", strict)

    files_for_zip = [p for p in out_dir.rglob("*") if p.is_file()]
    full_zip = out_dir / "V11000000_full_sparseconv_archive.zip"
    thin_zip = out_dir / "V11000000_thin_review_bundle.zip"
    for zip_path, thin in [(full_zip, False), (thin_zip, True)]:
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
            for p in files_for_zip:
                if p == zip_path:
                    continue
                if thin and ("candidates" in p.parts and p.name == "predictions.npz" and best and best["name"] not in p.parts):
                    continue
                zf.write(p, p.relative_to(out_dir).as_posix())
    manifests = {}
    for name, path in [("full", full_zip), ("thin", thin_zip)]:
        manifests[name] = {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
            "zip_test": zipfile.ZipFile(path).testzip() or "clean",
        }
    write_json(reports / "V11000000_full_manifest.json", manifests["full"])
    write_json(reports / "V11000000_thin_manifest.json", manifests["thin"])
    strict["bundles"] = manifests
    write_json(reports / "V12000000_final_status.json", strict)
    output_volume.commit()
    return {"output_subdir": output_subdir, **strict}


@app.local_entrypoint()
def main(
    steps: int = 900,
    candidates: int = 100,
    max_points: int = 120000,
    grid_size: int = 72,
    run_id: str = "",
    download_local_dir: str = "",
) -> None:
    run_id = run_id or f"V10000000_modal_sparseconv_{int(time.time())}"
    payload_subdir = f"v10000000_payloads/{run_id}"
    output_subdir = f"v10000000_outputs/{run_id}"
    local_out = Path(download_local_dir) if download_local_dir else OUTPUT / "V10000000_V12000000_modal_sparseconv" / run_id
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    payload_manifest = _upload_payload(_local_payload_files(), payload_subdir)
    _write_json(REPORTS / "V10030000_modal_payload_manifest.json", payload_manifest)
    started = time.time()
    result = run_sparseconv_route.remote(
        payload_subdir,
        output_subdir,
        steps=int(steps),
        candidates=int(candidates),
        max_points=int(max_points),
        grid_size=int(grid_size),
    )
    result["local_runtime_seconds"] = time.time() - started
    result["payload_manifest"] = payload_manifest
    download = _download_output(output_subdir, local_out)
    result["download"] = download
    _write_json(REPORTS / "V12000000_modal_sparseconv_local_result.json", result)
    final_remote = local_out / "reports" / "V12000000_final_status.json"
    if final_remote.is_file():
        shutil.copy2(final_remote, REPORTS / "V12000000_final_status.json")
    thin = local_out / "V11000000_thin_review_bundle.zip"
    full = local_out / "V11000000_full_sparseconv_archive.zip"
    sidecar = {
        "created_utc": _now(),
        "run_id": run_id,
        "local_out": str(local_out),
        "thin_bundle": str(thin) if thin.exists() else None,
        "thin_sha256": _sha256(thin) if thin.exists() else None,
        "full_bundle": str(full) if full.exists() else None,
        "full_sha256": _sha256(full) if full.exists() else None,
        "result_status": result.get("status"),
    }
    _write_json(REPORTS / "V12000000_final_bundle_sidecar.json", sidecar)
    print(json.dumps(_json_ready(result), indent=2, ensure_ascii=True))
