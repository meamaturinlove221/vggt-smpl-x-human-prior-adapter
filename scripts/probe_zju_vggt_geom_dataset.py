import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"

for root in (str(REPO_ROOT), str(TRAINING_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from training.data.datasets.zju_vggt_geom import ZjuVggtGeomDataset
from training.loss import (
    _build_human_prior_region_components,
    _build_human_prior_target_mask,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Probe the ZJU VGGT-geom pseudo-supervision dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional training config name or path. When set, inherit ZJU dataset/source-policy settings from it.",
    )
    parser.add_argument(
        "--zju_dir",
        type=str,
        default="",
        help="Optional local ZJU root. Auto-detected when omitted.",
    )
    parser.add_argument("--seq_names", nargs="+", default=["CoreView_390"])
    parser.add_argument("--geom_subdir", type=str, default="vggt_geom")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument(
        "--aggregate_samples",
        type=int,
        default=1,
        help="Number of deterministic sample indices to probe starting from --sample_index.",
    )
    parser.add_argument(
        "--aggregate_stride",
        type=int,
        default=1,
        help="Stride between probed sample indices when --aggregate_samples > 1.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed both random and numpy for reproducible source-view sampling.")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--camera_source", type=str, default="gt", choices=["gt", "geom"])
    parser.add_argument("--mask_source", type=str, default="mask", choices=["none", "mask", "mask_cihp"])
    parser.add_argument(
        "--source_policy",
        type=str,
        default="random",
        choices=["random", "nearest_ring", "uniform_ring", "nearest_plus_uniform_tail"],
    )
    parser.add_argument("--source_view_pool", type=str, default="cached_only", choices=["cached_only", "geom_plus_raw"])
    parser.add_argument("--source_view_pool_train_probability", type=float, default=1.0)
    parser.add_argument("--source_anchor_policy", type=str, default="random", choices=["random", "max_depth_conf"])
    parser.add_argument("--min_supervised_views", type=int, default=1)
    parser.add_argument(
        "--supervised_view_quality_filter",
        type=str,
        default="none",
        choices=["none", "drop_worst_by_depth_conf_if_multi_supervised"],
    )
    parser.add_argument(
        "--conf_depth_view_quality_filter",
        type=str,
        default="none",
        choices=["none", "drop_worst_by_depth_conf_if_multi_supervised"],
    )
    parser.add_argument("--allow_duplicate_img", action="store_true", help="Allow duplicate view sampling in the probe.")
    parser.add_argument("--min_depth_conf", type=float, default=0.0)
    parser.add_argument("--holdout_stride", type=int, default=10)
    parser.add_argument("--smpl_prior_pose_noise_prob", type=float, default=0.0)
    parser.add_argument("--smpl_prior_pose_noise_rot_deg", type=float, default=0.0)
    parser.add_argument("--smpl_prior_pose_noise_trans_scale", type=float, default=0.0)
    parser.add_argument("--smpl_prior_pose_noise_scale_std", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="output/zju_vggt_geom_probe")
    return parser.parse_args()


def resolve_config_path(config_ref: str) -> Path:
    raw = str(config_ref or "").strip()
    if not raw:
        raise ValueError("config_ref must be non-empty")
    candidate = Path(raw)
    if candidate.is_file():
        return candidate.resolve()
    config_dir = TRAINING_ROOT / "config"
    named = config_dir / raw
    if named.is_file():
        return named.resolve()
    if not named.suffix:
        named_yaml = named.with_suffix(".yaml")
        if named_yaml.is_file():
            return named_yaml.resolve()
    raise FileNotFoundError(f"Unable to resolve config '{config_ref}' from {config_dir}")


def load_config_with_defaults(config_ref: str):
    config_dir = TRAINING_ROOT / "config"

    def _resolve_default_path(base_dir: Path, item) -> Path:
        if isinstance(item, str):
            rel = Path(item)
        elif isinstance(item, dict):
            key, value = next(iter(item.items()))
            rel = Path(str(key)) / str(value)
        else:
            raise TypeError(f"Unsupported defaults item: {item!r}")
        candidate = base_dir / rel
        if candidate.is_file():
            return candidate.resolve()
        if not candidate.suffix:
            candidate_yaml = candidate.with_suffix(".yaml")
            if candidate_yaml.is_file():
                return candidate_yaml.resolve()
        candidate = config_dir / rel
        if candidate.is_file():
            return candidate.resolve()
        if not candidate.suffix:
            candidate_yaml = candidate.with_suffix(".yaml")
            if candidate_yaml.is_file():
                return candidate_yaml.resolve()
        raise FileNotFoundError(f"Unable to resolve defaults item {item!r} from {base_dir}")

    def _load(path: Path, seen):
        resolved = path.resolve()
        if resolved in seen:
            raise RuntimeError(f"Recursive config defaults detected at {resolved}")
        seen = set(seen)
        seen.add(resolved)
        current = OmegaConf.load(resolved)
        defaults = list(current.get("defaults", []))
        merged = OmegaConf.create()
        include_self = False
        for item in defaults:
            if item == "_self_":
                include_self = True
                continue
            if isinstance(item, dict) and "_self_" in item:
                include_self = True
                continue
            child_path = _resolve_default_path(resolved.parent, item)
            merged = OmegaConf.merge(merged, _load(child_path, seen))
        current_without_defaults = OmegaConf.create({k: v for k, v in current.items() if k != "defaults"})
        if include_self or not defaults:
            merged = OmegaConf.merge(merged, current_without_defaults)
        return merged

    config_path = resolve_config_path(config_ref)
    return config_path, _load(config_path, set())


def apply_config_defaults(args):
    config_ref = str(getattr(args, "config", "") or "").strip()
    if not config_ref:
        return None, None

    config_path, cfg = load_config_with_defaults(config_ref)

    def _resolved(key, default=None):
        if key not in cfg:
            return default
        value = cfg[key]
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        return value

    field_map = {
        "zju_source_policy": "source_policy",
        "zju_source_view_pool": "source_view_pool",
        "zju_source_view_pool_train_probability": "source_view_pool_train_probability",
        "zju_source_anchor_policy": "source_anchor_policy",
        "zju_min_supervised_views": "min_supervised_views",
        "zju_supervised_view_quality_filter": "supervised_view_quality_filter",
        "zju_conf_depth_view_quality_filter": "conf_depth_view_quality_filter",
        "zju_min_depth_conf": "min_depth_conf",
        "zju_holdout_stride": "holdout_stride",
        "zju_smpl_prior_pose_noise_prob": "smpl_prior_pose_noise_prob",
        "zju_smpl_prior_pose_noise_rot_deg": "smpl_prior_pose_noise_rot_deg",
        "zju_smpl_prior_pose_noise_trans_scale": "smpl_prior_pose_noise_trans_scale",
        "zju_smpl_prior_pose_noise_scale_std": "smpl_prior_pose_noise_scale_std",
        "zju_geom_subdir": "geom_subdir",
        "zju_camera_source": "camera_source",
        "zju_mask_source": "mask_source",
    }
    for config_key, arg_key in field_map.items():
        value = _resolved(config_key, None)
        if value is not None:
            setattr(args, arg_key, value)

    seq_names = _resolved("zju_seq_names", None)
    if seq_names is not None:
        if isinstance(seq_names, str):
            args.seq_names = [item.strip() for item in seq_names.split(",") if item.strip()]
        else:
            args.seq_names = [str(item) for item in seq_names]

    zju_dir = _resolved("zju_dir", None)
    if zju_dir not in (None, "", "/YOUR/PATH/TO/ZJU") and not str(getattr(args, "zju_dir", "") or "").strip():
        args.zju_dir = str(zju_dir)

    return config_path, cfg


def resolve_zju_dir(requested, seq_names, geom_subdir):
    def _normalize_candidate(candidate):
        try:
            return candidate.resolve()
        except OSError:
            return candidate.absolute()

    def _score_candidate(candidate):
        if not candidate.is_dir():
            return None
        valid_subdir_count = 0
        total_frame_count = 0
        for seq_name in seq_names:
            for geom_subdir_name in geom_subdirs:
                geom_dir = candidate / str(seq_name) / geom_subdir_name
                if not geom_dir.is_dir():
                    continue
                frame_count = sum(1 for _ in geom_dir.glob("frame_*.npz"))
                if frame_count > 0:
                    valid_subdir_count += 1
                    total_frame_count += frame_count
        if valid_subdir_count <= 0:
            return None
        return int(valid_subdir_count), int(total_frame_count)

    candidates = []
    requested = str(requested).strip()
    geom_subdirs = [item.strip() for item in str(geom_subdir).split(",") if item.strip()]
    if requested:
        explicit_candidate = Path(requested).absolute()
        explicit_score = _score_candidate(explicit_candidate)
        if explicit_score is not None:
            return explicit_candidate
        candidates.append(explicit_candidate)
    g_datasets = "G:\\" + chr(0x6570) + chr(0x636E) + chr(0x96C6)
    candidates.extend(
        [
            Path(r"F:\datasets\ZJU_MoCap\data\zju_mocap"),
            Path(g_datasets) / "datasets" / "ZJU_MoCap" / "data" / "zju_mocap",
        ]
    )

    seen = set()
    best_candidate = None
    best_score = None
    for candidate in candidates:
        candidate = _normalize_candidate(candidate)
        candidate_key = str(candidate).lower()
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        score = _score_candidate(candidate)
        if score is None:
            continue
        if best_candidate is None or score > best_score:
            best_candidate = candidate
            best_score = score
    if best_candidate is not None:
        return best_candidate
    raise FileNotFoundError(
        f"Unable to resolve a ZJU root with {seq_names[0]}/{geom_subdir}/frame_*.npz from candidates: {candidates}"
    )


def write_binary_ply(path, points, colors):
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    vertex_data = np.empty(
        points.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex_data["x"] = points[:, 0]
    vertex_data["y"] = points[:, 1]
    vertex_data["z"] = points[:, 2]
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        vertex_data.tofile(handle)


def normalize_extrinsics_matrices(extrinsics):
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    if extrinsics.ndim != 3:
        raise ValueError(f"Expected extrinsics with shape [N,3,4] or [N,4,4], got {extrinsics.shape}")
    if extrinsics.shape[1:] == (4, 4):
        return extrinsics
    if extrinsics.shape[1:] == (3, 4):
        padded = np.tile(np.eye(4, dtype=np.float32), (extrinsics.shape[0], 1, 1))
        padded[:, :3, :4] = extrinsics
        return padded
    raise ValueError(f"Unsupported extrinsics shape: {extrinsics.shape}")


def closed_form_inverse_se3_numpy(se3):
    se3 = normalize_extrinsics_matrices(se3)
    rotation = se3[:, :3, :3]
    translation = se3[:, :3, 3:]
    rotation_t = np.transpose(rotation, (0, 2, 1))
    top_right = -np.matmul(rotation_t, translation)
    inverted = np.tile(np.eye(4, dtype=se3.dtype), (len(rotation), 1, 1))
    inverted[:, :3, :3] = rotation_t
    inverted[:, :3, 3:] = top_right
    return inverted


def camera_centers_from_extrinsics(extrinsics):
    extrinsics = normalize_extrinsics_matrices(extrinsics)
    rotation = extrinsics[:, :3, :3]
    translation = extrinsics[:, :3, 3]
    return (-(np.transpose(rotation, (0, 2, 1)) @ translation[..., None])).squeeze(-1).astype(np.float32)


def unproject_depth_map_to_world_points(depth_map, extrinsics_cam, intrinsics_cam):
    depth_map = np.asarray(depth_map, dtype=np.float32)
    extrinsics_cam = normalize_extrinsics_matrices(extrinsics_cam)
    intrinsics_cam = np.asarray(intrinsics_cam, dtype=np.float32)
    cam_to_world = closed_form_inverse_se3_numpy(extrinsics_cam)
    world_points = []
    for frame_idx in range(depth_map.shape[0]):
        depth = depth_map[frame_idx]
        if depth.ndim == 3:
            depth = depth[..., 0]
        intrinsic = intrinsics_cam[frame_idx]

        height, width = depth.shape
        u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

        fu, fv = intrinsic[0, 0], intrinsic[1, 1]
        cu, cv = intrinsic[0, 2], intrinsic[1, 2]

        x_cam = (u - cu) * depth / max(float(fu), 1e-6)
        y_cam = (v - cv) * depth / max(float(fv), 1e-6)
        z_cam = depth
        cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1)

        rotation = cam_to_world[frame_idx, :3, :3]
        translation = cam_to_world[frame_idx, :3, 3]
        world = np.dot(cam_coords, rotation.T) + translation
        world_points.append(world.astype(np.float32))

    return np.stack(world_points, axis=0)


def render_point_cloud_views(output_path, points, colors, title, camera_xyz=None, max_points=180000):
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.shape[0] == 0:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=220, facecolor="black")
        ax.set_facecolor("black")
        ax.text(0.5, 0.5, "No points", color="white", ha="center", va="center", fontsize=20)
        ax.set_axis_off()
        fig.suptitle(title, color="white", fontsize=14)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05, facecolor=fig.get_facecolor())
        plt.close(fig)
        return

    points, colors = downsample_point_cloud(points, colors, max_points=max_points)
    camera_xyz = None if camera_xyz is None else np.asarray(camera_xyz, dtype=np.float32)
    norm_colors = np.clip(colors.astype(np.float32) / 255.0, 0.0, 1.0)
    point_size = float(np.clip(30000.0 / max(points.shape[0], 1), 0.06, 0.45))

    def _project(cloud, mode):
        if mode == "front":
            return cloud[:, 0], cloud[:, 1]
        if mode == "side":
            return cloud[:, 2], cloud[:, 1]
        if mode == "top":
            return cloud[:, 0], cloud[:, 2]
        if mode == "oblique":
            return cloud[:, 0] + 0.45 * cloud[:, 2], cloud[:, 1] - 0.18 * cloud[:, 2]
        raise ValueError(f"Unsupported render mode: {mode}")

    def _set_limits(ax, x, y):
        if x.size <= 1 or y.size <= 1:
            return
        x_lo, x_hi = np.percentile(x, [1.0, 99.0])
        y_lo, y_hi = np.percentile(y, [1.0, 99.0])
        span = max(float(x_hi - x_lo), float(y_hi - y_lo), 1e-3)
        pad = span * 0.08
        x_center = float((x_lo + x_hi) * 0.5)
        y_center = float((y_lo + y_hi) * 0.5)
        ax.set_xlim(x_center - span * 0.5 - pad, x_center + span * 0.5 + pad)
        ax.set_ylim(y_center - span * 0.5 - pad, y_center + span * 0.5 + pad)

    view_specs = (
        ("Front (X/Y)", "front"),
        ("Side (Z/Y)", "side"),
        ("Top (X/Z)", "top"),
        ("Oblique", "oblique"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=220, facecolor="black")
    for ax, (label, mode) in zip(axes.flatten(), view_specs):
        ax.set_facecolor("black")
        x_vals, y_vals = _project(points, mode)
        ax.scatter(x_vals, y_vals, s=point_size, c=norm_colors, linewidths=0, alpha=0.82)
        if camera_xyz is not None and camera_xyz.size > 0:
            cam_x, cam_y = _project(camera_xyz, mode)
            ax.scatter(cam_x, cam_y, s=28, c="#ff5a36", marker="^", linewidths=0, alpha=0.95)
        _set_limits(ax, x_vals, y_vals)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(label, color="white", fontsize=12)
    fig.suptitle(title, color="white", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.06, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_mosaic(path, images_hwc):
    labeled = [Image.fromarray(np.asarray(img, dtype=np.uint8)) for img in images_hwc]
    width = max(im.width for im in labeled)
    height = max(im.height for im in labeled)
    cols = min(4, len(labeled))
    rows = (len(labeled) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * width, rows * height), color=(16, 16, 16))
    for idx, image in enumerate(labeled):
        row = idx // cols
        col = idx % cols
        canvas.paste(image, (col * width, row * height))
    canvas.save(path)


def save_scalar_mosaic(path, arrays, normalize_mode="binary"):
    arrays = [np.asarray(arr) for arr in arrays]
    vis_images = []
    for arr in arrays:
        if normalize_mode == "binary":
            gray = (arr > 0).astype(np.uint8) * 255
        else:
            arr = np.asarray(arr, dtype=np.float32)
            finite = np.isfinite(arr)
            if finite.any():
                active = arr[finite]
                lo = float(active.min())
                hi = float(np.percentile(active, 99.0))
                if hi <= lo:
                    hi = lo + 1e-6
                scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
            else:
                scaled = np.zeros_like(arr, dtype=np.float32)
            gray = (scaled * 255.0).astype(np.uint8)
        vis_images.append(np.stack([gray, gray, gray], axis=-1))
    save_mosaic(path, vis_images)


def stack_optional(sample, key, dtype=None):
    if key not in sample or sample[key] is None:
        return None
    array = np.stack(sample[key])
    if dtype is not None:
        array = array.astype(dtype)
    return array


def summarize_optional_mask(mask_array):
    if mask_array is None:
        return {"present": False}
    mask_array = np.asarray(mask_array, dtype=bool)
    per_view_coverage = mask_array.reshape(mask_array.shape[0], -1).mean(axis=1)
    return {
        "present": True,
        "any_nonzero": bool(mask_array.any()),
        "nonzero_views": int(np.sum(per_view_coverage > 0.0)),
        "pixel_count": int(mask_array.sum()),
        "coverage_mean": float(per_view_coverage.mean()),
        "coverage_max": float(per_view_coverage.max()),
        "per_view_coverage": [float(x) for x in per_view_coverage.tolist()],
    }


def summarize_optional_feature_map(feature_array):
    if feature_array is None:
        return {"present": False}
    feature_array = np.asarray(feature_array, dtype=np.float32)
    flat = feature_array.reshape(feature_array.shape[0], -1)
    per_view_nonzero = (flat > 0.0).mean(axis=1)
    per_view_max = flat.max(axis=1)
    return {
        "present": True,
        "any_nonzero": bool(np.any(flat > 0.0)),
        "nonzero_views": int(np.sum(per_view_nonzero > 0.0)),
        "value_mean": float(feature_array.mean()),
        "value_max": float(feature_array.max()),
        "nonzero_coverage_mean": float(per_view_nonzero.mean()),
        "per_view_nonzero_coverage": [float(x) for x in per_view_nonzero.tolist()],
        "per_view_max": [float(x) for x in per_view_max.tolist()],
    }


def summarize_optional_depth_map(depth_array):
    if depth_array is None:
        return {"present": False}
    depth_array = np.asarray(depth_array, dtype=np.float32)
    valid = np.isfinite(depth_array) & (depth_array > 0.0)
    per_view_coverage = valid.reshape(valid.shape[0], -1).mean(axis=1)
    if valid.any():
        active = depth_array[valid]
        p50, p95 = np.percentile(active, [50.0, 95.0])
        value_mean = float(active.mean())
        value_max = float(active.max())
    else:
        p50 = 0.0
        p95 = 0.0
        value_mean = 0.0
        value_max = 0.0
    return {
        "present": True,
        "any_nonzero": bool(valid.any()),
        "nonzero_views": int(np.sum(per_view_coverage > 0.0)),
        "pixel_count": int(valid.sum()),
        "coverage_mean": float(per_view_coverage.mean()),
        "coverage_max": float(per_view_coverage.max()),
        "depth_mean": value_mean,
        "depth_p50": float(p50),
        "depth_p95": float(p95),
        "depth_max": value_max,
        "per_view_coverage": [float(x) for x in per_view_coverage.tolist()],
    }


def resolve_human_prior_target_spec(cfg):
    fallback = {
        "scope": "unproject_geometry",
        "mask_key": "human_prior_completion_masks",
        "feature_map_key": "smpl_prior_feature_maps",
        "train_only": False,
        "anchor_view_only": False,
    }
    if cfg is None:
        return fallback
    for scope in ("unproject_geometry", "depth"):
        scope_cfg = OmegaConf.select(cfg, f"loss.{scope}")
        if scope_cfg is None:
            continue
        mask_key = str(scope_cfg.get("human_prior_mask_key", "") or "").strip()
        feature_map_key = str(scope_cfg.get("human_prior_feature_map_key", "") or "").strip()
        if not mask_key and not feature_map_key:
            continue
        return {
            "scope": scope,
            "mask_key": mask_key,
            "feature_map_key": feature_map_key,
            "train_only": bool(scope_cfg.get("human_prior_train_only", True)),
            "anchor_view_only": bool(scope_cfg.get("human_prior_anchor_view_only", False)),
        }
    return fallback


def build_human_prior_target_mask(sample, split, prior_target_spec):
    reference_mask = stack_optional(sample, "point_masks", dtype=bool)
    if reference_mask is None:
        return None

    batch = {"_loss_phase": str(split).lower()}
    mask_key = str(prior_target_spec.get("mask_key", "") or "").strip()
    feature_map_key = str(prior_target_spec.get("feature_map_key", "") or "").strip()
    if mask_key:
        mask_array = stack_optional(sample, mask_key, dtype=bool)
        if mask_array is not None:
            batch[mask_key] = torch.from_numpy(mask_array[None].astype(bool))
        else:
            mask_key = ""
    if feature_map_key:
        feature_array = stack_optional(sample, feature_map_key, dtype=np.float32)
        if feature_array is not None:
            batch[feature_map_key] = torch.from_numpy(feature_array[None].astype(np.float32))
        else:
            feature_map_key = ""

    prior_mask, prior_feature_map = _build_human_prior_region_components(
        batch=batch,
        reference_mask=torch.from_numpy(reference_mask[None].astype(bool)),
        human_prior_mask_key=mask_key or None,
        human_prior_feature_map_key=feature_map_key or None,
        human_prior_train_only=bool(prior_target_spec.get("train_only", True)),
        human_prior_anchor_view_only=bool(prior_target_spec.get("anchor_view_only", False)),
    )
    prior_target_mask = _build_human_prior_target_mask(prior_mask, prior_feature_map)
    if prior_target_mask is None:
        return None
    return prior_target_mask.squeeze(0).cpu().numpy().astype(bool)


def summarize_prior_support(sample, prior_target_mask=None):
    return {
        "smpl_prior_masks": summarize_optional_mask(stack_optional(sample, "smpl_prior_masks", dtype=bool)),
        "smpl_prior_feature_maps": summarize_optional_feature_map(
            stack_optional(sample, "smpl_prior_feature_maps", dtype=np.float32)
        ),
        "human_prior_completion_masks": summarize_optional_mask(
            stack_optional(sample, "human_prior_completion_masks", dtype=bool)
        ),
        "human_prior_completion_point_masks": summarize_optional_mask(
            stack_optional(sample, "human_prior_completion_point_masks", dtype=bool)
        ),
        "human_prior_completion_depths": summarize_optional_depth_map(
            stack_optional(sample, "human_prior_completion_depths", dtype=np.float32)
        ),
        "human_prior_target_mask": summarize_optional_mask(
            None if prior_target_mask is None else np.asarray(prior_target_mask, dtype=bool)
        ),
        "head_hair_region_masks": summarize_optional_mask(
            stack_optional(sample, "head_hair_region_masks", dtype=bool)
        ),
        "head_hair_detail_masks": summarize_optional_mask(
            stack_optional(sample, "head_hair_detail_masks", dtype=bool)
        ),
    }


def flatten_masked_point_cloud(point_maps, point_masks, color_maps):
    point_maps = np.asarray(point_maps, dtype=np.float32)
    point_masks = np.asarray(point_masks, dtype=bool)
    color_maps = np.asarray(color_maps, dtype=np.uint8)
    flat_points = point_maps.reshape(-1, 3)[point_masks.reshape(-1)]
    flat_colors = color_maps.reshape(-1, 3)[point_masks.reshape(-1)]
    return flat_points.astype(np.float32), flat_colors.astype(np.uint8)


def downsample_point_cloud(points, colors, max_points=250000):
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.shape[0] <= int(max_points):
        return points, colors
    keep = np.linspace(0, points.shape[0] - 1, int(max_points), dtype=np.int64)
    return points[keep], colors[keep]


def build_completion_artifacts(sample, images):
    base_world_points = stack_optional(sample, "world_points", dtype=np.float32)
    base_point_masks = stack_optional(sample, "point_masks", dtype=bool)
    completion_world_points = stack_optional(sample, "human_prior_completion_world_points", dtype=np.float32)
    completion_point_masks = stack_optional(sample, "human_prior_completion_point_masks", dtype=bool)

    empty_points = np.zeros((0, 3), dtype=np.float32)
    empty_colors = np.zeros((0, 3), dtype=np.uint8)
    if (
        base_world_points is None
        or base_point_masks is None
        or completion_world_points is None
        or completion_point_masks is None
    ):
        return {
            "completion_point_masks": None,
            "completed_point_masks": None,
            "completion_points": empty_points,
            "completion_colors": empty_colors,
            "completed_points": empty_points,
            "completed_colors": empty_colors,
            "base_point_count": int(base_point_masks.sum()) if base_point_masks is not None else 0,
            "completion_point_count": 0,
            "completed_point_count": int(base_point_masks.sum()) if base_point_masks is not None else 0,
            "added_point_count": 0,
            "overlap_point_count": 0,
        }

    completion_points, completion_colors = flatten_masked_point_cloud(
        completion_world_points,
        completion_point_masks,
        images,
    )
    completed_point_masks = base_point_masks | completion_point_masks
    completed_world_points = np.where(base_point_masks[..., None], base_world_points, completion_world_points)
    completed_points, completed_colors = flatten_masked_point_cloud(
        completed_world_points,
        completed_point_masks,
        images,
    )
    added_point_masks = completed_point_masks & (~base_point_masks)
    return {
        "completion_point_masks": completion_point_masks,
        "completed_point_masks": completed_point_masks,
        "completion_points": completion_points,
        "completion_colors": completion_colors,
        "completed_points": completed_points,
        "completed_colors": completed_colors,
        "base_point_count": int(base_point_masks.sum()),
        "completion_point_count": int(completion_point_masks.sum()),
        "completed_point_count": int(completed_point_masks.sum()),
        "added_point_count": int(added_point_masks.sum()),
        "overlap_point_count": int((base_point_masks & completion_point_masks).sum()),
    }


def export_prior_artifacts(output_dir, sample, images, prior_target_mask=None, completion_artifacts=None):
    optional_specs = (
        ("smpl_prior_masks", "sample_smpl_prior_masks.png", "binary", bool),
        ("smpl_prior_feature_maps", "sample_smpl_prior_feature_maps.png", "continuous", np.float32),
        ("human_prior_completion_masks", "sample_human_prior_completion_masks.png", "binary", bool),
        ("human_prior_completion_point_masks", "sample_human_prior_completion_point_masks.png", "binary", bool),
        ("human_prior_completion_depths", "sample_human_prior_completion_depths.png", "continuous", np.float32),
        ("head_hair_region_masks", "sample_head_hair_region_masks.png", "binary", bool),
        ("head_hair_detail_masks", "sample_head_hair_detail_masks.png", "binary", bool),
    )
    for key, file_name, normalize_mode, dtype in optional_specs:
        array = stack_optional(sample, key, dtype=dtype)
        if array is not None:
            save_scalar_mosaic(output_dir / file_name, array, normalize_mode=normalize_mode)
    if prior_target_mask is not None:
        save_scalar_mosaic(
            output_dir / "sample_human_prior_target_mask.png",
            np.asarray(prior_target_mask, dtype=bool),
            normalize_mode="binary",
        )
    completion_artifacts = completion_artifacts or build_completion_artifacts(sample, images)
    completion_points, completion_colors = downsample_point_cloud(
        completion_artifacts["completion_points"],
        completion_artifacts["completion_colors"],
    )
    completed_points, completed_colors = downsample_point_cloud(
        completion_artifacts["completed_points"],
        completion_artifacts["completed_colors"],
    )
    write_binary_ply(output_dir / "sample_human_prior_completion_world_points.ply", completion_points, completion_colors)
    write_binary_ply(output_dir / "sample_completed_world_points.ply", completed_points, completed_colors)
    camera_xyz = camera_centers_from_extrinsics(np.stack(sample["extrinsics"]).astype(np.float32))
    render_point_cloud_views(
        output_dir / "sample_human_prior_completion_world_points_views.png",
        completion_artifacts["completion_points"],
        completion_artifacts["completion_colors"],
        title="Human Prior Completion Point Cloud",
        camera_xyz=camera_xyz,
    )
    render_point_cloud_views(
        output_dir / "sample_completed_world_points_views.png",
        completion_artifacts["completed_points"],
        completion_artifacts["completed_colors"],
        title="Completed Point Cloud",
        camera_xyz=camera_xyz,
    )


def build_case_package(output_dir, sample, prior_target_mask=None):
    payload = {
        "seq_name": np.asarray(str(sample["seq_name"])),
        "ids": np.asarray(sample["ids"]),
        "camera_names": np.asarray(sample.get("camera_names", []), dtype=object),
        "images": np.stack(sample["images"]).astype(np.uint8),
        "depths": np.stack(sample["depths"]).astype(np.float32),
        "world_points": np.stack(sample["world_points"]).astype(np.float32),
        "point_masks": np.stack(sample["point_masks"]).astype(bool),
        "extrinsics": np.stack(sample["extrinsics"]).astype(np.float32),
        "intrinsics": np.stack(sample["intrinsics"]).astype(np.float32),
    }
    for key, dtype in (
        ("foreground_masks", bool),
        ("depth_conf_maps", np.float32),
        ("smpl_prior_masks", bool),
        ("smpl_prior_feature_maps", np.float32),
        ("human_prior_completion_masks", bool),
        ("human_prior_completion_depths", np.float32),
        ("human_prior_completion_world_points", np.float32),
        ("human_prior_completion_point_masks", bool),
        ("head_hair_region_masks", bool),
        ("head_hair_detail_masks", bool),
    ):
        array = stack_optional(sample, key, dtype=dtype)
        if array is not None:
            payload[key] = array
    if prior_target_mask is not None:
        payload["human_prior_target_mask"] = np.asarray(prior_target_mask, dtype=bool)
    package_path = output_dir / "sample_prior_case_package.npz"
    np.savez_compressed(package_path, **payload)
    return package_path


def summarize_point_cloud(points, written_vertex_count):
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return {
            "point_count_full": 0,
            "point_count_written": int(written_vertex_count),
            "downsampled_for_ply": False,
            "bbox_min": [0.0, 0.0, 0.0],
            "bbox_max": [0.0, 0.0, 0.0],
            "extent": [0.0, 0.0, 0.0],
            "centroid": [0.0, 0.0, 0.0],
            "radius_percentiles": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
            "axis_percentiles": {
                "x": {"p05": 0.0, "p50": 0.0, "p95": 0.0},
                "y": {"p05": 0.0, "p50": 0.0, "p95": 0.0},
                "z": {"p05": 0.0, "p50": 0.0, "p95": 0.0},
            },
        }

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    extent = bbox_max - bbox_min
    centroid = points.mean(axis=0)
    centered = points - centroid[None, :]
    radii = np.linalg.norm(centered, axis=1)

    def _axis_stats(axis_values):
        p05, p50, p95 = np.percentile(axis_values, [5.0, 50.0, 95.0])
        return {"p05": float(p05), "p50": float(p50), "p95": float(p95)}

    p50, p95, p99 = np.percentile(radii, [50.0, 95.0, 99.0])
    return {
        "point_count_full": int(points.shape[0]),
        "point_count_written": int(written_vertex_count),
        "downsampled_for_ply": bool(points.shape[0] != int(written_vertex_count)),
        "bbox_min": [float(x) for x in bbox_min.tolist()],
        "bbox_max": [float(x) for x in bbox_max.tolist()],
        "extent": [float(x) for x in extent.tolist()],
        "centroid": [float(x) for x in centroid.tolist()],
        "radius_percentiles": {"p50": float(p50), "p95": float(p95), "p99": float(p99)},
        "axis_percentiles": {
            "x": _axis_stats(points[:, 0]),
            "y": _axis_stats(points[:, 1]),
            "z": _axis_stats(points[:, 2]),
        },
    }


def export_sample_artifacts(output_dir, images, depths, world_points, world_colors, extrinsics, intrinsics):
    flat_points, flat_colors = downsample_point_cloud(world_points, world_colors, max_points=250000)
    write_binary_ply(output_dir / "sample_world_points.ply", flat_points, flat_colors)
    save_mosaic(output_dir / "sample_images.png", images)
    camera_xyz = camera_centers_from_extrinsics(extrinsics)
    render_point_cloud_views(
        output_dir / "sample_world_points_views.png",
        world_points,
        world_colors,
        title="Dataset World Points",
        camera_xyz=camera_xyz,
    )

    depth_world_points = unproject_depth_map_to_world_points(depths, extrinsics, intrinsics)
    depth_maps = np.asarray(depths, dtype=np.float32)
    if depth_maps.ndim == 4:
        depth_valid = np.isfinite(depth_maps[..., 0]) & (depth_maps[..., 0] > 0.0)
    else:
        depth_valid = np.isfinite(depth_maps) & (depth_maps > 0.0)
    depth_flat_points = depth_world_points.reshape(-1, 3)[depth_valid.reshape(-1)]
    depth_flat_colors = images.reshape(-1, 3)[depth_valid.reshape(-1)]
    depth_ply_points, depth_ply_colors = downsample_point_cloud(depth_flat_points, depth_flat_colors, max_points=250000)
    write_binary_ply(output_dir / "sample_depth_unproject_world_points.ply", depth_ply_points, depth_ply_colors)
    render_point_cloud_views(
        output_dir / "sample_depth_unproject_world_points_views.png",
        depth_flat_points,
        depth_flat_colors,
        title="Depth Unprojected Point Cloud",
        camera_xyz=camera_xyz,
    )

    depth_vis = []
    for depth in depths:
        valid = depth > 0
        if valid.any():
            p95 = np.percentile(depth[valid], 95.0)
            scaled = np.clip(depth / max(p95, 1e-6), 0.0, 1.0)
        else:
            scaled = np.zeros_like(depth)
        depth_vis.append((scaled * 255.0).astype(np.uint8))
    save_mosaic(output_dir / "sample_depths.png", [np.stack([d, d, d], axis=-1) for d in depth_vis])


def build_sample_payload(
    args,
    dataset,
    sample,
    masks,
    pointcloud_stats,
    zju_dir,
    config_path,
    sample_index,
    prior_target_spec=None,
    prior_target_mask=None,
    completion_artifacts=None,
):
    prior_support = summarize_prior_support(sample, prior_target_mask=prior_target_mask)
    target_mask_stats = prior_support["human_prior_target_mask"]
    completion_artifacts = completion_artifacts or {
        "base_point_count": int(masks.sum()),
        "completion_point_count": 0,
        "completed_point_count": int(masks.sum()),
        "added_point_count": 0,
        "overlap_point_count": 0,
    }
    base_point_count = int(completion_artifacts.get("base_point_count", int(masks.sum())))
    added_point_count = int(completion_artifacts.get("added_point_count", 0))
    added_ratio = float(added_point_count / max(base_point_count, 1))
    return {
        "config_path": str(config_path) if config_path is not None else "",
        "seed": int(args.seed),
        "probe_sample_index": int(sample_index),
        "dataset_len": len(dataset),
        "sequence_list_len": dataset.sequence_list_len,
        "sample_seq_name": sample["seq_name"],
        "ids": sample["ids"].tolist(),
        "num_images": len(sample["images"]),
        "available_view_count": int(sample.get("available_view_count", len(sample["images"]))),
        "image_shape": list(np.asarray(sample["images"][0]).shape),
        "depth_shape": list(np.asarray(sample["depths"][0]).shape),
        "valid_ratio": float(masks.mean()),
        "valid_points": int(masks.sum()),
        "pointcloud_stats": pointcloud_stats,
        "camera_source": args.camera_source,
        "mask_source": args.mask_source,
        "source_policy": args.source_policy,
        "requested_source_view_pool": str(sample.get("selection_requested_source_view_pool", args.source_view_pool)),
        "effective_source_view_pool": str(sample.get("selection_source_view_pool", args.source_view_pool)),
        "source_view_pool_train_probability": float(
            sample.get("selection_source_view_pool_train_probability", args.source_view_pool_train_probability)
        ),
        "rawpool_candidate_pool_used": bool(
            sample.get(
                "selection_rawpool_candidate_pool_used",
                sample.get("selection_source_view_pool", args.source_view_pool) == "geom_plus_raw",
            )
        ),
        "source_anchor_policy": str(sample.get("selection_source_anchor_policy", args.source_anchor_policy)),
        "min_supervised_views": int(sample.get("selection_min_supervised_views", args.min_supervised_views)),
        "supervised_view_quality_filter": str(
            sample.get("selection_supervised_view_quality_filter", args.supervised_view_quality_filter)
        ),
        "camera_names": list(sample.get("camera_names", [])),
        "candidate_supervised_camera_names": list(sample.get("candidate_supervised_camera_names", [])),
        "supervised_camera_names": list(sample.get("supervised_camera_names", [])),
        "source_only_camera_names": list(sample.get("source_only_camera_names", [])),
        "source_only_view_count": int(len(sample.get("source_only_camera_names", []))),
        "supervised_view_count": int(len(sample.get("supervised_camera_names", []))),
        "dropped_supervised_camera_names": list(sample.get("dropped_supervised_camera_names", [])),
        "conf_depth_camera_names": list(sample.get("conf_depth_camera_names", [])),
        "conf_depth_dropped_camera_names": list(sample.get("conf_depth_dropped_camera_names", [])),
        "supervised_view_quality_scores": dict(sample.get("supervised_view_quality_scores", {})),
        "geom_subdirs_present": list(sample.get("geom_subdirs_present", [])),
        "selection_anchor_camera": sample.get("selection_anchor_camera"),
        "available_candidate_view_count": int(
            sample.get("available_candidate_view_count", sample.get("available_view_count", len(sample["images"])))
        ),
        "available_candidate_camera_names": list(sample.get("available_candidate_camera_names", [])),
        "zju_dir": str(zju_dir),
        "geom_subdir": args.geom_subdir,
        "seq_names": args.seq_names,
        "human_prior_target_scope": str((prior_target_spec or {}).get("scope", "")),
        "human_prior_target_mask_key": str((prior_target_spec or {}).get("mask_key", "")),
        "human_prior_target_feature_map_key": str((prior_target_spec or {}).get("feature_map_key", "")),
        "human_prior_target_mask_stats": target_mask_stats,
        "prior_coverage_mean": float(target_mask_stats.get("coverage_mean", 0.0)) if target_mask_stats.get("present") else 0.0,
        "completion_point_count": int(completion_artifacts.get("completion_point_count", 0)),
        "completed_point_count": int(completion_artifacts.get("completed_point_count", base_point_count)),
        "added_point_count": added_point_count,
        "completion_overlap_point_count": int(completion_artifacts.get("overlap_point_count", 0)),
        "added_ratio": added_ratio,
        "prior_support": prior_support,
    }


def write_sample_summary(output_dir, payload):
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# ZJU VGGT-Geom Dataset Probe",
                "",
                f"- dataset_len: `{payload['dataset_len']}`",
                f"- sequence_list_len: `{payload['sequence_list_len']}`",
                f"- probe_sample_index: `{payload['probe_sample_index']}`",
                f"- sample_seq_name: `{payload['sample_seq_name']}`",
                f"- ids: `{payload['ids']}`",
                f"- num_images: `{payload['num_images']}`",
                f"- available_view_count: `{payload['available_view_count']}`",
                f"- image_shape: `{payload['image_shape']}`",
                f"- depth_shape: `{payload['depth_shape']}`",
                f"- valid_ratio: `{payload['valid_ratio']:.4f}`",
                f"- valid_points: `{payload['valid_points']}`",
                f"- pointcloud_point_count_full: `{payload['pointcloud_stats']['point_count_full']}`",
                f"- pointcloud_point_count_written: `{payload['pointcloud_stats']['point_count_written']}`",
                f"- pointcloud_downsampled_for_ply: `{payload['pointcloud_stats']['downsampled_for_ply']}`",
                f"- pointcloud_centroid: `{payload['pointcloud_stats']['centroid']}`",
                f"- pointcloud_extent: `{payload['pointcloud_stats']['extent']}`",
                f"- pointcloud_radius_percentiles: `{payload['pointcloud_stats']['radius_percentiles']}`",
                f"- pointcloud_axis_percentiles: `{payload['pointcloud_stats']['axis_percentiles']}`",
                f"- camera_source: `{payload['camera_source']}`",
                f"- mask_source: `{payload['mask_source']}`",
                f"- source_policy: `{payload['source_policy']}`",
                f"- requested_source_view_pool: `{payload['requested_source_view_pool']}`",
                f"- effective_source_view_pool: `{payload['effective_source_view_pool']}`",
                f"- source_view_pool_train_probability: `{payload['source_view_pool_train_probability']}`",
                f"- rawpool_candidate_pool_used: `{payload['rawpool_candidate_pool_used']}`",
                f"- source_anchor_policy: `{payload['source_anchor_policy']}`",
                f"- min_supervised_views: `{payload['min_supervised_views']}`",
                f"- supervised_view_quality_filter: `{payload['supervised_view_quality_filter']}`",
                f"- camera_names: `{payload['camera_names']}`",
                f"- candidate_supervised_camera_names: `{payload['candidate_supervised_camera_names']}`",
                f"- supervised_camera_names: `{payload['supervised_camera_names']}`",
                f"- source_only_camera_names: `{payload['source_only_camera_names']}`",
                f"- supervised_view_count: `{payload['supervised_view_count']}`",
                f"- source_only_view_count: `{payload['source_only_view_count']}`",
                f"- dropped_supervised_camera_names: `{payload['dropped_supervised_camera_names']}`",
                f"- conf_depth_camera_names: `{payload['conf_depth_camera_names']}`",
                f"- conf_depth_dropped_camera_names: `{payload['conf_depth_dropped_camera_names']}`",
                f"- supervised_view_quality_scores: `{payload['supervised_view_quality_scores']}`",
                f"- geom_subdirs_present: `{payload['geom_subdirs_present']}`",
                f"- selection_anchor_camera: `{payload['selection_anchor_camera']}`",
                f"- available_candidate_view_count: `{payload['available_candidate_view_count']}`",
                f"- available_candidate_camera_names: `{payload['available_candidate_camera_names']}`",
                f"- human_prior_target_scope: `{payload['human_prior_target_scope']}`",
                f"- human_prior_target_mask_key: `{payload['human_prior_target_mask_key']}`",
                f"- human_prior_target_feature_map_key: `{payload['human_prior_target_feature_map_key']}`",
                f"- prior_coverage_mean: `{payload['prior_coverage_mean']:.6f}`",
                f"- completion_point_count: `{payload['completion_point_count']}`",
                f"- completed_point_count: `{payload['completed_point_count']}`",
                f"- added_point_count: `{payload['added_point_count']}`",
                f"- completion_overlap_point_count: `{payload['completion_overlap_point_count']}`",
                f"- added_ratio: `{payload['added_ratio']:.6f}`",
                f"- human_prior_target_mask_stats: `{payload['human_prior_target_mask_stats']}`",
                "",
                "## Files",
                "",
                "- `sample_images.png`",
                "- `sample_depths.png`",
                "- `sample_world_points.ply`",
                "- `sample_world_points_views.png`",
                "- `sample_depth_unproject_world_points.ply`",
                "- `sample_depth_unproject_world_points_views.png`",
                "- `sample_smpl_prior_masks.png`",
                "- `sample_smpl_prior_feature_maps.png`",
                "- `sample_human_prior_completion_masks.png`",
                "- `sample_human_prior_completion_point_masks.png`",
                "- `sample_human_prior_completion_depths.png`",
                "- `sample_human_prior_completion_world_points.ply`",
                "- `sample_human_prior_completion_world_points_views.png`",
                "- `sample_completed_world_points.ply`",
                "- `sample_completed_world_points_views.png`",
                "- `sample_human_prior_target_mask.png`",
                "- `sample_prior_case_package.npz`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_aggregate_summary(args, config_path, zju_dir, payloads):
    valid_ratios = np.asarray([item["valid_ratio"] for item in payloads], dtype=np.float64)
    valid_points = np.asarray([item["valid_points"] for item in payloads], dtype=np.float64)
    supervised_counts = np.asarray([item["supervised_view_count"] for item in payloads], dtype=np.float64)
    source_only_counts = np.asarray([item["source_only_view_count"] for item in payloads], dtype=np.float64)
    extents = np.asarray([item["pointcloud_stats"]["extent"] for item in payloads], dtype=np.float64)
    radius_p95 = np.asarray(
        [item["pointcloud_stats"]["radius_percentiles"]["p95"] for item in payloads], dtype=np.float64
    )
    prior_coverages = np.asarray([item.get("prior_coverage_mean", 0.0) for item in payloads], dtype=np.float64)
    completion_points = np.asarray([item.get("completion_point_count", 0) for item in payloads], dtype=np.float64)
    completed_points = np.asarray([item.get("completed_point_count", 0) for item in payloads], dtype=np.float64)
    added_points = np.asarray([item.get("added_point_count", 0) for item in payloads], dtype=np.float64)
    added_ratios = np.asarray([item.get("added_ratio", 0.0) for item in payloads], dtype=np.float64)
    target_mask_pixels = np.asarray(
        [
            item.get("human_prior_target_mask_stats", {}).get("pixel_count", 0)
            if item.get("human_prior_target_mask_stats", {}).get("present")
            else 0
            for item in payloads
        ],
        dtype=np.float64,
    )

    anchor_histogram = {}
    for item in payloads:
        anchor = str(item.get("selection_anchor_camera") or "")
        anchor_histogram[anchor] = anchor_histogram.get(anchor, 0) + 1

    return {
        "config_path": str(config_path) if config_path is not None else "",
        "seed": int(args.seed),
        "aggregate_samples": int(len(payloads)),
        "aggregate_stride": int(args.aggregate_stride),
        "sample_indices": [int(item["probe_sample_index"]) for item in payloads],
        "source_policy": args.source_policy,
        "camera_source": args.camera_source,
        "mask_source": args.mask_source,
        "zju_dir": str(zju_dir),
        "geom_subdir": args.geom_subdir,
        "seq_names": args.seq_names,
        "valid_ratio": {
            "mean": float(valid_ratios.mean()),
            "median": float(np.median(valid_ratios)),
            "min": float(valid_ratios.min()),
            "max": float(valid_ratios.max()),
        },
        "valid_points": {
            "mean": float(valid_points.mean()),
            "median": float(np.median(valid_points)),
            "min": int(valid_points.min()),
            "max": int(valid_points.max()),
        },
        "supervised_view_count": {
            "mean": float(supervised_counts.mean()),
            "min": int(supervised_counts.min()),
            "max": int(supervised_counts.max()),
        },
        "source_only_view_count": {
            "mean": float(source_only_counts.mean()),
            "min": int(source_only_counts.min()),
            "max": int(source_only_counts.max()),
        },
        "pointcloud_extent": {
            "mean": [float(x) for x in extents.mean(axis=0).tolist()],
            "median": [float(x) for x in np.median(extents, axis=0).tolist()],
            "max": [float(x) for x in extents.max(axis=0).tolist()],
        },
        "pointcloud_radius_p95": {
            "mean": float(radius_p95.mean()),
            "median": float(np.median(radius_p95)),
            "min": float(radius_p95.min()),
            "max": float(radius_p95.max()),
        },
        "prior_coverage": {
            "mean": float(prior_coverages.mean()),
            "median": float(np.median(prior_coverages)),
            "min": float(prior_coverages.min()),
            "max": float(prior_coverages.max()),
        },
        "completion_point_count": {
            "mean": float(completion_points.mean()),
            "median": float(np.median(completion_points)),
            "min": int(completion_points.min()),
            "max": int(completion_points.max()),
        },
        "completed_point_count": {
            "mean": float(completed_points.mean()),
            "median": float(np.median(completed_points)),
            "min": int(completed_points.min()),
            "max": int(completed_points.max()),
        },
        "added_point_count": {
            "mean": float(added_points.mean()),
            "median": float(np.median(added_points)),
            "min": int(added_points.min()),
            "max": int(added_points.max()),
        },
        "added_ratio": {
            "mean": float(added_ratios.mean()),
            "median": float(np.median(added_ratios)),
            "min": float(added_ratios.min()),
            "max": float(added_ratios.max()),
        },
        "human_prior_target_pixel_count": {
            "mean": float(target_mask_pixels.mean()),
            "median": float(np.median(target_mask_pixels)),
            "min": int(target_mask_pixels.min()),
            "max": int(target_mask_pixels.max()),
        },
        "samples_with_nonzero_completion": int(np.sum(completion_points > 0)),
        "selection_anchor_camera_histogram": anchor_histogram,
        "sample_seq_names": [str(item["sample_seq_name"]) for item in payloads],
    }


def write_aggregate_summary(output_dir, aggregate):
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "aggregate_summary.md").write_text(
        "\n".join(
            [
                "# ZJU VGGT-Geom Dataset Aggregate Probe",
                "",
                f"- aggregate_samples: `{aggregate['aggregate_samples']}`",
                f"- aggregate_stride: `{aggregate['aggregate_stride']}`",
                f"- sample_indices: `{aggregate['sample_indices']}`",
                f"- source_policy: `{aggregate['source_policy']}`",
                f"- valid_ratio_mean: `{aggregate['valid_ratio']['mean']:.6f}`",
                f"- valid_ratio_median: `{aggregate['valid_ratio']['median']:.6f}`",
                f"- valid_ratio_minmax: `[{aggregate['valid_ratio']['min']:.6f}, {aggregate['valid_ratio']['max']:.6f}]`",
                f"- valid_points_mean: `{aggregate['valid_points']['mean']:.2f}`",
                f"- supervised_view_count_mean: `{aggregate['supervised_view_count']['mean']:.2f}`",
                f"- source_only_view_count_mean: `{aggregate['source_only_view_count']['mean']:.2f}`",
                f"- pointcloud_extent_mean: `{aggregate['pointcloud_extent']['mean']}`",
                f"- pointcloud_extent_median: `{aggregate['pointcloud_extent']['median']}`",
                f"- pointcloud_radius_p95_mean: `{aggregate['pointcloud_radius_p95']['mean']:.6f}`",
                f"- pointcloud_radius_p95_median: `{aggregate['pointcloud_radius_p95']['median']:.6f}`",
                f"- prior_coverage_mean: `{aggregate['prior_coverage']['mean']:.6f}`",
                f"- prior_coverage_median: `{aggregate['prior_coverage']['median']:.6f}`",
                f"- completion_point_count_mean: `{aggregate['completion_point_count']['mean']:.2f}`",
                f"- completed_point_count_mean: `{aggregate['completed_point_count']['mean']:.2f}`",
                f"- added_point_count_mean: `{aggregate['added_point_count']['mean']:.2f}`",
                f"- added_ratio_mean: `{aggregate['added_ratio']['mean']:.6f}`",
                f"- human_prior_target_pixel_count_mean: `{aggregate['human_prior_target_pixel_count']['mean']:.2f}`",
                f"- samples_with_nonzero_completion: `{aggregate['samples_with_nonzero_completion']}`",
                f"- selection_anchor_camera_histogram: `{aggregate['selection_anchor_camera_histogram']}`",
                "",
                "## Files",
                "",
                "- `summary.json` / `summary.md` still describe the first sampled example",
                "- `per_sample_summaries.json` stores the deterministic sample basket",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    config_path, cfg = apply_config_defaults(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    zju_dir = resolve_zju_dir(args.zju_dir, args.seq_names, args.geom_subdir)
    prior_target_spec = resolve_human_prior_target_spec(cfg)

    common_conf = OmegaConf.create(
        {
            "debug": False,
            "training": args.split == "train",
            "inside_random": False,
            "allow_duplicate_img": bool(args.allow_duplicate_img),
            "load_depth": True,
            "img_size": 518,
            "patch_size": 14,
            "rescale": True,
            "rescale_aug": False,
            "landscape_check": False,
            "augs": {
                "scales": None,
            },
        }
    )

    dataset = ZjuVggtGeomDataset(
        common_conf=common_conf,
        split=args.split,
        ZJU_DIR=str(zju_dir),
        seq_names=args.seq_names,
        geom_subdir=args.geom_subdir,
        holdout_stride=args.holdout_stride,
        camera_source=args.camera_source,
        mask_source=args.mask_source,
        source_policy=args.source_policy,
        source_view_pool=args.source_view_pool,
        source_view_pool_train_probability=args.source_view_pool_train_probability,
        source_anchor_policy=args.source_anchor_policy,
        min_supervised_views=args.min_supervised_views,
        supervised_view_quality_filter=args.supervised_view_quality_filter,
        conf_depth_view_quality_filter=args.conf_depth_view_quality_filter,
        min_depth_conf=args.min_depth_conf,
        smpl_prior_pose_noise_prob=args.smpl_prior_pose_noise_prob,
        smpl_prior_pose_noise_rot_deg=args.smpl_prior_pose_noise_rot_deg,
        smpl_prior_pose_noise_trans_scale=args.smpl_prior_pose_noise_trans_scale,
        smpl_prior_pose_noise_scale_std=args.smpl_prior_pose_noise_scale_std,
        len_train=-1,
        len_test=-1,
    )
    sample_count = max(1, int(args.aggregate_samples))
    sample_stride = max(1, int(args.aggregate_stride))
    sample_indices = []
    for offset in range(sample_count):
        idx = int(args.sample_index) + offset * sample_stride
        if idx >= len(dataset):
            break
        sample_indices.append(idx)
    if not sample_indices:
        raise IndexError(f"No valid sample indices available from start={args.sample_index} with stride={sample_stride}")

    payloads = []
    first_sample_summary_path = output_dir / "summary.md"
    for loop_idx, sample_index in enumerate(sample_indices):
        sample = dataset.get_data(seq_index=sample_index, img_per_seq=args.num_images)
        images = np.stack(sample["images"]).astype(np.uint8)
        depths = np.stack(sample["depths"]).astype(np.float32)
        world_points = np.stack(sample["world_points"]).astype(np.float32)
        masks = np.stack(sample["point_masks"]).astype(bool)
        prior_target_mask = build_human_prior_target_mask(sample, args.split, prior_target_spec)
        completion_artifacts = build_completion_artifacts(sample, images)

        flat_points_full = world_points.reshape(-1, 3)[masks.reshape(-1)]
        flat_colors_full = images.reshape(-1, 3)[masks.reshape(-1)]
        flat_points, flat_colors = downsample_point_cloud(flat_points_full, flat_colors_full, max_points=250000)

        pointcloud_stats = summarize_point_cloud(flat_points_full, written_vertex_count=flat_points.shape[0])
        payload = build_sample_payload(
            args=args,
            dataset=dataset,
            sample=sample,
            masks=masks,
            pointcloud_stats=pointcloud_stats,
            zju_dir=zju_dir,
            config_path=config_path,
            sample_index=sample_index,
            prior_target_spec=prior_target_spec,
            prior_target_mask=prior_target_mask,
            completion_artifacts=completion_artifacts,
        )
        payloads.append(payload)

        if loop_idx == 0:
            export_sample_artifacts(
                output_dir,
                images,
                depths,
                flat_points_full,
                flat_colors_full,
                np.stack(sample["extrinsics"]).astype(np.float32),
                np.stack(sample["intrinsics"]).astype(np.float32),
            )
            export_prior_artifacts(
                output_dir,
                sample,
                images,
                prior_target_mask=prior_target_mask,
                completion_artifacts=completion_artifacts,
            )
            build_case_package(output_dir, sample, prior_target_mask=prior_target_mask)
            write_sample_summary(output_dir, payload)

    (output_dir / "per_sample_summaries.json").write_text(
        json.dumps(payloads, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    aggregate = build_aggregate_summary(args=args, config_path=config_path, zju_dir=zju_dir, payloads=payloads)
    write_aggregate_summary(output_dir, aggregate)

    print(first_sample_summary_path)


if __name__ == "__main__":
    main()
