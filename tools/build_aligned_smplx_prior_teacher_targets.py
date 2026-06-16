from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.normal_line_multiview_eval import build_roi_masks, load_scene_view  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build local diagnostic teacher_targets.npz by aligning scene SMPL-X "
            "camera-space prior maps into the VGGT prediction world. This avoids "
            "using VGGT extrinsics as if they were real 4K4D camera extrinsics."
        )
    )
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--align-roi", choices=("full", "head", "face", "head_face"), default="head_face")
    parser.add_argument("--apply-roi", choices=("full", "head", "face", "head_face"), default="head_face")
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--max-correspondences", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=20260429)
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def roi_mask(scene_mask: np.ndarray, roi: str) -> np.ndarray:
    masks = build_roi_masks(scene_mask.astype(bool))
    if roi == "head_face":
        return masks["head"] | masks["face"]
    return masks[roi]


def normalize(vec: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.maximum(norm, eps)


def estimate_umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if source.shape[0] < 16:
        raise RuntimeError(f"Need at least 16 correspondences, got {source.shape[0]}")
    source = source.astype(np.float64)
    target = target.astype(np.float64)
    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)
    src_centered = source - mu_source
    tgt_centered = target - mu_target
    covariance = (tgt_centered.T @ src_centered) / source.shape[0]
    u_mat, singular_values, vt_mat = np.linalg.svd(covariance)
    rotation = u_mat @ vt_mat
    if np.linalg.det(rotation) < 0:
        u_mat[:, -1] *= -1.0
        rotation = u_mat @ vt_mat
    variance = float((src_centered**2).sum() / max(source.shape[0], 1))
    scale = float(singular_values.sum() / max(variance, 1e-12))
    translation = mu_target - scale * (rotation @ mu_source)
    return scale, rotation.astype(np.float64), translation.astype(np.float64)


def apply_similarity(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    flat = points.reshape(-1, 3).astype(np.float64)
    out = scale * (flat @ rotation.T) + translation[None, :]
    return out.reshape(points.shape).astype(np.float32)


def robust_similarity(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_correspondences: int,
    seed: int,
) -> tuple[dict[str, Any], tuple[float, np.ndarray, np.ndarray]]:
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[valid]
    target = target[valid]
    if source.shape[0] > int(max_correspondences) > 0:
        rng = np.random.default_rng(seed)
        idx = rng.choice(source.shape[0], size=int(max_correspondences), replace=False)
        source = source[idx]
        target = target[idx]
    scale, rotation, translation = estimate_umeyama(source, target)
    pred = apply_similarity(source.reshape(1, -1, 3), scale, rotation, translation).reshape(-1, 3)
    residual = np.linalg.norm(pred - target, axis=1)
    keep = residual <= float(np.percentile(residual, 80.0))
    scale, rotation, translation = estimate_umeyama(source[keep], target[keep])
    pred = apply_similarity(source.reshape(1, -1, 3), scale, rotation, translation).reshape(-1, 3)
    residual = np.linalg.norm(pred - target, axis=1)
    summary = {
        "input_correspondences": int(valid.sum()),
        "used_correspondences": int(source.shape[0]),
        "refit_correspondences": int(keep.sum()),
        "scale": float(scale),
        "rotation": rotation,
        "translation": translation,
        "residual_percentiles": [float(v) for v in np.percentile(residual, [0, 25, 50, 75, 90, 95, 99])],
    }
    return summary, (scale, rotation, translation)


def main() -> int:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base = np.load(args.base_predictions, allow_pickle=False)
    base_world = np.asarray(base["world_points"], dtype=np.float32)
    base_conf = np.asarray(base["world_points_conf"], dtype=np.float32)
    view_count, height, width, _ = base_world.shape

    prior = np.load(scene_dir / "prior_maps.npz", allow_pickle=True)
    prior_maps = np.asarray(prior["prior_maps"], dtype=np.float32)
    channels = [str(item) for item in prior["prior_channels"]]
    channel_index = {name: idx for idx, name in enumerate(channels)}
    required = [
        "smplx_posed_cam_x",
        "smplx_posed_cam_y",
        "smplx_posed_cam_z",
        "smplx_cam_nx",
        "smplx_cam_ny",
        "smplx_cam_nz",
        "smplx_visible_mask",
    ]
    missing = [name for name in required if name not in channel_index]
    if missing:
        raise KeyError(f"Missing prior channels: {missing}")

    smplx_cam = np.stack(
        [
            prior_maps[:, channel_index["smplx_posed_cam_x"]],
            prior_maps[:, channel_index["smplx_posed_cam_y"]],
            prior_maps[:, channel_index["smplx_posed_cam_z"]],
        ],
        axis=-1,
    ).astype(np.float32)
    smplx_normals = np.stack(
        [
            prior_maps[:, channel_index["smplx_cam_nx"]],
            prior_maps[:, channel_index["smplx_cam_ny"]],
            prior_maps[:, channel_index["smplx_cam_nz"]],
        ],
        axis=-1,
    ).astype(np.float32)
    smplx_normals = normalize(smplx_normals)
    visible = (
        np.asarray(prior["prior_mask"], dtype=bool)
        & (prior_maps[:, channel_index["smplx_visible_mask"]] > 0.5)
        & np.isfinite(smplx_cam).all(axis=-1)
        & (smplx_cam[..., 2] > 0.0)
    )

    teacher_world = np.zeros_like(base_world, dtype=np.float32)
    teacher_normals = np.zeros_like(base_world, dtype=np.float32)
    teacher_mask = np.zeros(base_world.shape[:3], dtype=bool)
    roi_masks = np.zeros(base_world.shape[:3], dtype=bool)
    summaries: list[dict[str, Any]] = []
    for view_idx in range(view_count):
        scene = load_scene_view(scene_dir, view_idx, (height, width))
        align_roi = roi_mask(scene.mask, str(args.align_roi))
        apply_roi = roi_mask(scene.mask, str(args.apply_roi))
        support = scene.mask.astype(bool)
        conf_valid = support & np.isfinite(base_conf[view_idx]) & (base_conf[view_idx] > 0.0)
        threshold = float(np.percentile(base_conf[view_idx][conf_valid], float(args.conf_percentile))) if conf_valid.any() else 0.0
        align_mask = (
            visible[view_idx]
            & align_roi
            & np.isfinite(base_world[view_idx]).all(axis=-1)
            & (base_conf[view_idx] >= threshold)
        )
        transform_summary, transform = robust_similarity(
            smplx_cam[view_idx][align_mask],
            base_world[view_idx][align_mask],
            max_correspondences=int(args.max_correspondences),
            seed=int(args.seed) + view_idx,
        )
        scale, rotation, translation = transform
        view_teacher = apply_similarity(smplx_cam[view_idx], scale, rotation, translation)
        teacher_world[view_idx] = view_teacher
        teacher_normals[view_idx] = normalize(smplx_normals[view_idx] @ rotation.T)
        teacher_mask[view_idx] = visible[view_idx] & apply_roi
        roi_masks[view_idx] = apply_roi
        delta = np.linalg.norm(view_teacher - base_world[view_idx], axis=-1)
        valid_delta = teacher_mask[view_idx] & np.isfinite(delta)
        summaries.append(
            {
                "view_index": int(view_idx),
                "align_roi_pixels": int(align_roi.sum()),
                "apply_roi_pixels": int(apply_roi.sum()),
                "teacher_mask_pixels": int(teacher_mask[view_idx].sum()),
                "base_conf_threshold": threshold,
                "transform": transform_summary,
                "distance_to_base_percentiles_on_teacher_mask": [
                    float(v) for v in np.percentile(delta[valid_delta], [0, 25, 50, 75, 90, 95, 99])
                ]
                if valid_delta.any()
                else [],
            }
        )

    np.savez_compressed(
        output_dir / "teacher_targets.npz",
        world_points=teacher_world.astype(np.float32),
        teacher_mask=teacher_mask.astype(bool),
        teacher_normals=teacher_normals.astype(np.float32),
        roi_mask=roi_masks.astype(bool),
    )
    summary = {
        "task": "aligned_smplx_prior_teacher_targets",
        "truthful_status": "coarse_smplx_prior_not_final_face_teacher",
        "scene_dir": str(scene_dir),
        "base_predictions": str(args.base_predictions.resolve()),
        "output_dir": str(output_dir),
        "teacher_targets": str((output_dir / "teacher_targets.npz").resolve()),
        "align_roi": str(args.align_roi),
        "apply_roi": str(args.apply_roi),
        "per_view": summaries,
        "notes": [
            "This aligns SMPL-X camera-space prior maps to VGGT prediction world per view.",
            "It is a coarse topology teacher only; template-looking face/hair/clothes are still a failure.",
        ],
    }
    (output_dir / "aligned_smplx_teacher_summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
