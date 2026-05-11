from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from bridge_teacher_targets_between_vggt_worlds import (  # noqa: E402
    camera_center_and_rotation,
    camera_ids,
    estimate_similarity,
    load_manifest,
    median_baseline,
)
from render_open3d_pointcloud import load_2d_roi_mask_stack, load_mask_stack  # noqa: E402
from vggt.utils.normal_refiner import point_map_to_normal_numpy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse same-model predictions from a source crop into a canonical "
            "VGGT prediction only where both crops agree geometrically. This is "
            "a local self-consensus candidate, not an external teacher."
        )
    )
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--base-scene-dir", required=True, type=Path)
    parser.add_argument("--source-predictions", required=True, type=Path)
    parser.add_argument("--source-scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--roi", choices=("head", "face"), default="head")
    parser.add_argument("--source-conf-percentile", type=float, default=45.0)
    parser.add_argument("--base-conf-percentile", type=float, default=0.0)
    parser.add_argument("--max-world-distance", type=float, default=0.07)
    parser.add_argument("--max-depth-delta", type=float, default=0.07)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--confidence-boost", type=float, default=115.0)
    parser.add_argument("--normal-confidence-boost", type=float, default=115.0)
    parser.add_argument("--normal-dilate", type=int, default=1)
    parser.add_argument("--axis-scale", type=float, default=0.01)
    parser.add_argument("--no-axes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def resize_params(width: int, height: int, target: int) -> dict[str, float]:
    if width >= height:
        new_width = int(target)
        new_height = round(height * (new_width / max(1, width)) / 14) * 14
    else:
        new_height = int(target)
        new_width = round(width * (new_height / max(1, height)) / 14) * 14
    new_width = max(14, int(new_width))
    new_height = max(14, int(new_height))
    return {
        "new_width": float(new_width),
        "new_height": float(new_height),
        "left": float((int(target) - new_width) // 2),
        "top": float((int(target) - new_height) // 2),
        "scale_x": float(new_width) / float(max(1, width)),
        "scale_y": float(new_height) / float(max(1, height)),
    }


def source_pixels_to_raw_grid(view_meta: dict[str, Any], target_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    meta = view_meta.get("preprocess_meta", {})
    if meta.get("transform") != "raw_crop_pad_to_square":
        raise ValueError(f"Source scene must use raw_crop_pad_to_square, got {meta.get('transform')!r}")
    x0, y0, x1, y1 = [float(v) for v in meta["crop_bbox_xyxy"]]
    crop_width = max(1.0, x1 - x0)
    crop_height = max(1.0, y1 - y0)
    params = resize_params(int(round(crop_width)), int(round(crop_height)), int(target_size))
    uu, vv = np.meshgrid(np.arange(target_size, dtype=np.float32), np.arange(target_size, dtype=np.float32))
    inside = (
        (uu >= params["left"])
        & (uu < params["left"] + params["new_width"])
        & (vv >= params["top"])
        & (vv < params["top"] + params["new_height"])
    )
    raw_x = x0 + (uu - params["left"]) / max(params["scale_x"], 1e-8)
    raw_y = y0 + (vv - params["top"]) / max(params["scale_y"], 1e-8)
    return raw_x.astype(np.float32), raw_y.astype(np.float32), inside


def raw_to_base_pixels(
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    base_view_meta: dict[str, Any],
    target_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_size = (
        base_view_meta.get("source_image_size")
        or base_view_meta.get("original_source_image_size")
        or base_view_meta.get("image_size")
    )
    raw_width, raw_height = int(source_size[0]), int(source_size[1])
    full_params = resize_params(raw_width, raw_height, int(target_size))
    aligned_x = raw_x * full_params["scale_x"] + full_params["left"]
    aligned_y = raw_y * full_params["scale_y"] + full_params["top"]

    meta = base_view_meta.get("preprocess_meta", {})
    if meta.get("transform") != "crop_pad_to_square":
        raise ValueError(f"Base scene must use crop_pad_to_square, got {meta.get('transform')!r}")
    bx0, by0, bx1, by1 = [float(v) for v in meta["crop_bbox_xyxy"]]
    crop_width = max(1.0, bx1 - bx0)
    crop_height = max(1.0, by1 - by0)
    crop_params = resize_params(int(round(crop_width)), int(round(crop_height)), int(target_size))
    base_u = (aligned_x - bx0) * crop_params["scale_x"] + crop_params["left"]
    base_v = (aligned_y - by0) * crop_params["scale_y"] + crop_params["top"]
    target_u = np.rint(base_u).astype(np.int32)
    target_v = np.rint(base_v).astype(np.int32)
    inside = (
        (target_u >= 0)
        & (target_u < int(target_size))
        & (target_v >= 0)
        & (target_v < int(target_size))
        & (base_u >= crop_params["left"] - 0.5)
        & (base_u < crop_params["left"] + crop_params["new_width"] + 0.5)
        & (base_v >= crop_params["top"] - 0.5)
        & (base_v < crop_params["top"] + crop_params["new_height"] + 0.5)
    )
    return target_u, target_v, inside


def similarity_from_predictions(
    *,
    source_predictions: dict[str, np.ndarray],
    source_manifest: dict[str, Any],
    base_predictions: dict[str, np.ndarray],
    base_manifest: dict[str, Any],
    axis_scale: float,
    use_axes: bool,
) -> dict[str, Any]:
    source_ids = camera_ids(source_manifest)
    base_ids = camera_ids(base_manifest)
    source_index = {cam_id: idx for idx, cam_id in enumerate(source_ids)}
    source_extrinsic = np.asarray(source_predictions["extrinsic"], dtype=np.float64)
    base_extrinsic = np.asarray(base_predictions["extrinsic"], dtype=np.float64)

    source_corr: list[np.ndarray] = []
    base_corr: list[np.ndarray] = []
    per_camera: list[dict[str, Any]] = []
    for base_idx, cam_id in enumerate(base_ids):
        if cam_id not in source_index:
            continue
        source_idx = source_index[cam_id]
        src_center, src_rot = camera_center_and_rotation(source_extrinsic[source_idx])
        base_center, base_rot = camera_center_and_rotation(base_extrinsic[base_idx])
        source_corr.append(src_center)
        base_corr.append(base_center)
        per_camera.append({"camera_id": cam_id, "source_view_index": source_idx, "base_view_index": base_idx})

    if not source_corr:
        raise RuntimeError("No shared camera IDs between source and base scenes.")

    source_centers = np.stack(source_corr, axis=0)
    base_centers = np.stack(base_corr, axis=0)
    source_axis_len = float(axis_scale) * median_baseline(source_centers)
    base_axis_len = float(axis_scale) * median_baseline(base_centers)
    if use_axes:
        for item in per_camera:
            src_idx = int(item["source_view_index"])
            base_idx = int(item["base_view_index"])
            src_center, src_rot = camera_center_and_rotation(source_extrinsic[src_idx])
            base_center, base_rot = camera_center_and_rotation(base_extrinsic[base_idx])
            for axis_idx in range(3):
                source_corr.append(src_center + source_axis_len * src_rot[:, axis_idx])
                base_corr.append(base_center + base_axis_len * base_rot[:, axis_idx])

    source_arr = np.stack(source_corr, axis=0)
    base_arr = np.stack(base_corr, axis=0)
    scale, rotation, translation = estimate_similarity(source_arr, base_arr)
    mapped = scale * (source_arr @ rotation.T) + translation[None, :]
    residual = np.linalg.norm(mapped - base_arr, axis=1)
    return {
        "source_ids": source_ids,
        "base_ids": base_ids,
        "per_camera": per_camera,
        "scale": float(scale),
        "rotation": rotation.astype(np.float64),
        "translation": translation.astype(np.float64),
        "residual_percentiles": [float(v) for v in np.percentile(residual, [0, 25, 50, 75, 90, 95, 100])],
        "correspondence_count": int(source_arr.shape[0]),
        "used_axes": bool(use_axes),
        "axis_scale": float(axis_scale),
    }


def transform_points(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    flat = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    out = flat.copy()
    finite = np.isfinite(flat).all(axis=1)
    if finite.any():
        out[finite] = (float(scale) * (flat[finite].astype(np.float64) @ rotation.T) + translation[None, :]).astype(
            np.float32
        )
    return out.reshape(points.shape)


def world_to_camera_z(points_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = np.asarray(extrinsic[:3, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:3, 3], dtype=np.float32)
    cam = points_world.astype(np.float32) @ rotation.T + translation[None, :]
    return cam[:, 2]


def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        grown = np.zeros_like(out, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                grown |= padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
        out = grown
    return out


def recompute_normals(
    *,
    fused_world: np.ndarray,
    base: dict[str, np.ndarray],
    replace_mask: np.ndarray,
    confidence_boost: float,
    dilate: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    normal = np.asarray(base["normal"], dtype=np.float32).copy()
    normal_conf = np.asarray(base["normal_conf"], dtype=np.float32).copy()
    extrinsic = np.asarray(base["extrinsic"], dtype=np.float32)
    summary: dict[str, Any] = {"enabled": True, "per_view": {}}
    for view_idx in range(fused_world.shape[0]):
        seed = np.asarray(replace_mask[view_idx], dtype=bool)
        use_mask = dilate_mask(seed, int(dilate))
        rotation = extrinsic[view_idx, :3, :3]
        translation = extrinsic[view_idx, :3, 3]
        cam_points = np.einsum("ij,hwj->hwi", rotation, fused_world[view_idx].astype(np.float32)) + translation
        finite = np.isfinite(cam_points).all(axis=-1)
        normal_map, normal_valid = point_map_to_normal_numpy(cam_points, finite)
        use = use_mask & normal_valid
        mean_dot = 0.0
        flipped = False
        if use.any():
            dot = np.sum(normal[view_idx][use] * normal_map[use], axis=-1)
            mean_dot = float(np.nanmean(dot)) if dot.size else 0.0
            flipped = bool(mean_dot < 0.0)
            if flipped:
                normal_map = -normal_map
            normal[view_idx][use] = normal_map[use]
            normal_conf[view_idx][use] = np.maximum(normal_conf[view_idx][use], float(confidence_boost))
        summary["per_view"][str(view_idx)] = {
            "seed_pixels": int(seed.sum()),
            "candidate_pixels": int(use_mask.sum()),
            "normal_replaced_pixels": int(use.sum()),
            "mean_dot_before_optional_flip": float(mean_dot),
            "flipped_to_match_previous_convention": bool(flipped),
        }
    summary["normal_replaced_pixels_total"] = int(
        sum(row["normal_replaced_pixels"] for row in summary["per_view"].values())
    )
    return normal, normal_conf, summary


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


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = load_npz(args.base_predictions)
    source = load_npz(args.source_predictions)
    base_manifest = load_manifest(args.base_scene_dir)
    source_manifest = load_manifest(args.source_scene_dir)
    base_views = base_manifest["exported_views"]
    source_views = source_manifest["exported_views"]
    if len(base_views) != len(source_views):
        raise ValueError("Base/source view counts differ.")

    base_world = np.asarray(base["world_points"], dtype=np.float32)
    source_world = np.asarray(source["world_points"], dtype=np.float32)
    target_size = int(base_world.shape[1])
    if base_world.shape != source_world.shape:
        raise ValueError(f"Prediction shapes differ: {base_world.shape} vs {source_world.shape}")

    sim = similarity_from_predictions(
        source_predictions=source,
        source_manifest=source_manifest,
        base_predictions=base,
        base_manifest=base_manifest,
        axis_scale=float(args.axis_scale),
        use_axes=not bool(args.no_axes),
    )
    source_world_mapped = transform_points(
        source_world,
        float(sim["scale"]),
        np.asarray(sim["rotation"], dtype=np.float64),
        np.asarray(sim["translation"], dtype=np.float64),
    )

    base_masks = load_mask_stack(args.base_scene_dir / "masks", target_size=target_size).astype(bool)
    source_masks = load_mask_stack(args.source_scene_dir / "masks", target_size=target_size).astype(bool)
    base_roi_masks = load_2d_roi_mask_stack(args.base_scene_dir / "masks", target_size=target_size, roi=args.roi).astype(bool)
    source_conf = np.asarray(source["world_points_conf"], dtype=np.float32)
    base_conf = np.asarray(base["world_points_conf"], dtype=np.float32)
    base_depth = np.asarray(base["depth"], dtype=np.float32)[..., 0]
    fused_mask = np.zeros(base_world.shape[:3], dtype=bool)
    fused_world = base_world.copy()
    per_view: dict[str, Any] = {}

    for view_idx, (base_view, source_view) in enumerate(zip(base_views, source_views)):
        if str(base_view.get("camera_id")).zfill(2) != str(source_view.get("camera_id")).zfill(2):
            raise ValueError(f"View order mismatch at {view_idx}: {base_view.get('camera_id')} vs {source_view.get('camera_id')}")
        raw_x, raw_y, source_inside = source_pixels_to_raw_grid(source_view, target_size)
        target_u, target_v, base_inside = raw_to_base_pixels(raw_x, raw_y, base_view, target_size)
        source_valid_base = (
            source_inside
            & base_inside
            & source_masks[view_idx]
            & np.isfinite(source_world_mapped[view_idx]).all(axis=-1)
            & np.isfinite(source_conf[view_idx])
        )
        if source_valid_base.any():
            src_threshold = float(np.percentile(source_conf[view_idx][source_valid_base], float(args.source_conf_percentile)))
        else:
            src_threshold = float("inf")
        mapped_v = target_v[source_valid_base]
        mapped_u = target_u[source_valid_base]
        mapped_points = source_world_mapped[view_idx][source_valid_base]
        mapped_conf = source_conf[view_idx][source_valid_base]
        mapped_base_points = base_world[view_idx, mapped_v, mapped_u]
        mapped_base_conf = base_conf[view_idx, mapped_v, mapped_u]
        mapped_base_depth = base_depth[view_idx, mapped_v, mapped_u]
        mapped_base_roi = base_roi_masks[view_idx, mapped_v, mapped_u] & base_masks[view_idx, mapped_v, mapped_u]
        finite_base = np.isfinite(mapped_base_points).all(axis=-1) & np.isfinite(mapped_base_conf)
        if finite_base.any() and float(args.base_conf_percentile) > 0:
            base_threshold = float(np.percentile(mapped_base_conf[finite_base], float(args.base_conf_percentile)))
        else:
            base_threshold = 0.0

        world_distance = np.linalg.norm(mapped_points - mapped_base_points, axis=-1)
        source_z_in_base = world_to_camera_z(mapped_points, np.asarray(base["extrinsic"], dtype=np.float32)[view_idx])
        depth_delta = np.abs(source_z_in_base - mapped_base_depth)
        valid = (
            mapped_base_roi
            & finite_base
            & (mapped_conf >= src_threshold)
            & (mapped_base_conf >= base_threshold)
            & np.isfinite(world_distance)
            & np.isfinite(depth_delta)
            & (world_distance <= float(args.max_world_distance))
            & (depth_delta <= float(args.max_depth_delta))
        )

        chosen = np.zeros((target_size, target_size), dtype=bool)
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size:
            order = valid_indices[np.argsort(-mapped_conf[valid_indices])]
            for candidate_idx in order:
                y = int(mapped_v[candidate_idx])
                x = int(mapped_u[candidate_idx])
                if chosen[y, x]:
                    continue
                chosen[y, x] = True
                fused_world[view_idx, y, x] = (
                    base_world[view_idx, y, x] + float(args.alpha) * (mapped_points[candidate_idx] - base_world[view_idx, y, x])
                ).astype(np.float32)

        fused_mask[view_idx] = chosen
        per_view[str(view_idx)] = {
            "camera_id": str(base_view.get("camera_id")).zfill(2),
            "source_pixels_inside_base_crop": int(source_valid_base.sum()),
            "source_conf_threshold": float(src_threshold) if np.isfinite(src_threshold) else None,
            "base_conf_threshold": float(base_threshold),
            "valid_consensus_candidates": int(valid.sum()),
            "fused_pixels": int(chosen.sum()),
            "world_distance_percentiles_valid": [
                float(v) for v in np.percentile(world_distance[valid], [0, 25, 50, 75, 90, 95, 99])
            ]
            if valid.any()
            else [],
            "depth_delta_percentiles_valid": [
                float(v) for v in np.percentile(depth_delta[valid], [0, 25, 50, 75, 90, 95, 99])
            ]
            if valid.any()
            else [],
        }

    rotation = np.asarray(base["extrinsic"], dtype=np.float32)[:, :3, :3]
    translation = np.asarray(base["extrinsic"], dtype=np.float32)[:, :3, 3]
    fused_cam = np.einsum("vij,vhwj->vhwi", rotation, fused_world.astype(np.float32)) + translation[:, None, None, :]
    fused_depth = np.asarray(base["depth"], dtype=np.float32).copy()
    depth_valid = fused_mask & np.isfinite(fused_cam[..., 2]) & (fused_cam[..., 2] > 1e-6)
    fused_depth[depth_valid, 0] = fused_cam[..., 2][depth_valid]

    world_conf = base_conf.copy()
    depth_conf = np.asarray(base["depth_conf"], dtype=np.float32).copy()
    world_conf[fused_mask] = np.maximum(world_conf[fused_mask], float(args.confidence_boost))
    depth_conf[depth_valid] = np.maximum(depth_conf[depth_valid], float(args.confidence_boost))

    out = dict(base)
    out["world_points"] = fused_world.astype(base["world_points"].dtype, copy=False)
    out["depth"] = fused_depth.astype(base["depth"].dtype, copy=False)
    out["world_points_conf"] = world_conf.astype(base["world_points_conf"].dtype, copy=False)
    out["depth_conf"] = depth_conf.astype(base["depth_conf"].dtype, copy=False)
    out["self_crop_consensus_mask"] = fused_mask.astype(np.uint8)

    normal_summary: dict[str, Any] = {"enabled": False}
    if "normal" in base and "normal_conf" in base:
        normal, normal_conf, normal_summary = recompute_normals(
            fused_world=fused_world,
            base=base,
            replace_mask=fused_mask,
            confidence_boost=float(args.normal_confidence_boost),
            dilate=int(args.normal_dilate),
        )
        out["normal"] = normal.astype(base["normal"].dtype, copy=False)
        out["normal_conf"] = normal_conf.astype(base["normal_conf"].dtype, copy=False)

    output_path = args.output_dir / "predictions.npz"
    np.savez_compressed(output_path, **out)
    summary = {
        "task": "fuse_self_crop_consensus_predictions",
        "truthful_status": "local_same_model_crop_consensus_candidate_not_teacher_not_pass",
        "base_predictions": str(args.base_predictions.resolve()),
        "base_scene_dir": str(args.base_scene_dir.resolve()),
        "source_predictions": str(args.source_predictions.resolve()),
        "source_scene_dir": str(args.source_scene_dir.resolve()),
        "output_predictions": str(output_path.resolve()),
        "roi": args.roi,
        "alpha": float(args.alpha),
        "source_conf_percentile": float(args.source_conf_percentile),
        "base_conf_percentile": float(args.base_conf_percentile),
        "max_world_distance": float(args.max_world_distance),
        "max_depth_delta": float(args.max_depth_delta),
        "confidence_boost": float(args.confidence_boost),
        "fused_pixels_total": int(fused_mask.sum()),
        "fused_pixels_per_view": [int(fused_mask[idx].sum()) for idx in range(fused_mask.shape[0])],
        "similarity": {
            "scale": sim["scale"],
            "residual_percentiles": sim["residual_percentiles"],
            "correspondence_count": sim["correspondence_count"],
            "used_axes": sim["used_axes"],
            "axis_scale": sim["axis_scale"],
        },
        "normal_recompute": normal_summary,
        "per_view": per_view,
        "warning": (
            "This file is a postprocess candidate from same-model crop consensus. "
            "It can pass only through package_normal_candidate_gate.py with explicit Open3D visual review."
        ),
    }
    (args.output_dir / "self_crop_consensus_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
