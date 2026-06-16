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

from normal_line_multiview_eval import build_roi_masks, load_scene_view  # noqa: E402
from vggt.utils.normal_refiner import point_map_to_normal_numpy, points_world_to_camera  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch a VGGT predictions.npz with a bounded residual from another prediction bundle. "
            "This is a local teacher/candidate gate: confidence is preserved by default, and "
            "depth is recomputed from patched world points."
        )
    )
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--source-predictions", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--roi", choices=("face", "head", "full"), default="face")
    parser.add_argument(
        "--view-indices",
        default="",
        help="Comma-separated view indices to patch. Empty means all views.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min-displacement", type=float, default=1e-5)
    parser.add_argument("--max-displacement", type=float, default=0.02)
    parser.add_argument(
        "--update-normal",
        action="store_true",
        help="Replace patched pixels with the predicted-normal convention derived from the patched point map.",
    )
    parser.add_argument(
        "--copy-source-conf",
        action="store_true",
        help="Copy source confidence on patched pixels. Default preserves base confidence for fixed-threshold audits.",
    )
    return parser.parse_args()


def parse_view_indices(spec: str, view_count: int) -> np.ndarray:
    selected = np.zeros((view_count,), dtype=bool)
    if not spec.strip():
        selected[:] = True
        return selected
    for piece in spec.split(","):
        item = piece.strip()
        if not item:
            continue
        index = int(item)
        if index < 0 or index >= view_count:
            raise ValueError(f"view index {index} outside [0, {view_count})")
        selected[index] = True
    if not selected.any():
        raise ValueError("--view-indices did not select any views")
    return selected


def load_roi_stack(scene_dir: Path, view_count: int, height: int, width: int, roi: str) -> np.ndarray:
    masks: list[np.ndarray] = []
    for view_idx in range(view_count):
        scene = load_scene_view(scene_dir, view_idx, (height, width))
        masks.append(build_roi_masks(scene.mask.astype(bool))[roi])
    return np.stack(masks, axis=0)


def world_to_depth(world_points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    camera_points = np.empty_like(world_points, dtype=np.float32)
    for view_idx in range(world_points.shape[0]):
        camera_points[view_idx] = points_world_to_camera(world_points[view_idx], extrinsic[view_idx])
    return camera_points[..., 2].astype(np.float32), camera_points


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1)
    valid = np.isfinite(values).all(axis=-1) & (norms > eps)
    out = np.zeros_like(values, dtype=np.float32)
    out[valid] = values[valid] / norms[valid, None]
    return out, valid


def update_normals_from_points(
    normal: np.ndarray,
    patched_world_points: np.ndarray,
    extrinsic: np.ndarray,
    patch_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    updated = normal.copy()
    _, camera_points = world_to_depth(patched_world_points, extrinsic)
    finite = np.isfinite(camera_points).all(axis=-1)
    for view_idx in range(camera_points.shape[0]):
        point_normal, surface_valid = point_map_to_normal_numpy(camera_points[view_idx], finite[view_idx])
        point_normal, vector_valid = normalize_vectors(point_normal)
        valid = patch_mask[view_idx] & surface_valid & vector_valid
        # Raw point-map normals follow the derived-normal convention found in the
        # sign audit; VGGT predicted normals are opposite that winding.
        updated[view_idx][valid] = -point_normal[valid]
    updated, valid_vectors = normalize_vectors(updated)
    return updated, int((patch_mask & valid_vectors).sum())


def main() -> int:
    args = parse_args()
    base_path = Path(args.base_predictions)
    source_path = Path(args.source_predictions)
    scene_dir = Path(args.scene_dir)
    output_npz = Path(args.output_npz)
    output_summary = Path(args.output_summary) if args.output_summary else output_npz.with_suffix(".json")

    base = np.load(base_path, allow_pickle=False)
    source = np.load(source_path, allow_pickle=False)
    required = ("world_points", "depth", "normal", "extrinsic")
    for key in required:
        if key not in base.files or key not in source.files:
            raise KeyError(f"Both bundles must contain {key}")

    world_points = np.asarray(base["world_points"], dtype=np.float32).copy()
    source_world = np.asarray(source["world_points"], dtype=np.float32)
    if source_world.shape != world_points.shape:
        raise ValueError(f"source world_points shape {source_world.shape} != base {world_points.shape}")

    view_count, height, width, _ = world_points.shape
    selected_views = parse_view_indices(str(args.view_indices), view_count)
    roi_mask = load_roi_stack(scene_dir, view_count, height, width, str(args.roi))
    view_mask = selected_views[:, None, None]
    residual = source_world - world_points
    displacement = np.linalg.norm(residual, axis=-1)
    patch_mask = (
        roi_mask
        & view_mask
        & np.isfinite(residual).all(axis=-1)
        & (displacement >= float(args.min_displacement))
        & (displacement <= float(args.max_displacement))
    )

    alpha = float(args.alpha)
    world_points[patch_mask] = world_points[patch_mask] + alpha * residual[patch_mask]

    out: dict[str, Any] = {key: np.asarray(base[key]) for key in base.files}
    out["world_points"] = world_points.astype(np.float32)

    depth_z, _ = world_to_depth(world_points, np.asarray(base["extrinsic"], dtype=np.float32))
    depth = np.asarray(base["depth"], dtype=np.float32).copy()
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth[..., 0] = depth_z
    elif depth.ndim == 3:
        depth = depth_z
    else:
        raise ValueError(f"Unsupported depth shape {depth.shape}")
    out["depth"] = depth.astype(np.float32)

    normal_update_count = 0
    if bool(args.update_normal):
        normal, normal_update_count = update_normals_from_points(
            np.asarray(base["normal"], dtype=np.float32),
            world_points,
            np.asarray(base["extrinsic"], dtype=np.float32),
            patch_mask,
        )
        out["normal"] = normal.astype(np.float32)

    if bool(args.copy_source_conf):
        for key in ("world_points_conf", "depth_conf", "normal_conf"):
            if key in out and key in source.files:
                conf = np.asarray(out[key], dtype=np.float32).copy()
                conf[patch_mask] = np.asarray(source[key], dtype=np.float32)[patch_mask]
                out[key] = conf.astype(np.float32)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **out)

    selected_displacement = displacement[patch_mask]
    summary = {
        "base_predictions": str(base_path.resolve()),
        "source_predictions": str(source_path.resolve()),
        "scene_dir": str(scene_dir.resolve()),
        "output_npz": str(output_npz.resolve()),
        "roi": str(args.roi),
        "view_indices": [int(i) for i in np.flatnonzero(selected_views)],
        "alpha": alpha,
        "min_displacement": float(args.min_displacement),
        "max_displacement": float(args.max_displacement),
        "patch_pixels": int(patch_mask.sum()),
        "patch_pixels_by_view": [int(patch_mask[i].sum()) for i in range(view_count)],
        "displacement_mean": float(selected_displacement.mean()) if selected_displacement.size else 0.0,
        "displacement_p90": float(np.percentile(selected_displacement, 90.0)) if selected_displacement.size else 0.0,
        "displacement_max": float(selected_displacement.max()) if selected_displacement.size else 0.0,
        "depth_recomputed_from_world_points": True,
        "update_normal": bool(args.update_normal),
        "normal_update_count": normal_update_count,
        "copy_source_conf": bool(args.copy_source_conf),
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
