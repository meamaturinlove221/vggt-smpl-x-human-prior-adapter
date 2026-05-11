from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_kinect_depth_teacher_targets import (  # noqa: E402
    connected_component_stats,
    json_ready,
    load_json,
    load_mask,
    load_rgb_camera_params,
    make_overlay,
    project_zbuffer,
    resolve_kinect_smc,
    roi_mask_from_human,
    select_kinect_cameras,
    unproject_kinect_cloud,
)
from tools.dna_4k4d import normalize_camera_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project raw 4K4D Kinect depth into each exported RGB scene view and save "
            "per-view camera-space depth maps. This avoids fitting Kinect into VGGT "
            "world coordinates and is only a local diagnostic teacher source."
        )
    )
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--kinect-smc", type=Path, default=None)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--roi-kind", choices=("all", "head", "face", "face_core", "head_face"), default="all")
    parser.add_argument("--kinect-cameras", nargs="*", default=["all"], help="Kinect camera ids, or 'all'.")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--min-depth-m", type=float, default=0.4)
    parser.add_argument("--max-depth-m", type=float, default=6.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty. Re-run with --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_manifest = load_json(scene_dir / "scene_manifest.json")
    frame = int(args.frame if args.frame is not None else scene_manifest.get("frame_id", 0))
    kinect_smc = resolve_kinect_smc(scene_manifest, args.kinect_smc)
    with h5py.File(kinect_smc, "r") as handle:
        selected_cameras = select_kinect_cameras(handle, list(args.kinect_cameras))

    rgb_params = load_rgb_camera_params(scene_manifest)
    kinect_world, kinect_camera_ids, kinect_stats = unproject_kinect_cloud(
        kinect_smc,
        frame,
        depth_scale=float(args.depth_scale),
        min_depth_m=float(args.min_depth_m),
        max_depth_m=float(args.max_depth_m),
        selected_cameras=selected_cameras,
    )

    first_mask = load_mask(Path(scene_manifest["exported_views"][0]["mask_path"]), 518)
    height, width = first_mask.shape
    depth = np.zeros((len(scene_manifest["exported_views"]), height, width), dtype=np.float32)
    mask = np.zeros_like(depth, dtype=bool)
    roi_stack = np.zeros_like(depth, dtype=bool)
    real_world = np.zeros((len(scene_manifest["exported_views"]), height, width, 3), dtype=np.float32)
    real_view_cam = np.zeros_like(real_world)
    per_view: list[dict[str, Any]] = []

    overlay_dir = output_dir / "overlays"
    for view_idx, view in enumerate(scene_manifest["exported_views"]):
        camera_id = normalize_camera_id(view["camera_id"])
        view_mask = load_mask(Path(view["mask_path"]), 518)
        roi = roi_mask_from_human(view_mask, str(args.roi_kind))
        world_map, cam_map, hit_mask = project_zbuffer(kinect_world, rgb_params[camera_id], roi)
        view_depth = cam_map[..., 2]
        valid = hit_mask & roi & np.isfinite(view_depth) & (view_depth > 0.0)

        depth[view_idx] = np.where(valid, view_depth, 0.0).astype(np.float32)
        mask[view_idx] = valid
        roi_stack[view_idx] = roi
        real_world[view_idx] = world_map.astype(np.float32)
        real_view_cam[view_idx] = cam_map.astype(np.float32)
        make_overlay(
            Path(view["image_path"]),
            roi,
            valid,
            overlay_dir / f"{view_idx:02d}_cam{camera_id}_{args.roi_kind}_kinect_view_depth_hits.png",
        )

        depth_values = depth[view_idx][valid]
        per_view.append(
            {
                "view_index": int(view_idx),
                "camera_id": camera_id,
                "roi_pixels": int(roi.sum()),
                "hit_pixels": int(valid.sum()),
                "hit_ratio_in_roi": float(valid.sum() / max(int(roi.sum()), 1)),
                "hit_connected_components": connected_component_stats(valid),
                "depth_percentiles_m": [float(v) for v in np.percentile(depth_values, [0, 5, 50, 95, 100])]
                if depth_values.size
                else [],
            }
        )

    output_path = output_dir / "kinect_view_depth.npz"
    np.savez_compressed(
        output_path,
        depth=depth.astype(np.float32),
        mask=mask.astype(bool),
        roi_mask=roi_stack.astype(bool),
        real_world_points=real_world.astype(np.float32),
        real_view_cam_points=real_view_cam.astype(np.float32),
        kinect_camera_ids=kinect_camera_ids,
    )
    summary = {
        "task": "kinect_view_depth_npz",
        "truthful_status": "per_view_depth_teacher_diagnostic_not_final_pass",
        "scene_dir": str(scene_dir),
        "output_depth_npz": str(output_path),
        "kinect_smc": str(kinect_smc),
        "selected_kinect_cameras": selected_cameras,
        "frame": int(frame),
        "roi_kind": str(args.roi_kind),
        "kinect_stats": kinect_stats,
        "per_view": per_view,
        "notes": [
            "Depth is projected into each real RGB camera view before any VGGT-world fitting.",
            "This file is eligible only for local depth diagnostics; it is not a mentor-final point cloud.",
        ],
    }
    (output_dir / "kinect_view_depth_summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
