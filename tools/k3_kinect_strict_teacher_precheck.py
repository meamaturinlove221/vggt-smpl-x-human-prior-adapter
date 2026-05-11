from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from v10_surface_completion_pipeline import (
    LOCAL_ROOT,
    REGIONS,
    REPORTS,
    bbox_stats,
    contact_sheet,
    load_colmap_cameras,
    load_mask,
    load_ply_xyz_rgb,
    make_projection_mask,
    mask_iou,
    read_summary,
    region_masks_from_template,
    load_template,
    scalar_stats,
    select_region_by_bbox,
    write_json,
    write_report,
    json_ready,
)


SCENE = Path("output/4k4d_scenes/0012_11_frame0000_12views_tmf")
K2 = LOCAL_ROOT / "V13_K2_kinect_alignment_candidate_search"
OUT = LOCAL_ROOT / "V13_K3_kinect_strict_teacher_precheck"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _projection_rows(points: np.ndarray, scene: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    cams = load_colmap_cameras(scene)
    rows = []
    depths = {}
    eval_pts = points
    if len(eval_pts) > 120000:
        rng = np.random.default_rng(1331)
        eval_pts = eval_pts[rng.choice(len(eval_pts), 120000, replace=False)]
    for cam in cams[:6]:
        size = (int(cam["width"]), int(cam["height"]))
        mask_path = scene / "masks" / cam["name"]
        gt = load_mask(mask_path, size)
        pred, depth = make_projection_mask(eval_pts, cam, size, radius=2)
        finite = depth[np.isfinite(depth)]
        rows.append(
            {
                "view": cam["name"],
                "mask_iou": mask_iou(pred, gt),
                "pred_pixels": int(pred.sum()),
                "mask_pixels": int(gt.sum()),
                "depth_stats": scalar_stats(finite),
            }
        )
        depths[cam["name"]] = depth
    return rows, depths


def _region_coverage(points: np.ndarray) -> dict[str, Any]:
    template = load_template()
    masks = region_masks_from_template(template)
    tpts = template["hybrid_vertices"].astype(np.float32)
    rows = {}
    for name in REGIONS:
        fallback = (0.65, 1.0) if name in {"head", "face_core", "hairline"} else (0.0, 1.0)
        mask = select_region_by_bbox(points, tpts[masks[name]], fallback_fraction=fallback)
        rows[name] = {
            "point_count": int(mask.sum()),
            "coverage_nonempty": bool(mask.sum() > (100 if name == "full_body" else 20)),
            "bbox": bbox_stats(points[mask]),
        }
    return rows


def _save_depth_npz(depths: dict[str, np.ndarray], out: Path) -> dict[str, Path]:
    names = np.asarray(list(depths.keys()))
    arr = np.stack([depths[name] for name in depths], axis=0) if depths else np.zeros((0, 1, 1), np.float32)
    normals = np.zeros((*arr.shape, 3), dtype=np.float32)
    visibility = np.isfinite(arr)
    paths = {
        "depth": out / "kinect_teacher_depth_6view.npz",
        "normal": out / "kinect_teacher_normal_6view.npz",
        "visibility": out / "kinect_teacher_visibility_6view.npz",
    }
    np.savez_compressed(paths["depth"], view_names=names, depth=arr, research_only=True)
    np.savez_compressed(paths["normal"], view_names=names, normal=normals, research_only=True)
    np.savez_compressed(paths["visibility"], view_names=names, visibility=visibility, research_only=True)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="V13 K3 Kinect strict teacher precheck.")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    k2 = read_summary(K2 / "summary.json")
    surface = Path(k2.get("best_surface", K2 / "k2_best_kinect_tsdf_vggt_world.ply"))
    points, colors = load_ply_xyz_rgb(surface, max_points=180000)
    proj_rows, depths = _projection_rows(points, SCENE)
    depth_paths = _save_depth_npz(depths, out)
    regions = _region_coverage(points)
    contact_sheet(points, colors, out / "kinect_teacher_open3d_contact_sheet.png", "K3 Kinect precheck")
    mean_iou = float(np.mean([row["mask_iou"] for row in proj_rows])) if proj_rows else 0.0
    region_ok = all(regions[name]["coverage_nonempty"] for name in REGIONS)
    residual_ok = bool(k2.get("strict_alignment_candidate_ready"))
    reproj_ok = mean_iou > 0.35
    strict_teacher_precheck = bool(residual_ok and reproj_ok and region_ok)
    blockers = []
    if not residual_ok:
        blockers.append("K2 strict alignment candidate did not pass residual threshold.")
    if not reproj_ok:
        blockers.append(f"6-view reprojection IoU too low: mean={mean_iou:.4f}, threshold=0.35.")
    if not region_ok:
        blockers.append("Full/head/face/hairline/hands region coverage is incomplete.")
    summary = {
        "task": "v13_k3_kinect_strict_teacher_precheck",
        "created_utc": utc_now(),
        "status": "k3_teacher_precheck_pass_research_only" if strict_teacher_precheck else "k3_teacher_precheck_blocked",
        "source_surface": surface,
        "projection_rows": proj_rows,
        "mean_6view_mask_iou": mean_iou,
        "region_coverage": regions,
        "artifacts": {
            "depth": depth_paths["depth"],
            "normal": depth_paths["normal"],
            "visibility": depth_paths["visibility"],
            "contact_sheet": out / "kinect_teacher_open3d_contact_sheet.png",
        },
        "gates": {
            "alignment_residual_pass": residual_ok,
            "sixview_reprojection_pass": reproj_ok,
            "region_coverage_pass": region_ok,
            "strict_teacher_precheck_pass": strict_teacher_precheck,
        },
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "decision": "Kinect TSDF may proceed to D-line strict teacher transaction." if strict_teacher_precheck else "Kinect TSDF is still blocked before strict teacher promotion.",
        "blockers": blockers,
    }
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v13_k3_kinect_strict_teacher_precheck.json", summary)
    write_report(REPORTS / "20260508_v13_k3_kinect_strict_teacher_precheck.md", "V13 K3 Kinect Strict Teacher Precheck", summary)
    print(json.dumps(json_ready({"status": summary["status"], "mean_iou": mean_iou, "output": out}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
