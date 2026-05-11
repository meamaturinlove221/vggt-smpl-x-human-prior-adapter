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
    json_ready,
    load_ply_xyz_rgb,
    load_template,
    region_masks_from_template,
    scalar_stats,
    select_region_by_bbox,
    write_ascii_ply,
    write_json,
    write_report,
)


DEFAULT_30K = Path("output/surface_research_cloud_preflight/Cloud_G_V10/a5x3_2dgs_colmap_scene_30k/model_smoke/point_cloud/iteration_30000/point_cloud.ply")
G3_SUMMARY = LOCAL_ROOT / "V11_G3_2DGS_surface_anchor/summary.json"
OUT = LOCAL_ROOT / "V13_G1_2DGS_strict_surface"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _trim(points: np.ndarray, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    finite = np.isfinite(points).all(axis=1)
    pts = points[finite]
    cols = colors[finite] if len(colors) == len(points) else np.full((int(finite.sum()), 3), 210, np.uint8)
    if len(pts) == 0:
        return pts, cols, {"input": int(len(points)), "kept": 0}
    lo = np.percentile(pts, 0.5, axis=0)
    hi = np.percentile(pts, 99.5, axis=0)
    keep = np.logical_and(pts >= lo, pts <= hi).all(axis=1)
    pts = pts[keep]
    cols = cols[keep]
    return pts, cols, {"input": int(len(points)), "finite": int(finite.sum()), "kept": int(len(pts)), "p005": lo, "p995": hi}


def _region_rows(points: np.ndarray) -> dict[str, Any]:
    template = load_template()
    masks = region_masks_from_template(template)
    tpts = template["hybrid_vertices"].astype(np.float32)
    rows = {}
    for name in REGIONS:
        fallback = (0.65, 1.0) if name in {"head", "face_core", "hairline"} else (0.0, 1.0)
        mask = select_region_by_bbox(points, tpts[masks[name]], fallback_fraction=fallback)
        threshold = 500 if name == "full_body" else 50
        rows[name] = {
            "point_count": int(mask.sum()),
            "coverage_nonempty": bool(mask.sum() >= threshold),
            "bbox": bbox_stats(points[mask]),
        }
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="V13 G1 2DGS strict surface extraction audit.")
    parser.add_argument("--input-ply", type=Path, default=DEFAULT_30K)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    pts, cols = load_ply_xyz_rgb(args.input_ply, max_points=None)
    trimmed, trimmed_cols, trim_info = _trim(pts, cols)
    write_ascii_ply(out / "g1_2dgs_strict_surface_trimmed.ply", trimmed, trimmed_cols)
    contact_sheet(trimmed, trimmed_cols, out / "g1_2dgs_strict_surface_open3d_contact_sheet.png", "G1 2DGS")
    rows = _region_rows(trimmed)
    g3_summary = {}
    if G3_SUMMARY.is_file():
        g3_summary = json.loads(G3_SUMMARY.read_text(encoding="utf-8"))
    # Use G3's existing 6-view projection summary as provenance, but do not inherit pass blindly.
    projection_rows = g3_summary.get("selected_tier", {}).get("projection_rows") or g3_summary.get("projection_rows") or []
    mean_iou = float(np.mean([float(r.get("mask_iou", 0.0)) for r in projection_rows])) if projection_rows else float(g3_summary.get("mean_6view_mask_iou", 0.0) or 0.0)
    gates = {
        "not_sparse_points": len(trimmed) > 50000,
        "not_floating_noise": bool(trim_info.get("kept", 0) / max(trim_info.get("finite", 1), 1) > 0.75),
        "full_body_visual_pass": rows["full_body"]["coverage_nonempty"],
        "head_visual_pass": rows["head"]["coverage_nonempty"],
        "face_visual_pass": rows["face_core"]["coverage_nonempty"],
        "teacher_or_surface_anchor_pass": bool(len(trimmed) > 50000 and rows["full_body"]["coverage_nonempty"] and mean_iou > 0.25),
        "strict_teacher_precheck_pass": False,
    }
    blockers = []
    if mean_iou <= 0.25:
        blockers.append(f"2DGS/G3 mean 6-view IoU is too low for strict surface teacher: {mean_iou:.4f}.")
    if not gates["teacher_or_surface_anchor_pass"]:
        blockers.append("2DGS remains research anchor/weak-pool, not strict teacher.")
    summary = {
        "task": "v13_g1_2dgs_strict_surface_extractor",
        "created_utc": utc_now(),
        "status": "g1_2dgs_surface_anchor_ready_research_only" if gates["teacher_or_surface_anchor_pass"] else "g1_2dgs_strict_surface_blocked",
        "input_ply": args.input_ply,
        "trim": trim_info,
        "artifacts": {"surface": out / "g1_2dgs_strict_surface_trimmed.ply", "contact_sheet": out / "g1_2dgs_strict_surface_open3d_contact_sheet.png"},
        "region_coverage": rows,
        "mean_6view_mask_iou": mean_iou,
        "gates": gates,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "decision": "2DGS can serve as research surface anchor but does not pass strict teacher precheck." if not gates["strict_teacher_precheck_pass"] else "2DGS strict precheck passed.",
        "blockers": blockers,
    }
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v13_g1_2dgs_strict_surface_extractor.json", summary)
    write_report(REPORTS / "20260508_v13_g1_2dgs_strict_surface_extractor.md", "V13 G1 2DGS Strict Surface Extractor", summary)
    print(json.dumps(json_ready({"status": summary["status"], "points": len(trimmed), "mean_iou": mean_iou, "output": out}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
