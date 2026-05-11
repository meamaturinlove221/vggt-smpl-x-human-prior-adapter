from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from v10_surface_completion_pipeline import LOCAL_ROOT, REPORTS, bbox_stats, json_ready, load_ply_xyz_rgb, read_summary, write_json, write_report


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    parser = argparse.ArgumentParser(description="V14 2DGS protocol alignment audit.")
    parser.add_argument("--g1-summary", type=Path, default=REPORTS / "20260508_v13_g1_2dgs_strict_surface_extractor.json")
    parser.add_argument("--g3-summary", type=Path, default=LOCAL_ROOT / "V11_G3_2DGS_surface_anchor/summary.json")
    parser.add_argument("--scene-dir", type=Path, default=Path("output/4k4d_scenes/0012_11_frame0000_12views_tmf"))
    parser.add_argument("--output-dir", type=Path, default=LOCAL_ROOT / "V14_G14_2DGS_protocol_alignment_audit")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    g1 = read_summary(args.g1_summary)
    g3 = read_summary(args.g3_summary)
    surface_path = Path(g1.get("artifacts", {}).get("surface", ""))
    pts, _ = load_ply_xyz_rgb(surface_path, max_points=200000) if surface_path.is_file() else ([], [])
    mean_iou = float(g1.get("mean_6view_mask_iou", 0.0) or 0.0)
    scene_manifest = read_summary(args.scene_dir / "scene_manifest.json")
    exported_views = scene_manifest.get("exported_views", [])
    image_shapes = []
    for view in exported_views:
        image_path = Path(view.get("image_path", ""))
        mask_path = Path(view.get("mask_path", ""))
        image_shapes.append(
            {
                "camera_id": view.get("camera_id"),
                "image_exists": image_path.is_file(),
                "mask_exists": mask_path.is_file(),
                "image": str(image_path),
                "mask": str(mask_path),
            }
        )
    blockers = []
    if mean_iou <= 0.01:
        blockers.append("G1/G3 projection evidence reports mean_6view_mask_iou near zero; coordinate/projection protocol is not established.")
    if not g3.get("selected_tier_payload") and not g3.get("projection_rows"):
        blockers.append("G3 summary lacks reusable projection rows for strict 6-view protocol validation.")
    summary = {
        "task": "v14_g14_2dgs_protocol_alignment_audit",
        "created_utc": utc_now(),
        "status": "g14_protocol_blocked",
        "inputs": {"g1_summary": str(args.g1_summary.resolve()), "g3_summary": str(args.g3_summary.resolve()), "surface": str(surface_path)},
        "surface_bbox": bbox_stats(pts) if len(pts) else {"valid": False},
        "point_count": int(len(pts)),
        "mean_6view_mask_iou": mean_iou,
        "scene_view_count": len(exported_views),
        "scene_views": image_shapes,
        "strict_teacher_precheck_pass": False,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "decision": "2DGS remains blocked at protocol alignment; do not retrain or promote until projection protocol explains mean IoU near zero.",
        "blockers": blockers,
    }
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v14_2dgs_protocol_alignment_audit.json", summary)
    write_report(REPORTS / "20260508_v14_2dgs_protocol_alignment_audit.md", "V14 2DGS Protocol Alignment Audit", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": out}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
