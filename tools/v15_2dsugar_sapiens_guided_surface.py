from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v15_common import (
    LOCAL_ROOT,
    REPORTS,
    json_ready,
    read_json,
    safe_v15_output_dir,
    utc_now,
    write_json,
    write_report,
)


def _scan_backend_roots(roots: list[Path]) -> dict[str, Any]:
    needles = ("2d-sugar", "2dsugar", "sugar")
    matches = []
    for root in roots:
        if not root.exists():
            continue
        try:
            for child in root.rglob("*"):
                low = child.name.lower()
                if any(needle in low for needle in needles):
                    matches.append(str(child.resolve()))
        except Exception as exc:
            matches.append(f"scan_error:{root}:{exc}")
    return {"roots": [str(root.resolve()) for root in roots], "matches": sorted(set(matches))}


def main() -> int:
    parser = argparse.ArgumentParser(description="V15 2D-SuGaR/Sapiens guided surface feasibility audit.")
    parser.add_argument("--output-dir", type=Path, default=LOCAL_ROOT / "V15_GS_2DSuGaR_sapiens_guided_surface")
    parser.add_argument("--camera-remap-summary", type=Path, default=LOCAL_ROOT / "V15_GS_camera_view_contract_remap/summary.json")
    parser.add_argument("--normal-summary", type=Path, default=LOCAL_ROOT / "V15_GS_2DGS_true_normal_rasterizer/summary.json")
    parser.add_argument("--depth-summary", type=Path, default=LOCAL_ROOT / "V15_S_sapiens_metric_depth_calibration/summary.json")
    args = parser.parse_args()

    out = safe_v15_output_dir(args.output_dir)
    backend_scan = _scan_backend_roots([Path("external"), Path("external_models")])
    remap = read_json(args.camera_remap_summary)
    normal = read_json(args.normal_summary)
    depth = read_json(args.depth_summary)
    backend_found = bool(backend_scan.get("matches"))
    prerequisites = {
        "camera_remap_ready": str(remap.get("status", "")).endswith("research_only"),
        "two_dgs_normals_ready": str(normal.get("status", "")).endswith("research_only"),
        "sapiens_depth_fit_valid": bool(depth.get("gates", {}).get("global_fit_valid")),
        "two_d_sugar_backend_found": backend_found,
    }
    blockers = []
    if not backend_found:
        blockers.append("No local 2D-SuGaR/SuGaR backend checkout was found under external/ or external_models/.")
    if not prerequisites["camera_remap_ready"]:
        blockers.append("V15 camera remap did not produce a ready research bridge.")
    if not prerequisites["two_dgs_normals_ready"]:
        blockers.append("V15 2DGS normal raster was not ready.")
    if not prerequisites["sapiens_depth_fit_valid"]:
        blockers.append("Sapiens metric-depth calibration did not produce a valid global affine fit.")
    blockers.append("No guided surface was exported; this audit intentionally stops before fake 2D-SuGaR success.")
    summary = {
        "task": "v15_2dsugar_sapiens_guided_surface",
        "created_utc": utc_now(),
        "status": "v15_2dsugar_sapiens_guided_surface_blocked",
        "research_only": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "strict_teacher_passes": 0,
        "strict_candidate_passes": 0,
        "inputs": {
            "camera_remap_summary": str(args.camera_remap_summary.resolve()),
            "normal_summary": str(args.normal_summary.resolve()),
            "depth_summary": str(args.depth_summary.resolve()),
        },
        "metrics": {
            "backend_match_count": len(backend_scan.get("matches", [])),
            "camera_overlap_count": remap.get("metrics", {}).get("camera_id_overlap_count"),
            "normal_valid_pixels": normal.get("metrics", {}).get("normal_valid_pixels"),
            "depth_global_corr": depth.get("metrics", {}).get("global_corr"),
        },
        "backend_scan": backend_scan,
        "prerequisites": prerequisites,
        "decision": "2D-SuGaR/Sapiens-guided surface extraction is blocked on this local workspace; no surface export was produced.",
        "blockers": blockers,
    }
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v15_2dsugar_sapiens_guided_surface.json", summary)
    write_report(REPORTS / "20260508_v15_2dsugar_sapiens_guided_surface.md", "V15 2D-SuGaR Sapiens Guided Surface", summary)
    print(json.dumps(json_ready({"status": summary["status"], "metrics": summary["metrics"], "output": out}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
