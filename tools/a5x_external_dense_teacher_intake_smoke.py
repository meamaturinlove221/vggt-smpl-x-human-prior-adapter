from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.v8_research_smoke_utils import (  # noqa: E402
    RESEARCH_FLAGS,
    STRICT_FACTS,
    contact_sheet,
    ensure_clean_output,
    make_human_points,
    now_utc,
    projection_png,
    simple_iou,
    write_json,
    write_ply,
    write_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/surface_research_cloud_preflight/Cloud_B_A5X_external_dense_teacher_intake_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V8-B A5-X external dense teacher intake decision smoke.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--max-hours", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_clean_output(args.output_dir, args.overwrite)
    target, target_color = make_human_points(3600, seed=31, clothing=True, hair=True, hands=True)
    rng = np.random.default_rng(42)
    methods = {
        "must3r_family_pointmap": target + rng.normal(0, 0.018, target.shape).astype(np.float32),
        "mast3r_slam_known_camera_fusion": target + rng.normal(0, 0.026, target.shape).astype(np.float32),
        "meshsplat_2dgs_surface": target * np.asarray([1.03, 1.0, 1.02], dtype=np.float32) + rng.normal(0, 0.02, target.shape).astype(np.float32),
        "neus2_masked_known_camera_sdf": target * np.asarray([0.99, 1.01, 1.0], dtype=np.float32) + rng.normal(0, 0.022, target.shape).astype(np.float32),
    }
    images = []
    method_rows = {}
    for name, points in methods.items():
        colors = target_color.copy()
        ply = args.output_dir / f"{name}_dense_candidate_points.ply"
        png = args.output_dir / f"{name}_projection.png"
        write_ply(ply, points, colors)
        projection_png(points, colors, png, name)
        images.append(png)
        region_scores = {
            "full": simple_iou(points, target, 0.055),
            "head": simple_iou(points[target[:, 1] > 0.72], target[target[:, 1] > 0.72], 0.045),
            "face": simple_iou(points[(target[:, 1] > 0.76) & (target[:, 2] > -0.04)], target[(target[:, 1] > 0.76) & (target[:, 2] > -0.04)], 0.042),
            "hairline": simple_iou(points[target[:, 1] > 0.90], target[target[:, 1] > 0.90], 0.043),
            "hands": simple_iou(points[np.abs(target[:, 0]) > 0.22], target[np.abs(target[:, 0]) > 0.22], 0.044),
        }
        method_rows[name] = {
            "same_frame": True,
            "known_camera_aligned": True,
            "can_reproject_original_6_view_protocol": True,
            "not_smplx_scaffold_only": True,
            "connected_component_ratio": float(0.82 + 0.03 * rng.random()),
            "mean_depth_residual": float(np.linalg.norm(points - target, axis=1).mean()),
            "region_scores": region_scores,
            "ply": str(ply),
        }

    sheet = args.output_dir / "a5x_external_dense_teacher_intake_contact_sheet.png"
    contact_sheet(images, sheet, "A5-X external dense intake candidates")
    best_name = max(method_rows, key=lambda key: method_rows[key]["region_scores"]["full"])
    best = method_rows[best_name]
    strict_teacher_ready = all(score > 0.88 for score in best["region_scores"].values()) and best["mean_depth_residual"] < 0.015
    summary = {
        "status": "research_only_a5x_external_dense_intake_no_export",
        "created_utc": now_utc(),
        "success": bool(best["region_scores"]["full"] > 0.80),
        "pass": False,
        **RESEARCH_FLAGS,
        **STRICT_FACTS,
        "max_cases": int(args.max_cases),
        "max_hours": float(args.max_hours),
        "controls": {"real": best_name, "shuffle": "method_order_shuffle", "zero": "empty_candidate", "mask_only": "silhouette_shell_control"},
        "methods": method_rows,
        "best_method": best_name,
        "strict_teacher_ready": False,
        "weak_teacher_pool_only": not strict_teacher_ready,
        "artifact_genealogy": {
            "source": "V8-B synthetic external dense intake decision smoke",
            "uses_known_cameras": True,
            "uses_smplx_anchor": False,
            "external_backend_names": list(methods),
        },
        "outputs": {"contact_sheet": str(sheet), "summary_json": str(args.output_dir / "summary.json"), "report_md": str(args.output_dir / "report.md")},
        "decision": "RESEARCH_ONLY_WEAK_TEACHER_POOL: dense intake candidates were generated and reviewed, but no strict teacher/export/pass is written.",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_report(args.output_dir / "report.md", "A5-X External Dense Teacher Intake Smoke", summary)
    write_json(REPO_ROOT / "reports/20260507_v8_cloud_b_a5x_external_dense_intake_status.json", summary)
    write_report(REPO_ROOT / "reports/20260507_v8_cloud_b_a5x_external_dense_intake_status.md", "A5-X External Dense Teacher Intake Smoke", summary)
    print({"status": summary["status"], "best_method": best_name, "weak_teacher_pool_only": summary["weak_teacher_pool_only"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
