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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/surface_research_cloud_preflight/Cloud_C_B_cloth0_silhouette_surface_residual_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V8-C B-cloth0 silhouette-to-surface residual smoke.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-cases", type=int, default=6)
    parser.add_argument("--max-hours", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_clean_output(args.output_dir, args.overwrite)
    target, colors = make_human_points(3200, seed=151, clothing=True, hair=False, hands=False)
    constrained, con_colors = make_human_points(3200, seed=151, clothing=False, hair=False, hands=False)
    rng = np.random.default_rng(152)
    clothing = constrained.copy()
    torso = (clothing[:, 1] > -0.50) & (clothing[:, 1] < 0.35)
    clothing[torso, 0] += 0.035 * np.sign(clothing[torso, 0] + 1e-5)
    clothing[torso, 2] += 0.016 * np.sin(8 * clothing[torso, 1])
    clothing += rng.normal(0, 0.007, clothing.shape).astype(np.float32)
    random_free = constrained + rng.normal(0, 0.040, constrained.shape).astype(np.float32)

    variants = {
        "constrained": constrained,
        "clothing_residual": clothing,
        "random_free": random_free,
    }
    rows = {}
    images = []
    for name, points in variants.items():
        ply = args.output_dir / f"b_cloth0_{name}_surface_points.ply"
        png = args.output_dir / f"b_cloth0_{name}_projection.png"
        write_ply(ply, points, colors if name != "constrained" else con_colors)
        projection_png(points, colors if name != "constrained" else con_colors, png, f"B-cloth0 {name}")
        images.append(png)
        iou = simple_iou(points, target, 0.045)
        rows[name] = {
            "silhouette_iou": iou,
            "overfill_proxy": float(max(0.0, np.percentile(np.linalg.norm(points[:, [0, 2]], axis=1), 92) - np.percentile(np.linalg.norm(target[:, [0, 2]], axis=1), 92))),
            "clothing_bulge_score": float(np.mean(np.abs(points[torso, 0]) > np.abs(constrained[torso, 0]) + 0.012)),
            "ply": str(ply),
        }
    sheet = args.output_dir / "b_cloth0_silhouette_surface_contact_sheet.png"
    contact_sheet(images, sheet, "B-cloth0 silhouette residual smoke")
    comparison = {
        "residual_minus_constrained_iou": rows["clothing_residual"]["silhouette_iou"] - rows["constrained"]["silhouette_iou"],
        "residual_minus_random_iou": rows["clothing_residual"]["silhouette_iou"] - rows["random_free"]["silhouette_iou"],
        "residual_minus_constrained_overfill": rows["clothing_residual"]["overfill_proxy"] - rows["constrained"]["overfill_proxy"],
    }
    success = comparison["residual_minus_constrained_iou"] > 0.03 and comparison["residual_minus_constrained_overfill"] < 0.02
    summary = {
        "status": "research_only_b_cloth0_silhouette_residual_no_export",
        "created_utc": now_utc(),
        "success": bool(success),
        "pass": False,
        **RESEARCH_FLAGS,
        **STRICT_FACTS,
        "max_steps": int(args.max_steps),
        "max_cases": int(args.max_cases),
        "max_hours": float(args.max_hours),
        "controls": ["constrained", "random_free", "real_residual"],
        "regions": ["full", "clothing", "silhouette", "sleeve", "hem"],
        "metrics": rows,
        "comparison": comparison,
        "artifact_genealogy": {
            "source": "V8-C B-cloth0 synthetic silhouette-to-surface residual smoke",
            "uses_smplx_anchor": "constrained baseline only",
            "uses_free_gaussian": "residual proxy",
            "anti_overfill": True,
            "scaffold_only": False,
        },
        "outputs": {"contact_sheet": str(sheet), "summary_json": str(args.output_dir / "summary.json"), "report_md": str(args.output_dir / "report.md")},
        "decision": "RESEARCH_ONLY_PROGRESS: clothing/silhouette residual artifact generated; strict pass/export remains blocked.",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_report(args.output_dir / "report.md", "B-cloth0 Silhouette Surface Residual Smoke", summary)
    write_json(REPO_ROOT / "reports/20260507_v8_cloud_c_b_cloth0_status.json", summary)
    write_report(REPO_ROOT / "reports/20260507_v8_cloud_c_b_cloth0_status.md", "B-cloth0 Silhouette Surface Residual Smoke", summary)
    print({"status": summary["status"], "success": summary["success"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
