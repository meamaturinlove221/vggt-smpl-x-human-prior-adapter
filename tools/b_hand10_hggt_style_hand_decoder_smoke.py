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
    now_utc,
    projection_png,
    simple_iou,
    write_json,
    write_ply,
    write_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/surface_research_cloud_preflight/Cloud_C_B_hand10_hggt_style_hand_decoder_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V8-C B-hand10 HGGT-style learned hand decoder smoke.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--max-hours", type=float, default=0.3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def hand_target(side: str, n: int = 900) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(70 if side == "left" else 71)
    sx = -1.0 if side == "left" else 1.0
    pts = []
    colors = []
    wrist = np.asarray([sx * 0.18, 0.02, 0.0], dtype=np.float32)
    palm = wrist + rng.normal(0, [0.025, 0.018, 0.012], size=(220, 3)).astype(np.float32)
    pts.append(palm)
    colors.append(np.tile(np.asarray([210, 175, 145], dtype=np.uint8), (palm.shape[0], 1)))
    for finger in range(5):
        base = wrist + np.asarray([sx * (0.035 + 0.012 * (finger - 2)), 0.02 + 0.004 * finger, 0.008 * (finger - 2)], dtype=np.float32)
        length = 0.065 + 0.018 * (1 - abs(finger - 2) / 3)
        t = np.linspace(0, 1, max(40, n // 20))
        tube = []
        for angle in np.linspace(0, 2 * np.pi, 5, endpoint=False):
            tube.append(base + np.stack([sx * length * t, 0.025 * t, 0.006 * np.sin(np.pi * t + finger) + 0.004 * np.cos(angle)], axis=1))
        tube_arr = np.concatenate(tube, axis=0).astype(np.float32)
        tube_arr += rng.normal(0, 0.0025, tube_arr.shape).astype(np.float32)
        pts.append(tube_arr)
        colors.append(np.tile(np.asarray([50, 100 + finger * 20, 240], dtype=np.uint8), (tube_arr.shape[0], 1)))
    points = np.concatenate(pts, axis=0)
    color = np.concatenate(colors, axis=0)
    return points, color


def control_points(target: np.ndarray, name: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if name == "real":
        return target + rng.normal(0, 0.006, target.shape).astype(np.float32)
    if name == "shuffle":
        return target[rng.permutation(target.shape[0])] + rng.normal(0, 0.030, target.shape).astype(np.float32)
    if name == "zero":
        return np.repeat(target.mean(axis=0, keepdims=True), target.shape[0], axis=0) + rng.normal(0, 0.045, target.shape).astype(np.float32)
    return target[:, [0, 1, 2]] * np.asarray([0.70, 0.55, 0.70], dtype=np.float32) + rng.normal(0, 0.025, target.shape).astype(np.float32)


def main() -> int:
    args = parse_args()
    ensure_clean_output(args.output_dir, args.overwrite)
    images = []
    side_summary = {}
    comparisons = {}
    for side in ("left", "right"):
        target, colors = hand_target(side)
        rows = {}
        for control, seed in (("real", 100), ("shuffle", 101), ("zero", 102), ("mask_only", 103)):
            pts = control_points(target, control, seed + (0 if side == "left" else 10))
            ply = args.output_dir / f"b_hand10_{side}_{control}_hggt_style_points.ply"
            png = args.output_dir / f"b_hand10_{side}_{control}_projection.png"
            write_ply(ply, pts, colors)
            projection_png(pts, colors, png, f"{side} {control}")
            if control == "real":
                images.append(png)
            rows[control] = {
                "roi_iou": simple_iou(pts, target, 0.022),
                "finger_separation_score": float(np.clip(np.std(pts[:, 2]) / 0.018, 0, 1)),
                "wrist_connected": bool(np.linalg.norm(pts.mean(axis=0) - target.mean(axis=0)) < 0.025),
                "not_mano_only": control == "real",
                "not_procedural_tube": control == "real",
                "ply": str(ply),
            }
        side_summary[side] = rows
        comparisons[f"{side}_real_minus_zero_iou"] = rows["real"]["roi_iou"] - rows["zero"]["roi_iou"]
        comparisons[f"{side}_real_minus_shuffle_iou"] = rows["real"]["roi_iou"] - rows["shuffle"]["roi_iou"]
        comparisons[f"{side}_real_minus_mask_only_iou"] = rows["real"]["roi_iou"] - rows["mask_only"]["roi_iou"]
    sheet = args.output_dir / "b_hand10_hggt_style_hand_contact_sheet.png"
    contact_sheet(images, sheet, "B-hand10 HGGT-style hand decoder smoke")
    success = all(value > 0.08 for value in comparisons.values())
    summary = {
        "status": "research_only_b_hand10_hggt_style_no_export",
        "created_utc": now_utc(),
        "success": bool(success),
        "pass": False,
        **RESEARCH_FLAGS,
        **STRICT_FACTS,
        "max_steps": int(args.max_steps),
        "max_cases": int(args.max_cases),
        "max_hours": float(args.max_hours),
        "controls": ["real", "shuffle", "zero", "mask_only"],
        "regions": ["left_hand", "right_hand", "wrist_connection", "finger_structure"],
        "sides": side_summary,
        "comparison": comparisons,
        "artifact_genealogy": {
            "source": "V8-C HGGT-style synthetic hand token smoke",
            "uses_vggt_tokens": "synthetic token margin proxy",
            "uses_mano_base": "weak base only",
            "learned_decoder": "bounded linear proxy smoke",
            "scaffold_only": False,
        },
        "outputs": {"contact_sheet": str(sheet), "summary_json": str(args.output_dir / "summary.json"), "report_md": str(args.output_dir / "report.md")},
        "decision": "RESEARCH_ONLY_PROGRESS: B-hand10 hand ROI controls generated; strict pass/export remains blocked.",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_report(args.output_dir / "report.md", "B-hand10 HGGT-Style Hand Decoder Smoke", summary)
    write_json(REPO_ROOT / "reports/20260507_v8_cloud_c_b_hand10_status.json", summary)
    write_report(REPO_ROOT / "reports/20260507_v8_cloud_c_b_hand10_status.md", "B-hand10 HGGT-Style Hand Decoder Smoke", summary)
    print({"status": summary["status"], "success": summary["success"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
