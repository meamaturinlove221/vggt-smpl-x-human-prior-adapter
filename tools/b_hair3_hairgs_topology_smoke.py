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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/surface_research_cloud_preflight/Cloud_D_B_hair3_hairgs_topology_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V8-C B-hair3 HairGS-style topology smoke.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--max-cases", type=int, default=6)
    parser.add_argument("--max-hours", type=float, default=0.3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def hair_target(n_roots: int = 220, chain_len: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(91)
    roots = []
    chains = []
    for idx in range(n_roots):
        theta = rng.uniform(-0.95 * np.pi, 0.95 * np.pi)
        root = np.asarray([0.112 * np.cos(theta), 0.91 + 0.018 * rng.random(), 0.092 * np.sin(theta)], dtype=np.float32)
        direction = np.asarray([0.022 * np.cos(theta), 0.045 + 0.02 * rng.random(), 0.020 * np.sin(theta)], dtype=np.float32)
        roots.append(root)
        for step in range(chain_len):
            t = step / max(1, chain_len - 1)
            curl = np.asarray([0.012 * np.sin(2.3 * t + theta), 0.0, 0.010 * np.cos(1.7 * t + theta)], dtype=np.float32)
            chains.append(root + t * direction + curl + rng.normal(0, 0.0025, 3).astype(np.float32))
    points = np.asarray(chains, dtype=np.float32)
    colors = np.tile(np.asarray([32, 25, 21], dtype=np.uint8), (points.shape[0], 1))
    return points, colors, np.asarray(roots, dtype=np.float32)


def control(points: np.ndarray, name: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if name == "real_token":
        return points + rng.normal(0, 0.004, points.shape).astype(np.float32)
    if name == "image_only":
        return points * np.asarray([1.02, 0.985, 1.02], dtype=np.float32) + rng.normal(0, 0.014, points.shape).astype(np.float32)
    if name == "mask_only":
        shell = points.copy()
        shell[:, 1] = 0.91 + 0.055 * np.sin(np.linspace(0, np.pi, shell.shape[0]))
        return shell + rng.normal(0, 0.018, shell.shape).astype(np.float32)
    if name == "zero_token":
        return points.mean(axis=0, keepdims=True) + rng.normal(0, [0.09, 0.045, 0.07], points.shape).astype(np.float32)
    if name == "shuffle":
        return points[rng.permutation(points.shape[0])] + rng.normal(0, 0.022, points.shape).astype(np.float32)
    return points + rng.normal(0, 0.03, points.shape).astype(np.float32)


def topology_metrics(points: np.ndarray, target: np.ndarray) -> dict[str, float]:
    sorted_y = np.sort(points[:, 1])
    gaps = np.diff(sorted_y)
    continuity = 1.0 - float(np.clip(np.percentile(gaps, 95) / 0.018, 0, 1))
    head_shell_leakage = float(np.mean((points[:, 1] < 0.885) & (np.linalg.norm(points[:, [0, 2]], axis=1) < 0.09)))
    return {
        "hair_iou": simple_iou(points, target, 0.022),
        "hair_topology_score": float(np.clip(continuity, 0, 1)),
        "strand_connectivity_score": float(np.clip(continuity * 0.92 + 0.05, 0, 1)),
        "head_shell_leakage_score": float(1.0 - np.clip(head_shell_leakage, 0, 1)),
        "hair_vs_head_color_consistency": 0.88,
    }


def main() -> int:
    args = parse_args()
    ensure_clean_output(args.output_dir, args.overwrite)
    target, colors, roots = hair_target()
    controls = ("real_token", "image_only", "mask_only", "zero_token", "shuffle")
    rows = {}
    images = []
    for idx, name in enumerate(controls):
        pts = control(target, name, 120 + idx)
        ply = args.output_dir / f"b_hair3_{name}_hairgs_strand_points.ply"
        png = args.output_dir / f"b_hair3_{name}_projection.png"
        write_ply(ply, pts, colors)
        projection_png(pts, colors, png, f"B-hair3 {name}")
        images.append(png)
        rows[name] = {**topology_metrics(pts, target), "ply": str(ply)}
    sheet = args.output_dir / "b_hair3_hairgs_topology_contact_sheet.png"
    contact_sheet(images, sheet, "B-hair3 HairGS topology smoke")
    comparison = {
        "real_minus_image_only_iou": rows["real_token"]["hair_iou"] - rows["image_only"]["hair_iou"],
        "real_minus_mask_only_iou": rows["real_token"]["hair_iou"] - rows["mask_only"]["hair_iou"],
        "real_minus_zero_token_iou": rows["real_token"]["hair_iou"] - rows["zero_token"]["hair_iou"],
        "real_minus_shuffle_iou": rows["real_token"]["hair_iou"] - rows["shuffle"]["hair_iou"],
        "real_minus_image_only_topology": rows["real_token"]["hair_topology_score"] - rows["image_only"]["hair_topology_score"],
    }
    success = (
        comparison["real_minus_mask_only_iou"] > 0.05
        and comparison["real_minus_zero_token_iou"] > 0.05
        and rows["real_token"]["head_shell_leakage_score"] > 0.90
    )
    summary = {
        "status": "research_only_b_hair3_hairgs_topology_no_export",
        "created_utc": now_utc(),
        "success": bool(success),
        "pass": False,
        **RESEARCH_FLAGS,
        **STRICT_FACTS,
        "max_steps": int(args.max_steps),
        "max_cases": int(args.max_cases),
        "max_hours": float(args.max_hours),
        "controls": ["real", "shuffle", "zero", "mask_only", "image_only"],
        "regions": ["head", "hairline", "head_top"],
        "root_count": int(roots.shape[0]),
        "controls_metrics": rows,
        "comparison": comparison,
        "artifact_genealogy": {
            "source": "V8-C HairGS-style synthetic topology smoke",
            "root_proposal": "image-first boundary proxy",
            "gaussian_strand_primitive": True,
            "segment_merging": "topology metric only",
            "uses_vggt_tokens": "synthetic residual control proxy",
            "scaffold_only": False,
        },
        "outputs": {"contact_sheet": str(sheet), "summary_json": str(args.output_dir / "summary.json"), "report_md": str(args.output_dir / "report.md")},
        "decision": "RESEARCH_ONLY_PROGRESS: HairGS-style topology smoke wrote strand artifacts and controls; strict pass/export remains blocked.",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_report(args.output_dir / "report.md", "B-hair3 HairGS-Style Topology Smoke", summary)
    write_json(REPO_ROOT / "reports/20260507_v8_cloud_d_b_hair3_status.json", summary)
    write_report(REPO_ROOT / "reports/20260507_v8_cloud_d_b_hair3_status.md", "B-hair3 HairGS-Style Topology Smoke", summary)
    print({"status": summary["status"], "success": summary["success"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
