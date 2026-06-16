from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SDF_ARRAYS = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D7_query_sdf_smoke_hybrid6_layer23/"
    "b_fus3d_query_sdf_smoke_arrays.npz"
)
DEFAULT_ABLATION_ARRAYS = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D13_raw_image_linesearch_ablation_hybrid6_layer23/"
    "b_fus3d_raw_image_linesearch_ablation_arrays.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D14_raw_image_offset_proposal_precheck_hybrid6_layer23"
)
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_fus3d_raw_image_offset_proposal_status.md")

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
    "teacher_export": "blocked",
    "candidate_export": "blocked",
}

CONTRACT = {
    "research_only": True,
    "local_only": True,
    "offset_proposal_visual_precheck_only": True,
    "sparse_query_pointcloud_only": True,
    "not_mesh": True,
    "not_decoder": True,
    "not_teacher": True,
    "not_candidate": True,
    "no_train": True,
    "no_cloud": True,
    "no_candidate_export": True,
    "no_teacher_export": True,
    "no_registry_write": True,
    "no_strict_pass_write": True,
    "writes_predictions_npz": False,
    "writes_formal_prediction_arrays": False,
    "writes_research_diagnostic_arrays": True,
    "writes_checkpoint": False,
}

FAMILY_COLORS = {
    "full_body": (170, 170, 170),
    "face_core": (245, 190, 40),
    "hairline": (175, 90, 230),
    "left_hand": (40, 150, 255),
    "right_hand": (255, 95, 55),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only bounded visual precheck for B-Fus3D raw-image offset proposals. "
            "It writes sparse query-level PLYs colored by family/offset and summary stats. "
            "It does not write a mesh, predictions, teacher/candidate, checkpoint, or strict pass."
        )
    )
    parser.add_argument("--sdf-arrays", type=Path, default=DEFAULT_SDF_ARRAYS)
    parser.add_argument("--ablation-arrays", type=Path, default=DEFAULT_ABLATION_ARRAYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--require-rgb-decisive", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def scalar_stats(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(arr)
    if arr.size == 0 or not finite.any():
        return {"count": int(arr.size), "finite": int(finite.sum())}
    vals = arr[finite]
    return {
        "count": int(arr.size),
        "finite": int(finite.sum()),
        "min": float(np.min(vals)),
        "p10": float(np.percentile(vals, 10)),
        "median": float(np.median(vals)),
        "mean": float(np.mean(vals)),
        "p90": float(np.percentile(vals, 90)),
        "max": float(np.max(vals)),
    }


def colors_by_offset(families: np.ndarray, offsets: np.ndarray, selected: np.ndarray) -> np.ndarray:
    colors = np.zeros((families.shape[0], 3), dtype=np.uint8)
    for idx, family in enumerate(families.astype(str).tolist()):
        base = np.asarray(FAMILY_COLORS.get(family, (200, 200, 200)), dtype=np.float32)
        if not selected[idx]:
            colors[idx] = np.asarray([45, 45, 45], dtype=np.uint8)
        elif offsets[idx] < -1e-8:
            colors[idx] = np.clip(base * np.asarray([0.55, 0.85, 1.25]), 0, 255).astype(np.uint8)
        elif offsets[idx] > 1e-8:
            colors[idx] = np.clip(base * np.asarray([1.25, 0.75, 0.55]), 0, 255).astype(np.uint8)
        else:
            colors[idx] = np.clip(base, 0, 255).astype(np.uint8)
    return colors


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray, families: np.ndarray, offsets: np.ndarray, selected: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("property float selected_offset\nproperty int selected\nproperty int family_hash\n")
        handle.write("end_header\n")
        for point, color, family, offset, flag in zip(points, colors, families.astype(str), offsets, selected, strict=False):
            fam_hash = sum(ord(ch) for ch in family) % 997
            handle.write(
                f"{float(point[0]):.8f} {float(point[1]):.8f} {float(point[2]):.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{float(offset):.8f} {int(bool(flag))} {int(fam_hash)}\n"
            )


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Fus3D Raw-Image Offset Proposal Precheck",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is a sparse query-level visual proposal precheck. It writes PLYs for",
        "template query points and raw-image offset proposal points. It is not a mesh,",
        "not a decoder, not a teacher, not a candidate, and not a pass.",
        "",
        "## Strict Truth",
        "",
        "```text",
        "strict_candidate_passes = 0",
        "strict_teacher_passes = 0",
        "formal_cloud train/infer/export = blocked",
        "teacher_export = blocked",
        "candidate_export = blocked",
        "```",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary["aggregate"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Family Summary",
        "",
        "```json",
        json.dumps(summary["family_summary"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Decision",
        "",
        "```json",
        json.dumps(summary["decision"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Outputs",
        "",
    ]
    for value in summary["outputs"].values():
        lines.append(f"- `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    sdf = np.load(args.sdf_arrays.expanduser().resolve(), allow_pickle=True)
    abl = np.load(args.ablation_arrays.expanduser().resolve(), allow_pickle=True)
    positions = np.asarray(sdf["query_positions"], dtype=np.float32)
    normals = np.asarray(sdf["query_normals"], dtype=np.float32)
    families = np.asarray(sdf["query_families"]).astype(str)
    combined_offsets = np.asarray(abl["combined_best_offsets"], dtype=np.float32)
    rgb_offsets = np.asarray(abl["rgb_best_offsets"], dtype=np.float32)
    combined_decisive = np.asarray(abl["combined_decisive"], dtype=bool)
    rgb_decisive = np.asarray(abl["rgb_decisive"], dtype=bool)
    selected = combined_decisive & rgb_decisive if args.require_rgb_decisive else combined_decisive
    proposal_offsets = np.where(selected, combined_offsets, 0.0).astype(np.float32)
    proposal_points = positions + normals * proposal_offsets[:, None]
    colors = colors_by_offset(families, proposal_offsets, selected)

    template_ply = output_dir / "query_template_points_colored_by_family.ply"
    proposal_ply = output_dir / "query_offset_proposal_points_colored_by_offset.ply"
    write_ply(template_ply, positions, colors_by_offset(families, np.zeros_like(proposal_offsets), np.ones_like(selected)), families, np.zeros_like(proposal_offsets), np.ones_like(selected))
    write_ply(proposal_ply, proposal_points, colors, families, proposal_offsets, selected)

    family_summary = {}
    for family in sorted(set(families.tolist())):
        mask = families == family
        sel = mask & selected
        family_summary[family] = {
            "query_count": int(mask.sum()),
            "selected_count": int(sel.sum()),
            "selected_ratio": float(sel.sum() / max(mask.sum(), 1)),
            "negative_offset_count": int(np.sum(sel & (proposal_offsets < -1e-8))),
            "positive_offset_count": int(np.sum(sel & (proposal_offsets > 1e-8))),
            "offset_stats_selected": scalar_stats(proposal_offsets[sel]),
            "proposal_delta_norm_stats": scalar_stats(np.linalg.norm((proposal_points - positions)[sel], axis=1) if np.any(sel) else []),
        }
    aggregate = {
        "query_count": int(positions.shape[0]),
        "selected_count": int(selected.sum()),
        "selected_ratio": float(selected.sum() / max(positions.shape[0], 1)),
        "require_rgb_decisive": bool(args.require_rgb_decisive),
        "negative_offset_count": int(np.sum(selected & (proposal_offsets < -1e-8))),
        "positive_offset_count": int(np.sum(selected & (proposal_offsets > 1e-8))),
        "offset_stats_selected": scalar_stats(proposal_offsets[selected]),
        "proposal_delta_norm_stats": scalar_stats(np.linalg.norm((proposal_points - positions)[selected], axis=1) if np.any(selected) else []),
    }
    decision = {
        "status": "research_sparse_offset_proposal_no_pass",
        "proposal_is_sparse_query_points_only": True,
        "selected_ratio": aggregate["selected_ratio"],
        "interpretation": (
            "raw-image RGB-supported line-search produces a sparse local offset proposal for visual inspection only"
        ),
        "next_allowed_action": (
            "Inspect proposal PLY visually; only if directions look coherent should a future bounded rendered-mesh proposal be designed."
        ),
        "blocked_actions": [
            "do_not_use_sparse_query_points_as_mesh_or_teacher",
            "do_not_export_predictions_or_candidate",
            "do_not_train_from_this_precheck",
            "do_not_unblock_cloud",
        ],
    }
    arrays_path = output_dir / "b_fus3d_raw_image_offset_proposal_arrays.npz"
    np.savez_compressed(
        arrays_path,
        template_points=positions,
        proposal_points=proposal_points,
        proposal_offsets=proposal_offsets,
        selected=selected,
        families=families,
    )
    summary = {
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_train_infer_export": "blocked",
        "status": "research_only_sparse_offset_proposal_not_mesh_not_decoder_not_teacher_not_candidate",
        "contract": CONTRACT,
        "inputs": {
            "sdf_arrays": str(args.sdf_arrays.expanduser().resolve()),
            "ablation_arrays": str(args.ablation_arrays.expanduser().resolve()),
            "require_rgb_decisive": bool(args.require_rgb_decisive),
        },
        "aggregate": aggregate,
        "family_summary": family_summary,
        "decision": decision,
        "outputs": {
            "template_ply": str(template_ply),
            "proposal_ply": str(proposal_ply),
            "arrays": str(arrays_path),
            "summary_json": str(output_dir / "b_fus3d_raw_image_offset_proposal_summary.json"),
            "summary_md": str(output_dir / "b_fus3d_raw_image_offset_proposal_summary.md"),
            "status_report": str(args.status_report),
        },
    }
    summary_path = output_dir / "b_fus3d_raw_image_offset_proposal_summary.json"
    md_path = output_dir / "b_fus3d_raw_image_offset_proposal_summary.md"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, json_ready(summary))
    if args.status_report:
        write_markdown(args.status_report.expanduser().resolve(), json_ready(summary))
    print(json.dumps(json_ready({"summary": str(summary_path), "decision": decision, "aggregate": aggregate}), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
