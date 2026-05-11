from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_LINES_ARRAYS = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D12_raw_image_normal_linesearch_probe_hybrid6_layer23/"
    "b_fus3d_raw_image_linesearch_arrays.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D13_raw_image_linesearch_ablation_hybrid6_layer23"
)
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_fus3d_raw_image_linesearch_ablation_status.md")

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
    "raw_image_linesearch_ablation_only": True,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only ablation for B-Fus3D12 line-search arrays. It compares "
            "combined score, RGB-only variance, and mask-only support to check whether "
            "the non-zero offset preference is a photometric signal or a mask-score "
            "artifact. It does not train, decode, export teacher/candidate, or write pass state."
        )
    )
    parser.add_argument("--linesearch-arrays", type=Path, default=DEFAULT_LINES_ARRAYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--margin", type=float, default=1e-4)
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


def best_readout(score: np.ndarray, offsets: np.ndarray, center_index: int, margin: float) -> dict[str, np.ndarray]:
    finite = np.isfinite(score)
    any_valid = finite.any(axis=1)
    safe = np.where(finite, score, np.inf)
    best_idx = np.argmin(safe, axis=1)
    best_idx[~any_valid] = center_index
    best_offsets = offsets[best_idx]
    best_scores = safe[np.arange(score.shape[0]), best_idx]
    center_scores = score[:, center_index]
    center_margin = center_scores - best_scores
    nonzero = any_valid & (np.abs(best_offsets) > 1e-8)
    decisive = nonzero & np.isfinite(center_margin) & (center_margin > float(margin))
    return {
        "any_valid": any_valid,
        "best_idx": best_idx,
        "best_offsets": best_offsets,
        "center_margin": center_margin,
        "nonzero": nonzero,
        "decisive": decisive,
    }


def family_stats(families: np.ndarray, readout: dict[str, np.ndarray]) -> dict[str, Any]:
    out = {}
    for family in sorted(set(families.astype(str).tolist())):
        mask = families.astype(str) == family
        valid = mask & readout["any_valid"]
        out[family] = {
            "query_count": int(mask.sum()),
            "valid_count": int(valid.sum()),
            "valid_ratio": float(valid.sum() / max(mask.sum(), 1)),
            "nonzero_count": int(np.sum(mask & readout["nonzero"])),
            "nonzero_ratio": float(np.sum(mask & readout["nonzero"]) / max(valid.sum(), 1)),
            "decisive_count": int(np.sum(mask & readout["decisive"])),
            "decisive_ratio": float(np.sum(mask & readout["decisive"]) / max(valid.sum(), 1)),
            "best_offset_stats": scalar_stats(readout["best_offsets"][valid]),
            "center_margin_stats": scalar_stats(readout["center_margin"][valid]),
        }
    return out


def overlap_stats(a: dict[str, np.ndarray], b: dict[str, np.ndarray], valid: np.ndarray) -> dict[str, Any]:
    both = valid & a["decisive"] & b["decisive"]
    either = valid & (a["decisive"] | b["decisive"])
    same_offset = valid & a["decisive"] & b["decisive"] & np.isclose(a["best_offsets"], b["best_offsets"])
    return {
        "both_decisive": int(both.sum()),
        "either_decisive": int(either.sum()),
        "jaccard_decisive": float(both.sum() / max(either.sum(), 1)),
        "same_offset_when_both": int(same_offset.sum()),
        "same_offset_ratio_when_both": float(same_offset.sum() / max(both.sum(), 1)),
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Fus3D Raw-Image Line-Search Ablation",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This research-only ablation separates the B-Fus3D12 line-search score into",
        "combined, RGB-only, and mask-only readouts. It checks whether the non-zero",
        "offset preference is photometric or just mask/silhouette driven.",
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
        "## Aggregate",
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
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_readout(name: str, readout: dict[str, np.ndarray]) -> dict[str, Any]:
    valid = readout["any_valid"]
    return {
        "name": name,
        "valid_count": int(valid.sum()),
        "valid_ratio": float(valid.sum() / max(valid.size, 1)),
        "nonzero_count": int((valid & readout["nonzero"]).sum()),
        "nonzero_ratio": float((valid & readout["nonzero"]).sum() / max(valid.sum(), 1)),
        "decisive_count": int((valid & readout["decisive"]).sum()),
        "decisive_ratio": float((valid & readout["decisive"]).sum() / max(valid.sum(), 1)),
        "center_margin_stats": scalar_stats(readout["center_margin"][valid]),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.linesearch_arrays.expanduser().resolve(), allow_pickle=True)
    offsets = np.asarray(data["offsets"], dtype=np.float32)
    center_index = int(np.argmin(np.abs(offsets)))
    families = np.asarray(data["families"]).astype(str)
    combined_score = np.asarray(data["scores"], dtype=np.float32)
    rgb_variance = np.asarray(data["rgb_variance"], dtype=np.float32)
    mask_score = np.asarray(data["mask_score"], dtype=np.float32)

    combined = best_readout(combined_score, offsets, center_index, float(args.margin))
    rgb_only = best_readout(rgb_variance, offsets, center_index, float(args.margin))
    mask_only = best_readout(-mask_score, offsets, center_index, float(args.margin))
    valid = combined["any_valid"] | rgb_only["any_valid"] | mask_only["any_valid"]
    aggregate = {
        "offsets": offsets.tolist(),
        "margin": float(args.margin),
        "combined": summarize_readout("combined", combined),
        "rgb_only": summarize_readout("rgb_only", rgb_only),
        "mask_only": summarize_readout("mask_only", mask_only),
        "combined_vs_rgb": overlap_stats(combined, rgb_only, valid),
        "combined_vs_mask": overlap_stats(combined, mask_only, valid),
        "rgb_vs_mask": overlap_stats(rgb_only, mask_only, valid),
    }
    family_summary = {
        "combined": family_stats(families, combined),
        "rgb_only": family_stats(families, rgb_only),
        "mask_only": family_stats(families, mask_only),
    }
    rgb_decisive = float(aggregate["rgb_only"]["decisive_ratio"])
    mask_decisive = float(aggregate["mask_only"]["decisive_ratio"])
    combined_decisive = float(aggregate["combined"]["decisive_ratio"])
    combined_mask_jaccard = float(aggregate["combined_vs_mask"]["jaccard_decisive"])
    combined_rgb_jaccard = float(aggregate["combined_vs_rgb"]["jaccard_decisive"])
    likely_mask_driven = combined_decisive > 0.25 and combined_mask_jaccard > combined_rgb_jaccard + 0.10
    likely_photo_signal = rgb_decisive > 0.25 and combined_rgb_jaccard >= 0.30
    decision = {
        "status": "research_linesearch_ablation_no_pass",
        "likely_mask_driven": bool(likely_mask_driven),
        "likely_photometric_signal": bool(likely_photo_signal),
        "combined_decisive_ratio": combined_decisive,
        "rgb_only_decisive_ratio": rgb_decisive,
        "mask_only_decisive_ratio": mask_decisive,
        "combined_vs_rgb_jaccard": combined_rgb_jaccard,
        "combined_vs_mask_jaccard": combined_mask_jaccard,
        "interpretation": (
            "combined line-search has photometric support"
            if likely_photo_signal and not likely_mask_driven
            else "combined line-search is at risk of being mask/silhouette driven or ambiguous"
        ),
        "next_allowed_action": (
            "Use this to decide whether a bounded visual proposal precheck is warranted; do not train/export from it."
        ),
        "blocked_actions": [
            "do_not_tune_offset_margin_or_mask_weight_into_a_loop",
            "do_not_claim_surface_geometry_success",
            "do_not_train_decoder_from_ablation_alone",
            "do_not_unblock_cloud",
        ],
    }
    arrays_path = output_dir / "b_fus3d_raw_image_linesearch_ablation_arrays.npz"
    np.savez_compressed(
        arrays_path,
        offsets=offsets,
        combined_best_offsets=combined["best_offsets"],
        rgb_best_offsets=rgb_only["best_offsets"],
        mask_best_offsets=mask_only["best_offsets"],
        combined_decisive=combined["decisive"],
        rgb_decisive=rgb_only["decisive"],
        mask_decisive=mask_only["decisive"],
        families=families,
    )
    summary = {
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_train_infer_export": "blocked",
        "status": "research_only_raw_image_linesearch_ablation_not_decoder_not_teacher_not_candidate",
        "contract": CONTRACT,
        "inputs": {"linesearch_arrays": str(args.linesearch_arrays.expanduser().resolve())},
        "aggregate": aggregate,
        "family_summary": family_summary,
        "decision": decision,
        "outputs": {
            "arrays": str(arrays_path),
            "summary_json": str(output_dir / "b_fus3d_raw_image_linesearch_ablation_summary.json"),
            "summary_md": str(output_dir / "b_fus3d_raw_image_linesearch_ablation_summary.md"),
            "status_report": str(args.status_report),
        },
    }
    summary_path = output_dir / "b_fus3d_raw_image_linesearch_ablation_summary.json"
    md_path = output_dir / "b_fus3d_raw_image_linesearch_ablation_summary.md"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, json_ready(summary))
    if args.status_report:
        write_markdown(args.status_report.expanduser().resolve(), json_ready(summary))
    print(json.dumps(json_ready({"summary": str(summary_path), "decision": decision}), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
