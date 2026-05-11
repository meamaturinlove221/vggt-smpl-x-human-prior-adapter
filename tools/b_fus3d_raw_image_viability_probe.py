from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_QUERY_CACHE = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D6_query_evidence_cache_hybrid6_layer23/"
    "b_fus3d_query_evidence_cache.npz"
)
DEFAULT_QUERY_SUMMARY = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D6_query_evidence_cache_hybrid6_layer23/"
    "b_fus3d_query_evidence_cache_summary.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D11_raw_image_viability_probe_hybrid6_layer23"
)
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_fus3d_raw_image_viability_status.md")

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
    "raw_image_viability_probe_only": True,
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
            "Research-only raw-image viability probe for B-Fus3D queries. It samples "
            "raw RGB/mask at query projections and measures cross-view color variance "
            "and mask support. It does not train, decode, render, export a teacher/"
            "candidate, or write strict pass state."
        )
    )
    parser.add_argument("--query-cache", type=Path, default=DEFAULT_QUERY_CACHE)
    parser.add_argument("--query-summary", type=Path, default=DEFAULT_QUERY_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-mask-views", type=int, default=2)
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def load_image(path: Path, mode: str) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img.convert(mode))
    return arr


def scene_image_name(view_index: int) -> str:
    if view_index == 0:
        return "00_tgt_cam00.png"
    return f"{view_index:02d}_src_cam{view_index:02d}.png"


def sample_nearest(arr: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    x = int(np.rint(float(uv[0])))
    y = int(np.rint(float(uv[1])))
    if x < 0 or y < 0 or x >= w or y >= h:
        if arr.ndim == 2:
            return np.asarray(np.nan, dtype=np.float32)
        return np.full(arr.shape[2], np.nan, dtype=np.float32)
    return arr[y, x].astype(np.float32)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Fus3D Raw-Image Viability Probe",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is a research-only query projection probe. It samples raw RGB and mask",
        "values at the B-Fus3D query projections and measures whether each family has",
        "enough multi-view raw-image support for a future learned surface backend.",
        "It is not a decoder, not a teacher, not a candidate, and not a pass.",
        "",
        "## Strict Truth",
        "",
        "```text",
        "strict_candidate_passes = 0",
        "strict_teacher_passes = 0",
        "formal_cloud_train/infer/export = blocked",
        "teacher_export = blocked",
        "candidate_export = blocked",
        "```",
        "",
        "## Inputs",
        "",
        "```json",
        json.dumps(summary["inputs"], indent=2, ensure_ascii=False),
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
        "## Notes",
        "",
        "- Low color variance can mean texture consistency or template/shell ambiguity; it is not geometry success.",
        "- High mask support means the query projects inside the human crop/mask, not that Open3D looks correct.",
        "- This result can only guide whether raw-image/rendered losses are viable in a future backend.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    query = np.load(args.query_cache.expanduser().resolve(), allow_pickle=True)
    query_summary = load_json(args.query_summary)
    scene_dir = Path(query_summary["summary"]["scene_dir"])
    selected_views = np.asarray(query["selected_view_indices"], dtype=np.int32)
    uv = np.asarray(query["uv"], dtype=np.float32)
    families = np.asarray(query["query_families"]).astype(str)
    support = np.asarray(query["support"], dtype=np.int32)

    rgb_by_slot: list[np.ndarray] = []
    mask_by_slot: list[np.ndarray] = []
    view_files = []
    for view_index in selected_views.tolist():
        name = scene_image_name(int(view_index))
        image_path = scene_dir / "images" / name
        mask_path = scene_dir / "masks" / name
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing raw image/mask for view {view_index}: {image_path} / {mask_path}")
        rgb_by_slot.append(load_image(image_path, "RGB").astype(np.float32) / 255.0)
        mask_by_slot.append(load_image(mask_path, "L").astype(np.float32) / 255.0)
        view_files.append({"view_index": int(view_index), "image": str(image_path), "mask": str(mask_path)})

    n, slots, _ = uv.shape
    rgb_samples = np.full((n, slots, 3), np.nan, dtype=np.float32)
    mask_samples = np.full((n, slots), np.nan, dtype=np.float32)
    in_bounds = np.zeros((n, slots), dtype=bool)
    for qi in range(n):
        for si in range(slots):
            sample_mask = sample_nearest(mask_by_slot[si], uv[qi, si])
            sample_rgb = sample_nearest(rgb_by_slot[si], uv[qi, si])
            if np.isfinite(sample_mask).all() and np.isfinite(sample_rgb).all():
                mask_samples[qi, si] = float(sample_mask)
                rgb_samples[qi, si] = np.asarray(sample_rgb, dtype=np.float32)
                in_bounds[qi, si] = True

    visible = in_bounds & (mask_samples >= float(args.mask_threshold))
    visible_count = visible.sum(axis=1).astype(np.int32)
    rgb_mean = np.full((n, 3), np.nan, dtype=np.float32)
    rgb_var = np.full((n,), np.nan, dtype=np.float32)
    rgb_range = np.full((n,), np.nan, dtype=np.float32)
    for qi in range(n):
        vals = rgb_samples[qi, visible[qi]]
        if vals.shape[0] > 0:
            rgb_mean[qi] = vals.mean(axis=0)
        if vals.shape[0] >= 2:
            channel_var = vals.var(axis=0).mean()
            rgb_var[qi] = float(channel_var)
            rgb_range[qi] = float((vals.max(axis=0) - vals.min(axis=0)).mean())

    family_summary = {}
    viable_families = []
    for family in sorted(set(families.tolist())):
        mask = families == family
        support_mask = mask & (visible_count >= int(args.min_mask_views))
        family_summary[family] = {
            "query_count": int(mask.sum()),
            "cache_support_mean": float(np.mean(support[mask])) if mask.any() else 0.0,
            "mask_visible_ge_min_views": int(support_mask.sum()),
            "mask_visible_ge_min_ratio": float(support_mask.sum() / max(mask.sum(), 1)),
            "visible_count_stats": scalar_stats(visible_count[mask]),
            "rgb_variance_stats_ge2_views": scalar_stats(rgb_var[mask & np.isfinite(rgb_var)]),
            "rgb_range_stats_ge2_views": scalar_stats(rgb_range[mask & np.isfinite(rgb_range)]),
        }
        if family_summary[family]["mask_visible_ge_min_ratio"] >= 0.5:
            viable_families.append(family)

    decision = {
        "status": "research_raw_image_viability_no_pass",
        "families_with_mask_support_ge_50pct": viable_families,
        "face_core_raw_image_viable": "face_core" in viable_families,
        "hairline_raw_image_weak": "hairline" not in viable_families,
        "left_hand_raw_image_viable": "left_hand" in viable_families,
        "right_hand_raw_image_weak": "right_hand" not in viable_families,
        "interpretation": (
            "raw image support is uneven: face_core/left_hand may support rendered losses, "
            "while hairline/right_hand remain weak unless crop/view/backend changes are introduced"
        ),
        "next_allowed_action": (
            "Use these stats only to prioritize future raw-image rendered losses; do not train or export from this probe."
        ),
        "blocked_actions": [
            "do_not_claim_geometry_or_visual_pass",
            "do_not_train_decoder_from_raw_viability_stats_alone",
            "do_not_unblock_cloud",
            "do_not_restart_view_support_threshold_loop",
        ],
    }

    arrays_path = output_dir / "b_fus3d_raw_image_viability_arrays.npz"
    np.savez_compressed(
        arrays_path,
        rgb_samples=rgb_samples,
        mask_samples=mask_samples,
        visible=visible,
        visible_count=visible_count,
        rgb_mean=rgb_mean,
        rgb_var=rgb_var,
        rgb_range=rgb_range,
        families=families,
        support=support,
        selected_view_indices=selected_views,
    )
    summary = {
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_train_infer_export": "blocked",
        "status": "research_only_raw_image_viability_probe_not_decoder_not_teacher_not_candidate",
        "contract": CONTRACT,
        "inputs": {
            "query_cache": str(args.query_cache.expanduser().resolve()),
            "query_summary": str(args.query_summary.expanduser().resolve()),
            "scene_dir": str(scene_dir),
            "selected_views": selected_views.tolist(),
            "view_files": view_files,
            "mask_threshold": float(args.mask_threshold),
            "min_mask_views": int(args.min_mask_views),
        },
        "family_summary": family_summary,
        "decision": decision,
        "outputs": {
            "arrays": str(arrays_path),
            "summary_json": str(output_dir / "b_fus3d_raw_image_viability_summary.json"),
            "summary_md": str(output_dir / "b_fus3d_raw_image_viability_summary.md"),
            "status_report": str(args.status_report),
        },
    }
    summary_path = output_dir / "b_fus3d_raw_image_viability_summary.json"
    md_path = output_dir / "b_fus3d_raw_image_viability_summary.md"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, json_ready(summary))
    if args.status_report:
        write_markdown(args.status_report.expanduser().resolve(), json_ready(summary))
    print(json.dumps(json_ready({"summary": str(summary_path), "decision": decision}), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
