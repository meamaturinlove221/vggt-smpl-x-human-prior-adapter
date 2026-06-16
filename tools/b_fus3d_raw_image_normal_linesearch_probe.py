from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_SDF_ARRAYS = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D7_query_sdf_smoke_hybrid6_layer23/"
    "b_fus3d_query_sdf_smoke_arrays.npz"
)
DEFAULT_QUERY_SUMMARY = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D6_query_evidence_cache_hybrid6_layer23/"
    "b_fus3d_query_evidence_cache_summary.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D12_raw_image_normal_linesearch_probe_hybrid6_layer23"
)
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_fus3d_raw_image_linesearch_status.md")

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
    "raw_image_normal_linesearch_only": True,
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
            "Research-only raw-image normal line-search probe. It samples each "
            "B-Fus3D query at offsets along its template normal and asks whether "
            "raw multi-view RGB/mask consistency prefers a non-zero offset. It "
            "does not train, decode, render a mesh, export teacher/candidate, "
            "or write strict pass state."
        )
    )
    parser.add_argument("--sdf-arrays", type=Path, default=DEFAULT_SDF_ARRAYS)
    parser.add_argument("--query-summary", type=Path, default=DEFAULT_QUERY_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--offsets", default="-0.012,0.0,0.012")
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


def parse_offsets(text: str) -> np.ndarray:
    vals = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not vals:
        vals = [-0.012, 0.0, 0.012]
    if 0.0 not in vals:
        vals.append(0.0)
    return np.asarray(sorted(vals), dtype=np.float32)


def scene_image_name(view_index: int) -> str:
    if int(view_index) == 0:
        return "00_tgt_cam00.png"
    return f"{int(view_index):02d}_src_cam{int(view_index):02d}.png"


def load_image(path: Path, mode: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert(mode))


def sample_nearest(arr: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    x = int(np.rint(float(uv[0])))
    y = int(np.rint(float(uv[1])))
    if x < 0 or y < 0 or x >= w or y >= h:
        if arr.ndim == 2:
            return np.asarray(np.nan, dtype=np.float32)
        return np.full(arr.shape[2], np.nan, dtype=np.float32)
    return arr[y, x].astype(np.float32)


def project(points: np.ndarray, world_to_cam: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot = world_to_cam[:3, :3].astype(np.float32)
    trans = world_to_cam[:3, 3].astype(np.float32)
    cam = points @ rot.T + trans[None, :]
    z = cam[:, 2]
    uvw = (intrinsic @ cam.T).T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
    return uv.astype(np.float32), z.astype(np.float32)


def load_view_params(query_summary: dict[str, Any]) -> tuple[Path, list[int], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    # Reuse the already cached projections where possible would hide the effect
    # of the offset line-search. Load camera params through the sidecar summary.
    from prepare_4k4d_prior_training_case import (  # local import keeps this probe lightweight
        align_intrinsics_for_scene_view,
        load_scene_manifest,
        recover_legacy_crop_source_sizes,
        resolve_scene_camera_params,
    )

    scene_dir = Path(query_summary["summary"]["scene_dir"])
    selected_views = [int(v) for v in query_summary["summary"]["selected_view_indices"]]
    target_size = int(query_summary["summary"].get("target_size", 518))
    manifest = recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir))
    cameras, _source = resolve_scene_camera_params(manifest, Path(str(manifest.get("dataset_root", ""))), "data_used_in_4K4D")
    intrinsics = []
    extrinsics = []
    rgbs = []
    masks = []
    for view_index in selected_views:
        view = manifest["exported_views"][view_index]
        cam = cameras[str(view["camera_id"])]
        intrinsics.append(align_intrinsics_for_scene_view(np.asarray(cam["intrinsic"], dtype=np.float32), view, target_size))
        extrinsics.append(np.asarray(cam["world_to_cam"], dtype=np.float32))
        name = scene_image_name(view_index)
        rgbs.append(load_image(scene_dir / "images" / name, "RGB").astype(np.float32) / 255.0)
        masks.append(load_image(scene_dir / "masks" / name, "L").astype(np.float32) / 255.0)
    return scene_dir, selected_views, intrinsics, extrinsics, rgbs, masks


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Fus3D Raw-Image Normal Line-Search Probe",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is a research-only line-search over template-normal offsets. It checks",
        "whether raw RGB/mask consistency prefers a non-zero offset for B-Fus3D query",
        "points. It is not a decoder, teacher, candidate, or pass.",
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
        "- Non-zero offset preference is only a raw-image signal, not geometry success.",
        "- If the center offset is usually best or the margin is tiny, raw-image losses may be too weak for local refinement.",
        "- This probe cannot unblock formal cloud or strict gates.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    offsets = parse_offsets(args.offsets)
    center_index = int(np.argmin(np.abs(offsets)))
    sdf = np.load(args.sdf_arrays.expanduser().resolve(), allow_pickle=True)
    query_summary = load_json(args.query_summary)
    scene_dir, selected_views, intrinsics, extrinsics, rgbs, masks = load_view_params(query_summary)
    qpos = np.asarray(sdf["query_positions"], dtype=np.float32)
    normals = np.asarray(sdf["query_normals"], dtype=np.float32)
    families = np.asarray(sdf["query_families"]).astype(str)
    n = qpos.shape[0]
    scores = np.full((n, offsets.size), np.nan, dtype=np.float32)
    visible_counts = np.zeros((n, offsets.size), dtype=np.int32)
    rgb_variance = np.full((n, offsets.size), np.nan, dtype=np.float32)
    mask_score = np.full((n, offsets.size), np.nan, dtype=np.float32)

    for oi, offset in enumerate(offsets):
        points = qpos + normals * float(offset)
        rgb_values = np.full((n, len(selected_views), 3), np.nan, dtype=np.float32)
        mask_values = np.full((n, len(selected_views)), np.nan, dtype=np.float32)
        for vi, (_view_index, intrinsic, world_to_cam, rgb, mask) in enumerate(
            zip(selected_views, intrinsics, extrinsics, rgbs, masks, strict=False)
        ):
            uv, z = project(points, world_to_cam, intrinsic)
            for qi in range(n):
                if not np.isfinite(z[qi]) or z[qi] <= 1e-6:
                    continue
                m = sample_nearest(mask, uv[qi])
                r = sample_nearest(rgb, uv[qi])
                if np.isfinite(m).all() and np.isfinite(r).all():
                    mask_values[qi, vi] = float(m)
                    rgb_values[qi, vi] = np.asarray(r, dtype=np.float32)
        visible = mask_values >= float(args.mask_threshold)
        visible_counts[:, oi] = visible.sum(axis=1)
        for qi in range(n):
            vals = rgb_values[qi, visible[qi]]
            if vals.shape[0] >= int(args.min_mask_views):
                rgb_variance[qi, oi] = float(vals.var(axis=0).mean())
                mask_score[qi, oi] = float(np.nanmean(mask_values[qi]))
                # Lower is better: prefer low color variance and high mask support.
                scores[qi, oi] = float(rgb_variance[qi, oi] - 0.02 * mask_score[qi, oi])

    best_idx = np.nanargmin(np.where(np.isfinite(scores), scores, np.inf), axis=1)
    any_valid = np.isfinite(scores).any(axis=1)
    best_idx[~any_valid] = center_index
    best_offsets = offsets[best_idx]
    center_scores = scores[:, center_index]
    best_scores = scores[np.arange(n), best_idx]
    center_margin = center_scores - best_scores
    nonzero_best = any_valid & (np.abs(best_offsets) > 1e-8)
    decisive = nonzero_best & np.isfinite(center_margin) & (center_margin > 1e-4)

    family_summary = {}
    for family in sorted(set(families.tolist())):
        mask = families == family
        valid = mask & any_valid
        family_summary[family] = {
            "query_count": int(mask.sum()),
            "valid_linesearch_count": int(valid.sum()),
            "valid_linesearch_ratio": float(valid.sum() / max(mask.sum(), 1)),
            "nonzero_best_count": int(np.sum(nonzero_best & mask)),
            "nonzero_best_ratio": float(np.sum(nonzero_best & mask) / max(valid.sum(), 1)),
            "decisive_nonzero_count": int(np.sum(decisive & mask)),
            "decisive_nonzero_ratio": float(np.sum(decisive & mask) / max(valid.sum(), 1)),
            "best_offset_stats": scalar_stats(best_offsets[valid]),
            "center_margin_stats": scalar_stats(center_margin[valid]),
            "center_rgb_variance_stats": scalar_stats(rgb_variance[valid, center_index]),
        }

    decision = {
        "status": "research_raw_image_linesearch_no_pass",
        "offsets": offsets.tolist(),
        "valid_query_ratio": float(any_valid.sum() / max(n, 1)),
        "decisive_nonzero_ratio": float(decisive.sum() / max(any_valid.sum(), 1)),
        "families_with_decisive_nonzero_ge_25pct": [
            family for family, row in family_summary.items() if float(row["decisive_nonzero_ratio"]) >= 0.25
        ],
        "interpretation": (
            "raw image line-search provides a local geometric direction only for families "
            "with sufficient decisive_nonzero_ratio; otherwise rendered losses are likely weak/ambiguous"
        ),
        "next_allowed_action": "Use as backend-design evidence only; do not train/export/claim pass from line-search.",
        "blocked_actions": [
            "do_not_claim_geometry_or_visual_pass",
            "do_not_tune_offsets_or_thresholds_into_a_loop",
            "do_not_train_decoder_from_linesearch_alone",
            "do_not_unblock_cloud",
        ],
    }
    arrays_path = output_dir / "b_fus3d_raw_image_linesearch_arrays.npz"
    np.savez_compressed(
        arrays_path,
        offsets=offsets,
        scores=scores,
        rgb_variance=rgb_variance,
        mask_score=mask_score,
        visible_counts=visible_counts,
        best_offsets=best_offsets,
        center_margin=center_margin,
        families=families,
        any_valid=any_valid,
        decisive_nonzero=decisive,
    )
    summary = {
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_train_infer_export": "blocked",
        "status": "research_only_raw_image_normal_linesearch_not_decoder_not_teacher_not_candidate",
        "contract": CONTRACT,
        "inputs": {
            "sdf_arrays": str(args.sdf_arrays.expanduser().resolve()),
            "query_summary": str(args.query_summary.expanduser().resolve()),
            "scene_dir": str(scene_dir),
            "selected_views": selected_views,
            "offsets": offsets.tolist(),
            "mask_threshold": float(args.mask_threshold),
            "min_mask_views": int(args.min_mask_views),
        },
        "family_summary": family_summary,
        "decision": decision,
        "outputs": {
            "arrays": str(arrays_path),
            "summary_json": str(output_dir / "b_fus3d_raw_image_linesearch_summary.json"),
            "summary_md": str(output_dir / "b_fus3d_raw_image_linesearch_summary.md"),
            "status_report": str(args.status_report),
        },
    }
    summary_path = output_dir / "b_fus3d_raw_image_linesearch_summary.json"
    md_path = output_dir / "b_fus3d_raw_image_linesearch_summary.md"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, json_ready(summary))
    if args.status_report:
        write_markdown(args.status_report.expanduser().resolve(), json_ready(summary))
    print(json.dumps(json_ready({"summary": str(summary_path), "decision": decision}), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
