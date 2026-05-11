from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_colmap_depth_teacher_targets import read_colmap_depth  # noqa: E402


DEFAULT_HAND_EVIDENCE_CACHE = Path(
    "output/surface_research_preflight_local/"
    "B_hand0_evidence_cache_60v_humancrop_hybrid6/"
    "b_hand_evidence_cache.json"
)
DEFAULT_A5_WORKSPACE = Path(
    "output/surface_research_preflight/"
    "A5_known_camera_colmap_workspace_modal_colmap_execute_t256_hybrid12_known_direct_v1/"
    "A5_known_camera_colmap_workspace/workspace"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_hand6_colmap_depth_evidence_probe_hybrid12"
)
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_hand_colmap_depth_evidence_status.md")

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
    "depth_evidence_probe_only": True,
    "fixed_smoke_not_tuning_loop": True,
    "not_hand_decoder": True,
    "not_mano_or_smplx_success_claim": True,
    "not_teacher": True,
    "not_candidate": True,
    "no_train": True,
    "no_vggt_training": True,
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
            "Research-only B-hand6 probe. It maps B-hand ROI bboxes into an existing "
            "A5 known-camera COLMAP PatchMatch workspace and measures whether the hand "
            "regions have valid depth support. It does not train, infer, export a "
            "teacher/candidate, write predictions.npz, or touch the strict registry."
        )
    )
    parser.add_argument("--hand-evidence-cache", type=Path, default=DEFAULT_HAND_EVIDENCE_CACHE)
    parser.add_argument("--a5-workspace", type=Path, default=DEFAULT_A5_WORKSPACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--depth-valid-min", type=float, default=0.4)
    parser.add_argument("--depth-valid-max", type=float, default=8.0)
    parser.add_argument("--min-present-ratio", type=float, default=0.20)
    parser.add_argument("--strong-present-ratio", type=float, default=0.50)
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


def load_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON in {resolved}")
    payload["_resolved_path"] = str(resolved)
    return payload


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


def norm_camera_keys(value: Any) -> list[str]:
    text = str(value).strip()
    keys = [text]
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        as_int = str(int(digits))
        keys.extend([as_int, as_int.zfill(2), digits, digits.zfill(2)])
    return list(dict.fromkeys(keys))


def build_a5_view_map(workspace: Path) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    known = load_json(workspace / "known_camera_model.json")
    depth_dir = workspace / "dense" / "stereo" / "depth_maps"
    view_map: dict[str, dict[str, Any]] = {}
    for view in known.get("views", []):
        if not isinstance(view, dict):
            continue
        image_name = str(view.get("image_name", ""))
        depth_path = depth_dir / f"{image_name}.photometric.bin"
        entry = {
            "view_index": int(view.get("view_index", -1)),
            "scene_camera_id": str(view.get("scene_camera_id", "")),
            "image_name": image_name,
            "image_path": str(view.get("image_path", "")),
            "mask_path": str(view.get("mask_path", "")),
            "depth_path": str(depth_path),
            "output_size": view.get("output_size", known.get("target_size", None)),
            "source_size": view.get("source_size", None),
        }
        for key in norm_camera_keys(view.get("scene_camera_id", "")):
            view_map[key] = entry
        for key in norm_camera_keys(view.get("view_index", "")):
            view_map.setdefault(key, entry)
    depth_cache: dict[str, np.ndarray] = {}
    return view_map, depth_cache


def iter_hand_rois(cache: dict[str, Any]) -> list[dict[str, Any]]:
    per_view = cache.get("per_view", {})
    rows: list[dict[str, Any]] = []
    if isinstance(per_view, dict):
        view_items = per_view.items()
    elif isinstance(per_view, list):
        view_items = enumerate(per_view)
    else:
        return rows
    for _, view_payload in view_items:
        if not isinstance(view_payload, dict):
            continue
        rois = view_payload.get("hand_rois", [])
        if not isinstance(rois, list):
            continue
        for roi in rois:
            if isinstance(roi, dict):
                rows.append(roi)
    return rows


def clamp_bbox_xyxy(bbox: list[float], scale_x: float, scale_y: float, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    sx0 = int(np.floor(x0 * scale_x))
    sy0 = int(np.floor(y0 * scale_y))
    sx1 = int(np.ceil(x1 * scale_x))
    sy1 = int(np.ceil(y1 * scale_y))
    sx0 = max(0, min(width - 1, sx0))
    sy0 = max(0, min(height - 1, sy0))
    sx1 = max(sx0 + 1, min(width, sx1))
    sy1 = max(sy0 + 1, min(height, sy1))
    return sx0, sy0, sx1, sy1


def depth_for_view(a5_entry: dict[str, Any], depth_cache: dict[str, np.ndarray]) -> np.ndarray | None:
    path = str(a5_entry.get("depth_path", ""))
    if not path:
        return None
    if path in depth_cache:
        return depth_cache[path]
    depth_path = Path(path)
    if not depth_path.is_file():
        return None
    depth = read_colmap_depth(depth_path)
    depth_cache[path] = depth
    return depth


def draw_contact_sheet(
    output_path: Path,
    per_camera_depth: dict[str, np.ndarray],
    draw_rows: list[dict[str, Any]],
    *,
    valid_min: float,
    valid_max: float,
) -> None:
    tiles = []
    for camera_id, depth in sorted(per_camera_depth.items(), key=lambda item: norm_camera_keys(item[0])[-1]):
        valid = np.isfinite(depth) & (depth >= valid_min) & (depth <= valid_max)
        image = Image.fromarray((valid.astype(np.uint8) * 180 + 30).astype(np.uint8), mode="L").convert("RGB")
        draw = ImageDraw.Draw(image)
        for row in draw_rows:
            if str(row.get("mapped_camera_id")) != str(camera_id):
                continue
            bbox = row.get("bbox_scaled_xyxy")
            if not bbox:
                continue
            color = (255, 64, 64) if row.get("side") == "left" else (64, 128, 255)
            draw.rectangle([int(v) for v in bbox], outline=color, width=2)
            draw.text((int(bbox[0]) + 1, int(bbox[1]) + 1), str(row.get("side", "?"))[0].upper(), fill=color)
        draw.text((4, 4), f"cam{camera_id}", fill=(255, 255, 0))
        tiles.append(image)
    if not tiles:
        return
    tile_w, tile_h = tiles[0].size
    cols = min(4, len(tiles))
    rows = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), (0, 0, 0))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % cols) * tile_w, (idx // cols) * tile_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def summarize_side(rows: list[dict[str, Any]], side: str, min_present_ratio: float, strong_present_ratio: float) -> dict[str, Any]:
    side_rows = [row for row in rows if row.get("side") == side]
    mapped = [row for row in side_rows if row.get("depth_present")]
    valid_ratios = np.asarray([float(row.get("depth_valid_ratio_in_bbox", 0.0)) for row in mapped], dtype=np.float32)
    bbox_pixels = int(sum(int(row.get("bbox_scaled_area", 0)) for row in mapped))
    valid_pixels = int(sum(int(row.get("depth_valid_pixels_in_bbox", 0)) for row in mapped))
    return {
        "side": side,
        "roi_count": int(len(side_rows)),
        "mapped_depth_roi_count": int(len(mapped)),
        "missing_depth_roi_count": int(len(side_rows) - len(mapped)),
        "bbox_pixels_mapped": bbox_pixels,
        "depth_valid_pixels_mapped": valid_pixels,
        "depth_valid_ratio_total": float(valid_pixels / max(bbox_pixels, 1)),
        "depth_valid_ratio_per_roi": scalar_stats(valid_ratios),
        "rois_ge_min_present_ratio": int(np.sum(valid_ratios >= float(min_present_ratio))) if valid_ratios.size else 0,
        "rois_ge_strong_present_ratio": int(np.sum(valid_ratios >= float(strong_present_ratio))) if valid_ratios.size else 0,
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    side_summary = summary["side_summary"]
    lines = [
        "# B-Hand6 COLMAP Depth Evidence Probe",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is a research-only probe. It maps B-hand ROI boxes into an existing A5 known-camera",
        "COLMAP PatchMatch workspace and measures whether hand boxes contain valid depth pixels.",
        "It is not a hand decoder, not a teacher, not a candidate, and not a strict pass.",
        "",
        "## Strict Truth",
        "",
        "```text",
        "strict_candidate_passes = 0",
        "strict_teacher_passes = 0",
        "formal_cloud_train_infer_export = blocked",
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
        "## Aggregate Result",
        "",
        "```json",
        json.dumps(summary["aggregate"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Side Summary",
        "",
        "```json",
        json.dumps(side_summary, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Decision",
        "",
        "```json",
        json.dumps(summary["decision"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Contact Sheet",
        "",
        f"- `{summary.get('contact_sheet', '')}`",
        "",
        "## Notes",
        "",
        "- ROI boxes are scaled from the B-hand cache target resolution to the A5 COLMAP depth resolution.",
        "- This measures bbox-level depth presence, not hand mesh correctness or connected-arm Open3D pass.",
        "- A positive depth-presence signal still cannot override B-hand3/B-hand4 connected-hand visual failure.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    hand_cache = load_json(args.hand_evidence_cache)
    workspace = args.a5_workspace.expanduser().resolve()
    view_map, depth_cache = build_a5_view_map(workspace)
    target_size = int(hand_cache.get("target_size", 518) or 518)
    roi_rows = iter_hand_rois(hand_cache)

    results: list[dict[str, Any]] = []
    per_camera_depth: dict[str, np.ndarray] = {}
    draw_rows: list[dict[str, Any]] = []
    for roi_index, roi in enumerate(roi_rows):
        camera_id = roi.get("camera_id", roi.get("view_index", ""))
        a5_entry = None
        for key in norm_camera_keys(camera_id):
            if key in view_map:
                a5_entry = view_map[key]
                break
        row: dict[str, Any] = {
            "roi_index": int(roi_index),
            "side": str(roi.get("side", "unknown")),
            "cache_view_index": roi.get("view_index"),
            "cache_camera_id": str(camera_id),
            "bbox_xyxy": roi.get("bbox_xyxy"),
            "roi_pixels": roi.get("crop_metadata", {}).get("roi_pixels") if isinstance(roi.get("crop_metadata"), dict) else None,
            "bbox_area_pixels_cache": roi.get("crop_metadata", {}).get("bbox_area_pixels") if isinstance(roi.get("crop_metadata"), dict) else None,
            "smplx_visible_ratio_in_roi": roi.get("smplx_prior", {}).get("visible_ratio_in_roi") if isinstance(roi.get("smplx_prior"), dict) else None,
            "prediction_support_ratio_in_roi": roi.get("prediction_support", {}).get("support_ratio_in_roi") if isinstance(roi.get("prediction_support"), dict) else None,
            "depth_present": False,
            "missing_reason": None,
        }
        if a5_entry is None:
            row["missing_reason"] = "camera_not_in_a5_hybrid12_workspace"
            results.append(row)
            continue
        depth = depth_for_view(a5_entry, depth_cache)
        if depth is None:
            row["missing_reason"] = "depth_file_missing_or_unreadable"
            row["mapped_image_name"] = a5_entry.get("image_name")
            row["mapped_camera_id"] = a5_entry.get("scene_camera_id")
            results.append(row)
            continue
        bbox = roi.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            row["missing_reason"] = "missing_bbox_xyxy"
            row["mapped_image_name"] = a5_entry.get("image_name")
            row["mapped_camera_id"] = a5_entry.get("scene_camera_id")
            results.append(row)
            continue
        h, w = depth.shape[:2]
        sx = w / float(target_size)
        sy = h / float(target_size)
        x0, y0, x1, y1 = clamp_bbox_xyxy(bbox, sx, sy, w, h)
        crop = depth[y0:y1, x0:x1]
        valid = np.isfinite(crop) & (crop >= float(args.depth_valid_min)) & (crop <= float(args.depth_valid_max))
        row.update(
            {
                "depth_present": True,
                "mapped_image_name": a5_entry.get("image_name"),
                "mapped_camera_id": str(a5_entry.get("scene_camera_id")),
                "mapped_view_index": int(a5_entry.get("view_index", -1)),
                "depth_shape": [int(h), int(w)],
                "bbox_scaled_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                "bbox_scaled_area": int(max((x1 - x0) * (y1 - y0), 0)),
                "depth_valid_pixels_in_bbox": int(valid.sum()),
                "depth_valid_ratio_in_bbox": float(valid.sum() / max(valid.size, 1)),
                "depth_stats_valid_in_bbox": scalar_stats(crop[valid]),
            }
        )
        results.append(row)
        per_camera_depth[str(a5_entry.get("scene_camera_id"))] = depth
        draw_rows.append(row)

    sides = sorted({str(row.get("side", "unknown")) for row in results})
    side_summary = {
        side: summarize_side(results, side, args.min_present_ratio, args.strong_present_ratio)
        for side in sides
    }
    mapped_rows = [row for row in results if row.get("depth_present")]
    valid_ratios = np.asarray([float(row.get("depth_valid_ratio_in_bbox", 0.0)) for row in mapped_rows], dtype=np.float32)
    total_bbox_pixels = int(sum(int(row.get("bbox_scaled_area", 0)) for row in mapped_rows))
    total_valid_pixels = int(sum(int(row.get("depth_valid_pixels_in_bbox", 0)) for row in mapped_rows))
    aggregate = {
        "roi_count": int(len(results)),
        "mapped_depth_roi_count": int(len(mapped_rows)),
        "missing_depth_roi_count": int(len(results) - len(mapped_rows)),
        "mapped_camera_ids": sorted({str(row.get("mapped_camera_id")) for row in mapped_rows}),
        "missing_camera_ids": sorted({str(row.get("cache_camera_id")) for row in results if not row.get("depth_present")}),
        "bbox_pixels_mapped": total_bbox_pixels,
        "depth_valid_pixels_mapped": total_valid_pixels,
        "depth_valid_ratio_total": float(total_valid_pixels / max(total_bbox_pixels, 1)),
        "depth_valid_ratio_per_roi": scalar_stats(valid_ratios),
        "rois_ge_min_present_ratio": int(np.sum(valid_ratios >= float(args.min_present_ratio))) if valid_ratios.size else 0,
        "rois_ge_strong_present_ratio": int(np.sum(valid_ratios >= float(args.strong_present_ratio))) if valid_ratios.size else 0,
    }
    left = side_summary.get("left", {})
    right = side_summary.get("right", {})
    both_have_mapped = int(left.get("mapped_depth_roi_count", 0)) > 0 and int(right.get("mapped_depth_roi_count", 0)) > 0
    both_have_present = (
        float(left.get("depth_valid_ratio_total", 0.0)) >= float(args.min_present_ratio)
        and float(right.get("depth_valid_ratio_total", 0.0)) >= float(args.min_present_ratio)
    )
    decision = {
        "status": "research_depth_presence_only_no_pass",
        "colmap_depth_present_for_both_hands": bool(both_have_mapped and both_have_present),
        "depth_presence_interpretation": (
            "bbox_level_depth_signal_exists_but_not_connected_hand_surface"
            if both_have_mapped and both_have_present
            else "depth_signal_missing_or_too_sparse_for_at_least_one_hand"
        ),
        "next_allowed_action": (
            "Use as weak visibility/depth evidence for B-hand token backend only; do not export teacher/candidate."
        ),
        "blocked_actions": [
            "do_not_claim_hand_gate_pass",
            "do_not_use_bbox_depth_presence_as_teacher",
            "do_not_restart_view_or_threshold_loop",
            "do_not_unblock_formal_cloud",
        ],
    }

    contact_sheet = output_dir / "b_hand_colmap_depth_valid_bbox_contact.png"
    draw_contact_sheet(
        contact_sheet,
        per_camera_depth,
        draw_rows,
        valid_min=float(args.depth_valid_min),
        valid_max=float(args.depth_valid_max),
    )

    arrays_path = output_dir / "b_hand_colmap_depth_evidence_arrays.npz"
    np.savez_compressed(
        arrays_path,
        depth_valid_ratio=np.asarray([float(row.get("depth_valid_ratio_in_bbox", np.nan)) for row in results], dtype=np.float32),
        depth_present=np.asarray([bool(row.get("depth_present")) for row in results], dtype=bool),
        side=np.asarray([str(row.get("side", "unknown")) for row in results]),
        cache_camera_id=np.asarray([str(row.get("cache_camera_id", "")) for row in results]),
        mapped_camera_id=np.asarray([str(row.get("mapped_camera_id", "")) for row in results]),
    )

    summary = {
        "task": "b_hand_colmap_depth_evidence_probe",
        "status": "research_only_depth_evidence_probe_not_decoder_not_teacher_not_candidate",
        "contract": CONTRACT,
        "strict_gate_truth": STRICT_FACTS,
        "inputs": {
            "hand_evidence_cache": str(args.hand_evidence_cache.expanduser().resolve()),
            "a5_workspace": str(workspace),
            "depth_valid_min": float(args.depth_valid_min),
            "depth_valid_max": float(args.depth_valid_max),
            "hand_cache_target_size": int(target_size),
        },
        "aggregate": aggregate,
        "side_summary": side_summary,
        "per_roi": results,
        "decision": decision,
        "contact_sheet": str(contact_sheet),
        "arrays": str(arrays_path),
    }
    summary_path = output_dir / "b_hand_colmap_depth_evidence_summary.json"
    md_path = output_dir / "b_hand_colmap_depth_evidence_summary.md"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, json_ready(summary))
    if args.status_report:
        write_markdown(args.status_report.expanduser().resolve(), json_ready(summary))
    print(json.dumps(json_ready({"summary": str(summary_path), "decision": decision, "aggregate": aggregate}), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
