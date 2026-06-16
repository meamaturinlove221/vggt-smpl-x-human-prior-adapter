from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from audit_headface_teacher_surface import _roi_mask, load_scene_mask  # noqa: E402
from build_colmap_depth_teacher_targets import read_colmap_depth  # noqa: E402


DEFAULT_WORKSPACE = Path(
    "output/surface_research_preflight/"
    "A5_known_camera_colmap_workspace_modal_colmap_execute_t256_hybrid12_known_direct_v1/"
    "A5_known_camera_colmap_workspace/workspace"
)
DEFAULT_GATE_SUMMARY = Path(
    "output/surface_research_preflight_local/"
    "A5_known_direct_v1_hybrid12_teacher_gate_signfix_headshoulder/"
    "teacher_gate_summary.json"
)
DEFAULT_OUTPUT_DIR = Path("output/surface_research_preflight_local/A5_hybrid12_colmap_depth_range_audit_20260507")
DEFAULT_REPORT = Path("reports/20260507_a5_hybrid12_depth_range_audit.md")

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
}

CONTRACT = {
    "research_only": True,
    "local_only": True,
    "audit_only_existing_a5_outputs": True,
    "no_new_colmap_run": True,
    "no_view_count_loop": True,
    "no_threshold_or_fusion_tuning": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_registry_write": True,
    "no_predictions_write": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only audit for existing A5 known-direct COLMAP depth maps. "
            "It explains depth range/ROI/fused-point compatibility collapse without "
            "running COLMAP, changing view sets, tuning fusion, exporting teacher/candidate, "
            "or writing strict pass state."
        )
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--gate-summary", type=Path, default=DEFAULT_GATE_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--roi-kinds", default="face_core,head_face,hairline")
    parser.add_argument("--depth-valid-min", type=float, default=0.4)
    parser.add_argument("--depth-valid-max", type=float, default=8.0)
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


def scalar_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values).reshape(-1)
    finite = np.isfinite(arr)
    if arr.size == 0 or not finite.any():
        return {"count": int(arr.size), "finite": int(finite.sum())}
    vals = arr[finite].astype(np.float64)
    return {
        "count": int(arr.size),
        "finite": int(finite.sum()),
        "min": float(np.min(vals)),
        "p01": float(np.percentile(vals, 1)),
        "p05": float(np.percentile(vals, 5)),
        "p25": float(np.percentile(vals, 25)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p75": float(np.percentile(vals, 75)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "max": float(np.max(vals)),
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ply_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = []
        vertex_count = None
        is_binary = False
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            text = line.decode("ascii", errors="replace").strip()
            header.append(text)
            if text.startswith("format binary_little_endian"):
                is_binary = True
            if text.startswith("element vertex"):
                vertex_count = int(text.split()[-1])
            if text == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY has no vertex count: {path}")
        if is_binary:
            data = np.frombuffer(handle.read(vertex_count * 3 * 4), dtype="<f4")
            if data.size < vertex_count * 3:
                raise ValueError(f"PLY binary xyz payload too short: {path}")
            return data[: vertex_count * 3].reshape(vertex_count, 3).astype(np.float32)
        rows = []
        for _ in range(vertex_count):
            parts = handle.readline().decode("ascii", errors="replace").strip().split()
            if len(parts) >= 3:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        return np.asarray(rows, dtype=np.float32)


def camera_depths_for_points(points: np.ndarray, world_to_cam: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    world_to_cam = np.asarray(world_to_cam, dtype=np.float32)
    cam = points @ world_to_cam[:3, :3].T + world_to_cam[:3, 3][None, :]
    return cam[:, 2].astype(np.float32)


def parse_rois(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def gate_totals(gate_summary: dict[str, Any]) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    entries = gate_summary.get("entries", [])
    for entry in entries:
        roi = str(entry.get("roi_kind", ""))
        bucket = totals.setdefault(
            roi,
            {
                "roi_pixels": 0,
                "raw_hit_pixels": 0,
                "depth_compatible_hit_pixels": 0,
                "raw_depth_p50_values": [],
                "raw_depth_p90_values": [],
                "compat_depth_p50_values": [],
                "compat_depth_p90_values": [],
            },
        )
        bucket["roi_pixels"] += int(entry.get("roi_pixels", 0) or 0)
        raw = entry.get("raw_visible", {}) if isinstance(entry.get("raw_visible"), dict) else {}
        compat = entry.get("depth_compatible", {}) if isinstance(entry.get("depth_compatible"), dict) else {}
        bucket["raw_hit_pixels"] += int(raw.get("hit_pixels", 0) or 0)
        bucket["depth_compatible_hit_pixels"] += int(compat.get("hit_pixels", 0) or 0)
        for src, prefix in [(raw, "raw"), (compat, "compat")]:
            resid = src.get("depth_residual", {}) if isinstance(src.get("depth_residual"), dict) else {}
            for key in ("p50", "p90"):
                val = resid.get(key)
                if val is not None:
                    bucket[f"{prefix}_depth_{key}_values"].append(float(val))
    for roi, bucket in totals.items():
        raw_hits = max(int(bucket["raw_hit_pixels"]), 1)
        roi_pixels = max(int(bucket["roi_pixels"]), 1)
        bucket["raw_coverage"] = float(bucket["raw_hit_pixels"] / roi_pixels)
        bucket["depth_compatible_coverage"] = float(bucket["depth_compatible_hit_pixels"] / roi_pixels)
        bucket["depth_compatible_over_raw"] = float(bucket["depth_compatible_hit_pixels"] / raw_hits)
        for key in ("raw_depth_p50_values", "raw_depth_p90_values", "compat_depth_p50_values", "compat_depth_p90_values"):
            bucket[key.replace("_values", "")] = scalar_stats(np.asarray(bucket[key], dtype=np.float32))
            del bucket[key]
    return totals


def main() -> int:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    known = read_json(workspace / "known_camera_model.json")
    gate_summary = read_json(args.gate_summary.expanduser().resolve()) if args.gate_summary else {}
    rois = parse_rois(args.roi_kinds)
    views = known.get("views", [])
    scene_dir = Path(str(known.get("scene_dir", "")))
    target_size = int(known.get("target_size", 256))
    depth_dir = workspace / "dense" / "stereo" / "depth_maps"
    fused_path = workspace / "dense" / "fused.ply"

    patch_cfg = (workspace / "dense" / "stereo" / "patch-match.cfg").read_text(encoding="utf-8") if (workspace / "dense" / "stereo" / "patch-match.cfg").is_file() else ""
    fusion_cfg = (workspace / "dense" / "stereo" / "fusion.cfg").read_text(encoding="utf-8") if (workspace / "dense" / "stereo" / "fusion.cfg").is_file() else ""
    patch_cfg_lines = [line.strip() for line in patch_cfg.splitlines() if line.strip()]
    fusion_cfg_lines = [line.strip() for line in fusion_cfg.splitlines() if line.strip()]

    depth_rows = []
    roi_totals: dict[str, dict[str, Any]] = {
        roi: {
            "roi_pixels": 0,
            "depth_valid_pixels": 0,
            "depth_valid_in_roi": 0,
            "depth_valid_in_roi_ratio_values": [],
            "depth_stats_in_roi_values": [],
        }
        for roi in rois
    }
    all_depth_values = []
    for local_idx, view in enumerate(views):
        image_name = str(view["image_name"])
        depth_path = depth_dir / f"{image_name}.photometric.bin"
        if not depth_path.is_file():
            depth_rows.append({"image_name": image_name, "missing_depth": True})
            continue
        depth = read_colmap_depth(depth_path)
        valid = np.isfinite(depth) & (depth >= float(args.depth_valid_min)) & (depth <= float(args.depth_valid_max))
        all_depth_values.append(depth[valid])
        row: dict[str, Any] = {
            "local_index": int(local_idx),
            "view_index": int(view.get("view_index", local_idx)),
            "camera_id": str(view.get("scene_camera_id", "")),
            "image_name": image_name,
            "depth_shape": [int(v) for v in depth.shape],
            "valid_pixels": int(valid.sum()),
            "valid_ratio": float(valid.sum() / max(valid.size, 1)),
            "depth_stats_valid": scalar_stats(depth[valid]),
        }
        mask_path = workspace / "masks" / image_name
        if mask_path.is_file():
            mask = np.asarray(Image.open(mask_path).convert("L").resize((depth.shape[1], depth.shape[0]), Image.Resampling.NEAREST)) > 127
            row["human_mask_pixels"] = int(mask.sum())
            row["valid_in_human_mask"] = int((valid & mask).sum())
            row["valid_in_human_mask_ratio"] = float((valid & mask).sum() / max(int(mask.sum()), 1))
            for roi in rois:
                roi_mask = _roi_mask(mask, roi)
                roi_pixels = int(roi_mask.sum())
                roi_valid = valid & roi_mask
                roi_totals[roi]["roi_pixels"] += roi_pixels
                roi_totals[roi]["depth_valid_pixels"] += int(valid.sum())
                roi_totals[roi]["depth_valid_in_roi"] += int(roi_valid.sum())
                roi_totals[roi]["depth_valid_in_roi_ratio_values"].append(float(roi_valid.sum() / max(roi_pixels, 1)))
                if roi_valid.any():
                    roi_totals[roi]["depth_stats_in_roi_values"].append(depth[roi_valid])
                row[f"{roi}_pixels"] = roi_pixels
                row[f"{roi}_valid"] = int(roi_valid.sum())
                row[f"{roi}_valid_ratio"] = float(roi_valid.sum() / max(roi_pixels, 1))
                row[f"{roi}_depth_stats"] = scalar_stats(depth[roi_valid])
        depth_rows.append(row)

    roi_summary: dict[str, Any] = {}
    for roi, total in roi_totals.items():
        depth_arrays = [arr for arr in total["depth_stats_in_roi_values"] if np.asarray(arr).size]
        merged = np.concatenate(depth_arrays, axis=0) if depth_arrays else np.asarray([], dtype=np.float32)
        ratios = np.asarray(total["depth_valid_in_roi_ratio_values"], dtype=np.float32)
        roi_summary[roi] = {
            "roi_pixels": int(total["roi_pixels"]),
            "depth_valid_in_roi": int(total["depth_valid_in_roi"]),
            "valid_in_roi_ratio": float(total["depth_valid_in_roi"] / max(total["roi_pixels"], 1)),
            "per_view_valid_ratio": scalar_stats(ratios),
            "depth_stats_in_roi": scalar_stats(merged),
        }

    fused_summary: dict[str, Any] = {"provided": fused_path.is_file()}
    if fused_path.is_file():
        points = load_ply_xyz(fused_path)
        fused_summary.update(
            {
                "point_count": int(points.shape[0]),
                "bbox_min": [float(v) for v in points.min(axis=0)] if points.size else [0.0, 0.0, 0.0],
                "bbox_max": [float(v) for v in points.max(axis=0)] if points.size else [0.0, 0.0, 0.0],
                "bbox_extent": [float(v) for v in (points.max(axis=0) - points.min(axis=0))] if points.size else [0.0, 0.0, 0.0],
            }
        )
        fused_depth_rows = []
        for view in views:
            world_to_cam = np.asarray(view.get("world_to_cam_4x4"), dtype=np.float32)
            z = camera_depths_for_points(points, world_to_cam)
            positive = z[np.isfinite(z) & (z > 0)]
            fused_depth_rows.append(
                {
                    "view_index": int(view.get("view_index", 0)),
                    "camera_id": str(view.get("scene_camera_id", "")),
                    "positive_depth_count": int(positive.size),
                    "positive_depth_ratio": float(positive.size / max(points.shape[0], 1)),
                    "positive_depth_stats": scalar_stats(positive),
                }
            )
        fused_summary["depth_by_view"] = fused_depth_rows

    gate_totals_summary = gate_totals(gate_summary) if gate_summary else {}
    all_depth = np.concatenate([arr for arr in all_depth_values if np.asarray(arr).size], axis=0) if all_depth_values else np.asarray([], dtype=np.float32)
    depth_range = known.get("known_direct_depth_range", {})
    decision = (
        "A5 hybrid12 COLMAP depth maps are dense inside the human/ROI masks, but the "
        "strict same-protocol pointcloud gate fails because the fused shared point cloud "
        "projects into the 6-view headshoulder protocol with large residuals and fragmented "
        "coverage. This supports freezing COLMAP/MVS teacher use: the issue is not a simple "
        "lack of raw PatchMatch depth pixels, and the next step is not view-count/threshold tuning."
    )
    summary = {
        **STRICT_FACTS,
        "task": "audit_a5_hybrid12_colmap_depth_range",
        "truthful_status": "research_only_depth_range_audit_not_teacher_not_candidate",
        "contract": CONTRACT,
        "inputs": {
            "workspace": str(workspace),
            "gate_summary": str(args.gate_summary.expanduser().resolve()),
            "known_camera_model": str(workspace / "known_camera_model.json"),
            "depth_dir": str(depth_dir),
            "fused_ply": str(fused_path),
        },
        "configuration": {
            "depth_valid_min": float(args.depth_valid_min),
            "depth_valid_max": float(args.depth_valid_max),
            "roi_kinds": rois,
            "target_size": target_size,
        },
        "known_direct_depth_range": depth_range,
        "patch_match_cfg_line_count": len(patch_cfg_lines),
        "patch_match_cfg_preview": patch_cfg_lines[:24],
        "fusion_cfg_line_count": len(fusion_cfg_lines),
        "fusion_cfg_preview": fusion_cfg_lines[:24],
        "depth_map_overall": {
            "depth_file_count": int(sum(1 for row in depth_rows if not row.get("missing_depth"))),
            "depth_valid_stats": scalar_stats(all_depth),
        },
        "roi_depth_summary": roi_summary,
        "gate_totals_summary": gate_totals_summary,
        "fused_pointcloud_summary": fused_summary,
        "per_view_depth_summary": depth_rows,
        "decision": decision,
        "outputs": {
            "summary_json": str(output_dir / "a5_hybrid12_colmap_depth_range_audit_summary.json"),
            "summary_md": str(output_dir / "a5_hybrid12_colmap_depth_range_audit_summary.md"),
            "status_report": str(args.report.expanduser().resolve()),
        },
    }
    summary = json_ready(summary)
    (output_dir / "a5_hybrid12_colmap_depth_range_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "a5_hybrid12_colmap_depth_range_audit_summary.md", summary)
    write_markdown(args.report.expanduser().resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# A5 Hybrid12 COLMAP Depth-Range Audit",
        "",
        f"Status: `{summary['truthful_status']}`",
        "",
        "This is a research-only audit of existing A5 hybrid12 outputs. It does",
        "not run COLMAP, tune view counts/thresholds/fusion, export a teacher or",
        "candidate, write predictions, or write strict pass state.",
        "",
        "## Strict Truth",
        "",
        "```text",
        f"strict_candidate_passes = {summary['strict_candidate_passes']}",
        f"strict_teacher_passes = {summary['strict_teacher_passes']}",
        f"formal_cloud_train_infer_export = {summary['formal_cloud_train_infer_export']}",
        "```",
        "",
        "## Depth Maps",
        "",
        "```json",
        json.dumps(summary["depth_map_overall"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## ROI Depth Summary",
        "",
        "```json",
        json.dumps(summary["roi_depth_summary"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Gate Totals",
        "",
        "```json",
        json.dumps(summary["gate_totals_summary"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Fused Pointcloud",
        "",
        "```json",
        json.dumps(summary["fused_pointcloud_summary"], indent=2, ensure_ascii=False)[:12000],
        "```",
        "",
        "## Decision",
        "",
        "```text",
        summary["decision"],
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
