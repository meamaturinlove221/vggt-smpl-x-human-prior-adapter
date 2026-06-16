from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from v10_surface_completion_pipeline import (
    LOCAL_ROOT,
    REGIONS,
    REPORTS,
    bbox_stats,
    contact_sheet,
    json_ready,
    load_ply_xyz_rgb,
    load_template,
    paste_grid,
    read_summary,
    region_masks_from_template,
    scalar_stats,
    select_region_by_bbox,
    write_json,
)


K1_SUMMARY = REPORTS / "20260508_v13_kinect_alignment_autopsy.json"
K2_SUMMARY = REPORTS / "20260508_v13_k2_kinect_alignment_candidates.json"
K2B_SUMMARY = REPORTS / "20260508_v13_k2b_kinect_temporal_offset_sweep.json"
REGISTRY = REPORTS / "20260508_v13_strict_gate_registry_refresh.json"
K2_DIR = LOCAL_ROOT / "V13_K2_kinect_alignment_candidate_search"
K3_DIR = LOCAL_ROOT / "V13_K3_kinect_depth_teacher_targets_camera_axes"
G3_PLY = LOCAL_ROOT / "V11_G3_2DGS_surface_anchor/g3_2dgs_anchor_surface.ply"
OUT = LOCAL_ROOT / "V14_K14" / "kinect_teacher_gate_autopsy"
REPORT_DIR = REPORTS / "V14_K14"

CONTRACT = {
    "research_only": True,
    "no_predictions_write": True,
    "no_teacher_targets_write": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_registry_write": True,
    "no_package_write": True,
    "no_strict_pass_write": True,
    "formal_cloud_unblocked": False,
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def percentile_value(values: list[Any], index: int) -> float | None:
    if len(values) <= index:
        return None
    return as_float(values[index])


def ratio_to_limit(value: float | None, limit: float | None) -> float | None:
    if value is None or limit is None or limit <= 0:
        return None
    return float(value / limit)


def extent_volume(bbox: dict[str, Any]) -> float | None:
    extent = bbox.get("extent_p01_p99")
    if not isinstance(extent, list) or len(extent) != 3:
        return None
    vals = [as_float(v) for v in extent]
    if any(v is None or v <= 0 for v in vals):
        return None
    return float(vals[0] * vals[1] * vals[2])


def extent_ratio(src: dict[str, Any], dst: dict[str, Any]) -> list[float | None]:
    src_extent = src.get("extent_p01_p99") if isinstance(src, dict) else None
    dst_extent = dst.get("extent_p01_p99") if isinstance(dst, dict) else None
    if not isinstance(src_extent, list) or not isinstance(dst_extent, list):
        return [None, None, None]
    out: list[float | None] = []
    for a, b in zip(src_extent[:3], dst_extent[:3], strict=False):
        av = as_float(a)
        bv = as_float(b)
        out.append(float(av / bv) if av is not None and bv and bv > 0 else None)
    return out


def sample_points(points: np.ndarray, limit: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) <= limit:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), size=limit, replace=False)]


def nearest_neighbor_stats(src: np.ndarray, dst: np.ndarray, *, limit: int, seed: int) -> dict[str, Any]:
    src_sample = sample_points(src, limit, seed)
    dst_sample = sample_points(dst, limit, seed + 1)
    if len(src_sample) == 0 or len(dst_sample) == 0:
        return {"valid": False, "src_count": int(len(src_sample)), "dst_count": int(len(dst_sample))}
    tree = cKDTree(dst_sample)
    dist, _ = tree.query(src_sample, k=1, workers=-1)
    return {
        "valid": True,
        "src_count": int(len(src_sample)),
        "dst_count": int(len(dst_sample)),
        "nn_distance": scalar_stats(dist),
    }


def camera_baseline_stats(per_view: list[dict[str, Any]]) -> dict[str, Any]:
    real = []
    vggt = []
    for row in per_view:
        real_point = row.get("real_center_targetcam")
        vggt_point = row.get("vggt_center_world")
        if isinstance(real_point, list) and isinstance(vggt_point, list) and len(real_point) == 3 and len(vggt_point) == 3:
            real.append(real_point)
            vggt.append(vggt_point)
    if len(real) < 2:
        return {"valid": False, "count": len(real)}
    real_arr = np.asarray(real, dtype=np.float64)
    vggt_arr = np.asarray(vggt, dtype=np.float64)
    real_from_0 = np.linalg.norm(real_arr - real_arr[0:1], axis=1)
    vggt_from_0 = np.linalg.norm(vggt_arr - vggt_arr[0:1], axis=1)
    nz = real_from_0 > 1e-8
    baseline_ratio = vggt_from_0[nz] / np.clip(real_from_0[nz], 1e-12, None)
    return {
        "valid": True,
        "camera_count": int(len(real_arr)),
        "real_distance_from_view0": scalar_stats(real_from_0[nz]),
        "vggt_distance_from_view0": scalar_stats(vggt_from_0[nz]),
        "vggt_to_real_baseline_ratio": scalar_stats(baseline_ratio),
    }


def region_coverage(points: np.ndarray) -> dict[str, Any]:
    try:
        template = load_template()
        masks = region_masks_from_template(template)
        template_points = np.asarray(template["hybrid_vertices"], dtype=np.float32)
    except Exception as exc:
        return {"available": False, "reason": str(exc)}

    rows: dict[str, Any] = {"available": True}
    for name in REGIONS:
        fallback = (0.65, 1.0) if name in {"head", "face_core", "hairline"} else (0.0, 1.0)
        mask = select_region_by_bbox(points, template_points[masks[name]], fallback_fraction=fallback)
        threshold = 100 if name == "full_body" else 20
        rows[name] = {
            "point_count": int(mask.sum()),
            "coverage_nonempty": bool(mask.sum() > threshold),
            "threshold": int(threshold),
            "bbox": bbox_stats(points[mask]),
            "diagnostic_only": True,
        }
    rows["note"] = "BBox-normalized region buckets are an autopsy lens only; they do not certify anatomical teacher ownership."
    return rows


def summarize_registry(registry: dict[str, Any]) -> dict[str, Any]:
    rows = registry.get("kinect_coord_audits", [])
    if not isinstance(rows, list):
        rows = []
    roi_passes = []
    all_roi = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        slim = {
            "name": row.get("name"),
            "roi_kind": row.get("roi_kind"),
            "pass": bool(row.get("pass")),
            "teacher_targets_written": bool(row.get("teacher_targets_written")),
            "alignment_residual_p50": row.get("alignment_residual_p50"),
            "distance_to_base_p50": row.get("distance_to_base_p50"),
            "view_passes": row.get("view_passes"),
            "view_total": row.get("view_total"),
            "failed": row.get("failed", []),
        }
        if row.get("pass"):
            roi_passes.append(slim)
        if row.get("roi_kind") == "all":
            all_roi.append(slim)
    return {
        "strict_teacher_passes": registry.get("counts", {}).get("strict_teacher_passes"),
        "strict_candidate_passes": registry.get("counts", {}).get("strict_candidate_passes"),
        "kinect_coord_passes": registry.get("counts", {}).get("kinect_coord_passes"),
        "kinect_coord_pass_rows": roi_passes,
        "all_roi_kinect_rows": all_roi[:8],
        "note": "Registry Kinect coordinate passes are diagnostic coordinate audits, not strict full mentor teacher passes.",
    }


def make_overlay_sheet(overlay_dir: Path, out_path: Path, per_view: list[dict[str, Any]]) -> Path | None:
    if not overlay_dir.is_dir():
        return None
    row_by_index = {int(row.get("view_index", -1)): row for row in per_view if isinstance(row, dict)}
    images: list[Image.Image] = []
    for path in sorted(overlay_dir.glob("*.png"))[:12]:
        image = Image.open(path).convert("RGB").resize((220, 220), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(image)
        idx = int(path.name.split("_", 1)[0]) if path.name[:2].isdigit() else -1
        row = row_by_index.get(idx, {})
        hit = as_float(row.get("hit_ratio_in_roi"))
        label = f"{path.name[:12]} hit={hit:.3f}" if hit is not None else path.name[:18]
        draw.rectangle((0, 0, 219, 22), fill=(0, 0, 0))
        draw.text((5, 5), label, fill=(255, 255, 255))
        images.append(image)
    if not images:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paste_grid(images, cols=4, bg=(20, 20, 20)).save(out_path)
    return out_path


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    official = summary["official_teacher_gate"]
    protocol = summary["k2_protocol_mismatch"]
    camera = summary["camera_autopsy"]
    roi = summary["crop_roi_autopsy"]
    depth = summary["depth_autopsy"]
    lines = [
        "# V14 K14 Kinect Teacher Gate Autopsy",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only. This report writes no predictions, teacher targets, package, registry entry, or strict pass state.",
        "",
        "## Bottom Line",
        "",
        summary["decision"],
        "",
        "## Why K2 0.02024 Does Not Pass",
        "",
        (
            f"- K2 best is `{protocol.get('k2_best_candidate')}` with one-way NN median "
            f"{protocol.get('k2_best_score_to_g3_median')} against the G3 anchor, but the official gate is not that metric."
        ),
        (
            f"- Official camera/depth gate failed residual p50={official.get('alignment_residual_p50')} "
            f"(max {official.get('alignment_residual_p50_max')}) and distance-to-base p50={official.get('distance_to_base_p50')} "
            f"(max {official.get('distance_to_base_p50_max')})."
        ),
        (
            f"- Visibility is not the main blocker here: all-ROI view pass ratio is "
            f"{roi.get('view_pass_ratio')} with min hit ratio {roi.get('min_hit_ratio')}."
        ),
        (
            f"- Camera-axis similarity scale is {camera.get('similarity_scale')}; camera correspondence residual p50 remains "
            f"{official.get('alignment_residual_p50')}, so the real Kinect/RGB rig does not land in the VGGT/base-prediction world cleanly."
        ),
        "",
        "## Failed Official Checks",
        "",
    ]
    failed = official.get("failed_checks", {})
    if failed:
        for name, payload in failed.items():
            lines.append(f"- `{name}`: value={payload.get('value')} max={payload.get('max')}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Protocol Notes",
            "",
            "- K2/G3 nearest-neighbor scoring is a local surface proximity diagnostic and can be fooled by a small Kinect cloud sitting inside a larger anchor cloud.",
            "- K3 uses the official projected Kinect-depth protocol: calibrated RGB/Kinect camera geometry, ROI hit masks, camera-axis alignment, and depth distance to the base VGGT predictions.",
            "- K2b temporal offsets did not move the official residuals into gate range.",
            "",
            "## Artifacts",
            "",
        ]
    )
    for key, value in summary.get("artifacts", {}).items():
        if value:
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Blockers", ""])
    blockers = summary.get("blockers") or []
    lines.extend([f"- {item}" for item in blockers] if blockers else ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="V14 K14 Kinect teacher-gate autopsy.")
    parser.add_argument("--k1-summary", type=Path, default=K1_SUMMARY)
    parser.add_argument("--k2-summary", type=Path, default=K2_SUMMARY)
    parser.add_argument("--k2b-summary", type=Path, default=K2B_SUMMARY)
    parser.add_argument("--k3-summary", type=Path, default=K3_DIR / "kinect_teacher_summary.json")
    parser.add_argument("--registry-summary", type=Path, default=REGISTRY)
    parser.add_argument("--g3-ply", type=Path, default=G3_PLY)
    parser.add_argument("--k3-hits-ply", type=Path, default=K3_DIR / "kinect_teacher_vggt_world_hits.ply")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--max-nn-points", type=int, default=90000)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report_dir = REPORT_DIR.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    k1 = read_summary(args.k1_summary)
    k2 = read_summary(args.k2_summary)
    k2b = read_summary(args.k2b_summary)
    k3 = read_summary(args.k3_summary)
    registry = read_summary(args.registry_summary)

    gate = k3.get("teacher_gate", {})
    checks = gate.get("checks", {})
    residuals = gate.get("residual_percentiles", []) or k3.get("target_alignment", {}).get("residual_percentiles", [])
    distances = gate.get("distance_to_base_percentiles", []) or k3.get("distance_to_base_on_teacher_mask_percentiles", [])
    per_view_visibility = checks.get("per_view_visibility", {})
    per_view = k3.get("per_view", [])
    views = per_view_visibility.get("views", [])
    hit_ratios = [float(row.get("hit_ratio_in_roi", 0.0)) for row in views if isinstance(row, dict)]
    connected = [float(row.get("largest_hit_component_ratio", 0.0)) for row in views if isinstance(row, dict)]
    failed_checks = {name: item for name, item in checks.items() if isinstance(item, dict) and item.get("ok") is False}

    k2_best_path = Path(k2.get("best_surface") or K2_DIR / "k2_best_kinect_tsdf_vggt_world.ply")
    k2_points, k2_colors = load_ply_xyz_rgb(k2_best_path, max_points=None) if k2_best_path.is_file() else (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8))
    k3_points, k3_colors = load_ply_xyz_rgb(args.k3_hits_ply, max_points=None) if args.k3_hits_ply.is_file() else (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8))
    g3_points, g3_colors = load_ply_xyz_rgb(args.g3_ply, max_points=None) if args.g3_ply.is_file() else (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8))

    k2_bbox = bbox_stats(k2_points)
    k3_bbox = bbox_stats(k3_points)
    g3_bbox = bbox_stats(g3_points)
    k1_bboxes = k1.get("bbox", {})
    k1_vggt_bbox = k1_bboxes.get("kinect_vggt", {}) if isinstance(k1_bboxes, dict) else {}
    k2_vol = extent_volume(k2_bbox)
    g3_vol = extent_volume(g3_bbox)

    k2_contact = out / "k14_k2_best_contact_sheet.png"
    k3_contact = out / "k14_k3_official_hits_contact_sheet.png"
    g3_contact = out / "k14_g3_anchor_contact_sheet.png"
    if len(k2_points):
        contact_sheet(k2_points, k2_colors, k2_contact, "K14 K2 best")
    if len(k3_points):
        contact_sheet(k3_points, k3_colors, k3_contact, "K14 K3 official hits")
    if len(g3_points):
        contact_sheet(g3_points, g3_colors, g3_contact, "K14 G3 anchor")
    overlay_sheet = make_overlay_sheet(K3_DIR / "overlays", out / "k14_k3_roi_hit_overlay_sheet.png", per_view)

    official = {
        "pass": bool(gate.get("pass")),
        "teacher_targets_written": bool(k3.get("teacher_targets_written")),
        "roi_kind": k3.get("roi_kind"),
        "alignment_source": checks.get("non_circular_alignment", {}).get("alignment_source") or k3.get("alignment", {}).get("alignment_source"),
        "transform_mode": k3.get("transform_mode"),
        "alignment_residual_p50": percentile_value(residuals, 2),
        "alignment_residual_p50_max": as_float(checks.get("alignment_residual_p50", {}).get("max")),
        "alignment_residual_p95": percentile_value(residuals, 5),
        "alignment_residual_p95_max": as_float(checks.get("alignment_residual_p95", {}).get("max")),
        "distance_to_base_p50": percentile_value(distances, 2),
        "distance_to_base_p50_max": as_float(checks.get("distance_to_base_p50", {}).get("max")),
        "distance_to_base_p95": percentile_value(distances, 5),
        "distance_to_base_p95_max": as_float(checks.get("distance_to_base_p95", {}).get("max")),
        "failed_checks": failed_checks,
    }
    official["threshold_ratios"] = {
        "alignment_residual_p50": ratio_to_limit(official["alignment_residual_p50"], official["alignment_residual_p50_max"]),
        "alignment_residual_p95": ratio_to_limit(official["alignment_residual_p95"], official["alignment_residual_p95_max"]),
        "distance_to_base_p50": ratio_to_limit(official["distance_to_base_p50"], official["distance_to_base_p50_max"]),
        "distance_to_base_p95": ratio_to_limit(official["distance_to_base_p95"], official["distance_to_base_p95_max"]),
    }

    k2_protocol = {
        "k2_best_candidate": k2.get("best_candidate"),
        "k2_best_score_to_g3_median": k2.get("best_score_to_g3_median"),
        "k2_best_surface": str(k2_best_path),
        "k2_metric_is_one_way_nn_to_g3": True,
        "k2_bbox": k2_bbox,
        "k1_kinect_vggt_bbox": k1_vggt_bbox,
        "g3_bbox": g3_bbox,
        "k3_official_hits_bbox": k3_bbox,
        "k2_extent_ratio_vs_g3": extent_ratio(k2_bbox, g3_bbox),
        "k3_hits_extent_ratio_vs_g3": extent_ratio(k3_bbox, g3_bbox),
        "k2_bbox_volume": k2_vol,
        "g3_bbox_volume": g3_vol,
        "k2_volume_ratio_vs_g3": float(k2_vol / g3_vol) if k2_vol is not None and g3_vol else None,
        "bidirectional_nn_sample": {
            "k2_to_g3": nearest_neighbor_stats(k2_points, g3_points, limit=int(args.max_nn_points), seed=1410),
            "g3_to_k2": nearest_neighbor_stats(g3_points, k2_points, limit=int(args.max_nn_points), seed=1412),
            "k3_hits_to_g3": nearest_neighbor_stats(k3_points, g3_points, limit=int(args.max_nn_points), seed=1414),
            "g3_to_k3_hits": nearest_neighbor_stats(g3_points, k3_points, limit=int(args.max_nn_points), seed=1416),
        },
        "interpretation": (
            "Low K2 residual is a point-cloud proximity score against a dense anchor. It does not test calibrated "
            "camera correspondences, per-pixel teacher depth against base predictions, or full teacher eligibility."
        ),
    }

    alignment = k3.get("alignment", {})
    transform = k3.get("target_alignment", {})
    camera = {
        "alignment_source": alignment.get("alignment_source"),
        "camera_correspondences": alignment.get("camera_correspondences"),
        "camera_center_correspondences": alignment.get("camera_center_correspondences"),
        "axis_endpoint_correspondences": alignment.get("axis_endpoint_correspondences"),
        "real_center_baseline_median_targetcam_units": alignment.get("real_center_baseline_median_targetcam_units"),
        "axis_length_targetcam_units": alignment.get("axis_length_targetcam_units"),
        "similarity_scale": transform.get("scale"),
        "similarity_translation": transform.get("translation"),
        "camera_baselines": camera_baseline_stats(alignment.get("per_view", [])),
        "interpretation": (
            "The accepted non-circular camera_axes route still leaves large residuals, so the failure is a real camera/protocol "
            "alignment mismatch rather than a point-alignment bookkeeping pass."
        ),
    }

    temporal_rows = []
    for row in k2b.get("rows", []) if isinstance(k2b.get("rows", []), list) else []:
        metrics = row.get("metrics", {}) if isinstance(row, dict) else {}
        temporal_rows.append(
            {
                "frame": row.get("frame"),
                "returncode": row.get("returncode"),
                "teacher_gate_pass": metrics.get("teacher_gate_pass"),
                "residual_p50": metrics.get("residual_p50"),
                "distance_p50": metrics.get("distance_p50"),
                "view_pass_ratio": metrics.get("view_pass_ratio"),
                "min_hit_ratio": metrics.get("min_hit_ratio"),
                "score": row.get("score"),
            }
        )

    crop_roi = {
        "roi_kind": k3.get("roi_kind"),
        "view_pass_ratio": per_view_visibility.get("view_pass_ratio"),
        "view_passes": per_view_visibility.get("view_passes"),
        "view_total": per_view_visibility.get("view_total"),
        "min_hit_ratio": min(hit_ratios) if hit_ratios else None,
        "mean_hit_ratio": float(np.mean(hit_ratios)) if hit_ratios else None,
        "hit_ratio_stats": scalar_stats(np.asarray(hit_ratios, dtype=np.float64)) if hit_ratios else {"count": 0},
        "connected_component_ratio_stats": scalar_stats(np.asarray(connected, dtype=np.float64)) if connected else {"count": 0},
        "interpretation": "All-ROI crop visibility passes in K3; crop coverage is not the primary reason this K3 run fails.",
    }

    per_camera_depth_p50 = []
    for row in (k3.get("kinect_stats", {}).get("per_camera", {}) or {}).values():
        percentiles = row.get("depth_m_percentiles", []) if isinstance(row, dict) else []
        value = percentile_value(percentiles, 2)
        if value is not None:
            per_camera_depth_p50.append(value)
    depth = {
        "selected_kinect_cameras": k3.get("selected_kinect_cameras"),
        "per_camera_depth_median_stats": scalar_stats(np.asarray(per_camera_depth_p50, dtype=np.float64)) if per_camera_depth_p50 else {"count": 0},
        "official_distance_to_base_percentiles": distances,
        "temporal_sweep_best_frame": k2b.get("best_frame"),
        "temporal_sweep_best_metrics": k2b.get("best_metrics", {}),
        "temporal_sweep_rows": temporal_rows,
        "interpretation": "Temporal offsets do not fix the depth-to-base residual; the official teacher depth remains far from VGGT base predictions.",
    }

    regions = {
        "k2_best": region_coverage(k2_points),
        "k3_official_hits": region_coverage(k3_points),
        "note": "Region counts are diagnostic and are not a strict full-body/hand visual pass.",
    }

    registry_autopsy = summarize_registry(registry) if registry else {}

    blockers = []
    if failed_checks:
        blockers.append(f"Official teacher gate failed checks: {', '.join(failed_checks.keys())}.")
    if k2.get("best_score_to_g3_median") is not None:
        blockers.append("K2 0.02024 is a G3 nearest-neighbor diagnostic, not the official camera/depth teacher objective.")
    if crop_roi["view_pass_ratio"] is not None and crop_roi["view_pass_ratio"] >= 0.8:
        blockers.append("K3 visibility/crop coverage passes, so residual/camera/depth mismatch remains the blocker.")
    if k2_protocol["k2_volume_ratio_vs_g3"] is not None and k2_protocol["k2_volume_ratio_vs_g3"] < 0.01:
        blockers.append("K2 best cloud is orders of magnitude smaller than the G3 anchor by p01-p99 bbox volume.")
    blockers.append("No strict teacher pass, registry write, package, or training unblock is produced by K14.")

    artifacts = {
        "output_summary": str(out / "summary.json"),
        "output_report": str(out / "report.md"),
        "report_json": str(report_dir / "20260508_k14_kinect_teacher_gate_autopsy.json"),
        "report_md": str(report_dir / "20260508_k14_kinect_teacher_gate_autopsy.md"),
        "k2_best_contact_sheet": str(k2_contact) if k2_contact.is_file() else None,
        "k3_official_hits_contact_sheet": str(k3_contact) if k3_contact.is_file() else None,
        "g3_anchor_contact_sheet": str(g3_contact) if g3_contact.is_file() else None,
        "k3_roi_hit_overlay_sheet": str(overlay_sheet) if overlay_sheet else None,
    }

    summary = {
        "task": "v14_k14_kinect_teacher_gate_autopsy",
        "created_utc": utc_now(),
        "status": "k14_teacher_gate_autopsy_complete",
        **CONTRACT,
        "inputs": {
            "k1_summary": str(args.k1_summary.resolve()),
            "k2_summary": str(args.k2_summary.resolve()),
            "k2b_summary": str(args.k2b_summary.resolve()),
            "k3_summary": str(args.k3_summary.resolve()),
            "registry_summary": str(args.registry_summary.resolve()),
            "k2_best_surface": str(k2_best_path.resolve()) if k2_best_path.exists() else str(k2_best_path),
            "k3_hits_ply": str(args.k3_hits_ply.resolve()) if args.k3_hits_ply.exists() else str(args.k3_hits_ply),
            "g3_ply": str(args.g3_ply.resolve()) if args.g3_ply.exists() else str(args.g3_ply),
        },
        "official_teacher_gate": official,
        "k2_protocol_mismatch": k2_protocol,
        "camera_autopsy": camera,
        "depth_autopsy": depth,
        "crop_roi_autopsy": crop_roi,
        "region_autopsy": regions,
        "registry_context": registry_autopsy,
        "artifacts": artifacts,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "strict_teacher_precheck_pass": False,
        "teacher_targets_written": False,
        "decision": (
            "K2 residual 0.02024 is real but insufficient: it is a one-way G3 proximity diagnostic. "
            "The official Kinect teacher gate fails the calibrated camera/depth protocol, especially camera-axis "
            "alignment residual and distance-to-base thresholds, while all-ROI visibility itself is mostly adequate."
        ),
        "blockers": blockers,
    }

    write_json(out / "summary.json", summary)
    write_json(report_dir / "20260508_k14_kinect_teacher_gate_autopsy.json", summary)
    write_markdown(out / "report.md", summary)
    write_markdown(report_dir / "20260508_k14_kinect_teacher_gate_autopsy.md", summary)
    print(
        json.dumps(
            json_ready(
                {
                    "status": summary["status"],
                    "strict_teacher_precheck_pass": False,
                    "official_failed_checks": list(failed_checks.keys()),
                    "output": out,
                }
            ),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
