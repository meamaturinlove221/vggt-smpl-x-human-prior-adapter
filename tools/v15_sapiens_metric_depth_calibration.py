from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from v15_common import (
    DEFAULT_SAPIENS_DEPTH,
    LOCAL_ROOT,
    REPORTS,
    camera_id_overlap,
    fit_affine_depth,
    json_ready,
    resize_bilinear_float,
    resize_nearest,
    safe_v15_output_dir,
    scalar_stats,
    utc_now,
    write_json,
    write_report,
)


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def main() -> int:
    parser = argparse.ArgumentParser(description="V15 Sapiens relative-depth metric calibration audit against 2DGS depth.")
    parser.add_argument("--sapiens-depth-npz", type=Path, default=DEFAULT_SAPIENS_DEPTH)
    parser.add_argument("--metric-depth-npz", type=Path, default=LOCAL_ROOT / "V15_GS_2DGS_true_normal_rasterizer/v15_2dgs_true_depth_6view.npz")
    parser.add_argument("--output-dir", type=Path, default=LOCAL_ROOT / "V15_S_sapiens_metric_depth_calibration")
    args = parser.parse_args()

    out = safe_v15_output_dir(args.output_dir)
    sapiens = _load_npz(args.sapiens_depth_npz)
    metric = _load_npz(args.metric_depth_npz)
    rel_depth = np.asarray(sapiens["depth"], dtype=np.float32)
    rel_mask = np.asarray(sapiens["mask"], dtype=bool)
    rel_names = [str(x) for x in sapiens["image_names"]]
    metric_depth = np.asarray(metric["depth"], dtype=np.float32)
    metric_mask = np.asarray(metric["visibility"], dtype=bool) if "visibility" in metric else np.isfinite(metric_depth)
    metric_names = [str(x) for x in metric["view_names"]]
    overlap = camera_id_overlap(metric_names, rel_names)
    pairs = [(row["g3_index"], row["sapiens_matches"][0]["index"], row["camera_id"]) for row in overlap if row["match_count"] == 1]

    per_view = []
    global_rel = []
    global_metric = []
    global_valid = []
    calibrated_depth = np.zeros_like(rel_depth, dtype=np.float32)
    calibrated_mask = np.zeros_like(rel_mask, dtype=bool)
    for metric_idx, rel_idx, cam_id in pairs:
        m = metric_depth[metric_idx]
        m_valid = metric_mask[metric_idx] & np.isfinite(m)
        r = resize_bilinear_float(rel_depth[rel_idx], m.shape)
        r_valid = resize_nearest(rel_mask[rel_idx], m.shape)
        shared = m_valid & r_valid & np.isfinite(r)
        fit = fit_affine_depth(r, m, shared)
        fit.update({"camera_id": cam_id, "metric_name": metric_names[metric_idx], "sapiens_name": rel_names[rel_idx]})
        per_view.append(fit)
        if fit.get("fit_valid"):
            native_cal = rel_depth[rel_idx] * float(fit["scale"]) + float(fit["bias"])
            calibrated_depth[rel_idx] = native_cal.astype(np.float32)
            calibrated_mask[rel_idx] = rel_mask[rel_idx]
        global_rel.append(r.reshape(-1))
        global_metric.append(m.reshape(-1))
        global_valid.append(shared.reshape(-1))
    if global_rel:
        global_fit = fit_affine_depth(np.concatenate(global_rel), np.concatenate(global_metric), np.concatenate(global_valid))
    else:
        global_fit = {"fit_valid": False, "sample_count": 0}
    if global_fit.get("fit_valid"):
        for idx in range(rel_depth.shape[0]):
            if not calibrated_mask[idx].any():
                calibrated_depth[idx] = rel_depth[idx] * float(global_fit["scale"]) + float(global_fit["bias"])
                calibrated_mask[idx] = rel_mask[idx]

    output_path = out / "v15_sapiens_depth_metric_affine_calibrated_research.npz"
    np.savez_compressed(
        output_path,
        depth=calibrated_depth.astype(np.float32),
        mask=calibrated_mask,
        image_names=np.asarray(rel_names),
        global_scale=np.asarray(global_fit.get("scale", np.nan), dtype=np.float32),
        global_bias=np.asarray(global_fit.get("bias", np.nan), dtype=np.float32),
        research_only=True,
    )

    valid_fits = [row for row in per_view if row.get("fit_valid")]
    median_re = [float(row["median_relative_abs_error"]) for row in valid_fits if "median_relative_abs_error" in row]
    gates = {
        "sapiens_depth_exists": args.sapiens_depth_npz.is_file(),
        "metric_depth_exists": args.metric_depth_npz.is_file(),
        "has_camera_id_overlap": len(pairs) > 0,
        "global_fit_valid": bool(global_fit.get("fit_valid")),
        "global_corr_above_0_25": float(global_fit.get("corr", 0.0) or 0.0) > 0.25,
        "median_relative_error_below_0_25": bool(median_re and float(np.median(median_re)) < 0.25),
    }
    blockers = []
    if not gates["global_corr_above_0_25"]:
        blockers.append(f"Global Sapiens relative-depth vs 2DGS metric-depth correlation is too weak: {float(global_fit.get('corr', 0.0) or 0.0):.4f}.")
    if not gates["median_relative_error_below_0_25"]:
        med = float(np.median(median_re)) if median_re else float("nan")
        blockers.append(f"Per-view affine calibrated median relative depth error is not strict-ready: {med:.4f}.")
    blockers.append("2DGS depth is a sparse/raster research anchor, not a strict fused metric surface.")
    blockers.append("Only 6 overlapping cameras can be calibrated; the remaining 6 Sapiens depths use global affine fallback if exported.")
    status = "v15_sapiens_metric_depth_calibrated_research_only" if gates["global_fit_valid"] else "v15_sapiens_metric_depth_calibration_blocked"
    summary = {
        "task": "v15_sapiens_metric_depth_calibration",
        "created_utc": utc_now(),
        "status": status,
        "research_only": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "strict_teacher_passes": 0,
        "strict_candidate_passes": 0,
        "inputs": {"sapiens_depth_npz": str(args.sapiens_depth_npz.resolve()), "metric_depth_npz": str(args.metric_depth_npz.resolve())},
        "metrics": {
            "matched_view_count": len(pairs),
            "valid_fit_count": len(valid_fits),
            "global_sample_count": int(global_fit.get("sample_count", 0) or 0),
            "global_corr": global_fit.get("corr"),
            "global_mae": global_fit.get("mae"),
            "global_rmse": global_fit.get("rmse"),
            "median_per_view_relative_error": float(np.median(median_re)) if median_re else None,
        },
        "gates": gates,
        "overlap_rows": overlap,
        "global_fit": global_fit,
        "per_view": per_view,
        "outputs": {"calibrated_depth": str(output_path.resolve())},
        "decision": "Sapiens relative depth can be affine-fit to overlapping 2DGS raster depth for research diagnostics, but it is not a strict metric multiview surface calibration.",
        "blockers": blockers,
    }
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v15_sapiens_metric_depth_calibration.json", summary)
    write_report(REPORTS / "20260508_v15_sapiens_metric_depth_calibration.md", "V15 Sapiens Metric Depth Calibration", summary)
    print(json.dumps(json_ready({"status": status, "metrics": summary["metrics"], "output": out}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
