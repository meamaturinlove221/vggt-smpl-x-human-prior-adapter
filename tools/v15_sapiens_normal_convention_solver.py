from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from v15_common import (
    DEFAULT_SAPIENS_NORMAL,
    LOCAL_ROOT,
    REPORTS,
    camera_id_overlap,
    json_ready,
    normal_angle_metrics,
    normalize_vectors,
    parse_camera_id_from_name,
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


def _apply_convention(normals: np.ndarray, signs: tuple[int, int, int], swap_xy: bool) -> np.ndarray:
    arr = np.asarray(normals, dtype=np.float32).copy()
    if swap_xy:
        arr = arr[..., [1, 0, 2]]
    sign_arr = np.asarray(signs, dtype=np.float32)
    arr = arr * sign_arr.reshape((1,) * (arr.ndim - 1) + (3,))
    arr, _ = normalize_vectors(arr)
    return arr


def _candidate_specs() -> list[dict[str, Any]]:
    specs = []
    for signs in itertools.product((-1, 1), repeat=3):
        for swap_xy in (False, True):
            specs.append({"name": f"{'swapxy_' if swap_xy else ''}sign_{signs[0]}_{signs[1]}_{signs[2]}", "signs": signs, "swap_xy": swap_xy})
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description="V15 Sapiens normal convention solver against rasterized 2DGS normals.")
    parser.add_argument("--sapiens-normal-npz", type=Path, default=DEFAULT_SAPIENS_NORMAL)
    parser.add_argument(
        "--reference-normal-npz",
        type=Path,
        default=LOCAL_ROOT / "V15_GS_2DGS_true_normal_rasterizer/v15_2dgs_true_camera_normals_6view.npz",
    )
    parser.add_argument("--output-dir", type=Path, default=LOCAL_ROOT / "V15_S_sapiens_normal_convention_solver")
    args = parser.parse_args()

    out = safe_v15_output_dir(args.output_dir)
    sapiens = _load_npz(args.sapiens_normal_npz)
    ref = _load_npz(args.reference_normal_npz)
    sapiens_normals = np.asarray(sapiens["normal"], dtype=np.float32)
    sapiens_mask = np.asarray(sapiens["mask"], dtype=bool)
    sapiens_names = [str(x) for x in sapiens["image_names"]]
    ref_normals = np.asarray(ref["normal"], dtype=np.float32)
    ref_mask = np.asarray(ref["visibility"], dtype=bool) if "visibility" in ref else np.linalg.norm(ref_normals, axis=-1) > 0.5
    ref_names = [str(x) for x in ref["view_names"]]
    overlap = camera_id_overlap(ref_names, sapiens_names)
    pairs = [(row["g3_index"], row["sapiens_matches"][0]["index"], row["camera_id"]) for row in overlap if row["match_count"] == 1]

    candidate_rows = []
    best: dict[str, Any] | None = None
    for spec in _candidate_specs():
        per_pair = []
        weighted_cos = []
        weighted_abs_angle = []
        total_valid = 0
        for ref_idx, sap_idx, cam_id in pairs:
            ref_view = ref_normals[ref_idx]
            ref_valid = ref_mask[ref_idx] & (np.linalg.norm(ref_view, axis=-1) > 0.5)
            sap_view = resize_bilinear_float(sapiens_normals[sap_idx], ref_view.shape[:2])
            sap_valid = resize_nearest(sapiens_mask[sap_idx], ref_view.shape[:2])
            sap_view = _apply_convention(sap_view, tuple(spec["signs"]), bool(spec["swap_xy"]))
            shared = ref_valid & sap_valid & (np.linalg.norm(sap_view, axis=-1) > 0.5)
            metrics = normal_angle_metrics(sap_view, ref_view, shared)
            metrics.update({"camera_id": cam_id, "ref_name": ref_names[ref_idx], "sapiens_name": sapiens_names[sap_idx]})
            per_pair.append(metrics)
            if metrics.get("valid_pixels", 0):
                count = int(metrics["valid_pixels"])
                total_valid += count
                weighted_cos.append(float(metrics["signed_cos_mean"]) * count)
                weighted_abs_angle.append(float(metrics["abs_angle_mean_deg"]) * count)
        mean_cos = float(sum(weighted_cos) / total_valid) if total_valid else -1.0
        mean_abs_angle = float(sum(weighted_abs_angle) / total_valid) if total_valid else 180.0
        row = {
            "name": spec["name"],
            "signs": list(spec["signs"]),
            "swap_xy": bool(spec["swap_xy"]),
            "matched_views": len(pairs),
            "valid_pixels": int(total_valid),
            "weighted_signed_cos_mean": mean_cos,
            "weighted_abs_angle_mean_deg": mean_abs_angle,
            "per_view": per_pair,
        }
        candidate_rows.append(row)
        if best is None or row["weighted_signed_cos_mean"] > best["weighted_signed_cos_mean"]:
            best = row

    best = best or {}
    best_normals = _apply_convention(sapiens_normals, tuple(best.get("signs", [1, 1, 1])), bool(best.get("swap_xy", False)))
    best_path = out / "v15_sapiens_normals_best_camera_convention.npz"
    np.savez_compressed(
        best_path,
        normal=best_normals.astype(np.float32),
        mask=sapiens_mask,
        image_names=np.asarray(sapiens_names),
        convention_name=str(best.get("name", "")),
        research_only=True,
    )

    gates = {
        "reference_normal_exists": args.reference_normal_npz.is_file(),
        "sapiens_normal_exists": args.sapiens_normal_npz.is_file(),
        "has_camera_id_overlap": len(pairs) > 0,
        "best_has_valid_pixels": int(best.get("valid_pixels", 0) or 0) > 0,
        "best_mean_cos_above_0_5": float(best.get("weighted_signed_cos_mean", -1.0) or -1.0) > 0.5,
    }
    blockers = []
    if not gates["best_mean_cos_above_0_5"]:
        blockers.append("Best Sapiens convention does not strongly agree with 2DGS orientation-normal reference.")
    blockers.append("Reference normals come from 2DGS orientation rasterization, not a strict metric surface teacher.")
    blockers.append("Only the 6 overlapping camera IDs can be scored; the extra 6 Sapiens/TMF views remain unvalidated by G3.")
    status = "v15_sapiens_convention_solved_research_only" if gates["best_has_valid_pixels"] else "v15_sapiens_convention_blocked"
    summary = {
        "task": "v15_sapiens_normal_convention_solver",
        "created_utc": utc_now(),
        "status": status,
        "research_only": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "strict_teacher_passes": 0,
        "strict_candidate_passes": 0,
        "inputs": {"sapiens_normal_npz": str(args.sapiens_normal_npz.resolve()), "reference_normal_npz": str(args.reference_normal_npz.resolve())},
        "metrics": {
            "matched_view_count": len(pairs),
            "best_convention": best.get("name"),
            "best_valid_pixels": int(best.get("valid_pixels", 0) or 0),
            "best_weighted_signed_cos_mean": best.get("weighted_signed_cos_mean"),
            "best_weighted_abs_angle_mean_deg": best.get("weighted_abs_angle_mean_deg"),
        },
        "gates": gates,
        "overlap_rows": overlap,
        "best": best,
        "candidate_rows": sorted(candidate_rows, key=lambda item: item["weighted_signed_cos_mean"], reverse=True),
        "outputs": {"best_convention_normals": str(best_path.resolve())},
        "decision": "A best image-space Sapiens normal convention was selected for research comparisons only; it is not a world-normal proof or teacher promotion.",
        "blockers": blockers,
    }
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v15_sapiens_normal_convention_solver.json", summary)
    write_report(REPORTS / "20260508_v15_sapiens_normal_convention_solver.md", "V15 Sapiens Normal Convention Solver", summary)
    print(json.dumps(json_ready({"status": status, "metrics": summary["metrics"], "output": out}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
