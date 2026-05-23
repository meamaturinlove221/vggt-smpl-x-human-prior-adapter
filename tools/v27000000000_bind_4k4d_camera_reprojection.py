from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
OUTPUT = AUX / "output"
SMC = Path(r"G:\数据集\datasets\data_used_in_4K4D\annotations\0012_11_annots.smc")
PRED_ROOT = OUTPUT / "V25000000000_visual_first_predictions"
V11700 = AUX / "remote_pull" / "V11700_gap_reduction_branch_520" / "predictions.npz"
CAMERA_IDS = ["00", "01", "15", "30", "45", "59"]
GROUPS = [
    "true_surface_transformer",
    "random_semantic",
    "strong_shuffled_surface_semantic",
    "local_knn_smoothing_surface",
    "no_sparseconv_mlp",
    "no_surface_graph",
    "random_surface_graph",
    "observation_only",
    "support_only",
    "no_teacher",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_cameras() -> dict[str, dict[str, np.ndarray]]:
    cams: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(str(SMC), "r") as f:
        for cid in CAMERA_IDS:
            grp = f[f"Camera_Parameter/{cid}"]
            cams[cid] = {"K": grp["K"][()], "RT": grp["RT"][()], "D": grp["D"][()]}
    return cams


def project(points: np.ndarray, K: np.ndarray, RT: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = points.reshape(-1, 3).astype(np.float64)
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (RT @ homo.T).T[:, :3]
    z = cam[:, 2]
    pix_h = (K @ cam.T).T
    xy = pix_h[:, :2] / np.clip(pix_h[:, 2:3], 1e-9, None)
    return xy.reshape(points.shape[:2] + (2,)), z.reshape(points.shape[:2])


def run() -> dict[str, Any]:
    cams = load_cameras()
    with np.load(V11700, allow_pickle=False) as z:
        baseline_wp = z["world_points"].astype(np.float32)
    rows = []
    group_scores: dict[str, float] = {}
    for group in GROUPS:
        with np.load(PRED_ROOT / f"{group}_seed0" / "predictions.npz", allow_pickle=False) as z:
            wp = z["world_points"].astype(np.float32)
            valid = z["valid_mask"].astype(bool)
            region = z["region_label"].astype(np.int16)
        view_scores = []
        for vi, cid in enumerate(CAMERA_IDS):
            K = cams[cid]["K"]
            RT = cams[cid]["RT"]
            xy, zc = project(wp[vi], K, RT)
            xy0, z0 = project(baseline_wp[vi], K, RT)
            in_frame = (xy[..., 0] >= 0) & (xy[..., 0] < 518) & (xy[..., 1] >= 0) & (xy[..., 1] < 518) & (zc > 0)
            in_frame0 = (xy0[..., 0] >= 0) & (xy0[..., 0] < 518) & (xy0[..., 1] >= 0) & (xy0[..., 1] < 518) & (z0 > 0)
            mask = valid[vi]
            if mask.any():
                reproj_shift = np.linalg.norm(xy - xy0, axis=-1)
                score = float(in_frame[mask].mean() - 0.0005 * reproj_shift[mask].mean())
                view_scores.append(score)
                rows.append({
                    "group": group,
                    "view_index": vi,
                    "camera_id": cid,
                    "valid_pixels": int(mask.sum()),
                    "in_frame_ratio": float(in_frame[mask].mean()),
                    "baseline_in_frame_ratio": float(in_frame0[mask].mean()),
                    "mean_reprojection_shift_vs_v11700": float(reproj_shift[mask].mean()),
                    "reprojection_score": score,
                    "full_body": True,
                    "head_face_pixels": int((mask & (region[vi] == 1)).sum()),
                    "hairline_pixels": int((mask & (region[vi] == 2)).sum()),
                    "left_hand_pixels": int((mask & (region[vi] == 3)).sum()),
                    "right_hand_pixels": int((mask & (region[vi] == 4)).sum()),
                })
        group_scores[group] = float(np.mean(view_scores)) if view_scores else -1e9
    write_csv(REPORTS / "V27000000000_reprojection_metrics.csv", rows)
    ranking = sorted([{"group": g, "score": s} for g, s in group_scores.items()], key=lambda x: x["score"], reverse=True)
    true_score = group_scores["true_surface_transformer"]
    payload = {
        "created_utc": now(),
        "smc_path": str(SMC),
        "camera_ids": CAMERA_IDS,
        "camera_binding_available": True,
        "ranking": ranking,
        "true_reprojection_beats_all_controls": all(true_score > s for g, s in group_scores.items() if g != "true_surface_transformer"),
        "reprojection_consistency_available": True,
        "note": "Projection uses 4K4D SMC K/RT for camera ids matching V6850/V11700 outputs. Distortion D is loaded but not applied in this first binding.",
    }
    write_json(REPORTS / "V27000000000_camera_binding_report.json", payload)
    return payload


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
