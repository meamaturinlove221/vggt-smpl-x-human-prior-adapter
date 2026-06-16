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
V11700 = AUX / "remote_pull" / "V11700_gap_reduction_branch_520" / "predictions.npz"
TRUE_PRED = OUTPUT / "V24000000000_dense_fullview_predictions" / "true_surface_transformer_seed0" / "predictions.npz"
OUT = OUTPUT / "V28000000000_reprojection_scale_repair"
CAMERA_IDS = ["00", "01", "15", "30", "45", "59"]


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


def load_cams() -> dict[str, dict[str, np.ndarray]]:
    out = {}
    with h5py.File(str(SMC), "r") as f:
        for cid in CAMERA_IDS:
            g = f[f"Camera_Parameter/{cid}"]
            out[cid] = {"K": g["K"][()], "RT": g["RT"][()]}
    return out


def project(points: np.ndarray, K: np.ndarray, RT: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = points.reshape(-1, 3).astype(np.float64)
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    cam = (RT @ homo.T).T[:, :3]
    pix = (K @ cam.T).T
    xy = pix[:, :2] / np.clip(pix[:, 2:3], 1e-9, None)
    return xy.reshape(points.shape[:2] + (2,)), cam[:, 2].reshape(points.shape[:2])


def reproj_score(wp: np.ndarray, base: np.ndarray, valid: np.ndarray, cams: dict[str, dict[str, np.ndarray]]) -> tuple[float, float, float]:
    scores = []
    shifts = []
    inframes = []
    for vi, cid in enumerate(CAMERA_IDS):
        xy, z = project(wp[vi], cams[cid]["K"], cams[cid]["RT"])
        xy0, _ = project(base[vi], cams[cid]["K"], cams[cid]["RT"])
        in_frame = (xy[..., 0] >= 0) & (xy[..., 0] < 518) & (xy[..., 1] >= 0) & (xy[..., 1] < 518) & (z > 0)
        mask = valid[vi]
        shift = np.linalg.norm(xy - xy0, axis=-1)
        scores.append(float(in_frame[mask].mean() - 0.0005 * shift[mask].mean()))
        shifts.append(float(shift[mask].mean()))
        inframes.append(float(in_frame[mask].mean()))
    return float(np.mean(scores)), float(np.mean(shifts)), float(np.mean(inframes))


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    cams = load_cams()
    with np.load(V11700, allow_pickle=False) as z:
        base_wp = z["world_points"].astype(np.float32)
        depth = z["depth"].astype(np.float32)
        conf = z["world_points_conf"].astype(np.float32)
        base_normal = z["normal"].astype(np.float32)
        normal_conf = z["normal_conf"].astype(np.float32)
    with np.load(TRUE_PRED, allow_pickle=False) as z:
        true_wp = z["world_points"].astype(np.float32)
        learned_normal = z["normal"].astype(np.float32)
        valid = z["valid_mask"].astype(bool)
        region = z["region_label"].astype(np.int16)
    residual = true_wp - base_wp
    normal_residual = learned_normal - base_normal
    rows = []
    scales = [0.0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.2, 0.3, 0.5, 0.75, 1.0]
    best = None
    for scale in scales:
        wp = base_wp + residual * scale
        n = base_normal + normal_residual * scale
        n = n / np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6, None)
        rscore, shift, in_frame = reproj_score(wp, base_wp, valid, cams)
        region_delta = {}
        for name, rid in {"full_body": None, "head_face": 1, "hairline": 2, "left_hand": 3, "right_hand": 4}.items():
            mask = valid if rid is None else (valid & (region == rid))
            region_delta[name] = float(np.linalg.norm(wp - base_wp, axis=-1)[mask].mean()) if mask.any() else 0.0
        combined = rscore + 40.0 * min(region_delta["full_body"], 0.0015)
        row = {
            "scale": scale,
            "reprojection_score": rscore,
            "mean_reprojection_shift": shift,
            "in_frame_ratio": in_frame,
            "combined_score": combined,
            **{f"delta_{k}": v for k, v in region_delta.items()},
        }
        rows.append(row)
        if best is None or combined > best["combined_score"]:
            best = row
            best_wp = wp
            best_n = n
    write_csv(REPORTS / "V28000000000_scale_search.csv", rows)
    assert best is not None
    out_dir = OUT / "true_surface_transformer_seed0"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "predictions.npz",
        world_points=best_wp.astype(np.float32),
        depth=depth,
        confidence=conf,
        normal=best_n.astype(np.float32),
        normal_conf=normal_conf,
        learned_normal=best_n.astype(np.float32),
        geometric_normal=base_normal,
        normal_residual=(best_n - base_normal).astype(np.float32),
        valid_mask=valid,
        region_label=region,
    )
    decision = {
        "created_utc": now(),
        "best": best,
        "reprojection_aware_scale_found": best["scale"] > 0,
        "visible_region_delta_nonzero": best["delta_full_body"] > 0,
        "mentor_ready": False,
        "remaining_limitations": [
            "Scale repair improves camera-space trust region but is still not a trained differentiable renderer.",
            "Controls need equivalent scale-search rerun before any mentor-ready claim.",
        ],
        "next_route_required": True,
        "next_route": "V290 differentiable renderer training with camera-bound loss",
    }
    write_json(REPORTS / "V28000000000_decision.json", decision)
    return decision


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
