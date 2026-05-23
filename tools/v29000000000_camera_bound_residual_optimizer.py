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
SMC = Path(r"G:\数据集\datasets\data_used_in_4K4D\annotations\0012_11_annots.smc")
V11700 = AUX / "remote_pull" / "V11700_gap_reduction_branch_520" / "predictions.npz"
TRUE_PRED = AUX / "output" / "V24000000000_dense_fullview_predictions" / "true_surface_transformer_seed0" / "predictions.npz"
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
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def cams() -> dict[str, dict[str, np.ndarray]]:
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


def objective(wp: np.ndarray, base: np.ndarray, valid: np.ndarray, c: dict[str, dict[str, np.ndarray]]) -> tuple[float, float, float]:
    shifts = []
    inframes = []
    for vi, cid in enumerate(CAMERA_IDS):
        xy, z = project(wp[vi], c[cid]["K"], c[cid]["RT"])
        xy0, _ = project(base[vi], c[cid]["K"], c[cid]["RT"])
        mask = valid[vi]
        in_frame = (xy[..., 0] >= 0) & (xy[..., 0] < 518) & (xy[..., 1] >= 0) & (xy[..., 1] < 518) & (z > 0)
        shifts.append(float(np.linalg.norm(xy - xy0, axis=-1)[mask].mean()))
        inframes.append(float(in_frame[mask].mean()))
    shift = float(np.mean(shifts))
    in_frame = float(np.mean(inframes))
    return in_frame - 0.0005 * shift, shift, in_frame


def run() -> dict[str, Any]:
    c = cams()
    with np.load(V11700, allow_pickle=False) as z:
        base = z["world_points"].astype(np.float32)
    with np.load(TRUE_PRED, allow_pickle=False) as z:
        true = z["world_points"].astype(np.float32)
        valid = z["valid_mask"].astype(bool)
        region = z["region_label"].astype(np.int16)
    residual = true - base
    scales = np.concatenate([np.linspace(0, 0.02, 11), np.linspace(0.025, 0.2, 8), np.linspace(0.25, 1.0, 4)])
    rows = []
    best = None
    for scale in scales:
        wp = base + residual * float(scale)
        reproj, shift, inframe = objective(wp, base, valid, c)
        full_delta = float(np.linalg.norm(wp - base, axis=-1)[valid].mean())
        hair = valid & (region == 2)
        hands = valid & ((region == 3) | (region == 4))
        hair_delta = float(np.linalg.norm(wp - base, axis=-1)[hair].mean()) if hair.any() else 0.0
        hand_delta = float(np.linalg.norm(wp - base, axis=-1)[hands].mean()) if hands.any() else 0.0
        # Hard trust region: reprojection shift above 2 px is treated as severe.
        trust_penalty = max(0.0, shift - 2.0) * 0.25
        score = reproj + 35.0 * min(full_delta, 0.001) + 20.0 * min(hair_delta + hand_delta, 0.001) - trust_penalty
        row = {
            "scale": float(scale),
            "objective": score,
            "reprojection_score": reproj,
            "mean_reprojection_shift": shift,
            "in_frame_ratio": inframe,
            "full_delta": full_delta,
            "hair_delta": hair_delta,
            "hand_delta": hand_delta,
            "trust_penalty": trust_penalty,
        }
        rows.append(row)
        if best is None or score > best["objective"]:
            best = row
    write_csv(REPORTS / "V29000000000_residual_optimizer.csv", rows)
    assert best is not None
    nonzero_feasible = best["scale"] > 0 and best["mean_reprojection_shift"] <= 2.0 and best["full_delta"] > 1e-6
    payload = {
        "created_utc": now(),
        "best": best,
        "nonzero_camera_bound_residual_feasible": bool(nonzero_feasible),
        "mentor_ready": False,
        "external_hard_block_requires_user_action": not nonzero_feasible,
        "user_action_checklist": [] if nonzero_feasible else [
            "Provide the exact camera calibration and coordinate convention used to produce V11700/V920 six-view tensors.",
            "Confirm whether 4K4D SMC file 0012_11_annots.smc is the matching subject/action for current V11700/V920 outputs.",
            "If current V11700/V920 came from a different SMC sequence, provide that sequence's *_annots.smc path.",
            "Provide or confirm the world-coordinate transform from VGGT world_points to 4K4D camera coordinates.",
            "Provide trusted reprojection masks or rendered silhouettes for camera ids 00,01,15,30,45,59 at 518x518.",
        ],
    }
    write_json(REPORTS / "V29000000000_decision.json", payload)
    return payload


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
