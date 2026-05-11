from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial import cKDTree

from v10_surface_completion_pipeline import LOCAL_ROOT, REPORTS, bbox_stats, contact_sheet, json_ready, load_ply_xyz_rgb, scalar_stats, write_ascii_ply, write_json, write_report


DATASET_ROOT = Path(r"G:\数据集\datasets\data_used_in_4K4D")
ANN = DATASET_ROOT / "annotations/0012_11_annots.smc"
KINECT_REAL = LOCAL_ROOT / "V12_Kinect_TSDF_frame0000/kinect_tsdf_real_world.ply"
KINECT_VGGT = LOCAL_ROOT / "V12_Kinect_TSDF_VGGT_world_frame0000/kinect_tsdf_vggt_world.ply"
G3 = LOCAL_ROOT / "V11_G3_2DGS_surface_anchor/g3_2dgs_anchor_surface.ply"
OUT = LOCAL_ROOT / "V13_K2_kinect_alignment_candidate_search"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_k3d(frame: int = 0) -> np.ndarray:
    with h5py.File(ANN, "r") as ann:
        arr = np.asarray(ann["Keypoints_3D"]["keypoints3d"][frame], dtype=np.float32)
    valid = np.isfinite(arr[:, :3]).all(axis=1) & (arr[:, 3] > 0.01)
    return arr[valid, :3]


def _similarity(src: np.ndarray, dst: np.ndarray, allow_scale: bool = True) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = min(len(src), len(dst))
    src = src[:n]
    dst = dst[:n]
    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)
    xs = src - cs
    xd = dst - cd
    cov = xs.T @ xd / max(n, 1)
    u, s, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    scale = float(np.sum(s) / max(np.sum(xs * xs) / max(n, 1), 1e-12)) if allow_scale else 1.0
    t = cd - scale * (r @ cs)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = scale * r
    mat[:3, 3] = t
    return mat


def _apply(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    return points @ mat[:3, :3].T + mat[:3, 3]


def _quantile_cloud(points: np.ndarray) -> np.ndarray:
    return np.asarray([np.percentile(points, q, axis=0) for q in (2, 5, 25, 50, 75, 95, 98)], dtype=np.float64)


def _score(points: np.ndarray, target: np.ndarray, sample: int = 90000) -> dict[str, Any]:
    if len(points) == 0 or len(target) == 0:
        return {"valid": False, "score": 1e9}
    rng = np.random.default_rng(1320)
    src = points[rng.choice(len(points), min(sample, len(points)), replace=False)] if len(points) > sample else points
    dst = target[rng.choice(len(target), min(sample, len(target)), replace=False)] if len(target) > sample else target
    tree = cKDTree(dst)
    d, _ = tree.query(src, k=1, workers=-1)
    stats = scalar_stats(d)
    return {"valid": True, "score": float(stats.get("median", 1e9) or 1e9), "nn_distance": stats}


def _trimmed_icp(src_points: np.ndarray, dst_points: np.ndarray, init: np.ndarray, iterations: int = 8, trim: float = 0.65) -> tuple[np.ndarray, list[dict[str, Any]]]:
    mat = init.copy()
    rng = np.random.default_rng(1321)
    src = src_points[rng.choice(len(src_points), min(80000, len(src_points)), replace=False)] if len(src_points) > 80000 else src_points
    dst = dst_points[rng.choice(len(dst_points), min(100000, len(dst_points)), replace=False)] if len(dst_points) > 100000 else dst_points
    log = []
    tree = cKDTree(dst)
    for idx in range(iterations):
        cur = _apply(src, mat)
        dist, nn = tree.query(cur, k=1, workers=-1)
        keep_n = max(16, int(len(dist) * trim))
        keep = np.argsort(dist)[:keep_n]
        delta = _similarity(cur[keep], dst[nn[keep]], allow_scale=False)
        mat = delta @ mat
        log.append({"iteration": idx, "kept": int(keep_n), "distance": scalar_stats(dist[keep])})
    return mat, log


def main() -> int:
    parser = argparse.ArgumentParser(description="V13 K2 Kinect alignment candidate search.")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    real, colors = load_ply_xyz_rgb(KINECT_REAL, max_points=240000)
    v12_vggt, v12_cols = load_ply_xyz_rgb(KINECT_VGGT, max_points=240000)
    g3, _ = load_ply_xyz_rgb(G3, max_points=200000)
    k3d = _load_k3d(0)
    candidates: dict[str, dict[str, Any]] = {}
    mats: dict[str, np.ndarray] = {}
    mats["identity_real_world"] = np.eye(4)
    # KINECT_VGGT is already transformed by V12 camera-axis method, represented as direct points.
    candidates["v12_camera_axis_vggt"] = {"transformed_points": v12_vggt, "score_to_g3": _score(v12_vggt, g3)}
    mats["bbox_similarity_real_to_g3"] = _similarity(_quantile_cloud(real), _quantile_cloud(g3), allow_scale=True)
    mats["bbox_rigid_real_to_g3"] = _similarity(_quantile_cloud(real), _quantile_cloud(g3), allow_scale=False)
    if len(k3d) >= 16:
        mats["bbox_similarity_real_to_keypoints3d"] = _similarity(_quantile_cloud(real), _quantile_cloud(k3d), allow_scale=True)
    base = mats["bbox_similarity_real_to_g3"]
    icp_mat, icp_log = _trimmed_icp(real, g3, base, iterations=8, trim=0.55)
    mats["trimmed_icp_after_bbox_similarity"] = icp_mat
    for name, mat in mats.items():
        transformed = _apply(real, mat)
        score_g3 = _score(transformed, g3)
        score_k3d = _score(transformed, k3d, sample=20000)
        ply = out / f"k2_{name}.ply"
        write_ascii_ply(ply, transformed, colors)
        candidates[name] = {
            "matrix": mat,
            "surface": ply,
            "score_to_g3": score_g3,
            "score_to_keypoints3d": score_k3d,
            "bbox": bbox_stats(transformed),
        }
        if name == "trimmed_icp_after_bbox_similarity":
            candidates[name]["icp_log"] = icp_log
    best_name = min(candidates, key=lambda n: float(candidates[n].get("score_to_g3", {}).get("score", 1e9)))
    best_path = out / "k2_best_kinect_tsdf_vggt_world.ply"
    best_pts = candidates[best_name].get("transformed_points")
    if best_pts is None:
        best_pts, _ = load_ply_xyz_rgb(Path(candidates[best_name]["surface"]), max_points=None)
    write_ascii_ply(best_path, best_pts, colors[: len(best_pts)] if len(colors) >= len(best_pts) else None)
    contact_sheet(best_pts, colors[: len(best_pts)] if len(colors) >= len(best_pts) else None, out / "k2_best_open3d_contact_sheet.png", f"K2 best {best_name}")
    median = float(candidates[best_name].get("score_to_g3", {}).get("nn_distance", {}).get("median", 1e9) or 1e9)
    strict_ready = bool(median < 0.05)
    summary = {
        "task": "v13_k2_kinect_alignment_candidate_search",
        "created_utc": utc_now(),
        "status": "k2_alignment_candidate_search_complete",
        "best_candidate": best_name,
        "best_surface": best_path,
        "best_score_to_g3_median": median,
        "strict_alignment_candidate_ready": strict_ready,
        "candidates": candidates,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "decision": "K2 found an alignment candidate under the residual threshold." if strict_ready else "K2 did not reduce Kinect/G3 residual enough for strict teacher precheck.",
        "blockers": [] if strict_ready else [f"Best median NN residual {median:.6f} >= 0.05 threshold."],
    }
    write_json(out / "k2_alignment_candidates.json", summary)
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v13_k2_kinect_alignment_candidates.json", summary)
    write_report(REPORTS / "20260508_v13_k2_kinect_alignment_candidates.md", "V13 K2 Kinect Alignment Candidates", summary)
    print(json.dumps(json_ready({"status": summary["status"], "best": best_name, "median": median, "output": out}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
