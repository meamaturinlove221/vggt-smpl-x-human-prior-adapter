from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from v10_surface_completion_pipeline import (
    LOCAL_ROOT,
    REPORTS,
    bbox_stats,
    contact_sheet,
    json_ready,
    load_ply_xyz_rgb,
    paste_grid,
    scalar_stats,
    write_ascii_ply,
    write_json,
    write_report,
)


DATASET_ROOT = Path(r"G:\数据集\datasets\data_used_in_4K4D")
ANN = DATASET_ROOT / "annotations/0012_11_annots.smc"
SCENE = Path("output/4k4d_scenes/0012_11_frame0000_12views_tmf")
KINECT_REAL = LOCAL_ROOT / "V12_Kinect_TSDF_frame0000/kinect_tsdf_real_world.ply"
KINECT_VGGT = LOCAL_ROOT / "V12_Kinect_TSDF_VGGT_world_frame0000/kinect_tsdf_vggt_world.ply"
G3 = LOCAL_ROOT / "V11_G3_2DGS_surface_anchor/g3_2dgs_anchor_surface.ply"
OUT = LOCAL_ROOT / "V13_K1_kinect_alignment_autopsy"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_k3d(frame: int) -> np.ndarray:
    with h5py.File(ANN, "r") as ann:
        arr = np.asarray(ann["Keypoints_3D"]["keypoints3d"][frame], dtype=np.float32)
    valid = np.isfinite(arr[:, :3]).all(axis=1) & (arr[:, 3] > 0.01)
    return arr[valid, :3]


def _similarity(src: np.ndarray, dst: np.ndarray, allow_scale: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = min(len(src), len(dst))
    if n < 8:
        return np.eye(4), {"valid": False, "reason": "too_few_points", "count": int(n)}
    src = src[:n]
    dst = dst[:n]
    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)
    xs = src - cs
    xd = dst - cd
    cov = xs.T @ xd / n
    u, s, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    scale = float(np.sum(s) / max(np.sum(xs * xs) / n, 1e-12)) if allow_scale else 1.0
    t = cd - scale * (r @ cs)
    m = np.eye(4)
    m[:3, :3] = scale * r
    m[:3, 3] = t
    pred = src @ m[:3, :3].T + t
    err = np.linalg.norm(pred - dst, axis=1)
    return m, {"valid": True, "count": int(n), "scale": scale, "residual": scalar_stats(err)}


def _nn_residual(src: np.ndarray, dst: np.ndarray, sample: int = 80000) -> dict[str, Any]:
    if len(src) == 0 or len(dst) == 0:
        return {"valid": False}
    if len(src) > sample:
        rng = np.random.default_rng(1301)
        src = src[rng.choice(len(src), sample, replace=False)]
    if len(dst) > sample:
        rng = np.random.default_rng(1302)
        dst = dst[rng.choice(len(dst), sample, replace=False)]
    tree = cKDTree(dst)
    dist, _ = tree.query(src, k=1, workers=-1)
    return {"valid": True, "src_count": int(len(src)), "dst_count": int(len(dst)), "nn_distance": scalar_stats(dist)}


def _apply(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    return points @ mat[:3, :3].T + mat[:3, 3]


def _mask_sheet(scene: Path, output: Path) -> None:
    ims = []
    for path in sorted((scene / "masks").glob("*.png"))[:12]:
        im = Image.open(path).convert("RGB").resize((180, 180))
        draw = ImageDraw.Draw(im)
        draw.text((5, 5), path.name[:18], fill=(255, 0, 0))
        ims.append(im)
    output.parent.mkdir(parents=True, exist_ok=True)
    paste_grid(ims, cols=4, bg=(20, 20, 20)).save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="V13 K1 Kinect/VGGT alignment autopsy.")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    real_pts, real_cols = load_ply_xyz_rgb(KINECT_REAL, max_points=220000)
    vggt_pts, vggt_cols = load_ply_xyz_rgb(KINECT_VGGT, max_points=220000)
    g3_pts, g3_cols = load_ply_xyz_rgb(G3, max_points=180000)
    k3d = _load_k3d(0)
    candidates: dict[str, Any] = {}
    candidates["v12_camera_axis_vggt_vs_g3"] = _nn_residual(vggt_pts, g3_pts)
    # bbox center/extent alignment gives a geometry-only baseline without fitting to masks.
    if len(real_pts) and len(g3_pts):
        rp = np.asarray([np.percentile(real_pts, q, axis=0) for q in (5, 50, 95)]).reshape(-1, 3)
        gp = np.asarray([np.percentile(g3_pts, q, axis=0) for q in (5, 50, 95)]).reshape(-1, 3)
        mat, info = _similarity(rp, gp, allow_scale=True)
        transformed = _apply(real_pts, mat)
        write_ascii_ply(out / "k1_bbox_similarity_kinect_to_g3.ply", transformed, real_cols)
        contact_sheet(transformed, real_cols, out / "k1_bbox_similarity_contact_sheet.png", "K1 bbox sim")
        candidates["bbox_similarity_real_to_g3"] = {**info, "nn_to_g3": _nn_residual(transformed, g3_pts)}
    if len(k3d) and len(real_pts):
        kp = np.asarray([np.percentile(k3d, q, axis=0) for q in (5, 50, 95)]).reshape(-1, 3)
        rp = np.asarray([np.percentile(real_pts, q, axis=0) for q in (5, 50, 95)]).reshape(-1, 3)
        mat, info = _similarity(rp, kp, allow_scale=True)
        transformed = _apply(real_pts, mat)
        write_ascii_ply(out / "k1_bbox_similarity_kinect_to_keypoints3d.ply", transformed, real_cols)
        candidates["bbox_similarity_real_to_keypoints3d"] = {**info, "nn_to_keypoints3d": _nn_residual(transformed, k3d, sample=20000)}
    _mask_sheet(SCENE, out / "k1_12view_mask_sheet.png")
    contact_sheet(real_pts, real_cols, out / "k1_kinect_real_world_contact_sheet.png", "K1 Kinect real")
    contact_sheet(vggt_pts, vggt_cols, out / "k1_kinect_vggt_world_contact_sheet.png", "K1 Kinect VGGT")
    contact_sheet(g3_pts, g3_cols, out / "k1_g3_anchor_contact_sheet.png", "K1 G3")
    summary = {
        "task": "v13_k1_kinect_alignment_autopsy",
        "created_utc": utc_now(),
        "status": "k1_alignment_autopsy_complete",
        "inputs": {"kinect_real": KINECT_REAL, "kinect_vggt": KINECT_VGGT, "g3_anchor": G3, "annotations": ANN, "scene": SCENE},
        "bbox": {"kinect_real": bbox_stats(real_pts), "kinect_vggt": bbox_stats(vggt_pts), "g3_anchor": bbox_stats(g3_pts), "keypoints3d": bbox_stats(k3d)},
        "candidate_residuals": candidates,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "decision": "K1 autopsy generated geometry baselines; K2 must search transforms and reprojection/depth residual before any teacher precheck.",
        "blockers": ["No K1 candidate writes strict teacher pass."],
    }
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v13_kinect_alignment_autopsy.json", summary)
    write_report(REPORTS / "20260508_v13_kinect_alignment_autopsy.md", "V13 K1 Kinect Alignment Autopsy", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": out}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
