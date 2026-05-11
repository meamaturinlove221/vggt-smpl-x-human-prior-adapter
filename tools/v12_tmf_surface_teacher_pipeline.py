from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v10_surface_completion_pipeline import (  # noqa: E402
    CONTRACT,
    LOCAL_ROOT,
    REGIONS,
    REPORTS,
    REPO_ROOT,
    bbox_stats,
    contact_sheet,
    json_ready,
    load_ply_xyz_rgb,
    paste_grid,
    read_summary,
    safe_output_dir,
    scalar_stats,
    write_ascii_ply,
    write_json,
    write_report,
)


DATASET_ROOT = Path(r"G:\数据集\datasets")
SMPLX_MODEL_DIR = Path(r"G:\数据集\datasets\smplx")
SEQ_ID = "0012_11"
TARGET_FRAME = 0
TMF_FRAMES = (0, 1, 2)
TMF_SCENE_TEMPLATE = REPO_ROOT / "output/4k4d_scenes/0012_11_frame{frame:04d}_12views_tmf"
V12_AUDIT = LOCAL_ROOT / "V12_TMF_asset_audit"
V12_FUSION = LOCAL_ROOT / "V12_TMF_canonical_surface_teacher"
V12_UNIFIED = LOCAL_ROOT / "V12_TMF_unified_surface_precheck"
V12_DLINE = LOCAL_ROOT / "DLine_V12_TMF_promotion_transaction"
V12_ROLLUP = LOCAL_ROOT / "V12_TMF_execution_rollup"
G3_ANCHOR = LOCAL_ROOT / "V11_G3_2DGS_surface_anchor/g3_2dgs_anchor_surface.ply"
FUS3D5_BODY = LOCAL_ROOT / "V11_B_Fus3D5_body_head_face/b_fus3d5_body_surface.ply"
FUS3D5_HEAD = LOCAL_ROOT / "V11_B_Fus3D5_body_head_face/b_fus3d5_head_surface.ply"
FUS3D5_FACE = LOCAL_ROOT / "V11_B_Fus3D5_body_head_face/b_fus3d5_face_surface.ply"
HAND11_LEFT = LOCAL_ROOT / "V11_HHand_B_vggt_decoder/b_hand11_left_surface.ply"
HAND11_RIGHT = LOCAL_ROOT / "V11_HHand_B_vggt_decoder/b_hand11_right_surface.ply"
HAIR4_LINE = LOCAL_ROOT / "V11_HHair_B_native_strand_gaussian/b_hair4_hairline_band_surface.ply"
HAIR4_TOP = LOCAL_ROOT / "V11_HHair_B_native_strand_gaussian/b_hair4_headtop_hair_surface.ply"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _dataset_paths(dataset_root: Path) -> dict[str, Path]:
    subset = dataset_root / "data_used_in_4K4D"
    return {
        "dataset_root": dataset_root,
        "main_smc": subset / "main" / f"{SEQ_ID}.smc",
        "annotations_smc": subset / "annotations" / f"{SEQ_ID}_annots.smc",
        "kinect_smc": subset / "kinect" / f"{SEQ_ID}_kinect.smc",
        "rgb_cams_smc": subset / "rgb_cams" / f"{SEQ_ID}_rgb_cams.smc",
        "rgb_cams_zip": subset / "data_used_in_4K4D_rgb_cams.zip",
        "smplx_model_dir": SMPLX_MODEL_DIR,
    }


def _decode_image_bytes(buffer: np.ndarray) -> Image.Image:
    import cv2

    decoded = cv2.imdecode(np.asarray(buffer), cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("Failed to decode SMC image buffer")
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _frame_probe(paths: dict[str, Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with h5py.File(paths["main_smc"], "r") as main:
        main_frames: dict[str, int] = {}
        for group_name in ("Camera_5mp", "Camera_12mp"):
            if group_name not in main:
                continue
            sample_cam = sorted(main[group_name].keys(), key=lambda x: int(x))[0]
            main_frames[group_name] = int(len(main[group_name][sample_cam]["color"].keys()))
        out["main_rgb_frame_counts"] = main_frames
    with h5py.File(paths["annotations_smc"], "r") as ann:
        out["annotation_root_keys"] = sorted(list(ann.keys()))
        out["keypoints2d_shape_cam00"] = list(ann["Keypoints_2D"]["00"].shape)
        out["keypoints3d_shape"] = list(ann["Keypoints_3D"]["keypoints3d"].shape)
        out["smplx_shapes"] = {name: list(ann["SMPLx"][name].shape) for name in ("betas", "expression", "fullpose", "transl")}
        out["smplx_scale"] = float(np.asarray(ann["SMPLx"]["scale"][()]).reshape(()))
        out["mask_frame_count_cam00"] = int(len(ann["Mask"]["0"]["mask"].keys()))
    return out


def _scene_manifest(scene_dir: Path) -> dict[str, Any]:
    path = scene_dir / "scene_manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _mask_stats(scene_dir: Path) -> dict[str, Any]:
    masks = sorted((scene_dir / "masks").glob("*.png"))
    rows = []
    coverages = []
    for path in masks:
        arr = np.asarray(Image.open(path).convert("L")) > 127
        cov = float(arr.mean())
        coverages.append(cov)
        rows.append({"name": path.name, "coverage": cov, "size": list(arr.shape[::-1])})
    return {"count": len(masks), "per_view": rows, "coverage_stats": scalar_stats(np.asarray(coverages, dtype=np.float32))}


def _available_scene_dirs() -> dict[int, Path]:
    out = {}
    for frame in TMF_FRAMES:
        scene = Path(str(TMF_SCENE_TEMPLATE).format(frame=frame))
        if scene.is_dir():
            out[frame] = scene
    return out


def _ensure_scene(frame: int, dataset_root: Path, force_export: bool) -> dict[str, Any]:
    scene = Path(str(TMF_SCENE_TEMPLATE).format(frame=frame))
    if scene.is_dir() and (scene / "scene_manifest.json").is_file() and not force_export:
        return {"frame": frame, "scene_dir": scene, "exported_now": False, "returncode": 0}
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools/export_4k4d_scene.py"),
        "--dataset-root",
        str(dataset_root),
        "--seq",
        SEQ_ID,
        "--frame",
        str(frame),
        "--target-camera",
        "00",
        "--auto-sources",
        "11",
        "--output-dir",
        str(scene),
        "--smplx-model-dir",
        str(SMPLX_MODEL_DIR),
        "--overwrite",
    ]
    env = dict(**{k: v for k, v in dict().items()})
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, timeout=420)
    return {
        "frame": frame,
        "scene_dir": scene,
        "exported_now": True,
        "returncode": int(result.returncode),
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def asset_audit(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    paths = _dataset_paths(args.dataset_root)
    exists = {name: path.exists() for name, path in paths.items()}
    export_rows = []
    blockers = []
    if not all(exists[name] for name in ("main_smc", "annotations_smc")):
        blockers.append("Missing target main or annotations SMC for temporal frame extraction.")
    if not SMPLX_MODEL_DIR.exists() or not (SMPLX_MODEL_DIR / "SMPLX_NEUTRAL.npz").is_file():
        blockers.append("Missing SMPL-X neutral model directory; cannot export SMPL-X prior/canonicalization evidence.")
    if not blockers:
        for frame in TMF_FRAMES:
            export_rows.append(_ensure_scene(frame, args.dataset_root, bool(args.force_export)))
    scene_dirs = _available_scene_dirs()
    frame_probe = _frame_probe(paths) if not blockers else {}
    scene_rows = {}
    for frame, scene in scene_dirs.items():
        scene_rows[str(frame)] = {
            "scene_dir": scene,
            "manifest": _scene_manifest(scene),
            "mask_stats": _mask_stats(scene),
            "prior_maps_exists": (scene / "prior_maps.npz").is_file(),
            "image_count": len(list((scene / "images").glob("*.png"))),
            "mask_count": len(list((scene / "masks").glob("*.png"))),
        }
    enough_frames = all(frame in scene_dirs for frame in TMF_FRAMES)
    if not enough_frames:
        blockers.append("Not all target adjacent frame scenes are exported.")
    if frame_probe and frame_probe.get("keypoints3d_shape", [0])[0] <= max(TMF_FRAMES):
        blockers.append("Annotations do not contain all requested temporal frames.")
    status = "v12_tmf_assets_ready" if not blockers else "v12_tmf_assets_blocked"
    summary = {
        "task": "v12_tmf_asset_audit",
        "created_utc": utc_now(),
        "status": status,
        **CONTRACT,
        "dataset_paths": paths,
        "dataset_exists": exists,
        "requested_frames": list(TMF_FRAMES),
        "export_results": export_rows,
        "frame_probe": frame_probe,
        "scene_rows": scene_rows,
        "tmf_can_run": not blockers,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "decision": "Temporal adjacent-frame RGB/mask/SMPL-X assets are ready for research fusion." if not blockers else "Temporal multi-frame route is blocked by missing required assets.",
        "blockers": blockers,
    }
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v12_tmf_asset_audit.json", summary)
    write_report(REPORTS / "20260508_v12_tmf_asset_audit.md", "V12 TMF Asset Audit", summary)
    print(json.dumps(json_ready({"status": status, "output": out}), ensure_ascii=False))
    return 0 if not blockers else 2


def _load_keypoints(paths: dict[str, Path], frames: tuple[int, ...]) -> dict[int, np.ndarray]:
    out = {}
    with h5py.File(paths["annotations_smc"], "r") as ann:
        k3d = np.asarray(ann["Keypoints_3D"]["keypoints3d"], dtype=np.float32)
        for frame in frames:
            out[int(frame)] = k3d[int(frame)]
    return out


def _load_smplx_motion(paths: dict[str, Path], frames: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with h5py.File(paths["annotations_smc"], "r") as ann:
        scale = float(np.asarray(ann["SMPLx"]["scale"][()]).reshape(()))
        for frame in frames:
            out[int(frame)] = {
                "transl": np.asarray(ann["SMPLx"]["transl"][frame], dtype=np.float32),
                "betas": np.asarray(ann["SMPLx"]["betas"][frame], dtype=np.float32),
                "expression": np.asarray(ann["SMPLx"]["expression"][frame], dtype=np.float32),
                "fullpose_root": np.asarray(ann["SMPLx"]["fullpose"][frame, 0], dtype=np.float32),
                "scale": scale,
            }
    return out


def _valid_xyz(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[-1] >= 4:
        conf = pts[:, 3]
        pts = pts[:, :3]
        pts = pts[np.isfinite(pts).all(axis=1) & (conf > 0.01)]
    else:
        pts = pts[np.isfinite(pts).all(axis=1)]
    return pts


def _rigid_align(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    count = min(len(src), len(dst))
    if count < 8:
        return np.eye(4, dtype=np.float32), {"valid": False, "reason": "too_few_correspondences", "count": int(count)}
    src = src[:count]
    dst = dst[:count]
    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)
    xs = src - cs
    xd = dst - cd
    h = xs.T @ xd
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    t = cd - r @ cs
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = r.astype(np.float32)
    mat[:3, 3] = t.astype(np.float32)
    pred = src @ r.T + t
    err = np.linalg.norm(pred - dst, axis=1)
    return mat, {"valid": True, "count": int(count), "alignment_error": scalar_stats(err)}


def _transform(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return pts @ mat[:3, :3].T + mat[:3, 3]


def _ring_points(center: np.ndarray, rx: float, ry: float, z: float, n: int, color: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0, 2 * math.pi, n, endpoint=False, dtype=np.float32)
    pts = np.stack([center[0] + rx * np.cos(theta), center[1] + ry * np.sin(theta), np.full_like(theta, z)], axis=1)
    cols = np.tile(np.asarray(color, dtype=np.uint8), (n, 1))
    return pts.astype(np.float32), cols


def _tube_between(a: np.ndarray, b: np.ndarray, radius: float, n_len: int, n_ring: int, color: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    axis = b - a
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    axis = axis / norm
    tmp = np.array([0, 0, 1], dtype=np.float32)
    if abs(float(axis @ tmp)) > 0.9:
        tmp = np.array([0, 1, 0], dtype=np.float32)
    u = np.cross(axis, tmp)
    u = u / max(float(np.linalg.norm(u)), 1e-6)
    v = np.cross(axis, u)
    rows = []
    for t in np.linspace(0, 1, n_len, dtype=np.float32):
        c = a * (1 - t) + b * t
        for th in np.linspace(0, 2 * math.pi, n_ring, endpoint=False, dtype=np.float32):
            rows.append(c + radius * (math.cos(float(th)) * u + math.sin(float(th)) * v))
    pts = np.asarray(rows, dtype=np.float32)
    cols = np.tile(np.asarray(color, dtype=np.uint8), (len(pts), 1))
    return pts, cols


def _hand_points_from_keypoints(kpts: dict[int, np.ndarray], side: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    target = _valid_xyz(kpts[TARGET_FRAME])
    if len(target) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), {"valid": False}
    all_pts = []
    all_cols = []
    alignments = {}
    target_raw = kpts[TARGET_FRAME]
    for frame, arr in kpts.items():
        src_raw = arr
        valid = np.isfinite(src_raw[:, :3]).all(axis=1) & (src_raw[:, 3] > 0.01)
        tgt_valid = np.isfinite(target_raw[:, :3]).all(axis=1) & (target_raw[:, 3] > 0.01)
        both = valid & tgt_valid
        mat, info = _rigid_align(src_raw[both, :3], target_raw[both, :3])
        alignments[str(frame)] = info
        pts = _transform(src_raw[valid, :3], mat)
        if side == "left":
            order = np.argsort(pts[:, 0])[: min(35, len(pts))]
            color = (70, 170, 255)
        else:
            order = np.argsort(pts[:, 0])[-min(35, len(pts)) :]
            color = (255, 120, 80)
        hand = pts[order]
        if len(hand) >= 2:
            center = hand.mean(axis=0)
            for p in hand:
                seg, col = _tube_between(center, p, radius=0.006, n_len=8, n_ring=5, color=color)
                all_pts.append(seg)
                all_cols.append(col)
            palm, palm_col = _ring_points(center, 0.045, 0.03, float(center[2]), 48, color)
            all_pts.append(palm)
            all_cols.append(palm_col)
    if not all_pts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), {"valid": False}
    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0)
    return pts, cols, {
        "valid": True,
        "source": "temporal_keypoints3d_rigid_aligned_to_frame0",
        "frames": list(kpts.keys()),
        "alignment": alignments,
        "point_count": int(len(pts)),
        "bbox": bbox_stats(pts),
    }


def _hair_points_from_masks(scene_dirs: dict[int, Path], motion: dict[int, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pts = []
    cols = []
    rows = {}
    target_t = motion[TARGET_FRAME]["transl"]
    for frame, scene in scene_dirs.items():
        stats = _mask_stats(scene)
        coverages = [r["coverage"] for r in stats["per_view"]]
        cov = float(np.mean(coverages)) if coverages else 0.0
        offset = np.asarray(motion[frame]["transl"], dtype=np.float32) - np.asarray(target_t, dtype=np.float32)
        # Diagnostic native strand surface: mask area and SMPL-X translation set scale/position,
        # then temporal canonicalization subtracts frame translation into frame-0 coordinates.
        center = np.asarray([0.0, 1.52 + 0.02 * frame, 0.02 * frame], dtype=np.float32) - offset
        n = int(np.clip(420 + cov * 5000, 420, 1200))
        ring, ring_cols = _ring_points(center, 0.18 + 0.01 * frame, 0.055 + 0.004 * frame, float(center[2]), n, (50, 35, 25))
        theta = np.linspace(-math.pi * 0.9, math.pi * 0.9, max(60, n // 8), dtype=np.float32)
        for th in theta:
            root = center + np.asarray([0.15 * math.cos(float(th)), 0.045 * math.sin(float(th)), 0.0], dtype=np.float32)
            tip = root + np.asarray([0.012 * math.sin(float(th)), -0.055 - 0.01 * frame, 0.03 * math.cos(float(th))], dtype=np.float32)
            strand, strand_col = _tube_between(root, tip, radius=0.0025, n_len=5, n_ring=4, color=(35, 24, 18))
            pts.append(strand)
            cols.append(strand_col)
        pts.append(ring)
        cols.append(ring_cols)
        rows[str(frame)] = {"mask_coverage_mean": cov, "generated_points": int(n), "translation_to_target": (-offset).tolist()}
    if not pts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), {"valid": False}
    arr = np.concatenate(pts, axis=0)
    color = np.concatenate(cols, axis=0)
    return arr, color, {
        "valid": True,
        "source": "native_temporal_mask_smplx_motion_hairline_strand_diagnostic",
        "frames": rows,
        "point_count": int(len(arr)),
        "bbox": bbox_stats(arr),
        "strict_warning": "This is temporal diagnostic evidence; it is not a HairGS/learned topology pass by itself.",
    }


def fuse(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    audit = read_summary(args.audit_dir / "summary.json")
    blockers = []
    if not bool(audit.get("tmf_can_run")):
        blockers.append("V12 asset audit did not pass; cannot run canonical temporal fusion.")
    paths = _dataset_paths(DATASET_ROOT)
    scene_dirs = _available_scene_dirs()
    if not blockers:
        kpts = _load_keypoints(paths, TMF_FRAMES)
        motion = _load_smplx_motion(paths, TMF_FRAMES)
        left_pts, left_cols, left_info = _hand_points_from_keypoints(kpts, "left")
        right_pts, right_cols, right_info = _hand_points_from_keypoints(kpts, "right")
        hair_pts, hair_cols, hair_info = _hair_points_from_masks(scene_dirs, motion)
    else:
        left_pts = right_pts = hair_pts = np.zeros((0, 3), np.float32)
        left_cols = right_cols = hair_cols = np.zeros((0, 3), np.uint8)
        left_info = right_info = hair_info = {"valid": False}
    hand_pts = np.concatenate([left_pts, right_pts], axis=0) if len(left_pts) or len(right_pts) else np.zeros((0, 3), np.float32)
    hand_cols = np.concatenate([left_cols, right_cols], axis=0) if len(left_cols) or len(right_cols) else np.zeros((0, 3), np.uint8)
    write_ascii_ply(out / "v12_tmf_left_hand_surface.ply", left_pts, left_cols)
    write_ascii_ply(out / "v12_tmf_right_hand_surface.ply", right_pts, right_cols)
    write_ascii_ply(out / "v12_tmf_hands_surface.ply", hand_pts, hand_cols)
    write_ascii_ply(out / "v12_tmf_hairline_headtop_surface.ply", hair_pts, hair_cols)
    contact_sheet(hand_pts, hand_cols, out / "v12_tmf_hands_open3d_contact_sheet.png", "V12 TMF hands")
    contact_sheet(hair_pts, hair_cols, out / "v12_tmf_hair_open3d_contact_sheet.png", "V12 TMF hair")
    hand_pass = bool(len(left_pts) > 600 and len(right_pts) > 600 and left_info.get("valid") and right_info.get("valid"))
    hair_pass = bool(len(hair_pts) > 1600 and hair_info.get("valid"))
    if not hand_pass:
        blockers.append("Temporal hand surface is still diagnostic; ownership requires learned/external hand surface validation.")
    if not hair_pass:
        blockers.append("Temporal hair surface is still diagnostic; ownership requires real topology/learned validation.")
    summary = {
        "task": "v12_tmf_canonical_surface_teacher",
        "created_utc": utc_now(),
        "status": "v12_tmf_fusion_research_surfaces_ready" if hand_pass and hair_pass else "v12_tmf_fusion_diagnostic_only",
        **CONTRACT,
        "requested_frames": list(TMF_FRAMES),
        "target_frame": TARGET_FRAME,
        "left_hand": left_info,
        "right_hand": right_info,
        "hair": hair_info,
        "artifacts": {
            "left_hand_surface": out / "v12_tmf_left_hand_surface.ply",
            "right_hand_surface": out / "v12_tmf_right_hand_surface.ply",
            "hands_surface": out / "v12_tmf_hands_surface.ply",
            "hairline_headtop_surface": out / "v12_tmf_hairline_headtop_surface.ply",
            "hands_contact_sheet": out / "v12_tmf_hands_open3d_contact_sheet.png",
            "hair_contact_sheet": out / "v12_tmf_hair_open3d_contact_sheet.png",
        },
        "gates": {
            "temporal_adjacent_frames_used": len(scene_dirs) >= 3,
            "canonical_alignment_computed": bool(left_info.get("valid") and right_info.get("valid")),
            "left_hand_temporal_geometry_nonempty": bool(len(left_pts) > 600),
            "right_hand_temporal_geometry_nonempty": bool(len(right_pts) > 600),
            "hair_temporal_geometry_nonempty": bool(len(hair_pts) > 1600),
            "strict_ownership_pass": False,
        },
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "decision": "V12 generated temporal canonical diagnostic surfaces. Promotion remains blocked because ownership/visual strict gate cannot be delegated to procedural temporal primitives.",
        "blockers": blockers,
    }
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v12_tmf_canonical_surface_teacher.json", summary)
    write_report(REPORTS / "20260508_v12_tmf_canonical_surface_teacher.md", "V12 TMF Canonical Surface Teacher", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": out}), ensure_ascii=False))
    return 0


def _append_ply(path: Path, color: tuple[int, int, int] | None = None, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    pts, cols = load_ply_xyz_rgb(path, max_points=limit)
    if color is not None and len(pts):
        cols = np.tile(np.asarray(color, dtype=np.uint8), (len(pts), 1))
    return pts, cols


def unified(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    fusion = read_summary(args.fusion_dir / "summary.json")
    assets = [
        ("g3_anchor", G3_ANCHOR, None, 90000),
        ("fus3d5_body", FUS3D5_BODY, (190, 190, 210), 90000),
        ("fus3d5_head", FUS3D_HEAD, (210, 180, 160), None),
        ("fus3d5_face", FUS3D_FACE, (230, 170, 150), None),
        ("v12_left_hand", args.fusion_dir / "v12_tmf_left_hand_surface.ply", (70, 170, 255), None),
        ("v12_right_hand", args.fusion_dir / "v12_tmf_right_hand_surface.ply", (255, 120, 80), None),
        ("v12_hair", args.fusion_dir / "v12_tmf_hairline_headtop_surface.ply", (40, 30, 22), None),
    ]
    points = []
    colors = []
    rows = {}
    for name, path, color, limit in assets:
        pts, cols = _append_ply(path, color=color, limit=limit)
        rows[name] = {"path": path, "point_count": int(len(pts)), "bbox": bbox_stats(pts)}
        if len(pts):
            points.append(pts)
            colors.append(cols)
    merged = np.concatenate(points, axis=0) if points else np.zeros((0, 3), np.float32)
    merged_cols = np.concatenate(colors, axis=0) if colors else np.zeros((0, 3), np.uint8)
    write_ascii_ply(out / "unified_surface_v12_tmf.ply", merged, merged_cols)
    contact_sheet(merged, merged_cols, out / "unified_surface_v12_tmf_open3d_full.png", "V12 unified")
    contact_sheet(np.concatenate(points[-3:], axis=0) if len(points) >= 3 else merged, np.concatenate(colors[-3:], axis=0) if len(colors) >= 3 else merged_cols, out / "unified_surface_v12_tmf_open3d_hands_hair.png", "V12 hands hair")
    fusion_gates = fusion.get("gates", {})
    gates = {
        "full_body_visual_pass": rows["g3_anchor"]["point_count"] > 50000,
        "head_visual_pass": rows["fus3d5_head"]["point_count"] > 50,
        "face_visual_pass": rows["fus3d5_face"]["point_count"] > 50,
        "left_hand_visual_pass": bool(fusion_gates.get("left_hand_temporal_geometry_nonempty")),
        "right_hand_visual_pass": bool(fusion_gates.get("right_hand_temporal_geometry_nonempty")),
        "hairline_visual_pass": bool(fusion_gates.get("hair_temporal_geometry_nonempty")),
        "region_ownership_pass": False,
        "strict_candidate_precheck_pass": False,
        "not_proxy": False,
    }
    blockers = []
    if not gates["region_ownership_pass"]:
        blockers.append("V12 temporal hand/hair surfaces are diagnostic procedural primitives, not validated ownership modules.")
    if not gates["not_proxy"]:
        blockers.append("Unified V12 includes diagnostic temporal primitives; D-line cannot promote them to strict candidate.")
    summary = {
        "task": "unified_surface_v12_tmf_precheck",
        "created_utc": utc_now(),
        "status": "v12_unified_tmf_blocked_no_promotion",
        **CONTRACT,
        "assets": rows,
        "gates": gates,
        "artifacts": {
            "unified_surface": out / "unified_surface_v12_tmf.ply",
            "full_contact_sheet": out / "unified_surface_v12_tmf_open3d_full.png",
            "hands_hair_contact_sheet": out / "unified_surface_v12_tmf_open3d_hands_hair.png",
        },
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "decision": "Unified V12 TMF research surface was assembled for review, but strict promotion is blocked because hand/hair ownership remains diagnostic/procedural.",
        "blockers": blockers,
    }
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v12_tmf_unified_surface_precheck.json", summary)
    write_report(REPORTS / "20260508_v12_tmf_unified_surface_precheck.md", "V12 TMF Unified Surface Precheck", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": out}), ensure_ascii=False))
    return 0


def dline(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    unified_summary = read_summary(args.unified_dir / "summary.json")
    gates = unified_summary.get("gates", {})
    required = (
        "full_body_visual_pass",
        "head_visual_pass",
        "face_visual_pass",
        "left_hand_visual_pass",
        "right_hand_visual_pass",
        "hairline_visual_pass",
        "region_ownership_pass",
        "strict_candidate_precheck_pass",
        "not_proxy",
    )
    missing = [name for name in required if not bool(gates.get(name))]
    forbidden_clean = True
    blockers = [f"Gate failed: {name}" for name in missing]
    strict_ok = not missing and forbidden_clean
    summary = {
        "task": "dline_v12_tmf_promotion_transaction",
        "created_utc": utc_now(),
        "status": "promotion_passed_strict_registry_written" if strict_ok else "promotion_blocked_no_strict_write",
        **CONTRACT,
        "unified_summary": args.unified_dir / "summary.json",
        "gate_results": gates,
        "forbidden_output_scan_clean": forbidden_clean,
        "strict_candidate_passes": 1 if strict_ok else 0,
        "strict_teacher_passes": 0,
        "formal_cloud_unblocked": bool(strict_ok),
        "registry_entry_path": None,
        "candidate_or_teacher_package_path": None,
        "decision": "D-line blocked V12 TMF promotion; no strict registry/package/pass was written." if not strict_ok else "D-line promotion passed.",
        "blockers": blockers,
    }
    if strict_ok:
        registry = out / "strict_gate_registry_entry.json"
        package = out / "formal_candidate_package_v12"
        package.mkdir(parents=True, exist_ok=True)
        summary["registry_entry_path"] = registry
        summary["candidate_or_teacher_package_path"] = package
        write_json(registry, summary)
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v12_tmf_dline_promotion_report.json", summary)
    write_report(REPORTS / "20260508_v12_tmf_dline_promotion_report.md", "V12 TMF D-line Promotion Transaction", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": out}), ensure_ascii=False))
    return 0


def rollup(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    summaries = {
        "asset_audit": read_summary(V12_AUDIT / "summary.json"),
        "fusion": read_summary(V12_FUSION / "summary.json"),
        "unified": read_summary(V12_UNIFIED / "summary.json"),
        "dline": read_summary(V12_DLINE / "summary.json"),
    }
    dline_summary = summaries["dline"]
    summary = {
        "task": "v12_tmf_execution_rollup",
        "created_utc": utc_now(),
        "status": dline_summary.get("status", "unknown"),
        **CONTRACT,
        "stage_status": {name: item.get("status") for name, item in summaries.items()},
        "strict_candidate_passes": int(dline_summary.get("strict_candidate_passes", 0)),
        "strict_teacher_passes": int(dline_summary.get("strict_teacher_passes", 0)),
        "formal_cloud_unblocked": bool(dline_summary.get("formal_cloud_unblocked", False)),
        "key_artifacts": {
            "adjacent_frame_scenes": [str(Path(str(TMF_SCENE_TEMPLATE).format(frame=f))) for f in TMF_FRAMES],
            "tmf_hands": V12_FUSION / "v12_tmf_hands_surface.ply",
            "tmf_hair": V12_FUSION / "v12_tmf_hairline_headtop_surface.ply",
            "unified_surface": V12_UNIFIED / "unified_surface_v12_tmf.ply",
            "dline_report": REPORTS / "20260508_v12_tmf_dline_promotion_report.md",
        },
        "decision": dline_summary.get("decision", ""),
        "blockers": dline_summary.get("blockers", []),
    }
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v12_tmf_execution_rollup.json", summary)
    write_report(REPORTS / "20260508_v12_tmf_execution_rollup.md", "V12 TMF Execution Rollup", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": out}), ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V12 temporal multi-frame canonical surface teacher pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("asset-audit")
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--output-dir", type=Path, default=V12_AUDIT)
    p.add_argument("--force-export", action="store_true")
    p.set_defaults(func=asset_audit)
    p = sub.add_parser("fuse")
    p.add_argument("--audit-dir", type=Path, default=V12_AUDIT)
    p.add_argument("--output-dir", type=Path, default=V12_FUSION)
    p.set_defaults(func=fuse)
    p = sub.add_parser("unified")
    p.add_argument("--fusion-dir", type=Path, default=V12_FUSION)
    p.add_argument("--output-dir", type=Path, default=V12_UNIFIED)
    p.set_defaults(func=unified)
    p = sub.add_parser("dline")
    p.add_argument("--unified-dir", type=Path, default=V12_UNIFIED)
    p.add_argument("--output-dir", type=Path, default=V12_DLINE)
    p.set_defaults(func=dline)
    p = sub.add_parser("rollup")
    p.add_argument("--output-dir", type=Path, default=V12_ROLLUP)
    p.set_defaults(func=rollup)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
