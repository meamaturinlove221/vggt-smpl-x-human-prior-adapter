from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "output" / "training_cases" / "0012_11_frame0000_6views_smplx_native_prior_v15"
PKG = ROOT / "output" / "frozen_candidates" / "V50R2_rebuilt_from_sessions_gdrive_modal" / "package_files"
V32 = ROOT / "output" / "surface_research_preflight_local" / "V32_candidate_inference_research"
OUT = ROOT / "output" / "mentor_report_v50r2" / "pointcloud_sources"
REPORTS = ROOT / "reports"
CANONICAL_V50R2_SOURCE_SCRIPT = ROOT / "tools" / "v223_v50r2_view_consistent_sources.py"


def _run_v50r2_view_consistent_replacement() -> int:
    import runpy

    runpy.run_path(str(CANONICAL_V50R2_SOURCE_SCRIPT), run_name="__main__")
    return 0


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def despill_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.int16).copy()
    r = arr[:, 0]
    g = arr[:, 1]
    b = arr[:, 2]
    spill = (g > 38) & (g > r + 5) & (g > b + 5) & (r < 190) & (b < 190)
    if spill.any():
        arr[spill, 1] = np.minimum(arr[spill, 1], np.maximum(r[spill], b[spill]) + 2)
    return np.clip(arr, 0, 255).astype(np.uint8)


def write_ply(path: Path, points: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    )
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write(header + "\n")
        for p, c in zip(points, rgb):
            f.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def depth_to_cam_points(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    z = depth.astype(np.float32)
    x = (xx - cx) * z / max(fx, 1e-8)
    y = (yy - cy) * z / max(fy, 1e-8)
    return np.stack([x, y, z], axis=-1)


def finite_point_mask(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=-1) & (np.linalg.norm(points, axis=-1) > 1e-8)


def export_masked_map(
    path: Path,
    points_map: np.ndarray,
    image: np.ndarray,
    mask: np.ndarray,
) -> int:
    valid = mask.astype(bool) & finite_point_mask(points_map)
    pts = points_map[valid].astype(np.float32)
    rgb = despill_rgb(image[valid])
    write_ply(path, pts, rgb)
    return int(len(pts))


def main() -> int:
    return _run_v50r2_view_consistent_replacement()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    send_dir = OUT / "send_to_mentor"
    send_dir.mkdir(parents=True, exist_ok=True)

    inp = load_npz(CASE / "inputs.npz")
    targets = load_npz(CASE / "targets.npz")
    cand = load_npz(PKG / "candidate_files__candidate_points.npz")
    head = load_npz(PKG / "candidate_files__head_face_patch.npz")
    hand = load_npz(PKG / "candidate_files__hand_patch.npz")
    depths = load_npz(V32 / "candidate_depths_research.npz")["candidate_depths"]

    images = inp["images"]
    masks = inp["point_masks"].astype(bool)
    intrinsics = targets["intrinsics"]
    cams = [str(x) for x in inp["camera_ids"]]
    candidate_world = cand["candidate_points_world"]
    head_face_world = head["refined_points_world"]
    head_face_mask = head["head_mask"].astype(bool) | head["face_mask"].astype(bool)
    hand_world = hand["hand_points_world"]
    hand_region = hand["hand_region_id_map"]

    paths: dict[str, str] = {}
    stats: dict[str, object] = {
        "created_utc": now(),
        "source_candidate_points": str((PKG / "candidate_files__candidate_points.npz").resolve()),
        "source_candidate_depths": str((V32 / "candidate_depths_research.npz").resolve()),
        "source_inputs": str((CASE / "inputs.npz").resolve()),
        "policy": (
            "Human-only PLY exports. Uses point_masks / ROI masks to remove background/full-frame point-map pixels. "
            "Per-view visible-surface PLYs are safe to open in MeshLab. A direct 6-view concatenation is not a valid "
            "global fused human point cloud and is explicitly marked do-not-send."
        ),
        "views": [],
        "recommended_send_to_mentor": [],
        "invalid_do_not_send": [],
    }

    for i, cam in enumerate(cams):
        mask = masks[i] & finite_point_mask(candidate_world[i])
        pts = candidate_world[i][mask].astype(np.float32)
        rgb = despill_rgb(images[i][mask])
        out = OUT / f"v50r2_human_only_candidate_world_cam{cam}.ply"
        write_ply(out, pts, rgb)
        paths[f"candidate_world_cam{cam}"] = str(out.resolve())

        depth_mask = masks[i] & np.isfinite(depths[i]) & (depths[i] > 1e-4)
        cam_points = depth_to_cam_points(depths[i], intrinsics[i])
        dpts = cam_points[depth_mask].astype(np.float32)
        drgb = despill_rgb(images[i][depth_mask])
        dout = OUT / f"v50r2_human_only_depth_unprojected_cam{cam}.ply"
        write_ply(dout, dpts, drgb)
        paths[f"depth_unprojected_cam{cam}"] = str(dout.resolve())

        stats["views"].append(
            {
                "camera": cam,
                "candidate_world_points": int(len(pts)),
                "depth_unprojected_points": int(len(dpts)),
                "candidate_world_ply": str(out.resolve()),
                "depth_unprojected_ply": str(dout.resolve()),
            }
        )

    merged_out = OUT / "v50r2_human_only_candidate_world_merged6v.ply"
    invalid_dir = OUT / "invalid_do_not_send"
    invalid_dir.mkdir(parents=True, exist_ok=True)
    invalid_marker = OUT / "v50r2_human_only_candidate_world_merged6v_INVALID_DO_NOT_SEND.txt"
    invalid_reason = (
        "This file is a stale/legacy 6-view concatenation of per-camera visible-surface point maps. "
        "It is not a calibrated fused point cloud. In MeshLab it can look like multiple bodies, flat sheets, "
        "or severe tearing. Do not send it to the mentor; use the per-view files under send_to_mentor/ instead."
    )
    invalid_marker.write_text(invalid_reason + "\n", encoding="utf-8")
    if merged_out.exists():
        invalid_merged = invalid_dir / merged_out.name
        if invalid_merged.exists():
            invalid_merged.unlink()
        merged_out.replace(invalid_merged)
        stats["invalid_do_not_send"].append({"path": str(invalid_merged.resolve()), "reason": invalid_reason})
    old_v32 = ROOT / "output" / "surface_research_preflight_local" / "V32_candidate_inference_research" / "v32_candidate_open3d_review_points.ply"
    if old_v32.exists():
        stats["invalid_do_not_send"].append(
            {
                "path": str(old_v32.resolve()),
                "reason": "Legacy full-frame/debug review PLY. It includes non-human/full-frame point-map pixels and is not a mentor-facing human point cloud.",
            }
        )

    cam_to_idx = {cam: i for i, cam in enumerate(cams)}

    def add_recommended(label: str, category: str, coordinate_space: str, path: Path, points: int, reason: str) -> None:
        item = {
            "label": label,
            "category": category,
            "coordinate_space": coordinate_space,
            "points": int(points),
            "path": str(path.resolve()),
            "reason": reason,
        }
        stats["recommended_send_to_mentor"].append(item)
        paths[label] = str(path.resolve())

    # MeshLab-safe primary sources: one camera/view per file, no multi-view concatenation.
    primary_exports = [
        ("full_body_cam30_depth", "30", "full_body", "camera_depth_unprojected", "front/three-quarter full body view with one visible surface"),
        ("full_body_cam15_depth", "15", "full_body", "camera_depth_unprojected", "side full body view with one visible surface"),
        ("full_body_cam59_depth", "59", "full_body", "camera_depth_unprojected", "back/side full body view with one visible surface"),
    ]
    for label, cam, category, coord, reason in primary_exports:
        i = cam_to_idx[cam]
        cam_points = depth_to_cam_points(depths[i], intrinsics[i])
        roi = masks[i] & np.isfinite(depths[i]) & (depths[i] > 1e-4)
        out = send_dir / f"v50r2_SEND_{label}.ply"
        n = export_masked_map(out, cam_points, images[i], roi)
        add_recommended(label, category, coord, out, n, reason)

    # Region-specific PLYs for mentor inspection. These are still per-view PLYs.
    roi_exports = [
        ("head_face_cam30_depth", "30", "head_face", "camera_depth_unprojected", head_face_mask, "head/face ROI from the readable front/three-quarter view"),
        ("head_face_cam30_refined_world", "30", "head_face", "refined_world_points", head_face_mask, "head/face refined candidate ROI, one view only"),
        ("left_hand_cam15_depth", "15", "left_hand", "camera_depth_unprojected", hand_region == 1, "left-hand visible view"),
        ("left_hand_cam15_refined_world", "15", "left_hand", "refined_world_points", hand_region == 1, "left-hand SMPL-X native local patch, one view only"),
        ("right_hand_cam30_depth", "30", "right_hand", "camera_depth_unprojected", hand_region == 2, "right-hand visible view"),
        ("right_hand_cam30_refined_world", "30", "right_hand", "refined_world_points", hand_region == 2, "right-hand SMPL-X native local patch, one view only"),
    ]
    for label, cam, category, coord, roi_masks, reason in roi_exports:
        i = cam_to_idx[cam]
        roi = masks[i] & roi_masks[i]
        if coord == "camera_depth_unprojected":
            cam_points = depth_to_cam_points(depths[i], intrinsics[i])
            roi = roi & np.isfinite(depths[i]) & (depths[i] > 1e-4)
            points_map = cam_points
        elif category == "head_face":
            points_map = head_face_world[i]
        else:
            points_map = hand_world[i]
        out = send_dir / f"v50r2_SEND_{label}.ply"
        n = export_masked_map(out, points_map, images[i], roi)
        add_recommended(label, category, coord, out, n, reason)

    stats["outputs"] = paths

    json_path = REPORTS / "20260509_v50r2_pointcloud_ply_validity_audit.json"
    md_path = REPORTS / "20260509_v50r2_pointcloud_ply_validity_audit.md"
    json_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# V50R2 Point Cloud PLY Validity Audit",
        "",
        "Conclusion: the multi-body / flat-sheet MeshLab view came from an invalid export pattern, not from a mentor-facing point cloud source. V50R2 stores per-view visible-surface point maps. Concatenating six views directly creates overlapping shells and apparent multiple bodies.",
        "",
        "Do not send these files:",
    ]
    for item in stats["invalid_do_not_send"]:
        lines.append(f"- `{item['path']}`")
        lines.append(f"  - reason: {item['reason']}")
    lines += [
        "",
        "Open these MeshLab-safe files first:",
    ]
    for item in stats["recommended_send_to_mentor"]:
        lines.append(f"- {item['category']} / {item['coordinate_space']} / {item['label']}: `{item['path']}`")
        lines.append(f"  - points: {item['points']}")
        lines.append(f"  - reason: {item['reason']}")
    lines += [
        "",
        "All per-view human-only exports:",
    ]
    for view in stats["views"]:
        lines.append(f"- cam{view['camera']} candidate world: `{view['candidate_world_ply']}`")
        lines.append(f"- cam{view['camera']} depth unprojected: `{view['depth_unprojected_ply']}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Keep the earlier report name as a compatibility pointer.
    (REPORTS / "20260509_v50r2_human_only_pointcloud_sources.md").write_text(
        "See `20260509_v50r2_pointcloud_ply_validity_audit.md`. The old merged6v export is invalid for mentor display.\n",
        encoding="utf-8",
    )
    (REPORTS / "20260509_v50r2_human_only_pointcloud_sources.json").write_text(
        json.dumps({"superseded_by": str(json_path.resolve())}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
