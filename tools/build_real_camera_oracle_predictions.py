from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_kinect_depth_teacher_targets import (  # noqa: E402
    build_camera_alignment_correspondences,
    camera_center_from_world_to_cam,
    load_json,
    load_rgb_camera_params,
    robust_transform,
)
from tools.dna_4k4d import normalize_camera_id  # noqa: E402
from tools.render_open3d_pointcloud import unproject_depth_map_to_point_map_numpy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local real-camera oracle diagnostic predictions.npz. This replaces "
            "VGGT-predicted camera intrinsics/extrinsics with crop-corrected 4K4D RGB cameras "
            "aligned into the VGGT gauge by camera geometry. It is an inference/eval diagnostic, "
            "not a HART/PnP route and not a teacher/candidate pass by itself."
        )
    )
    parser.add_argument("--predictions-npz", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument(
        "--alignment-source",
        choices=("camera", "camera_axes"),
        default="camera_axes",
        help="Use real RGB camera centers, optionally with axis endpoints, to align into the VGGT gauge.",
    )
    parser.add_argument("--camera-axis-scale", type=float, default=0.25)
    parser.add_argument("--transform-mode", choices=("similarity", "axis_affine"), default="similarity")
    parser.add_argument("--max-correspondences", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--world-mode",
        choices=("keep_original", "sync_to_oracle_depth"),
        default="keep_original",
        help=(
            "keep_original tests only whether real cameras rescue depth_unprojection. "
            "sync_to_oracle_depth is depth-authoritative and should be treated as a pseudo-positive-risk diagnostic."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(v) for v in value]
    return value


def orthonormalize(rotation: np.ndarray) -> np.ndarray:
    u_mat, _, vt_mat = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    out = u_mat @ vt_mat
    if np.linalg.det(out) < 0.0:
        u_mat[:, -1] *= -1.0
        out = u_mat @ vt_mat
    return out.astype(np.float64)


def real_camera_pose_in_target_camera(
    scene_manifest: dict[str, Any],
    rgb_params: dict[str, dict[str, np.ndarray]],
    view_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    target_camera_id = normalize_camera_id(scene_manifest["exported_views"][0]["camera_id"])
    target_w2c_real = np.asarray(rgb_params[target_camera_id]["world_to_cam"], dtype=np.float64)
    view = scene_manifest["exported_views"][view_index]
    camera_id = normalize_camera_id(view["camera_id"])
    real_c2w = np.asarray(rgb_params[camera_id]["cam_to_world"], dtype=np.float64)
    real_center_world = real_c2w[:3, 3]
    center_target = target_w2c_real[:3, :3] @ real_center_world + target_w2c_real[:3, 3]
    rotation_c2w_target = target_w2c_real[:3, :3] @ real_c2w[:3, :3]
    return center_target.astype(np.float64), orthonormalize(rotation_c2w_target)


def similarity_parts(matrix: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    linear = np.asarray(matrix[:3, :3], dtype=np.float64)
    scale = float(np.cbrt(abs(np.linalg.det(linear))))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError(f"Invalid similarity scale from transform matrix: {scale}")
    rotation = orthonormalize(linear / scale)
    translation = np.asarray(matrix[:3, 3], dtype=np.float64)
    return scale, rotation, translation


def axis_affine_parts(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    linear = np.asarray(matrix[:3, :3], dtype=np.float64)
    scale_vec = np.diag(linear).astype(np.float64)
    if np.any(~np.isfinite(scale_vec)) or np.any(np.abs(scale_vec) <= 1e-12):
        raise ValueError(f"Invalid axis-affine scale from transform matrix: {scale_vec}")
    translation = np.asarray(matrix[:3, 3], dtype=np.float64)
    return scale_vec, translation


def transform_center_and_rotation(
    center_target: np.ndarray,
    rotation_c2w_target: np.ndarray,
    transform_matrix: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if mode == "similarity":
        scale, rotation, translation = similarity_parts(transform_matrix)
        center_vggt = scale * (rotation @ center_target) + translation
        rotation_c2w_vggt = orthonormalize(rotation @ rotation_c2w_target)
        return center_vggt, rotation_c2w_vggt, {
            "scale": float(scale),
            "rotation": rotation,
            "translation": translation,
        }

    scale_vec, translation = axis_affine_parts(transform_matrix)
    center_vggt = scale_vec * center_target + translation
    # Axis-affine is not a rigid transform. For camera orientation we keep the
    # real target-camera axes, then re-orthonormalize so downstream SE3 inverse
    # code remains valid. This is diagnostic only.
    rotation_c2w_vggt = orthonormalize(rotation_c2w_target)
    return center_vggt, rotation_c2w_vggt, {
        "scale_xyz": scale_vec,
        "translation_xyz": translation,
        "orientation_note": "axis_affine scales centers only; rotations keep real target-camera orientation",
    }


def extrinsic_from_center_rotation(center_world: np.ndarray, rotation_c2w: np.ndarray) -> np.ndarray:
    rotation_c2w = orthonormalize(rotation_c2w)
    rotation_w2c = rotation_c2w.T
    translation_w2c = -rotation_w2c @ np.asarray(center_world, dtype=np.float64)
    out = np.zeros((3, 4), dtype=np.float32)
    out[:3, :3] = rotation_w2c.astype(np.float32)
    out[:3, 3] = translation_w2c.astype(np.float32)
    return out


def try_build_pose_encoding(extrinsic: np.ndarray, intrinsic: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray | None:
    try:
        import torch

        from vggt.utils.pose_enc import extri_intri_to_pose_encoding

        with torch.no_grad():
            ext = torch.from_numpy(extrinsic[None].astype(np.float32))
            intr = torch.from_numpy(intrinsic[None].astype(np.float32))
            pose = extri_intri_to_pose_encoding(ext, intr, image_size_hw=image_hw)
        return pose[0].cpu().numpy().astype(np.float32)
    except Exception:
        return None


def compare_camera_sets(original_extrinsic: np.ndarray, oracle_extrinsic: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(original_extrinsic.shape[0]):
        original_center, original_rot = camera_center_from_world_to_cam(original_extrinsic[idx])
        oracle_center, oracle_rot = camera_center_from_world_to_cam(oracle_extrinsic[idx])
        rot_delta = original_rot.T @ oracle_rot
        trace = float(np.clip((np.trace(rot_delta) - 1.0) * 0.5, -1.0, 1.0))
        angle = float(np.degrees(np.arccos(trace)))
        rows.append(
            {
                "view_index": int(idx),
                "center_delta": float(np.linalg.norm(original_center - oracle_center)),
                "rotation_delta_deg": angle,
                "original_center": original_center,
                "oracle_center": oracle_center,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    output_npz = args.output_npz.resolve()
    summary_json = args.summary_json.resolve()
    if output_npz.exists() and not args.overwrite:
        raise FileExistsError(f"{output_npz} exists. Use --overwrite to replace it.")
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    scene_dir = args.scene_dir.resolve()
    scene_manifest = load_json(scene_dir / "scene_manifest.json")
    predictions_path = args.predictions_npz.resolve()
    with np.load(predictions_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}

    original_extrinsic = np.asarray(arrays["extrinsic"], dtype=np.float32)
    original_intrinsic = np.asarray(arrays["intrinsic"], dtype=np.float32)
    view_count = int(original_extrinsic.shape[0])
    if len(scene_manifest["exported_views"]) != view_count:
        raise ValueError(
            f"Scene view count {len(scene_manifest['exported_views'])} does not match predictions {view_count}"
        )
    height = int(np.asarray(arrays["depth"]).shape[1])
    width = int(np.asarray(arrays["depth"]).shape[2])

    rgb_params = load_rgb_camera_params(scene_manifest)
    source_corr, target_corr, alignment_summary = build_camera_alignment_correspondences(
        scene_manifest,
        rgb_params,
        original_extrinsic,
        include_axes=args.alignment_source == "camera_axes",
        axis_scale=float(args.camera_axis_scale),
    )
    transform_summary, transform_matrix = robust_transform(
        source_corr,
        target_corr,
        mode=str(args.transform_mode),
        max_correspondences=int(args.max_correspondences),
        seed=int(args.seed),
    )

    oracle_extrinsic = np.zeros_like(original_extrinsic, dtype=np.float32)
    oracle_intrinsic = np.zeros_like(original_intrinsic, dtype=np.float32)
    per_view: list[dict[str, Any]] = []
    transform_detail_for_summary: dict[str, Any] | None = None
    for view_idx, view in enumerate(scene_manifest["exported_views"]):
        camera_id = normalize_camera_id(view["camera_id"])
        center_target, rotation_c2w_target = real_camera_pose_in_target_camera(scene_manifest, rgb_params, view_idx)
        center_vggt, rotation_c2w_vggt, transform_detail = transform_center_and_rotation(
            center_target,
            rotation_c2w_target,
            transform_matrix,
            str(args.transform_mode),
        )
        if transform_detail_for_summary is None:
            transform_detail_for_summary = transform_detail
        oracle_extrinsic[view_idx] = extrinsic_from_center_rotation(center_vggt, rotation_c2w_vggt)
        oracle_intrinsic[view_idx] = np.asarray(rgb_params[camera_id]["aligned_intrinsic"], dtype=np.float32)
        per_view.append(
            {
                "view_index": int(view_idx),
                "camera_id": camera_id,
                "center_target_camera_coord": center_target,
                "center_oracle_vggt_gauge": center_vggt,
                "original_intrinsic": original_intrinsic[view_idx],
                "oracle_aligned_intrinsic": oracle_intrinsic[view_idx],
            }
        )

    output_arrays = dict(arrays)
    output_arrays["extrinsic"] = oracle_extrinsic.astype(np.float32)
    output_arrays["intrinsic"] = oracle_intrinsic.astype(np.float32)
    pose_encoding = try_build_pose_encoding(oracle_extrinsic, oracle_intrinsic, (height, width))
    if pose_encoding is not None and "pose_enc" in output_arrays:
        output_arrays["pose_enc"] = pose_encoding

    if args.world_mode == "sync_to_oracle_depth":
        output_arrays["world_points"] = unproject_depth_map_to_point_map_numpy(
            np.asarray(output_arrays["depth"], dtype=np.float32),
            oracle_extrinsic,
            oracle_intrinsic,
        ).astype(np.float32)
        output_arrays["world_points_conf"] = np.asarray(output_arrays["depth_conf"], dtype=np.float32)

    np.savez_compressed(output_npz, **output_arrays)

    summary = {
        "task": "real_camera_oracle_predictions",
        "truthful_status": "diagnostic_only_not_hart_not_pnp_not_teacher_not_mentor_pass",
        "predictions_npz": str(predictions_path),
        "scene_dir": str(scene_dir),
        "output_npz": str(output_npz),
        "world_mode": str(args.world_mode),
        "alignment_source": str(args.alignment_source),
        "transform_mode": str(args.transform_mode),
        "alignment_summary": alignment_summary,
        "transform_summary": transform_summary,
        "transform_detail": transform_detail_for_summary,
        "camera_delta_vs_vggt_head": compare_camera_sets(original_extrinsic, oracle_extrinsic),
        "per_view": per_view,
        "risk_notes": [
            "This does not remove the need for cameras; it uses calibrated 4K4D RGB cameras.",
            "This is not HART-style PnP and does not replace the VGGT camera head in the method.",
            "Predicted depth scale remains VGGT-gauge; a real-camera oracle cannot prove geometry success unless strict Open3D gates pass.",
            "keep_original world_mode intentionally leaves point branch unchanged so depth-only camera gains cannot be misreported as point-branch success.",
        ],
    }
    summary_json.write_text(json.dumps(as_jsonable(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(as_jsonable(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
