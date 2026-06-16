from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch a VGGT prediction bundle with a cautious SMPL-X face/head scaffold. "
            "This is a local diagnostic candidate, not a training result."
        )
    )
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--targets-npz", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--face-alpha", type=float, default=0.85)
    parser.add_argument("--hairline-alpha", type=float, default=0.45)
    parser.add_argument("--ear-alpha", type=float, default=0.65)
    parser.add_argument("--head-alpha", type=float, default=0.0)
    parser.add_argument("--world-conf", type=float, default=80.0)
    parser.add_argument("--depth-conf", type=float, default=110.0)
    parser.add_argument("--normal-conf", type=float, default=1.0)
    parser.add_argument(
        "--normal-sign",
        choices=("predicted", "raw"),
        default="predicted",
        help="Use 'predicted' to flip raw SMPL-X normals into VGGT predicted-normal convention.",
    )
    return parser.parse_args()


def invert_se3(extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    rotation_t = np.transpose(rotation, (0, 2, 1))
    out = np.tile(np.eye(4, dtype=np.float32), (extrinsic.shape[0], 1, 1))
    out[:, :3, :3] = rotation_t
    out[:, :3, 3] = -np.einsum("bij,bj->bi", rotation_t, translation)
    return out


def camera_to_world(points_cam: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    cam_to_world = invert_se3(extrinsic.astype(np.float32))
    rotation = cam_to_world[:, :3, :3]
    translation = cam_to_world[:, :3, 3]
    return np.einsum("vij,vhwj->vhwi", rotation, points_cam.astype(np.float32)) + translation[:, None, None, :]


def normalized(vec: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.maximum(norm, eps)


def load_targets_masks(path: Path, shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    targets = np.load(path, allow_pickle=False)
    masks: dict[str, np.ndarray] = {}
    for key in ("head_roi_mask", "face_roi_mask", "hairline_mask", "ear_band_mask", "teacher_mask"):
        if key in targets:
            arr = np.asarray(targets[key]).astype(bool)
            if arr.shape != shape:
                raise ValueError(f"{key} has shape {arr.shape}, expected {shape}")
            masks[key] = arr
    return masks


def blend_field(base: np.ndarray, target: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_expanded = alpha[..., None].astype(np.float32)
    return base * (1.0 - alpha_expanded) + target * alpha_expanded


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_predictions)
    scene_dir = Path(args.scene_dir)
    targets_path = Path(args.targets_npz)
    output_npz = Path(args.output_npz)
    output_summary = Path(args.output_summary) if args.output_summary else output_npz.with_suffix(".json")

    base = np.load(base_path, allow_pickle=False)
    prior_bundle = np.load(scene_dir / "prior_maps.npz", allow_pickle=True)
    prior_maps = np.asarray(prior_bundle["prior_maps"]).astype(np.float32)
    prior_mask = np.asarray(prior_bundle["prior_mask"]).astype(bool)
    channels = [str(x) for x in prior_bundle["prior_channels"]]

    channel_index = {name: idx for idx, name in enumerate(channels)}
    required = [
        "smplx_posed_cam_x",
        "smplx_posed_cam_y",
        "smplx_posed_cam_z",
        "smplx_cam_nx",
        "smplx_cam_ny",
        "smplx_cam_nz",
        "smplx_visible_mask",
    ]
    missing = [name for name in required if name not in channel_index]
    if missing:
        raise KeyError(f"Missing prior channels: {missing}")

    world_points = np.asarray(base["world_points"]).astype(np.float32).copy()
    depth = np.asarray(base["depth"]).astype(np.float32).copy()
    normal = np.asarray(base["normal"]).astype(np.float32).copy()
    world_conf = np.asarray(base["world_points_conf"]).astype(np.float32).copy()
    depth_conf = np.asarray(base["depth_conf"]).astype(np.float32).copy()
    normal_conf = np.asarray(base["normal_conf"]).astype(np.float32).copy()
    extrinsic = np.asarray(base["extrinsic"]).astype(np.float32)

    view_count, height, width = world_conf.shape
    masks = load_targets_masks(targets_path, (view_count, height, width))

    smplx_cam = np.stack(
        [
            prior_maps[:, channel_index["smplx_posed_cam_x"]],
            prior_maps[:, channel_index["smplx_posed_cam_y"]],
            prior_maps[:, channel_index["smplx_posed_cam_z"]],
        ],
        axis=-1,
    )
    smplx_normal = np.stack(
        [
            prior_maps[:, channel_index["smplx_cam_nx"]],
            prior_maps[:, channel_index["smplx_cam_ny"]],
            prior_maps[:, channel_index["smplx_cam_nz"]],
        ],
        axis=-1,
    )
    if str(args.normal_sign) == "predicted":
        smplx_normal = -smplx_normal
    smplx_normal = normalized(smplx_normal)

    smplx_visible = prior_mask & (prior_maps[:, channel_index["smplx_visible_mask"]] > 0.5)
    finite = np.isfinite(smplx_cam).all(axis=-1) & (smplx_cam[..., 2] > 0.0)
    support = smplx_visible & finite

    alpha = np.zeros((view_count, height, width), dtype=np.float32)
    if float(args.head_alpha) > 0.0 and "head_roi_mask" in masks:
        alpha = np.maximum(alpha, masks["head_roi_mask"].astype(np.float32) * float(args.head_alpha))
    if "face_roi_mask" in masks:
        alpha = np.maximum(alpha, masks["face_roi_mask"].astype(np.float32) * float(args.face_alpha))
    if "hairline_mask" in masks:
        alpha = np.maximum(alpha, masks["hairline_mask"].astype(np.float32) * float(args.hairline_alpha))
    if "ear_band_mask" in masks:
        alpha = np.maximum(alpha, masks["ear_band_mask"].astype(np.float32) * float(args.ear_alpha))
    alpha *= support.astype(np.float32)
    patch_mask = alpha > 0.0

    smplx_world = camera_to_world(smplx_cam, extrinsic)
    world_points = blend_field(world_points, smplx_world, alpha)
    depth[..., 0] = depth[..., 0] * (1.0 - alpha) + smplx_cam[..., 2] * alpha
    normal = normalized(blend_field(normal, smplx_normal, np.clip(alpha, 0.0, 1.0)))

    world_conf[patch_mask] = np.maximum(world_conf[patch_mask], float(args.world_conf))
    depth_conf[patch_mask] = np.maximum(depth_conf[patch_mask], float(args.depth_conf))
    normal_conf[patch_mask] = np.maximum(normal_conf[patch_mask], float(args.normal_conf))

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {key: np.asarray(base[key]) for key in base.files}
    out["world_points"] = world_points.astype(np.float32)
    out["depth"] = depth.astype(np.float32)
    out["normal"] = normal.astype(np.float32)
    out["world_points_conf"] = world_conf.astype(np.float32)
    out["depth_conf"] = depth_conf.astype(np.float32)
    out["normal_conf"] = normal_conf.astype(np.float32)
    np.savez_compressed(output_npz, **out)

    summary = {
        "base_predictions": str(base_path),
        "scene_dir": str(scene_dir),
        "targets_npz": str(targets_path),
        "output_npz": str(output_npz),
        "patch_pixels": int(patch_mask.sum()),
        "support_pixels": int(support.sum()),
        "alpha_nonzero_min": float(alpha[patch_mask].min()) if np.any(patch_mask) else 0.0,
        "alpha_nonzero_max": float(alpha[patch_mask].max()) if np.any(patch_mask) else 0.0,
        "face_alpha": float(args.face_alpha),
        "hairline_alpha": float(args.hairline_alpha),
        "ear_alpha": float(args.ear_alpha),
        "head_alpha": float(args.head_alpha),
        "world_conf": float(args.world_conf),
        "depth_conf": float(args.depth_conf),
        "normal_conf": float(args.normal_conf),
        "normal_sign": str(args.normal_sign),
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
