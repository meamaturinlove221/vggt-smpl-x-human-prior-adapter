from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse external teacher target points into a VGGT predictions.npz for local "
            "diagnostics. It updates both world_points and depth so world/depth "
            "Open3D gates can test whether the teacher creates real geometry."
        )
    )
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--teacher-targets", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-distance", type=float, default=0.18)
    parser.add_argument("--confidence-boost", type=float, default=120.0)
    parser.add_argument("--depth-confidence-boost", type=float, default=None)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def world_to_camera(points_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:, :3, :3].astype(np.float32)
    translation = extrinsic[:, :3, 3].astype(np.float32)
    return np.einsum("vij,vhwj->vhwi", rotation, points_world.astype(np.float32)) + translation[:, None, None, :]


def main() -> int:
    args = parse_args()
    base = load_npz(args.base_predictions)
    teacher = load_npz(args.teacher_targets)
    if "world_points" not in base or "depth" not in base:
        raise KeyError("base predictions must contain world_points and depth")
    if "world_points" not in teacher or "teacher_mask" not in teacher:
        raise KeyError("teacher targets must contain world_points and teacher_mask")

    base_world = np.asarray(base["world_points"], dtype=np.float32)
    teacher_world = np.asarray(teacher["world_points"], dtype=np.float32)
    teacher_mask = np.asarray(teacher["teacher_mask"], dtype=bool)
    if teacher_world.shape != base_world.shape:
        raise ValueError(f"world_points shape mismatch: {teacher_world.shape} vs {base_world.shape}")
    if teacher_mask.shape != base_world.shape[:3]:
        raise ValueError(f"teacher_mask shape mismatch: {teacher_mask.shape} vs {base_world.shape[:3]}")

    delta = teacher_world - base_world
    distance = np.linalg.norm(delta, axis=-1)
    valid = (
        teacher_mask
        & np.isfinite(base_world).all(axis=-1)
        & np.isfinite(teacher_world).all(axis=-1)
        & np.isfinite(distance)
    )
    if float(args.max_distance) > 0.0:
        valid &= distance <= float(args.max_distance)

    alpha = float(args.alpha)
    fused_world = base_world.copy()
    fused_world[valid] = base_world[valid] + alpha * delta[valid]

    fused_cam = world_to_camera(fused_world, np.asarray(base["extrinsic"], dtype=np.float32))
    depth = np.asarray(base["depth"], dtype=np.float32).copy()
    depth_valid = valid & np.isfinite(fused_cam[..., 2]) & (fused_cam[..., 2] > 1e-6)
    depth[depth_valid, 0] = fused_cam[..., 2][depth_valid]

    world_conf = np.asarray(base["world_points_conf"], dtype=np.float32).copy()
    depth_conf = np.asarray(base["depth_conf"], dtype=np.float32).copy()
    world_conf[valid] = np.maximum(world_conf[valid], float(args.confidence_boost))
    depth_boost = float(args.depth_confidence_boost) if args.depth_confidence_boost is not None else float(args.confidence_boost)
    depth_conf[depth_valid] = np.maximum(depth_conf[depth_valid], depth_boost)

    out = dict(base)
    out["world_points"] = fused_world.astype(base["world_points"].dtype, copy=False)
    out["depth"] = depth.astype(base["depth"].dtype, copy=False)
    out["world_points_conf"] = world_conf.astype(base["world_points_conf"].dtype, copy=False)
    out["depth_conf"] = depth_conf.astype(base["depth_conf"].dtype, copy=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "predictions.npz"
    np.savez_compressed(output_path, **out)

    teacher_distances = distance[teacher_mask & np.isfinite(distance)]
    valid_distances = distance[valid & np.isfinite(distance)]
    summary = {
        "base_predictions": str(args.base_predictions.resolve()),
        "teacher_targets": str(args.teacher_targets.resolve()),
        "output_predictions": str(output_path.resolve()),
        "alpha": alpha,
        "max_distance": float(args.max_distance),
        "confidence_boost": float(args.confidence_boost),
        "depth_confidence_boost": depth_boost,
        "teacher_mask_pixels": int(teacher_mask.sum()),
        "valid_fused_pixels": int(valid.sum()),
        "depth_fused_pixels": int(depth_valid.sum()),
        "distance_percentiles_on_teacher_mask": [
            float(value) for value in np.percentile(teacher_distances, [0, 25, 50, 75, 90, 95, 99])
        ]
        if teacher_distances.size
        else [],
        "distance_percentiles_on_fused_pixels": [
            float(value) for value in np.percentile(valid_distances, [0, 25, 50, 75, 90, 95, 99])
        ]
        if valid_distances.size
        else [],
        "truthful_status": "local_diagnostic_candidate_not_training_result",
    }
    (args.output_dir / "fusion_summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
