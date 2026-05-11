from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an existing 4K4D pseudo training case and replace only the "
            "prior supervision tensors from a teacher_targets.npz. The main "
            "depth/world_points targets are left untouched, so this can layer a "
            "coarse continuity prior on top of a local high-confidence teacher."
        )
    )
    parser.add_argument("--source-case-dir", required=True, type=Path)
    parser.add_argument("--teacher-targets", required=True, type=Path)
    parser.add_argument("--output-case-dir", required=True, type=Path)
    parser.add_argument("--max-distance", type=float, default=0.18)
    parser.add_argument("--mask-dilate", type=int, default=0)
    parser.add_argument("--prior-mask-source", choices=("teacher", "union"), default="teacher")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def copy_case(source: Path, output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    shutil.copytree(source, output)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    if int(iterations) <= 0:
        return out
    for _ in range(int(iterations)):
        padded = np.pad(out, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=False)
        grown = np.zeros_like(out, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                grown |= padded[:, dy : dy + out.shape[1], dx : dx + out.shape[2]]
        out = grown
    return out


def world_to_camera(points_world: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    rotation = extrinsics[:, :3, :3].astype(np.float32)
    translation = extrinsics[:, :3, 3].astype(np.float32)
    return np.einsum("vij,vhwj->vhwi", rotation, points_world.astype(np.float32)) + translation[:, None, None, :]


def normals_world_to_camera(normals_world: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    rotation = extrinsics[:, :3, :3].astype(np.float32)
    normals = np.einsum("vij,vhwj->vhwi", rotation, normals_world.astype(np.float32))
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    return (normals / np.clip(norm, 1e-6, None)).astype(np.float32)


def main() -> int:
    args = parse_args()
    source_case = args.source_case_dir.resolve()
    output_case = args.output_case_dir.resolve()
    teacher_path = args.teacher_targets.resolve()
    copy_case(source_case, output_case, overwrite=bool(args.overwrite))

    inputs_path = output_case / "inputs.npz"
    targets_path = output_case / "targets.npz"
    inputs = load_npz(inputs_path)
    targets = load_npz(targets_path)
    teacher = load_npz(teacher_path)

    if "world_points" not in teacher or "teacher_mask" not in teacher:
        raise KeyError("teacher_targets must contain world_points and teacher_mask")
    teacher_world = np.asarray(teacher["world_points"], dtype=np.float32)
    teacher_mask = np.asarray(teacher["teacher_mask"], dtype=bool)
    target_world = np.asarray(targets["world_points"], dtype=np.float32)
    if teacher_world.shape != target_world.shape:
        raise ValueError(f"teacher world shape {teacher_world.shape} != case world shape {target_world.shape}")
    if teacher_mask.shape != target_world.shape[:3]:
        raise ValueError(f"teacher mask shape {teacher_mask.shape} != case mask shape {target_world.shape[:3]}")

    distance = np.linalg.norm(teacher_world - target_world, axis=-1)
    valid = (
        teacher_mask
        & np.isfinite(teacher_world).all(axis=-1)
        & np.isfinite(target_world).all(axis=-1)
        & np.isfinite(distance)
    )
    if float(args.max_distance) > 0.0:
        valid &= distance <= float(args.max_distance)
    valid = dilate_mask(valid, int(args.mask_dilate))
    valid &= teacher_mask

    if str(args.prior_mask_source) == "union" and "prior_mask" in inputs:
        prior_mask = np.asarray(inputs["prior_mask"], dtype=bool) | valid
    else:
        prior_mask = valid

    extrinsics = np.asarray(targets["extrinsics"], dtype=np.float32)
    teacher_cam = world_to_camera(teacher_world, extrinsics)
    prior_depths = np.zeros(target_world.shape[:3], dtype=np.float32)
    depth_valid = prior_mask & np.isfinite(teacher_cam[..., 2]) & (teacher_cam[..., 2] > 1e-6)
    prior_depths[depth_valid] = teacher_cam[..., 2][depth_valid]

    prior_points = np.zeros_like(target_world, dtype=np.float32)
    prior_points[prior_mask] = teacher_world[prior_mask]

    prior_normals = np.zeros_like(target_world, dtype=np.float32)
    if "teacher_normals" in teacher:
        teacher_normals = np.asarray(teacher["teacher_normals"], dtype=np.float32)
        if teacher_normals.shape != target_world.shape:
            raise ValueError(f"teacher_normals shape {teacher_normals.shape} != {target_world.shape}")
        prior_normals[prior_mask] = normals_world_to_camera(teacher_normals, extrinsics)[prior_mask]

    inputs["prior_mask"] = prior_mask.astype(bool)
    targets["prior_depths"] = prior_depths.astype(np.float32)
    targets["prior_points"] = prior_points.astype(np.float32)
    targets["prior_normals"] = prior_normals.astype(np.float32)
    np.savez_compressed(inputs_path, **inputs)
    np.savez_compressed(targets_path, **targets)

    distances_teacher = distance[teacher_mask & np.isfinite(distance)]
    distances_valid = distance[prior_mask & np.isfinite(distance)]
    summary = {
        "task": "build_teacher_prior_training_case",
        "truthful_status": "training_case_prior_only_not_final_result",
        "source_case_dir": str(source_case),
        "teacher_targets": str(teacher_path),
        "output_case_dir": str(output_case),
        "max_distance": float(args.max_distance),
        "mask_dilate": int(args.mask_dilate),
        "prior_mask_source": str(args.prior_mask_source),
        "teacher_mask_pixels": int(teacher_mask.sum()),
        "prior_mask_pixels": int(prior_mask.sum()),
        "depth_valid_pixels": int(depth_valid.sum()),
        "distance_percentiles_teacher_mask": [
            float(v) for v in np.percentile(distances_teacher, [0, 25, 50, 75, 90, 95, 99])
        ]
        if distances_teacher.size
        else [],
        "distance_percentiles_prior_mask": [
            float(v) for v in np.percentile(distances_valid, [0, 25, 50, 75, 90, 95, 99])
        ]
        if distances_valid.size
        else [],
        "notes": [
            "Main depth/world_points supervision is preserved from the source case.",
            "prior_points/prior_depths are intended for loss.human_prior.point/depth.",
            "prior_normals are camera-space normals converted from the teacher normal field when available.",
        ],
    }
    (output_case / "teacher_prior_training_case_summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path = output_case / "case_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest["teacher_prior_training_case_patch"] = summary
    manifest_path.write_text(json.dumps(json_ready(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
