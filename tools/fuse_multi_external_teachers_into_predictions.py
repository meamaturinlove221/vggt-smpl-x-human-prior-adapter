from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def fuse_multi(
    base_predictions: Path,
    teacher_targets: list[Path],
    output_dir: Path,
    *,
    alpha_xyz: tuple[float, float, float],
    max_distance: float,
    min_votes: int,
    reducer: str,
) -> dict:
    base = _load_npz(base_predictions)
    base_points = np.asarray(base["world_points"], dtype=np.float32)
    base_conf = np.asarray(base["world_points_conf"], dtype=np.float32)

    deltas = []
    valid_masks = []
    teacher_summaries = []
    for target_path in teacher_targets:
        teacher = _load_npz(target_path)
        teacher_points = np.asarray(teacher["world_points"], dtype=np.float32)
        teacher_mask = np.asarray(teacher["teacher_mask"], dtype=bool)
        if teacher_points.shape != base_points.shape:
            raise ValueError(f"shape mismatch for {target_path}: {teacher_points.shape} vs {base_points.shape}")
        delta = teacher_points - base_points
        distance = np.linalg.norm(delta, axis=-1)
        valid = (
            teacher_mask
            & np.isfinite(base_points).all(axis=-1)
            & np.isfinite(teacher_points).all(axis=-1)
            & np.isfinite(distance)
        )
        if max_distance > 0:
            valid &= distance <= float(max_distance)
        deltas.append(delta)
        valid_masks.append(valid)
        valid_distances = distance[teacher_mask & np.isfinite(distance)]
        teacher_summaries.append(
            {
                "teacher_targets": str(target_path),
                "teacher_mask_pixels": int(teacher_mask.sum()),
                "valid_pixels": int(valid.sum()),
                "distance_percentiles": [
                    float(value) for value in np.percentile(valid_distances, [0, 25, 50, 75, 90, 95, 99])
                ]
                if valid_distances.size
                else [],
            }
        )

    stacked_delta = np.stack(deltas, axis=0)
    stacked_valid = np.stack(valid_masks, axis=0)
    votes = stacked_valid.sum(axis=0)
    use = votes >= int(min_votes)
    masked_delta = np.where(stacked_valid[..., None], stacked_delta, np.nan)
    if reducer == "median":
        fused_delta = np.nanmedian(masked_delta, axis=0)
    elif reducer == "mean":
        fused_delta = np.nanmean(masked_delta, axis=0)
    else:
        raise ValueError(f"Unsupported reducer: {reducer}")
    fused_delta = np.nan_to_num(fused_delta, nan=0.0, posinf=0.0, neginf=0.0)

    alpha = np.asarray(alpha_xyz, dtype=np.float32).reshape(1, 1, 1, 3)
    fused_points = base_points.copy()
    fused_points[use] = base_points[use] + fused_delta[use] * alpha.reshape(3)

    out = dict(base)
    out["world_points"] = fused_points.astype(base["world_points"].dtype, copy=False)
    out["world_points_conf"] = base_conf.astype(base["world_points_conf"].dtype, copy=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.npz"
    np.savez_compressed(output_path, **out)
    summary = {
        "base_predictions": str(base_predictions),
        "teacher_targets": [str(path) for path in teacher_targets],
        "output_predictions": str(output_path),
        "alpha_xyz": [float(value) for value in alpha_xyz],
        "max_distance": float(max_distance),
        "min_votes": int(min_votes),
        "reducer": reducer,
        "fused_pixels": int(use.sum()),
        "vote_histogram": {str(i): int((votes == i).sum()) for i in range(len(teacher_targets) + 1)},
        "teachers": teacher_summaries,
    }
    (output_dir / "fusion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--teacher-targets", required=True, help="Comma-separated targets.npz paths")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alpha-x", type=float, default=0.5)
    parser.add_argument("--alpha-y", type=float, default=0.0)
    parser.add_argument("--alpha-z", type=float, default=0.5)
    parser.add_argument("--max-distance", type=float, default=0.04)
    parser.add_argument("--min-votes", type=int, default=1)
    parser.add_argument("--reducer", choices=["median", "mean"], default="median")
    args = parser.parse_args()
    teacher_targets = [Path(item.strip()) for item in args.teacher_targets.split(",") if item.strip()]
    if not teacher_targets:
        raise ValueError("at least one teacher target is required")
    summary = fuse_multi(
        base_predictions=args.base_predictions,
        teacher_targets=teacher_targets,
        output_dir=args.output_dir,
        alpha_xyz=(args.alpha_x, args.alpha_y, args.alpha_z),
        max_distance=args.max_distance,
        min_votes=args.min_votes,
        reducer=args.reducer,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
