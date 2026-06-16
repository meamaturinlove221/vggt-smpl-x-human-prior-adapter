from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from render_open3d_pointcloud import unproject_depth_map_to_point_map_numpy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a diagnostic predictions.npz with confidence reweighted by "
            "cross-view nearest-neighbor support. Coordinates are not changed."
        )
    )
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--support-source", choices=("world_points", "depth_unprojection"), default="depth_unprojection")
    parser.add_argument("--radius", type=float, default=0.035)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--base-conf-weight", type=float, default=0.25)
    parser.add_argument("--min-other-views", type=int, default=1)
    parser.add_argument("--write-depth-conf", action="store_true")
    return parser.parse_args()


def resolve_support_points(data: np.lib.npyio.NpzFile, source: str) -> np.ndarray:
    if source == "world_points":
        return np.asarray(data["world_points"], dtype=np.float32)
    if source == "depth_unprojection":
        return unproject_depth_map_to_point_map_numpy(data["depth"], data["extrinsic"], data["intrinsic"])
    raise ValueError(source)


def support_score_per_view(points: np.ndarray, radius: float, stride: int, min_other_views: int) -> tuple[np.ndarray, dict]:
    view_count, height, width, _ = points.shape
    yy, xx = np.mgrid[0:height:stride, 0:width:stride]
    flat_y = yy.reshape(-1)
    flat_x = xx.reshape(-1)
    support_small = np.zeros((view_count, len(flat_y)), dtype=np.float32)
    valid_small = np.zeros((view_count, len(flat_y)), dtype=bool)

    sampled_points = []
    trees = []
    for view_idx in range(view_count):
        p = np.asarray(points[view_idx, flat_y, flat_x], dtype=np.float32)
        valid = np.isfinite(p).all(axis=-1)
        sampled_points.append(p)
        valid_small[view_idx] = valid
        trees.append(cKDTree(p[valid]) if np.any(valid) else None)

    for view_idx in range(view_count):
        p = sampled_points[view_idx]
        valid = valid_small[view_idx]
        if not np.any(valid):
            continue
        counts = np.zeros(len(p), dtype=np.float32)
        for other_idx, tree in enumerate(trees):
            if other_idx == view_idx or tree is None:
                continue
            hits = np.zeros(len(p), dtype=bool)
            query_points = p[valid]
            neighbors = tree.query_ball_point(query_points, r=float(radius), workers=-1)
            hits_valid = np.asarray([len(item) > 0 for item in neighbors], dtype=bool)
            hits[np.flatnonzero(valid)] = hits_valid
            counts += hits.astype(np.float32)
        support_small[view_idx] = counts

    denom = max(1.0, float(view_count - 1))
    support_small = support_small / denom
    if int(min_other_views) > 1:
        support_small = np.where(support_small >= float(min_other_views) / denom, support_small, 0.0)
    support_maps = np.zeros((view_count, height, width), dtype=np.float32)
    for view_idx in range(view_count):
        small_img = support_small[view_idx].reshape(yy.shape)
        support_maps[view_idx] = np.repeat(np.repeat(small_img, stride, axis=0), stride, axis=1)[:height, :width]
    summary = {
        "radius": float(radius),
        "sample_stride": int(stride),
        "min_other_views": int(min_other_views),
        "support_mean": float(support_maps.mean()),
        "support_p50": float(np.percentile(support_maps, 50)),
        "support_p90": float(np.percentile(support_maps, 90)),
    }
    return support_maps, summary


def reweight_conf(conf: np.ndarray, support: np.ndarray, base_weight: float) -> np.ndarray:
    conf = np.asarray(conf, dtype=np.float32)
    support = np.asarray(support, dtype=np.float32)
    max_conf = np.nanpercentile(conf[np.isfinite(conf)], 99.5) if np.isfinite(conf).any() else 1.0
    normalized = np.clip(conf / max(float(max_conf), 1e-6), 0.0, 1.0)
    weight = float(base_weight) + (1.0 - float(base_weight)) * np.clip(support, 0.0, 1.0)
    return (normalized * weight * max(float(max_conf), 1e-6)).astype(np.float32)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_npz).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(input_path, allow_pickle=False)
    support_points = resolve_support_points(data, str(args.support_source))
    support, support_summary = support_score_per_view(
        support_points,
        radius=float(args.radius),
        stride=max(1, int(args.sample_stride)),
        min_other_views=int(args.min_other_views),
    )

    payload = {key: data[key] for key in data.files}
    original_world_conf = np.asarray(payload["world_points_conf"], dtype=np.float32)
    original_depth_conf = np.asarray(payload["depth_conf"], dtype=np.float32)
    payload["world_points_conf"] = reweight_conf(original_world_conf, support, base_weight=float(args.base_conf_weight))
    if bool(args.write_depth_conf):
        payload["depth_conf"] = reweight_conf(original_depth_conf, support, base_weight=float(args.base_conf_weight))
    payload["cross_view_support_conf"] = support.astype(np.float32)

    output_npz = output_dir / "predictions.npz"
    np.savez_compressed(output_npz, **payload)
    summary = {
        "name": args.name,
        "input_npz": str(input_path),
        "output_npz": str(output_npz),
        "support_source": str(args.support_source),
        "write_depth_conf": bool(args.write_depth_conf),
        "base_conf_weight": float(args.base_conf_weight),
        "support_summary": support_summary,
        "note": "Diagnostic confidence reweight only; coordinates are unchanged.",
    }
    (output_dir / "cross_view_support_conf_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
