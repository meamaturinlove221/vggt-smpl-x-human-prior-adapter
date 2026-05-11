from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from render_open3d_pointcloud import unproject_depth_map_to_point_map_numpy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local diagnostic backend: replace world_points with the candidate's "
            "own depth+camera unprojection. This checks whether a candidate fails "
            "because the point-map branch is inconsistent with the depth branch. "
            "It is not a teacher, not a camera replacement, and not a pass claim."
        )
    )
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--world-conf-source",
        choices=("depth_conf", "max_depth_world", "keep_world"),
        default="depth_conf",
        help="Confidence assigned to synced world_points.",
    )
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.input_npz, allow_pickle=False) as payload:
        data = {key: np.asarray(payload[key]) for key in payload.files}

    required = ("depth", "extrinsic", "intrinsic", "world_points", "world_points_conf", "depth_conf")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing required arrays: {missing}")

    old_world = np.asarray(data["world_points"], dtype=np.float32)
    synced_world = unproject_depth_map_to_point_map_numpy(
        np.asarray(data["depth"], dtype=np.float32),
        np.asarray(data["extrinsic"], dtype=np.float32),
        np.asarray(data["intrinsic"], dtype=np.float32),
    ).astype(np.float32)
    if synced_world.shape != old_world.shape:
        raise ValueError(f"Synced world shape {synced_world.shape} != old world shape {old_world.shape}")

    delta = synced_world - old_world
    delta_norm = np.linalg.norm(delta, axis=-1)
    finite = np.isfinite(delta_norm)

    out = dict(data)
    out["world_points"] = synced_world.astype(data["world_points"].dtype, copy=False)
    depth_conf = np.asarray(data["depth_conf"], dtype=np.float32)
    world_conf = np.asarray(data["world_points_conf"], dtype=np.float32)
    if args.world_conf_source == "depth_conf":
        out["world_points_conf"] = depth_conf.astype(data["world_points_conf"].dtype, copy=False)
    elif args.world_conf_source == "max_depth_world":
        out["world_points_conf"] = np.maximum(world_conf, depth_conf).astype(data["world_points_conf"].dtype, copy=False)
    else:
        out["world_points_conf"] = data["world_points_conf"]

    output_npz = args.output_dir / "predictions.npz"
    np.savez_compressed(output_npz, **out)

    summary = {
        "task": "sync_world_points_to_depth_unprojection",
        "input_npz": str(args.input_npz.resolve()),
        "output_npz": str(output_npz.resolve()),
        "world_conf_source": args.world_conf_source,
        "shape": list(synced_world.shape),
        "finite_delta_pixels": int(finite.sum()),
        "delta_l2_mean": float(np.mean(delta_norm[finite])) if finite.any() else None,
        "delta_l2_median": float(np.median(delta_norm[finite])) if finite.any() else None,
        "delta_l2_p90": float(np.percentile(delta_norm[finite], 90)) if finite.any() else None,
        "truthful_note": (
            "This is a local backend diagnostic. It uses only the candidate's own "
            "depth/camera output to synchronize world_points. It cannot pass the "
            "mentor gate unless the strict candidate package and explicit Open3D "
            "visual review pass."
        ),
    }
    (args.output_dir / "sync_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
