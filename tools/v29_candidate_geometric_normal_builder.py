from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from v29_normal_utils import (
    V24_TARGETS,
    V25_POINTS,
    V29_OUT,
    all_regions_nonempty,
    jr,
    load_npz,
    normals_from_point_map,
    region_support_from_masks,
    utc_now,
    vector_length_stats,
    write_json,
)


DEFAULT_OUT = V29_OUT / "v29_candidate_geometric_normals.npz"
DEFAULT_SUMMARY = V29_OUT / "v29_candidate_geometric_normal_summary.json"


def build_candidate_normals(points_path: Path, v24_targets: Path) -> dict[str, Any]:
    if not points_path.exists():
        raise FileNotFoundError(points_path)
    points_npz = load_npz(points_path, allow_pickle=True)
    v24 = load_npz(v24_targets)
    region_masks = np.asarray(v24["teacher_region_masks"], dtype=bool)
    region_names = tuple(str(x) for x in np.asarray(v24["teacher_region_names"]).tolist())

    frame_keys = [str(x) for x in np.asarray(points_npz.get("frame_keys", np.asarray(["frame0000"]))).tolist()]
    payload: dict[str, Any] = {
        "frame_keys": np.asarray(frame_keys),
        "source": np.asarray(["geometric_from_candidate_world_points", str(points_path)]),
    }
    frame_summaries: dict[str, Any] = {}
    best_region_support: dict[str, int] = {name: 0 for name in region_names}
    best_frame = None

    for frame_key in frame_keys:
        if frame_key not in points_npz:
            continue
        points = np.asarray(points_npz[frame_key], dtype=np.float32)
        normals, valid = normals_from_point_map(points)
        payload[f"{frame_key}_candidate_geometric_normals"] = normals.astype(np.float32)
        payload[f"{frame_key}_candidate_normal_valid"] = valid.astype(np.uint8)

        # V24 region masks are 6 views; use the first 6 candidate views for same-frame region support.
        comparable_views = min(valid.shape[0], region_masks.shape[1])
        region_support = region_support_from_masks(valid[:comparable_views], region_masks[:, :comparable_views], region_names)  # type: ignore[arg-type]
        if sum(region_support.values()) > sum(best_region_support.values()):
            best_region_support = region_support
            best_frame = frame_key
        frame_summaries[frame_key] = {
            "shape": list(points.shape),
            "normal_valid_count": int(valid.sum()),
            "normal_length_stats": vector_length_stats(normals, valid),
            "region_normal_support_first6_against_v24_masks": region_support,
        }

    summary = {
        "status": "DONE_PASS" if all_regions_nonempty(best_region_support, region_names) else "DONE_FAIL_ROUTED",
        "created_at": utc_now(),
        "source": points_path,
        "v24_region_mask_source": v24_targets,
        "output": DEFAULT_OUT,
        "normal_source": "geometric_from_candidate_points_world_finite_difference",
        "not_model_normal_head": True,
        "frame_summaries": frame_summaries,
        "best_region_support_frame": best_frame,
        "region_normal_support": best_region_support,
        "all_required_regions_nonempty": all_regions_nonempty(best_region_support, region_names),
        "forbidden_writes": {
            "predictions_npz": False,
            "candidate_package": False,
            "teacher_package": False,
            "strict_registry": False,
            "strict_pass": False,
        },
    }
    return {"payload": payload, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build candidate geometric normals from V25 research world point maps.")
    parser.add_argument("--points", type=Path, default=V25_POINTS)
    parser.add_argument("--v24-targets", type=Path, default=V24_TARGETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    result = build_candidate_normals(args.points, args.v24_targets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result["payload"])
    result["summary"]["output"] = args.output
    write_json(args.summary, result["summary"])
    print(jr(result["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
