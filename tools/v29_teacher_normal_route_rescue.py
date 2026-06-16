from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from v29_normal_utils import (
    REGION_NAMES,
    V24_TARGETS,
    V29_OUT,
    all_regions_nonempty,
    geometric_consistency,
    jr,
    load_npz,
    normals_from_point_map,
    normalize_vectors,
    region_support_from_masks,
    utc_now,
    vector_length_stats,
    write_json,
)


DEFAULT_OUT = V29_OUT / "v29_teacher_normals_world.npz"
DEFAULT_SUMMARY = V29_OUT / "v29_teacher_normal_route_rescue_summary.json"


def build_teacher_normals(targets: Path) -> dict[str, Any]:
    z = load_npz(targets)
    required = [
        "teacher_points_world",
        "teacher_normals_world",
        "teacher_visibility",
        "teacher_uncertainty",
        "teacher_mask",
        "teacher_region_masks",
        "teacher_region_names",
    ]
    missing = [k for k in required if k not in z]
    if missing:
        raise KeyError(f"Missing V24 teacher keys: {missing}")

    points = np.asarray(z["teacher_points_world"], dtype=np.float32)
    normals = np.asarray(z["teacher_normals_world"], dtype=np.float32)
    mask = np.asarray(z["teacher_mask"], dtype=bool)
    visibility = np.asarray(z["teacher_visibility"], dtype=np.float32)
    uncertainty = np.asarray(z["teacher_uncertainty"], dtype=np.float32)
    region_masks = np.asarray(z["teacher_region_masks"], dtype=bool)
    region_names = tuple(str(x) for x in np.asarray(z["teacher_region_names"]).tolist())

    propagated_normals, normal_valid = normalize_vectors(normals, mask=mask)
    normal_visibility = np.where(normal_valid, np.clip(visibility, 0.0, 1.0), 0.0).astype(np.float32)
    normal_uncertainty = np.where(normal_valid, np.clip(uncertainty, 0.0, 1.0), 1.0).astype(np.float32)
    region_support = region_support_from_masks(normal_valid, region_masks, region_names)  # type: ignore[arg-type]

    geometric_normals, geometric_valid = normals_from_point_map(points, mask=mask)
    consistency = geometric_consistency(propagated_normals, geometric_normals, mask=normal_valid & geometric_valid)

    payload = {
        "v29_teacher_normals_world": propagated_normals.astype(np.float32),
        "v29_teacher_normal_visibility": normal_visibility,
        "v29_teacher_normal_uncertainty": normal_uncertainty,
        "v29_teacher_region_masks": region_masks.astype(np.uint8),
        "v29_teacher_region_names": np.asarray(region_names),
        "v29_teacher_normal_valid": normal_valid.astype(np.uint8),
        "v29_teacher_geometric_normals_world": geometric_normals.astype(np.float32),
        "v29_teacher_geometric_normal_valid": geometric_valid.astype(np.uint8),
        "source": np.asarray(["V24_residual_teacher_v2.teacher_normals_world"]),
    }
    summary = {
        "status": "DONE_PASS" if all_regions_nonempty(region_support, region_names) else "DONE_FAIL_ROUTED",
        "created_at": utc_now(),
        "source": targets,
        "output": DEFAULT_OUT,
        "normal_source": "propagated_from_v24_teacher_normals_world",
        "geometric_consistency_source": "finite_difference_from_v24_teacher_points_world",
        "shape": list(propagated_normals.shape),
        "normal_valid_count": int(normal_valid.sum()),
        "region_normal_support": region_support,
        "all_required_regions_nonempty": all_regions_nonempty(region_support, region_names),
        "normal_length_stats": vector_length_stats(propagated_normals, normal_valid),
        "teacher_vs_geometric_normal_consistency": consistency,
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
    parser = argparse.ArgumentParser(description="Propagate V24 teacher normals into V29 strict-compatible normal evidence.")
    parser.add_argument("--targets", type=Path, default=V24_TARGETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    result = build_teacher_normals(args.targets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result["payload"])
    result["summary"]["output"] = args.output
    write_json(args.summary, result["summary"])
    print(jr(result["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
