from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from v29_normal_utils import (
    V26_TARGETS,
    V29_OUT,
    all_regions_nonempty,
    jr,
    load_npz,
    normals_from_point_map,
    region_support_from_id_map,
    utc_now,
    vector_length_stats,
    write_json,
)


DEFAULT_OUT = V29_OUT / "v29_temporal_normals_world.npz"
DEFAULT_CONF = V29_OUT / "v29_temporal_normal_confidence.npz"
DEFAULT_SUMMARY = V29_OUT / "v29_temporal_normal_audit.json"


def build_temporal_normals(targets: Path) -> dict[str, Any]:
    z = load_npz(targets)
    required = [
        "target_frame_points",
        "target_frame_region_id_map",
        "canonical_residual",
        "canonical_support",
        "temporal_confidence",
        "temporal_uncertainty",
        "temporal_variance",
        "frame_keys",
        "normal_available",
    ]
    missing = [k for k in required if k not in z]
    if missing:
        raise KeyError(f"Missing V26 temporal keys: {missing}")

    points = np.asarray(z["target_frame_points"], dtype=np.float32)
    residual = np.asarray(z["canonical_residual"], dtype=np.float32)
    support = np.asarray(z["canonical_support"], dtype=bool)
    region_id_map = np.asarray(z["target_frame_region_id_map"], dtype=np.uint8)
    confidence = np.asarray(z["temporal_confidence"], dtype=np.float32)
    uncertainty = np.asarray(z["temporal_uncertainty"], dtype=np.float32)
    variance = np.asarray(z["temporal_variance"], dtype=np.float32)
    frame_keys = np.asarray(z["frame_keys"])

    normals, valid = normals_from_point_map(points, mask=support)
    residual_normals, residual_valid = normals_from_point_map(residual, mask=support)
    normal_confidence = np.where(valid, np.clip(confidence, 0.0, 1.0), 0.0).astype(np.float32)
    variance_norm = variance / (float(np.nanmax(variance)) + 1e-6)
    normal_uncertainty = np.where(valid, np.clip(0.5 * uncertainty + 0.5 * variance_norm, 0.0, 1.0), 1.0).astype(np.float32)
    region_support = region_support_from_id_map(valid, region_id_map)

    payload = {
        "v29_temporal_normals_world": normals.astype(np.float32),
        "v29_temporal_normal_valid": valid.astype(np.uint8),
        "v29_temporal_normal_confidence": normal_confidence,
        "v29_temporal_normal_uncertainty": normal_uncertainty,
        "v29_temporal_region_id_map": region_id_map,
        "v29_temporal_residual_normals": residual_normals.astype(np.float32),
        "v29_temporal_residual_normal_valid": residual_valid.astype(np.uint8),
        "frame_keys": frame_keys,
        "source": np.asarray(["V26.target_frame_points finite_difference", "V26.canonical_residual finite_difference"]),
    }
    conf_payload = {
        "v29_temporal_normal_confidence": normal_confidence,
        "v29_temporal_normal_uncertainty": normal_uncertainty,
        "v29_temporal_normal_valid": valid.astype(np.uint8),
    }
    summary = {
        "status": "DONE_PASS" if all_regions_nonempty(region_support) else "DONE_FAIL_ROUTED",
        "created_at": utc_now(),
        "source": targets,
        "output": DEFAULT_OUT,
        "confidence_output": DEFAULT_CONF,
        "normal_source": "geometric_finite_difference_from_v26_target_frame_points",
        "canonical_residual_normal_source": "geometric_finite_difference_from_v26_canonical_residual",
        "v26_declared_normal_available": bool(np.asarray(z["normal_available"]).item()),
        "v29_reconstructed_normal_available": bool(valid.sum() > 0),
        "shape": list(normals.shape),
        "normal_valid_count": int(valid.sum()),
        "residual_normal_valid_count": int(residual_valid.sum()),
        "region_normal_support": region_support,
        "all_required_regions_nonempty": all_regions_nonempty(region_support),
        "normal_length_stats": vector_length_stats(normals, valid),
        "forbidden_writes": {
            "predictions_npz": False,
            "candidate_package": False,
            "teacher_package": False,
            "strict_registry": False,
            "strict_pass": False,
        },
    }
    return {"payload": payload, "confidence": conf_payload, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconstruct temporal normals from V26 target-frame and canonical residual geometry.")
    parser.add_argument("--targets", type=Path, default=V26_TARGETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--confidence-output", type=Path, default=DEFAULT_CONF)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    result = build_temporal_normals(args.targets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result["payload"])
    np.savez_compressed(args.confidence_output, **result["confidence"])
    result["summary"]["output"] = args.output
    result["summary"]["confidence_output"] = args.confidence_output
    write_json(args.summary, result["summary"])
    print(jr(result["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
