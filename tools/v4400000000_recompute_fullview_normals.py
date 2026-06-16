"""Recompute geometric normals for full-view VGGT-style point maps.

This tool is intentionally conservative: it does not claim a learned normal
head. It fills `normal` from local finite-difference geometry and records that
source in the caller's manifest.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def recompute_normals(world_points: np.ndarray, valid_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return finite-difference normals and confidence for (V,H,W,3) points."""
    wp = np.asarray(world_points, dtype=np.float32)
    if wp.ndim != 4 or wp.shape[-1] != 3:
        raise ValueError(f"world_points must have shape (V,H,W,3), got {wp.shape}")

    v, h, w, _ = wp.shape
    normals = np.zeros_like(wp, dtype=np.float32)

    dx = wp[:, 1:-1, 2:, :] - wp[:, 1:-1, :-2, :]
    dy = wp[:, 2:, 1:-1, :] - wp[:, :-2, 1:-1, :]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    good = norm[..., 0] > 1e-12
    n = np.divide(n, np.maximum(norm, 1e-12), out=np.zeros_like(n), where=norm > 1e-12)
    normals[:, 1:-1, 1:-1, :] = n

    if valid_mask is None:
        finite = np.isfinite(wp).all(axis=-1)
    else:
        finite = np.asarray(valid_mask).astype(bool) & np.isfinite(wp).all(axis=-1)

    conf = np.zeros((v, h, w), dtype=np.float32)
    conf[:, 1:-1, 1:-1] = good.astype(np.float32)
    conf *= finite.astype(np.float32)
    normals *= conf[..., None]
    return normals, conf


def normal_stats(normals: np.ndarray, valid_mask: np.ndarray | None = None) -> dict[str, Any]:
    mag = np.linalg.norm(normals, axis=-1)
    valid = np.isfinite(mag)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask).astype(bool)
    denom = int(valid.sum())
    nonzero = (mag > 1e-8) & valid
    return {
        "valid_count": denom,
        "normal_nonzero_count": int(nonzero.sum()),
        "normal_nonzero_ratio": float(nonzero.sum() / max(denom, 1)),
        "normal_magnitude_mean": float(mag[valid].mean()) if denom else 0.0,
        "normal_magnitude_std": float(mag[valid].std()) if denom else 0.0,
    }


def repair_prediction(input_npz: Path, output_npz: Path) -> dict[str, Any]:
    with np.load(input_npz, allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    if "world_points" not in arrays:
        raise KeyError(f"{input_npz} lacks world_points")

    valid_mask = None
    if "world_points_conf" in arrays:
        valid_mask = np.asarray(arrays["world_points_conf"]) > 0
    normals, normal_conf = recompute_normals(arrays["world_points"], valid_mask)
    arrays["normal_learned_input"] = np.asarray(arrays.get("normal", np.zeros_like(normals)), dtype=np.float32)
    arrays["normal"] = normals.astype(np.float32)
    arrays["normal_conf"] = normal_conf.astype(np.float32)
    arrays["normal_source_code"] = np.asarray("geometric_finite_difference_recomputed")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)

    stats = normal_stats(normals, valid_mask)
    return {
        "input_npz": str(input_npz),
        "output_npz": str(output_npz),
        "normal_source": "geometric_finite_difference_recomputed",
        "learned_normal_preserved_as": "normal_learned_input",
        "created_utc": now_utc(),
        **stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = repair_prediction(args.input, args.output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
