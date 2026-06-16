#!/usr/bin/env python
"""Create lightweight V32 Open3D-style PLY review artifacts without requiring Open3D."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "surface_research_preflight_local" / "V32_candidate_inference_research"
REPORT_JSON = ROOT / "reports" / "20260508_v32_candidate_open3d_review.json"
REPORT_MD = ROOT / "reports" / "20260508_v32_candidate_open3d_review.md"


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def main() -> None:
    pts_npz = np.load(OUT / "candidate_points_world_research.npz")
    vis_npz = np.load(OUT / "candidate_visibility_research.npz")
    points = pts_npz["candidate_points_world"]
    vis = vis_npz["candidate_visibility"] > 0.5
    flat = points[vis]
    if len(flat) > 120000:
        idx = np.linspace(0, len(flat) - 1, 120000).astype(np.int64)
        flat = flat[idx]
    colors = np.tile(np.array([[90, 180, 230]], dtype=np.uint8), (len(flat), 1))
    ply = OUT / "v32_candidate_open3d_review_points.ply"
    _write_ply(ply, flat.astype(np.float32), colors)
    audit = {
        "status": "DONE_PASS" if len(flat) > 0 else "DONE_FAIL_ROUTED",
        "ply_path": str(ply),
        "sampled_vertex_count": int(len(flat)),
    }
    REPORT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    REPORT_MD.write_text("# V32 Candidate Open3D Review\n\nstatus: `{}`\n\nply: `{}`\n\nsampled_vertex_count: `{}`\n".format(audit["status"], ply, len(flat)), encoding="utf-8")
    print(json.dumps({"status": audit["status"], "ply_path": str(ply)}, indent=2))


if __name__ == "__main__":
    main()
