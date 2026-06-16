#!/usr/bin/env python
"""Audit V32 candidate normal/depth consistency."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "surface_research_preflight_local" / "V32_candidate_inference_research"
REPORT_JSON = ROOT / "reports" / "20260508_v32_candidate_normal_depth_audit.json"
REPORT_MD = ROOT / "reports" / "20260508_v32_candidate_normal_depth_audit.md"


def main() -> None:
    depth = np.load(OUT / "candidate_depths_research.npz")["candidate_depths"]
    normals = np.load(OUT / "candidate_normals_geometric_research.npz")["candidate_normals_geometric"]
    vis = np.load(OUT / "candidate_visibility_research.npz")["candidate_visibility"] > 0.5
    nlen = np.linalg.norm(normals, axis=-1)
    finite_depth = np.isfinite(depth) & vis
    metrics = {
        "visible_pixels": int(vis.sum()),
        "finite_depth_ratio": float(finite_depth.sum() / max(1, int(vis.sum()))),
        "normal_unit_ratio": float(((nlen > 0.8) & (nlen < 1.2) & vis).sum() / max(1, int(vis.sum()))),
        "depth_min": float(np.min(depth[finite_depth])) if finite_depth.any() else None,
        "depth_max": float(np.max(depth[finite_depth])) if finite_depth.any() else None,
    }
    blockers = []
    if metrics["finite_depth_ratio"] < 0.95:
        blockers.append("depth_not_finite")
    if metrics["normal_unit_ratio"] < 0.80:
        blockers.append("normal_not_unit_enough")
    audit = {"status": "DONE_PASS" if not blockers else "DONE_FAIL_ROUTED", "blockers": blockers, "metrics": metrics}
    REPORT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    REPORT_MD.write_text("# V32 Candidate Normal Depth Audit\n\nstatus: `{}`\n\nmetrics: `{}`\n\nblockers: `{}`\n".format(audit["status"], metrics, blockers), encoding="utf-8")
    print(json.dumps({"status": audit["status"], "blockers": blockers}, indent=2))


if __name__ == "__main__":
    main()
