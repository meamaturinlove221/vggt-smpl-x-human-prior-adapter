from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from v10_surface_completion_pipeline import LOCAL_ROOT, REPORTS, REPO_ROOT, json_ready, read_summary, write_json, write_report


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    parser = argparse.ArgumentParser(description="V14 F14 Fus3D region backend readiness.")
    parser.add_argument("--output-json", type=Path, default=REPORTS / "20260508_v14_fus3d_region_backend_readiness.json")
    parser.add_argument("--output-md", type=Path, default=REPORTS / "20260508_v14_fus3d_region_backend_readiness.md")
    args = parser.parse_args()
    candidates = {
        "b_fus3d5_script": REPO_ROOT / "tools/b_fus3d5_body_head_face_surface_backend_train.py",
        "b_fus3d4_export": REPO_ROOT / "tools/b_fus3d4_export_surface_candidate_precheck.py",
        "sapiens_normals": REPO_ROOT / "output/surface_research_cloud_preflight/V13_Sapiens_Normal/sapiens_normals.npz",
        "sapiens_depths": REPO_ROOT / "output/surface_research_cloud_preflight/V13_Sapiens_Depth/sapiens_depths.npz",
        "g1_surface": LOCAL_ROOT / "V13_G1_2DGS_strict_surface/g1_2dgs_strict_surface_trimmed.ply",
        "k2_surface": LOCAL_ROOT / "V13_K2_kinect_alignment_candidate_search/k2_best_kinect_tsdf_vggt_world.ply",
    }
    rows = {name: {"path": str(path.resolve()), "exists": path.exists(), "size": path.stat().st_size if path.is_file() else 0} for name, path in candidates.items()}
    k14 = read_summary(REPORTS / "20260508_v14_kinect_teacher_gate_autopsy.json")
    g14 = read_summary(REPORTS / "20260508_v14_2dgs_protocol_alignment_audit.json")
    s14 = read_summary(REPORTS / "20260508_v14_sapiens_normal_depth_qa.json")
    body_head_face_ready = bool(False)
    blockers = []
    if not rows["b_fus3d5_script"]["exists"]:
        blockers.append("B-Fus3D5/F14 training script is missing.")
    if not (s14.get("status") == "s14_sapiens_qa_ready"):
        blockers.append("Sapiens supervision is not QA-ready.")
    if not (k14.get("strict_teacher_precheck_pass") or g14.get("strict_teacher_precheck_pass")):
        blockers.append("No strict K/G teacher or protocol-pass anchor is available for F14 supervised body/head/face training.")
    summary = {
        "task": "v14_f14_fus3d_region_backend_readiness",
        "created_utc": utc_now(),
        "status": "f14_region_backend_blocked" if blockers else "f14_region_backend_ready",
        "assets": rows,
        "body_head_face_ready": body_head_face_ready,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "decision": "F14 is not launched as a formal region backend because K/G teacher anchors still fail strict protocol." if blockers else "F14 assets are ready for bounded research training.",
        "blockers": blockers,
    }
    write_json(args.output_json, summary)
    write_report(args.output_md, "V14 Fus3D Region Backend Readiness", summary)
    print(json.dumps(json_ready({"status": summary["status"], "body_head_face_ready": body_head_face_ready}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
