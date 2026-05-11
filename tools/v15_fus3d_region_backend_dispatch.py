from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from v10_surface_completion_pipeline import LOCAL_ROOT, REPORTS, REPO_ROOT, json_ready, read_summary, write_json


DEFAULT_REGISTRY = REPORTS / "20260508_v14_strict_gate_registry_refresh.json"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_row(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size": path.stat().st_size if path.is_file() else 0,
    }


def run_json_command(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    try:
        payload = json.loads(proc.stdout)
    except Exception:
        payload = {"stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:]}
    payload["returncode"] = int(proc.returncode)
    return payload


def formal_guards(registry: Path) -> dict[str, Any]:
    return {
        "candidate": run_json_command([sys.executable, "tools/check_cloud_gate_status.py", "--registry", str(registry), "--json"]),
        "teacher_supervised": run_json_command(
            [sys.executable, "tools/check_cloud_gate_status.py", "--registry", str(registry), "--teacher-supervised", "--json"]
        ),
    }


def research_gate_probe() -> dict[str, Any]:
    return run_json_command([sys.executable, "tools/check_research_cloud_gate_status.py", "--json"])


def local_region_prep_command() -> list[str]:
    return ["python", "tools/b_fus3d5_body_head_face_surface_backend_train.py"]


def cloud_region_train_command() -> list[str]:
    return [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "tools\\run_modal_utf8.ps1",
        "run",
        "modal_v9_b_fus3d3_real_asset_train.py::run_train",
        "--remote-asset-subdir",
        "surface_research_cloud_preflight/V9_cloud_asset_staging/assets",
        "--remote-output-subdir",
        "surface_research_cloud_preflight/V15_Fus3D_region_backend_dispatch/b_fus3d3_real_asset_train_preflight",
        "--max-steps",
        "500",
        "--max-cases",
        "8",
        "--max-hours",
        "1.0",
        "--download-local-dir",
        "output\\surface_research_cloud_preflight\\V15_Fus3D_region_backend_dispatch\\b_fus3d3_real_asset_train_preflight",
    ]


def research_cloud_lane_command() -> list[str]:
    return [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "tools\\run_modal_utf8.ps1",
        "run",
        "modal_v8_research_cloud.py::run_job",
        "--job-id",
        "Cloud-A-V15-Fus3D",
        "--lane",
        "b_fus3d2_human_dataset_train",
        "--output-subdir",
        "surface_research_cloud_preflight/V15_Fus3D_region_backend_dispatch/b_fus3d2_human_dataset_train",
        "--max-steps",
        "120",
        "--max-cases",
        "8",
        "--max-hours",
        "1.0",
        "--download-local-dir",
        "output\\surface_research_cloud_preflight\\V15_Fus3D_region_backend_dispatch\\b_fus3d2_human_dataset_train",
    ]


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V15 Fus3D Region Backend Dispatch",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only dispatch/readiness report. This script does not write predictions, teacher/candidate packages, registries, or strict pass state.",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Source Gates",
        "",
        "```json",
        json.dumps(summary["source_gate_results"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Formal Guard",
        "",
        "```json",
        json.dumps(summary["formal_guards"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Research Cloud Gate",
        "",
        "```json",
        json.dumps(summary["research_cloud_gate"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Dispatch Commands",
        "",
    ]
    for label, row in summary.get("dispatch_commands", {}).items():
        lines.extend(
            [
                f"### {label}",
                "",
                f"- allowed_by_this_report: `{str(row['allowed_by_this_report']).lower()}`",
                f"- reason: {row['reason']}",
                "",
                "```powershell",
                " ".join(row["command"]),
                "```",
                "",
            ]
        )
    lines.extend(["## Blockers", ""])
    blockers = summary.get("blockers") or []
    lines.extend([f"- {item}" for item in blockers] if blockers else ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def first_summary(*paths: Path) -> dict[str, Any]:
    for path in paths:
        data = read_summary(path)
        if data:
            return data
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="V15 F-line Fus3D region backend dispatch/readiness.")
    parser.add_argument("--output-json", type=Path, default=REPORTS / "20260508_v15_fus3d_region_backend_dispatch.json")
    parser.add_argument("--output-md", type=Path, default=REPORTS / "20260508_v15_fus3d_region_backend_dispatch.md")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()

    t15 = read_summary(REPORTS / "20260508_v15_tmf_prediction_dispatch.json")
    t14 = read_summary(REPORTS / "20260508_v14_tmf_prediction_readiness.json")
    f14 = read_summary(REPORTS / "20260508_v14_fus3d_region_backend_readiness.json")
    k14 = first_summary(REPORTS / "V14_K14/20260508_k14_kinect_teacher_gate_autopsy.json", REPORTS / "20260508_v14_kinect_teacher_gate_autopsy.json")
    g14 = read_summary(REPORTS / "20260508_v14_2dgs_protocol_alignment_audit.json")
    s14 = read_summary(REPORTS / "20260508_v14_sapiens_normal_depth_qa.json")
    h14 = first_summary(REPORTS / "V14_H14_R14/readiness.json", REPORTS / "20260508_v14_external_hand_hair_asset_manager.json")
    dline = read_summary(REPORTS / "20260508_v14_dline_promotion_report.json")
    source_graph = read_summary(REPORTS / "20260508_v15_source_graph.json")

    guards = formal_guards(args.registry.expanduser().resolve())
    research_gate = research_gate_probe()
    formal_cloud_unblocked = bool(guards["candidate"].get("cloud_allowed") or guards["teacher_supervised"].get("cloud_allowed"))

    assets = {
        "b_fus3d5_script": file_row(REPO_ROOT / "tools/b_fus3d5_body_head_face_surface_backend_train.py"),
        "b_fus3d4_export": file_row(REPO_ROOT / "tools/b_fus3d4_export_surface_candidate_precheck.py"),
        "sapiens_normals": file_row(REPO_ROOT / "output/surface_research_cloud_preflight/V13_Sapiens_Normal/sapiens_normals.npz"),
        "sapiens_depths": file_row(REPO_ROOT / "output/surface_research_cloud_preflight/V13_Sapiens_Depth/sapiens_depths.npz"),
        "g1_surface": file_row(LOCAL_ROOT / "V13_G1_2DGS_strict_surface/g1_2dgs_strict_surface_trimmed.ply"),
        "k2_surface": file_row(LOCAL_ROOT / "V13_K2_kinect_alignment_candidate_search/k2_best_kinect_tsdf_vggt_world.ply"),
        "existing_body": file_row(LOCAL_ROOT / "V11_B_Fus3D5_body_head_face/b_fus3d5_body_surface.ply"),
        "existing_head": file_row(LOCAL_ROOT / "V11_B_Fus3D5_body_head_face/b_fus3d5_head_surface.ply"),
        "existing_face": file_row(LOCAL_ROOT / "V11_B_Fus3D5_body_head_face/b_fus3d5_face_surface.ply"),
        "existing_summary": file_row(LOCAL_ROOT / "V11_B_Fus3D5_body_head_face/summary.json"),
    }

    t_predictions_ready = bool(t15.get("frame0001_0002_predictions_ready") or t14.get("frame0001_0002_predictions_ready"))
    temporal_dispatch_blocked = bool(t15 and not t15.get("frame0001_0002_predictions_ready"))
    strict_teacher_anchor_ready = bool(k14.get("strict_teacher_precheck_pass") or g14.get("strict_teacher_precheck_pass") or t15.get("canonical_teacher_ready") or t14.get("canonical_teacher_ready"))
    sapiens_ready = bool(s14.get("status") == "s14_sapiens_qa_ready")
    research_inputs_ready = bool(
        assets["b_fus3d5_script"]["exists"]
        and sapiens_ready
        and (assets["g1_surface"]["exists"] or assets["k2_surface"]["exists"])
    )
    hand_ready = bool(h14.get("hand_ownership_ready"))
    hair_ready = bool(h14.get("hair_ownership_ready"))
    existing_research_region_surfaces = bool(
        assets["existing_body"]["exists"] and assets["existing_head"]["exists"] and assets["existing_face"]["exists"]
    )
    body_head_face_ready = bool(False)

    blockers: list[str] = []
    if not research_inputs_ready:
        blockers.append("Fus3D/Sapiens/Kinect/2DGS research inputs are incomplete.")
    if not strict_teacher_anchor_ready:
        blockers.append("No strict K/G/T teacher or protocol-pass anchor is available for supervised body/head/face training.")
    if not t_predictions_ready:
        blockers.append("Frame0001/frame0002 VGGT predictions are missing; temporal region training cannot claim readiness.")
    if temporal_dispatch_blocked:
        blockers.append(f"T15 dispatch is blocked: {t15.get('status')}.")
    if not hand_ready:
        blockers.append("Hand ownership remains unavailable, so F-line cannot form a unified strict candidate.")
    if not hair_ready:
        blockers.append("Hair topology ownership remains unavailable, so F-line cannot form a unified strict candidate.")
    if not formal_cloud_unblocked:
        blockers.append("Formal cloud guard remains blocked; formal/cloud Fus3D training commands are not authorized.")

    dispatch_commands = {
        "bounded_local_region_prep": {
            "command": local_region_prep_command(),
            "allowed_by_this_report": bool(research_inputs_ready),
            "reason": (
                "bounded local research-prep wrapper may be re-run; it writes V11_B_Fus3D5 research outputs only"
                if research_inputs_ready
                else "local research inputs are incomplete"
            ),
        },
        "research_cloud_lane_dry_run_constraint": {
            "command": research_cloud_lane_command(),
            "allowed_by_this_report": bool(research_gate.get("research_gate", {}).get("allowed") and not formal_cloud_unblocked),
            "reason": (
                "research cloud gate allows bounded non-prediction preflight lanes only; post-job referee still required"
                if research_gate.get("research_gate", {}).get("allowed")
                else "research cloud guard did not explicitly allow this lane"
            ),
        },
        "formal_or_cloud_region_training": {
            "command": cloud_region_train_command(),
            "allowed_by_this_report": False,
            "reason": "formal/cloud region training is blocked by strict gates and must not be launched as an unbounded or promotion-producing job",
        },
    }

    source_gate_results = {
        "k14_teacher_precheck_pass": bool(k14.get("strict_teacher_precheck_pass")),
        "g14_teacher_precheck_pass": bool(g14.get("strict_teacher_precheck_pass")),
        "s14_supervision_ready": sapiens_ready,
        "t15_predictions_ready": t_predictions_ready,
        "t15_status": t15.get("status"),
        "f14_body_head_face_ready": bool(f14.get("body_head_face_ready")),
        "hand_ownership_ready": hand_ready,
        "hair_ownership_ready": hair_ready,
        "research_inputs_ready": research_inputs_ready,
        "strict_teacher_anchor_ready": strict_teacher_anchor_ready,
        "existing_research_region_surfaces": existing_research_region_surfaces,
        "source_graph_status": source_graph.get("status"),
    }

    if research_inputs_ready and t_predictions_ready and strict_teacher_anchor_ready and hand_ready and hair_ready:
        status = "v15_fus3d_dispatch_ready_for_review_not_promoted"
        decision = "All named blockers appear cleared, but this dispatcher still writes no strict pass; D-line must review before promotion."
    elif research_inputs_ready:
        status = "v15_fus3d_research_prep_feasible_formal_blocked"
        decision = "Fus3D region research prep is feasible, but formal body/head/face readiness is blocked by missing strict teacher/temporal/hand/hair gates."
    else:
        status = "v15_fus3d_dispatch_blocked_inputs_missing"
        decision = "Fus3D region backend dispatch is blocked because required research inputs are incomplete."

    summary = {
        "task": "v15_fus3d_region_backend_dispatch",
        "created_utc": utc_now(),
        "status": status,
        "assets": assets,
        "source_status": {
            "t15": t15.get("status"),
            "t14": t14.get("status"),
            "f14": f14.get("status"),
            "k14": k14.get("status"),
            "g14": g14.get("status"),
            "s14": s14.get("status"),
            "h14_r14": h14.get("status"),
            "dline": dline.get("status"),
            "source_graph": source_graph.get("status"),
        },
        "source_gate_results": source_gate_results,
        "formal_guards": guards,
        "research_cloud_gate": research_gate,
        "formal_cloud_unblocked": False,
        "formal_cloud_actual_guard_allowed": formal_cloud_unblocked,
        "dispatch_commands": dispatch_commands,
        "research_cloud_constraints": {
            "allowed_root": "output/surface_research_cloud_preflight",
            "max_hours_in_commands": 1.0,
            "post_job_referee_required": True,
            "forbidden_outputs": [
                "predictions.npz",
                "teacher package",
                "candidate package",
                "strict registry write",
                "strict pass write",
            ],
            "formal_modal_region_training_allowed_by_this_report": False,
        },
        "body_head_face_ready": body_head_face_ready,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_strict_pass_write": True,
        "decision": decision,
        "blockers": blockers,
    }
    write_json(args.output_json, summary)
    write_markdown(args.output_md, summary)
    print(json.dumps(json_ready({"status": status, "output": args.output_json, "body_head_face_ready": body_head_face_ready}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
