from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from v10_surface_completion_pipeline import REPORTS, json_ready, read_summary, write_json, write_report


REPO_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def first_summary(*paths: Path) -> dict[str, Any]:
    for path in paths:
        data = read_summary(path)
        if data:
            return data
    return {}


def compact_source(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": data.get("status"),
        "decision": data.get("decision"),
        "strict_candidate_passes": data.get("strict_candidate_passes"),
        "strict_teacher_passes": data.get("strict_teacher_passes"),
    }


def root_dependency_dirs() -> list[str]:
    names = [
        "must3r-main",
        "MASt3R-SLAM-main",
        "2d-gaussian-splatting-main",
        "WiLoR",
        "HaMeR",
        "OSX",
        "SMPLer-X",
        "SMPLest-X",
        "HairGS",
        "GaussianHaircut",
    ]
    found: list[str] = []
    for name in names:
        path = Path("D:/") / name
        if path.exists():
            found.append(str(path))
    return found


def process_snapshot() -> list[dict[str, Any]]:
    ps = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { ($_.Name -match '^(python|python.exe|git|git.exe|modal|modal.exe|powershell|pwsh)$') "
        "-and ($_.CommandLine -match 'v14|VGGT|vggt|run_local_vggt|modal_4k4d|SMPLer|WiLoR|HaMeR|OSX|GaussianHaircut|sapiens|2dgs|hairgs|external_hand_hair') } | "
        "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Depth 4"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return [{"parse_error": proc.stdout[-1000:]}]
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Write V14 execution rollup.")
    parser.add_argument("--output-json", type=Path, default=REPORTS / "20260508_v14_execution_rollup.json")
    parser.add_argument("--output-md", type=Path, default=REPORTS / "20260508_v14_execution_rollup.md")
    args = parser.parse_args()

    sources = {
        "truth_lock": read_summary(REPORTS / "20260508_v14_truth_registry.json"),
        "k14": first_summary(
            REPORTS / "V14_K14/20260508_k14_kinect_teacher_gate_autopsy.json",
            REPORTS / "20260508_v14_kinect_teacher_gate_autopsy.json",
        ),
        "g14": read_summary(REPORTS / "20260508_v14_2dgs_protocol_alignment_audit.json"),
        "s14": read_summary(REPORTS / "20260508_v14_sapiens_normal_depth_qa.json"),
        "h14_r14": first_summary(
            REPORTS / "V14_H14_R14/readiness.json",
            REPORTS / "20260508_v14_external_hand_hair_asset_manager.json",
        ),
        "f14": read_summary(REPORTS / "20260508_v14_fus3d_region_backend_readiness.json"),
        "t14": read_summary(REPORTS / "20260508_v14_tmf_prediction_readiness.json"),
        "dline": read_summary(REPORTS / "20260508_v14_dline_promotion_report.json"),
        "cloud_gate_candidate": read_summary(REPORTS / "20260508_v14_cloud_gate_status.json"),
        "cloud_gate_teacher": read_summary(REPORTS / "20260508_v14_cloud_gate_teacher_status.json"),
    }

    dline = sources["dline"]
    strict_candidate_passes = int(dline.get("strict_candidate_passes", 0) or 0)
    strict_teacher_passes = int(dline.get("strict_teacher_passes", 0) or 0)
    gate_results = dline.get("gate_results", {}) if isinstance(dline.get("gate_results"), dict) else {}

    summary = {
        "task": "v14_execution_rollup",
        "created_utc": utc_now(),
        "status": "v14_terminal_state_no_strict_pass",
        "strict_candidate_passes": strict_candidate_passes,
        "strict_teacher_passes": strict_teacher_passes,
        "formal_cloud_unblocked": bool(dline.get("formal_cloud_unblocked")),
        "promotion_status": dline.get("status"),
        "promotion_blockers": dline.get("blockers", []),
        "key_outcomes": {
            "truth_lock_active": gate_results.get("truth_lock_active"),
            "k14_teacher_precheck_pass": gate_results.get("k14_teacher_precheck_pass"),
            "g14_teacher_precheck_pass": gate_results.get("g14_teacher_precheck_pass"),
            "s14_supervision_ready": gate_results.get("s14_supervision_ready"),
            "hand_ownership_ready": gate_results.get("hand_ownership_ready"),
            "hair_ownership_ready": gate_results.get("hair_ownership_ready"),
            "f14_body_head_face_ready": gate_results.get("f14_body_head_face_ready"),
            "t14_canonical_teacher_ready": gate_results.get("t14_canonical_teacher_ready"),
        },
        "local_root_dependencies_found": root_dependency_dirs(),
        "process_snapshot_after_cleanup": process_snapshot(),
        "artifacts": {
            "truth_registry": str((REPORTS / "20260508_v14_truth_registry.json").resolve()),
            "k14_report": str((REPORTS / "V14_K14/20260508_k14_kinect_teacher_gate_autopsy.json").resolve()),
            "g14_report": str((REPORTS / "20260508_v14_2dgs_protocol_alignment_audit.json").resolve()),
            "s14_report": str((REPORTS / "20260508_v14_sapiens_normal_depth_qa.json").resolve()),
            "h14_r14_report": str((REPORTS / "V14_H14_R14/readiness.json").resolve()),
            "f14_report": str((REPORTS / "20260508_v14_fus3d_region_backend_readiness.json").resolve()),
            "t14_report": str((REPORTS / "20260508_v14_tmf_prediction_readiness.json").resolve()),
            "dline_report": str((REPORTS / "20260508_v14_dline_promotion_report.json").resolve()),
            "cloud_gate_candidate": str((REPORTS / "20260508_v14_cloud_gate_status.json").resolve()),
            "cloud_gate_teacher": str((REPORTS / "20260508_v14_cloud_gate_teacher_status.json").resolve()),
        },
        "sources": {name: compact_source(data) for name, data in sources.items()},
        "decision": (
            "V14 ran all available local/reportable branches to D-line. Promotion remains blocked because no strict "
            "K/G/T teacher precheck passed, no hand ownership asset exists, no hair topology ownership asset exists, "
            "and no unified candidate can legally be built from ownership-pass regions."
        ),
    }
    write_json(args.output_json, summary)
    write_report(args.output_md, "V14 Execution Rollup", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": args.output_json}), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
