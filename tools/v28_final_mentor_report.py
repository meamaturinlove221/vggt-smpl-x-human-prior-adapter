from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
OUT_MD = REPORTS / "20260508_v28_final_mentor_report.md"
OUT_JSON = REPORTS / "20260508_v28_final_mentor_report.json"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    gate = read_json(REPORTS / "20260508_v28_strict_teacher_candidate_gate.json")
    package = read_json(REPORTS / "20260508_v28_final_package_builder.json")
    summaries = {
        "V21": read_json(REPORTS / "20260508_v21_completion_contract.json").get("status"),
        "V22": read_json(REPORTS / "20260508_v22_true_vggt_smplx_microfit.json").get("final_status")
        or read_json(REPORTS / "20260508_v22_true_vggt_smplx_microfit.json").get("status"),
        "V23": read_json(REPORTS / "20260508_v23_residual_surface_v2.json").get("status"),
        "V24": read_json(REPORTS / "20260508_v24_residual_teacher_v2.json").get("status"),
        "V25": read_json(REPORTS / "20260508_v25_research_predictions_3frames.json").get("status"),
        "V26": read_json(REPORTS / "20260508_v26_temporal_canonical_teacher.json").get("status"),
        "V27": read_json(REPORTS / "20260508_v27_teacher_supervised_training.json").get("status"),
        "V28": gate.get("status"),
    }
    payload = {
        "task": "v28_final_mentor_report",
        "created_utc": utc_now(),
        "strict_teacher_passes": int(gate.get("strict_teacher_passes", 0)),
        "strict_candidate_passes": int(gate.get("strict_candidate_passes", 0)),
        "formal_cloud_unblocked": bool(gate.get("formal_cloud_unblocked", False)),
        "all_required_branches_executed": bool(gate.get("all_required_branches_executed", False)),
        "failure_proof_complete": bool(gate.get("failure_proof_complete", False)),
        "next_manual_mentor_decision_needed": bool(gate.get("next_manual_mentor_decision_needed", False)),
        "stage_statuses": summaries,
        "primary_blockers": gate.get("blockers", []),
        "key_artifacts": {
            "v24_teacher_targets": str((REPO_ROOT / "output" / "surface_research_preflight_local" / "V24_residual_teacher_v2" / "v24_residual_teacher_targets_v2.npz").resolve()),
            "v24_region_contact_sheet": str((REPO_ROOT / "output" / "surface_research_preflight_local" / "V24_residual_teacher_v2" / "v24_teacher_region_contact_sheet.png").resolve()),
            "v25_research_predictions": str((REPO_ROOT / "output" / "surface_research_cloud_preflight" / "V25_research_vggt_predictions").resolve()),
            "v26_temporal_targets": str((REPO_ROOT / "output" / "surface_research_preflight_local" / "V26_temporal_canonical_teacher" / "v26_temporal_canonical_teacher_targets.npz").resolve()),
            "failure_proof_pack": package.get("failure_proof_pack"),
        },
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# V28 Final Mentor Report",
        "",
        "## Final Gate Status",
        "",
        f"- strict_teacher_passes: `{payload['strict_teacher_passes']}`",
        f"- strict_candidate_passes: `{payload['strict_candidate_passes']}`",
        f"- formal_cloud_unblocked: `{payload['formal_cloud_unblocked']}`",
        f"- all_required_branches_executed: `{payload['all_required_branches_executed']}`",
        f"- failure_proof_complete: `{payload['failure_proof_complete']}`",
        "",
        "## Stage Statuses",
        "",
    ]
    for key, status in summaries.items():
        lines.append(f"- {key}: `{status}`")
    lines.extend(["", "## Primary Blockers", ""])
    lines.extend([f"- {b}" for b in payload["primary_blockers"]] or ["- none"])
    lines.extend(
        [
            "",
            "## Mentor Decision Needed",
            "",
            "The SMPL-X native route now has repaired head/face/hands residual evidence, residual teacher v2, three-frame research VGGT predictions, temporal canonical evidence, and a teacher-supervised research config. It still cannot be promoted because the Modal V25 base VGGT model has no `normal_head` and no human-prior channels, so the strict normal/prior route remains incomplete.",
            "",
            "## Key Artifacts",
            "",
        ]
    )
    for key, value in payload["key_artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "DONE_PASS", "json": str(OUT_JSON), "md": str(OUT_MD)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
