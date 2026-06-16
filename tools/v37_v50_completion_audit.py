from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "reports" / "20260509_v37_v50_completion_audit.json"
REPORT_MD = ROOT / "reports" / "20260509_v37_v50_completion_audit.md"
LOCAL = ROOT / "output" / "surface_research_preflight_local"
CLOUD = ROOT / "output" / "surface_research_cloud_preflight"


STAGES = {
    "v37": {
        "report": ROOT / "reports/20260509_v37_checkpoint_forensic_audit.json",
        "allowed": {"DONE_PASS", "DONE_FAIL_ROUTED", "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE"},
    },
    "v38": {
        "report": ROOT / "reports/20260509_v38_prior_enabled_checkpoint_scaffold.json",
        "allowed": {"DONE_PASS"},
    },
    "v39": {
        "report": ROOT / "reports/20260509_v39_adapter_microfit.json",
        "allowed": {"DONE_PASS", "DONE_FAIL_ROUTED"},
    },
    "v40": {
        "report": ROOT / "reports/20260509_v40_region_balanced_adapter_rescue.json",
        "allowed": {"DONE_PASS", "DONE_FAIL_ROUTED", "NOT_REQUIRED"},
    },
    "v41": {
        "report": ROOT / "reports/20260509_v41_prior_sensitive_head_unfreeze.json",
        "allowed": {"DONE_PASS", "DONE_FAIL_ROUTED", "NOT_REQUIRED"},
    },
    "v41b": {
        "report": ROOT / "reports/20260509_v41b_real_prior_enabled_checkpoint_builder.json",
        "allowed": {"DONE_PASS", "DONE_FAIL_ROUTED", "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE"},
    },
    "v42": {
        "report": ROOT / "reports/20260509_v42_prior_enabled_predictions_rerun.json",
        "allowed": {"DONE_PASS", "DONE_FAIL_ROUTED", "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE"},
    },
    "v42_cloud_attempt": {
        "report": ROOT / "reports/20260509_v42_cloud_attempt_audit.json",
        "allowed": {"DONE_PASS", "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE"},
    },
    "v43": {
        "report": ROOT / "reports/20260509_v43_replay_with_prior_enabled_predictions.json",
        "allowed": {"DONE_PASS", "DONE_FAIL_ROUTED"},
    },
    "v44": {
        "report": ROOT / "reports/20260509_v44_strict_visual_pre_promotion_gate.json",
        "allowed": {"DONE_PASS", "DONE_FAIL_ROUTED"},
    },
    "v45": {
        "report": ROOT / "reports/20260509_v45_head_face_correction.json",
        "allowed": {"DONE_PASS", "DONE_FAIL_ROUTED"},
    },
    "v46": {
        "report": ROOT / "reports/20260509_v46_hand_correction.json",
        "allowed": {"DONE_PASS", "DONE_FAIL_ROUTED"},
    },
    "v47": {
        "report": ROOT / "reports/20260509_v47_60view_correction.json",
        "allowed": {"DONE_PASS", "DONE_FAIL_ROUTED"},
    },
    "v48": {
        "report": ROOT / "reports/20260509_v48_temporal_correction.json",
        "allowed": {"DONE_PASS", "DONE_FAIL_ROUTED"},
    },
    "v49": {
        "report": ROOT / "reports/20260509_v49_package_dry_run.json",
        "allowed": {"DONE_PASS", "DONE_FAIL_ROUTED"},
    },
    "v50": {
        "report": ROOT / "reports/20260509_v50_final_promotion_transaction.json",
        "allowed": {"DONE_PASS", "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE"},
    },
    "v44_v50_audit": {
        "report": ROOT / "reports/20260509_v44_v50_completion_audit.json",
        "allowed": {"COMPLETE_AUDIT_PASS"},
    },
}

FORBIDDEN_FILENAMES = {"predictions.npz"}
FORBIDDEN_PATH_TOKENS = ("formal_candidate", "candidate_package_v50", "teacher_package_v50", "strict_registry_entry_v50")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def scan_forbidden() -> list[str]:
    hits: list[str] = []
    for root in (LOCAL, CLOUD):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            lower = path.as_posix().lower()
            if path.name.lower() in FORBIDDEN_FILENAMES or any(token in lower for token in FORBIDDEN_PATH_TOKENS):
                hits.append(str(path.resolve()))
    return hits


def main() -> None:
    rows: dict[str, Any] = {}
    blockers: list[str] = []
    for name, spec in STAGES.items():
        path = spec["report"]
        data = read_json(path)
        status = data.get("status", data.get("final_status", "MISSING"))
        ok = path.is_file() and status in spec["allowed"]
        if not ok:
            blockers.append(f"{name} missing or status not allowed: {status}")
        rows[name] = {
            "report": str(path.resolve()),
            "report_exists": path.is_file(),
            "status": status,
            "allowed": sorted(spec["allowed"]),
            "ok": ok,
        }
    v50 = read_json(ROOT / "reports/20260509_v50_final_promotion_transaction.json")
    v42 = read_json(ROOT / "reports/20260509_v42_prior_enabled_predictions_rerun.json")
    forbidden = scan_forbidden()
    if forbidden:
        blockers.append(f"forbidden outputs detected: {len(forbidden)}")
    all_executed = not blockers
    summary = {
        "task": "v37_v50_completion_audit",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "COMPLETE_AUDIT_PASS" if all_executed else "COMPLETE_AUDIT_FAIL",
        "all_required_branches_executed": all_executed,
        "stage_rows": rows,
        "forbidden_hit_count": len(forbidden),
        "forbidden_hits": forbidden,
        "strict_candidate_passes": int(v50.get("strict_candidate_passes", 0) or 0),
        "strict_teacher_passes": int(v50.get("strict_teacher_passes", 0) or 0),
        "formal_cloud_unblocked": bool(v50.get("formal_cloud_unblocked")),
        "remaining_blockers": v50.get("remaining_blockers", []),
        "v42_status": v42.get("status"),
        "v42_blockers": v42.get("blockers", []),
        "completion_result": (
            "strict_pass" if int(v50.get("strict_candidate_passes", 0) or 0) > 0 or int(v50.get("strict_teacher_passes", 0) or 0) > 0
            else "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE"
        ),
        "blockers": blockers,
    }
    write_json(REPORT_JSON, summary)
    lines = [
        "# V37-V50 Completion Audit",
        "",
        f"Status: `{summary['status']}`",
        f"completion_result: `{summary['completion_result']}`",
        f"strict_candidate_passes: `{summary['strict_candidate_passes']}`",
        f"strict_teacher_passes: `{summary['strict_teacher_passes']}`",
        f"forbidden_hit_count: `{summary['forbidden_hit_count']}`",
        "",
        "## Remaining Blockers",
        "",
    ]
    lines.extend([f"- {item}" for item in (summary["remaining_blockers"] or summary["v42_blockers"] or ["none"])])
    REPORT_MD.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(summary["status"])


if __name__ == "__main__":
    main()
