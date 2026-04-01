import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

STATUS_DISCUSSION_DECISION_JSON = OUTPUT_ROOT / "execution_ready_status_discussion_decision.camera_focal_objective_isolation.20260401.json"
STATUS_DISCUSSION_PACKET_JSON = OUTPUT_ROOT / "execution_ready_status_discussion_packet.camera_focal_objective_isolation.20260401.json"
BASELINE_VALIDATION_JSON = OUTPUT_ROOT / "execution_prep_baseline_validation.camera_focal_objective_isolation.20260331.json"
LOCALIZATION_JSON = OUTPUT_ROOT / "early_fl_tax_localization.20260330.json"
OBJECT_ALIGNMENT_JSON = OUTPUT_ROOT / "fl_tax_object_alignment_matrix.20260330.json"
ROOT_CAUSE_JSON = OUTPUT_ROOT / "objective_balance_root_cause_decision.20260330.json"

CANDIDATE_CONFIG = (
    "training/config/"
    "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_"
    "gradconfmask_lossflisolation0_minimal.yaml"
)
CANDIDATE_SHAPE = "stablelead_global_lossfl_isolation0"

DECISION_JSON = OUTPUT_ROOT / "execution_ready_promotion_decision.camera_focal_objective_isolation.20260401.json"
DECISION_MD = OUTPUT_ROOT / "execution_ready_promotion_decision.camera_focal_objective_isolation.20260401.md"

NEXT_REQUIREMENT = "manual_approval_to_arm_execution_ready_camera_focal_objective_isolation"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def render_md(payload: dict) -> str:
    lines = [
        "# Execution-Ready Promotion Decision: Camera Focal Objective Isolation (2026-04-01)",
        "",
        f"- decision: `{payload['decision']}`",
        f"- family: `{payload['family']}`",
        f"- ready_for_execution: `{payload['ready_for_execution']}`",
        f"- first_candidate_shape: `{payload['first_candidate_shape']}`",
        f"- first_candidate_config: `{payload['first_candidate_config']}`",
        f"- first_allowed_next_step: `{payload['first_allowed_next_step']}`",
        "",
        "## Promotion Scope",
        "",
        f"- {payload['promotion_scope']}",
        "",
        "## Why This Candidate",
        "",
    ]
    for item in payload["why_this_candidate"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Still Forbidden", ""])
    for item in payload["still_forbidden"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Supporting Refs", ""])
    for key, value in payload["supporting_refs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Next Requirement", "", f"- `{payload['next_requirement']}`", ""])
    return "\n".join(lines)


def main() -> int:
    status_discussion_decision = load_json(STATUS_DISCUSSION_DECISION_JSON)
    status_discussion_packet = load_json(STATUS_DISCUSSION_PACKET_JSON)
    baseline_validation = load_json(BASELINE_VALIDATION_JSON)
    localization = load_json(LOCALIZATION_JSON)
    object_alignment = load_json(OBJECT_ALIGNMENT_JSON)
    root_cause = load_json(ROOT_CAUSE_JSON)

    payload = {
        "checked_at": now_iso(),
        "artifact_kind": "execution_ready_promotion_decision",
        "family": "camera_focal_objective_isolation",
        "decision": "PROMOTE_TO_EXECUTION_READY",
        "ready_for_execution": True,
        "do_not_auto_open_ticket": True,
        "cloud_must_remain_off": True,
        "first_candidate_shape": CANDIDATE_SHAPE,
        "first_candidate_config": CANDIDATE_CONFIG,
        "promotion_scope": (
            "This decision promotes camera_focal_objective_isolation from execution-ready status discussion into "
            "execution-ready pending manual arm approval. It authorizes one concrete first candidate on the current "
            "repo but still does not auto-arm, auto-run, or allow cloud."
        ),
        "why_this_candidate": [
            "The status-discussion packet passed with one stable loss.py-only execution-prep baseline and no unresolved contract drift.",
            "The early FL-tax localization says the residual conflict is global, FL-dominant, and already visible by smoke val step 0.",
            "Default-only and hardtail-only focal scaling families failed to move that plateau, so the next honest first candidate is objective isolation itself rather than another stream-local focal tweak.",
            "The reviewed loss_fl_isolation_scale hook is already locally validated, and setting it to 0.0 gives the narrowest direct test of whether FL pressure inside loss_camera is the blocking objective term while still preserving loss_FL readout for review.",
        ],
        "first_allowed_next_step": (
            "Arm exactly one approved camera_focal_objective_isolation ticket with the prebuilt lossflisolation0 config, "
            "run it through the normal local smoke/10x5/100x20 gate path, and keep cloud off."
        ),
        "still_forbidden": [
            "Do not auto-arm an approved problem without an explicit manual approval step.",
            "Do not open a second candidate or cousin sweep.",
            "Do not use cloud or Modal.",
            "Do not reopen tail-contract derivative families or two-stage cousins.",
        ],
        "supporting_refs": {
            "execution_ready_status_discussion_decision": repo_rel(STATUS_DISCUSSION_DECISION_JSON),
            "execution_ready_status_discussion_packet": repo_rel(STATUS_DISCUSSION_PACKET_JSON),
            "execution_prep_baseline_validation": repo_rel(BASELINE_VALIDATION_JSON),
            "objective_balance_root_cause_decision": repo_rel(ROOT_CAUSE_JSON),
            "early_fl_tax_localization": repo_rel(LOCALIZATION_JSON),
            "fl_tax_object_alignment_matrix": repo_rel(OBJECT_ALIGNMENT_JSON),
            "status_discussion_decision_value": status_discussion_decision.get("decision", ""),
            "status_discussion_packet_state": status_discussion_packet.get("state", ""),
            "baseline_validation_status": baseline_validation.get("overall_status", ""),
            "dominant_component": localization.get("dominant_component", ""),
            "most_supported_scope": localization.get("most_supported_scope", ""),
            "root_cause_label": root_cause.get("label", ""),
            "default_stream_focal_family_delta_loss_camera": next(
                (
                    row.get("delta_loss_camera")
                    for row in object_alignment.get("rows", [])
                    if row.get("family") == "default_stream_intrinsics_counterbalance"
                ),
                None,
            ),
        },
        "next_requirement": NEXT_REQUIREMENT,
    }

    write_json(DECISION_JSON, payload)
    write_text(DECISION_MD, render_md(payload))

    print(
        json.dumps(
            {"execution_ready_promotion_decision": repo_rel(DECISION_JSON)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
