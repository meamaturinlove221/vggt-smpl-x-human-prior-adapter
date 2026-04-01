import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

PREPARATION_DESIGN_JSON = OUTPUT_ROOT / "execution_ready_preparation_design.camera_focal_objective_isolation.20260331.json"
DISCUSSION_DECISION_JSON = OUTPUT_ROOT / "execution_ready_discussion_decision.camera_focal_objective_isolation.20260331.json"
STATUS_DISCUSSION_JSON = OUTPUT_ROOT / "execution_ready_status_discussion_decision.camera_focal_objective_isolation.20260401.json"
STATUS_DISCUSSION_MD = OUTPUT_ROOT / "execution_ready_status_discussion_decision.camera_focal_objective_isolation.20260401.md"

NEXT_REQUIREMENT = "manual_review_execution_ready_status_discussion_before_any_execution_ready_promotion"


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
        "# Execution-Ready Status Discussion Decision: Camera Focal Objective Isolation (2026-04-01)",
        "",
        f"- decision: `{payload['decision']}`",
        f"- ready_for_execution: `{payload['ready_for_execution']}`",
        f"- do_not_arm_now: `{payload['do_not_arm_now']}`",
        f"- do_not_run_candidate_now: `{payload['do_not_run_candidate_now']}`",
        f"- cloud_must_remain_off: `{payload['cloud_must_remain_off']}`",
        "",
        "## Decision Scope",
        "",
        f"- {payload['decision_scope']}",
        "",
        "## First Allowed Next Step",
        "",
        f"- {payload['first_allowed_next_step']}",
        "",
        "## Still Forbidden",
        "",
    ]
    for item in payload["still_forbidden"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Supporting Refs", ""])
    for key, value in payload["supporting_refs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Next Requirement", "", f"- `{payload['next_requirement']}`", ""])
    return "\n".join(lines)


def main() -> int:
    preparation_design = load_json(PREPARATION_DESIGN_JSON)
    discussion_decision = load_json(DISCUSSION_DECISION_JSON)

    payload = {
        "checked_at": now_iso(),
        "artifact_kind": "execution_ready_status_discussion_decision",
        "family": "camera_focal_objective_isolation",
        "decision": "PROMOTE_TO_EXECUTION_READY_STATUS_DISCUSSION",
        "decision_scope": (
            "This decision accepts the current execution-ready preparation design as the "
            "manual-review baseline and promotes the packet into execution-ready status "
            "discussion. It still does not grant execution-ready status and still does "
            "not authorize arm/run/training/cloud."
        ),
        "ready_for_execution": False,
        "do_not_arm_now": True,
        "do_not_run_candidate_now": True,
        "cloud_must_remain_off": True,
        "first_allowed_next_step": (
            "Conduct the bounded manual execution-ready status discussion using the "
            "existing training/loss.py-only packet and preparation design as the baseline, "
            "while still not executing anything."
        ),
        "still_forbidden": [
            "Do not mark ready_for_execution=true automatically.",
            "Do not arm any approved problem.",
            "Do not run candidate or training flows.",
            "Do not generate config or yaml.",
            "Do not use cloud or Modal.",
        ],
        "supporting_refs": {
            "execution_ready_preparation_design": repo_rel(PREPARATION_DESIGN_JSON),
            "execution_ready_discussion_decision": repo_rel(DISCUSSION_DECISION_JSON),
            "preparation_design_state": preparation_design.get("state", ""),
            "discussion_decision_value": discussion_decision.get("decision", ""),
            "preparation_design_next_requirement": preparation_design.get("next_requirement", ""),
        },
        "next_requirement": NEXT_REQUIREMENT,
    }

    write_json(STATUS_DISCUSSION_JSON, payload)
    write_text(STATUS_DISCUSSION_MD, render_md(payload))

    print(
        json.dumps(
            {"execution_ready_status_discussion_decision": repo_rel(STATUS_DISCUSSION_JSON)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
