import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

APPROVAL_NOTE_JSON = OUTPUT_ROOT / "execution_ready_discussion_approval_note.camera_focal_objective_isolation.20260331.json"
GATE_JSON = OUTPUT_ROOT / "execution_ready_gate.camera_focal_objective_isolation.20260331.json"
DECISION_JSON = OUTPUT_ROOT / "execution_ready_discussion_decision.camera_focal_objective_isolation.20260331.json"
DECISION_MD = OUTPUT_ROOT / "execution_ready_discussion_decision.camera_focal_objective_isolation.20260331.md"

NEXT_REQUIREMENT = "design_execution_ready_preparation_design_for_validated_loss_py_patch"


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
        "# Execution-Ready Discussion Decision: Camera Focal Objective Isolation (2026-03-31)",
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
        "## Supporting Refs",
        "",
    ]
    for key, value in payload["supporting_refs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Next Requirement", "", f"- `{payload['next_requirement']}`", ""])
    return "\n".join(lines)


def main() -> int:
    approval_note = load_json(APPROVAL_NOTE_JSON)
    gate = load_json(GATE_JSON)

    payload = {
        "checked_at": now_iso(),
        "artifact_kind": "execution_ready_discussion_decision",
        "family": "camera_focal_objective_isolation",
        "decision": "PROMOTE_TO_EXECUTION_READY_DISCUSSION",
        "decision_scope": (
            "This decision promotes the validated loss.py-only packet into execution-ready discussion. "
            "It does not grant execution-ready status, does not authorize config materialization, and does not permit arm/run/training/cloud."
        ),
        "ready_for_execution": False,
        "do_not_arm_now": True,
        "do_not_run_candidate_now": True,
        "cloud_must_remain_off": True,
        "first_allowed_next_step": (
            "Design an execution-ready preparation design that defines the minimal local-only preparation boundary, keeps all checks non-training, and still does not execute anything."
        ),
        "supporting_refs": {
            "execution_ready_discussion_approval_note": repo_rel(APPROVAL_NOTE_JSON),
            "execution_ready_gate": repo_rel(GATE_JSON),
            "approval_note_recommended_decision": approval_note.get("recommended_decision", ""),
            "gate_recommended_decision": gate.get("recommended_decision", ""),
        },
        "next_requirement": NEXT_REQUIREMENT,
    }

    write_json(DECISION_JSON, payload)
    write_text(DECISION_MD, render_md(payload))

    print(json.dumps({"execution_ready_discussion_decision": repo_rel(DECISION_JSON)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
