import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

VALIDATION_JSON = OUTPUT_ROOT / "execution_prep_baseline_validation.camera_focal_objective_isolation.20260331.json"
PROMOTION_NOTE_JSON = OUTPUT_ROOT / "execution_prep_promotion_note.camera_focal_objective_isolation.20260331.json"
AUTHORED_PATCH_REVIEW_JSON = OUTPUT_ROOT / "authored_patch_review_note.camera_focal_objective_isolation.20260331.json"
HYGIENE_REVIEW_JSON = OUTPUT_ROOT / "hygiene_review.camera_focal_objective_isolation.20260331.json"

DECISION_JSON = OUTPUT_ROOT / "execution_prep_baseline_decision.camera_focal_objective_isolation.20260331.json"
DECISION_MD = OUTPUT_ROOT / "execution_prep_baseline_decision.camera_focal_objective_isolation.20260331.md"


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
        "# Execution-Prep Baseline Decision: Camera Focal Objective Isolation (2026-03-31)",
        "",
        f"- decision: `{payload['decision']}`",
        f"- target_file: `{payload['target_file']}`",
        f"- target_object: `{payload['target_object']}`",
        f"- ready_for_execution: `{payload['ready_for_execution']}`",
        f"- do_not_arm_now: `{payload['do_not_arm_now']}`",
        f"- do_not_run_candidate_now: `{payload['do_not_run_candidate_now']}`",
        f"- cloud_must_remain_off: `{payload['cloud_must_remain_off']}`",
        "",
        "## Decision Scope",
        "",
        f"- {payload['decision_scope']}",
        "",
        "## Summary",
        "",
        f"- {payload['summary']}",
        "",
        "## Supporting Refs",
        "",
    ]
    for key, value in payload["supporting_refs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Next Requirement", "", f"- `{payload['next_requirement']}`", ""])
    return "\n".join(lines)


def main() -> int:
    validation = load_json(VALIDATION_JSON)
    promotion_note = load_json(PROMOTION_NOTE_JSON)
    authored_patch_review = load_json(AUTHORED_PATCH_REVIEW_JSON)
    hygiene_review = load_json(HYGIENE_REVIEW_JSON)

    payload = {
        "checked_at": now_iso(),
        "artifact_kind": "execution_prep_baseline_decision",
        "family": "camera_focal_objective_isolation",
        "decision": "PROMOTE_TO_EXECUTION_PREP_BASELINE",
        "target_file": "training/loss.py",
        "target_object": "loss_FL inside compute_camera_loss / camera loss aggregation",
        "ready_for_execution": False,
        "do_not_arm_now": True,
        "do_not_run_candidate_now": True,
        "cloud_must_remain_off": True,
        "decision_scope": (
            "This promotion only recognizes the reviewed and locally validated training/loss.py patch as the execution-prep baseline. "
            "It does not grant execution-ready status, does not authorize arm/run, and does not permit cloud use."
        ),
        "summary": (
            "The stronger local validation passed, the hygiene review stayed clean, and the authored patch still preserves the default "
            "camera-loss contract. Therefore the current training/loss.py-only patch is approved as the execution-prep baseline while the workflow remains manual-review-only."
        ),
        "supporting_refs": {
            "execution_prep_baseline_validation": repo_rel(VALIDATION_JSON),
            "execution_prep_promotion_note": repo_rel(PROMOTION_NOTE_JSON),
            "authored_patch_review_note": repo_rel(AUTHORED_PATCH_REVIEW_JSON),
            "hygiene_review": repo_rel(HYGIENE_REVIEW_JSON),
            "validation_status": validation.get("overall_status", ""),
            "promotion_recommendation": promotion_note.get("recommended_decision", ""),
            "authored_patch_default_semantics_preserved": authored_patch_review.get("default_semantics_preserved"),
            "hygiene_review_status": hygiene_review.get("conclusion", hygiene_review.get("overall_status", "")),
        },
        "next_requirement": "design_execution_ready_boundary_for_validated_loss_py_patch",
    }

    write_json(DECISION_JSON, payload)
    write_text(DECISION_MD, render_md(payload))

    print(json.dumps({"execution_prep_baseline_decision": repo_rel(DECISION_JSON)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
