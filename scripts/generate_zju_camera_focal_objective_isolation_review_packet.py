import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

LOCALIZATION_JSON = OUTPUT_ROOT / "early_fl_tax_localization.20260330.json"
FAILURE_JSON = OUTPUT_ROOT / "two_stage_failure_interpretation.20260330.json"
SEED_JSON = OUTPUT_ROOT / "approved_problem.seed.camera_focal_objective_isolation.json"
BLUEPRINT_JSON = OUTPUT_ROOT / "family_blueprint.camera_focal_objective_isolation.json"
PLAN_JSON = OUTPUT_ROOT / "candidate_patch_plan.camera_focal_objective_isolation.json"
DRAFT_JSON = OUTPUT_ROOT / "next_manual_problem_draft.camera_focal_objective_isolation.20260330.json"

REVIEW_NOTE_JSON = OUTPUT_ROOT / "review_note.camera_focal_objective_isolation.20260330.json"
REVIEW_NOTE_MD = OUTPUT_ROOT / "review_note.camera_focal_objective_isolation.20260330.md"
CHECKLIST_JSON = OUTPUT_ROOT / "decision_checklist.camera_focal_objective_isolation.20260330.json"
CHECKLIST_MD = OUTPUT_ROOT / "decision_checklist.camera_focal_objective_isolation.20260330.md"


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


def render_review_md(payload: dict) -> str:
    lines = [
        "# Review Note: Camera Focal Objective Isolation (2026-03-30)",
        "",
        "## 1. Why Now",
        "",
    ]
    lines.extend([f"- {item}" for item in payload["why_now"]])
    lines.extend(
        [
            "",
            "## 2. What It Is Not",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in payload["what_it_is_not"]])
    lines.extend(
        [
            "",
            "## 3. Objective Target",
            "",
            f"- {payload['objective_target']}",
            "",
            "## 4. First Shape Scope",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in payload["first_shape_scope"]])
    lines.extend(
        [
            "",
            "## 5. Promotion Gate To Execution Prep",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in payload["promotion_gate_to_execution_prep"]])
    lines.append("")
    return "\n".join(lines)


def render_checklist_md(payload: dict) -> str:
    lines = [
        "# Decision Checklist: Camera Focal Objective Isolation (2026-03-30)",
        "",
    ]
    for item in payload["checks"]:
        status = "x" if item["satisfied"] else " "
        lines.append(f"- [{status}] {item['label']}")
        lines.append(f"  - status: `{item['status']}`")
        lines.append(f"  - note: {item['note']}")
    lines.extend(
        [
            "",
            "## Allowed Decision Buttons",
            "",
        ]
    )
    lines.extend([f"- `{item}`" for item in payload["allowed_decision_buttons"]])
    return "\n".join(lines) + "\n"


def main() -> int:
    checked_at = now_iso()
    localization = load_json(LOCALIZATION_JSON)
    failure = load_json(FAILURE_JSON)
    seed = load_json(SEED_JSON)
    blueprint = load_json(BLUEPRINT_JSON)
    plan = load_json(PLAN_JSON)
    draft = load_json(DRAFT_JSON)

    review_note = {
        "checked_at": checked_at,
        "artifact_kind": "manual_review_note",
        "family": "camera_focal_objective_isolation",
        "why_now": [
            f"dominant_component = {localization.get('dominant_component', '')}",
            f"appears_by = {localization.get('appears_by', '')}",
            f"most_supported_scope = {localization.get('most_supported_scope', '')}",
            f"{failure.get('conclusion', '')}",
        ],
        "what_it_is_not": [
            "not tail-contract derivative reopen",
            "not two-stage retry",
            "not stream-local focal tweak",
            "not execution-ready ticket",
        ],
        "objective_target": (
            "Localize how the FL-dominant camera tax is already bound to the depth-win regime by step0, rather than ranking another family candidate."
        ),
        "first_shape_scope": [
            f"first shape fixed as {seed.get('first_candidate_shape', '')}",
            "scope limited to audit / logging / bounded objective-isolation design",
            "do not materialize candidate config today",
        ],
        "promotion_gate_to_execution_prep": [
            "Only promote if review identifies one single minimal write surface.",
            "Only promote if that write surface isolates an objective object rather than reopening an old family.",
            "Only promote to execution-prep design; do not arm, do not run, and do not touch cloud from this packet.",
        ],
        "supporting_refs": {
            "localization": str(LOCALIZATION_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
            "failure": str(FAILURE_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
            "seed": str(SEED_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
            "blueprint": str(BLUEPRINT_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
            "candidate_patch_plan": str(PLAN_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
            "draft": str(DRAFT_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        },
    }

    checks = [
        {
            "id": "dominant_component_loss_fl",
            "label": "dominant_component still loss_FL",
            "status": "supported",
            "satisfied": localization.get("dominant_component") == "loss_FL",
            "note": f"localization reports dominant_component={localization.get('dominant_component', '')}",
        },
        {
            "id": "appears_by_smoke_val_step0",
            "label": "appears_by still smoke_val_step0",
            "status": "supported",
            "satisfied": localization.get("appears_by") == "smoke_val_step0",
            "note": f"localization reports appears_by={localization.get('appears_by', '')}",
        },
        {
            "id": "scope_global",
            "label": "scope still global",
            "status": "supported",
            "satisfied": localization.get("most_supported_scope") == "global",
            "note": f"localization reports most_supported_scope={localization.get('most_supported_scope', '')}",
        },
        {
            "id": "scalar_schedule_insufficient",
            "label": "scalar schedule insufficient still supported",
            "status": "supported",
            "satisfied": failure.get("conclusion") == "SCALAR_SCHEDULE_INSUFFICIENT",
            "note": f"failure interpretation concludes {failure.get('conclusion', '')}",
        },
        {
            "id": "no_tail_contract_reopen",
            "label": "no tail-contract derivative reopen",
            "status": "supported",
            "satisfied": bool(seed.get("mutation_dsl", {}).get("disallow_tail_contract_derivative_reopen", False)),
            "note": "seed explicitly disallows tail-contract derivative reopen",
        },
        {
            "id": "first_shape_fixed",
            "label": "first shape fixed as fl_only_camera_objective_isolation_audit",
            "status": "supported",
            "satisfied": seed.get("first_candidate_shape") == "fl_only_camera_objective_isolation_audit",
            "note": f"seed/draft/plan all point to {seed.get('first_candidate_shape', '')}",
        },
        {
            "id": "no_arm_run_cloud",
            "label": "no arm/run/cloud",
            "status": "supported",
            "satisfied": (
                bool(seed.get("mutation_dsl", {}).get("disallow_arm_now", False))
                and bool(seed.get("mutation_dsl", {}).get("disallow_run_now", False))
                and bool(seed.get("mutation_dsl", {}).get("disallow_cloud", False))
                and bool(plan.get("do_not_arm_now"))
                and bool(plan.get("do_not_run_candidate_now"))
            ),
            "note": "seed and candidate patch plan both keep arm/run/cloud disabled",
        },
        {
            "id": "execution_prep_gate_only",
            "label": "enough evidence to promote only to execution-prep design",
            "status": "manual_judgment_required",
            "satisfied": False,
            "note": "diagnostic evidence is strong, but the packet still lacks a single agreed minimal write surface, so keep manual review only for now",
        },
    ]

    checklist = {
        "checked_at": checked_at,
        "artifact_kind": "manual_review_decision_checklist",
        "family": "camera_focal_objective_isolation",
        "checks": checks,
        "allowed_decision_buttons": [
            "KEEP_MANUAL_REVIEW_ONLY",
            "PROMOTE_TO_EXECUTION_PREP_DESIGN",
        ],
        "recommended_decision": "KEEP_MANUAL_REVIEW_ONLY",
        "recommended_decision_reason": (
            "The packet strongly supports the problem choice, but it does not yet identify one single minimal write surface honest enough for execution-prep."
        ),
        "execution_guardrails": {
            "ready_for_execution_must_remain_false_today": not blueprint.get("ready_for_execution", True) and not draft.get("ready_for_execution", True),
            "do_not_arm_now": bool(plan.get("do_not_arm_now")),
            "do_not_run_candidate_now": bool(plan.get("do_not_run_candidate_now")),
            "cloud_must_remain_off": bool(seed.get("cloud_must_remain_off", False)),
        },
    }

    write_json(REVIEW_NOTE_JSON, review_note)
    write_text(REVIEW_NOTE_MD, render_review_md(review_note))
    write_json(CHECKLIST_JSON, checklist)
    write_text(CHECKLIST_MD, render_checklist_md(checklist))

    print(json.dumps({
        "review_note": str(REVIEW_NOTE_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        "decision_checklist": str(CHECKLIST_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
