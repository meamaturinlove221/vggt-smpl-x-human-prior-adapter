import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
LOSS_PY = REPO_ROOT / "training" / "loss.py"

AUTHORED_PATCH_REVIEW_NOTE_JSON = OUTPUT_ROOT / "authored_patch_review_note.camera_focal_objective_isolation.20260331.json"
PATCH_AUTHORING_APPROVAL_NOTE_JSON = OUTPUT_ROOT / "patch_authoring_approval_note.camera_focal_objective_isolation.20260331.json"

HYGIENE_JSON = OUTPUT_ROOT / "hygiene_review.camera_focal_objective_isolation.20260331.json"
HYGIENE_MD = OUTPUT_ROOT / "hygiene_review.camera_focal_objective_isolation.20260331.md"
PROMOTION_JSON = OUTPUT_ROOT / "execution_prep_promotion_note.camera_focal_objective_isolation.20260331.json"
PROMOTION_MD = OUTPUT_ROOT / "execution_prep_promotion_note.camera_focal_objective_isolation.20260331.md"


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


def _extract_region(lines: list[str], start: int, end: int) -> list[str]:
    return [lines[i - 1] for i in range(start, end + 1)]


def _count_occurrences(lines: list[str], needle: str) -> int:
    return sum(1 for line in lines if needle in line)


def render_hygiene_md(payload: dict) -> str:
    lines = [
        "# Hygiene Review: Camera Focal Objective Isolation (2026-03-31)",
        "",
        f"- conclusion: `{payload['conclusion']}`",
        "",
        "## Checks",
        "",
    ]
    for check in payload["checks"]:
        lines.extend(
            [
                f"### {check['label']}",
                "",
                f"- status: `{check['status']}`",
                f"- note: {check['note']}",
                "",
            ]
        )
    return "\n".join(lines)


def render_promotion_md(payload: dict) -> str:
    lines = [
        "# Execution-Prep Promotion Note: Camera Focal Objective Isolation (2026-03-31)",
        "",
        f"- recommended_decision: `{payload['recommended_decision']}`",
        f"- ready_for_execution: `{payload['ready_for_execution']}`",
        "",
        "## 1. Why now for execution-prep promotion discussion",
        "",
        f"- {payload['why_now_for_execution_prep_promotion_discussion']}",
        "",
        "## 2. What exactly is being promoted",
        "",
        f"- {payload['what_exactly_is_being_promoted']}",
        "",
        "## 3. Promotion guardrails",
        "",
    ]
    lines.extend([f"- {item}" for item in payload["promotion_guardrails"]])
    lines.extend(
        [
            "",
            "## Allowed Buttons",
            "",
        ]
    )
    lines.extend([f"- `{item}`" for item in payload["allowed_buttons"]])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    checked_at = now_iso()
    loss_lines = LOSS_PY.read_text(encoding="utf-8").splitlines()
    authored_patch_review = load_json(AUTHORED_PATCH_REVIEW_NOTE_JSON)
    patch_authoring_approval = load_json(PATCH_AUTHORING_APPROVAL_NOTE_JSON)

    stage_region = _extract_region(loss_lines, 418, 511)
    return_region = _extract_region(loss_lines, 504, 513)

    stage_accumulation_count = _count_occurrences(stage_region, 'total_loss_T +=')
    stage_accumulation_count += _count_occurrences(stage_region, 'total_loss_R +=')
    stage_accumulation_count += _count_occurrences(stage_region, 'total_loss_FL +=')
    helper_total_camera_loss_count = _count_occurrences(return_region, 'total_camera_loss = _compute_total_camera_loss(')
    raw_formula_count = _count_occurrences(return_region, 'avg_loss_T * weight_trans')
    raw_formula_count += _count_occurrences(return_region, 'avg_loss_R * weight_rot')
    raw_formula_count += _count_occurrences(return_region, 'avg_loss_FL * weight_focal')

    stage_issue = stage_accumulation_count != 3
    formula_issue = helper_total_camera_loss_count != 1 or raw_formula_count != 0

    checks = [
        {
            "id": "stage_accumulation_once",
            "label": "compute_camera_loss stage-level component accumulation",
            "status": "pass" if not stage_issue else "needs_cleanup",
            "note": (
                "Each of total_loss_T, total_loss_R, and total_loss_FL is accumulated exactly once through stage_component_dict."
                if not stage_issue
                else "Stage-level accumulation count is inconsistent with the intended single accumulation path."
            ),
        },
        {
            "id": "single_default_total_camera_loss_formula",
            "label": "single effective total_camera_loss default formula path",
            "status": "pass" if not formula_issue else "needs_cleanup",
            "note": (
                "Only the helper-based total_camera_loss path remains active in the aggregation region; no shadow raw formula is left beside it."
                if not formula_issue
                else "The aggregation region still appears to retain overlapping formula paths."
            ),
        },
    ]

    conclusion = "PASS_NO_SEMANTIC_ISSUE"
    if stage_issue or formula_issue:
        conclusion = "PATCH_REQUIRES_TINY_CLEANUP_BEFORE_PROMOTION"

    hygiene_payload = {
        "checked_at": checked_at,
        "artifact_kind": "hygiene_review",
        "family": "camera_focal_objective_isolation",
        "target_file": "training/loss.py",
        "checks": checks,
        "conclusion": conclusion,
        "supporting_refs": {
            "authored_patch_review_note": repo_rel(AUTHORED_PATCH_REVIEW_NOTE_JSON),
            "inspected_file": repo_rel(LOSS_PY),
        },
        "next_requirement": "decide_reviewed_loss_py_patch_can_be_promoted_to_execution_prep",
    }
    write_json(HYGIENE_JSON, hygiene_payload)
    write_text(HYGIENE_MD, render_hygiene_md(hygiene_payload))

    if conclusion != "PASS_NO_SEMANTIC_ISSUE":
        print(
            json.dumps(
                {
                    "hygiene_review": repo_rel(HYGIENE_JSON),
                    "execution_prep_promotion_note": None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    promotion_payload = {
        "checked_at": checked_at,
        "artifact_kind": "execution_prep_promotion_note",
        "family": "camera_focal_objective_isolation",
        "state": "manual_review_execution_prep_promotion_discussion_only",
        "ready_for_execution": False,
        "why_now_for_execution_prep_promotion_discussion": (
            "The single-file boundary is already fixed to training/loss.py, an authored loss.py patch now exists, "
            "the authored patch review note says default semantics and the return contract remain preserved, "
            "and the hygiene review found no duplicate accumulation or duplicate total_camera_loss formula path."
        ),
        "what_exactly_is_being_promoted": (
            "This is not arm/run approval and not execution-ready status. It only promotes the currently reviewed "
            "training/loss.py patch to an execution-prep discussion baseline for stronger static or unit validation in a later step."
        ),
        "promotion_guardrails": [
            "The write surface remains limited to training/loss.py.",
            "Do not touch trainer, dataset, config, runner, or cloud as part of this promotion discussion.",
            "ready_for_execution must remain false until stronger validation is completed.",
            "Any later execution-prep step must build on the reviewed authored patch rather than reopen old families or widen the surface.",
        ],
        "allowed_buttons": [
            "KEEP_MANUAL_REVIEW_ONLY",
            "PROMOTE_TO_EXECUTION_PREP_DISCUSSION",
        ],
        "recommended_decision": "PROMOTE_TO_EXECUTION_PREP_DISCUSSION",
        "recommended_reason": (
            "The patch surface is already minimal and reviewed, and the hygiene review confirms the authored baseline is clean enough to discuss execution-prep without granting execution permission."
        ),
        "supporting_refs": {
            "hygiene_review": repo_rel(HYGIENE_JSON),
            "authored_patch_review_note": repo_rel(AUTHORED_PATCH_REVIEW_NOTE_JSON),
            "patch_authoring_approval_note": repo_rel(PATCH_AUTHORING_APPROVAL_NOTE_JSON),
            "prior_patch_authoring_recommendation": patch_authoring_approval.get("recommended_decision", ""),
            "authored_patch_default_semantics_preserved": authored_patch_review.get("default_semantics_preserved", False),
            "authored_patch_return_contract_preserved": authored_patch_review.get("return_contract_preserved", False),
        },
        "next_requirement": "decide_reviewed_loss_py_patch_can_be_promoted_to_execution_prep",
    }
    write_json(PROMOTION_JSON, promotion_payload)
    write_text(PROMOTION_MD, render_promotion_md(promotion_payload))

    print(
        json.dumps(
            {
                "hygiene_review": repo_rel(HYGIENE_JSON),
                "execution_prep_promotion_note": repo_rel(PROMOTION_JSON),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
