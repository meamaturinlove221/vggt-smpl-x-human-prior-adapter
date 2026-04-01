import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

DESIGN_CONTRACT_JSON = OUTPUT_ROOT / "design_contract.camera_focal_objective_isolation.20260330.json"
PATCH_BOUNDARY_JSON = OUTPUT_ROOT / "patch_boundary_note.camera_focal_objective_isolation.20260330.json"
IMPLEMENTATION_SKETCH_JSON = OUTPUT_ROOT / "execution_prep_implementation_sketch.camera_focal_objective_isolation.20260331.json"
PSEUDODIFF_MAP_JSON = OUTPUT_ROOT / "pseudodiff_map.camera_focal_objective_isolation.20260331.json"
PATCH_AUTHORING_GATE_JSON = OUTPUT_ROOT / "patch_authoring_gate.camera_focal_objective_isolation.20260331.json"

APPROVAL_NOTE_JSON = OUTPUT_ROOT / "patch_authoring_approval_note.camera_focal_objective_isolation.20260331.json"
APPROVAL_NOTE_MD = OUTPUT_ROOT / "patch_authoring_approval_note.camera_focal_objective_isolation.20260331.md"


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
        "# Patch Authoring Approval Note: Camera Focal Objective Isolation (2026-03-31)",
        "",
        f"- recommended_decision: `{payload['recommended_decision']}`",
        f"- ready_for_execution: `{payload['ready_for_execution']}`",
        "",
        "## 1. Why now for patch-authoring decision",
        "",
        f"- {payload['why_now_for_patch_authoring_decision']}",
        "",
        "## 2. What exactly is being approved",
        "",
        f"- {payload['what_exactly_is_being_approved']}",
        "",
        "## 3. Allowed patch boundary after approval",
        "",
    ]
    lines.extend([f"- {item}" for item in payload["allowed_patch_boundary_after_approval"]])
    lines.extend(
        [
            "",
            "## 4. First patch acceptance criteria",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in payload["first_patch_acceptance_criteria"]])
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
    design_contract = load_json(DESIGN_CONTRACT_JSON)
    patch_boundary = load_json(PATCH_BOUNDARY_JSON)
    implementation_sketch = load_json(IMPLEMENTATION_SKETCH_JSON)
    pseudodiff_map = load_json(PSEUDODIFF_MAP_JSON)
    patch_gate = load_json(PATCH_AUTHORING_GATE_JSON)

    payload = {
        "checked_at": checked_at,
        "artifact_kind": "patch_authoring_approval_note",
        "family": "camera_focal_objective_isolation",
        "state": "manual_review_patch_authoring_decision_only",
        "ready_for_execution": False,
        "why_now_for_patch_authoring_decision": (
            "Two-stage objective decoupling is already closed as dead_same_day, the early FL tax has already been localized, "
            "the design contract already narrowed the write surface to training/loss.py only, and the pseudodiff map now shows "
            "that every planned first-patch block remains inside that one file with preserved default semantics."
        ),
        "what_exactly_is_being_approved": (
            "This does not approve execution-ready status, candidate config generation, arm/run, or cloud usage. "
            "It only approves authoring one training/loss.py-only patch for loss_FL isolation design, still under manual review governance."
        ),
        "allowed_patch_boundary_after_approval": [
            "Only training/loss.py may be edited.",
            "Only the compute_camera_loss / loss_FL / camera component assembly path may be touched.",
            "The first authored patch must not rewrite MultitaskLoss.forward.",
            "The first authored patch must not touch training/trainer.py, training/data/*, training/config/*, runner scripts, or cloud paths.",
            "The first authored patch must remain an objective-isolation patch and must not reopen tail-contract or two-stage family variants.",
        ],
        "first_patch_acceptance_criteria": [
            "Default numeric semantics remain unchanged when any new local hook is inactive.",
            "The return contract for loss_camera, loss_T, loss_R, and loss_FL remains unchanged.",
            "No new external dependency, config materialization, or batch-field requirement is introduced.",
            "The patch remains describable as objective isolation inside training/loss.py rather than a reopen of an older family axis.",
        ],
        "allowed_buttons": [
            "KEEP_MANUAL_REVIEW_ONLY",
            "PROMOTE_TO_PATCH_AUTHORING",
        ],
        "recommended_decision": "PROMOTE_TO_PATCH_AUTHORING",
        "recommended_reason": (
            "The manual packet has now reached the point where the first authored surface is narrowly and honestly defined: "
            "one file, one object boundary, one preserved return contract, and no requirement to enable execution. "
            "That is sufficient to authorize patch authoring while still keeping ready_for_execution=false and all arm/run/cloud paths disabled."
        ),
        "execution_guardrails": {
            "ready_for_execution_must_remain_false": True,
            "do_not_arm_now": True,
            "do_not_run_candidate_now": True,
            "cloud_must_remain_off": True,
            "do_not_generate_yaml": True,
        },
        "supporting_refs": {
            "design_contract": repo_rel(DESIGN_CONTRACT_JSON),
            "patch_boundary_note": repo_rel(PATCH_BOUNDARY_JSON),
            "execution_prep_implementation_sketch": repo_rel(IMPLEMENTATION_SKETCH_JSON),
            "pseudodiff_map": repo_rel(PSEUDODIFF_MAP_JSON),
            "patch_authoring_gate": repo_rel(PATCH_AUTHORING_GATE_JSON),
            "patch_gate_previous_recommendation": patch_gate.get("recommended_decision", ""),
            "target_file": pseudodiff_map.get("target_file", ""),
            "allowed_write_surface": design_contract.get("allowed_write_surface", []),
            "target_object": patch_boundary.get("target_object", ""),
            "pseudodiff_blocks": [block.get("label", "") for block in pseudodiff_map.get("blocks", [])],
            "first_patch_scope": implementation_sketch.get("exact_first_patch_scope", []),
        },
        "next_requirement": "decide_keep_manual_review_only_or_promote_to_patch_authoring",
    }

    write_json(APPROVAL_NOTE_JSON, payload)
    write_text(APPROVAL_NOTE_MD, render_md(payload))

    print(
        json.dumps(
            {
                "patch_authoring_approval_note": repo_rel(APPROVAL_NOTE_JSON),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
