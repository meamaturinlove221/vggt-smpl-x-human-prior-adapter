import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

RECOMMEND_JSON = OUTPUT_ROOT / "minimal_write_surface_recommendation.camera_focal_objective_isolation.20260330.json"
DESIGN_JSON = OUTPUT_ROOT / "execution_prep_design_sketch.camera_focal_objective_isolation.20260330.json"
CHECKLIST_JSON = OUTPUT_ROOT / "decision_checklist.camera_focal_objective_isolation.20260330.json"

CONTRACT_JSON = OUTPUT_ROOT / "design_contract.camera_focal_objective_isolation.20260330.json"
CONTRACT_MD = OUTPUT_ROOT / "design_contract.camera_focal_objective_isolation.20260330.md"
BOUNDARY_JSON = OUTPUT_ROOT / "patch_boundary_note.camera_focal_objective_isolation.20260330.json"
BOUNDARY_MD = OUTPUT_ROOT / "patch_boundary_note.camera_focal_objective_isolation.20260330.md"


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


def render_contract_md(payload: dict) -> str:
    lines = [
        "# Design Contract: Camera Focal Objective Isolation (2026-03-30)",
        "",
        "## 1. Problem Statement",
        "",
        f"- {payload['problem_statement']}",
        "",
        "## 2. Allowed Write Surface",
        "",
    ]
    lines.extend([f"- `{item}`" for item in payload["allowed_write_surface"]])
    lines.extend(
        [
            "",
            "## 3. Allowed Patch Kinds",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in payload["allowed_patch_kinds"]])
    lines.extend(
        [
            "",
            "## 4. Forbidden Changes",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in payload["forbidden_changes"]])
    lines.extend(
        [
            "",
            "## 5. Promotion Gate To Execution Prep",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in payload["promotion_gate_to_execution_prep"]])
    lines.extend(
        [
            "",
            "## 6. Minimal Validation Signals",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in payload["minimal_validation_signals"]])
    lines.append("")
    return "\n".join(lines)


def render_boundary_md(payload: dict) -> str:
    lines = [
        "# Patch Boundary Note: Camera Focal Objective Isolation (2026-03-30)",
        "",
        f"- target_file: `{payload['target_file']}`",
        f"- target_object: `{payload['target_object']}`",
        "",
        "## Do Not Touch",
        "",
    ]
    lines.extend([f"- {item}" for item in payload["do_not_touch"]])
    lines.extend(
        [
            "",
            f"- future_first_patch_kind: `{payload['future_first_patch_kind']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    checked_at = now_iso()
    recommendation = load_json(RECOMMEND_JSON)
    design = load_json(DESIGN_JSON)
    checklist = load_json(CHECKLIST_JSON)

    contract = {
        "checked_at": checked_at,
        "artifact_kind": "design_contract",
        "family": "camera_focal_objective_isolation",
        "problem_statement": (
            "In the absence of any dataset, routing, stream-contract, or candidate-config changes, determine whether training/loss.py alone can isolate the loss_FL-related camera objective from the current loss_camera aggregation into an object that can later be scheduled, accounted, and constrained independently."
        ),
        "allowed_write_surface": [
            "training/loss.py",
        ],
        "allowed_patch_kinds": [
            "loss_FL single-object isolation",
            "component-level separation before loss_camera aggregation",
            "bounded scalar/gate/schedule hook design positions for loss_FL",
            "design-level definition of any extra logging key needed for the isolated FL object",
        ],
        "forbidden_changes": [
            "Do not reopen tail-contract derivative families.",
            "Do not modify manifest, reserve, or hardtail bucket behavior.",
            "Do not add another two-stage cousin or schedule family.",
            "Do not introduce new dataset label plumbing dependencies.",
            "Do not write candidate yaml.",
            "Do not touch arm or run paths.",
            "Do not touch cloud.",
            "Do not modify training/trainer.py in this contract round.",
            "Do not modify training/data/* in this contract round.",
            "Do not modify training/config/* in this contract round.",
            "Do not modify scripts/run_zju_source_policy_research_candidate.py in this contract round.",
        ],
        "promotion_gate_to_execution_prep": [
            "A single patch boundary is agreed: training/loss.py only.",
            "A single target object is agreed: loss_FL isolation inside the camera-loss aggregation path.",
            "The minimal validation signal is defined without first expanding trainer, dataset, config, runner, or cloud surfaces.",
        ],
        "minimal_validation_signals": [
            "Future execution-prep must be able to isolate or scale loss_FL independently.",
            "Future execution-prep must preserve the original total loss_camera aggregation semantics unless the design explicitly says otherwise.",
            "Existing logging surfaces must remain sufficient to observe loss_FL movement without requiring a trainer or config expansion first.",
            "No new stream family may be introduced and the tail contract must remain unchanged.",
        ],
        "supporting_refs": {
            "minimal_write_surface_recommendation": repo_rel(RECOMMEND_JSON),
            "execution_prep_design_sketch": repo_rel(DESIGN_JSON),
            "decision_checklist": repo_rel(CHECKLIST_JSON),
        },
    }

    boundary = {
        "checked_at": checked_at,
        "artifact_kind": "patch_boundary_note",
        "family": "camera_focal_objective_isolation",
        "target_file": "training/loss.py",
        "target_object": "loss_FL inside compute_camera_loss / camera loss aggregation path",
        "do_not_touch": [
            "training/trainer.py",
            "training/data/*",
            "training/config/*",
            "scripts/run_zju_source_policy_research_candidate.py",
            "cloud",
        ],
        "future_first_patch_kind": "isolate_FL_object_without_redefining_tail_contract",
        "boundary_reason": (
            "The recommendation already narrowed the honest execution-prep surface to training/loss.py alone, and the checklist still keeps the family in manual-review-only mode until that boundary is explicitly approved."
        ),
        "current_recommended_button": checklist.get("recommended_decision", ""),
    }

    write_json(CONTRACT_JSON, contract)
    write_text(CONTRACT_MD, render_contract_md(contract))
    write_json(BOUNDARY_JSON, boundary)
    write_text(BOUNDARY_MD, render_boundary_md(boundary))

    print(
        json.dumps(
            {
                "design_contract": repo_rel(CONTRACT_JSON),
                "patch_boundary_note": repo_rel(BOUNDARY_JSON),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
