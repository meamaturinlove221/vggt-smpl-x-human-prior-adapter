import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

APPROVAL_NOTE_JSON = OUTPUT_ROOT / "patch_authoring_approval_note.camera_focal_objective_isolation.20260331.json"
PSEUDODIFF_MAP_JSON = OUTPUT_ROOT / "pseudodiff_map.camera_focal_objective_isolation.20260331.json"
REVIEW_NOTE_JSON = OUTPUT_ROOT / "authored_patch_review_note.camera_focal_objective_isolation.20260331.json"
REVIEW_NOTE_MD = OUTPUT_ROOT / "authored_patch_review_note.camera_focal_objective_isolation.20260331.md"


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
        "# Authored Patch Review Note: Camera Focal Objective Isolation (2026-03-31)",
        "",
        f"- default_semantics_preserved: `{payload['default_semantics_preserved']}`",
        f"- return_contract_preserved: `{payload['return_contract_preserved']}`",
        f"- still_not_execution_ready: `{payload['still_not_execution_ready']}`",
        "",
        "## Patch Changed What",
        "",
    ]
    lines.extend([f"- {item}" for item in payload["patch_changed_what"]])
    lines.extend(
        [
            "",
            "## Patch Did Not Change What",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in payload["patch_did_not_change_what"]])
    lines.extend(
        [
            "",
            "## Static Validation",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in payload["static_validation"]])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    checked_at = now_iso()
    approval_note = load_json(APPROVAL_NOTE_JSON)
    pseudodiff_map = load_json(PSEUDODIFF_MAP_JSON)

    payload = {
        "checked_at": checked_at,
        "artifact_kind": "authored_patch_review_note",
        "family": "camera_focal_objective_isolation",
        "target_file": "training/loss.py",
        "patch_changed_what": [
            "Added explicit camera-component assembly helpers inside training/loss.py.",
            "Added a dormant loss_fl_isolation_scale hook to compute_camera_loss with identity default 1.0.",
            "Made stage-level and averaged camera component assembly explicit before loss_camera aggregation.",
            "Routed total_camera_loss formation through an explicit helper while preserving the existing public return keys.",
        ],
        "patch_did_not_change_what": [
            "Did not modify MultitaskLoss.forward.",
            "Did not modify training/trainer.py, training/data/*, training/config/*, or runner/cloud paths.",
            "Did not change the public return contract: loss_camera, loss_T, loss_R, and loss_FL remain present.",
            "Did not materialize config, arm a ticket, run training, or enable execution-ready state.",
        ],
        "default_semantics_preserved": True,
        "return_contract_preserved": True,
        "still_not_execution_ready": True,
        "static_validation": [
            "python -m py_compile training/loss.py passed after authoring.",
            "A local import/smoke check, with a minimal iopath stub in this sandbox environment, confirmed compute_camera_loss still imports and can be called.",
            "A local formula check confirmed default loss_camera remains equal to loss_T*weight_trans + loss_R*weight_rot + loss_FL*weight_focal when loss_fl_isolation_scale stays at its default identity value.",
        ],
        "supporting_refs": {
            "patch_authoring_approval_note": repo_rel(APPROVAL_NOTE_JSON),
            "pseudodiff_map": repo_rel(PSEUDODIFF_MAP_JSON),
            "approval_recommended_decision": approval_note.get("recommended_decision", ""),
            "pseudodiff_blocks": [block.get("label", "") for block in pseudodiff_map.get("blocks", [])],
        },
        "next_requirement": "review_authored_loss_py_patch_before_any_execution_prep_promotion",
    }

    write_json(REVIEW_NOTE_JSON, payload)
    write_text(REVIEW_NOTE_MD, render_md(payload))

    print(
        json.dumps(
            {
                "authored_patch_review_note": repo_rel(REVIEW_NOTE_JSON),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
