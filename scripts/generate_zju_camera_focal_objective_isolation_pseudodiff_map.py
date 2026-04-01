import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

DESIGN_CONTRACT_JSON = OUTPUT_ROOT / "design_contract.camera_focal_objective_isolation.20260330.json"
PATCH_BOUNDARY_JSON = OUTPUT_ROOT / "patch_boundary_note.camera_focal_objective_isolation.20260330.json"
IMPLEMENTATION_SKETCH_JSON = OUTPUT_ROOT / "execution_prep_implementation_sketch.camera_focal_objective_isolation.20260331.json"

PSEUDODIFF_JSON = OUTPUT_ROOT / "pseudodiff_map.camera_focal_objective_isolation.20260331.json"
PSEUDODIFF_MD = OUTPUT_ROOT / "pseudodiff_map.camera_focal_objective_isolation.20260331.md"
PATCH_GATE_JSON = OUTPUT_ROOT / "patch_authoring_gate.camera_focal_objective_isolation.20260331.json"
PATCH_GATE_MD = OUTPUT_ROOT / "patch_authoring_gate.camera_focal_objective_isolation.20260331.md"


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


def render_pseudodiff_md(payload: dict) -> str:
    lines = [
        "# Pseudodiff Map: Camera Focal Objective Isolation (2026-03-31)",
        "",
        f"- state: `{payload['state']}`",
        f"- target_file: `{payload['target_file']}`",
        f"- ready_for_execution: `{payload['ready_for_execution']}`",
        "",
    ]
    for block in payload["blocks"]:
        lines.extend(
            [
                f"## {block['block_id']}: {block['label']}",
                "",
                f"- target_region: `{block['target_region']}`",
                f"- patch_intent: {block['patch_intent']}",
                f"- default_semantics_preserved: `{block['default_semantics_preserved']}`",
                f"- expected_local_invariant: {block['expected_local_invariant']}",
                "",
                "### Do Not Change",
                "",
            ]
        )
        lines.extend([f"- {item}" for item in block["do_not_change"]])
        lines.append("")
    return "\n".join(lines)


def render_patch_gate_md(payload: dict) -> str:
    lines = [
        "# Patch Authoring Gate: Camera Focal Objective Isolation (2026-03-31)",
        "",
        f"- recommended_decision: `{payload['recommended_decision']}`",
        f"- recommended_reason: {payload['recommended_reason']}",
        "",
        "## Gate Checks",
        "",
    ]
    for check in payload["checks"]:
        lines.extend(
            [
                f"- {check['label']}: `{check['status']}`",
                f"  - note: {check['note']}",
            ]
        )
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

    blocks = [
        {
            "block_id": "block_A",
            "label": "compute_camera_loss signature region",
            "target_region": "training/loss.py:271-315",
            "patch_intent": (
                "At most introduce dormant loss.py-local hook parameters or local defaults that make FL isolation explicit without requiring any new config reads or batch keys."
            ),
            "do_not_change": [
                "Do not change existing call sites outside training/loss.py.",
                "Do not add new required kwargs.",
                "Do not add new batch_data dependencies.",
                "Do not change pose encoding, valid_frame_mask logic, or tensor shapes.",
            ],
            "default_semantics_preserved": True,
            "expected_local_invariant": (
                "compute_camera_loss remains callable through existing camera config forwarding, and identity defaults preserve current numeric behavior."
            ),
        },
        {
            "block_id": "block_B",
            "label": "focal loss path region",
            "target_region": "training/loss.py:397-447 plus support at training/loss.py:657-696",
            "patch_intent": (
                "Make the FL-specific path easier to name and inspect by adding pseudodiff-level placeholders such as a local focal-component handle or comments around the loss_FL path, without changing raw focal error math."
            ),
            "do_not_change": [
                "Do not change how loss_FL is computed from pred_pose_valid[..., 7:] vs gt_pose_valid[..., 7:].",
                "Do not change weighting tensor shapes or masking rules.",
                "Do not change l1/l2 branch behavior.",
                "Do not move focal computation into trainer, dataset, or config.",
            ],
            "default_semantics_preserved": True,
            "expected_local_invariant": (
                "The focal-loss path remains numerically identical under the default dormant design, and any future isolation still starts from the same FL tensor path."
            ),
        },
        {
            "block_id": "block_C",
            "label": "component assembly region",
            "target_region": "training/loss.py:455-458",
            "patch_intent": (
                "Introduce an explicit local camera_component_dict or equivalent assembly object so avg_loss_T, avg_loss_R, and avg_loss_FL are assembled before total_camera_loss is formed."
            ),
            "do_not_change": [
                "Do not rename the public returned keys.",
                "Do not change stage averaging semantics.",
                "Do not introduce external logging requirements.",
                "Do not alter the definition of avg_loss_T, avg_loss_R, or avg_loss_FL.",
            ],
            "default_semantics_preserved": True,
            "expected_local_invariant": (
                "The three averaged component scalars remain the same values as before; only the assembly becomes more explicit."
            ),
        },
        {
            "block_id": "block_D",
            "label": "total_camera_loss aggregation and return region",
            "target_region": "training/loss.py:460-472",
            "patch_intent": (
                "Rewrite the aggregation into explicit assembly terms, optionally with a dormant isolated_fl_scale = 1.0 local variable, while keeping loss_camera/loss_T/loss_R/loss_FL return keys unchanged."
            ),
            "do_not_change": [
                "Do not change outer MultitaskLoss.forward composition.",
                "Do not change the default numeric meaning of loss_camera.",
                "Do not add new return keys that trainer/config must know about first.",
                "Do not add arm/run/cloud-facing behavior.",
            ],
            "default_semantics_preserved": True,
            "expected_local_invariant": (
                "When the dormant local variable stays at identity, total_camera_loss and all returned component values are unchanged from the current implementation."
            ),
        },
    ]

    pseudodiff = {
        "checked_at": checked_at,
        "artifact_kind": "pseudodiff_map",
        "family": "camera_focal_objective_isolation",
        "state": "manual_review_only_pseudodiff_map",
        "ready_for_execution": False,
        "target_file": "training/loss.py",
        "target_object": patch_boundary.get("target_object", "loss_FL"),
        "allowed_write_surface": design_contract.get("allowed_write_surface", []),
        "blocks": blocks,
        "supporting_refs": {
            "design_contract": repo_rel(DESIGN_CONTRACT_JSON),
            "patch_boundary_note": repo_rel(PATCH_BOUNDARY_JSON),
            "execution_prep_implementation_sketch": repo_rel(IMPLEMENTATION_SKETCH_JSON),
        },
        "next_requirement": "manual_review_loss_py_pseudodiff_before_any_patch_authoring",
    }

    checks = [
        {
            "id": "loss_py_only_blocks",
            "label": "all planned blocks remain inside training/loss.py",
            "status": "supported",
            "note": "All four pseudodiff blocks map to training/loss.py regions only.",
        },
        {
            "id": "no_trainer_config_prereq",
            "label": "no trainer/config pre-change is required",
            "status": "supported",
            "note": "Current loss_FL logging already exists, so the pseudodiff stays loss.py-only.",
        },
        {
            "id": "return_contract_preserved",
            "label": "loss_camera/loss_T/loss_R/loss_FL contract remains preserved",
            "status": "supported",
            "note": "Every block keeps the current returned keys and default semantics unchanged.",
        },
        {
            "id": "manual_authoring_approval_still_needed",
            "label": "manual approval is still required before real patch authoring",
            "status": "required",
            "note": "The pseudodiff proves local scope, but governance still requires an explicit human decision before any code-authoring step.",
        },
    ]

    patch_gate = {
        "checked_at": checked_at,
        "artifact_kind": "patch_authoring_gate",
        "family": "camera_focal_objective_isolation",
        "state": "manual_review_only",
        "allowed_buttons": [
            "KEEP_MANUAL_REVIEW_ONLY",
            "PROMOTE_TO_PATCH_AUTHORING",
        ],
        "checks": checks,
        "recommended_decision": "KEEP_MANUAL_REVIEW_ONLY",
        "recommended_reason": (
            "The pseudodiff now shows a clean training/loss.py-only path, but the workflow is still manual-review-governed and should not auto-advance into patch authoring without an explicit human approval step."
        ),
        "ready_for_execution": False,
        "do_not_arm_now": True,
        "do_not_run_candidate_now": True,
        "cloud_must_remain_off": True,
        "supporting_refs": {
            "pseudodiff_map": repo_rel(PSEUDODIFF_JSON),
            "execution_prep_implementation_sketch": repo_rel(IMPLEMENTATION_SKETCH_JSON),
        },
        "next_requirement": "manual_review_loss_py_pseudodiff_before_any_patch_authoring",
    }

    write_json(PSEUDODIFF_JSON, pseudodiff)
    write_text(PSEUDODIFF_MD, render_pseudodiff_md(pseudodiff))
    write_json(PATCH_GATE_JSON, patch_gate)
    write_text(PATCH_GATE_MD, render_patch_gate_md(patch_gate))

    print(
        json.dumps(
            {
                "pseudodiff_map": repo_rel(PSEUDODIFF_JSON),
                "patch_authoring_gate": repo_rel(PATCH_GATE_JSON),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
