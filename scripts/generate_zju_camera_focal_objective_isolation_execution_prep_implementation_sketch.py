import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

DESIGN_CONTRACT_JSON = OUTPUT_ROOT / "design_contract.camera_focal_objective_isolation.20260330.json"
PATCH_BOUNDARY_JSON = OUTPUT_ROOT / "patch_boundary_note.camera_focal_objective_isolation.20260330.json"
RECOMMEND_JSON = OUTPUT_ROOT / "minimal_write_surface_recommendation.camera_focal_objective_isolation.20260330.json"

SKETCH_JSON = OUTPUT_ROOT / "execution_prep_implementation_sketch.camera_focal_objective_isolation.20260331.json"
SKETCH_MD = OUTPUT_ROOT / "execution_prep_implementation_sketch.camera_focal_objective_isolation.20260331.md"


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
        "# Execution-Prep Implementation Sketch: Camera Focal Objective Isolation (2026-03-31)",
        "",
        f"- state: `{payload['state']}`",
        f"- ready_for_execution: `{payload['ready_for_execution']}`",
        f"- target_file: `{payload['target_file']}`",
        "",
        "## Primary Goal",
        "",
        f"- {payload['implementation_goal']}",
        "",
        "## Local Code Evidence",
        "",
    ]
    lines.extend([f"- {item}" for item in payload["local_code_evidence"]])
    lines.extend(
        [
            "",
            "## Exact First-Patch Scope",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in payload["exact_first_patch_scope"]])
    lines.extend(
        [
            "",
            "## Patch Units",
            "",
        ]
    )
    for item in payload["patch_units"]:
        lines.extend(
            [
                f"### {item['id']}: {item['title']}",
                "",
                f"- target_region: `{item['target_region']}`",
                f"- patch_kind: {item['patch_kind']}",
                f"- default_semantics_preserved: `{item['default_semantics_preserved']}`",
                f"- design_note: {item['design_note']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Do Not Change In First Patch",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in payload["do_not_change_in_first_patch"]])
    lines.extend(
        [
            "",
            "## Future Acceptance Gates",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in payload["future_acceptance_gates"]])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    checked_at = now_iso()
    design_contract = load_json(DESIGN_CONTRACT_JSON)
    patch_boundary = load_json(PATCH_BOUNDARY_JSON)
    recommendation = load_json(RECOMMEND_JSON)

    payload = {
        "checked_at": checked_at,
        "artifact_kind": "execution_prep_implementation_sketch",
        "family": "camera_focal_objective_isolation",
        "state": "execution_prep_design_implementation_only",
        "ready_for_execution": False,
        "manual_action_kind": "manual_review",
        "target_file": "training/loss.py",
        "target_object": patch_boundary.get("target_object", "loss_FL"),
        "implementation_goal": (
            "Define the first loss.py-only implementation shape that would isolate the FL-dominant camera objective "
            "inside compute_camera_loss while preserving the existing loss_camera contract and without requiring trainer, "
            "dataset, config, runner, or cloud changes."
        ),
        "local_code_evidence": [
            "training/loss.py:29 _loss_call_kwargs already forwards non-meta camera kwargs, so a future loss-only hook does not require trainer-side plumbing first.",
            "training/loss.py:63 MultitaskLoss.forward consumes loss_camera as the camera objective entry, so preserving loss_camera keeps outer loss composition stable.",
            "training/loss.py:271 compute_camera_loss already computes camera components in the one file approved by the design contract.",
            "training/loss.py:468-472 compute_camera_loss already returns loss_camera, loss_T, loss_R, and loss_FL as distinct values.",
            "training/loss.py:657 camera_loss_single already isolates the raw focal-error tensor path, so the first patch does not need dataset or routing changes to find the FL object.",
            "training/trainer.py:856-870 scalar logging is key-based and does not need trainer changes just to preserve existing loss_FL visibility.",
            "training/config/default.yaml:62-82 and training/config/zju_vggt_geom_unproject_source_policy_nearest_rawpool_minimal.yaml:106-126 already log loss_FL for train and val.",
        ],
        "exact_first_patch_scope": [
            "Keep the write surface inside training/loss.py only.",
            "Touch compute_camera_loss first; keep MultitaskLoss.forward behavior unchanged in the first patch.",
            "Preserve returned keys loss_camera, loss_T, loss_R, and loss_FL.",
            "If a dormant isolation hook is introduced, it must default to identity semantics and remain inactive without new config materialization.",
        ],
        "patch_units": [
            {
                "id": "unit_1",
                "title": "explicit camera-component assembly inside compute_camera_loss",
                "target_region": "training/loss.py:271-472 compute_camera_loss",
                "patch_kind": "refactor-only component assembly",
                "default_semantics_preserved": True,
                "design_note": (
                    "Gather avg_loss_T, avg_loss_R, and avg_loss_FL into an explicit local component structure before total_camera_loss is formed. "
                    "This makes the FL object boundary concrete without changing current math."
                ),
            },
            {
                "id": "unit_2",
                "title": "single local aggregation hook for FL isolation",
                "target_region": "training/loss.py:460-472 total_camera_loss aggregation and return block",
                "patch_kind": "bounded local hook site",
                "default_semantics_preserved": True,
                "design_note": (
                    "Introduce one local composition point where the FL component could later be isolated, scaled, or constrained independently, "
                    "while the default path still computes loss_camera exactly as today."
                ),
            },
            {
                "id": "unit_3",
                "title": "signature-safe future hook placeholders only if needed",
                "target_region": "training/loss.py:271 compute_camera_loss signature and local helper area",
                "patch_kind": "optional dormant parameter design",
                "default_semantics_preserved": True,
                "design_note": (
                    "Only if the design needs an explicit future hook, add compute_camera_loss-local parameters with identity defaults. "
                    "_loss_call_kwargs already forwards camera kwargs, so this remains loss.py-only and does not require trainer or config edits today."
                ),
            },
            {
                "id": "unit_4",
                "title": "leave MultitaskLoss.forward unchanged in first patch",
                "target_region": "training/loss.py:63-171 MultitaskLoss.forward",
                "patch_kind": "no-op boundary decision",
                "default_semantics_preserved": True,
                "design_note": (
                    "Do not rewrite the outer objective composition in the first patch. Keep forward consuming loss_camera so the first authored patch stays inside the approved object boundary."
                ),
            },
        ],
        "do_not_change_in_first_patch": [
            "Do not edit training/trainer.py.",
            "Do not edit training/data/*.",
            "Do not edit training/config/*.",
            "Do not edit scripts/run_zju_source_policy_research_candidate.py.",
            "Do not materialize a candidate yaml.",
            "Do not redefine tail-contract behavior, reserve routing, or hardtail bucket behavior.",
            "Do not introduce a two-stage cousin, cloud path, or arm/run integration.",
            "Do not require any new logging key to validate the first authored patch; existing loss_FL visibility must remain enough.",
        ],
        "future_acceptance_gates": [
            "A future authored patch must preserve the default loss_camera numeric behavior when the new hook is inactive.",
            "A future authored patch must keep the first authored surface inside training/loss.py only.",
            "A future authored patch must preserve existing returned keys so current trainer/config logging does not regress.",
            "A future authored patch must make the FL object boundary easier to inspect without reopening old families or adding config-first complexity.",
        ],
        "supporting_refs": {
            "design_contract": repo_rel(DESIGN_CONTRACT_JSON),
            "patch_boundary_note": repo_rel(PATCH_BOUNDARY_JSON),
            "minimal_write_surface_recommendation": repo_rel(RECOMMEND_JSON),
            "allowed_write_surface": design_contract.get("allowed_write_surface", []),
            "recommended_minimal_write_surface": recommendation.get("recommended_minimal_write_surface", ""),
        },
        "next_requirement": "manual_review_loss_py_execution_prep_implementation_sketch_before_any_patch_authoring",
    }

    write_json(SKETCH_JSON, payload)
    write_text(SKETCH_MD, render_md(payload))

    print(
        json.dumps(
            {
                "execution_prep_implementation_sketch": repo_rel(SKETCH_JSON),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
