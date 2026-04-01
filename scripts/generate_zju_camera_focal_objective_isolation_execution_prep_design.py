import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

LOCALIZATION_JSON = OUTPUT_ROOT / "early_fl_tax_localization.20260330.json"
FAILURE_JSON = OUTPUT_ROOT / "two_stage_failure_interpretation.20260330.json"
REVIEW_NOTE_JSON = OUTPUT_ROOT / "review_note.camera_focal_objective_isolation.20260330.json"
CHECKLIST_JSON = OUTPUT_ROOT / "decision_checklist.camera_focal_objective_isolation.20260330.json"

DESIGN_JSON = OUTPUT_ROOT / "execution_prep_design_sketch.camera_focal_objective_isolation.20260330.json"
DESIGN_MD = OUTPUT_ROOT / "execution_prep_design_sketch.camera_focal_objective_isolation.20260330.md"
CANDIDATES_JSON = OUTPUT_ROOT / "minimal_write_surface_candidates.camera_focal_objective_isolation.20260330.json"
CANDIDATES_MD = OUTPUT_ROOT / "minimal_write_surface_candidates.camera_focal_objective_isolation.20260330.md"
RECOMMEND_JSON = OUTPUT_ROOT / "minimal_write_surface_recommendation.camera_focal_objective_isolation.20260330.json"
RECOMMEND_MD = OUTPUT_ROOT / "minimal_write_surface_recommendation.camera_focal_objective_isolation.20260330.md"


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


def render_design_md(payload: dict) -> str:
    lines = [
        "# Execution-Prep Design Sketch: Camera Focal Objective Isolation (2026-03-30)",
        "",
        f"- state: `{payload['state']}`",
        f"- family: `{payload['family']}`",
        f"- ready_for_execution: `{payload['ready_for_execution']}`",
        "",
        "## Constraints",
        "",
    ]
    lines.extend([f"- {item}" for item in payload["constraints"]])
    lines.extend(["", "## Candidate Surfaces", ""])
    for item in payload["candidate_surfaces"]:
        lines.extend(
            [
                f"### {item['label']}",
                "",
                f"- candidate_name: `{item['candidate_name']}`",
                f"- files_to_touch: `{item['files_to_touch']}`",
                f"- core_question: {item['core_question']}",
                f"- why_this_is_honest: {item['why_this_is_honest']}",
                "",
            ]
        )
    return "\n".join(lines)


def render_candidates_md(payload: dict) -> str:
    lines = [
        "# Minimal Write Surface Candidates: Camera Focal Objective Isolation (2026-03-30)",
        "",
        "| candidate_name | files_to_touch | loss | trainer | dataset | config | reopen_tail | supports_honestly | risk |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in payload["candidates"]:
        lines.append(
            f"| {item['candidate_name']} | {', '.join(item['files_to_touch'])} | "
            f"{int(item['touches_loss'])} | {int(item['touches_trainer'])} | {int(item['touches_dataset'])} | "
            f"{int(item['touches_config'])} | {int(item['reopens_tail_contract_family'])} | "
            f"{int(item['supports_manual_hypothesis_honestly'])} | {item['execution_prep_risk']} |"
        )
    lines.extend(["", "## Per-Candidate Notes", ""])
    for item in payload["candidates"]:
        lines.extend(
            [
                f"### {item['candidate_name']}",
                "",
                f"- why_not_smaller: {item['why_not_smaller']}",
                f"- why_not_larger: {item['why_not_larger']}",
                "",
            ]
        )
    return "\n".join(lines)


def render_recommend_md(payload: dict) -> str:
    lines = [
        "# Minimal Write Surface Recommendation: Camera Focal Objective Isolation (2026-03-30)",
        "",
        f"- recommended_minimal_write_surface: `{payload['recommended_minimal_write_surface']}`",
        f"- files_to_touch: `{payload['files_to_touch']}`",
        "",
        "## Why",
        "",
        f"- why_this_is_the_smallest_honest_surface: {payload['why_this_is_the_smallest_honest_surface']}",
        f"- why_it_is_enough_for_execution_prep: {payload['why_it_is_enough_for_execution_prep']}",
        f"- why_it_is_not_a_reopen_of_an_old_family: {payload['why_it_is_not_a_reopen_of_an_old_family']}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    checked_at = now_iso()
    localization = load_json(LOCALIZATION_JSON)
    failure = load_json(FAILURE_JSON)
    review_note = load_json(REVIEW_NOTE_JSON)
    checklist = load_json(CHECKLIST_JSON)

    candidates = [
        {
            "candidate_name": "loss-only camera focal term isolation",
            "label": "A. loss-only camera focal term isolation",
            "files_to_touch": ["training/loss.py"],
            "touches_loss": True,
            "touches_trainer": False,
            "touches_dataset": False,
            "touches_config": False,
            "reopens_tail_contract_family": False,
            "supports_manual_hypothesis_honestly": True,
            "execution_prep_risk": "low",
            "core_question": (
                "Can the FL-dominant tax be isolated as its own loss object inside the existing camera loss composition, "
                "without touching dataset, stream contract, or runner surfaces?"
            ),
            "why_this_is_honest": (
                "training/loss.py already computes loss_T, loss_R, and loss_FL separately before combining them into loss_camera, "
                "so the narrowest honest next design is to isolate the FL object where it is actually formed."
            ),
            "why_not_smaller": (
                "Anything smaller than training/loss.py would stay at note level only and would not actually isolate the objective object under review."
            ),
            "why_not_larger": (
                "Trainer and config surfaces are not required for the first execution-prep sketch because loss_FL already exists as a first-class component and is already returned by the loss module."
            ),
        },
        {
            "candidate_name": "camera loss component logging/gating scaffold",
            "label": "B. camera loss component logging/gating scaffold",
            "files_to_touch": ["training/loss.py", "training/trainer.py"],
            "touches_loss": True,
            "touches_trainer": True,
            "touches_dataset": False,
            "touches_config": False,
            "reopens_tail_contract_family": False,
            "supports_manual_hypothesis_honestly": True,
            "execution_prep_risk": "medium",
            "core_question": (
                "Should execution prep first add more explicit per-component camera-objective accounting and bounded gating hooks before changing the objective itself?"
            ),
            "why_this_is_honest": (
                "If the team decides the current aggregate logging is still too coarse, the next minimal expansion is trainer-side accounting around the existing loss components rather than a wider family reopen."
            ),
            "why_not_smaller": (
                "A pure loss.py patch would not help if the real blocker is missing component-level training-time accounting or gate-safety scaffolding."
            ),
            "why_not_larger": (
                "Even this scaffold still avoids dataset changes, routing changes, and candidate-yaml materialization."
            ),
        },
        {
            "candidate_name": "default-stream-only camera-objective isolation hook",
            "label": "C. default-stream-only camera-objective isolation hook",
            "files_to_touch": ["training/loss.py", "training/config/<future_single_audit_config_placeholder>"],
            "touches_loss": True,
            "touches_trainer": False,
            "touches_dataset": False,
            "touches_config": True,
            "reopens_tail_contract_family": False,
            "supports_manual_hypothesis_honestly": True,
            "execution_prep_risk": "medium",
            "core_question": (
                "If a local entry point is eventually required, is the smallest credible entry a default-stream-only camera/focal isolation hook rather than another hardtail/reserve intervention?"
            ),
            "why_this_is_honest": (
                "It still targets the camera objective object, but delays the leap to a future config gate until after the write surface is agreed."
            ),
            "why_not_smaller": (
                "Without at least a future single-config entry point, this option cannot express a bounded default-stream-only hook if the review later insists on one."
            ),
            "why_not_larger": (
                "It still avoids trainer, dataset, routing, cloud, and family-reopen surfaces."
            ),
        },
    ]

    design = {
        "checked_at": checked_at,
        "artifact_kind": "execution_prep_design_sketch",
        "family": "camera_focal_objective_isolation",
        "state": "execution_prep_design_only",
        "ready_for_execution": False,
        "constraints": [
            "Do not arm, run, or touch cloud.",
            "Do not materialize candidate yaml today.",
            "Do not reopen tail-contract derivative families.",
            "Do not require dataset or routing changes.",
            "Choose the smallest honest write surface before any later execution-prep step.",
        ],
        "starting_facts": {
            "dominant_component": localization.get("dominant_component", ""),
            "appears_by": localization.get("appears_by", ""),
            "most_supported_scope": localization.get("most_supported_scope", ""),
            "failure_conclusion": failure.get("conclusion", ""),
        },
        "candidate_surfaces": candidates,
        "current_recommended_button": checklist.get("recommended_decision", ""),
        "next_requirement": "choose_single_minimal_write_surface_before_execution_prep",
        "supporting_refs": {
            "review_note": repo_rel(REVIEW_NOTE_JSON),
            "decision_checklist": repo_rel(CHECKLIST_JSON),
            "localization": repo_rel(LOCALIZATION_JSON),
            "failure": repo_rel(FAILURE_JSON),
            "loss_surface": "training/loss.py",
            "trainer_surface": "training/trainer.py",
            "logging_config_surface": "training/config/default.yaml",
        },
    }

    candidate_matrix = {
        "checked_at": checked_at,
        "artifact_kind": "minimal_write_surface_candidates",
        "family": "camera_focal_objective_isolation",
        "candidates": candidates,
        "selection_goal": "choose the narrowest write surface that still isolates the FL-dominant global objective conflict honestly",
    }

    recommendation = {
        "checked_at": checked_at,
        "artifact_kind": "minimal_write_surface_recommendation",
        "family": "camera_focal_objective_isolation",
        "recommended_minimal_write_surface": "loss-only camera focal term isolation",
        "files_to_touch": ["training/loss.py"],
        "why_this_is_the_smallest_honest_surface": (
            "training/loss.py already computes loss_T, loss_R, and loss_FL separately and then combines them into loss_camera, so the objective object under review already lives in that one file. "
            "A narrower surface would not actually isolate anything beyond documentation."
        ),
        "why_it_is_enough_for_execution_prep": (
            "Because loss_FL is already returned as its own scalar and the default training configs already log loss_FL, execution prep can stay at the loss-object layer without requiring trainer, dataset, or config expansion just to define the design."
        ),
        "why_it_is_not_a_reopen_of_an_old_family": (
            "This recommendation does not change hardtail/reserve manifests, stream routing, two-stage schedules, or tail cousins. It isolates the FL-dominant camera objective object itself rather than reopening any prior family axis."
        ),
        "rejected_alternatives": {
            "camera loss component logging/gating scaffold": "Useful only if current component accounting proves insufficient; larger than needed for the first honest execution-prep sketch.",
            "default-stream-only camera-objective isolation hook": "Potential later local entry point, but it is not the smallest honest surface because it adds a config concept before the loss-object surface has been settled.",
        },
        "next_requirement": "choose_single_minimal_write_surface_before_execution_prep",
        "code_evidence": [
            "training/loss.py: compute_camera_loss already forms loss_T, loss_R, loss_FL separately before combining them into loss_camera",
            "training/trainer.py: _update_and_log_scalars already logs scalar keys supplied by config",
            "training/config/default.yaml: scalar_keys_to_log already includes loss_camera, loss_T, loss_R, loss_FL for train and val",
        ],
    }

    write_json(DESIGN_JSON, design)
    write_text(DESIGN_MD, render_design_md(design))
    write_json(CANDIDATES_JSON, candidate_matrix)
    write_text(CANDIDATES_MD, render_candidates_md(candidate_matrix))
    write_json(RECOMMEND_JSON, recommendation)
    write_text(RECOMMEND_MD, render_recommend_md(recommendation))

    print(
        json.dumps(
            {
                "design_sketch": repo_rel(DESIGN_JSON),
                "candidate_matrix": repo_rel(CANDIDATES_JSON),
                "recommendation": repo_rel(RECOMMEND_JSON),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
