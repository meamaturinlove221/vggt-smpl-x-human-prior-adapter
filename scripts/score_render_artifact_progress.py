import argparse
import json
import zipfile
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize render-artifact deltas for a baseline/candidate pair.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--baseline-variant", default="baseline_depth_unproject")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--reference-summary-json")
    parser.add_argument("--reference-variant")
    parser.add_argument("--family")
    parser.add_argument("--key-panels-zip")
    parser.add_argument("--artifact-completeness-json")
    parser.add_argument("--zip-manifest-json")
    return parser.parse_args()


def _rows_by_variant(summary: dict, variant: str) -> dict[str, dict]:
    return {str(row["case_id"]): row for row in summary.get("rows", []) if str(row.get("variant")) == variant}


BASELINE_KEY_PANEL_KEYS = [
    "target_baseline_candidate_renderdiff_fgmask_png",
    "fg_visible_components_colored_png",
    "fg_peak_map_png",
    "fg_hole_bridge_panel_png",
    "masked_error_regression_map_png",
    "component_rank_ledger_json",
]

MERGE_RULE_KEY_PANEL_KEYS = BASELINE_KEY_PANEL_KEYS + [
    "before_after_render_operator_panel_png",
    "render_operator_conflict_overlay_png",
    "render_operator_choice_overlay_png",
    "before_after_component_merge_panel_png",
    "before_after_peak_component_panel_png",
    "render_operator_guard_regression_panel_png",
    "render_operator_merge_intent_overlay_png",
    "render_operator_merge_adoption_overlay_png",
    "render_operator_merge_veto_overlay_png",
    "render_operator_hard_anchor_failure_overlay_png",
]

PEAK_STABILIZATION_KEY_PANEL_KEYS = MERGE_RULE_KEY_PANEL_KEYS + [
    "render_operator_post_merge_peak_risk_overlay_png",
    "render_operator_post_merge_peak_containment_overlay_png",
    "render_operator_post_merge_realization_overlay_png",
    "before_after_post_merge_peak_panel_png",
    "before_after_post_merge_component_panel_png",
    "render_operator_hard_anchor_peak_breakout_overlay_png",
]

POST_MERGE_REALIZATION_KEY_PANEL_KEYS = PEAK_STABILIZATION_KEY_PANEL_KEYS + [
    "render_operator_post_merge_realization_eligibility_overlay_png",
    "render_operator_post_merge_realization_binding_overlay_png",
    "render_operator_post_merge_component_consolidation_overlay_png",
    "before_after_post_merge_realization_panel_png",
    "before_after_post_merge_component_consolidation_panel_png",
    "render_operator_hard_anchor_realization_breakout_overlay_png",
]

COMPONENT_CONSOLIDATION_KEY_PANEL_KEYS = POST_MERGE_REALIZATION_KEY_PANEL_KEYS + [
    "render_operator_post_merge_component_eligibility_overlay_png",
    "render_operator_post_merge_component_binding_overlay_png",
    "render_operator_post_merge_component_consolidation_contract_overlay_png",
    "before_after_post_merge_component_binding_panel_png",
    "before_after_post_merge_component_consolidation_contract_panel_png",
    "render_operator_hard_anchor_component_consolidation_breakout_overlay_png",
]

COMPONENT_ADJACENCY_KEY_PANEL_KEYS = COMPONENT_CONSOLIDATION_KEY_PANEL_KEYS + [
    "render_operator_post_merge_component_adjacency_overlay_png",
    "render_operator_post_merge_component_gap_closure_overlay_png",
    "render_operator_post_merge_component_adjacency_contract_overlay_png",
    "before_after_post_merge_component_adjacency_panel_png",
    "before_after_post_merge_component_gap_closure_panel_png",
    "render_operator_hard_anchor_component_adjacency_breakout_overlay_png",
]

COMPONENT_CLOSURE_REALIZATION_KEY_PANEL_KEYS = COMPONENT_ADJACENCY_KEY_PANEL_KEYS + [
    "render_operator_post_merge_component_closure_realization_eligibility_overlay_png",
    "render_operator_post_merge_component_closure_realization_binding_overlay_png",
    "render_operator_post_merge_component_closure_realization_contract_overlay_png",
    "before_after_post_merge_component_closure_realization_panel_png",
    "before_after_post_merge_component_closure_binding_panel_png",
    "render_operator_hard_anchor_component_closure_realization_breakout_overlay_png",
]

HARD_ANCHOR_CLOSURE_REALIZATION_KEY_PANEL_KEYS = COMPONENT_CLOSURE_REALIZATION_KEY_PANEL_KEYS + [
    "render_operator_hard_anchor_local_closure_focus_overlay_png",
    "render_operator_hard_anchor_local_closure_binding_overlay_png",
    "render_operator_hard_anchor_local_closure_veto_overlay_png",
    "before_after_hard_anchor_local_closure_panel_png",
    "render_operator_hard_anchor_local_peak_rebound_guard_overlay_png",
    "render_operator_hard_anchor_local_closure_breakout_overlay_png",
]

HARD_ANCHOR_BREAKOUT_ALIGNMENT_KEY_PANEL_KEYS = HARD_ANCHOR_CLOSURE_REALIZATION_KEY_PANEL_KEYS + [
    "render_operator_hard_anchor_breakout_target_overlay_png",
    "render_operator_hard_anchor_local_shrinkage_witness_overlay_png",
    "render_operator_hard_anchor_breakout_alignment_overlap_overlay_png",
    "render_operator_hard_anchor_breakout_misalignment_veto_overlay_png",
    "before_after_hard_anchor_breakout_alignment_panel_png",
    "render_operator_hard_anchor_breakout_alignment_overlay_png",
]

HARD_ANCHOR_PEAK_REBOUND_SUPPRESSION_KEY_PANEL_KEYS = HARD_ANCHOR_BREAKOUT_ALIGNMENT_KEY_PANEL_KEYS + [
    "render_operator_hard_anchor_peak_rebound_focus_overlay_png",
    "render_operator_hard_anchor_peak_rebound_binding_overlay_png",
    "render_operator_hard_anchor_peak_rebound_veto_overlay_png",
    "before_after_hard_anchor_peak_rebound_suppression_panel_png",
    "render_operator_hard_anchor_peak_rebound_suppression_overlay_png",
    "render_operator_hard_anchor_peak_rebound_residual_overlay_png",
]

HARD_ANCHOR_PEAK_RESIDUAL_ACTIVATION_KEY_PANEL_KEYS = HARD_ANCHOR_PEAK_REBOUND_SUPPRESSION_KEY_PANEL_KEYS + [
    "render_operator_hard_anchor_peak_residual_activation_overlay_png",
    "render_operator_hard_anchor_peak_residual_binding_overlay_png",
    "render_operator_hard_anchor_peak_residual_veto_overlay_png",
    "before_after_hard_anchor_peak_residual_activation_panel_png",
    "render_operator_hard_anchor_peak_residual_contraction_witness_overlay_png",
    "render_operator_hard_anchor_peak_residual_guard_overlay_png",
]

HARD_ANCHOR_PEAK_RESIDUAL_CONTRACTION_REALIZATION_KEY_PANEL_KEYS = HARD_ANCHOR_PEAK_RESIDUAL_ACTIVATION_KEY_PANEL_KEYS + [
    "render_operator_hard_anchor_peak_residual_contraction_target_overlay_png",
    "render_operator_hard_anchor_peak_residual_contraction_binding_overlay_png",
    "render_operator_hard_anchor_peak_residual_contraction_realization_overlay_png",
    "before_after_hard_anchor_peak_residual_contraction_panel_png",
    "render_operator_hard_anchor_peak_residual_contraction_veto_overlay_png",
    "render_operator_hard_anchor_peak_residual_contraction_guard_overlay_png",
]

HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_ALIGNMENT_REALIZATION_KEY_PANEL_KEYS = HARD_ANCHOR_PEAK_RESIDUAL_CONTRACTION_REALIZATION_KEY_PANEL_KEYS + [
    "render_operator_hard_anchor_peak_residual_overlap_target_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_binding_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_realization_overlay_png",
    "before_after_hard_anchor_peak_residual_overlap_panel_png",
    "render_operator_hard_anchor_peak_residual_overlap_veto_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_alignment_overlay_png",
]

HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_GAIN_REALIZATION_KEY_PANEL_KEYS = HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_ALIGNMENT_REALIZATION_KEY_PANEL_KEYS + [
    "render_operator_hard_anchor_peak_residual_overlap_gain_target_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_binding_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_realization_overlay_png",
    "before_after_hard_anchor_peak_residual_overlap_gain_panel_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_veto_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_alignment_overlay_png",
]

HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_GAIN_AMPLIFICATION_REALIZATION_KEY_PANEL_KEYS = HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_GAIN_REALIZATION_KEY_PANEL_KEYS + [
    "render_operator_hard_anchor_peak_residual_overlap_gain_amplification_target_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_amplification_binding_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_amplification_realization_overlay_png",
    "before_after_hard_anchor_peak_residual_overlap_gain_amplification_panel_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_amplification_veto_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_amplification_alignment_overlay_png",
]

HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_GAIN_DENSITY_AMPLIFICATION_REALIZATION_KEY_PANEL_KEYS = HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_GAIN_AMPLIFICATION_REALIZATION_KEY_PANEL_KEYS + [
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_amplification_target_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_amplification_binding_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_overlay_png",
    "before_after_hard_anchor_peak_residual_overlap_gain_density_amplification_panel_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_amplification_veto_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_amplification_alignment_overlay_png",
]

HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_GAIN_DENSITY_LANDING_REALIZATION_KEY_PANEL_KEYS = HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_GAIN_DENSITY_AMPLIFICATION_REALIZATION_KEY_PANEL_KEYS + [
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_target_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_binding_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_realization_overlay_png",
    "before_after_hard_anchor_peak_residual_overlap_gain_density_landing_panel_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_veto_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_alignment_overlay_png",
]

HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_GAIN_DENSITY_LANDING_TO_COMPONENT_COLLAPSE_CONTRACT_KEY_PANEL_KEYS = HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_GAIN_DENSITY_LANDING_REALIZATION_KEY_PANEL_KEYS + [
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_target_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_binding_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_contract_overlay_png",
    "before_after_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_panel_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_veto_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_alignment_overlay_png",
]

HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_GAIN_DENSITY_LANDING_TO_COMPONENT_COLLAPSE_CONNECTIVITY_CONTRACT_KEY_PANEL_KEYS = HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_GAIN_DENSITY_LANDING_TO_COMPONENT_COLLAPSE_CONTRACT_KEY_PANEL_KEYS + [
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_target_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_binding_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_contract_overlay_png",
    "before_after_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_panel_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_veto_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_alignment_overlay_png",
]

HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_GAIN_DENSITY_LANDING_TO_COMPONENT_COLLAPSE_CONNECTIVITY_GRAPH_REALIZATION_KEY_PANEL_KEYS = HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_GAIN_DENSITY_LANDING_TO_COMPONENT_COLLAPSE_CONNECTIVITY_CONTRACT_KEY_PANEL_KEYS + [
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_target_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_binding_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_realization_overlay_png",
    "before_after_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_panel_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_veto_overlay_png",
    "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_alignment_overlay_png",
]

MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_FLOOR = 1.0e-3
MEANINGFUL_MEAN_OVERLAP_GAIN_FLOOR = 3.0e-4
MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_AMPLIFICATION_FLOOR = 1.5e-3
MEANINGFUL_MEAN_OVERLAP_GAIN_AMPLIFICATION_FLOOR = 5.0e-4
MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_AMPLIFICATION_FLOOR = 2.25e-3
MEANINGFUL_MEAN_OVERLAP_GAIN_DENSITY_AMPLIFICATION_FLOOR = 7.5e-4
MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_LANDING_FLOOR = 2.5e-3
MEANINGFUL_MEAN_OVERLAP_GAIN_DENSITY_LANDING_FLOOR = 8.5e-4
MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_FLOOR = 5.0e-4
MEANINGFUL_MEAN_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_FLOOR = 1.5e-4
MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_CONNECTIVITY_FLOOR = 1.0e-4
MEANINGFUL_MEAN_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_CONNECTIVITY_FLOOR = 3.5e-5
MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_CONNECTIVITY_GRAPH_FLOOR = 5.0e-5
MEANINGFUL_MEAN_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_CONNECTIVITY_GRAPH_FLOOR = 2.0e-5


def _is_peak_stabilization_family(*, family: str, variant: str) -> bool:
    return bool(
        "peak_stabilization" in family
        or variant == "stablelead_rehydrated_operator_peak_stabilization_v1"
    )


def _is_post_merge_realization_family(*, family: str, variant: str) -> bool:
    return bool(
        "post_merge_realization" in family
        or variant == "stablelead_rehydrated_operator_post_merge_realization_v1"
    )


def _is_component_consolidation_family(*, family: str, variant: str) -> bool:
    return bool(
        "component_consolidation_contract" in family
        or variant == "stablelead_rehydrated_operator_component_consolidation_contract_v1"
    )


def _is_component_adjacency_family(*, family: str, variant: str) -> bool:
    return bool(
        "component_adjacency_closure" in family
        or variant == "stablelead_rehydrated_operator_component_adjacency_closure_v1"
    )


def _is_component_closure_realization_family(*, family: str, variant: str) -> bool:
    return bool(
        "component_closure_realization" in family
        or variant == "stablelead_rehydrated_operator_component_closure_realization_v1"
    )


def _is_hard_anchor_closure_realization_family(*, family: str, variant: str) -> bool:
    return bool(
        "hard_anchor_closure_realization" in family
        or variant == "stablelead_rehydrated_operator_hard_anchor_closure_realization_v1"
    )


def _is_hard_anchor_breakout_alignment_family(*, family: str, variant: str) -> bool:
    return bool(
        "hard_anchor_breakout_alignment" in family
        or variant == "stablelead_rehydrated_operator_hard_anchor_breakout_alignment_v1"
    )


def _is_hard_anchor_peak_rebound_suppression_family(*, family: str, variant: str) -> bool:
    return bool(
        "hard_anchor_peak_rebound_suppression" in family
        or variant == "stablelead_rehydrated_operator_hard_anchor_peak_rebound_suppression_v1"
    )


def _is_hard_anchor_peak_residual_activation_family(*, family: str, variant: str) -> bool:
    return bool(
        "hard_anchor_peak_residual_activation" in family
        or variant == "stablelead_rehydrated_operator_hard_anchor_peak_residual_activation_v1"
    )


def _is_hard_anchor_peak_residual_contraction_realization_family(*, family: str, variant: str) -> bool:
    return bool(
        "hard_anchor_peak_residual_contraction_realization" in family
        or variant == "stablelead_rehydrated_operator_hard_anchor_peak_residual_contraction_realization_v1"
    )


def _is_hard_anchor_peak_residual_overlap_alignment_realization_family(*, family: str, variant: str) -> bool:
    return bool(
        "hard_anchor_peak_residual_overlap_alignment_realization" in family
        or variant == "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_alignment_realization_v1"
    )


def _is_hard_anchor_peak_residual_overlap_gain_realization_family(*, family: str, variant: str) -> bool:
    return bool(
        "hard_anchor_peak_residual_overlap_gain_realization" in family
        or variant == "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_realization_v1"
    )


def _is_hard_anchor_peak_residual_overlap_gain_amplification_realization_family(*, family: str, variant: str) -> bool:
    return bool(
        "hard_anchor_peak_residual_overlap_gain_amplification_realization" in family
        or variant == "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_amplification_realization_v1"
    )


def _is_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_family(*, family: str, variant: str) -> bool:
    return bool(
        "hard_anchor_peak_residual_overlap_gain_density_amplification_realization" in family
        or variant == "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_v1"
    )


def _is_hard_anchor_peak_residual_overlap_gain_density_landing_realization_family(*, family: str, variant: str) -> bool:
    return bool(
        "hard_anchor_peak_residual_overlap_gain_density_landing_realization" in family
        or variant == "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_realization_v1"
    )


def _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_contract_family(*, family: str, variant: str) -> bool:
    return bool(
        "hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_contract" in family
        or variant == "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_contract_v1"
    )


def _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_contract_family(*, family: str, variant: str) -> bool:
    return bool(
        "hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_contract" in family
        or variant == "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_contract_v1"
    )


def _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_realization_family(*, family: str, variant: str) -> bool:
    return bool(
        "hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_realization" in family
        or variant == "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_realization_v1"
    )


def _required_artifact_keys(*, family: str, variant: str) -> list[str]:
    if _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_realization_family(family=family, variant=variant):
        return list(HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_GAIN_DENSITY_LANDING_TO_COMPONENT_COLLAPSE_CONNECTIVITY_GRAPH_REALIZATION_KEY_PANEL_KEYS)
    if _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_contract_family(family=family, variant=variant):
        return list(HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_GAIN_DENSITY_LANDING_TO_COMPONENT_COLLAPSE_CONNECTIVITY_CONTRACT_KEY_PANEL_KEYS)
    if _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_contract_family(family=family, variant=variant):
        return list(HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_GAIN_DENSITY_LANDING_TO_COMPONENT_COLLAPSE_CONTRACT_KEY_PANEL_KEYS)
    if _is_hard_anchor_peak_residual_overlap_gain_density_landing_realization_family(family=family, variant=variant):
        return list(HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_GAIN_DENSITY_LANDING_REALIZATION_KEY_PANEL_KEYS)
    if _is_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_family(family=family, variant=variant):
        return list(HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_GAIN_DENSITY_AMPLIFICATION_REALIZATION_KEY_PANEL_KEYS)
    if _is_hard_anchor_peak_residual_overlap_gain_amplification_realization_family(family=family, variant=variant):
        return list(HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_GAIN_AMPLIFICATION_REALIZATION_KEY_PANEL_KEYS)
    if _is_hard_anchor_peak_residual_overlap_gain_realization_family(family=family, variant=variant):
        return list(HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_GAIN_REALIZATION_KEY_PANEL_KEYS)
    if _is_hard_anchor_peak_residual_overlap_alignment_realization_family(family=family, variant=variant):
        return list(HARD_ANCHOR_PEAK_RESIDUAL_OVERLAP_ALIGNMENT_REALIZATION_KEY_PANEL_KEYS)
    if _is_hard_anchor_peak_residual_contraction_realization_family(family=family, variant=variant):
        return list(HARD_ANCHOR_PEAK_RESIDUAL_CONTRACTION_REALIZATION_KEY_PANEL_KEYS)
    if _is_hard_anchor_peak_residual_activation_family(family=family, variant=variant):
        return list(HARD_ANCHOR_PEAK_RESIDUAL_ACTIVATION_KEY_PANEL_KEYS)
    if _is_hard_anchor_peak_rebound_suppression_family(family=family, variant=variant):
        return list(HARD_ANCHOR_PEAK_REBOUND_SUPPRESSION_KEY_PANEL_KEYS)
    if _is_hard_anchor_breakout_alignment_family(family=family, variant=variant):
        return list(HARD_ANCHOR_BREAKOUT_ALIGNMENT_KEY_PANEL_KEYS)
    if _is_hard_anchor_closure_realization_family(family=family, variant=variant):
        return list(HARD_ANCHOR_CLOSURE_REALIZATION_KEY_PANEL_KEYS)
    if _is_component_closure_realization_family(family=family, variant=variant):
        return list(COMPONENT_CLOSURE_REALIZATION_KEY_PANEL_KEYS)
    if _is_component_adjacency_family(family=family, variant=variant):
        return list(COMPONENT_ADJACENCY_KEY_PANEL_KEYS)
    if _is_component_consolidation_family(family=family, variant=variant):
        return list(COMPONENT_CONSOLIDATION_KEY_PANEL_KEYS)
    if _is_post_merge_realization_family(family=family, variant=variant):
        return list(POST_MERGE_REALIZATION_KEY_PANEL_KEYS)
    if _is_peak_stabilization_family(family=family, variant=variant):
        return list(PEAK_STABILIZATION_KEY_PANEL_KEYS)
    if "operator_merge" in family or variant in {
        "stablelead_rehydrated_operator_component_merge_rule_v1",
        "stablelead_rehydrated_operator_merge_binding_v1",
    }:
        return list(MERGE_RULE_KEY_PANEL_KEYS)
    return list(BASELINE_KEY_PANEL_KEYS)


def _repo_relative_string(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()).as_posix())
    except ValueError:
        return str(path.resolve().as_posix())


def _build_artifact_completeness(
    *,
    summary_path: Path,
    summary: dict,
    variant: str,
    family: str,
    key_panels_zip_path: Path | None,
) -> tuple[dict, dict]:
    base_dir = summary_path.parent
    candidate_rows = [row for row in summary.get("rows", []) if str(row.get("variant")) == variant]
    zip_members: list[str] = []
    if key_panels_zip_path is not None:
        key_panels_zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(key_panels_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            seen_members: set[str] = set()
            for row in candidate_rows:
                files = row.get("files", {})
                for key, relative in sorted(files.items()):
                    if not relative:
                        continue
                    if not (key.endswith("_png") or key.endswith("_json")):
                        continue
                    resolved = (base_dir / str(relative)).resolve()
                    if not resolved.exists():
                        continue
                    arcname = _repo_relative_string(Path(str(relative)))
                    if arcname in seen_members:
                        continue
                    zf.write(resolved, arcname=arcname)
                    seen_members.add(arcname)
        with zipfile.ZipFile(key_panels_zip_path, "r") as zf:
            zip_members = sorted(zf.namelist())
    zip_member_set = set(zip_members)

    required_keys = _required_artifact_keys(family=family, variant=variant)
    per_case = []
    all_listed_png_present = True
    all_required_present = True
    for row in candidate_rows:
        case_id = str(row.get("case_id"))
        files = dict(row.get("files", {}))
        listed_png_keys = sorted(key for key in files if key.endswith("_png"))
        expected_keys = sorted(set(required_keys + listed_png_keys))
        expected_panels = []
        existing_panels = []
        zipped_panels = []
        missing_panels = []
        null_required_panels = []
        listed_png_missing = []
        for key in expected_keys:
            relative = str(files.get(key) or "")
            item = {"key": key, "path": relative}
            expected_panels.append(item)
            if not relative:
                if key.endswith("_png"):
                    missing_panels.append(item)
                    listed_png_missing.append(item)
                    all_listed_png_present = False
                if key in required_keys:
                    if item not in missing_panels:
                        missing_panels.append(item)
                    null_required_panels.append(item)
                    all_required_present = False
                continue
            resolved = (base_dir / relative).resolve()
            exists = resolved.exists()
            if resolved.exists():
                existing_panels.append(item)
            else:
                missing_panels.append(item)
                if key.endswith("_png"):
                    listed_png_missing.append(item)
                    all_listed_png_present = False
                if key in required_keys:
                    all_required_present = False
            if relative.replace("\\", "/") in zip_member_set:
                zipped_panels.append(item)
            elif key in required_keys and exists:
                missing_panels.append({**item, "missing_in_zip": True})
                all_required_present = False
        per_case.append(
            {
                "case_id": case_id,
                "expected_panels": expected_panels,
                "existing_panels": existing_panels,
                "zipped_panels": zipped_panels,
                "missing_panels": missing_panels,
                "null_required_panels": null_required_panels,
                "listed_png_missing": listed_png_missing,
            }
        )

    artifact_incomplete_fail = bool(not all_required_present or not all_listed_png_present)
    completeness_report = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "family": family,
        "variant": variant,
        "summary_json": _repo_relative_string(summary_path),
        "key_panels_zip": (
            _repo_relative_string(key_panels_zip_path)
            if key_panels_zip_path is not None and key_panels_zip_path.exists()
            else ""
        ),
        "artifact_incomplete_fail": artifact_incomplete_fail,
        "all_listed_png_present": all_listed_png_present,
        "all_required_panels_present": all_required_present,
        "required_keys": required_keys,
        "per_case": per_case,
    }
    zip_manifest = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "family": family,
        "variant": variant,
        "zip_path": (
            _repo_relative_string(key_panels_zip_path)
            if key_panels_zip_path is not None and key_panels_zip_path.exists()
            else ""
        ),
        "member_count": len(zip_members),
        "members": zip_members,
        "per_case": [
            {
                "case_id": item["case_id"],
                "zipped_panels": item["zipped_panels"],
            }
            for item in per_case
        ],
    }
    return completeness_report, zip_manifest


def _artifact_type(row: dict) -> str:
    if row["fg_multilayer_overlap_ratio"] >= 0.12 or row["delta_fg_multilayer_overlap_ratio"] > 0.01:
        return "multilayer_residual"
    if row["fg_visible_component_count"] >= 4 or row["fg_visible_component_entropy"] >= 0.12 or row["delta_fg_visible_component_count"] > 0:
        return "multi_component_fragmentation"
    if row["fg_secondary_mass_ratio"] >= 0.10 or row["delta_fg_secondary_mass_ratio"] > 0.01:
        return "primary_secondary_lobe_competition"
    if row["fg_peak_count_after_render"] >= 20 or row["delta_fg_peak_count_after_render"] > 0:
        return "peak_rebound"
    return "mixed_render_artifact"


def _dominant_artifact(rows: list[dict]) -> str:
    scores: dict[str, float] = {}
    for row in rows:
        artifact = row["artifact_type"]
        severity = (
            max(row["delta_fg_peak_count_after_render"], 0.0)
            + max(row["delta_fg_visible_component_count"], 0.0)
            + max(row["delta_fg_visible_component_entropy"], 0.0) * 10.0
            + max(row["delta_fg_secondary_mass_ratio"], 0.0) * 10.0
            + max(row["delta_fg_multilayer_overlap_ratio"], 0.0) * 10.0
        )
        scores[artifact] = scores.get(artifact, 0.0) + severity
    if not scores:
        return "insufficient_evidence"
    return max(scores.items(), key=lambda item: item[1])[0]


def _smoke_case_pass(row: dict, *, family: str, variant: str) -> bool:
    guard = row.get("guard_report", {})
    if _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_realization_family(family=family, variant=variant):
        quality_guard = bool(
            row["delta_masked_l1"] <= 0.0
            and row["delta_masked_ssim"] >= 0.0
            and row["delta_off_body_support_ratio"] <= 0.0
            and not bool(guard.get("human_erasure_guard_violation", False))
            and bool(guard.get("visible_mass_floor", {}).get("pass", True))
            and bool(guard.get("largest_component_floor", {}).get("pass", True))
            and bool(guard.get("fg_coverage_floor", {}).get("pass", True))
            and bool(guard.get("masked_quality_guard", {}).get("pass", True))
        )
        if row.get("case_id") == "CoreView_390_frame_001170_Camera_B4":
            return bool(
                quality_guard
                and not bool(row.get("local_rewrite_without_merge_rebound", False))
                and row["delta_fg_connected_components"] < 0.0
                and row["delta_fg_peak_count_after_render"] <= 0.0
                and row["delta_hard_anchor_peak_residual"] <= 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_target_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_binding_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_realization_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_gain"] > MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_CONNECTIVITY_GRAPH_FLOOR
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_alignment_score"] > 0.0
            )
        return quality_guard
    if _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_contract_family(family=family, variant=variant):
        quality_guard = bool(
            row["delta_masked_l1"] <= 0.0
            and row["delta_masked_ssim"] >= 0.0
            and row["delta_off_body_support_ratio"] <= 0.0
            and not bool(guard.get("human_erasure_guard_violation", False))
            and bool(guard.get("visible_mass_floor", {}).get("pass", True))
            and bool(guard.get("largest_component_floor", {}).get("pass", True))
            and bool(guard.get("fg_coverage_floor", {}).get("pass", True))
            and bool(guard.get("masked_quality_guard", {}).get("pass", True))
        )
        if row.get("case_id") == "CoreView_390_frame_001170_Camera_B4":
            return bool(
                quality_guard
                and row["delta_fg_connected_components"] < 0.0
                and row["delta_fg_peak_count_after_render"] <= 0.0
                and row["delta_hard_anchor_peak_residual"] <= 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_target_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_binding_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_contract_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_gain"] > MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_CONNECTIVITY_FLOOR
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_alignment_score"] > 0.0
            )
        return quality_guard
    if _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_contract_family(family=family, variant=variant):
        quality_guard = bool(
            row["delta_masked_l1"] <= 0.0
            and row["delta_masked_ssim"] >= 0.0
            and row["delta_off_body_support_ratio"] <= 0.0
            and not bool(guard.get("human_erasure_guard_violation", False))
            and bool(guard.get("visible_mass_floor", {}).get("pass", True))
            and bool(guard.get("largest_component_floor", {}).get("pass", True))
            and bool(guard.get("fg_coverage_floor", {}).get("pass", True))
            and bool(guard.get("masked_quality_guard", {}).get("pass", True))
        )
        if row.get("case_id") == "CoreView_390_frame_001170_Camera_B4":
            return bool(
                quality_guard
                and row["delta_fg_connected_components"] < 0.0
                and row["delta_fg_peak_count_after_render"] <= 0.0
                and row["delta_hard_anchor_peak_residual"] <= 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_target_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_binding_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_contract_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_gain"] > MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_FLOOR
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_alignment_score"] > 0.0
            )
        return quality_guard
    if _is_hard_anchor_peak_residual_overlap_gain_density_landing_realization_family(family=family, variant=variant):
        quality_guard = bool(
            row["delta_masked_l1"] <= 0.0
            and row["delta_masked_ssim"] >= 0.0
            and row["delta_off_body_support_ratio"] <= 0.0
            and not bool(guard.get("human_erasure_guard_violation", False))
            and bool(guard.get("visible_mass_floor", {}).get("pass", True))
            and bool(guard.get("largest_component_floor", {}).get("pass", True))
            and bool(guard.get("fg_coverage_floor", {}).get("pass", True))
            and bool(guard.get("masked_quality_guard", {}).get("pass", True))
        )
        if row.get("case_id") == "CoreView_390_frame_001170_Camera_B4":
            return bool(
                quality_guard
                and row["delta_fg_connected_components"] < 0.0
                and row["delta_fg_peak_count_after_render"] <= 0.0
                and row["delta_hard_anchor_peak_residual"] <= 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_target_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_binding_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_realization_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_gain"] > MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_LANDING_FLOOR
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_alignment_score"] > 0.0
            )
        return quality_guard
    if _is_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_family(family=family, variant=variant):
        quality_guard = bool(
            row["delta_masked_l1"] <= 0.0
            and row["delta_masked_ssim"] >= 0.0
            and row["delta_off_body_support_ratio"] <= 0.0
            and not bool(guard.get("human_erasure_guard_violation", False))
            and bool(guard.get("visible_mass_floor", {}).get("pass", True))
            and bool(guard.get("largest_component_floor", {}).get("pass", True))
            and bool(guard.get("fg_coverage_floor", {}).get("pass", True))
            and bool(guard.get("masked_quality_guard", {}).get("pass", True))
        )
        if row.get("case_id") == "CoreView_390_frame_001170_Camera_B4":
            return bool(
                quality_guard
                and row["delta_fg_connected_components"] < 0.0
                and row["delta_fg_peak_count_after_render"] <= 0.0
                and row["delta_hard_anchor_peak_residual"] <= 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_amplification_target_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_amplification_binding_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_amplification_gain"] > MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_AMPLIFICATION_FLOOR
                and row["delta_hard_anchor_peak_residual_overlap_gain_density_amplification_alignment_score"] > 0.0
            )
        return quality_guard
    if _is_hard_anchor_peak_residual_overlap_gain_amplification_realization_family(family=family, variant=variant):
        quality_guard = bool(
            row["delta_masked_l1"] <= 0.0
            and row["delta_masked_ssim"] >= 0.0
            and row["delta_off_body_support_ratio"] <= 0.0
            and not bool(guard.get("human_erasure_guard_violation", False))
            and bool(guard.get("visible_mass_floor", {}).get("pass", True))
            and bool(guard.get("largest_component_floor", {}).get("pass", True))
            and bool(guard.get("fg_coverage_floor", {}).get("pass", True))
            and bool(guard.get("masked_quality_guard", {}).get("pass", True))
        )
        if row.get("case_id") == "CoreView_390_frame_001170_Camera_B4":
            return bool(
                quality_guard
                and row["delta_fg_connected_components"] < 0.0
                and row["delta_fg_peak_count_after_render"] <= 0.0
                and row["delta_hard_anchor_peak_residual"] <= 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_amplification_target_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_amplification_binding_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_amplification_realization_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_amplification_gain"] > MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_AMPLIFICATION_FLOOR
                and row["delta_hard_anchor_peak_residual_overlap_gain_amplification_alignment_score"] > 0.0
            )
        return quality_guard
    if _is_hard_anchor_peak_residual_overlap_gain_realization_family(family=family, variant=variant):
        quality_guard = bool(
            row["delta_masked_l1"] <= 0.0
            and row["delta_masked_ssim"] >= 0.0
            and row["delta_off_body_support_ratio"] <= 0.0
            and not bool(guard.get("human_erasure_guard_violation", False))
            and bool(guard.get("visible_mass_floor", {}).get("pass", True))
            and bool(guard.get("largest_component_floor", {}).get("pass", True))
            and bool(guard.get("fg_coverage_floor", {}).get("pass", True))
            and bool(guard.get("masked_quality_guard", {}).get("pass", True))
        )
        if row.get("case_id") == "CoreView_390_frame_001170_Camera_B4":
            return bool(
                quality_guard
                and row["delta_fg_connected_components"] < 0.0
                and row["delta_fg_peak_count_after_render"] <= 0.0
                and row["delta_hard_anchor_peak_residual"] <= 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_target_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_binding_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain_realization_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain"] > MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_FLOOR
                and row["delta_hard_anchor_peak_residual_overlap_gain_alignment_score"] > 0.0
            )
        return quality_guard
    if _is_hard_anchor_peak_residual_overlap_alignment_realization_family(family=family, variant=variant):
        quality_guard = bool(
            row["delta_masked_l1"] <= 0.0
            and row["delta_masked_ssim"] >= 0.0
            and row["delta_off_body_support_ratio"] <= 0.0
            and not bool(guard.get("human_erasure_guard_violation", False))
            and bool(guard.get("visible_mass_floor", {}).get("pass", True))
            and bool(guard.get("largest_component_floor", {}).get("pass", True))
            and bool(guard.get("fg_coverage_floor", {}).get("pass", True))
            and bool(guard.get("masked_quality_guard", {}).get("pass", True))
        )
        if row.get("case_id") == "CoreView_390_frame_001170_Camera_B4":
            return bool(
                quality_guard
                and row["delta_fg_connected_components"] < 0.0
                and row["delta_fg_peak_count_after_render"] <= 0.0
                and row["delta_hard_anchor_peak_residual"] <= 0.0
                and row["delta_hard_anchor_peak_residual_overlap_target_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_binding_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_realization_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_gain"] > 0.0
                and row["delta_hard_anchor_peak_residual_overlap_alignment_score"] > 0.0
            )
        return quality_guard
    if _is_hard_anchor_peak_residual_contraction_realization_family(family=family, variant=variant):
        quality_guard = bool(
            row["delta_masked_l1"] <= 0.0
            and row["delta_masked_ssim"] >= 0.0
            and row["delta_off_body_support_ratio"] <= 0.0
            and not bool(guard.get("human_erasure_guard_violation", False))
            and bool(guard.get("visible_mass_floor", {}).get("pass", True))
            and bool(guard.get("largest_component_floor", {}).get("pass", True))
            and bool(guard.get("fg_coverage_floor", {}).get("pass", True))
            and bool(guard.get("masked_quality_guard", {}).get("pass", True))
        )
        if row.get("case_id") == "CoreView_390_frame_001170_Camera_B4":
            return bool(
                quality_guard
                and row["delta_fg_connected_components"] < 0.0
                and row["delta_fg_peak_count_after_render"] <= 0.0
                and row["delta_hard_anchor_peak_residual"] <= 0.0
                and row["delta_hard_anchor_peak_residual_contraction_target_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_contraction_realization_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_contraction_overlap_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_contraction_gain"] > 0.0
                and row["delta_hard_anchor_peak_residual_contraction_alignment_score"] > 0.0
            )
        return quality_guard
    if _is_hard_anchor_peak_residual_activation_family(family=family, variant=variant):
        quality_guard = bool(
            row["delta_masked_l1"] <= 0.0
            and row["delta_masked_ssim"] >= 0.0
            and row["delta_off_body_support_ratio"] <= 0.0
            and not bool(guard.get("human_erasure_guard_violation", False))
            and bool(guard.get("visible_mass_floor", {}).get("pass", True))
            and bool(guard.get("largest_component_floor", {}).get("pass", True))
            and bool(guard.get("fg_coverage_floor", {}).get("pass", True))
            and bool(guard.get("masked_quality_guard", {}).get("pass", True))
        )
        if row.get("case_id") == "CoreView_390_frame_001170_Camera_B4":
            return bool(
                quality_guard
                and row["delta_fg_connected_components"] < 0.0
                and row["delta_fg_peak_count_after_render"] <= 0.0
                and row["delta_hard_anchor_peak_residual"] <= 0.0
                and row["delta_post_operator_component_merge_gain"] > 0.0
                and row["delta_hard_anchor_peak_rebound_contract_gain"] > 0.0
                and row["delta_hard_anchor_peak_residual_activation_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_binding_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_contraction_fraction"] > 0.0
                and row["delta_hard_anchor_peak_residual_activation_gain"] > 0.0
            )
        return quality_guard
    if _is_hard_anchor_peak_rebound_suppression_family(family=family, variant=variant):
        quality_guard = bool(
            row["delta_masked_l1"] <= 0.0
            and row["delta_masked_ssim"] >= 0.0
            and row["delta_off_body_support_ratio"] <= 0.0
            and not bool(guard.get("human_erasure_guard_violation", False))
            and bool(guard.get("visible_mass_floor", {}).get("pass", True))
            and bool(guard.get("largest_component_floor", {}).get("pass", True))
            and bool(guard.get("fg_coverage_floor", {}).get("pass", True))
            and bool(guard.get("masked_quality_guard", {}).get("pass", True))
        )
        if row.get("case_id") == "CoreView_390_frame_001170_Camera_B4":
            return bool(
                quality_guard
                and row["delta_fg_connected_components"] < 0.0
                and row["delta_fg_peak_count_after_render"] <= 0.0
                and row["delta_post_operator_component_merge_gain"] > 0.0
                and row["delta_hard_anchor_breakout_alignment_gain"] > 0.0
                and row["delta_hard_anchor_peak_rebound_focus_fraction"] > 0.0
                and row["delta_hard_anchor_peak_rebound_binding_fraction"] > 0.0
                and row["delta_hard_anchor_peak_rebound_suppression_fraction"] > 0.0
                and row["delta_hard_anchor_peak_rebound_contract_gain"] > 0.0
            )
        return quality_guard
    if _is_hard_anchor_breakout_alignment_family(family=family, variant=variant):
        quality_guard = bool(
            row["delta_masked_l1"] <= 0.0
            and row["delta_masked_ssim"] >= 0.0
            and row["delta_off_body_support_ratio"] <= 0.0
            and not bool(guard.get("human_erasure_guard_violation", False))
            and bool(guard.get("visible_mass_floor", {}).get("pass", True))
            and bool(guard.get("largest_component_floor", {}).get("pass", True))
            and bool(guard.get("fg_coverage_floor", {}).get("pass", True))
            and bool(guard.get("masked_quality_guard", {}).get("pass", True))
        )
        if row.get("case_id") == "CoreView_390_frame_001170_Camera_B4":
            return bool(
                quality_guard
                and row["delta_fg_connected_components"] < 0.0
                and row["delta_fg_peak_count_after_render"] <= 0.0
                and row["delta_post_operator_component_merge_gain"] > 0.0
                and row["delta_hard_anchor_breakout_alignment_fraction"] > 0.0
                and row["delta_hard_anchor_breakout_alignment_gain"] > 0.0
                and row["delta_hard_anchor_local_shrinkage_witness_fraction"] > 0.0
                and row["delta_hard_anchor_breakout_alignment_centroid_score"] > 0.0
            )
        return quality_guard
    if _is_hard_anchor_closure_realization_family(family=family, variant=variant):
        quality_guard = bool(
            row["delta_masked_l1"] <= 0.0
            and row["delta_masked_ssim"] >= 0.0
            and row["delta_off_body_support_ratio"] <= 0.0
            and not bool(guard.get("human_erasure_guard_violation", False))
            and bool(guard.get("visible_mass_floor", {}).get("pass", True))
            and bool(guard.get("largest_component_floor", {}).get("pass", True))
            and bool(guard.get("fg_coverage_floor", {}).get("pass", True))
            and bool(guard.get("masked_quality_guard", {}).get("pass", True))
        )
        if row.get("case_id") == "CoreView_390_frame_001170_Camera_B4":
            return bool(
                quality_guard
                and row["delta_fg_connected_components"] < 0.0
                and row["delta_fg_peak_count_after_render"] <= 0.0
                and row["delta_post_operator_component_merge_gain"] > 0.0
                and row["delta_hard_anchor_local_closure_fraction"] > 0.0
                and row["delta_hard_anchor_local_closure_gain"] > 0.0
                and row["delta_hard_anchor_local_closure_weight"] > 0.0
                and row["delta_hard_anchor_local_peak_rebound_suppression_gain"] > 0.0
            )
        return quality_guard
    if _is_component_closure_realization_family(family=family, variant=variant):
        return bool(
            row["delta_post_operator_component_merge_gain"] > 0.0
            and row["delta_post_merge_component_consolidation_gain"] > 0.0
            and row["delta_post_merge_component_realization_gain"] > 0.0
            and row["delta_post_merge_component_closure_realization_gain"] > 0.0
            and row["delta_post_merge_component_closure_realization_fraction"] > 0.0
            and row["delta_post_merge_component_closure_binding_fraction"] > 0.0
            and row["delta_fg_connected_components"] < 0.0
            and row["delta_fg_peak_count_after_render"] <= 0.0
            and row["delta_masked_l1"] <= 0.0
            and row["delta_masked_ssim"] >= 0.0
            and row["delta_off_body_support_ratio"] <= 0.0
            and not bool(guard.get("human_erasure_guard_violation", False))
            and bool(guard.get("visible_mass_floor", {}).get("pass", True))
            and bool(guard.get("largest_component_floor", {}).get("pass", True))
            and bool(guard.get("fg_coverage_floor", {}).get("pass", True))
            and bool(guard.get("masked_quality_guard", {}).get("pass", True))
        )
    if _is_component_adjacency_family(family=family, variant=variant):
        return bool(
            row["delta_post_operator_component_merge_gain"] > 0.0
            and row["delta_post_merge_component_consolidation_gain"] > 0.0
            and row["delta_post_merge_component_realization_gain"] > 0.0
            and row["delta_post_merge_component_adjacency_gain"] > 0.0
            and row["delta_post_merge_component_closure_realization_gain"] > 0.0
            and row["delta_fg_connected_components"] < 0.0
            and row["delta_fg_peak_count_after_render"] <= 0.0
            and row["delta_masked_l1"] <= 0.0
            and row["delta_masked_ssim"] >= 0.0
            and row["delta_off_body_support_ratio"] <= 0.0
            and not bool(guard.get("human_erasure_guard_violation", False))
            and bool(guard.get("visible_mass_floor", {}).get("pass", True))
            and bool(guard.get("largest_component_floor", {}).get("pass", True))
            and bool(guard.get("fg_coverage_floor", {}).get("pass", True))
            and bool(guard.get("masked_quality_guard", {}).get("pass", True))
        )
    if _is_component_consolidation_family(family=family, variant=variant):
        return bool(
            row["delta_post_operator_component_merge_gain"] > 0.0
            and row["delta_post_merge_component_consolidation_gain"] > 0.0
            and row["delta_post_merge_component_realization_gain"] > 0.0
            and row["delta_fg_connected_components"] < 0.0
            and row["delta_fg_peak_count_after_render"] <= 0.0
            and row["delta_masked_l1"] <= 0.0
            and row["delta_masked_ssim"] >= 0.0
            and row["delta_off_body_support_ratio"] <= 0.0
            and not bool(guard.get("human_erasure_guard_violation", False))
            and bool(guard.get("visible_mass_floor", {}).get("pass", True))
            and bool(guard.get("largest_component_floor", {}).get("pass", True))
            and bool(guard.get("fg_coverage_floor", {}).get("pass", True))
            and bool(guard.get("masked_quality_guard", {}).get("pass", True))
        )
    if _is_post_merge_realization_family(family=family, variant=variant):
        return bool(
            row["delta_post_operator_component_merge_gain"] > 0.0
            and row["delta_post_merge_component_realization_gain"] > 0.0
            and row["delta_fg_connected_components"] < 0.0
            and row["delta_fg_peak_count_after_render"] <= 0.0
            and row["delta_masked_l1"] <= 0.0
            and row["delta_masked_ssim"] >= 0.0
            and row["delta_off_body_support_ratio"] <= 0.0
            and not bool(guard.get("human_erasure_guard_violation", False))
            and bool(guard.get("visible_mass_floor", {}).get("pass", True))
            and bool(guard.get("largest_component_floor", {}).get("pass", True))
            and bool(guard.get("fg_coverage_floor", {}).get("pass", True))
            and bool(guard.get("masked_quality_guard", {}).get("pass", True))
        )
    if _is_peak_stabilization_family(family=family, variant=variant):
        return bool(
            row["delta_post_operator_component_merge_gain"] > 0.0
            and row["delta_post_operator_peak_gain"] > 0.0
            and row["delta_fg_peak_count_after_render"] < 0.0
            and row["delta_fg_connected_components"] < 0.0
            and row["delta_masked_l1"] <= 0.0
            and row["delta_masked_ssim"] >= 0.0
            and row["delta_off_body_support_ratio"] <= 0.0
            and not bool(guard.get("human_erasure_guard_violation", False))
            and bool(guard.get("visible_mass_floor", {}).get("pass", True))
            and bool(guard.get("largest_component_floor", {}).get("pass", True))
            and bool(guard.get("fg_coverage_floor", {}).get("pass", True))
            and bool(guard.get("masked_quality_guard", {}).get("pass", True))
        )
    return bool(
        row["delta_resolved_to_primary_fraction"] > 0.0
        and row["delta_post_operator_component_merge_gain"] > 0.0
        and row["delta_fg_peak_count_after_render"] < 0.0
        and row["delta_fg_connected_components"] < 0.0
        and row["delta_masked_l1"] <= 0.0
        and row["delta_masked_ssim"] >= 0.0
        and row["delta_off_body_support_ratio"] <= 0.0
        and not bool(guard.get("human_erasure_guard_violation", False))
        and bool(guard.get("visible_mass_floor", {}).get("pass", True))
        and bool(guard.get("largest_component_floor", {}).get("pass", True))
        and bool(guard.get("fg_coverage_floor", {}).get("pass", True))
        and bool(guard.get("masked_quality_guard", {}).get("pass", True))
    )


def _control_case_pass(row: dict) -> bool:
    guard = row.get("guard_report", {})
    return bool(
        row.get("smoke_case_pass", False)
        and not bool(guard.get("human_erasure_guard_violation", False))
        and not bool(guard.get("shard_only_collapse_suspected", False))
    )


def _local_rewrite_without_merge_rebound(row: dict) -> bool:
    local_rewrite = bool(
        abs(float(row.get("delta_primary_owned_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_secondary_owned_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_conflict_resolved_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_merge_intent_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_merge_adoption_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_peak_stabilization_intent_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_post_merge_realization_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_post_merge_realization_intent_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_post_merge_realization_binding_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_post_merge_realization_adoption_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_post_merge_component_binding_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_post_merge_component_consolidation_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_post_merge_component_adjacency_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_post_merge_component_gap_closure_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_post_merge_component_closure_realization_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_post_merge_component_closure_binding_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_post_merge_component_consolidation_gain", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_local_closure_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_local_closure_gain", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_local_closure_residual_split_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_local_peak_rebound_suppression_gain", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_local_closure_weight", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_breakout_target_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_breakout_alignment_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_local_shrinkage_witness_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_breakout_alignment_gain", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_rebound_focus_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_rebound_binding_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_rebound_suppression_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_rebound_contract_gain", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_activation_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_binding_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_contraction_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_activation_gain", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_contraction_target_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_contraction_binding_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_contraction_realization_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_contraction_overlap_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_contraction_gain", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_contraction_alignment_score", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_target_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_binding_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_realization_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_alignment_score", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_target_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_binding_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_realization_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_alignment_score", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_remaining_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_target_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_binding_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_realization_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_gain", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_alignment_score", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_remaining_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_target_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_binding_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_gain", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_alignment_score", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_remaining_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_target_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_binding_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_realization_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_gain", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_alignment_score", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_remaining_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_target_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_binding_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_contract_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_gain", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_alignment_score", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_remaining_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_target_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_binding_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_contract_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_gain", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_alignment_score", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_remaining_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_target_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_binding_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_realization_fraction", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_gain", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_alignment_score", 0.0))) > 0.02
        or abs(float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_remaining_fraction", 0.0))) > 0.02
    )
    component_or_peak_rebound = bool(
        float(row.get("delta_post_operator_component_merge_gain", 0.0)) <= 0.0
        or float(row.get("delta_fg_peak_count_after_render", 0.0)) >= 0.0
        or float(row.get("delta_post_merge_component_realization_gain", 0.0)) <= 0.0
        or float(row.get("delta_post_merge_component_consolidation_gain", 0.0)) <= 0.0
        or float(row.get("delta_post_merge_component_adjacency_gain", 0.0)) <= 0.0
        or float(row.get("delta_post_merge_component_closure_realization_gain", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_local_closure_gain", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_local_peak_rebound_suppression_gain", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_breakout_alignment_gain", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_rebound_contract_gain", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_activation_gain", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_contraction_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_contraction_target_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_contraction_realization_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_contraction_overlap_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_contraction_gain", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_contraction_alignment_score", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_target_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_binding_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_realization_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_alignment_score", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_target_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_binding_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_realization_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain", 0.0)) <= MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_FLOOR
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_alignment_score", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_target_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_binding_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_realization_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_gain", 0.0)) <= MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_AMPLIFICATION_FLOOR
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_alignment_score", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_target_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_binding_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_gain", 0.0)) <= MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_AMPLIFICATION_FLOOR
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_alignment_score", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_target_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_binding_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_realization_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_gain", 0.0)) <= MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_LANDING_FLOOR
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_alignment_score", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_target_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_binding_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_contract_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_gain", 0.0)) <= MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_FLOOR
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_alignment_score", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_target_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_binding_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_contract_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_gain", 0.0)) <= MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_CONNECTIVITY_FLOOR
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_alignment_score", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_target_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_binding_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_realization_fraction", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_gain", 0.0)) <= MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_CONNECTIVITY_GRAPH_FLOOR
        or float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_alignment_score", 0.0)) <= 0.0
        or float(row.get("delta_hard_anchor_peak_residual", 0.0)) > 0.0
    )
    no_real_merge_gain = bool(
        float(row.get("delta_post_operator_component_merge_gain", 0.0)) <= 0.0
        or (
            float(row.get("delta_post_operator_peak_gain", 0.0)) <= 0.0
            and float(row.get("delta_post_merge_component_realization_gain", 0.0)) <= 0.0
            and float(row.get("delta_post_merge_component_consolidation_gain", 0.0)) <= 0.0
            and float(row.get("delta_post_merge_component_adjacency_gain", 0.0)) <= 0.0
            and float(row.get("delta_post_merge_component_closure_realization_gain", 0.0)) <= 0.0
            and float(row.get("delta_hard_anchor_local_closure_gain", 0.0)) <= 0.0
            and float(row.get("delta_hard_anchor_local_peak_rebound_suppression_gain", 0.0)) <= 0.0
            and float(row.get("delta_hard_anchor_breakout_alignment_gain", 0.0)) <= 0.0
            and float(row.get("delta_hard_anchor_peak_rebound_contract_gain", 0.0)) <= 0.0
            and float(row.get("delta_hard_anchor_peak_residual_activation_gain", 0.0)) <= 0.0
            and float(row.get("delta_hard_anchor_peak_residual_contraction_gain", 0.0)) <= 0.0
            and float(row.get("delta_hard_anchor_peak_residual_overlap_gain", 0.0)) <= 0.0
            and float(row.get("delta_hard_anchor_peak_residual_overlap_gain", 0.0)) <= MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_FLOOR
            and float(row.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_gain", 0.0)) <= 0.0
            and float(row.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_gain", 0.0)) <= MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_AMPLIFICATION_FLOOR
            and float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_gain", 0.0)) <= 0.0
            and float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_gain", 0.0)) <= MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_AMPLIFICATION_FLOOR
            and float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_gain", 0.0)) <= 0.0
            and float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_gain", 0.0)) <= MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_LANDING_FLOOR
            and float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_gain", 0.0)) <= 0.0
            and float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_gain", 0.0)) <= MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_FLOOR
            and float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_gain", 0.0)) <= 0.0
            and float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_gain", 0.0)) <= MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_CONNECTIVITY_FLOOR
            and float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_gain", 0.0)) <= 0.0
            and float(row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_gain", 0.0)) <= MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_CONNECTIVITY_GRAPH_FLOOR
        )
    )
    return bool(local_rewrite and component_or_peak_rebound and no_real_merge_gain)


def _anchor_component_breakout(rows: list[dict]) -> dict:
    anchor_cases = {
        "CoreView_390_frame_001170_Camera_B4",
        "CoreView_390_frame_000600_Camera_B4",
    }
    improved_cases = [
        row["case_id"]
        for row in rows
        if row["case_id"] in anchor_cases and float(row["delta_fg_connected_components"]) < 0.0
    ]
    return {
        "anchor_cases": sorted(anchor_cases),
        "improved_cases": improved_cases,
        "pass": bool(improved_cases),
    }


def _hard_anchor_operator_core_guard(rows: list[dict], *, family: str, variant: str) -> dict:
    anchor_case_id = "CoreView_390_frame_001170_Camera_B4"
    anchor_row = next((row for row in rows if row["case_id"] == anchor_case_id), None)
    if anchor_row is None:
        return {
            "anchor_case": anchor_case_id,
            "present": False,
            "local_rewrite_without_merge_rebound": False,
            "pass": False,
        }
    local_rewrite_without_merge_rebound = _local_rewrite_without_merge_rebound(anchor_row)
    anchor_component_merge_gain = float(anchor_row.get("delta_post_operator_component_merge_gain", 0.0))
    anchor_peak_gain = float(anchor_row.get("delta_post_operator_peak_gain", 0.0))
    anchor_peak_count = float(anchor_row.get("delta_fg_peak_count_after_render", 0.0))
    anchor_connected_components = float(anchor_row.get("delta_fg_connected_components", 0.0))
    anchor_realization_gain = float(anchor_row.get("delta_post_merge_component_realization_gain", 0.0))
    anchor_consolidation_gain = float(anchor_row.get("delta_post_merge_component_consolidation_gain", 0.0))
    anchor_adjacency_gain = float(anchor_row.get("delta_post_merge_component_adjacency_gain", 0.0))
    anchor_closure_realization_gain = float(anchor_row.get("delta_post_merge_component_closure_realization_gain", 0.0))
    anchor_closure_realization_fraction = float(anchor_row.get("delta_post_merge_component_closure_realization_fraction", 0.0))
    anchor_closure_binding_fraction = float(anchor_row.get("delta_post_merge_component_closure_binding_fraction", 0.0))
    anchor_closure_residual_split_fraction = float(anchor_row.get("delta_post_merge_component_closure_residual_split_fraction", 0.0))
    anchor_closure_breakout_score = float(anchor_row.get("delta_hard_anchor_component_closure_realization_breakout_score", 0.0))
    anchor_local_closure_fraction = float(anchor_row.get("delta_hard_anchor_local_closure_fraction", 0.0))
    anchor_local_closure_gain = float(anchor_row.get("delta_hard_anchor_local_closure_gain", 0.0))
    anchor_local_closure_residual_split_fraction = float(anchor_row.get("delta_hard_anchor_local_closure_residual_split_fraction", 0.0))
    anchor_local_peak_rebound_suppression_gain = float(anchor_row.get("delta_hard_anchor_local_peak_rebound_suppression_gain", 0.0))
    anchor_local_closure_weight = float(anchor_row.get("delta_hard_anchor_local_closure_weight", 0.0))
    anchor_breakout_target_fraction = float(anchor_row.get("delta_hard_anchor_breakout_target_fraction", 0.0))
    anchor_breakout_alignment_fraction = float(anchor_row.get("delta_hard_anchor_breakout_alignment_fraction", 0.0))
    anchor_local_shrinkage_witness_fraction = float(anchor_row.get("delta_hard_anchor_local_shrinkage_witness_fraction", 0.0))
    anchor_breakout_misalignment_veto_fraction = float(anchor_row.get("delta_hard_anchor_breakout_misalignment_veto_fraction", 0.0))
    anchor_breakout_alignment_centroid_score = float(anchor_row.get("delta_hard_anchor_breakout_alignment_centroid_score", 0.0))
    anchor_breakout_alignment_gain = float(anchor_row.get("delta_hard_anchor_breakout_alignment_gain", 0.0))
    anchor_peak_rebound_focus_fraction = float(anchor_row.get("delta_hard_anchor_peak_rebound_focus_fraction", 0.0))
    anchor_peak_rebound_binding_fraction = float(anchor_row.get("delta_hard_anchor_peak_rebound_binding_fraction", 0.0))
    anchor_peak_rebound_veto_fraction = float(anchor_row.get("delta_hard_anchor_peak_rebound_veto_fraction", 0.0))
    anchor_peak_rebound_suppression_fraction = float(anchor_row.get("delta_hard_anchor_peak_rebound_suppression_fraction", 0.0))
    anchor_peak_rebound_residual_fraction = float(anchor_row.get("delta_hard_anchor_peak_rebound_residual_fraction", 0.0))
    anchor_peak_rebound_contract_gain = float(anchor_row.get("delta_hard_anchor_peak_rebound_contract_gain", 0.0))
    anchor_peak_residual_activation_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_activation_fraction", 0.0))
    anchor_peak_residual_binding_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_binding_fraction", 0.0))
    anchor_peak_residual_veto_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_veto_fraction", 0.0))
    anchor_peak_residual_contraction_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_contraction_fraction", 0.0))
    anchor_peak_residual_activation_gain = float(anchor_row.get("delta_hard_anchor_peak_residual_activation_gain", 0.0))
    anchor_peak_residual_remaining_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_remaining_fraction", 0.0))
    anchor_peak_residual_contraction_target_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_contraction_target_fraction", 0.0))
    anchor_peak_residual_contraction_binding_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_contraction_binding_fraction", 0.0))
    anchor_peak_residual_contraction_realization_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_contraction_realization_fraction", 0.0))
    anchor_peak_residual_contraction_overlap_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_contraction_overlap_fraction", 0.0))
    anchor_peak_residual_contraction_gain = float(anchor_row.get("delta_hard_anchor_peak_residual_contraction_gain", 0.0))
    anchor_peak_residual_contraction_alignment_score = float(anchor_row.get("delta_hard_anchor_peak_residual_contraction_alignment_score", 0.0))
    anchor_peak_residual_overlap_target_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_target_fraction", 0.0))
    anchor_peak_residual_overlap_binding_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_binding_fraction", 0.0))
    anchor_peak_residual_overlap_realization_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_realization_fraction", 0.0))
    anchor_peak_residual_overlap_gain = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain", 0.0))
    anchor_peak_residual_overlap_alignment_score = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_alignment_score", 0.0))
    anchor_peak_residual_overlap_gain_target_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_target_fraction", 0.0))
    anchor_peak_residual_overlap_gain_binding_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_binding_fraction", 0.0))
    anchor_peak_residual_overlap_gain_realization_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_realization_fraction", 0.0))
    anchor_peak_residual_overlap_gain_alignment_score = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_alignment_score", 0.0))
    anchor_peak_residual_overlap_gain_remaining_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_remaining_fraction", 0.0))
    anchor_peak_residual_overlap_gain_amplification_target_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_target_fraction", 0.0))
    anchor_peak_residual_overlap_gain_amplification_binding_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_binding_fraction", 0.0))
    anchor_peak_residual_overlap_gain_amplification_realization_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_realization_fraction", 0.0))
    anchor_peak_residual_overlap_gain_amplification_gain = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_gain", 0.0))
    anchor_peak_residual_overlap_gain_amplification_alignment_score = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_alignment_score", 0.0))
    anchor_peak_residual_overlap_gain_amplification_remaining_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_remaining_fraction", 0.0))
    anchor_peak_residual_overlap_gain_density_amplification_target_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_target_fraction", 0.0))
    anchor_peak_residual_overlap_gain_density_amplification_binding_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_binding_fraction", 0.0))
    anchor_peak_residual_overlap_gain_density_amplification_realization_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_fraction", 0.0))
    anchor_peak_residual_overlap_gain_density_amplification_gain = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_gain", 0.0))
    anchor_peak_residual_overlap_gain_density_amplification_alignment_score = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_alignment_score", 0.0))
    anchor_peak_residual_overlap_gain_density_amplification_remaining_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_remaining_fraction", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_target_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_target_fraction", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_binding_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_binding_fraction", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_realization_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_realization_fraction", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_gain = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_gain", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_alignment_score = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_alignment_score", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_remaining_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_remaining_fraction", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_collapse_target_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_target_fraction", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_collapse_binding_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_binding_fraction", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_collapse_contract_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_contract_fraction", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_collapse_gain = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_gain", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_collapse_alignment_score = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_alignment_score", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_collapse_remaining_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_remaining_fraction", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_target_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_target_fraction", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_binding_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_binding_fraction", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_contract_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_contract_fraction", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_gain = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_gain", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_alignment_score = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_alignment_score", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_remaining_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_remaining_fraction", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_target_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_target_fraction", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_binding_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_binding_fraction", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_realization_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_realization_fraction", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_gain = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_gain", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_alignment_score = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_alignment_score", 0.0))
    anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_remaining_fraction = float(anchor_row.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_remaining_fraction", 0.0))
    return {
        "anchor_case": anchor_case_id,
        "present": True,
        "local_rewrite_without_merge_rebound": bool(local_rewrite_without_merge_rebound),
        "delta_fg_connected_components": anchor_connected_components,
        "delta_fg_peak_count_after_render": anchor_peak_count,
        "delta_post_operator_component_merge_gain": anchor_component_merge_gain,
        "delta_post_operator_peak_gain": anchor_peak_gain,
        "delta_post_merge_peak_containment_gain": float(anchor_row.get("delta_post_merge_peak_containment_gain", 0.0)),
        "delta_post_merge_component_realization_gain": anchor_realization_gain,
        "delta_post_merge_component_binding_fraction": float(anchor_row.get("delta_post_merge_component_binding_fraction", 0.0)),
        "delta_post_merge_component_consolidation_fraction": float(anchor_row.get("delta_post_merge_component_consolidation_fraction", 0.0)),
        "delta_post_merge_component_adjacency_fraction": float(anchor_row.get("delta_post_merge_component_adjacency_fraction", 0.0)),
        "delta_post_merge_component_gap_closure_fraction": float(anchor_row.get("delta_post_merge_component_gap_closure_fraction", 0.0)),
        "delta_post_merge_component_consolidation_gain": anchor_consolidation_gain,
        "delta_post_merge_component_adjacency_gain": anchor_adjacency_gain,
        "delta_post_merge_component_closure_realization_gain": anchor_closure_realization_gain,
        "delta_post_merge_component_adjacency_residual_split_fraction": float(anchor_row.get("delta_post_merge_component_adjacency_residual_split_fraction", 0.0)),
        "delta_post_merge_component_closure_realization_fraction": anchor_closure_realization_fraction,
        "delta_post_merge_component_closure_binding_fraction": anchor_closure_binding_fraction,
        "delta_post_merge_component_closure_residual_split_fraction": anchor_closure_residual_split_fraction,
        "delta_hard_anchor_local_closure_fraction": anchor_local_closure_fraction,
        "delta_hard_anchor_local_closure_gain": anchor_local_closure_gain,
        "delta_hard_anchor_local_closure_residual_split_fraction": anchor_local_closure_residual_split_fraction,
        "delta_hard_anchor_local_peak_rebound_suppression_gain": anchor_local_peak_rebound_suppression_gain,
        "delta_hard_anchor_local_closure_weight": anchor_local_closure_weight,
        "delta_hard_anchor_breakout_target_fraction": anchor_breakout_target_fraction,
        "delta_hard_anchor_breakout_alignment_fraction": anchor_breakout_alignment_fraction,
        "delta_hard_anchor_local_shrinkage_witness_fraction": anchor_local_shrinkage_witness_fraction,
        "delta_hard_anchor_breakout_misalignment_veto_fraction": anchor_breakout_misalignment_veto_fraction,
        "delta_hard_anchor_breakout_alignment_centroid_score": anchor_breakout_alignment_centroid_score,
        "delta_hard_anchor_breakout_alignment_gain": anchor_breakout_alignment_gain,
        "delta_hard_anchor_peak_rebound_focus_fraction": anchor_peak_rebound_focus_fraction,
        "delta_hard_anchor_peak_rebound_binding_fraction": anchor_peak_rebound_binding_fraction,
        "delta_hard_anchor_peak_rebound_veto_fraction": anchor_peak_rebound_veto_fraction,
        "delta_hard_anchor_peak_rebound_suppression_fraction": anchor_peak_rebound_suppression_fraction,
        "delta_hard_anchor_peak_rebound_residual_fraction": anchor_peak_rebound_residual_fraction,
        "delta_hard_anchor_peak_rebound_contract_gain": anchor_peak_rebound_contract_gain,
        "delta_hard_anchor_peak_residual_activation_fraction": anchor_peak_residual_activation_fraction,
        "delta_hard_anchor_peak_residual_binding_fraction": anchor_peak_residual_binding_fraction,
        "delta_hard_anchor_peak_residual_veto_fraction": anchor_peak_residual_veto_fraction,
        "delta_hard_anchor_peak_residual_contraction_fraction": anchor_peak_residual_contraction_fraction,
        "delta_hard_anchor_peak_residual_activation_gain": anchor_peak_residual_activation_gain,
        "delta_hard_anchor_peak_residual_remaining_fraction": anchor_peak_residual_remaining_fraction,
        "delta_hard_anchor_peak_residual_contraction_target_fraction": anchor_peak_residual_contraction_target_fraction,
        "delta_hard_anchor_peak_residual_contraction_binding_fraction": anchor_peak_residual_contraction_binding_fraction,
        "delta_hard_anchor_peak_residual_contraction_realization_fraction": anchor_peak_residual_contraction_realization_fraction,
        "delta_hard_anchor_peak_residual_contraction_overlap_fraction": anchor_peak_residual_contraction_overlap_fraction,
        "delta_hard_anchor_peak_residual_contraction_gain": anchor_peak_residual_contraction_gain,
        "delta_hard_anchor_peak_residual_contraction_alignment_score": anchor_peak_residual_contraction_alignment_score,
        "delta_hard_anchor_peak_residual_overlap_target_fraction": anchor_peak_residual_overlap_target_fraction,
        "delta_hard_anchor_peak_residual_overlap_binding_fraction": anchor_peak_residual_overlap_binding_fraction,
        "delta_hard_anchor_peak_residual_overlap_realization_fraction": anchor_peak_residual_overlap_realization_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain": anchor_peak_residual_overlap_gain,
        "delta_hard_anchor_peak_residual_overlap_alignment_score": anchor_peak_residual_overlap_alignment_score,
        "delta_hard_anchor_peak_residual_overlap_gain_target_fraction": anchor_peak_residual_overlap_gain_target_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_binding_fraction": anchor_peak_residual_overlap_gain_binding_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_realization_fraction": anchor_peak_residual_overlap_gain_realization_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_alignment_score": anchor_peak_residual_overlap_gain_alignment_score,
        "delta_hard_anchor_peak_residual_overlap_gain_remaining_fraction": anchor_peak_residual_overlap_gain_remaining_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_amplification_target_fraction": anchor_peak_residual_overlap_gain_amplification_target_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_amplification_binding_fraction": anchor_peak_residual_overlap_gain_amplification_binding_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_amplification_realization_fraction": anchor_peak_residual_overlap_gain_amplification_realization_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_amplification_gain": anchor_peak_residual_overlap_gain_amplification_gain,
        "delta_hard_anchor_peak_residual_overlap_gain_amplification_alignment_score": anchor_peak_residual_overlap_gain_amplification_alignment_score,
        "delta_hard_anchor_peak_residual_overlap_gain_amplification_remaining_fraction": anchor_peak_residual_overlap_gain_amplification_remaining_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_density_amplification_target_fraction": anchor_peak_residual_overlap_gain_density_amplification_target_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_density_amplification_binding_fraction": anchor_peak_residual_overlap_gain_density_amplification_binding_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_fraction": anchor_peak_residual_overlap_gain_density_amplification_realization_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_density_amplification_gain": anchor_peak_residual_overlap_gain_density_amplification_gain,
        "delta_hard_anchor_peak_residual_overlap_gain_density_amplification_alignment_score": anchor_peak_residual_overlap_gain_density_amplification_alignment_score,
        "delta_hard_anchor_peak_residual_overlap_gain_density_amplification_remaining_fraction": anchor_peak_residual_overlap_gain_density_amplification_remaining_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_target_fraction": anchor_peak_residual_overlap_gain_density_landing_target_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_binding_fraction": anchor_peak_residual_overlap_gain_density_landing_binding_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_realization_fraction": anchor_peak_residual_overlap_gain_density_landing_realization_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_gain": anchor_peak_residual_overlap_gain_density_landing_gain,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_alignment_score": anchor_peak_residual_overlap_gain_density_landing_alignment_score,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_remaining_fraction": anchor_peak_residual_overlap_gain_density_landing_remaining_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_target_fraction": anchor_peak_residual_overlap_gain_density_landing_collapse_target_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_binding_fraction": anchor_peak_residual_overlap_gain_density_landing_collapse_binding_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_contract_fraction": anchor_peak_residual_overlap_gain_density_landing_collapse_contract_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_gain": anchor_peak_residual_overlap_gain_density_landing_collapse_gain,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_alignment_score": anchor_peak_residual_overlap_gain_density_landing_collapse_alignment_score,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_remaining_fraction": anchor_peak_residual_overlap_gain_density_landing_collapse_remaining_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_target_fraction": anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_target_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_binding_fraction": anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_binding_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_contract_fraction": anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_contract_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_gain": anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_gain,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_alignment_score": anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_alignment_score,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_remaining_fraction": anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_remaining_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_target_fraction": anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_target_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_binding_fraction": anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_binding_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_realization_fraction": anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_realization_fraction,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_gain": anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_gain,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_alignment_score": anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_alignment_score,
        "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_remaining_fraction": anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_remaining_fraction,
        "delta_post_merge_component_residual_split_fraction": float(anchor_row.get("delta_post_merge_component_residual_split_fraction", 0.0)),
        "delta_post_merge_peak_residual_after_realization": float(anchor_row.get("delta_post_merge_peak_residual_after_realization", 0.0)),
        "delta_hard_anchor_peak_residual": float(anchor_row.get("delta_hard_anchor_peak_residual", 0.0)),
        "delta_hard_anchor_realization_breakout_score": float(anchor_row.get("delta_hard_anchor_realization_breakout_score", 0.0)),
        "delta_hard_anchor_component_consolidation_breakout_score": float(anchor_row.get("delta_hard_anchor_component_consolidation_breakout_score", 0.0)),
        "delta_hard_anchor_component_adjacency_breakout_score": float(anchor_row.get("delta_hard_anchor_component_adjacency_breakout_score", 0.0)),
        "delta_hard_anchor_component_closure_realization_breakout_score": anchor_closure_breakout_score,
        "pass": bool(
            (
                not local_rewrite_without_merge_rebound
                and anchor_peak_count <= (
                    0.0
                    if (
                        _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_realization_family(family=family, variant=variant)
                        or
                        _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_contract_family(family=family, variant=variant)
                        or
                        _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_contract_family(family=family, variant=variant)
                        or
                        _is_hard_anchor_peak_residual_overlap_gain_density_landing_realization_family(family=family, variant=variant)
                        or
                        _is_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_family(family=family, variant=variant)
                        or
                        _is_hard_anchor_peak_residual_overlap_gain_amplification_realization_family(family=family, variant=variant)
                        or
                        _is_hard_anchor_peak_residual_overlap_gain_realization_family(family=family, variant=variant)
                        or
                        _is_hard_anchor_peak_residual_overlap_alignment_realization_family(family=family, variant=variant)
                        or
                        _is_hard_anchor_peak_residual_contraction_realization_family(family=family, variant=variant)
                        or
                        _is_hard_anchor_peak_residual_activation_family(family=family, variant=variant)
                        or
                        _is_hard_anchor_peak_rebound_suppression_family(family=family, variant=variant)
                        or
                        _is_hard_anchor_breakout_alignment_family(family=family, variant=variant)
                        or
                        _is_component_closure_realization_family(family=family, variant=variant)
                        or
                        _is_post_merge_realization_family(family=family, variant=variant)
                        or _is_component_consolidation_family(family=family, variant=variant)
                        or _is_component_adjacency_family(family=family, variant=variant)
                    )
                    else -1e-12
                )
                and (
                    _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_realization_family(family=family, variant=variant)
                    or
                    _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_contract_family(family=family, variant=variant)
                    or
                    _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_contract_family(family=family, variant=variant)
                    or
                    _is_hard_anchor_peak_residual_overlap_gain_density_landing_realization_family(family=family, variant=variant)
                    or
                    _is_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_family(family=family, variant=variant)
                    or
                    _is_hard_anchor_peak_residual_overlap_gain_amplification_realization_family(family=family, variant=variant)
                    or
                    _is_hard_anchor_peak_residual_overlap_gain_realization_family(family=family, variant=variant)
                    or
                    _is_hard_anchor_peak_residual_overlap_alignment_realization_family(family=family, variant=variant)
                    or
                    _is_hard_anchor_peak_residual_contraction_realization_family(family=family, variant=variant)
                    or anchor_component_merge_gain > 0.0
                )
                and (
                    (
                        _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_realization_family(family=family, variant=variant)
                        and anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_target_fraction > 0.0
                        and anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_binding_fraction > 0.0
                        and anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_realization_fraction > 0.0
                        and anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_gain > MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_CONNECTIVITY_GRAPH_FLOOR
                        and anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_alignment_score > 0.0
                        and float(anchor_row.get("delta_hard_anchor_peak_residual", 0.0)) <= 0.0
                        and anchor_connected_components < 0.0
                    )
                    or
                    (
                        _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_contract_family(family=family, variant=variant)
                        and anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_target_fraction > 0.0
                        and anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_binding_fraction > 0.0
                        and anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_contract_fraction > 0.0
                        and anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_gain > MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_CONNECTIVITY_FLOOR
                        and anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_alignment_score > 0.0
                        and float(anchor_row.get("delta_hard_anchor_peak_residual", 0.0)) <= 0.0
                        and anchor_connected_components < 0.0
                    )
                    or
                    (
                        _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_contract_family(family=family, variant=variant)
                        and anchor_peak_residual_overlap_gain_density_landing_collapse_target_fraction > 0.0
                        and anchor_peak_residual_overlap_gain_density_landing_collapse_binding_fraction > 0.0
                        and anchor_peak_residual_overlap_gain_density_landing_collapse_contract_fraction > 0.0
                        and anchor_peak_residual_overlap_gain_density_landing_collapse_gain > MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_FLOOR
                        and anchor_peak_residual_overlap_gain_density_landing_collapse_alignment_score > 0.0
                        and float(anchor_row.get("delta_hard_anchor_peak_residual", 0.0)) <= 0.0
                        and anchor_connected_components < 0.0
                    )
                    or
                    (
                        _is_hard_anchor_peak_residual_overlap_gain_density_landing_realization_family(family=family, variant=variant)
                        and anchor_peak_residual_overlap_gain_density_landing_target_fraction > 0.0
                        and anchor_peak_residual_overlap_gain_density_landing_binding_fraction > 0.0
                        and anchor_peak_residual_overlap_gain_density_landing_realization_fraction > 0.0
                        and anchor_peak_residual_overlap_gain_density_landing_gain > MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_LANDING_FLOOR
                        and anchor_peak_residual_overlap_gain_density_landing_alignment_score > 0.0
                        and float(anchor_row.get("delta_hard_anchor_peak_residual", 0.0)) <= 0.0
                        and anchor_connected_components < 0.0
                    )
                    or
                    (
                        _is_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_family(family=family, variant=variant)
                        and anchor_peak_residual_overlap_gain_density_amplification_target_fraction > 0.0
                        and anchor_peak_residual_overlap_gain_density_amplification_binding_fraction > 0.0
                        and anchor_peak_residual_overlap_gain_density_amplification_realization_fraction > 0.0
                        and anchor_peak_residual_overlap_gain_density_amplification_gain > MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_AMPLIFICATION_FLOOR
                        and anchor_peak_residual_overlap_gain_density_amplification_alignment_score > 0.0
                        and float(anchor_row.get("delta_hard_anchor_peak_residual", 0.0)) <= 0.0
                        and anchor_connected_components < 0.0
                    )
                    or
                    (
                        _is_hard_anchor_peak_residual_overlap_gain_amplification_realization_family(family=family, variant=variant)
                        and anchor_peak_residual_overlap_gain_amplification_target_fraction > 0.0
                        and anchor_peak_residual_overlap_gain_amplification_binding_fraction > 0.0
                        and anchor_peak_residual_overlap_gain_amplification_realization_fraction > 0.0
                        and anchor_peak_residual_overlap_gain_amplification_gain > MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_AMPLIFICATION_FLOOR
                        and anchor_peak_residual_overlap_gain_amplification_alignment_score > 0.0
                        and float(anchor_row.get("delta_hard_anchor_peak_residual", 0.0)) <= 0.0
                        and anchor_connected_components < 0.0
                    )
                    or
                    (
                        _is_hard_anchor_peak_residual_overlap_gain_realization_family(family=family, variant=variant)
                        and anchor_peak_residual_overlap_gain_target_fraction > 0.0
                        and anchor_peak_residual_overlap_gain_binding_fraction > 0.0
                        and anchor_peak_residual_overlap_gain_realization_fraction > 0.0
                        and anchor_peak_residual_overlap_gain > MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_FLOOR
                        and anchor_peak_residual_overlap_gain_alignment_score > 0.0
                        and float(anchor_row.get("delta_hard_anchor_peak_residual", 0.0)) <= 0.0
                        and anchor_connected_components < 0.0
                    )
                    or
                    (
                        _is_hard_anchor_peak_residual_overlap_alignment_realization_family(family=family, variant=variant)
                        and anchor_peak_residual_overlap_target_fraction > 0.0
                        and anchor_peak_residual_overlap_binding_fraction > 0.0
                        and anchor_peak_residual_overlap_realization_fraction > 0.0
                        and anchor_peak_residual_overlap_gain > 0.0
                        and anchor_peak_residual_overlap_alignment_score > 0.0
                        and float(anchor_row.get("delta_hard_anchor_peak_residual", 0.0)) <= 0.0
                        and anchor_connected_components < 0.0
                    )
                    or
                    (
                        _is_hard_anchor_peak_residual_contraction_realization_family(family=family, variant=variant)
                        and anchor_peak_residual_contraction_target_fraction > 0.0
                        and anchor_peak_residual_contraction_realization_fraction > 0.0
                        and anchor_peak_residual_contraction_overlap_fraction > 0.0
                        and anchor_peak_residual_contraction_gain > 0.0
                        and anchor_peak_residual_contraction_alignment_score > 0.0
                        and float(anchor_row.get("delta_hard_anchor_peak_residual", 0.0)) <= 0.0
                        and anchor_connected_components < 0.0
                    )
                    or
                    (
                        _is_hard_anchor_peak_residual_activation_family(family=family, variant=variant)
                        and anchor_breakout_alignment_gain > 0.0
                        and anchor_peak_rebound_contract_gain > 0.0
                        and anchor_peak_residual_activation_fraction > 0.0
                        and anchor_peak_residual_binding_fraction > 0.0
                        and anchor_peak_residual_contraction_fraction > 0.0
                        and anchor_peak_residual_activation_gain > 0.0
                        and float(anchor_row.get("delta_hard_anchor_peak_residual", 0.0)) <= 0.0
                        and anchor_connected_components < 0.0
                    )
                    or
                    (
                        _is_hard_anchor_peak_rebound_suppression_family(family=family, variant=variant)
                        and anchor_breakout_alignment_gain > 0.0
                        and anchor_peak_rebound_focus_fraction > 0.0
                        and anchor_peak_rebound_binding_fraction > 0.0
                        and anchor_peak_rebound_suppression_fraction > 0.0
                        and anchor_peak_rebound_contract_gain > 0.0
                        and anchor_connected_components < 0.0
                    )
                    or
                    (
                        _is_hard_anchor_breakout_alignment_family(family=family, variant=variant)
                        and anchor_breakout_alignment_fraction > 0.0
                        and anchor_breakout_alignment_gain > 0.0
                        and anchor_local_shrinkage_witness_fraction > 0.0
                        and anchor_breakout_alignment_centroid_score > 0.0
                        and anchor_connected_components < 0.0
                    )
                    or
                    (
                        _is_hard_anchor_closure_realization_family(family=family, variant=variant)
                        and anchor_local_closure_fraction > 0.0
                        and anchor_local_closure_gain > 0.0
                        and anchor_local_closure_weight > 0.0
                        and anchor_local_peak_rebound_suppression_gain > 0.0
                        and anchor_closure_breakout_score > 0.0
                        and anchor_connected_components < 0.0
                    )
                    or
                    (
                        _is_component_closure_realization_family(family=family, variant=variant)
                        and anchor_consolidation_gain > 0.0
                        and anchor_realization_gain > 0.0
                        and anchor_closure_realization_gain > 0.0
                        and anchor_closure_realization_fraction > 0.0
                        and anchor_closure_binding_fraction > 0.0
                        and anchor_closure_breakout_score > 0.0
                        and anchor_connected_components < 0.0
                    )
                    or
                    (
                        _is_component_adjacency_family(family=family, variant=variant)
                        and anchor_consolidation_gain > 0.0
                        and anchor_realization_gain > 0.0
                        and anchor_adjacency_gain > 0.0
                        and anchor_closure_realization_gain > 0.0
                        and anchor_connected_components < 0.0
                    )
                    or
                    (
                        _is_component_consolidation_family(family=family, variant=variant)
                        and anchor_consolidation_gain > 0.0
                        and anchor_realization_gain > 0.0
                        and anchor_connected_components < 0.0
                    )
                    or
                    (
                        _is_post_merge_realization_family(family=family, variant=variant)
                        and anchor_realization_gain > 0.0
                        and anchor_connected_components < 0.0
                    )
                    or (
                        not _is_post_merge_realization_family(family=family, variant=variant)
                        and (
                            not _is_peak_stabilization_family(family=family, variant=variant)
                            or anchor_peak_gain > 0.0
                        )
                    )
                )
            )
        ),
    }


def _failure_routing(
    aggregate: dict,
    *,
    rows: list[dict],
    is_guard_failure_eliminated: bool,
    hard_anchor_guard: dict,
    family: str,
    variant: str,
) -> dict:
    if _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_realization_family(family=family, variant=variant):
        if bool(aggregate.get("artifact_incomplete_fail", False)):
            return {
                "route": "G0",
                "next_family": family,
                "next_candidate": variant,
                "reason": "graph-realization packaging or required panels failed completeness, so the honest next move is to repair artifacts without changing families.",
            }
        anchor_components_delta = float(hard_anchor_guard.get("delta_fg_connected_components", 0.0))
        anchor_peak_delta = float(hard_anchor_guard.get("delta_fg_peak_count_after_render", 0.0))
        anchor_peak_residual_delta = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual", 0.0))
        anchor_target_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_target_fraction", 0.0))
        anchor_binding_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_binding_fraction", 0.0))
        anchor_realization_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_realization_fraction", 0.0))
        anchor_graph_gain = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_gain", 0.0))
        anchor_alignment_score = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_alignment_score", 0.0))
        graph_moved = bool(
            anchor_target_fraction > 0.0
            or anchor_binding_fraction > 0.0
            or anchor_realization_fraction > 0.0
            or anchor_graph_gain > 0.0
            or anchor_alignment_score > 0.0
        )
        aggregate_generalization_failed = bool(
            aggregate.get("mean_delta_masked_l1", 0.0) > 0.0
            or aggregate.get("mean_delta_masked_ssim", 0.0) < 0.0
            or aggregate.get("worst_delta_off_body_support_ratio", 0.0) > 0.0
            or aggregate.get("mean_delta_fg_connected_components", 0.0) >= 0.0
            or aggregate.get("mean_delta_fg_peak_count_after_render", 0.0) > 0.0
            or aggregate.get("mean_delta_hard_anchor_peak_residual", 0.0) > 0.0
            or any(not bool(row.get("smoke_case_pass", False)) for row in rows)
        )
        if not graph_moved:
            return {
                "route": "G1",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_binding_realization_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_binding_realization_v1",
                "reason": "graph target, binding, and realization stayed near zero, so the next honest wall is graph binding realization rather than more graph pressure.",
            }
        if (
            aggregate.get("mean_delta_masked_l1", 0.0) > 0.0
            or aggregate.get("mean_delta_masked_ssim", 0.0) < 0.0
            or aggregate.get("worst_delta_off_body_support_ratio", 0.0) > 0.0
        ):
            return {
                "route": "G4",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_guard_preservation_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_guard_preservation_v1",
                "reason": "graph realization started to move, but masked quality or off-body support regressed, so the next honest wall is a guard-preserving graph contract.",
            }
        if bool(hard_anchor_guard.get("pass", False)) and aggregate_generalization_failed:
            return {
                "route": "G5",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_generalization_contract_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_generalization_contract_v1",
                "reason": "the hard anchor passed, but the remaining smoke cases still blocked honest generalization above graph realization.",
            }
        if (
            anchor_graph_gain <= MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_CONNECTIVITY_GRAPH_FLOOR
            or float(aggregate.get("mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_gain", 0.0))
            <= MEANINGFUL_MEAN_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_CONNECTIVITY_GRAPH_FLOOR
        ):
            return {
                "route": "G2",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_landing_realization_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_landing_realization_v1",
                "reason": "graph overlays lit up, but actual graph realization gain stayed too small to count as honest collapse pressure on the hard anchor.",
            }
        return {
            "route": "G3",
            "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_to_visible_component_collapse_contract_audit",
            "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_to_visible_component_collapse_contract_v1",
            "reason": (
                "graph realization gain became positive, but the hard anchor still did not honestly collapse connected components, post-render peaks, or residual peaks."
                if (
                    anchor_components_delta >= 0.0
                    or anchor_peak_delta > 0.0
                    or anchor_peak_residual_delta > 0.0
                )
                else "graph realization landed locally, but its coupling to visible component collapse still failed to materialize honestly."
            ),
        }
    if _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_contract_family(family=family, variant=variant):
        if bool(aggregate.get("artifact_incomplete_fail", False)):
            return {
                "route": "K0",
                "next_family": family,
                "next_candidate": variant,
                "reason": "connectivity-contract packaging or required panels failed completeness, so the honest next move is to repair artifacts without changing families.",
            }
        anchor_components_delta = float(hard_anchor_guard.get("delta_fg_connected_components", 0.0))
        anchor_peak_delta = float(hard_anchor_guard.get("delta_fg_peak_count_after_render", 0.0))
        anchor_peak_residual_delta = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual", 0.0))
        anchor_target_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_target_fraction", 0.0))
        anchor_binding_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_binding_fraction", 0.0))
        anchor_contract_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_contract_fraction", 0.0))
        anchor_connectivity_gain = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_gain", 0.0))
        anchor_alignment_score = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_alignment_score", 0.0))
        connectivity_moved = bool(
            anchor_target_fraction > 0.0
            or anchor_binding_fraction > 0.0
            or anchor_contract_fraction > 0.0
            or anchor_connectivity_gain > 0.0
            or anchor_alignment_score > 0.0
        )
        aggregate_generalization_failed = bool(
            aggregate.get("mean_delta_masked_l1", 0.0) > 0.0
            or aggregate.get("mean_delta_masked_ssim", 0.0) < 0.0
            or aggregate.get("worst_delta_off_body_support_ratio", 0.0) > 0.0
            or aggregate.get("mean_delta_fg_connected_components", 0.0) >= 0.0
            or aggregate.get("mean_delta_fg_peak_count_after_render", 0.0) > 0.0
            or aggregate.get("mean_delta_hard_anchor_peak_residual", 0.0) > 0.0
            or any(not bool(row.get("smoke_case_pass", False)) for row in rows)
        )
        if not connectivity_moved:
            return {
                "route": "K1",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_binding_realization_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_binding_realization_v1",
                "reason": "connectivity target, binding, and contract maps stayed near zero, so the next honest wall is connectivity binding realization.",
            }
        if (
            aggregate.get("mean_delta_masked_l1", 0.0) > 0.0
            or aggregate.get("mean_delta_masked_ssim", 0.0) < 0.0
            or aggregate.get("worst_delta_off_body_support_ratio", 0.0) > 0.0
        ):
            return {
                "route": "K4",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_quality_preserving_contract_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_quality_preserving_contract_v1",
                "reason": "connectivity coupling started to move, but masked quality or off-body support regressed, so the next honest wall is a quality-preserving connectivity contract.",
            }
        if bool(hard_anchor_guard.get("pass", False)) and aggregate_generalization_failed:
            return {
                "route": "G",
                "next_family": "teacher_geometry_render_operator_post_anchor_generalization_realization_audit",
                "next_candidate": "stablelead_rehydrated_operator_post_anchor_generalization_realization_v1",
                "reason": "the hard anchor passed, but the remaining smoke cases still blocked honest aggregate generalization.",
            }
        if anchor_connectivity_gain > 0.0 and anchor_components_delta >= 0.0:
            return {
                "route": "K2",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_realization_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_realization_v1",
                "reason": "connectivity gain turned positive, but connected components still did not fall honestly, so the next honest wall is connectivity graph realization.",
            }
        if anchor_components_delta < 0.0 and (anchor_peak_delta > 0.0 or anchor_peak_residual_delta > 0.0):
            return {
                "route": "K3",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_peak_residual_cleanup_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_peak_residual_cleanup_v1",
                "reason": "connectivity closure started to reduce components, but post-render peaks or residual peaks still failed honestly, so the next honest wall is peak-residual cleanup.",
            }
        return {
            "route": "K2",
            "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_realization_audit",
            "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_realization_v1",
            "reason": "landed collapse connectivity remained positive, but honest connectivity collapse still did not materialize on the hard anchor.",
        }
    if _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_contract_family(family=family, variant=variant):
        if bool(aggregate.get("artifact_incomplete_fail", False)):
            return {
                "route": "C0",
                "next_family": family,
                "next_candidate": variant,
                "reason": "collapse-contract packaging or required panels failed completeness, so the honest next move is to repair artifacts without changing families.",
            }
        anchor_components_delta = float(hard_anchor_guard.get("delta_fg_connected_components", 0.0))
        anchor_peak_delta = float(hard_anchor_guard.get("delta_fg_peak_count_after_render", 0.0))
        anchor_peak_residual_delta = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual", 0.0))
        anchor_target_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_target_fraction", 0.0))
        anchor_binding_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_binding_fraction", 0.0))
        anchor_contract_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_contract_fraction", 0.0))
        anchor_collapse_gain = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_gain", 0.0))
        anchor_alignment_score = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_alignment_score", 0.0))
        collapse_moved = bool(
            anchor_target_fraction > 0.0
            or anchor_binding_fraction > 0.0
            or anchor_contract_fraction > 0.0
            or anchor_collapse_gain > 0.0
            or anchor_alignment_score > 0.0
        )
        if not collapse_moved:
            return {
                "route": "C1",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_binding_realization_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_binding_realization_v1",
                "reason": "collapse target, binding, and contract maps stayed near zero, so the next honest wall is collapse binding realization.",
            }
        if (
            aggregate.get("mean_delta_masked_l1", 0.0) > 0.0
            or aggregate.get("mean_delta_masked_ssim", 0.0) < 0.0
            or aggregate.get("worst_delta_off_body_support_ratio", 0.0) > 0.0
        ):
            return {
                "route": "C4",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_quality_preserving_contract_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_quality_preserving_contract_v1",
                "reason": "collapse coupling started to move, but masked quality or off-body support regressed, so the next honest wall is a quality-preserving collapse contract.",
            }
        if anchor_collapse_gain > 0.0 and anchor_components_delta >= 0.0:
            return {
                "route": "C2",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_contract_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_contract_v1",
                "reason": "collapse gain turned positive, but connected components still did not fall honestly, so the next honest wall is connectivity coupling.",
            }
        if anchor_components_delta < 0.0 and (anchor_peak_delta > 0.0 or anchor_peak_residual_delta > 0.0):
            return {
                "route": "C3",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_peak_residual_cleanup_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_peak_residual_cleanup_v1",
                "reason": "components started to collapse, but post-render peaks or residual peaks still failed honestly, so the next honest wall is peak-residual cleanup.",
            }
        return {
            "route": "C2",
            "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_contract_audit",
            "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_contract_v1",
            "reason": "landed density remained aligned, but honest component collapse still did not materialize on the hard anchor.",
        }
    if _is_hard_anchor_peak_residual_overlap_gain_density_landing_realization_family(family=family, variant=variant):
        if bool(aggregate.get("artifact_incomplete_fail", False)):
            return {
                "route": "L0",
                "next_family": family,
                "next_candidate": variant,
                "reason": "density-landing packaging or required panels failed completeness, so the honest next move is to repair artifacts without changing families.",
            }
        anchor_components_delta = float(hard_anchor_guard.get("delta_fg_connected_components", 0.0))
        anchor_peak_delta = float(hard_anchor_guard.get("delta_fg_peak_count_after_render", 0.0))
        anchor_peak_residual_delta = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual", 0.0))
        anchor_target_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_target_fraction", 0.0))
        anchor_binding_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_binding_fraction", 0.0))
        anchor_realization_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_realization_fraction", 0.0))
        anchor_landing_gain = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_gain", 0.0))
        anchor_alignment_score = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_landing_alignment_score", 0.0))
        landing_moved = bool(
            anchor_target_fraction > 0.0
            or anchor_binding_fraction > 0.0
            or anchor_realization_fraction > 0.0
            or anchor_landing_gain > 0.0
            or anchor_alignment_score > 0.0
        )
        aggregate_generalization_failed = bool(
            aggregate.get("mean_delta_masked_l1", 0.0) > 0.0
            or aggregate.get("mean_delta_masked_ssim", 0.0) < 0.0
            or aggregate.get("worst_delta_off_body_support_ratio", 0.0) > 0.0
            or aggregate.get("mean_delta_fg_connected_components", 0.0) >= 0.0
            or aggregate.get("mean_delta_fg_peak_count_after_render", 0.0) > 0.0
            or aggregate.get("mean_delta_hard_anchor_peak_residual", 0.0) > 0.0
            or any(not bool(row.get("smoke_case_pass", False)) for row in rows)
        )
        if not landing_moved:
            return {
                "route": "L1",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_binding_realization_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_binding_realization_v1",
                "reason": "density-landing panels rendered, but landing target, binding, and realization stayed near zero, so the next honest wall is landing binding realization.",
            }
        if (
            anchor_landing_gain <= 0.0
            or float(aggregate.get("mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_gain", 0.0)) <= 0.0
        ):
            return {
                "route": "L2",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_footprint_alignment_realization_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_footprint_alignment_realization_v1",
                "reason": "density-landing ledgers lit up, but landed gain on the final shrinkage footprint stayed nonpositive, so the next honest wall is footprint alignment realization.",
            }
        if (
            (
                anchor_components_delta < 0.0
                or anchor_peak_delta <= 0.0
                or anchor_peak_residual_delta <= 0.0
                or float(aggregate.get("mean_delta_fg_connected_components", 0.0)) < 0.0
            )
            and (
                aggregate.get("mean_delta_masked_l1", 0.0) > 0.0
                or aggregate.get("mean_delta_masked_ssim", 0.0) < 0.0
                or aggregate.get("worst_delta_off_body_support_ratio", 0.0) > 0.0
                or any(not bool(row.get("guard_report", {}).get("all_pass", True)) for row in rows)
            )
        ):
            return {
                "route": "L4",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_quality_preserving_contract_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_quality_preserving_contract_v1",
                "reason": "density landing started to improve shrinkage, but masked quality or coverage guards regressed, so the next honest wall is a quality-preserving landing contract.",
            }
        if bool(hard_anchor_guard.get("pass", False)) and aggregate_generalization_failed:
            return {
                "route": "G",
                "next_family": "teacher_geometry_render_operator_post_anchor_generalization_realization_audit",
                "next_candidate": "stablelead_rehydrated_operator_post_anchor_generalization_realization_v1",
                "reason": "the hard anchor passed, but the remaining smoke cases still blocked honest aggregate generalization.",
            }
        return {
            "route": "L3",
            "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_contract_audit",
            "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_contract_v1",
            "reason": (
                "density landing gain became positive, but hard-anchor connected components, post-render peaks, or residual peaks still did not collapse honestly."
                if (
                    anchor_components_delta >= 0.0
                    or anchor_peak_delta > 0.0
                    or anchor_peak_residual_delta > 0.0
                )
                else "density landing improved locally without producing honest component collapse across the smoke set."
            ),
        }
    if _is_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_family(family=family, variant=variant):
        anchor_components_delta = float(hard_anchor_guard.get("delta_fg_connected_components", 0.0))
        anchor_peak_delta = float(hard_anchor_guard.get("delta_fg_peak_count_after_render", 0.0))
        anchor_peak_residual_delta = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual", 0.0))
        anchor_target_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_target_fraction", 0.0))
        anchor_binding_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_binding_fraction", 0.0))
        anchor_realization_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_fraction", 0.0))
        anchor_overlap_gain = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_gain", 0.0))
        anchor_alignment_score = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_density_amplification_alignment_score", 0.0))
        density_moved = bool(
            anchor_target_fraction > 0.0
            or anchor_binding_fraction > 0.0
            or anchor_realization_fraction > 0.0
            or anchor_overlap_gain > 0.0
            or anchor_alignment_score > 0.0
        )
        aggregate_generalization_failed = bool(
            aggregate.get("mean_delta_masked_l1", 0.0) > 0.0
            or aggregate.get("mean_delta_masked_ssim", 0.0) < 0.0
            or aggregate.get("worst_delta_off_body_support_ratio", 0.0) > 0.0
            or aggregate.get("mean_delta_fg_connected_components", 0.0) >= 0.0
            or aggregate.get("mean_delta_fg_peak_count_after_render", 0.0) > 0.0
            or aggregate.get("mean_delta_hard_anchor_peak_residual", 0.0) > 0.0
            or any(not bool(row.get("smoke_case_pass", False)) for row in rows)
        )
        if not density_moved:
            return {
                "route": "F1",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_binding_realization_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_binding_realization_v1",
                "reason": "density-amplification panels rendered, but density target, binding, realization, and gain stayed at zero, so the next honest wall is density binding realization rather than more landing pressure.",
            }
        if (
            anchor_overlap_gain <= MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_DENSITY_AMPLIFICATION_FLOOR
            or float(aggregate.get("mean_delta_hard_anchor_peak_residual_overlap_gain_density_amplification_gain", 0.0)) <= MEANINGFUL_MEAN_OVERLAP_GAIN_DENSITY_AMPLIFICATION_FLOOR
        ):
            return {
                "route": "F2",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_realization_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_landing_realization_v1",
                "reason": "density ledgers turned on, but the landed density gain on final visible shrinkage stayed too weak to count as honest overlap gain.",
            }
        if (
            (
                anchor_components_delta < 0.0
                or anchor_peak_delta <= 0.0
                or anchor_peak_residual_delta <= 0.0
                or float(aggregate.get("mean_delta_fg_connected_components", 0.0)) < 0.0
            )
            and (
                aggregate.get("mean_delta_masked_l1", 0.0) > 0.0
                or aggregate.get("mean_delta_masked_ssim", 0.0) < 0.0
                or aggregate.get("worst_delta_off_body_support_ratio", 0.0) > 0.0
                or any(not bool(row.get("guard_report", {}).get("all_pass", True)) for row in rows)
            )
        ):
            return {
                "route": "F4",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_quality_preserving_contract_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_quality_preserving_contract_v1",
                "reason": "density landing began to improve collapse signals, but masked quality or coverage guards regressed, so the next honest wall is a quality-preserving density contract.",
            }
        if bool(hard_anchor_guard.get("pass", False)) and aggregate_generalization_failed:
            return {
                "route": "G",
                "next_family": "teacher_geometry_render_operator_post_anchor_generalization_realization_audit",
                "next_candidate": "stablelead_rehydrated_operator_post_anchor_generalization_realization_v1",
                "reason": "the hard anchor passed, but the remaining smoke cases still blocked honest aggregate generalization.",
            }
        return {
            "route": "F3",
            "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_to_component_collapse_contract_audit",
            "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_to_component_collapse_contract_v1",
            "reason": (
                "density gain landed locally, but hard-anchor connected components, post-render peaks, or residual peaks still did not collapse honestly."
                if (
                    anchor_components_delta >= 0.0
                    or anchor_peak_delta > 0.0
                    or anchor_peak_residual_delta > 0.0
                )
                else "density landing became real, but it still failed to couple into honest component collapse."
            ),
        }
    if _is_hard_anchor_peak_residual_overlap_gain_amplification_realization_family(family=family, variant=variant):
        anchor_components_delta = float(hard_anchor_guard.get("delta_fg_connected_components", 0.0))
        anchor_peak_delta = float(hard_anchor_guard.get("delta_fg_peak_count_after_render", 0.0))
        anchor_peak_residual_delta = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual", 0.0))
        anchor_target_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_target_fraction", 0.0))
        anchor_binding_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_binding_fraction", 0.0))
        anchor_realization_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_realization_fraction", 0.0))
        anchor_overlap_gain = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_gain", 0.0))
        anchor_alignment_score = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_amplification_alignment_score", 0.0))
        amplification_moved = bool(
            anchor_target_fraction > 0.0
            or anchor_binding_fraction > 0.0
            or anchor_realization_fraction > 0.0
            or anchor_overlap_gain > 0.0
            or anchor_alignment_score > 0.0
        )
        aggregate_generalization_failed = bool(
            aggregate.get("mean_delta_masked_l1", 0.0) > 0.0
            or aggregate.get("mean_delta_masked_ssim", 0.0) < 0.0
            or aggregate.get("worst_delta_off_body_support_ratio", 0.0) > 0.0
            or aggregate.get("mean_delta_fg_connected_components", 0.0) >= 0.0
            or aggregate.get("mean_delta_fg_peak_count_after_render", 0.0) > 0.0
            or aggregate.get("mean_delta_hard_anchor_peak_residual", 0.0) > 0.0
            or any(not bool(row.get("smoke_case_pass", False)) for row in rows)
        )
        if not amplification_moved:
            return {
                "route": "A",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_amplification_wiring_repair_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_amplification_wiring_repair_v1",
                "reason": "amplification panels rendered, but the new amplification ledgers stayed at zero, so the next honest wall is amplification wiring repair rather than more landing pressure.",
            }
        if bool(hard_anchor_guard.get("pass", False)) and aggregate_generalization_failed:
            return {
                "route": "D",
                "next_family": "teacher_geometry_render_operator_post_anchor_generalization_realization_audit",
                "next_candidate": "stablelead_rehydrated_operator_post_anchor_generalization_realization_v1",
                "reason": "the hard anchor passed, but the remaining smoke cases still blocked honest aggregate generalization.",
            }
        if anchor_overlap_gain <= MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_AMPLIFICATION_FLOOR or float(aggregate.get("mean_delta_hard_anchor_peak_residual_overlap_gain_amplification_gain", 0.0)) <= MEANINGFUL_MEAN_OVERLAP_GAIN_AMPLIFICATION_FLOOR:
            return {
                "route": "B",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_v1",
                "reason": "amplification ledgers turned on, but actual overlap-gain amplification landing on final visible shrinkage stayed too small to count as honest gain.",
            }
        return {
            "route": "C",
            "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_collapse_coupled_amplification_realization_audit",
            "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_collapse_coupled_amplification_realization_v1",
            "reason": (
                "actual overlap-gain amplification improved, but hard-anchor residual peaks, post-render peaks, or connected components still did not collapse honestly."
                if (
                    anchor_components_delta >= 0.0
                    or anchor_peak_delta > 0.0
                    or anchor_peak_residual_delta > 0.0
                )
                else "amplification landed locally, but its residual-collapse coupling still failed to materialize honestly."
            ),
        }
    if _is_hard_anchor_peak_residual_overlap_gain_realization_family(family=family, variant=variant):
        anchor_components_delta = float(hard_anchor_guard.get("delta_fg_connected_components", 0.0))
        anchor_peak_delta = float(hard_anchor_guard.get("delta_fg_peak_count_after_render", 0.0))
        anchor_peak_residual_delta = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual", 0.0))
        anchor_target_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_target_fraction", 0.0))
        anchor_binding_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_binding_fraction", 0.0))
        anchor_realization_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_realization_fraction", 0.0))
        anchor_overlap_gain = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain", 0.0))
        anchor_alignment_score = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain_alignment_score", 0.0))
        overlap_gain_moved = bool(
            anchor_target_fraction > 0.0
            or anchor_binding_fraction > 0.0
            or anchor_realization_fraction > 0.0
            or anchor_overlap_gain > 0.0
            or anchor_alignment_score > 0.0
        )
        aggregate_generalization_failed = bool(
            aggregate.get("mean_delta_masked_l1", 0.0) > 0.0
            or aggregate.get("mean_delta_masked_ssim", 0.0) < 0.0
            or aggregate.get("worst_delta_off_body_support_ratio", 0.0) > 0.0
            or aggregate.get("mean_delta_fg_connected_components", 0.0) >= 0.0
            or aggregate.get("mean_delta_fg_peak_count_after_render", 0.0) > 0.0
            or aggregate.get("mean_delta_hard_anchor_peak_residual", 0.0) > 0.0
            or any(not bool(row.get("smoke_case_pass", False)) for row in rows)
        )
        if not overlap_gain_moved:
            return {
                "route": "A",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_wiring_repair_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_wiring_repair_v1",
                "reason": "overlap-gain panels rendered, but the new overlap-gain ledgers stayed at zero, so the next honest wall is overlap-gain wiring repair rather than more gain pressure.",
            }
        if bool(hard_anchor_guard.get("pass", False)) and aggregate_generalization_failed:
            return {
                "route": "C",
                "next_family": "teacher_geometry_render_operator_post_hard_anchor_generalization_audit",
                "next_candidate": "stablelead_rehydrated_operator_post_hard_anchor_generalization_v1",
                "reason": "the hard anchor passed, but the non-anchor smoke cases still blocked honest aggregate generalization.",
            }
        if anchor_overlap_gain > MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_FLOOR and anchor_peak_residual_delta > 0.0:
            return {
                "route": "D",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_neutralization_realization_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_neutralization_realization_v1",
                "reason": "overlap gain became materially positive on the hard anchor, but residual peaks still stayed above zero, so the next honest wall is residual neutralization rather than more overlap-gain pressure.",
            }
        return {
            "route": "B",
            "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_amplification_realization_audit",
            "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_amplification_realization_v1",
            "reason": (
                "overlap-gain ledgers turned on, but actual overlap gain on final visible shrinkage stayed too small to reduce hard-anchor fragmentation, post-render peaks, or residual peaks honestly."
                if (
                    anchor_components_delta >= 0.0
                    or anchor_peak_delta > 0.0
                    or anchor_peak_residual_delta > 0.0
                    or anchor_overlap_gain <= MEANINGFUL_HARD_ANCHOR_OVERLAP_GAIN_FLOOR
                    or float(aggregate.get("mean_delta_hard_anchor_peak_residual_overlap_gain", 0.0)) <= MEANINGFUL_MEAN_OVERLAP_GAIN_FLOOR
                )
                else "overlap-gain moved locally, but its amplification still did not materialize honestly on the final visible shrinkage region."
            ),
        }
    if _is_hard_anchor_peak_residual_overlap_alignment_realization_family(family=family, variant=variant):
        anchor_components_delta = float(hard_anchor_guard.get("delta_fg_connected_components", 0.0))
        anchor_peak_delta = float(hard_anchor_guard.get("delta_fg_peak_count_after_render", 0.0))
        anchor_peak_residual_delta = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual", 0.0))
        anchor_target_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_target_fraction", 0.0))
        anchor_binding_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_binding_fraction", 0.0))
        anchor_realization_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_realization_fraction", 0.0))
        anchor_overlap_gain = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_gain", 0.0))
        anchor_alignment_score = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_overlap_alignment_score", 0.0))
        overlap_moved = bool(
            anchor_target_fraction > 0.0
            or anchor_binding_fraction > 0.0
            or anchor_realization_fraction > 0.0
            or anchor_overlap_gain > 0.0
            or anchor_alignment_score > 0.0
        )
        aggregate_generalization_failed = bool(
            aggregate.get("mean_delta_masked_l1", 0.0) > 0.0
            or aggregate.get("mean_delta_masked_ssim", 0.0) < 0.0
            or aggregate.get("worst_delta_off_body_support_ratio", 0.0) > 0.0
            or aggregate.get("mean_delta_fg_connected_components", 0.0) >= 0.0
            or aggregate.get("mean_delta_fg_peak_count_after_render", 0.0) > 0.0
            or aggregate.get("mean_delta_hard_anchor_peak_residual", 0.0) > 0.0
            or any(not bool(row.get("smoke_case_pass", False)) for row in rows)
        )
        if not overlap_moved:
            return {
                "route": "A",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_wiring_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_wiring_v1",
                "reason": "overlap/alignment panels rendered, but the new overlap ledgers stayed at zero, so the next honest wall is overlap wiring rather than more local overlap pressure.",
            }
        if bool(hard_anchor_guard.get("pass", False)) and aggregate_generalization_failed:
            return {
                "route": "C",
                "next_family": "teacher_geometry_render_operator_post_hard_anchor_generalization_audit",
                "next_candidate": "stablelead_rehydrated_operator_post_hard_anchor_generalization_v1",
                "reason": "the hard anchor passed, but the non-anchor smoke cases still blocked honest aggregate generalization.",
            }
        return {
            "route": "B",
            "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_gain_realization_audit",
            "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_gain_realization_v1",
            "reason": (
                "overlap/alignment ledgers turned on, but the hard anchor still did not land overlap gain on the final visible shrinkage region "
                "because components, post-render peaks, or residual peaks stayed positive."
                if (
                    anchor_components_delta >= 0.0
                    or anchor_peak_delta > 0.0
                    or anchor_peak_residual_delta > 0.0
                    or anchor_overlap_gain <= 0.0
                )
                else "overlap/alignment moved locally, but its gain still did not materialize honestly on the hard anchor."
            ),
        }
    if _is_hard_anchor_peak_residual_contraction_realization_family(family=family, variant=variant):
        anchor_components_delta = float(hard_anchor_guard.get("delta_fg_connected_components", 0.0))
        anchor_peak_delta = float(hard_anchor_guard.get("delta_fg_peak_count_after_render", 0.0))
        anchor_peak_residual_delta = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual", 0.0))
        anchor_target_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_contraction_target_fraction", 0.0))
        anchor_realization_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_contraction_realization_fraction", 0.0))
        anchor_overlap_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_contraction_overlap_fraction", 0.0))
        anchor_contraction_gain = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_contraction_gain", 0.0))
        anchor_alignment_score = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_contraction_alignment_score", 0.0))
        contraction_moved = bool(
            anchor_target_fraction > 0.0
            or anchor_realization_fraction > 0.0
            or anchor_overlap_fraction > 0.0
            or anchor_contraction_gain > 0.0
            or anchor_alignment_score > 0.0
        )
        aggregate_generalization_failed = bool(
            aggregate.get("mean_delta_masked_l1", 0.0) > 0.0
            or aggregate.get("mean_delta_masked_ssim", 0.0) < 0.0
            or aggregate.get("worst_delta_off_body_support_ratio", 0.0) > 0.0
            or aggregate.get("mean_delta_fg_connected_components", 0.0) >= 0.0
            or aggregate.get("mean_delta_fg_peak_count_after_render", 0.0) > 0.0
            or aggregate.get("mean_delta_hard_anchor_peak_residual", 0.0) > 0.0
            or any(not bool(row.get("smoke_case_pass", False)) for row in rows)
        )
        if not contraction_moved:
            return {
                "route": "A",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_contraction_wiring_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_contraction_wiring_v1",
                "reason": "contraction overlays rendered, but the new contraction ledgers stayed at zero, so the next honest wall is contraction wiring rather than more contraction pressure.",
            }
        if bool(hard_anchor_guard.get("pass", False)) and aggregate_generalization_failed:
            return {
                "route": "C",
                "next_family": "teacher_geometry_render_operator_post_hard_anchor_generalization_audit",
                "next_candidate": "stablelead_rehydrated_operator_post_hard_anchor_generalization_v1",
                "reason": "the hard anchor passed, but the non-anchor smoke cases still blocked honest aggregate generalization.",
            }
        return {
            "route": "B",
            "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_overlap_alignment_realization_audit",
            "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_overlap_alignment_realization_v1",
            "reason": (
                "contraction ledgers turned on, but the hard anchor still did not land contraction on final visible shrinkage "
                "because components, post-render peaks, or residual peaks stayed positive."
                if (
                    anchor_components_delta >= 0.0
                    or anchor_peak_delta > 0.0
                    or anchor_peak_residual_delta > 0.0
                )
                else "contraction moved locally, but overlap alignment still did not materialize honestly on the hard anchor."
            ),
        }
    if _is_hard_anchor_peak_residual_activation_family(family=family, variant=variant):
        anchor_components_delta = float(hard_anchor_guard.get("delta_fg_connected_components", 0.0))
        anchor_peak_delta = float(hard_anchor_guard.get("delta_fg_peak_count_after_render", 0.0))
        anchor_peak_residual_delta = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual", 0.0))
        anchor_activation_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_activation_fraction", 0.0))
        anchor_binding_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_binding_fraction", 0.0))
        anchor_contraction_fraction = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_contraction_fraction", 0.0))
        anchor_activation_gain = float(hard_anchor_guard.get("delta_hard_anchor_peak_residual_activation_gain", 0.0))
        activation_moved = bool(
            anchor_activation_fraction > 0.0
            or anchor_binding_fraction > 0.0
            or anchor_contraction_fraction > 0.0
            or anchor_activation_gain > 0.0
        )
        aggregate_generalization_failed = bool(
            aggregate.get("mean_delta_masked_l1", 0.0) > 0.0
            or aggregate.get("mean_delta_masked_ssim", 0.0) < 0.0
            or aggregate.get("worst_delta_off_body_support_ratio", 0.0) > 0.0
            or aggregate.get("mean_delta_fg_connected_components", 0.0) >= 0.0
            or aggregate.get("mean_delta_fg_peak_count_after_render", 0.0) > 0.0
            or any(not bool(row.get("smoke_case_pass", False)) for row in rows)
        )
        if not activation_moved:
            return {
                "route": "A",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_activation_wiring_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_activation_wiring_v1",
                "reason": "peak-residual activation panels rendered, but every activation ledger stayed at zero, so the next honest wall is activation wiring rather than more residual tuning.",
            }
        if anchor_peak_residual_delta > 0.0:
            return {
                "route": "B",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_contraction_realization_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_contraction_realization_v1",
                "reason": "residual-activation ledgers moved, but the hard anchor still kept positive residual post-render peaks, so contraction realization is now the honest wall.",
            }
        if bool(hard_anchor_guard.get("pass", False)) and aggregate_generalization_failed:
            return {
                "route": "C",
                "next_family": "teacher_geometry_render_operator_post_hard_anchor_generalization_audit",
                "next_candidate": "stablelead_rehydrated_operator_post_hard_anchor_generalization_v1",
                "reason": "the hard anchor passed, but the non-anchor smoke cases still blocked honest aggregate generalization.",
            }
        if anchor_peak_delta > 0.0 or anchor_components_delta >= 0.0:
            return {
                "route": "B",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_contraction_realization_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_contraction_realization_v1",
                "reason": "residual activation moved, but the hard anchor still failed to contract components and post-render peaks together.",
            }
        return {
            "route": "A",
            "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_activation_wiring_audit",
            "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_activation_wiring_v1",
            "reason": "peak-residual activation still did not resolve the hard anchor honestly, so the remaining wall is tighter activation wiring.",
        }
    if _is_hard_anchor_peak_rebound_suppression_family(family=family, variant=variant):
        anchor_components_delta = float(hard_anchor_guard.get("delta_fg_connected_components", 0.0))
        anchor_peak_delta = float(hard_anchor_guard.get("delta_fg_peak_count_after_render", 0.0))
        anchor_peak_focus = float(hard_anchor_guard.get("delta_hard_anchor_peak_rebound_focus_fraction", 0.0))
        anchor_peak_binding = float(hard_anchor_guard.get("delta_hard_anchor_peak_rebound_binding_fraction", 0.0))
        anchor_peak_suppression = float(hard_anchor_guard.get("delta_hard_anchor_peak_rebound_suppression_fraction", 0.0))
        anchor_peak_contract_gain = float(hard_anchor_guard.get("delta_hard_anchor_peak_rebound_contract_gain", 0.0))
        suppression_moved = bool(
            anchor_peak_focus > 0.0
            or anchor_peak_binding > 0.0
            or anchor_peak_suppression > 0.0
            or anchor_peak_contract_gain > 0.0
        )
        aggregate_generalization_failed = bool(
            aggregate.get("mean_delta_masked_l1", 0.0) > 0.0
            or aggregate.get("mean_delta_masked_ssim", 0.0) < 0.0
            or aggregate.get("worst_delta_off_body_support_ratio", 0.0) > 0.0
            or aggregate.get("mean_delta_fg_connected_components", 0.0) >= 0.0
            or aggregate.get("mean_delta_fg_peak_count_after_render", 0.0) > 0.0
            or any(not bool(row.get("smoke_case_pass", False)) for row in rows)
        )
        if not suppression_moved:
            return {
                "route": "P1",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_activation_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_activation_v1",
                "reason": "peak-rebound-suppression panels rendered, but the new hard-anchor peak ledgers still stayed at zero, so activation remains the next honest wall.",
            }
        if anchor_peak_delta <= 0.0 and anchor_components_delta >= 0.0:
            return {
                "route": "P2",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_component_contraction_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_component_contraction_v1",
                "reason": "local peak rebound finally came down, but the hard anchor still did not contract visible components honestly.",
            }
        if anchor_peak_delta > 0.0:
            return {
                "route": "P3",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_residual_activation_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_residual_activation_v1",
                "reason": "peak-rebound suppression activated, but residual post-render peaks still remained positive on the hard anchor.",
            }
        if bool(hard_anchor_guard.get("pass", False)) and aggregate_generalization_failed:
            return {
                "route": "C",
                "next_family": "teacher_geometry_render_operator_post_hard_anchor_generalization_audit",
                "next_candidate": "stablelead_rehydrated_operator_post_hard_anchor_generalization_v1",
                "reason": "the hard anchor passed, but the non-anchor smoke cases still blocked honest aggregate generalization.",
            }
        return {
            "route": "P2",
            "next_family": "teacher_geometry_render_operator_hard_anchor_peak_component_contraction_audit",
            "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_component_contraction_v1",
            "reason": "peak rebound moved, but the hard anchor still needs coupled component contraction after suppression.",
        }
    if _is_hard_anchor_breakout_alignment_family(family=family, variant=variant):
        anchor_components_delta = float(hard_anchor_guard.get("delta_fg_connected_components", 0.0))
        anchor_peak_delta = float(hard_anchor_guard.get("delta_fg_peak_count_after_render", 0.0))
        anchor_alignment_fraction = float(hard_anchor_guard.get("delta_hard_anchor_breakout_alignment_fraction", 0.0))
        anchor_alignment_gain = float(hard_anchor_guard.get("delta_hard_anchor_breakout_alignment_gain", 0.0))
        anchor_alignment_centroid = float(hard_anchor_guard.get("delta_hard_anchor_breakout_alignment_centroid_score", 0.0))
        anchor_shrinkage_witness = float(hard_anchor_guard.get("delta_hard_anchor_local_shrinkage_witness_fraction", 0.0))
        alignment_moved = bool(
            anchor_alignment_fraction > 0.0
            or anchor_alignment_gain > 0.0
            or anchor_alignment_centroid > 0.0
            or anchor_shrinkage_witness > 0.0
        )
        aggregate_generalization_failed = bool(
            aggregate.get("mean_delta_masked_l1", 0.0) > 0.0
            or aggregate.get("mean_delta_masked_ssim", 0.0) < 0.0
            or aggregate.get("worst_delta_off_body_support_ratio", 0.0) > 0.0
            or aggregate.get("mean_delta_fg_connected_components", 0.0) >= 0.0
            or aggregate.get("mean_delta_fg_peak_count_after_render", 0.0) > 0.0
            or any(not bool(row.get("smoke_case_pass", False)) for row in rows)
        )
        if not alignment_moved:
            return {
                "route": "A",
                "next_family": "teacher_geometry_render_operator_hard_anchor_alignment_activation_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_alignment_activation_v1",
                "reason": "the breakout-alignment panels rendered, but every alignment ledger stayed at zero, which still looks like an activation or score-plumbing wall.",
            }
        if anchor_peak_delta > 0.0:
            return {
                "route": "B",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_rebound_suppression_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_rebound_suppression_v1",
                "reason": "breakout alignment started to move, but the hard anchor still rebounded on post-render peaks.",
            }
        if bool(hard_anchor_guard.get("pass", False)) and aggregate_generalization_failed:
            return {
                "route": "C",
                "next_family": "teacher_geometry_render_operator_post_hard_anchor_generalization_audit",
                "next_candidate": "stablelead_rehydrated_operator_post_hard_anchor_generalization_v1",
                "reason": "the hard anchor passed, but the two non-anchor smoke cases still blocked honest aggregate generalization.",
            }
        return {
            "route": "A",
            "next_family": "teacher_geometry_render_operator_hard_anchor_alignment_activation_audit",
            "next_candidate": "stablelead_rehydrated_operator_hard_anchor_alignment_activation_v1",
            "reason": "breakout-alignment ledgers moved, but final visible shrinkage still did not contract the hard anchor honestly.",
        }
    if _is_hard_anchor_closure_realization_family(family=family, variant=variant):
        anchor_components_delta = float(hard_anchor_guard.get("delta_fg_connected_components", 0.0))
        anchor_peak_delta = float(hard_anchor_guard.get("delta_fg_peak_count_after_render", 0.0))
        anchor_local_fraction = float(hard_anchor_guard.get("delta_hard_anchor_local_closure_fraction", 0.0))
        anchor_local_gain = float(hard_anchor_guard.get("delta_hard_anchor_local_closure_gain", 0.0))
        anchor_local_weight = float(hard_anchor_guard.get("delta_hard_anchor_local_closure_weight", 0.0))
        anchor_local_peak_gain = float(hard_anchor_guard.get("delta_hard_anchor_local_peak_rebound_suppression_gain", 0.0))
        local_breakout_moved = bool(
            anchor_local_fraction > 0.0
            or anchor_local_gain > 0.0
            or anchor_local_weight > 0.0
            or anchor_local_peak_gain > 0.0
        )
        aggregate_generalization_failed = bool(
            aggregate.get("mean_delta_masked_l1", 0.0) > 0.0
            or aggregate.get("mean_delta_masked_ssim", 0.0) < 0.0
            or aggregate.get("worst_delta_off_body_support_ratio", 0.0) > 0.0
            or aggregate.get("mean_delta_fg_connected_components", 0.0) >= 0.0
            or aggregate.get("mean_delta_fg_peak_count_after_render", 0.0) > 0.0
            or any(not bool(row.get("smoke_case_pass", False)) for row in rows)
        )
        if anchor_components_delta < 0.0 and anchor_peak_delta > 0.0:
            return {
                "route": "H1",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_rebound_suppression_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_rebound_suppression_v1",
                "reason": "the hard anchor finally contracted components, but post-render peak rebound still remained positive.",
            }
        if anchor_components_delta >= 0.0 and not local_breakout_moved:
            return {
                "route": "H2",
                "next_family": "teacher_geometry_render_operator_hard_anchor_breakout_alignment_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_breakout_alignment_v1",
                "reason": "the hard-anchor-local closure breakout never materialized into any measurable local movement, so alignment between breakout intent and final visible shrinkage is still the next honest wall.",
            }
        if local_breakout_moved and (
            anchor_components_delta >= 0.0
            or float(hard_anchor_guard.get("delta_hard_anchor_component_closure_realization_breakout_score", 0.0)) <= 0.0
        ):
            return {
                "route": "H2",
                "next_family": "teacher_geometry_render_operator_hard_anchor_breakout_alignment_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_breakout_alignment_v1",
                "reason": "hard-anchor-local closure maps moved, but the final visible component shrinkage still did not align with the breakout corridor.",
            }
        if bool(hard_anchor_guard.get("pass", False)) and aggregate_generalization_failed:
            return {
                "route": "H3",
                "next_family": "teacher_geometry_render_operator_post_hard_anchor_generalization_audit",
                "next_candidate": "stablelead_rehydrated_operator_post_hard_anchor_generalization_v1",
                "reason": "the hard anchor passed locally, but the other smoke cases still blocked honest aggregate generalization.",
            }
        if anchor_peak_delta > 0.0:
            return {
                "route": "H1",
                "next_family": "teacher_geometry_render_operator_hard_anchor_peak_rebound_suppression_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_peak_rebound_suppression_v1",
                "reason": "the hard anchor still failed on post-render peak suppression.",
            }
        return {
            "route": "H2",
            "next_family": "teacher_geometry_render_operator_hard_anchor_breakout_alignment_audit",
            "next_candidate": "stablelead_rehydrated_operator_hard_anchor_breakout_alignment_v1",
            "reason": "the hard anchor still did not land true local closure realization in the same region as the final visible-component contraction.",
        }
    if _is_component_closure_realization_family(family=family, variant=variant):
        closure_fraction = float(aggregate.get("mean_delta_post_merge_component_closure_realization_fraction", 0.0))
        closure_binding_fraction = float(aggregate.get("mean_delta_post_merge_component_closure_binding_fraction", 0.0))
        closure_gain = float(aggregate.get("mean_delta_post_merge_component_closure_realization_gain", 0.0))
        connected_components_delta = float(aggregate.get("mean_delta_fg_connected_components", 0.0))
        peak_delta = float(aggregate.get("mean_delta_fg_peak_count_after_render", 0.0))
        hard_anchor_blocked = bool(
            not bool(hard_anchor_guard.get("pass", False))
            or float(hard_anchor_guard.get("delta_fg_connected_components", 0.0)) >= 0.0
            or float(hard_anchor_guard.get("delta_fg_peak_count_after_render", 0.0)) > 0.0
            or float(hard_anchor_guard.get("delta_hard_anchor_component_closure_realization_breakout_score", 0.0)) <= 0.0
        )
        if (closure_gain > 0.0 or closure_fraction > 0.0 or closure_binding_fraction > 0.0) and connected_components_delta >= 0.0:
            return {
                "route": "R1",
                "next_family": "teacher_geometry_render_operator_component_gap_realization_contract_audit",
                "next_candidate": "stablelead_rehydrated_operator_component_gap_realization_contract_v1",
                "reason": "closure-realization gains turned on, but final connected components still did not contract, so render binding remains incomplete.",
            }
        if connected_components_delta < 0.0 and peak_delta > 0.0:
            return {
                "route": "R2",
                "next_family": "teacher_geometry_render_operator_component_peak_rebound_suppression_audit",
                "next_candidate": "stablelead_rehydrated_operator_component_peak_rebound_suppression_v1",
                "reason": "closure realization started to reduce fragmentation, but post-render peaks still rebounded.",
            }
        if hard_anchor_blocked:
            return {
                "route": "R3",
                "next_family": "teacher_geometry_render_operator_hard_anchor_closure_realization_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_closure_realization_v1",
                "reason": "aggregate movement improved, but CoreView_390_frame_001170_Camera_B4 still blocks true local closure realization.",
            }
        return {
            "route": "R1",
            "next_family": "teacher_geometry_render_operator_component_gap_realization_contract_audit",
            "next_candidate": "stablelead_rehydrated_operator_component_gap_realization_contract_v1",
            "reason": "closure realization still failed to close the remaining image-space split, so the next honest wall is explicit local gap realization.",
        }
    if _is_component_adjacency_family(family=family, variant=variant):
        adjacency_fraction = float(aggregate.get("mean_delta_post_merge_component_adjacency_fraction", 0.0))
        gap_fraction = float(aggregate.get("mean_delta_post_merge_component_gap_closure_fraction", 0.0))
        adjacency_gain = float(aggregate.get("mean_delta_post_merge_component_adjacency_gain", 0.0))
        closure_realization_gain = float(aggregate.get("mean_delta_post_merge_component_closure_realization_gain", 0.0))
        connected_components_delta = float(aggregate.get("mean_delta_fg_connected_components", 0.0))
        continuity_or_bridge_hole_bad = bool(
            float(aggregate.get("mean_delta_fg_hole_ratio", 0.0)) > 0.0
            or float(aggregate.get("mean_delta_fg_bridge_break_ratio", 0.0)) > 0.0
            or not bool(is_guard_failure_eliminated)
        )
        hard_anchor_blocked = bool(
            bool(hard_anchor_guard.get("local_rewrite_without_merge_rebound", False))
            or float(hard_anchor_guard.get("delta_post_merge_component_adjacency_gain", 0.0)) <= 0.0
            or float(hard_anchor_guard.get("delta_post_merge_component_closure_realization_gain", 0.0)) <= 0.0
            or float(hard_anchor_guard.get("delta_fg_connected_components", 0.0)) >= 0.0
        )
        if adjacency_gain > 0.0 and continuity_or_bridge_hole_bad:
            return {
                "route": "A2",
                "next_family": "teacher_geometry_render_operator_component_closure_continuity_contract_audit",
                "next_candidate": "stablelead_rehydrated_operator_component_closure_continuity_contract_v1",
                "reason": "adjacency closure started to move, but continuity or bridge-hole behavior regressed.",
            }
        if adjacency_gain > 0.0 and hard_anchor_blocked and connected_components_delta < 0.0:
            return {
                "route": "A3",
                "next_family": "teacher_geometry_render_operator_hard_anchor_local_closure_audit",
                "next_candidate": "stablelead_rehydrated_operator_hard_anchor_local_closure_v1",
                "reason": "aggregate fragmentation improved, but the hard anchor still blocked local adjacency closure.",
            }
        if (adjacency_fraction > 0.01 or gap_fraction > 0.01) and (
            closure_realization_gain <= 0.0 or connected_components_delta >= 0.0
        ):
            return {
                "route": "A1",
                "next_family": "teacher_geometry_render_operator_component_closure_realization_audit",
                "next_candidate": "stablelead_rehydrated_operator_component_closure_realization_v1",
                "reason": "adjacency and gap-closure maps moved, but image-space closure still did not realize into fewer components.",
            }
        return {
            "route": "A1",
            "next_family": "teacher_geometry_render_operator_component_closure_realization_audit",
            "next_candidate": "stablelead_rehydrated_operator_component_closure_realization_v1",
            "reason": "the adjacency contract still failed to bind overlay movement into image-space closure, so closure realization is the next honest wall.",
        }
    if _is_component_consolidation_family(family=family, variant=variant):
        consolidation_fraction = float(aggregate.get("mean_delta_post_merge_component_consolidation_fraction", 0.0))
        consolidation_gain = float(aggregate.get("mean_delta_post_merge_component_consolidation_gain", 0.0))
        realization_gain = float(aggregate.get("mean_delta_post_merge_component_realization_gain", 0.0))
        connected_components_delta = float(aggregate.get("mean_delta_fg_connected_components", 0.0))
        continuity_or_bridge_hole_bad = bool(
            float(aggregate.get("mean_delta_fg_hole_ratio", 0.0)) > 0.0
            or float(aggregate.get("mean_delta_fg_bridge_break_ratio", 0.0)) > 0.0
            or not bool(is_guard_failure_eliminated)
        )
        hard_anchor_peak_rebound = bool(
            bool(hard_anchor_guard.get("local_rewrite_without_merge_rebound", False))
            or float(hard_anchor_guard.get("delta_fg_peak_count_after_render", 0.0)) > 0.0
        )
        hard_anchor_not_closed = bool(
            float(hard_anchor_guard.get("delta_fg_connected_components", 0.0)) >= 0.0
            or float(hard_anchor_guard.get("delta_post_merge_component_consolidation_gain", 0.0)) <= 0.0
            or float(hard_anchor_guard.get("delta_post_merge_component_realization_gain", 0.0)) <= 0.0
        )
        if consolidation_gain > 0.0 and (hard_anchor_peak_rebound or hard_anchor_not_closed):
            return {
                "route": "C1",
                "next_family": "teacher_geometry_render_operator_component_breakout_localization_audit",
                "next_candidate": "stablelead_rehydrated_operator_component_breakout_localization_v1",
                "reason": "consolidation gain appeared, but the hard anchor still rebounded on local peak or closure behavior.",
            }
        if consolidation_gain > 0.0 and continuity_or_bridge_hole_bad:
            return {
                "route": "C3",
                "next_family": "teacher_geometry_render_operator_component_consolidation_continuity_contract_audit",
                "next_candidate": "stablelead_rehydrated_operator_component_consolidation_continuity_contract_v1",
                "reason": "component consolidation improved, but continuity or bridge-hole behavior regressed.",
            }
        if consolidation_fraction > 0.01 and connected_components_delta >= 0.0:
            return {
                "route": "C2",
                "next_family": "teacher_geometry_render_operator_component_adjacency_closure_audit",
                "next_candidate": "stablelead_rehydrated_operator_component_adjacency_closure_v1",
                "reason": "the consolidation contract moved, but connected components still did not come down.",
            }
        return {
            "route": "C2",
            "next_family": "teacher_geometry_render_operator_component_adjacency_closure_audit",
            "next_candidate": "stablelead_rehydrated_operator_component_adjacency_closure_v1",
            "reason": "the component-consolidation contract did not convert into lower connected-component counts, so adjacency closure is the next honest wall.",
        }
    if _is_post_merge_realization_family(family=family, variant=variant):
        realization_binding = float(aggregate.get("mean_delta_post_merge_realization_binding_fraction", 0.0))
        realization_gain = float(aggregate.get("mean_delta_post_merge_component_realization_gain", 0.0))
        consolidation_gain = float(aggregate.get("mean_delta_post_merge_component_consolidation_gain", 0.0))
        connected_components_delta = float(aggregate.get("mean_delta_fg_connected_components", 0.0))
        continuity_or_bridge_hole_bad = bool(
            float(aggregate.get("mean_delta_fg_hole_ratio", 0.0)) > 0.0
            or float(aggregate.get("mean_delta_fg_bridge_break_ratio", 0.0)) > 0.0
            or not bool(is_guard_failure_eliminated)
        )
        hard_anchor_peak_rebound = bool(
            bool(hard_anchor_guard.get("local_rewrite_without_merge_rebound", False))
            or float(hard_anchor_guard.get("delta_fg_peak_count_after_render", 0.0)) > 0.0
        )
        hard_anchor_consolidation_unstable = bool(
            float(hard_anchor_guard.get("delta_fg_connected_components", 0.0)) >= 0.0
            or float(hard_anchor_guard.get("delta_post_merge_component_consolidation_gain", 0.0)) <= 0.0
        )
        if realization_gain > 0.0 and hard_anchor_peak_rebound and hard_anchor_consolidation_unstable:
            return {
                "route": "R1",
                "next_family": "teacher_geometry_render_operator_realization_peak_localization_audit",
                "next_candidate": "stablelead_rehydrated_operator_realization_peak_localization_v1",
                "reason": "realization gain appeared, but the hard anchor still rebounded on peak localization and consolidation stayed unstable.",
            }
        if consolidation_gain > 0.0 and continuity_or_bridge_hole_bad:
            return {
                "route": "R3",
                "next_family": "teacher_geometry_render_operator_realization_continuity_contract_audit",
                "next_candidate": "stablelead_rehydrated_operator_realization_continuity_contract_v1",
                "reason": "realization improved consolidation, but continuity or bridge-hole behavior regressed.",
            }
        if realization_binding > 0.01 and connected_components_delta >= 0.0:
            return {
                "route": "R2",
                "next_family": "teacher_geometry_render_operator_component_consolidation_contract_audit",
                "next_candidate": "stablelead_rehydrated_operator_component_consolidation_contract_v1",
                "reason": "realization binding moved, but component counts did not come down so consolidation is still missing.",
            }
        return {
            "route": "",
            "next_family": "",
            "next_candidate": "",
            "reason": "",
        }
    if _is_peak_stabilization_family(family=family, variant=variant):
        merge_gain = float(aggregate.get("mean_delta_post_operator_component_merge_gain", 0.0))
        peak_gain = float(aggregate.get("mean_delta_post_operator_peak_gain", 0.0))
        peak_count_after = float(aggregate.get("mean_delta_fg_peak_count_after_render", 0.0))
        hard_anchor_old_peak_stable = bool(
            float(hard_anchor_guard.get("delta_fg_peak_count_after_render", 0.0)) >= 0.0
            and float(hard_anchor_guard.get("delta_hard_anchor_peak_residual", 0.0)) >= 0.0
        )
        other_rows = [row for row in rows if row["case_id"] != "CoreView_390_frame_001170_Camera_B4"]
        other_continuity_regressed = bool(
            any(
                float(row.get("delta_fg_hole_ratio", 0.0)) > 0.0
                or float(row.get("delta_fg_bridge_break_ratio", 0.0)) > 0.0
                for row in other_rows
            )
            or not bool(is_guard_failure_eliminated)
        )
        hard_anchor_improved = bool(
            float(hard_anchor_guard.get("delta_post_operator_component_merge_gain", 0.0)) > 0.0
            and float(hard_anchor_guard.get("delta_post_operator_peak_gain", 0.0)) > 0.0
        )
        if merge_gain > 0.0 and peak_count_after >= 0.0 and hard_anchor_old_peak_stable:
            return {
                "route": "P1",
                "next_family": "teacher_geometry_render_operator_peak_localization_contract_audit",
                "next_candidate": "stablelead_rehydrated_operator_peak_localization_contract_v1",
                "reason": "component merge gain moved, but peak count did not fall and the hard anchor kept the old peak location.",
            }
        if peak_gain > 0.0 and merge_gain <= 0.01:
            return {
                "route": "P2",
                "next_family": "teacher_geometry_render_operator_post_merge_realization_audit",
                "next_candidate": "stablelead_rehydrated_operator_post_merge_realization_v1",
                "reason": "peak gain moved, but component merge gain stayed near zero so realization is still missing.",
            }
        if hard_anchor_improved and other_continuity_regressed:
            return {
                "route": "P3",
                "next_family": "teacher_geometry_render_operator_peak_continuity_contract_audit",
                "next_candidate": "stablelead_rehydrated_operator_peak_continuity_contract_v1",
                "reason": "the hard anchor improved, but the other smoke cases regressed on continuity or bridge-hole behavior.",
            }
        return {
            "route": "",
            "next_family": "",
            "next_candidate": "",
            "reason": "",
        }
    merge_intent = float(aggregate.get("mean_delta_merge_intent_fraction", 0.0))
    merge_adoption = float(aggregate.get("mean_delta_merge_adoption_fraction", 0.0))
    merge_gain = float(aggregate.get("mean_delta_post_operator_component_merge_gain", 0.0))
    continuity_or_bridge_hole_bad = bool(
        float(aggregate.get("mean_delta_fg_hole_ratio", 0.0)) > 0.0
        or float(aggregate.get("mean_delta_fg_bridge_break_ratio", 0.0)) > 0.0
        or float(aggregate.get("mean_delta_merge_veto_continuity_fraction", 0.0)) > 0.0
        or float(aggregate.get("mean_delta_merge_veto_bridge_hole_fraction", 0.0)) > 0.0
        or not bool(is_guard_failure_eliminated)
    )
    hard_anchor_peak_rebound = bool(float(hard_anchor_guard.get("delta_fg_peak_count_after_render", 0.0)) >= 0.0)
    components_reduced = bool(float(aggregate.get("mean_delta_fg_connected_components", 0.0)) < 0.0)
    if merge_intent > 0.01 and merge_adoption <= 0.01:
        return {
            "route": "M1",
            "next_family": "teacher_geometry_render_operator_merge_binding_audit",
            "next_candidate": "stablelead_rehydrated_operator_merge_binding_v1",
            "reason": "merge intent appeared, but adoption stayed near zero.",
        }
    if merge_gain > 0.01 and continuity_or_bridge_hole_bad:
        return {
            "route": "M2",
            "next_family": "teacher_geometry_render_operator_merge_continuity_contract_audit",
            "next_candidate": "stablelead_rehydrated_operator_merge_continuity_contract_v1",
            "reason": "merge gain moved, but continuity or bridge-hole behavior regressed.",
        }
    if merge_gain > 0.01 and components_reduced and hard_anchor_peak_rebound:
        return {
            "route": "M3",
            "next_family": "teacher_geometry_render_operator_peak_stabilization_audit",
            "next_candidate": "stablelead_rehydrated_operator_peak_stabilization_v1",
            "reason": "merge gain moved and components dropped, but the hard anchor still showed peak rebound.",
        }
    return {
        "route": "",
        "next_family": "",
        "next_candidate": "",
        "reason": "",
    }


def _guard_attribution(row: dict) -> dict:
    guard = row.get("guard_report", {})
    coverage_due_to = []
    largest_due_to = []
    masked_l1_due_to = []
    masked_ssim_due_to = []
    if not bool(guard.get("fg_coverage_floor", {}).get("pass", True)):
        if row["delta_fg_hole_ratio"] > 0:
            coverage_due_to.append("fg_hole_ratio_up")
        if row["delta_fg_bridge_break_ratio"] > 0:
            coverage_due_to.append("fg_bridge_break_ratio_up")
        if row["delta_fg_visible_component_count"] > 0 or row["delta_fg_visible_component_entropy"] > 0:
            coverage_due_to.append("fragmentation_persistence")
        if row["fg_visible_rgb_coverage_ratio"] < 0.60:
            coverage_due_to.append("visible_rgb_coverage_low")
    if not bool(guard.get("largest_component_floor", {}).get("pass", True)):
        if row["delta_fg_visible_component_count"] >= 0:
            largest_due_to.append("components_not_reduced")
        if row["delta_fg_duplicate_lobe_ratio"] >= 0:
            largest_due_to.append("duplicate_lobes_not_reduced")
        if row["delta_fg_peak_count_after_render"] >= 0:
            largest_due_to.append("peak_rebound")
    if not bool(guard.get("masked_quality_guard", {}).get("pass", True)):
        if row["delta_masked_l1"] > 0:
            if row["delta_fg_nonblack_residual_ratio"] > 0:
                masked_l1_due_to.append("inside_fg_residual_regression")
            if row["delta_fg_hole_ratio"] > 0:
                masked_l1_due_to.append("coverage_hole_regression")
            if row["delta_fg_bridge_break_ratio"] > 0:
                masked_l1_due_to.append("bridge_break_regression")
        if row["delta_masked_ssim"] < 0:
            if row["delta_fg_visible_component_entropy"] > 0:
                masked_ssim_due_to.append("fragmentation_entropy_up")
            if row["delta_fg_duplicate_lobe_ratio"] >= 0 or row["delta_fg_multilayer_overlap_ratio"] >= 0:
                masked_ssim_due_to.append("secondary_lobe_structure_persists")
            if row["delta_fg_hole_ratio"] > 0 or row["delta_fg_bridge_break_ratio"] > 0:
                masked_ssim_due_to.append("topology_regression")
    return {
        "coverage_floor_fail_due_to": coverage_due_to,
        "largest_component_floor_fail_due_to": largest_due_to,
        "masked_l1_regression_due_to": masked_l1_due_to,
        "masked_ssim_regression_due_to": masked_ssim_due_to,
    }


def build_summary(summary: dict, variant: str, baseline_variant: str, *, family: str = "") -> dict:
    baseline = _rows_by_variant(summary, baseline_variant)
    candidate = _rows_by_variant(summary, variant)
    case_ids = sorted(set(baseline) & set(candidate))
    rows = []
    for case_id in case_ids:
        base_row = baseline[case_id]
        cand_row = candidate[case_id]
        base_support = base_row["support_metrics"]
        cand_support = cand_row["support_metrics"]
        base_comp = base_row.get("composition_metrics", {})
        cand_comp = cand_row.get("composition_metrics", {})
        row = {
            "case_id": case_id,
            "delta_fg_visible_component_count": float(cand_support["fg_visible_component_count"] - base_support["fg_visible_component_count"]),
            "delta_largest_fg_visible_component_ratio": float(cand_support["largest_fg_visible_component_ratio"] - base_support["largest_fg_visible_component_ratio"]),
            "delta_second_fg_visible_component_ratio": float(cand_support["second_fg_visible_component_ratio"] - base_support["second_fg_visible_component_ratio"]),
            "delta_fg_visible_component_entropy": float(cand_support["fg_visible_component_entropy"] - base_support["fg_visible_component_entropy"]),
            "delta_fg_secondary_mass_ratio": float(cand_support["fg_secondary_mass_ratio"] - base_support["fg_secondary_mass_ratio"]),
            "delta_fg_top2_visible_lobe_gap": float(cand_support["fg_top2_visible_lobe_gap"] - base_support["fg_top2_visible_lobe_gap"]),
            "delta_fg_multilayer_overlap_ratio": float(cand_support["fg_multilayer_overlap_ratio"] - base_support["fg_multilayer_overlap_ratio"]),
            "delta_fg_duplicate_lobe_ratio": float(cand_support["fg_duplicate_lobe_ratio"] - base_support["fg_duplicate_lobe_ratio"]),
            "delta_fg_nonblack_residual_ratio": float(cand_support["fg_nonblack_residual_ratio"] - base_support["fg_nonblack_residual_ratio"]),
            "delta_fg_hole_ratio": float(cand_support["fg_hole_ratio"] - base_support["fg_hole_ratio"]),
            "delta_fg_bridge_break_ratio": float(cand_support["fg_bridge_break_ratio"] - base_support["fg_bridge_break_ratio"]),
            "delta_fg_peak_count_after_render": float(cand_support["fg_peak_count_after_render"] - base_support["fg_peak_count_after_render"]),
            "delta_fg_peak_count": float(cand_support["fg_peak_count"] - base_support["fg_peak_count"]),
            "delta_fg_connected_components": float(cand_support["fg_connected_components"] - base_support["fg_connected_components"]),
            "delta_masked_l1": float(cand_row["metrics"]["fg_masked"]["l1"] - base_row["metrics"]["fg_masked"]["l1"]),
            "delta_masked_ssim": float(cand_row["metrics"]["fg_masked"]["ssim"] - base_row["metrics"]["fg_masked"]["ssim"]),
            "delta_off_body_support_ratio": float(cand_support["off_body_support_ratio"] - base_support["off_body_support_ratio"]),
            "delta_bg_bottom_support_ratio": float(cand_support["bg_bottom_support_ratio"] - base_support["bg_bottom_support_ratio"]),
            "delta_primary_consistency_score": float(cand_comp.get("primary_consistency_score", 0.0) - base_comp.get("primary_consistency_score", 0.0)),
            "delta_secondary_conflict_score": float(cand_comp.get("secondary_conflict_score", 0.0) - base_comp.get("secondary_conflict_score", 0.0)),
            "delta_composition_penalty_score": float(cand_comp.get("composition_penalty_score", 0.0) - base_comp.get("composition_penalty_score", 0.0)),
            "delta_resolved_to_primary_fraction": float(cand_comp.get("resolved_to_primary_fraction", 0.0) - base_comp.get("resolved_to_primary_fraction", 0.0)),
            "delta_resolved_to_secondary_fraction": float(cand_comp.get("resolved_to_secondary_fraction", 0.0) - base_comp.get("resolved_to_secondary_fraction", 0.0)),
            "delta_resolved_to_abstain_fraction": float(cand_comp.get("resolved_to_abstain_fraction", 0.0) - base_comp.get("resolved_to_abstain_fraction", 0.0)),
            "delta_primary_owned_fraction": float(cand_comp.get("primary_owned_fraction", cand_comp.get("resolved_to_primary_fraction", 0.0)) - base_comp.get("primary_owned_fraction", base_comp.get("resolved_to_primary_fraction", 0.0))),
            "delta_secondary_owned_fraction": float(cand_comp.get("secondary_owned_fraction", cand_comp.get("resolved_to_secondary_fraction", 0.0)) - base_comp.get("secondary_owned_fraction", base_comp.get("resolved_to_secondary_fraction", 0.0))),
            "delta_continuity_protected_fraction": float(cand_comp.get("continuity_protected_fraction", 0.0) - base_comp.get("continuity_protected_fraction", 0.0)),
            "delta_conflict_resolved_fraction": float(cand_comp.get("conflict_resolved_fraction", 0.0) - base_comp.get("conflict_resolved_fraction", 0.0)),
            "delta_merge_intent_fraction": float(cand_comp.get("merge_intent_fraction", 0.0) - base_comp.get("merge_intent_fraction", 0.0)),
            "delta_merge_adoption_fraction": float(cand_comp.get("merge_adoption_fraction", 0.0) - base_comp.get("merge_adoption_fraction", 0.0)),
            "delta_merge_veto_fraction": float(cand_comp.get("merge_veto_fraction", 0.0) - base_comp.get("merge_veto_fraction", 0.0)),
            "delta_merge_veto_continuity_fraction": float(cand_comp.get("merge_veto_continuity_fraction", 0.0) - base_comp.get("merge_veto_continuity_fraction", 0.0)),
            "delta_merge_veto_bridge_hole_fraction": float(cand_comp.get("merge_veto_bridge_hole_fraction", 0.0) - base_comp.get("merge_veto_bridge_hole_fraction", 0.0)),
            "delta_merge_veto_overlap_insufficient_fraction": float(cand_comp.get("merge_veto_overlap_insufficient_fraction", 0.0) - base_comp.get("merge_veto_overlap_insufficient_fraction", 0.0)),
            "delta_peak_stabilization_intent_fraction": float(cand_comp.get("peak_stabilization_intent_fraction", 0.0) - base_comp.get("peak_stabilization_intent_fraction", 0.0)),
            "delta_peak_stabilization_adoption_fraction": float(cand_comp.get("peak_stabilization_adoption_fraction", 0.0) - base_comp.get("peak_stabilization_adoption_fraction", 0.0)),
            "delta_peak_stabilization_veto_fraction": float(cand_comp.get("peak_stabilization_veto_fraction", 0.0) - base_comp.get("peak_stabilization_veto_fraction", 0.0)),
            "delta_post_merge_realization_fraction": float(cand_comp.get("post_merge_realization_fraction", 0.0) - base_comp.get("post_merge_realization_fraction", 0.0)),
            "delta_post_merge_realization_intent_fraction": float(cand_comp.get("post_merge_realization_intent_fraction", 0.0) - base_comp.get("post_merge_realization_intent_fraction", 0.0)),
            "delta_post_merge_realization_binding_fraction": float(cand_comp.get("post_merge_realization_binding_fraction", 0.0) - base_comp.get("post_merge_realization_binding_fraction", 0.0)),
            "delta_post_merge_realization_adoption_fraction": float(cand_comp.get("post_merge_realization_adoption_fraction", 0.0) - base_comp.get("post_merge_realization_adoption_fraction", 0.0)),
            "delta_post_merge_component_binding_fraction": float(cand_comp.get("post_merge_component_binding_fraction", 0.0) - base_comp.get("post_merge_component_binding_fraction", 0.0)),
            "delta_post_merge_component_consolidation_fraction": float(cand_comp.get("post_merge_component_consolidation_fraction", 0.0) - base_comp.get("post_merge_component_consolidation_fraction", 0.0)),
            "delta_post_merge_component_adjacency_fraction": float(cand_comp.get("post_merge_component_adjacency_fraction", 0.0) - base_comp.get("post_merge_component_adjacency_fraction", 0.0)),
            "delta_post_merge_component_gap_closure_fraction": float(cand_comp.get("post_merge_component_gap_closure_fraction", 0.0) - base_comp.get("post_merge_component_gap_closure_fraction", 0.0)),
            "delta_post_merge_component_closure_realization_fraction": float(cand_comp.get("post_merge_component_closure_realization_fraction", 0.0) - base_comp.get("post_merge_component_closure_realization_fraction", 0.0)),
            "delta_post_merge_component_closure_binding_fraction": float(cand_comp.get("post_merge_component_closure_binding_fraction", 0.0) - base_comp.get("post_merge_component_closure_binding_fraction", 0.0)),
            "delta_post_merge_peak_containment_gain": float(cand_comp.get("post_merge_peak_containment_gain", 0.0) - base_comp.get("post_merge_peak_containment_gain", 0.0)),
            "delta_post_merge_component_consolidation_gain": float(cand_comp.get("post_merge_component_consolidation_gain", 0.0) - base_comp.get("post_merge_component_consolidation_gain", 0.0)),
            "delta_post_merge_component_realization_gain": float(cand_comp.get("post_merge_component_realization_gain", 0.0) - base_comp.get("post_merge_component_realization_gain", 0.0)),
            "delta_post_merge_component_adjacency_gain": float(cand_comp.get("post_merge_component_adjacency_gain", 0.0) - base_comp.get("post_merge_component_adjacency_gain", 0.0)),
            "delta_post_merge_component_closure_realization_gain": float(cand_comp.get("post_merge_component_closure_realization_gain", 0.0) - base_comp.get("post_merge_component_closure_realization_gain", 0.0)),
            "delta_post_merge_component_adjacency_residual_split_fraction": float(cand_comp.get("post_merge_component_adjacency_residual_split_fraction", 0.0) - base_comp.get("post_merge_component_adjacency_residual_split_fraction", 0.0)),
            "delta_post_merge_component_closure_residual_split_fraction": float(cand_comp.get("post_merge_component_closure_residual_split_fraction", 0.0) - base_comp.get("post_merge_component_closure_residual_split_fraction", 0.0)),
            "delta_post_merge_component_residual_split_fraction": float(cand_comp.get("post_merge_component_residual_split_fraction", 0.0) - base_comp.get("post_merge_component_residual_split_fraction", 0.0)),
            "delta_post_merge_peak_residual_after_realization": float(cand_comp.get("post_merge_peak_residual_after_realization", 0.0) - base_comp.get("post_merge_peak_residual_after_realization", 0.0)),
            "delta_hard_anchor_peak_residual": float(cand_comp.get("hard_anchor_peak_residual", 0.0) - base_comp.get("hard_anchor_peak_residual", 0.0)),
            "delta_hard_anchor_realization_breakout_score": float(cand_comp.get("hard_anchor_realization_breakout_score", 0.0) - base_comp.get("hard_anchor_realization_breakout_score", 0.0)),
            "delta_hard_anchor_component_consolidation_breakout_score": float(cand_comp.get("hard_anchor_component_consolidation_breakout_score", 0.0) - base_comp.get("hard_anchor_component_consolidation_breakout_score", 0.0)),
            "delta_hard_anchor_component_adjacency_breakout_score": float(cand_comp.get("hard_anchor_component_adjacency_breakout_score", 0.0) - base_comp.get("hard_anchor_component_adjacency_breakout_score", 0.0)),
            "delta_hard_anchor_component_closure_realization_breakout_score": float(cand_comp.get("hard_anchor_component_closure_realization_breakout_score", 0.0) - base_comp.get("hard_anchor_component_closure_realization_breakout_score", 0.0)),
            "delta_hard_anchor_local_closure_fraction": float(cand_comp.get("hard_anchor_local_closure_fraction", 0.0) - base_comp.get("hard_anchor_local_closure_fraction", 0.0)),
            "delta_hard_anchor_local_closure_gain": float(cand_comp.get("hard_anchor_local_closure_gain", 0.0) - base_comp.get("hard_anchor_local_closure_gain", 0.0)),
            "delta_hard_anchor_local_closure_residual_split_fraction": float(cand_comp.get("hard_anchor_local_closure_residual_split_fraction", 0.0) - base_comp.get("hard_anchor_local_closure_residual_split_fraction", 0.0)),
            "delta_hard_anchor_local_peak_rebound_suppression_gain": float(cand_comp.get("hard_anchor_local_peak_rebound_suppression_gain", 0.0) - base_comp.get("hard_anchor_local_peak_rebound_suppression_gain", 0.0)),
            "delta_hard_anchor_local_closure_weight": float(cand_comp.get("hard_anchor_local_closure_weight", 0.0) - base_comp.get("hard_anchor_local_closure_weight", 0.0)),
            "delta_hard_anchor_breakout_target_fraction": float(cand_comp.get("hard_anchor_breakout_target_fraction", 0.0) - base_comp.get("hard_anchor_breakout_target_fraction", 0.0)),
            "delta_hard_anchor_breakout_alignment_fraction": float(cand_comp.get("hard_anchor_breakout_alignment_fraction", 0.0) - base_comp.get("hard_anchor_breakout_alignment_fraction", 0.0)),
            "delta_hard_anchor_local_shrinkage_witness_fraction": float(cand_comp.get("hard_anchor_local_shrinkage_witness_fraction", 0.0) - base_comp.get("hard_anchor_local_shrinkage_witness_fraction", 0.0)),
            "delta_hard_anchor_breakout_misalignment_veto_fraction": float(cand_comp.get("hard_anchor_breakout_misalignment_veto_fraction", 0.0) - base_comp.get("hard_anchor_breakout_misalignment_veto_fraction", 0.0)),
            "delta_hard_anchor_breakout_alignment_centroid_score": float(cand_comp.get("hard_anchor_breakout_alignment_centroid_score", 0.0) - base_comp.get("hard_anchor_breakout_alignment_centroid_score", 0.0)),
            "delta_hard_anchor_breakout_alignment_gain": float(cand_comp.get("hard_anchor_breakout_alignment_gain", 0.0) - base_comp.get("hard_anchor_breakout_alignment_gain", 0.0)),
            "delta_hard_anchor_peak_rebound_focus_fraction": float(cand_comp.get("hard_anchor_peak_rebound_focus_fraction", 0.0) - base_comp.get("hard_anchor_peak_rebound_focus_fraction", 0.0)),
            "delta_hard_anchor_peak_rebound_binding_fraction": float(cand_comp.get("hard_anchor_peak_rebound_binding_fraction", 0.0) - base_comp.get("hard_anchor_peak_rebound_binding_fraction", 0.0)),
            "delta_hard_anchor_peak_rebound_veto_fraction": float(cand_comp.get("hard_anchor_peak_rebound_veto_fraction", 0.0) - base_comp.get("hard_anchor_peak_rebound_veto_fraction", 0.0)),
            "delta_hard_anchor_peak_rebound_suppression_fraction": float(cand_comp.get("hard_anchor_peak_rebound_suppression_fraction", 0.0) - base_comp.get("hard_anchor_peak_rebound_suppression_fraction", 0.0)),
            "delta_hard_anchor_peak_rebound_residual_fraction": float(cand_comp.get("hard_anchor_peak_rebound_residual_fraction", 0.0) - base_comp.get("hard_anchor_peak_rebound_residual_fraction", 0.0)),
            "delta_hard_anchor_peak_rebound_contract_gain": float(cand_comp.get("hard_anchor_peak_rebound_contract_gain", 0.0) - base_comp.get("hard_anchor_peak_rebound_contract_gain", 0.0)),
            "delta_hard_anchor_peak_residual_activation_fraction": float(cand_comp.get("hard_anchor_peak_residual_activation_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_activation_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_binding_fraction": float(cand_comp.get("hard_anchor_peak_residual_binding_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_binding_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_veto_fraction": float(cand_comp.get("hard_anchor_peak_residual_veto_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_veto_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_contraction_fraction": float(cand_comp.get("hard_anchor_peak_residual_contraction_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_contraction_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_activation_gain": float(cand_comp.get("hard_anchor_peak_residual_activation_gain", 0.0) - base_comp.get("hard_anchor_peak_residual_activation_gain", 0.0)),
            "delta_hard_anchor_peak_residual_remaining_fraction": float(cand_comp.get("hard_anchor_peak_residual_remaining_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_remaining_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_contraction_target_fraction": float(cand_comp.get("hard_anchor_peak_residual_contraction_target_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_contraction_target_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_contraction_binding_fraction": float(cand_comp.get("hard_anchor_peak_residual_contraction_binding_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_contraction_binding_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_contraction_realization_fraction": float(cand_comp.get("hard_anchor_peak_residual_contraction_realization_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_contraction_realization_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_contraction_overlap_fraction": float(cand_comp.get("hard_anchor_peak_residual_contraction_overlap_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_contraction_overlap_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_contraction_gain": float(cand_comp.get("hard_anchor_peak_residual_contraction_gain", 0.0) - base_comp.get("hard_anchor_peak_residual_contraction_gain", 0.0)),
            "delta_hard_anchor_peak_residual_contraction_alignment_score": float(cand_comp.get("hard_anchor_peak_residual_contraction_alignment_score", 0.0) - base_comp.get("hard_anchor_peak_residual_contraction_alignment_score", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_target_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_target_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_target_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_binding_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_binding_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_binding_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_realization_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_realization_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_realization_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_alignment_score": float(cand_comp.get("hard_anchor_peak_residual_overlap_alignment_score", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_alignment_score", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_target_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_target_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_target_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_binding_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_binding_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_binding_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_realization_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_realization_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_realization_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_alignment_score": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_alignment_score", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_alignment_score", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_remaining_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_remaining_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_remaining_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_amplification_target_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_amplification_target_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_amplification_target_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_amplification_binding_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_amplification_binding_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_amplification_binding_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_amplification_realization_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_amplification_realization_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_amplification_realization_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_amplification_gain": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_amplification_gain", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_amplification_gain", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_amplification_alignment_score": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_amplification_alignment_score", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_amplification_alignment_score", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_amplification_remaining_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_amplification_remaining_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_amplification_remaining_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_amplification_target_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_amplification_target_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_amplification_target_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_amplification_binding_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_amplification_binding_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_amplification_binding_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_amplification_realization_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_amplification_realization_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_amplification_gain": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_amplification_gain", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_amplification_gain", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_amplification_alignment_score": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_amplification_alignment_score", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_amplification_alignment_score", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_amplification_remaining_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_amplification_remaining_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_amplification_remaining_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_target_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_target_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_target_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_binding_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_binding_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_binding_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_realization_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_realization_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_realization_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_gain": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_gain", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_gain", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_alignment_score": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_alignment_score", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_alignment_score", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_remaining_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_remaining_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_remaining_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_target_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_target_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_target_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_binding_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_binding_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_binding_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_contract_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_contract_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_contract_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_gain": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_gain", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_gain", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_alignment_score": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_alignment_score", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_alignment_score", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_remaining_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_remaining_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_remaining_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_target_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_target_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_target_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_binding_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_binding_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_binding_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_contract_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_contract_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_contract_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_gain": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_gain", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_gain", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_alignment_score": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_alignment_score", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_alignment_score", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_remaining_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_remaining_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_remaining_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_target_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_target_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_target_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_binding_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_binding_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_binding_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_realization_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_realization_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_realization_fraction", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_gain": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_gain", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_gain", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_alignment_score": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_alignment_score", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_alignment_score", 0.0)),
            "delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_remaining_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_remaining_fraction", 0.0) - base_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_remaining_fraction", 0.0)),
            "delta_post_operator_component_merge_gain": float(cand_comp.get("post_operator_component_merge_gain", 0.0) - base_comp.get("post_operator_component_merge_gain", 0.0)),
            "delta_post_operator_peak_gain": float(cand_comp.get("post_operator_peak_gain", 0.0) - base_comp.get("post_operator_peak_gain", 0.0)),
            "delta_ownership_primary_affinity": float(cand_comp.get("ownership_primary_affinity", 0.0) - base_comp.get("ownership_primary_affinity", 0.0)),
            "delta_ownership_secondary_affinity": float(cand_comp.get("ownership_secondary_affinity", 0.0) - base_comp.get("ownership_secondary_affinity", 0.0)),
            "delta_ownership_margin": float(cand_comp.get("ownership_margin", 0.0) - base_comp.get("ownership_margin", 0.0)),
            "delta_ownership_actionability": float(cand_comp.get("ownership_actionability", 0.0) - base_comp.get("ownership_actionability", 0.0)),
            "delta_ownership_bridge_risk": float(cand_comp.get("ownership_bridge_risk", 0.0) - base_comp.get("ownership_bridge_risk", 0.0)),
            "delta_ownership_hole_risk": float(cand_comp.get("ownership_hole_risk", 0.0) - base_comp.get("ownership_hole_risk", 0.0)),
            "delta_ownership_thin_structure_risk": float(cand_comp.get("ownership_thin_structure_risk", 0.0) - base_comp.get("ownership_thin_structure_risk", 0.0)),
            "fg_visible_component_count": int(cand_support["fg_visible_component_count"]),
            "largest_fg_visible_component_ratio": float(cand_support["largest_fg_visible_component_ratio"]),
            "second_fg_visible_component_ratio": float(cand_support["second_fg_visible_component_ratio"]),
            "fg_visible_component_entropy": float(cand_support["fg_visible_component_entropy"]),
            "fg_secondary_mass_ratio": float(cand_support["fg_secondary_mass_ratio"]),
            "fg_top2_visible_lobe_gap": float(cand_support["fg_top2_visible_lobe_gap"]),
            "fg_multilayer_overlap_ratio": float(cand_support["fg_multilayer_overlap_ratio"]),
            "fg_duplicate_lobe_ratio": float(cand_support["fg_duplicate_lobe_ratio"]),
            "fg_nonblack_residual_ratio": float(cand_support["fg_nonblack_residual_ratio"]),
            "fg_hole_ratio": float(cand_support["fg_hole_ratio"]),
            "fg_bridge_break_ratio": float(cand_support["fg_bridge_break_ratio"]),
            "fg_peak_count_after_render": int(cand_support["fg_peak_count_after_render"]),
            "fg_visible_rgb_coverage_ratio": float(cand_support.get("fg_visible_rgb_coverage_ratio", 0.0)),
            "fg_visible_mass_ratio": float(cand_support.get("fg_visible_mass_ratio", 0.0)),
            "off_body_support_ratio": float(cand_support["off_body_support_ratio"]),
            "primary_consistency_score": float(cand_comp.get("primary_consistency_score", 0.0)),
            "secondary_conflict_score": float(cand_comp.get("secondary_conflict_score", 0.0)),
            "composition_penalty_score": float(cand_comp.get("composition_penalty_score", 0.0)),
            "resolved_to_primary_fraction": float(cand_comp.get("resolved_to_primary_fraction", 0.0)),
            "resolved_to_secondary_fraction": float(cand_comp.get("resolved_to_secondary_fraction", 0.0)),
            "resolved_to_abstain_fraction": float(cand_comp.get("resolved_to_abstain_fraction", 0.0)),
            "primary_owned_fraction": float(cand_comp.get("primary_owned_fraction", cand_comp.get("resolved_to_primary_fraction", 0.0))),
            "secondary_owned_fraction": float(cand_comp.get("secondary_owned_fraction", cand_comp.get("resolved_to_secondary_fraction", 0.0))),
            "continuity_protected_fraction": float(cand_comp.get("continuity_protected_fraction", 0.0)),
            "conflict_resolved_fraction": float(cand_comp.get("conflict_resolved_fraction", 0.0)),
            "peak_stabilization_intent_fraction": float(cand_comp.get("peak_stabilization_intent_fraction", 0.0)),
            "peak_stabilization_adoption_fraction": float(cand_comp.get("peak_stabilization_adoption_fraction", 0.0)),
            "peak_stabilization_veto_fraction": float(cand_comp.get("peak_stabilization_veto_fraction", 0.0)),
            "post_merge_realization_fraction": float(cand_comp.get("post_merge_realization_fraction", 0.0)),
            "post_merge_realization_intent_fraction": float(cand_comp.get("post_merge_realization_intent_fraction", 0.0)),
            "post_merge_realization_binding_fraction": float(cand_comp.get("post_merge_realization_binding_fraction", 0.0)),
            "post_merge_realization_adoption_fraction": float(cand_comp.get("post_merge_realization_adoption_fraction", 0.0)),
            "post_merge_component_binding_fraction": float(cand_comp.get("post_merge_component_binding_fraction", 0.0)),
            "post_merge_component_consolidation_fraction": float(cand_comp.get("post_merge_component_consolidation_fraction", 0.0)),
            "post_merge_component_adjacency_fraction": float(cand_comp.get("post_merge_component_adjacency_fraction", 0.0)),
            "post_merge_component_gap_closure_fraction": float(cand_comp.get("post_merge_component_gap_closure_fraction", 0.0)),
            "post_merge_component_closure_realization_fraction": float(cand_comp.get("post_merge_component_closure_realization_fraction", 0.0)),
            "post_merge_component_closure_binding_fraction": float(cand_comp.get("post_merge_component_closure_binding_fraction", 0.0)),
            "post_merge_peak_containment_gain": float(cand_comp.get("post_merge_peak_containment_gain", 0.0)),
            "post_merge_component_consolidation_gain": float(cand_comp.get("post_merge_component_consolidation_gain", 0.0)),
            "post_merge_component_realization_gain": float(cand_comp.get("post_merge_component_realization_gain", 0.0)),
            "post_merge_component_adjacency_gain": float(cand_comp.get("post_merge_component_adjacency_gain", 0.0)),
            "post_merge_component_closure_realization_gain": float(cand_comp.get("post_merge_component_closure_realization_gain", 0.0)),
            "post_merge_component_adjacency_residual_split_fraction": float(cand_comp.get("post_merge_component_adjacency_residual_split_fraction", 0.0)),
            "post_merge_component_closure_residual_split_fraction": float(cand_comp.get("post_merge_component_closure_residual_split_fraction", 0.0)),
            "post_merge_component_residual_split_fraction": float(cand_comp.get("post_merge_component_residual_split_fraction", 0.0)),
            "post_merge_peak_residual_after_realization": float(cand_comp.get("post_merge_peak_residual_after_realization", 0.0)),
            "hard_anchor_peak_residual": float(cand_comp.get("hard_anchor_peak_residual", 0.0)),
            "hard_anchor_realization_breakout_score": float(cand_comp.get("hard_anchor_realization_breakout_score", 0.0)),
            "hard_anchor_component_consolidation_breakout_score": float(cand_comp.get("hard_anchor_component_consolidation_breakout_score", 0.0)),
            "hard_anchor_component_adjacency_breakout_score": float(cand_comp.get("hard_anchor_component_adjacency_breakout_score", 0.0)),
            "hard_anchor_component_closure_realization_breakout_score": float(cand_comp.get("hard_anchor_component_closure_realization_breakout_score", 0.0)),
            "hard_anchor_local_closure_fraction": float(cand_comp.get("hard_anchor_local_closure_fraction", 0.0)),
            "hard_anchor_local_closure_gain": float(cand_comp.get("hard_anchor_local_closure_gain", 0.0)),
            "hard_anchor_local_closure_residual_split_fraction": float(cand_comp.get("hard_anchor_local_closure_residual_split_fraction", 0.0)),
            "hard_anchor_local_peak_rebound_suppression_gain": float(cand_comp.get("hard_anchor_local_peak_rebound_suppression_gain", 0.0)),
            "hard_anchor_local_closure_weight": float(cand_comp.get("hard_anchor_local_closure_weight", 0.0)),
            "hard_anchor_breakout_target_fraction": float(cand_comp.get("hard_anchor_breakout_target_fraction", 0.0)),
            "hard_anchor_breakout_alignment_fraction": float(cand_comp.get("hard_anchor_breakout_alignment_fraction", 0.0)),
            "hard_anchor_local_shrinkage_witness_fraction": float(cand_comp.get("hard_anchor_local_shrinkage_witness_fraction", 0.0)),
            "hard_anchor_breakout_misalignment_veto_fraction": float(cand_comp.get("hard_anchor_breakout_misalignment_veto_fraction", 0.0)),
            "hard_anchor_breakout_alignment_centroid_score": float(cand_comp.get("hard_anchor_breakout_alignment_centroid_score", 0.0)),
            "hard_anchor_breakout_alignment_gain": float(cand_comp.get("hard_anchor_breakout_alignment_gain", 0.0)),
            "hard_anchor_peak_rebound_focus_fraction": float(cand_comp.get("hard_anchor_peak_rebound_focus_fraction", 0.0)),
            "hard_anchor_peak_rebound_binding_fraction": float(cand_comp.get("hard_anchor_peak_rebound_binding_fraction", 0.0)),
            "hard_anchor_peak_rebound_veto_fraction": float(cand_comp.get("hard_anchor_peak_rebound_veto_fraction", 0.0)),
            "hard_anchor_peak_rebound_suppression_fraction": float(cand_comp.get("hard_anchor_peak_rebound_suppression_fraction", 0.0)),
            "hard_anchor_peak_rebound_residual_fraction": float(cand_comp.get("hard_anchor_peak_rebound_residual_fraction", 0.0)),
            "hard_anchor_peak_rebound_contract_gain": float(cand_comp.get("hard_anchor_peak_rebound_contract_gain", 0.0)),
            "hard_anchor_peak_residual_activation_fraction": float(cand_comp.get("hard_anchor_peak_residual_activation_fraction", 0.0)),
            "hard_anchor_peak_residual_binding_fraction": float(cand_comp.get("hard_anchor_peak_residual_binding_fraction", 0.0)),
            "hard_anchor_peak_residual_veto_fraction": float(cand_comp.get("hard_anchor_peak_residual_veto_fraction", 0.0)),
            "hard_anchor_peak_residual_contraction_fraction": float(cand_comp.get("hard_anchor_peak_residual_contraction_fraction", 0.0)),
            "hard_anchor_peak_residual_activation_gain": float(cand_comp.get("hard_anchor_peak_residual_activation_gain", 0.0)),
            "hard_anchor_peak_residual_remaining_fraction": float(cand_comp.get("hard_anchor_peak_residual_remaining_fraction", 0.0)),
            "hard_anchor_peak_residual_contraction_target_fraction": float(cand_comp.get("hard_anchor_peak_residual_contraction_target_fraction", 0.0)),
            "hard_anchor_peak_residual_contraction_binding_fraction": float(cand_comp.get("hard_anchor_peak_residual_contraction_binding_fraction", 0.0)),
            "hard_anchor_peak_residual_contraction_realization_fraction": float(cand_comp.get("hard_anchor_peak_residual_contraction_realization_fraction", 0.0)),
            "hard_anchor_peak_residual_contraction_overlap_fraction": float(cand_comp.get("hard_anchor_peak_residual_contraction_overlap_fraction", 0.0)),
            "hard_anchor_peak_residual_contraction_gain": float(cand_comp.get("hard_anchor_peak_residual_contraction_gain", 0.0)),
            "hard_anchor_peak_residual_contraction_alignment_score": float(cand_comp.get("hard_anchor_peak_residual_contraction_alignment_score", 0.0)),
            "hard_anchor_peak_residual_overlap_target_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_target_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_binding_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_binding_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_realization_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_realization_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain", 0.0)),
            "hard_anchor_peak_residual_overlap_alignment_score": float(cand_comp.get("hard_anchor_peak_residual_overlap_alignment_score", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_target_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_target_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_binding_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_binding_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_realization_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_realization_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_alignment_score": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_alignment_score", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_remaining_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_remaining_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_amplification_target_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_amplification_target_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_amplification_binding_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_amplification_binding_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_amplification_realization_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_amplification_realization_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_amplification_gain": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_amplification_gain", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_amplification_alignment_score": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_amplification_alignment_score", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_amplification_remaining_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_amplification_remaining_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_amplification_target_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_amplification_target_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_amplification_binding_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_amplification_binding_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_amplification_realization_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_amplification_realization_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_amplification_gain": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_amplification_gain", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_amplification_alignment_score": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_amplification_alignment_score", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_amplification_remaining_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_amplification_remaining_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_target_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_target_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_binding_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_binding_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_realization_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_realization_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_gain": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_gain", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_alignment_score": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_alignment_score", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_remaining_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_remaining_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_collapse_target_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_target_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_collapse_binding_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_binding_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_collapse_contract_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_contract_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_collapse_gain": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_gain", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_collapse_alignment_score": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_alignment_score", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_collapse_remaining_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_remaining_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_target_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_target_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_binding_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_binding_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_contract_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_contract_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_gain": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_gain", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_alignment_score": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_alignment_score", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_remaining_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_remaining_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_target_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_target_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_binding_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_binding_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_realization_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_realization_fraction", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_gain": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_gain", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_alignment_score": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_alignment_score", 0.0)),
            "hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_remaining_fraction": float(cand_comp.get("hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_remaining_fraction", 0.0)),
            "post_operator_component_merge_gain": float(cand_comp.get("post_operator_component_merge_gain", 0.0)),
            "post_operator_peak_gain": float(cand_comp.get("post_operator_peak_gain", 0.0)),
            "ownership_primary_affinity": float(cand_comp.get("ownership_primary_affinity", 0.0)),
            "ownership_secondary_affinity": float(cand_comp.get("ownership_secondary_affinity", 0.0)),
            "ownership_margin": float(cand_comp.get("ownership_margin", 0.0)),
            "ownership_actionability": float(cand_comp.get("ownership_actionability", 0.0)),
            "ownership_bridge_risk": float(cand_comp.get("ownership_bridge_risk", 0.0)),
            "ownership_hole_risk": float(cand_comp.get("ownership_hole_risk", 0.0)),
            "ownership_thin_structure_risk": float(cand_comp.get("ownership_thin_structure_risk", 0.0)),
            "effective_source_subset": cand_row.get("effective_source_subset", []),
            "applied_anchor_rules": cand_row.get("applied_anchor_rules", []),
            "prototype_parent_variant": cand_row.get("prototype_parent_variant", ""),
            "guard_report": cand_row.get("guard_report", {}),
            "files": {
                "target_baseline_candidate_renderdiff_fgmask_png": cand_row["files"].get("target_baseline_candidate_renderdiff_fgmask_png"),
                "fg_visible_components_colored_png": cand_row["files"].get("fg_visible_components_colored_png"),
                "fg_primary_vs_secondary_lobe_png": cand_row["files"].get("fg_primary_vs_secondary_lobe_png"),
                "fg_multilayer_overlap_heatmap_png": cand_row["files"].get("fg_multilayer_overlap_heatmap_png"),
                "fg_peak_map_png": cand_row["files"].get("fg_peak_map_png"),
                "fg_hole_bridge_panel_png": cand_row["files"].get("fg_hole_bridge_panel_png"),
                "secondary_lobe_score_map_png": cand_row["files"].get("secondary_lobe_score_map_png"),
                "before_after_primary_secondary_overlay_png": cand_row["files"].get("before_after_primary_secondary_overlay_png"),
                "before_after_duplicate_lobe_heatmap_png": cand_row["files"].get("before_after_duplicate_lobe_heatmap_png"),
                "before_after_multilayer_overlap_heatmap_png": cand_row["files"].get("before_after_multilayer_overlap_heatmap_png"),
                "before_after_render_peak_map_png": cand_row["files"].get("before_after_render_peak_map_png"),
                "before_after_hole_bridge_panel_png": cand_row["files"].get("before_after_hole_bridge_panel_png"),
                "before_after_target_baseline_candidate_suppressed_png": cand_row["files"].get("before_after_target_baseline_candidate_suppressed_png"),
                "suppression_alpha_delta_map_png": cand_row["files"].get("suppression_alpha_delta_map_png"),
                "suppression_rgb_blend_map_png": cand_row["files"].get("suppression_rgb_blend_map_png"),
                "coverage_loss_inside_fg_map_png": cand_row["files"].get("coverage_loss_inside_fg_map_png"),
                "masked_error_regression_map_png": cand_row["files"].get("masked_error_regression_map_png"),
                "baseline_primary_overlap_map_png": cand_row["files"].get("baseline_primary_overlap_map_png"),
                "thin_structure_protection_map_png": cand_row["files"].get("thin_structure_protection_map_png"),
                "component_rank_overlay_png": cand_row["files"].get("component_rank_overlay_png"),
                "component_touch_graph_overlay_png": cand_row["files"].get("component_touch_graph_overlay_png"),
                "post_suppression_component_colored_png": cand_row["files"].get("post_suppression_component_colored_png"),
                "before_after_component_entropy_panel_png": cand_row["files"].get("before_after_component_entropy_panel_png"),
                "component_rank_ledger_json": cand_row["files"].get("component_rank_ledger_json"),
                "candidate_support_inside_outside_png": cand_row["files"].get("support_outside_fg_png"),
                "candidate_support_overlay_on_fg_png": cand_row["files"].get("support_overlay_on_fg_png"),
                "candidate_fg_coverage_overlay_png": cand_row["files"].get("fg_coverage_overlay_png"),
                "render_operator_primary_contribution_map_png": cand_row["files"].get("render_operator_primary_contribution_map_png"),
                "render_operator_secondary_contribution_map_png": cand_row["files"].get("render_operator_secondary_contribution_map_png"),
                "render_operator_conflict_resolution_map_png": cand_row["files"].get("render_operator_conflict_resolution_map_png"),
                "render_operator_continuity_bypass_map_png": cand_row["files"].get("render_operator_continuity_bypass_map_png"),
                "render_operator_choice_map_png": cand_row["files"].get("render_operator_choice_map_png"),
                "render_operator_delta_map_png": cand_row["files"].get("render_operator_delta_map_png"),
                "before_after_render_operator_panel_png": cand_row["files"].get("before_after_render_operator_panel_png"),
                "render_operator_conflict_overlay_png": cand_row["files"].get("render_operator_conflict_overlay_png"),
                "render_operator_choice_overlay_png": cand_row["files"].get("render_operator_choice_overlay_png"),
                "before_after_component_merge_panel_png": cand_row["files"].get("before_after_component_merge_panel_png"),
                "before_after_peak_component_panel_png": cand_row["files"].get("before_after_peak_component_panel_png"),
                "render_operator_merge_intent_overlay_png": cand_row["files"].get("render_operator_merge_intent_overlay_png"),
                "render_operator_merge_adoption_overlay_png": cand_row["files"].get("render_operator_merge_adoption_overlay_png"),
                "render_operator_merge_veto_overlay_png": cand_row["files"].get("render_operator_merge_veto_overlay_png"),
                "render_operator_hard_anchor_failure_overlay_png": cand_row["files"].get("render_operator_hard_anchor_failure_overlay_png"),
                "render_operator_post_merge_peak_risk_overlay_png": cand_row["files"].get("render_operator_post_merge_peak_risk_overlay_png"),
                "render_operator_post_merge_peak_containment_overlay_png": cand_row["files"].get("render_operator_post_merge_peak_containment_overlay_png"),
                "render_operator_post_merge_realization_overlay_png": cand_row["files"].get("render_operator_post_merge_realization_overlay_png"),
                "before_after_post_merge_peak_panel_png": cand_row["files"].get("before_after_post_merge_peak_panel_png"),
                "before_after_post_merge_component_panel_png": cand_row["files"].get("before_after_post_merge_component_panel_png"),
                "render_operator_hard_anchor_peak_breakout_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_breakout_overlay_png"),
                "render_operator_post_merge_realization_eligibility_overlay_png": cand_row["files"].get("render_operator_post_merge_realization_eligibility_overlay_png"),
                "render_operator_post_merge_realization_binding_overlay_png": cand_row["files"].get("render_operator_post_merge_realization_binding_overlay_png"),
                "render_operator_post_merge_component_consolidation_overlay_png": cand_row["files"].get("render_operator_post_merge_component_consolidation_overlay_png"),
                "before_after_post_merge_realization_panel_png": cand_row["files"].get("before_after_post_merge_realization_panel_png"),
                "before_after_post_merge_component_consolidation_panel_png": cand_row["files"].get("before_after_post_merge_component_consolidation_panel_png"),
                "render_operator_hard_anchor_realization_breakout_overlay_png": cand_row["files"].get("render_operator_hard_anchor_realization_breakout_overlay_png"),
                "render_operator_post_merge_component_eligibility_overlay_png": cand_row["files"].get("render_operator_post_merge_component_eligibility_overlay_png"),
                "render_operator_post_merge_component_binding_overlay_png": cand_row["files"].get("render_operator_post_merge_component_binding_overlay_png"),
                "render_operator_post_merge_component_consolidation_contract_overlay_png": cand_row["files"].get("render_operator_post_merge_component_consolidation_contract_overlay_png"),
                "before_after_post_merge_component_binding_panel_png": cand_row["files"].get("before_after_post_merge_component_binding_panel_png"),
                "before_after_post_merge_component_consolidation_contract_panel_png": cand_row["files"].get("before_after_post_merge_component_consolidation_contract_panel_png"),
                "render_operator_hard_anchor_component_consolidation_breakout_overlay_png": cand_row["files"].get("render_operator_hard_anchor_component_consolidation_breakout_overlay_png"),
                "render_operator_post_merge_component_adjacency_overlay_png": cand_row["files"].get("render_operator_post_merge_component_adjacency_overlay_png"),
                "render_operator_post_merge_component_gap_closure_overlay_png": cand_row["files"].get("render_operator_post_merge_component_gap_closure_overlay_png"),
                "render_operator_post_merge_component_adjacency_contract_overlay_png": cand_row["files"].get("render_operator_post_merge_component_adjacency_contract_overlay_png"),
                "before_after_post_merge_component_adjacency_panel_png": cand_row["files"].get("before_after_post_merge_component_adjacency_panel_png"),
                "before_after_post_merge_component_gap_closure_panel_png": cand_row["files"].get("before_after_post_merge_component_gap_closure_panel_png"),
                "render_operator_hard_anchor_component_adjacency_breakout_overlay_png": cand_row["files"].get("render_operator_hard_anchor_component_adjacency_breakout_overlay_png"),
                "render_operator_post_merge_component_closure_realization_eligibility_overlay_png": cand_row["files"].get("render_operator_post_merge_component_closure_realization_eligibility_overlay_png"),
                "render_operator_post_merge_component_closure_realization_binding_overlay_png": cand_row["files"].get("render_operator_post_merge_component_closure_realization_binding_overlay_png"),
                "render_operator_post_merge_component_closure_realization_contract_overlay_png": cand_row["files"].get("render_operator_post_merge_component_closure_realization_contract_overlay_png"),
                "before_after_post_merge_component_closure_realization_panel_png": cand_row["files"].get("before_after_post_merge_component_closure_realization_panel_png"),
                "before_after_post_merge_component_closure_binding_panel_png": cand_row["files"].get("before_after_post_merge_component_closure_binding_panel_png"),
                "render_operator_hard_anchor_component_closure_realization_breakout_overlay_png": cand_row["files"].get("render_operator_hard_anchor_component_closure_realization_breakout_overlay_png"),
                "render_operator_hard_anchor_local_closure_focus_overlay_png": cand_row["files"].get("render_operator_hard_anchor_local_closure_focus_overlay_png"),
                "render_operator_hard_anchor_local_closure_binding_overlay_png": cand_row["files"].get("render_operator_hard_anchor_local_closure_binding_overlay_png"),
                "render_operator_hard_anchor_local_closure_veto_overlay_png": cand_row["files"].get("render_operator_hard_anchor_local_closure_veto_overlay_png"),
                "before_after_hard_anchor_local_closure_panel_png": cand_row["files"].get("before_after_hard_anchor_local_closure_panel_png"),
                "render_operator_hard_anchor_local_peak_rebound_guard_overlay_png": cand_row["files"].get("render_operator_hard_anchor_local_peak_rebound_guard_overlay_png"),
                "render_operator_hard_anchor_local_closure_breakout_overlay_png": cand_row["files"].get("render_operator_hard_anchor_local_closure_breakout_overlay_png"),
                "render_operator_hard_anchor_breakout_target_overlay_png": cand_row["files"].get("render_operator_hard_anchor_breakout_target_overlay_png"),
                "render_operator_hard_anchor_local_shrinkage_witness_overlay_png": cand_row["files"].get("render_operator_hard_anchor_local_shrinkage_witness_overlay_png"),
                "render_operator_hard_anchor_breakout_alignment_overlap_overlay_png": cand_row["files"].get("render_operator_hard_anchor_breakout_alignment_overlap_overlay_png"),
                "render_operator_hard_anchor_breakout_misalignment_veto_overlay_png": cand_row["files"].get("render_operator_hard_anchor_breakout_misalignment_veto_overlay_png"),
                "before_after_hard_anchor_breakout_alignment_panel_png": cand_row["files"].get("before_after_hard_anchor_breakout_alignment_panel_png"),
                "render_operator_hard_anchor_breakout_alignment_overlay_png": cand_row["files"].get("render_operator_hard_anchor_breakout_alignment_overlay_png"),
                "render_operator_hard_anchor_peak_rebound_focus_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_rebound_focus_overlay_png"),
                "render_operator_hard_anchor_peak_rebound_binding_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_rebound_binding_overlay_png"),
                "render_operator_hard_anchor_peak_rebound_veto_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_rebound_veto_overlay_png"),
                "before_after_hard_anchor_peak_rebound_suppression_panel_png": cand_row["files"].get("before_after_hard_anchor_peak_rebound_suppression_panel_png"),
                "render_operator_hard_anchor_peak_rebound_suppression_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_rebound_suppression_overlay_png"),
                "render_operator_hard_anchor_peak_rebound_residual_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_rebound_residual_overlay_png"),
                "render_operator_hard_anchor_peak_residual_activation_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_activation_overlay_png"),
                "render_operator_hard_anchor_peak_residual_binding_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_binding_overlay_png"),
                "render_operator_hard_anchor_peak_residual_veto_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_veto_overlay_png"),
                "before_after_hard_anchor_peak_residual_activation_panel_png": cand_row["files"].get("before_after_hard_anchor_peak_residual_activation_panel_png"),
                "render_operator_hard_anchor_peak_residual_contraction_witness_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_contraction_witness_overlay_png"),
                "render_operator_hard_anchor_peak_residual_guard_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_guard_overlay_png"),
                "render_operator_hard_anchor_peak_residual_contraction_target_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_contraction_target_overlay_png"),
                "render_operator_hard_anchor_peak_residual_contraction_binding_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_contraction_binding_overlay_png"),
                "render_operator_hard_anchor_peak_residual_contraction_realization_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_contraction_realization_overlay_png"),
                "before_after_hard_anchor_peak_residual_contraction_panel_png": cand_row["files"].get("before_after_hard_anchor_peak_residual_contraction_panel_png"),
                "render_operator_hard_anchor_peak_residual_contraction_veto_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_contraction_veto_overlay_png"),
                "render_operator_hard_anchor_peak_residual_contraction_guard_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_contraction_guard_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_target_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_target_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_binding_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_binding_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_realization_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_realization_overlay_png"),
                "before_after_hard_anchor_peak_residual_overlap_panel_png": cand_row["files"].get("before_after_hard_anchor_peak_residual_overlap_panel_png"),
                "render_operator_hard_anchor_peak_residual_overlap_veto_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_veto_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_alignment_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_alignment_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_target_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_target_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_binding_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_binding_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_realization_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_realization_overlay_png"),
                "before_after_hard_anchor_peak_residual_overlap_gain_panel_png": cand_row["files"].get("before_after_hard_anchor_peak_residual_overlap_gain_panel_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_veto_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_veto_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_alignment_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_alignment_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_amplification_target_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_amplification_target_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_amplification_binding_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_amplification_binding_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_amplification_realization_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_amplification_realization_overlay_png"),
                "before_after_hard_anchor_peak_residual_overlap_gain_amplification_panel_png": cand_row["files"].get("before_after_hard_anchor_peak_residual_overlap_gain_amplification_panel_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_amplification_veto_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_amplification_veto_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_amplification_alignment_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_amplification_alignment_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_amplification_target_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_amplification_target_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_amplification_binding_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_amplification_binding_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_overlay_png"),
                "before_after_hard_anchor_peak_residual_overlap_gain_density_amplification_panel_png": cand_row["files"].get("before_after_hard_anchor_peak_residual_overlap_gain_density_amplification_panel_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_amplification_veto_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_amplification_veto_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_amplification_alignment_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_amplification_alignment_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_target_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_target_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_binding_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_binding_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_realization_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_realization_overlay_png"),
                "before_after_hard_anchor_peak_residual_overlap_gain_density_landing_panel_png": cand_row["files"].get("before_after_hard_anchor_peak_residual_overlap_gain_density_landing_panel_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_veto_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_veto_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_alignment_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_alignment_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_target_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_target_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_binding_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_binding_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_contract_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_contract_overlay_png"),
                "before_after_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_panel_png": cand_row["files"].get("before_after_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_panel_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_veto_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_veto_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_alignment_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_alignment_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_target_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_target_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_binding_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_binding_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_contract_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_contract_overlay_png"),
                "before_after_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_panel_png": cand_row["files"].get("before_after_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_panel_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_veto_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_veto_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_alignment_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_alignment_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_target_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_target_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_binding_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_binding_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_realization_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_realization_overlay_png"),
                "before_after_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_panel_png": cand_row["files"].get("before_after_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_panel_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_veto_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_veto_overlay_png"),
                "render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_alignment_overlay_png": cand_row["files"].get("render_operator_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_alignment_overlay_png"),
                "render_operator_guard_regression_panel_png": cand_row["files"].get("render_operator_guard_regression_panel_png"),
            },
        }
        row["artifact_type"] = _artifact_type(row)
        row["local_rewrite_without_merge_rebound"] = _local_rewrite_without_merge_rebound(row)
        row["smoke_case_pass"] = _smoke_case_pass(row, family=family, variant=variant)
        row["control_case_pass"] = _control_case_pass(row)
        row["guard_attribution"] = _guard_attribution(row)
        rows.append(row)

    count = max(len(rows), 1)
    honest_primary_count = sum(1 for row in rows if row["smoke_case_pass"])
    control_primary_count = sum(1 for row in rows if row["control_case_pass"])
    aggregate = {
        "case_count": len(rows),
        "mean_delta_fg_visible_component_count": float(sum(row["delta_fg_visible_component_count"] for row in rows) / count),
        "mean_delta_largest_fg_visible_component_ratio": float(sum(row["delta_largest_fg_visible_component_ratio"] for row in rows) / count),
        "mean_delta_second_fg_visible_component_ratio": float(sum(row["delta_second_fg_visible_component_ratio"] for row in rows) / count),
        "mean_delta_fg_visible_component_entropy": float(sum(row["delta_fg_visible_component_entropy"] for row in rows) / count),
        "mean_delta_fg_secondary_mass_ratio": float(sum(row["delta_fg_secondary_mass_ratio"] for row in rows) / count),
        "mean_delta_fg_top2_visible_lobe_gap": float(sum(row["delta_fg_top2_visible_lobe_gap"] for row in rows) / count),
        "mean_delta_fg_multilayer_overlap_ratio": float(sum(row["delta_fg_multilayer_overlap_ratio"] for row in rows) / count),
        "mean_delta_fg_duplicate_lobe_ratio": float(sum(row["delta_fg_duplicate_lobe_ratio"] for row in rows) / count),
        "mean_delta_fg_nonblack_residual_ratio": float(sum(row["delta_fg_nonblack_residual_ratio"] for row in rows) / count),
        "mean_delta_fg_hole_ratio": float(sum(row["delta_fg_hole_ratio"] for row in rows) / count),
        "mean_delta_fg_bridge_break_ratio": float(sum(row["delta_fg_bridge_break_ratio"] for row in rows) / count),
        "mean_delta_fg_peak_count_after_render": float(sum(row["delta_fg_peak_count_after_render"] for row in rows) / count),
        "mean_delta_fg_peak_count": float(sum(row["delta_fg_peak_count"] for row in rows) / count),
        "mean_delta_fg_connected_components": float(sum(row["delta_fg_connected_components"] for row in rows) / count),
        "mean_delta_masked_l1": float(sum(row["delta_masked_l1"] for row in rows) / count),
        "mean_delta_masked_ssim": float(sum(row["delta_masked_ssim"] for row in rows) / count),
        "mean_delta_off_body_support_ratio": float(sum(row["delta_off_body_support_ratio"] for row in rows) / count),
        "mean_delta_bg_bottom_support_ratio": float(sum(row["delta_bg_bottom_support_ratio"] for row in rows) / count),
        "mean_delta_primary_consistency_score": float(sum(row["delta_primary_consistency_score"] for row in rows) / count),
        "mean_delta_secondary_conflict_score": float(sum(row["delta_secondary_conflict_score"] for row in rows) / count),
        "mean_delta_composition_penalty_score": float(sum(row["delta_composition_penalty_score"] for row in rows) / count),
        "mean_delta_resolved_to_primary_fraction": float(sum(row["delta_resolved_to_primary_fraction"] for row in rows) / count),
        "mean_delta_resolved_to_secondary_fraction": float(sum(row["delta_resolved_to_secondary_fraction"] for row in rows) / count),
        "mean_delta_resolved_to_abstain_fraction": float(sum(row["delta_resolved_to_abstain_fraction"] for row in rows) / count),
        "mean_delta_primary_owned_fraction": float(sum(row["delta_primary_owned_fraction"] for row in rows) / count),
        "mean_delta_secondary_owned_fraction": float(sum(row["delta_secondary_owned_fraction"] for row in rows) / count),
        "mean_delta_continuity_protected_fraction": float(sum(row["delta_continuity_protected_fraction"] for row in rows) / count),
        "mean_delta_conflict_resolved_fraction": float(sum(row["delta_conflict_resolved_fraction"] for row in rows) / count),
        "mean_delta_merge_intent_fraction": float(sum(row["delta_merge_intent_fraction"] for row in rows) / count),
        "mean_delta_merge_adoption_fraction": float(sum(row["delta_merge_adoption_fraction"] for row in rows) / count),
        "mean_delta_merge_veto_fraction": float(sum(row["delta_merge_veto_fraction"] for row in rows) / count),
        "mean_delta_merge_veto_continuity_fraction": float(sum(row["delta_merge_veto_continuity_fraction"] for row in rows) / count),
        "mean_delta_merge_veto_bridge_hole_fraction": float(sum(row["delta_merge_veto_bridge_hole_fraction"] for row in rows) / count),
        "mean_delta_merge_veto_overlap_insufficient_fraction": float(sum(row["delta_merge_veto_overlap_insufficient_fraction"] for row in rows) / count),
        "mean_delta_peak_stabilization_intent_fraction": float(sum(row["delta_peak_stabilization_intent_fraction"] for row in rows) / count),
        "mean_delta_peak_stabilization_adoption_fraction": float(sum(row["delta_peak_stabilization_adoption_fraction"] for row in rows) / count),
        "mean_delta_peak_stabilization_veto_fraction": float(sum(row["delta_peak_stabilization_veto_fraction"] for row in rows) / count),
        "mean_delta_post_merge_realization_fraction": float(sum(row["delta_post_merge_realization_fraction"] for row in rows) / count),
        "mean_delta_post_merge_realization_intent_fraction": float(sum(row["delta_post_merge_realization_intent_fraction"] for row in rows) / count),
        "mean_delta_post_merge_realization_binding_fraction": float(sum(row["delta_post_merge_realization_binding_fraction"] for row in rows) / count),
        "mean_delta_post_merge_realization_adoption_fraction": float(sum(row["delta_post_merge_realization_adoption_fraction"] for row in rows) / count),
        "mean_delta_post_merge_component_binding_fraction": float(sum(row["delta_post_merge_component_binding_fraction"] for row in rows) / count),
        "mean_delta_post_merge_component_consolidation_fraction": float(sum(row["delta_post_merge_component_consolidation_fraction"] for row in rows) / count),
        "mean_delta_post_merge_component_adjacency_fraction": float(sum(row["delta_post_merge_component_adjacency_fraction"] for row in rows) / count),
        "mean_delta_post_merge_component_gap_closure_fraction": float(sum(row["delta_post_merge_component_gap_closure_fraction"] for row in rows) / count),
        "mean_delta_post_merge_component_closure_realization_fraction": float(sum(row["delta_post_merge_component_closure_realization_fraction"] for row in rows) / count),
        "mean_delta_post_merge_component_closure_binding_fraction": float(sum(row["delta_post_merge_component_closure_binding_fraction"] for row in rows) / count),
        "mean_delta_post_merge_peak_containment_gain": float(sum(row["delta_post_merge_peak_containment_gain"] for row in rows) / count),
        "mean_delta_post_merge_component_consolidation_gain": float(sum(row["delta_post_merge_component_consolidation_gain"] for row in rows) / count),
        "mean_delta_post_merge_component_realization_gain": float(sum(row["delta_post_merge_component_realization_gain"] for row in rows) / count),
        "mean_delta_post_merge_component_adjacency_gain": float(sum(row["delta_post_merge_component_adjacency_gain"] for row in rows) / count),
        "mean_delta_post_merge_component_closure_realization_gain": float(sum(row["delta_post_merge_component_closure_realization_gain"] for row in rows) / count),
        "mean_delta_post_merge_component_adjacency_residual_split_fraction": float(sum(row["delta_post_merge_component_adjacency_residual_split_fraction"] for row in rows) / count),
        "mean_delta_post_merge_component_closure_residual_split_fraction": float(sum(row["delta_post_merge_component_closure_residual_split_fraction"] for row in rows) / count),
        "mean_delta_post_merge_component_residual_split_fraction": float(sum(row["delta_post_merge_component_residual_split_fraction"] for row in rows) / count),
        "mean_delta_post_merge_peak_residual_after_realization": float(sum(row["delta_post_merge_peak_residual_after_realization"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual": float(sum(row["delta_hard_anchor_peak_residual"] for row in rows) / count),
        "mean_delta_hard_anchor_realization_breakout_score": float(sum(row["delta_hard_anchor_realization_breakout_score"] for row in rows) / count),
        "mean_delta_hard_anchor_component_consolidation_breakout_score": float(sum(row["delta_hard_anchor_component_consolidation_breakout_score"] for row in rows) / count),
        "mean_delta_hard_anchor_component_adjacency_breakout_score": float(sum(row["delta_hard_anchor_component_adjacency_breakout_score"] for row in rows) / count),
        "mean_delta_hard_anchor_component_closure_realization_breakout_score": float(sum(row["delta_hard_anchor_component_closure_realization_breakout_score"] for row in rows) / count),
        "mean_delta_hard_anchor_local_closure_fraction": float(sum(row["delta_hard_anchor_local_closure_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_local_closure_gain": float(sum(row["delta_hard_anchor_local_closure_gain"] for row in rows) / count),
        "mean_delta_hard_anchor_local_closure_residual_split_fraction": float(sum(row["delta_hard_anchor_local_closure_residual_split_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_local_peak_rebound_suppression_gain": float(sum(row["delta_hard_anchor_local_peak_rebound_suppression_gain"] for row in rows) / count),
        "mean_delta_hard_anchor_local_closure_weight": float(sum(row["delta_hard_anchor_local_closure_weight"] for row in rows) / count),
        "mean_delta_hard_anchor_breakout_target_fraction": float(sum(row["delta_hard_anchor_breakout_target_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_breakout_alignment_fraction": float(sum(row["delta_hard_anchor_breakout_alignment_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_local_shrinkage_witness_fraction": float(sum(row["delta_hard_anchor_local_shrinkage_witness_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_breakout_misalignment_veto_fraction": float(sum(row["delta_hard_anchor_breakout_misalignment_veto_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_breakout_alignment_centroid_score": float(sum(row["delta_hard_anchor_breakout_alignment_centroid_score"] for row in rows) / count),
        "mean_delta_hard_anchor_breakout_alignment_gain": float(sum(row["delta_hard_anchor_breakout_alignment_gain"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_rebound_focus_fraction": float(sum(row["delta_hard_anchor_peak_rebound_focus_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_rebound_binding_fraction": float(sum(row["delta_hard_anchor_peak_rebound_binding_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_rebound_veto_fraction": float(sum(row["delta_hard_anchor_peak_rebound_veto_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_rebound_suppression_fraction": float(sum(row["delta_hard_anchor_peak_rebound_suppression_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_rebound_residual_fraction": float(sum(row["delta_hard_anchor_peak_rebound_residual_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_rebound_contract_gain": float(sum(row["delta_hard_anchor_peak_rebound_contract_gain"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_activation_fraction": float(sum(row["delta_hard_anchor_peak_residual_activation_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_binding_fraction": float(sum(row["delta_hard_anchor_peak_residual_binding_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_veto_fraction": float(sum(row["delta_hard_anchor_peak_residual_veto_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_contraction_fraction": float(sum(row["delta_hard_anchor_peak_residual_contraction_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_activation_gain": float(sum(row["delta_hard_anchor_peak_residual_activation_gain"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_remaining_fraction": float(sum(row["delta_hard_anchor_peak_residual_remaining_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_contraction_target_fraction": float(sum(row["delta_hard_anchor_peak_residual_contraction_target_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_contraction_binding_fraction": float(sum(row["delta_hard_anchor_peak_residual_contraction_binding_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_contraction_realization_fraction": float(sum(row["delta_hard_anchor_peak_residual_contraction_realization_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_contraction_overlap_fraction": float(sum(row["delta_hard_anchor_peak_residual_contraction_overlap_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_contraction_gain": float(sum(row["delta_hard_anchor_peak_residual_contraction_gain"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_contraction_alignment_score": float(sum(row["delta_hard_anchor_peak_residual_contraction_alignment_score"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_target_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_target_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_binding_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_binding_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_realization_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_realization_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_alignment_score": float(sum(row["delta_hard_anchor_peak_residual_overlap_alignment_score"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_target_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_target_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_binding_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_binding_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_realization_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_realization_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_alignment_score": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_alignment_score"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_remaining_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_remaining_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_amplification_target_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_amplification_target_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_amplification_binding_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_amplification_binding_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_amplification_realization_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_amplification_realization_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_amplification_gain": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_amplification_gain"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_amplification_alignment_score": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_amplification_alignment_score"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_amplification_remaining_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_amplification_remaining_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_amplification_target_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_amplification_target_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_amplification_binding_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_amplification_binding_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_amplification_gain": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_amplification_gain"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_amplification_alignment_score": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_amplification_alignment_score"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_amplification_remaining_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_amplification_remaining_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_target_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_target_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_binding_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_binding_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_realization_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_realization_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_gain": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_gain"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_alignment_score": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_alignment_score"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_remaining_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_remaining_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_target_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_target_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_binding_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_binding_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_contract_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_contract_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_gain": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_gain"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_alignment_score": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_alignment_score"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_remaining_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_remaining_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_target_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_target_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_binding_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_binding_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_contract_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_contract_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_gain": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_gain"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_alignment_score": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_alignment_score"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_remaining_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_remaining_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_target_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_target_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_binding_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_binding_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_realization_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_realization_fraction"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_gain": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_gain"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_alignment_score": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_alignment_score"] for row in rows) / count),
        "mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_remaining_fraction": float(sum(row["delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_remaining_fraction"] for row in rows) / count),
        "mean_delta_post_operator_component_merge_gain": float(sum(row["delta_post_operator_component_merge_gain"] for row in rows) / count),
        "mean_delta_post_operator_peak_gain": float(sum(row["delta_post_operator_peak_gain"] for row in rows) / count),
        "mean_delta_ownership_primary_affinity": float(sum(row["delta_ownership_primary_affinity"] for row in rows) / count),
        "mean_delta_ownership_secondary_affinity": float(sum(row["delta_ownership_secondary_affinity"] for row in rows) / count),
        "mean_delta_ownership_margin": float(sum(row["delta_ownership_margin"] for row in rows) / count),
        "mean_delta_ownership_actionability": float(sum(row["delta_ownership_actionability"] for row in rows) / count),
        "mean_delta_ownership_bridge_risk": float(sum(row["delta_ownership_bridge_risk"] for row in rows) / count),
        "mean_delta_ownership_hole_risk": float(sum(row["delta_ownership_hole_risk"] for row in rows) / count),
        "mean_delta_ownership_thin_structure_risk": float(sum(row["delta_ownership_thin_structure_risk"] for row in rows) / count),
        "mean_fg_visible_rgb_coverage_ratio": float(sum(row["fg_visible_rgb_coverage_ratio"] for row in rows) / count),
        "mean_fg_visible_mass_ratio": float(sum(row["fg_visible_mass_ratio"] for row in rows) / count),
        "worst_delta_fg_peak_count": float(max(row["delta_fg_peak_count"] for row in rows)) if rows else 0.0,
        "worst_delta_fg_peak_count_after_render": float(max(row["delta_fg_peak_count_after_render"] for row in rows)) if rows else 0.0,
        "worst_delta_off_body_support_ratio": float(max(row["delta_off_body_support_ratio"] for row in rows)) if rows else 0.0,
        "max_fg_duplicate_lobe_ratio": float(max(row["fg_duplicate_lobe_ratio"] for row in rows)) if rows else 0.0,
        "max_fg_multilayer_overlap_ratio": float(max(row["fg_multilayer_overlap_ratio"] for row in rows)) if rows else 0.0,
        "dominant_artifact_type": _dominant_artifact(rows),
        "honest_primary_count": int(honest_primary_count),
        "control_primary_count": int(control_primary_count),
    }
    anchor_component_breakout = _anchor_component_breakout(rows)
    hard_anchor_operator_core_guard = _hard_anchor_operator_core_guard(rows, family=family, variant=variant)
    all_guard_pass = all(bool(row.get("guard_report", {}).get("all_pass", True)) for row in rows)
    if _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_graph_realization_family(family=family, variant=variant):
        smoke_1x3_pass = bool(
            len(rows) == 3
            and honest_primary_count == len(rows)
            and bool(hard_anchor_operator_core_guard["pass"])
            and aggregate["mean_delta_masked_l1"] <= 0.0
            and aggregate["mean_delta_masked_ssim"] >= 0.0
            and aggregate["worst_delta_off_body_support_ratio"] <= 0.0
            and aggregate["mean_delta_fg_connected_components"] < 0.0
            and aggregate["mean_delta_fg_peak_count_after_render"] <= 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual"] <= 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_target_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_binding_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_realization_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_gain"] > MEANINGFUL_MEAN_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_CONNECTIVITY_GRAPH_FLOOR
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_graph_alignment_score"] > 0.0
            and all_guard_pass
        )
    elif _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_connectivity_contract_family(family=family, variant=variant):
        smoke_1x3_pass = bool(
            len(rows) == 3
            and honest_primary_count == len(rows)
            and bool(hard_anchor_operator_core_guard["pass"])
            and aggregate["mean_delta_masked_l1"] <= 0.0
            and aggregate["mean_delta_masked_ssim"] >= 0.0
            and aggregate["worst_delta_off_body_support_ratio"] <= 0.0
            and aggregate["mean_delta_fg_connected_components"] < 0.0
            and aggregate["mean_delta_fg_peak_count_after_render"] <= 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual"] <= 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_target_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_binding_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_contract_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_gain"] > MEANINGFUL_MEAN_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_CONNECTIVITY_FLOOR
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_connectivity_alignment_score"] > 0.0
            and all_guard_pass
        )
    elif _is_hard_anchor_peak_residual_overlap_gain_density_landing_to_component_collapse_contract_family(family=family, variant=variant):
        smoke_1x3_pass = bool(
            len(rows) == 3
            and honest_primary_count == len(rows)
            and bool(hard_anchor_operator_core_guard["pass"])
            and aggregate["mean_delta_masked_l1"] <= 0.0
            and aggregate["mean_delta_masked_ssim"] >= 0.0
            and aggregate["worst_delta_off_body_support_ratio"] <= 0.0
            and aggregate["mean_delta_fg_connected_components"] < 0.0
            and aggregate["mean_delta_fg_peak_count_after_render"] <= 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual"] <= 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_target_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_binding_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_contract_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_gain"] > MEANINGFUL_MEAN_OVERLAP_GAIN_DENSITY_LANDING_COLLAPSE_FLOOR
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_collapse_alignment_score"] > 0.0
            and all_guard_pass
        )
    elif _is_hard_anchor_peak_residual_overlap_gain_density_landing_realization_family(family=family, variant=variant):
        smoke_1x3_pass = bool(
            len(rows) == 3
            and honest_primary_count == len(rows)
            and bool(hard_anchor_operator_core_guard["pass"])
            and aggregate["mean_delta_masked_l1"] <= 0.0
            and aggregate["mean_delta_masked_ssim"] >= 0.0
            and aggregate["worst_delta_off_body_support_ratio"] <= 0.0
            and aggregate["mean_delta_fg_connected_components"] < 0.0
            and aggregate["mean_delta_fg_peak_count_after_render"] <= 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual"] <= 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_target_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_binding_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_realization_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_gain"] > MEANINGFUL_MEAN_OVERLAP_GAIN_DENSITY_LANDING_FLOOR
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_landing_alignment_score"] > 0.0
            and all_guard_pass
        )
    elif _is_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_family(family=family, variant=variant):
        smoke_1x3_pass = bool(
            len(rows) == 3
            and honest_primary_count == len(rows)
            and bool(hard_anchor_operator_core_guard["pass"])
            and aggregate["mean_delta_masked_l1"] <= 0.0
            and aggregate["mean_delta_masked_ssim"] >= 0.0
            and aggregate["worst_delta_off_body_support_ratio"] <= 0.0
            and aggregate["mean_delta_fg_connected_components"] < 0.0
            and aggregate["mean_delta_fg_peak_count_after_render"] <= 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual"] <= 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_amplification_target_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_amplification_binding_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_amplification_realization_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_amplification_gain"] > MEANINGFUL_MEAN_OVERLAP_GAIN_DENSITY_AMPLIFICATION_FLOOR
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_density_amplification_alignment_score"] > 0.0
            and all_guard_pass
        )
    elif _is_hard_anchor_peak_residual_overlap_gain_amplification_realization_family(family=family, variant=variant):
        smoke_1x3_pass = bool(
            len(rows) == 3
            and honest_primary_count == len(rows)
            and bool(hard_anchor_operator_core_guard["pass"])
            and aggregate["mean_delta_masked_l1"] <= 0.0
            and aggregate["mean_delta_masked_ssim"] >= 0.0
            and aggregate["worst_delta_off_body_support_ratio"] <= 0.0
            and aggregate["mean_delta_fg_connected_components"] < 0.0
            and aggregate["mean_delta_fg_peak_count_after_render"] <= 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual"] <= 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_amplification_target_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_amplification_binding_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_amplification_realization_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_amplification_gain"] > MEANINGFUL_MEAN_OVERLAP_GAIN_AMPLIFICATION_FLOOR
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_amplification_alignment_score"] > 0.0
            and all_guard_pass
        )
    elif _is_hard_anchor_peak_residual_overlap_gain_realization_family(family=family, variant=variant):
        smoke_1x3_pass = bool(
            len(rows) == 3
            and honest_primary_count == len(rows)
            and bool(hard_anchor_operator_core_guard["pass"])
            and aggregate["mean_delta_masked_l1"] <= 0.0
            and aggregate["mean_delta_masked_ssim"] >= 0.0
            and aggregate["worst_delta_off_body_support_ratio"] <= 0.0
            and aggregate["mean_delta_fg_connected_components"] < 0.0
            and aggregate["mean_delta_fg_peak_count_after_render"] <= 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual"] <= 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_target_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_binding_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_realization_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain"] > MEANINGFUL_MEAN_OVERLAP_GAIN_FLOOR
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain_alignment_score"] > 0.0
            and all_guard_pass
        )
    elif _is_hard_anchor_peak_residual_overlap_alignment_realization_family(family=family, variant=variant):
        smoke_1x3_pass = bool(
            len(rows) == 3
            and honest_primary_count == len(rows)
            and bool(hard_anchor_operator_core_guard["pass"])
            and aggregate["mean_delta_masked_l1"] <= 0.0
            and aggregate["mean_delta_masked_ssim"] >= 0.0
            and aggregate["worst_delta_off_body_support_ratio"] <= 0.0
            and aggregate["mean_delta_fg_connected_components"] < 0.0
            and aggregate["mean_delta_fg_peak_count_after_render"] <= 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual"] <= 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_target_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_binding_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_realization_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_gain"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_overlap_alignment_score"] > 0.0
            and all_guard_pass
        )
    elif _is_hard_anchor_peak_residual_contraction_realization_family(family=family, variant=variant):
        smoke_1x3_pass = bool(
            len(rows) == 3
            and honest_primary_count == len(rows)
            and bool(hard_anchor_operator_core_guard["pass"])
            and aggregate["mean_delta_masked_l1"] <= 0.0
            and aggregate["mean_delta_masked_ssim"] >= 0.0
            and aggregate["worst_delta_off_body_support_ratio"] <= 0.0
            and aggregate["mean_delta_fg_connected_components"] < 0.0
            and aggregate["mean_delta_fg_peak_count_after_render"] <= 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual"] <= 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_contraction_target_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_contraction_realization_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_contraction_overlap_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_contraction_gain"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_contraction_alignment_score"] > 0.0
            and all_guard_pass
        )
    elif _is_hard_anchor_peak_residual_activation_family(family=family, variant=variant):
        smoke_1x3_pass = bool(
            len(rows) == 3
            and honest_primary_count == len(rows)
            and bool(hard_anchor_operator_core_guard["pass"])
            and aggregate["mean_delta_masked_l1"] <= 0.0
            and aggregate["mean_delta_masked_ssim"] >= 0.0
            and aggregate["worst_delta_off_body_support_ratio"] <= 0.0
            and aggregate["mean_delta_fg_connected_components"] < 0.0
            and aggregate["mean_delta_fg_peak_count_after_render"] <= 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual"] <= 0.0
            and aggregate["mean_delta_hard_anchor_peak_rebound_contract_gain"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_activation_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_binding_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_contraction_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_residual_activation_gain"] > 0.0
            and all_guard_pass
        )
    elif _is_hard_anchor_peak_rebound_suppression_family(family=family, variant=variant):
        smoke_1x3_pass = bool(
            len(rows) == 3
            and honest_primary_count == len(rows)
            and bool(hard_anchor_operator_core_guard["pass"])
            and aggregate["mean_delta_masked_l1"] <= 0.0
            and aggregate["mean_delta_masked_ssim"] >= 0.0
            and aggregate["worst_delta_off_body_support_ratio"] <= 0.0
            and aggregate["mean_delta_fg_connected_components"] < 0.0
            and aggregate["mean_delta_fg_peak_count_after_render"] <= 0.0
            and aggregate["mean_delta_hard_anchor_breakout_alignment_gain"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_rebound_focus_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_rebound_binding_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_rebound_suppression_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_peak_rebound_contract_gain"] > 0.0
            and all_guard_pass
        )
    elif _is_hard_anchor_breakout_alignment_family(family=family, variant=variant):
        smoke_1x3_pass = bool(
            len(rows) == 3
            and honest_primary_count == len(rows)
            and bool(hard_anchor_operator_core_guard["pass"])
            and aggregate["mean_delta_masked_l1"] <= 0.0
            and aggregate["mean_delta_masked_ssim"] >= 0.0
            and aggregate["worst_delta_off_body_support_ratio"] <= 0.0
            and aggregate["mean_delta_fg_connected_components"] < 0.0
            and aggregate["mean_delta_fg_peak_count_after_render"] <= 0.0
            and aggregate["mean_delta_hard_anchor_breakout_alignment_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_breakout_alignment_gain"] > 0.0
            and aggregate["mean_delta_hard_anchor_local_shrinkage_witness_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_breakout_alignment_centroid_score"] > 0.0
            and all_guard_pass
        )
    elif _is_hard_anchor_closure_realization_family(family=family, variant=variant):
        smoke_1x3_pass = bool(
            len(rows) == 3
            and honest_primary_count == len(rows)
            and bool(hard_anchor_operator_core_guard["pass"])
            and aggregate["mean_delta_masked_l1"] <= 0.0
            and aggregate["mean_delta_masked_ssim"] >= 0.0
            and aggregate["worst_delta_off_body_support_ratio"] <= 0.0
            and aggregate["mean_delta_fg_connected_components"] < 0.0
            and aggregate["mean_delta_fg_peak_count_after_render"] <= 0.0
            and aggregate["mean_delta_hard_anchor_local_closure_fraction"] > 0.0
            and aggregate["mean_delta_hard_anchor_local_closure_gain"] > 0.0
            and aggregate["mean_delta_hard_anchor_local_closure_weight"] > 0.0
            and aggregate["mean_delta_hard_anchor_local_peak_rebound_suppression_gain"] > 0.0
            and all_guard_pass
        )
    elif _is_component_closure_realization_family(family=family, variant=variant):
        smoke_1x3_pass = bool(
            len(rows) == 3
            and honest_primary_count == len(rows)
            and aggregate["mean_delta_post_operator_component_merge_gain"] > 0.0
            and aggregate["mean_delta_post_merge_component_consolidation_gain"] > 0.0
            and aggregate["mean_delta_post_merge_component_realization_gain"] > 0.0
            and aggregate["mean_delta_post_merge_component_closure_realization_gain"] > 0.0
            and aggregate["mean_delta_post_merge_component_closure_realization_fraction"] > 0.0
            and aggregate["mean_delta_post_merge_component_closure_binding_fraction"] > 0.0
            and aggregate["mean_delta_fg_connected_components"] < 0.0
            and aggregate["mean_delta_fg_peak_count_after_render"] <= 0.0
            and aggregate["mean_delta_masked_l1"] <= 0.0
            and aggregate["mean_delta_masked_ssim"] >= 0.0
            and aggregate["worst_delta_off_body_support_ratio"] <= 0.0
            and bool(hard_anchor_operator_core_guard["pass"])
            and all_guard_pass
        )
    elif _is_component_adjacency_family(family=family, variant=variant):
        smoke_1x3_pass = bool(
            len(rows) == 3
            and honest_primary_count == len(rows)
            and aggregate["mean_delta_post_operator_component_merge_gain"] > 0.0
            and aggregate["mean_delta_post_merge_component_consolidation_gain"] > 0.0
            and aggregate["mean_delta_post_merge_component_realization_gain"] > 0.0
            and aggregate["mean_delta_post_merge_component_adjacency_gain"] > 0.0
            and aggregate["mean_delta_post_merge_component_closure_realization_gain"] > 0.0
            and aggregate["mean_delta_fg_connected_components"] < 0.0
            and aggregate["mean_delta_fg_peak_count_after_render"] <= 0.0
            and aggregate["mean_delta_masked_l1"] <= 0.0
            and aggregate["mean_delta_masked_ssim"] >= 0.0
            and aggregate["worst_delta_off_body_support_ratio"] <= 0.0
            and bool(hard_anchor_operator_core_guard["pass"])
            and all_guard_pass
        )
    elif _is_component_consolidation_family(family=family, variant=variant):
        smoke_1x3_pass = bool(
            len(rows) == 3
            and honest_primary_count == len(rows)
            and aggregate["mean_delta_post_operator_component_merge_gain"] > 0.0
            and aggregate["mean_delta_post_merge_component_consolidation_gain"] > 0.0
            and aggregate["mean_delta_post_merge_component_realization_gain"] > 0.0
            and aggregate["mean_delta_fg_connected_components"] < 0.0
            and aggregate["mean_delta_fg_peak_count_after_render"] <= 0.0
            and aggregate["mean_delta_masked_l1"] <= 0.0
            and aggregate["mean_delta_masked_ssim"] >= 0.0
            and aggregate["worst_delta_off_body_support_ratio"] <= 0.0
            and bool(hard_anchor_operator_core_guard["pass"])
            and all_guard_pass
        )
    elif _is_post_merge_realization_family(family=family, variant=variant):
        smoke_1x3_pass = bool(
            len(rows) == 3
            and honest_primary_count == len(rows)
            and aggregate["mean_delta_post_operator_component_merge_gain"] > 0.0
            and aggregate["mean_delta_post_merge_component_realization_gain"] > 0.0
            and aggregate["mean_delta_fg_connected_components"] < 0.0
            and aggregate["mean_delta_fg_peak_count_after_render"] <= 0.0
            and aggregate["mean_delta_masked_l1"] <= 0.0
            and aggregate["mean_delta_masked_ssim"] >= 0.0
            and aggregate["worst_delta_off_body_support_ratio"] <= 0.0
            and bool(hard_anchor_operator_core_guard["pass"])
            and all_guard_pass
        )
    elif _is_peak_stabilization_family(family=family, variant=variant):
        smoke_1x3_pass = bool(
            len(rows) == 3
            and honest_primary_count == len(rows)
            and aggregate["mean_delta_post_operator_component_merge_gain"] > 0.0
            and aggregate["mean_delta_post_operator_peak_gain"] > 0.0
            and aggregate["mean_delta_fg_peak_count_after_render"] < 0.0
            and aggregate["mean_delta_fg_connected_components"] < 0.0
            and aggregate["mean_delta_masked_l1"] <= 0.0
            and aggregate["mean_delta_masked_ssim"] >= 0.0
            and aggregate["worst_delta_off_body_support_ratio"] <= 0.0
            and bool(hard_anchor_operator_core_guard["pass"])
            and all_guard_pass
        )
    else:
        smoke_1x3_pass = bool(
            len(rows) == 3
            and honest_primary_count == len(rows)
            and aggregate["mean_delta_resolved_to_primary_fraction"] > 0.0
            and aggregate["mean_delta_post_operator_component_merge_gain"] > 0.0
            and aggregate["mean_delta_fg_peak_count_after_render"] < 0.0
            and aggregate["mean_delta_fg_connected_components"] < 0.0
            and aggregate["mean_delta_masked_l1"] <= 0.0
            and aggregate["mean_delta_masked_ssim"] >= 0.0
            and aggregate["worst_delta_off_body_support_ratio"] <= 0.0
            and bool(hard_anchor_operator_core_guard["pass"])
            and all_guard_pass
        )
    control_3x_pass = bool(
        smoke_1x3_pass
        and len(rows) == 3
        and control_primary_count == len(rows)
    )
    is_fragmentation_truly_reduced = bool(
        aggregate["mean_delta_fg_visible_component_count"] < 0.0
        and aggregate["mean_delta_fg_visible_component_entropy"] < 0.0
        and aggregate["mean_delta_fg_peak_count_after_render"] < 0.0
        and aggregate["mean_delta_fg_connected_components"] < 0.0
    )
    is_quality_regression_removed = bool(
        aggregate["mean_delta_masked_l1"] <= 0.0
        and aggregate["mean_delta_masked_ssim"] >= 0.0
    )
    is_guard_failure_eliminated = bool(all_guard_pass)
    correspondence_not_primary = (
        aggregate["mean_delta_off_body_support_ratio"] <= 0.0
        and aggregate["mean_fg_visible_rgb_coverage_ratio"] >= 0.55
        and (
            aggregate["dominant_artifact_type"] in {
                "multilayer_residual",
                "multi_component_fragmentation",
                "primary_secondary_lobe_competition",
                "peak_rebound",
            }
            or aggregate["max_fg_duplicate_lobe_ratio"] >= 0.15
            or aggregate["max_fg_multilayer_overlap_ratio"] >= 0.10
        )
    )
    failure_routing = _failure_routing(
        aggregate,
        rows=rows,
        is_guard_failure_eliminated=bool(all_guard_pass),
        hard_anchor_guard=hard_anchor_operator_core_guard,
        family=family,
        variant=variant,
    )
    return {
        "checked_at": datetime.now().astimezone().isoformat(),
        "family": family,
        "variant": variant,
        "baseline_variant": baseline_variant,
        "aggregate": aggregate,
        "anchor_component_breakout": anchor_component_breakout,
        "hard_anchor_operator_core_guard": hard_anchor_operator_core_guard,
        "smoke_1x3_pass": smoke_1x3_pass,
        "control_3x_pass": control_3x_pass,
        "is_fragmentation_truly_reduced": is_fragmentation_truly_reduced,
        "is_quality_regression_removed": is_quality_regression_removed,
        "is_guard_failure_eliminated": is_guard_failure_eliminated,
        "correspondence_not_primary": bool(correspondence_not_primary),
        "failure_routing": failure_routing,
        "summary_reason": (
            "Correspondence-side control improved masking and off-body leakage, but the remaining failure is dominated by inside-fg render artifact."
            if correspondence_not_primary
            else "The remaining failure still looks mixed; correspondence may still contribute materially."
        ),
        "per_case": rows,
    }


def _compare_with_reference(payload: dict, reference_payload: dict) -> dict:
    current = payload.get("aggregate", {})
    reference = reference_payload.get("aggregate", {})
    if not reference:
        return {
            "is_v2_better_than_v1_on_same_3_cases": False,
            "reference_variant": reference_payload.get("variant"),
            "comparison_ready": False,
        }
    better = bool(
        float(current.get("mean_delta_fg_connected_components", 0.0)) <= float(reference.get("mean_delta_fg_connected_components", 0.0))
        and float(current.get("mean_delta_fg_peak_count_after_render", 0.0)) <= float(reference.get("mean_delta_fg_peak_count_after_render", 0.0))
        and float(current.get("mean_delta_masked_l1", 0.0)) <= float(reference.get("mean_delta_masked_l1", 0.0))
        and float(current.get("mean_delta_masked_ssim", 0.0)) >= float(reference.get("mean_delta_masked_ssim", 0.0))
        and int(current.get("honest_primary_count", 0)) >= int(reference.get("honest_primary_count", 0))
        and int(current.get("control_primary_count", 0)) >= int(reference.get("control_primary_count", 0))
    )
    return {
        "is_v2_better_than_v1_on_same_3_cases": better,
        "reference_variant": reference_payload.get("variant"),
        "comparison_ready": True,
    }


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary_json).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    family = str(args.family or "")
    payload = build_summary(summary, args.variant, args.baseline_variant, family=family)
    completeness_report = None
    zip_manifest = None
    if args.key_panels_zip or args.artifact_completeness_json or args.zip_manifest_json:
        completeness_report, zip_manifest = _build_artifact_completeness(
            summary_path=summary_path,
            summary=summary,
            variant=args.variant,
            family=family,
            key_panels_zip_path=Path(args.key_panels_zip).resolve() if args.key_panels_zip else None,
        )
        payload["artifact_completeness"] = completeness_report
        payload["artifact_incomplete_fail"] = bool(completeness_report["artifact_incomplete_fail"])
        if payload["artifact_incomplete_fail"]:
            payload["smoke_1x3_pass"] = False
            payload["control_3x_pass"] = False
            payload["failure_routing"] = {
                "route": "artifact_incomplete_fail",
                "next_family": family,
                "next_candidate": args.variant,
                "reason": "Artifact completeness gate failed before smoke verdict.",
            }
            payload["summary_reason"] = "Artifact completeness gate failed before smoke verdict."
    if args.reference_summary_json:
        reference_payload = json.loads(Path(args.reference_summary_json).read_text(encoding="utf-8"))
        payload.update(_compare_with_reference(payload, reference_payload))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.artifact_completeness_json and completeness_report is not None:
        artifact_path = Path(args.artifact_completeness_json)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(completeness_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.zip_manifest_json and zip_manifest is not None:
        manifest_path = Path(args.zip_manifest_json)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(zip_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
