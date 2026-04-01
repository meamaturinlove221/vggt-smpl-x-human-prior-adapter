import json
import subprocess
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
RAWPOOL_STATUS_ROOT = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current"
WATCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_watch"
GUARD_ROOT = REPO_ROOT / "output" / "zju_source_policy_rawpool_overnight_watch"

CANDIDATE_VERDICT_PATH = OUTPUT_ROOT / "candidate_verdict.json"
RESEARCH_STATUS_PATH = OUTPUT_ROOT / "research_loop_status.json"
LATEST_WATCH_PATH = WATCH_ROOT / "latest_watch_snapshot.json"
FRONTIER_LEDGER_PATH = OUTPUT_ROOT / "frontier_ledger.json"
FAMILY_STOP_REASON_PATH = OUTPUT_ROOT / "family_stop_reason.json"
TASK_PLAN_PATH = RAWPOOL_STATUS_ROOT / "task_plan.json"
EARLY_STEP_TRACE_PATH = OUTPUT_ROOT / "early_step_objective_balance_trace.20260330.json"
PER_STREAM_AUDIT_PATH = OUTPUT_ROOT / "per_stream_camera_component_audit.20260330.json"
FAMILY_MATRIX_PATH = OUTPUT_ROOT / "family_outcome_alignment_matrix.20260330.json"
ROOT_CAUSE_PATH = OUTPUT_ROOT / "objective_balance_root_cause_decision.20260330.json"
LATEST_GUARD_PATH = GUARD_ROOT / "latest_guard_snapshot.json"

AXIS_CLOSED_JSON_PATH = OUTPUT_ROOT / "two_stage_axis_closed.20260330.json"
AXIS_CLOSED_MD_PATH = OUTPUT_ROOT / "two_stage_axis_closed.20260330.md"
LOCALIZATION_JSON_PATH = OUTPUT_ROOT / "early_fl_tax_localization.20260330.json"
LOCALIZATION_MD_PATH = OUTPUT_ROOT / "early_fl_tax_localization.20260330.md"
OBJECT_MATRIX_JSON_PATH = OUTPUT_ROOT / "fl_tax_object_alignment_matrix.20260330.json"
OBJECT_MATRIX_MD_PATH = OUTPUT_ROOT / "fl_tax_object_alignment_matrix.20260330.md"
FAILURE_JSON_PATH = OUTPUT_ROOT / "two_stage_failure_interpretation.20260330.json"
FAILURE_MD_PATH = OUTPUT_ROOT / "two_stage_failure_interpretation.20260330.md"
TWO_STAGE_SHORT_VS_LEAD_PATH = (
    OUTPUT_ROOT
    / "runs"
    / "20260330_130251_zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropwo_9012e0a99e"
    / "short_vs_lead"
    / "summary.json"
)


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


def repo_rel(path_like: str | Path) -> str:
    path = Path(path_like)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def metric_delta(summary: dict, metric: str) -> float | None:
    for row in summary.get("val", {}).get("rows", []):
        if row.get("metric") == metric:
            return row.get("delta")
    return None


def run_json_command(args: list[str]) -> list | dict | None:
    result = subprocess.run(
        args,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def list_modal_apps() -> list[dict]:
    payload = run_json_command(["modal", "app", "list", "--json"])
    return payload if isinstance(payload, list) else []


def list_repo_processes() -> list[dict]:
    command = r"""
$patterns = @(
  'run_zju_source_policy_research_candidate.py',
  'run_zju_vggt_geom_minimal_finetune.ps1',
  'compare_zju_finetune_runs.py',
  'run_zju_source_policy_research_loop.py',
  'run_zju_source_policy_research_watch.py'
)
$procs = Get-CimInstance Win32_Process |
  Where-Object {
    $_.CommandLine -and
    $_.CommandLine -like '*vggt-main*' -and
    (
      $matched = $false
      foreach ($pattern in $patterns) {
        if ($_.CommandLine -like ('*' + $pattern + '*')) {
          $matched = $true
        }
      }
      $matched
    )
  } |
  Select-Object ProcessId, Name, CommandLine
$procs | ConvertTo-Json -Depth 4
"""
    payload = run_json_command(["powershell", "-NoProfile", "-Command", command])
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    return []


def intervention_row(
    *,
    family: str,
    intervention_object: str,
    changes_default_focal: bool,
    changes_hardtail_focal: bool,
    changes_manifest_contract: bool,
    changes_depth_weight: bool,
    changes_camera_supervision: bool,
    delta_loss_fl: float | None,
    delta_loss_camera: float,
    delta_conf_depth: float,
    delta_reg_depth: float,
) -> dict:
    return {
        "family": family,
        "intervention_object": intervention_object,
        "changes_default_focal": changes_default_focal,
        "changes_hardtail_focal": changes_hardtail_focal,
        "changes_manifest_contract": changes_manifest_contract,
        "changes_depth_weight": changes_depth_weight,
        "changes_camera_supervision": changes_camera_supervision,
        "delta_loss_FL": delta_loss_fl,
        "delta_loss_camera": delta_loss_camera,
        "delta_conf_depth": delta_conf_depth,
        "delta_reg_depth": delta_reg_depth,
    }


def render_axis_closed_md(payload: dict) -> str:
    return "\n".join(
        [
            "# Two-Stage Axis Closed (2026-03-30)",
            "",
            f"- Two-stage objective decoupling is formally closed with verdict `{payload['verdict']}`.",
            f"- Same-family retry remains forbidden: `{payload['same_family_retry_forbidden']}`.",
            "- Any next move must come from a new manual diagnosis rather than another automatic ticket.",
            "",
        ]
    )


def render_localization_md(payload: dict) -> str:
    lines = [
        "# Early FL Tax Localization (2026-03-30)",
        "",
        f"- dominant_component: `{payload['dominant_component']}`",
        f"- appears_by: `{payload['appears_by']}`",
        f"- most_supported_scope: `{payload['most_supported_scope']}`",
        f"- evidence_strength: `{payload['evidence_strength']}`",
        f"- support_new_manual_problem: `{payload['support_new_manual_problem']}`",
        "",
        "## Why",
        "",
    ]
    lines.extend([f"- {item}" for item in payload["headline_evidence"]])
    lines.extend(
        [
            "",
            "## Representative Readout",
            "",
            f"- immediate_depth_win_family_count: `{payload['immediate_depth_win_family_count']}` / `{payload['families_traced_count']}`",
            f"- plateau_family_count: `{payload['plateau_family_count']}`",
            f"- two_stage_short_delta: `{payload['two_stage_short_delta']}`",
            "",
        ]
    )
    return "\n".join(lines)


def render_object_matrix_md(payload: dict) -> str:
    lines = [
        "# FL Tax Object Alignment Matrix (2026-03-30)",
        "",
        f"- checked_at: `{payload['checked_at']}`",
        f"- leading_object_hypothesis: `{payload['leading_object_hypothesis']}`",
        "",
        "| family | intervention_object | default_focal | hardtail_focal | manifest_contract | depth_weight | camera_supervision | ΔFL | Δcamera | Δconf | Δreg |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        def fmt(value):
            if value is None:
                return "n/a"
            if isinstance(value, bool):
                return "1" if value else "0"
            return f"{value:.4f}" if isinstance(value, float) else str(value)

        lines.append(
            "| "
            + " | ".join(
                [
                    row["family"],
                    row["intervention_object"],
                    fmt(row["changes_default_focal"]),
                    fmt(row["changes_hardtail_focal"]),
                    fmt(row["changes_manifest_contract"]),
                    fmt(row["changes_depth_weight"]),
                    fmt(row["changes_camera_supervision"]),
                    fmt(row["delta_loss_FL"]),
                    fmt(row["delta_loss_camera"]),
                    fmt(row["delta_conf_depth"]),
                    fmt(row["delta_reg_depth"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- object_level_takeaway: `{payload['object_level_takeaway']}`",
            "",
        ]
    )
    return "\n".join(lines)


def render_failure_md(payload: dict) -> str:
    lines = [
        "# Two-Stage Failure Interpretation (2026-03-30)",
        "",
        f"- conclusion: `{payload['conclusion']}`",
        f"- confidence: `{payload['confidence']}`",
        "",
        "## Why",
        "",
    ]
    lines.extend([f"- {item}" for item in payload["supporting_points"]])
    return "\n".join(lines) + "\n"


def main() -> int:
    checked_at = now_iso()
    candidate_verdict = load_json(CANDIDATE_VERDICT_PATH)
    research_status = load_json(RESEARCH_STATUS_PATH)
    latest_watch = load_json(LATEST_WATCH_PATH)
    frontier_ledger = load_json(FRONTIER_LEDGER_PATH)
    family_stop_reason = load_json(FAMILY_STOP_REASON_PATH)
    task_plan = load_json(TASK_PLAN_PATH)
    early_trace = load_json(EARLY_STEP_TRACE_PATH)
    per_stream = load_json(PER_STREAM_AUDIT_PATH)
    family_matrix = load_json(FAMILY_MATRIX_PATH)
    root_cause = load_json(ROOT_CAUSE_PATH)
    latest_guard = load_json(LATEST_GUARD_PATH)
    two_stage_short = load_json(TWO_STAGE_SHORT_VS_LEAD_PATH)

    modal_apps = list_modal_apps()
    repo_processes = list_repo_processes()
    active_modal_apps = [row for row in modal_apps if str(row.get("State", "")).lower() != "stopped"]

    watch_research_summary = (((latest_watch.get("research") or {}).get("summary")) or {})

    axis_closed = {
        "checked_at": checked_at,
        "artifact_kind": "two_stage_axis_closed",
        "family": "two_stage_objective_decoupling",
        "verdict": candidate_verdict.get("status", ""),
        "family_in_verdict": candidate_verdict.get("family", ""),
        "research_state": research_status.get("state", ""),
        "approved_problem_present": bool(research_status.get("approved_problem_present")),
        "watch_manual_action_required": bool(watch_research_summary.get("manual_action_required")),
        "watch_manual_action_kind": str(watch_research_summary.get("manual_action_kind", "")),
        "same_family_retry_forbidden": bool(research_status.get("same_family_retry_forbidden")),
        "preferred_first_family": str(research_status.get("preferred_first_family", "")),
        "no_auto_next_ticket": not bool(str(research_status.get("preferred_first_family", "")).strip()),
        "active_modal_app_count": len(active_modal_apps),
        "repo_process_count_snapshot": int(latest_guard.get("repo_process_count", 0) or 0),
        "local_research_process_count": len(repo_processes),
        "closure_contract_satisfied": (
            candidate_verdict.get("status") == "dead_same_day"
            and candidate_verdict.get("family") == "two_stage_objective_decoupling"
            and research_status.get("state") == "IDLE_GUARD"
            and not bool(research_status.get("approved_problem_present"))
            and bool(research_status.get("same_family_retry_forbidden"))
            and len(active_modal_apps) == 0
            and len(repo_processes) == 0
        ),
        "supporting_refs": {
            "candidate_verdict": repo_rel(CANDIDATE_VERDICT_PATH),
            "research_loop_status": repo_rel(RESEARCH_STATUS_PATH),
            "latest_watch_snapshot": repo_rel(LATEST_WATCH_PATH),
            "frontier_ledger": repo_rel(FRONTIER_LEDGER_PATH),
            "family_stop_reason": repo_rel(FAMILY_STOP_REASON_PATH),
            "task_plan": repo_rel(TASK_PLAN_PATH),
        },
    }
    write_json(AXIS_CLOSED_JSON_PATH, axis_closed)
    write_text(AXIS_CLOSED_MD_PATH, render_axis_closed_md(axis_closed))

    early_answers = early_trace.get("answers", {})
    aggregate_readout = early_trace.get("aggregate_readout", {})
    stream_answers = per_stream.get("answers", {})
    persistent_component = per_stream.get("persistent_camera_component", {})

    two_stage_short_delta = {
        "loss_FL": metric_delta(two_stage_short, "loss_FL"),
        "loss_camera": metric_delta(two_stage_short, "loss_camera"),
        "loss_T": metric_delta(two_stage_short, "loss_T"),
        "loss_conf_depth": metric_delta(two_stage_short, "loss_conf_depth"),
        "loss_reg_depth": metric_delta(two_stage_short, "loss_reg_depth"),
    }

    localization = {
        "checked_at": checked_at,
        "artifact_kind": "early_fl_tax_localization",
        "dominant_component": persistent_component.get("dominant_component", "loss_FL"),
        "appears_by": "smoke_val_step0"
        if any(item.get("camera_tax_present_at_smoke_val_step0") for item in early_trace.get("families_traced", []))
        else "short_val_step0",
        "most_supported_scope": "global",
        "evidence_strength": "strong",
        "support_new_manual_problem": True,
        "immediate_depth_win_family_count": aggregate_readout.get("families_with_immediate_short_val_camera_tax_and_depth_gain", 0),
        "families_traced_count": aggregate_readout.get("families_traced_count", 0),
        "plateau_family_count": family_matrix.get("pattern_counts", {}).get("DEPTH_WIN_SMALL_FL_TAX", 0),
        "scope_counterevidence": {
            "default_only_supported": bool(early_answers.get("default_stream_only_supported")),
            "all_streams_or_global_pattern_supported": bool(early_answers.get("all_streams_or_global_pattern_supported")),
            "hardtail_stream_with_strongest_depth_gain_cooccurrence": stream_answers.get("stream_with_strongest_depth_gain_cooccurrence", ""),
            "reserve_changes_plateau_materially": False,
        },
        "headline_evidence": [
            root_cause.get("rationale", ""),
            aggregate_readout.get("primary_finding", ""),
            aggregate_readout.get("default_vs_global_localization", ""),
            persistent_component.get("finding", ""),
        ],
        "two_stage_short_delta": two_stage_short_delta,
        "supporting_refs": {
            "early_step_trace": repo_rel(EARLY_STEP_TRACE_PATH),
            "per_stream_camera_component_audit": repo_rel(PER_STREAM_AUDIT_PATH),
            "family_outcome_alignment_matrix": repo_rel(FAMILY_MATRIX_PATH),
            "objective_balance_root_cause_decision": repo_rel(ROOT_CAUSE_PATH),
            "two_stage_short_vs_lead": repo_rel(TWO_STAGE_SHORT_VS_LEAD_PATH),
        },
    }
    write_json(LOCALIZATION_JSON_PATH, localization)
    write_text(LOCALIZATION_MD_PATH, render_localization_md(localization))

    trace_by_family = {item["family"]: item for item in early_trace.get("families_traced", [])}
    matrix_rows = {item["family"]: item for item in family_matrix.get("rows", [])}
    object_rows = [
        intervention_row(
            family="hardtail_bucket_granularity_refinement",
            intervention_object="tail_manifest_contract",
            changes_default_focal=False,
            changes_hardtail_focal=False,
            changes_manifest_contract=True,
            changes_depth_weight=False,
            changes_camera_supervision=False,
            delta_loss_fl=(trace_by_family.get("hardtail_bucket_granularity_refinement", {}).get("short_val_step4_vs_stable_short", {}) or {}).get("FL"),
            delta_loss_camera=matrix_rows["hardtail_bucket_granularity_refinement"]["loss_camera"],
            delta_conf_depth=matrix_rows["hardtail_bucket_granularity_refinement"]["loss_conf_depth"],
            delta_reg_depth=matrix_rows["hardtail_bucket_granularity_refinement"]["loss_reg_depth"],
        ),
        intervention_row(
            family="tail_anchor_stabilization",
            intervention_object="tail_manifest_contract",
            changes_default_focal=False,
            changes_hardtail_focal=False,
            changes_manifest_contract=True,
            changes_depth_weight=False,
            changes_camera_supervision=True,
            delta_loss_fl=(trace_by_family.get("tail_anchor_stabilization", {}).get("short_val_step4_vs_stable_short", {}) or {}).get("FL"),
            delta_loss_camera=matrix_rows["tail_anchor_stabilization"]["loss_camera"],
            delta_conf_depth=matrix_rows["tail_anchor_stabilization"]["loss_conf_depth"],
            delta_reg_depth=matrix_rows["tail_anchor_stabilization"]["loss_reg_depth"],
        ),
        intervention_row(
            family="tail_anchor_reserve_hybridization",
            intervention_object="tail_manifest_contract",
            changes_default_focal=False,
            changes_hardtail_focal=False,
            changes_manifest_contract=True,
            changes_depth_weight=False,
            changes_camera_supervision=False,
            delta_loss_fl=(trace_by_family.get("tail_anchor_reserve_hybridization", {}).get("short_val_step4_vs_stable_short", {}) or {}).get("FL"),
            delta_loss_camera=matrix_rows["tail_anchor_reserve_hybridization"]["loss_camera"],
            delta_conf_depth=matrix_rows["tail_anchor_reserve_hybridization"]["loss_conf_depth"],
            delta_reg_depth=matrix_rows["tail_anchor_reserve_hybridization"]["loss_reg_depth"],
        ),
        intervention_row(
            family="tail_stream_selective_focal_reinforcement",
            intervention_object="hardtail_focal_scale",
            changes_default_focal=False,
            changes_hardtail_focal=True,
            changes_manifest_contract=False,
            changes_depth_weight=False,
            changes_camera_supervision=False,
            delta_loss_fl=(trace_by_family.get("tail_stream_selective_focal_reinforcement", {}).get("short_val_step4_vs_stable_short", {}) or {}).get("FL"),
            delta_loss_camera=matrix_rows["tail_stream_selective_focal_reinforcement"]["loss_camera"],
            delta_conf_depth=matrix_rows["tail_stream_selective_focal_reinforcement"]["loss_conf_depth"],
            delta_reg_depth=matrix_rows["tail_stream_selective_focal_reinforcement"]["loss_reg_depth"],
        ),
        intervention_row(
            family="default_stream_intrinsics_counterbalance",
            intervention_object="default_focal_scale",
            changes_default_focal=True,
            changes_hardtail_focal=False,
            changes_manifest_contract=False,
            changes_depth_weight=False,
            changes_camera_supervision=False,
            delta_loss_fl=(trace_by_family.get("default_stream_intrinsics_counterbalance", {}).get("short_val_step4_vs_stable_short", {}) or {}).get("FL"),
            delta_loss_camera=matrix_rows["default_stream_intrinsics_counterbalance"]["loss_camera"],
            delta_conf_depth=matrix_rows["default_stream_intrinsics_counterbalance"]["loss_conf_depth"],
            delta_reg_depth=matrix_rows["default_stream_intrinsics_counterbalance"]["loss_reg_depth"],
        ),
        intervention_row(
            family="tail_intrinsics_branch_decoupling",
            intervention_object="camera_supervision_branch_off",
            changes_default_focal=False,
            changes_hardtail_focal=False,
            changes_manifest_contract=False,
            changes_depth_weight=False,
            changes_camera_supervision=True,
            delta_loss_fl=None,
            delta_loss_camera=matrix_rows["tail_intrinsics_branch_decoupling"]["loss_camera"],
            delta_conf_depth=matrix_rows["tail_intrinsics_branch_decoupling"]["loss_conf_depth"],
            delta_reg_depth=matrix_rows["tail_intrinsics_branch_decoupling"]["loss_reg_depth"],
        ),
        intervention_row(
            family="tail_pose_branch_decoupling",
            intervention_object="camera_supervision_branch_off",
            changes_default_focal=False,
            changes_hardtail_focal=False,
            changes_manifest_contract=False,
            changes_depth_weight=False,
            changes_camera_supervision=True,
            delta_loss_fl=None,
            delta_loss_camera=matrix_rows["tail_pose_branch_decoupling"]["loss_camera"],
            delta_conf_depth=matrix_rows["tail_pose_branch_decoupling"]["loss_conf_depth"],
            delta_reg_depth=matrix_rows["tail_pose_branch_decoupling"]["loss_reg_depth"],
        ),
        intervention_row(
            family="tail_dual_supervision_rebalancing",
            intervention_object="camera_supervision_count",
            changes_default_focal=False,
            changes_hardtail_focal=False,
            changes_manifest_contract=False,
            changes_depth_weight=False,
            changes_camera_supervision=True,
            delta_loss_fl=(trace_by_family.get("tail_dual_supervision_rebalancing", {}).get("short_val_step4_vs_stable_short", {}) or {}).get("FL"),
            delta_loss_camera=matrix_rows["tail_dual_supervision_rebalancing"]["loss_camera"],
            delta_conf_depth=matrix_rows["tail_dual_supervision_rebalancing"]["loss_conf_depth"],
            delta_reg_depth=matrix_rows["tail_dual_supervision_rebalancing"]["loss_reg_depth"],
        ),
        intervention_row(
            family="two_stage_objective_decoupling",
            intervention_object="camera_depth_objective_schedule",
            changes_default_focal=True,
            changes_hardtail_focal=False,
            changes_manifest_contract=False,
            changes_depth_weight=True,
            changes_camera_supervision=False,
            delta_loss_fl=two_stage_short_delta["loss_FL"],
            delta_loss_camera=two_stage_short_delta["loss_camera"],
            delta_conf_depth=two_stage_short_delta["loss_conf_depth"],
            delta_reg_depth=two_stage_short_delta["loss_reg_depth"],
        ),
    ]
    object_matrix = {
        "checked_at": checked_at,
        "artifact_kind": "fl_tax_object_alignment_matrix",
        "rows": object_rows,
        "leading_object_hypothesis": "camera_depth_objective_conflict",
        "object_level_takeaway": (
            "Neither default-only focal scaling nor hardtail-only focal scaling nor reserve-tail contract changes remove the FL tax, while the two-stage scalar schedule also lands on the same short-gate pattern. That points to an objective-layer coupling rather than a single stream-local focal underweight."
        ),
        "supporting_refs": {
            "family_outcome_alignment_matrix": repo_rel(FAMILY_MATRIX_PATH),
            "per_stream_camera_component_audit": repo_rel(PER_STREAM_AUDIT_PATH),
            "early_step_objective_balance_trace": repo_rel(EARLY_STEP_TRACE_PATH),
            "two_stage_short_vs_lead": repo_rel(TWO_STAGE_SHORT_VS_LEAD_PATH),
        },
    }
    write_json(OBJECT_MATRIX_JSON_PATH, object_matrix)
    write_text(OBJECT_MATRIX_MD_PATH, render_object_matrix_md(object_matrix))

    failure = {
        "checked_at": checked_at,
        "artifact_kind": "two_stage_failure_interpretation",
        "conclusion": "SCALAR_SCHEDULE_INSUFFICIENT",
        "confidence": "strong",
        "supporting_points": [
            "The two-stage ticket preserved the same short-gate residual camera/T tax (+0.0006 / +0.0001) while depth terms improved by the same plateau-sized amounts (-0.0522 / -0.0139).",
            "Default-only focal counterbalance, hardtail-only focal reinforcement, reserve addition, and blanket tail focal reinforcement all left the short-gate plateau effectively unchanged.",
            "Early-step traces show the FL-dominant camera tax already present by the first validation snapshots, so a late scalar schedule boundary does not touch the origin of the conflict.",
            "The objective-balance root-cause decision already selected GLOBAL_OBJECTIVE_CONFLICT, and the two-stage result did not falsify that diagnosis.",
        ],
        "why_not_schedule_too_weak": (
            "The tested schedule changed both global depth weight and default-stream focal scale late in training, yet the short-gate result still matched the established FL-tax plateau. Given the broader family matrix, the failure is better explained by scalar schedule insufficiency than by merely needing a slightly stronger scalar."
        ),
        "supporting_refs": {
            "two_stage_axis_closed": repo_rel(AXIS_CLOSED_JSON_PATH),
            "early_fl_tax_localization": repo_rel(LOCALIZATION_JSON_PATH),
            "fl_tax_object_alignment_matrix": repo_rel(OBJECT_MATRIX_JSON_PATH),
            "objective_balance_root_cause_decision": repo_rel(ROOT_CAUSE_PATH),
        },
    }
    write_json(FAILURE_JSON_PATH, failure)
    write_text(FAILURE_MD_PATH, render_failure_md(failure))

    print(
        json.dumps(
            {
                "axis_closed": repo_rel(AXIS_CLOSED_JSON_PATH),
                "early_fl_tax_localization": repo_rel(LOCALIZATION_JSON_PATH),
                "fl_tax_object_alignment_matrix": repo_rel(OBJECT_MATRIX_JSON_PATH),
                "two_stage_failure_interpretation": repo_rel(FAILURE_JSON_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
