import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
RAWPOOL_STATUS_ROOT = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current"
LOCAL_MANIFEST_PATH = REPO_ROOT / "scripts" / "manifests" / "zju_source_policy_rawpool_local_nightly_v1.json"
TRAINING_QUESTION_MANIFEST_PATH = REPO_ROOT / "scripts" / "manifests" / "zju_next_training_question_v1.json"
TASK_PLAN_JSON_PATH = RAWPOOL_STATUS_ROOT / "task_plan.json"
TASK_PLAN_MD_PATH = RAWPOOL_STATUS_ROOT / "task_plan.md"
SUMMARY_MD_PATH = RAWPOOL_STATUS_ROOT / "summary.md"
WATCH_SCRIPT_PATH = REPO_ROOT / "scripts" / "run_zju_source_policy_research_watch.py"
RESEARCH_LOOP_SCRIPT_PATH = REPO_ROOT / "scripts" / "run_zju_source_policy_research_loop.py"
GUARD_SNAPSHOT_PATH = REPO_ROOT / "output" / "zju_source_policy_rawpool_overnight_watch" / "latest_guard_snapshot.json"
CURRENT_LEAD_PROMOTION_DECISION_PATH = OUTPUT_ROOT / "promotion_decision.20260328.json"
RESIDUAL_PROMOTION_PACKET_PATH = OUTPUT_ROOT / "promotion_review_packet.residual_case_coverage_rebalancing.20260329.md"
RESIDUAL_PROMOTION_PACKET_JSON_PATH = OUTPUT_ROOT / "promotion_review_packet.residual_case_coverage_rebalancing.20260329.json"
RESIDUAL_RUN_STATUS_PATH = (
    OUTPUT_ROOT
    / "runs"
    / "20260329_173643_zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_hardcasebucketmix4to1_minimal"
    / "status.json"
)
RESIDUAL_VS_CURRENT_SHORT_PATH = OUTPUT_ROOT / "tmp" / "residual_vs_hybrid_short" / "summary.json"
RESIDUAL_VS_CURRENT_LONG_PATH = OUTPUT_ROOT / "tmp" / "residual_vs_hybrid_long" / "summary.json"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def today_tag() -> str:
    return datetime.now().strftime("%Y%m%d")


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
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def safe_repo_rel(path_like: str | Path) -> str:
    text = str(path_like or "").strip()
    if not text:
        return ""
    path = Path(text)
    if path.is_absolute():
        try:
            return repo_rel(path)
        except ValueError:
            return text.replace("\\", "/")
    return text.replace("\\", "/")


def metric_rows(summary: dict) -> dict:
    return {str(row["metric"]): row for row in summary.get("val", {}).get("rows", [])}


def four_metrics(summary: dict, side: str) -> dict:
    rows = metric_rows(summary)
    key = "baseline" if side == "baseline" else "candidate"
    return {
        "camera": rows["loss_camera"][key],
        "T": rows["loss_T"][key],
        "conf_depth": rows["loss_conf_depth"][key],
        "reg_depth": rows["loss_reg_depth"][key],
    }


def four_deltas(summary: dict) -> dict:
    rows = metric_rows(summary)
    return {
        "delta_camera": rows["loss_camera"]["delta"],
        "delta_T": rows["loss_T"]["delta"],
        "delta_conf_depth": rows["loss_conf_depth"]["delta"],
        "delta_reg_depth": rows["loss_reg_depth"]["delta"],
    }


def gate_pass(summary: dict) -> bool:
    rows = metric_rows(summary)
    return (
        rows["loss_camera"]["delta"] <= 0
        and rows["loss_T"]["delta"] <= 0
        and rows["loss_conf_depth"]["delta"] <= 0
        and rows["loss_reg_depth"]["delta"] <= 0
    )


def upsert_by_key(items: list[dict], payload: dict, key: str) -> list[dict]:
    result: list[dict] = []
    inserted = False
    for item in items:
        if item.get(key) == payload.get(key):
            result.append(payload)
            inserted = True
        else:
            result.append(item)
    if not inserted:
        result.append(payload)
    return result


def run_checked(args: list[str]) -> None:
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
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(args)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def render_task_plan_md(now: str, guard_checked_at: str, decision: str, current_lead_family: str, current_lead_shape: str, current_lead_config: str) -> str:
    lead_line = (
        "- the existing hybrid-ring local lead stays synced after rejecting the residual ticket"
        if decision == "REJECT"
        else "- the residual-case-coverage local lead is synced"
    )
    return "\n".join(
        [
            f"# ZJU Source-Policy Rawpool Task Plan ({now[:10]})",
            "",
            f"- checked_at: `{now}`",
            f"- latest_guard_checked_at: `{guard_checked_at}`",
            "- task_mode_status: `active`",
            "- current_mode: `steady_hold`",
            "- research_loop_mode: `IDLE_GUARD`",
            "- task_mode_focus: `promoted_local_lead_synced_cloud_off`",
            "",
            "## Current State",
            "",
            "- Guard snapshot remains clean.",
            "- Research loop is clean: no active approval and empty allowlist.",
            "- Planning is on manual-decision sync complete:",
            "- no current priority family",
            "- no auto-next ticket",
            lead_line,
            "- cloud remains off",
            "",
            "## Current Local Lead",
            "",
            f"- family: `{current_lead_family}`",
            f"- first_candidate_shape: `{current_lead_shape}`",
            f"- manual decision: `{decision}`",
            f"- config: `{current_lead_config}`",
            "",
            "## Fastest Next Path",
            "",
            "- Keep guard and research idle/clean.",
            f"- Treat {current_lead_family} / {current_lead_shape} as the current local lead.",
            "- Do not auto-open another ticket.",
            "- If later needed, define a genuinely new manual problem instead of rerunning the same family.",
            "",
        ]
    )


def render_summary_md(now: str, guard: dict, decision: str, current_lead_config: str) -> str:
    headline = (
        "- The residual hardcase-mix ticket was executed but not promoted after a corrected recheck against the true current hybrid-ring lead."
        if decision == "REJECT"
        else "- The residual hardcase-mix ticket was manually promoted after a corrected recheck against the true current lead."
    )
    lead_line = (
        "- `source_policy_hybrid_ring_regularization / stablelead_nearest_plus_uniform_tail` remains the current local lead."
        if decision == "REJECT"
        else "- `residual_case_coverage_rebalancing / promotedlead_hardcase_bucket_mix` is now the current local lead."
    )
    return "\n".join(
        [
            f"# ZJU Source-Policy Rawpool Status ({now[:10]})",
            "",
            f"- checked_at: `{now}`",
            "- current_status: `steady_hold`",
            "- research_loop_status: `IDLE_GUARD`",
            f"- current_lead_config: `{current_lead_config}`",
            f"- consistency_ok: `{bool(guard.get('consistency_ok', True))}`",
            f"- cloud_gate: `{bool(guard.get('state_cloud_gate', False))}`",
            f"- launch_cloud_now: `{bool(guard.get('state_launch_cloud_now', False))}`",
            f"- active_modal_app_count: `{int(guard.get('active_modal_app_count', 0) or 0)}`",
            f"- repo_process_count: `{int(guard.get('repo_process_count', 0) or 0)}`",
            "",
            "## Current Conclusion",
            "",
            headline,
            lead_line,
            "- There is no active `approved_problem.json`.",
            "- Research is back in `IDLE_GUARD`.",
            "- Planning is clean: no current priority family, no auto-next ticket, cloud stays off.",
            "",
            "## Closed Batch",
            "",
            "- `interpolated_eligibility_shaping`",
            "- `partial_joint_depth_routing`",
            "- `conf_reg_disagreement_routing`",
            "- `unproject_consistency_routing`",
            "- `unproject_aux_confgate`",
            "- `residual_case_coverage_rebalancing_same_family_retry`",
            "",
            "Do not reopen cousins from those closed lines without a fresh manual problem.",
            "",
        ]
    )


def main() -> int:
    now = now_iso()
    tag = today_tag()
    guard = load_json(GUARD_SNAPSHOT_PATH)
    local_manifest = load_json(LOCAL_MANIFEST_PATH)
    training_question = load_json(TRAINING_QUESTION_MANIFEST_PATH)
    task_plan = load_json(TASK_PLAN_JSON_PATH)
    current_lead_decision = load_json(CURRENT_LEAD_PROMOTION_DECISION_PATH)
    residual_packet = load_json(RESIDUAL_PROMOTION_PACKET_JSON_PATH)
    residual_run_status = load_json(RESIDUAL_RUN_STATUS_PATH)
    short_recheck = load_json(RESIDUAL_VS_CURRENT_SHORT_PATH)
    long_recheck = load_json(RESIDUAL_VS_CURRENT_LONG_PATH)

    short_pass = gate_pass(short_recheck)
    long_pass = gate_pass(long_recheck)
    decision = "PROMOTE" if short_pass and long_pass else "REJECT"
    decision_status = "promoted_local_lead_synced" if decision == "PROMOTE" else "executed_but_not_promoted"

    prior_current_lead_config = str((local_manifest.get("current_lead", {}) or {}).get("config", "")).strip()
    residual_config = safe_repo_rel(residual_packet.get("config", ""))
    residual_run_dir_rel = safe_repo_rel(residual_packet.get("run_dir", ""))
    residual_short_log_rel = repo_rel(residual_run_status["artifacts"]["short_log"])
    residual_long_log_rel = repo_rel(residual_run_status["artifacts"]["long_candidate_log"])
    residual_short_summary_rel = repo_rel(RESIDUAL_VS_CURRENT_SHORT_PATH)
    residual_long_summary_rel = repo_rel(RESIDUAL_VS_CURRENT_LONG_PATH)
    residual_run_status_rel = repo_rel(RESIDUAL_RUN_STATUS_PATH)
    residual_packet_md_rel = repo_rel(RESIDUAL_PROMOTION_PACKET_PATH)
    current_lead_decision_md_rel = repo_rel(CURRENT_LEAD_PROMOTION_DECISION_PATH.with_suffix(".md"))

    decision_json_path = OUTPUT_ROOT / f"promotion_decision.{tag}.json"
    decision_md_path = OUTPUT_ROOT / f"promotion_decision.{tag}.md"

    short_baseline = four_metrics(short_recheck, "baseline")
    short_candidate = four_metrics(short_recheck, "candidate")
    long_baseline = four_metrics(long_recheck, "baseline")
    long_candidate = four_metrics(long_recheck, "candidate")
    short_delta = four_deltas(short_recheck)
    long_delta = four_deltas(long_recheck)

    next_requirement = (
        "Manual rejection sync complete. Keep research in IDLE_GUARD, keep cloud off, and do not auto-select a new family or auto-launch cloud. Wait for a fresh manual problem only if later needed."
        if decision == "REJECT"
        else "Manual promoted-lead sync complete. Keep research in IDLE_GUARD, keep cloud off, and do not auto-select a new family or auto-launch cloud. Wait for a fresh manual problem only if later needed."
    )
    blocking_reason = (
        "The residual hardcase mix did not beat the actual current hybrid-ring lead on the direct recheck: short gate regressed loss_camera, and long gate regressed both loss_camera and loss_T."
        if decision == "REJECT"
        else "The residual hardcase mix beat the actual current lead on both short and long gate."
    )

    decision_payload = {
        "checked_at": now,
        "decision": decision,
        "decision_status": decision_status,
        "decision_family": "residual_case_coverage_rebalancing",
        "reviewed_candidate_shape": "promotedlead_hardcase_bucket_mix",
        "reviewed_candidate_config": residual_config,
        "current_local_lead_config_before_decision": prior_current_lead_config,
        "current_local_lead_family_before_decision": "source_policy_hybrid_ring_regularization",
        "source_packet": residual_packet_md_rel,
        "source_run_dir": residual_run_dir_rel,
        "formal_verdict": str(residual_packet.get("formal_status", "")),
        "formal_gate_stage": str(residual_packet.get("gate_reached", "")),
        "direct_recheck_vs_current_lead": {
            "short_gate_summary": residual_short_summary_rel,
            "long_gate_summary": residual_long_summary_rel,
            "short_gate_pass": short_pass,
            "long_gate_pass": long_pass,
        },
        "current_lead_short_gate_metrics": short_baseline,
        "candidate_short_gate_metrics": short_candidate,
        "candidate_short_gate_delta_vs_current_lead": short_delta,
        "current_lead_long_gate_metrics": long_baseline,
        "candidate_long_gate_metrics": long_candidate,
        "candidate_long_gate_delta_vs_current_lead": long_delta,
        "blocking_reason": blocking_reason,
        "current_lead_reference_source": repo_rel(CURRENT_LEAD_PROMOTION_DECISION_PATH),
        "cloud_must_remain_off": True,
        "no_auto_next_ticket": True,
        "same_night_second_candidate_forbidden": True,
        "same_family_retry_forbidden": True,
        "next_requirement": next_requirement,
    }
    write_json(decision_json_path, decision_payload)
    write_text(
        decision_md_path,
        "\n".join(
            [
                f"# Promotion Decision ({tag[:4]}-{tag[4:6]}-{tag[6:]})",
                "",
                f"- decision: `{decision}`",
                f"- decision_status: `{decision_status}`",
                "- decision_family: `residual_case_coverage_rebalancing`",
                "- reviewed_candidate_shape: `promotedlead_hardcase_bucket_mix`",
                f"- reviewed_candidate_config: `{residual_config}`",
                f"- current_local_lead_config_before_decision: `{prior_current_lead_config}`",
                f"- source_packet: `{residual_packet_md_rel}`",
                f"- formal_verdict: `{residual_packet.get('formal_status', '')}`",
                f"- formal_gate_stage: `{residual_packet.get('gate_reached', '')}`",
                f"- short_gate_pass_vs_current_lead: `{short_pass}`",
                f"- long_gate_pass_vs_current_lead: `{long_pass}`",
                "",
                "## Short Gate Recheck vs Current Lead",
                "",
                f"- current_lead camera/T/conf_depth/reg_depth: `{short_baseline}`",
                f"- candidate camera/T/conf_depth/reg_depth: `{short_candidate}`",
                f"- delta candidate-current: `{short_delta}`",
                "",
                "## Long Gate Recheck vs Current Lead",
                "",
                f"- current_lead camera/T/conf_depth/reg_depth: `{long_baseline}`",
                f"- candidate camera/T/conf_depth/reg_depth: `{long_candidate}`",
                f"- delta candidate-current: `{long_delta}`",
                "",
                "## Result",
                "",
                f"- blocking_reason: `{blocking_reason}`",
                f"- next_requirement: `{next_requirement}`",
                "",
            ]
        ),
    )

    hybrid_gate_refs = {
        "short_gate_reference_summary": str(current_lead_decision.get("short_gate_reference_summary", "")),
        "short_gate_reference_log": str(current_lead_decision.get("short_gate_reference_log", "")),
        "long_gate_reference_summary": str(current_lead_decision.get("long_gate_reference_summary", "")),
        "long_gate_reference_log": str(current_lead_decision.get("long_gate_reference_log", "")),
        "long_gate_reference_status": str(current_lead_decision.get("source_run_dir", "")) + "/status.json",
        "long_gate_reference_source_field": "candidate_log_from_promoted_hybrid_ring_run",
        "stable_lead_reference_note": "The canonical stable-lead reference now comes from the manually promoted hybrid-ring candidate logs, because that config remains the current local lead.",
    }
    residual_gate_refs = {
        "short_gate_reference_summary": residual_short_summary_rel,
        "short_gate_reference_log": residual_short_log_rel,
        "long_gate_reference_summary": residual_long_summary_rel,
        "long_gate_reference_log": residual_long_log_rel,
        "long_gate_reference_status": residual_run_status_rel,
        "long_gate_reference_source_field": "candidate_log_from_promoted_residual_case_coverage_run",
        "stable_lead_reference_note": "The canonical stable-lead reference now comes from the manually promoted residual-case-coverage candidate logs, because that config is the current local lead.",
    }

    if decision == "PROMOTE":
        current_lead_family = "residual_case_coverage_rebalancing"
        current_lead_shape = "promotedlead_hardcase_bucket_mix"
        current_lead_config = residual_config
        current_lead_packet = residual_packet_md_rel
        current_lead_gate_refs = residual_gate_refs
        main_direction = "Hold residual_case_coverage_rebalancing / promotedlead_hardcase_bucket_mix as the current local lead, keep research in steady_hold + IDLE_GUARD, keep no auto-next ticket, and keep cloud off until a fresh manual problem and a separate explicit downstream decision exist."
        cloud_blocker = "The manually promoted residual-case-coverage local lead is now synced as the current local lead, but cloud must remain off. This PROMOTE decision resolves local lead selection only; it does not authorize cloud execution, auto-next ticket generation, or any second candidate. Any later forward move still requires a fresh manual problem and a separate explicit downstream decision."
        why_lead = [
            "It beat the actual current hybrid-ring lead on both short and long gate under the corrected recheck.",
            "Manual PROMOTE has now resolved the hold, so this config is the current local lead while cloud stays off and no auto-next ticket is active.",
        ]
        local_manifest["problem_id"] = "residual_case_coverage_rebalancing_v1_promoted"
    else:
        current_lead_family = "source_policy_hybrid_ring_regularization"
        current_lead_shape = "stablelead_nearest_plus_uniform_tail"
        current_lead_config = prior_current_lead_config
        current_lead_packet = current_lead_decision_md_rel
        current_lead_gate_refs = hybrid_gate_refs
        main_direction = "Hold source_policy_hybrid_ring_regularization / stablelead_nearest_plus_uniform_tail as the current local lead, keep research in steady_hold + IDLE_GUARD, keep no auto-next ticket, and keep cloud off until a fresh manual problem and a separate explicit downstream decision exist."
        cloud_blocker = "The manually promoted hybrid-ring local lead remains the current local lead, and cloud must remain off. The residual_case_coverage_rebalancing ticket was executed but not promoted after a corrected recheck against the actual current lead. Any later forward move still requires a fresh manual problem and a separate explicit downstream decision."
        why_lead = [
            "It beat the previous stable lead at the short 10/5 gate on camera, conf_depth, and reg_depth while keeping translation flat.",
            "It also beat the previous stable lead again at the long 100/20 gate and therefore earned a formal provisional_lead before manual promotion.",
            "A later residual_case_coverage_rebalancing ticket improved depth terms but regressed camera on the corrected 10/5 recheck against this lead, so manual REJECT kept this config current.",
        ]
        local_manifest["problem_id"] = "source_policy_hybrid_ring_regularization_v1_promoted"

    local_manifest["status"] = "promoted_local_lead_synced_cloud_off"
    local_manifest["main_direction"] = main_direction
    local_manifest["current_cloud_blocker"] = cloud_blocker
    local_manifest["current_lead"] = {
        "config": current_lead_config,
        "gate_summary": current_lead_packet,
        "attribution_summary": current_lead_packet,
        "family": current_lead_family,
        "first_candidate_shape": current_lead_shape,
        "gate_references": current_lead_gate_refs,
        "why_lead": why_lead,
    }
    local_manifest["latest_manual_decision"] = {
        "decision": decision,
        "decision_status": decision_status,
        "decision_path": repo_rel(decision_json_path),
        "decision_family": "residual_case_coverage_rebalancing",
    }
    local_manifest["latest_rejected_candidate"] = (
        {
            "family": "residual_case_coverage_rebalancing",
            "candidate_shape": "promotedlead_hardcase_bucket_mix",
            "candidate_config": residual_config,
            "decision_path": repo_rel(decision_json_path),
            "reason": blocking_reason,
        }
        if decision == "REJECT"
        else {}
    )
    frozen = list(local_manifest.get("frozen_non_reentry_candidates", []) or [])
    frozen = upsert_by_key(
        frozen,
        {
            "name": "residual_case_coverage_rebalancing_same_family_retry",
            "reason": (
                "Executed but not promoted after corrected current-lead recheck."
                if decision == "REJECT"
                else "Promoted lead already spent the single-ticket family budget."
            ),
        },
        "name",
    )
    local_manifest["frozen_non_reentry_candidates"] = frozen
    local_manifest["next_local_diagnostic"] = {
        "name": "fresh_manual_problem_only_if_later_needed",
        "status": "deferred_until_future_manual_problem",
        "next_allowed_candidate_family": "",
        "goal": "Do not auto-open another ticket. Only define a fresh manual problem later if another genuinely new question is needed.",
    }
    local_manifest["non_goals"] = [
        "Do not arm or run another ticket tonight.",
        "Do not reopen the closed failure batch or the residual same-family retry.",
        "Do not auto-promote to cloud or auto-generate the next question.",
    ]
    local_manifest["single_source_of_truth"] = [
        current_lead_packet,
        repo_rel(decision_json_path),
        repo_rel(LOCAL_MANIFEST_PATH),
        repo_rel(TRAINING_QUESTION_MANIFEST_PATH),
    ]
    local_manifest["cloud_gate"] = False
    local_manifest["launch_cloud_now"] = False
    write_json(LOCAL_MANIFEST_PATH, local_manifest)

    training_question["status"] = "ready_for_new_training_question"
    training_question["problem_family"] = ""
    training_question["recommended_problem_id"] = ""
    training_question["candidate_config"] = current_lead_config
    training_question["current_cloud_blocker"] = cloud_blocker
    training_question["candidate_config_status"] = "promoted_local_lead_synced_no_active_next_training_question"
    training_question["geometry_direction"] = main_direction
    training_question["question"] = "No active next training question is open. The promoted local lead is synced; define a fresh manual problem only if later needed."
    training_question["why_now"] = [
        (
            "Manual REJECT has closed the residual_case_coverage_rebalancing hold after a corrected recheck against the actual current hybrid-ring lead."
            if decision == "REJECT"
            else "Manual PROMOTE has resolved the residual_case_coverage_rebalancing hold."
        ),
        "There is no current priority family and no auto-next ticket.",
        "The next forward move, if later needed, is a fresh manual problem rather than another same-family or cousin run.",
    ]
    training_question["candidate_implementation_requirements"] = [
        "Keep the current local lead as the reference config until a future manual problem is defined.",
        "Do not auto-arm another ticket from this manifest alone.",
        "Keep cloud off.",
    ]
    training_question["non_goals"] = [
        "Do not define a new manual problem automatically.",
        "Do not reopen the residual same-family retry or slot_3 ticket.",
        "Do not use this manifest as implicit approval for a new run.",
    ]
    training_question["patch_collection_stop"] = True
    training_question["ready_for_new_training_question"] = True
    training_question["cloud_gate"] = False
    training_question["launch_cloud_now"] = False
    write_json(TRAINING_QUESTION_MANIFEST_PATH, training_question)

    for payload in [
        {
            "id": "phase_17_new_manual_problem",
            "status": "completed",
            "details": "Defined and approved the new manual problem for residual_case_coverage_rebalancing.",
        },
        {
            "id": "phase_18_execution_readiness",
            "status": "completed",
            "details": "Made residual_case_coverage_rebalancing execution-ready via the official hard-tail manifest and hardcasebucketmix4to1 config.",
        },
        {
            "id": "phase_19_single_ticket_execution",
            "status": "completed",
            "details": "Ran the approved residual_case_coverage_rebalancing first ticket through long gate under the single-ticket contract.",
        },
        {
            "id": "phase_20_post_verdict_cleanup",
            "status": "completed",
            "details": "Archived the residual approval, removed active approved_problem.json, and returned research to IDLE_GUARD with no auto-next ticket.",
        },
        {
            "id": "phase_21_manual_promotion_decision",
            "status": "completed",
            "details": (
                "A corrected current-lead recheck forced REJECT for residual_case_coverage_rebalancing / promotedlead_hardcase_bucket_mix."
                if decision == "REJECT"
                else "A corrected current-lead recheck supported PROMOTE for residual_case_coverage_rebalancing / promotedlead_hardcase_bucket_mix."
            ),
        },
        {
            "id": "phase_22_rejection_sync" if decision == "REJECT" else "phase_22_promoted_lead_sync",
            "status": "completed",
            "details": (
                "Planning, task mode, manifests, and references were synced back to the existing hybrid-ring local lead while keeping cloud off and forbidding residual same-family retry."
                if decision == "REJECT"
                else "Planning, task mode, manifests, and references were synced to the promoted residual-case-coverage local lead while keeping cloud off."
            ),
        },
    ]:
        task_plan["completed_this_round"] = upsert_by_key(list(task_plan.get("completed_this_round", []) or []), payload, "id")

    loaded = dict(task_plan.get("loaded_long_process_plan", {}) or {})
    phases = list(loaded.get("phases", []) or [])
    for payload in [
        {
            "id": "phase_17_new_manual_problem",
            "status": "completed",
            "goal": "Define a genuinely new manual problem around official promoted hard-tail coverage.",
            "delivered_family": "residual_case_coverage_rebalancing",
        },
        {
            "id": "phase_18_execution_readiness",
            "status": "completed",
            "goal": "Convert residual_case_coverage_rebalancing from manual-review-ready to execution-ready.",
            "candidate_shape": "promotedlead_hardcase_bucket_mix",
        },
        {
            "id": "phase_19_single_ticket_execution",
            "status": "completed",
            "goal": "Run exactly one approved residual_case_coverage_rebalancing ticket.",
            "verdict": str(residual_packet.get("formal_status", "")),
            "gate_stage_reached": str(residual_packet.get("gate_reached", "")),
        },
        {
            "id": "phase_20_post_verdict_cleanup",
            "status": "completed",
            "goal": "Archive the residual approval, return to IDLE_GUARD, and clear auto-next planning residue while holding for manual decision.",
        },
        {
            "id": "phase_21_manual_promotion_decision",
            "status": "completed",
            "goal": "Make a fresh manual decision on residual_case_coverage_rebalancing / promotedlead_hardcase_bucket_mix.",
            "decision": decision,
        },
        {
            "id": "phase_22_rejection_sync" if decision == "REJECT" else "phase_22_promoted_lead_sync",
            "status": "completed",
            "goal": (
                "Sync the manual rejection outcome while keeping the existing hybrid-ring lead current, research idle, and cloud off."
                if decision == "REJECT"
                else "Sync the manually promoted residual-case-coverage result as the current local lead while keeping research idle and cloud off."
            ),
        },
    ]:
        phases = upsert_by_key(phases, payload, "id")
    loaded["mode"] = "TASK_MODE_ACTIVE_PROMOTED_LOCAL_LEAD_SYNCED"
    loaded["current_priority_family"] = ""
    loaded["auto_next_ticket_enabled"] = False
    loaded["phases"] = phases
    loaded["promoted_local_lead_config"] = current_lead_config

    frontier_item = {
        "family": "residual_case_coverage_rebalancing",
        "label": "promotedlead_hardcase_bucket_mix",
        "verdict": "executed_but_not_promoted" if decision == "REJECT" else "promoted_to_current_local_lead",
        "gate_stage_reached": "manual_decision_resolved_after_current_lead_recheck",
        "short_delta_camera_vs_current_lead": short_delta["delta_camera"],
        "short_delta_T_vs_current_lead": short_delta["delta_T"],
        "short_delta_conf_depth_vs_current_lead": short_delta["delta_conf_depth"],
        "short_delta_reg_depth_vs_current_lead": short_delta["delta_reg_depth"],
        "long_delta_camera_vs_current_lead": long_delta["delta_camera"],
        "long_delta_T_vs_current_lead": long_delta["delta_T"],
        "long_delta_conf_depth_vs_current_lead": long_delta["delta_conf_depth"],
        "long_delta_reg_depth_vs_current_lead": long_delta["delta_reg_depth"],
    }
    frontier = upsert_by_key(list(task_plan.get("frontier_progression", []) or []), frontier_item, "family")

    task_plan.update(
        {
            "checked_at": now,
            "latest_guard_checked_at": str(guard.get("checked_at", "")),
            "task_mode_status": "active",
            "current_mode": "steady_hold",
            "research_loop_mode": "IDLE_GUARD",
            "task_mode_focus": "promoted_local_lead_synced_cloud_off",
            "task_mode_contract": "single_problem_single_candidate_cross_night",
            "loaded_long_process_plan": loaded,
            "frontier_progression": frontier,
            "manual_promotion_decision": {
                "decision": decision,
                "decision_status": decision_status,
                "decision_path": str(decision_json_path.resolve()),
                "decision_family": "residual_case_coverage_rebalancing",
                "reviewed_candidate_shape": "promotedlead_hardcase_bucket_mix",
                "kept_current_local_lead_family": current_lead_family,
                "kept_current_local_lead_config": current_lead_config,
                "cloud_must_remain_off": True,
                "no_auto_next_ticket": True,
                "same_family_retry_forbidden": True,
            },
            "current_local_lead": {
                "family": current_lead_family,
                "first_candidate_shape": current_lead_shape,
                "config": current_lead_config,
            },
        }
    )
    write_json(TASK_PLAN_JSON_PATH, task_plan)
    write_text(
        TASK_PLAN_MD_PATH,
        render_task_plan_md(
            now,
            str(guard.get("checked_at", "")),
            decision,
            current_lead_family,
            current_lead_shape,
            current_lead_config,
        ),
    )
    write_text(SUMMARY_MD_PATH, render_summary_md(now, guard, decision, current_lead_config))

    run_checked([sys.executable, "-m", "py_compile", str(RESEARCH_LOOP_SCRIPT_PATH)])
    run_checked([sys.executable, str(RESEARCH_LOOP_SCRIPT_PATH)])
    run_checked([sys.executable, str(WATCH_SCRIPT_PATH), "--once"])
    print(
        json.dumps(
            {
                "decision": decision,
                "decision_status": decision_status,
                "decision_json": str(decision_json_path.resolve()),
                "decision_md": str(decision_md_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
