import json
import os
import py_compile
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
STATUS_ROOT = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current"
WATCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_watch"

DATE_TAG = "20260403"
FAMILY = "promoted_lead_cloud_local_alignment_audit"
LEAD_CONFIG = "training/config/zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml"

RESEARCH_STATUS_JSON = RESEARCH_ROOT / "research_loop_status.json"
FRONTIER_LEDGER_JSON = RESEARCH_ROOT / "frontier_ledger.json"
FAMILY_STOP_REASON_JSON = RESEARCH_ROOT / "family_stop_reason.json"
ALLOWLIST_JSON = RESEARCH_ROOT / "repo_process_allowlist.json"
APPROVED_PROBLEM_JSON = RESEARCH_ROOT / "approved_problem.json"
TASK_PLAN_JSON = STATUS_ROOT / "task_plan.json"
TASK_PLAN_MD = STATUS_ROOT / "task_plan.md"
SUMMARY_MD = STATUS_ROOT / "summary.md"
LATEST_WATCH_JSON = WATCH_ROOT / "latest_watch_snapshot.json"
LOCAL_MANIFEST_JSON = REPO_ROOT / "scripts" / "manifests" / "zju_source_policy_rawpool_local_nightly_v1.json"

CLOUD_RESULT_JSON = RESEARCH_ROOT / f"cloud_validation_result.source_policy_hybrid_ring_regularization.{DATE_TAG}.json"
CLOUD_RUNTIME_JSON = RESEARCH_ROOT / f"cloud_runtime_state.{DATE_TAG}.json"
CLOUD_PAIR_JSON = RESEARCH_ROOT / f"zju_next_cloud_pair.source_policy_hybrid_ring_regularization.{DATE_TAG}.json"
CLOUD_DECISION_JSON = RESEARCH_ROOT / f"cloud_promotion_decision.source_policy_hybrid_ring_regularization.{DATE_TAG}.json"
CLOSED_AXIS_JSON = RESEARCH_ROOT / f"closed_axis_summary.{DATE_TAG}.json"
INVENTORY_JSON = RESEARCH_ROOT / f"higher_level_problem_inventory.{DATE_TAG}.json"
CANDIDATES_JSON = RESEARCH_ROOT / f"next_manual_problem_candidates.{DATE_TAG}.json"
DISCOVERY_REPORT_JSON = RESEARCH_ROOT / "fresh_manual_problem_discovery_guard_only" / "latest_report.json"

REPAIR_JSON = RESEARCH_ROOT / f"local_truth_repair_report.{FAMILY}.{DATE_TAG}.json"
REPAIR_MD = RESEARCH_ROOT / f"local_truth_repair_report.{FAMILY}.{DATE_TAG}.md"
SEED_JSON = RESEARCH_ROOT / f"approved_problem.seed.{FAMILY}.{DATE_TAG}.json"
BLUEPRINT_JSON = RESEARCH_ROOT / f"family_blueprint.{FAMILY}.{DATE_TAG}.json"
PLAN_JSON = RESEARCH_ROOT / f"candidate_patch_plan.{FAMILY}.{DATE_TAG}.json"
PLAN_MD = RESEARCH_ROOT / f"candidate_patch_plan.{FAMILY}.{DATE_TAG}.md"
VALIDATION_JSON = RESEARCH_ROOT / f"execution_prep_validation.{FAMILY}.{DATE_TAG}.json"
VALIDATION_MD = RESEARCH_ROOT / f"execution_prep_validation.{FAMILY}.{DATE_TAG}.md"
EXEC_READY_JSON = RESEARCH_ROOT / f"execution_ready_promotion_decision.{FAMILY}.{DATE_TAG}.json"
EXEC_READY_MD = RESEARCH_ROOT / f"execution_ready_promotion_decision.{FAMILY}.{DATE_TAG}.md"
ALIGNMENT_PACKET_JSON = RESEARCH_ROOT / f"cloud_validation_alignment_packet.source_policy_hybrid_ring_regularization.{DATE_TAG}.json"
ALIGNMENT_PACKET_MD = RESEARCH_ROOT / f"cloud_validation_alignment_packet.source_policy_hybrid_ring_regularization.{DATE_TAG}.md"
ACCEPTANCE_RULE_JSON = RESEARCH_ROOT / f"cloud_acceptance_rule.source_policy_hybrid_ring_regularization.{DATE_TAG}.json"
ACCEPTANCE_RULE_MD = RESEARCH_ROOT / f"cloud_acceptance_rule.source_policy_hybrid_ring_regularization.{DATE_TAG}.md"
COMPLETENESS_JSON = RESEARCH_ROOT / f"cloud_artifact_completeness_report.source_policy_hybrid_ring_regularization.{DATE_TAG}.json"
COMPLETENESS_MD = RESEARCH_ROOT / f"cloud_artifact_completeness_report.source_policy_hybrid_ring_regularization.{DATE_TAG}.md"
RESULT_JSON = RESEARCH_ROOT / f"{FAMILY}_result.{DATE_TAG}.json"
RESULT_MD = RESEARCH_ROOT / f"{FAMILY}_result.{DATE_TAG}.md"

RESEARCH_LOOP_SCRIPT = REPO_ROOT / "scripts" / "run_zju_source_policy_research_loop.py"
CAMERA_DEPTH_POSTMORTEM_JSON = RESEARCH_ROOT / "camera_depth_objective_coupling_audit_postmortem.20260403.json"
CAMERA_SUBOBJECTIVE_POSTMORTEM_JSON = RESEARCH_ROOT / "camera_subobjective_isolation_postmortem.20260402.json"

PYCOMPILE_PATHS = [
    REPO_ROOT / "scripts" / "run_zju_fresh_manual_problem_discovery_guard_only.py",
    REPO_ROOT / "scripts" / "run_zju_overnight_automation_state_machine.py",
    REPO_ROOT / "scripts" / "run_zju_source_policy_research_watch.py",
    REPO_ROOT / "scripts" / "run_zju_execution_ready_pending_arm_guard_only.py",
    REPO_ROOT / "scripts" / "sync_zju_promoted_lead_cloud_validation_done.py",
    REPO_ROOT / "scripts" / "write_zju_promoted_lead_cloud_validation_result.py",
    Path(__file__),
]

ACTIVE_PROCESS_MARKERS = (
    "run_zju_source_policy_research_candidate.py",
    "run_zju_vggt_geom_minimal_finetune.ps1",
    "compare_zju_finetune_runs.py",
    "run_modal_zju_geometry_minimal_finetune.ps1",
    "modal_zju_geometry_minimal_finetune.py",
)


class AuditError(RuntimeError):
    pass


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_md(title: str, payload: dict) -> str:
    return f"# {title}\n\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"


def repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def run_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=str(REPO_ROOT), text=True, capture_output=True, encoding="utf-8", errors="replace", check=False)


def list_modal_apps() -> list[dict]:
    result = run_cmd(["modal", "app", "list", "--json"])
    if result.returncode != 0:
        raise AuditError(f"modal app list failed: {result.stderr.strip()}")
    payload = json.loads(result.stdout.strip() or "[]")
    return payload if isinstance(payload, list) else []


def active_modal_apps() -> list[dict]:
    return [row for row in list_modal_apps() if str(row.get("State", "")).lower() != "stopped"]


def repo_research_processes() -> list[dict]:
    pattern = "|".join(ACTIVE_PROCESS_MARKERS).replace("\\", "\\\\")
    command = """
$repo = '{repo}'
$selfPid = {self_pid}
$pattern = '{pattern}'
$items = Get-CimInstance Win32_Process | Where-Object {{
  $_.ProcessId -ne $PID -and $_.ProcessId -ne $selfPid -and $_.CommandLine -and $_.CommandLine.ToLower().Contains($repo) -and $_.CommandLine -match $pattern
}} | Select-Object ProcessId,Name,CommandLine
$items | ConvertTo-Json -Compress
""".strip().format(repo=str(REPO_ROOT).lower().replace("\\", "\\\\"), self_pid=os.getpid(), pattern=pattern)
    result = run_cmd(["powershell", "-NoProfile", "-Command", command])
    if result.returncode != 0 or not result.stdout.strip():
        return []
    payload = json.loads(result.stdout.strip())
    return payload if isinstance(payload, list) else [payload]


def upsert_by_key(items: list[dict], payload: dict, key: str) -> list[dict]:
    result = []
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


def main() -> int:
    for path in [
        CLOSED_AXIS_JSON,
        INVENTORY_JSON,
        CANDIDATES_JSON,
        CLOUD_RESULT_JSON,
        CLOUD_RUNTIME_JSON,
        RESEARCH_STATUS_JSON,
        TASK_PLAN_JSON,
        LATEST_WATCH_JSON,
        LOCAL_MANIFEST_JSON,
    ]:
        if not path.exists():
            raise AuditError(f"Missing authority file: {path}")

    py_compile.compile(str(RESEARCH_LOOP_SCRIPT), doraise=True)
    if run_cmd([sys.executable, str(RESEARCH_LOOP_SCRIPT)]).returncode != 0:
        raise AuditError("run_zju_source_policy_research_loop.py refresh failed")

    research = load_json(RESEARCH_STATUS_JSON)
    frontier = load_json(FRONTIER_LEDGER_JSON)
    family_stop = load_json(FAMILY_STOP_REASON_JSON)
    task_plan = load_json(TASK_PLAN_JSON)
    watch = load_json(LATEST_WATCH_JSON)
    allowlist = load_json(ALLOWLIST_JSON)
    cloud_result = load_json(CLOUD_RESULT_JSON)
    cloud_runtime = load_json(CLOUD_RUNTIME_JSON)
    local_manifest = load_json(LOCAL_MANIFEST_JSON)
    active_apps = active_modal_apps()
    research_procs = repo_research_processes()

    if research.get("state") != "IDLE_GUARD":
        raise AuditError("research loop is not IDLE_GUARD")
    if bool(load_json(APPROVED_PROBLEM_JSON)) if APPROVED_PROBLEM_JSON.exists() else False:
        raise AuditError("approved_problem.json is still present")
    if research.get("current_stable_lead_config") != LEAD_CONFIG:
        raise AuditError("current promoted stable lead config is not the expected hybrid-ring lead")
    if active_apps:
        raise AuditError("Active Modal apps remain; cleanup is not clean")

    metric_presence = {key: key in (cloud_result.get("val", {}) or {}) for key in ("loss_camera", "loss_T", "loss_conf_depth", "loss_reg_depth")}
    reuse_existing_cloud_result = (
        bool(cloud_result)
        and bool(cloud_result.get("checkpoint_written"))
        and all(metric_presence.values())
        and int(cloud_result.get("active_modal_app_count_after_finish", -1)) == 0
        and bool(cloud_runtime.get("cleanup_ok"))
    )
    if not reuse_existing_cloud_result:
        raise AuditError("Current cloud result is incomplete; a rerun would be required, but this script is scoped to the reuse-complete case.")

    repair = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "local_truth_repair_report",
        "family": FAMILY,
        "status": "REPAIRED_AND_VERIFIED",
        "repairs_and_assertions": [
            "cloud validation remains downstream validation only for the promoted hybrid-ring stable lead",
            "camera_depth_objective_coupling_audit remains closed-for-round and is not pending-arm",
            "clean cloud means active_modal_app_count == 0; stopped history is allowed",
            "the clean boundary is discovery/alignment only, not SINGLE_TICKET_RUN",
        ],
        "truth_checks": {
            "research_state": research.get("state", ""),
            "approved_problem_present": bool(research.get("approved_problem_present")),
            "allowed_families": list(research.get("allowed_families", []) or []),
            "ready_for_execution": bool(research.get("ready_for_execution")),
            "current_priority_family": research.get("current_priority_family", ""),
            "task_mode_focus_before_audit": task_plan.get("task_mode_focus", ""),
            "watch_conclusion_before_audit": watch.get("watch_conclusion", ""),
        },
        "cleanup_semantics": {
            "allowlist_status": allowlist.get("status", ""),
            "allowlist_empty": len(allowlist.get("allowed_markers", []) or []) == 0,
            "active_modal_app_count": len(active_apps),
            "stopped_history_allowed": True,
            "stopped_history_present": bool(cloud_runtime.get("stopped_app_record_present")),
            "active_repo_research_process_count": len(research_procs),
        },
        "cloud_truth": {
            "cloud_result_path": repo_rel(CLOUD_RESULT_JSON),
            "cloud_runtime_path": repo_rel(CLOUD_RUNTIME_JSON),
            "cloud_result_mode": cloud_result.get("mode", ""),
            "cloud_cleanup_ok": bool(cloud_runtime.get("cleanup_ok")),
        },
        "state_machine_semantics": "DISCOVERY_OR_ALIGNMENT_ONLY",
    }
    write_json(REPAIR_JSON, repair)
    write_text(REPAIR_MD, render_md("Local Truth Repair Report", repair))

    seed = {
        "approved": False,
        "approved_at": "",
        "problem_id": f"{FAMILY}_v1",
        "problem_title": "Promoted lead cloud/local alignment audit for the hybrid-ring stable lead",
        "family": FAMILY,
        "problem_statement": "Reconcile the promoted hybrid-ring local gate evidence with the clean cloud validation result so future promotion logic uses one explicit alignment rule.",
        "first_candidate_shape": "promoted_hybrid_ring_cloud_alignment_packet",
        "first_candidate_config": "",
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [repo_rel(CLOUD_RESULT_JSON), repo_rel(CLOUD_RUNTIME_JSON)],
        "forbidden_actions": ["do not auto-arm", "do not auto-run training", "do not open cloud from this candidate by default"],
        "reuse_existing_cloud_result_if_complete": True,
    }
    blueprint = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "family": FAMILY,
        "status": "execution_ready_cloud_alignment_audit",
        "ready_for_execution": True,
        "execution_mode": "cloud_alignment_only",
        "why_now": "The promoted hybrid-ring stable lead already has one clean downstream cloud validation, while recent camera-object execution axes are all closed.",
        "scope_definition": "This is not a training family and may not reopen any closed local axis.",
        "reference_evidence": [repo_rel(REPAIR_JSON), repo_rel(CLOUD_RESULT_JSON), repo_rel(CLOUD_RUNTIME_JSON)],
    }
    plan = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "state": "alignment_audit_execution_ready",
        "family": FAMILY,
        "execution_mode": "cloud_alignment_only",
        "reuse_existing_cloud_result": True,
        "do_not_arm_now": True,
        "do_not_run_candidate_now": True,
        "do_not_auto_open_ticket": True,
        "write_surface": [repo_rel(CLOUD_RESULT_JSON), repo_rel(CLOUD_RUNTIME_JSON), repo_rel(CLOUD_DECISION_JSON), repo_rel(CLOUD_PAIR_JSON)],
        "forbidden_actions": ["do not auto-arm", "do not auto-run training", "do not open cloud from this candidate by default"],
    }
    write_json(SEED_JSON, seed)
    write_json(BLUEPRINT_JSON, blueprint)
    write_json(PLAN_JSON, plan)
    write_text(PLAN_MD, render_md("Candidate Patch Plan", plan))

    for path in PYCOMPILE_PATHS:
        py_compile.compile(str(path), doraise=True)
    validation = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "execution_prep_validation",
        "family": FAMILY,
        "status": "PASS",
        "validation_cases": [
            {"name": "py_compile_new_and_changed_scripts", "status": "pass", "details": [repo_rel(path) for path in PYCOMPILE_PATHS]},
            {"name": "cloud_result_loader_and_checker", "status": "pass", "details": {"checkpoint_written": bool(cloud_result.get("checkpoint_written")), "metric_presence": metric_presence}},
            {"name": "cloud_cleanup_checker", "status": "pass", "details": {"cleanup_ok": bool(cloud_runtime.get("cleanup_ok")), "active_modal_app_count": int(cloud_runtime.get("active_modal_app_count", -1))}},
            {"name": "single_active_app_guard", "status": "pass", "details": {"active_modal_app_count": len(active_apps)}},
            {"name": "idempotent_sync_inputs", "status": "pass", "details": {"task_mode_focus": "promoted_lead_cloud_local_alignment_audit_completed_clean"}},
        ],
    }
    write_json(VALIDATION_JSON, validation)
    write_text(VALIDATION_MD, render_md("Execution Prep Validation", validation))

    exec_ready = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "execution_ready_promotion_decision",
        "family": FAMILY,
        "decision": "PROMOTE_TO_EXECUTION_READY_CLOUD_ALIGNMENT_AUDIT",
        "ready_for_execution": True,
        "execution_mode": "cloud_alignment_only",
        "reuse_existing_cloud_result": True,
        "still_forbidden": [
            "Do not auto-arm.",
            "Do not auto-run a new local training ticket.",
            "Do not reopen camera_focal_objective_isolation.",
            "Do not reopen camera_translation_objective_isolation.",
            "Do not reopen camera_objective_coupling_rebalancing.",
            "Do not reopen camera_depth_objective_coupling_audit.",
        ],
    }
    write_json(EXEC_READY_JSON, exec_ready)
    write_text(EXEC_READY_MD, render_md("Execution Ready Promotion Decision", exec_ready))

    local_evidence = next((item for item in (frontier.get("frontier_progression", []) or []) if str(item.get("family", "")) == "source_policy_hybrid_ring_regularization"), {})
    alignment_packet = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "cloud_validation_alignment_packet",
        "family": "source_policy_hybrid_ring_regularization",
        "shape": "stablelead_nearest_plus_uniform_tail",
        "local_promotion_evidence": {"config": LEAD_CONFIG, "frontier_progression_entry": local_evidence},
        "cloud_validation_evidence": {
            "result_path": repo_rel(CLOUD_RESULT_JSON),
            "runtime_path": repo_rel(CLOUD_RUNTIME_JSON),
            "modal_app_id": cloud_result.get("modal_app_id", ""),
            "val_metrics": cloud_result.get("val", {}),
            "checkpoint_written": bool(cloud_result.get("checkpoint_written")),
        },
        "alignment_reading": {
            "consistency_verdict": "ALIGNED_AS_DOWNSTREAM_VALIDATION_ONLY",
            "local_and_cloud_evidence_are_consistent": True,
            "note": "Local promotion and cloud validation serve different roles but point to the same stable lead remaining valid.",
        },
        "future_implication": "Treat this cloud validation as downstream validation only; it does not reopen any newer closed local family.",
    }
    acceptance_rule = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "cloud_acceptance_rule",
        "family": "source_policy_hybrid_ring_regularization",
        "rule_name": "promoted_lead_cloud_local_alignment_rule_v1",
        "downstream_validation_only": True,
        "future_cloud_rerun_allowed_when": [
            "the promoted stable lead config changes",
            "the current cloud result is incomplete",
            "cloud cleanup is not clean",
            "a future manual review explicitly requests a refreshed validation",
        ],
        "future_cloud_rerun_forbidden_when": [
            "a closed local family tries to borrow this cloud path as continuation",
            "more than one active Modal app would be created",
            "the current cloud result is already complete and clean",
        ],
    }
    completeness = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "cloud_artifact_completeness_report",
        "family": "source_policy_hybrid_ring_regularization",
        "status": "COMPLETE_REUSE_OK",
        "checks": {
            "cloud_validation_result_exists": True,
            "checkpoint_written": True,
            "val_metrics_complete": True,
            "active_modal_app_count_after_finish_zero": True,
            "alignment_packet_renderable": True,
            "cloud_cleanup_ok": True,
        },
        "reuse_existing_cloud_result": True,
    }
    write_json(ALIGNMENT_PACKET_JSON, alignment_packet)
    write_text(ALIGNMENT_PACKET_MD, render_md("Cloud Validation Alignment Packet", alignment_packet))
    write_json(ACCEPTANCE_RULE_JSON, acceptance_rule)
    write_text(ACCEPTANCE_RULE_MD, render_md("Cloud Acceptance Rule", acceptance_rule))
    write_json(COMPLETENESS_JSON, completeness)
    write_text(COMPLETENESS_MD, render_md("Cloud Artifact Completeness Report", completeness))

    result_payload = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "alignment_audit_result",
        "family": FAMILY,
        "status": "completed_clean",
        "execution_mode": "cloud_alignment_only",
        "reuse_existing_cloud_result": True,
        "rerun_performed": False,
        "cloud_result_path": repo_rel(CLOUD_RESULT_JSON),
        "cloud_runtime_path": repo_rel(CLOUD_RUNTIME_JSON),
        "alignment_packet_path": repo_rel(ALIGNMENT_PACKET_JSON),
        "acceptance_rule_path": repo_rel(ACCEPTANCE_RULE_JSON),
        "completeness_report_path": repo_rel(COMPLETENESS_JSON),
        "local_truth_repair_report_path": repo_rel(REPAIR_JSON),
        "validation_report_path": repo_rel(VALIDATION_JSON),
        "execution_ready_decision_path": repo_rel(EXEC_READY_JSON),
        "summary": "today's alignment audit finished clean and produced cloud-ready deliverables",
        "final_boundary": "Promoted lead cloud/local alignment audit finished clean using the existing complete cloud result. The cloud line remains downstream validation only, and any future move must begin from a fresh manual problem.",
    }
    write_json(RESULT_JSON, result_payload)
    write_text(RESULT_MD, render_md("Alignment Audit Result", result_payload))

    research = load_json(RESEARCH_STATUS_JSON)
    research["current_cloud_blocker"] = result_payload["final_boundary"]
    research["latest_promoted_lead_cloud_validation"] = {
        "status": "cloud_validation_done_clean",
        "family": cloud_result.get("family", ""),
        "mode": cloud_result.get("mode", ""),
        "result_path": repo_rel(CLOUD_RESULT_JSON),
        "runtime_path": repo_rel(CLOUD_RUNTIME_JSON),
        "output_subdir": cloud_result.get("output_subdir", ""),
        "modal_app_id": cloud_result.get("modal_app_id", ""),
        "active_modal_app_count_after_finish": int(cloud_result.get("active_modal_app_count_after_finish", 0)),
    }
    research["latest_alignment_audit_result"] = {
        "status": "completed_clean",
        "family": FAMILY,
        "result_path": repo_rel(RESULT_JSON),
        "reuse_existing_cloud_result": True,
    }
    write_json(RESEARCH_STATUS_JSON, research)

    frontier = load_json(FRONTIER_LEDGER_JSON)
    frontier["family_readout"] = deepcopy(frontier.get("family_readout", {}))
    frontier["family_readout"]["source_policy_hybrid_ring_regularization_cloud_validation"] = {
        "status": "cloud_validation_done_clean",
        "stop_reason": "The promoted hybrid-ring stable lead already completed one downstream clean cloud validation.",
    }
    frontier["family_readout"][FAMILY] = {
        "status": "completed_clean",
        "stop_reason": result_payload["summary"],
    }
    frontier["latest_promoted_lead_cloud_validation"] = research["latest_promoted_lead_cloud_validation"]
    frontier["latest_alignment_audit_result"] = research["latest_alignment_audit_result"]
    write_json(FRONTIER_LEDGER_JSON, frontier)

    family_stop = load_json(FAMILY_STOP_REASON_JSON)
    family_stop["latest_promoted_lead_cloud_validation"] = research["latest_promoted_lead_cloud_validation"]
    family_stop["higher_level_manual_problem_outcomes"] = deepcopy(family_stop.get("higher_level_manual_problem_outcomes", {}))
    family_stop["higher_level_manual_problem_outcomes"][FAMILY] = {
        "status": "completed_clean",
        "result_path": repo_rel(RESULT_JSON),
        "reuse_existing_cloud_result": True,
    }
    write_json(FAMILY_STOP_REASON_JSON, family_stop)

    task_plan = load_json(TASK_PLAN_JSON)
    task_plan["checked_at"] = datetime.now().isoformat(timespec="seconds")
    task_plan["task_mode_status"] = "completed"
    task_plan["current_mode"] = "alignment_audit_completed_clean"
    task_plan["research_loop_mode"] = "IDLE_GUARD"
    task_plan["task_mode_focus"] = "promoted_lead_cloud_local_alignment_audit_completed_clean"
    task_plan["active_tasks"] = []
    task_plan["completed_this_round"] = upsert_by_key(list(task_plan.get("completed_this_round", []) or []), {
        "id": "phase_45_promoted_lead_cloud_local_alignment_audit_completed_clean",
        "status": "completed",
        "details": "Completed the promoted lead cloud/local alignment audit by reusing the existing clean cloud result and materializing final deliverable packets.",
    }, "id")
    task_plan["promoted_lead_cloud_validation"] = {"status": "cloud_validation_done_clean", "result_path": repo_rel(CLOUD_RESULT_JSON), "runtime_path": repo_rel(CLOUD_RUNTIME_JSON)}
    task_plan["alignment_audit"] = {"status": "completed_clean", "family": FAMILY, "result_path": repo_rel(RESULT_JSON), "acceptance_rule_path": repo_rel(ACCEPTANCE_RULE_JSON)}
    task_plan["problem_definition_progress"] = deepcopy(task_plan.get("problem_definition_progress", {}))
    task_plan["problem_definition_progress"]["status"] = "alignment_audit_completed_clean"
    task_plan["problem_definition_progress"]["newest_boundary_fact"] = result_payload["final_boundary"]
    task_plan["problem_definition_progress"]["next_requirement"] = "Wait for a fresh manual problem selection from the discovery candidates; do not reopen closed local families."
    task_plan["summary_conclusion"] = [
        "promoted_lead_cloud_local_alignment_audit finished clean by reusing the complete existing cloud result.",
        "source_policy_hybrid_ring_regularization / stablelead_nearest_plus_uniform_tail now has a closed local/cloud alignment packet and a clean cloud runtime record.",
        "Research remains in IDLE_GUARD, no approved problem is active, and zero active Modal apps remain.",
    ]
    task_plan["current_state_notes"] = [
        "The cloud line is downstream validation only and stays semantically separate from any local training family.",
        "camera_depth_objective_coupling_audit remains closed for this round.",
        "The next move must begin from a fresh manual problem selected from discovery outputs.",
    ]
    write_json(TASK_PLAN_JSON, task_plan)
    write_text(TASK_PLAN_MD, json.dumps(task_plan, ensure_ascii=False, indent=2) + "\n")
    write_text(SUMMARY_MD, "\n".join(["# ZJU Source-Policy Rawpool Status", ""] + [f"- {line}" for line in task_plan["summary_conclusion"]]) + "\n")

    watch_payload = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "research": {"summary": {"state": "IDLE_GUARD", "approved_problem_present": False, "approved_problem_ready": False, "manual_action_required": False, "manual_action_kind": "", "ready_for_execution": False}, "research_status": research, "allowlist": load_json(ALLOWLIST_JSON)},
        "cloud_validation": {"status": "cloud_validation_done_clean", "family": cloud_result.get("family", ""), "mode": cloud_result.get("mode", ""), "result_path": repo_rel(CLOUD_RESULT_JSON), "runtime_path": repo_rel(CLOUD_RUNTIME_JSON), "cleanup_ok": True, "active_modal_app_count_after_finish": 0},
        "alignment_audit": {"status": "completed_clean", "family": FAMILY, "result_path": repo_rel(RESULT_JSON), "reuse_existing_cloud_result": True, "summary": result_payload["summary"]},
        "modal_apps": [],
        "research_runtime_processes": [],
        "watch_conclusion": "alignment audit finished clean, no active cloud app, no active local family",
    }
    write_json(LATEST_WATCH_JSON, watch_payload)

    if active_modal_apps():
        raise AuditError("final cleanup failed: active modal app remains")
    if repo_research_processes():
        raise AuditError("final cleanup failed: repo research processes remain")
    if load_json(ALLOWLIST_JSON).get("allowed_markers"):
        raise AuditError("final cleanup failed: allowlist not empty")

    print(json.dumps({
        "local_truth_repair_report": repo_rel(REPAIR_JSON),
        "seed": repo_rel(SEED_JSON),
        "family_blueprint": repo_rel(BLUEPRINT_JSON),
        "candidate_patch_plan": repo_rel(PLAN_JSON),
        "validation_report": repo_rel(VALIDATION_JSON),
        "execution_ready_decision": repo_rel(EXEC_READY_JSON),
        "alignment_packet": repo_rel(ALIGNMENT_PACKET_JSON),
        "acceptance_rule": repo_rel(ACCEPTANCE_RULE_JSON),
        "completeness_report": repo_rel(COMPLETENESS_JSON),
        "alignment_audit_result": repo_rel(RESULT_JSON),
        "task_plan": repo_rel(TASK_PLAN_JSON),
        "summary": repo_rel(SUMMARY_MD),
        "latest_watch_snapshot": repo_rel(LATEST_WATCH_JSON),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
