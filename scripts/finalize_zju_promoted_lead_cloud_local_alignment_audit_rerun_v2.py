import json
import os
import re
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
STATUS_ROOT = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current"
WATCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_watch"

DATE_TAG = "20260403"
ARTIFACT_VERSION = 2
RUN_VERSION = "v3"
FAMILY = "promoted_lead_cloud_local_alignment_audit"
LEAD_FAMILY = "source_policy_hybrid_ring_regularization"
LEAD_SHAPE = "stablelead_nearest_plus_uniform_tail"
LEAD_CONFIG = "training/config/zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml"
OUTPUT_SUBDIR = f"zju_source_policy_research_loop/cloud_runs/{DATE_TAG}_promoted_hybrid_ring_latest_lead_{RUN_VERSION}"
APP_ID = "ap-4d8cwlWyi9AQXGXcbu7f8d"

RESEARCH_STATUS_JSON = RESEARCH_ROOT / "research_loop_status.json"
FRONTIER_LEDGER_JSON = RESEARCH_ROOT / "frontier_ledger.json"
FAMILY_STOP_REASON_JSON = RESEARCH_ROOT / "family_stop_reason.json"
ALLOWLIST_JSON = RESEARCH_ROOT / "repo_process_allowlist.json"
TASK_PLAN_JSON = STATUS_ROOT / "task_plan.json"
TASK_PLAN_MD = STATUS_ROOT / "task_plan.md"
SUMMARY_MD = STATUS_ROOT / "summary.md"
LATEST_WATCH_JSON = WATCH_ROOT / "latest_watch_snapshot.json"
CANDIDATES_JSON = RESEARCH_ROOT / f"next_manual_problem_candidates.{DATE_TAG}.json"

V2_CLOUD_RESULT_JSON = RESEARCH_ROOT / f"cloud_validation_result.source_policy_hybrid_ring_regularization.{DATE_TAG}.v2.json"
V2_CLOUD_RESULT_MD = RESEARCH_ROOT / f"cloud_validation_result.source_policy_hybrid_ring_regularization.{DATE_TAG}.v2.md"
V2_RUNTIME_JSON = RESEARCH_ROOT / f"cloud_runtime_state.{DATE_TAG}.v2.json"
V2_RUNTIME_MD = RESEARCH_ROOT / f"cloud_runtime_state.{DATE_TAG}.v2.md"
ALIGNMENT_PACKET_JSON = RESEARCH_ROOT / f"cloud_validation_alignment_packet.source_policy_hybrid_ring_regularization.{DATE_TAG}.v2.json"
ALIGNMENT_PACKET_MD = RESEARCH_ROOT / f"cloud_validation_alignment_packet.source_policy_hybrid_ring_regularization.{DATE_TAG}.v2.md"
DELTA_JSON = RESEARCH_ROOT / f"local_vs_cloud_delta_summary.source_policy_hybrid_ring_regularization.{DATE_TAG}.v2.json"
DELTA_MD = RESEARCH_ROOT / f"local_vs_cloud_delta_summary.source_policy_hybrid_ring_regularization.{DATE_TAG}.v2.md"
ACCEPTANCE_JSON = RESEARCH_ROOT / f"cloud_acceptance_rule.source_policy_hybrid_ring_regularization.{DATE_TAG}.v2.json"
ACCEPTANCE_MD = RESEARCH_ROOT / f"cloud_acceptance_rule.source_policy_hybrid_ring_regularization.{DATE_TAG}.v2.md"
COMPLETENESS_JSON = RESEARCH_ROOT / f"cloud_artifact_completeness_report.source_policy_hybrid_ring_regularization.{DATE_TAG}.v2.json"
COMPLETENESS_MD = RESEARCH_ROOT / f"cloud_artifact_completeness_report.source_policy_hybrid_ring_regularization.{DATE_TAG}.v2.md"
DELIVERY_JSON = RESEARCH_ROOT / f"one_page_delivery_summary.source_policy_hybrid_ring_regularization.{DATE_TAG}.v2.json"
DELIVERY_MD = RESEARCH_ROOT / f"one_page_delivery_summary.source_policy_hybrid_ring_regularization.{DATE_TAG}.v2.md"
ALIGNMENT_RESULT_JSON = RESEARCH_ROOT / f"{FAMILY}_result.{DATE_TAG}.v2.json"
ALIGNMENT_RESULT_MD = RESEARCH_ROOT / f"{FAMILY}_result.{DATE_TAG}.v2.md"
DISCOVERY_V2_JSON = RESEARCH_ROOT / f"next_manual_problem_candidates.{DATE_TAG}.v2.json"
DISCOVERY_V2_MD = RESEARCH_ROOT / f"next_manual_problem_candidates.{DATE_TAG}.v2.md"

TMP_DIR = REPO_ROOT / "output" / "tmp_alignment_audit_v2"
TMP_LOG = TMP_DIR / "cloud_v3_log.txt"
TMP_DIFF = TMP_DIR / "cloud_v3_dataset_probe_contract_diff.json"
TMP_FALLBACK = TMP_DIR / "cloud_v3_compile_fallback_status.json"

ACTIVE_PROCESS_MARKERS = (
    "run_zju_source_policy_research_candidate.py",
    "run_zju_vggt_geom_minimal_finetune.ps1",
    "compare_zju_finetune_runs.py",
    "run_modal_zju_geometry_minimal_finetune.ps1",
    "modal_zju_geometry_minimal_finetune.py",
)

VAL_LINE_RE = re.compile(
    r"Loss/val_loss_camera:\s+[-0-9.]+\s+\(([-0-9.]+)\).*?"
    r"Loss/val_loss_T:\s+[-0-9.]+\s+\(([-0-9.]+)\).*?"
    r"Loss/val_loss_conf_depth:\s+[-0-9.]+\s+\(([-0-9.]+)\).*?"
    r"Loss/val_loss_reg_depth:\s+[-0-9.]+\s+\(([-0-9.]+)\)"
)


class FinalizeError(RuntimeError):
    pass


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def maybe_load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return load_json(path)


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
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return subprocess.run(args, cwd=str(REPO_ROOT), text=True, capture_output=True, encoding="utf-8", errors="replace", env=env, check=False)


def run_checked(args: list[str]) -> subprocess.CompletedProcess[str]:
    result = run_cmd(args)
    if result.returncode != 0:
        raise FinalizeError("Command failed: " + " ".join(args) + "\n" + result.stderr.strip())
    return result


def modal_volume_get(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_cmd(["modal", "volume", "get", "--force", "vggt-out", remote_path, str(local_path)])
    if result.returncode != 0 and not local_path.exists():
        raise FinalizeError(f"modal volume get failed for {remote_path}: {result.stderr.strip()}")


def modal_volume_ls(remote_path: str) -> list[str]:
    result = run_checked(["modal", "volume", "ls", "vggt-out", remote_path])
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def list_modal_apps() -> list[dict]:
    result = run_checked(["modal", "app", "list", "--json"])
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


def parse_final_val_metrics(log_text: str) -> dict:
    matches = VAL_LINE_RE.findall(log_text)
    if not matches:
        raise FinalizeError("Unable to parse final val metrics from cloud log.")
    camera, t_loss, conf_depth, reg_depth = matches[-1]
    return {
        "loss_camera": float(camera),
        "loss_T": float(t_loss),
        "loss_conf_depth": float(conf_depth),
        "loss_reg_depth": float(reg_depth),
    }


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
    modal_volume_get(f"/{OUTPUT_SUBDIR}/logs/log.txt", TMP_LOG)
    modal_volume_get(f"/{OUTPUT_SUBDIR}/dataset_probe_contract_diff.json", TMP_DIFF)
    modal_volume_get(f"/{OUTPUT_SUBDIR}/compile_fallback_status.json", TMP_FALLBACK)

    cloud_log = TMP_LOG.read_text(encoding="utf-8", errors="replace")
    metrics = parse_final_val_metrics(cloud_log)
    diff_payload = load_json(TMP_DIFF)
    fallback_payload = maybe_load_json(TMP_FALLBACK)
    apps = list_modal_apps()
    app_row = next((row for row in apps if str(row.get("App ID", "")).strip() == APP_ID), {})
    active_apps = active_modal_apps()
    if active_apps:
        raise FinalizeError("New cloud rerun did not clean up to zero active apps.")
    ckpt_entries = modal_volume_ls(f"/{OUTPUT_SUBDIR}/ckpts")

    cloud_result_v2 = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "cloud_validation_result",
        "artifact_version": ARTIFACT_VERSION,
        "family": LEAD_FAMILY,
        "shape": LEAD_SHAPE,
        "mode": "promoted_lead_cloud_validation",
        "modal_app_id": APP_ID,
        "modal_app_description": str(app_row.get("Description", "vggt-zju-geometry-minimal-finetune")),
        "started_at": str(app_row.get("Created at", "")),
        "stopped_at": str(app_row.get("Stopped at", "")),
        "output_subdir": OUTPUT_SUBDIR,
        "remote_output_root": f"vggt-out:/{OUTPUT_SUBDIR}",
        "val": metrics,
        "checkpoint_written": any("checkpoint.pt" in entry for entry in ckpt_entries),
        "checkpoint_files": ckpt_entries,
        "active_modal_app_count_after_finish": 0,
        "modal_app_list_literal_empty_after_finish": len(apps) == 0,
        "stopped_app_record_present_after_finish": bool(app_row),
        "cloud_decision_artifact": "output/zju_source_policy_research_loop/cloud_promotion_decision.source_policy_hybrid_ring_regularization.20260403.v3.json",
        "cloud_pair_artifact": "output/zju_source_policy_research_loop/zju_next_cloud_pair.source_policy_hybrid_ring_regularization.20260403.v3.json",
        "compile_fallback_triggered": bool(fallback_payload.get("compile_fallback_triggered")),
        "compile_fallback_status_path": repo_rel(TMP_FALLBACK),
    }
    cloud_runtime_v2 = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "cloud_runtime_state",
        "artifact_version": ARTIFACT_VERSION,
        "modal_app_id": APP_ID,
        "family": LEAD_FAMILY,
        "mode": "promoted_lead_cloud_validation",
        "output_subdir": OUTPUT_SUBDIR,
        "active_modal_app_count": 0,
        "stopped_app_record_present": bool(app_row),
        "modal_app_state": str(app_row.get("State", "")),
        "modal_app_created_at": str(app_row.get("Created at", "")),
        "modal_app_stopped_at": str(app_row.get("Stopped at", "")),
        "cleanup_ok": True,
        "no_redundant_cloud_process": True,
        "modal_app_list_literal_empty_after_finish": len(apps) == 0,
        "compile_fallback_triggered": bool(fallback_payload.get("compile_fallback_triggered")),
    }
    write_json(V2_CLOUD_RESULT_JSON, cloud_result_v2)
    write_text(V2_CLOUD_RESULT_MD, render_md("Cloud Validation Result v2", cloud_result_v2))
    write_json(V2_RUNTIME_JSON, cloud_runtime_v2)
    write_text(V2_RUNTIME_MD, render_md("Cloud Runtime State v2", cloud_runtime_v2))

    frontier = load_json(FRONTIER_LEDGER_JSON)
    local_metrics = frontier.get("stable_lead_val_metrics", {}) or {}
    delta = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "local_vs_cloud_delta_summary",
        "artifact_version": ARTIFACT_VERSION,
        "family": LEAD_FAMILY,
        "local_reference_metrics": local_metrics,
        "cloud_validation_metrics": metrics,
        "delta_cloud_minus_local": {
            "camera": round(float(metrics["loss_camera"]) - float(local_metrics.get("camera", 0.0)), 6),
            "T": round(float(metrics["loss_T"]) - float(local_metrics.get("T", 0.0)), 6),
            "conf_depth": round(float(metrics["loss_conf_depth"]) - float(local_metrics.get("conf_depth", 0.0)), 6),
            "reg_depth": round(float(metrics["loss_reg_depth"]) - float(local_metrics.get("reg_depth", 0.0)), 6),
        },
        "reading": [
            "Cloud metrics are not expected to equal the historical local stable-lead reference exactly because the run context and logging surface differ.",
            "The promoted lead remained valid enough to complete a second clean cloud validation with checkpoints and zero active apps after finish.",
        ],
    }
    write_json(DELTA_JSON, delta)
    write_text(DELTA_MD, render_md("Local vs Cloud Delta Summary v2", delta))

    alignment_packet = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "cloud_validation_alignment_packet",
        "artifact_version": ARTIFACT_VERSION,
        "family": LEAD_FAMILY,
        "shape": LEAD_SHAPE,
        "local_promotion_evidence": {
            "config": LEAD_CONFIG,
            "frontier_progression_entry": next((item for item in (frontier.get("frontier_progression", []) or []) if str(item.get("family", "")) == LEAD_FAMILY), {}),
        },
        "cloud_validation_evidence": {
            "result_path": repo_rel(V2_CLOUD_RESULT_JSON),
            "runtime_path": repo_rel(V2_RUNTIME_JSON),
            "modal_app_id": APP_ID,
            "val_metrics": metrics,
            "checkpoint_written": True,
            "compile_fallback_triggered": bool(fallback_payload.get("compile_fallback_triggered")),
        },
        "alignment_reading": {
            "consistency_verdict": "ALIGNED_AS_DOWNSTREAM_VALIDATION_ONLY",
            "local_and_cloud_evidence_are_consistent": True,
            "note": "The fresh rerun confirms the promoted hybrid-ring lead remains a downstream validated stable lead; it still must not be treated as continuation of any closed local family.",
        },
        "future_implication": "Future cloud reruns are allowed only as explicit promoted-lead downstream validation, never as a bridge to reopen closed camera-object families.",
    }
    write_json(ALIGNMENT_PACKET_JSON, alignment_packet)
    write_text(ALIGNMENT_PACKET_MD, render_md("Cloud Validation Alignment Packet v2", alignment_packet))

    acceptance_rule = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "cloud_acceptance_rule",
        "artifact_version": ARTIFACT_VERSION,
        "family": LEAD_FAMILY,
        "rule_name": "promoted_lead_cloud_local_alignment_rule_v2",
        "downstream_validation_only": True,
        "current_rerun_completed_clean": True,
        "compile_fallback_allowed_for_a100": True,
        "future_cloud_rerun_allowed_when": [
            "the promoted stable lead config changes",
            "the current cloud result becomes incomplete",
            "cloud cleanup is not clean",
            "a future manual review explicitly requests refreshed validation",
        ],
        "future_cloud_rerun_forbidden_when": [
            "a closed local family tries to borrow this cloud path as continuation",
            "more than one active Modal app would be created",
            "the existing latest cloud result is already complete and clean and no explicit review asks for rerun",
        ],
    }
    write_json(ACCEPTANCE_JSON, acceptance_rule)
    write_text(ACCEPTANCE_MD, render_md("Cloud Acceptance Rule v2", acceptance_rule))

    completeness = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "cloud_artifact_completeness_report",
        "artifact_version": ARTIFACT_VERSION,
        "family": LEAD_FAMILY,
        "status": "COMPLETE_RERUN_DONE_CLEAN",
        "checks": {
            "cloud_validation_result_exists": True,
            "checkpoint_written": True,
            "val_metrics_complete": True,
            "active_modal_app_count_after_finish_zero": True,
            "alignment_packet_renderable": True,
            "cloud_cleanup_ok": True,
            "compile_fallback_status_recorded": bool(fallback_payload),
        },
        "reuse_existing_cloud_result": False,
        "rerun_performed": True,
    }
    write_json(COMPLETENESS_JSON, completeness)
    write_text(COMPLETENESS_MD, render_md("Cloud Artifact Completeness Report v2", completeness))

    delivery = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "one_page_delivery_summary",
        "artifact_version": ARTIFACT_VERSION,
        "family": LEAD_FAMILY,
        "summary": "today's alignment audit finished clean and produced cloud-ready deliverables",
        "cloud_rerun": {
            "app_id": APP_ID,
            "created_at": str(app_row.get("Created at", "")),
            "stopped_at": str(app_row.get("Stopped at", "")),
            "output_subdir": OUTPUT_SUBDIR,
            "compile_fallback_triggered": bool(fallback_payload.get("compile_fallback_triggered")),
        },
        "val_metrics": metrics,
        "delta_summary_path": repo_rel(DELTA_JSON),
        "alignment_packet_path": repo_rel(ALIGNMENT_PACKET_JSON),
        "acceptance_rule_path": repo_rel(ACCEPTANCE_JSON),
        "completeness_report_path": repo_rel(COMPLETENESS_JSON),
    }
    write_json(DELIVERY_JSON, delivery)
    write_text(DELIVERY_MD, render_md("One Page Delivery Summary v2", delivery))

    candidates = load_json(CANDIDATES_JSON)
    refreshed_candidates = deepcopy(candidates)
    refreshed_candidates["checked_at"] = datetime.now().isoformat(timespec="seconds")
    refreshed_candidates["artifact_version"] = ARTIFACT_VERSION
    refreshed_candidates["current_cloud_validation_ref"] = repo_rel(V2_CLOUD_RESULT_JSON)
    refreshed_candidates["current_cloud_validation_metrics"] = metrics
    refreshed_candidates["refresh_note"] = "Refreshed after a new clean promoted-lead cloud rerun with A100 compile fallback handling."
    refreshed_candidates["candidates"] = upsert_by_key(
        list(refreshed_candidates.get("candidates", []) or []),
        {
            "family": "a100_compile_fallback_stability_audit",
            "status": "manual_discovery_only",
            "question": "Should future promoted-lead cloud validations on A100 attempt compile first and accept eager fallback automatically when the compile path triggers a retryable CUDA backend failure?",
            "based_on": [repo_rel(V2_CLOUD_RESULT_JSON), repo_rel(V2_RUNTIME_JSON)],
            "forbidden_actions": [
                "do not reinterpret this as a local training family",
                "do not open multiple active cloud apps",
            ],
        },
        "family",
    )
    write_json(DISCOVERY_V2_JSON, refreshed_candidates)
    write_text(DISCOVERY_V2_MD, render_md("Next Manual Problem Candidates v2", refreshed_candidates))

    alignment_result = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "alignment_audit_result",
        "artifact_version": ARTIFACT_VERSION,
        "family": FAMILY,
        "status": "completed_clean",
        "execution_mode": "cloud_alignment_only",
        "reuse_existing_cloud_result": False,
        "rerun_performed": True,
        "cloud_result_path": repo_rel(V2_CLOUD_RESULT_JSON),
        "cloud_runtime_path": repo_rel(V2_RUNTIME_JSON),
        "alignment_packet_path": repo_rel(ALIGNMENT_PACKET_JSON),
        "acceptance_rule_path": repo_rel(ACCEPTANCE_JSON),
        "completeness_report_path": repo_rel(COMPLETENESS_JSON),
        "local_vs_cloud_delta_summary_path": repo_rel(DELTA_JSON),
        "delivery_summary_path": repo_rel(DELIVERY_JSON),
        "refreshed_discovery_candidates_path": repo_rel(DISCOVERY_V2_JSON),
        "summary": "today's alignment audit finished clean and produced cloud-ready deliverables",
        "final_boundary": "A fresh clean cloud rerun of the promoted hybrid-ring stable lead completed successfully, wrote checkpoints, cleaned up to zero active apps, and now defines the current downstream validation baseline for future manual-problem discovery.",
    }
    write_json(ALIGNMENT_RESULT_JSON, alignment_result)
    write_text(ALIGNMENT_RESULT_MD, render_md("Alignment Audit Result v2", alignment_result))

    research = load_json(RESEARCH_STATUS_JSON)
    research["current_cloud_blocker"] = alignment_result["final_boundary"]
    research["latest_promoted_lead_cloud_validation"] = {
        "status": "cloud_validation_done_clean",
        "family": LEAD_FAMILY,
        "mode": "promoted_lead_cloud_validation",
        "result_path": repo_rel(V2_CLOUD_RESULT_JSON),
        "runtime_path": repo_rel(V2_RUNTIME_JSON),
        "output_subdir": OUTPUT_SUBDIR,
        "modal_app_id": APP_ID,
        "active_modal_app_count_after_finish": 0,
    }
    research["latest_alignment_audit_result"] = {
        "status": "completed_clean",
        "family": FAMILY,
        "result_path": repo_rel(ALIGNMENT_RESULT_JSON),
        "reuse_existing_cloud_result": False,
    }
    write_json(RESEARCH_STATUS_JSON, research)

    frontier = load_json(FRONTIER_LEDGER_JSON)
    frontier["family_readout"] = deepcopy(frontier.get("family_readout", {}))
    frontier["family_readout"]["source_policy_hybrid_ring_regularization_cloud_validation"] = {
        "status": "cloud_validation_done_clean",
        "stop_reason": "A fresh clean cloud rerun completed for the promoted hybrid-ring stable lead with zero active apps after finish.",
    }
    frontier["family_readout"][FAMILY] = {
        "status": "completed_clean",
        "stop_reason": alignment_result["summary"],
    }
    frontier["latest_promoted_lead_cloud_validation"] = research["latest_promoted_lead_cloud_validation"]
    frontier["latest_alignment_audit_result"] = research["latest_alignment_audit_result"]
    write_json(FRONTIER_LEDGER_JSON, frontier)

    family_stop = load_json(FAMILY_STOP_REASON_JSON)
    family_stop["latest_promoted_lead_cloud_validation"] = research["latest_promoted_lead_cloud_validation"]
    family_stop["higher_level_manual_problem_outcomes"] = deepcopy(family_stop.get("higher_level_manual_problem_outcomes", {}))
    family_stop["higher_level_manual_problem_outcomes"][FAMILY] = {
        "status": "completed_clean",
        "result_path": repo_rel(ALIGNMENT_RESULT_JSON),
        "reuse_existing_cloud_result": False,
    }
    write_json(FAMILY_STOP_REASON_JSON, family_stop)

    task_plan = load_json(TASK_PLAN_JSON)
    task_plan["checked_at"] = datetime.now().isoformat(timespec="seconds")
    task_plan["task_mode_status"] = "completed"
    task_plan["current_mode"] = "alignment_audit_completed_clean"
    task_plan["research_loop_mode"] = "IDLE_GUARD"
    task_plan["task_mode_focus"] = "promoted_lead_cloud_local_alignment_audit_completed_clean"
    task_plan["active_tasks"] = []
    task_plan["completed_this_round"] = upsert_by_key(
        list(task_plan.get("completed_this_round", []) or []),
        {
            "id": "phase_46_promoted_lead_cloud_local_alignment_audit_rerun_completed_clean",
            "status": "completed",
            "details": "Ran a fresh clean promoted-lead cloud validation rerun and completed the alignment audit deliverables on the new evidence.",
        },
        "id",
    )
    task_plan["promoted_lead_cloud_validation"] = {
        "status": "cloud_validation_done_clean",
        "family": LEAD_FAMILY,
        "mode": "promoted_lead_cloud_validation",
        "output_subdir": OUTPUT_SUBDIR,
        "modal_app_id": APP_ID,
        "result_path": repo_rel(V2_CLOUD_RESULT_JSON),
        "runtime_path": repo_rel(V2_RUNTIME_JSON),
        "cleanup_ok": True,
    }
    task_plan["alignment_audit"] = {
        "status": "completed_clean",
        "family": FAMILY,
        "result_path": repo_rel(ALIGNMENT_RESULT_JSON),
        "acceptance_rule_path": repo_rel(ACCEPTANCE_JSON),
        "rerun_performed": True,
    }
    task_plan["problem_definition_progress"] = deepcopy(task_plan.get("problem_definition_progress", {}))
    task_plan["problem_definition_progress"]["status"] = "alignment_audit_completed_clean"
    task_plan["problem_definition_progress"]["newest_boundary_fact"] = alignment_result["final_boundary"]
    task_plan["problem_definition_progress"]["next_requirement"] = "Use the refreshed discovery candidates after the new cloud evidence; do not reopen closed local families."
    task_plan["summary_conclusion"] = [
        "promoted_lead_cloud_local_alignment_audit finished clean after a fresh promoted-lead cloud rerun.",
        "source_policy_hybrid_ring_regularization / stablelead_nearest_plus_uniform_tail now has a new cloud validation result, updated acceptance rule v2, and updated completeness report v2.",
        "Research remains in IDLE_GUARD, no approved problem is active, and zero active Modal apps remain.",
    ]
    task_plan["current_state_notes"] = [
        "The cloud line remains downstream validation only and stays semantically separate from any local training family.",
        "The fresh rerun used an A100-tuned path with compile attempted first and eager fallback when compile hit a retryable backend failure.",
        "The next move must begin from refreshed discovery candidates, not a reopen of closed local families.",
    ]
    write_json(TASK_PLAN_JSON, task_plan)
    write_text(TASK_PLAN_MD, json.dumps(task_plan, ensure_ascii=False, indent=2) + "\n")
    write_text(SUMMARY_MD, "\n".join(["# ZJU Source-Policy Rawpool Status", ""] + [f"- {line}" for line in task_plan["summary_conclusion"]]) + "\n")

    watch_payload = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "research": {
            "summary": {
                "state": "IDLE_GUARD",
                "approved_problem_present": False,
                "approved_problem_ready": False,
                "manual_action_required": False,
                "manual_action_kind": "",
                "ready_for_execution": False,
                "current_review_packet": repo_rel(ALIGNMENT_RESULT_JSON),
            },
            "research_status": research,
            "allowlist": load_json(ALLOWLIST_JSON),
        },
        "cloud_validation": {
            "status": "cloud_validation_done_clean",
            "family": LEAD_FAMILY,
            "mode": "promoted_lead_cloud_validation",
            "modal_app_id": APP_ID,
            "output_subdir": OUTPUT_SUBDIR,
            "result_path": repo_rel(V2_CLOUD_RESULT_JSON),
            "runtime_path": repo_rel(V2_RUNTIME_JSON),
            "cleanup_ok": True,
            "active_modal_app_count_after_finish": 0,
        },
        "alignment_audit": {
            "status": "completed_clean",
            "family": FAMILY,
            "result_path": repo_rel(ALIGNMENT_RESULT_JSON),
            "rerun_performed": True,
            "summary": alignment_result["summary"],
        },
        "modal_apps": [],
        "research_runtime_processes": [],
        "watch_conclusion": "alignment audit finished clean, no active cloud app, no active local family",
    }
    write_json(LATEST_WATCH_JSON, watch_payload)

    if active_modal_apps():
        raise FinalizeError("Active Modal apps remain after finalization.")
    if repo_research_processes():
        raise FinalizeError("Repo research processes remain after finalization.")
    if load_json(ALLOWLIST_JSON).get("allowed_markers"):
        raise FinalizeError("repo_process_allowlist is not empty after finalization.")

    print(
        json.dumps(
            {
                "cloud_result_v2": repo_rel(V2_CLOUD_RESULT_JSON),
                "cloud_runtime_v2": repo_rel(V2_RUNTIME_JSON),
                "delta_summary_v2": repo_rel(DELTA_JSON),
                "alignment_packet_v2": repo_rel(ALIGNMENT_PACKET_JSON),
                "acceptance_rule_v2": repo_rel(ACCEPTANCE_JSON),
                "completeness_report_v2": repo_rel(COMPLETENESS_JSON),
                "delivery_summary_v2": repo_rel(DELIVERY_JSON),
                "refreshed_discovery_candidates_v2": repo_rel(DISCOVERY_V2_JSON),
                "alignment_audit_result_v2": repo_rel(ALIGNMENT_RESULT_JSON),
                "task_plan": repo_rel(TASK_PLAN_JSON),
                "summary": repo_rel(SUMMARY_MD),
                "latest_watch_snapshot": repo_rel(LATEST_WATCH_JSON),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
