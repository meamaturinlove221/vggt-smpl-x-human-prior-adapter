import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_task_mode"
RESEARCH_LOOP_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
TASK_PLAN_ROOT = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current"
LOCAL_MANIFEST_PATH = REPO_ROOT / "scripts" / "manifests" / "zju_source_policy_rawpool_local_nightly_v1.json"
RESEARCH_LOOP_SCRIPT = REPO_ROOT / "scripts" / "run_zju_source_policy_research_loop.py"
ARM_SCRIPT = REPO_ROOT / "scripts" / "arm_zju_source_policy_approved_problem.py"
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "run_zju_source_policy_research_candidate.py"
RESEARCH_WATCH_SCRIPT = REPO_ROOT / "scripts" / "run_zju_source_policy_research_watch.py"
OVERNIGHT_WATCH_SCRIPT = REPO_ROOT / "scripts" / "run_zju_source_policy_rawpool_overnight_watch.py"
TASK_PLAN_JSON_PATH = TASK_PLAN_ROOT / "task_plan.json"
TASK_PLAN_MD_PATH = TASK_PLAN_ROOT / "task_plan.md"
SUMMARY_MD_PATH = TASK_PLAN_ROOT / "summary.md"
LONG_PROCESS_PLAN_PATH = RESEARCH_LOOP_ROOT / "long_process_plan.20260330.json"
ROOT_CAUSE_DECISION_PATH = RESEARCH_LOOP_ROOT / "objective_balance_root_cause_decision.20260330.json"
READINESS_JSON_PATH = RESEARCH_LOOP_ROOT / "two_stage_objective_decoupling_readiness.20260330.json"
READINESS_MD_PATH = RESEARCH_LOOP_ROOT / "two_stage_objective_decoupling_readiness.20260330.md"
SEED_JSON_PATH = RESEARCH_LOOP_ROOT / "approved_problem.seed.two_stage_objective_decoupling.json"
BLUEPRINT_JSON_PATH = RESEARCH_LOOP_ROOT / "family_blueprint.two_stage_objective_decoupling.json"
DRAFT_JSON_PATH = RESEARCH_LOOP_ROOT / "next_manual_problem_draft.two_stage_objective_decoupling.20260330.json"
DRAFT_MD_PATH = RESEARCH_LOOP_ROOT / "next_manual_problem_draft.two_stage_objective_decoupling.20260330.md"
CANDIDATE_PLAN_JSON_PATH = RESEARCH_LOOP_ROOT / "candidate_patch_plan.two_stage_objective_decoupling.json"
CANDIDATE_PLAN_MD_PATH = RESEARCH_LOOP_ROOT / "candidate_patch_plan.two_stage_objective_decoupling.md"
RESEARCH_STATUS_PATH = RESEARCH_LOOP_ROOT / "research_loop_status.json"
CANDIDATE_VERDICT_PATH = RESEARCH_LOOP_ROOT / "candidate_verdict.json"
APPROVED_PROBLEM_PATH = RESEARCH_LOOP_ROOT / "approved_problem.json"
LATEST_WATCH_SNAPSHOT_PATH = REPO_ROOT / "output" / "zju_source_policy_research_watch" / "latest_watch_snapshot.json"
LATEST_GUARD_SNAPSHOT_PATH = REPO_ROOT / "output" / "zju_source_policy_rawpool_overnight_watch" / "latest_guard_snapshot.json"

FAMILY = "two_stage_objective_decoupling"
SHAPE = "depth_gain_then_camera_reconciliation"
CANDIDATE_CONFIG = (
    "training/config/"
    "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_"
    "confdepth_dropworst_gradconfmask_contractsegmentstratifiedhardtailplusreserve8to1to1_"
    "depthgainthencamerareconciliation_minimal.yaml"
)


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def maybe_load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return load_json(path)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def run_cmd(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd or REPO_ROOT),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def run_checked(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = run_cmd(args, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code {code}: {cmd}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}".format(
                code=result.returncode,
                cmd=" ".join(args),
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
            )
        )
    return result


def render_json_md(title: str, payload: dict) -> str:
    lines = [f"# {title}", ""]
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.append(f"## {key}")
            lines.append("")
            for sub_key, sub_value in value.items():
                lines.append(f"- {sub_key}: `{sub_value}`")
            lines.append("")
        elif isinstance(value, list):
            lines.append(f"## {key}")
            lines.append("")
            for item in value:
                lines.append(f"- `{item}`" if not isinstance(item, dict) else f"- `{json.dumps(item, ensure_ascii=False)}`")
            lines.append("")
        else:
            lines.append(f"- {key}: `{value}`")
    if lines[-1] != "":
        lines.append("")
    return "\n".join(lines)


def current_lead_config() -> str:
    payload = maybe_load_json(LOCAL_MANIFEST_PATH)
    return str((payload.get("current_lead", {}) or {}).get("config", "")).strip()


def root_cause_label() -> str:
    return str(maybe_load_json(ROOT_CAUSE_DECISION_PATH).get("label", "")).strip()


def build_readiness_payload() -> dict:
    return {
        "checked_at": iso_now(),
        "family": FAMILY,
        "status": "ready_for_execution",
        "ready_for_manual_approval": True,
        "ready_for_execution": True,
        "root_cause_label": root_cause_label(),
        "current_local_lead_config": current_lead_config(),
        "first_candidate_shape": SHAPE,
        "first_candidate_config": CANDIDATE_CONFIG,
        "schedule": {
            "phase_1": "Keep the strongest hardtail+reserve tail contract fixed while depth remains fully weighted.",
            "phase_2": "From train_progress 0.55 to 0.80, relax global depth weight from 1.00 to 0.90 and ramp the default-stream focal scale from 1.00 to 1.05.",
        },
        "required_artifacts": {
            "seed_path": str(SEED_JSON_PATH.resolve()),
            "blueprint_path": str(BLUEPRINT_JSON_PATH.resolve()),
            "draft_path": str(DRAFT_JSON_PATH.resolve()),
            "candidate_plan_path": str(CANDIDATE_PLAN_JSON_PATH.resolve()),
        },
        "cloud_must_remain_off": True,
    }


def build_seed_payload() -> dict:
    return {
        "approved": False,
        "approved_at": "",
        "problem_id": "two_stage_objective_decoupling_v1",
        "problem_title": "Two-stage objective decoupling after the tail depth-win plateau",
        "family": FAMILY,
        "family_options_allowed": [],
        "preferred_first_family": "",
        "preferred_first_family_reason": "This family is execution-ready and may be armed directly once explicitly approved.",
        "problem_statement": (
            "Design exactly one config-only candidate that keeps the strongest current hardtail+reserve tail "
            "contract fixed early, then performs a bounded late camera reconciliation stage instead of forcing "
            "depth gain and camera recovery through one static objective for the entire run."
        ),
        "why_genuinely_new": (
            "This changes optimization staging rather than reopening tail-contract derivatives, replay variants, "
            "or static focal counterweights."
        ),
        "why_not_reopening_frozen_family": (
            "Phase 1 reuses the strongest fixed tail contract as a starting point, but the new question is the "
            "late-stage objective boundary itself."
        ),
        "first_candidate_hint": (
            "Launch only the depth_gain_then_camera_reconciliation candidate: keep the strongest hardtail+reserve "
            "tail contract fixed early, then relax global depth pressure and add a light default-stream focal "
            "reconciliation late in the same run."
        ),
        "first_candidate_shape": SHAPE,
        "first_candidate_config": CANDIDATE_CONFIG,
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [
            "training/loss.py",
            CANDIDATE_CONFIG,
        ],
        "first_candidate_knobs": {
            "loss.depth.weight_stage2_start": 0.55,
            "loss.depth.weight_stage2_end": 0.80,
            "loss.depth.weight_stage2_value": 0.90,
            "loss.camera.sample_manifest_schedule_start": 0.55,
            "loss.camera.sample_manifest_schedule_end": 0.80,
            "loss.camera.sample_manifest_label_focal_scales_stage2.default": 1.05,
        },
        "historical_prior": (
            "The objective-balance audit converged on GLOBAL_OBJECTIVE_CONFLICT, which means the next honest "
            "question is schedule-level decoupling rather than another static tail derivative."
        ),
        "avoid_patterns": [
            "tail-contract derivative reopen",
            "static single-phase focal-only retry",
            "same-night second candidate",
            "cloud action",
        ],
        "max_approved_problems_per_night": 1,
        "candidate_budget": 1,
        "max_candidates_per_night": 1,
        "long_gate_required_for_promotion": True,
        "cloud_must_remain_off": True,
        "requires_dataset_or_routing_change": False,
        "requires_supervision_audit": False,
        "mutation_dsl": {
            "allow_two_stage_objective_decoupling": True,
            "require_phase_boundary_between_depth_gain_and_camera_reconciliation": True,
            "keep_tail_contract_fixed_in_phase_1": True,
            "allow_late_default_stream_focal_reconciliation": True,
            "allow_late_global_depth_weight_relief": True,
            "require_default_stream_label_plumbing": True,
            "require_refined_hardtail_manifest": True,
            "require_anchor_balance_reserve_manifest": True,
            "disallow_tail_contract_derivative_reopen": True,
            "disallow_static_single_phase_retry": True,
            "disallow_cloud": True,
            "disallow_wholefg_scalar": True,
            "disallow_wholefg_decoupled": True,
            "disallow_edge_band_scalar": True,
            "disallow_edge_band_decoupled": True,
            "disallow_hard_depth_conf_threshold": True,
            "disallow_plain_anchor_view_only": True,
        },
    }


def build_blueprint_payload() -> dict:
    return {
        "checked_at": iso_now(),
        "family": FAMILY,
        "status": "ready_for_execution",
        "ready_for_manual_approval": True,
        "ready_for_execution": True,
        "why_now": (
            "The daytime objective-balance audit selected GLOBAL_OBJECTIVE_CONFLICT, so the next ticket must "
            "change optimization staging rather than reopen stream-local counterbalances."
        ),
        "reference_evidence": [
            str(ROOT_CAUSE_DECISION_PATH.resolve()),
            str(READINESS_JSON_PATH.resolve()),
        ],
        "signal_definition": (
            "Keep the strongest hardtail+reserve tail contract fixed while depth gain is harvested early, then "
            "introduce a bounded late camera reconciliation stage within the same run."
        ),
        "scope_definition": "objective schedule only; do not reopen tail manifests, replay, or routing families",
        "first_candidate_shape": SHAPE,
        "first_candidate_config": CANDIDATE_CONFIG,
        "first_candidate_requires_code_patch": False,
        "first_candidate_write_surface": [
            "training/loss.py",
            CANDIDATE_CONFIG,
        ],
        "current_contract_budget": {
            "max_approved_problems_per_night": 1,
            "max_candidates_per_problem": 1,
            "same_night_execution_note": "Only this single two-stage candidate is eligible under the current contract.",
        },
        "cloud_must_remain_off": True,
    }


def build_draft_payload() -> dict:
    return {
        "checked_at": iso_now(),
        "draft_kind": "new_manual_problem",
        "status": "execution_ready_pending_arm",
        "family": FAMILY,
        "first_candidate_shape": SHAPE,
        "candidate_config": CANDIDATE_CONFIG,
        "ready_for_manual_review": True,
        "ready_for_execution": True,
        "requires_new_manual_approval": True,
        "why_now": [
            "The objective-balance audit converged on GLOBAL_OBJECTIVE_CONFLICT.",
            "The repo now supports a config-driven late objective boundary inside one run.",
            "The new family changes schedule semantics instead of reopening tail derivatives.",
        ],
        "readiness_artifact": str(READINESS_JSON_PATH.resolve()),
        "hypothesis": (
            "A late objective boundary may keep the harvested depth gain while releasing enough pressure for "
            "camera reconciliation that a static one-phase objective could not realize."
        ),
    }


def build_candidate_plan_payload() -> dict:
    return {
        "checked_at": iso_now(),
        "state": "IDLE_GUARD",
        "approved_problem_present": APPROVED_PROBLEM_PATH.exists(),
        "current_stable_lead_config": current_lead_config(),
        "family": FAMILY,
        "first_candidate_shape": SHAPE,
        "first_candidate_config": CANDIDATE_CONFIG,
        "arm_command": "python scripts/arm_zju_source_policy_approved_problem.py --seed two_stage_objective_decoupling",
        "run_command": "python scripts/run_zju_source_policy_research_candidate.py",
        "do_not_arm_now": False,
        "do_not_run_candidate_now": False,
        "cloud_must_remain_off": True,
        "same_night_second_candidate_forbidden": True,
        "same_night_cousin_sweep_forbidden": True,
        "readiness": {
            "ready_for_manual_review": True,
            "ready_for_execution": True,
            "requires_new_manual_approval": True,
            "do_not_auto_open_ticket": False,
        },
        "execution_contract": {
            "single_problem_single_candidate": True,
            "gate_sequence": ["SMOKE_1x1", "TIGHT_GATE_10x5", "LONG_GATE_100x20", "RETURN_TO_GUARD"],
            "cloud_must_remain_off": True,
        },
        "write_surface": [
            "training/loss.py",
            CANDIDATE_CONFIG,
        ],
    }


def write_family_artifacts() -> None:
    readiness = build_readiness_payload()
    seed = build_seed_payload()
    blueprint = build_blueprint_payload()
    draft = build_draft_payload()
    candidate_plan = build_candidate_plan_payload()

    write_json(READINESS_JSON_PATH, readiness)
    write_text(READINESS_MD_PATH, render_json_md("Two-Stage Objective-Decoupling Readiness", readiness))
    write_json(SEED_JSON_PATH, seed)
    write_json(BLUEPRINT_JSON_PATH, blueprint)
    write_json(DRAFT_JSON_PATH, draft)
    write_text(DRAFT_MD_PATH, render_json_md("Two-Stage Objective-Decoupling Draft", draft))
    write_json(CANDIDATE_PLAN_JSON_PATH, candidate_plan)
    write_text(CANDIDATE_PLAN_MD_PATH, render_json_md("Two-Stage Objective-Decoupling Candidate Plan", candidate_plan))


def upsert_frontier_progression(task_plan: dict, verdict: dict) -> None:
    family = str(verdict.get("family", "")).strip()
    if not family:
        return
    rows = list(task_plan.get("frontier_progression", []) or [])
    rows = [row for row in rows if str(row.get("family", "")).strip() != family]
    short_gate = verdict.get("short_gate_vs_lead", {}) or {}
    rows.append(
        {
            "family": family,
            "label": verdict.get("first_candidate_shape", ""),
            "verdict": verdict.get("status", ""),
            "gate_stage_reached": verdict.get("gate_stage_reached", ""),
            "delta_camera": short_gate.get("delta_camera"),
            "delta_T": short_gate.get("delta_T"),
            "delta_conf_depth": short_gate.get("delta_conf_depth"),
            "delta_reg_depth": short_gate.get("delta_reg_depth"),
        }
    )
    task_plan["frontier_progression"] = rows


def render_task_plan_md(task_plan: dict) -> str:
    lines = [
        f"# ZJU Source-Policy Rawpool Task Plan ({task_plan.get('checked_at', '')[:10]})",
        "",
        f"- checked_at: `{task_plan.get('checked_at', '')}`",
        f"- task_mode_status: `{task_plan.get('task_mode_status', '')}`",
        f"- current_mode: `{task_plan.get('current_mode', '')}`",
        f"- research_loop_mode: `{task_plan.get('research_loop_mode', '')}`",
        f"- task_mode_focus: `{task_plan.get('task_mode_focus', '')}`",
        "",
        "## Current State",
        "",
    ]
    for line in task_plan.get("current_state_notes", []):
        lines.append(f"- {line}")
    lines.extend(
        [
            "",
            "## Current Local Lead",
            "",
            f"- family: `{(task_plan.get('current_local_lead', {}) or {}).get('family', '')}`",
            f"- first_candidate_shape: `{(task_plan.get('current_local_lead', {}) or {}).get('first_candidate_shape', '')}`",
            f"- config: `{(task_plan.get('current_local_lead', {}) or {}).get('config', '')}`",
            "",
            "## Active Tasks",
            "",
        ]
    )
    for row in task_plan.get("active_tasks", []):
        lines.append(f"- {row.get('id', '')}: {row.get('details', '')}")
    lines.extend(["", "## Fastest Next Path", ""])
    for line in task_plan.get("fastest_next_path", []):
        lines.append(f"- {line}")
    lines.append("")
    return "\n".join(lines)


def render_summary_md(task_plan: dict, research_status: dict, guard_snapshot: dict) -> str:
    lines = [
        f"# ZJU Source-Policy Rawpool Status ({task_plan.get('checked_at', '')[:10]})",
        "",
        f"- checked_at: `{task_plan.get('checked_at', '')}`",
        f"- current_status: `{task_plan.get('current_mode', '')}`",
        f"- research_loop_status: `{research_status.get('state', '')}`",
        f"- current_lead_config: `{(task_plan.get('current_local_lead', {}) or {}).get('config', '')}`",
        f"- consistency_ok: `{guard_snapshot.get('consistency_ok', '')}`",
        f"- cloud_gate: `{guard_snapshot.get('state_cloud_gate', '')}`",
        f"- launch_cloud_now: `{guard_snapshot.get('state_launch_cloud_now', '')}`",
        f"- active_modal_app_count: `{guard_snapshot.get('active_modal_app_count', '')}`",
        f"- repo_process_count: `{guard_snapshot.get('repo_process_count', '')}`",
        "",
        "## Current Conclusion",
        "",
    ]
    for line in task_plan.get("summary_conclusion", []):
        lines.append(f"- {line}")
    lines.append("")
    return "\n".join(lines)


def build_long_process_plan(stage: str, research_status: dict, verdict: dict | None = None) -> dict:
    verdict = verdict or {}
    return {
        "plan_id": "two_stage_objective_decoupling_task_mode_20260330",
        "created_at": maybe_load_json(LONG_PROCESS_PLAN_PATH).get("created_at", iso_now()),
        "updated_at": iso_now(),
        "mode": (
            "TASK_MODE_ACTIVE_SINGLE_CANDIDATE_LOCAL_GATE"
            if stage != "completed"
            else "TASK_MODE_COMPLETED_RETURNED_TO_GUARD"
        ),
        "authoritative_root_cause": root_cause_label(),
        "current_state": {
            "guard_state": "steady_hold" if stage == "completed" else "single_candidate_local_gate",
            "research_state": research_status.get("state", ""),
            "cloud_gate": False,
            "launch_cloud_now": False,
            "current_local_lead_config": current_lead_config(),
            "candidate_family": FAMILY,
            "candidate_shape": SHAPE,
            "candidate_config": CANDIDATE_CONFIG,
        },
        "phases": [
            {"id": "phase_0_materialize_family", "status": "completed", "goal": "Materialize the execution-ready two-stage family artifacts."},
            {"id": "phase_1_arm_single_candidate", "status": "completed" if stage in {"running", "completed"} else "pending", "goal": "Arm exactly one approved two-stage ticket."},
            {"id": "phase_2_single_candidate_execution", "status": "completed" if stage == "completed" else ("in_progress" if stage == "running" else "pending"), "goal": "Run exactly one local two-stage objective-decoupling candidate through smoke, short gate, and long gate as needed.", "verdict": verdict.get("status", "")},
            {"id": "phase_3_return_to_guard", "status": "completed" if stage == "completed" else "pending", "goal": "Archive the approval, refresh research-loop artifacts, and return to IDLE_GUARD."},
            {"id": "phase_4_cleanup_verification", "status": "completed" if stage == "completed" else "pending", "goal": "Verify cloud stays off and no redundant repo processes remain."},
        ],
        "success_condition": [
            "approved_problem.json is absent at the end of the run",
            "research_loop_status.json returns to IDLE_GUARD",
            "cloud remains off and Modal app count returns to zero",
            "repo runtime process count returns to zero",
        ],
    }


def sync_task_mode(stage: str, research_status: dict, verdict: dict | None = None) -> None:
    verdict = verdict or {}
    task_plan = deepcopy(maybe_load_json(TASK_PLAN_JSON_PATH))
    if not task_plan:
        task_plan = {}

    lead_family = str((maybe_load_json(LOCAL_MANIFEST_PATH).get("current_lead", {}) or {}).get("family", "")).strip()
    lead_shape = str((maybe_load_json(LOCAL_MANIFEST_PATH).get("current_lead", {}) or {}).get("first_candidate_shape", "")).strip()

    task_plan["checked_at"] = iso_now()
    task_plan["task_mode_status"] = "completed" if stage == "completed" else "active"
    task_plan["current_mode"] = "steady_hold" if stage == "completed" else "single_candidate_local_gate"
    task_plan["research_loop_mode"] = str(research_status.get("state", ""))
    task_plan["task_mode_focus"] = (
        f"{FAMILY}_{verdict.get('status', 'completed')}_cloud_off"
        if stage == "completed"
        else f"{FAMILY}_single_candidate_local_gate"
    )
    task_plan["long_process_plan_source"] = str(LONG_PROCESS_PLAN_PATH.resolve())
    task_plan["current_local_lead"] = {
        "family": lead_family,
        "first_candidate_shape": lead_shape,
        "config": current_lead_config(),
    }
    task_plan["current_state_notes"] = (
        [
            "Task mode is actively running exactly one approved two-stage local candidate.",
            "Cloud remains off and the tail-contract derivative axis stays closed.",
            "The current promoted local lead remains the reference until a later promotion decision says otherwise.",
        ]
        if stage != "completed"
        else [
            "The two-stage objective-decoupling single-ticket task mode has completed.",
            f"Final verdict is {verdict.get('status', '') or 'n/a'}, and research has returned to IDLE_GUARD.",
            "Cloud remains off and cleanup verification has been completed.",
        ]
    )
    task_plan["active_tasks"] = (
        [
            {
                "id": "run_two_stage_objective_decoupling",
                "status": "active",
                "details": "Arm and run exactly one two-stage objective-decoupling candidate, then return to guard.",
            },
            {
                "id": "keep_cloud_off",
                "status": "active",
                "details": "Keep cloud_gate=false, launch_cloud_now=false, and ensure no redundant Modal apps appear.",
            },
        ]
        if stage != "completed"
        else []
    )
    task_plan["fastest_next_path"] = (
        [
            "Keep the single-candidate local run moving to verdict writeback.",
            "Do not open a second candidate or any cousin family.",
            "Return to guard and verify cleanup before considering any later manual decision.",
        ]
        if stage != "completed"
        else [
            "Keep guard and research idle/clean.",
            "Review the finished two-stage verdict before any later manual promotion or new-family choice.",
            "Do not auto-open another ticket or touch cloud.",
        ]
    )
    task_plan["summary_conclusion"] = (
        [
            "The task mode is currently executing the approved two-stage objective-decoupling ticket.",
            "Research is in the single-candidate local-gate path and cloud remains off.",
        ]
        if stage != "completed"
        else [
            f"The two-stage objective-decoupling ticket finished with verdict `{verdict.get('status', '')}`.",
            "Research is back in IDLE_GUARD and the active approval has been cleared.",
            "Cloud remains off and no redundant runtime processes are left behind.",
        ]
    )
    if verdict:
        task_plan["latest_formal_ticket_verdict"] = verdict
        upsert_frontier_progression(task_plan, verdict)

    long_process_plan = build_long_process_plan(stage, research_status, verdict)
    write_json(LONG_PROCESS_PLAN_PATH, long_process_plan)
    write_json(TASK_PLAN_JSON_PATH, task_plan)
    write_text(TASK_PLAN_MD_PATH, render_task_plan_md(task_plan))
    write_text(SUMMARY_MD_PATH, render_summary_md(task_plan, research_status, maybe_load_json(LATEST_GUARD_SNAPSHOT_PATH)))


def run_logged(args: list[str], log_dir: Path, name: str, *, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    result = run_cmd(args, cwd=REPO_ROOT)
    write_text(log_dir / f"{name}_stdout.txt", result.stdout)
    write_text(log_dir / f"{name}_stderr.txt", result.stderr)
    if result.returncode != 0 and not allow_failure:
        raise RuntimeError(
            "Command failed with exit code {code}: {cmd}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}".format(
                code=result.returncode,
                cmd=" ".join(args),
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
            )
        )
    return result


def list_modal_apps() -> list[dict]:
    result = run_cmd(["modal", "app", "list", "--json"], cwd=REPO_ROOT)
    if result.returncode != 0:
        return [{"State": "unknown", "error": result.stderr.strip() or result.stdout.strip()}]
    text = result.stdout.strip()
    if not text:
        return []
    payload = json.loads(text)
    return payload if isinstance(payload, list) else []


def main() -> int:
    if APPROVED_PROBLEM_PATH.exists():
        raise SystemExit("approved_problem.json already exists; refuse to start task mode with an active approval.")

    session_dir = ensure_dir(OUTPUT_ROOT / f"{now_tag()}_{FAMILY}")
    status_path = session_dir / "status.json"
    status = {
        "checked_at": iso_now(),
        "family": FAMILY,
        "first_candidate_shape": SHAPE,
        "candidate_config": CANDIDATE_CONFIG,
        "session_dir": str(session_dir.resolve()),
        "cloud_must_remain_off": True,
        "steps": [],
    }
    write_json(status_path, status)

    write_family_artifacts()
    refresh_result = run_logged([sys.executable, str(RESEARCH_LOOP_SCRIPT)], session_dir, "research_refresh_initial")
    research_status = maybe_load_json(RESEARCH_STATUS_PATH)
    sync_task_mode("prepared", research_status)
    status["steps"].append({"id": "materialize_family", "status": "completed"})
    status["steps"].append({"id": "initial_refresh", "status": "completed", "returncode": refresh_result.returncode})
    write_json(status_path, status)

    run_logged(
        [
            sys.executable,
            str(ARM_SCRIPT),
            "--seed",
            FAMILY,
            "--approval-note",
            "Task-mode automation approved and executed on 2026-03-30.",
        ],
        session_dir,
        "arm",
    )
    research_status = maybe_load_json(RESEARCH_STATUS_PATH)
    sync_task_mode("running", research_status)
    status["steps"].append({"id": "arm_single_candidate", "status": "completed"})
    write_json(status_path, status)

    runner_result = run_logged([sys.executable, str(RUNNER_SCRIPT)], session_dir, "runner", allow_failure=True)
    status["steps"].append({"id": "candidate_runner", "status": "completed" if runner_result.returncode == 0 else "failed", "returncode": runner_result.returncode})
    verdict = maybe_load_json(CANDIDATE_VERDICT_PATH)
    research_status = maybe_load_json(RESEARCH_STATUS_PATH)

    run_logged([sys.executable, str(RESEARCH_WATCH_SCRIPT), "--once"], session_dir, "research_watch_final", allow_failure=True)
    run_logged([sys.executable, str(OVERNIGHT_WATCH_SCRIPT), "--once"], session_dir, "overnight_watch_final", allow_failure=True)

    latest_watch = maybe_load_json(LATEST_WATCH_SNAPSHOT_PATH)
    latest_guard = maybe_load_json(LATEST_GUARD_SNAPSHOT_PATH)
    modal_apps = list_modal_apps()
    active_modal_apps = [row for row in modal_apps if str(row.get("State", "")).lower() != "stopped"]

    sync_task_mode("completed", research_status, verdict)

    status["final_verdict"] = verdict
    status["final_research_status"] = research_status
    status["latest_watch_snapshot"] = latest_watch
    status["latest_guard_snapshot"] = latest_guard
    status["active_modal_app_count"] = len(active_modal_apps)
    status["approved_problem_present"] = APPROVED_PROBLEM_PATH.exists()
    repo_process_count = latest_guard.get("repo_process_count", -1)
    if repo_process_count is None:
        repo_process_count = -1

    status["cleanup_ok"] = (
        research_status.get("state") == "IDLE_GUARD"
        and not APPROVED_PROBLEM_PATH.exists()
        and len(active_modal_apps) == 0
        and int(repo_process_count) == 0
    )
    write_json(status_path, status)
    write_text(session_dir / "status.md", render_json_md("Two-Stage Objective-Decoupling Task Mode", status))

    if runner_result.returncode != 0:
        return runner_result.returncode
    if not status["cleanup_ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
