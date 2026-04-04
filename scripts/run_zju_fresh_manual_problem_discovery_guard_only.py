import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
OUTPUT_ROOT = RESEARCH_ROOT / "fresh_manual_problem_discovery_guard_only"

RESEARCH_STATUS_JSON = RESEARCH_ROOT / "research_loop_status.json"
FRONTIER_LEDGER_JSON = RESEARCH_ROOT / "frontier_ledger.json"
FAMILY_STOP_REASON_JSON = RESEARCH_ROOT / "family_stop_reason.json"
TASK_PLAN_JSON = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current" / "task_plan.json"
ALLOWLIST_JSON = RESEARCH_ROOT / "repo_process_allowlist.json"
CLOUD_RESULT_JSON = RESEARCH_ROOT / "cloud_validation_result.source_policy_hybrid_ring_regularization.20260403.json"
CLOUD_RUNTIME_JSON = RESEARCH_ROOT / "cloud_runtime_state.20260403.json"
CAMERA_SUBOBJECTIVE_POSTMORTEM_JSON = RESEARCH_ROOT / "camera_subobjective_isolation_postmortem.20260402.json"
CAMERA_DEPTH_POSTMORTEM_JSON = RESEARCH_ROOT / "camera_depth_objective_coupling_audit_postmortem.20260403.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guard-only refresh for fresh manual problem discovery artifacts."
    )
    return parser.parse_args()


def now() -> datetime:
    return datetime.now()


def now_iso() -> str:
    return now().isoformat(timespec="seconds")


def date_tag() -> str:
    return now().strftime("%Y%m%d")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def maybe_load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return load_json(path)


def latest_existing_path(*patterns: str) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(REPO_ROOT.glob(pattern))
    if not matches:
        return None
    return sorted(path.resolve() for path in matches)[-1]


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def list_active_modal_apps() -> list[dict]:
    result = subprocess.run(
        ["modal", "app", "list", "--json"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"modal app list failed: {result.stderr.strip()}")
    payload = json.loads(result.stdout.strip() or "[]")
    return [row for row in payload if str(row.get("State", "")).lower() != "stopped"]


def discovery_boundary_ok(research: dict, allowlist: dict, active_modal_apps: list[dict], cloud_runtime: dict) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if research.get("state") != "IDLE_GUARD":
        issues.append("research_state_not_idle_guard")
    if bool(research.get("approved_problem_present")):
        issues.append("approved_problem_present")
    if bool(research.get("allowed_families", [])):
        issues.append("allowed_families_not_empty")
    if bool(research.get("ready_for_execution")):
        issues.append("ready_for_execution_true")
    if str(allowlist.get("status", "")).strip() != "idle_empty_allowlist":
        issues.append("allowlist_not_idle_empty")
    if list(allowlist.get("allowed_markers", []) or []):
        issues.append("allowlist_markers_not_empty")
    if active_modal_apps:
        issues.append("active_modal_apps_present")
    if not bool(cloud_runtime.get("cleanup_ok")):
        issues.append("cloud_cleanup_not_ok")
    return (len(issues) == 0, issues)


def build_closed_axis_summary(frontier: dict, family_stop: dict, camera_subobjective: dict, camera_depth_postmortem: dict) -> dict:
    axes = [
        {
            "axis": "camera_focal_objective_isolation",
            "status": str((camera_subobjective.get("focal_isolation", {}) or {}).get("status", "")),
            "gate_stage_reached": str((camera_subobjective.get("focal_isolation", {}) or {}).get("gate_stage_reached", "")),
            "reason": str((camera_subobjective.get("focal_isolation", {}) or {}).get("reason", "")),
        },
        {
            "axis": "camera_translation_objective_isolation",
            "status": str((camera_subobjective.get("translation_isolation", {}) or {}).get("status", "")),
            "gate_stage_reached": str((camera_subobjective.get("translation_isolation", {}) or {}).get("gate_stage_reached", "")),
            "reason": str((camera_subobjective.get("translation_isolation", {}) or {}).get("reason", "")),
        },
        {
            "axis": "camera_objective_coupling_rebalancing",
            "status": str(
                (((camera_depth_postmortem.get("closed_axes", {}) or {}).get("camera_objective_coupling_rebalancing", {}) or {}).get("status", ""))
            ),
            "gate_stage_reached": str(
                (((camera_depth_postmortem.get("closed_axes", {}) or {}).get("camera_objective_coupling_rebalancing", {}) or {}).get("gate_stage_reached", ""))
            ),
            "reason": str(
                (((camera_depth_postmortem.get("closed_axes", {}) or {}).get("camera_objective_coupling_rebalancing", {}) or {}).get("reason", ""))
            ),
        },
        {
            "axis": "camera_depth_objective_coupling_audit",
            "status": str((frontier.get("latest_formal_verdict", {}) or {}).get("status", "")),
            "gate_stage_reached": str((frontier.get("latest_formal_verdict", {}) or {}).get("gate_stage_reached", "")),
            "reason": str((frontier.get("latest_formal_verdict", {}) or {}).get("reason", "")),
        },
        {
            "axis": "two_stage_objective_decoupling",
            "status": next(
                (
                    str(item.get("verdict", ""))
                    for item in (frontier.get("frontier_progression", []) or [])
                    if str(item.get("family", "")) == "two_stage_objective_decoupling"
                ),
                "",
            ),
            "gate_stage_reached": next(
                (
                    str(item.get("gate_stage_reached", ""))
                    for item in (frontier.get("frontier_progression", []) or [])
                    if str(item.get("family", "")) == "two_stage_objective_decoupling"
                ),
                "",
            ),
            "reason": "Two-stage objective decoupling already closed earlier in the round and should not be reopened automatically.",
        },
    ]
    return {
        "checked_at": now_iso(),
        "artifact_kind": "closed_axis_summary",
        "status": "DISCOVERY_BOUNDARY_REACHED",
        "closed_axes": axes,
        "frozen_low_level_families": family_stop.get("frozen_families", {}),
        "tail_contract_derivative_batch": family_stop.get("tail_contract_derivative_batch", {}),
        "summary": [
            "All recent camera-object execution families are closed for this round.",
            "The latest local camera-depth audit is dead_same_day and may not be re-armed automatically.",
            "Discovery should target a fresh higher-level manual problem rather than another ticket on a closed axis.",
        ],
    }


def build_higher_level_problem_inventory(research: dict, cloud_result: dict, cloud_runtime: dict, closed_summary: dict) -> dict:
    return {
        "checked_at": now_iso(),
        "artifact_kind": "higher_level_problem_inventory",
        "status": "DISCOVERY_BOUNDARY_REACHED",
        "current_stable_lead": {
            "config": research.get("current_stable_lead_config", ""),
            "cloud_validation_status": cloud_result.get("artifact_kind", ""),
            "cloud_validation_mode": cloud_result.get("mode", ""),
            "cloud_validation_metrics": cloud_result.get("val", {}),
            "cloud_cleanup_ok": bool(cloud_runtime.get("cleanup_ok")),
        },
        "boundary_facts": [
            "Promoted hybrid-ring stable lead now has both local promotion and one clean downstream cloud validation.",
            "The latest camera-depth objective coupling audit failed locally at short_gate_10x5 with zero metric movement versus the stable lead.",
            "There is no active approved problem, no active allowed family, and no active Modal app.",
        ],
        "inventory": [
            {
                "problem_kind": "lead_validation_alignment",
                "status": "candidate_for_manual_selection_only",
                "question": "How should local long-gate evidence and cloud validation evidence be aligned for the promoted stable lead?",
                "why_now": "The lead is now validated in both places, so the remaining question is interpretive rather than executional.",
            },
            {
                "problem_kind": "selection_contract_mechanism",
                "status": "candidate_for_manual_selection_only",
                "question": "Why does nearest_plus_uniform_tail remain the best lead even though the probe diff shows tighter valid-ratio/point coverage and late-slot churn?",
                "why_now": "The cloud run reinforces that the lead is real, so the next higher-level work can focus on mechanism rather than more ticket attempts.",
            },
            {
                "problem_kind": "cross_axis_plateau_root_cause",
                "status": "candidate_for_manual_selection_only",
                "question": "What higher-level root cause explains why all recent camera-object families plateaued while the promoted source-policy lead stayed best?",
                "why_now": "This is the cleanest synthesis target after the focal, translation, FL/T coupling, and camera-depth audit axes all closed.",
            },
        ],
        "closed_axis_count": len(closed_summary.get("closed_axes", [])),
    }


def build_next_manual_problem_candidates(cloud_result: dict) -> dict:
    candidates = [
        {
            "family": "promoted_lead_cloud_local_alignment_audit",
            "status": "manual_discovery_only",
            "question": "Reconcile the promoted hybrid-ring local gate evidence with the clean cloud validation result so future promotion logic uses one explicit alignment rule.",
            "based_on": [
                "local promoted lead",
                repo_rel(CLOUD_RESULT_JSON),
            ],
            "forbidden_actions": [
                "do not auto-arm",
                "do not auto-run training",
                "do not open cloud from this candidate by default",
            ],
        },
        {
            "family": "hybrid_ring_selection_contract_mechanism_audit",
            "status": "manual_discovery_only",
            "question": "Explain why slot_3 source churn and tighter point-cloud coverage still produce the strongest promoted lead under nearest_plus_uniform_tail.",
            "based_on": [
                repo_rel(CLOUD_RESULT_JSON),
                "dataset_probe_contract_diff for the promoted hybrid-ring cloud validation",
            ],
            "forbidden_actions": [
                "do not reinterpret this as a new execution-ready family",
                "do not reopen source-policy cousins automatically",
            ],
        },
        {
            "family": "cross_axis_plateau_boundary_synthesis",
            "status": "manual_discovery_only",
            "question": "Synthesize the closed focal, translation, FL/T coupling, and camera-depth audit axes into one higher-level root-cause problem statement.",
            "based_on": [
                repo_rel(CAMERA_SUBOBJECTIVE_POSTMORTEM_JSON),
                repo_rel(CAMERA_DEPTH_POSTMORTEM_JSON),
            ],
            "forbidden_actions": [
                "do not auto-materialize a ticket",
                "do not mark ready_for_execution automatically",
            ],
        },
        {
            "family": "tail_contract_boundary_reconciliation_audit",
            "status": "manual_discovery_only",
            "question": "Resolve whether the historical tail-contract final-discriminator boundary still matters after the promoted lead and later camera-object closures.",
            "based_on": [
                repo_rel(FAMILY_STOP_REASON_JSON),
                repo_rel(CLOUD_RESULT_JSON),
            ],
            "forbidden_actions": [
                "do not reopen default_stream_intrinsics_counterbalance automatically",
                "do not treat this as permission to run a dormant tail-family ticket",
            ],
        },
    ]
    return {
        "checked_at": now_iso(),
        "artifact_kind": "next_manual_problem_candidates",
        "status": "DISCOVERY_BOUNDARY_REACHED",
        "promoted_lead_cloud_validation_done_clean": True,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selection_rule": "Human daytime review must choose whether any of these becomes the next fresh manual problem; no candidate here is execution-ready by itself.",
        "current_cloud_validation_ref": repo_rel(CLOUD_RESULT_JSON),
        "current_cloud_validation_metrics": cloud_result.get("val", {}),
    }


def render_md(title: str, payload: dict) -> str:
    return f"# {title}\n\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"


def main() -> int:
    _ = parse_args()
    report_dir = ensure_dir(OUTPUT_ROOT / f"{date_tag()}_discovery")

    research = load_json(RESEARCH_STATUS_JSON)
    frontier = load_json(FRONTIER_LEDGER_JSON)
    family_stop = load_json(FAMILY_STOP_REASON_JSON)
    task_plan = load_json(TASK_PLAN_JSON)
    allowlist = load_json(ALLOWLIST_JSON)
    cloud_result = maybe_load_json(
        latest_existing_path("output/zju_source_policy_research_loop/cloud_validation_result.source_policy_hybrid_ring_regularization.*.json")
        or CLOUD_RESULT_JSON
    )
    cloud_runtime = maybe_load_json(
        latest_existing_path("output/zju_source_policy_research_loop/cloud_runtime_state.*.json")
        or CLOUD_RUNTIME_JSON
    )
    camera_subobjective = maybe_load_json(CAMERA_SUBOBJECTIVE_POSTMORTEM_JSON)
    camera_depth_postmortem = maybe_load_json(CAMERA_DEPTH_POSTMORTEM_JSON)
    active_modal_apps = list_active_modal_apps()

    boundary_ok, issues = discovery_boundary_ok(research, allowlist, active_modal_apps, cloud_runtime)
    report = {
        "checked_at": now_iso(),
        "artifact_kind": "fresh_manual_problem_discovery_guard_report",
        "status": "DISCOVERY_BOUNDARY_REACHED" if boundary_ok else "NOT_DISCOVERY_BOUNDARY",
        "boundary_ok": boundary_ok,
        "boundary_issues": issues,
        "truth": {
            "research_state": research.get("state", ""),
            "approved_problem_present": bool(research.get("approved_problem_present")),
            "allowed_families": list(research.get("allowed_families", []) or []),
            "ready_for_execution": bool(research.get("ready_for_execution")),
            "current_priority_family": research.get("current_priority_family", ""),
        },
        "cleanup": {
            "allowlist_status": allowlist.get("status", ""),
            "active_modal_app_count": len(active_modal_apps),
            "cloud_cleanup_ok": bool(cloud_runtime.get("cleanup_ok")),
        },
    }

    if boundary_ok:
        closed_summary = build_closed_axis_summary(frontier, family_stop, camera_subobjective, camera_depth_postmortem)
        inventory = build_higher_level_problem_inventory(research, cloud_result, cloud_runtime, closed_summary)
        candidates = build_next_manual_problem_candidates(cloud_result)

        closed_json = RESEARCH_ROOT / f"closed_axis_summary.{date_tag()}.json"
        closed_md = RESEARCH_ROOT / f"closed_axis_summary.{date_tag()}.md"
        inventory_json = RESEARCH_ROOT / f"higher_level_problem_inventory.{date_tag()}.json"
        inventory_md = RESEARCH_ROOT / f"higher_level_problem_inventory.{date_tag()}.md"
        candidates_json = RESEARCH_ROOT / f"next_manual_problem_candidates.{date_tag()}.json"
        candidates_md = RESEARCH_ROOT / f"next_manual_problem_candidates.{date_tag()}.md"

        write_json(closed_json, closed_summary)
        write_text(closed_md, render_md("Closed Axis Summary", closed_summary))
        write_json(inventory_json, inventory)
        write_text(inventory_md, render_md("Higher-Level Problem Inventory", inventory))
        write_json(candidates_json, candidates)
        write_text(candidates_md, render_md("Next Manual Problem Candidates", candidates))

        report["artifacts"] = {
            "closed_axis_summary": repo_rel(closed_json),
            "higher_level_problem_inventory": repo_rel(inventory_json),
            "next_manual_problem_candidates": repo_rel(candidates_json),
        }
        report["task_mode_focus"] = task_plan.get("task_mode_focus", "")

    report_json = OUTPUT_ROOT / "latest_report.json"
    report_md = OUTPUT_ROOT / "latest_report.md"
    write_json(report_json, report)
    write_text(report_md, render_md("Fresh Manual Problem Discovery Guard Report", report))
    write_json(report_dir / "report.json", report)
    write_text(report_dir / "report.md", render_md("Fresh Manual Problem Discovery Guard Report", report))
    print(json.dumps({"latest_report_json": str(report_json.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
