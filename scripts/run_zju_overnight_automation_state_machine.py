import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "overnight_automation"
RESEARCH_STATUS_JSON = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "research_loop_status.json"
APPROVED_PROBLEM_JSON = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "approved_problem.json"
ALLOWLIST_JSON = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "repo_process_allowlist.json"
CLOUD_RESULT_JSON = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "cloud_validation_result.source_policy_hybrid_ring_regularization.20260403.json"
CLOUD_RUNTIME_JSON = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "cloud_runtime_state.20260403.json"

RESEARCH_LOOP_SCRIPT = REPO_ROOT / "scripts" / "run_zju_source_policy_research_loop.py"
CANDIDATE_RUNNER = REPO_ROOT / "scripts" / "run_zju_source_policy_research_candidate.py"
WATCH_SCRIPT = REPO_ROOT / "scripts" / "run_zju_source_policy_research_watch.py"
DISCOVERY_GUARD_SCRIPT = REPO_ROOT / "scripts" / "run_zju_fresh_manual_problem_discovery_guard_only.py"

STATE_IDLE_GUARD = "IDLE_GUARD"
STATE_MANUAL_PROBLEM_PACKAGING = "MANUAL_PROBLEM_PACKAGING"
STATE_LOCAL_SINGLE_TICKET_EXECUTION = "LOCAL_SINGLE_TICKET_EXECUTION"
STATE_LOCAL_PROVISIONAL_LEAD_HOLD = "LOCAL_PROVISIONAL_LEAD_HOLD"
MODE_GUARD_ONLY = "GUARD_ONLY"
MODE_SINGLE_TICKET_RUN = "SINGLE_TICKET_RUN"
MODE_DISCOVERY_GUARD_ONLY = "DISCOVERY_GUARD_ONLY"


class OvernightAutomationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ZJU overnight automation state machine with only guard-only or single-ticket execution modes."
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--mode", choices=["auto", "guard_only", "single_ticket_run", "discovery_guard_only"], default="auto")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


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


def run_checked(args: list[str], *, stdout_path: Path, stderr_path: Path) -> None:
    result = subprocess.run(
        args,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    write_text(stdout_path, result.stdout)
    write_text(stderr_path, result.stderr)
    if result.returncode != 0:
        raise OvernightAutomationError(
            "Command failed with exit code {code}: {cmd}\nSTDERR:\n{stderr}".format(
                code=result.returncode,
                cmd=" ".join(args),
                stderr=result.stderr.strip(),
            )
        )


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
        raise OvernightAutomationError(f"modal app list --json failed: {result.stderr.strip()}")
    payload = json.loads(result.stdout.strip() or "[]")
    return [row for row in payload if str(row.get("State", "")).lower() != "stopped"]


def verify_idle_cleanup() -> dict:
    allowlist = load_json(ALLOWLIST_JSON)
    if str(allowlist.get("status", "")).strip() != "idle_empty_allowlist":
        raise OvernightAutomationError("repo_process_allowlist.json is not idle_empty_allowlist.")
    active_apps = list_active_modal_apps()
    if active_apps:
        raise OvernightAutomationError("Modal apps are still active; overnight automation requires a clean cloud state.")
    return {
        "allowlist_status": allowlist.get("status", ""),
        "active_modal_app_count": len(active_apps),
    }


def detect_state(research_status: dict, approved_problem: dict) -> str:
    if approved_problem and bool(approved_problem.get("approved")):
        return STATE_LOCAL_SINGLE_TICKET_EXECUTION
    latest_verdict = research_status.get("latest_formal_verdict", {}) or {}
    if str(latest_verdict.get("status", "")).strip() == "provisional_lead":
        return STATE_LOCAL_PROVISIONAL_LEAD_HOLD
    if bool(research_status.get("ready_for_execution")) and str(research_status.get("manual_action_kind", "")).strip() == "manual_approval":
        return STATE_MANUAL_PROBLEM_PACKAGING
    return STATE_IDLE_GUARD


def discovery_boundary_ready(research_status: dict) -> bool:
    cloud_result = maybe_load_json(
        latest_existing_path("output/zju_source_policy_research_loop/cloud_validation_result.source_policy_hybrid_ring_regularization.*.json")
        or CLOUD_RESULT_JSON
    )
    cloud_runtime = maybe_load_json(
        latest_existing_path("output/zju_source_policy_research_loop/cloud_runtime_state.*.json")
        or CLOUD_RUNTIME_JSON
    )
    return (
        str(research_status.get("state", "")).strip() == STATE_IDLE_GUARD
        and not bool(research_status.get("approved_problem_present"))
        and not bool(research_status.get("allowed_families", []))
        and not bool(research_status.get("ready_for_execution"))
        and str(research_status.get("current_priority_family", "")).strip() == ""
        and str(cloud_result.get("artifact_kind", "")).strip() == "cloud_validation_result"
        and str(cloud_result.get("mode", "")).strip() == "promoted_lead_cloud_validation"
        and bool(cloud_runtime.get("cleanup_ok"))
    )


def resolve_mode(requested_mode: str, state: str, research_status: dict) -> str:
    if requested_mode == "discovery_guard_only":
        return MODE_DISCOVERY_GUARD_ONLY
    if requested_mode == "guard_only":
        return MODE_GUARD_ONLY
    if requested_mode == "single_ticket_run":
        return MODE_SINGLE_TICKET_RUN
    if state == STATE_LOCAL_SINGLE_TICKET_EXECUTION:
        return MODE_SINGLE_TICKET_RUN
    if discovery_boundary_ready(research_status):
        return MODE_DISCOVERY_GUARD_ONLY
    return MODE_GUARD_ONLY


def main() -> int:
    args = parse_args()
    run_dir = ensure_dir(args.output_root / f"{now_tag()}_overnight_state_machine")

    run_checked(
        [args.python_exe, str(RESEARCH_LOOP_SCRIPT)],
        stdout_path=run_dir / "research_loop_refresh.stdout.txt",
        stderr_path=run_dir / "research_loop_refresh.stderr.txt",
    )

    research_status = load_json(RESEARCH_STATUS_JSON)
    approved_problem = maybe_load_json(APPROVED_PROBLEM_JSON)
    state = detect_state(research_status, approved_problem)
    mode = resolve_mode(args.mode, state, research_status)

    records = [
        {
            "step": "research_loop_refresh",
            "status": "pass",
        }
    ]

    if mode == MODE_DISCOVERY_GUARD_ONLY:
        cleanup = verify_idle_cleanup()
        records.append({"step": "cleanup_verify", "status": "pass"})
        run_checked(
            [args.python_exe, str(DISCOVERY_GUARD_SCRIPT)],
            stdout_path=run_dir / "discovery_guard.stdout.txt",
            stderr_path=run_dir / "discovery_guard.stderr.txt",
        )
        records.append({"step": "discovery_guard", "status": "pass"})
        run_checked(
            [args.python_exe, str(WATCH_SCRIPT), "--once"],
            stdout_path=run_dir / "watch_once.stdout.txt",
            stderr_path=run_dir / "watch_once.stderr.txt",
        )
        records.append({"step": "watch_once", "status": "pass"})
        reason = "discovery-guard mode: refresh truth, verify cleanup, and refresh fresh-manual-problem discovery artifacts without opening a ticket"
    elif mode == MODE_GUARD_ONLY:
        cleanup = verify_idle_cleanup()
        records.append({"step": "cleanup_verify", "status": "pass"})
        run_checked(
            [args.python_exe, str(WATCH_SCRIPT), "--once"],
            stdout_path=run_dir / "watch_once.stdout.txt",
            stderr_path=run_dir / "watch_once.stderr.txt",
        )
        records.append({"step": "watch_once", "status": "pass"})
        reason = "guard-only mode: refresh truth, verify cleanup, and keep research idle with no automatic ticket launch"
    else:
        if state != STATE_LOCAL_SINGLE_TICKET_EXECUTION:
            raise OvernightAutomationError(
                "single_ticket_run requires an active approved_problem.json with approved=true."
            )
        run_checked(
            [args.python_exe, str(CANDIDATE_RUNNER)],
            stdout_path=run_dir / "candidate_runner.stdout.txt",
            stderr_path=run_dir / "candidate_runner.stderr.txt",
        )
        records.append({"step": "candidate_runner", "status": "pass"})
        run_checked(
            [args.python_exe, str(RESEARCH_LOOP_SCRIPT)],
            stdout_path=run_dir / "research_loop_post_run.stdout.txt",
            stderr_path=run_dir / "research_loop_post_run.stderr.txt",
        )
        records.append({"step": "research_loop_post_run_refresh", "status": "pass"})
        cleanup = verify_idle_cleanup()
        records.append({"step": "cleanup_verify", "status": "pass"})
        run_checked(
            [args.python_exe, str(WATCH_SCRIPT), "--once"],
            stdout_path=run_dir / "watch_once.stdout.txt",
            stderr_path=run_dir / "watch_once.stderr.txt",
        )
        records.append({"step": "watch_once", "status": "pass"})
        reason = "single-ticket mode: execute exactly one approved ticket, return to guard, and verify cleanup"

    payload = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "state": state,
        "nightly_mode": mode,
        "reason": reason,
        "research_ready_for_execution": bool(research_status.get("ready_for_execution")),
        "manual_action_kind": str(research_status.get("manual_action_kind", "")),
        "current_priority_family": str(research_status.get("current_priority_family", "")),
        "cleanup_verification": cleanup,
        "run_dir": str(run_dir.resolve()),
        "steps": records,
    }
    write_json(run_dir / "status.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
