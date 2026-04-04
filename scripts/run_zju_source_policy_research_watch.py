import argparse
import atexit
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_LOOP_SCRIPT_REL = Path("scripts") / "run_zju_source_policy_research_loop.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_watch"
DEFAULT_LOCK_PATH = DEFAULT_OUTPUT_ROOT / "active_watch.json"
DEFAULT_LATEST_SESSION_PATH = DEFAULT_OUTPUT_ROOT / "latest_session.json"
DEFAULT_LATEST_SNAPSHOT_PATH = DEFAULT_OUTPUT_ROOT / "latest_watch_snapshot.json"
DEFAULT_GUARD_SNAPSHOT_PATH = (
    REPO_ROOT / "output" / "zju_source_policy_rawpool_overnight_watch" / "latest_guard_snapshot.json"
)
DEFAULT_RESEARCH_STATUS_PATH = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "research_loop_status.json"
DEFAULT_APPROVED_PROBLEM_PATH = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "approved_problem.json"
DEFAULT_ALLOWLIST_PATH = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "repo_process_allowlist.json"
DEFAULT_LOCAL_MANIFEST_PATH = REPO_ROOT / "scripts" / "manifests" / "zju_source_policy_rawpool_local_nightly_v1.json"

EXPECTED_STABLE_LEAD_CONFIG = (
    "training/config/zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml"
)
WATCH_SCRIPT_MARKER = "run_zju_source_policy_research_watch.py"
RESEARCH_RUNTIME_MARKERS = (
    "run_zju_source_policy_research_candidate.py",
    "run_zju_vggt_geom_minimal_finetune.ps1",
    "run_zju_source_policy_rawpool_long_gate.py",
    "compare_zju_finetune_runs.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a guard-safe research watch loop that only refreshes research-loop artifacts and records "
            "whether the repo remains in the expected idle-guard state."
        )
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--latest-session-path", type=Path, default=DEFAULT_LATEST_SESSION_PATH)
    parser.add_argument("--latest-snapshot-path", type=Path, default=DEFAULT_LATEST_SNAPSHOT_PATH)
    parser.add_argument("--guard-snapshot-path", type=Path, default=DEFAULT_GUARD_SNAPSHOT_PATH)
    parser.add_argument("--research-status-path", type=Path, default=DEFAULT_RESEARCH_STATUS_PATH)
    parser.add_argument("--approved-problem-path", type=Path, default=DEFAULT_APPROVED_PROBLEM_PATH)
    parser.add_argument("--allowlist-path", type=Path, default=DEFAULT_ALLOWLIST_PATH)
    parser.add_argument("--local-manifest-path", type=Path, default=DEFAULT_LOCAL_MANIFEST_PATH)
    parser.add_argument("--interval-sec", type=int, default=600)
    parser.add_argument("--sleep-heartbeat-sec", type=int, default=60)
    parser.add_argument("--duration-hours", type=float, default=12.0)
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means unlimited until duration-hours is reached.")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def maybe_load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return load_json(path)


def safe_int(value: object, default: int = -1) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def run_cmd(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def process_info(pid: int) -> dict | None:
    result = run_cmd(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "$p = Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\"; "
                "if ($p) {{ $p | Select-Object ProcessId, Name, CommandLine | ConvertTo-Json -Compress }}"
            ).format(pid=pid),
        ],
        cwd=REPO_ROOT,
    )
    text = result.stdout.strip()
    if result.returncode != 0 or not text:
        return None
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload[0] if payload else None
    return payload


def acquire_lock(lock_path: Path, session_dir: Path, args: argparse.Namespace) -> None:
    if lock_path.exists():
        prior = maybe_load_json(lock_path)
        prior_pid = int(prior.get("pid", 0) or 0)
        if prior_pid > 0 and prior_pid != os.getpid():
            info = process_info(prior_pid)
            if info and WATCH_SCRIPT_MARKER in str(info.get("CommandLine", "")):
                raise RuntimeError(
                    "Another research watch is already active: pid={pid} session_dir={session_dir}".format(
                        pid=prior_pid,
                        session_dir=prior.get("session_dir", ""),
                    )
                )
    payload = {
        "pid": os.getpid(),
        "started_at": iso_now(),
        "session_dir": str(session_dir.resolve()),
        "script": str(Path(__file__).resolve()),
        "interval_sec": int(args.interval_sec),
        "sleep_heartbeat_sec": int(args.sleep_heartbeat_sec),
        "duration_hours": float(args.duration_hours),
        "max_cycles": int(args.max_cycles),
        "once": bool(args.once),
        "last_heartbeat": iso_now(),
        "last_cycle": 0,
        "last_status": "starting",
    }
    write_json(lock_path, payload)


def update_lock(lock_path: Path, *, cycle_index: int, status: str, latest_snapshot_path: str = "") -> None:
    payload = maybe_load_json(lock_path)
    payload.update(
        {
            "pid": os.getpid(),
            "last_heartbeat": iso_now(),
            "last_cycle": int(cycle_index),
            "last_status": status,
            "latest_snapshot_path": latest_snapshot_path,
        }
    )
    write_json(lock_path, payload)


def release_lock(lock_path: Path) -> None:
    if not lock_path.exists():
        return
    payload = maybe_load_json(lock_path)
    if int(payload.get("pid", 0) or 0) == os.getpid():
        lock_path.unlink(missing_ok=True)


def list_modal_apps() -> list[dict]:
    result = run_cmd(["modal", "app", "list", "--json"], cwd=REPO_ROOT)
    if result.returncode != 0:
        return [{"State": "unknown", "error": result.stderr.strip() or result.stdout.strip()}]
    text = result.stdout.strip()
    if not text:
        return []
    payload = json.loads(text)
    return payload if isinstance(payload, list) else []


def get_research_runtime_processes() -> list[dict]:
    pattern = "|".join(RESEARCH_RUNTIME_MARKERS).replace("\\", "\\\\")
    command = """
$selfPid = {self_pid}
$pattern = '{pattern}'
$items = Get-CimInstance Win32_Process | Where-Object {{
  $_.ProcessId -ne $PID -and
  $_.ProcessId -ne $selfPid -and
  $_.CommandLine -and
  ($_.CommandLine -match $pattern)
}} | Select-Object ProcessId, Name, CommandLine
$items | ConvertTo-Json -Compress
""".strip().format(
        self_pid=os.getpid(),
        pattern=pattern,
    )
    result = run_cmd(["powershell", "-NoProfile", "-Command", command], cwd=REPO_ROOT)
    text = result.stdout.strip()
    if result.returncode != 0 or not text:
        return []
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    return [payload]


def resolve_expected_stable_lead(local_manifest_path: Path) -> str:
    payload = maybe_load_json(local_manifest_path)
    current_lead = str((payload.get("current_lead", {}) or {}).get("config", "")).strip()
    return current_lead or EXPECTED_STABLE_LEAD_CONFIG


def build_guard_summary(guard_snapshot: dict, local_manifest_path: Path) -> dict:
    expected_lead = resolve_expected_stable_lead(local_manifest_path)
    lead_value = str(
        guard_snapshot.get("state_current_lead_config")
        or guard_snapshot.get("current_lead_config")
        or ""
    )
    checks = {
        "steady_hold": str(guard_snapshot.get("state_latest_decision", "")) == "steady_hold",
        "stable_lead_expected": lead_value == expected_lead,
        "consistency_ok": bool(guard_snapshot.get("consistency_ok")),
        "cloud_gate_off": not bool(guard_snapshot.get("state_cloud_gate")),
        "launch_cloud_now_off": not bool(guard_snapshot.get("state_launch_cloud_now")),
        "active_modal_app_count_zero": safe_int(guard_snapshot.get("active_modal_app_count"), default=-1) == 0,
        "repo_process_count_zero": safe_int(guard_snapshot.get("repo_process_count"), default=-1) == 0,
    }
    return {
        "all_green": all(checks.values()),
        "checks": checks,
        "expected_stable_lead_config": expected_lead,
        "observed_stable_lead_config": lead_value,
    }


def build_research_summary(
    *,
    research_status: dict,
    approved_problem_path: Path,
    allowlist_payload: dict,
    modal_apps: list[dict],
    research_runtime_processes: list[dict],
) -> dict:
    allowed_markers = allowlist_payload.get("allowed_markers", [])
    active_modal_apps = [row for row in modal_apps if str(row.get("State", "")).lower() != "stopped"]
    latest_alignment_audit = research_status.get("latest_alignment_audit_result", {}) or {}
    latest_promoted_cloud_validation = research_status.get("latest_promoted_lead_cloud_validation", {}) or {}
    next_requirement = str(research_status.get("next_requirement", "")).strip().lower()
    runner_launch_required = (
        str(research_status.get("state", "")) == "ARMED_PROBLEM"
        and bool(research_status.get("approved_problem_present"))
    )
    promotion_decision_required = "manual promotion decision" in next_requirement
    ready_for_execution = bool(research_status.get("ready_for_execution"))
    explicit_manual_action_kind = str(research_status.get("manual_action_kind", "")).strip()
    manual_review_required = (
        not bool(research_status.get("approved_problem_present"))
        and bool(str(research_status.get("preferred_first_family", "")).strip())
    )
    manual_action_kind = (
        explicit_manual_action_kind
        or (
            "runner_launch"
            if runner_launch_required
            else (
                "promotion_decision"
                if promotion_decision_required
                else ("manual_review" if manual_review_required else "")
            )
        )
    )
    return {
        "state": str(research_status.get("state", "")),
        "approved_problem_present": bool(research_status.get("approved_problem_present")) or approved_problem_path.exists(),
        "approved_problem_ready": bool(research_status.get("approved_problem_ready")),
        "allowlist_empty": isinstance(allowed_markers, list) and len(allowed_markers) == 0,
        "allowlist_status": str(allowlist_payload.get("status", "")),
        "active_modal_app_count": len(active_modal_apps),
        "runtime_process_count": len(research_runtime_processes),
        "manual_action_required": bool(research_status.get("manual_action_required")) or bool(manual_action_kind),
        "manual_action_kind": manual_action_kind,
        "ready_for_execution": ready_for_execution,
        "alignment_audit_completed_clean": (
            str(latest_alignment_audit.get("status", "")).strip() == "completed_clean"
            and not ready_for_execution
            and not str(research_status.get("current_priority_family", "")).strip()
        ),
        "alignment_audit_family": str(latest_alignment_audit.get("family", "")).strip(),
        "promoted_lead_cloud_validation_done_clean": (
            str(latest_promoted_cloud_validation.get("status", "")).strip() == "cloud_validation_done_clean"
            or (
                str(latest_promoted_cloud_validation.get("artifact_kind", "")).strip() == "cloud_validation_result"
                and int(latest_promoted_cloud_validation.get("active_modal_app_count_after_finish", -1)) == 0
            )
        ),
    }


def build_snapshot(
    *,
    cycle_index: int,
    stage: str,
    guard_snapshot_path: Path,
    research_status_path: Path,
    approved_problem_path: Path,
    allowlist_path: Path,
    local_manifest_path: Path,
    refresh_exit_code: int | None,
    refresh_error: str,
    cycle_dir: Path,
) -> dict:
    guard_snapshot = maybe_load_json(guard_snapshot_path)
    research_status = maybe_load_json(research_status_path)
    allowlist_payload = maybe_load_json(allowlist_path)
    modal_apps = list_modal_apps()
    research_runtime_processes = get_research_runtime_processes()
    guard_summary = build_guard_summary(guard_snapshot, local_manifest_path)
    research_summary = build_research_summary(
        research_status=research_status,
        approved_problem_path=approved_problem_path,
        allowlist_payload=allowlist_payload,
        modal_apps=modal_apps,
        research_runtime_processes=research_runtime_processes,
    )
    snapshot = {
        "checked_at": iso_now(),
        "stage": stage,
        "cycle_index": int(cycle_index),
        "watch_contract_mode": "status_refresh_only",
        "refresh_exit_code": refresh_exit_code,
        "refresh_error": refresh_error,
        "cycle_dir": str(cycle_dir.resolve()),
        "guard": {
            "latest_guard_snapshot_path": str(guard_snapshot_path.resolve()),
            "summary": guard_summary,
            "snapshot": guard_snapshot,
        },
        "research": {
            "research_status_path": str(research_status_path.resolve()),
            "approved_problem_path": str(approved_problem_path.resolve()),
            "allowlist_path": str(allowlist_path.resolve()),
            "summary": research_summary,
            "research_status": research_status,
            "allowlist": allowlist_payload,
        },
        "modal_apps": modal_apps,
        "research_runtime_processes": research_runtime_processes,
        "watch_conclusion": build_watch_conclusion(
            guard_all_green=guard_summary["all_green"],
            research_summary=research_summary,
            refresh_exit_code=refresh_exit_code,
            refresh_error=refresh_error,
        ),
    }
    return snapshot


def build_watch_conclusion(*, guard_all_green: bool, research_summary: dict, refresh_exit_code: int | None, refresh_error: str) -> str:
    if refresh_exit_code not in (None, 0):
        return f"research refresh failed but the watch remains passive: {refresh_error}"
    if not guard_all_green:
        return "guard drift detected; watch remains passive and records the failure without opening research"
    if research_summary.get("alignment_audit_completed_clean"):
        return "guard is green; alignment audit finished clean, no active cloud app remains, and no active local family is open"
    if research_summary.get("manual_action_kind") == "manual_approval" and research_summary.get("ready_for_execution"):
        return "guard is green and research remains idle; exactly one execution-ready family is pending explicit manual arm approval before any run"
    if research_summary.get("manual_action_kind") == "runner_launch":
        return "an approved problem is armed; watch remains passive and waits for an explicit manual runner launch"
    if research_summary.get("manual_action_kind") == "promotion_decision":
        return "guard is green and research remains idle; a provisional local lead is waiting for a fresh manual promotion decision"
    if research_summary.get("manual_action_kind") == "manual_review":
        return "guard is green and research remains idle; exactly one pending manual-review family is waiting for an explicit approval before any run"
    if research_summary.get("runtime_process_count", 0):
        return "research execution processes are active; watch records them but does not intervene"
    if research_summary.get("state") == "IDLE_GUARD":
        return "guard is green and research remains idle; no research progress is occurring"
    return f"research state is {research_summary.get('state', 'unknown')}; watch records it without taking action"


def write_runtime_status(
    *,
    session_dir: Path,
    latest_session_path: Path,
    status: str,
    reason: str,
    cycle_count: int,
    lock_path: Path,
    latest_snapshot_path: Path,
) -> None:
    payload = {
        "checked_at": iso_now(),
        "status": status,
        "reason": reason,
        "cycle_count": int(cycle_count),
        "session_dir": str(session_dir.resolve()),
        "lock_path": str(lock_path.resolve()),
        "latest_snapshot_path": str(latest_snapshot_path.resolve()),
    }
    write_json(session_dir / "runtime_status.json", payload)
    write_json(latest_session_path, payload)


def run_refresh(python_exe: str, cycle_dir: Path) -> tuple[int, str]:
    result = run_cmd([python_exe, str(RESEARCH_LOOP_SCRIPT_REL)], cwd=REPO_ROOT)
    write_text(cycle_dir / "refresh_stdout.txt", result.stdout)
    write_text(cycle_dir / "refresh_stderr.txt", result.stderr)
    error = ""
    if result.returncode != 0:
        error = (
            f"run_zju_source_policy_research_loop.py exited with {result.returncode}. "
            f"stderr={result.stderr.strip()!r}"
        )
    return result.returncode, error


def sleep_with_heartbeat(
    *,
    lock_path: Path,
    session_dir: Path,
    latest_session_path: Path,
    latest_snapshot_path: Path,
    guard_snapshot_path: Path,
    research_status_path: Path,
    approved_problem_path: Path,
    allowlist_path: Path,
    local_manifest_path: Path,
    cycle_index: int,
    total_sleep_sec: int,
    heartbeat_sec: int,
) -> None:
    total_sleep_sec = max(int(total_sleep_sec), 0)
    step = max(int(heartbeat_sec), 5)
    if total_sleep_sec <= 0:
        return
    deadline = time.monotonic() + total_sleep_sec
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(step, remaining))
        if time.monotonic() >= deadline:
            break
        heartbeat_dir = ensure_dir(session_dir / f"cycle_{cycle_index:03d}" / "heartbeat")
        snapshot = build_snapshot(
            cycle_index=cycle_index,
            stage="sleep_heartbeat",
            guard_snapshot_path=guard_snapshot_path,
            research_status_path=research_status_path,
            approved_problem_path=approved_problem_path,
            allowlist_path=allowlist_path,
            local_manifest_path=local_manifest_path,
            refresh_exit_code=None,
            refresh_error="",
            cycle_dir=heartbeat_dir,
        )
        write_json(heartbeat_dir / f"{now_tag()}_snapshot.json", snapshot)
        write_json(latest_snapshot_path, snapshot)
        update_lock(
            lock_path,
            cycle_index=cycle_index,
            status="sleeping",
            latest_snapshot_path=str(latest_snapshot_path.resolve()),
        )
        write_runtime_status(
            session_dir=session_dir,
            latest_session_path=latest_session_path,
            status="running",
            reason="sleeping between passive research refresh cycles",
            cycle_count=cycle_index,
            lock_path=lock_path,
            latest_snapshot_path=latest_snapshot_path,
        )


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_root)
    session_dir = ensure_dir(args.output_root / f"{now_tag()}_watch")
    acquire_lock(args.lock_path, session_dir, args)
    atexit.register(release_lock, args.lock_path)

    write_runtime_status(
        session_dir=session_dir,
        latest_session_path=args.latest_session_path,
        status="starting",
        reason="research watch is starting in passive status-refresh-only mode",
        cycle_count=0,
        lock_path=args.lock_path,
        latest_snapshot_path=args.latest_snapshot_path,
    )

    start_time = time.monotonic()
    cycle_index = 0
    while True:
        if args.max_cycles and cycle_index >= args.max_cycles:
            break
        if not args.once and (time.monotonic() - start_time) >= args.duration_hours * 3600:
            break
        cycle_index += 1
        cycle_dir = ensure_dir(session_dir / f"cycle_{cycle_index:03d}")
        refresh_exit_code, refresh_error = run_refresh(args.python_exe, cycle_dir)
        snapshot = build_snapshot(
            cycle_index=cycle_index,
            stage="cycle_complete",
            guard_snapshot_path=args.guard_snapshot_path,
            research_status_path=args.research_status_path,
            approved_problem_path=args.approved_problem_path,
            allowlist_path=args.allowlist_path,
            local_manifest_path=args.local_manifest_path,
            refresh_exit_code=refresh_exit_code,
            refresh_error=refresh_error,
            cycle_dir=cycle_dir,
        )
        write_json(cycle_dir / "cycle_snapshot.json", snapshot)
        write_json(args.latest_snapshot_path, snapshot)
        update_lock(
            args.lock_path,
            cycle_index=cycle_index,
            status="running_degraded" if refresh_exit_code else "running",
            latest_snapshot_path=str(args.latest_snapshot_path.resolve()),
        )
        write_runtime_status(
            session_dir=session_dir,
            latest_session_path=args.latest_session_path,
            status="running_degraded" if refresh_exit_code else "running",
            reason=snapshot["watch_conclusion"],
            cycle_count=cycle_index,
            lock_path=args.lock_path,
            latest_snapshot_path=args.latest_snapshot_path,
        )
        if args.once:
            break
        sleep_with_heartbeat(
            lock_path=args.lock_path,
            session_dir=session_dir,
            latest_session_path=args.latest_session_path,
            latest_snapshot_path=args.latest_snapshot_path,
            guard_snapshot_path=args.guard_snapshot_path,
            research_status_path=args.research_status_path,
            approved_problem_path=args.approved_problem_path,
            allowlist_path=args.allowlist_path,
            local_manifest_path=args.local_manifest_path,
            cycle_index=cycle_index,
            total_sleep_sec=args.interval_sec,
            heartbeat_sec=args.sleep_heartbeat_sec,
        )

    write_runtime_status(
        session_dir=session_dir,
        latest_session_path=args.latest_session_path,
        status="completed",
        reason="research watch finished its scheduled passive monitoring window",
        cycle_count=cycle_index,
        lock_path=args.lock_path,
        latest_snapshot_path=args.latest_snapshot_path,
    )
    update_lock(
        args.lock_path,
        cycle_index=cycle_index,
        status="completed",
        latest_snapshot_path=str(args.latest_snapshot_path.resolve()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
