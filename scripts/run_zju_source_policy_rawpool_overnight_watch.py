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
NIGHTLY_RUNNER = REPO_ROOT / "scripts" / "run_zju_source_policy_rawpool_local_nightly.py"
DEFAULT_LOCAL_MANIFEST_PATH = REPO_ROOT / "scripts" / "manifests" / "zju_source_policy_rawpool_local_nightly_v1.json"
DEFAULT_STATE_PATH = REPO_ROOT / "output" / "geometry_post_v9_nightly_state" / "state.json"
DEFAULT_CONSISTENCY_PATH = REPO_ROOT / "output" / "geometry_post_v9_nightly_state" / "consistency_check.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_rawpool_overnight_watch"
DEFAULT_LOCK_PATH = DEFAULT_OUTPUT_ROOT / "active_watch.json"
DEFAULT_LATEST_SESSION_PATH = DEFAULT_OUTPUT_ROOT / "latest_session.json"
DEFAULT_LATEST_GUARD_SNAPSHOT_PATH = DEFAULT_OUTPUT_ROOT / "latest_guard_snapshot.json"
DEFAULT_REPO_PROCESS_ALLOWLIST_PATH = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "repo_process_allowlist.json"
ALLOWED_REPO_PROCESS_MARKERS = (
    "run_zju_source_policy_rawpool_overnight_watch.py",
    "run_zju_source_policy_rawpool_guard_daemon.py",
)


class WatchError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local-only overnight watch loop for the ZJU source-policy rawpool line."
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--local-manifest", type=Path, default=DEFAULT_LOCAL_MANIFEST_PATH)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--consistency-path", type=Path, default=DEFAULT_CONSISTENCY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--latest-session-path", type=Path, default=DEFAULT_LATEST_SESSION_PATH)
    parser.add_argument("--latest-guard-snapshot-path", type=Path, default=DEFAULT_LATEST_GUARD_SNAPSHOT_PATH)
    parser.add_argument("--repo-process-allowlist-path", type=Path, default=DEFAULT_REPO_PROCESS_ALLOWLIST_PATH)
    parser.add_argument("--interval-sec", type=int, default=1800)
    parser.add_argument("--sleep-heartbeat-sec", type=int, default=60)
    parser.add_argument("--duration-hours", type=float, default=10.0)
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


def run_checked(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = run_cmd(args, cwd=cwd)
    if result.returncode != 0:
        raise WatchError(
            "Command failed with exit code {code}: {cmd}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}".format(
                code=result.returncode,
                cmd=" ".join(args),
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
            )
        )
    return result


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
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, list):
        return payload[0] if payload else None
    return payload


def acquire_lock(lock_path: Path, session_dir: Path, args: argparse.Namespace) -> None:
    if lock_path.exists():
        try:
            prior = load_json(lock_path)
        except Exception:
            prior = {}
        prior_pid = int(prior.get("pid", 0) or 0)
        if prior_pid > 0 and prior_pid != os.getpid():
            info = process_info(prior_pid)
            if info and "run_zju_source_policy_rawpool_overnight_watch.py" in str(info.get("CommandLine", "")):
                raise WatchError(
                    "Another overnight watch is already active: pid={pid} session_dir={session_dir}".format(
                        pid=prior_pid,
                        session_dir=prior.get("session_dir", ""),
                    )
                )
    payload = {
        "pid": os.getpid(),
        "started_at": iso_now(),
        "session_dir": str(session_dir.resolve()),
        "interval_sec": int(args.interval_sec),
        "duration_hours": float(args.duration_hours),
        "max_cycles": int(args.max_cycles),
        "once": bool(args.once),
        "script": str(Path(__file__).resolve()),
        "last_heartbeat": iso_now(),
        "last_cycle": 0,
        "last_status": "starting",
    }
    write_json(lock_path, payload)


def update_lock(lock_path: Path, *, cycle_index: int, status: str, latest_decision_json: str = "") -> None:
    payload = {}
    if lock_path.exists():
        try:
            payload = load_json(lock_path)
        except Exception:
            payload = {}
    payload.update(
        {
            "pid": os.getpid(),
            "last_heartbeat": iso_now(),
            "last_cycle": int(cycle_index),
            "last_status": status,
            "latest_decision_json": latest_decision_json,
        }
    )
    write_json(lock_path, payload)


def sleep_with_heartbeat(
    *,
    lock_path: Path,
    session_dir: Path,
    latest_session_path: Path,
    latest_guard_snapshot_path: Path,
    state_path: Path,
    consistency_path: Path,
    cycle_index: int,
    total_sleep_sec: int,
    heartbeat_sec: int,
    latest_decision_json: str,
    current_lead_config: str,
    expected_lead_config: str,
    repo_process_allowlist_path: Path,
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
        chunk = min(step, remaining)
        time.sleep(chunk)
        if time.monotonic() >= deadline:
            break
        state = load_json(state_path)
        consistency = load_json(consistency_path)
        modal_apps = list_modal_apps()
        repo_processes = get_repo_processes(repo_process_allowlist_path)
        validate_live_guards(modal_apps=modal_apps, repo_processes=repo_processes)
        validate_live_state(
            state=state,
            consistency=consistency,
            expected_lead_config=expected_lead_config,
        )
        write_guard_snapshot(
            session_dir=session_dir,
            latest_guard_snapshot_path=latest_guard_snapshot_path,
            stage="sleep_heartbeat",
            cycle_index=cycle_index,
            latest_decision_json=latest_decision_json,
            current_lead_config=current_lead_config,
            state=state,
            consistency=consistency,
            modal_apps=modal_apps,
            repo_processes=repo_processes,
        )
        update_lock(
            lock_path,
            cycle_index=cycle_index,
            status="sleeping",
            latest_decision_json=latest_decision_json,
        )
        write_runtime_status(
            session_dir=session_dir,
            latest_session_path=latest_session_path,
            status="running",
            reason="sleeping until the next steady_hold cycle",
            cycle_count=cycle_index,
            lock_path=lock_path,
            latest_decision_json=latest_decision_json,
            latest_run_dir=str(Path(latest_decision_json).resolve().parent) if latest_decision_json else "",
        )


def release_lock(lock_path: Path) -> None:
    if not lock_path.exists():
        return
    try:
        payload = load_json(lock_path)
    except Exception:
        payload = {}
    if int(payload.get("pid", 0) or 0) == os.getpid():
        lock_path.unlink(missing_ok=True)


def parse_decision_path(stdout_text: str) -> Path:
    lines = [line.strip() for line in stdout_text.splitlines() if line.strip()]
    if not lines:
        raise WatchError("Nightly runner stdout did not contain a decision path.")
    path = Path(lines[-1])
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise WatchError(f"Nightly runner reported a missing decision path: {path}")
    return path


def normalize_repo_path(path_like: str | Path) -> str:
    path = Path(path_like)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return str(path.resolve())


def list_modal_apps() -> list[dict]:
    result = run_checked(["modal", "app", "list", "--json"], cwd=REPO_ROOT)
    text = result.stdout.strip()
    if not text:
        return []
    payload = json.loads(text)
    return payload if isinstance(payload, list) else []


def load_dynamic_repo_process_markers(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    try:
        payload = load_json(path)
    except Exception:
        return ()
    markers = payload.get("allowed_markers", [])
    if not isinstance(markers, list):
        return ()
    return tuple(str(marker) for marker in markers if str(marker).strip())


def get_repo_processes(repo_process_allowlist_path: Path) -> list[dict]:
    command = """
$repo = '{repo}'
$selfPid = {self_pid}
$items = Get-CimInstance Win32_Process | Where-Object {{
  $_.ProcessId -ne $PID -and
  $_.ProcessId -ne $selfPid -and
  $_.CommandLine -and
  $_.CommandLine.ToLower().Contains($repo) -and
  (($_.Name.ToLower().Replace('.exe','')) -in @('python','powershell','pwsh','modal'))
}} | Select-Object ProcessId, Name, CommandLine
$items | ConvertTo-Json -Compress
""".strip().format(
        repo=str(REPO_ROOT).lower().replace("\\", "\\\\"),
        self_pid=os.getpid(),
    )
    result = run_checked(["powershell", "-NoProfile", "-Command", command], cwd=REPO_ROOT)
    text = result.stdout.strip()
    if not text:
        return []
    payload = json.loads(text)
    if isinstance(payload, list):
        rows = payload
    else:
        rows = [payload]
    allowed_markers = tuple(marker.lower() for marker in (ALLOWED_REPO_PROCESS_MARKERS + load_dynamic_repo_process_markers(repo_process_allowlist_path)))
    filtered = []
    for row in rows:
        cmd = str(row.get("CommandLine", ""))
        cmd_lower = cmd.lower()
        if any(marker in cmd_lower for marker in allowed_markers):
            continue
        filtered.append(row)
    return filtered


def validate_cycle_outputs(decision: dict, state: dict, consistency: dict, modal_apps: list[dict], repo_processes: list[dict]) -> None:
    if decision.get("nightly_mode") != "steady_hold":
        raise WatchError(f"Unexpected nightly_mode from runner: {decision.get('nightly_mode')}")
    if bool(decision.get("cloud_gate")):
        raise WatchError("Nightly decision unexpectedly set cloud_gate=true.")
    if bool(decision.get("launch_cloud_now")):
        raise WatchError("Nightly decision unexpectedly set launch_cloud_now=true.")
    if state.get("latest_decision") != "steady_hold":
        raise WatchError(f"State drift: expected latest_decision=steady_hold, got {state.get('latest_decision')}")
    if bool(state.get("cloud_gate")):
        raise WatchError("State drift: cloud_gate became true.")
    if bool(state.get("launch_cloud_now")):
        raise WatchError("State drift: launch_cloud_now became true.")
    if not bool(consistency.get("ok")):
        raise WatchError("Consistency check is not clean.")
    active_apps = [row for row in modal_apps if str(row.get("State", "")).lower() != "stopped"]
    if active_apps:
        raise WatchError(f"Modal app drift: found {len(active_apps)} active apps.")
    if repo_processes:
        raise WatchError(f"Repo process drift: found {len(repo_processes)} unexpected repo-scoped processes.")


def validate_live_guards(*, modal_apps: list[dict], repo_processes: list[dict]) -> None:
    active_apps = [row for row in modal_apps if str(row.get("State", "")).lower() != "stopped"]
    if active_apps:
        raise WatchError(f"Modal app drift during sleep heartbeat: found {len(active_apps)} active apps.")
    if repo_processes:
        raise WatchError(f"Repo process drift during sleep heartbeat: found {len(repo_processes)} unexpected repo-scoped processes.")


def validate_expected_lead(*, decision: dict | None, state: dict, expected_lead_config: str, phase: str) -> None:
    if decision is not None:
        decision_lead = normalize_repo_path(str(decision.get("current_lead_config", "")))
        if decision_lead != expected_lead_config:
            raise WatchError(
                f"Lead drift {phase}: decision current_lead_config={decision_lead} expected {expected_lead_config}"
            )
    state_lead = normalize_repo_path(str(state.get("current_lead_config", "")))
    if state_lead != expected_lead_config:
        raise WatchError(
            f"Lead drift {phase}: state current_lead_config={state_lead} expected {expected_lead_config}"
        )


def validate_live_state(*, state: dict, consistency: dict, expected_lead_config: str) -> None:
    if state.get("latest_decision") != "steady_hold":
        raise WatchError(f"State drift during sleep heartbeat: expected latest_decision=steady_hold, got {state.get('latest_decision')}")
    if bool(state.get("cloud_gate")):
        raise WatchError("State drift during sleep heartbeat: cloud_gate became true.")
    if bool(state.get("launch_cloud_now")):
        raise WatchError("State drift during sleep heartbeat: launch_cloud_now became true.")
    if not bool(consistency.get("ok")):
        raise WatchError("Consistency drift during sleep heartbeat: consistency_check is not clean.")
    validate_expected_lead(
        decision=None,
        state=state,
        expected_lead_config=expected_lead_config,
        phase="during sleep heartbeat",
    )


def run_cycle(args: argparse.Namespace, session_dir: Path, cycle_index: int, expected_lead_config: str) -> dict:
    cycle_dir = ensure_dir(session_dir / f"cycle_{cycle_index:03d}")
    result = run_checked(
        [
            args.python_exe,
            str(NIGHTLY_RUNNER),
            "--mode",
            "auto",
            "--python-exe",
            args.python_exe,
        ],
        cwd=REPO_ROOT,
    )
    write_text(cycle_dir / "runner_stdout.txt", result.stdout)
    write_text(cycle_dir / "runner_stderr.txt", result.stderr)

    decision_path = parse_decision_path(result.stdout)
    decision = load_json(decision_path)
    state = load_json(args.state_path)
    consistency = load_json(args.consistency_path)
    modal_apps = list_modal_apps()
    repo_processes = get_repo_processes(args.repo_process_allowlist_path)
    validate_cycle_outputs(decision, state, consistency, modal_apps, repo_processes)
    validate_expected_lead(
        decision=decision,
        state=state,
        expected_lead_config=expected_lead_config,
        phase="after cycle",
    )

    active_apps = [row for row in modal_apps if str(row.get("State", "")).lower() != "stopped"]
    cycle_summary = {
        "cycle_index": cycle_index,
        "checked_at": iso_now(),
        "nightly_decision_json": str(decision_path.resolve()),
        "latest_run_dir": str(decision_path.parent.resolve()),
        "nightly_mode": decision.get("nightly_mode"),
        "current_lead_config": decision.get("current_lead_config"),
        "cloud_gate": bool(decision.get("cloud_gate")),
        "launch_cloud_now": bool(decision.get("launch_cloud_now")),
        "consistency_ok": bool(consistency.get("ok")),
        "active_modal_app_count": len(active_apps),
        "repo_process_count": len(repo_processes),
    }
    write_guard_snapshot(
        session_dir=session_dir,
        latest_guard_snapshot_path=args.latest_guard_snapshot_path,
        stage="cycle_complete",
        cycle_index=cycle_index,
        latest_decision_json=cycle_summary["nightly_decision_json"],
        current_lead_config=str(cycle_summary["current_lead_config"]),
        state=state,
        consistency=consistency,
        modal_apps=modal_apps,
        repo_processes=repo_processes,
    )
    write_json(cycle_dir / "cycle_summary.json", cycle_summary)
    write_text(
        cycle_dir / "cycle_summary.md",
        "\n".join(
            [
                "# ZJU Source-Policy Rawpool Overnight Watch Cycle",
                "",
                f"- cycle_index: `{cycle_index}`",
                f"- checked_at: `{cycle_summary['checked_at']}`",
                f"- nightly_mode: `{cycle_summary['nightly_mode']}`",
                f"- current_lead_config: `{cycle_summary['current_lead_config']}`",
                f"- cloud_gate: `{cycle_summary['cloud_gate']}`",
                f"- launch_cloud_now: `{cycle_summary['launch_cloud_now']}`",
                f"- consistency_ok: `{cycle_summary['consistency_ok']}`",
                f"- active_modal_app_count: `{cycle_summary['active_modal_app_count']}`",
                f"- repo_process_count: `{cycle_summary['repo_process_count']}`",
                f"- nightly_decision_json: `{cycle_summary['nightly_decision_json']}`",
                "",
            ]
        ),
    )
    write_json(cycle_dir / "modal_apps.json", {"rows": modal_apps})
    write_json(cycle_dir / "repo_processes.json", {"rows": repo_processes})
    return cycle_summary


def write_session_status(session_dir: Path, payload: dict, latest_session_path: Path | None = None) -> None:
    write_json(session_dir / "watch_status.json", payload)
    lines = ["# ZJU Source-Policy Rawpool Overnight Watch", ""]
    for key, value in payload.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    write_text(session_dir / "watch_status.md", "\n".join(lines))
    if latest_session_path is not None:
        write_json(latest_session_path, payload)


def write_runtime_status(
    *,
    session_dir: Path,
    latest_session_path: Path,
    status: str,
    reason: str,
    cycle_count: int,
    lock_path: Path,
    latest_decision_json: str = "",
    latest_run_dir: str = "",
) -> None:
    payload = {
        "status": status,
        "reason": reason,
        "updated_at": iso_now(),
        "cycle_count": int(cycle_count),
        "last_cycle_nightly_decision_json": latest_decision_json,
        "last_cycle_latest_run_dir": latest_run_dir,
        "session_dir": str(session_dir.resolve()),
        "lock_path": str(lock_path.resolve()),
    }
    write_session_status(session_dir, payload, latest_session_path)


def write_guard_snapshot(
    *,
    session_dir: Path,
    latest_guard_snapshot_path: Path,
    stage: str,
    cycle_index: int,
    latest_decision_json: str,
    current_lead_config: str,
    state: dict,
    consistency: dict,
    modal_apps: list[dict],
    repo_processes: list[dict],
) -> None:
    active_apps = [row for row in modal_apps if str(row.get("State", "")).lower() != "stopped"]
    payload = {
        "checked_at": iso_now(),
        "stage": stage,
        "cycle_index": int(cycle_index),
        "latest_decision_json": latest_decision_json,
        "current_lead_config": current_lead_config,
        "state_latest_decision": state.get("latest_decision", ""),
        "state_current_lead_config": state.get("current_lead_config", ""),
        "state_cloud_gate": bool(state.get("cloud_gate")),
        "state_launch_cloud_now": bool(state.get("launch_cloud_now")),
        "consistency_ok": bool(consistency.get("ok")),
        "active_modal_app_count": len(active_apps),
        "repo_process_count": len(repo_processes),
        "session_dir": str(session_dir.resolve()),
    }
    write_json(session_dir / "guard_snapshot.json", payload)
    write_json(latest_guard_snapshot_path, payload)


def main() -> None:
    args = parse_args()
    local_manifest = load_json(args.local_manifest)
    expected_lead_config = normalize_repo_path(str(local_manifest["current_lead"]["config"]))
    output_root = ensure_dir(args.output_root)
    session_dir = ensure_dir(output_root / f"{now_tag()}_watch")
    acquire_lock(args.lock_path, session_dir, args)
    atexit.register(release_lock, args.lock_path)

    settings = {
        "started_at": iso_now(),
        "pid": os.getpid(),
        "python_exe": args.python_exe,
        "local_manifest": str(args.local_manifest.resolve()),
        "expected_lead_config": expected_lead_config,
        "interval_sec": int(args.interval_sec),
        "sleep_heartbeat_sec": int(args.sleep_heartbeat_sec),
        "duration_hours": float(args.duration_hours),
        "max_cycles": int(args.max_cycles),
        "once": bool(args.once),
        "state_path": str(args.state_path.resolve()),
        "consistency_path": str(args.consistency_path.resolve()),
        "lock_path": str(args.lock_path.resolve()),
        "latest_session_path": str(args.latest_session_path.resolve()),
        "latest_guard_snapshot_path": str(args.latest_guard_snapshot_path.resolve()),
        "repo_process_allowlist_path": str(args.repo_process_allowlist_path.resolve()),
    }
    write_json(session_dir / "settings.json", settings)
    write_runtime_status(
        session_dir=session_dir,
        latest_session_path=args.latest_session_path,
        status="running",
        reason="watch started",
        cycle_count=0,
        lock_path=args.lock_path,
    )

    cycle_index = 0
    deadline = time.time() + max(args.duration_hours, 0.0) * 3600.0
    final_status = "success"
    final_reason = "completed requested watch window"
    last_cycle_summary = {}

    try:
        while True:
            cycle_index += 1
            update_lock(args.lock_path, cycle_index=cycle_index, status="running")
            write_runtime_status(
                session_dir=session_dir,
                latest_session_path=args.latest_session_path,
                status="running",
                reason="running steady_hold cycle",
                cycle_count=cycle_index - 1,
                lock_path=args.lock_path,
                latest_decision_json=last_cycle_summary.get("nightly_decision_json", ""),
                latest_run_dir=last_cycle_summary.get("latest_run_dir", ""),
            )
            last_cycle_summary = run_cycle(args, session_dir, cycle_index, expected_lead_config)
            update_lock(
                args.lock_path,
                cycle_index=cycle_index,
                status="sleeping",
                latest_decision_json=last_cycle_summary.get("nightly_decision_json", ""),
            )
            write_runtime_status(
                session_dir=session_dir,
                latest_session_path=args.latest_session_path,
                status="running",
                reason="sleeping until the next steady_hold cycle",
                cycle_count=cycle_index,
                lock_path=args.lock_path,
                latest_decision_json=last_cycle_summary.get("nightly_decision_json", ""),
                latest_run_dir=last_cycle_summary.get("latest_run_dir", ""),
            )

            if args.once:
                final_reason = "completed single validation cycle"
                break
            if args.max_cycles > 0 and cycle_index >= args.max_cycles:
                final_reason = "completed max_cycles"
                break
            if time.time() >= deadline:
                final_reason = "completed duration_hours"
                break
            sleep_with_heartbeat(
                lock_path=args.lock_path,
                session_dir=session_dir,
                latest_session_path=args.latest_session_path,
                latest_guard_snapshot_path=args.latest_guard_snapshot_path,
                state_path=args.state_path,
                consistency_path=args.consistency_path,
                cycle_index=cycle_index,
                total_sleep_sec=max(args.interval_sec, 10),
                heartbeat_sec=args.sleep_heartbeat_sec,
                latest_decision_json=last_cycle_summary.get("nightly_decision_json", ""),
                current_lead_config=str(last_cycle_summary.get("current_lead_config", "")),
                expected_lead_config=expected_lead_config,
                repo_process_allowlist_path=args.repo_process_allowlist_path,
            )
    except Exception as exc:
        final_status = "failed"
        final_reason = str(exc)
        raise
    finally:
        payload = {
            "status": final_status,
            "reason": final_reason,
            "finished_at": iso_now(),
            "cycle_count": cycle_index,
            "last_cycle_nightly_decision_json": last_cycle_summary.get("nightly_decision_json", ""),
            "last_cycle_latest_run_dir": last_cycle_summary.get("latest_run_dir", ""),
            "session_dir": str(session_dir.resolve()),
            "lock_path": str(args.lock_path.resolve()),
        }
        write_session_status(session_dir, payload, args.latest_session_path)
        update_lock(
            args.lock_path,
            cycle_index=cycle_index,
            status=final_status,
            latest_decision_json=last_cycle_summary.get("nightly_decision_json", ""),
        )
        release_lock(args.lock_path)

    print(session_dir / "watch_status.json")


if __name__ == "__main__":
    try:
        main()
    except WatchError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
