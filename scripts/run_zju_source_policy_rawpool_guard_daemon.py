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
WATCH_SCRIPT = REPO_ROOT / "scripts" / "run_zju_source_policy_rawpool_overnight_watch.py"
DEFAULT_LOCAL_MANIFEST_PATH = REPO_ROOT / "scripts" / "manifests" / "zju_source_policy_rawpool_local_nightly_v1.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_rawpool_guard_daemon"
DEFAULT_LOCK_PATH = DEFAULT_OUTPUT_ROOT / "active_daemon.json"
DEFAULT_LATEST_SESSION_PATH = DEFAULT_OUTPUT_ROOT / "latest_session.json"
DEFAULT_WATCH_LOCK_PATH = REPO_ROOT / "output" / "zju_source_policy_rawpool_overnight_watch" / "active_watch.json"
DEFAULT_WATCH_LATEST_SESSION_PATH = REPO_ROOT / "output" / "zju_source_policy_rawpool_overnight_watch" / "latest_session.json"


class GuardDaemonError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep the ZJU source-policy rawpool steady_hold watch alive across multiple watch windows."
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--watch-script", type=Path, default=WATCH_SCRIPT)
    parser.add_argument("--local-manifest", type=Path, default=DEFAULT_LOCAL_MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--latest-session-path", type=Path, default=DEFAULT_LATEST_SESSION_PATH)
    parser.add_argument("--watch-lock-path", type=Path, default=DEFAULT_WATCH_LOCK_PATH)
    parser.add_argument("--watch-latest-session-path", type=Path, default=DEFAULT_WATCH_LATEST_SESSION_PATH)
    parser.add_argument("--poll-sec", type=int, default=60)
    parser.add_argument("--watch-interval-sec", type=int, default=1800)
    parser.add_argument("--watch-sleep-heartbeat-sec", type=int, default=60)
    parser.add_argument("--watch-duration-hours", type=float, default=10.0)
    parser.add_argument("--total-duration-hours", type=float, default=72.0, help="0 means unlimited.")
    parser.add_argument("--max-watch-sessions", type=int, default=0, help="0 means unlimited until total-duration-hours is reached.")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
            if info and "run_zju_source_policy_rawpool_guard_daemon.py" in str(info.get("CommandLine", "")):
                raise GuardDaemonError(
                    "Another guard daemon is already active: pid={pid} session_dir={session_dir}".format(
                        pid=prior_pid,
                        session_dir=prior.get("session_dir", ""),
                    )
                )
    payload = {
        "pid": os.getpid(),
        "started_at": iso_now(),
        "session_dir": str(session_dir.resolve()),
        "poll_sec": int(args.poll_sec),
        "watch_duration_hours": float(args.watch_duration_hours),
        "total_duration_hours": float(args.total_duration_hours),
        "max_watch_sessions": int(args.max_watch_sessions),
        "script": str(Path(__file__).resolve()),
        "last_heartbeat": iso_now(),
        "last_status": "starting",
        "watch_session_count": 0,
    }
    write_json(lock_path, payload)


def update_lock(lock_path: Path, *, status: str, watch_session_count: int, last_watch_pid: int = 0, last_watch_session_dir: str = "") -> None:
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
            "last_status": status,
            "watch_session_count": int(watch_session_count),
            "last_watch_pid": int(last_watch_pid),
            "last_watch_session_dir": last_watch_session_dir,
        }
    )
    write_json(lock_path, payload)


def release_lock(lock_path: Path) -> None:
    if not lock_path.exists():
        return
    try:
        payload = load_json(lock_path)
    except Exception:
        payload = {}
    if int(payload.get("pid", 0) or 0) == os.getpid():
        lock_path.unlink(missing_ok=True)


def write_session_status(session_dir: Path, payload: dict, latest_session_path: Path | None = None) -> None:
    write_json(session_dir / "daemon_status.json", payload)
    lines = ["# ZJU Source-Policy Rawpool Guard Daemon", ""]
    for key, value in payload.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    write_text(session_dir / "daemon_status.md", "\n".join(lines))
    if latest_session_path is not None:
        write_json(latest_session_path, payload)


def write_runtime_status(
    *,
    session_dir: Path,
    latest_session_path: Path,
    status: str,
    reason: str,
    watch_session_count: int,
    lock_path: Path,
    last_watch_pid: int = 0,
    last_watch_session_dir: str = "",
) -> None:
    payload = {
        "status": status,
        "reason": reason,
        "updated_at": iso_now(),
        "watch_session_count": int(watch_session_count),
        "last_watch_pid": int(last_watch_pid),
        "last_watch_session_dir": last_watch_session_dir,
        "session_dir": str(session_dir.resolve()),
        "lock_path": str(lock_path.resolve()),
    }
    write_session_status(session_dir, payload, latest_session_path)


def get_active_watch(watch_lock_path: Path) -> dict | None:
    if not watch_lock_path.exists():
        return None
    try:
        payload = load_json(watch_lock_path)
    except Exception:
        return None
    pid = int(payload.get("pid", 0) or 0)
    if pid <= 0:
        return None
    info = process_info(pid)
    if not info:
        return None
    if "run_zju_source_policy_rawpool_overnight_watch.py" not in str(info.get("CommandLine", "")):
        return None
    return payload


def load_watch_latest_session(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}


def should_stop_for_failed_watch(payload: dict) -> bool:
    return str(payload.get("status", "")).lower() == "failed"


def launch_watch(
    *,
    args: argparse.Namespace,
    session_dir: Path,
    watch_session_count: int,
) -> dict:
    launch_dir = ensure_dir(session_dir / f"watch_launch_{watch_session_count:03d}")
    stdout_path = launch_dir / "watch_stdout.txt"
    stderr_path = launch_dir / "watch_stderr.txt"
    cmd = [
        args.python_exe,
        str(args.watch_script),
        "--python-exe",
        args.python_exe,
        "--local-manifest",
        str(args.local_manifest),
        "--interval-sec",
        str(args.watch_interval_sec),
        "--sleep-heartbeat-sec",
        str(args.watch_sleep_heartbeat_sec),
        "--duration-hours",
        str(args.watch_duration_hours),
    ]
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    try:
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=creationflags,
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()
    launch_record = {
        "launched_at": iso_now(),
        "watch_pid": int(proc.pid),
        "watch_command": cmd,
        "stdout_path": str(stdout_path.resolve()),
        "stderr_path": str(stderr_path.resolve()),
    }
    write_json(launch_dir / "launch_record.json", launch_record)
    return launch_record


def main() -> None:
    args = parse_args()
    session_dir = ensure_dir(args.output_root / f"{now_tag()}_guard_daemon")
    acquire_lock(args.lock_path, session_dir, args)
    atexit.register(release_lock, args.lock_path)

    settings = {
        "started_at": iso_now(),
        "pid": os.getpid(),
        "python_exe": args.python_exe,
        "watch_script": str(args.watch_script.resolve()),
        "local_manifest": str(args.local_manifest.resolve()),
        "poll_sec": int(args.poll_sec),
        "watch_interval_sec": int(args.watch_interval_sec),
        "watch_sleep_heartbeat_sec": int(args.watch_sleep_heartbeat_sec),
        "watch_duration_hours": float(args.watch_duration_hours),
        "total_duration_hours": float(args.total_duration_hours),
        "max_watch_sessions": int(args.max_watch_sessions),
        "watch_lock_path": str(args.watch_lock_path.resolve()),
        "watch_latest_session_path": str(args.watch_latest_session_path.resolve()),
        "daemon_lock_path": str(args.lock_path.resolve()),
        "daemon_latest_session_path": str(args.latest_session_path.resolve()),
    }
    write_json(session_dir / "settings.json", settings)

    total_deadline = None
    if args.total_duration_hours > 0:
        total_deadline = time.time() + args.total_duration_hours * 3600.0

    watch_session_count = 0
    final_status = "success"
    final_reason = "completed requested daemon window"
    last_watch_pid = 0
    last_watch_session_dir = ""

    write_runtime_status(
        session_dir=session_dir,
        latest_session_path=args.latest_session_path,
        status="running",
        reason="guard daemon started",
        watch_session_count=watch_session_count,
        lock_path=args.lock_path,
    )

    try:
        while True:
            active_watch = get_active_watch(args.watch_lock_path)
            if active_watch is not None:
                last_watch_pid = int(active_watch.get("pid", 0) or 0)
                last_watch_session_dir = str(active_watch.get("session_dir", "") or "")
                update_lock(
                    args.lock_path,
                    status="monitoring_active_watch",
                    watch_session_count=watch_session_count,
                    last_watch_pid=last_watch_pid,
                    last_watch_session_dir=last_watch_session_dir,
                )
                write_runtime_status(
                    session_dir=session_dir,
                    latest_session_path=args.latest_session_path,
                    status="running",
                    reason="monitoring active steady_hold watch",
                    watch_session_count=watch_session_count,
                    lock_path=args.lock_path,
                    last_watch_pid=last_watch_pid,
                    last_watch_session_dir=last_watch_session_dir,
                )
                time.sleep(max(int(args.poll_sec), 5))
                continue

            latest_watch = load_watch_latest_session(args.watch_latest_session_path)
            if should_stop_for_failed_watch(latest_watch):
                final_status = "failed"
                final_reason = "stopped after watch failure: {reason}".format(
                    reason=latest_watch.get("reason", "unknown watch failure")
                )
                break

            if args.max_watch_sessions > 0 and watch_session_count >= args.max_watch_sessions:
                final_reason = "completed max_watch_sessions"
                break

            if total_deadline is not None and time.time() >= total_deadline:
                final_reason = "completed total_duration_hours"
                break

            watch_session_count += 1
            launch_record = launch_watch(
                args=args,
                session_dir=session_dir,
                watch_session_count=watch_session_count,
            )
            last_watch_pid = int(launch_record["watch_pid"])
            last_watch_session_dir = ""
            update_lock(
                args.lock_path,
                status="launched_new_watch",
                watch_session_count=watch_session_count,
                last_watch_pid=last_watch_pid,
                last_watch_session_dir=last_watch_session_dir,
            )
            write_runtime_status(
                session_dir=session_dir,
                latest_session_path=args.latest_session_path,
                status="running",
                reason="launched new steady_hold watch",
                watch_session_count=watch_session_count,
                lock_path=args.lock_path,
                last_watch_pid=last_watch_pid,
                last_watch_session_dir=last_watch_session_dir,
            )
            time.sleep(max(int(args.poll_sec), 5))
    except Exception as exc:
        final_status = "failed"
        final_reason = str(exc)
        raise
    finally:
        payload = {
            "status": final_status,
            "reason": final_reason,
            "finished_at": iso_now(),
            "watch_session_count": int(watch_session_count),
            "last_watch_pid": int(last_watch_pid),
            "last_watch_session_dir": last_watch_session_dir,
            "session_dir": str(session_dir.resolve()),
            "lock_path": str(args.lock_path.resolve()),
        }
        write_session_status(session_dir, payload, args.latest_session_path)
        update_lock(
            args.lock_path,
            status=final_status,
            watch_session_count=watch_session_count,
            last_watch_pid=last_watch_pid,
            last_watch_session_dir=last_watch_session_dir,
        )
        release_lock(args.lock_path)

    print(session_dir / "daemon_status.json")


if __name__ == "__main__":
    try:
        main()
    except GuardDaemonError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
