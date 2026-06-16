import argparse
import atexit
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAUNCHER_SCRIPT = REPO_ROOT / "scripts" / "run_modal_zju_geometry_smplprior_headhair_longrun.ps1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "modal_zju_geometry_guard_daemon"
DEFAULT_LOCK_PATH = DEFAULT_OUTPUT_ROOT / "active_guard.json"
DEFAULT_LATEST_SESSION_PATH = DEFAULT_OUTPUT_ROOT / "latest_session.json"
DEFAULT_MODAL_APP_NAME = "vggt-zju-geometry-smplprior-headhair"
DEFAULT_CHECKPOINT_SUBPATH = (
    "20260415_smplprior_headhair_longrun_eager_shmsafe_resume2/ckpts/checkpoint_6.pt"
)
DEFAULT_OUTPUT_SUBDIR = "20260416_smplprior_headhair_longrun_eager_shmsafe_resume4_guarded"
DEFAULT_OUTPUT_VOLUME = "vggt-out"
STOPPED_STATES = {"stopped", "stopping", "completed", "failed"}
PROGRESS_RE = re.compile(
    r"^INFO (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*?"
    r"(?P<phase>Train|Val) Epoch: \[(?P<epoch>\d+)\]\[\s*(?P<step>\d+)/(?P<total>\d+)\]"
)
CHECKPOINT_RE = re.compile(r"Saving checkpoint at epoch (?P<epoch>\d+) to (?P<path>\S+)")
ERROR_MARKERS = ("traceback", "runtimeerror", "exception", "keyboardinterrupt", "cuda error")


class GuardError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch and watch a detached Modal ZJU geometry training run."
    )
    parser.add_argument("--modal-exe", default="")
    parser.add_argument("--launcher-script", type=Path, default=DEFAULT_LAUNCHER_SCRIPT)
    parser.add_argument("--modal-app-name", default=DEFAULT_MODAL_APP_NAME)
    parser.add_argument("--checkpoint-subpath", default=DEFAULT_CHECKPOINT_SUBPATH)
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--output-volume", default=DEFAULT_OUTPUT_VOLUME)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--latest-session-path", type=Path, default=DEFAULT_LATEST_SESSION_PATH)
    parser.add_argument("--poll-sec", type=int, default=300)
    parser.add_argument("--launch-poll-sec", type=int, default=20)
    parser.add_argument("--launch-timeout-sec", type=int, default=1800)
    parser.add_argument("--stale-hours", type=float, default=3.0)
    parser.add_argument("--remote-tail-bytes", type=int, default=1_000_000)
    parser.add_argument("--launcher-tail-bytes", type=int, default=1_000_000)
    parser.add_argument("--status-name", default="smplprior_headhair_resume_guard")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def resolve_modal_exe(preferred: str) -> str:
    if preferred and Path(preferred).exists():
        return str(Path(preferred).resolve())
    candidates = (
        REPO_ROOT / ".venv5080" / "Scripts" / "modal.exe",
        REPO_ROOT / ".venv" / "Scripts" / "modal.exe",
        REPO_ROOT / "venv" / "Scripts" / "modal.exe",
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return "modal"


def run_cmd(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        args,
        cwd=str(cwd or REPO_ROOT),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    result = run_cmd(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"$p = Get-Process -Id {pid} -ErrorAction SilentlyContinue; if ($p) {{ 'alive' }}",
        ]
    )
    return result.stdout.strip() == "alive"


def acquire_lock(lock_path: Path, session_dir: Path, args: argparse.Namespace) -> None:
    if lock_path.exists():
        try:
            payload = load_json(lock_path)
        except Exception:
            payload = {}
        prior_pid = int(payload.get("pid", 0) or 0)
        if prior_pid and prior_pid != os.getpid() and pid_alive(prior_pid):
            raise GuardError(
                "Another modal guard daemon is already active: "
                f"pid={prior_pid} session_dir={payload.get('session_dir', '')}"
            )
    write_json(
        lock_path,
        {
            "pid": os.getpid(),
            "started_at": iso_now(),
            "session_dir": str(session_dir.resolve()),
            "status_name": args.status_name,
            "output_subdir": args.output_subdir,
            "checkpoint_subpath": args.checkpoint_subpath,
            "last_heartbeat": iso_now(),
            "last_status": "starting",
        },
    )


def update_lock(
    lock_path: Path,
    *,
    status: str,
    launcher_pid: int = 0,
    primary_app_id: str = "",
) -> None:
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
            "launcher_pid": int(launcher_pid),
            "primary_app_id": primary_app_id,
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


def parse_app_time(value: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def list_modal_apps(modal_exe: str, description: str) -> list[dict]:
    result = run_cmd([modal_exe, "app", "list", "--json"])
    if result.returncode != 0:
        raise GuardError("modal app list failed: " + result.stderr.strip())
    raw = result.stdout.strip() or "[]"
    payload = json.loads(raw)
    items = payload if isinstance(payload, list) else [payload]
    return [item for item in items if str(item.get("Description", "")) == description]


def active_modal_apps(modal_exe: str, description: str) -> list[dict]:
    items = [
        item
        for item in list_modal_apps(modal_exe, description)
        if str(item.get("State", "")).strip().lower() not in STOPPED_STATES
    ]
    return sorted(items, key=lambda item: parse_app_time(str(item.get("Created at", ""))), reverse=True)


def stop_modal_app(modal_exe: str, app_id: str) -> None:
    result = run_cmd([modal_exe, "app", "stop", app_id])
    if result.returncode != 0:
        raise GuardError(f"modal app stop failed for {app_id}: {result.stderr.strip()}")


def stop_redundant_apps(modal_exe: str, apps: list[dict]) -> list[str]:
    if len(apps) <= 1:
        return []
    stopped = []
    for app in apps[1:]:
        app_id = str(app.get("App ID", "")).strip()
        if not app_id:
            continue
        stop_modal_app(modal_exe, app_id)
        stopped.append(app_id)
    return stopped


def kill_process_tree(pid: int) -> None:
    if pid <= 0:
        return
    run_cmd(["taskkill", "/PID", str(pid), "/T", "/F"])


def safe_wait(proc: subprocess.Popen[str], timeout: int = 30) -> None:
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass


def launch_launcher(
    args: argparse.Namespace,
    modal_exe: str,
    session_dir: Path,
) -> tuple[subprocess.Popen[str], Path, Path]:
    stdout_path = session_dir / "launcher_stdout.txt"
    stderr_path = session_dir / "launcher_stderr.txt"
    cmd = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(args.launcher_script),
        "-ModalExe",
        modal_exe,
        "-ModalAppName",
        args.modal_app_name,
        "-CheckpointSubpath",
        args.checkpoint_subpath,
        "-OutputSubdir",
        args.output_subdir,
        "-DisableCompile",
        "-StopExistingApps",
        "-SkipPreflight",
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
            text=True,
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()
    write_json(
        session_dir / "launch_record.json",
        {
            "launched_at": iso_now(),
            "launcher_pid": int(proc.pid),
            "command": cmd,
            "stdout_path": str(stdout_path.resolve()),
            "stderr_path": str(stderr_path.resolve()),
        },
    )
    return proc, stdout_path, stderr_path


def modal_volume_get(modal_exe: str, volume_name: str, remote_path: str, local_path: Path) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.unlink(missing_ok=True)
    result = run_cmd([modal_exe, "volume", "get", "--force", volume_name, remote_path, str(local_path)])
    return result.returncode == 0 or local_path.exists()


def read_tail_text(path: Path, max_bytes: int) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(size - max_bytes, 0), os.SEEK_SET)
        return handle.read().decode("utf-8", errors="replace")


def parse_progress_snapshot(text: str) -> dict:
    latest_train = None
    latest_val = None
    latest_checkpoint = None
    error_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = PROGRESS_RE.match(line)
        if match:
            payload = {
                "line": line,
                "timestamp_utc": match.group("ts"),
                "phase": match.group("phase"),
                "epoch": int(match.group("epoch")),
                "step": int(match.group("step")),
                "total": int(match.group("total")),
            }
            if payload["phase"] == "Train":
                latest_train = payload
            else:
                latest_val = payload

        checkpoint_match = CHECKPOINT_RE.search(line)
        if checkpoint_match:
            latest_checkpoint = {
                "line": line,
                "epoch": int(checkpoint_match.group("epoch")),
                "path": checkpoint_match.group("path"),
            }

        lowered = line.lower()
        if any(marker in lowered for marker in ERROR_MARKERS):
            error_lines.append(line)

    latest_progress = latest_train or latest_val
    if latest_train and latest_val:
        train_ts = datetime.strptime(latest_train["timestamp_utc"], "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
        val_ts = datetime.strptime(latest_val["timestamp_utc"], "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
        latest_progress = latest_train if train_ts >= val_ts else latest_val

    return {
        "latest_train": latest_train,
        "latest_val": latest_val,
        "latest_progress": latest_progress,
        "latest_checkpoint": latest_checkpoint,
        "error_lines": error_lines[-20:],
    }


def progress_key(payload: dict | None) -> tuple | None:
    if not payload:
        return None
    return (
        payload.get("phase"),
        payload.get("epoch"),
        payload.get("step"),
        payload.get("timestamp_utc"),
    )


def write_status(session_dir: Path, latest_session_path: Path, payload: dict) -> None:
    write_json(session_dir / "status.json", payload)
    lines = ["# Modal ZJU Geometry Guard", ""]
    for key, value in payload.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    write_text(session_dir / "status.md", "\n".join(lines))
    write_json(latest_session_path, payload)


def wait_for_active_app(
    *,
    modal_exe: str,
    description: str,
    timeout_sec: int,
    poll_sec: int,
) -> list[dict]:
    deadline = time.monotonic() + max(timeout_sec, 1)
    while time.monotonic() < deadline:
        apps = active_modal_apps(modal_exe, description)
        if apps:
            return apps
        time.sleep(max(poll_sec, 5))
    return active_modal_apps(modal_exe, description)


def main() -> int:
    args = parse_args()
    modal_exe = resolve_modal_exe(args.modal_exe)
    session_dir = ensure_dir(args.output_root / f"{now_tag()}_{args.status_name}")
    remote_dir = ensure_dir(session_dir / "remote_pull")

    acquire_lock(args.lock_path, session_dir, args)
    atexit.register(release_lock, args.lock_path)

    settings = {
        "started_at": iso_now(),
        "pid": os.getpid(),
        "status_name": args.status_name,
        "modal_exe": modal_exe,
        "launcher_script": str(args.launcher_script.resolve()),
        "modal_app_name": args.modal_app_name,
        "checkpoint_subpath": args.checkpoint_subpath,
        "output_subdir": args.output_subdir,
        "output_volume": args.output_volume,
        "poll_sec": int(args.poll_sec),
        "launch_timeout_sec": int(args.launch_timeout_sec),
        "stale_hours": float(args.stale_hours),
        "session_dir": str(session_dir.resolve()),
        "lock_path": str(args.lock_path.resolve()),
    }
    write_json(session_dir / "settings.json", settings)

    launch_started_monotonic = time.monotonic()
    launcher_proc, launcher_stdout_path, launcher_stderr_path = launch_launcher(args, modal_exe, session_dir)
    update_lock(args.lock_path, status="launching", launcher_pid=launcher_proc.pid)

    apps = wait_for_active_app(
        modal_exe=modal_exe,
        description=args.modal_app_name,
        timeout_sec=args.launch_timeout_sec,
        poll_sec=args.launch_poll_sec,
    )
    stopped_redundant = stop_redundant_apps(modal_exe, apps) if apps else []
    if apps:
        apps = active_modal_apps(modal_exe, args.modal_app_name)
    saw_active_app_once = bool(apps)

    latest_progress_key = None
    latest_progress_seen_utc = utc_now()
    latest_progress_source = ""
    latest_checkpoint = None
    last_error_lines: list[str] = []
    last_reason = "launched"

    while True:
        primary_app_id = str(apps[0].get("App ID", "")).strip() if apps else ""
        update_lock(
            args.lock_path,
            status="watching" if apps else "waiting",
            launcher_pid=launcher_proc.pid,
            primary_app_id=primary_app_id,
        )

        remote_log_path = remote_dir / "log.txt"
        remote_driver_path = remote_dir / "driver_live.log"
        remote_log_ok = modal_volume_get(
            modal_exe,
            args.output_volume,
            f"/{args.output_subdir}/logs/log.txt",
            remote_log_path,
        )
        remote_driver_ok = modal_volume_get(
            modal_exe,
            args.output_volume,
            f"/{args.output_subdir}/driver_live.log",
            remote_driver_path,
        )

        snapshot = parse_progress_snapshot(read_tail_text(remote_log_path, args.remote_tail_bytes))
        progress_source = "remote_log" if snapshot["latest_progress"] else ""
        if not snapshot["latest_progress"]:
            snapshot = parse_progress_snapshot(read_tail_text(launcher_stdout_path, args.launcher_tail_bytes))
            progress_source = "launcher_stdout" if snapshot["latest_progress"] else ""

        current_progress = snapshot["latest_progress"]
        current_progress_key = progress_key(current_progress)
        if current_progress_key and current_progress_key != latest_progress_key:
            latest_progress_key = current_progress_key
            latest_progress_seen_utc = utc_now()
            latest_progress_source = progress_source
        if snapshot["latest_checkpoint"]:
            latest_checkpoint = snapshot["latest_checkpoint"]
        if snapshot["error_lines"]:
            last_error_lines = snapshot["error_lines"]

        try:
            apps = active_modal_apps(modal_exe, args.modal_app_name)
        except GuardError as exc:
            last_reason = str(exc)
            apps = []

        stopped_now = stop_redundant_apps(modal_exe, apps) if apps else []
        if stopped_now:
            stopped_redundant.extend(stopped_now)
            apps = active_modal_apps(modal_exe, args.modal_app_name)
        if apps:
            saw_active_app_once = True

        launcher_running = launcher_proc.poll() is None
        primary_app = apps[0] if apps else {}
        primary_app_id = str(primary_app.get("App ID", "")).strip()
        progress_age_hours = round((utc_now() - latest_progress_seen_utc).total_seconds() / 3600.0, 4)

        status = "running" if apps else "waiting_for_app"
        reason = last_reason

        if not apps and not launcher_running and saw_active_app_once:
            status = "stopped"
            reason = "active Modal app has ended after launcher exit"
        elif apps and progress_age_hours >= args.stale_hours:
            status = "stale_stopped"
            reason = (
                f"latest progress has been unchanged for {progress_age_hours:.2f}h; "
                "stopping active Modal app(s)"
            )
            for app in apps:
                app_id = str(app.get("App ID", "")).strip()
                if app_id:
                    stop_modal_app(modal_exe, app_id)
            if launcher_running:
                kill_process_tree(launcher_proc.pid)
                safe_wait(launcher_proc, timeout=30)
            apps = []
        elif not apps and (time.monotonic() - launch_started_monotonic) > args.launch_timeout_sec:
            status = "launch_timeout"
            reason = "no active Modal app appeared before the launch timeout"
            if launcher_running:
                kill_process_tree(launcher_proc.pid)
                safe_wait(launcher_proc, timeout=30)
        else:
            status = "running" if apps else "waiting_for_app"
            reason = "watching active Modal app progress" if apps else "waiting for Modal app activation"

        payload = {
            "updated_at": iso_now(),
            "status": status,
            "reason": reason,
            "session_dir": str(session_dir.resolve()),
            "launcher_pid": int(launcher_proc.pid),
            "launcher_running": launcher_running,
            "launcher_returncode": launcher_proc.poll(),
            "launcher_stdout_path": str(launcher_stdout_path.resolve()),
            "launcher_stderr_path": str(launcher_stderr_path.resolve()),
            "remote_log_path": str(remote_log_path.resolve()) if remote_log_ok else "",
            "remote_driver_path": str(remote_driver_path.resolve()) if remote_driver_ok else "",
            "modal_app_name": args.modal_app_name,
            "primary_app_id": primary_app_id,
            "active_app_ids": [str(app.get("App ID", "")).strip() for app in apps],
            "active_app_states": [str(app.get("State", "")).strip() for app in apps],
            "stopped_redundant_app_ids": stopped_redundant,
            "output_subdir": args.output_subdir,
            "checkpoint_subpath": args.checkpoint_subpath,
            "latest_progress_source": latest_progress_source,
            "latest_progress_seen_utc": latest_progress_seen_utc.isoformat(timespec="seconds"),
            "progress_age_hours": progress_age_hours,
            "latest_train": snapshot["latest_train"],
            "latest_val": snapshot["latest_val"],
            "latest_checkpoint": latest_checkpoint,
            "last_error_lines": last_error_lines[-10:],
        }
        write_status(session_dir, args.latest_session_path, payload)

        if status in {"stopped", "stale_stopped", "launch_timeout"}:
            break

        time.sleep(max(args.poll_sec, 30))

    update_lock(
        args.lock_path,
        status=payload["status"],
        launcher_pid=launcher_proc.pid,
        primary_app_id=payload["primary_app_id"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
