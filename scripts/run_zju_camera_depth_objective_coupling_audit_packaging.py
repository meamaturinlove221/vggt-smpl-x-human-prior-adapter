import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "packaging_runs"
ALLOWLIST_JSON = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "repo_process_allowlist.json"
WATCH_SCRIPT = REPO_ROOT / "scripts" / "run_zju_source_policy_research_watch.py"
RESEARCH_LOOP_SCRIPT = REPO_ROOT / "scripts" / "run_zju_source_policy_research_loop.py"

STEPS = [
    REPO_ROOT / "scripts" / "write_zju_camera_depth_objective_coupling_audit_postmortem.py",
    REPO_ROOT / "scripts" / "validate_zju_camera_depth_objective_coupling_audit_execution_prep_baseline.py",
    REPO_ROOT / "scripts" / "package_zju_camera_depth_objective_coupling_audit_execution_ready.py",
]
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync_zju_camera_depth_objective_coupling_audit_ready.py"
ARM_SCRIPT = REPO_ROOT / "scripts" / "arm_zju_source_policy_approved_problem.py"


class PackagingError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package camera_depth_objective_coupling_audit into execution-ready pending manual arm."
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def run_checked(args: list[str], *, cwd: Path, stdout_path: Path, stderr_path: Path) -> None:
    result = subprocess.run(
        args,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    write_text(stdout_path, result.stdout)
    write_text(stderr_path, result.stderr)
    if result.returncode != 0:
        raise PackagingError(
            "Command failed with exit code {code}: {cmd}\nSTDERR:\n{stderr}".format(
                code=result.returncode,
                cmd=" ".join(args),
                stderr=result.stderr.strip(),
            )
        )


def verify_cleanup() -> dict:
    allowlist = load_json(ALLOWLIST_JSON)
    if str(allowlist.get("status", "")).strip() != "idle_empty_allowlist":
        raise PackagingError("repo_process_allowlist.json is not idle_empty_allowlist after packaging.")

    modal = subprocess.run(
        ["modal", "app", "list", "--json"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if modal.returncode != 0:
        raise PackagingError(f"modal app list --json failed during cleanup verification: {modal.stderr.strip()}")
    apps = json.loads(modal.stdout.strip() or "[]")
    active_apps = [row for row in apps if str(row.get("State", "")).lower() != "stopped"]
    if active_apps:
        raise PackagingError("Modal apps are still active after packaging; cleanup verification failed.")

    return {
        "allowlist_status": allowlist.get("status", ""),
        "active_modal_app_count": len(active_apps),
    }


def main() -> int:
    args = parse_args()
    run_dir = ensure_dir(args.output_root / f"{now_tag()}_camera_depth_objective_coupling_audit")

    records = []
    for step in STEPS:
        stem = step.stem
        run_checked(
            [args.python_exe, str(step)],
            cwd=REPO_ROOT,
            stdout_path=run_dir / f"{stem}.stdout.txt",
            stderr_path=run_dir / f"{stem}.stderr.txt",
        )
        records.append({"step": stem, "status": "pass"})

    run_checked(
        [args.python_exe, str(RESEARCH_LOOP_SCRIPT)],
        cwd=REPO_ROOT,
        stdout_path=run_dir / "research_loop_refresh.stdout.txt",
        stderr_path=run_dir / "research_loop_refresh.stderr.txt",
    )
    records.append({"step": "research_loop_refresh", "status": "pass"})

    run_checked(
        [args.python_exe, str(ARM_SCRIPT), "--seed", "camera_depth_objective_coupling_audit", "--dry-run"],
        cwd=REPO_ROOT,
        stdout_path=run_dir / "arm_dry_run.stdout.txt",
        stderr_path=run_dir / "arm_dry_run.stderr.txt",
    )
    records.append({"step": "arm_dry_run", "status": "pass"})

    run_checked(
        [args.python_exe, str(SYNC_SCRIPT)],
        cwd=REPO_ROOT,
        stdout_path=run_dir / "sync_ready.stdout.txt",
        stderr_path=run_dir / "sync_ready.stderr.txt",
    )
    records.append({"step": "sync_ready", "status": "pass"})

    run_checked(
        [args.python_exe, str(WATCH_SCRIPT), "--once"],
        cwd=REPO_ROOT,
        stdout_path=run_dir / "watch_once.stdout.txt",
        stderr_path=run_dir / "watch_once.stderr.txt",
    )
    records.append({"step": "watch_once", "status": "pass"})

    cleanup = verify_cleanup()
    payload = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "family": "camera_depth_objective_coupling_audit",
        "status": "PACKAGED_READY_FOR_MANUAL_ARM",
        "run_dir": str(run_dir.resolve()),
        "steps": records,
        "cleanup_verification": cleanup,
    }
    write_json(run_dir / "packaging_status.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
