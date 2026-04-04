import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modal_zju_visual_lift_benchmark import run_local_contract_rehearsal
OUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the teacher-fixed visual-lift cloud contract audit and local rehearsal.")
    parser.add_argument("--date-tag", default="20260404")
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _poll_rehearsal(output_root: Path, should_fail: bool) -> dict:
    result_holder: dict = {}

    def _runner() -> None:
        result_holder["result"] = run_local_contract_rehearsal(
            str(output_root),
            should_fail=should_fail,
            heartbeat_interval_sec=1,
        )

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()

    launched_seen = False
    manifest_seen = False
    stdout_seen = False
    heartbeat_count_during_run = 0
    while thread.is_alive():
        launched_seen = launched_seen or (output_root / "cloud_status.json").exists()
        manifest_seen = manifest_seen or (output_root / "manifest.copy.json").exists()
        stdout_seen = stdout_seen or (output_root / "stdout.log").exists()
        heartbeat_history = output_root / "heartbeat.history.jsonl"
        if heartbeat_history.exists():
            heartbeat_count_during_run = max(
                heartbeat_count_during_run,
                len([line for line in heartbeat_history.read_text(encoding="utf-8").splitlines() if line.strip()]),
            )
        time.sleep(0.4)
    thread.join()

    heartbeat_history = output_root / "heartbeat.history.jsonl"
    heartbeat_rows = []
    if heartbeat_history.exists():
        heartbeat_rows = [json.loads(line) for line in heartbeat_history.read_text(encoding="utf-8").splitlines() if line.strip()]
    final_status = _load_json(output_root / "cloud_status.json")
    timing = _load_json(output_root / "timing.json")
    command = _load_json(output_root / "command.json")
    success_path = output_root / "success.json"
    exception_path = output_root / "exception.json"
    return {
        "output_root": str(output_root),
        "expected_failure": bool(should_fail),
        "launched_seen_during_run": bool(launched_seen),
        "manifest_seen_during_run": bool(manifest_seen),
        "stdout_seen_during_run": bool(stdout_seen),
        "heartbeat_count_during_run": int(heartbeat_count_during_run),
        "heartbeat_count_final": int(len(heartbeat_rows)),
        "final_status": final_status,
        "timing": timing,
        "command": command,
        "success_json_present": success_path.exists(),
        "exception_json_present": exception_path.exists(),
        "stdout_log_size": (output_root / "stdout.log").stat().st_size if (output_root / "stdout.log").exists() else 0,
    }


def main() -> int:
    args = parse_args()
    checked_at = datetime.now().astimezone().isoformat()
    date_tag = str(args.date_tag)

    audit_result_json = OUT_ROOT / f"cloud_contract_audit_result.teacher_fixed_visual_lift_benchmark.{date_tag}.json"
    audit_result_md = OUT_ROOT / f"cloud_contract_audit_result.teacher_fixed_visual_lift_benchmark.{date_tag}.md"
    patch_plan_json = OUT_ROOT / f"cloud_contract_patch_plan.teacher_fixed_visual_lift_benchmark.{date_tag}.json"
    patch_plan_md = OUT_ROOT / f"cloud_contract_patch_plan.teacher_fixed_visual_lift_benchmark.{date_tag}.md"
    rehearsal_json = OUT_ROOT / f"cloud_contract_rehearsal_report.teacher_fixed_visual_lift_benchmark.{date_tag}.json"
    rehearsal_md = OUT_ROOT / f"cloud_contract_rehearsal_report.teacher_fixed_visual_lift_benchmark.{date_tag}.md"
    go_no_go_json = OUT_ROOT / f"cloud_contract_go_no_go.teacher_fixed_visual_lift_benchmark.{date_tag}.json"
    go_no_go_md = OUT_ROOT / f"cloud_contract_go_no_go.teacher_fixed_visual_lift_benchmark.{date_tag}.md"

    patch_plan = {
        "checked_at": checked_at,
        "family": "teacher_fixed_visual_lift_benchmark_cloud_contract_audit",
        "patch_items": [
            "Write output_root artifacts before subprocess launch and commit immediately.",
            "Create manifest.copy.json, cloud_status.json, heartbeat.json, heartbeat.history.jsonl, stdout.log, command.json, timing.json, and resolved_checkpoint_path.txt at launch.",
            "Run a monitored subprocess that updates heartbeat and commits while stdout is still streaming.",
            "Force final success.json or exception.json plus cloud_status/timing/heartbeat in finally.",
            "Expose a local no-cloud rehearsal path that verifies both success and failure preservation.",
            "Teach evaluate_teacher_visual_lift_cases.py to write progress.json and summary.partial.json after every case.",
        ],
        "files_touched": [
            "modal_zju_visual_lift_benchmark.py",
            "scripts/evaluate_teacher_visual_lift_cases.py",
            "scripts/run_teacher_fixed_visual_lift_cloud_contract_audit.py",
        ],
    }
    _write_json(patch_plan_json, patch_plan)
    _write_md(patch_plan_md, json.dumps(patch_plan, ensure_ascii=False, indent=2))

    rehearsal_root = REPO_ROOT / "output" / "teacher_fixed_visual_lift_benchmark" / f"cloud_contract_rehearsal.{date_tag}"
    success_root = rehearsal_root / "success"
    failure_root = rehearsal_root / "failure"
    success_result = _poll_rehearsal(success_root, should_fail=False)
    failure_result = _poll_rehearsal(failure_root, should_fail=True)

    rehearsal_report = {
        "checked_at": checked_at,
        "family": "teacher_fixed_visual_lift_benchmark_cloud_contract_audit",
        "success_rehearsal": success_result,
        "failure_rehearsal": failure_result,
    }
    _write_json(rehearsal_json, rehearsal_report)
    _write_md(rehearsal_md, json.dumps(rehearsal_report, ensure_ascii=False, indent=2))

    go_no_go = {
        "checked_at": checked_at,
        "family": "teacher_fixed_visual_lift_benchmark_cloud_contract_audit",
        "go": bool(
            success_result["launched_seen_during_run"]
            and success_result["manifest_seen_during_run"]
            and success_result["stdout_seen_during_run"]
            and success_result["heartbeat_count_final"] >= 3
            and success_result["success_json_present"]
            and failure_result["launched_seen_during_run"]
            and failure_result["manifest_seen_during_run"]
            and failure_result["stdout_seen_during_run"]
            and failure_result["heartbeat_count_final"] >= 3
            and failure_result["exception_json_present"]
        ),
        "reason": "GO" if (
            success_result["heartbeat_count_final"] >= 3
            and success_result["success_json_present"]
            and failure_result["heartbeat_count_final"] >= 3
            and failure_result["exception_json_present"]
        ) else "NO_GO",
    }
    _write_json(go_no_go_json, go_no_go)
    _write_md(go_no_go_md, json.dumps(go_no_go, ensure_ascii=False, indent=2))

    audit_result = {
        "checked_at": checked_at,
        "family": "teacher_fixed_visual_lift_benchmark_cloud_contract_audit",
        "status": "passed" if go_no_go["go"] else "failed",
        "go_no_go_path": str(go_no_go_json.relative_to(REPO_ROOT)),
        "rehearsal_report_path": str(rehearsal_json.relative_to(REPO_ROOT)),
        "patch_plan_path": str(patch_plan_json.relative_to(REPO_ROOT)),
        "summary": (
            "The cloud execution contract now materializes launch artifacts immediately, updates heartbeat while running, and preserves final success/failure state locally without opening any cloud app."
            if go_no_go["go"]
            else "The local rehearsal still failed to prove one or more required contract guarantees."
        ),
    }
    _write_json(audit_result_json, audit_result)
    _write_md(audit_result_md, json.dumps(audit_result, ensure_ascii=False, indent=2))

    print(audit_result_json)
    print(rehearsal_json)
    print(go_no_go_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
