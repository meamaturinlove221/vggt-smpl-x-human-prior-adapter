import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
OUTPUT_ROOT = RESEARCH_ROOT / "execution_ready_pending_arm_guard_only"

RESEARCH_STATUS_JSON = RESEARCH_ROOT / "research_loop_status.json"
ALLOWLIST_JSON = RESEARCH_ROOT / "repo_process_allowlist.json"
APPROVED_PROBLEM_JSON = RESEARCH_ROOT / "approved_problem.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guard-only watcher for execution-ready pending-arm families."
    )
    parser.add_argument("--family", default="camera_depth_objective_coupling_audit")
    parser.add_argument("--python-exe", default=sys.executable)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


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


def list_modal_apps() -> list[dict]:
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


def render_md(payload: dict) -> str:
    return f"# Execution-Ready Pending-Arm Guard Report\n\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"


def main() -> int:
    args = parse_args()
    checked_at = now_iso()
    research = load_json(RESEARCH_STATUS_JSON)
    allowlist = load_json(ALLOWLIST_JSON)
    approved_problem = maybe_load_json(APPROVED_PROBLEM_JSON)
    active_modal_apps = list_modal_apps()
    latest_formal_verdict = research.get("latest_formal_verdict", {}) or {}

    pending_arm_ok = (
        bool(research.get("ready_for_execution"))
        and str(research.get("manual_action_kind", "")).strip() == "manual_approval"
        and not bool(research.get("approved_problem_present"))
        and bool(research.get("cloud_must_remain_off"))
        and str(research.get("current_priority_family", "")).strip() == args.family
    )

    family_closed = (
        str(latest_formal_verdict.get("family", "")).strip() == args.family
        and str(latest_formal_verdict.get("status", "")).strip() in {"dead_same_day", "failed_long_gate", "provisional_lead"}
    )

    status = "PENDING_ARM_GUARD_OK" if pending_arm_ok else ("FAMILY_NOT_PENDING_ARM_CURRENTLY" if family_closed else "DRIFT_DETECTED")
    payload = {
        "checked_at": checked_at,
        "family": args.family,
        "status": status,
        "pending_arm_ok": pending_arm_ok,
        "family_closed_for_round": family_closed,
        "truth": {
            "state": research.get("state", ""),
            "ready_for_execution": bool(research.get("ready_for_execution")),
            "manual_action_kind": research.get("manual_action_kind", ""),
            "approved_problem_present": bool(research.get("approved_problem_present")),
            "current_priority_family": research.get("current_priority_family", ""),
            "cloud_must_remain_off": bool(research.get("cloud_must_remain_off")),
            "latest_formal_verdict": latest_formal_verdict,
        },
        "guard_checks": {
            "allowlist_status": allowlist.get("status", ""),
            "allowlist_empty": len(allowlist.get("allowed_markers", []) or []) == 0,
            "approved_problem_absent": not bool(approved_problem),
            "active_modal_app_count": len(active_modal_apps),
        },
        "note": (
            "The target family is still correctly held in execution-ready pending-arm."
            if pending_arm_ok
            else (
                "The target family is not pending-arm anymore; it already has a formal local verdict, so this guard now acts as a drift detector rather than a hold-open watcher."
                if family_closed
                else "The target family is neither pending-arm nor cleanly closed; inspect live truth before any further automation."
            )
        ),
    }

    report_json = OUTPUT_ROOT / f"latest_report.{args.family}.json"
    report_md = OUTPUT_ROOT / f"latest_report.{args.family}.md"
    write_json(report_json, payload)
    write_text(report_md, render_md(payload))
    print(json.dumps({"report_json": str(report_json.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
