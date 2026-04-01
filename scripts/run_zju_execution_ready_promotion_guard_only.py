import argparse
import hashlib
import importlib
import inspect
import json
import subprocess
import sys
import types
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
OUTPUT_ROOT = RESEARCH_ROOT / "execution_ready_promotion_guard_only"
LOSS_PY = REPO_ROOT / "training" / "loss.py"
RESEARCH_STATUS = RESEARCH_ROOT / "research_loop_status.json"
TASK_PLAN = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current" / "task_plan.json"
LATEST_WATCH = REPO_ROOT / "output" / "zju_source_policy_research_watch" / "latest_watch_snapshot.json"
LATEST_GUARD = REPO_ROOT / "output" / "zju_source_policy_rawpool_overnight_watch" / "latest_guard_snapshot.json"
PROMOTION_DECISION = RESEARCH_ROOT / "execution_ready_promotion_decision.camera_focal_objective_isolation.20260401.json"
SEED_JSON = RESEARCH_ROOT / "approved_problem.seed.camera_focal_objective_isolation.json"
BLUEPRINT_JSON = RESEARCH_ROOT / "family_blueprint.camera_focal_objective_isolation.json"
FAMILY_PLAN_JSON = RESEARCH_ROOT / "candidate_patch_plan.camera_focal_objective_isolation.json"

WRITERS = [
    REPO_ROOT / "scripts" / "write_zju_execution_ready_promotion_decision.py",
    REPO_ROOT / "scripts" / "sync_zju_execution_ready_promotion.py",
]

EXPECTED_NEXT = "manual_approval_to_arm_execution_ready_camera_focal_objective_isolation"
EXPECTED_FOCUS = "camera_focal_objective_isolation_execution_ready_pending_arm_cloud_off"


def parse_args():
    parser = argparse.ArgumentParser(description="Verify execution-ready promotion truth for camera_focal_objective_isolation.")
    parser.add_argument("--python-exe", default=sys.executable)
    return parser.parse_args()


def iso_now():
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict):
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str):
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def render_json_md(title: str, payload: dict) -> str:
    return f"# {title}\n\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"


def run_cmd(args):
    return subprocess.run(args, cwd=str(REPO_ROOT), text=True, capture_output=True, encoding="utf-8", errors="replace", check=False)


def refresh_truth(python_exe: str, cycle_dir: Path):
    rows = []
    for script in WRITERS:
        result = run_cmd([python_exe, str(script)])
        write_text(cycle_dir / f"{script.stem}.stdout.txt", result.stdout)
        write_text(cycle_dir / f"{script.stem}.stderr.txt", result.stderr)
        rows.append({"script": str(script), "returncode": result.returncode})
        if result.returncode != 0:
            raise RuntimeError(f"{script.name} failed: {result.stderr.strip()!r}")
    return rows


def loss_smoke():
    import py_compile

    def ensure_iopath_stub():
        try:
            import iopath.common.file_io  # noqa: F401
            return
        except ModuleNotFoundError:
            iopath = types.ModuleType("iopath")
            common = types.ModuleType("iopath.common")
            file_io = types.ModuleType("iopath.common.file_io")

            class _DummyPathMgr:
                def isdir(self, path): return False
                def isfile(self, path): return False
                def open(self, *args, **kwargs): raise FileNotFoundError

            file_io.g_pathmgr = _DummyPathMgr()
            common.file_io = file_io
            sys.modules["iopath"] = iopath
            sys.modules["iopath.common"] = common
            sys.modules["iopath.common.file_io"] = file_io

    py_compile.compile(str(LOSS_PY), doraise=True)
    ensure_iopath_stub()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    training_root = str((REPO_ROOT / "training").resolve())
    if training_root not in sys.path:
        sys.path.insert(0, training_root)
    loss_mod = importlib.reload(importlib.import_module("loss"))
    sig = inspect.signature(loss_mod.compute_camera_loss)
    return {
        "py_compile": "pass",
        "loss_fl_isolation_scale_default": float(sig.parameters["loss_fl_isolation_scale"].default),
        "loss_py_sha256": hashlib.sha256(LOSS_PY.read_bytes()).hexdigest(),
    }


def main() -> int:
    args = parse_args()
    session_dir = ensure_dir(OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_guard"))
    cycle_dir = ensure_dir(session_dir / "cycle_001")
    refresh_rows = refresh_truth(args.python_exe, cycle_dir)
    smoke = loss_smoke()

    research = read_json(RESEARCH_STATUS)
    task = read_json(TASK_PLAN)
    watch = read_json(LATEST_WATCH)
    guard = read_json(LATEST_GUARD)
    decision = read_json(PROMOTION_DECISION)
    seed = read_json(SEED_JSON)
    blueprint = read_json(BLUEPRINT_JSON)
    family_plan = read_json(FAMILY_PLAN_JSON)

    drifts = []
    if research.get("ready_for_execution") is not True: drifts.append("research_loop_status.ready_for_execution")
    if research.get("manual_action_kind") != "manual_approval": drifts.append("research_loop_status.manual_action_kind")
    if research.get("current_priority_family") != "camera_focal_objective_isolation": drifts.append("research_loop_status.current_priority_family")
    if research.get("next_requirement") != EXPECTED_NEXT: drifts.append("research_loop_status.next_requirement")
    if task.get("task_mode_focus") != EXPECTED_FOCUS: drifts.append("task_plan.task_mode_focus")
    if (task.get("problem_definition_progress", {}) or {}).get("status") != "ready_for_execution": drifts.append("task_plan.problem_definition_progress.status")
    if (watch.get("research", {}).get("summary", {}) or {}).get("ready_for_execution") is not True: drifts.append("latest_watch_snapshot.research.summary.ready_for_execution")
    if decision.get("decision") != "PROMOTE_TO_EXECUTION_READY": drifts.append("promotion_decision.decision")
    if decision.get("ready_for_execution") is not True: drifts.append("promotion_decision.ready_for_execution")
    if seed.get("first_candidate_config") != decision.get("first_candidate_config"): drifts.append("seed.first_candidate_config")
    if blueprint.get("status") != "ready_for_execution": drifts.append("family_blueprint.status")
    if family_plan.get("state") != "execution_ready_pending_arm": drifts.append("family_plan.state")
    if family_plan.get("do_not_arm_now") is not False: drifts.append("family_plan.do_not_arm_now")
    if guard.get("state_cloud_gate") is True: drifts.append("latest_guard_snapshot.state_cloud_gate")
    if guard.get("state_launch_cloud_now") is True: drifts.append("latest_guard_snapshot.state_launch_cloud_now")

    report = {
        "checked_at": iso_now(),
        "family": "camera_focal_objective_isolation",
        "status": "NO_DRIFT" if not drifts else "DRIFT_DETECTED",
        "review_readiness": "execution_ready_pending_manual_arm" if not drifts else "",
        "drift_fields": drifts,
        "current_truth": {
            "research_state": research.get("state", ""),
            "manual_action_kind": research.get("manual_action_kind", ""),
            "ready_for_execution": research.get("ready_for_execution"),
            "current_priority_family": research.get("current_priority_family", ""),
            "next_requirement": research.get("next_requirement", ""),
            "task_mode_focus": task.get("task_mode_focus", ""),
            "promotion_decision": decision.get("decision", ""),
            "family_blueprint_status": blueprint.get("status", ""),
            "family_plan_state": family_plan.get("state", ""),
            "watch_conclusion": watch.get("watch_conclusion", ""),
        },
        "loss_smoke": smoke,
        "refresh_chain": refresh_rows,
    }

    latest_report = OUTPUT_ROOT / "latest_report.json"
    report_json = RESEARCH_ROOT / "overnight_execution_ready_promotion_guard_report.camera_focal_objective_isolation.20260401.json"
    report_md = RESEARCH_ROOT / "overnight_execution_ready_promotion_guard_report.camera_focal_objective_isolation.20260401.md"
    write_json(latest_report, report)
    write_json(report_json, report)
    write_text(report_md, render_json_md("Execution-Ready Promotion Guard Report", report))
    write_json(cycle_dir / "cycle_report.json", report)
    write_text(cycle_dir / "cycle_report.md", render_json_md("Execution-Ready Promotion Guard Report", report))

    print(json.dumps({"latest_report_json": str(latest_report.resolve()), "report_json": str(report_json.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
