import argparse
import hashlib
import importlib
import inspect
import json
import subprocess
import sys
import time
import types
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
OUTPUT_ROOT = RESEARCH_ROOT / "status_discussion_guard_only"
LOSS_PY = REPO_ROOT / "training" / "loss.py"
LOCAL_MANIFEST = REPO_ROOT / "scripts" / "manifests" / "zju_source_policy_rawpool_local_nightly_v1.json"
ALLOWLIST = RESEARCH_ROOT / "repo_process_allowlist.json"
GUARD_SNAPSHOT = REPO_ROOT / "output" / "zju_source_policy_rawpool_overnight_watch" / "latest_guard_snapshot.json"
RESEARCH_STATUS = RESEARCH_ROOT / "research_loop_status.json"
TASK_PLAN = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current" / "task_plan.json"
LATEST_WATCH = REPO_ROOT / "output" / "zju_source_policy_research_watch" / "latest_watch_snapshot.json"
SUMMARY_MD = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current" / "summary.md"
PREP_DESIGN = RESEARCH_ROOT / "execution_ready_preparation_design.camera_focal_objective_isolation.20260331.json"
STATUS_DISCUSSION_DECISION = RESEARCH_ROOT / "execution_ready_status_discussion_decision.camera_focal_objective_isolation.20260401.json"
STATUS_DISCUSSION_PACKET = RESEARCH_ROOT / "execution_ready_status_discussion_packet.camera_focal_objective_isolation.20260401.json"
FAMILY_BLUEPRINT = RESEARCH_ROOT / "family_blueprint.camera_focal_objective_isolation.json"
FAMILY_PLAN = RESEARCH_ROOT / "candidate_patch_plan.camera_focal_objective_isolation.json"

WRITER_SCRIPTS = [
    REPO_ROOT / "scripts" / "write_zju_execution_ready_boundary.py",
    REPO_ROOT / "scripts" / "write_zju_execution_ready_discussion_approval_note.py",
    REPO_ROOT / "scripts" / "write_zju_execution_ready_discussion_decision.py",
    REPO_ROOT / "scripts" / "write_zju_execution_ready_prep_plan.py",
    REPO_ROOT / "scripts" / "write_zju_execution_ready_preparation_design.py",
    REPO_ROOT / "scripts" / "write_zju_execution_ready_status_discussion_decision.py",
    REPO_ROOT / "scripts" / "write_zju_execution_ready_status_discussion_packet.py",
    REPO_ROOT / "scripts" / "sync_zju_daytime_manual_diagnosis_after_two_stage_failure.py",
]

FAMILY = "camera_focal_objective_isolation"
MODE = "STATUS_DISCUSSION_GUARD_ONLY"
EXPECTED_NEXT = "manual_review_execution_ready_status_discussion_before_any_execution_ready_promotion"
EXPECTED_TASK_MODE_FOCUS = "execution_ready_status_discussion_promoted_cloud_off"
EXPECTED_PROGRESS_STATUS = "execution_ready_status_discussion_promoted_not_execution_ready"
EXPECTED_BLUEPRINT_STATUS = "execution_ready_status_discussion_promoted_not_execution_ready"
EXPECTED_PLAN_STATE = "manual_review_execution_ready_status_discussion_only"
EXPECTED_PACKET_STATE = "manual_review_execution_ready_status_discussion_packet_ready"
EXPECTED_STATUS_DISCUSSION_DECISION = "PROMOTE_TO_EXECUTION_READY_STATUS_DISCUSSION"
ALLOWED_PROCESS_MARKERS = (
    "run_zju_execution_ready_status_discussion_guard_only.py",
    "run_zju_source_policy_rawpool_overnight_watch.py",
    "run_zju_source_policy_rawpool_guard_daemon.py",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the status-discussion overnight guard.")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--interval-sec", type=int, default=3600)
    parser.add_argument("--duration-hours", type=float, default=12.0)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def day_tag():
    return datetime.now().strftime("%Y%m%d")


def iso_now():
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path):
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


def write_runtime_status(session_dir: Path, latest_session_path: Path, payload: dict):
    write_json(session_dir / "runtime_status.json", payload)
    write_json(latest_session_path, payload)


def run_cmd(args):
    return subprocess.run(
        args,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh_truth(python_exe: str, cycle_dir: Path):
    rows = []
    for script in WRITER_SCRIPTS:
        result = run_cmd([python_exe, str(script)])
        stem = script.stem
        write_text(cycle_dir / f"{stem}.stdout.txt", result.stdout)
        write_text(cycle_dir / f"{stem}.stderr.txt", result.stderr)
        rows.append({"script": str(script), "returncode": result.returncode})
        if result.returncode != 0:
            raise RuntimeError(f"{script.name} failed: {result.stderr.strip()!r}")
    return rows


def modal_apps():
    result = run_cmd(["modal", "app", "list", "--json"])
    if result.returncode != 0:
        raise RuntimeError(f"modal app list failed: {result.stderr.strip()!r}")
    text = result.stdout.strip()
    return json.loads(text) if text else []


def repo_processes():
    repo = str(REPO_ROOT).lower().replace("\\", "\\\\")
    self_pid = subprocess.os.getpid()
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
""".strip().format(repo=repo, self_pid=self_pid)
    result = run_cmd(["powershell", "-NoProfile", "-Command", command])
    if result.returncode != 0:
        raise RuntimeError(f"repo process query failed: {result.stderr.strip()!r}")
    text = result.stdout.strip()
    rows = json.loads(text) if text else []
    rows = rows if isinstance(rows, list) else [rows]
    allow = tuple(m.lower() for m in ALLOWED_PROCESS_MARKERS + tuple(read_json(ALLOWLIST).get("allowed_markers", [])))
    return [row for row in rows if not any(marker in str(row.get("CommandLine", "")).lower() for marker in allow)]


def loss_smokes():
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
                def isdir(self, path):
                    return False

                def isfile(self, path):
                    return False

                def open(self, *args, **kwargs):
                    raise FileNotFoundError

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
    param = sig.parameters.get("loss_fl_isolation_scale")
    if param is None or float(param.default) != 1.0:
        raise RuntimeError("loss_fl_isolation_scale signature/default drifted.")
    components = loss_mod._assemble_camera_component_dict(1.25, 0.5, 0.75)
    total = loss_mod._compute_total_camera_loss(
        components,
        weight_trans=1.0,
        weight_rot=2.0,
        weight_focal=0.5,
        loss_fl_isolation_scale=1.0,
    )
    if abs(float(total) - (1.25 + 1.0 + 0.375)) > 1e-9:
        raise RuntimeError("default identity formula drifted.")
    return {
        "py_compile": "pass",
        "import_smoke": "pass",
        "signature_smoke": "pass",
        "default_identity_formula_smoke": "pass",
        "loss_py_sha256": sha256(LOSS_PY),
    }


def expected_lead():
    return str((read_json(LOCAL_MANIFEST).get("current_lead", {}) or {}).get("config", "")).strip()


def detect_drift(smoke: dict, baseline_hash: str):
    research = read_json(RESEARCH_STATUS)
    task = read_json(TASK_PLAN)
    watch = read_json(LATEST_WATCH)
    guard = read_json(GUARD_SNAPSHOT)
    design = read_json(PREP_DESIGN)
    status_decision = read_json(STATUS_DISCUSSION_DECISION)
    packet = read_json(STATUS_DISCUSSION_PACKET)
    blueprint = read_json(FAMILY_BLUEPRINT)
    family_plan = read_json(FAMILY_PLAN)
    modal = [row for row in modal_apps() if str(row.get("State", "")).lower() != "stopped"]
    repo = repo_processes()
    lead = expected_lead()
    drifts = []

    if research.get("state") != "IDLE_GUARD":
        drifts.append("research_loop_status.state")
    if research.get("approved_problem_present"):
        drifts.append("research_loop_status.approved_problem_present")
    if research.get("manual_action_kind") != "manual_review":
        drifts.append("research_loop_status.manual_action_kind")
    if research.get("ready_for_execution"):
        drifts.append("research_loop_status.ready_for_execution")
    if research.get("preferred_first_family") != FAMILY:
        drifts.append("research_loop_status.preferred_first_family")
    if research.get("next_requirement") != EXPECTED_NEXT:
        drifts.append("research_loop_status.next_requirement")

    if task.get("task_mode_status") != "completed":
        drifts.append("task_plan.task_mode_status")
    if task.get("task_mode_focus") != EXPECTED_TASK_MODE_FOCUS:
        drifts.append("task_plan.task_mode_focus")
    if ((task.get("problem_definition_progress", {}) or {}).get("status") != EXPECTED_PROGRESS_STATUS):
        drifts.append("task_plan.problem_definition_progress.status")

    watch_summary = (((watch.get("research", {}) or {}).get("summary", {})) or {})
    if watch_summary.get("state") != "IDLE_GUARD":
        drifts.append("latest_watch_snapshot.research.summary.state")
    if watch_summary.get("manual_action_kind") != "manual_review":
        drifts.append("latest_watch_snapshot.research.summary.manual_action_kind")

    if guard.get("state_latest_decision") != "steady_hold":
        drifts.append("latest_guard_snapshot.state_latest_decision")
    if guard.get("state_cloud_gate"):
        drifts.append("latest_guard_snapshot.state_cloud_gate")
    if guard.get("state_launch_cloud_now"):
        drifts.append("latest_guard_snapshot.state_launch_cloud_now")
    if str(guard.get("state_current_lead_config", "")).strip() != lead:
        drifts.append("latest_guard_snapshot.state_current_lead_config")

    if design.get("artifact_kind") != "execution_ready_preparation_design":
        drifts.append("preparation_design.artifact_kind")
    if design.get("ready_for_execution"):
        drifts.append("preparation_design.ready_for_execution")

    if status_decision.get("artifact_kind") != "execution_ready_status_discussion_decision":
        drifts.append("status_discussion_decision.artifact_kind")
    if status_decision.get("decision") != EXPECTED_STATUS_DISCUSSION_DECISION:
        drifts.append("status_discussion_decision.decision")
    if status_decision.get("next_requirement") != EXPECTED_NEXT:
        drifts.append("status_discussion_decision.next_requirement")
    if status_decision.get("ready_for_execution"):
        drifts.append("status_discussion_decision.ready_for_execution")

    if packet.get("artifact_kind") != "execution_ready_status_discussion_packet":
        drifts.append("status_discussion_packet.artifact_kind")
    if packet.get("state") != EXPECTED_PACKET_STATE:
        drifts.append("status_discussion_packet.state")
    if packet.get("next_requirement") != EXPECTED_NEXT:
        drifts.append("status_discussion_packet.next_requirement")
    if packet.get("ready_for_execution"):
        drifts.append("status_discussion_packet.ready_for_execution")

    if blueprint.get("status") != EXPECTED_BLUEPRINT_STATUS:
        drifts.append("family_blueprint.status")
    if blueprint.get("ready_for_execution"):
        drifts.append("family_blueprint.ready_for_execution")
    if blueprint.get("next_requirement") != EXPECTED_NEXT:
        drifts.append("family_blueprint.next_requirement")

    if family_plan.get("state") != EXPECTED_PLAN_STATE:
        drifts.append("family_plan.state")
    if ((family_plan.get("readiness", {}) or {}).get("ready_for_execution")):
        drifts.append("family_plan.readiness.ready_for_execution")
    if not ((family_plan.get("readiness", {}) or {}).get("status_discussion_packet_ready")):
        drifts.append("family_plan.readiness.status_discussion_packet_ready")
    if family_plan.get("next_requirement") != EXPECTED_NEXT:
        drifts.append("family_plan.next_requirement")

    if modal:
        drifts.append("active_modal_app_count")
    if repo:
        drifts.append("repo_process_count")
    if smoke.get("loss_py_sha256") != baseline_hash:
        drifts.append("training/loss.py.sha256")

    return drifts, research, task, watch, design, status_decision, packet, blueprint, family_plan, modal, repo, lead


def render_md(report):
    lines = [
        "# Overnight Status Discussion Guard Report",
        "",
        f"- checked_at: `{report['checked_at']}`",
        f"- overnight_mode: `{report['overnight_mode']}`",
        f"- status: `{report['status']}`",
        f"- cycle_index: `{report['cycle_index']}`",
        f"- review_readiness: `{report['review_readiness']}`",
        "",
        "## Current Truth",
        "",
    ]
    for key, value in report["current_truth"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Drift Fields", ""])
    lines.extend([f"- {item}" for item in report["drift_fields"]] or ["- `NONE`"])
    lines.extend(["", "## Loss Smokes", ""])
    for key, value in report["loss_smokes"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Key Paths", ""])
    for key, value in report["key_paths"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def run_cycle(args, session_dir: Path, cycle_index: int, baseline_hash: str, report_json: Path, report_md: Path):
    cycle_dir = ensure_dir(session_dir / f"cycle_{cycle_index:03d}")
    error = ""
    refresh_rows = []
    smoke = {}
    drifts = ["refresh_or_smoke_failure"]
    research = {}
    task = {}
    watch = {}
    design = {}
    status_decision = {}
    packet = {}
    blueprint = {}
    family_plan = {}
    modal = []
    repo = []
    lead = ""
    try:
        refresh_rows = refresh_truth(args.python_exe, cycle_dir)
        smoke = loss_smokes()
        (
            drifts,
            research,
            task,
            watch,
            design,
            status_decision,
            packet,
            blueprint,
            family_plan,
            modal,
            repo,
            lead,
        ) = detect_drift(smoke, baseline_hash)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        research = read_json(RESEARCH_STATUS) if RESEARCH_STATUS.exists() else {}
        task = read_json(TASK_PLAN) if TASK_PLAN.exists() else {}
        watch = read_json(LATEST_WATCH) if LATEST_WATCH.exists() else {}
        design = read_json(PREP_DESIGN) if PREP_DESIGN.exists() else {}
        status_decision = read_json(STATUS_DISCUSSION_DECISION) if STATUS_DISCUSSION_DECISION.exists() else {}
        packet = read_json(STATUS_DISCUSSION_PACKET) if STATUS_DISCUSSION_PACKET.exists() else {}
        blueprint = read_json(FAMILY_BLUEPRINT) if FAMILY_BLUEPRINT.exists() else {}
        family_plan = read_json(FAMILY_PLAN) if FAMILY_PLAN.exists() else {}

    report = {
        "checked_at": iso_now(),
        "family": FAMILY,
        "overnight_mode": MODE,
        "status": "NO_DRIFT" if not drifts and not error else "DRIFT_DETECTED",
        "review_readiness": "manual_status_discussion_packet_ready" if not drifts and not error else "",
        "cycle_index": cycle_index,
        "session_dir": str(session_dir.resolve()),
        "error": error,
        "drift_fields": drifts,
        "current_truth": {
            "state": research.get("state", ""),
            "manual_action_kind": research.get("manual_action_kind", ""),
            "ready_for_execution": research.get("ready_for_execution"),
            "approved_problem_present": research.get("approved_problem_present"),
            "preferred_first_family": research.get("preferred_first_family", ""),
            "next_requirement": research.get("next_requirement", ""),
            "task_mode_status": task.get("task_mode_status", ""),
            "task_mode_focus": task.get("task_mode_focus", ""),
            "watch_conclusion": watch.get("watch_conclusion", ""),
            "preparation_design_state": design.get("state", ""),
            "status_discussion_decision": status_decision.get("decision", ""),
            "status_discussion_packet_state": packet.get("state", ""),
            "family_blueprint_status": blueprint.get("status", ""),
            "family_plan_state": family_plan.get("state", ""),
            "expected_lead_config": lead,
            "active_modal_app_count": len(modal),
            "repo_process_count": len(repo),
        },
        "loss_smokes": smoke,
        "refresh_chain": refresh_rows,
        "key_paths": {
            "research_loop_status": str(RESEARCH_STATUS.resolve()),
            "task_plan": str(TASK_PLAN.resolve()),
            "latest_watch_snapshot": str(LATEST_WATCH.resolve()),
            "summary_md": str(SUMMARY_MD.resolve()),
            "preparation_design_json": str(PREP_DESIGN.resolve()),
            "status_discussion_decision_json": str(STATUS_DISCUSSION_DECISION.resolve()),
            "status_discussion_packet_json": str(STATUS_DISCUSSION_PACKET.resolve()),
            "family_blueprint_json": str(FAMILY_BLUEPRINT.resolve()),
            "family_plan_json": str(FAMILY_PLAN.resolve()),
        },
    }
    write_json(report_json, report)
    write_text(report_md, render_md(report))
    write_json(cycle_dir / "cycle_report.json", report)
    write_text(cycle_dir / "cycle_report.md", render_md(report))
    return report


def main():
    args = parse_args()
    session_dir = ensure_dir(OUTPUT_ROOT / f"{now_tag()}_guard")
    latest_session_path = OUTPUT_ROOT / "latest_session.json"
    latest_report_path = OUTPUT_ROOT / "latest_report.json"
    baseline_hash = sha256(LOSS_PY)
    write_json(
        session_dir / "baseline.json",
        {
            "checked_at": iso_now(),
            "loss_py_sha256": baseline_hash,
            "expected_lead_config": expected_lead(),
        },
    )
    report_json = RESEARCH_ROOT / f"overnight_status_discussion_guard_report.{FAMILY}.{day_tag()}.json"
    report_md = RESEARCH_ROOT / f"overnight_status_discussion_guard_report.{FAMILY}.{day_tag()}.md"

    write_runtime_status(
        session_dir,
        latest_session_path,
        {
            "checked_at": iso_now(),
            "mode": MODE,
            "status": "starting",
            "cycle_index": 0,
            "session_dir": str(session_dir.resolve()),
            "latest_report_json": str(latest_report_path.resolve()),
        },
    )

    start = time.monotonic()
    cycle_index = 0
    report = {}
    while True:
        if args.max_cycles and cycle_index >= args.max_cycles:
            break
        if not args.once and (time.monotonic() - start) >= args.duration_hours * 3600:
            break
        cycle_index += 1
        report = run_cycle(args, session_dir, cycle_index, baseline_hash, report_json, report_md)
        write_json(latest_report_path, report)
        write_runtime_status(
            session_dir,
            latest_session_path,
            {
                "checked_at": iso_now(),
                "mode": MODE,
                "status": "running" if report.get("status") == "NO_DRIFT" else "running_with_drift",
                "cycle_index": cycle_index,
                "session_dir": str(session_dir.resolve()),
                "latest_report_json": str(latest_report_path.resolve()),
                "latest_report_status": report.get("status", ""),
            },
        )
        if args.once:
            break
        time.sleep(max(1, args.interval_sec))

    write_runtime_status(
        session_dir,
        latest_session_path,
        {
            "checked_at": iso_now(),
            "mode": MODE,
            "status": "completed" if report.get("status") == "NO_DRIFT" else "completed_with_drift",
            "cycle_index": cycle_index,
            "session_dir": str(session_dir.resolve()),
            "latest_report_json": str(latest_report_path.resolve()),
            "latest_report_status": report.get("status", ""),
        },
    )

    print(
        json.dumps(
            {
                "session_dir": str(session_dir.resolve()),
                "report_json": str(report_json.resolve()),
                "report_md": str(report_md.resolve()),
                "latest_report_json": str(latest_report_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
