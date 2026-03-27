import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_LOOP_SCRIPT = REPO_ROOT / "scripts" / "run_zju_source_policy_research_loop.py"
APPROVED_PROBLEM_PATH = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "approved_problem.json"
APPROVED_PROBLEM_ARCHIVE_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "approved_problem_archive"
REPO_PROCESS_ALLOWLIST_PATH = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "repo_process_allowlist.json"
RESEARCH_STATUS_PATH = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "research_loop_status.json"
CANDIDATE_VERDICT_PATH = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "candidate_verdict.json"
FRONTIER_LEDGER_PATH = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "frontier_ledger.json"
FAMILY_STOP_REASON_PATH = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "family_stop_reason.json"
RESUME_TOKEN_PATH = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "resume_token.json"
GATE_REFERENCE_LOGS_PATH = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "gate_reference_logs.json"
RUN_OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "runs"
PREFLIGHT_PS1 = REPO_ROOT / "scripts" / "invoke_modal_zju_preflight.ps1"
FINETUNE_PS1 = REPO_ROOT / "scripts" / "run_zju_vggt_geom_minimal_finetune.ps1"
COMPARE_PY = REPO_ROOT / "scripts" / "compare_zju_finetune_runs.py"


class ResearchCandidateError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run exactly one approved config-only ZJU source-policy research candidate under the "
            "single-problem single-candidate overnight contract."
        )
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--approved-problem-path", type=Path, default=APPROVED_PROBLEM_PATH)
    parser.add_argument("--approved-problem-archive-root", type=Path, default=APPROVED_PROBLEM_ARCHIVE_ROOT)
    parser.add_argument("--repo-process-allowlist-path", type=Path, default=REPO_PROCESS_ALLOWLIST_PATH)
    parser.add_argument("--gate-reference-logs-path", type=Path, default=GATE_REFERENCE_LOGS_PATH)
    parser.add_argument("--smoke-train-batches", type=int, default=1)
    parser.add_argument("--smoke-val-batches", type=int, default=1)
    parser.add_argument("--short-train-batches", type=int, default=10)
    parser.add_argument("--short-val-batches", type=int, default=5)
    parser.add_argument("--long-train-batches", type=int, default=100)
    parser.add_argument("--long-val-batches", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=RUN_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def try_load_json(path: Path) -> tuple[dict, str]:
    try:
        return load_json(path), ""
    except Exception as exc:
        return {}, str(exc)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def normalize_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path.resolve()


def slugify(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum():
            keep.append(ch.lower())
        else:
            keep.append("_")
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "approved_problem"


def config_name_from_path(path_like: str | Path) -> str:
    return normalize_repo_path(path_like).stem


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
        raise ResearchCandidateError(
            "Command failed with exit code {code}: {cmd}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}".format(
                code=result.returncode,
                cmd=" ".join(args),
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
            )
        )
    return result


def import_research_loop_module():
    spec = importlib.util.spec_from_file_location("zju_research_loop_module", RESEARCH_LOOP_SCRIPT)
    if spec is None or spec.loader is None:
        raise ResearchCandidateError(f"Unable to load research-loop module from {RESEARCH_LOOP_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_average(summary: dict, metric_name: str) -> float | None:
    for row in summary.get("val", {}).get("rows", []):
        if row.get("metric") == metric_name:
            return row.get("candidate")
    return None


def delta_average(summary: dict, metric_name: str) -> float | None:
    for row in summary.get("val", {}).get("rows", []):
        if row.get("metric") == metric_name:
            return row.get("delta")
    return None


def run_research_loop_refresh(python_exe: str) -> None:
    run_checked([python_exe, str(RESEARCH_LOOP_SCRIPT)], cwd=REPO_ROOT)


def load_gate_reference_logs(path: Path) -> dict:
    payload = load_json(path)
    short_gate = payload.get("short_gate", {}) or {}
    long_gate = payload.get("long_gate", {}) or {}
    stable_short_log = normalize_repo_path(short_gate.get("stable_lead_reference_log", ""))
    baseline_short_log = normalize_repo_path(short_gate.get("baseline_reference_log", ""))
    stable_long_log = normalize_repo_path(long_gate.get("stable_lead_reference_log", ""))
    baseline_long_log = normalize_repo_path(long_gate.get("baseline_reference_log", ""))
    for required_path, label in [
        (stable_short_log, "stable short-gate"),
        (baseline_short_log, "baseline short-gate"),
        (stable_long_log, "stable long-gate"),
        (baseline_long_log, "baseline long-gate"),
    ]:
        if not required_path.exists():
            raise ResearchCandidateError(f"Missing canonical {label} reference log: {required_path}")
    return {
        "payload": payload,
        "stable_short_log": stable_short_log,
        "baseline_short_log": baseline_short_log,
        "stable_long_log": stable_long_log,
        "baseline_long_log": baseline_long_log,
    }


def archive_approved_problem(
    *,
    approved_problem_path: Path,
    archive_root: Path,
    run_dir: Path,
    verdict_status: str,
    status_path: Path,
) -> Path | None:
    if not approved_problem_path.exists():
        return None
    raw_text = approved_problem_path.read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        payload = {
            "malformed_json": True,
            "archive_read_error": str(exc),
            "raw_text": raw_text,
        }
    archive_dir = ensure_dir(archive_root)
    problem_id = slugify(str(payload.get("problem_id", "")))
    archive_path = archive_dir / f"{now_tag()}_{problem_id}.json"
    payload["archived_at"] = iso_now()
    payload["archive_reason"] = "return_to_guard_after_candidate_runner"
    payload["archive_verdict_status"] = verdict_status
    payload["archive_run_dir"] = str(run_dir.resolve())
    payload["archive_status_path"] = str(status_path.resolve())
    payload["active_contract_consumed"] = True
    write_json(archive_path, payload)
    approved_problem_path.unlink()
    return archive_path


def fallback_archive_approved_problem(
    *,
    approved_problem_path: Path,
    archive_root: Path,
    run_dir: Path,
    status_path: Path,
    archive_error: str,
) -> Path | None:
    if not approved_problem_path.exists():
        return None
    archive_dir = ensure_dir(archive_root)
    raw_text = approved_problem_path.read_text(encoding="utf-8-sig")
    archive_path = archive_dir / f"{now_tag()}_archive_error_{approved_problem_path.stem}.json"
    write_json(
        archive_path,
        {
            "checked_at": iso_now(),
            "archive_reason": "fallback_after_archive_error",
            "archive_error": archive_error,
            "archive_run_dir": str(run_dir.resolve()),
            "archive_status_path": str(status_path.resolve()),
            "raw_text": raw_text,
        },
    )
    approved_problem_path.unlink(missing_ok=True)
    return archive_path


def build_allowlist(markers: list[str]) -> dict:
    return {
        "checked_at": iso_now(),
        "status": "research_window_open",
        "guard_track_must_continue": True,
        "notes": "Temporary allowlist for one approved research candidate run; clear on return to guard.",
        "allowed_markers": markers,
    }


def clear_allowlist(path: Path) -> None:
    write_json(
        path,
        {
            "checked_at": iso_now(),
            "status": "idle_empty_allowlist",
            "guard_track_must_continue": True,
            "notes": "No active approved research candidate is running.",
            "allowed_markers": [],
        },
    )


def run_preflight() -> None:
    run_checked(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PREFLIGHT_PS1),
        ],
        cwd=REPO_ROOT,
    )


def run_finetune(
    *,
    python_exe: str,
    config_path: Path,
    exp_name: str,
    limit_train_batches: int,
    limit_val_batches: int,
    run_dir: Path,
    stage_name: str,
) -> Path:
    config_name = config_name_from_path(config_path)
    result = run_checked(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(FINETUNE_PS1),
            "-PythonExe",
            python_exe,
            "-Config",
            config_name,
            "-ExpName",
            exp_name,
            "-LimitTrainBatches",
            str(limit_train_batches),
            "-LimitValBatches",
            str(limit_val_batches),
        ],
        cwd=REPO_ROOT,
    )
    stage_dir = ensure_dir(run_dir / stage_name)
    write_text(stage_dir / "stdout.txt", result.stdout)
    write_text(stage_dir / "stderr.txt", result.stderr)
    log_path = REPO_ROOT / "training" / "logs" / exp_name / "log.txt"
    if not log_path.exists():
        raise ResearchCandidateError(
            "Expected training log was not created: {log_path}\n"
            "Captured stage stdout: {stdout_path}\n"
            "Captured stage stderr: {stderr_path}".format(
                log_path=log_path,
                stdout_path=stage_dir / "stdout.txt",
                stderr_path=stage_dir / "stderr.txt",
            )
        )
    return log_path


def run_compare(
    *,
    python_exe: str,
    baseline_log: Path,
    candidate_log: Path,
    baseline_label: str,
    candidate_label: str,
    output_dir: Path,
    title: str,
) -> dict:
    run_checked(
        [
            python_exe,
            str(COMPARE_PY),
            "--baseline-log",
            str(baseline_log),
            "--candidate-log",
            str(candidate_log),
            "--baseline-label",
            baseline_label,
            "--candidate-label",
            candidate_label,
            "--output-dir",
            str(output_dir),
            "--title",
            title,
        ],
        cwd=REPO_ROOT,
    )
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise ResearchCandidateError(f"Expected compare summary was not created: {summary_path}")
    return load_json(summary_path)


def short_gate_beats_stable_lead(summary_vs_lead: dict) -> bool:
    camera_delta = delta_average(summary_vs_lead, "loss_camera")
    t_delta = delta_average(summary_vs_lead, "loss_T")
    conf_delta = delta_average(summary_vs_lead, "loss_conf_depth")
    reg_delta = delta_average(summary_vs_lead, "loss_reg_depth")
    return (
        camera_delta is not None
        and t_delta is not None
        and conf_delta is not None
        and reg_delta is not None
        and camera_delta <= 0.0
        and t_delta <= 0.0
        and conf_delta < 0.0
        and reg_delta < 0.0
    )


def attach_verdict_context(payload: dict, approved_problem: dict, gate_stage_reached: str) -> dict:
    payload["problem_id"] = approved_problem.get("problem_id", "")
    payload["family"] = approved_problem.get("family", "")
    payload["first_candidate_shape"] = approved_problem.get("first_candidate_shape", "")
    payload["gate_stage_reached"] = gate_stage_reached
    payload.setdefault("approved_problem_archive_path", "")
    return payload


def build_candidate_verdict_payload(
    *,
    status: str,
    candidate_config: str,
    approved_problem: dict,
    gate_stage_reached: str,
    summary_vs_lead: dict | None = None,
    summary_vs_baseline: dict | None = None,
    reason: str = "",
) -> dict:
    payload = {
        "checked_at": iso_now(),
        "status": status,
        "active_candidate": candidate_config,
        "reason": reason,
    }
    attach_verdict_context(payload, approved_problem, gate_stage_reached)
    if summary_vs_lead is not None:
        payload["short_gate_vs_lead"] = {
            "camera": metric_average(summary_vs_lead, "loss_camera"),
            "T": metric_average(summary_vs_lead, "loss_T"),
            "conf_depth": metric_average(summary_vs_lead, "loss_conf_depth"),
            "reg_depth": metric_average(summary_vs_lead, "loss_reg_depth"),
            "delta_camera": delta_average(summary_vs_lead, "loss_camera"),
            "delta_T": delta_average(summary_vs_lead, "loss_T"),
            "delta_conf_depth": delta_average(summary_vs_lead, "loss_conf_depth"),
            "delta_reg_depth": delta_average(summary_vs_lead, "loss_reg_depth"),
        }
    if summary_vs_baseline is not None:
        payload["short_gate_vs_baseline"] = {
            "camera": metric_average(summary_vs_baseline, "loss_camera"),
            "T": metric_average(summary_vs_baseline, "loss_T"),
            "conf_depth": metric_average(summary_vs_baseline, "loss_conf_depth"),
            "reg_depth": metric_average(summary_vs_baseline, "loss_reg_depth"),
        }
    return payload


def build_contract_exit_payload(
    *,
    status: str,
    reason: str,
    approved_problem: dict,
    gate_stage_reached: str,
    candidate_config: str = "",
) -> dict:
    payload = {
        "checked_at": iso_now(),
        "status": status,
        "active_candidate": candidate_config,
        "reason": reason,
        "approved_problem_archive_path": "",
    }
    return attach_verdict_context(payload, approved_problem, gate_stage_reached)


def mark_finalization_error(verdict_payload: dict, *, status: str, reason: str) -> None:
    if verdict_payload.get("status") != status:
        verdict_payload.setdefault("underlying_candidate_status", verdict_payload.get("status", ""))
        verdict_payload.setdefault("underlying_candidate_reason", verdict_payload.get("reason", ""))
    verdict_payload["status"] = status
    verdict_payload["reason"] = reason


def finalize_runner(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    status: dict,
    status_path: Path,
    verdict_payload: dict,
    verdict_status_for_archive: str,
) -> None:
    clear_allowlist(args.repo_process_allowlist_path)
    archived_approved_problem_path = ""
    archive_error = ""
    try:
        if args.approved_problem_path.exists():
            archived_path = archive_approved_problem(
                approved_problem_path=args.approved_problem_path,
                archive_root=args.approved_problem_archive_root,
                run_dir=run_dir,
                verdict_status=verdict_status_for_archive,
                status_path=status_path,
            )
            if archived_path is not None:
                archived_approved_problem_path = str(archived_path.resolve())
    except Exception as exc:
        archive_error = str(exc)
        status["approved_problem_archive_error"] = archive_error
        try:
            fallback_path = fallback_archive_approved_problem(
                approved_problem_path=args.approved_problem_path,
                archive_root=args.approved_problem_archive_root,
                run_dir=run_dir,
                status_path=status_path,
                archive_error=archive_error,
            )
            if fallback_path is not None:
                archived_approved_problem_path = str(fallback_path.resolve())
        except Exception as fallback_exc:
            status["approved_problem_archive_fallback_error"] = str(fallback_exc)

    if archive_error:
        mark_finalization_error(
            verdict_payload,
            status="archive_error",
            reason=f"Approved problem archive failed during return_to_guard: {archive_error}",
        )
    if args.approved_problem_path.exists():
        try:
            args.approved_problem_path.unlink(missing_ok=True)
        except Exception as exc:
            status["active_contract_cleanup_error"] = str(exc)

    verdict_payload["approved_problem_archive_path"] = archived_approved_problem_path
    status["approved_problem_archive_path"] = archived_approved_problem_path
    status["return_to_guard_started_at"] = iso_now()
    status["research_loop_status_path"] = str(RESEARCH_STATUS_PATH.resolve())
    status["frontier_ledger_path"] = str(FRONTIER_LEDGER_PATH.resolve())
    status["family_stop_reason_path"] = str(FAMILY_STOP_REASON_PATH.resolve())
    status["resume_token_path"] = str(RESUME_TOKEN_PATH.resolve())
    write_json(CANDIDATE_VERDICT_PATH, verdict_payload)
    try:
        run_research_loop_refresh(args.python_exe)
        status["research_loop_refresh_status"] = "ok"
    except Exception as exc:
        status["research_loop_refresh_status"] = f"failed: {exc}"
        mark_finalization_error(
            verdict_payload,
            status="writeback_error",
            reason=f"Research-loop writeback failed during return_to_guard: {exc}",
        )

    research_status, research_status_error = try_load_json(RESEARCH_STATUS_PATH)
    if research_status:
        status["final_research_loop_state"] = research_status.get("state", "")
        status["final_research_loop_reason"] = research_status.get("reason", "")
        if status["final_research_loop_state"] != "IDLE_GUARD":
            mark_finalization_error(
                verdict_payload,
                status="return_to_guard_failed",
                reason=(
                    "Research loop did not settle back to IDLE_GUARD after finalize; "
                    f"observed state={status['final_research_loop_state']}"
                ),
            )
    elif research_status_error:
        status["final_research_loop_state"] = ""
        status["final_research_loop_reason"] = research_status_error
        mark_finalization_error(
            verdict_payload,
            status="writeback_error",
            reason=f"Unable to read final research_loop_status.json: {research_status_error}",
        )

    frontier_ledger, frontier_error = try_load_json(FRONTIER_LEDGER_PATH)
    if frontier_ledger:
        status["latest_frontier_verdict_status"] = (
            (frontier_ledger.get("latest_formal_verdict", {}) or {}).get("status", "")
        )
    elif frontier_error:
        status["latest_frontier_verdict_status"] = frontier_error

    family_stop_reason, family_stop_error = try_load_json(FAMILY_STOP_REASON_PATH)
    if family_stop_reason:
        status["latest_family_outcomes"] = family_stop_reason.get("latest_family_outcomes", {})
    elif family_stop_error:
        status["latest_family_outcomes"] = {"error": family_stop_error}

    status["phase"] = "return_to_guard_complete"
    status["decision"] = verdict_payload
    write_json(CANDIDATE_VERDICT_PATH, verdict_payload)
    write_json(status_path, status)


def main() -> int:
    args = parse_args()
    approved_problem = {}
    approved_problem_load_error = ""
    if args.approved_problem_path.exists():
        approved_problem, approved_problem_load_error = try_load_json(args.approved_problem_path)
    candidate_config = str(approved_problem.get("first_candidate_config", "")).strip()
    run_tag = now_tag()
    candidate_stem = normalize_repo_path(candidate_config).stem if candidate_config else "no_candidate"
    run_dir = ensure_dir(args.output_root / f"{run_tag}_{candidate_stem}")
    status_path = run_dir / "status.json"
    status = {
        "checked_at": iso_now(),
        "phase": "initialized",
        "approved_problem_path": str(args.approved_problem_path.resolve()),
        "approved_problem_archive_root": str(args.approved_problem_archive_root.resolve()),
        "gate_reference_logs_path": str(args.gate_reference_logs_path.resolve()),
        "candidate_config": candidate_config,
        "candidate_verdict_path": str(CANDIDATE_VERDICT_PATH.resolve()),
        "research_loop_status_path": str(RESEARCH_STATUS_PATH.resolve()),
        "frontier_ledger_path": str(FRONTIER_LEDGER_PATH.resolve()),
        "family_stop_reason_path": str(FAMILY_STOP_REASON_PATH.resolve()),
        "resume_token_path": str(RESUME_TOKEN_PATH.resolve()),
        "artifacts": {},
    }
    if approved_problem_load_error:
        status["contract_load_error"] = approved_problem_load_error
    write_json(status_path, status)

    if args.dry_run:
        status["phase"] = "dry_run_ready"
        write_json(status_path, status)
        print(status_path)
        return 0

    return_code = 0
    verdict_payload = build_contract_exit_payload(
        status="runner_error",
        reason="Runner exited without producing a verdict.",
        approved_problem=approved_problem,
        gate_stage_reached="initialized",
        candidate_config=candidate_config,
    )
    verdict_status_for_archive = verdict_payload["status"]
    candidate_config_path: Path | None = normalize_repo_path(candidate_config) if candidate_config else None
    validation_issues: list[str] = []
    try:
        if approved_problem_load_error:
            verdict_payload = build_contract_exit_payload(
                status="contract_load_error",
                reason=f"approved_problem.json could not be parsed: {approved_problem_load_error}",
                approved_problem=approved_problem,
                gate_stage_reached="contract_load",
            )
            status["phase"] = "contract_load_error"
            return_code = 1
            return return_code

        try:
            run_research_loop_refresh(args.python_exe)
            status["research_loop_preflight_refresh"] = "ok"
        except Exception as exc:
            verdict_payload = build_contract_exit_payload(
                status="research_loop_refresh_failed",
                reason=str(exc),
                approved_problem=approved_problem,
                gate_stage_reached="preflight_refresh",
                candidate_config=candidate_config,
            )
            status["phase"] = "research_loop_refresh_failed"
            status["error"] = str(exc)
            return_code = 1
            return return_code

        research_loop = import_research_loop_module()
        validation_issues = (
            research_loop.validate_approved_problem(
                approved_problem,
                max_approved_problems_per_night=1,
                max_candidates_per_problem=1,
            )
            if approved_problem
            else []
        )
        if not approved_problem:
            verdict_payload = build_contract_exit_payload(
                status="idle_no_approved_problem",
                reason="No approved_problem.json exists; research runner must remain idle.",
                approved_problem=approved_problem,
                gate_stage_reached="contract_entry",
            )
            status["phase"] = "idle_no_approved_problem"
            return_code = 1
            return return_code
        if not approved_problem.get("approved"):
            verdict_payload = build_contract_exit_payload(
                status="contract_not_approved",
                reason="approved_problem.json exists but approved=false.",
                approved_problem=approved_problem,
                gate_stage_reached="contract_entry",
                candidate_config=candidate_config,
            )
            status["phase"] = "contract_not_approved"
            return_code = 1
            return return_code
        if validation_issues:
            verdict_payload = build_contract_exit_payload(
                status="contract_rejected",
                reason=f"approved_problem.json failed contract validation: {validation_issues}",
                approved_problem=approved_problem,
                gate_stage_reached="contract_validation",
                candidate_config=candidate_config,
            )
            verdict_payload["validation_issues"] = validation_issues
            status["phase"] = "contract_rejected"
            return_code = 1
            return return_code
        if not candidate_config:
            verdict_payload = build_contract_exit_payload(
                status="contract_rejected",
                reason="approved_problem.json does not specify first_candidate_config.",
                approved_problem=approved_problem,
                gate_stage_reached="contract_validation",
            )
            status["phase"] = "contract_rejected"
            return_code = 1
            return return_code
        if bool(approved_problem.get("first_candidate_requires_code_patch", False)):
            verdict_payload = build_contract_exit_payload(
                status="contract_rejected",
                reason="This runner only supports config-only approved candidates.",
                approved_problem=approved_problem,
                gate_stage_reached="contract_validation",
                candidate_config=candidate_config,
            )
            status["phase"] = "contract_rejected"
            return_code = 1
            return return_code

        candidate_config_path = normalize_repo_path(candidate_config)
        if not candidate_config_path.exists():
            verdict_payload = build_contract_exit_payload(
                status="contract_rejected",
                reason=f"Candidate config does not exist: {candidate_config_path}",
                approved_problem=approved_problem,
                gate_stage_reached="contract_validation",
                candidate_config=str(candidate_config_path),
            )
            status["phase"] = "contract_rejected"
            return_code = 1
            return return_code

        try:
            gate_reference_logs = load_gate_reference_logs(args.gate_reference_logs_path)
        except Exception as exc:
            verdict_payload = build_contract_exit_payload(
                status="reference_missing",
                reason=str(exc),
                approved_problem=approved_problem,
                gate_stage_reached="reference_load",
                candidate_config=str(candidate_config_path),
            )
            status["phase"] = "reference_missing"
            return_code = 1
            return return_code

        stable_short_log = gate_reference_logs["stable_short_log"]
        baseline_short_log = gate_reference_logs["baseline_short_log"]
        stable_long_log = gate_reference_logs["stable_long_log"]
        baseline_long_log = gate_reference_logs["baseline_long_log"]
        status.update(
            {
                "candidate_config": str(candidate_config_path),
                "stable_short_reference_log": str(stable_short_log),
                "baseline_short_reference_log": str(baseline_short_log),
                "stable_long_reference_log": str(stable_long_log),
                "baseline_long_reference_log": str(baseline_long_log),
            }
        )
        write_json(status_path, status)

        allowlist_markers = [
            "run_zju_source_policy_research_candidate.py",
            "run_zju_vggt_geom_minimal_finetune.ps1",
            "compare_zju_finetune_runs.py",
        ]
        write_json(args.repo_process_allowlist_path, build_allowlist(allowlist_markers))
        status["allowlist_markers"] = allowlist_markers
        status["phase"] = "preflight"
        write_json(status_path, status)
        run_preflight()

        smoke_exp = f"zju_source_policy_candidate_{candidate_stem}_smoke1x1_{run_tag}"
        status["phase"] = "smoke_1x1"
        write_json(status_path, status)
        smoke_log = run_finetune(
            python_exe=args.python_exe,
            config_path=candidate_config_path,
            exp_name=smoke_exp,
            limit_train_batches=args.smoke_train_batches,
            limit_val_batches=args.smoke_val_batches,
            run_dir=run_dir,
            stage_name="smoke_1x1",
        )
        status["artifacts"]["smoke_log"] = str(smoke_log)
        write_json(status_path, status)

        short_exp = f"zju_source_policy_candidate_{candidate_stem}_gate10x5_{run_tag}"
        status["phase"] = "short_gate_10x5"
        write_json(status_path, status)
        short_log = run_finetune(
            python_exe=args.python_exe,
            config_path=candidate_config_path,
            exp_name=short_exp,
            limit_train_batches=args.short_train_batches,
            limit_val_batches=args.short_val_batches,
            run_dir=run_dir,
            stage_name="short_gate_10x5",
        )
        status["artifacts"]["short_log"] = str(short_log)

        short_vs_lead_dir = ensure_dir(run_dir / "short_vs_lead")
        short_vs_lead = run_compare(
            python_exe=args.python_exe,
            baseline_log=stable_short_log,
            candidate_log=short_log,
            baseline_label="stable_lead",
            candidate_label=candidate_stem,
            output_dir=short_vs_lead_dir,
            title=f"Approved Research Candidate 10/5: stable lead vs {candidate_stem}",
        )
        short_vs_baseline_dir = ensure_dir(run_dir / "short_vs_baseline")
        short_vs_baseline = run_compare(
            python_exe=args.python_exe,
            baseline_log=baseline_short_log,
            candidate_log=short_log,
            baseline_label="baseline",
            candidate_label=candidate_stem,
            output_dir=short_vs_baseline_dir,
            title=f"Approved Research Candidate 10/5: baseline vs {candidate_stem}",
        )
        status["artifacts"]["short_vs_lead_summary"] = str(short_vs_lead_dir / "summary.md")
        status["artifacts"]["short_vs_baseline_summary"] = str(short_vs_baseline_dir / "summary.md")
        write_json(status_path, status)

        if not short_gate_beats_stable_lead(short_vs_lead):
            verdict_payload = build_candidate_verdict_payload(
                status="dead_same_day",
                candidate_config=str(candidate_config_path),
                approved_problem=approved_problem,
                gate_stage_reached="short_gate_10x5",
                summary_vs_lead=short_vs_lead,
                summary_vs_baseline=short_vs_baseline,
                reason="Short gate did not beat the stable lead on camera/T/conf_depth/reg_depth simultaneously.",
            )
            verdict_status_for_archive = verdict_payload["status"]
            status["phase"] = "completed_dead_same_day"
            return_code = 0
            return return_code

        long_gate_tag = run_tag
        status["phase"] = "long_gate_100x20"
        write_json(status_path, status)
        status["artifacts"]["long_gate_baseline_reference_log"] = str(baseline_long_log)
        status["artifacts"]["long_gate_stable_reference_log"] = str(stable_long_log)
        candidate_long_log = run_finetune(
            python_exe=args.python_exe,
            config_path=candidate_config_path,
            exp_name=f"zju_source_policy_candidate_{candidate_stem}_longgate100x20_{long_gate_tag}",
            limit_train_batches=args.long_train_batches,
            limit_val_batches=args.long_val_batches,
            run_dir=run_dir,
            stage_name="long_gate_candidate",
        )
        status["artifacts"]["long_candidate_log"] = str(candidate_long_log)
        long_vs_lead_dir = ensure_dir(run_dir / "long_vs_lead")
        long_vs_lead = run_compare(
            python_exe=args.python_exe,
            baseline_log=stable_long_log,
            candidate_log=candidate_long_log,
            baseline_label="stable_lead",
            candidate_label=candidate_stem,
            output_dir=long_vs_lead_dir,
            title=f"Approved Research Candidate 100/20: stable lead vs {candidate_stem}",
        )
        long_vs_baseline_dir = ensure_dir(run_dir / "long_vs_baseline")
        long_vs_baseline = run_compare(
            python_exe=args.python_exe,
            baseline_log=baseline_long_log,
            candidate_log=candidate_long_log,
            baseline_label="baseline",
            candidate_label=candidate_stem,
            output_dir=long_vs_baseline_dir,
            title=f"Approved Research Candidate 100/20: baseline vs {candidate_stem}",
        )
        status["artifacts"]["long_vs_lead_summary"] = str(long_vs_lead_dir / "summary.md")
        status["artifacts"]["long_vs_baseline_summary"] = str(long_vs_baseline_dir / "summary.md")
        if short_gate_beats_stable_lead(long_vs_lead):
            verdict_payload = build_candidate_verdict_payload(
                status="provisional_lead",
                candidate_config=str(candidate_config_path),
                approved_problem=approved_problem,
                gate_stage_reached="long_gate_100x20",
                summary_vs_lead=long_vs_lead,
                summary_vs_baseline=long_vs_baseline,
                reason="Long gate also beat the stable lead.",
            )
            status["phase"] = "completed_provisional_lead"
        else:
            verdict_payload = build_candidate_verdict_payload(
                status="failed_long_gate",
                candidate_config=str(candidate_config_path),
                approved_problem=approved_problem,
                gate_stage_reached="long_gate_100x20",
                summary_vs_lead=long_vs_lead,
                summary_vs_baseline=long_vs_baseline,
                reason="Short gate won, but long gate failed to preserve promotion metrics against the stable lead.",
            )
            status["phase"] = "completed_failed_long_gate"
        verdict_status_for_archive = verdict_payload["status"]
        return_code = 0
        return return_code
    except Exception as exc:
        return_code = 1
        active_candidate = str(candidate_config_path) if candidate_config_path is not None else candidate_config
        verdict_payload = build_contract_exit_payload(
            status="runner_error",
            reason=str(exc),
            approved_problem=approved_problem,
            gate_stage_reached=str(status.get("phase", "runner_error")),
            candidate_config=active_candidate,
        )
        status["phase"] = "runner_error"
        status["error"] = str(exc)
    finally:
        verdict_status_for_archive = verdict_payload.get("status", verdict_status_for_archive)
        finalize_runner(
            args=args,
            run_dir=run_dir,
            status=status,
            status_path=status_path,
            verdict_payload=verdict_payload,
            verdict_status_for_archive=verdict_status_for_archive,
        )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
