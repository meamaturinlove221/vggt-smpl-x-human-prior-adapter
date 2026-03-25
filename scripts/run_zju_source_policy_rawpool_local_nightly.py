import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_MANIFEST = REPO_ROOT / "scripts" / "manifests" / "zju_source_policy_rawpool_local_nightly_v1.json"
DEFAULT_TRAINING_QUESTION_MANIFEST = REPO_ROOT / "scripts" / "manifests" / "zju_next_training_question_v1.json"
DEFAULT_STATE_DIR = REPO_ROOT / "output" / "geometry_post_v9_nightly_state"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_rawpool_local_nightly"
PREFLIGHT_PS1 = REPO_ROOT / "scripts" / "invoke_modal_zju_preflight.ps1"
CONSISTENCY_PY = REPO_ROOT / "scripts" / "check_zju_post_v9_consistency.py"


class NightlyError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the local-only ZJU source-policy nightly hold protocol from machine-readable manifests."
    )
    parser.add_argument(
        "--local-manifest",
        type=Path,
        default=DEFAULT_LOCAL_MANIFEST,
        help="Path to the rawpool local nightly manifest.",
    )
    parser.add_argument(
        "--training-question-manifest",
        type=Path,
        default=DEFAULT_TRAINING_QUESTION_MANIFEST,
        help="Path to the next training question manifest.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=DEFAULT_STATE_DIR,
        help="Persistent nightly state directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for nightly run artifacts.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "steady_hold", "single_candidate_local_gate"],
        help="Nightly mode. 'auto' resolves to the manifest default.",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable used for subordinate checks.",
    )
    parser.add_argument(
        "--candidate-config",
        type=str,
        default="",
        help="Required only for single_candidate_local_gate.",
    )
    parser.add_argument(
        "--approval-note",
        type=str,
        default="",
        help="Required only for single_candidate_local_gate.",
    )
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
        raise NightlyError(
            "Command failed with exit code {code}: {cmd}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}".format(
                code=result.returncode,
                cmd=" ".join(args),
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
            )
        )
    return result


def resolve_mode(requested_mode: str, local_manifest: dict) -> str:
    if requested_mode != "auto":
        return requested_mode
    for mode in local_manifest.get("nightly_modes", []):
        if bool(mode.get("default")):
            return str(mode["name"])
    return "steady_hold"


def normalize_repo_path(path_like: str | Path) -> str:
    path = Path(path_like)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return str(path.resolve())


def validate_machine_readable_alignment(local_manifest: dict, training_question_manifest: dict) -> None:
    lead_config = normalize_repo_path(local_manifest["current_lead"]["config"])
    candidate_config = normalize_repo_path(training_question_manifest["candidate_config"])
    if lead_config != candidate_config:
        raise NightlyError(
            f"Lead mismatch between local manifest and training-question manifest: {lead_config} != {candidate_config}"
        )

    checks = (
        ("patch_collection_stop", bool(local_manifest["patch_collection_stop"]), bool(training_question_manifest["patch_collection_stop"])),
        ("ready_for_new_training_question", bool(local_manifest["ready_for_new_training_question"]), bool(training_question_manifest["ready_for_new_training_question"])),
        ("cloud_gate", bool(local_manifest["cloud_gate"]), bool(training_question_manifest["cloud_gate"])),
        ("launch_cloud_now", bool(local_manifest["launch_cloud_now"]), bool(training_question_manifest["launch_cloud_now"])),
        ("current_cloud_blocker", str(local_manifest["current_cloud_blocker"]), str(training_question_manifest["current_cloud_blocker"])),
    )
    mismatches = [name for name, lhs, rhs in checks if lhs != rhs]
    if mismatches:
        raise NightlyError(f"Manifest mismatch on fields: {', '.join(mismatches)}")


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    return load_json(state_path)


def build_training_question_brief(local_manifest: dict, training_question_manifest: dict, mode: str, reason: str) -> dict:
    return {
        "status": str(training_question_manifest["status"]),
        "candidate_config": str(training_question_manifest["candidate_config"]),
        "problem_family": str(training_question_manifest.get("problem_family", "")),
        "recommended_problem_id": str(training_question_manifest.get("recommended_problem_id", "")),
        "geometry_direction": str(training_question_manifest.get("geometry_direction", local_manifest.get("main_direction", ""))),
        "current_cloud_blocker": str(local_manifest["current_cloud_blocker"]),
        "patch_collection_stop": bool(local_manifest["patch_collection_stop"]),
        "ready_for_new_training_question": bool(local_manifest["ready_for_new_training_question"]),
        "cloud_gate": bool(local_manifest["cloud_gate"]),
        "launch_cloud_now": bool(local_manifest["launch_cloud_now"]),
        "nightly_mode": mode,
        "reason": reason,
        "single_source_of_truth": list(local_manifest.get("single_source_of_truth", [])),
    }


def build_decision(local_manifest: dict, mode: str, state: str, reason: str, artifacts: dict) -> dict:
    return {
        "state": state,
        "nightly_mode": mode,
        "latest_decision": mode,
        "current_lead_config": str(local_manifest["current_lead"]["config"]),
        "patch_collection_stop": bool(local_manifest["patch_collection_stop"]),
        "ready_for_new_training_question": bool(local_manifest["ready_for_new_training_question"]),
        "cloud_gate": bool(local_manifest["cloud_gate"]),
        "launch_cloud_now": bool(local_manifest["launch_cloud_now"]),
        "current_cloud_blocker": str(local_manifest["current_cloud_blocker"]),
        "reason": reason,
        "artifacts": dict(artifacts),
    }


def update_state(
    state_path: Path,
    prior_state: dict,
    local_manifest: dict,
    run_dir: Path,
    brief_json: Path,
    latest_decision: str,
    reason: str,
) -> dict:
    next_state = dict(prior_state)
    next_state["patch_collection_stop"] = bool(local_manifest["patch_collection_stop"])
    next_state["ready_for_new_training_question"] = bool(local_manifest["ready_for_new_training_question"])
    next_state["cloud_gate"] = bool(local_manifest["cloud_gate"])
    next_state["launch_cloud_now"] = bool(local_manifest["launch_cloud_now"])
    next_state["current_lead_config"] = str(local_manifest["current_lead"]["config"])
    next_state["current_cloud_blocker"] = str(local_manifest["current_cloud_blocker"])
    next_state["latest_decision"] = latest_decision
    next_state["last_reason"] = reason
    next_state["latest_run_dir"] = str(run_dir.resolve())
    next_state["generated_training_question_brief"] = str(brief_json.resolve())
    next_state["generated_cloud_template"] = ""
    next_state["suggested_manifest_path"] = ""
    write_json(state_path, next_state)
    return next_state


def write_markdown(path: Path, title: str, fields: dict) -> None:
    lines = [f"# {title}", ""]
    for key, value in fields.items():
        if isinstance(value, dict):
            lines.extend([f"## {key}", ""])
            for sub_key, sub_value in value.items():
                lines.append(f"- {sub_key}: `{sub_value}`")
            lines.append("")
        else:
            lines.append(f"- {key}: `{value}`")
    if lines[-1] != "":
        lines.append("")
    write_text(path, "\n".join(lines))


def validate_single_candidate_request(local_manifest: dict, candidate_config: str, approval_note: str) -> None:
    if not candidate_config.strip():
        raise NightlyError("single_candidate_local_gate requires --candidate-config.")
    if not approval_note.strip():
        raise NightlyError("single_candidate_local_gate requires --approval-note.")

    normalized_candidate = normalize_repo_path(candidate_config)
    normalized_lead = normalize_repo_path(local_manifest["current_lead"]["config"])
    if normalized_candidate == normalized_lead:
        raise NightlyError("Candidate config matches the current lead; no new candidate exists.")

    frozen = {
        normalize_repo_path(item["config"])
        for item in local_manifest.get("frozen_non_reentry_candidates", [])
        if item.get("config")
    }
    if normalized_candidate in frozen:
        raise NightlyError("Candidate config is already in the frozen non-reentry list.")

    raise NightlyError(
        "single_candidate_local_gate is intentionally blocked until the candidate gate inputs are fully machine-readable. "
        "Current local automation only supports steady_hold without markdown parsing."
    )


def main():
    args = parse_args()
    local_manifest = load_json(args.local_manifest)
    training_question_manifest = load_json(args.training_question_manifest)
    validate_machine_readable_alignment(local_manifest, training_question_manifest)

    mode = resolve_mode(args.mode, local_manifest)
    if mode == "single_candidate_local_gate":
        validate_single_candidate_request(local_manifest, args.candidate_config, args.approval_note)

    if mode != "steady_hold":
        raise NightlyError(f"Unsupported mode for current local automation: {mode}")

    state_dir = ensure_dir(args.state_dir)
    output_root = ensure_dir(args.output_root)
    state_path = state_dir / "state.json"
    prior_state = load_state(state_path)
    run_dir = ensure_dir(output_root / f"{now_tag()}_{mode}")

    preflight = run_checked(
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
    preflight_stdout = run_dir / "preflight_stdout.txt"
    preflight_stderr = run_dir / "preflight_stderr.txt"
    write_text(preflight_stdout, preflight.stdout)
    write_text(preflight_stderr, preflight.stderr)

    reason = (
        "steady_hold: keep confdepth_dropworst_gradconfmask as the local lead, do not train, "
        "do not open a new candidate automatically, and keep cloud off until a fresh manual training question exists."
    )
    brief = build_training_question_brief(local_manifest, training_question_manifest, mode, reason)
    brief_json = run_dir / "next_training_question_brief.json"
    write_json(brief_json, brief)
    write_markdown(run_dir / "next_training_question_brief.md", "ZJU Next Training Question Brief", brief)

    artifacts = {
        "preflight_stdout": str(preflight_stdout.resolve()),
        "preflight_stderr": str(preflight_stderr.resolve()),
        "training_question_brief_json": str(brief_json.resolve()),
    }
    decision = build_decision(local_manifest, mode, "success", reason, artifacts)
    decision_json = run_dir / "nightly_decision.json"
    write_json(decision_json, decision)
    write_markdown(run_dir / "nightly_decision.md", "ZJU Source-Policy Rawpool Nightly Decision", decision)

    update_state(
        state_path=state_path,
        prior_state=prior_state,
        local_manifest=local_manifest,
        run_dir=run_dir,
        brief_json=brief_json,
        latest_decision=mode,
        reason=reason,
    )

    consistency = run_checked(
        [args.python_exe, str(CONSISTENCY_PY)],
        cwd=REPO_ROOT,
    )
    consistency_json = state_dir / "consistency_check.json"
    consistency_md = state_dir / "consistency_check.md"
    artifacts["consistency_check_json"] = str(consistency_json.resolve())
    artifacts["consistency_check_md"] = str(consistency_md.resolve())
    artifacts["consistency_stdout"] = consistency.stdout.strip()
    decision = build_decision(local_manifest, mode, "success", reason, artifacts)
    write_json(decision_json, decision)
    write_markdown(run_dir / "nightly_decision.md", "ZJU Source-Policy Rawpool Nightly Decision", decision)

    print(run_dir / "nightly_decision.json")


if __name__ == "__main__":
    try:
        main()
    except NightlyError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
