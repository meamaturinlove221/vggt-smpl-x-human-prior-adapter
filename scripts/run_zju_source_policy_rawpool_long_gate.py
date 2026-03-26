import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MANIFEST = REPO_ROOT / "scripts" / "manifests" / "zju_source_policy_rawpool_local_nightly_v1.json"
TRAINING_QUESTION_MANIFEST = REPO_ROOT / "scripts" / "manifests" / "zju_next_training_question_v1.json"
NIGHTLY_RUNNER = REPO_ROOT / "scripts" / "run_zju_source_policy_rawpool_local_nightly.py"
FINETUNE_PS1 = REPO_ROOT / "scripts" / "run_zju_vggt_geom_minimal_finetune.ps1"
COMPARE_PY = REPO_ROOT / "scripts" / "compare_zju_finetune_runs.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_rawpool_long_gate"
DEFAULT_ABLATION_ROOT = REPO_ROOT / "output" / "zju_training_ablation"
DEFAULT_PREVIOUS_LEAD = (
    REPO_ROOT
    / "training"
    / "config"
    / "zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml"
)


class LongGateError(RuntimeError):
    pass


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


def normalize_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path.resolve()


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
        raise LongGateError(
            "Command failed with exit code {code}: {cmd}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}".format(
                code=result.returncode,
                cmd=" ".join(args),
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
            )
        )
    return result


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


def make_exp_name(label: str, gate_tag: str) -> str:
    return f"zju_source_policy_{label}_{gate_tag}"


def run_steady_hold(python_exe: str, run_dir: Path, stage_name: str) -> Path:
    result = run_checked(
        [python_exe, str(NIGHTLY_RUNNER), "--mode", "auto"],
        cwd=REPO_ROOT,
    )
    stdout = result.stdout.strip().splitlines()[-1].strip()
    decision_path = Path(stdout)
    if not decision_path.is_absolute():
        decision_path = (REPO_ROOT / decision_path).resolve()
    if not decision_path.exists():
        raise LongGateError(f"Nightly runner did not produce a valid decision path: {stdout}")
    stage_dir = ensure_dir(run_dir / stage_name)
    write_text(stage_dir / "stdout.txt", result.stdout)
    write_text(stage_dir / "stderr.txt", result.stderr)
    write_json(stage_dir / "decision_pointer.json", {"nightly_decision_json": str(decision_path)})
    return decision_path.parent


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
    args = [
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
    ]
    result = run_checked(args, cwd=REPO_ROOT)
    stage_dir = ensure_dir(run_dir / stage_name)
    write_text(stage_dir / "stdout.txt", result.stdout)
    write_text(stage_dir / "stderr.txt", result.stderr)
    log_path = REPO_ROOT / "training" / "logs" / exp_name / "log.txt"
    if not log_path.exists():
        raise LongGateError(f"Expected training log was not created: {log_path}")
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
        raise LongGateError(f"Expected compare summary was not created: {summary_path}")
    return load_json(summary_path)


def build_decision(current_vs_previous: dict | None, current_vs_baseline: dict) -> dict:
    baseline_gap = {
        "val_loss_conf_depth": metric_average(current_vs_baseline, "loss_conf_depth"),
        "val_loss_reg_depth": metric_average(current_vs_baseline, "loss_reg_depth"),
        "delta_vs_baseline_conf_depth": delta_average(current_vs_baseline, "loss_conf_depth"),
        "delta_vs_baseline_reg_depth": delta_average(current_vs_baseline, "loss_reg_depth"),
    }
    if current_vs_previous is None:
        return {
            "status": "baseline_only_completed",
            "reason": "Previous-lead control was skipped; baseline comparison only.",
            "baseline_gap": baseline_gap,
        }

    camera_delta = delta_average(current_vs_previous, "loss_camera")
    t_delta = delta_average(current_vs_previous, "loss_T")
    conf_delta = delta_average(current_vs_previous, "loss_conf_depth")
    reg_delta = delta_average(current_vs_previous, "loss_reg_depth")
    stable = (
        camera_delta is not None
        and t_delta is not None
        and conf_delta is not None
        and reg_delta is not None
        and camera_delta <= 0.0
        and t_delta <= 0.0
        and conf_delta < 0.0
        and reg_delta < 0.0
    )
    if stable:
        return {
            "status": "keep_current_local_lead_cloud_off",
            "reason": "Long gate preserves camera/T and keeps both conf_depth/reg_depth below the previous local lead, but baseline gaps remain.",
            "delta_vs_previous": {
                "val_loss_camera": camera_delta,
                "val_loss_T": t_delta,
                "val_loss_conf_depth": conf_delta,
                "val_loss_reg_depth": reg_delta,
            },
            "baseline_gap": baseline_gap,
        }
    return {
        "status": "current_lead_not_stable_on_long_gate",
        "reason": "Long gate did not preserve the current lead over the previous local lead on the promotion metrics.",
        "delta_vs_previous": {
            "val_loss_camera": camera_delta,
            "val_loss_T": t_delta,
            "val_loss_conf_depth": conf_delta,
            "val_loss_reg_depth": reg_delta,
        },
        "baseline_gap": baseline_gap,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a longer local validation gate around the current ZJU source-policy rawpool lead."
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--limit-train-batches", type=int, default=200)
    parser.add_argument("--limit-val-batches", type=int, default=40)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION_ROOT)
    parser.add_argument("--tag", default="20260326_v1")
    parser.add_argument(
        "--previous-lead-config",
        type=Path,
        default=DEFAULT_PREVIOUS_LEAD,
        help="Optional previous-lead config to compare against the current lead.",
    )
    parser.add_argument("--skip-previous-lead", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    local_manifest = load_json(LOCAL_MANIFEST)
    training_question_manifest = load_json(TRAINING_QUESTION_MANIFEST)

    current_lead_config = normalize_repo_path(local_manifest["current_lead"]["config"])
    baseline_config = normalize_repo_path(training_question_manifest["baseline_config"])
    previous_lead_config = normalize_repo_path(args.previous_lead_config)

    run_dir = ensure_dir(
        args.output_root
        / f"{now_tag()}_lead_validation_{args.limit_train_batches}x{args.limit_val_batches}"
    )
    status_path = run_dir / "status.json"
    status = {
        "phase": "initialized",
        "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_val_batches,
        "baseline_config": str(baseline_config),
        "current_lead_config": str(current_lead_config),
        "previous_lead_config": None if args.skip_previous_lead else str(previous_lead_config),
        "artifacts": {},
    }
    write_json(status_path, status)

    gate_tag = f"longgate_{args.limit_train_batches}t_{args.limit_val_batches}v_{args.tag}"
    baseline_exp = make_exp_name("baseline", gate_tag)
    current_exp = make_exp_name("current_lead", gate_tag)
    previous_exp = make_exp_name("previous_lead", gate_tag)

    status["phase"] = "initial_steady_hold"
    write_json(status_path, status)
    initial_hold_dir = run_steady_hold(args.python_exe, run_dir, "initial_steady_hold")
    status["artifacts"]["initial_steady_hold_dir"] = str(initial_hold_dir)
    write_json(status_path, status)

    status["phase"] = "run_baseline"
    write_json(status_path, status)
    baseline_log = run_finetune(
        python_exe=args.python_exe,
        config_path=baseline_config,
        exp_name=baseline_exp,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        run_dir=run_dir,
        stage_name="baseline",
    )
    status["artifacts"]["baseline_log"] = str(baseline_log)
    write_json(status_path, status)

    if not args.skip_previous_lead:
        status["phase"] = "run_previous_lead"
        write_json(status_path, status)
        previous_log = run_finetune(
            python_exe=args.python_exe,
            config_path=previous_lead_config,
            exp_name=previous_exp,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
            run_dir=run_dir,
            stage_name="previous_lead",
        )
        status["artifacts"]["previous_lead_log"] = str(previous_log)
        write_json(status_path, status)
    else:
        previous_log = None

    status["phase"] = "run_current_lead"
    write_json(status_path, status)
    current_log = run_finetune(
        python_exe=args.python_exe,
        config_path=current_lead_config,
        exp_name=current_exp,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        run_dir=run_dir,
        stage_name="current_lead",
    )
    status["artifacts"]["current_lead_log"] = str(current_log)
    write_json(status_path, status)

    status["phase"] = "compare"
    write_json(status_path, status)
    current_vs_baseline_dir = ensure_dir(
        args.ablation_root / f"{current_exp}_vs_baseline"
    )
    current_vs_baseline = run_compare(
        python_exe=args.python_exe,
        baseline_log=baseline_log,
        candidate_log=current_log,
        baseline_label="baseline",
        candidate_label=current_lead_config.stem,
        output_dir=current_vs_baseline_dir,
        title=f"ZJU Source-Policy Long Gate {args.limit_train_batches}/{args.limit_val_batches}: Current Lead vs Baseline",
    )
    status["artifacts"]["current_vs_baseline_summary"] = str(current_vs_baseline_dir / "summary.md")

    if previous_log is not None:
        current_vs_previous_dir = ensure_dir(
            args.ablation_root / f"{current_exp}_vs_previous_lead"
        )
        current_vs_previous = run_compare(
            python_exe=args.python_exe,
            baseline_log=previous_log,
            candidate_log=current_log,
            baseline_label=previous_lead_config.stem,
            candidate_label=current_lead_config.stem,
            output_dir=current_vs_previous_dir,
            title=f"ZJU Source-Policy Long Gate {args.limit_train_batches}/{args.limit_val_batches}: Current Lead vs Previous Lead",
        )
        status["artifacts"]["current_vs_previous_summary"] = str(current_vs_previous_dir / "summary.md")
    else:
        current_vs_previous = None

    decision = build_decision(current_vs_previous, current_vs_baseline)
    status["decision"] = decision
    write_json(run_dir / "long_gate_decision.json", decision)
    write_json(status_path, status)

    status["phase"] = "final_steady_hold"
    write_json(status_path, status)
    final_hold_dir = run_steady_hold(args.python_exe, run_dir, "final_steady_hold")
    status["artifacts"]["final_steady_hold_dir"] = str(final_hold_dir)
    status["phase"] = "completed"
    write_json(status_path, status)

    print(status_path)


if __name__ == "__main__":
    main()
