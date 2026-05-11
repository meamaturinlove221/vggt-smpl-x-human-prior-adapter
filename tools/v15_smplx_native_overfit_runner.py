from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
LOCAL_ROOT = REPO_ROOT / "output" / "surface_research_preflight_local"
WORKER_C_ROOT = LOCAL_ROOT / "V15_SMPLX_native_worker_C"

DEFAULT_OUTPUT_JSON = REPORTS / "20260508_v15_smplx_native_overfit_runner.json"
DEFAULT_OUTPUT_MD = REPORTS / "20260508_v15_smplx_native_overfit_runner.md"
DEFAULT_RUN_DIR = WORKER_C_ROOT / "native_overfit_runner"
DEFAULT_SCENE = REPO_ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_12views_tmf"
DEFAULT_RAW_SCENE = REPO_ROOT / "output" / "4k4d_preprocessed_scene_variants" / "0012_11_frame0000_60views_human_crop"
DEFAULT_ASSET_REPORT = REPORTS / "20260508_v15_required_licensed_assets.json"
DEFAULT_HAIR_HAND_REPORT = REPORTS / "20260508_v15_hair_hand_readiness.json"

FORBIDDEN_OUTPUT_TOKENS = (
    "predictions",
    "teacher_export",
    "candidate_export",
    "strict_gate_registry",
    "strict_pass",
    "formal_candidate",
    "candidate_gate",
)

RESEARCH_CONTRACT = {
    "research_only": True,
    "formal_cloud_unblocked": False,
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "no_predictions_write": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_registry_write": True,
    "no_strict_pass_write": True,
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def file_row(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve() if path.exists() else path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "size": path.stat().st_size if path.is_file() else 0,
    }


def safe_v15_research_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    lower = resolved.as_posix().lower()
    if "surface_research_preflight_local" not in lower or "v15_smplx_native_worker_c" not in lower:
        raise ValueError(f"Refusing non-Worker-C research output path: {resolved}")
    for token in FORBIDDEN_OUTPUT_TOKENS:
        if token in lower:
            raise ValueError(f"Refusing forbidden output token {token!r}: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def discover_worker_inputs(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = read_json(path)
        row = file_row(path)
        row.update(
            {
                "status": data.get("status"),
                "task": data.get("task"),
                "decision": data.get("decision"),
                "blocker_count": len(data.get("blockers", []) or []),
                "hand_ownership_ready": data.get("hand_ownership_ready"),
                "hair_ownership_ready": data.get("hair_ownership_ready"),
            }
        )
        rows.append(row)
    return rows


def smplx_asset_status(asset_report: dict[str, Any]) -> dict[str, Any]:
    asset_sets = asset_report.get("asset_sets", {}) if isinstance(asset_report.get("asset_sets"), dict) else {}
    smplx = asset_sets.get("smplx", {}) if isinstance(asset_sets.get("smplx"), dict) else {}
    return {
        "asset_report_status": asset_report.get("status"),
        "smplx_ready": bool(smplx.get("ok")),
        "smplx_present": smplx.get("present", []),
        "smplx_missing": smplx.get("missing", []),
        "manual_source": smplx.get("manual_source"),
    }


def command_for(args: argparse.Namespace, output_dir: Path) -> list[str]:
    scene = args.scene_dir.expanduser().resolve()
    if args.raw_scene_dir is not None and args.raw_scene_dir.expanduser().exists():
        scene = args.raw_scene_dir.expanduser().resolve()
    return [
        sys.executable,
        "tools/optimize_raw_smplx_softsurfel_torch.py",
        "--scene-dir",
        str(scene),
        "--output-dir",
        str(output_dir / "raw_softsurfel_local_attempt"),
        "--target-size",
        str(args.target_size),
        "--max-views",
        str(args.max_views),
        "--steps",
        str(args.steps),
        "--surfel-samples",
        str(args.surfel_samples),
        "--renderer",
        args.renderer,
        "--device",
        args.device,
        "--overwrite",
    ]


def run_command(cmd: list[str], timeout_sec: int) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": int(proc.returncode),
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-4000:],
            "elapsed_sec": round(time.time() - started, 3),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "cmd": cmd,
            "returncode": "timeout",
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
            "elapsed_sec": round(time.time() - started, 3),
            "timed_out": True,
        }


def load_attempt_summary(output_dir: Path) -> dict[str, Any]:
    candidates = [
        output_dir / "raw_softsurfel_local_attempt" / "raw_softsurfel_surface_summary.json",
        output_dir / "raw_softsurfel_local_attempt" / "raw_smplx_silhouette_optimization_summary.json",
        output_dir / "summary.json",
    ]
    for path in candidates:
        data = read_json(path)
        if data:
            data["_summary_path"] = str(path.resolve())
            return data
    return {}


def summarize_attempt(summary: dict[str, Any]) -> dict[str, Any]:
    if not summary:
        return {"present": False}
    metrics: dict[str, Any] = {"present": True, "summary_path": summary.get("_summary_path")}
    for key in ("status", "decision"):
        if key in summary:
            metrics[key] = summary[key]
    for key in ("final_loss", "initial_loss", "loss", "best_loss"):
        if key in summary:
            metrics[key] = summary[key]
    for key in ("mask_iou", "mean_iou", "final_mean_iou", "target_recall", "mean_target_recall"):
        if key in summary:
            metrics[key] = summary[key]
    if isinstance(summary.get("metrics"), dict):
        for key in ("mean_iou", "final_mean_iou", "mask_iou", "target_recall", "rgb_residual"):
            if key in summary["metrics"]:
                metrics[key] = summary["metrics"][key]
    return metrics


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V15 SMPL-X Native Overfit Runner",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only Worker C runner. It does not write predictions, teacher/candidate packages, registries, strict pass state, or formal cloud jobs.",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Command",
        "",
        "```powershell",
        " ".join(summary["overfit_command"]),
        "```",
        "",
        "## Inputs",
        "",
        f"- scene_ready: `{summary['input_readiness']['scene_ready']}`",
        f"- smplx_ready: `{summary['input_readiness']['smplx_ready']}`",
        f"- worker_a_b_present: `{summary['input_readiness']['worker_a_b_present']}`",
        "",
        "## Attempt",
        "",
        "```json",
        json.dumps(summary["attempt"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Research Gate / D-Line",
        "",
        f"- research_gate_result: `{summary['research_gate_result']}`",
        f"- dline_allowed: `{summary['dline_allowed']}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = summary.get("blockers") or []
    lines.extend([f"- {item}" for item in blockers] if blockers else ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="V15 Worker C SMPL-X native overfit runner/report.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--raw-scene-dir", type=Path, default=DEFAULT_RAW_SCENE)
    parser.add_argument("--asset-report", type=Path, default=DEFAULT_ASSET_REPORT)
    parser.add_argument("--hair-hand-report", type=Path, default=DEFAULT_HAIR_HAND_REPORT)
    parser.add_argument("--worker-input", type=Path, action="append", default=[])
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--max-views", type=int, default=6)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--surfel-samples", type=int, default=900)
    parser.add_argument("--renderer", choices=("surfel", "triangle"), default="surfel")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    output_dir = safe_v15_research_dir(args.output_dir)
    worker_paths = [
        REPORTS / "20260508_v15_hair_hand_readiness.json",
        REPORTS / "20260508_v15_fus3d_region_backend_dispatch.json",
        REPORTS / "20260508_v15_required_licensed_assets.json",
        LOCAL_ROOT / "B_Hair4_native_4k4d_smplx_hair_topology" / "summary.json",
        LOCAL_ROOT / "B_GS0_smplx_anchored_free_gaussian_smoke" / "b_gs0_summary.json",
        *args.worker_input,
    ]
    worker_inputs = discover_worker_inputs(worker_paths)
    asset_report = read_json(args.asset_report)
    asset_status = smplx_asset_status(asset_report)
    hair_hand = read_json(args.hair_hand_report)
    command = command_for(args, output_dir)

    scene_ready = args.scene_dir.expanduser().is_dir() or (args.raw_scene_dir is not None and args.raw_scene_dir.expanduser().is_dir())
    worker_a_b_present = any(row.get("exists") for row in worker_inputs)
    can_run = bool(scene_ready and asset_status["smplx_ready"])
    blockers: list[str] = []
    if not scene_ready:
        blockers.append("No bounded local 4K4D scene directory was found for the native overfit runner.")
    if not asset_status["smplx_ready"]:
        blockers.append("SMPL-X asset set is not ready according to the V15 licensed asset report.")
    if not worker_a_b_present:
        blockers.append("No Worker A/B SMPL-X native output summary was found; runner can only emit the next command contract.")
    if bool(hair_hand.get("hand_ownership_ready")) is False:
        blockers.append("Hand ownership remains false; native SMPL-X overfit cannot become a unified candidate.")
    if bool(hair_hand.get("hair_ownership_ready")) is False:
        blockers.append("Hair ownership remains false; native SMPL-X overfit cannot become a unified candidate.")

    run_result: dict[str, Any] = {"executed": False, "reason": "not requested"}
    if args.execute:
        if can_run:
            run_result = {"executed": True, **run_command(command, args.timeout_sec)}
        else:
            run_result = {"executed": False, "reason": "required inputs are missing"}

    attempt_summary = load_attempt_summary(output_dir)
    attempt = {
        "run_result": run_result,
        "summary": summarize_attempt(attempt_summary),
        "output_dir": str(output_dir),
    }
    overfit_observed = bool(attempt["summary"].get("present"))

    if overfit_observed:
        status = "v15_smplx_native_overfit_observed_research_only"
        decision = "A local native SMPL-X overfit artifact was observed, but Worker C writes no candidate, registry, strict pass, or formal cloud state."
    elif can_run:
        status = "v15_smplx_native_overfit_ready_to_run_research_only"
        decision = "Required local inputs are present. Worker C recorded a bounded local overfit command; execute with --execute only inside the research-only output root."
    else:
        status = "v15_smplx_native_overfit_blocked_missing_inputs"
        decision = "Worker C could not run the native SMPL-X overfit because required local artifacts are missing."

    summary = {
        "task": "v15_smplx_native_overfit_runner",
        "created_utc": utc_now(),
        "status": status,
        **RESEARCH_CONTRACT,
        "output_dir": output_dir,
        "input_readiness": {
            "scene_ready": scene_ready,
            "scene": file_row(args.scene_dir),
            "raw_scene": file_row(args.raw_scene_dir) if args.raw_scene_dir is not None else None,
            **asset_status,
            "worker_a_b_present": worker_a_b_present,
            "worker_inputs": worker_inputs,
        },
        "hair_hand_status": {
            "status": hair_hand.get("status"),
            "hand_ownership_ready": hair_hand.get("hand_ownership_ready"),
            "hair_ownership_ready": hair_hand.get("hair_ownership_ready"),
        },
        "overfit_command": command,
        "attempt": attempt,
        "overfit_observed": overfit_observed,
        "research_gate_result": "local_research_only_no_formal_cloud",
        "dline_allowed": False,
        "decision": decision,
        "blockers": blockers,
    }
    write_json(args.output_json, summary)
    write_markdown(args.output_md, summary)
    print(json.dumps(json_ready({"status": status, "json": args.output_json, "md": args.output_md}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
