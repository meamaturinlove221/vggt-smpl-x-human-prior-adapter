import json
import os
import shlex
import subprocess
import sys
import threading
import time
import traceback
import base64
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import modal

from modal_zju_geometry_minimal_finetune import (
    CPU_COUNT,
    LIVE_COMMIT_INTERVAL_SEC,
    MEMORY_MB,
    REMOTE_CODE_DIR,
    REMOTE_DATA_DIR,
    REMOTE_OUTPUT_DIR,
    TIMEOUT_SEC,
    TRAINING_IMAGE,
    _remote_data_path,
    _resolve_checkpoint_path,
    data_volume,
    output_volume,
)


APP_NAME = os.environ.get("VGGT_ZJU_VISUAL_LIFT_APP_NAME", "vggt-zju-visual-lift-benchmark")
_GPU_SPEC_RAW = os.environ.get("VGGT_ZJU_VISUAL_LIFT_GPU", os.environ.get("VGGT_ZJU_MODAL_GPU", "A100-80GB"))
GPU_SPEC = None if str(_GPU_SPEC_RAW).strip().lower() in {"", "cpu", "none"} else _GPU_SPEC_RAW
VISUAL_LIFT_CPU_COUNT = float(os.environ.get("VGGT_ZJU_VISUAL_LIFT_CPU", "8"))
VISUAL_LIFT_MEMORY_MB = int(os.environ.get("VGGT_ZJU_VISUAL_LIFT_MEMORY_MB", "32768"))


app = modal.App(APP_NAME)


@dataclass
class VisualLiftBenchmarkConfig:
    zju_subdir: str
    checkpoint_subpath: str
    output_subdir: str
    manifest_blob: str
    case_set: str = "benchmark_cases"
    variants: str = "mask_hole_fill_plus_guided"
    device: str = "cuda"
    dtype: str = "bfloat16"
    render_size_h: int = 518
    render_size_w: int = 518
    render_max_points: int = 750000
    z_tolerance: float = 0.02
    min_conf: float = 1e-6
    target_mask_source: str = "mask"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(blob: str) -> "VisualLiftBenchmarkConfig":
        text = str(blob)
        if text.startswith("base64:"):
            text = base64.b64decode(text[len("base64:") :]).decode("utf-8")
        return VisualLiftBenchmarkConfig(**json.loads(text))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _commit_if_needed(commit_callback) -> None:
    if commit_callback is None:
        return
    commit_callback()


def _extract_progress(progress_path: Path) -> dict:
    if not progress_path.exists():
        return {}
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        "completed_case_count": int(payload.get("completed_case_count", 0)),
        "case_count": int(payload.get("case_count", 0)),
        "latest_case_id": str(payload.get("latest_case_id", "")),
    }


def _build_eval_command(cfg: VisualLiftBenchmarkConfig, manifest_path: Path, checkpoint_path: Path, output_root: Path) -> tuple[list[str], dict[str, str]]:
    eval_output_dir = output_root / "eval"
    remote_code_dir = Path(str(REMOTE_CODE_DIR))
    eval_script = remote_code_dir / "scripts" / "evaluate_teacher_visual_lift_cases.py"
    cmd = [
        sys.executable,
        str(eval_script),
        "--manifest-json",
        str(manifest_path),
        "--case-set",
        cfg.case_set,
        "--output-dir",
        str(eval_output_dir),
        "--checkpoint",
        checkpoint_path.as_posix(),
        "--variants",
        cfg.variants,
        "--local-zju-root",
        _remote_data_path(cfg.zju_subdir).as_posix(),
        "--device",
        cfg.device,
        "--dtype",
        cfg.dtype,
        "--render-size",
        str(int(cfg.render_size_h)),
        str(int(cfg.render_size_w)),
        "--render-max-points",
        str(int(cfg.render_max_points)),
        "--z-tolerance",
        str(float(cfg.z_tolerance)),
        "--min-conf",
        str(float(cfg.min_conf)),
        "--target-mask-source",
        cfg.target_mask_source,
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = str(remote_code_dir)
    return cmd, env


def _write_launch_artifacts(
    *,
    output_root: Path,
    manifest_text: str,
    checkpoint_path: Path,
    cmd: list[str],
    cwd: Path,
    heartbeat_interval_sec: int,
    commit_callback,
) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_copy = output_root / "manifest.copy.json"
    stdout_log = output_root / "stdout.log"
    cloud_status = output_root / "cloud_status.json"
    heartbeat = output_root / "heartbeat.json"
    heartbeat_history = output_root / "heartbeat.history.jsonl"
    success_json = output_root / "success.json"
    exception_json = output_root / "exception.json"
    command_json = output_root / "command.json"
    timing_json = output_root / "timing.json"
    resolved_checkpoint_path_txt = output_root / "resolved_checkpoint_path.txt"
    _write_text(manifest_copy, manifest_text)
    _write_text(stdout_log, "")
    _write_text(resolved_checkpoint_path_txt, checkpoint_path.as_posix() + "\n")
    launched_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_json(
        command_json,
        {
            "cmd": cmd,
            "cwd": str(cwd),
            "cmd_shell_joined": shlex.join(cmd),
        },
    )
    _write_json(
        timing_json,
        {
            "launched_at": launched_at,
            "started_at": None,
            "ended_at": None,
            "duration_sec": None,
            "heartbeat_interval_sec": int(heartbeat_interval_sec),
        },
    )
    launch_payload = {
        "status": "launched",
        "launched_at": launched_at,
        "output_root": output_root.as_posix(),
        "stdout_log": stdout_log.as_posix(),
        "manifest_copy": manifest_copy.as_posix(),
        "resolved_checkpoint_path": checkpoint_path.as_posix(),
    }
    _write_json(cloud_status, launch_payload)
    _write_json(
        heartbeat,
        {
            "status": "launched",
            "checked_at": launched_at,
            "stdout_log_size_bytes": 0,
            "stdout_line_count": 0,
            "recent_stdout_tail": [],
            "progress": {},
        },
    )
    _append_jsonl(
        heartbeat_history,
        {
            "status": "launched",
            "checked_at": launched_at,
            "stdout_log_size_bytes": 0,
            "stdout_line_count": 0,
            "recent_stdout_tail": [],
            "progress": {},
        },
    )
    _commit_if_needed(commit_callback)
    return {
        "manifest_copy": manifest_copy,
        "stdout_log": stdout_log,
        "cloud_status": cloud_status,
        "heartbeat": heartbeat,
        "heartbeat_history": heartbeat_history,
        "success_json": success_json,
        "exception_json": exception_json,
        "command_json": command_json,
        "timing_json": timing_json,
        "resolved_checkpoint_path_txt": resolved_checkpoint_path_txt,
    }


def _run_monitored_subprocess(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    output_root: Path,
    manifest_text: str,
    checkpoint_path: Path,
    commit_callback=None,
    heartbeat_interval_sec: int = LIVE_COMMIT_INTERVAL_SEC,
) -> dict:
    paths = _write_launch_artifacts(
        output_root=output_root,
        manifest_text=manifest_text,
        checkpoint_path=checkpoint_path,
        cmd=cmd,
        cwd=cwd,
        heartbeat_interval_sec=heartbeat_interval_sec,
        commit_callback=commit_callback,
    )
    eval_output_dir = output_root / "eval"
    progress_path = eval_output_dir / "progress.json"
    status_path = paths["cloud_status"]
    heartbeat_path = paths["heartbeat"]
    heartbeat_history_path = paths["heartbeat_history"]
    stdout_log_path = paths["stdout_log"]
    timing_path = paths["timing_json"]
    success_path = paths["success_json"]
    exception_path = paths["exception_json"]

    start_time = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_json(
        status_path,
        {
            "status": "running",
            "launched_at": start_iso,
            "started_at": start_iso,
            "output_root": output_root.as_posix(),
            "stdout_log": stdout_log_path.as_posix(),
            "resolved_checkpoint_path": checkpoint_path.as_posix(),
        },
    )
    _write_json(
        timing_path,
        {
            "launched_at": start_iso,
            "started_at": start_iso,
            "ended_at": None,
            "duration_sec": None,
            "heartbeat_interval_sec": int(heartbeat_interval_sec),
        },
    )
    _commit_if_needed(commit_callback)

    stdout_state = {
        "line_count": 0,
        "log_size_bytes": 0,
        "tail": deque(maxlen=32),
    }
    stdout_lock = threading.Lock()

    process = None
    reader_thread = None
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        def _reader() -> None:
            assert process is not None and process.stdout is not None
            with stdout_log_path.open("a", encoding="utf-8", errors="replace") as handle:
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    with stdout_lock:
                        stdout_state["line_count"] += 1
                        stdout_state["log_size_bytes"] = handle.tell()
                        stdout_state["tail"].append(line.rstrip("\n"))

        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()

        while process.poll() is None:
            time.sleep(max(min(int(heartbeat_interval_sec), 5), 1))
            with stdout_lock:
                heartbeat_payload = {
                    "status": "running",
                    "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "stdout_log_size_bytes": int(stdout_state["log_size_bytes"]),
                    "stdout_line_count": int(stdout_state["line_count"]),
                    "recent_stdout_tail": list(stdout_state["tail"]),
                    "progress": _extract_progress(progress_path),
                }
            _write_json(heartbeat_path, heartbeat_payload)
            _append_jsonl(heartbeat_history_path, heartbeat_payload)
            _commit_if_needed(commit_callback)

        return_code = process.wait()
        if reader_thread is not None:
            reader_thread.join(timeout=max(int(heartbeat_interval_sec), 1))
        end_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        duration_sec = float(time.time() - start_time)
        with stdout_lock:
            final_heartbeat = {
                "status": "succeeded" if return_code == 0 else "failed",
                "checked_at": end_iso,
                "stdout_log_size_bytes": int(stdout_state["log_size_bytes"]),
                "stdout_line_count": int(stdout_state["line_count"]),
                "recent_stdout_tail": list(stdout_state["tail"]),
                "progress": _extract_progress(progress_path),
            }
        _write_json(heartbeat_path, final_heartbeat)
        _append_jsonl(heartbeat_history_path, final_heartbeat)
        _write_json(
            timing_path,
            {
                "launched_at": start_iso,
                "started_at": start_iso,
                "ended_at": end_iso,
                "duration_sec": duration_sec,
                "heartbeat_interval_sec": int(heartbeat_interval_sec),
            },
        )
        if return_code == 0:
            success_payload = {
                "status": "succeeded",
                "ended_at": end_iso,
                "duration_sec": duration_sec,
                "summary_json": (eval_output_dir / "summary.json").as_posix(),
                "summary_md": (eval_output_dir / "summary.md").as_posix(),
            }
            _write_json(success_path, success_payload)
            _write_json(
                status_path,
                {
                    "status": "succeeded",
                    "launched_at": start_iso,
                    "started_at": start_iso,
                    "ended_at": end_iso,
                    "duration_sec": duration_sec,
                    "output_root": output_root.as_posix(),
                    "stdout_log": stdout_log_path.as_posix(),
                    "summary_json": (eval_output_dir / "summary.json").as_posix(),
                    "summary_md": (eval_output_dir / "summary.md").as_posix(),
                    "resolved_checkpoint_path": checkpoint_path.as_posix(),
                },
            )
            _commit_if_needed(commit_callback)
            return success_payload

        exception_payload = {
            "status": "failed",
            "ended_at": end_iso,
            "duration_sec": duration_sec,
            "returncode": int(return_code),
            "stdout_log": stdout_log_path.as_posix(),
            "recent_stdout_tail": list(final_heartbeat["recent_stdout_tail"]),
            "progress": final_heartbeat["progress"],
        }
        _write_json(exception_path, exception_payload)
        _write_json(
            status_path,
            {
                "status": "failed",
                "launched_at": start_iso,
                "started_at": start_iso,
                "ended_at": end_iso,
                "duration_sec": duration_sec,
                "returncode": int(return_code),
                "output_root": output_root.as_posix(),
                "stdout_log": stdout_log_path.as_posix(),
                "exception_json": exception_path.as_posix(),
                "resolved_checkpoint_path": checkpoint_path.as_posix(),
            },
        )
        _commit_if_needed(commit_callback)
        raise subprocess.CalledProcessError(return_code, cmd)
    except Exception as exc:
        end_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        duration_sec = float(time.time() - start_time)
        exception_payload = {
            "status": "failed",
            "ended_at": end_iso,
            "duration_sec": duration_sec,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "stdout_log": stdout_log_path.as_posix(),
        }
        _write_json(exception_path, exception_payload)
        _write_json(
            timing_path,
            {
                "launched_at": start_iso,
                "started_at": start_iso,
                "ended_at": end_iso,
                "duration_sec": duration_sec,
                "heartbeat_interval_sec": int(heartbeat_interval_sec),
            },
        )
        _write_json(
            status_path,
            {
                "status": "failed",
                "launched_at": start_iso,
                "started_at": start_iso,
                "ended_at": end_iso,
                "duration_sec": duration_sec,
                "exception_json": exception_path.as_posix(),
                "stdout_log": stdout_log_path.as_posix(),
                "output_root": output_root.as_posix(),
                "resolved_checkpoint_path": checkpoint_path.as_posix(),
            },
        )
        with stdout_lock:
            failure_heartbeat = {
                "status": "failed",
                "checked_at": end_iso,
                "stdout_log_size_bytes": int(stdout_state["log_size_bytes"]),
                "stdout_line_count": int(stdout_state["line_count"]),
                "recent_stdout_tail": list(stdout_state["tail"]),
                "progress": _extract_progress(progress_path),
            }
        _write_json(heartbeat_path, failure_heartbeat)
        _append_jsonl(heartbeat_history_path, failure_heartbeat)
        _commit_if_needed(commit_callback)
        raise


def _run_benchmark(cfg: VisualLiftBenchmarkConfig) -> str:
    checkpoint_path = _resolve_checkpoint_path(cfg.checkpoint_subpath)
    output_root = Path(str(REMOTE_OUTPUT_DIR / cfg.output_subdir.strip("/")))
    manifest_path = output_root / "manifest.copy.json"
    cmd, env = _build_eval_command(cfg, manifest_path, checkpoint_path, output_root)
    _run_monitored_subprocess(
        cmd=cmd,
        cwd=Path(str(REMOTE_CODE_DIR)),
        env=env,
        output_root=output_root,
        manifest_text=cfg.manifest_blob,
        checkpoint_path=checkpoint_path,
        commit_callback=output_volume.commit,
        heartbeat_interval_sec=LIVE_COMMIT_INTERVAL_SEC,
    )
    return output_root.as_posix()


def run_local_contract_rehearsal(output_root: str, *, should_fail: bool, heartbeat_interval_sec: int = 1) -> dict:
    output_root_path = Path(output_root).resolve()
    manifest_text = json.dumps({"cases": [{"case_id": "rehearsal_case"}]}, ensure_ascii=False, indent=2)
    checkpoint_path = output_root_path / "dummy_checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("dummy-checkpoint\n", encoding="utf-8")
    script = (
        "import sys, time, json, pathlib;"
        "root = pathlib.Path(sys.argv[1]);"
        "eval_root = root / 'eval';"
        "eval_root.mkdir(parents=True, exist_ok=True);"
        "print('rehearsal: start', flush=True);"
        "time.sleep(1.2);"
        "print('rehearsal: midpoint', flush=True);"
        "progress = {'completed_case_count': 1, 'case_count': 1, 'latest_case_id': 'rehearsal_case'};"
        "(eval_root / 'progress.json').write_text(json.dumps(progress), encoding='utf-8');"
        "time.sleep(1.2);"
        "print('rehearsal: finalizing', flush=True);"
        "(eval_root / 'summary.partial.json').write_text(json.dumps({'status':'partial'}), encoding='utf-8');"
        "time.sleep(1.2);"
        "ret = int(sys.argv[2]);"
        "success = {'ok': True};"
        "(eval_root / 'summary.json').write_text(json.dumps(success), encoding='utf-8');"
        "(eval_root / 'summary.md').write_text('# rehearsal\\n', encoding='utf-8');"
        "sys.exit(ret)"
    )
    cmd = [sys.executable, "-c", script, str(output_root_path), "3" if should_fail else "0"]
    result = {"status": "unknown"}
    try:
        result = _run_monitored_subprocess(
            cmd=cmd,
            cwd=Path.cwd(),
            env=os.environ.copy(),
            output_root=output_root_path,
            manifest_text=manifest_text,
            checkpoint_path=checkpoint_path,
            commit_callback=None,
            heartbeat_interval_sec=max(int(heartbeat_interval_sec), 1),
        )
        result["status"] = "succeeded"
    except subprocess.CalledProcessError:
        result = {"status": "failed_expected" if should_fail else "failed_unexpected"}
    result["output_root"] = output_root_path.as_posix()
    return result


@app.function(
    image=TRAINING_IMAGE,
    gpu=GPU_SPEC,
    cpu=VISUAL_LIFT_CPU_COUNT,
    memory=VISUAL_LIFT_MEMORY_MB,
    timeout=TIMEOUT_SEC,
    volumes={
        REMOTE_DATA_DIR.as_posix(): data_volume,
        REMOTE_OUTPUT_DIR.as_posix(): output_volume,
    },
)
def run_remote_visual_lift_benchmark(cfg_json: str) -> str:
    cfg = VisualLiftBenchmarkConfig.from_json(cfg_json)
    return _run_benchmark(cfg)


@app.local_entrypoint()
def run_zju_visual_lift_benchmark(
    manifest_path: str,
    checkpoint_subpath: str,
    output_subdir: str,
    zju_subdir: str = "zju_mocap",
    case_set: str = "benchmark_cases",
    variants: str = "mask_hole_fill_plus_guided",
    device: str = "cuda",
    dtype: str = "bfloat16",
    render_size_h: int = 518,
    render_size_w: int = 518,
    render_max_points: int = 750000,
    z_tolerance: float = 0.02,
    min_conf: float = 1e-6,
    target_mask_source: str = "mask",
) -> None:
    manifest_text = Path(manifest_path).read_text(encoding="utf-8")
    cfg = VisualLiftBenchmarkConfig(
        zju_subdir=zju_subdir,
        checkpoint_subpath=checkpoint_subpath,
        output_subdir=output_subdir,
        manifest_blob=manifest_text,
        case_set=case_set,
        variants=variants,
        device=device,
        dtype=dtype,
        render_size_h=render_size_h,
        render_size_w=render_size_w,
        render_max_points=render_max_points,
        z_tolerance=z_tolerance,
        min_conf=min_conf,
        target_mask_source=target_mask_source,
    )
    output_root = run_remote_visual_lift_benchmark.remote(cfg.to_json())
    print(output_root)
