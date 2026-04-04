import base64
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import modal


REPO_ROOT = Path(__file__).resolve().parent
REMOTE_CODE_DIR = PurePosixPath("/workspace/vggt")
REMOTE_DATA_DIR = PurePosixPath("/mnt/data")
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")

APP_NAME = os.environ.get("VGGT_ZJU_VISUAL_LIFT_MIN_APP_NAME", "vggt-zju-visual-lift-benchmark-minimal-cpu")
DATA_VOLUME_NAME = os.environ.get("VGGT_ZJU_MODAL_DATA_VOLUME", os.environ.get("VGGT_MODAL_DATA_VOLUME", "vggt-zju-data"))
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_ZJU_MODAL_OUTPUT_VOLUME", os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-out"))
CPU_COUNT = float(os.environ.get("VGGT_ZJU_VISUAL_LIFT_MIN_CPU", "8"))
MEMORY_MB = int(os.environ.get("VGGT_ZJU_VISUAL_LIFT_MIN_MEMORY_MB", "32768"))
TIMEOUT_SEC = int(os.environ.get("VGGT_ZJU_VISUAL_LIFT_MIN_TIMEOUT_SEC", str(12 * 60 * 60)))
HEARTBEAT_SEC = int(os.environ.get("VGGT_ZJU_VISUAL_LIFT_MIN_HEARTBEAT_SEC", "5"))

REQUIREMENTS = [
    "torch==2.3.1",
    "torchvision==0.18.1",
    "numpy==1.26.1",
    "Pillow",
    "huggingface_hub",
    "einops",
    "safetensors",
    "opencv-python",
]

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install(*REQUIREMENTS)
    .add_local_dir(str(REPO_ROOT / "vggt"), remote_path=(REMOTE_CODE_DIR / "vggt").as_posix())
    .add_local_file(str(REPO_ROOT / "scripts" / "evaluate_teacher_visual_lift_cases.py"), remote_path=(REMOTE_CODE_DIR / "scripts" / "evaluate_teacher_visual_lift_cases.py").as_posix())
    .add_local_file(str(REPO_ROOT / "scripts" / "compare_geometry_branches_zju_report.py"), remote_path=(REMOTE_CODE_DIR / "scripts" / "compare_geometry_branches_zju_report.py").as_posix())
    .add_local_file(str(REPO_ROOT / "scripts" / "zju_geometry_region_utils.py"), remote_path=(REMOTE_CODE_DIR / "scripts" / "zju_geometry_region_utils.py").as_posix())
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=False)


@dataclass
class VisualLiftBenchmarkConfig:
    zju_subdir: str
    checkpoint_subpath: str
    output_subdir: str
    manifest_blob: str = ""
    manifest_subpath: str = ""
    case_set: str = "benchmark_cases"
    variants: str = "mask_hole_fill_plus_guided"
    device: str = "cpu"
    dtype: str = "float32"
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
        text = str(blob).lstrip("\ufeff")
        if text.startswith("base64:"):
            text = base64.b64decode(text[len("base64:") :]).decode("utf-8-sig")
        text = text.lstrip("\ufeff")
        return VisualLiftBenchmarkConfig(**json.loads(text))


def _normalize_subpath(value: str) -> str:
    cleaned = (value or "").strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("Expected a non-empty volume-relative path.")
    return cleaned


def _remote_data_path(subpath: str) -> Path:
    return Path(str(REMOTE_DATA_DIR / _normalize_subpath(subpath)))


def _remote_output_path(subpath: str) -> Path:
    return Path(str(REMOTE_OUTPUT_DIR / _normalize_subpath(subpath)))


def _resolve_manifest_text(cfg: VisualLiftBenchmarkConfig) -> str:
    if str(cfg.manifest_blob).strip():
        return str(cfg.manifest_blob)
    if str(cfg.manifest_subpath).strip():
        candidate = _remote_output_path(cfg.manifest_subpath)
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
        candidate = _remote_data_path(cfg.manifest_subpath)
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Could not resolve manifest_subpath={cfg.manifest_subpath}")
    raise ValueError("Either manifest_blob or manifest_subpath must be provided.")


def _resolve_checkpoint_path(remote_subpath: str) -> Path:
    candidate = _remote_data_path(remote_subpath)
    if candidate.exists():
        return candidate
    fallback = _remote_output_path(remote_subpath)
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not resolve checkpoint path for {remote_subpath}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def _run_monitored_subprocess(*, cmd: list[str], cwd: Path, env: dict[str, str], output_root: Path, manifest_text: str, checkpoint_path: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "cloud_status.json"
    heartbeat_path = output_root / "heartbeat.json"
    heartbeat_history = output_root / "heartbeat.history.jsonl"
    stdout_log = output_root / "stdout.log"
    command_json = output_root / "command.json"
    timing_json = output_root / "timing.json"
    exception_json = output_root / "exception.json"
    success_json = output_root / "success.json"
    manifest_copy = output_root / "manifest.copy.json"
    resolved_checkpoint_txt = output_root / "resolved_checkpoint_path.txt"
    eval_output_dir = output_root / "eval"
    progress_path = eval_output_dir / "progress.json"

    launched_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_text(stdout_log, "")
    _write_text(manifest_copy, manifest_text)
    _write_text(resolved_checkpoint_txt, checkpoint_path.as_posix() + "\n")
    _write_json(command_json, {"cmd": cmd, "cwd": str(cwd), "cmd_shell_joined": shlex.join(cmd)})
    _write_json(timing_json, {"launched_at": launched_at, "started_at": launched_at, "ended_at": None, "duration_sec": None})
    launch_payload = {
        "status": "launched",
        "launched_at": launched_at,
        "output_root": output_root.as_posix(),
        "stdout_log": stdout_log.as_posix(),
        "manifest_copy": manifest_copy.as_posix(),
        "resolved_checkpoint_path": checkpoint_path.as_posix(),
    }
    _write_json(status_path, launch_payload)
    _write_json(heartbeat_path, {"status": "launched", "checked_at": launched_at, "stdout_log_size_bytes": 0, "stdout_line_count": 0, "recent_stdout_tail": [], "progress": {}})
    _append_jsonl(heartbeat_history, {"status": "launched", "checked_at": launched_at, "stdout_log_size_bytes": 0, "stdout_line_count": 0, "recent_stdout_tail": [], "progress": {}})
    output_volume.commit()

    stdout_state = {"line_count": 0, "log_size_bytes": 0, "tail": deque(maxlen=32)}
    stdout_lock = threading.Lock()
    stop_event = threading.Event()

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
        assert process.stdout is not None
        with stdout_log.open("a", encoding="utf-8", errors="replace") as handle:
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                with stdout_lock:
                    stdout_state["line_count"] += 1
                    stdout_state["log_size_bytes"] = handle.tell()
                    stdout_state["tail"].append(line.rstrip("\n"))

    def _heartbeat() -> None:
        while not stop_event.wait(HEARTBEAT_SEC):
            with stdout_lock:
                payload = {
                    "status": "running",
                    "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "stdout_log_size_bytes": int(stdout_state["log_size_bytes"]),
                    "stdout_line_count": int(stdout_state["line_count"]),
                    "recent_stdout_tail": list(stdout_state["tail"]),
                    "progress": _extract_progress(progress_path),
                }
            _write_json(heartbeat_path, payload)
            _append_jsonl(heartbeat_history, payload)
            output_volume.commit()

    reader = threading.Thread(target=_reader, daemon=True)
    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    reader.start()
    heartbeat_thread.start()

    start_time = time.time()
    try:
        return_code = process.wait()
        reader.join(timeout=5)
        stop_event.set()
        heartbeat_thread.join(timeout=5)
        ended_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        duration = float(time.time() - start_time)
        with stdout_lock:
            final_hb = {
                "status": "succeeded" if return_code == 0 else "failed",
                "checked_at": ended_at,
                "stdout_log_size_bytes": int(stdout_state["log_size_bytes"]),
                "stdout_line_count": int(stdout_state["line_count"]),
                "recent_stdout_tail": list(stdout_state["tail"]),
                "progress": _extract_progress(progress_path),
            }
        _write_json(heartbeat_path, final_hb)
        _append_jsonl(heartbeat_history, final_hb)
        _write_json(timing_json, {"launched_at": launched_at, "started_at": launched_at, "ended_at": ended_at, "duration_sec": duration})
        if return_code == 0:
            _write_json(success_json, {"status": "succeeded", "summary_json": (eval_output_dir / "summary.json").as_posix(), "summary_md": (eval_output_dir / "summary.md").as_posix()})
            _write_json(status_path, {"status": "succeeded", "output_root": output_root.as_posix(), "stdout_log": stdout_log.as_posix(), "summary_json": (eval_output_dir / "summary.json").as_posix(), "summary_md": (eval_output_dir / "summary.md").as_posix(), "resolved_checkpoint_path": checkpoint_path.as_posix()})
            output_volume.commit()
            return
        _write_json(exception_json, {"status": "failed", "returncode": int(return_code), "stdout_log": stdout_log.as_posix(), "recent_stdout_tail": final_hb["recent_stdout_tail"], "progress": final_hb["progress"]})
        _write_json(status_path, {"status": "failed", "returncode": int(return_code), "output_root": output_root.as_posix(), "stdout_log": stdout_log.as_posix(), "exception_json": exception_json.as_posix(), "resolved_checkpoint_path": checkpoint_path.as_posix()})
        output_volume.commit()
        raise subprocess.CalledProcessError(return_code, cmd)
    except Exception as exc:
        stop_event.set()
        _write_json(exception_json, {"status": "failed", "exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(), "stdout_log": stdout_log.as_posix()})
        _write_json(status_path, {"status": "failed", "output_root": output_root.as_posix(), "stdout_log": stdout_log.as_posix(), "exception_json": exception_json.as_posix(), "resolved_checkpoint_path": checkpoint_path.as_posix()})
        output_volume.commit()
        raise


def _run_benchmark(cfg: VisualLiftBenchmarkConfig) -> str:
    checkpoint_path = _resolve_checkpoint_path(cfg.checkpoint_subpath)
    output_root = Path(str(REMOTE_OUTPUT_DIR / cfg.output_subdir.strip("/")))
    manifest_path = output_root / "manifest.copy.json"
    manifest_text = _resolve_manifest_text(cfg)
    eval_output_dir = output_root / "eval"
    remote_code_dir = Path(str(REMOTE_CODE_DIR))
    cmd = [
        sys.executable,
        str(remote_code_dir / "scripts" / "evaluate_teacher_visual_lift_cases.py"),
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
    _run_monitored_subprocess(
        cmd=cmd,
        cwd=remote_code_dir,
        env=env,
        output_root=output_root,
        manifest_text=manifest_text,
        checkpoint_path=checkpoint_path,
    )
    return output_root.as_posix()


@app.function(
    image=IMAGE,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SEC,
    volumes={
        REMOTE_DATA_DIR.as_posix(): data_volume,
        REMOTE_OUTPUT_DIR.as_posix(): output_volume,
    },
)
def run_remote_visual_lift_benchmark(cfg_json: str) -> str:
    cfg = VisualLiftBenchmarkConfig.from_json(cfg_json)
    return _run_benchmark(cfg)
