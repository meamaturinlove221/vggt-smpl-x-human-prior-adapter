from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import modal


REPO_ROOT = Path(__file__).resolve().parent
REMOTE_CODE_DIR = PurePosixPath("/workspace/vggt")
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
APP_NAME = os.environ.get("VGGT_MODAL_V8_RESEARCH_APP_NAME", "vggt-v8-research-cloud")
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_V8_RESEARCH_TIMEOUT_SEC", str(2 * 60 * 60)))


CODE_SYNC_IGNORE = [
    ".git",
    ".git/**",
    "__pycache__",
    "__pycache__/**",
    ".venv*",
    ".venv*/**",
    "output",
    "output/**",
    "reports",
    "reports/**",
]


IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install("numpy==1.26.1", "Pillow", "scipy", "scikit-image")
    .add_local_dir(str(REPO_ROOT / "tools"), remote_path=(REMOTE_CODE_DIR / "tools").as_posix(), ignore=CODE_SYNC_IGNORE)
    .add_local_dir(str(REPO_ROOT / "training"), remote_path=(REMOTE_CODE_DIR / "training").as_posix(), ignore=CODE_SYNC_IGNORE)
)

app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


@dataclass
class V8ResearchJob:
    job_id: str
    lane: str
    output_subdir: str
    max_steps: int = 80
    max_cases: int = 20
    max_hours: float = 2.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(blob: str) -> "V8ResearchJob":
        return V8ResearchJob(**json.loads(blob))


def _normalize_subpath(value: str) -> str:
    cleaned = (value or "").strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("Expected a non-empty output subpath.")
    parts = Path(cleaned).parts
    if ".." in parts:
        raise ValueError(f"Parent traversal is not allowed: {value!r}")
    lower = cleaned.lower()
    if "surface_research_cloud_preflight" not in lower:
        raise ValueError("V8 research cloud output must include surface_research_cloud_preflight.")
    forbidden = ("strict_pass", "teacher_export", "candidate_export", "predictions", "formal_candidate")
    if any(word in lower for word in forbidden):
        raise ValueError(f"V8 research cloud output contains forbidden token: {value!r}")
    return cleaned


def _lane_command(job: V8ResearchJob, output_dir: Path) -> list[str]:
    remote_tools = Path(str(REMOTE_CODE_DIR)) / "tools"
    if job.lane == "b_fus3d2_human_dataset_train":
        return [
            sys.executable,
            str(remote_tools / "b_fus3d2_human_dataset_train.py"),
            "--output-dir",
            str(output_dir),
            "--max-steps",
            str(job.max_steps),
            "--max-cases",
            str(job.max_cases),
            "--max-hours",
            str(job.max_hours),
            "--overwrite",
        ]
    if job.lane == "a5x_external_dense_teacher_intake":
        return [
            sys.executable,
            str(remote_tools / "a5x_external_dense_teacher_intake_smoke.py"),
            "--output-dir",
            str(output_dir),
            "--max-cases",
            str(job.max_cases),
            "--max-hours",
            str(job.max_hours),
            "--overwrite",
        ]
    if job.lane == "b_hand10_hggt_style_hand_decoder":
        return [
            sys.executable,
            str(remote_tools / "b_hand10_hggt_style_hand_decoder_smoke.py"),
            "--output-dir",
            str(output_dir),
            "--max-steps",
            str(job.max_steps),
            "--max-cases",
            str(job.max_cases),
            "--max-hours",
            str(job.max_hours),
            "--overwrite",
        ]
    if job.lane == "b_hair3_hairgs_topology":
        return [
            sys.executable,
            str(remote_tools / "b_hair3_hairgs_topology_smoke.py"),
            "--output-dir",
            str(output_dir),
            "--max-steps",
            str(job.max_steps),
            "--max-cases",
            str(job.max_cases),
            "--max-hours",
            str(job.max_hours),
            "--overwrite",
        ]
    raise ValueError(f"Unsupported V8 lane: {job.lane}")


@app.function(
    image=IMAGE,
    cpu=4.0,
    memory=24 * 1024,
    timeout=TIMEOUT_SEC,
    volumes={REMOTE_OUTPUT_DIR.as_posix(): output_volume},
)
def run_v8_research_job(job_json: str) -> dict:
    job = V8ResearchJob.from_json(job_json)
    output_subdir = _normalize_subpath(job.output_subdir)
    output_dir = Path(str(REMOTE_OUTPUT_DIR / output_subdir))
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    guard = {
        "research_only": True,
        "no_export": True,
        "no_predictions_write": True,
        "no_registry_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_strict_pass_write": True,
        "formal_cloud_train_infer_export": "blocked",
        "job": asdict(job),
    }
    (output_dir / "v8_research_cloud_launch_guard.json").write_text(
        json.dumps(guard, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    cmd = _lane_command(job, output_dir)
    result = subprocess.run(cmd, cwd=str(REMOTE_CODE_DIR), check=False, capture_output=True, text=True)
    summary = {
        **guard,
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": int(result.returncode),
        "cmd": cmd,
        "stdout_tail": result.stdout[-12000:],
        "stderr_tail": result.stderr[-12000:],
        "elapsed_seconds": round(time.time() - started, 3),
        "output_subdir": output_subdir,
    }
    (output_dir / "v8_research_cloud_job_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_volume.commit()
    if result.returncode != 0:
        raise RuntimeError(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def _download_volume_dir(remote_subdir: str, local_dir: Path) -> None:
    remote_subdir = _normalize_subpath(remote_subdir)
    local_dir = local_dir.expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_prefix = Path(remote_subdir)
    for entry in output_volume.listdir(remote_subdir, recursive=True):
        rel_path = Path(entry.path)
        try:
            rel_path = rel_path.relative_to(remote_prefix)
        except ValueError:
            pass
        dest_path = local_dir / rel_path
        if entry.type == modal.volume.FileEntryType.DIRECTORY:
            dest_path.mkdir(parents=True, exist_ok=True)
            continue
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with dest_path.open("wb") as handle:
            output_volume.read_file_into_fileobj(entry.path, handle)


@app.local_entrypoint()
def run_job(
    job_id: str,
    lane: str,
    output_subdir: str,
    max_steps: int = 80,
    max_cases: int = 20,
    max_hours: float = 2.0,
    download_local_dir: str = "",
) -> None:
    job = V8ResearchJob(
        job_id=job_id,
        lane=lane,
        output_subdir=_normalize_subpath(output_subdir),
        max_steps=int(max_steps),
        max_cases=int(max_cases),
        max_hours=float(max_hours),
    )
    print("[v8-research-cloud] launch job:")
    print(json.dumps(asdict(job), indent=2, ensure_ascii=False))
    summary = run_v8_research_job.remote(job.to_json())
    print("[v8-research-cloud] remote summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    local_dir = Path(download_local_dir).expanduser().resolve() if download_local_dir.strip() else REPO_ROOT / "output" / summary["output_subdir"]
    _download_volume_dir(summary["output_subdir"], local_dir)
    print(f"[v8-research-cloud] downloaded artifacts to {local_dir}")


@app.local_entrypoint()
def download_run(remote_output_subdir: str, local_output_dir: str) -> None:
    _download_volume_dir(remote_output_subdir, Path(local_output_dir))
    print(f"[v8-research-cloud] downloaded artifacts to {Path(local_output_dir).resolve()}")
