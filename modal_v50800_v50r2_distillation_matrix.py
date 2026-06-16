from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import modal


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
APP_NAME = os.environ.get("VGGT_MODAL_V508_APP_NAME", "vggt-v508-v50r2-distillation-matrix")
VOLUME_NAME = os.environ.get("VGGT_MODAL_V508_VOLUME", "vggt-v508-v50r2-distillation-output")
GPU_NAME = os.environ.get("VGGT_MODAL_V508_GPU", "A10G")
REMOTE_REPO = PurePosixPath("/workspace/repo")
REMOTE_OUT = PurePosixPath("/v508_out")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install("torch==2.5.1", "numpy==1.26.4", "Pillow==10.4.0")
    .add_local_dir(str(REPO / "models"), remote_path=str(REMOTE_REPO / "models"))
    .add_local_dir(str(REPO / "training"), remote_path=str(REMOTE_REPO / "training"))
    .add_local_dir(str(REPO / "tools"), remote_path=str(REMOTE_REPO / "tools"))
    .add_local_dir(str(REPO / "reports"), remote_path=str(REMOTE_REPO / "reports"))
    .add_local_dir(str(REPO / "boards"), remote_path=str(REMOTE_REPO / "boards"))
    .add_local_dir(
        str(REPO / "output" / "V5040000000000000000000_v50r2_teacher_bank"),
        remote_path=str(REMOTE_REPO / "output" / "V5040000000000000000000_v50r2_teacher_bank"),
    )
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@app.function(
    image=image,
    gpu=GPU_NAME,
    cpu=4.0,
    memory=24 * 1024,
    timeout=8 * 60 * 60,
    volumes={str(REMOTE_OUT): volume},
)
def run_v508(steps: int = 3, max_samples: int = 384, seed: int = 508) -> dict[str, Any]:
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(str(REMOTE_REPO))
    os.chdir(repo)
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["VGGT_MODAL_V508_GPU"] = GPU_NAME
    proc = subprocess.run(
        [
            sys.executable,
            "tools/V508_v50r2_distillation_matrix.py",
            "--steps",
            str(int(steps)),
            "--max-samples",
            str(int(max_samples)),
            "--seed",
            str(int(seed)),
            "--mode",
            "modal_smoke",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        timeout=7 * 60 * 60,
    )
    out_root = Path(str(REMOTE_OUT))
    out_root.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for rel in [
        "reports/V5080000000000000000000_training_manifest.csv",
        "reports/V5080000000000000000000_seed_metrics.csv",
        "reports/V5080000000000000000000_hash_reconciliation.json",
        "reports/V5080000000000000000000_failed_jobs.json",
    ]:
        src = repo / rel
        if src.exists():
            dst = out_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
    matrix_src = repo / "output" / "V5080000000000000000000_v50r2_distillation_matrix"
    matrix_dst = out_root / "output" / "V5080000000000000000000_v50r2_distillation_matrix"
    if matrix_src.exists():
        if matrix_dst.exists():
            shutil.rmtree(matrix_dst)
        shutil.copytree(matrix_src, matrix_dst)
        copied.append("output/V5080000000000000000000_v50r2_distillation_matrix")
    volume.commit()
    return {
        "created_at": now(),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-5000:],
        "stderr_tail": proc.stderr[-5000:],
        "copied": copied,
        "remote_output": str(REMOTE_OUT),
        "gpu": GPU_NAME,
        "remote_env_gpu_label": os.environ.get("VGGT_MODAL_V508_GPU"),
        "steps": int(steps),
        "max_samples": int(max_samples),
        "seed": int(seed),
        "interpretation": "V508 Modal smoke is not final training completion; target checkpoints 300/600/1000/2000/4000 remain the full matrix.",
    }


@app.local_entrypoint()
def main(steps: int = 3, max_samples: int = 384, seed: int = 508) -> None:
    result = run_v508.remote(steps=steps, max_samples=max_samples, seed=seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
