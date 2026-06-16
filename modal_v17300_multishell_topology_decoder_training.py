from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import modal


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
APP_NAME = os.environ.get("VGGT_MODAL_V17300_APP_NAME", "vggt-v17300-multishell-topology-decoder")
VOLUME_NAME = os.environ.get("VGGT_MODAL_V17300_VOLUME", "vggt-v17300-multishell-output")
REMOTE_REPO = PurePosixPath("/workspace/repo")
REMOTE_OUT = PurePosixPath("/v17300_out")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install("torch==2.5.1", "numpy==1.26.4", "Pillow")
    .add_local_dir(str(REPO / "models"), remote_path=str(REMOTE_REPO / "models"))
    .add_local_dir(str(REPO / "tools"), remote_path=str(REMOTE_REPO / "tools"))
    .add_local_dir(str(REPO / "reports"), remote_path=str(REMOTE_REPO / "reports"))
    .add_local_dir(str(REPO / "output" / "V17100000000000000000_part_graph_cross_section_targets"), remote_path=str(REMOTE_REPO / "output" / "V17100000000000000000_part_graph_cross_section_targets"))
    .add_local_dir(str(REPO / "output" / "V10700000000000000000_volume_aware_training_matrix"), remote_path=str(REMOTE_REPO / "output" / "V10700000000000000000_volume_aware_training_matrix"))
    .add_local_dir(str(REPO / "output" / "V1400000000000000000_learned_residual_matrix"), remote_path=str(REMOTE_REPO / "output" / "V1400000000000000000_learned_residual_matrix"))
    .add_local_dir(str(REPO / "output" / "V5360000000000000000_geometry_part_binding_repair"), remote_path=str(REMOTE_REPO / "output" / "V5360000000000000000_geometry_part_binding_repair"))
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@app.function(
    image=image,
    gpu=os.environ.get("VGGT_MODAL_V17300_GPU", "A10G"),
    cpu=4.0,
    memory=24 * 1024,
    timeout=8 * 60 * 60,
    volumes={str(REMOTE_OUT): volume},
)
def run_v17300(steps: int = 300, max_points: int = 8192) -> dict[str, Any]:
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(str(REMOTE_REPO))
    os.chdir(repo)
    env = dict(os.environ)
    env["VGGT_REPO_ROOT"] = str(repo)
    env["V17300_STEPS"] = str(steps)
    env["V17300_MAX_POINTS"] = str(max_points)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "tools/V17300_multishell_topology_decoder_training.py"],
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
        "reports/V17300000000000000000_training_manifest.csv",
        "reports/V17300000000000000000_seed_metrics.csv",
        "reports/V17300000000000000000_training_decision.json",
        "reports/V17300000000000000000_runtime_environment.json",
        "boards/V17300000000000000000_multishell_training_board.png",
        "boards/V17300000000000000000_turntable_cross_section.png",
    ]:
        src = repo / rel
        if src.exists():
            dst = out_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
    matrix_src = repo / "output" / "V17300000000000000000_multishell_topology_decoder_training"
    matrix_dst = out_root / "output" / "V17300000000000000000_multishell_topology_decoder_training"
    if matrix_src.exists():
        if matrix_dst.exists():
            shutil.rmtree(matrix_dst)
        shutil.copytree(matrix_src, matrix_dst)
        copied.append("output/V17300000000000000000_multishell_topology_decoder_training")
    volume.commit()
    return {
        "created_at": now(),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "copied": copied,
        "remote_output": str(REMOTE_OUT),
        "steps": steps,
        "max_points": max_points,
    }


@app.local_entrypoint()
def main(steps: int = 300, max_points: int = 8192) -> None:
    result = run_v17300.remote(steps=steps, max_points=max_points)
    print(json.dumps(result, ensure_ascii=False, indent=2))
