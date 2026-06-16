from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import modal


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
APP_NAME = os.environ.get("VGGT_MODAL_V21100_APP_NAME", "vggt-v21100-patch-local-decoder")
VOLUME_NAME = os.environ.get("VGGT_MODAL_V21100_VOLUME", "vggt-v21100-patch-local-decoder-output")
REMOTE_REPO = PurePosixPath("/workspace/repo")
REMOTE_OUT = PurePosixPath("/v21100_out")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install("torch==2.5.1", "numpy==1.26.4", "Pillow")
    .add_local_dir(str(REPO / "models"), remote_path=str(REMOTE_REPO / "models"))
    .add_local_dir(str(REPO / "tools"), remote_path=str(REMOTE_REPO / "tools"))
    .add_local_dir(str(REPO / "reports"), remote_path=str(REMOTE_REPO / "reports"))
    .add_local_dir(str(REPO / "output" / "V9500000000000000_smpl_feature_bank_v4"), remote_path=str(REMOTE_REPO / "output" / "V9500000000000000_smpl_feature_bank_v4"))
    .add_local_dir(str(REPO / "output" / "V5360000000000000000_geometry_part_binding_repair"), remote_path=str(REMOTE_REPO / "output" / "V5360000000000000000_geometry_part_binding_repair"))
    .add_local_dir(str(REPO / "output" / "V161000000000000_repaired_detail_regions"), remote_path=str(REMOTE_REPO / "output" / "V161000000000000_repaired_detail_regions"))
    .add_local_dir(str(REPO / "output" / "V1400000000000000000_learned_residual_matrix"), remote_path=str(REMOTE_REPO / "output" / "V1400000000000000000_learned_residual_matrix"))
    .add_local_dir(str(REPO / "output" / "V18600000000000000000_part_coverage_canonical_surfel_training"), remote_path=str(REMOTE_REPO / "output" / "V18600000000000000000_part_coverage_canonical_surfel_training"))
    .add_local_dir(str(REPO / "output" / "V18700000000000000000_visible_anchor_canonical_surfel_training"), remote_path=str(REMOTE_REPO / "output" / "V18700000000000000000_visible_anchor_canonical_surfel_training"))
    .add_local_dir(str(REPO / "output" / "V19000000000000000000_pose_frame_occupancy_repair"), remote_path=str(REMOTE_REPO / "output" / "V19000000000000000000_pose_frame_occupancy_repair"))
    .add_local_dir(str(REPO / "output" / "V19200000000000000000_upright_pose_frame_layout"), remote_path=str(REMOTE_REPO / "output" / "V19200000000000000000_upright_pose_frame_layout"))
    .add_local_dir(str(REPO / "output" / "V19400000000000000000_visible_surface_preserving_infill"), remote_path=str(REMOTE_REPO / "output" / "V19400000000000000000_visible_surface_preserving_infill"))
    .add_local_dir(str(REPO / "output" / "V19700000000000000000_moderate_offset_surface_completion"), remote_path=str(REMOTE_REPO / "output" / "V19700000000000000000_moderate_offset_surface_completion"))
    .add_local_dir(str(REPO / "output" / "V20000000000000000000_visible_baseline_locked_completion"), remote_path=str(REMOTE_REPO / "output" / "V20000000000000000000_visible_baseline_locked_completion"))
    .add_local_dir(str(REPO / "output" / "V20200000000000000000_visible_nonregression_connected_infill"), remote_path=str(REMOTE_REPO / "output" / "V20200000000000000000_visible_nonregression_connected_infill"))
    .add_local_dir(str(REPO / "output" / "V20300000000000000000_part_specific_nonregression_infill"), remote_path=str(REMOTE_REPO / "output" / "V20300000000000000000_part_specific_nonregression_infill"))
    .add_local_dir(str(REPO / "output" / "V20420000000000000000_part_local_target_student"), remote_path=str(REMOTE_REPO / "output" / "V20420000000000000000_part_local_target_student"))
    .add_local_dir(str(REPO / "output" / "V20410000000000000000_part_local_targets"), remote_path=str(REMOTE_REPO / "output" / "V20410000000000000000_part_local_targets"))
    .add_local_dir(str(REPO / "output" / "V20500000000000000000_learned_part_proposals"), remote_path=str(REMOTE_REPO / "output" / "V20500000000000000000_learned_part_proposals"))
    .add_local_dir(str(REPO / "output" / "V21000000000000000000_patch_geometry_sources"), remote_path=str(REMOTE_REPO / "output" / "V21000000000000000000_patch_geometry_sources"))
    .add_local_dir(str(REPO / "output" / "V20600000000000000000_learned_part_proposal_student"), remote_path=str(REMOTE_REPO / "output" / "V20600000000000000000_learned_part_proposal_student"))
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@app.function(
    image=image,
    gpu=os.environ.get("VGGT_MODAL_V21100_GPU", "A10G"),
    cpu=4.0,
    memory=24 * 1024,
    timeout=8 * 60 * 60,
    volumes={str(REMOTE_OUT): volume},
)
def run_v21100(steps: int = 220, max_points: int = 8192) -> dict[str, Any]:
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(str(REMOTE_REPO))
    os.chdir(repo)
    env = dict(os.environ)
    env["VGGT_REPO_ROOT"] = str(repo)
    env["V21100_STEPS"] = str(steps)
    env["V21100_MAX_POINTS"] = str(max_points)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "tools/V21100_patch_local_decoder.py"],
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
        "reports/V21100000000000000000_patch_local_decoder_manifest.csv",
        "reports/V21100000000000000000_patch_local_decoder_scores.csv",
        "reports/V21100000000000000000_patch_local_decoder_decision.json",
        "reports/V21100000000000000000_runtime_environment.json",
        "boards/V21100000000000000000_patch_local_decoder_board.png",
        "boards/V21100000000000000000_patch_local_decoder_turntable_cross_section.png",
    ]:
        src = repo / rel
        if src.exists():
            dst = out_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
    matrix_src = repo / "output" / "V21100000000000000000_patch_local_decoder"
    matrix_dst = out_root / "output" / "V21100000000000000000_patch_local_decoder"
    if matrix_src.exists():
        if matrix_dst.exists():
            shutil.rmtree(matrix_dst)
        shutil.copytree(matrix_src, matrix_dst)
        copied.append("output/V21100000000000000000_patch_local_decoder")
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
def main(steps: int = 220, max_points: int = 8192) -> None:
    result = run_v21100.remote(steps=steps, max_points=max_points)
    print(json.dumps(result, ensure_ascii=False, indent=2))
