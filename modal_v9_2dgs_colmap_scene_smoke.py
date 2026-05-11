from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

import modal


REPO_ROOT = Path(__file__).resolve().parent
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
APP_NAME = os.environ.get("VGGT_MODAL_V9_2DGS_SCENE_APP_NAME", "vggt-v9-2dgs-colmap-scene-smoke")
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_V9_2DGS_SCENE_TIMEOUT_SEC", str(3 * 60 * 60)))
FORBIDDEN_OUTPUT_TOKENS = ("strict_pass", "teacher_export", "candidate_export", "predictions", "formal_candidate", "strict_gate_registry")


IMAGE_2DGS = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "git",
        "build-essential",
        "gcc-11",
        "g++-11",
        "cmake",
        "ninja-build",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libgl1",
        "wget",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "FORCE_CUDA": "1",
            "CC": "/usr/bin/gcc-11",
            "CXX": "/usr/bin/g++-11",
            "CUDAHOSTCXX": "/usr/bin/g++-11",
            "TORCH_CUDA_ARCH_LIST": "8.6",
        }
    )
    .run_commands("python -m pip install --upgrade pip wheel 'setuptools<70'")
)

app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


def _normalize_subpath(value: str) -> str:
    cleaned = (value or "").strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("Expected a non-empty remote subpath.")
    if ".." in Path(cleaned).parts:
        raise ValueError(f"Parent traversal is not allowed: {value!r}")
    lower = cleaned.lower()
    if "surface_research_cloud_preflight" not in lower:
        raise ValueError("V9 research output must include surface_research_cloud_preflight.")
    if any(word in lower for word in FORBIDDEN_OUTPUT_TOKENS):
        raise ValueError(f"V9 research output contains forbidden token: {value!r}")
    return cleaned


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 900) -> dict:
    started = time.time()
    print(f"[v9-2dgs-scene] RUN cwd={cwd or os.getcwd()} cmd={' '.join(cmd)}", flush=True)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines: list[str] = []
        deadline = time.time() + timeout
        assert proc.stdout is not None
        while True:
            if time.time() > deadline:
                proc.kill()
                lines.append(f"[timeout after {timeout}s]\n")
                break
            line = proc.stdout.readline()
            if line:
                lines.append(line)
                print(f"[v9-2dgs-scene] {line.rstrip()}", flush=True)
                continue
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        returncode = proc.wait(timeout=10)
        combined = "".join(lines)
        return {
            "command": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": int(returncode),
            "stdout_tail": combined[-12000:],
            "stderr_tail": "",
            "elapsed_sec": float(time.time() - started),
        }
    except Exception as exc:
        return {
            "command": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "error": repr(exc),
            "stdout_tail": "",
            "stderr_tail": "",
            "elapsed_sec": float(time.time() - started),
        }


def _copy_volume_tree(remote_subdir: str, dst: Path) -> dict:
    remote_subdir = _normalize_subpath(remote_subdir)
    dst.mkdir(parents=True, exist_ok=True)
    remote_prefix = Path(remote_subdir)
    copied = 0
    bytes_total = 0
    for entry in output_volume.listdir(remote_subdir, recursive=True):
        rel_path = Path(entry.path)
        try:
            rel_path = rel_path.relative_to(remote_prefix)
        except ValueError:
            pass
        dest_path = dst / rel_path
        if entry.type == modal.volume.FileEntryType.DIRECTORY:
            dest_path.mkdir(parents=True, exist_ok=True)
            continue
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with dest_path.open("wb") as handle:
            count = output_volume.read_file_into_fileobj(entry.path, handle)
        copied += 1
        bytes_total += int(count or 0)
    return {"remote_subdir": remote_subdir, "local_dir": str(dst), "file_count": int(copied), "bytes": int(bytes_total)}


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


def _write_result(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# V9 2DGS COLMAP Scene Smoke",
                "",
                f"Status: `{payload['status']}`",
                "",
                payload["decision"],
                "",
                "Research-only. No teacher, candidate, predictions, registry, or strict pass artifact was written.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


@app.function(image=IMAGE_2DGS, gpu=os.environ.get("VGGT_MODAL_V9_2DGS_SCENE_GPU", "A10G"), memory=48 * 1024, timeout=TIMEOUT_SEC, volumes={REMOTE_OUTPUT_DIR.as_posix(): output_volume})
def run_scene_smoke(remote_scene_subdir: str, remote_output_subdir: str, iterations: int = 30) -> dict:
    scene_subdir = _normalize_subpath(remote_scene_subdir)
    output_subdir = _normalize_subpath(remote_output_subdir)
    output_dir = Path(str(REMOTE_OUTPUT_DIR / output_subdir))
    work_root = Path("/tmp/v9_2dgs_scene_smoke")
    scene_root = work_root / "scene"
    repo = work_root / "repo"
    model_dir = work_root / "model"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    common = {
        "task": "v9_2dgs_colmap_scene_smoke",
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": "2dgs",
        "research_only": True,
        "formal_cloud_unblocked": False,
        "no_export": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_strict_pass_write": True,
        "remote_scene_subdir": scene_subdir,
        "remote_output_subdir": output_subdir,
    }
    steps: dict[str, dict] = {}
    steps["copy_scene_from_volume"] = _copy_volume_tree(scene_subdir, scene_root)
    steps["clone"] = _run(["git", "clone", "--depth", "1", "--recursive", "https://github.com/hbb1/2d-gaussian-splatting.git", str(repo)], None, timeout=900)
    if not repo.is_dir():
        status = "blocked_clone_failed"
    else:
        steps["torch_install"] = _run([sys.executable, "-m", "pip", "install", "torch==2.0.0", "torchvision==0.15.0", "--index-url", "https://download.pytorch.org/whl/cu118"], None, timeout=1200)
        steps["setuptools_pin"] = _run([sys.executable, "-m", "pip", "install", "setuptools<70"], None, timeout=300)
        steps["base_deps"] = _run([sys.executable, "-m", "pip", "install", "plyfile", "tqdm", "trimesh", "opencv-python", "mediapy", "scikit-image", "lpips"], None, timeout=900)
        steps["diff_surfel_install"] = _run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-cache-dir", "submodules/diff-surfel-rasterization"], repo, timeout=1200)
        steps["simple_knn_install"] = _run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-cache-dir", "submodules/simple-knn"], repo, timeout=1200)
        steps["loader_probe"] = _run(
            [
                sys.executable,
                "-c",
                (
                    "from scene.dataset_readers import readColmapSceneInfo; "
                    f"s=readColmapSceneInfo({str(scene_root)!r}, None, False); "
                    "print(len(s.train_cameras), len(s.test_cameras), s.point_cloud.points.shape[0], s.nerf_normalization)"
                ),
            ],
            repo,
            timeout=180,
        )
        if steps["loader_probe"].get("returncode") == 0 and int(iterations) > 0:
            steps["train_smoke"] = _run(
                [
                    sys.executable,
                    "train.py",
                    "-s",
                    str(scene_root),
                    "-m",
                    str(model_dir),
                    "--iterations",
                    str(int(iterations)),
                    "--test_iterations",
                    str(int(iterations)),
                    "--save_iterations",
                    str(int(iterations)),
                    "--quiet",
                    "--port",
                    "6019",
                ],
                repo,
                timeout=1800,
            )
        if steps.get("train_smoke", {}).get("returncode") == 0:
            status = "train_smoke_completed_weak_pool_only"
        elif steps["loader_probe"].get("returncode") == 0:
            status = "loader_ready_train_smoke_blocked_or_skipped"
        else:
            status = "blocked_loader_probe_failed"
    if model_dir.is_dir():
        dst_model = output_dir / "model_smoke"
        if dst_model.exists():
            shutil.rmtree(dst_model)
        shutil.copytree(model_dir, dst_model)
    decision = (
        "2DGS read the staged known-camera COLMAP scene and completed a short research-only training smoke; initial points are MUSt3R weak-pool only."
        if status == "train_smoke_completed_weak_pool_only"
        else "2DGS scene smoke did not complete training; see step logs."
    )
    payload = {**common, "status": status, "decision": decision, "steps": steps, "elapsed_sec": float(time.time() - started)}
    _write_result(output_dir, payload)
    output_volume.commit()
    return payload


@app.local_entrypoint()
def run_smoke(
    remote_scene_subdir: str = "surface_research_cloud_preflight/Cloud_B_V9/a5x2_2dgs_colmap_scene/2dgs_colmap_scene",
    remote_output_subdir: str = "surface_research_cloud_preflight/Cloud_B_V9/a5x2_2dgs_colmap_scene_smoke",
    download_local_dir: str = "",
    iterations: int = 30,
) -> None:
    scene_subdir = _normalize_subpath(remote_scene_subdir)
    output_subdir = _normalize_subpath(remote_output_subdir)
    payload = run_scene_smoke.remote(scene_subdir, output_subdir, int(iterations))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    local_dir = Path(download_local_dir).expanduser().resolve() if download_local_dir.strip() else REPO_ROOT / "output" / output_subdir
    _download_volume_dir(output_subdir, local_dir)
    print(f"[v9-2dgs-scene] downloaded artifacts to {local_dir}")
