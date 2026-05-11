from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

import modal


REPO_ROOT = Path(__file__).resolve().parent
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
APP_NAME = os.environ.get("VGGT_MODAL_V9_BACKEND_SMOKES_APP_NAME", "vggt-v9-backend-cloud-dependency-smokes")
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_V9_BACKEND_SMOKES_TIMEOUT_SEC", str(2 * 60 * 60)))

FORBIDDEN_OUTPUT_TOKENS = ("strict_pass", "teacher_export", "candidate_export", "predictions", "formal_candidate", "strict_gate_registry")


BASE_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake", "ninja-build", "libglib2.0-0", "libsm6", "libxext6", "libxrender1", "libgl1", "wget")
    .run_commands("python -m pip install --upgrade pip setuptools wheel")
)

app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


def _normalize_subpath(value: str) -> str:
    cleaned = (value or "").strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("Expected a non-empty remote subpath.")
    parts = Path(cleaned).parts
    if ".." in parts:
        raise ValueError(f"Parent traversal is not allowed: {value!r}")
    lower = cleaned.lower()
    if "surface_research_cloud_preflight" not in lower:
        raise ValueError("V9 research output must include surface_research_cloud_preflight.")
    if any(word in lower for word in FORBIDDEN_OUTPUT_TOKENS):
        raise ValueError(f"V9 research output contains forbidden token: {value!r}")
    return cleaned


def _run(cmd: list[str], cwd: Path | None, timeout: int = 900) -> dict:
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "command": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": int(proc.returncode),
            "stdout_tail": proc.stdout[-10000:],
            "stderr_tail": proc.stderr[-10000:],
            "elapsed_sec": float(time.time() - started),
        }
    except Exception as exc:
        return {"command": cmd, "cwd": str(cwd) if cwd else None, "error": repr(exc), "elapsed_sec": float(time.time() - started)}


def _write_result(output_dir: Path, name: str, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / f"{name}_report.md").write_text(
        "\n".join(
            [
                f"# V9 {name} Cloud Dependency Smoke",
                "",
                f"Status: `{payload['status']}`",
                "",
                payload["decision"],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


@app.function(image=BASE_IMAGE, gpu=os.environ.get("VGGT_MODAL_V9_BACKEND_SMOKE_GPU", "A10G"), memory=48 * 1024, timeout=TIMEOUT_SEC, volumes={REMOTE_OUTPUT_DIR.as_posix(): output_volume})
def run_dependency_smoke(backend: str, remote_output_subdir: str) -> dict:
    backend_key = backend.strip().lower()
    out_subdir = _normalize_subpath(remote_output_subdir)
    output_dir = Path(str(REMOTE_OUTPUT_DIR / out_subdir))
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    common = {
        "task": f"v9_{backend_key}_cloud_dependency_smoke",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": backend,
        "research_only": True,
        "no_export": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_strict_pass_write": True,
        "formal_cloud_unblocked": False,
    }
    steps: dict[str, dict] = {}
    status = "blocked_unknown_backend"
    decision = "Unsupported backend smoke."
    repo = output_dir / "repo"

    if backend_key == "2dgs":
        steps["clone"] = _run(["git", "clone", "--depth", "1", "--recursive", "https://github.com/hbb1/2d-gaussian-splatting.git", str(repo)], None, timeout=900)
        if not repo.is_dir():
            status = "blocked_clone_failed"
        else:
            steps["torch_install"] = _run([sys.executable, "-m", "pip", "install", "torch==2.0.0", "torchvision==0.15.0", "--index-url", "https://download.pytorch.org/whl/cu118"], None, timeout=1200)
            steps["base_deps"] = _run([sys.executable, "-m", "pip", "install", "plyfile", "tqdm", "trimesh", "opencv-python", "mediapy", "scikit-image", "lpips"], None, timeout=900)
            steps["diff_surfel_install"] = _run([sys.executable, "-m", "pip", "install", "submodules/diff-surfel-rasterization"], repo, timeout=1200)
            steps["simple_knn_install"] = _run([sys.executable, "-m", "pip", "install", "submodules/simple-knn"], repo, timeout=1200)
            steps["import_probe"] = _run([sys.executable, "-c", "import torch; import gaussian_renderer; import scene; print(torch.__version__, torch.cuda.is_available())"], repo, timeout=120)
            ok = steps["import_probe"].get("returncode") == 0
            status = "dependency_import_ready_needs_colmap_scene" if ok else "blocked_dependency_build_or_import_failed"
        decision = (
            "2DGS cloud dependencies imported; next blocker is preparing a COLMAP-format 4K4D scene."
            if status == "dependency_import_ready_needs_colmap_scene"
            else "2DGS cloud dependency smoke failed; see build/import step logs."
        )

    elif backend_key == "mast3r-slam":
        steps["clone"] = _run(["git", "clone", "--depth", "1", "--recursive", "https://github.com/rmurai0610/MASt3R-SLAM.git", str(repo)], None, timeout=1200)
        if not repo.is_dir():
            status = "blocked_clone_failed"
        else:
            steps["torch_install"] = _run([sys.executable, "-m", "pip", "install", "torch==2.5.1", "torchvision==0.20.1", "--index-url", "https://download.pytorch.org/whl/cu124"], None, timeout=1200)
            steps["mast3r_install"] = _run([sys.executable, "-m", "pip", "install", "-e", "thirdparty/mast3r"], repo, timeout=1200)
            steps["in3d_install"] = _run([sys.executable, "-m", "pip", "install", "-e", "thirdparty/in3d"], repo, timeout=900)
            steps["self_install"] = _run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "-e", "."], repo, timeout=1200)
            steps["import_probe"] = _run([sys.executable, "-c", "import torch; import mast3r; import mast3r_slam; print(torch.__version__, torch.cuda.is_available())"], repo, timeout=120)
            ok = steps["import_probe"].get("returncode") == 0
            checkpoints = [repo / "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"]
            status = "dependency_import_ready_missing_checkpoints" if ok and not all(p.is_file() for p in checkpoints) else "dependency_import_ready" if ok else "blocked_dependency_build_or_import_failed"
        decision = (
            "MASt3R-SLAM imports on cloud; next blocker is checkpoint download plus 4K4D folder/calibration run."
            if status.startswith("dependency_import_ready")
            else "MASt3R-SLAM cloud dependency smoke failed; see build/import step logs."
        )

    elif backend_key == "hair-gs":
        steps["clone"] = _run(["git", "clone", "--depth", "1", "--recursive", "https://github.com/yimin-pan/hair-gs.git", str(repo)], None, timeout=1200)
        if not repo.is_dir():
            status = "blocked_clone_failed"
        else:
            steps["torch_install"] = _run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124"], None, timeout=1200)
            steps["base_deps"] = _run([sys.executable, "-m", "pip", "install", "plyfile", "pyrr", "scipy", "opencv-python", "pyvista", "dreifus", "glfw", "smplx", "chumpy-fix", "tqdm", "tensorboard"], None, timeout=900)
            steps["c_utils_install"] = _run([sys.executable, "-m", "pip", "install", "./c_utils"], repo, timeout=900)
            steps["diff_gaussian_install"] = _run([sys.executable, "-m", "pip", "install", "submodules/diff-gaussian-rasterization"], repo, timeout=1200)
            steps["simple_knn_install"] = _run([sys.executable, "-m", "pip", "install", "submodules/simple-knn"], repo, timeout=1200)
            steps["self_install"] = _run([sys.executable, "-m", "pip", "install", "-e", "."], repo, timeout=900)
            steps["import_probe"] = _run([sys.executable, "-c", "import torch; from scene import HairGaussianModel; print(torch.__version__, torch.cuda.is_available(), HairGaussianModel)"], repo, timeout=120)
            ok = steps["import_probe"].get("returncode") == 0
            status = "dependency_import_ready_missing_flame_and_hair_dataset" if ok else "blocked_dependency_build_or_import_failed"
        decision = (
            "Hair-GS imports on cloud; next blocker is FLAME files and parsed hair dataset/4K4D hair conversion."
            if status == "dependency_import_ready_missing_flame_and_hair_dataset"
            else "Hair-GS cloud dependency smoke failed; see build/import step logs."
        )

    payload = {**common, "status": status, "steps": steps, "elapsed_sec": float(time.time() - started), "decision": decision}
    _write_result(output_dir, backend_key.replace("-", "_"), payload)
    output_volume.commit()
    return payload


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
def run_smoke(
    backend: str,
    remote_output_subdir: str = "surface_research_cloud_preflight/V9_backend_dependency_smokes",
    download_local_dir: str = "",
) -> None:
    remote_output_subdir = _normalize_subpath(remote_output_subdir)
    summary = run_dependency_smoke.remote(backend, remote_output_subdir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    local_dir = Path(download_local_dir).expanduser().resolve() if download_local_dir.strip() else REPO_ROOT / "output" / remote_output_subdir
    _download_volume_dir(remote_output_subdir, local_dir)
    print(f"[v9-backend-smoke] downloaded artifacts to {local_dir}")
