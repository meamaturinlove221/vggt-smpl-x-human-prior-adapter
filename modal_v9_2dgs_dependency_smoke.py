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
APP_NAME = os.environ.get("VGGT_MODAL_V9_2DGS_SMOKE_APP_NAME", "vggt-v9-2dgs-dependency-smoke")
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_V9_2DGS_SMOKE_TIMEOUT_SEC", str(2 * 60 * 60)))
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


def _run(cmd: list[str], cwd: Path | None, timeout: int = 900) -> dict:
    started = time.time()
    try:
        print(f"[v9-2dgs-smoke] RUN cwd={cwd or os.getcwd()} cmd={' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
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
                print(f"[v9-2dgs-smoke] {line.rstrip()}", flush=True)
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
            "stdout_tail": combined[-10000:],
            "stderr_tail": "",
            "elapsed_sec": float(time.time() - started),
        }
    except Exception as exc:
        return {"command": cmd, "cwd": str(cwd) if cwd else None, "error": repr(exc), "elapsed_sec": float(time.time() - started)}


def _write_result(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "2dgs_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "2dgs_report.md").write_text(
        "\n".join(
            [
                "# V9 2DGS Cloud Dependency Smoke",
                "",
                f"Status: `{payload['status']}`",
                "",
                payload["decision"],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


@app.function(image=IMAGE_2DGS, gpu=os.environ.get("VGGT_MODAL_V9_2DGS_SMOKE_GPU", "A10G"), memory=48 * 1024, timeout=TIMEOUT_SEC, volumes={REMOTE_OUTPUT_DIR.as_posix(): output_volume})
def run_dependency_smoke(remote_output_subdir: str) -> dict:
    out_subdir = _normalize_subpath(remote_output_subdir)
    output_dir = Path(str(REMOTE_OUTPUT_DIR / out_subdir))
    output_dir.mkdir(parents=True, exist_ok=True)
    repo = output_dir / "repo"
    started = time.time()
    common = {
        "task": "v9_2dgs_cloud_dependency_smoke",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": "2dgs",
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
    steps["clone"] = _run(["git", "clone", "--depth", "1", "--recursive", "https://github.com/hbb1/2d-gaussian-splatting.git", str(repo)], None, timeout=900)
    if not repo.is_dir():
        status = "blocked_clone_failed"
    else:
        steps["torch_install"] = _run([sys.executable, "-m", "pip", "install", "torch==2.0.0", "torchvision==0.15.0", "--index-url", "https://download.pytorch.org/whl/cu118"], None, timeout=1200)
        steps["setuptools_pin"] = _run([sys.executable, "-m", "pip", "install", "setuptools<70"], None, timeout=300)
        steps["base_deps"] = _run([sys.executable, "-m", "pip", "install", "plyfile", "tqdm", "trimesh", "opencv-python", "mediapy", "scikit-image", "lpips"], None, timeout=900)
        steps["diff_surfel_install"] = _run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-cache-dir", "submodules/diff-surfel-rasterization"], repo, timeout=1200)
        steps["simple_knn_install"] = _run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-cache-dir", "submodules/simple-knn"], repo, timeout=1200)
        steps["import_probe"] = _run([sys.executable, "-c", "import torch; import gaussian_renderer; import scene; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"], repo, timeout=120)
        status = "dependency_import_ready_needs_colmap_scene" if steps["import_probe"].get("returncode") == 0 else "blocked_dependency_build_or_import_failed"
    decision = (
        "2DGS cloud dependencies imported; next blocker is preparing a COLMAP-format 4K4D scene."
        if status == "dependency_import_ready_needs_colmap_scene"
        else "2DGS cloud dependency smoke failed; see build/import step logs."
    )
    payload = {**common, "status": status, "steps": steps, "elapsed_sec": float(time.time() - started), "decision": decision}
    _write_result(output_dir, payload)
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
    remote_output_subdir: str = "surface_research_cloud_preflight/V9_backend_dependency_smokes/2dgs_fixed",
    download_local_dir: str = "",
) -> None:
    remote_output_subdir = _normalize_subpath(remote_output_subdir)
    summary = run_dependency_smoke.remote(remote_output_subdir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    local_dir = Path(download_local_dir).expanduser().resolve() if download_local_dir.strip() else REPO_ROOT / "output" / remote_output_subdir
    _download_volume_dir(remote_output_subdir, local_dir)
    print(f"[v9-2dgs-smoke] downloaded artifacts to {local_dir}")
