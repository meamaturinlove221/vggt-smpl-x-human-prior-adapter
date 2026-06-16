from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path, PurePosixPath

import modal


REPO_ROOT = Path(__file__).resolve().parent
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
APP_NAME = os.environ.get("VGGT_MODAL_V9_A5X2_MUST3R_APP_NAME", "vggt-v9-a5x2-must3r-true-backend")
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_V9_A5X2_MUST3R_TIMEOUT_SEC", str(3 * 60 * 60)))

CHECKPOINT_URLS = {
    "MUSt3R_224_cvpr.pth": "https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_224_cvpr.pth",
    "MUSt3R_512_cvpr.pth": "https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_512_cvpr.pth",
}
FORBIDDEN_OUTPUT_TOKENS = ("strict_pass", "teacher_export", "candidate_export", "predictions", "formal_candidate", "strict_gate_registry")


IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake", "libglib2.0-0", "libsm6", "libxext6", "libxrender1", "libgl1", "wget")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126",
        "python -m pip install must3r@git+https://github.com/naver/must3r.git",
    )
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _download_file(url: str, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 1024 * 1024:
        return {"url": url, "path": str(path), "downloaded": False, "bytes": int(path.stat().st_size)}
    tmp = path.with_suffix(path.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return {"url": url, "path": str(path), "downloaded": True, "bytes": int(path.stat().st_size)}


@app.function(
    image=IMAGE,
    gpu=os.environ.get("VGGT_MODAL_V9_A5X2_MUST3R_GPU", "A10G"),
    memory=48 * 1024,
    timeout=TIMEOUT_SEC,
    volumes={REMOTE_OUTPUT_DIR.as_posix(): output_volume},
)
def run_must3r_true_backend(
    remote_asset_subdir: str,
    remote_output_subdir: str,
    image_count: int,
    image_size: int,
    checkpoint_name: str,
) -> dict:
    asset_subdir = _normalize_subpath(remote_asset_subdir)
    output_subdir = _normalize_subpath(remote_output_subdir)
    asset_dir = Path(str(REMOTE_OUTPUT_DIR / asset_subdir))
    output_dir = Path(str(REMOTE_OUTPUT_DIR / output_subdir))
    output_dir.mkdir(parents=True, exist_ok=True)
    image_src_dir = asset_dir / "scene" / "images"
    run_image_dir = output_dir / "input_images"
    run_image_dir.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in image_src_dir.glob("*") if p.is_file()])[: int(image_count)]
    for idx, src in enumerate(images):
        shutil.copy2(src, run_image_dir / f"{idx:03d}_{src.name}")

    ckpt_name = checkpoint_name if checkpoint_name in CHECKPOINT_URLS else "MUSt3R_224_cvpr.pth"
    checkpoint_path = Path(str(REMOTE_OUTPUT_DIR / "surface_research_cloud_preflight/V9_checkpoints/must3r" / ckpt_name))
    started = time.time()
    summary = {
        "task": "v9_a5x2_must3r_true_backend",
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "research_only": True,
        "no_export": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_strict_pass_write": True,
        "formal_cloud_unblocked": False,
        "backend": "MUSt3R",
        "official_repo": "https://github.com/naver/must3r",
        "remote_asset_subdir": asset_subdir,
        "remote_output_subdir": output_subdir,
        "input_image_count": len(images),
        "image_size": int(image_size),
        "checkpoint_name": ckpt_name,
        "checkpoint_path": str(checkpoint_path),
        "status": "not_started",
    }
    try:
        summary["checkpoint_download"] = _download_file(CHECKPOINT_URLS[ckpt_name], checkpoint_path)
        import_probe = subprocess.run(
            [sys.executable, "-c", "import torch, must3r; print(torch.__version__, torch.cuda.is_available(), must3r.__file__)"],
            text=True,
            capture_output=True,
            check=False,
        )
        summary["import_probe"] = {
            "returncode": int(import_probe.returncode),
            "stdout_tail": import_probe.stdout[-4000:],
            "stderr_tail": import_probe.stderr[-4000:],
        }
        cmd = [
            sys.executable,
            "-m",
            "must3r.get_reconstruction",
            "--image_dir",
            str(run_image_dir),
            "--output",
            str(output_dir / "must3r_run"),
            "--weights",
            str(checkpoint_path),
            "--image_size",
            str(int(image_size)),
            "--execution_mode",
            "linseq",
            "--max_bs",
            "1",
            "--num_mem_imgs",
            str(max(1, len(images))),
            "--file_type",
            "ply",
            "--device",
            "cuda",
            "--render_once",
        ]
        run = subprocess.run(cmd, text=True, capture_output=True, check=False)
        ply_files = sorted((output_dir / "must3r_run").glob("*.ply")) if (output_dir / "must3r_run").is_dir() else []
        summary.update(
            {
                "command": cmd,
                "returncode": int(run.returncode),
                "stdout_tail": run.stdout[-12000:],
                "stderr_tail": run.stderr[-12000:],
                "ply_files": [str(p) for p in ply_files],
                "ply_count": len(ply_files),
                "elapsed_sec": float(time.time() - started),
            }
        )
        if run.returncode == 0 and ply_files:
            summary["status"] = "true_backend_completed"
            summary["decision"] = "A5X2_MUST3R_TRUE_BACKEND_COMPLETED: official MUSt3R ran on staged 4K4D images and produced PLY artifacts. Research-only; not teacher/candidate."
        else:
            summary["status"] = "true_backend_runtime_failed"
            summary["decision"] = "A5X2_MUST3R_TRUE_BACKEND_FAILED: installation/import/checkpoint may be present, but reconstruction did not produce a PLY artifact."
    except Exception as exc:
        summary.update(
            {
                "status": "true_backend_install_or_runtime_exception",
                "exception": repr(exc),
                "elapsed_sec": float(time.time() - started),
                "decision": "A5X2_MUST3R_TRUE_BACKEND_FAILED: hard install/runtime exception; no synthetic substitute was generated.",
            }
        )
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# V9 A5-X2 MUSt3R True Backend",
                "",
                f"Status: `{summary['status']}`",
                "",
                f"- backend: `{summary['backend']}`",
                f"- input_image_count: `{summary['input_image_count']}`",
                f"- checkpoint_name: `{summary['checkpoint_name']}`",
                f"- returncode: `{summary.get('returncode')}`",
                f"- ply_count: `{summary.get('ply_count', 0)}`",
                "",
                "## Decision",
                "",
                summary["decision"],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_volume.commit()
    if summary["status"] != "true_backend_completed":
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
def run_backend(
    remote_asset_subdir: str = "surface_research_cloud_preflight/V9_cloud_asset_staging/assets",
    remote_output_subdir: str = "surface_research_cloud_preflight/Cloud_B_V9/a5x2_must3r_true_backend",
    image_count: int = 6,
    image_size: int = 224,
    checkpoint_name: str = "MUSt3R_224_cvpr.pth",
    download_local_dir: str = "",
) -> None:
    remote_asset_subdir = _normalize_subpath(remote_asset_subdir)
    remote_output_subdir = _normalize_subpath(remote_output_subdir)
    try:
        summary = run_must3r_true_backend.remote(remote_asset_subdir, remote_output_subdir, int(image_count), int(image_size), checkpoint_name)
        print("[v9-a5x2-must3r] remote summary:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        local_dir = Path(download_local_dir).expanduser().resolve() if download_local_dir.strip() else REPO_ROOT / "output" / remote_output_subdir
        _download_volume_dir(remote_output_subdir, local_dir)
        print(f"[v9-a5x2-must3r] downloaded remote artifacts to {local_dir}")
