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
APP_NAME = os.environ.get("VGGT_MODAL_V9_MAST3R_SLAM_APP_NAME", "vggt-v9-mast3r-slam-true-backend-smoke")
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_V9_MAST3R_SLAM_TIMEOUT_SEC", str(3 * 60 * 60)))

FORBIDDEN_OUTPUT_TOKENS = (
    "strict_pass",
    "teacher_export",
    "candidate_export",
    "predictions",
    "formal_candidate",
    "strict_gate_registry",
)

CHECKPOINT_URLS = {
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth": "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth": "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl": "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl",
}

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "git",
        "build-essential",
        "gcc-11",
        "g++-11",
        "clang",
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
    .run_commands(
        "python -m pip install --upgrade pip wheel 'setuptools<70'",
        "python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124",
    )
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
    print(f"[v9-mast3r-slam] RUN cwd={cwd or os.getcwd()} cmd={' '.join(cmd)}", flush=True)
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
                print(f"[v9-mast3r-slam] {line.rstrip()}", flush=True)
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
            "stdout_tail": combined[-14000:],
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _file_size(path: Path) -> int:
    return int(path.stat().st_size) if path.is_file() else 0


def _download_file(url: str, path: Path, min_bytes: int = 1024 * 1024) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size >= min_bytes:
        return {
            "url": url,
            "path": str(path),
            "status": "already_present",
            "bytes": int(path.stat().st_size),
        }
    tmp = path.with_suffix(path.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    started = time.time()
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
        return {
            "url": url,
            "path": str(path),
            "status": "downloaded",
            "bytes": int(path.stat().st_size),
            "elapsed_sec": float(time.time() - started),
        }
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return {
            "url": url,
            "path": str(path),
            "status": "blocked_download_failed",
            "bytes": _file_size(path),
            "error": repr(exc),
            "elapsed_sec": float(time.time() - started),
        }


def _copy_mounted_volume_tree(remote_subdir: str, dst: Path) -> dict:
    remote_subdir = _normalize_subpath(remote_subdir)
    src = Path(str(REMOTE_OUTPUT_DIR / remote_subdir))
    if not src.is_dir():
        return {
            "remote_subdir": remote_subdir,
            "source": str(src),
            "local_dir": str(dst),
            "status": "missing_source_dir",
            "file_count": 0,
            "bytes": 0,
        }
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    files = [item for item in dst.rglob("*") if item.is_file()]
    return {
        "remote_subdir": remote_subdir,
        "source": str(src),
        "local_dir": str(dst),
        "status": "copied",
        "file_count": int(len(files)),
        "bytes": int(sum(item.stat().st_size for item in files)),
    }


def _stage_rgb_sequence(asset_dir: Path, run_image_dir: Path, image_count: int) -> dict:
    src_dir = asset_dir / "4k4d_scene" / "images"
    if not src_dir.is_dir():
        return {
            "status": "missing_4k4d_images_dir",
            "source_dir": str(src_dir),
            "run_image_dir": str(run_image_dir),
            "image_count": 0,
            "files": [],
        }
    images = sorted([p for p in src_dir.glob("*.png") if p.is_file()])[: int(image_count)]
    if run_image_dir.exists():
        shutil.rmtree(run_image_dir)
    run_image_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for idx, src in enumerate(images):
        dst = run_image_dir / f"{idx:04d}.png"
        shutil.copy2(src, dst)
        staged.append({"source": str(src), "path": str(dst), "bytes": int(dst.stat().st_size)})
    return {
        "status": "staged" if staged else "blocked_no_png_images",
        "source_dir": str(src_dir),
        "run_image_dir": str(run_image_dir),
        "image_count": int(len(staged)),
        "files": staged[:8],
    }


def _write_calib_from_sidecar(asset_dir: Path, calib_path: Path) -> dict:
    sidecar = asset_dir / "4k4d_scene" / "camera_params_sidecar.npz"
    if not sidecar.is_file():
        return {"status": "missing_camera_sidecar", "sidecar": str(sidecar), "calib_path": str(calib_path)}
    try:
        import numpy as np
        from PIL import Image
        import yaml

        with np.load(sidecar, allow_pickle=False) as payload:
            intrinsics = payload["intrinsics"][0].astype(float)
        image_files = sorted((asset_dir / "4k4d_scene" / "images").glob("*.png"))
        if image_files:
            with Image.open(image_files[0]) as img:
                width, height = img.size
        else:
            width = int(round(intrinsics[0, 2] * 2))
            height = int(round(intrinsics[1, 2] * 2))
        calib = [
            float(intrinsics[0, 0]),
            float(intrinsics[1, 1]),
            float(intrinsics[0, 2]),
            float(intrinsics[1, 2]),
        ]
        calib_path.parent.mkdir(parents=True, exist_ok=True)
        calib_path.write_text(
            yaml.safe_dump({"width": width, "height": height, "calibration": calib}, sort_keys=False),
            encoding="utf-8",
        )
        return {
            "status": "written",
            "sidecar": str(sidecar),
            "calib_path": str(calib_path),
            "width": int(width),
            "height": int(height),
            "calibration": calib,
        }
    except Exception as exc:
        return {"status": "blocked_calib_write_failed", "sidecar": str(sidecar), "calib_path": str(calib_path), "error": repr(exc)}


def _collect_logs(repo: Path, save_as: str) -> dict:
    log_dir = repo / "logs" / save_as
    files = sorted([p for p in log_dir.rglob("*") if p.is_file()]) if log_dir.is_dir() else []
    copied = []
    return {
        "log_dir": str(log_dir),
        "exists": log_dir.is_dir(),
        "file_count": int(len(files)),
        "files": [{"path": str(path), "bytes": int(path.stat().st_size)} for path in files],
        "copied_files": copied,
    }


def _write_result(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", payload)
    report_lines = [
        "# V9 MASt3R-SLAM True Backend Smoke",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"- backend: `{payload['backend']}`",
        f"- input_image_count: `{payload.get('input_image_count', 0)}`",
        f"- checkpoint_status: `{payload.get('checkpoint_status')}`",
        f"- main_returncode: `{payload.get('main_returncode')}`",
        f"- log_file_count: `{payload.get('logs', {}).get('file_count', 0)}`",
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
        "Research-only. No predictions, teacher, candidate, registry, or strict-pass artifact was written.",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


@app.function(
    image=IMAGE,
    gpu=os.environ.get("VGGT_MODAL_V9_MAST3R_SLAM_GPU", "A10G"),
    memory=48 * 1024,
    timeout=TIMEOUT_SEC,
    volumes={REMOTE_OUTPUT_DIR.as_posix(): output_volume},
)
def run_true_backend_smoke(
    remote_asset_subdir: str,
    remote_output_subdir: str,
    image_count: int = 8,
    use_calib: bool = False,
    main_timeout_sec: int = 1800,
) -> dict:
    asset_subdir = _normalize_subpath(remote_asset_subdir)
    output_subdir = _normalize_subpath(remote_output_subdir)
    output_dir = Path(str(REMOTE_OUTPUT_DIR / output_subdir))
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    work_root = Path("/tmp/v9_mast3r_slam_true_backend_smoke")
    repo = work_root / "repo"
    asset_dir = work_root / "assets"
    run_image_dir = work_root / "input_png_sequence"
    save_as = "v9_mast3r_slam_true_backend_smoke"

    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "task": "v9_mast3r_slam_true_backend_smoke",
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": "mast3r-slam",
        "official_repo": "https://github.com/rmurai0610/MASt3R-SLAM",
        "research_only": True,
        "formal_cloud_unblocked": False,
        "no_export": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_strict_pass_write": True,
        "remote_asset_subdir": asset_subdir,
        "remote_output_subdir": output_subdir,
        "checkpoint_urls": CHECKPOINT_URLS,
        "repo_readme_confirmed": {
            "checkpoint_host": "download.europe.naverlabs.com",
            "run_entrypoint": "python main.py --dataset <path/to/folder-or-video> --config config/base.yaml --calib optional --no-viz",
            "folder_input_format": "top-level PNG files are loaded by mast3r_slam.dataloader.RGBFiles",
            "4k4d_source_format": "staged V9 assets use assets/4k4d_scene/images/*.png plus camera_params_sidecar.npz",
        },
        "steps": {},
        "status": "not_started",
    }

    try:
        summary["steps"]["copy_assets_from_volume"] = _copy_mounted_volume_tree(asset_subdir, asset_dir)
        summary["steps"]["stage_rgb_sequence"] = _stage_rgb_sequence(asset_dir, run_image_dir, int(image_count))
        summary["input_image_count"] = summary["steps"]["stage_rgb_sequence"]["image_count"]
        if summary["input_image_count"] < 2:
            summary.update(
                {
                    "status": "blocked_missing_4k4d_png_sequence",
                    "decision": "MASt3R-SLAM true backend smoke blocked before runtime: staged 4K4D assets did not contain enough top-level PNG input frames after conversion.",
                    "elapsed_sec": float(time.time() - started),
                }
            )
            _write_result(output_dir, summary)
            output_volume.commit()
            return summary

        summary["steps"]["clone"] = _run(
            ["git", "clone", "--depth", "1", "--recursive", "https://github.com/rmurai0610/MASt3R-SLAM.git", str(repo)],
            None,
            timeout=1200,
        )
        if not repo.is_dir():
            summary.update(
                {
                    "status": "blocked_clone_failed",
                    "decision": "MASt3R-SLAM true backend smoke blocked: official repo clone failed on Modal.",
                    "elapsed_sec": float(time.time() - started),
                }
            )
            _write_result(output_dir, summary)
            output_volume.commit()
            return summary

        summary["steps"]["setuptools_pin"] = _run([sys.executable, "-m", "pip", "install", "setuptools<70"], None, timeout=300)
        summary["steps"]["mast3r_install"] = _run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "-e", "thirdparty/mast3r"], repo, timeout=1200)
        summary["steps"]["in3d_install"] = _run([sys.executable, "-m", "pip", "install", "-e", "thirdparty/in3d"], repo, timeout=1200)
        summary["steps"]["self_install"] = _run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "-e", "."], repo, timeout=1200)
        summary["steps"]["import_probe"] = _run(
            [
                sys.executable,
                "-c",
                "import torch; import mast3r; import mast3r_slam; import mast3r_slam_backends; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())",
            ],
            repo,
            timeout=180,
        )
        if summary["steps"]["import_probe"].get("returncode") != 0:
            summary.update(
                {
                    "status": "blocked_dependency_build_or_import_failed",
                    "decision": "MASt3R-SLAM true backend smoke blocked: dependency/import probe failed before checkpoint or backend run.",
                    "elapsed_sec": float(time.time() - started),
                }
            )
            _write_result(output_dir, summary)
            output_volume.commit()
            return summary

        ckpt_dir = repo / "checkpoints"
        checkpoint_results = {}
        for name, url in CHECKPOINT_URLS.items():
            checkpoint_results[name] = _download_file(url, ckpt_dir / name)
        summary["checkpoint_downloads"] = checkpoint_results
        failed_checkpoints = {name: item for name, item in checkpoint_results.items() if item.get("status") == "blocked_download_failed"}
        summary["checkpoint_status"] = "ready" if not failed_checkpoints else "blocked_download_failed"
        if failed_checkpoints:
            summary.update(
                {
                    "status": "blocked_checkpoint_host_or_download_failed",
                    "decision": "MASt3R-SLAM true backend smoke blocked: official Naver checkpoint host/download failed, so the real model/backend was not run.",
                    "elapsed_sec": float(time.time() - started),
                }
            )
            _write_result(output_dir, summary)
            output_volume.commit()
            return summary

        calib_path = work_root / "intrinsics.yaml"
        summary["steps"]["calib_prepare"] = _write_calib_from_sidecar(asset_dir, calib_path)
        config_path = "config/calib.yaml" if use_calib and summary["steps"]["calib_prepare"]["status"] == "written" else "config/base.yaml"
        cmd = [
            sys.executable,
            "main.py",
            "--dataset",
            str(run_image_dir),
            "--config",
            config_path,
            "--save-as",
            save_as,
            "--no-viz",
        ]
        if use_calib and summary["steps"]["calib_prepare"]["status"] == "written":
            cmd += ["--calib", str(calib_path)]
        summary["steps"]["main_true_backend_run"] = _run(cmd, repo, timeout=int(main_timeout_sec))
        summary["main_returncode"] = summary["steps"]["main_true_backend_run"].get("returncode")
        summary["logs"] = _collect_logs(repo, save_as)

        if summary["logs"]["exists"]:
            dst_logs = output_dir / "mast3r_slam_logs"
            if dst_logs.exists():
                shutil.rmtree(dst_logs)
            shutil.copytree(repo / "logs" / save_as, dst_logs)
            summary["logs"]["copied_to"] = str(dst_logs)

        if summary["main_returncode"] == 0 and summary["logs"]["file_count"] > 0:
            summary["status"] = "true_backend_completed_research_only"
            summary["decision"] = "MASt3R-SLAM official backend ran on staged 4K4D PNG input and produced its own research-only SLAM logs. This is not a VGGT teacher/candidate/predictions artifact."
        elif summary["main_returncode"] == 0:
            summary["status"] = "true_backend_completed_no_saved_logs"
            summary["decision"] = "MASt3R-SLAM official backend returned successfully on staged 4K4D PNG input, but no saved logs were found under the expected MASt3R-SLAM logs directory."
        else:
            summary["status"] = "true_backend_runtime_failed"
            summary["decision"] = "MASt3R-SLAM dependencies and checkpoints were ready, but the official backend run failed or timed out on the staged 4K4D PNG sequence. See main_true_backend_run stdout tail."
        summary["elapsed_sec"] = float(time.time() - started)
    except Exception as exc:
        summary.update(
            {
                "status": "true_backend_exception",
                "exception": repr(exc),
                "decision": "MASt3R-SLAM true backend smoke hit an exception; no synthetic substitute artifact was generated.",
                "elapsed_sec": float(time.time() - started),
            }
        )

    _write_result(output_dir, summary)
    output_volume.commit()
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
def run_smoke(
    remote_asset_subdir: str = "surface_research_cloud_preflight/V9_cloud_asset_staging/assets",
    remote_output_subdir: str = "surface_research_cloud_preflight/Cloud_B_V9/mast3r_slam_true_backend_smoke",
    download_local_dir: str = "",
    image_count: int = 8,
    use_calib: bool = False,
    main_timeout_sec: int = 1800,
) -> None:
    asset_subdir = _normalize_subpath(remote_asset_subdir)
    output_subdir = _normalize_subpath(remote_output_subdir)
    payload = run_true_backend_smoke.remote(asset_subdir, output_subdir, int(image_count), bool(use_calib), int(main_timeout_sec))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    local_dir = Path(download_local_dir).expanduser().resolve() if download_local_dir.strip() else REPO_ROOT / "output" / output_subdir
    _download_volume_dir(output_subdir, local_dir)
    print(f"[v9-mast3r-slam] downloaded artifacts to {local_dir}")
