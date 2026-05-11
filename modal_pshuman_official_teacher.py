from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
import zipfile
from collections import deque
from pathlib import Path, PurePosixPath

import modal


REMOTE_DATA_DIR = PurePosixPath("/mnt/data")
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")
REMOTE_CACHE_DIR = PurePosixPath("/mnt/cache")

APP_NAME = os.environ.get("VGGT_MODAL_PSHUMAN_OFFICIAL_APP_NAME", "vggt-pshuman-official-teacher")
DATA_VOLUME_NAME = os.environ.get("VGGT_MODAL_DATA_VOLUME", "vggt-4k4d-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
CACHE_VOLUME_NAME = os.environ.get("VGGT_MODAL_PSHUMAN_OFFICIAL_CACHE_VOLUME", "vggt-pshuman-official-cache")
GPU_SPEC = os.environ.get("VGGT_MODAL_PSHUMAN_OFFICIAL_GPU", "A100-80GB")
CPU_COUNT = float(os.environ.get("VGGT_MODAL_PSHUMAN_OFFICIAL_CPU", "16"))
MEMORY_MB = int(os.environ.get("VGGT_MODAL_PSHUMAN_OFFICIAL_MEMORY_MB", str(96 * 1024)))
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_PSHUMAN_OFFICIAL_TIMEOUT_SEC", str(6 * 60 * 60)))

DEFAULT_SCENE_SUBDIR = (
    "4k4d_preprocessed_scene_variants/"
    "0012_11_frame0000_6views_sparseproto_headshoulder_crop"
)
DEFAULT_LOCAL_SCENE_DIR = f"output/{DEFAULT_SCENE_SUBDIR}"
DEFAULT_REMOTE_SCENE_SUBDIR = f"pshuman_official_teacher_smoke/{DEFAULT_SCENE_SUBDIR}"
DEFAULT_OUTPUT_SUBDIR = "detail_normal_refiner_20260426/pshuman_official_teacher_cam30"
DEFAULT_DOWNLOAD_LOCAL_DIR = f"output/{DEFAULT_OUTPUT_SUBDIR}"
DEFAULT_IMAGE_NAME = "30_src_cam30.png"

PSHUMAN_REPO = "https://github.com/pengHTYX/PSHuman"
PSHUMAN_REF = os.environ.get("VGGT_MODAL_PSHUMAN_OFFICIAL_REF", "main")
PSHUMAN_SOURCE_ZIP_URL = os.environ.get(
    "VGGT_MODAL_PSHUMAN_OFFICIAL_SOURCE_ZIP_URL",
    f"https://codeload.github.com/pengHTYX/PSHuman/zip/refs/heads/{PSHUMAN_REF}",
)
PSHUMAN_REQUIREMENTS_URL = (
    "https://raw.githubusercontent.com/pengHTYX/PSHuman/main/requirements.txt"
)
MIN_SOURCE_ZIP_BYTES = 100_000


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "build-essential",
        "ca-certificates",
        "curl",
        "ffmpeg",
        "git",
        "libegl1",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
        "libsm6",
        "libxext6",
        "libxrender1",
        "ninja-build",
        "wget",
    )
    .pip_install(
        "Pillow==10.2.0",
        "huggingface_hub==0.24.5",
        "numpy==1.26.3",
        "omegaconf==2.3.0",
        "packaging",
        "requests==2.32.3",
        "wheel",
    )
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)


def _norm(value: str) -> str:
    value = (value or "").replace("\\", "/").strip("/")
    if not value:
        raise ValueError("empty subpath")
    return value


def _upload_dir(local_dir: Path, remote_subdir: str) -> str:
    local_dir = local_dir.expanduser().resolve()
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Local scene directory not found: {local_dir}")
    remote_subdir = _norm(remote_subdir)
    with data_volume.batch_upload(force=True) as batch:
        for path in local_dir.rglob("*"):
            if path.is_file():
                batch.put_file(str(path), f"{remote_subdir}/{path.relative_to(local_dir).as_posix()}")
    return remote_subdir


def _download_dir(remote_subdir: str, local_dir: Path) -> None:
    remote_subdir = _norm(remote_subdir)
    local_dir = local_dir.expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    prefix = Path(remote_subdir)
    for entry in output_volume.listdir(remote_subdir, recursive=True):
        rel_path = Path(entry.path)
        try:
            rel_path = rel_path.relative_to(prefix)
        except ValueError:
            pass
        dest_path = local_dir / rel_path
        if entry.type == modal.volume.FileEntryType.DIRECTORY:
            dest_path.mkdir(parents=True, exist_ok=True)
            continue
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with dest_path.open("wb") as file_obj:
            output_volume.read_file_into_fileobj(entry.path, file_obj)


def _remote_data_path(subdir: str) -> Path:
    return Path(str(REMOTE_DATA_DIR / _norm(subdir)))


def _remote_output_path(subdir: str) -> Path:
    return Path(str(REMOTE_OUTPUT_DIR / _norm(subdir)))


def _run_command(
    cmd: list[str],
    cwd: Path | None,
    log_path: Path,
    env_extra: dict[str, str] | None = None,
    tail_lines: int = 240,
) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if env_extra:
        env.update(env_extra)

    started = time.time()
    tail = deque(maxlen=tail_lines)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n")
        log_file.write(f"cwd={cwd.as_posix() if cwd else os.getcwd()}\n\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            tail.append(line.rstrip("\n"))
        returncode = proc.wait()

    return {
        "cmd": cmd,
        "cwd": cwd.as_posix() if cwd else None,
        "returncode": int(returncode),
        "seconds": round(time.time() - started, 3),
        "log": log_path.name,
        "stdout_tail": "\n".join(tail),
    }


def _download_url(url: str, dest_path: Path, min_bytes: int) -> dict:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + ".part")
    tmp_path.unlink(missing_ok=True)
    started = time.time()
    request = urllib.request.Request(url, headers={"User-Agent": "vggt-pshuman-official-modal-smoke"})
    bytes_written = 0
    with urllib.request.urlopen(request, timeout=180) as response:
        with tmp_path.open("wb") as file_obj:
            while True:
                chunk = response.read(16 * 1024 * 1024)
                if not chunk:
                    break
                file_obj.write(chunk)
                bytes_written += len(chunk)
                if bytes_written and bytes_written % (256 * 1024 * 1024) < len(chunk):
                    print(f"[pshuman] downloaded {bytes_written / (1024 ** 2):.1f} MiB from {url}")
    size = tmp_path.stat().st_size
    if size < min_bytes:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is too small: {url} -> {size} bytes")
    tmp_path.replace(dest_path)
    return {"url": url, "path": dest_path.as_posix(), "bytes": int(size), "seconds": round(time.time() - started, 3)}


def _ensure_pshuman_source(cache_root: Path, source_zip_url: str) -> tuple[Path, dict]:
    source_root = cache_root / f"PSHuman-{PSHUMAN_REF}"
    marker = source_root / "inference.py"
    if marker.is_file():
        return source_root, {"cache_hit": True, "path": source_root.as_posix()}

    cache_root.mkdir(parents=True, exist_ok=True)
    zip_path = cache_root / f"PSHuman-{PSHUMAN_REF}.zip"
    download_info = None
    if not zip_path.is_file() or zip_path.stat().st_size < MIN_SOURCE_ZIP_BYTES:
        download_info = _download_url(source_zip_url, zip_path, MIN_SOURCE_ZIP_BYTES)

    tmp_dir = Path(tempfile.mkdtemp(prefix="pshuman_src_"))
    try:
        with zipfile.ZipFile(zip_path) as zip_obj:
            zip_obj.extractall(tmp_dir)
        extracted = next((path for path in tmp_dir.iterdir() if (path / "inference.py").is_file()), None)
        if extracted is None:
            raise RuntimeError(f"PSHuman archive did not contain inference.py: {zip_path}")
        if source_root.exists():
            shutil.rmtree(source_root)
        shutil.copytree(extracted, source_root)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return source_root, {
        "cache_hit": False,
        "path": source_root.as_posix(),
        "download": download_info,
        "repo": PSHUMAN_REPO,
        "ref": PSHUMAN_REF,
    }


def _select_input_paths(scene_root: Path, image_name: str, mask_name: str) -> tuple[Path, Path | None]:
    images_dir = scene_root / "images"
    masks_dir = scene_root / "masks"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Remote images directory not found: {images_dir}")
    image_path = images_dir / image_name
    if not image_path.is_file():
        available = sorted(path.name for path in images_dir.glob("*") if path.is_file())
        raise FileNotFoundError(f"Requested image not found: {image_path}; available={available}")
    resolved_mask_name = mask_name or image_path.name
    mask_path = masks_dir / resolved_mask_name
    if mask_name and not mask_path.is_file():
        raise FileNotFoundError(f"Requested mask not found: {mask_path}")
    return image_path, mask_path if mask_path.is_file() else None


def _prepare_rgba_input(image_path: Path, mask_path: Path | None, out_path: Path, background: int) -> dict:
    from PIL import Image
    import numpy as np

    rgb = Image.open(image_path).convert("RGB")
    width, height = rgb.size
    if mask_path is not None:
        alpha = Image.open(mask_path).convert("L").resize((width, height), Image.Resampling.NEAREST)
        mask_arr = np.asarray(alpha, dtype=np.uint8) > 127
    else:
        alpha = Image.new("L", (width, height), 255)
        mask_arr = np.ones((height, width), dtype=bool)

    rgb_arr = np.asarray(rgb, dtype=np.uint8)
    bg = np.full_like(rgb_arr, int(max(0, min(255, background))))
    composited = np.where(mask_arr[..., None], rgb_arr, bg)
    rgba = Image.merge("RGBA", (*Image.fromarray(composited, mode="RGB").split(), alpha))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(out_path)
    return {
        "source_image": image_path.name,
        "source_mask": mask_path.name if mask_path is not None else None,
        "prepared_rgba": out_path.name,
        "size_wh": [width, height],
        "mask_used": mask_path is not None,
        "mask_pixels": int(mask_arr.sum()),
        "background": int(background),
    }


def _inspect_assets(source_dir: Path) -> dict:
    relative_paths = [
        "configs/inference-768-6view.yaml",
        "mvdiffusion/data/fixed_prompt_embeds_7view",
        "mvdiffusion/data/six_human_pose",
        "smpl_related/HPS/pixie_data/pixie_model.tar",
        "smpl_related/HPS/pixie_data/SMPLX_NEUTRAL_2020.npz",
        "smpl_related/HPS/pixie_data/SMPL_X_template_FLAME_uv.obj",
        "data/HPS/pymaf_data/pretrained_model/PyMAF_model_checkpoint.pt",
        "data/smpl_related/models/smpl/SMPL_NEUTRAL.pkl",
        "data/smpl_related/models/smplx/SMPLX_NEUTRAL.npz",
    ]
    records = []
    missing = []
    for rel_path in relative_paths:
        path = source_dir / rel_path
        exists = path.exists()
        record = {"path": rel_path, "exists": bool(exists)}
        if path.is_file():
            record["bytes"] = int(path.stat().st_size)
        elif path.is_dir():
            record["entries"] = len(list(path.iterdir()))
        if not exists:
            missing.append(rel_path)
        records.append(record)
    return {"records": records, "missing": missing}


def _install_runtime_deps(out_root: Path) -> list[dict]:
    logs_dir = out_root / "install_logs"
    common_packages = [
        "accelerate==1.1.1",
        "diffusers==0.27.2",
        "einops==0.8.0",
        "icecream==2.1.3",
        "imageio==2.36.0",
        "imageio-ffmpeg==0.5.1",
        "kornia==0.7.4",
        "matplotlib==3.9.2",
        "mediapipe==0.10.18",
        "onnxruntime-gpu==1.20.0",
        "open3d==0.18.0",
        "opencv-python-headless==4.10.0.84",
        "peft==0.13.2",
        "pymeshlab==2023.12.post2",
        "PyMatting==1.1.13",
        "rembg==2.0.59",
        "safetensors==0.4.5",
        "scikit-image==0.24.0",
        "scikit-learn==1.5.2",
        "scipy==1.14.1",
        "termcolor==2.5.0",
        "tqdm==4.67.0",
        "transformers==4.46.2",
        "trimesh==4.5.2",
        "yacs==0.1.8",
    ]
    commands = [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
            "ninja",
            "packaging",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
            "torch==2.1.2",
            "torchvision==0.16.2",
            "xformers==0.0.23.post1",
        ],
        [sys.executable, "-m", "pip", "install", *common_packages],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch-scatter==2.1.2",
            "-f",
            "https://data.pyg.org/whl/torch-2.1.0+cu121.html",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "git+https://github.com/NVlabs/nvdiffrast.git@729261dc64c4241ea36efda84fbf532cc8b425b8",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47",
        ],
        [sys.executable, "-m", "pip", "install", "kaolin==0.17.0"],
    ]

    results = []
    for index, cmd in enumerate(commands, start=1):
        result = _run_command(
            cmd,
            cwd=None,
            log_path=logs_dir / f"{index:02d}_{cmd[-1].split('==')[0].replace('/', '_')}.log",
            env_extra={
                "CUDA_HOME": "/usr/local/cuda",
                "FORCE_CUDA": "1",
                "MAX_JOBS": "8",
            },
        )
        results.append(result)
        if result["returncode"] != 0:
            raise RuntimeError(f"Dependency install failed at step {index}; see {result['log']}")
    return results


def _import_probe(out_root: Path, source_dir: Path) -> dict:
    probe_code = """
import importlib, json
mods = [
    'torch', 'torchvision', 'diffusers', 'transformers', 'accelerate',
    'pytorch3d', 'nvdiffrast.torch', 'kaolin', 'pymeshlab', 'open3d',
    'rembg', 'kornia', 'trimesh', 'mvdiffusion.pipelines.pipeline_mvdiffusion_unclip',
]
result = {}
for mod in mods:
    try:
        obj = importlib.import_module(mod)
        result[mod] = {'ok': True, 'version': getattr(obj, '__version__', None)}
    except Exception as exc:
        result[mod] = {'ok': False, 'error': repr(exc)}
print(json.dumps(result, indent=2, ensure_ascii=False))
bad = [k for k, v in result.items() if not v['ok']]
raise SystemExit(1 if bad else 0)
""".strip()
    probe_path = out_root / "pshuman_import_probe.py"
    probe_path.write_text(probe_code, encoding="utf-8")
    result = _run_command(
        [sys.executable, probe_path.as_posix()],
        cwd=source_dir,
        log_path=out_root / "pshuman_import_probe.log",
        env_extra={"PYTHONPATH": source_dir.as_posix()},
    )
    return result


def _mesh_record(path: Path) -> dict:
    record = {"path": path.name, "bytes": int(path.stat().st_size), "suffix": path.suffix.lower()}
    if path.suffix.lower() == ".obj":
        vertices = 0
        faces = 0
        with path.open("r", encoding="utf-8", errors="ignore") as file_obj:
            for line in file_obj:
                if line.startswith("v "):
                    vertices += 1
                elif line.startswith("f "):
                    faces += 1
        record.update({"vertices": vertices, "faces": faces})
    elif path.suffix.lower() == ".ply":
        with path.open("rb") as file_obj:
            header = file_obj.read(4096).decode("utf-8", errors="ignore")
        for line in header.splitlines():
            if line.startswith("element vertex "):
                record["vertices"] = int(line.split()[-1])
            elif line.startswith("element face "):
                record["faces"] = int(line.split()[-1])
    return record


def _copy_artifacts(stage_root: Path, out_root: Path) -> list[dict]:
    artifact_exts = {".obj", ".ply", ".glb", ".mp4"}
    records = []
    for path in sorted(stage_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in artifact_exts:
            continue
        dest = out_root / path.name
        if dest.exists():
            dest = out_root / f"{path.stem}_{len(records):02d}{path.suffix}"
        shutil.copy2(path, dest)
        records.append(_mesh_record(dest))
    return records


@app.function(
    image=image,
    gpu=GPU_SPEC,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SEC,
    volumes={
        REMOTE_OUTPUT_DIR.as_posix(): output_volume,
        REMOTE_CACHE_DIR.as_posix(): cache_volume,
    },
)
def probe_pshuman_official_remote(
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    source_zip_url: str = PSHUMAN_SOURCE_ZIP_URL,
    install_deps: bool = False,
) -> dict:
    out_root = _remote_output_path(output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "ok": False,
        "mode": "probe",
        "output_subdir": _norm(output_subdir),
        "pshuman_repo": PSHUMAN_REPO,
        "pshuman_ref": PSHUMAN_REF,
        "source_zip_url": source_zip_url,
        "requirements_url": PSHUMAN_REQUIREMENTS_URL,
        "gpu": GPU_SPEC,
        "started_at_unix": time.time(),
    }
    try:
        summary["nvidia_smi"] = _run_command(
            ["nvidia-smi"],
            cwd=None,
            log_path=out_root / "nvidia_smi.log",
        )
        cache_root = Path(str(REMOTE_CACHE_DIR))
        source_dir, source_info = _ensure_pshuman_source(cache_root, source_zip_url)
        summary["pshuman_source"] = source_info
        summary["assets"] = _inspect_assets(source_dir)
        if install_deps:
            summary["install"] = _install_runtime_deps(out_root)
            summary["import_probe"] = _import_probe(out_root, source_dir)
            if summary["import_probe"]["returncode"] != 0:
                raise RuntimeError("Import probe failed after dependency install")
        cache_volume.commit()
        summary["ok"] = True
        summary["finished_at_unix"] = time.time()
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        (out_root / "pshuman_official_blocker.txt").write_text(
            "PSHuman official probe did not complete.\n\n"
            f"Error: {repr(exc)}\n\n"
            f"Traceback:\n{summary['traceback']}\n",
            encoding="utf-8",
        )
    finally:
        (out_root / "pshuman_official_teacher_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        output_volume.commit()
    return summary


@app.function(
    image=image,
    gpu=GPU_SPEC,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SEC,
    volumes={
        REMOTE_DATA_DIR.as_posix(): data_volume,
        REMOTE_OUTPUT_DIR.as_posix(): output_volume,
        REMOTE_CACHE_DIR.as_posix(): cache_volume,
    },
)
def run_pshuman_official_remote(
    scene_subdir: str,
    output_subdir: str,
    image_name: str = DEFAULT_IMAGE_NAME,
    mask_name: str = "",
    config_name: str = "configs/inference-768-6view.yaml",
    source_zip_url: str = PSHUMAN_SOURCE_ZIP_URL,
    install_deps: bool = True,
    run_import_probe: bool = True,
    num_inference_steps: int = 40,
    recon_iters: int = 700,
    color_iters: int = 200,
    recon_resolution: int = 1024,
    background: int = 255,
) -> dict:
    out_root = _remote_output_path(output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "ok": False,
        "mode": "run",
        "scene_subdir": _norm(scene_subdir),
        "output_subdir": _norm(output_subdir),
        "image_name": image_name,
        "mask_name": mask_name,
        "config_name": config_name,
        "pshuman_repo": PSHUMAN_REPO,
        "pshuman_ref": PSHUMAN_REF,
        "source_zip_url": source_zip_url,
        "requirements_url": PSHUMAN_REQUIREMENTS_URL,
        "gpu": GPU_SPEC,
        "official_route_notes": [
            "Uses official pengHTYX/PSHuman source, not the queued HF Space API.",
            "The 768 6-view config is designed for high-memory GPUs; this smoke requests A100-80GB.",
            "Mesh reconstruction imports PIXIE/SMPLX/HPS assets in addition to diffusion/runtime packages.",
        ],
        "started_at_unix": time.time(),
    }
    try:
        summary["nvidia_smi"] = _run_command(
            ["nvidia-smi"],
            cwd=None,
            log_path=out_root / "nvidia_smi.log",
        )

        scene_root = _remote_data_path(scene_subdir)
        image_path, mask_path = _select_input_paths(scene_root, image_name, mask_name)
        stage_root = Path(tempfile.mkdtemp(prefix="pshuman_official_teacher_"))
        input_root = stage_root / "input_images"
        input_path = input_root / f"{Path(image_name).stem}.png"
        summary["input"] = _prepare_rgba_input(image_path, mask_path, input_path, background=background)
        shutil.copy2(input_path, out_root / input_path.name)

        cache_root = Path(str(REMOTE_CACHE_DIR))
        source_dir, source_info = _ensure_pshuman_source(cache_root, source_zip_url)
        summary["pshuman_source"] = source_info
        summary["assets"] = _inspect_assets(source_dir)
        cache_volume.commit()

        config_path = source_dir / config_name
        if not config_path.is_file():
            candidates = sorted(source_dir.glob("configs/*inference*6view*.yaml"))
            if candidates:
                config_path = candidates[0]
            else:
                raise FileNotFoundError(f"PSHuman config not found: {source_dir / config_name}")
        shutil.copy2(config_path, out_root / f"input_{config_path.name}")

        if install_deps:
            summary["install"] = _install_runtime_deps(out_root)

        if run_import_probe:
            summary["import_probe"] = _import_probe(out_root, source_dir)
            if summary["import_probe"]["returncode"] != 0:
                raise RuntimeError("Import probe failed; see pshuman_import_probe.log")

        mv_results_dir = stage_root / "mv_results"
        recon_dir = stage_root / "recon"
        cmd = [
            sys.executable,
            "inference.py",
            "--config",
            config_path.as_posix(),
            f"validation_dataset.root_dir={input_root.as_posix()}",
            "validation_dataset.num_validation_samples=1",
            "validation_dataset.bg_color=white",
            "validation_batch_size=1",
            "dataloader_num_workers=0",
            f"save_dir={mv_results_dir.as_posix()}",
            f"recon_opt.res_path={recon_dir.as_posix()}",
            f"recon_opt.iters={int(recon_iters)}",
            f"recon_opt.clr_iters={int(color_iters)}",
            f"recon_opt.resolution={int(recon_resolution)}",
            "recon_opt.gpu_id=0",
            f"pipe_validation_kwargs.num_inference_steps={int(num_inference_steps)}",
        ]
        (out_root / "pshuman_official_command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
        command_result = _run_command(
            cmd,
            cwd=source_dir,
            log_path=out_root / "pshuman_official_command.log",
            env_extra={
                "PYTHONPATH": source_dir.as_posix(),
                "HF_HOME": (cache_root / "huggingface").as_posix(),
                "HUGGINGFACE_HUB_CACHE": (cache_root / "huggingface" / "hub").as_posix(),
                "TORCH_HOME": (cache_root / "torch").as_posix(),
                "CUDA_HOME": "/usr/local/cuda",
                "FORCE_CUDA": "1",
            },
        )
        summary["command"] = command_result
        if command_result["returncode"] != 0:
            raise RuntimeError("PSHuman official inference command failed; see pshuman_official_command.log")

        artifacts = _copy_artifacts(stage_root, out_root)
        summary["artifacts"] = artifacts
        mesh_artifacts = [item for item in artifacts if item["suffix"] in {".obj", ".ply", ".glb"}]
        if not mesh_artifacts:
            raise RuntimeError(f"PSHuman completed but produced no mesh artifact under {stage_root}")

        summary["ok"] = True
        summary["finished_at_unix"] = time.time()
        blocker_path = out_root / "pshuman_official_blocker.txt"
        blocker_path.unlink(missing_ok=True)
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        (out_root / "pshuman_official_blocker.txt").write_text(
            "PSHuman official/self-hosted mesh teacher smoke did not complete.\n\n"
            f"Error: {repr(exc)}\n\n"
            f"Traceback:\n{summary['traceback']}\n",
            encoding="utf-8",
        )
    finally:
        (out_root / "pshuman_official_teacher_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        output_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    mode: str = "run",
    local_scene_dir: str = DEFAULT_LOCAL_SCENE_DIR,
    remote_scene_subdir: str = DEFAULT_REMOTE_SCENE_SUBDIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    download_local_dir: str = DEFAULT_DOWNLOAD_LOCAL_DIR,
    image_name: str = DEFAULT_IMAGE_NAME,
    mask_name: str = "",
    config_name: str = "configs/inference-768-6view.yaml",
    source_zip_url: str = PSHUMAN_SOURCE_ZIP_URL,
    install_deps: bool = True,
    run_import_probe: bool = True,
    num_inference_steps: int = 40,
    recon_iters: int = 700,
    color_iters: int = 200,
    recon_resolution: int = 1024,
    background: int = 255,
):
    if mode == "probe":
        summary = probe_pshuman_official_remote.remote(
            output_subdir=output_subdir,
            source_zip_url=source_zip_url,
            install_deps=install_deps,
        )
    elif mode == "run":
        remote = _upload_dir(Path(local_scene_dir), remote_scene_subdir)
        summary = run_pshuman_official_remote.remote(
            scene_subdir=remote,
            output_subdir=output_subdir,
            image_name=image_name,
            mask_name=mask_name,
            config_name=config_name,
            source_zip_url=source_zip_url,
            install_deps=install_deps,
            run_import_probe=run_import_probe,
            num_inference_steps=num_inference_steps,
            recon_iters=recon_iters,
            color_iters=color_iters,
            recon_resolution=recon_resolution,
            background=background,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if download_local_dir:
        _download_dir(summary["output_subdir"], Path(download_local_dir))
