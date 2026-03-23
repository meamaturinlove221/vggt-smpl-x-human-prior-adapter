import json
import os
import shlex
import subprocess
import sys
import time
import base64
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import modal


REPO_ROOT = Path(__file__).resolve().parent
REMOTE_CODE_DIR = PurePosixPath("/workspace/vggt")
REMOTE_DATA_DIR = PurePosixPath("/mnt/data")
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")
REQUIREMENTS_TRAINING = "requirements_training.txt"
DEFAULT_TRAINING_REQUIREMENTS = [
    "torch==2.3.1",
    "torchvision==0.18.1",
    "numpy==1.26.1",
    "Pillow",
    "huggingface_hub",
    "einops",
    "safetensors",
    "hydra-core",
    "omegaconf",
    "fvcore",
    "iopath",
    "wcmatch",
    "tensorboard",
    "tqdm",
    "opencv-python",
]


def _load_requirements(path: Path, seen: set[str] | None = None) -> list[str]:
    seen = set() if seen is None else seen
    packages: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            nested = (path.parent / line[3:].strip()).resolve()
            packages.extend(_load_requirements(nested, seen))
            continue
        if line not in seen:
            seen.add(line)
            packages.append(line)
    return packages


def _resolve_training_requirements() -> list[str]:
    candidates = [
        REPO_ROOT / REQUIREMENTS_TRAINING,
        Path.cwd() / REQUIREMENTS_TRAINING,
    ]
    for candidate in candidates:
        if candidate.exists():
            return _load_requirements(candidate)
    return list(DEFAULT_TRAINING_REQUIREMENTS)


TRAINING_REQUIREMENTS = _resolve_training_requirements()


def _decode_config_blob(blob: str) -> dict:
    text = blob
    if text.startswith("base64:"):
        text = base64.b64decode(text[len("base64:") :]).decode("utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoded = base64.b64decode(text).decode("utf-8")
        return json.loads(decoded)

APP_NAME = os.environ.get("VGGT_MODAL_APP_NAME", "vggt-geometry-minimal-finetune")
DATA_VOLUME_NAME = os.environ.get("VGGT_MODAL_DATA_VOLUME", "vggt-geometry-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-geometry-output")
GPU_SPEC = os.environ.get("VGGT_MODAL_GPU", "A100-40GB")
CPU_COUNT = float(os.environ.get("VGGT_MODAL_CPU", "8"))
MEMORY_MB = int(os.environ.get("VGGT_MODAL_MEMORY_MB", "65536"))
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_TIMEOUT_SEC", str(24 * 60 * 60)))

CODE_SYNC_IGNORE = [
    ".git",
    ".git/**",
    ".github/**",
    "__pycache__",
    "__pycache__/**",
    ".pytest_cache",
    ".pytest_cache/**",
    ".venv*",
    ".venv*/**",
    "training/logs",
    "training/logs/**",
    "training/**/ckpts",
    "training/**/ckpts/**",
    "**/ckpts",
    "**/ckpts/**",
    "output",
    "output/**",
    "logs",
    "logs/**",
    "infer_out",
    "infer_out/**",
    "ckpt",
    "ckpt/**",
    "checkpoints_view_decoder",
    "checkpoints_view_decoder/**",
    "debug_viewdec",
    "debug_viewdec/**",
    "debug_viewdec_ablation",
    "debug_viewdec_ablation/**",
    "debug_vis",
    "debug_vis/**",
    "out_vis",
    "out_vis/**",
    "overfit_debug",
    "overfit_debug/**",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.npz",
    "*.ply",
    "*.obj",
    "*.glb",
    "*.zip",
    "*.tar",
    "*.tar.*",
]

TRAINING_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "build-essential",
        "ffmpeg",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .pip_install(*TRAINING_REQUIREMENTS)
    .add_local_dir(
        str(REPO_ROOT / "training"),
        remote_path=(REMOTE_CODE_DIR / "training").as_posix(),
        ignore=CODE_SYNC_IGNORE,
    )
    .add_local_dir(
        str(REPO_ROOT / "vggt"),
        remote_path=(REMOTE_CODE_DIR / "vggt").as_posix(),
        ignore=CODE_SYNC_IGNORE,
    )
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


@dataclass
class FinetuneConfig:
    co3d_subdir: str
    co3d_annotation_subdir: str
    checkpoint_subpath: str = "checkpoints/model.pt"
    config: str = "default"
    exp_name: str = "geometry_minimal_modal"
    output_subdir: str = ""
    max_img_per_gpu: int = 8
    accum_steps: int = 2
    max_epochs: int = 5
    learning_rate: float = 5e-5
    limit_train_batches: int = 200
    limit_val_batches: int = 100
    freeze_aggregator: bool = True
    extra_overrides: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(blob: str) -> "FinetuneConfig":
        return FinetuneConfig(**_decode_config_blob(blob))


def _normalize_subpath(value: str) -> str:
    cleaned = (value or "").strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("Expected a non-empty volume-relative path.")
    return cleaned


def _remote_data_path(subpath: str) -> Path:
    return Path(str(REMOTE_DATA_DIR / _normalize_subpath(subpath)))


def _resolve_output_root(exp_name: str, output_subdir: str) -> Path:
    if output_subdir.strip():
        return Path(str(REMOTE_OUTPUT_DIR / _normalize_subpath(output_subdir)))
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    safe_exp = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in exp_name).strip("_")
    safe_exp = safe_exp or "geometry_minimal_modal"
    return Path(str(REMOTE_OUTPUT_DIR / "geometry_minimal" / f"{run_tag}_{safe_exp}"))


def _build_overrides(cfg: FinetuneConfig, output_root: Path) -> list[str]:
    ckpt_path = _remote_data_path(cfg.checkpoint_subpath)
    co3d_dir = _remote_data_path(cfg.co3d_subdir)
    anno_dir = _remote_data_path(cfg.co3d_annotation_subdir)
    logging_dir = output_root / "logs"
    checkpoint_dir = output_root / "ckpts"

    overrides = [
        f"exp_name={cfg.exp_name}",
        f"logging.log_dir={logging_dir.as_posix()}",
        f"checkpoint.save_dir={checkpoint_dir.as_posix()}",
        f"checkpoint.resume_checkpoint_path={ckpt_path.as_posix()}",
        f"data.train.dataset.dataset_configs[0].CO3D_DIR={co3d_dir.as_posix()}",
        f"data.train.dataset.dataset_configs[0].CO3D_ANNOTATION_DIR={anno_dir.as_posix()}",
        f"data.val.dataset.dataset_configs[0].CO3D_DIR={co3d_dir.as_posix()}",
        f"data.val.dataset.dataset_configs[0].CO3D_ANNOTATION_DIR={anno_dir.as_posix()}",
        f"max_img_per_gpu={int(cfg.max_img_per_gpu)}",
        f"accum_steps={int(cfg.accum_steps)}",
        f"max_epochs={int(cfg.max_epochs)}",
        f"optim.optimizer.lr={float(cfg.learning_rate)}",
        f"limit_train_batches={int(cfg.limit_train_batches)}",
        f"limit_val_batches={int(cfg.limit_val_batches)}",
        "model.enable_camera=True",
        "model.enable_depth=True",
        "model.enable_point=False",
        "model.enable_track=False",
        "loss.point=null",
        "loss.track=null",
    ]

    if not cfg.freeze_aggregator:
        overrides.append("optim.frozen_module_names=[]")

    if cfg.extra_overrides.strip():
        overrides.extend(shlex.split(cfg.extra_overrides))

    return overrides


def _upload_checkpoint(local_path: str, remote_subpath: str) -> str:
    checkpoint_path = Path(local_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    remote_path = _normalize_subpath(remote_subpath)
    print(f"[modal-geometry] upload checkpoint: {checkpoint_path} -> {DATA_VOLUME_NAME}:{remote_path}")
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(checkpoint_path), remote_path)
    return remote_path


@app.function(
    image=TRAINING_IMAGE,
    gpu=GPU_SPEC,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SEC,
    volumes={
        REMOTE_DATA_DIR.as_posix(): data_volume,
        REMOTE_OUTPUT_DIR.as_posix(): output_volume,
    },
)
def run_remote_geometry_finetune(cfg_json: str) -> None:
    cfg = FinetuneConfig.from_json(cfg_json)
    co3d_dir = _remote_data_path(cfg.co3d_subdir)
    anno_dir = _remote_data_path(cfg.co3d_annotation_subdir)
    ckpt_path = _remote_data_path(cfg.checkpoint_subpath)

    missing = [str(path) for path in (co3d_dir, anno_dir, ckpt_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required Modal volume paths are missing:\n- " + "\n- ".join(missing)
        )

    output_root = _resolve_output_root(cfg.exp_name, cfg.output_subdir)
    output_root.mkdir(parents=True, exist_ok=True)
    overrides = _build_overrides(cfg, output_root)
    remote_code_dir = Path(str(REMOTE_CODE_DIR))
    launch_path = remote_code_dir / "training" / "launch.py"
    cmd = [sys.executable, str(launch_path), "--config", cfg.config, *overrides]

    print("[modal-geometry] data_volume =", DATA_VOLUME_NAME, flush=True)
    print("[modal-geometry] output_volume =", OUTPUT_VOLUME_NAME, flush=True)
    print("[modal-geometry] gpu =", GPU_SPEC, flush=True)
    print("[modal-geometry] output_root =", output_root.as_posix(), flush=True)
    print("[modal-geometry] command =", shlex.join(cmd), flush=True)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = str(remote_code_dir)
    env["PYTHONPATH"] = (
        repo_pythonpath
        if not existing_pythonpath
        else repo_pythonpath + os.pathsep + existing_pythonpath
    )
    env["PYTHONUNBUFFERED"] = "1"

    subprocess.run(
        cmd,
        cwd=str(remote_code_dir),
        env=env,
        check=True,
    )

    output_volume.commit()
    print("[modal-geometry] committed output volume", flush=True)


@app.local_entrypoint()
def upload_checkpoint(
    local_path: str,
    remote_subpath: str = "checkpoints/model.pt",
) -> None:
    remote_path = _upload_checkpoint(local_path, remote_subpath)
    print(f"[modal-geometry] checkpoint uploaded to {DATA_VOLUME_NAME}:{remote_path}")


@app.local_entrypoint()
def run_geometry_finetune(
    co3d_subdir: str,
    co3d_annotation_subdir: str,
    checkpoint_subpath: str = "checkpoints/model.pt",
    local_checkpoint: str = "",
    config: str = "default",
    exp_name: str = "geometry_minimal_modal",
    output_subdir: str = "",
    max_img_per_gpu: int = 8,
    accum_steps: int = 2,
    max_epochs: int = 5,
    learning_rate: float = 5e-5,
    limit_train_batches: int = 200,
    limit_val_batches: int = 100,
    freeze_aggregator: bool = True,
    extra_overrides: str = "",
) -> None:
    resolved_checkpoint_subpath = checkpoint_subpath
    if local_checkpoint.strip():
        resolved_checkpoint_subpath = _upload_checkpoint(local_checkpoint, checkpoint_subpath)

    cfg = FinetuneConfig(
        co3d_subdir=co3d_subdir,
        co3d_annotation_subdir=co3d_annotation_subdir,
        checkpoint_subpath=resolved_checkpoint_subpath,
        config=config,
        exp_name=exp_name,
        output_subdir=output_subdir,
        max_img_per_gpu=max_img_per_gpu,
        accum_steps=accum_steps,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        freeze_aggregator=freeze_aggregator,
        extra_overrides=extra_overrides,
    )

    print("[modal-geometry] launch config:")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    run_remote_geometry_finetune.remote(cfg.to_json())
