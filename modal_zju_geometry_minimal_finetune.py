import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import base64
from dataclasses import asdict, dataclass, replace
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

PROMOTED_SOURCE_POLICY_CONFIG = (
    "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_minimal"
)
PREVIOUS_SOURCE_POLICY_CONFIG = (
    "zju_vggt_geom_unproject_source_policy_nearest_rawpool_confdepth_dropworst_gradconfmask_minimal"
)


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

APP_NAME = os.environ.get(
    "VGGT_ZJU_MODAL_APP_NAME",
    os.environ.get("VGGT_MODAL_APP_NAME", "vggt-zju-geometry-minimal-finetune"),
)
DATA_VOLUME_NAME = os.environ.get(
    "VGGT_ZJU_MODAL_DATA_VOLUME",
    os.environ.get("VGGT_MODAL_DATA_VOLUME", "vggt-zju-data"),
)
OUTPUT_VOLUME_NAME = os.environ.get(
    "VGGT_ZJU_MODAL_OUTPUT_VOLUME",
    os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-out"),
)
OUTPUT_CHECKPOINT_FALLBACKS = (
    "weights/model.pt",
    "pretrained_weights/model.pt",
)
GPU_SPEC = os.environ.get("VGGT_ZJU_MODAL_GPU", os.environ.get("VGGT_MODAL_GPU", "A100-80GB"))
CPU_COUNT = float(os.environ.get("VGGT_ZJU_MODAL_CPU", os.environ.get("VGGT_MODAL_CPU", "24")))
MEMORY_MB = int(
    os.environ.get("VGGT_ZJU_MODAL_MEMORY_MB", os.environ.get("VGGT_MODAL_MEMORY_MB", "196608"))
)
TIMEOUT_SEC = int(
    os.environ.get("VGGT_ZJU_MODAL_TIMEOUT_SEC", os.environ.get("VGGT_MODAL_TIMEOUT_SEC", str(24 * 60 * 60)))
)
LIVE_COMMIT_INTERVAL_SEC = int(
    os.environ.get("VGGT_ZJU_MODAL_LIVE_COMMIT_SEC", os.environ.get("VGGT_MODAL_LIVE_COMMIT_SEC", "180"))
)

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
    .add_local_dir(
        str(REPO_ROOT / "scripts"),
        remote_path=(REMOTE_CODE_DIR / "scripts").as_posix(),
        ignore=CODE_SYNC_IGNORE,
    )
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=False)


@dataclass
class ZjuFinetuneConfig:
    zju_subdir: str
    seq_names: str = "CoreView_390"
    geom_subdir: str = "vggt_geom"
    checkpoint_subpath: str = "checkpoints/model.pt"
    config: str = "zju_vggt_geom_minimal"
    exp_name: str = "zju_geometry_minimal_modal"
    output_subdir: str = ""
    num_images: int = 4
    max_img_per_gpu: int = 16
    accum_steps: int = 1
    max_epochs: int = 1
    learning_rate: float = 5e-5
    limit_train_batches: int = 100
    limit_val_batches: int = 20
    num_workers: int = 20
    train_prefetch_factor: int = 12
    val_prefetch_factor: int = 6
    holdout_stride: int = 10
    camera_source: str = "gt"
    mask_source: str = "mask"
    min_depth_conf: float = 0.0
    freeze_aggregator: bool = True
    enable_compile: bool = True
    emit_dataset_probe: bool = True
    emit_dataset_probe_contract_diff: bool = True
    probe_sample_index: int = 0
    probe_aggregate_samples: int = 4
    probe_aggregate_stride: int = 50
    probe_seed: int = 42
    probe_reference_config: str = ""
    probe_label_current: str = "current_config"
    probe_label_reference: str = "reference_config"
    extra_overrides: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(blob: str) -> "ZjuFinetuneConfig":
        return ZjuFinetuneConfig(**_decode_config_blob(blob))


@dataclass
class ZjuAblationPairConfig:
    zju_subdir: str
    seq_names: str = "CoreView_390"
    geom_subdir: str = "vggt_geom"
    checkpoint_subpath: str = "checkpoints/model.pt"
    baseline_config: str = "zju_vggt_geom_minimal"
    candidate_config: str = "zju_vggt_geom_unproject_minimal"
    exp_prefix: str = "zju_geom_modal_pair"
    output_subdir_base: str = ""
    num_images: int = 4
    max_img_per_gpu: int = 16
    accum_steps: int = 1
    max_epochs: int = 1
    learning_rate: float = 5e-5
    limit_train_batches: int = 100
    limit_val_batches: int = 20
    num_workers: int = 20
    train_prefetch_factor: int = 12
    val_prefetch_factor: int = 6
    holdout_stride: int = 10
    camera_source: str = "gt"
    mask_source: str = "mask"
    min_depth_conf: float = 0.0
    baseline_min_depth_conf: float | None = None
    candidate_min_depth_conf: float | None = None
    freeze_aggregator: bool = True
    enable_compile: bool = True
    extra_overrides: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(blob: str) -> "ZjuAblationPairConfig":
        return ZjuAblationPairConfig(**_decode_config_blob(blob))


@dataclass
class ZjuResidualAuditConfig:
    zju_subdir: str
    seq_names: str = "CoreView_390"
    geom_subdir: str = "vggt_geom"
    checkpoint_subpath: str = "checkpoints/model.pt"
    config: str = "zju_vggt_geom_minimal"
    output_subdir: str = ""
    label: str = "promoted_lead"
    split: str = "train"
    num_samples: int = 2048
    num_images: int = 4
    device: str = "cuda"
    reference_checkpoint_subpath: str = ""
    reference_label: str = "reference"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(blob: str) -> "ZjuResidualAuditConfig":
        return ZjuResidualAuditConfig(**_decode_config_blob(blob))


def _normalize_subpath(value: str) -> str:
    cleaned = (value or "").strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("Expected a non-empty volume-relative path.")
    return cleaned


def _remote_data_path(subpath: str) -> Path:
    return Path(str(REMOTE_DATA_DIR / _normalize_subpath(subpath)))


def _remote_output_path(subpath: str) -> Path:
    return Path(str(REMOTE_OUTPUT_DIR / _normalize_subpath(subpath)))


def _resolve_output_root(exp_name: str, output_subdir: str) -> Path:
    if output_subdir.strip():
        return Path(str(REMOTE_OUTPUT_DIR / _normalize_subpath(output_subdir)))
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    safe_exp = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in exp_name).strip("_")
    safe_exp = safe_exp or "zju_geometry_minimal_modal"
    return Path(str(REMOTE_OUTPUT_DIR / "zju_geometry_minimal" / f"{run_tag}_{safe_exp}"))


def _resolve_pair_output_root(exp_prefix: str, output_subdir_base: str) -> Path:
    if output_subdir_base.strip():
        return Path(str(REMOTE_OUTPUT_DIR / _normalize_subpath(output_subdir_base)))
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    safe_prefix = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in exp_prefix).strip("_")
    safe_prefix = safe_prefix or "zju_geom_modal_pair"
    return Path(str(REMOTE_OUTPUT_DIR / "geometry_pairs" / f"{run_tag}_{safe_prefix}"))


def _resolve_residual_audit_output_root(output_subdir: str) -> Path:
    if output_subdir.strip():
        return Path(str(REMOTE_OUTPUT_DIR / _normalize_subpath(output_subdir)))
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    return Path(str(REMOTE_OUTPUT_DIR / "zju_source_policy_research_loop" / "promoted_residual_regen_cloud" / run_tag))


def _build_overrides(
    cfg: ZjuFinetuneConfig,
    output_root: Path,
    resolved_checkpoint_path: Path | None = None,
) -> list[str]:
    zju_root = _remote_data_path(cfg.zju_subdir)
    ckpt_path = resolved_checkpoint_path or _remote_data_path(cfg.checkpoint_subpath)
    logging_dir = output_root / "logs"
    checkpoint_dir = output_root / "ckpts"

    overrides = [
        f"exp_name={cfg.exp_name}",
        f"logging.log_dir={logging_dir.as_posix()}",
        f"checkpoint.save_dir={checkpoint_dir.as_posix()}",
        f"checkpoint.resume_checkpoint_path={ckpt_path.as_posix()}",
        f"zju_dir={zju_root.as_posix()}",
        f"zju_seq_names={cfg.seq_names}",
        f"zju_geom_subdir={cfg.geom_subdir}",
        f"zju_camera_source={cfg.camera_source}",
        f"zju_mask_source={cfg.mask_source}",
        f"zju_min_depth_conf={float(cfg.min_depth_conf)}",
        f"zju_holdout_stride={int(cfg.holdout_stride)}",
        f"data.train.common_config.fix_img_num={int(cfg.num_images)}",
        f"data.val.common_config.fix_img_num={int(cfg.num_images)}",
        "data.train.common_config.fix_aspect_ratio=1.0",
        "data.val.common_config.fix_aspect_ratio=1.0",
        "data.train.common_config.allow_duplicate_img=False",
        "data.val.common_config.allow_duplicate_img=False",
        "data.train.common_config.load_depth=True",
        "data.val.common_config.load_depth=True",
        f"data.train.num_workers={int(cfg.num_workers)}",
        f"data.val.num_workers={int(cfg.num_workers)}",
        f"num_workers={int(cfg.num_workers)}",
        f"max_img_per_gpu={int(cfg.max_img_per_gpu)}",
        f"accum_steps={int(cfg.accum_steps)}",
        f"max_epochs={int(cfg.max_epochs)}",
        f"optim.optimizer.lr={float(cfg.learning_rate)}",
        f"limit_train_batches={int(cfg.limit_train_batches)}",
        f"limit_val_batches={int(cfg.limit_val_batches)}",
        "data.train.pin_memory=True",
        "data.val.pin_memory=True",
        "+data.train.pin_memory_device=cuda",
        "+data.val.pin_memory_device=cuda",
        "cuda.allow_tf32=True",
        "cuda.cudnn_benchmark=True",
        "model.enable_camera=True",
        "model.enable_depth=True",
        "model.enable_point=False",
        "model.enable_track=False",
        "loss.point=null",
        "loss.track=null",
    ]

    if bool(getattr(cfg, "enable_compile", False)):
        overrides.extend(
            [
                "++optim.compile.enabled=True",
                "++optim.compile.backend=inductor",
                "++optim.compile.mode=max-autotune",
                "++optim.compile.dynamic=False",
                "++optim.compile.fullgraph=False",
                "++optim.compile.suppress_errors=True",
            ]
        )
    else:
        overrides.append("++optim.compile.enabled=False")

    if cfg.num_workers > 0:
        overrides.extend(
            [
                "+data.train.persistent_workers=True",
                "+data.val.persistent_workers=True",
                f"+data.train.prefetch_factor={int(cfg.train_prefetch_factor)}",
                f"+data.val.prefetch_factor={int(cfg.val_prefetch_factor)}",
            ]
        )

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
    print(f"[modal-zju-geometry] upload checkpoint: {checkpoint_path} -> {DATA_VOLUME_NAME}:{remote_path}")
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(checkpoint_path), remote_path)
    return remote_path


def _materialize_checkpoint_from_parts(parts: list[Path], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as writer:
        for part in parts:
            with part.open("rb") as reader:
                shutil.copyfileobj(reader, writer, length=64 * 1024 * 1024)
    data_volume.commit()
    return destination


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    output_volume.commit()


def _resolve_checkpoint_path(remote_subpath: str) -> Path:
    target = _remote_data_path(remote_subpath)
    if target.exists():
        return target

    candidate_exact = [_remote_output_path(remote_subpath)] + [
        _remote_output_path(path) for path in OUTPUT_CHECKPOINT_FALLBACKS
    ]
    for candidate in candidate_exact:
        if candidate.exists():
            print(f"[modal-zju-geometry] use checkpoint directly from output volume: {candidate}", flush=True)
            return candidate

    candidate_parts = candidate_exact
    for candidate in candidate_parts:
        parts = sorted(candidate.parent.glob(candidate.name + ".part*"))
        if parts:
            print(
                f"[modal-zju-geometry] assemble split checkpoint from output volume: "
                f"{candidate} -> {target} ({len(parts)} parts)",
                flush=True,
            )
            return _materialize_checkpoint_from_parts(parts, target)

    raise FileNotFoundError(
        "Could not resolve checkpoint from data volume or output volume fallbacks.\n"
        f"- requested data path: {target}\n"
        + "\n".join(f"- tried output fallback: {path}" for path in candidate_exact)
    )


def _normalize_config_stem(config_ref: str) -> str:
    raw = str(config_ref or "").strip().replace("\\", "/")
    if not raw:
        return ""
    stem = raw.split("/")[-1]
    if stem.endswith(".yaml"):
        stem = stem[: -len(".yaml")]
    return stem


def _default_probe_reference_bundle(config_ref: str) -> tuple[str, str, str]:
    stem = _normalize_config_stem(config_ref)
    if stem == PROMOTED_SOURCE_POLICY_CONFIG:
        return (
            PREVIOUS_SOURCE_POLICY_CONFIG,
            "promoted_nearest_plus_uniform_tail",
            "previous_nearest_ring",
        )
    return ("", "current_config", "reference_config")


def _stream_subprocess_with_live_mirror(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    mirror_log_path: Path,
    commit_callback=None,
    commit_interval_sec: int = 30,
) -> None:
    mirror_log_path.parent.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    commit_thread = None

    if commit_callback is not None and commit_interval_sec > 0:
        def _commit_worker():
            while not stop_event.wait(commit_interval_sec):
                try:
                    commit_callback()
                except Exception as exc:
                    print(f"[modal-zju-geometry] live commit warning: {exc}", flush=True)

        commit_thread = threading.Thread(target=_commit_worker, daemon=True)
        commit_thread.start()

    return_code = None
    with mirror_log_path.open("w", encoding="utf-8", errors="replace") as mirror_handle:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                mirror_handle.write(line)
                mirror_handle.flush()
                print(line, end="", flush=True)
        finally:
            return_code = process.wait()
            stop_event.set()
            if commit_thread is not None:
                commit_thread.join(timeout=max(commit_interval_sec, 1))
            if commit_callback is not None:
                try:
                    commit_callback()
                except Exception as exc:
                    print(f"[modal-zju-geometry] final commit warning: {exc}", flush=True)

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def _compile_failure_looks_retryable(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace").lower()
    retryable_markers = (
        "misaligned address",
        "xid",
        "torch.compile",
        "inductor",
        "cuda error",
    )
    return any(marker in text for marker in retryable_markers)


def _run_training_subprocess(
    cfg: ZjuFinetuneConfig,
    *,
    resolved_checkpoint_path: Path,
    output_root: Path | None = None,
) -> Path:
    zju_root = _remote_data_path(cfg.zju_subdir)
    missing = [str(path) for path in (zju_root, resolved_checkpoint_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required Modal volume paths are missing:\n- " + "\n- ".join(missing)
        )

    output_root = output_root or _resolve_output_root(cfg.exp_name, cfg.output_subdir)
    output_root.mkdir(parents=True, exist_ok=True)
    overrides = _build_overrides(cfg, output_root, resolved_checkpoint_path)
    remote_code_dir = Path(str(REMOTE_CODE_DIR))
    launch_path = remote_code_dir / "training" / "launch.py"
    cmd = [sys.executable, str(launch_path), "--config", cfg.config, *overrides]

    print("[modal-zju-geometry] data_volume =", DATA_VOLUME_NAME, flush=True)
    print("[modal-zju-geometry] output_volume =", OUTPUT_VOLUME_NAME, flush=True)
    print("[modal-zju-geometry] gpu =", GPU_SPEC, flush=True)
    print("[modal-zju-geometry] output_root =", output_root.as_posix(), flush=True)
    print("[modal-zju-geometry] resume_ckpt =", resolved_checkpoint_path.as_posix(), flush=True)
    print("[modal-zju-geometry] command =", shlex.join(cmd), flush=True)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = str(remote_code_dir)
    env["PYTHONPATH"] = (
        repo_pythonpath if not existing_pythonpath else repo_pythonpath + os.pathsep + existing_pythonpath
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    if bool(getattr(cfg, "emit_dataset_probe", False)):
        probe_path = remote_code_dir / "scripts" / "probe_zju_vggt_geom_dataset.py"
        compare_probe_path = remote_code_dir / "scripts" / "compare_zju_vggt_geom_probes.py"
        if probe_path.exists():
            default_reference_config, default_label_current, default_label_reference = _default_probe_reference_bundle(
                cfg.config
            )
            probe_reference_config = str(
                getattr(cfg, "probe_reference_config", "") or default_reference_config
            ).strip()
            probe_label_current = str(
                getattr(cfg, "probe_label_current", "") or default_label_current
            ).strip()
            probe_label_reference = str(
                getattr(cfg, "probe_label_reference", "") or default_label_reference
            ).strip()

            def _build_probe_cmd(config_ref: str, output_dir: Path) -> list[str]:
                return [
                    sys.executable,
                    str(probe_path),
                    "--config",
                    str(config_ref),
                    "--zju_dir",
                    str(zju_root),
                    "--output_dir",
                    str(output_dir),
                    "--seed",
                    str(int(getattr(cfg, "probe_seed", 42))),
                    "--sample_index",
                    str(int(getattr(cfg, "probe_sample_index", 0))),
                    "--aggregate_samples",
                    str(int(getattr(cfg, "probe_aggregate_samples", 4))),
                    "--aggregate_stride",
                    str(int(getattr(cfg, "probe_aggregate_stride", 50))),
                    "--num_images",
                    str(int(cfg.num_images)),
                ]

            probe_output_dir = output_root / "dataset_probe"
            probe_log_path = output_root / "dataset_probe_live.log"
            probe_cmd = _build_probe_cmd(cfg.config, probe_output_dir)
            print("[modal-zju-geometry] dataset_probe =", shlex.join(probe_cmd), flush=True)
            _stream_subprocess_with_live_mirror(
                cmd=probe_cmd,
                cwd=remote_code_dir,
                env=env,
                mirror_log_path=probe_log_path,
                commit_callback=output_volume.commit,
                commit_interval_sec=LIVE_COMMIT_INTERVAL_SEC,
            )
            output_volume.commit()

            if (
                bool(getattr(cfg, "emit_dataset_probe_contract_diff", True))
                and probe_reference_config
                and compare_probe_path.exists()
            ):
                reference_probe_output_dir = output_root / "dataset_probe_reference"
                reference_probe_log_path = output_root / "dataset_probe_reference_live.log"
                reference_probe_cmd = _build_probe_cmd(probe_reference_config, reference_probe_output_dir)
                print(
                    "[modal-zju-geometry] dataset_probe_reference =",
                    shlex.join(reference_probe_cmd),
                    flush=True,
                )
                _stream_subprocess_with_live_mirror(
                    cmd=reference_probe_cmd,
                    cwd=remote_code_dir,
                    env=env,
                    mirror_log_path=reference_probe_log_path,
                    commit_callback=output_volume.commit,
                    commit_interval_sec=LIVE_COMMIT_INTERVAL_SEC,
                )
                output_volume.commit()

                diff_json_path = output_root / "dataset_probe_contract_diff.json"
                diff_md_path = output_root / "dataset_probe_contract_diff.md"
                diff_log_path = output_root / "dataset_probe_contract_diff_live.log"
                diff_cmd = [
                    sys.executable,
                    str(compare_probe_path),
                    "--probe-a",
                    str(probe_output_dir),
                    "--probe-b",
                    str(reference_probe_output_dir),
                    "--label-a",
                    probe_label_current,
                    "--label-b",
                    probe_label_reference,
                    "--output-json",
                    str(diff_json_path),
                    "--output-md",
                    str(diff_md_path),
                ]
                print("[modal-zju-geometry] dataset_probe_contract_diff =", shlex.join(diff_cmd), flush=True)
                _stream_subprocess_with_live_mirror(
                    cmd=diff_cmd,
                    cwd=remote_code_dir,
                    env=env,
                    mirror_log_path=diff_log_path,
                    commit_callback=output_volume.commit,
                    commit_interval_sec=LIVE_COMMIT_INTERVAL_SEC,
                )
                output_volume.commit()
        else:
            print("[modal-zju-geometry] dataset probe script missing; skipping probe export.", flush=True)

    live_log_path = output_root / ("driver_live_compile.log" if bool(getattr(cfg, "enable_compile", False)) else "driver_live.log")
    try:
        _stream_subprocess_with_live_mirror(
            cmd=cmd,
            cwd=remote_code_dir,
            env=env,
            mirror_log_path=live_log_path,
            commit_callback=output_volume.commit,
            commit_interval_sec=LIVE_COMMIT_INTERVAL_SEC,
        )
    except subprocess.CalledProcessError:
        if bool(getattr(cfg, "enable_compile", False)) and _compile_failure_looks_retryable(live_log_path):
            fallback_cfg = replace(cfg, enable_compile=False)
            fallback_overrides = _build_overrides(fallback_cfg, output_root, resolved_checkpoint_path)
            fallback_cmd = [sys.executable, str(launch_path), "--config", fallback_cfg.config, *fallback_overrides]
            fallback_status = {
                "compile_attempted": True,
                "compile_fallback_triggered": True,
                "reason": "retryable_compile_failure_detected",
                "compile_log": live_log_path.as_posix(),
                "fallback_log": (output_root / "driver_live.log").as_posix(),
            }
            _write_json(output_root / "compile_fallback_status.json", fallback_status)
            print("[modal-zju-geometry] compile path failed; retrying eagerly on the same A100 run", flush=True)
            _stream_subprocess_with_live_mirror(
                cmd=fallback_cmd,
                cwd=remote_code_dir,
                env=env,
                mirror_log_path=output_root / "driver_live.log",
                commit_callback=output_volume.commit,
                commit_interval_sec=LIVE_COMMIT_INTERVAL_SEC,
            )
        else:
            raise

    output_volume.commit()
    print("[modal-zju-geometry] committed output volume", flush=True)
    return output_root


def _run_residual_audit_subprocess(
    cfg: ZjuResidualAuditConfig,
    *,
    resolved_checkpoint_path: Path,
) -> Path:
    zju_root = _remote_data_path(cfg.zju_subdir)
    missing = [str(path) for path in (zju_root, resolved_checkpoint_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required Modal volume paths are missing:\n- " + "\n- ".join(missing)
        )

    output_root = _resolve_residual_audit_output_root(cfg.output_subdir)
    output_root.mkdir(parents=True, exist_ok=True)
    remote_code_dir = Path(str(REMOTE_CODE_DIR))
    audit_path = remote_code_dir / "scripts" / "audit_zju_conf_depth_attribution.py"
    if not audit_path.exists():
        raise FileNotFoundError(f"Residual audit script missing on Modal image: {audit_path}")

    cmd = [
        sys.executable,
        str(audit_path),
        "--config",
        cfg.config,
        "--checkpoint",
        resolved_checkpoint_path.as_posix(),
        "--label",
        cfg.label,
        "--split",
        cfg.split,
        "--num-samples",
        str(int(cfg.num_samples)),
        "--num-images",
        str(int(cfg.num_images)),
        "--zju-dir",
        zju_root.as_posix(),
        "--device",
        cfg.device,
        "--output-dir",
        output_root.as_posix(),
    ]
    if cfg.reference_checkpoint_subpath.strip():
        reference_checkpoint_path = _resolve_checkpoint_path(cfg.reference_checkpoint_subpath)
        cmd.extend(
            [
                "--reference-checkpoint",
                reference_checkpoint_path.as_posix(),
                "--reference-label",
                cfg.reference_label,
            ]
        )

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = str(remote_code_dir)
    env["PYTHONPATH"] = (
        repo_pythonpath if not existing_pythonpath else repo_pythonpath + os.pathsep + existing_pythonpath
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    print("[modal-zju-geometry] data_volume =", DATA_VOLUME_NAME, flush=True)
    print("[modal-zju-geometry] output_volume =", OUTPUT_VOLUME_NAME, flush=True)
    print("[modal-zju-geometry] gpu =", GPU_SPEC, flush=True)
    print("[modal-zju-geometry] residual_audit_output_root =", output_root.as_posix(), flush=True)
    print("[modal-zju-geometry] residual_audit_checkpoint =", resolved_checkpoint_path.as_posix(), flush=True)
    print("[modal-zju-geometry] residual_audit_command =", shlex.join(cmd), flush=True)

    live_log_path = output_root / "residual_audit_live.log"
    _stream_subprocess_with_live_mirror(
        cmd=cmd,
        cwd=remote_code_dir,
        env=env,
        mirror_log_path=live_log_path,
        commit_callback=output_volume.commit,
        commit_interval_sec=LIVE_COMMIT_INTERVAL_SEC,
    )

    artifact_status = {
        "config": cfg.config,
        "label": cfg.label,
        "split": cfg.split,
        "num_samples_requested": int(cfg.num_samples),
        "num_images": int(cfg.num_images),
        "checkpoint": resolved_checkpoint_path.as_posix(),
        "output_root": output_root.as_posix(),
        "summary_json": (output_root / "summary.json").as_posix(),
        "summary_md": (output_root / "summary.md").as_posix(),
        "primary_frame_rows_jsonl": (output_root / "primary_frame_rows.jsonl").as_posix(),
        "residual_audit_live_log": live_log_path.as_posix(),
        "state": "completed",
    }
    _write_json(output_root / "residual_audit_status.json", artifact_status)
    output_volume.commit()
    print("[modal-zju-geometry] committed residual audit output volume", flush=True)
    return output_root


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
def run_remote_zju_geometry_finetune(cfg_json: str) -> None:
    cfg = ZjuFinetuneConfig.from_json(cfg_json)
    ckpt_path = _resolve_checkpoint_path(cfg.checkpoint_subpath)
    _run_training_subprocess(cfg, resolved_checkpoint_path=ckpt_path)


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
def run_remote_zju_geometry_ablation_pair(cfg_json: str) -> None:
    cfg = ZjuAblationPairConfig.from_json(cfg_json)
    ckpt_path = _resolve_checkpoint_path(cfg.checkpoint_subpath)
    pair_output_root = _resolve_pair_output_root(cfg.exp_prefix, cfg.output_subdir_base)
    pair_output_root.mkdir(parents=True, exist_ok=True)
    status_path = pair_output_root / "pair_status.json"
    status = {
        "exp_prefix": cfg.exp_prefix,
        "pair_output_root": pair_output_root.as_posix(),
        "resume_checkpoint": ckpt_path.as_posix(),
        "state": "running",
        "stages": {},
    }
    _write_json(status_path, status)

    baseline_cfg = ZjuFinetuneConfig(
        zju_subdir=cfg.zju_subdir,
        seq_names=cfg.seq_names,
        geom_subdir=cfg.geom_subdir,
        checkpoint_subpath=cfg.checkpoint_subpath,
        config=cfg.baseline_config,
        exp_name=f"{cfg.exp_prefix}_baseline",
        num_images=cfg.num_images,
        max_img_per_gpu=cfg.max_img_per_gpu,
        accum_steps=cfg.accum_steps,
        max_epochs=cfg.max_epochs,
        learning_rate=cfg.learning_rate,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        num_workers=cfg.num_workers,
        train_prefetch_factor=cfg.train_prefetch_factor,
        val_prefetch_factor=cfg.val_prefetch_factor,
        holdout_stride=cfg.holdout_stride,
        camera_source=cfg.camera_source,
        mask_source=cfg.mask_source,
        min_depth_conf=cfg.min_depth_conf if cfg.baseline_min_depth_conf is None else cfg.baseline_min_depth_conf,
        freeze_aggregator=cfg.freeze_aggregator,
        extra_overrides=cfg.extra_overrides,
    )
    candidate_cfg = ZjuFinetuneConfig(
        zju_subdir=cfg.zju_subdir,
        seq_names=cfg.seq_names,
        geom_subdir=cfg.geom_subdir,
        checkpoint_subpath=cfg.checkpoint_subpath,
        config=cfg.candidate_config,
        exp_name=f"{cfg.exp_prefix}_unproject",
        num_images=cfg.num_images,
        max_img_per_gpu=cfg.max_img_per_gpu,
        accum_steps=cfg.accum_steps,
        max_epochs=cfg.max_epochs,
        learning_rate=cfg.learning_rate,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        num_workers=cfg.num_workers,
        train_prefetch_factor=cfg.train_prefetch_factor,
        val_prefetch_factor=cfg.val_prefetch_factor,
        holdout_stride=cfg.holdout_stride,
        camera_source=cfg.camera_source,
        mask_source=cfg.mask_source,
        min_depth_conf=cfg.min_depth_conf if cfg.candidate_min_depth_conf is None else cfg.candidate_min_depth_conf,
        freeze_aggregator=cfg.freeze_aggregator,
        extra_overrides=cfg.extra_overrides,
    )

    stage_specs = (
        ("baseline", baseline_cfg),
        ("unproject", candidate_cfg),
    )
    try:
        for stage_name, stage_cfg in stage_specs:
            status["stages"][stage_name] = {
                "state": "running",
                "config": stage_cfg.config,
                "exp_name": stage_cfg.exp_name,
                "output_root": (pair_output_root / stage_name).as_posix(),
                "driver_live_log": (pair_output_root / stage_name / "driver_live.log").as_posix(),
            }
            _write_json(status_path, status)
            _run_training_subprocess(
                stage_cfg,
                resolved_checkpoint_path=ckpt_path,
                output_root=pair_output_root / stage_name,
            )
            status["stages"][stage_name]["state"] = "completed"
            _write_json(status_path, status)
    except Exception as exc:
        status["state"] = "failed"
        status["error"] = str(exc)
        _write_json(status_path, status)
        raise

    status["state"] = "completed"
    _write_json(status_path, status)


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
def run_remote_zju_geometry_residual_audit(cfg_json: str) -> str:
    cfg = ZjuResidualAuditConfig.from_json(cfg_json)
    ckpt_path = _resolve_checkpoint_path(cfg.checkpoint_subpath)
    output_root = _run_residual_audit_subprocess(cfg, resolved_checkpoint_path=ckpt_path)
    return output_root.as_posix()


@app.local_entrypoint()
def upload_checkpoint(
    local_path: str,
    remote_subpath: str = "checkpoints/model.pt",
) -> None:
    remote_path = _upload_checkpoint(local_path, remote_subpath)
    print(f"[modal-zju-geometry] checkpoint uploaded to {DATA_VOLUME_NAME}:{remote_path}")


@app.local_entrypoint()
def run_zju_geometry_finetune(
    zju_subdir: str,
    seq_names: str = "CoreView_390",
    geom_subdir: str = "vggt_geom",
    checkpoint_subpath: str = "checkpoints/model.pt",
    local_checkpoint: str = "",
    config: str = "zju_vggt_geom_minimal",
    exp_name: str = "zju_geometry_minimal_modal",
    output_subdir: str = "",
    num_images: int = 4,
    max_img_per_gpu: int = 16,
    accum_steps: int = 1,
    max_epochs: int = 1,
    learning_rate: float = 5e-5,
    limit_train_batches: int = 100,
    limit_val_batches: int = 20,
    num_workers: int = 20,
    train_prefetch_factor: int = 12,
    val_prefetch_factor: int = 6,
    holdout_stride: int = 10,
    camera_source: str = "gt",
    mask_source: str = "mask",
    min_depth_conf: float = 0.0,
    freeze_aggregator: bool = True,
    enable_compile: bool = True,
    extra_overrides: str = "",
) -> None:
    resolved_checkpoint_subpath = checkpoint_subpath
    if local_checkpoint.strip():
        resolved_checkpoint_subpath = _upload_checkpoint(local_checkpoint, checkpoint_subpath)

    cfg = ZjuFinetuneConfig(
        zju_subdir=zju_subdir,
        seq_names=seq_names,
        geom_subdir=geom_subdir,
        checkpoint_subpath=resolved_checkpoint_subpath,
        config=config,
        exp_name=exp_name,
        output_subdir=output_subdir,
        num_images=num_images,
        max_img_per_gpu=max_img_per_gpu,
        accum_steps=accum_steps,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        num_workers=num_workers,
        train_prefetch_factor=train_prefetch_factor,
        val_prefetch_factor=val_prefetch_factor,
        holdout_stride=holdout_stride,
        camera_source=camera_source,
        mask_source=mask_source,
        min_depth_conf=min_depth_conf,
        freeze_aggregator=freeze_aggregator,
        enable_compile=enable_compile,
        extra_overrides=extra_overrides,
    )

    print("[modal-zju-geometry] launch config:")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    run_remote_zju_geometry_finetune.remote(cfg.to_json())


@app.local_entrypoint()
def run_zju_geometry_ablation_pair(
    zju_subdir: str,
    seq_names: str = "CoreView_390",
    geom_subdir: str = "vggt_geom",
    checkpoint_subpath: str = "checkpoints/model.pt",
    local_checkpoint: str = "",
    baseline_config: str = "zju_vggt_geom_minimal",
    candidate_config: str = "zju_vggt_geom_unproject_minimal",
    exp_prefix: str = "zju_geom_modal_pair",
    output_subdir_base: str = "",
    num_images: int = 4,
    max_img_per_gpu: int = 16,
    accum_steps: int = 1,
    max_epochs: int = 1,
    learning_rate: float = 5e-5,
    limit_train_batches: int = 100,
    limit_val_batches: int = 20,
    num_workers: int = 20,
    train_prefetch_factor: int = 12,
    val_prefetch_factor: int = 6,
    holdout_stride: int = 10,
    camera_source: str = "gt",
    mask_source: str = "mask",
    min_depth_conf: float = 0.0,
    freeze_aggregator: bool = True,
    enable_compile: bool = True,
    extra_overrides: str = "",
) -> None:
    resolved_checkpoint_subpath = checkpoint_subpath
    if local_checkpoint.strip():
        resolved_checkpoint_subpath = _upload_checkpoint(local_checkpoint, checkpoint_subpath)

    cfg = ZjuAblationPairConfig(
        zju_subdir=zju_subdir,
        seq_names=seq_names,
        geom_subdir=geom_subdir,
        checkpoint_subpath=resolved_checkpoint_subpath,
        baseline_config=baseline_config,
        candidate_config=candidate_config,
        exp_prefix=exp_prefix,
        output_subdir_base=output_subdir_base,
        num_images=num_images,
        max_img_per_gpu=max_img_per_gpu,
        accum_steps=accum_steps,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        num_workers=num_workers,
        train_prefetch_factor=train_prefetch_factor,
        val_prefetch_factor=val_prefetch_factor,
        holdout_stride=holdout_stride,
        camera_source=camera_source,
        mask_source=mask_source,
        min_depth_conf=min_depth_conf,
        freeze_aggregator=freeze_aggregator,
        enable_compile=enable_compile,
        extra_overrides=extra_overrides,
    )

    print("[modal-zju-geometry] launch pair config:")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    run_remote_zju_geometry_ablation_pair.remote(cfg.to_json())


@app.local_entrypoint()
def run_zju_geometry_residual_audit(
    zju_subdir: str,
    seq_names: str = "CoreView_390",
    geom_subdir: str = "vggt_geom",
    checkpoint_subpath: str = "checkpoints/model.pt",
    local_checkpoint: str = "",
    config: str = "zju_vggt_geom_minimal",
    output_subdir: str = "",
    label: str = "promoted_lead",
    split: str = "train",
    num_samples: int = 2048,
    num_images: int = 4,
    device: str = "cuda",
    reference_checkpoint_subpath: str = "",
    reference_label: str = "reference",
) -> None:
    resolved_checkpoint_subpath = checkpoint_subpath
    if local_checkpoint.strip():
        resolved_checkpoint_subpath = _upload_checkpoint(local_checkpoint, checkpoint_subpath)

    cfg = ZjuResidualAuditConfig(
        zju_subdir=zju_subdir,
        seq_names=seq_names,
        geom_subdir=geom_subdir,
        checkpoint_subpath=resolved_checkpoint_subpath,
        config=config,
        output_subdir=output_subdir,
        label=label,
        split=split,
        num_samples=num_samples,
        num_images=num_images,
        device=device,
        reference_checkpoint_subpath=reference_checkpoint_subpath,
        reference_label=reference_label,
    )

    print("[modal-zju-geometry] residual audit launch config:")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    output_root = run_remote_zju_geometry_residual_audit.remote(cfg.to_json())
    print(f"[modal-zju-geometry] residual audit output_root={output_root}")
