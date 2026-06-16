from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import modal


REPO_ROOT = Path(__file__).resolve().parent
REMOTE_CODE_DIR = PurePosixPath("/workspace/vggt")
REMOTE_DATA_DIR = PurePosixPath("/mnt/data")
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")

APP_NAME = os.environ.get("VGGT_MODAL_V25_RESEARCH_APP_NAME", "vggt-v25-research-predictions")
DATA_VOLUME_NAME = os.environ.get("VGGT_MODAL_DATA_VOLUME", "vggt-4k4d-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
GPU_SPEC = os.environ.get("VGGT_MODAL_V25_GPU", os.environ.get("VGGT_MODAL_GPU", "A100-40GB"))
CPU_COUNT = float(os.environ.get("VGGT_MODAL_V25_CPU", "8"))
MEMORY_MB = int(os.environ.get("VGGT_MODAL_V25_MEMORY_MB", str(80 * 1024)))
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_V25_TIMEOUT_SEC", str(6 * 60 * 60)))

V25_REMOTE_ROOT = "surface_research_cloud_preflight/V25_research_vggt_predictions"
V25_LOCAL_CLOUD_ROOT = REPO_ROOT / "output" / V25_REMOTE_ROOT
V25_LOCAL_PREFLIGHT_ROOT = REPO_ROOT / "output/surface_research_preflight_local/V25_research_vggt_predictions"

REQUIRED_RESEARCH_FILES = (
    "research_depths.npz",
    "research_points_world.npz",
    "research_normals.npz",
    "research_confidence.npz",
    "research_summary.json",
)
FORBIDDEN_FORMAL_TOKENS = (
    "strict_pass",
    "teacher_export",
    "candidate_export",
    "formal_candidate",
    "strict_gate_registry",
    "candidate_gate",
)


def _load_requirements(path: Path) -> list[str]:
    packages: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line not in seen:
            seen.add(line)
            packages.append(line)
    return packages


def _resolve_requirements() -> list[str]:
    candidate = REPO_ROOT / "requirements.txt"
    if candidate.is_file():
        return _load_requirements(candidate)
    return [
        "torch==2.3.1",
        "torchvision==0.18.1",
        "numpy==1.26.1",
        "Pillow",
        "huggingface_hub",
        "einops",
        "safetensors",
    ]


def _normalize_subpath(value: str, *, require_v25_root: bool) -> str:
    cleaned = (value or "").strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("Expected a non-empty volume-relative path.")
    if ".." in Path(cleaned).parts:
        raise ValueError(f"Parent traversal is forbidden: {value!r}")
    lower = cleaned.lower()
    if require_v25_root and not lower.startswith(V25_REMOTE_ROOT.lower()):
        raise ValueError(f"V25 output must stay under {V25_REMOTE_ROOT}: {value!r}")
    if any(token in lower for token in FORBIDDEN_FORMAL_TOKENS):
        raise ValueError(f"V25 path contains a formal-output token: {value!r}")
    if Path(cleaned).name.lower() == "predictions.npz":
        raise ValueError("V25 must not write or address predictions.npz.")
    return cleaned


def _remote_data_path(subpath: str) -> Path:
    return Path(str(REMOTE_DATA_DIR / _normalize_subpath(subpath, require_v25_root=False)))


def _remote_output_path(subpath: str) -> Path:
    return Path(str(REMOTE_OUTPUT_DIR / _normalize_subpath(subpath, require_v25_root=True)))


def _json_ready(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - only used before remote deps are present.
        np = None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if np is not None and isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else 0,
    }


def _write_markdown(path: Path, title: str, summary: dict[str, Any]) -> None:
    lines = [
        f"# {title}",
        "",
        f"Status: `{summary.get('status')}`",
        "",
        "Research-only. No formal package, registry, strict pass, teacher export, candidate export, or predictions.npz is written.",
        "",
        "## Decision",
        "",
        str(summary.get("decision", "")),
        "",
        "## Files",
        "",
    ]
    for name in REQUIRED_RESEARCH_FILES:
        info = summary.get("files", {}).get(name, {})
        lines.append(f"- `{name}`: exists={str(info.get('exists')).lower()} size={info.get('size', 0)}")
    blockers = summary.get("blockers") or []
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- {item}" for item in blockers] if blockers else ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _upload_dir(local_dir: Path, remote_subdir: str) -> str:
    local_dir = local_dir.expanduser().resolve()
    if not local_dir.is_dir():
        raise NotADirectoryError(f"V25 scene directory not found: {local_dir}")
    remote_subdir = _normalize_subpath(remote_subdir, require_v25_root=False)
    print(f"[v25-research] upload scene: {local_dir} -> {DATA_VOLUME_NAME}:{remote_subdir}", flush=True)
    with data_volume.batch_upload(force=True) as batch:
        for path in local_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.name.lower() == "predictions.npz":
                raise ValueError(f"Refusing to upload formal prediction file from scene: {path}")
            rel = path.relative_to(local_dir).as_posix()
            batch.put_file(str(path), f"{remote_subdir}/{rel}")
    return remote_subdir


def _download_volume_dir(remote_subdir: str, local_dir: Path) -> None:
    remote_subdir = _normalize_subpath(remote_subdir, require_v25_root=True)
    local_dir = local_dir.expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_prefix = Path(remote_subdir)
    downloaded = 0
    for entry in output_volume.listdir(remote_subdir, recursive=True):
        rel_path = Path(entry.path)
        try:
            rel_path = rel_path.relative_to(remote_prefix)
        except ValueError:
            pass
        if rel_path.name.lower() == "predictions.npz":
            raise RuntimeError(f"Safety stop: remote V25 output contains predictions.npz at {entry.path}")
        dest_path = local_dir / rel_path
        if entry.type == modal.volume.FileEntryType.DIRECTORY:
            dest_path.mkdir(parents=True, exist_ok=True)
            continue
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        last_error = None
        for attempt in range(1, 6):
            tmp_path = dest_path.with_suffix(dest_path.suffix + f".download{attempt}.tmp")
            try:
                with tmp_path.open("wb") as handle:
                    output_volume.read_file_into_fileobj(entry.path, handle)
                tmp_path.replace(dest_path)
                downloaded += 1
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                time.sleep(min(20, 2 * attempt))
        if last_error is not None:
            raise RuntimeError(f"Failed to download {entry.path} after retries") from last_error
    print(f"[v25-research] downloaded {downloaded} files from {remote_subdir} to {local_dir}", flush=True)


CODE_SYNC_IGNORE = [
    ".git",
    ".git/**",
    "__pycache__",
    "__pycache__/**",
    ".venv*",
    ".venv*/**",
    "output",
    "output/**",
    "reports",
    "reports/**",
]

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(*_resolve_requirements())
    .add_local_dir(str(REPO_ROOT / "vggt"), remote_path=(REMOTE_CODE_DIR / "vggt").as_posix(), ignore=CODE_SYNC_IGNORE)
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


@dataclass
class V25Config:
    scene_subdirs: dict[str, str]
    output_subdir: str = V25_REMOTE_ROOT
    image_mode: str = "pad"
    target_size: int = 518
    hf_repo: str = "facebook/VGGT-1B"
    checkpoint_relpath: str = ""
    prior_policy: str = "auto"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(blob: str) -> "V25Config":
        return V25Config(**json.loads(blob))


def _extract_model_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        if "model" in payload and isinstance(payload["model"], dict):
            return payload["model"]
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        return payload
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")


def _infer_model_kwargs_from_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    camera_token = state_dict.get("aggregator.camera_token")
    embed_dim = int(camera_token.shape[-1]) if camera_token is not None else 1024
    proj0 = state_dict.get("aggregator.human_prior_adapter.proj.0.weight")
    summary_proj0 = state_dict.get("aggregator.human_prior_adapter.summary_proj.0.weight")
    gate = state_dict.get("aggregator.human_prior_adapter.input_fusion.gate")
    scale_factors = state_dict.get("aggregator.human_prior_adapter.scale_factors_tensor")
    if scale_factors is not None:
        scale_factors = [int(value) for value in scale_factors.tolist()]
    else:
        scale_factors = [1]
    return {
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": embed_dim,
        "enable_camera": any(key.startswith("camera_head.") for key in state_dict),
        "enable_point": any(key.startswith("point_head.") for key in state_dict),
        "enable_depth": any(key.startswith("depth_head.") for key in state_dict),
        "enable_normal": any(key.startswith("normal_head.") for key in state_dict),
        "enable_track": any(key.startswith("track_head.") for key in state_dict),
        "human_prior_channels": int(proj0.shape[1]) if proj0 is not None else 0,
        "human_prior_summary_channels": int(summary_proj0.shape[1]) if summary_proj0 is not None else 0,
        "human_prior_hidden_dim": int(proj0.shape[0]) if proj0 is not None else 64,
        "human_prior_gate_init": float(gate.item()) if gate is not None else 0.0,
        "human_prior_multi_scale_factors": scale_factors,
    }


def _to_numpy(tensor: Any):
    return tensor.detach().float().cpu().numpy()


def _np_scalar_text(value: str):
    import numpy as np

    return np.asarray(value)


def _array_stats(array: Any) -> dict[str, Any]:
    import numpy as np

    arr = np.asarray(array)
    finite = np.isfinite(arr)
    stats: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite_count": int(finite.sum()),
        "total_count": int(arr.size),
    }
    if finite.any():
        vals = arr[finite]
        stats.update(
            {
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "mean": float(np.mean(vals)),
                "p05": float(np.percentile(vals, 5)),
                "p50": float(np.percentile(vals, 50)),
                "p95": float(np.percentile(vals, 95)),
            }
        )
    return stats


def _write_hard_impossible(output_root: Path, status: str, evidence: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "task": "v25_research_vggt_predictions_3frames",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "hard_impossible": True,
        "hard_impossible_evidence": evidence,
        "research_only": True,
        "no_export": True,
        "no_predictions_write": True,
        "no_registry_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_strict_pass_write": True,
        "formal_cloud_unblocked": False,
        "decision": "V25 could not produce research arrays because the requested VGGT model/checkpoint was unavailable or incompatible; no formal artifact was written.",
        "blockers": [str(evidence.get("reason", "model/checkpoint unavailable"))],
    }
    _write_json(output_root / "research_summary.json", summary)
    _write_markdown(output_root / "research_report.md", "V25 Research VGGT Predictions", summary)
    output_volume.commit()
    return summary


@app.function(
    image=IMAGE,
    gpu=GPU_SPEC,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SEC,
    volumes={
        REMOTE_DATA_DIR.as_posix(): data_volume,
        REMOTE_OUTPUT_DIR.as_posix(): output_volume,
    },
)
def run_v25_remote(cfg_json: str) -> dict[str, Any]:
    cfg = V25Config.from_json(cfg_json)
    output_root = _remote_output_path(cfg.output_subdir)
    output_root.mkdir(parents=True, exist_ok=True)

    remote_code_dir = Path(str(REMOTE_CODE_DIR))
    if str(remote_code_dir) not in sys.path:
        sys.path.insert(0, str(remote_code_dir))

    import numpy as np
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    started = time.time()
    guard = {
        "research_only": True,
        "no_export": True,
        "no_predictions_write": True,
        "no_registry_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_strict_pass_write": True,
        "formal_cloud_unblocked": False,
        "formal_package_registry_pass": "not_requested_not_written",
        "output_subdir": _normalize_subpath(cfg.output_subdir, require_v25_root=True),
        "required_research_files": list(REQUIRED_RESEARCH_FILES),
    }
    _write_json(output_root / "v25_research_guard.json", guard)

    device = "cuda"
    try:
        if cfg.checkpoint_relpath.strip():
            checkpoint_path = _remote_output_path(cfg.checkpoint_relpath)
            if not checkpoint_path.is_file():
                return _write_hard_impossible(
                    output_root,
                    "hard_impossible_missing_checkpoint",
                    {
                        "reason": "checkpoint_relpath was provided but the checkpoint file is missing on the Modal output volume",
                        "checkpoint_relpath": cfg.checkpoint_relpath,
                        "checkpoint_path": str(checkpoint_path),
                    },
                )
            payload = torch.load(checkpoint_path, map_location="cpu")
            state_dict = _extract_model_state_dict(payload)
            model_kwargs = payload.get("model_kwargs") if isinstance(payload, dict) else None
            if not isinstance(model_kwargs, dict):
                model_kwargs = _infer_model_kwargs_from_state_dict(state_dict)
            model = VGGT(**model_kwargs)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                return _write_hard_impossible(
                    output_root,
                    "hard_impossible_checkpoint_load_mismatch",
                    {
                        "reason": "checkpoint state dict did not match the local VGGT model",
                        "checkpoint_relpath": cfg.checkpoint_relpath,
                        "missing": list(map(str, missing)),
                        "unexpected": list(map(str, unexpected)),
                    },
                )
            load_summary: dict[str, Any] = {
                "source": "checkpoint",
                "checkpoint_relpath": cfg.checkpoint_relpath,
                "model_kwargs": model_kwargs,
            }
        else:
            model = VGGT.from_pretrained(cfg.hf_repo)
            load_summary = {"source": "huggingface", "hf_repo": cfg.hf_repo}
    except Exception as exc:  # noqa: BLE001
        return _write_hard_impossible(
            output_root,
            "hard_impossible_model_load_failed",
            {
                "reason": "VGGT model load failed",
                "hf_repo": cfg.hf_repo,
                "checkpoint_relpath": cfg.checkpoint_relpath,
                "exception_type": type(exc).__name__,
                "exception": repr(exc),
            },
        )

    model = model.to(device)
    model.eval()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model_prior_channels = int(getattr(model.aggregator, "human_prior_channels", 0) or 0)
    model_prior_summary_channels = int(getattr(model.aggregator, "human_prior_summary_channels", 0) or 0)
    normal_head_available = bool(getattr(model, "normal_head", None) is not None)

    depth_payload: dict[str, Any] = {}
    points_payload: dict[str, Any] = {}
    normals_payload: dict[str, Any] = {
        "normal_available": np.asarray(normal_head_available),
        "normal_reason": _np_scalar_text("" if normal_head_available else "Loaded VGGT model has no normal_head; no normals were fabricated."),
    }
    confidence_payload: dict[str, Any] = {
        "normal_available": np.asarray(normal_head_available),
    }
    frame_summaries: dict[str, Any] = {}
    blockers: list[str] = []

    for frame_key in sorted(cfg.scene_subdirs):
        scene_subdir = cfg.scene_subdirs[frame_key]
        scene_dir = _remote_data_path(scene_subdir)
        image_dir = scene_dir / "images"
        if not image_dir.is_dir():
            blockers.append(f"{frame_key}: remote image directory missing: {image_dir}")
            continue
        image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
        if len(image_paths) < 12:
            blockers.append(f"{frame_key}: expected at least 12 TMF images, found {len(image_paths)}")
            continue

        images = load_and_preprocess_images(
            [str(path) for path in image_paths],
            mode=cfg.image_mode,
            target_size=int(cfg.target_size),
        ).to(device)

        prior_maps = None
        prior_summary_tokens = None
        prior_maps_path = scene_dir / "prior_maps.npz"
        prior_maps_present = prior_maps_path.is_file()
        prior_used = False
        prior_reason = "prior_maps.npz not present"
        if prior_maps_present:
            if cfg.prior_policy == "disable":
                prior_reason = "prior_policy=disable"
            elif model_prior_channels <= 0:
                if cfg.prior_policy == "require":
                    return _write_hard_impossible(
                        output_root,
                        "hard_impossible_prior_checkpoint_missing",
                        {
                            "reason": "TMF scene has prior_maps.npz but the loaded VGGT model has no human prior adapter",
                            "frame": frame_key,
                            "scene_subdir": scene_subdir,
                            "prior_policy": cfg.prior_policy,
                            "model_prior_channels": model_prior_channels,
                            "hf_repo": cfg.hf_repo,
                            "checkpoint_relpath": cfg.checkpoint_relpath,
                        },
                    )
                prior_reason = "prior_maps present but loaded model has human_prior_channels=0; ran base VGGT research inference without prior injection"
            else:
                with np.load(prior_maps_path, allow_pickle=False) as prior_payload:
                    prior_maps_np = np.array(prior_payload["prior_maps"])
                    prior_maps = torch.from_numpy(prior_maps_np).to(device=device, dtype=torch.float32)
                    if tuple(prior_maps.shape[-2:]) != tuple(images.shape[-2:]):
                        if prior_maps.ndim == 4:
                            prior_maps = torch.nn.functional.interpolate(
                                prior_maps,
                                size=tuple(images.shape[-2:]),
                                mode="bilinear",
                                align_corners=False,
                            )
                        elif prior_maps.ndim == 5:
                            batch_size, seq_len, channels, _, _ = prior_maps.shape
                            prior_maps = torch.nn.functional.interpolate(
                                prior_maps.reshape(
                                    batch_size * seq_len,
                                    channels,
                                    prior_maps.shape[-2],
                                    prior_maps.shape[-1],
                                ),
                                size=tuple(images.shape[-2:]),
                                mode="bilinear",
                                align_corners=False,
                            ).reshape(batch_size, seq_len, channels, images.shape[-2], images.shape[-1])
                        else:
                            raise ValueError(f"Expected 4D/5D prior_maps, got {prior_maps.shape}")
                    if "prior_summary_tokens" in prior_payload.files and model_prior_summary_channels > 0:
                        prior_summary_tokens = torch.from_numpy(np.array(prior_payload["prior_summary_tokens"])).to(
                            device=device,
                            dtype=torch.float32,
                        )
                    prior_used = True
                    prior_reason = "prior maps injected"

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(
                    images,
                    prior_maps=prior_maps,
                    prior_summary_tokens=prior_summary_tokens,
                )

        pose_enc = predictions["pose_enc"]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        depth = _to_numpy(predictions["depth"].squeeze(0))
        depth_conf = _to_numpy(predictions["depth_conf"].squeeze(0))
        world_points = _to_numpy(predictions["world_points"].squeeze(0))
        world_points_conf = _to_numpy(predictions["world_points_conf"].squeeze(0))
        pose_enc_np = _to_numpy(pose_enc.squeeze(0))
        extrinsic_np = _to_numpy(extrinsic.squeeze(0))
        intrinsic_np = _to_numpy(intrinsic.squeeze(0))

        depth_payload[frame_key] = depth
        points_payload[frame_key] = world_points
        points_payload[f"{frame_key}_pose_enc"] = pose_enc_np
        points_payload[f"{frame_key}_extrinsic"] = extrinsic_np
        points_payload[f"{frame_key}_intrinsic"] = intrinsic_np
        confidence_payload[f"{frame_key}_depth_conf"] = depth_conf
        confidence_payload[f"{frame_key}_world_points_conf"] = world_points_conf
        if "normal" in predictions:
            normal = _to_numpy(predictions["normal"].squeeze(0))
            normal_conf = _to_numpy(predictions["normal_conf"].squeeze(0))
            normals_payload[frame_key] = normal
            confidence_payload[f"{frame_key}_normal_conf"] = normal_conf

        frame_summaries[frame_key] = {
            "scene_subdir": scene_subdir,
            "scene_dir": str(scene_dir),
            "image_count": len(image_paths),
            "image_names": [path.name for path in image_paths],
            "input_tensor_shape": list(images.shape),
            "prior_maps_present": bool(prior_maps_present),
            "prior_used": bool(prior_used),
            "prior_reason": prior_reason,
            "prior_tensor_shape": list(prior_maps.shape) if prior_maps is not None else None,
            "prior_summary_tensor_shape": list(prior_summary_tokens.shape) if prior_summary_tokens is not None else None,
            "depth": _array_stats(depth),
            "depth_conf": _array_stats(depth_conf),
            "world_points": _array_stats(world_points),
            "world_points_conf": _array_stats(world_points_conf),
            "normal_available": bool("normal" in predictions),
        }
        if "normal" in predictions:
            frame_summaries[frame_key]["normal"] = _array_stats(normals_payload[frame_key])
            frame_summaries[frame_key]["normal_conf"] = _array_stats(confidence_payload[f"{frame_key}_normal_conf"])

        del images, predictions, pose_enc, extrinsic, intrinsic
        torch.cuda.empty_cache()

    if blockers:
        summary = {
            "task": "v25_research_vggt_predictions_3frames",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "blocked_scene_inputs_incomplete",
            "hard_impossible": False,
            **guard,
            "blockers": blockers,
            "frame_summaries": frame_summaries,
            "decision": "V25 did not write partial research arrays because one or more requested TMF scenes were incomplete.",
        }
        _write_json(output_root / "research_summary.json", summary)
        _write_markdown(output_root / "research_report.md", "V25 Research VGGT Predictions", summary)
        output_volume.commit()
        return summary

    frame_keys = sorted(frame_summaries)
    depth_payload["frame_keys"] = np.asarray(frame_keys)
    points_payload["frame_keys"] = np.asarray(frame_keys)
    normals_payload["frame_keys"] = np.asarray(frame_keys)
    confidence_payload["frame_keys"] = np.asarray(frame_keys)

    np.savez_compressed(output_root / "research_depths.npz", **depth_payload)
    np.savez_compressed(output_root / "research_points_world.npz", **points_payload)
    np.savez_compressed(output_root / "research_normals.npz", **normals_payload)
    np.savez_compressed(output_root / "research_confidence.npz", **confidence_payload)

    files = {name: _file_info(output_root / name) for name in REQUIRED_RESEARCH_FILES}
    summary = {
        "task": "v25_research_vggt_predictions_3frames",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - started, 3),
        "status": "completed_research_only",
        "hard_impossible": False,
        **guard,
        "scene_subdirs": cfg.scene_subdirs,
        "image_mode": cfg.image_mode,
        "target_size": int(cfg.target_size),
        "device": device,
        "dtype": str(dtype),
        "gpu_name": torch.cuda.get_device_name(0),
        "load_summary": load_summary,
        "model_prior_channels": model_prior_channels,
        "model_prior_summary_channels": model_prior_summary_channels,
        "normal_head_available": normal_head_available,
        "frame_summaries": frame_summaries,
        "files": files,
        "blockers": [],
        "decision": "V25 wrote research-only VGGT depth, world-point, normal-capability, and confidence arrays for frame0000/frame0001/frame0002 without writing predictions.npz or any formal package/registry/pass.",
    }
    _write_json(output_root / "research_summary.json", summary)
    summary["files"] = {name: _file_info(output_root / name) for name in REQUIRED_RESEARCH_FILES}
    _write_json(output_root / "research_summary.json", summary)
    _write_markdown(output_root / "research_report.md", "V25 Research VGGT Predictions", summary)
    output_volume.commit()
    return summary


@app.local_entrypoint()
def run(
    frames: str = "0,1,2",
    scene_root: str = "output/4k4d_scenes",
    remote_output_subdir: str = V25_REMOTE_ROOT,
    image_mode: str = "pad",
    target_size: int = 518,
    hf_repo: str = "facebook/VGGT-1B",
    checkpoint_relpath: str = "",
    prior_policy: str = "auto",
    download_local_dir: str = "",
) -> None:
    frame_ids = [int(item.strip()) for item in frames.split(",") if item.strip()]
    if frame_ids != [0, 1, 2]:
        raise ValueError("V25 is scoped to exactly frames 0,1,2.")
    output_subdir = _normalize_subpath(remote_output_subdir, require_v25_root=True)
    scene_root_path = (REPO_ROOT / scene_root).resolve() if not Path(scene_root).is_absolute() else Path(scene_root).resolve()
    scene_subdirs: dict[str, str] = {}
    for frame in frame_ids:
        frame_key = f"frame{frame:04d}"
        local_scene = scene_root_path / f"0012_11_frame{frame:04d}_12views_tmf"
        remote_scene_subdir = f"{V25_REMOTE_ROOT}/input_scenes/{local_scene.name}"
        scene_subdirs[frame_key] = _upload_dir(local_scene, remote_scene_subdir)

    cfg = V25Config(
        scene_subdirs=scene_subdirs,
        output_subdir=output_subdir,
        image_mode=image_mode,
        target_size=int(target_size),
        hf_repo=hf_repo,
        checkpoint_relpath=checkpoint_relpath,
        prior_policy=prior_policy,
    )
    print("[v25-research] launch config:")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    summary = run_v25_remote.remote(cfg.to_json())
    print("[v25-research] remote summary:")
    print(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False))

    local_dir = Path(download_local_dir).expanduser().resolve() if download_local_dir.strip() else V25_LOCAL_CLOUD_ROOT
    _download_volume_dir(output_subdir, local_dir)
    V25_LOCAL_PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    local_manifest = {
        "task": "v25_research_modal_download_manifest",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "research_only": True,
        "remote_output_subdir": output_subdir,
        "download_local_dir": str(local_dir),
        "local_preflight_dir": str(V25_LOCAL_PREFLIGHT_ROOT),
        "summary_status": summary.get("status"),
        "no_predictions_write": True,
        "no_registry_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_strict_pass_write": True,
    }
    _write_json(V25_LOCAL_PREFLIGHT_ROOT / "modal_download_manifest.json", local_manifest)
    print(f"[v25-research] downloaded research artifacts to {local_dir}")
