from __future__ import annotations

import argparse
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

APP_NAME = os.environ.get("VGGT_MODAL_V42_APP_NAME", "vggt-v42-prior-enabled-predictions")
DATA_VOLUME_NAME = os.environ.get("VGGT_MODAL_DATA_VOLUME", "vggt-4k4d-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
GPU_SPEC = os.environ.get("VGGT_MODAL_V42_GPU", os.environ.get("VGGT_MODAL_GPU", "A100-40GB"))
CPU_COUNT = float(os.environ.get("VGGT_MODAL_V42_CPU", "8"))
MEMORY_MB = int(os.environ.get("VGGT_MODAL_V42_MEMORY_MB", str(96 * 1024)))
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_V42_TIMEOUT_SEC", str(6 * 60 * 60)))

V42_REMOTE_ROOT = "surface_research_cloud_preflight/V42_prior_enabled_predictions"
V42_LOCAL_ROOT = REPO_ROOT / "output/surface_research_cloud_preflight/V42_prior_enabled_predictions"
V42_LOCAL_PREFLIGHT_ROOT = REPO_ROOT / "output/surface_research_preflight_local/V42_prior_enabled_predictions"

REQUIRED_RESEARCH_FILES = (
    "research_depths.npz",
    "research_points_world.npz",
    "research_confidence.npz",
    "research_normals_geometric.npz",
    "research_prior_effect.json",
    "control_real_zero_shuffle_random_dropout.json",
)
FORBIDDEN_FORMAL_TOKENS = (
    "strict_pass",
    "teacher_export",
    "candidate_export",
    "formal_candidate",
    "strict_gate_registry",
    "candidate_package",
    "teacher_package",
    "registry_refresh",
)
CONTROLS = ("real", "zero", "shuffle", "random-region", "prior-dropout")


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
    return ["torch==2.3.1", "torchvision==0.18.1", "numpy==1.26.1", "Pillow", "huggingface_hub", "einops", "safetensors"]


def _normalize_subpath(value: str, *, require_v42_root: bool) -> str:
    cleaned = (value or "").strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("Expected a non-empty volume-relative path.")
    if ".." in Path(cleaned).parts:
        raise ValueError(f"Parent traversal is forbidden: {value!r}")
    lower = cleaned.lower()
    if require_v42_root and not lower.startswith(V42_REMOTE_ROOT.lower()):
        raise ValueError(f"V42 output must stay under {V42_REMOTE_ROOT}: {value!r}")
    if any(token in lower for token in FORBIDDEN_FORMAL_TOKENS):
        raise ValueError(f"V42 path contains a formal-output token: {value!r}")
    if Path(cleaned).name.lower() == "predictions.npz":
        raise ValueError("V42 must not write or address predictions.npz.")
    return cleaned


def _remote_data_path(subpath: str) -> Path:
    return Path(str(REMOTE_DATA_DIR / _normalize_subpath(subpath, require_v42_root=False)))


def _remote_output_path(subpath: str) -> Path:
    return Path(str(REMOTE_OUTPUT_DIR / _normalize_subpath(subpath, require_v42_root=True)))


def _json_ready(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
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
    return {"path": str(path), "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else 0}


def _upload_dir(local_dir: Path, remote_subdir: str) -> str:
    local_dir = local_dir.expanduser().resolve()
    if not local_dir.is_dir():
        raise NotADirectoryError(f"V42 local directory not found: {local_dir}")
    remote_subdir = _normalize_subpath(remote_subdir, require_v42_root=False)
    print(f"[v42] upload dir: {local_dir} -> {DATA_VOLUME_NAME}:{remote_subdir}", flush=True)
    with data_volume.batch_upload(force=True) as batch:
        for path in local_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.name.lower() == "predictions.npz":
                raise ValueError(f"Refusing to upload formal prediction file: {path}")
            rel = path.relative_to(local_dir).as_posix()
            batch.put_file(str(path), f"{remote_subdir}/{rel}")
    return remote_subdir


def _upload_file(local_file: Path, remote_subdir: str) -> str:
    local_file = local_file.expanduser().resolve()
    if not local_file.is_file():
        raise FileNotFoundError(local_file)
    if local_file.name.lower() == "predictions.npz":
        raise ValueError(f"Refusing to upload formal prediction file: {local_file}")
    remote_subdir = _normalize_subpath(remote_subdir, require_v42_root=False)
    print(f"[v42] upload file: {local_file} -> {DATA_VOLUME_NAME}:{remote_subdir}/{local_file.name}", flush=True)
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_file), f"{remote_subdir}/{local_file.name}")
    return f"{remote_subdir}/{local_file.name}"


def _download_volume_dir(remote_subdir: str, local_dir: Path) -> None:
    remote_subdir = _normalize_subpath(remote_subdir, require_v42_root=True)
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
            raise RuntimeError(f"Safety stop: remote V42 output contains predictions.npz at {entry.path}")
        dest_path = local_dir / rel_path
        if entry.type == modal.volume.FileEntryType.DIRECTORY:
            dest_path.mkdir(parents=True, exist_ok=True)
            continue
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with dest_path.open("wb") as handle:
            output_volume.read_file_into_fileobj(entry.path, handle)
        downloaded += 1
    print(f"[v42] downloaded {downloaded} files from {remote_subdir} to {local_dir}", flush=True)


CODE_SYNC_IGNORE = [".git", ".git/**", "__pycache__", "__pycache__/**", ".venv*", ".venv*/**", "output", "output/**", "reports", "reports/**"]

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
class V42Config:
    scene_subdirs: dict[str, str]
    checkpoint_relpath: str
    output_subdir: str = V42_REMOTE_ROOT
    image_mode: str = "pad"
    target_size: int = 518
    controls: tuple[str, ...] = CONTROLS

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(blob: str) -> "V42Config":
        payload = json.loads(blob)
        if isinstance(payload.get("controls"), list):
            payload["controls"] = tuple(payload["controls"])
        return V42Config(**payload)


def _extract_model_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        if "model" in payload and isinstance(payload["model"], dict):
            return payload["model"]
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        return payload
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")


def _to_numpy(tensor: Any):
    return tensor.detach().float().cpu().numpy()


def _array_stats(array: Any) -> dict[str, Any]:
    import numpy as np

    arr = np.asarray(array)
    finite = np.isfinite(arr)
    stats: dict[str, Any] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "finite_count": int(finite.sum()), "total_count": int(arr.size)}
    if finite.any():
        vals = arr[finite]
        stats.update({"min": float(np.min(vals)), "max": float(np.max(vals)), "mean": float(np.mean(vals)), "p50": float(np.percentile(vals, 50))})
    return stats


def _make_control_prior(prior_maps, prior_summary, control: str):
    import torch

    maps = prior_maps
    summary = prior_summary
    if control == "real":
        return maps, summary
    if control == "zero":
        return torch.zeros_like(maps), torch.zeros_like(summary) if summary is not None else None
    if control == "shuffle":
        return torch.roll(maps, shifts=1, dims=0), torch.roll(summary, shifts=1, dims=0) if summary is not None else None
    if control == "random-region":
        return torch.roll(maps, shifts=max(1, maps.shape[1] // 3), dims=1), summary
    if control == "prior-dropout":
        out = maps.clone()
        out[:, ::2] = 0
        if summary is not None:
            summary = summary.clone()
            summary[:, :, ::2] = 0
        return out, summary
    raise ValueError(control)


@app.function(
    image=IMAGE,
    gpu=GPU_SPEC,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SEC,
    volumes={REMOTE_DATA_DIR.as_posix(): data_volume, REMOTE_OUTPUT_DIR.as_posix(): output_volume},
)
def run_v42_remote(cfg_json: str) -> dict[str, Any]:
    import numpy as np
    import torch

    sys.path.insert(0, str(REMOTE_CODE_DIR))
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    cfg = V42Config.from_json(cfg_json)
    output_root = _remote_output_path(cfg.output_subdir)
    output_root.mkdir(parents=True, exist_ok=True)
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
        "output_subdir": _normalize_subpath(cfg.output_subdir, require_v42_root=True),
        "required_research_files": list(REQUIRED_RESEARCH_FILES),
    }
    _write_json(output_root / "v42_research_guard.json", guard)

    checkpoint_path = _remote_data_path(cfg.checkpoint_relpath)
    if not checkpoint_path.is_file():
        summary = {
            "task": "v42_prior_enabled_predictions_remote",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE",
            **guard,
            "blockers": [f"checkpoint missing on data volume: {checkpoint_path}"],
        }
        _write_json(output_root / "research_prior_effect.json", summary)
        _write_json(output_root / "research_summary.json", summary)
        output_volume.commit()
        return summary

    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_model_state_dict(payload)
    model_kwargs = payload.get("model_kwargs") if isinstance(payload, dict) else None
    if not isinstance(model_kwargs, dict):
        summary = {
            "task": "v42_prior_enabled_predictions_remote",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE",
            **guard,
            "blockers": ["checkpoint missing explicit model_kwargs"],
        }
        _write_json(output_root / "research_prior_effect.json", summary)
        _write_json(output_root / "research_summary.json", summary)
        output_volume.commit()
        return summary

    model = VGGT(**model_kwargs)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    device = "cuda"
    model = model.to(device).eval()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model_prior_channels = int(getattr(model.aggregator, "human_prior_channels", 0) or 0)
    model_prior_summary_channels = int(getattr(model.aggregator, "human_prior_summary_channels", 0) or 0)
    normal_head_available = bool(getattr(model, "normal_head", None) is not None)

    depth_payload: dict[str, Any] = {}
    points_payload: dict[str, Any] = {}
    normals_payload: dict[str, Any] = {"normal_available": np.asarray(normal_head_available)}
    confidence_payload: dict[str, Any] = {"normal_available": np.asarray(normal_head_available)}
    frame_summaries: dict[str, Any] = {}
    control_audit: dict[str, Any] = {"controls": {}, "frames": {}}
    blockers: list[str] = []

    for frame_key in sorted(cfg.scene_subdirs):
        scene_dir = _remote_data_path(cfg.scene_subdirs[frame_key])
        image_dir = scene_dir / "images"
        image_paths = sorted(path for path in image_dir.iterdir() if path.is_file()) if image_dir.is_dir() else []
        if len(image_paths) < 12:
            blockers.append(f"{frame_key}: expected at least 12 images, found {len(image_paths)}")
            continue
        images = load_and_preprocess_images([str(path) for path in image_paths], mode=cfg.image_mode, target_size=int(cfg.target_size)).to(device)
        prior_maps_path = scene_dir / "prior_maps.npz"
        if not prior_maps_path.is_file():
            blockers.append(f"{frame_key}: prior_maps.npz missing")
            continue
        with np.load(prior_maps_path, allow_pickle=False) as prior_payload:
            prior_maps_np = np.asarray(prior_payload["prior_maps"], dtype=np.float32)
            prior_maps = torch.from_numpy(prior_maps_np).to(device=device, dtype=torch.float32)
            prior_summary = None
            if "prior_summary_tokens" in prior_payload.files and model_prior_summary_channels > 0:
                prior_summary = torch.from_numpy(np.asarray(prior_payload["prior_summary_tokens"], dtype=np.float32)).to(device=device, dtype=torch.float32)
        if prior_maps.shape[1] != model_prior_channels:
            blockers.append(f"{frame_key}: prior channel mismatch {prior_maps.shape[1]} vs {model_prior_channels}")
            continue

        frame_control_stats: dict[str, Any] = {}
        real_outputs: dict[str, Any] | None = None
        real_signature = None
        for control in cfg.controls:
            c_maps, c_summary = _make_control_prior(prior_maps, prior_summary, control)
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    pred = model(images, prior_maps=c_maps, prior_summary_tokens=c_summary)
            depth = _to_numpy(pred["depth"].squeeze(0))
            depth_conf = _to_numpy(pred["depth_conf"].squeeze(0))
            world_points = _to_numpy(pred["world_points"].squeeze(0))
            world_points_conf = _to_numpy(pred["world_points_conf"].squeeze(0))
            pose_enc = pred["pose_enc"]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            normal = _to_numpy(pred["normal"].squeeze(0)) if "normal" in pred else np.zeros((*world_points.shape[:-1], 3), dtype=np.float32)
            normal_conf = _to_numpy(pred["normal_conf"].squeeze(0)) if "normal_conf" in pred else np.zeros(world_points.shape[:-1], dtype=np.float32)
            signature = np.concatenate([depth.reshape(-1)[:2048], world_points.reshape(-1)[:2048], normal.reshape(-1)[:2048]]).astype(np.float32)
            if control == "real":
                real_signature = signature
                real_outputs = {
                    "depth": depth,
                    "depth_conf": depth_conf,
                    "world_points": world_points,
                    "world_points_conf": world_points_conf,
                    "normal": normal,
                    "normal_conf": normal_conf,
                    "pose_enc": _to_numpy(pose_enc.squeeze(0)),
                    "extrinsic": _to_numpy(extrinsic.squeeze(0)),
                    "intrinsic": _to_numpy(intrinsic.squeeze(0)),
                }
            delta_l2 = None
            delta_max = None
            if real_signature is not None:
                diff = signature - real_signature
                delta_l2 = float(np.linalg.norm(diff))
                delta_max = float(np.max(np.abs(diff))) if diff.size else 0.0
            frame_control_stats[control] = {
                "depth": _array_stats(depth),
                "world_points": _array_stats(world_points),
                "normal": _array_stats(normal),
                "delta_from_real_l2": delta_l2,
                "delta_from_real_maxabs": delta_max,
            }
            del pred, c_maps, c_summary
            torch.cuda.empty_cache()

        if real_outputs is None:
            blockers.append(f"{frame_key}: real control did not run")
            continue
        depth_payload[frame_key] = real_outputs["depth"]
        points_payload[frame_key] = real_outputs["world_points"]
        points_payload[f"{frame_key}_pose_enc"] = real_outputs["pose_enc"]
        points_payload[f"{frame_key}_extrinsic"] = real_outputs["extrinsic"]
        points_payload[f"{frame_key}_intrinsic"] = real_outputs["intrinsic"]
        normals_payload[frame_key] = real_outputs["normal"]
        confidence_payload[f"{frame_key}_depth_conf"] = real_outputs["depth_conf"]
        confidence_payload[f"{frame_key}_world_points_conf"] = real_outputs["world_points_conf"]
        confidence_payload[f"{frame_key}_normal_conf"] = real_outputs["normal_conf"]
        frame_summaries[frame_key] = {
            "scene_dir": str(scene_dir),
            "image_count": len(image_paths),
            "input_tensor_shape": list(images.shape),
            "prior_used": True,
            "prior_tensor_shape": list(prior_maps.shape),
            "prior_summary_tensor_shape": list(prior_summary.shape) if prior_summary is not None else None,
            "depth": _array_stats(real_outputs["depth"]),
            "world_points": _array_stats(real_outputs["world_points"]),
            "normal": _array_stats(real_outputs["normal"]),
        }
        control_audit["frames"][frame_key] = frame_control_stats
        del images, prior_maps, prior_summary
        torch.cuda.empty_cache()

    if blockers:
        summary = {
            "task": "v42_prior_enabled_predictions_remote",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "DONE_FAIL_ROUTED",
            **guard,
            "blockers": blockers,
            "frame_summaries": frame_summaries,
            "decision": "V42 did not write partial research arrays because one or more requested frames failed.",
        }
        _write_json(output_root / "research_prior_effect.json", summary)
        _write_json(output_root / "research_summary.json", summary)
        output_volume.commit()
        return summary

    frame_keys = sorted(frame_summaries)
    for payload_dict in (depth_payload, points_payload, normals_payload, confidence_payload):
        payload_dict["frame_keys"] = np.asarray(frame_keys)
    np.savez_compressed(output_root / "research_depths.npz", **depth_payload)
    np.savez_compressed(output_root / "research_points_world.npz", **points_payload)
    np.savez_compressed(output_root / "research_normals_geometric.npz", **normals_payload)
    np.savez_compressed(output_root / "research_confidence.npz", **confidence_payload)

    real_beats_controls = True
    for frame_stats in control_audit["frames"].values():
        for control, row in frame_stats.items():
            if control == "real":
                continue
            if float(row.get("delta_from_real_l2") or 0.0) <= 1e-8:
                real_beats_controls = False
    control_audit.update(
        {
            "status": "DONE_PASS" if real_beats_controls else "DONE_FAIL_ROUTED",
            "real_differs_from_all_controls": bool(real_beats_controls),
            "note": "This audit measures prior-conditioned prediction deltas, not strict visual pass.",
        }
    )
    _write_json(output_root / "control_real_zero_shuffle_random_dropout.json", control_audit)

    files = {name: _file_info(output_root / name) for name in REQUIRED_RESEARCH_FILES}
    summary = {
        "task": "v42_prior_enabled_predictions_remote",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - started, 3),
        "status": "DONE_PASS" if real_beats_controls else "DONE_FAIL_ROUTED",
        **guard,
        "checkpoint_relpath": cfg.checkpoint_relpath,
        "model_kwargs": model_kwargs,
        "model_prior_channels": model_prior_channels,
        "model_prior_summary_channels": model_prior_summary_channels,
        "normal_head_available": normal_head_available,
        "strict_load_missing_keys": list(map(str, missing)),
        "strict_load_unexpected_keys": list(map(str, unexpected)),
        "frame_summaries": frame_summaries,
        "control_audit": control_audit,
        "files": files,
        "blockers": [] if real_beats_controls else ["prior-enabled checkpoint produced identical outputs for at least one control"],
        "decision": "V42 wrote research-only prior-enabled depth, world-point, normal, confidence, and control audit arrays.",
    }
    _write_json(output_root / "research_prior_effect.json", summary)
    _write_json(output_root / "research_summary.json", summary)
    output_volume.commit()
    return summary


@app.local_entrypoint()
def run(
    frames: str = "0,1,2",
    scene_root: str = "output/4k4d_scenes",
    checkpoint: str = "output/surface_research_preflight_local/V41b_real_prior_enabled_checkpoint/checkpoint.pt",
    remote_output_subdir: str = V42_REMOTE_ROOT,
    image_mode: str = "pad",
    target_size: int = 518,
) -> None:
    frame_ids = [int(item.strip()) for item in frames.split(",") if item.strip()]
    if frame_ids != [0, 1, 2]:
        raise ValueError("V42 is scoped to exactly frames 0,1,2.")
    output_subdir = _normalize_subpath(remote_output_subdir, require_v42_root=True)
    checkpoint_relpath = _upload_file(REPO_ROOT / checkpoint, "v42_input/prior_enabled_checkpoint")
    scene_root_path = (REPO_ROOT / scene_root).resolve() if not Path(scene_root).is_absolute() else Path(scene_root).resolve()
    scene_subdirs: dict[str, str] = {}
    for frame in frame_ids:
        frame_key = f"frame{frame:04d}"
        local_scene = scene_root_path / f"0012_11_frame{frame:04d}_12views_tmf"
        scene_subdirs[frame_key] = _upload_dir(local_scene, f"v42_input/scenes/{local_scene.name}")
    cfg = V42Config(scene_subdirs=scene_subdirs, checkpoint_relpath=checkpoint_relpath, output_subdir=output_subdir, image_mode=image_mode, target_size=int(target_size))
    print("[v42] launch config:")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    summary = run_v42_remote.remote(cfg.to_json())
    print("[v42] remote summary:")
    print(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False))
    _download_volume_dir(output_subdir, V42_LOCAL_ROOT)
    V42_LOCAL_PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(
        V42_LOCAL_PREFLIGHT_ROOT / "modal_download_manifest.json",
        {
            "task": "v42_modal_download_manifest",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "remote_output_subdir": output_subdir,
            "download_local_dir": str(V42_LOCAL_ROOT),
            "summary_status": summary.get("status"),
            "research_only": True,
            "no_predictions_write": True,
            "no_registry_write": True,
            "no_teacher_export": True,
            "no_candidate_export": True,
            "no_strict_pass_write": True,
        },
    )
    print(f"[v42] downloaded research artifacts to {V42_LOCAL_ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="Use `modal run modal_v42_prior_enabled_predictions.py`.")
    args = parser.parse_args()
    print(args.note)
