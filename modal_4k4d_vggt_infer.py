from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import modal


REPO_ROOT = Path(__file__).resolve().parent
REMOTE_CODE_DIR = PurePosixPath("/workspace/vggt")
REMOTE_DATA_DIR = PurePosixPath("/mnt/data")
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")
DEFAULT_MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


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


DEFAULT_REQUIREMENTS = [
    "torch==2.3.1",
    "torchvision==0.18.1",
    "numpy==1.26.1",
    "Pillow",
    "huggingface_hub",
    "einops",
    "safetensors",
]


def _resolve_requirements() -> list[str]:
    candidate = REPO_ROOT / "requirements.txt"
    if candidate.exists():
        return _load_requirements(candidate)
    return list(DEFAULT_REQUIREMENTS)


APP_NAME = os.environ.get("VGGT_MODAL_APP_NAME", "vggt-4k4d-infer")
DATA_VOLUME_NAME = os.environ.get("VGGT_MODAL_DATA_VOLUME", "vggt-4k4d-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
GPU_SPEC = os.environ.get("VGGT_MODAL_GPU", "A100-40GB")
CPU_COUNT = float(os.environ.get("VGGT_MODAL_CPU", "8"))
MEMORY_MB = int(os.environ.get("VGGT_MODAL_MEMORY_MB", "65536"))
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_TIMEOUT_SEC", str(6 * 60 * 60)))

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

INFER_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(*_resolve_requirements(), "matplotlib")
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
class InferenceConfig:
    scene_subdir: str
    output_subdir: str = ""
    image_mode: str = "pad"
    hf_repo: str = "facebook/VGGT-1B"
    checkpoint_subpath: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(blob: str) -> "InferenceConfig":
        return InferenceConfig(**json.loads(blob))


def _normalize_subpath(value: str) -> str:
    cleaned = (value or "").strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("Expected a non-empty volume-relative path.")
    return cleaned


def _remote_data_path(subpath: str) -> Path:
    return Path(str(REMOTE_DATA_DIR / _normalize_subpath(subpath)))


def _remote_output_path(subpath: str) -> Path:
    return Path(str(REMOTE_OUTPUT_DIR / _normalize_subpath(subpath)))


def _resolve_checkpoint_path(subpath: str) -> Path:
    candidates = (
        _remote_data_path(subpath),
        _remote_output_path(subpath),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Checkpoint not found on either Modal data/output volume.\n"
        + "\n".join(f"- tried: {candidate}" for candidate in candidates)
    )


def _resolve_output_root(scene_subdir: str, output_subdir: str) -> Path:
    if output_subdir.strip():
        return Path(str(REMOTE_OUTPUT_DIR / _normalize_subpath(output_subdir)))
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    safe_scene = Path(scene_subdir).name.replace(" ", "_")
    return Path(str(REMOTE_OUTPUT_DIR / "vggt_4k4d_infer" / f"{run_tag}_{safe_scene}"))


def _upload_dir(local_dir: Path, remote_subdir: str) -> str:
    local_dir = local_dir.expanduser().resolve()
    if not local_dir.is_dir():
        raise NotADirectoryError(f"Scene directory not found: {local_dir}")
    remote_subdir = _normalize_subpath(remote_subdir)
    print(f"[modal-4k4d] upload scene: {local_dir} -> {DATA_VOLUME_NAME}:{remote_subdir}")
    with data_volume.batch_upload(force=True) as batch:
        for path in local_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(local_dir).as_posix()
            batch.put_file(str(path), f"{remote_subdir}/{rel}")
    return remote_subdir


def _upload_file(local_path: Path, remote_subpath: str) -> str:
    local_path = local_path.expanduser().resolve()
    if not local_path.is_file():
        raise FileNotFoundError(f"Local file not found: {local_path}")
    remote_subpath = _normalize_subpath(remote_subpath)
    print(f"[modal-4k4d] upload file: {local_path} -> {DATA_VOLUME_NAME}:{remote_subpath}")
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_path), remote_subpath)
    return remote_subpath


def _to_numpy(tensor):
    return tensor.detach().float().cpu().numpy()


def _write_preview_png(array, output_path: Path) -> None:
    import numpy as np
    from PIL import Image

    arr = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        preview = np.zeros(arr.shape, dtype=np.uint8)
    else:
        lo = float(np.percentile(arr[finite], 2))
        hi = float(np.percentile(arr[finite], 98))
        if hi <= lo:
            hi = lo + 1e-6
        scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        preview = (scaled * 255.0).astype(np.uint8)
    Image.fromarray(preview).save(output_path)


def _write_normal_preview_png(array, output_path: Path) -> None:
    import numpy as np
    from PIL import Image

    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected normal preview array with shape (H, W, 3), got {arr.shape}")
    finite = np.isfinite(arr).all(axis=-1, keepdims=True)
    arr = np.where(finite, arr, 0.0)
    arr = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
    preview = (arr * 255.0).round().astype(np.uint8)
    Image.fromarray(preview).save(output_path)


def _preprocess_mask(mask_path: Path, target_size: int = 518):
    import numpy as np
    from PIL import Image

    img = Image.open(mask_path).convert("L")
    width, height = img.size
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14

    img = img.resize((new_width, new_height), Image.Resampling.NEAREST)
    arr = np.asarray(img, dtype=np.uint8)
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = arr
    return canvas


def _load_mask_stack(mask_dir: Path):
    import numpy as np

    mask_paths = sorted(path for path in mask_dir.iterdir() if path.is_file())
    masks = [_preprocess_mask(path) for path in mask_paths]
    return np.stack(masks, axis=0)


def _write_ascii_ply(points, colors, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _camera_centers(extrinsic):
    import numpy as np

    centers = []
    for camera in extrinsic:
        rotation = camera[:, :3]
        translation = camera[:, 3]
        centers.append(-(rotation.T @ translation))
    return np.asarray(centers, dtype=np.float32)


def _render_views(points, colors, camera_xyz, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=180)
    views = [("Front (X/Y)", 0, 1), ("Side (Z/Y)", 2, 1), ("Top (X/Z)", 0, 2)]
    norm_colors = colors.astype(np.float32) / 255.0
    for ax, (label, ax_x, ax_y) in zip(axes, views):
        ax.scatter(points[:, ax_x], points[:, ax_y], s=0.15, c=norm_colors, linewidths=0, alpha=0.65)
        ax.scatter(
            camera_xyz[:, ax_x],
            camera_xyz[:, ax_y],
            s=24,
            c="red",
            marker="^",
            edgecolors="white",
            linewidths=0.4,
            alpha=0.9,
        )
        ax.set_title(label)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.15)
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _build_filtered_cloud(world_points, world_points_conf, colors, masks, max_points, conf_percentile, rng):
    import numpy as np

    points = world_points.reshape(-1, 3)
    conf = world_points_conf.reshape(-1)
    rgb = colors.reshape(-1, 3)

    valid = np.isfinite(points).all(axis=1) & np.isfinite(conf) & (conf > 0)
    if masks is not None:
        valid &= masks.reshape(-1) > 0

    if not np.any(valid):
        raise RuntimeError("No valid points after filtering.")

    conf_valid = conf[valid]
    conf_threshold = float(np.percentile(conf_valid, conf_percentile))
    keep = valid & (conf >= conf_threshold)
    if not np.any(keep):
        keep = valid

    kept_indices = np.flatnonzero(keep)
    if len(kept_indices) > max_points:
        kept_indices = rng.choice(kept_indices, size=max_points, replace=False)

    kept_points = points[kept_indices]
    kept_rgb = rgb[kept_indices]
    return kept_points, kept_rgb, {
        "valid_points_before_conf": int(valid.sum()),
        "conf_threshold": conf_threshold,
        "points_after_conf": int(keep.sum()),
        "points_written": int(len(kept_indices)),
    }


def _closed_form_inverse_se3_numpy(se3):
    import numpy as np

    rotation = se3[:, :3, :3]
    translation = se3[:, :3, 3:]
    rotation_t = np.transpose(rotation, (0, 2, 1))
    top_right = -np.matmul(rotation_t, translation)
    inverted = np.tile(np.eye(4, dtype=se3.dtype), (len(rotation), 1, 1))
    inverted[:, :3, :3] = rotation_t
    inverted[:, :3, 3:] = top_right
    return inverted


def _unproject_depth_map_to_point_map_numpy(depth_map, extrinsics_cam, intrinsics_cam):
    import numpy as np

    world_points = []
    cam_to_world = _closed_form_inverse_se3_numpy(extrinsics_cam)
    for frame_idx in range(depth_map.shape[0]):
        depth = depth_map[frame_idx].squeeze(-1).astype(np.float32)
        intrinsic = intrinsics_cam[frame_idx].astype(np.float32)
        height, width = depth.shape
        u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        fu, fv = intrinsic[0, 0], intrinsic[1, 1]
        cu, cv = intrinsic[0, 2], intrinsic[1, 2]
        x_cam = (u - cu) * depth / fu
        y_cam = (v - cv) * depth / fv
        z_cam = depth
        cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1)
        rotation = cam_to_world[frame_idx, :3, :3]
        translation = cam_to_world[frame_idx, :3, 3]
        world = np.dot(cam_coords, rotation.T) + translation
        world_points.append(world.astype(np.float32))
    return np.stack(world_points, axis=0)


def _render_pointcloud_artifacts(output_dir: Path, scene_dir: Path, colors, arrays, point_source: str) -> dict:
    import json
    import numpy as np

    if point_source == "world_points":
        world_points = arrays["world_points"]
        world_points_conf = arrays["world_points_conf"]
        subdir_name = "pointcloud"
    elif point_source == "depth_unprojection":
        world_points = _unproject_depth_map_to_point_map_numpy(arrays["depth"], arrays["extrinsic"], arrays["intrinsic"])
        world_points_conf = arrays["depth_conf"]
        subdir_name = "pointcloud_depth_unprojection"
    else:
        raise ValueError(f"Unsupported point source: {point_source}")

    artifact_dir = output_dir / subdir_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = scene_dir / "masks"
    masks = _load_mask_stack(mask_dir) if mask_dir.is_dir() else None
    camera_xyz = _camera_centers(arrays["extrinsic"])
    rng = np.random.default_rng(0)

    raw_points, raw_colors, raw_summary = _build_filtered_cloud(
        world_points, world_points_conf, colors, None, max_points=180000, conf_percentile=70.0, rng=rng
    )
    _write_ascii_ply(raw_points, raw_colors, artifact_dir / "fused_pointcloud_raw.ply")
    _render_views(
        raw_points,
        raw_colors,
        camera_xyz,
        artifact_dir / "fused_pointcloud_raw_views.png",
        f"VGGT Fused Point Cloud ({point_source}, raw)",
    )

    summary = {
        "point_source": point_source,
        "num_views": int(world_points.shape[0]),
        "input_image_shape": list(colors.shape),
        "conf_percentile": 70.0,
        "max_points": 180000,
        "raw": raw_summary,
    }

    if masks is not None:
        masked_points, masked_colors, masked_summary = _build_filtered_cloud(
            world_points, world_points_conf, colors, masks, max_points=180000, conf_percentile=70.0, rng=rng
        )
        _write_ascii_ply(masked_points, masked_colors, artifact_dir / "fused_pointcloud_masked.ply")
        _render_views(
            masked_points,
            masked_colors,
            camera_xyz,
            artifact_dir / "fused_pointcloud_masked_views.png",
            f"VGGT Fused Point Cloud ({point_source}, mask filtered)",
        )
        summary["masked"] = masked_summary

    (artifact_dir / "pointcloud_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"dir": str(artifact_dir), "summary": summary}


def _load_checkpoint_state_dict(source: str):
    import torch

    if source.startswith("http://") or source.startswith("https://"):
        state_dict = torch.hub.load_state_dict_from_url(source, map_location="cpu")
    else:
        state_dict = torch.load(source, map_location="cpu")
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    return state_dict


def _infer_vggt_model_kwargs_from_state_dict(state_dict) -> dict:
    if not isinstance(state_dict, dict):
        return {}
    model_kwargs = {
        "enable_camera": any(str(key).startswith("camera_head.") for key in state_dict.keys()),
        "enable_point": any(str(key).startswith("point_head.") for key in state_dict.keys()),
        "enable_depth": any(str(key).startswith("depth_head.") for key in state_dict.keys()),
        "enable_track": any(str(key).startswith("track_head.") for key in state_dict.keys()),
    }
    if any(str(key).startswith("normal_head.") for key in state_dict.keys()):
        model_kwargs["enable_normal"] = True
    prior_conv = state_dict.get("aggregator.human_prior_patch_embeds.0.0.weight")
    legacy_proj = state_dict.get("aggregator.human_prior_adapter.proj.0.weight")
    summary_ln = state_dict.get("aggregator.human_prior_summary_proj.0.weight")
    legacy_summary = state_dict.get("aggregator.human_prior_adapter.summary_proj.0.weight")
    if prior_conv is not None:
        model_kwargs["human_prior_in_chans"] = int(prior_conv.shape[1])
        model_kwargs["human_prior_hidden_dim"] = int(prior_conv.shape[0])
    elif legacy_proj is not None:
        model_kwargs["human_prior_in_chans"] = int(legacy_proj.shape[1])
        model_kwargs["human_prior_hidden_dim"] = int(legacy_proj.shape[0])
    if summary_ln is not None:
        model_kwargs["human_prior_summary_in_dim"] = int(summary_ln.shape[0])
    elif legacy_summary is not None:
        model_kwargs["human_prior_summary_in_dim"] = int(legacy_summary.shape[1])
    scale_count = len({
        str(key).split(".")[2]
        for key in state_dict.keys()
        if str(key).startswith("aggregator.human_prior_patch_embeds.") and str(key).endswith(".0.weight")
    })
    if scale_count == 3:
        model_kwargs["human_prior_scales"] = [1, 2, 4]
    elif scale_count > 0:
        model_kwargs["human_prior_scales"] = list(range(1, scale_count + 1))
    if not model_kwargs["enable_camera"]:
        model_kwargs.pop("enable_camera", None)
    if not model_kwargs["enable_point"]:
        model_kwargs.pop("enable_point", None)
    if not model_kwargs["enable_depth"]:
        model_kwargs.pop("enable_depth", None)
    if not model_kwargs["enable_track"]:
        model_kwargs.pop("enable_track", None)
    return model_kwargs


def _load_model_weights(model, source: str, state_dict=None) -> None:
    if state_dict is None:
        state_dict = _load_checkpoint_state_dict(source)
    current_state = model.state_dict()
    filtered_state = {}
    skipped = {}
    for key, value in state_dict.items():
        if key not in current_state:
            continue
        if current_state[key].shape != value.shape:
            skipped[key] = {
                "checkpoint_shape": tuple(value.shape),
                "model_shape": tuple(current_state[key].shape),
            }
            continue
        filtered_state[key] = value
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if skipped:
        preview = {
            key: value
            for key, value in list(skipped.items())[:12]
        }
        print(
            f"[modal-4k4d] skipped {len(skipped)} shape-mismatched checkpoint key(s): {preview}",
            flush=True,
        )
    if missing:
        print(f"[modal-4k4d] missing keys after filtered load: {missing[:16]}", flush=True)
    if unexpected:
        print(f"[modal-4k4d] unexpected keys after filtered load: {unexpected[:16]}", flush=True)


def _load_vggt_model(preferred_source: str, device: str, checkpoint_path: Path | None = None):
    import torch
    from vggt.models.vggt import VGGT

    if checkpoint_path is not None:
        checkpoint_path = checkpoint_path.resolve()
        state_dict = _load_checkpoint_state_dict(str(checkpoint_path))
        model_kwargs = _infer_vggt_model_kwargs_from_state_dict(state_dict)
        if model_kwargs:
            print(f"[modal-4k4d] inferred model kwargs from uploaded checkpoint: {model_kwargs}", flush=True)
        model = VGGT(**model_kwargs)
        print(f"[modal-4k4d] loading uploaded checkpoint: {checkpoint_path}", flush=True)
        _load_model_weights(model, str(checkpoint_path), state_dict=state_dict)
        return model.to(device).eval(), str(checkpoint_path), "uploaded_checkpoint"

    if preferred_source and (preferred_source.startswith("http://") or preferred_source.startswith("https://") or Path(preferred_source).exists()):
        state_dict = _load_checkpoint_state_dict(preferred_source)
        model_kwargs = _infer_vggt_model_kwargs_from_state_dict(state_dict)
        if model_kwargs:
            print(f"[modal-4k4d] inferred model kwargs from checkpoint source: {model_kwargs}", flush=True)
        model = VGGT(**model_kwargs)
        print(f"[modal-4k4d] loading checkpoint source: {preferred_source}", flush=True)
        _load_model_weights(model, preferred_source, state_dict=state_dict)
        return model.to(device).eval(), preferred_source, "checkpoint"

    if preferred_source:
        try:
            print(f"[modal-4k4d] loading HF repo: {preferred_source}", flush=True)
            model = VGGT.from_pretrained(preferred_source)
            return model.to(device).eval(), preferred_source, "hf_repo"
        except Exception as exc:
            print(
                f"[modal-4k4d] HF repo load failed for {preferred_source}: {exc}\n"
                f"[modal-4k4d] falling back to direct checkpoint URL: {DEFAULT_MODEL_URL}",
                flush=True,
            )

    model = VGGT()
    _load_model_weights(model, DEFAULT_MODEL_URL)
    return model.to(device).eval(), DEFAULT_MODEL_URL, "checkpoint_fallback"


def _load_human_prior_feature_maps(scene_dir: Path):
    import numpy as np
    import torch

    bundle_path = scene_dir / "human_prior" / "smplx_vertex_feature_maps.npz"
    if not bundle_path.is_file():
        return None, None, {}

    with np.load(bundle_path, allow_pickle=False) as bundle:
        feature_key = "smpl_surface_feature_maps" if "smpl_surface_feature_maps" in bundle else "smpl_vertex_feature_maps"
        feature_maps = np.asarray(bundle[feature_key], dtype=np.float32)
        channel_key = "surface_channel_names" if "surface_channel_names" in bundle else "channel_names"
        channel_names = [str(name) for name in bundle[channel_key].tolist()]
        camera_ids = [str(camera_id) for camera_id in bundle["camera_ids"].tolist()]
        summary_tokens = None
        summary_feature_names = []
        summary_bin_names = []
        if "smpl_summary_tokens" in bundle:
            summary_tokens = np.asarray(bundle["smpl_summary_tokens"], dtype=np.float32)
            summary_feature_names = [str(name) for name in bundle["summary_feature_names"].tolist()]
            summary_bin_names = [str(name) for name in bundle["summary_bin_names"].tolist()]

    return torch.from_numpy(feature_maps).unsqueeze(0), (
        torch.from_numpy(summary_tokens).unsqueeze(0) if summary_tokens is not None else None
    ), {
        "bundle_path": str(bundle_path),
        "shape": list(feature_maps.shape),
        "channel_names": channel_names,
        "camera_ids": camera_ids,
        "summary_shape": list(summary_tokens.shape) if summary_tokens is not None else [],
        "summary_feature_names": summary_feature_names,
        "summary_bin_names": summary_bin_names,
    }


@app.function(
    image=INFER_IMAGE,
    gpu=GPU_SPEC,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SEC,
    volumes={
        REMOTE_DATA_DIR.as_posix(): data_volume,
        REMOTE_OUTPUT_DIR.as_posix(): output_volume,
    },
)
def run_remote_vggt_inference(cfg_json: str) -> None:
    cfg = InferenceConfig.from_json(cfg_json)
    remote_code_dir = Path(str(REMOTE_CODE_DIR))
    if str(remote_code_dir) not in sys.path:
        sys.path.insert(0, str(remote_code_dir))

    import numpy as np
    import torch
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    scene_dir = _remote_data_path(cfg.scene_subdir)
    image_dir = scene_dir / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Remote image dir not found: {image_dir}")
    image_paths = sorted([path for path in image_dir.iterdir() if path.is_file()])
    if not image_paths:
        raise FileNotFoundError(f"No images found under {image_dir}")

    output_root = _resolve_output_root(cfg.scene_subdir, cfg.output_subdir)
    output_root.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    start_time = time.time()

    checkpoint_path = None
    if cfg.checkpoint_subpath.strip():
        checkpoint_path = _resolve_checkpoint_path(cfg.checkpoint_subpath)

    model, resolved_model_source, model_load_mode = _load_vggt_model(
        cfg.hf_repo,
        device,
        checkpoint_path=checkpoint_path,
    )
    images = load_and_preprocess_images([str(path) for path in image_paths], mode=cfg.image_mode).to(device)
    human_prior_feature_maps, human_prior_summary_tokens, human_prior_summary = _load_human_prior_feature_maps(scene_dir)
    if human_prior_feature_maps is not None:
        human_prior_feature_maps = human_prior_feature_maps.to(device=device)
    if human_prior_summary_tokens is not None:
        human_prior_summary_tokens = human_prior_summary_tokens.to(device=device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(
                images,
                human_prior_feature_maps=human_prior_feature_maps,
                human_prior_summary_tokens=human_prior_summary_tokens,
            )

    pose_enc = predictions["pose_enc"]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    arrays = {
        "pose_enc": _to_numpy(pose_enc.squeeze(0)),
        "extrinsic": _to_numpy(extrinsic.squeeze(0)),
        "intrinsic": _to_numpy(intrinsic.squeeze(0)),
        "depth": _to_numpy(predictions["depth"].squeeze(0)),
        "depth_conf": _to_numpy(predictions["depth_conf"].squeeze(0)),
        "world_points": _to_numpy(predictions["world_points"].squeeze(0)),
        "world_points_conf": _to_numpy(predictions["world_points_conf"].squeeze(0)),
    }
    if "normal" in predictions:
        arrays["normal"] = _to_numpy(predictions["normal"].squeeze(0))
    if "normal_conf" in predictions:
        arrays["normal_conf"] = _to_numpy(predictions["normal_conf"].squeeze(0))
    np.savez_compressed(output_root / "predictions.npz", **arrays)
    colors = (
        images.detach().float().cpu().numpy().transpose(0, 2, 3, 1).clip(0.0, 1.0) * 255.0
    ).round().astype("uint8")

    preview_dir = output_root / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for idx, image_path in enumerate(image_paths):
        stem = image_path.stem
        _write_preview_png(arrays["depth"][idx, ..., 0], preview_dir / f"{stem}_depth.png")
        _write_preview_png(arrays["depth_conf"][idx], preview_dir / f"{stem}_depth_conf.png")
        _write_preview_png(arrays["world_points_conf"][idx], preview_dir / f"{stem}_point_conf.png")
        if "normal" in arrays:
            _write_normal_preview_png(arrays["normal"][idx], preview_dir / f"{stem}_normal.png")
        if "normal_conf" in arrays:
            _write_preview_png(arrays["normal_conf"][idx], preview_dir / f"{stem}_normal_conf.png")

    pointcloud_outputs = {
        "world_points": _render_pointcloud_artifacts(output_root, scene_dir, colors, arrays, "world_points"),
        "depth_unprojection": _render_pointcloud_artifacts(output_root, scene_dir, colors, arrays, "depth_unprojection"),
    }

    scene_manifest_path = scene_dir / "scene_manifest.json"
    scene_manifest = {}
    if scene_manifest_path.exists():
        scene_manifest = json.loads(scene_manifest_path.read_text(encoding="utf-8"))

    summary = {
        "scene_subdir": cfg.scene_subdir,
        "image_mode": cfg.image_mode,
        "hf_repo": cfg.hf_repo,
        "checkpoint_subpath": cfg.checkpoint_subpath,
        "resolved_model_source": resolved_model_source,
        "model_load_mode": model_load_mode,
        "image_names": [path.name for path in image_paths],
        "num_images": len(image_paths),
        "device": device,
        "dtype": str(dtype),
        "input_tensor_shape": list(images.shape),
        "output_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "gpu_name": torch.cuda.get_device_name(0),
        "elapsed_seconds": round(time.time() - start_time, 3),
        "pointcloud_outputs": {
            key: {
                "dir": value["dir"],
                "summary": value["summary"],
            }
            for key, value in pointcloud_outputs.items()
        },
        "human_prior": human_prior_summary,
        "scene_manifest": scene_manifest,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    output_volume.commit()
    print("[modal-4k4d] output_root =", output_root.as_posix(), flush=True)
    print("[modal-4k4d] committed output volume", flush=True)


@app.local_entrypoint()
def upload_scene(
    local_scene_dir: str,
    remote_scene_subdir: str = "",
) -> None:
    local_dir = Path(local_scene_dir).expanduser().resolve()
    if not remote_scene_subdir.strip():
        remote_scene_subdir = f"scenes/{local_dir.name}"
    remote_subdir = _upload_dir(local_dir, remote_scene_subdir)
    print(f"[modal-4k4d] scene uploaded to {DATA_VOLUME_NAME}:{remote_subdir}")


@app.local_entrypoint()
def run_scene(
    scene_subdir: str,
    output_subdir: str = "",
    image_mode: str = "pad",
    hf_repo: str = "facebook/VGGT-1B",
    checkpoint_subpath: str = "",
) -> None:
    cfg = InferenceConfig(
        scene_subdir=scene_subdir,
        output_subdir=output_subdir,
        image_mode=image_mode,
        hf_repo=hf_repo,
        checkpoint_subpath=checkpoint_subpath,
    )
    print("[modal-4k4d] launch config:")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    run_remote_vggt_inference.remote(cfg.to_json())


@app.local_entrypoint()
def run_scene_from_local(
    local_scene_dir: str,
    remote_scene_subdir: str = "",
    output_subdir: str = "",
    image_mode: str = "pad",
    hf_repo: str = "facebook/VGGT-1B",
    local_checkpoint: str = "",
    checkpoint_subpath: str = "",
) -> None:
    local_dir = Path(local_scene_dir).expanduser().resolve()
    if not remote_scene_subdir.strip():
        remote_scene_subdir = f"scenes/{local_dir.name}"
    remote_subdir = _upload_dir(local_dir, remote_scene_subdir)
    resolved_checkpoint_subpath = checkpoint_subpath
    if local_checkpoint.strip():
        local_checkpoint_path = Path(local_checkpoint).expanduser().resolve()
        if not resolved_checkpoint_subpath.strip():
            resolved_checkpoint_subpath = f"checkpoints/{local_checkpoint_path.name}"
        resolved_checkpoint_subpath = _upload_file(local_checkpoint_path, resolved_checkpoint_subpath)
    cfg = InferenceConfig(
        scene_subdir=remote_subdir,
        output_subdir=output_subdir,
        image_mode=image_mode,
        hf_repo=hf_repo,
        checkpoint_subpath=resolved_checkpoint_subpath,
    )
    print("[modal-4k4d] upload+run config:")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    run_remote_vggt_inference.remote(cfg.to_json())
