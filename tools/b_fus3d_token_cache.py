from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.lib.format as np_format


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
}
CONTRACT = {
    "research_only": True,
    "local_only": True,
    "no_train": True,
    "no_cloud": True,
    "no_candidate_export": True,
    "no_teacher_export": True,
    "no_strict_pass_write": True,
    "uses_vggt_depth_point_normal_as_hard_teacher": False,
    "writes_predictions_npz": False,
    "writes_strict_registry": False,
    "writes_candidate": False,
    "writes_teacher": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "B-Fus3D token evidence cache skeleton. Default mode is local metadata/dry-run only; "
            "explicit --extract plus a local --checkpoint is required to call VGGT. This never "
            "writes strict pass state, teachers, candidates, formal train/infer/export jobs, or "
            "cloud unblock signals."
        )
    )
    parser.add_argument("--scene-dir", type=Path, default=None, help="Optional scene dir with images/, masks/, scene_manifest.json.")
    parser.add_argument("--predictions", type=Path, default=None, help="Optional existing predictions.npz to scan by header only.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Local VGGT checkpoint path. No download is attempted.")
    parser.add_argument("--device", default="cuda", help="Extraction device. Ignored unless --extract is set.")
    parser.add_argument("--image-mode", choices=("pad", "crop"), default="pad")
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--view-indices", default="", help="Comma-separated scene image indices for extraction, e.g. 0,10,20.")
    parser.add_argument("--max-images", type=int, default=2, help="Metadata/default extraction view cap when --view-indices is omitted.")
    parser.add_argument("--extract", action="store_true", help="Actually load local VGGT and extract aggregator intermediate token stats.")
    parser.add_argument("--save-token-arrays", action="store_true", help="Save selected full token tensors into token_cache/*.npz.")
    parser.add_argument("--cache-layers", default="last", help="Layer selector for --save-token-arrays: last, all, or comma indices.")
    parser.add_argument("--token-array-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--max-stat-values", type=int, default=200000, help="Maximum sampled values per tensor for numeric stats.")
    parser.add_argument("--strict-load", action="store_true", help="Treat local checkpoint load mismatch as blocking.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def line_lookup(path: Path, patterns: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "matches": {}}
    if not path.is_file():
        return out
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for name, pattern in patterns.items():
        match = None
        for idx, line in enumerate(lines, start=1):
            if pattern in line:
                match = {"line": idx, "text": line.strip()}
                break
        out["matches"][name] = match
    return out


def scan_source_entrypoints() -> dict[str, Any]:
    aggregator = line_lookup(
        REPO_ROOT / "vggt" / "models" / "aggregator.py",
        {
            "aggregator_forward": "def forward(",
            "patch_tokens": "patch_tokens = self.patch_embed(images)",
            "special_token_concat": "tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)",
            "output_list": "output_list = []",
            "frame_attention": "def _process_frame_attention",
            "frame_intermediate_append": "intermediates.append(tokens.reshape(B, S, P, C))",
            "global_attention": "def _process_global_attention",
            "concat_frame_global": "concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)",
            "aggregator_return": "return output_list, self.patch_start_idx",
        },
    )
    vggt = line_lookup(
        REPO_ROOT / "vggt" / "models" / "vggt.py",
        {
            "vggt_aggregator_call": "aggregated_tokens_list, patch_start_idx = self.aggregator(",
            "predictions_dict": "predictions = {}",
            "depth_output": 'predictions["depth"] = depth',
            "point_output": 'predictions["world_points"] = pts3d',
            "normal_output": 'predictions["normal"] = normals',
            "images_output": 'predictions["images"] = images',
            "vggt_return": "return predictions",
        },
    )
    local_infer = line_lookup(
        REPO_ROOT / "tools" / "run_local_vggt_inference.py",
        {
            "image_dir": 'image_dir = scene_dir / "images"',
            "npz_arrays": "arrays = {",
            "predictions_npz": 'np.savez_compressed(output_dir / "predictions.npz", **arrays)',
            "summary_json": '(output_dir / "summary.json").write_text',
        },
    )
    required = [
        aggregator["matches"].get("aggregator_return"),
        aggregator["matches"].get("concat_frame_global"),
        vggt["matches"].get("vggt_aggregator_call"),
        local_infer["matches"].get("predictions_npz"),
    ]
    status = "clear" if all(required) else "blocked_unclear_entrypoint"
    blockers = []
    if status != "clear":
        blockers.append(
            "Could not verify one or more local source patterns for Aggregator token output or predictions.npz layout."
        )
    return {
        "status": status,
        "aggregator": aggregator,
        "vggt_forward": vggt,
        "local_inference": local_infer,
        "token_extraction_summary": (
            "Call model.aggregator(images) directly. It returns aggregated_tokens_list and patch_start_idx; "
            "each list item is torch.cat(frame_intermediate, global_intermediate, dim=-1) with shape [B, S, P, 2C]."
        ),
        "blockers": blockers,
    }


def parse_view_indices(value: str, count: int, max_images: int) -> list[int]:
    if count <= 0:
        return []
    if not value.strip():
        return list(range(min(max(0, int(max_images)), count)))
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            out.extend(range(start, end + step, step))
        else:
            out.append(int(part))
    deduped: list[int] = []
    for idx in out:
        if idx < 0:
            idx = count + idx
        if 0 <= idx < count and idx not in deduped:
            deduped.append(idx)
    return deduped


def image_paths_for_scene(scene_dir: Path | None) -> list[Path]:
    if scene_dir is None:
        return []
    image_dir = scene_dir / "images"
    if not image_dir.is_dir():
        return []
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def scan_image_headers(paths: list[Path], selected_indices: list[int], max_headers: int = 8) -> dict[str, Any]:
    suffix_histogram: dict[str, int] = {}
    for path in paths:
        suffix = path.suffix.lower()
        suffix_histogram[suffix] = suffix_histogram.get(suffix, 0) + 1
    header_rows: list[dict[str, Any]] = []
    pil_status = "not_imported"
    try:
        from PIL import Image

        pil_status = "available"
        for idx in selected_indices[:max_headers]:
            path = paths[idx]
            with Image.open(path) as image:
                width, height = image.size
                header_rows.append(
                    {
                        "index": int(idx),
                        "name": path.name,
                        "width": int(width),
                        "height": int(height),
                        "mode": image.mode,
                    }
                )
    except Exception as exc:  # pragma: no cover - depends on local imaging stack
        pil_status = f"unavailable_or_failed: {exc}"
    return {
        "count": len(paths),
        "suffix_histogram": suffix_histogram,
        "selected_indices": selected_indices,
        "selected_names": [paths[idx].name for idx in selected_indices],
        "header_scan_status": pil_status,
        "selected_headers": header_rows,
    }


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_parse_error": str(exc)}
    return payload if isinstance(payload, dict) else {"_non_dict_payload_type": type(payload).__name__}


def scan_npz_headers(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"provided": False}
    out: dict[str, Any] = {"provided": True, "path": str(path), "exists": path.is_file(), "arrays": {}}
    if not path.is_file():
        return out
    out["bytes"] = int(path.stat().st_size)
    try:
        with zipfile.ZipFile(path, "r") as archive:
            for info in archive.infolist():
                if not info.filename.endswith(".npy"):
                    continue
                key = Path(info.filename).stem
                with archive.open(info.filename, "r") as handle:
                    version = np_format.read_magic(handle)
                    if version == (1, 0):
                        shape, fortran_order, dtype = np_format.read_array_header_1_0(handle)
                    elif version == (2, 0):
                        shape, fortran_order, dtype = np_format.read_array_header_2_0(handle)
                    else:
                        shape, fortran_order, dtype = np_format._read_array_header(handle, version)  # noqa: SLF001
                out["arrays"][key] = {
                    "shape": [int(v) for v in shape],
                    "dtype": str(dtype),
                    "fortran_order": bool(fortran_order),
                    "compressed_bytes": int(info.compress_size),
                    "file_bytes": int(info.file_size),
                }
        out["status"] = "ok"
    except Exception as exc:
        out["status"] = "failed"
        out["error"] = str(exc)
    return out


def scan_scene(scene_dir: Path | None, view_indices_text: str, max_images: int) -> dict[str, Any]:
    if scene_dir is None:
        return {"provided": False}
    scene_dir = scene_dir.resolve()
    image_paths = image_paths_for_scene(scene_dir)
    selected_indices = parse_view_indices(view_indices_text, len(image_paths), max_images)
    manifest = read_json_if_exists(scene_dir / "scene_manifest.json")
    exported_views = manifest.get("exported_views") if isinstance(manifest.get("exported_views"), list) else []
    mask_paths = sorted(
        path
        for path in (scene_dir / "masks").iterdir()
        if (scene_dir / "masks").is_dir() and path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    return {
        "provided": True,
        "path": str(scene_dir),
        "exists": scene_dir.is_dir(),
        "images_dir": str(scene_dir / "images"),
        "images": scan_image_headers(image_paths, selected_indices),
        "masks_dir": str(scene_dir / "masks"),
        "masks": {
            "exists": (scene_dir / "masks").is_dir(),
            "count": len(mask_paths),
            "sample_names": [path.name for path in mask_paths[:8]],
        },
        "scene_manifest": {
            "exists": (scene_dir / "scene_manifest.json").is_file(),
            "exported_view_count": len(exported_views),
            "keys": sorted(str(key) for key in manifest.keys()) if manifest else [],
            "parse_error": manifest.get("_parse_error"),
        },
        "camera_params_sidecar": {
            "path": str(scene_dir / "camera_params_sidecar.npz"),
            "exists": (scene_dir / "camera_params_sidecar.npz").is_file(),
        },
        "prior_maps": scan_npz_headers(scene_dir / "prior_maps.npz"),
    }


def discover_local_checkpoint_candidates() -> list[dict[str, Any]]:
    candidates: list[Path] = []
    common_torch_hub = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "model.pt"
    if common_torch_hub.is_file():
        candidates.append(common_torch_hub)
    external_models = REPO_ROOT / "external_models"
    if external_models.is_dir():
        for suffix in ("*.pt", "*.pth", "*.ckpt", "*.safetensors"):
            candidates.extend(sorted(external_models.glob(suffix)))

    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            size = int(resolved.stat().st_size)
        except OSError:
            size = None
        rows.append(
            {
                "path": str(resolved),
                "exists": resolved.is_file(),
                "bytes": size,
                "name": resolved.name,
                "looks_like_vggt_1b_by_size": bool(size is not None and size > 1_000_000_000),
            }
        )
    return rows


def scan_model_inputs(checkpoint: Path | None) -> dict[str, Any]:
    checkpoint_info = {"provided": checkpoint is not None}
    if checkpoint is not None:
        resolved = checkpoint.resolve()
        checkpoint_info.update(
            {
                "path": str(resolved),
                "exists": resolved.is_file(),
                "bytes": int(resolved.stat().st_size) if resolved.is_file() else None,
            }
        )
    return {
        "checkpoint": checkpoint_info,
        "local_checkpoint_candidates": discover_local_checkpoint_candidates(),
        "downloads_attempted": False,
        "hf_or_cloud_used": False,
    }


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


def tensor_sample_stats(tensor: Any, max_values: int) -> dict[str, Any]:
    import torch

    detached = tensor.detach()
    flat = detached.reshape(-1)
    total = int(flat.numel())
    max_values = max(1, int(max_values))
    stride = max(1, math.ceil(total / max_values))
    sample = flat[::stride][:max_values].float()
    finite = torch.isfinite(sample)
    finite_sample = sample[finite]
    stats: dict[str, Any] = {
        "shape": [int(v) for v in detached.shape],
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "sampled_values": int(sample.numel()),
        "sample_stride": int(stride),
        "finite_fraction_sample": float(finite.float().mean().item()) if sample.numel() else None,
    }
    if int(finite_sample.numel()) <= 0:
        stats["numeric_status"] = "no_finite_sample"
        return stats
    cpu_sample = finite_sample.detach().cpu()
    abs_sample = cpu_sample.abs()
    stats.update(
        {
            "numeric_status": "ok",
            "mean_sample": float(cpu_sample.mean().item()),
            "std_sample": float(cpu_sample.std(unbiased=False).item()) if cpu_sample.numel() > 1 else 0.0,
            "min_sample": float(cpu_sample.min().item()),
            "max_sample": float(cpu_sample.max().item()),
            "abs_p50_sample": float(torch.quantile(abs_sample, 0.50).item()),
            "abs_p95_sample": float(torch.quantile(abs_sample, 0.95).item()),
        }
    )
    return stats


def resolve_cache_layers(spec: str, layer_count: int) -> list[int]:
    if layer_count <= 0:
        return []
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(layer_count))
    if spec == "last":
        return [layer_count - 1]
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0:
            idx = layer_count + idx
        if 0 <= idx < layer_count and idx not in layers:
            layers.append(idx)
    return layers


def run_extraction(args: argparse.Namespace, scene_scan: dict[str, Any]) -> dict[str, Any]:
    if not args.extract:
        return {
            "status": "not_requested",
            "reason": "Default dry-run/metadata mode. Pass --extract with a local --checkpoint to call VGGT.",
        }
    if args.checkpoint is None or not args.checkpoint.is_file():
        return {"status": "blocked_missing_checkpoint", "reason": "--extract requires an existing local --checkpoint."}

    scene_dir = args.scene_dir.resolve() if args.scene_dir else None
    image_paths = image_paths_for_scene(scene_dir)
    selected_indices = scene_scan.get("images", {}).get("selected_indices", [])
    if not image_paths or not selected_indices:
        return {"status": "blocked_no_images", "reason": "--extract requires scene images and selected view indices."}
    selected_paths = [image_paths[int(idx)] for idx in selected_indices]

    start = time.time()
    try:
        import torch

        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
    except Exception as exc:
        return {"status": "blocked_import_error", "reason": str(exc)}

    device_name = str(args.device)
    if device_name == "cuda" and not torch.cuda.is_available():
        return {"status": "blocked_no_cuda", "reason": "CUDA requested but torch.cuda.is_available() is false."}
    device = torch.device(device_name)

    try:
        payload = torch.load(args.checkpoint.resolve(), map_location="cpu")
        state_dict = _extract_model_state_dict(payload)
        model_kwargs = payload.get("model_kwargs") if isinstance(payload, dict) else None
        if not isinstance(model_kwargs, dict):
            model_kwargs = _infer_model_kwargs_from_state_dict(state_dict)
        model = VGGT(**model_kwargs)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if args.strict_load and (missing or unexpected):
            return {
                "status": "blocked_load_mismatch",
                "missing": list(missing),
                "unexpected": list(unexpected),
            }
        model.to(device)
        model.eval()
        images = load_and_preprocess_images(
            [str(path) for path in selected_paths],
            mode=str(args.image_mode),
            target_size=int(args.target_size),
        ).to(device)
        images_batch = images.unsqueeze(0)
        dtype = torch.float32
        autocast_enabled = device.type == "cuda"
        if autocast_enabled:
            dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=dtype):
                aggregated_tokens_list, patch_start_idx = model.aggregator(images_batch)
    except Exception as exc:
        return {"status": "blocked_extraction_error", "reason": repr(exc)}

    layer_stats: list[dict[str, Any]] = []
    for layer_idx, tokens in enumerate(aggregated_tokens_list):
        stats = tensor_sample_stats(tokens, int(args.max_stat_values))
        stats.update(
            {
                "layer_index": int(layer_idx),
                "patch_start_idx": int(patch_start_idx),
                "special_token_count": int(patch_start_idx),
                "patch_token_count": int(tokens.shape[2] - patch_start_idx) if len(tokens.shape) >= 3 else None,
            }
        )
        layer_stats.append(stats)

    saved_arrays: list[str] = []
    if args.save_token_arrays:
        import torch

        cache_dir = args.output_dir.resolve() / "token_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        selected_layers = resolve_cache_layers(str(args.cache_layers), len(aggregated_tokens_list))
        for layer_idx in selected_layers:
            tensor = aggregated_tokens_list[layer_idx].detach()
            if args.token_array_dtype == "float16":
                tensor = tensor.to(dtype=torch.float16)
            else:
                tensor = tensor.to(dtype=torch.float32)
            array_path = cache_dir / f"aggregator_layer_{layer_idx:02d}.npz"
            np.savez_compressed(
                array_path,
                tokens=tensor.cpu().numpy(),
                patch_start_idx=np.asarray([int(patch_start_idx)], dtype=np.int32),
                selected_view_indices=np.asarray(selected_indices, dtype=np.int32),
            )
            saved_arrays.append(str(array_path))

    gpu_name = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    return {
        "status": "extracted",
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "gpu_name": gpu_name,
        "autocast_dtype": str(dtype),
        "image_mode": str(args.image_mode),
        "target_size": int(args.target_size),
        "selected_images": [path.name for path in selected_paths],
        "input_tensor_shape": [int(v) for v in images_batch.shape],
        "model_kwargs": model_kwargs,
        "checkpoint_load_missing": list(missing),
        "checkpoint_load_unexpected": list(unexpected),
        "patch_start_idx": int(patch_start_idx),
        "layer_count": int(len(aggregated_tokens_list)),
        "layer_stats": layer_stats,
        "saved_token_arrays": saved_arrays,
        "elapsed_seconds": round(time.time() - start, 3),
    }


def expected_token_layout(args: argparse.Namespace, scene_scan: dict[str, Any]) -> dict[str, Any]:
    target_size = int(args.target_size)
    patch_size = 14
    grid_h = target_size // patch_size if target_size % patch_size == 0 else None
    grid_w = target_size // patch_size if target_size % patch_size == 0 else None
    patch_tokens = grid_h * grid_w if grid_h is not None and grid_w is not None else None
    selected_views = len(scene_scan.get("images", {}).get("selected_indices", [])) if scene_scan.get("provided") else 0
    return {
        "status": "estimated_from_local_source_defaults",
        "aggregator_depth_default": 24,
        "aa_order_default": ["frame", "global"],
        "aa_block_size_default": 1,
        "expected_layer_count": 24,
        "patch_size_default": patch_size,
        "target_size": target_size,
        "patch_grid_estimate": [grid_h, grid_w] if grid_h is not None else None,
        "patch_tokens_estimate": patch_tokens,
        "special_tokens": 5,
        "token_count_estimate": patch_tokens + 5 if patch_tokens is not None else None,
        "embed_dim_default": 1024,
        "concat_dim_default": 2048,
        "placeholder_shape_estimate": [1, selected_views, patch_tokens + 5, 2048]
        if patch_tokens is not None and selected_views
        else None,
    }


def build_token_layer_stats(
    args: argparse.Namespace,
    source_scan: dict[str, Any],
    scene_scan: dict[str, Any],
    extraction: dict[str, Any],
) -> dict[str, Any]:
    layout = expected_token_layout(args, scene_scan)
    if extraction.get("status") == "extracted":
        return {
            "status": "extracted",
            "research_only": True,
            "entrypoint_status": source_scan.get("status"),
            "entrypoint": source_scan.get("token_extraction_summary"),
            "patch_start_idx": extraction.get("patch_start_idx"),
            "layer_count": extraction.get("layer_count"),
            "layers": extraction.get("layer_stats", []),
            "saved_token_arrays": extraction.get("saved_token_arrays", []),
            "expected_layout": layout,
        }

    placeholder_layers: list[dict[str, Any]] = []
    for idx in range(int(layout["expected_layer_count"])):
        placeholder_layers.append(
            {
                "layer_index": idx,
                "status": "placeholder_not_extracted",
                "expected_shape": layout.get("placeholder_shape_estimate"),
                "expected_semantics": "concat(frame_intermediate, global_intermediate, dim=-1)",
            }
        )
    return {
        "status": "placeholder_only",
        "research_only": True,
        "entrypoint_status": source_scan.get("status"),
        "entrypoint": source_scan.get("token_extraction_summary"),
        "extraction_status": extraction.get("status"),
        "expected_layout": layout,
        "layers": placeholder_layers,
    }


def build_roi_placeholders(scene_scan: dict[str, Any]) -> dict[str, Any]:
    masks = scene_scan.get("masks", {}) if scene_scan.get("provided") else {}
    prior_maps = scene_scan.get("prior_maps", {}) if scene_scan.get("provided") else {}
    mask_assets_present = bool(masks.get("exists") and masks.get("count", 0) > 0)
    prior_keys = sorted((prior_maps.get("arrays") or {}).keys())
    families = {}
    for family in ("full_body", "left_hand", "right_hand", "face", "hair", "matting"):
        families[family] = {
            "status": "placeholder_not_evaluated",
            "roi_mask_source": None,
            "token_to_roi_mapping": "not_wired",
            "coverage_fraction": None,
            "covered_token_count": None,
            "total_token_count": None,
            "hard_gate_allowed": False,
        }
    return {
        "status": "placeholder_only",
        "research_only": True,
        "mask_assets_present": mask_assets_present,
        "mask_sample_names": masks.get("sample_names", []),
        "prior_map_keys": prior_keys,
        "families": families,
        "blocked_until": [
            "Connect image-space human crop/matting masks to aggregator patch-token grid.",
            "Add explicit face/left_hand/right_hand ROI mask provenance before any hard gate language.",
            "Keep full-body/hands gates diagnostic-only until separate strict coord and visual passes exist.",
        ],
        "hard_teacher_note": (
            "VGGT depth, point, and normal predictions are not used as hard teachers by this cache skeleton."
        ),
    }


def build_min_command(args: argparse.Namespace) -> dict[str, str]:
    scene_part = f" --scene-dir {args.scene_dir}" if args.scene_dir else ""
    pred_part = f" --predictions {args.predictions}" if args.predictions else ""
    dry = f"python tools\\b_fus3d_token_cache.py{scene_part}{pred_part} --output-dir {args.output_dir} --overwrite"
    checkpoint = args.checkpoint or (Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "model.pt")
    extract = (
        f"python tools\\b_fus3d_token_cache.py{scene_part}{pred_part} --output-dir {args.output_dir} "
        f"--checkpoint {checkpoint} --extract --view-indices 0,10 --target-size {int(args.target_size)} --overwrite"
    )
    return {"dry_run_metadata": dry.strip(), "optional_local_extraction": extract.strip()}


def build_summary(
    args: argparse.Namespace,
    source_scan: dict[str, Any],
    scene_scan: dict[str, Any],
    predictions_scan: dict[str, Any],
    model_scan: dict[str, Any],
    extraction: dict[str, Any],
    token_stats_path: Path,
    roi_path: Path,
) -> dict[str, Any]:
    missing_assets: list[str] = []
    if not scene_scan.get("provided"):
        missing_assets.append("scene_dir not provided; only source/model metadata was scanned")
    elif not scene_scan.get("images", {}).get("count"):
        missing_assets.append("scene images not found under scene_dir/images")
    if not model_scan.get("checkpoint", {}).get("exists"):
        missing_assets.append("explicit local VGGT checkpoint not provided")
    if not scene_scan.get("masks", {}).get("count"):
        missing_assets.append("human crop/matting masks not found or not scanned")
    missing_assets.append("ROI-to-aggregator-token coverage mapping is placeholder only")

    status = "metadata_only"
    if extraction.get("status") == "extracted":
        status = "extracted"
    elif args.extract and str(extraction.get("status", "")).startswith("blocked"):
        status = "blocked_extraction"

    return {
        "task": "b_fus3d_token_cache",
        "schema_version": 1,
        "status": status,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "strict_facts": STRICT_FACTS,
        "contract": CONTRACT,
        "inputs": {
            "scene_dir": str(args.scene_dir.resolve()) if args.scene_dir else None,
            "predictions": str(args.predictions.resolve()) if args.predictions else None,
            "output_dir": str(args.output_dir.resolve()),
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "target_size": int(args.target_size),
            "image_mode": str(args.image_mode),
            "view_indices": str(args.view_indices),
            "max_images": int(args.max_images),
            "extract_requested": bool(args.extract),
        },
        "source_scan": source_scan,
        "scene_scan": scene_scan,
        "predictions_scan": predictions_scan,
        "model_scan": model_scan,
        "extraction": {key: value for key, value in extraction.items() if key != "layer_stats"},
        "local_assets_sufficient_for": {
            "metadata_dry_run": source_scan.get("status") == "clear",
            "token_extraction": bool(
                source_scan.get("status") == "clear"
                and scene_scan.get("images", {}).get("count")
                and model_scan.get("checkpoint", {}).get("exists")
            ),
            "roi_coverage_hard_gate": False,
            "teacher_or_candidate_export": False,
            "strict_pass": False,
            "formal_cloud_train_infer_export": False,
        },
        "missing_or_placeholder_assets": missing_assets,
        "outputs": {
            "summary_json": str(args.output_dir.resolve() / "b_fus3d_token_cache_summary.json"),
            "token_layer_stats_json": str(token_stats_path),
            "roi_coverage_placeholders_json": str(roi_path),
        },
        "next_min_commands": build_min_command(args),
        "notes": [
            "This cache is token evidence for B-Fus3D research diagnostics only.",
            "No VGGT depth/point/normal output is treated as hard teacher evidence.",
            "No strict pass, teacher, candidate, cloud, train, infer, or export artifact is written.",
        ],
    }


def main() -> int:
    args = parse_args()
    if int(args.target_size) <= 0 or int(args.target_size) % 14 != 0:
        raise ValueError(f"--target-size must be positive and divisible by 14, got {args.target_size}")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_scan = scan_source_entrypoints()
    scene_scan = scan_scene(args.scene_dir, str(args.view_indices), int(args.max_images))
    predictions_scan = scan_npz_headers(args.predictions)
    model_scan = scan_model_inputs(args.checkpoint)
    extraction = run_extraction(args, scene_scan)

    token_stats = build_token_layer_stats(args, source_scan, scene_scan, extraction)
    roi_placeholders = build_roi_placeholders(scene_scan)
    token_stats_path = output_dir / "token_layer_stats.json"
    roi_path = output_dir / "roi_coverage_placeholders.json"
    write_json(token_stats_path, token_stats)
    write_json(roi_path, roi_placeholders)
    summary = build_summary(
        args,
        source_scan,
        scene_scan,
        predictions_scan,
        model_scan,
        extraction,
        token_stats_path,
        roi_path,
    )
    summary_path = output_dir / "b_fus3d_token_cache_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(json_ready({"summary": summary_path, "status": summary["status"]}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
