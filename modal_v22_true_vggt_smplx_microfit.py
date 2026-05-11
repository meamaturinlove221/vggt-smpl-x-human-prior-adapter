from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
REPORTS = REPO_ROOT / "reports"
LOCAL_ROOT = REPO_ROOT / "output" / "surface_research_preflight_local"
CLOUD_ROOT = REPO_ROOT / "output" / "surface_research_cloud_preflight"
DEFAULT_LOCAL_DIR = LOCAL_ROOT / "V22_true_vggt_smplx_microfit"
DEFAULT_CLOUD_DIR = CLOUD_ROOT / "V22_true_vggt_smplx_microfit"
DEFAULT_REPORT_JSON = REPORTS / "20260508_v22_true_vggt_smplx_microfit.json"
DEFAULT_REPORT_MD = REPORTS / "20260508_v22_true_vggt_smplx_microfit.md"
DEFAULT_AUDIT_JSON = REPORTS / "20260508_v22_true_vggt_smplx_microfit.audit.json"
DEFAULT_AUDIT_MD = REPORTS / "20260508_v22_true_vggt_smplx_microfit.audit.md"

DEFAULT_EXISTING6_CASE = REPO_ROOT / "output" / "training_cases" / "0012_11_frame0000_6views_smplx_native_prior_v15"
DEFAULT_BALANCED12_CASE = (
    REPO_ROOT / "output" / "training_cases" / "0012_11_frame0000_12views_sparseproto_smplxsurfacepose_v2"
)

FORBIDDEN_OUTPUT_TOKENS = (
    "teacher_export",
    "candidate_export",
    "formal_candidate",
    "strict_gate_registry",
    "candidate_gate",
    "strict_pass",
)

RESEARCH_CONTRACT = {
    "research_only": True,
    "formal_candidate_train_infer_export": "blocked",
    "formal_cloud_unblocked": False,
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "no_formal_predictions_package_registry_or_pass": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_registry_write": True,
    "no_package_write": True,
    "no_strict_pass_write": True,
}

REQUIRED_CONTROLS = ("real", "zero", "shuffle", "random-region", "prior-dropout")


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    label: str
    trainable_mode: str
    lr: float
    steps_scale: float


@dataclass(frozen=True)
class ViewSetSpec:
    viewset_id: str
    label: str
    case_root: Path
    indices: tuple[int, ...]
    required: bool
    asset_note: str


METHODS = (
    MethodSpec("M2", "adapter + depth/point heads", "adapter_heads", 2.0e-4, 1.0),
    MethodSpec("M3", "low-lr bounded full microfit", "all", 2.0e-5, 1.0),
)

VIEWSETS = (
    ViewSetSpec(
        "existing6",
        "existing sparse six-view SMPL-X native case",
        DEFAULT_EXISTING6_CASE,
        (0, 1, 2, 3, 4, 5),
        True,
        "V15 native six-view prior case.",
    ),
    ViewSetSpec(
        "hand_head6",
        "six-view hand/head-support order",
        DEFAULT_EXISTING6_CASE,
        (2, 3, 4, 1, 5, 0),
        True,
        "Same V15 assets as existing6, ordered by V16 hand/head support: cams 15,30,45,01,59,00.",
    ),
    ViewSetSpec(
        "balanced12",
        "balanced twelve-view SMPL-X surfacepose case",
        DEFAULT_BALANCED12_CASE,
        tuple(range(12)),
        False,
        "Uses the available 12-view SMPL-X surfacepose prior case when present.",
    ),
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def file_row(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve() if path.exists() else path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "size": path.stat().st_size if path.is_file() else 0,
    }


def safe_v22_dir(path: Path, *, cloud: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    lower = resolved.as_posix().lower()
    required_root = "surface_research_cloud_preflight" if cloud else "surface_research_preflight_local"
    if required_root not in lower or "v22_true_vggt_smplx_microfit" not in lower:
        raise ValueError(f"Refusing non-V22 research output path: {resolved}")
    for token in FORBIDDEN_OUTPUT_TOKENS:
        if token in lower:
            raise ValueError(f"Refusing forbidden formal-output token {token!r}: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def safe_v22_report_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.parent != REPORTS.resolve():
        raise ValueError(f"Refusing report outside reports/: {resolved}")
    if not resolved.name.startswith("20260508_v22_true_vggt_smplx_microfit."):
        raise ValueError(f"Unexpected V22 report name: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def clean_previous_run_files(output_dir: Path) -> list[str]:
    cleaned: list[str] = []
    if output_dir.resolve() != DEFAULT_LOCAL_DIR.resolve():
        return cleaned
    for pattern in ("M*_*.metrics.json", "summary.json", "summary.md", "cloud_guard.json"):
        for path in output_dir.glob(pattern):
            if path.is_file() and path.parent.resolve() == output_dir.resolve():
                cleaned.append(str(path))
                path.unlink()
    return cleaned


def tensor_stats(tensor: Any, mask: Any | None = None) -> dict[str, Any]:
    import torch

    arr = tensor.detach().float()
    if mask is not None:
        mask_arr = mask.detach().bool()
        while mask_arr.ndim < arr.ndim:
            mask_arr = mask_arr.unsqueeze(-1)
        arr = arr[mask_arr.expand_as(arr)]
    else:
        arr = arr.reshape(-1)
    finite = arr[torch.isfinite(arr)]
    if int(finite.numel()) == 0:
        return {"count": int(arr.numel()), "finite": 0}
    return {
        "count": int(arr.numel()),
        "finite": int(finite.numel()),
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()) if int(finite.numel()) > 1 else 0.0,
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
    }


def masked_mean_abs(diff: Any, mask: Any) -> Any:
    import torch

    finite = torch.isfinite(diff)
    while mask.ndim < diff.ndim:
        mask = mask.unsqueeze(-1)
    valid = finite & mask.expand_as(diff)
    if int(valid.sum().item()) == 0:
        return diff.sum() * 0.0
    return diff[valid].abs().mean()


def probe_torch_environment() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "torch_import_ok": False,
            "torch_import_error": repr(exc),
            "bounded_local_route": "blocked_torch_import_error",
        }

    info: dict[str, Any] = {
        "torch_import_ok": True,
        "torch_version": torch.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "torch_arch_list": list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else [],
    }
    cuda_supported = False
    if torch.cuda.is_available():
        try:
            capability = torch.cuda.get_device_capability(0)
            arch = f"sm_{capability[0]}{capability[1]}"
            cuda_supported = arch in set(info["torch_arch_list"])
            info.update(
                {
                    "cuda_device_name": torch.cuda.get_device_name(0),
                    "cuda_capability": list(capability),
                    "cuda_arch": arch,
                    "cuda_arch_supported_by_torch": bool(cuda_supported),
                }
            )
        except Exception as exc:
            info["cuda_probe_error"] = repr(exc)
    info["bounded_local_route"] = "cuda_available_for_bounded_attempt" if cuda_supported else "cpu_bounded_true_module_route"
    return info


def choose_device(requested: str, env: dict[str, Any]) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if env.get("cuda_arch_supported_by_torch") else "cpu"
    return "cuda" if env.get("cuda_arch_supported_by_torch") else "cpu"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    arr = np.asarray(value)
    if arr.ndim == 0:
        return [str(arr.item())]
    return [str(item) for item in arr.tolist()]


def load_case_arrays(case_root: Path) -> dict[str, Any]:
    case_root = case_root.expanduser().resolve()
    inputs_path = case_root / "inputs.npz"
    targets_path = case_root / "targets.npz"
    manifest_path = case_root / "case_manifest.json"
    if not inputs_path.is_file():
        raise FileNotFoundError(f"inputs.npz not found: {inputs_path}")
    if not targets_path.is_file():
        raise FileNotFoundError(f"targets.npz not found: {targets_path}")
    manifest = read_json(manifest_path)
    return {
        "case_root": case_root,
        "inputs_path": inputs_path,
        "targets_path": targets_path,
        "manifest_path": manifest_path,
        "inputs": load_npz(inputs_path),
        "targets": load_npz(targets_path),
        "manifest": manifest,
    }


def infer_camera_ids(case: dict[str, Any], view_count: int) -> list[str]:
    manifest = case.get("manifest", {})
    inputs = case.get("inputs", {})
    camera_ids = as_string_list(manifest.get("camera_ids"))
    if not camera_ids and "camera_ids" in inputs:
        camera_ids = as_string_list(inputs["camera_ids"])
    if not camera_ids:
        camera_ids = [f"{idx:02d}" for idx in range(view_count)]
    return camera_ids[:view_count]


def validate_viewset_assets(spec: ViewSetSpec) -> dict[str, Any]:
    required = [
        spec.case_root / "inputs.npz",
        spec.case_root / "targets.npz",
        spec.case_root / "case_manifest.json",
    ]
    row = {
        "viewset_id": spec.viewset_id,
        "label": spec.label,
        "case_root": str(spec.case_root.resolve() if spec.case_root.exists() else spec.case_root),
        "required": spec.required,
        "asset_note": spec.asset_note,
        "files": {path.name: file_row(path) for path in required},
        "available": all(path.is_file() for path in required),
        "requested_indices": list(spec.indices),
        "blockers": [],
    }
    if not row["available"]:
        row["blockers"] = [f"Missing required viewset input: {path}" for path in required if not path.is_file()]
        return row
    try:
        with np.load(spec.case_root / "inputs.npz", allow_pickle=False) as inputs:
            view_count = int(inputs["images"].shape[0]) if "images" in inputs.files else 0
            row["available_view_count"] = view_count
            row["input_keys"] = list(inputs.files)
        with np.load(spec.case_root / "targets.npz", allow_pickle=False) as targets:
            row["target_keys"] = list(targets.files)
        if view_count <= max(spec.indices):
            row["available"] = False
            row["blockers"] = [f"Requested index {max(spec.indices)} but only {view_count} views are available."]
    except Exception as exc:
        row["available"] = False
        row["blockers"] = [repr(exc)]
    return row


def prepare_micro_batch(
    case: dict[str, Any],
    *,
    viewset: ViewSetSpec,
    target_size: int,
    device_name: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    inputs = case["inputs"]
    targets = case["targets"]
    required_inputs = ("images", "prior_maps", "prior_mask")
    required_targets = ("prior_depths", "prior_points")
    missing = [key for key in required_inputs if key not in inputs]
    missing.extend(key for key in required_targets if key not in targets)
    if missing:
        raise KeyError("V22 microfit case missing required arrays: " + ", ".join(missing))
    if target_size % 14 != 0:
        raise ValueError("--target-size must be divisible by VGGT patch size 14")

    view_count = int(inputs["images"].shape[0])
    indices = [int(idx) for idx in viewset.indices if int(idx) < view_count]
    if len(indices) != len(viewset.indices):
        raise ValueError(f"Viewset {viewset.viewset_id} is not available in case with {view_count} views.")
    device = torch.device(device_name)

    images_np = np.asarray(inputs["images"][indices])
    if images_np.ndim != 4 or images_np.shape[-1] != 3:
        raise ValueError(f"Expected images [V,H,W,3], got {images_np.shape}")
    images = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
    images = F.interpolate(images, size=(target_size, target_size), mode="bilinear", align_corners=False)

    prior_maps = torch.from_numpy(np.asarray(inputs["prior_maps"][indices], dtype=np.float32))
    prior_maps = F.interpolate(prior_maps, size=(target_size, target_size), mode="bilinear", align_corners=False)

    prior_mask = torch.from_numpy(np.asarray(inputs["prior_mask"][indices]).astype(np.float32)).unsqueeze(1)
    prior_mask = F.interpolate(prior_mask, size=(target_size, target_size), mode="nearest").squeeze(1) > 0.5

    prior_summary = None
    if "prior_summary_tokens" in inputs:
        prior_summary = torch.from_numpy(np.asarray(inputs["prior_summary_tokens"][indices], dtype=np.float32))

    target_depth = torch.from_numpy(np.asarray(targets["prior_depths"][indices], dtype=np.float32)).unsqueeze(1)
    target_depth = F.interpolate(target_depth, size=(target_size, target_size), mode="bilinear", align_corners=False)
    target_depth = target_depth.squeeze(1)

    target_points = torch.from_numpy(np.asarray(targets["prior_points"][indices], dtype=np.float32)).permute(0, 3, 1, 2)
    target_points = F.interpolate(target_points, size=(target_size, target_size), mode="bilinear", align_corners=False)
    target_points = target_points.permute(0, 2, 3, 1)

    valid = prior_mask & torch.isfinite(target_depth) & (target_depth > 0) & torch.isfinite(target_points).all(dim=-1)
    if int(valid.sum().item()) > 0:
        scale = torch.median(target_depth[valid]).float().clamp(min=1.0e-6)
    else:
        scale = torch.tensor(1.0)
    target_depth = torch.nan_to_num(target_depth / scale, nan=0.0, posinf=0.0, neginf=0.0)
    target_points = torch.nan_to_num(target_points / scale, nan=0.0, posinf=0.0, neginf=0.0)

    all_cameras = infer_camera_ids(case, view_count)
    selected_cameras = [all_cameras[idx] if idx < len(all_cameras) else str(idx) for idx in indices]
    batch = {
        "images": images.unsqueeze(0).to(device),
        "prior_maps_real": prior_maps.unsqueeze(0).to(device),
        "prior_summary_real": None if prior_summary is None else prior_summary.unsqueeze(0).to(device),
        "target_depth": target_depth.unsqueeze(0).unsqueeze(-1).to(device),
        "target_points": target_points.unsqueeze(0).to(device),
        "prior_mask": valid.unsqueeze(0).to(device),
        "viewset_id": viewset.viewset_id,
        "case_root": case["case_root"],
        "case_id": case["manifest"].get("case_id"),
        "view_count": len(indices),
        "view_indices": indices,
        "camera_ids": selected_cameras,
        "target_size": int(target_size),
        "depth_scale": float(scale.item()),
        "prior_channels": int(prior_maps.shape[1]),
        "prior_summary_channels": 0 if prior_summary is None else int(prior_summary.shape[-1]),
        "prior_summary_token_count": 0 if prior_summary is None else int(prior_summary.shape[1]),
        "valid_pixels": int(valid.sum().item()),
        "valid_pixels_per_view": [int(v) for v in valid.reshape(len(indices), -1).sum(dim=1).tolist()],
        "image_stats": tensor_stats(images),
        "prior_map_stats": tensor_stats(prior_maps),
        "target_depth_stats": tensor_stats(target_depth, valid),
        "target_point_stats": tensor_stats(target_points, valid),
    }
    return batch


class MicroVGGTDepthPointModel:
    @staticmethod
    def build(
        *,
        img_size: int,
        prior_channels: int,
        prior_summary_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        head_features: int,
    ) -> Any:
        import torch.nn as nn

        from vggt.heads.dpt_head import DPTHead
        from vggt.models.aggregator import Aggregator

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.aggregator = Aggregator(
                    img_size=img_size,
                    patch_size=14,
                    embed_dim=embed_dim,
                    depth=depth,
                    num_heads=num_heads,
                    mlp_ratio=2.0,
                    num_register_tokens=2,
                    patch_embed="conv",
                    qk_norm=True,
                    rope_freq=100,
                    human_prior_channels=prior_channels,
                    human_prior_summary_channels=prior_summary_channels,
                    human_prior_hidden_dim=max(16, head_features),
                    human_prior_gate_init=0.0,
                    human_prior_multi_scale_factors=(1, 2),
                    human_prior_enable_input_fusion=True,
                    human_prior_enable_frame_fusion=True,
                    human_prior_enable_global_fusion=True,
                    human_prior_enable_summary_fusion=prior_summary_channels > 0,
                )
                layer_ids = list(range(depth))
                out_channels = [head_features, head_features, head_features, head_features]
                self.depth_head = DPTHead(
                    dim_in=2 * embed_dim,
                    patch_size=14,
                    output_dim=2,
                    activation="exp",
                    conf_activation="expp1",
                    features=head_features,
                    out_channels=out_channels,
                    intermediate_layer_idx=layer_ids,
                    pos_embed=True,
                )
                self.point_head = DPTHead(
                    dim_in=2 * embed_dim,
                    patch_size=14,
                    output_dim=4,
                    activation="inv_log",
                    conf_activation="expp1",
                    features=head_features,
                    out_channels=out_channels,
                    intermediate_layer_idx=layer_ids,
                    pos_embed=True,
                )

            def forward(self, images, prior_maps=None, prior_summary_tokens=None):
                aggregated_tokens_list, patch_start_idx = self.aggregator(
                    images,
                    prior_maps=prior_maps,
                    prior_summary_tokens=prior_summary_tokens,
                )
                depth_pred, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                    frames_chunk_size=None,
                )
                points_pred, points_conf = self.point_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                    frames_chunk_size=None,
                )
                return {
                    "depth": depth_pred,
                    "depth_conf": depth_conf,
                    "world_points": points_pred,
                    "world_points_conf": points_conf,
                    "aggregated_token_layers": len(aggregated_tokens_list),
                    "patch_start_idx": int(patch_start_idx),
                }

        return _Model()


def apply_control(batch: dict[str, Any], control: str, *, seed: int, dropout_keep: float) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    prior_maps = batch["prior_maps_real"]
    prior_summary = batch["prior_summary_real"]
    meta: dict[str, Any] = {"control": control}
    if control == "real":
        meta["description"] = "Unmodified SMPL-X prior maps and summary tokens."
        return prior_maps, prior_summary, meta
    if control == "zero":
        meta["description"] = "All prior maps and summary tokens are zeroed."
        zero_summary = None if prior_summary is None else torch.zeros_like(prior_summary)
        return torch.zeros_like(prior_maps), zero_summary, meta
    if control == "shuffle":
        meta["description"] = "Prior maps and summary tokens are rolled across the view dimension."
        if int(prior_maps.shape[1]) <= 1:
            meta["shuffle_is_effective"] = False
            return prior_maps.clone(), None if prior_summary is None else prior_summary.clone(), meta
        meta["shuffle_is_effective"] = True
        shuffled_maps = torch.roll(prior_maps, shifts=1, dims=1)
        shuffled_summary = None if prior_summary is None else torch.roll(prior_summary, shifts=1, dims=1)
        return shuffled_maps, shuffled_summary, meta
    if control == "random-region":
        meta["description"] = "Only a deterministic random rectangle of each prior map is kept; summary tokens are zeroed."
        _, views, _, height, width = prior_maps.shape
        out = torch.zeros_like(prior_maps)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        kept_pixels = 0
        box_rows = []
        box_h = max(1, height // 3)
        box_w = max(1, width // 3)
        for view_idx in range(views):
            y0 = int(torch.randint(0, max(1, height - box_h + 1), (1,), generator=generator).item())
            x0 = int(torch.randint(0, max(1, width - box_w + 1), (1,), generator=generator).item())
            out[:, view_idx, :, y0 : y0 + box_h, x0 : x0 + box_w] = prior_maps[
                :, view_idx, :, y0 : y0 + box_h, x0 : x0 + box_w
            ]
            kept_pixels += int(box_h * box_w)
            box_rows.append({"view": int(view_idx), "y0": y0, "x0": x0, "height": box_h, "width": box_w})
        meta["region_boxes"] = box_rows
        meta["kept_pixel_fraction_per_channel"] = float(kept_pixels / max(1, views * height * width))
        zero_summary = None if prior_summary is None else torch.zeros_like(prior_summary)
        return out, zero_summary, meta
    if control == "prior-dropout":
        meta["description"] = "Deterministic Bernoulli dropout is applied to prior maps and summary tokens."
        keep = max(0.0, min(1.0, float(dropout_keep)))
        generator = torch.Generator(device=prior_maps.device)
        generator.manual_seed(int(seed))
        mask = (torch.rand(prior_maps.shape, generator=generator, device=prior_maps.device) < keep).float()
        dropped_maps = prior_maps * mask
        if prior_summary is None:
            dropped_summary = None
            summary_keep_fraction = None
        else:
            summary_mask = (torch.rand(prior_summary.shape, generator=generator, device=prior_summary.device) < keep).float()
            dropped_summary = prior_summary * summary_mask
            summary_keep_fraction = float(summary_mask.mean().item())
        meta["dropout_keep_requested"] = keep
        meta["map_keep_fraction"] = float(mask.mean().item())
        meta["summary_keep_fraction"] = summary_keep_fraction
        return dropped_maps, dropped_summary, meta
    raise ValueError(f"Unknown control: {control}")


def set_trainable(model: Any, trainable_mode: str) -> dict[str, Any]:
    if trainable_mode == "adapter_heads":
        patterns = ("aggregator.human_prior_adapter.", "depth_head.", "point_head.")
    elif trainable_mode == "all":
        patterns = ("",)
    else:
        raise ValueError(f"Unsupported V22 trainable mode: {trainable_mode}")

    total = 0
    trainable = 0
    groups: dict[str, int] = {"adapter": 0, "depth_head": 0, "point_head": 0, "aggregator_other": 0, "other": 0}
    trainable_names: list[str] = []
    for name, param in model.named_parameters():
        total += int(param.numel())
        enabled = any(name.startswith(pattern) for pattern in patterns)
        param.requires_grad_(enabled)
        if enabled:
            trainable += int(param.numel())
            trainable_names.append(name)
            if name.startswith("aggregator.human_prior_adapter."):
                groups["adapter"] += int(param.numel())
            elif name.startswith("depth_head."):
                groups["depth_head"] += int(param.numel())
            elif name.startswith("point_head."):
                groups["point_head"] += int(param.numel())
            elif name.startswith("aggregator."):
                groups["aggregator_other"] += int(param.numel())
            else:
                groups["other"] += int(param.numel())
    return {
        "mode": trainable_mode,
        "total_params": total,
        "trainable_params": trainable,
        "trainable_groups": groups,
        "trainable_name_preview": trainable_names[:24],
    }


def compute_losses(predictions: dict[str, Any], batch: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    depth_loss = masked_mean_abs(predictions["depth"] - batch["target_depth"], batch["prior_mask"])
    point_loss = masked_mean_abs(predictions["world_points"] - batch["target_points"], batch["prior_mask"])
    loss = depth_loss + point_loss
    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "depth_l1": float(depth_loss.detach().cpu().item()),
        "point_l1": float(point_loss.detach().cpu().item()),
    }


def gate_stats(model: Any) -> dict[str, Any]:
    import torch

    values = []
    for name, param in model.named_parameters():
        if "human_prior_adapter" in name and name.endswith(".gate"):
            values.append(param.detach().reshape(-1).float().cpu())
    if not values:
        return {"gate_count": 0}
    merged = torch.cat(values)
    return {
        "gate_count": int(merged.numel()),
        "mean": float(merged.mean().item()),
        "mean_abs": float(merged.abs().mean().item()),
        "max_abs": float(merged.abs().max().item()),
    }


def run_one(
    *,
    method: MethodSpec,
    control: str,
    batch: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = MicroVGGTDepthPointModel.build(
        img_size=int(args.target_size),
        prior_channels=int(batch["prior_channels"]),
        prior_summary_channels=int(batch["prior_summary_channels"]),
        embed_dim=int(args.embed_dim),
        depth=int(args.depth),
        num_heads=int(args.num_heads),
        head_features=int(args.head_features),
    ).to(batch["images"].device)
    model.train()
    trainable = set_trainable(model, method.trainable_mode)
    prior_maps, prior_summary, control_meta = apply_control(
        batch,
        control,
        seed=seed + 17,
        dropout_keep=float(args.prior_dropout_keep),
    )

    def forward_loss() -> tuple[Any, dict[str, Any], dict[str, Any]]:
        preds = model(batch["images"], prior_maps=prior_maps, prior_summary_tokens=prior_summary)
        loss, loss_row = compute_losses(preds, batch)
        meta = {
            "depth_shape": [int(v) for v in preds["depth"].shape],
            "world_points_shape": [int(v) for v in preds["world_points"].shape],
            "aggregated_token_layers": int(preds["aggregated_token_layers"]),
            "patch_start_idx": int(preds["patch_start_idx"]),
        }
        return loss, loss_row, meta

    with torch.no_grad():
        initial_loss, initial_metrics, output_meta = forward_loss()

    step_count = max(0, int(round(float(args.steps) * method.steps_scale)))
    history: list[dict[str, Any]] = []
    if trainable["trainable_params"] > 0 and step_count > 0:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(method.lr),
            weight_decay=float(args.weight_decay),
        )
        for step in range(step_count):
            optimizer.zero_grad(set_to_none=True)
            loss, loss_row, _ = forward_loss()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=float(args.grad_clip),
            )
            optimizer.step()
            history.append(
                {
                    "step": int(step + 1),
                    **loss_row,
                    "grad_norm": float(grad_norm.detach().cpu().item()),
                }
            )

    model.eval()
    with torch.no_grad():
        final_loss, final_metrics, _ = forward_loss()

    row = {
        "task": "v22_true_vggt_smplx_microfit_metric",
        "research_only": True,
        "method_id": method.method_id,
        "method_label": method.label,
        "viewset_id": batch["viewset_id"],
        "control": control,
        "seed": int(seed),
        "trainable": trainable,
        "lr": float(method.lr),
        "requested_steps": int(args.steps),
        "executed_steps": int(step_count),
        "control_meta": control_meta,
        "batch": {
            key: value
            for key, value in batch.items()
            if key
            not in {
                "images",
                "prior_maps_real",
                "prior_summary_real",
                "target_depth",
                "target_points",
                "prior_mask",
            }
        },
        "initial": initial_metrics,
        "final": final_metrics,
        "loss_delta": float(final_metrics["loss"] - initial_metrics["loss"]),
        "history": history,
        "output_meta": output_meta,
        "gate_stats": gate_stats(model),
    }
    run_path = output_dir / f"{method.method_id}_{batch['viewset_id']}_{control}.metrics.json"
    write_json(run_path, row)
    row["metrics_path"] = str(run_path.resolve())
    del model
    if batch["images"].device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def compare_controls(results: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in results:
        by_key.setdefault((row["method_id"], row["viewset_id"]), {})[row["control"]] = row
    comparison: dict[str, Any] = {}
    any_positive = False
    for (method_id, viewset_id), rows in sorted(by_key.items()):
        key = f"{method_id}:{viewset_id}"
        real = rows.get("real")
        if not real:
            comparison[key] = {"available": False, "reason": "real control missing"}
            continue
        real_loss = float(real["final"]["loss"])
        controls = {name: row for name, row in rows.items() if name != "real"}
        deltas = {name: float(row["final"]["loss"] - real_loss) for name, row in controls.items()}
        beats = {name: bool(delta > float(tolerance)) for name, delta in deltas.items()}
        positive = bool(controls) and all(beats.values())
        any_positive = any_positive or positive
        comparison[key] = {
            "available": True,
            "method_id": method_id,
            "viewset_id": viewset_id,
            "real_final_loss": real_loss,
            "control_final_loss": {name: float(row["final"]["loss"]) for name, row in controls.items()},
            "control_minus_real_loss": deltas,
            "real_beats_each_control": beats,
            "real_beats_all_controls": positive,
            "tolerance": float(tolerance),
        }
    comparison["any_m2_m3_viewset_control_positive"] = bool(any_positive)
    return comparison


def inspect_code_paths() -> dict[str, Any]:
    checks = {
        "vggt_forward_prior_args": (REPO_ROOT / "vggt" / "models" / "vggt.py", "prior_maps: torch.Tensor = None"),
        "vggt_forward_aggregator_prior_call": (REPO_ROOT / "vggt" / "models" / "vggt.py", "prior_maps=prior_maps"),
        "aggregator_human_prior_adapter": (REPO_ROOT / "vggt" / "models" / "aggregator.py", "HumanPriorAdapter"),
        "aggregator_prior_forward": (REPO_ROOT / "vggt" / "models" / "aggregator.py", "prior_maps=prior_maps"),
        "dpt_head_dense_route": (REPO_ROOT / "vggt" / "heads" / "dpt_head.py", "class DPTHead"),
        "trainer_prior_plumbing": (REPO_ROOT / "training" / "trainer.py", "prior_summary_tokens=batch.get"),
    }
    rows: dict[str, Any] = {}
    for label, (path, needle) in checks.items():
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        rows[label] = {"file": str(path), "needle": needle, "present": needle in text}
    return rows


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V22 True VGGT SMPL-X Microfit",
        "",
        f"Status: `{summary['final_status']}`",
        "",
        "Research-only bounded M2/M3 run. It uses VGGT Aggregator, HumanPriorAdapter, and DPTHead modules, and it writes no formal predictions, package, registry, teacher export, candidate export, or strict pass state.",
        "",
        "## Execution",
        "",
        f"- executed: `{summary['execution'].get('executed')}`",
        f"- device: `{summary['execution'].get('device')}`",
        f"- elapsed_sec: `{summary['execution'].get('elapsed_sec')}`",
        f"- metric_file_count: `{summary['execution'].get('metric_file_count')}`",
        f"- methods: `{summary['required_methods']}`",
        f"- controls: `{summary['required_controls']}`",
        "",
        "## View Sets",
        "",
        "| View set | Available | Executed | Case | Cameras | Valid pixels |",
        "|---|---:|---:|---|---|---:|",
    ]
    for row in summary.get("viewset_runs", []):
        lines.append(
            f"| {row['viewset_id']} | {row.get('available')} | {row.get('executed')} | "
            f"`{row.get('case_root')}` | {' '.join(row.get('camera_ids', []))} | {row.get('valid_pixels')} |"
        )
    lines.extend(["", "## Control Comparison", "", "```json"])
    lines.append(json.dumps(json_ready(summary.get("comparison", {})), indent=2, ensure_ascii=False, sort_keys=True))
    lines.extend(
        [
            "```",
            "",
            "## Outputs",
            "",
        ]
    )
    for key, value in summary.get("outputs", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Decision", "", summary.get("decision", ""), "", "## Blockers", ""])
    blockers = summary.get("blockers") or []
    lines.extend([f"- {item}" for item in blockers] if blockers else ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_cloud_guard(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    guard = {
        "task": "v22_true_vggt_smplx_microfit_cloud_guard",
        "created_utc": utc_now(),
        **RESEARCH_CONTRACT,
        "status": "cloud_not_required_local_bounded_job_executed"
        if summary.get("execution", {}).get("executed")
        else "cloud_route_available_but_not_executed",
        "local_report_json": str(DEFAULT_REPORT_JSON.resolve()),
        "local_output_dir": str(DEFAULT_LOCAL_DIR.resolve()),
        "modal_entrypoint": "modal_v22_true_vggt_smplx_microfit.py::run_modal_v22_microfit",
        "decision": "V22 keeps a cloud guard under the owned cloud preflight path. The concrete bounded job in this turn ran locally unless the Modal entrypoint is invoked separately.",
    }
    write_json(path / "cloud_guard.json", guard)
    (path / "summary.md").write_text(
        "\n".join(
            [
                "# V22 Cloud Guard",
                "",
                f"Status: `{guard['status']}`",
                "",
                guard["decision"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return guard


def run_local(args: argparse.Namespace) -> dict[str, Any]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    local_dir = safe_v22_dir(args.output_dir, cloud=False)
    cloud_dir = safe_v22_dir(args.cloud_output_dir, cloud=True)
    report_json = safe_v22_report_path(args.report_json)
    report_md = safe_v22_report_path(args.report_md)
    audit_json = safe_v22_report_path(args.audit_json)
    audit_md = safe_v22_report_path(args.audit_md)
    cleaned = clean_previous_run_files(local_dir) if args.clean else []

    env = probe_torch_environment()
    code_paths = inspect_code_paths()
    viewset_assets = [validate_viewset_assets(spec) for spec in VIEWSETS]
    blockers: list[str] = []
    if not all(row["present"] for row in code_paths.values()):
        blockers.append("One or more VGGT prior path inspection needles were not found.")
    for row in viewset_assets:
        if row["required"] and not row["available"]:
            blockers.extend(row["blockers"])

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    viewset_runs: list[dict[str, Any]] = []
    started = time.time()
    execution = {
        "executed": bool(args.execute),
        "cleaned_previous_files": cleaned,
        "device": None,
        "torch_threads": int(args.torch_threads),
        "target_size": int(args.target_size),
        "steps": int(args.steps),
        "embed_dim": int(args.embed_dim),
        "depth": int(args.depth),
        "num_heads": int(args.num_heads),
        "head_features": int(args.head_features),
    }

    selected_controls = [item.strip() for item in args.controls.split(",") if item.strip()]
    unknown_controls = [item for item in selected_controls if item not in REQUIRED_CONTROLS]
    if unknown_controls:
        raise ValueError(f"Unknown controls: {unknown_controls}")

    if args.execute and not env.get("torch_import_ok"):
        blockers.append("Torch import failed; V22 local bounded microfit could not execute.")
    elif args.execute:
        import torch

        torch.set_num_threads(max(1, int(args.torch_threads)))
        device_name = choose_device(args.device, env)
        execution["device"] = device_name
        for spec, asset in zip(VIEWSETS, viewset_assets):
            if not asset.get("available"):
                if spec.required:
                    blockers.extend(asset.get("blockers", []))
                viewset_runs.append(
                    {
                        "viewset_id": spec.viewset_id,
                        "available": False,
                        "executed": False,
                        "case_root": asset.get("case_root"),
                        "blockers": asset.get("blockers", []),
                    }
                )
                continue
            try:
                case = load_case_arrays(spec.case_root)
                batch = prepare_micro_batch(case, viewset=spec, target_size=int(args.target_size), device_name=device_name)
                viewset_runs.append(
                    {
                        "viewset_id": spec.viewset_id,
                        "available": True,
                        "executed": True,
                        "case_root": str(case["case_root"]),
                        "case_id": batch.get("case_id"),
                        "view_indices": batch.get("view_indices"),
                        "camera_ids": batch.get("camera_ids"),
                        "valid_pixels": batch.get("valid_pixels"),
                        "prior_channels": batch.get("prior_channels"),
                        "prior_summary_channels": batch.get("prior_summary_channels"),
                    }
                )
                for method_idx, method in enumerate(METHODS):
                    for control_idx, control in enumerate(selected_controls):
                        seed = int(args.seed) + len(results) * 101 + method_idx * 17 + control_idx
                        results.append(
                            run_one(
                                method=method,
                                control=control,
                                batch=batch,
                                args=args,
                                output_dir=local_dir,
                                seed=seed,
                            )
                        )
            except Exception as exc:
                error = {"viewset_id": spec.viewset_id, "error": repr(exc)}
                errors.append(error)
                blockers.append(f"Viewset {spec.viewset_id} failed during bounded run: {exc!r}")
                viewset_runs.append(
                    {
                        "viewset_id": spec.viewset_id,
                        "available": True,
                        "executed": False,
                        "case_root": str(spec.case_root),
                        "blockers": [repr(exc)],
                    }
                )
    execution["elapsed_sec"] = round(time.time() - started, 3)
    execution["metric_file_count"] = len(results)

    comparison = compare_controls(results, float(args.tolerance)) if results else {
        "available": False,
        "reason": "No bounded microfit metrics were produced.",
        "any_m2_m3_viewset_control_positive": False,
    }

    required_viewsets = [row["viewset_id"] for row in viewset_assets if row.get("available") or row.get("required")]
    required_metric_count = len(METHODS) * len(selected_controls) * len(required_viewsets)
    full_contract_executed = (
        bool(args.execute)
        and len(results) >= required_metric_count
        and not errors
        and all(row.get("executed") for row in viewset_runs if row.get("viewset_id") in required_viewsets)
    )
    if full_contract_executed:
        final_status = "DONE_PASS"
        decision = (
            "V22 executed the bounded local research job for M2/M3 across available required view sets and all required controls. "
            "This is a research-contract pass only, not a formal model/candidate pass."
        )
    elif results:
        final_status = "DONE_FAIL_ROUTED"
        decision = (
            "V22 produced concrete bounded metrics but did not complete every required method/control/view-set cell; keep it fail-closed and route the missing cells to the Modal entrypoint."
        )
    else:
        final_status = "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE"
        decision = "V22 could not produce any bounded local microfit metrics; see blockers and environment evidence."

    summary = {
        "task": "v22_true_vggt_smplx_microfit",
        "created_utc": utc_now(),
        "final_status": final_status,
        **RESEARCH_CONTRACT,
        "environment": env,
        "code_path_inspection": code_paths,
        "viewset_assets": viewset_assets,
        "viewset_runs": viewset_runs,
        "required_methods": [item.method_id for item in METHODS],
        "method_specs": [
            {
                "method_id": item.method_id,
                "label": item.label,
                "trainable_mode": item.trainable_mode,
                "lr": item.lr,
                "steps_scale": item.steps_scale,
            }
            for item in METHODS
        ],
        "required_controls": list(REQUIRED_CONTROLS),
        "selected_controls": selected_controls,
        "execution": execution,
        "results": results,
        "comparison": comparison,
        "errors": errors,
        "blockers": blockers,
        "decision": decision,
        "outputs": {
            "local_output_dir": str(local_dir),
            "cloud_output_dir": str(cloud_dir),
            "report_json": str(report_json),
            "report_md": str(report_md),
            "audit_json": str(audit_json),
            "audit_md": str(audit_md),
            "modal_file": str((REPO_ROOT / "modal_v22_true_vggt_smplx_microfit.py").resolve()),
            "auditor": str((REPO_ROOT / "tools" / "v22_microfit_result_auditor.py").resolve()),
        },
    }
    cloud_guard = write_cloud_guard(cloud_dir, summary)
    summary["cloud_guard"] = cloud_guard

    write_json(report_json, summary)
    write_markdown(report_md, summary)
    write_json(local_dir / "summary.json", summary)
    write_markdown(local_dir / "summary.md", summary)
    print(
        json.dumps(
            json_ready(
                {
                    "final_status": final_status,
                    "executed": bool(args.execute),
                    "metric_file_count": len(results),
                    "viewsets": [row.get("viewset_id") for row in viewset_runs if row.get("executed")],
                    "report_json": report_json,
                    "local_output_dir": local_dir,
                }
            ),
            ensure_ascii=False,
        )
    )
    return summary


def run_auditor(args: argparse.Namespace) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "v22_microfit_result_auditor.py"),
        "--report-json",
        str(args.report_json),
        "--output-json",
        str(args.audit_json),
        "--output-md",
        str(args.audit_md),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip(), file=sys.stderr)
    return {"returncode": int(proc.returncode), "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:]}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V22 bounded true VGGT SMPL-X M2/M3 microfit research runner.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--cloud-output-dir", type=Path, default=DEFAULT_CLOUD_DIR)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--audit-md", type=Path, default=DEFAULT_AUDIT_MD)
    parser.add_argument("--target-size", type=int, default=28)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-features", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2200)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--controls", default=",".join(REQUIRED_CONTROLS))
    parser.add_argument("--prior-dropout-keep", type=float, default=0.5)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--audit", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_local(args)
    audit_result = None
    if args.audit:
        audit_result = run_auditor(args)
    return 0 if summary.get("final_status") in {"DONE_PASS", "DONE_FAIL_ROUTED"} and not (audit_result or {}).get("returncode") else int((audit_result or {}).get("returncode", 0))


# Optional Modal route for the same research-only bounded job. Local execution does not require modal.
try:
    import modal  # type: ignore
except Exception:
    modal = None


if modal is not None:
    REMOTE_CODE_DIR = PurePosixPath("/workspace/vggt")
    REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")
    OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
    APP_NAME = os.environ.get("VGGT_MODAL_V22_APP_NAME", "vggt-v22-true-vggt-smplx-microfit")
    TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_V22_TIMEOUT_SEC", str(2 * 60 * 60)))
    CODE_SYNC_IGNORE = [".git", ".git/**", "__pycache__", "__pycache__/**", ".venv*", ".venv*/**", "output", "output/**"]

    IMAGE = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
        .pip_install("numpy==1.26.4", "torch==2.3.1", "torchvision==0.18.1")
        .add_local_file(str(REPO_ROOT / "modal_v22_true_vggt_smplx_microfit.py"), remote_path=(REMOTE_CODE_DIR / "modal_v22_true_vggt_smplx_microfit.py").as_posix())
        .add_local_file(str(REPO_ROOT / "tools" / "v22_microfit_result_auditor.py"), remote_path=(REMOTE_CODE_DIR / "tools" / "v22_microfit_result_auditor.py").as_posix())
        .add_local_dir(str(REPO_ROOT / "vggt"), remote_path=(REMOTE_CODE_DIR / "vggt").as_posix(), ignore=CODE_SYNC_IGNORE)
    )
    app = modal.App(APP_NAME)
    output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)

    @app.function(image=IMAGE, cpu=4.0, memory=24 * 1024, timeout=TIMEOUT_SEC, volumes={REMOTE_OUTPUT_DIR.as_posix(): output_volume})
    def run_modal_v22_microfit(remote_case_root: str = "", target_size: int = 28, steps: int = 1) -> dict[str, Any]:
        output_dir = Path(str(REMOTE_OUTPUT_DIR / "surface_research_cloud_preflight/V22_true_vggt_smplx_microfit"))
        output_dir.mkdir(parents=True, exist_ok=True)
        guard = {
            "task": "v22_modal_research_guard",
            "created_utc": utc_now(),
            **RESEARCH_CONTRACT,
            "remote_case_root": remote_case_root,
            "remote_output_dir": str(output_dir),
        }
        write_json(output_dir / "modal_guard.json", guard)
        cmd = [
            sys.executable,
            str(Path(str(REMOTE_CODE_DIR)) / "modal_v22_true_vggt_smplx_microfit.py"),
            "--output-dir",
            str(output_dir),
            "--cloud-output-dir",
            str(output_dir),
            "--report-json",
            str(output_dir / "20260508_v22_true_vggt_smplx_microfit.json"),
            "--report-md",
            str(output_dir / "20260508_v22_true_vggt_smplx_microfit.md"),
            "--audit-json",
            str(output_dir / "20260508_v22_true_vggt_smplx_microfit.audit.json"),
            "--audit-md",
            str(output_dir / "20260508_v22_true_vggt_smplx_microfit.audit.md"),
            "--target-size",
            str(int(target_size)),
            "--steps",
            str(int(steps)),
            "--execute",
            "--audit",
        ]
        proc = subprocess.run(cmd, cwd=str(REMOTE_CODE_DIR), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        result = {
            **guard,
            "returncode": int(proc.returncode),
            "stdout_tail": proc.stdout[-8000:],
            "stderr_tail": proc.stderr[-8000:],
            "status": "completed_research_only" if proc.returncode == 0 else "failed_research_only",
        }
        write_json(output_dir / "modal_summary.json", result)
        output_volume.commit()
        if proc.returncode != 0:
            raise RuntimeError(json.dumps(json_ready(result), indent=2, ensure_ascii=False))
        return result

    @app.local_entrypoint()
    def run(target_size: int = 28, steps: int = 1) -> None:
        print(json.dumps(run_modal_v22_microfit.remote("", int(target_size), int(steps)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
