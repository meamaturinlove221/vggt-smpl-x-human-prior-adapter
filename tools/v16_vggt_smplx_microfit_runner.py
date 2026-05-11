from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORTS = REPO_ROOT / "reports"
LOCAL_ROOT = REPO_ROOT / "output" / "surface_research_preflight_local"
DEFAULT_OUTPUT_DIR = LOCAL_ROOT / "V16_vggt_smplx_microfit_runner"
DEFAULT_CASE_ROOT = (
    REPO_ROOT
    / "output"
    / "training_cases"
    / "0012_11_frame0000_6views_smplx_native_prior_v15"
)
DEFAULT_OUTPUT_JSON = REPORTS / "20260508_v16_vggt_smplx_microfit_runner.json"
DEFAULT_OUTPUT_MD = REPORTS / "20260508_v16_vggt_smplx_microfit_runner.md"
DEFAULT_ROLLUP_JSON = REPORTS / "20260508_v16_execution_rollup.json"
DEFAULT_ROLLUP_MD = REPORTS / "20260508_v16_execution_rollup.md"
DEFAULT_V17_STUB_MD = REPORTS / "20260508_v17_plan_stub.md"

FORBIDDEN_OUTPUT_TOKENS = (
    "predictions",
    "teacher_export",
    "candidate_export",
    "formal_candidate",
    "strict_gate_registry",
    "strict_pass",
    "candidate_gate",
)

RESEARCH_CONTRACT = {
    "research_only": True,
    "formal_candidate_train_infer_export": "blocked",
    "formal_cloud_unblocked": False,
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "no_predictions_npz_formal_path": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_registry_write": True,
    "no_package_write": True,
    "no_strict_pass_write": True,
}


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    label: str
    trainable_mode: str
    lr: float
    steps_scale: float


METHODS = (
    MethodSpec("M0", "no-train baseline", "none", 0.0, 0.0),
    MethodSpec("M1", "adapter-only", "adapter", 5.0e-4, 1.0),
    MethodSpec("M2", "adapter + depth/point heads", "adapter_heads", 2.0e-4, 1.0),
    MethodSpec("M3", "low-lr bounded full microfit", "all", 2.0e-5, 1.0),
)

CONTROLS = ("real", "zero", "shuffle")


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def safe_v16_research_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    lower = resolved.as_posix().lower()
    if "surface_research" not in lower or "v16_vggt_smplx_microfit_runner" not in lower:
        raise ValueError(f"Refusing non-V16 research output path: {resolved}")
    for token in FORBIDDEN_OUTPUT_TOKENS:
        if token in lower:
            raise ValueError(f"Refusing forbidden output token {token!r}: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def safe_report_path(path: Path, stem_prefix: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.parent != REPORTS.resolve() and "surface_research" not in resolved.as_posix().lower():
        raise ValueError(f"Refusing report path outside reports or research output root: {resolved}")
    if not resolved.name.startswith(stem_prefix) and "v16_vggt_smplx_microfit_runner" not in resolved.as_posix().lower():
        raise ValueError(f"Unexpected V16 report name: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


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
            "full_local_vggt_status": "blocked_torch_import_error_cpu_smoke_unavailable",
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
            gpu_name = torch.cuda.get_device_name(0)
            arch = f"sm_{capability[0]}{capability[1]}"
            cuda_supported = arch in set(info["torch_arch_list"])
            info.update(
                {
                    "cuda_device_name": gpu_name,
                    "cuda_capability": list(capability),
                    "cuda_arch": arch,
                    "cuda_arch_supported_by_torch": bool(cuda_supported),
                }
            )
        except Exception as exc:
            info["cuda_probe_error"] = repr(exc)
    info["full_local_vggt_status"] = (
        "available_for_cuda_attempt"
        if cuda_supported
        else "blocked_cuda_arch_or_cuda_unavailable_cpu_micro_smoke_route"
    )
    return info


def choose_device(requested: str, env: dict[str, Any]) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if env.get("cuda_arch_supported_by_torch"):
            return "cuda"
        return "cpu"
    if env.get("cuda_arch_supported_by_torch"):
        return "cuda"
    return "cpu"


def load_case_arrays(case_root: Path) -> dict[str, Any]:
    case_root = case_root.expanduser().resolve()
    inputs_path = case_root / "inputs.npz"
    targets_path = case_root / "targets.npz"
    manifest_path = case_root / "case_manifest.json"
    if not inputs_path.is_file():
        raise FileNotFoundError(f"inputs.npz not found: {inputs_path}")
    if not targets_path.is_file():
        raise FileNotFoundError(f"targets.npz not found: {targets_path}")

    with np.load(inputs_path, allow_pickle=False) as inputs_npz:
        inputs = {key: inputs_npz[key] for key in inputs_npz.files}
    with np.load(targets_path, allow_pickle=False) as targets_npz:
        targets = {key: targets_npz[key] for key in targets_npz.files}
    manifest = read_json(manifest_path)
    return {
        "case_root": case_root,
        "inputs_path": inputs_path,
        "targets_path": targets_path,
        "manifest_path": manifest_path,
        "inputs": inputs,
        "targets": targets,
        "manifest": manifest,
    }


def prepare_micro_batch(
    case: dict[str, Any],
    *,
    max_views: int,
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
        raise KeyError("V16 microfit case missing required arrays: " + ", ".join(missing))

    view_count = int(min(max_views, inputs["images"].shape[0], inputs["prior_maps"].shape[0]))
    if view_count <= 0:
        raise ValueError("No views available for V16 microfit")
    if target_size % 14 != 0:
        raise ValueError("--target-size must be divisible by VGGT patch size 14")

    device = torch.device(device_name)
    images_np = np.asarray(inputs["images"][:view_count])
    if images_np.ndim != 4 or images_np.shape[-1] != 3:
        raise ValueError(f"Expected images [V,H,W,3], got {images_np.shape}")
    images = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
    images = F.interpolate(images, size=(target_size, target_size), mode="bilinear", align_corners=False)

    prior_maps = torch.from_numpy(np.asarray(inputs["prior_maps"][:view_count], dtype=np.float32))
    prior_maps = F.interpolate(prior_maps, size=(target_size, target_size), mode="bilinear", align_corners=False)

    prior_mask = torch.from_numpy(np.asarray(inputs["prior_mask"][:view_count]).astype(np.float32)).unsqueeze(1)
    prior_mask = F.interpolate(prior_mask, size=(target_size, target_size), mode="nearest").squeeze(1) > 0.5

    prior_summary = None
    if "prior_summary_tokens" in inputs:
        prior_summary = torch.from_numpy(np.asarray(inputs["prior_summary_tokens"][:view_count], dtype=np.float32))

    target_depth = torch.from_numpy(np.asarray(targets["prior_depths"][:view_count], dtype=np.float32)).unsqueeze(1)
    target_depth = F.interpolate(target_depth, size=(target_size, target_size), mode="bilinear", align_corners=False)
    target_depth = target_depth.squeeze(1)

    target_points = torch.from_numpy(np.asarray(targets["prior_points"][:view_count], dtype=np.float32))
    target_points = target_points.permute(0, 3, 1, 2)
    target_points = F.interpolate(target_points, size=(target_size, target_size), mode="bilinear", align_corners=False)
    target_points = target_points.permute(0, 2, 3, 1)

    valid = (
        prior_mask
        & torch.isfinite(target_depth)
        & (target_depth > 0)
        & torch.isfinite(target_points).all(dim=-1)
    )
    if int(valid.sum().item()) > 0:
        scale = torch.median(target_depth[valid]).float().clamp(min=1.0e-6)
    else:
        scale = torch.tensor(1.0)
    target_depth = torch.nan_to_num(target_depth / scale, nan=0.0, posinf=0.0, neginf=0.0)
    target_points = torch.nan_to_num(target_points / scale, nan=0.0, posinf=0.0, neginf=0.0)

    batch = {
        "images": images.unsqueeze(0).to(device),
        "prior_maps_real": prior_maps.unsqueeze(0).to(device),
        "prior_summary_real": None if prior_summary is None else prior_summary.unsqueeze(0).to(device),
        "target_depth": target_depth.unsqueeze(0).unsqueeze(-1).to(device),
        "target_points": target_points.unsqueeze(0).to(device),
        "prior_mask": valid.unsqueeze(0).to(device),
        "view_count": view_count,
        "target_size": int(target_size),
        "depth_scale": float(scale.item()),
        "prior_channels": int(prior_maps.shape[1]),
        "prior_summary_channels": 0 if prior_summary is None else int(prior_summary.shape[-1]),
        "prior_summary_token_count": 0 if prior_summary is None else int(prior_summary.shape[1]),
        "valid_pixels": int(valid.sum().item()),
        "valid_pixels_per_view": [int(v) for v in valid.reshape(view_count, -1).sum(dim=1).tolist()],
        "image_stats": tensor_stats(images),
        "prior_map_stats": tensor_stats(prior_maps),
        "target_depth_stats": tensor_stats(target_depth, valid),
        "target_point_stats": tensor_stats(target_points, valid),
    }
    return batch


class MicroVGGTDepthPointModel:  # thin factory wrapper keeps imports local to execution.
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


def apply_control(batch: dict[str, Any], control: str) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    prior_maps = batch["prior_maps_real"]
    prior_summary = batch["prior_summary_real"]
    meta = {"control": control, "shuffle_is_effective": None}
    if control == "real":
        return prior_maps, prior_summary, meta
    if control == "zero":
        zero_summary = None if prior_summary is None else torch.zeros_like(prior_summary)
        return torch.zeros_like(prior_maps), zero_summary, meta
    if control == "shuffle":
        if int(prior_maps.shape[1]) <= 1:
            meta["shuffle_is_effective"] = False
            return prior_maps.clone(), None if prior_summary is None else prior_summary.clone(), meta
        meta["shuffle_is_effective"] = True
        shuffled_maps = torch.roll(prior_maps, shifts=1, dims=1)
        shuffled_summary = None if prior_summary is None else torch.roll(prior_summary, shifts=1, dims=1)
        return shuffled_maps, shuffled_summary, meta
    raise ValueError(f"Unknown control: {control}")


def set_trainable(model: Any, trainable_mode: str) -> dict[str, Any]:
    patterns: tuple[str, ...]
    if trainable_mode == "none":
        patterns = ()
    elif trainable_mode == "adapter":
        patterns = ("aggregator.human_prior_adapter.",)
    elif trainable_mode == "adapter_heads":
        patterns = ("aggregator.human_prior_adapter.", "depth_head.", "point_head.")
    elif trainable_mode == "all":
        patterns = ("",)
    else:
        raise ValueError(f"Unknown trainable mode: {trainable_mode}")

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
    model.eval()
    trainable = set_trainable(model, method.trainable_mode)
    prior_maps, prior_summary, control_meta = apply_control(batch, control)

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

    step_count = int(round(float(args.steps) * method.steps_scale))
    history: list[dict[str, Any]] = []
    optimizer = None
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
            row = {"step": int(step + 1), **loss_row, "grad_norm": float(grad_norm.detach().cpu().item())}
            history.append(row)

    with torch.no_grad():
        final_loss, final_metrics, _ = forward_loss()

    row = {
        "method_id": method.method_id,
        "method_label": method.label,
        "control": control,
        "seed": int(seed),
        "trainable": trainable,
        "lr": float(method.lr),
        "requested_steps": int(args.steps),
        "executed_steps": int(step_count if trainable["trainable_params"] > 0 else 0),
        "control_meta": control_meta,
        "initial": initial_metrics,
        "final": final_metrics,
        "loss_delta": float(final_metrics["loss"] - initial_metrics["loss"]),
        "history": history,
        "output_meta": output_meta,
        "gate_stats": gate_stats(model),
    }
    run_path = output_dir / f"{method.method_id}_{control}_metrics.json"
    write_json(run_path, row)
    row["metrics_path"] = run_path
    del model
    if batch["images"].device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def compare_controls(results: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for row in results:
        by_method.setdefault(row["method_id"], {})[row["control"]] = row
    comparison: dict[str, Any] = {}
    any_positive = False
    for method_id, rows in by_method.items():
        real = rows.get("real")
        if not real:
            comparison[method_id] = {"available": False, "reason": "real control missing"}
            continue
        real_loss = float(real["final"]["loss"])
        control_rows = {key: value for key, value in rows.items() if key != "real"}
        deltas = {
            key: float(value["final"]["loss"] - real_loss)
            for key, value in control_rows.items()
        }
        beats = {
            key: bool(delta > float(tolerance))
            for key, delta in deltas.items()
        }
        positive = bool(control_rows) and all(beats.values())
        any_positive = any_positive or (method_id in {"M1", "M2", "M3"} and positive)
        comparison[method_id] = {
            "available": True,
            "real_final_loss": real_loss,
            "control_final_loss": {
                key: float(value["final"]["loss"])
                for key, value in control_rows.items()
            },
            "control_minus_real_loss": deltas,
            "real_beats_all_controls": positive,
            "tolerance": float(tolerance),
        }
    comparison["any_trainable_method_control_positive"] = bool(any_positive)
    return comparison


def inspect_code_paths() -> dict[str, Any]:
    checks = {
        "vggt_forward_prior_maps": (
            REPO_ROOT / "vggt" / "models" / "vggt.py",
            "prior_maps=batch.get",
        ),
        "vggt_forward_prior_args": (
            REPO_ROOT / "vggt" / "models" / "vggt.py",
            "prior_maps: torch.Tensor = None",
        ),
        "aggregator_human_prior_adapter": (
            REPO_ROOT / "vggt" / "models" / "aggregator.py",
            "HumanPriorAdapter",
        ),
        "aggregator_prior_forward": (
            REPO_ROOT / "vggt" / "models" / "aggregator.py",
            "prior_maps=prior_maps",
        ),
        "trainer_prior_plumbing": (
            REPO_ROOT / "training" / "trainer.py",
            "prior_summary_tokens=batch.get",
        ),
        "dataset_prior_loading": (
            REPO_ROOT / "training" / "data" / "datasets" / "dna4k4d_pseudo.py",
            "prior_summary_tokens",
        ),
    }
    rows: dict[str, Any] = {}
    for label, (path, needle) in checks.items():
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        rows[label] = {
            "file": str(path),
            "needle": needle,
            "present": needle in text,
        }
    return rows


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V16 VGGT SMPL-X Microfit Runner",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only Agent E runner. It uses real VGGT Aggregator/HumanPriorAdapter/DPTHead modules on the local SMPL-X prior case, and it does not write formal prediction, teacher, candidate, registry, package, or pass artifacts.",
        "",
        "## D-Line Route",
        "",
        f"- route: `{summary['dline_route']}`",
        f"- dline_allowed: `{summary['dline_allowed']}`",
        f"- full_local_vggt_status: `{summary['environment'].get('full_local_vggt_status')}`",
        f"- device_used: `{summary['execution'].get('device')}`",
        "",
        "## Methods",
        "",
    ]
    for item in summary["method_specs"]:
        lines.append(
            f"- {item['method_id']}: {item['label']}; trainable=`{item['trainable_mode']}`; lr=`{item['lr']}`"
        )
    lines.extend(
        [
            "",
            "## Control Comparison",
            "",
            "```json",
            json.dumps(json_ready(summary["comparison"]), indent=2, ensure_ascii=False, sort_keys=True),
            "```",
            "",
            "## Case",
            "",
            f"- case_root: `{summary['case']['case_root']}`",
            f"- views: `{summary['batch']['view_count']}`",
            f"- target_size: `{summary['batch']['target_size']}`",
            f"- prior_channels: `{summary['batch']['prior_channels']}`",
            f"- prior_summary_channels: `{summary['batch']['prior_summary_channels']}`",
            f"- valid_pixels: `{summary['batch']['valid_pixels']}`",
            "",
            "## Outputs",
            "",
        ]
    )
    for item in summary.get("key_outputs", []):
        lines.append(f"- {item['label']}: `{item['path']}`")
    lines.extend(["", "## Blockers", ""])
    blockers = summary.get("blockers") or []
    lines.extend([f"- {item}" for item in blockers] if blockers else ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_rollup(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V16 Execution Rollup",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"- task: `{summary['task']}`",
        f"- route: `{summary['dline_route']}`",
        f"- research_only: `{summary['research_only']}`",
        f"- full_local_vggt_status: `{summary['environment'].get('full_local_vggt_status')}`",
        f"- smoke_executed: `{summary['execution'].get('executed')}`",
        f"- control_positive: `{summary['comparison'].get('any_trainable_method_control_positive')}`",
        f"- v17_stub_written: `{summary.get('v17_stub_written')}`",
        "",
        "## Key Outputs",
        "",
    ]
    for item in summary.get("key_outputs", []):
        lines.append(f"- {item['label']}: `{item['path']}`")
    lines.extend(["", "## Decision", "", summary["decision"], ""])
    blockers = summary.get("blockers") or []
    lines.extend(["## Blockers", ""])
    lines.extend([f"- {item}" for item in blockers] if blockers else ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_v17_stub(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V17 Plan Stub",
        "",
        "Minimal follow-up stub written only because V16 could not claim a full local VGGT CUDA microfit or did not produce a formal-positive D-line result.",
        "",
        "## Trigger",
        "",
        f"- v16_status: `{summary['status']}`",
        f"- full_local_vggt_status: `{summary['environment'].get('full_local_vggt_status')}`",
        f"- control_positive: `{summary['comparison'].get('any_trainable_method_control_positive')}`",
        "",
        "## Scope",
        "",
        "- Keep V17 research-only until a cloud or local environment can run full VGGT with compatible CUDA.",
        "- Preserve M0/M1/M2/M3 controls, but move full-VGGT execution to a compatible Modal GPU lane.",
        "- Do not write formal candidate, teacher, registry, package, pass, or formal prediction artifacts.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="V16 true VGGT SMPL-X microfit research runner.")
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--rollup-json", type=Path, default=DEFAULT_ROLLUP_JSON)
    parser.add_argument("--rollup-md", type=Path, default=DEFAULT_ROLLUP_MD)
    parser.add_argument("--v17-stub-md", type=Path, default=DEFAULT_V17_STUB_MD)
    parser.add_argument("--max-views", type=int, default=2)
    parser.add_argument("--target-size", type=int, default=56)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-features", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1600)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--controls", default="real,zero,shuffle")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--write-v17-stub-if-needed", action="store_true")
    args = parser.parse_args()

    output_dir = safe_v16_research_dir(args.output_dir)
    output_json = safe_report_path(args.output_json, "20260508_v16")
    output_md = safe_report_path(args.output_md, "20260508_v16")
    rollup_json = safe_report_path(args.rollup_json, "20260508_v16")
    rollup_md = safe_report_path(args.rollup_md, "20260508_v16")

    env = probe_torch_environment()
    code_paths = inspect_code_paths()
    blockers: list[str] = []
    if not all(row["present"] for row in code_paths.values()):
        blockers.append("One or more VGGT prior-path inspection needles were not found; see code_path_inspection.")

    case = load_case_arrays(args.case_root)
    manifest = case["manifest"]
    case_info = {
        "case_root": str(case["case_root"]),
        "inputs": file_row(case["inputs_path"]),
        "targets": file_row(case["targets_path"]),
        "manifest": file_row(case["manifest_path"]),
        "case_id": manifest.get("case_id"),
        "prior_channels": manifest.get("prior_channels", []),
        "prior_summary_channels": manifest.get("prior_summary_channels", []),
        "prior_geometry_source": manifest.get("prior_geometry_source"),
    }

    results: list[dict[str, Any]] = []
    execution: dict[str, Any] = {"executed": bool(args.execute)}
    batch_meta: dict[str, Any] = {}
    selected_controls = [item.strip() for item in args.controls.split(",") if item.strip()]
    unknown_controls = [item for item in selected_controls if item not in CONTROLS]
    if unknown_controls:
        raise ValueError(f"Unknown controls: {unknown_controls}")

    started = time.time()
    if args.execute:
        if not env.get("torch_import_ok"):
            blockers.append("Torch import failed; V16 local microfit could not execute.")
        else:
            import torch

            torch.set_num_threads(max(1, int(args.torch_threads)))
            device_name = choose_device(args.device, env)
            execution["device"] = device_name
            execution["torch_threads"] = int(args.torch_threads)
            batch = prepare_micro_batch(
                case,
                max_views=int(args.max_views),
                target_size=int(args.target_size),
                device_name=device_name,
            )
            batch_meta = {key: value for key, value in batch.items() if key not in {"images", "prior_maps_real", "prior_summary_real", "target_depth", "target_points", "prior_mask"}}
            for method_idx, method in enumerate(METHODS):
                for control_idx, control in enumerate(selected_controls):
                    seed = int(args.seed) + method_idx * 100 + control_idx
                    row = run_one(
                        method=method,
                        control=control,
                        batch=batch,
                        args=args,
                        output_dir=output_dir,
                        seed=seed,
                    )
                    results.append(row)
            execution["elapsed_sec"] = round(time.time() - started, 3)
    else:
        execution["reason"] = "not requested; pass --execute for local CPU-safe microfit smoke"

    comparison = compare_controls(results, float(args.tolerance)) if results else {
        "any_trainable_method_control_positive": False,
        "available": False,
        "reason": "microfit execution was not requested or could not run",
    }

    full_local_blocked = env.get("full_local_vggt_status") != "available_for_cuda_attempt"
    control_positive = bool(comparison.get("any_trainable_method_control_positive"))
    if results and control_positive and not full_local_blocked:
        status = "v16_vggt_smplx_microfit_control_positive_research_only"
        dline_route = "DLINE_V16_RESEARCH_ONLY_CONTINUE_MODAL_RESEARCH_NO_FORMAL_CANDIDATE"
        decision = "V16 observed a bounded control-positive microfit and the local CUDA probe did not block full VGGT, but outputs remain research-only."
    elif results and control_positive:
        status = "v16_vggt_smplx_microfit_cpu_smoke_positive_full_vggt_local_blocked"
        dline_route = "DLINE_V16_CPU_SMOKE_TO_MODAL_RESEARCH_FULL_VGGT_REQUIRED"
        decision = "V16 CPU microfit observed a control-positive research signal, but full local VGGT CUDA is blocked; use the Modal research lane for any full-VGGT follow-up."
        blockers.append("Full local VGGT CUDA is blocked or unsupported; this run is a CPU-safe microfit smoke only.")
    elif results:
        status = "v16_vggt_smplx_microfit_negative_or_inconclusive_research_only"
        dline_route = "DLINE_V16_FAIL_CLOSED_TO_V17_PLAN"
        decision = "V16 executed the bounded research microfit, but real priors did not beat zero/shuffle controls across trainable methods within tolerance."
        blockers.append("Real prior control did not robustly beat zero/shuffle controls in the bounded V16 smoke.")
        if full_local_blocked:
            blockers.append("Full local VGGT CUDA is blocked or unsupported; V16 cannot claim a full local VGGT result.")
    else:
        status = "v16_vggt_smplx_microfit_not_executed_research_only"
        dline_route = "DLINE_V16_RESEARCH_CONTRACT_ONLY"
        decision = "V16 wrote the runner/research contract but did not execute the local microfit smoke in this invocation."

    need_v17_stub = bool((results and not control_positive) or full_local_blocked)
    v17_stub_written = False
    v17_stub_path = None
    if args.write_v17_stub_if_needed and need_v17_stub:
        v17_stub_path = args.v17_stub_md.expanduser().resolve()

    key_outputs = [
        {"label": "output_dir", "path": str(output_dir)},
        {"label": "runner_json", "path": str(output_json)},
        {"label": "runner_md", "path": str(output_md)},
        {"label": "rollup_json", "path": str(rollup_json)},
        {"label": "rollup_md", "path": str(rollup_md)},
    ]
    if v17_stub_path is not None:
        key_outputs.append({"label": "v17_plan_stub", "path": str(v17_stub_path)})

    summary = {
        "task": "v16_vggt_smplx_microfit_runner",
        "created_utc": utc_now(),
        "status": status,
        **RESEARCH_CONTRACT,
        "environment": env,
        "code_path_inspection": code_paths,
        "case": case_info,
        "batch": batch_meta,
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
        "controls": selected_controls,
        "execution": execution,
        "results": results,
        "comparison": comparison,
        "dline_route": dline_route,
        "dline_allowed": False,
        "decision": decision,
        "blockers": blockers,
        "key_outputs": key_outputs,
        "v17_stub_needed": need_v17_stub,
        "v17_stub_written": v17_stub_written,
        "v17_stub_path": str(v17_stub_path) if v17_stub_path is not None else None,
    }

    if v17_stub_path is not None:
        write_v17_stub(v17_stub_path, summary)
        summary["v17_stub_written"] = True
        v17_stub_written = True

    write_json(output_json, summary)
    write_markdown(output_md, summary)
    write_json(rollup_json, summary)
    write_rollup(rollup_md, summary)
    write_json(output_dir / "summary.json", summary)
    write_markdown(output_dir / "summary.md", summary)

    print(
        json.dumps(
            json_ready(
                {
                    "status": status,
                    "dline_route": dline_route,
                    "executed": bool(args.execute),
                    "control_positive": control_positive,
                    "full_local_vggt_status": env.get("full_local_vggt_status"),
                    "v17_stub_written": v17_stub_written,
                    "output_json": output_json,
                    "rollup_json": rollup_json,
                }
            ),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
