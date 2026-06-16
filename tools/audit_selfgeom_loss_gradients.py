#!/usr/bin/env python
"""Audit whether self-geometry losses send gradients to VGGT geometry heads.

This is intentionally a one-batch diagnostic, not a training launcher.  It
loads an existing training config/checkpoint, runs one forward pass per selected
loss component, and reports gradient norms on prediction tensors and named model
modules.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"

for root in (REPO_ROOT, TRAINING_ROOT):
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

from trainer import Trainer  # noqa: E402
from train_utils.general import copy_data_to_device  # noqa: E402


DEFAULT_COMPONENTS = [
    "loss_prior_depth_normal",
    "loss_prior_depth_point",
    "loss_prior_cross_view_depth",
    "loss_prior_cross_view_point",
    "loss_prior_cross_view_normal",
    "loss_prior_cross_view",
    "loss_prior_point_normal",
    "loss_human_prior",
    "objective",
]

PREDICTION_KEYS = [
    "depth",
    "world_points",
    "normal",
    "pose_enc",
    "depth_conf",
    "world_points_conf",
    "normal_conf",
]

PARAM_GROUP_PREFIXES = {
    "aggregator": ("aggregator.",),
    "camera_head": ("camera_head.",),
    "depth_head": ("depth_head.",),
    "point_head": ("point_head.",),
    "normal_head": ("normal_head.",),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Hydra config name without .yaml")
    parser.add_argument("--checkpoint", default="", help="Optional checkpoint to load")
    parser.add_argument("--case-root", action="append", default=[], help="Training case root; repeatable")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--img-num", type=int, default=6)
    parser.add_argument("--max-img-per-gpu", type=int, default=6)
    parser.add_argument("--components", nargs="*", default=DEFAULT_COMPONENTS)
    parser.add_argument("--keep-base-loss", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override. Repeat for multiple overrides.",
    )
    return parser.parse_args()


def _as_posix_path(path: str) -> str:
    return Path(path).expanduser().resolve().as_posix()


def _compose_cfg(args: argparse.Namespace):
    with initialize_config_dir(version_base=None, config_dir=str(TRAINING_ROOT / "config")):
        cfg = compose(config_name=args.config, overrides=list(args.override))

    case_roots = [_as_posix_path(path) for path in args.case_root]

    with open_dict(cfg):
        cfg.device = args.device
        cfg.mode = "train"
        cfg.distributed.backend = "none"
        cfg.limit_train_batches = 0
        cfg.limit_val_batches = 0
        cfg.val_epoch_freq = 999
        cfg.max_epochs = 1
        cfg.num_workers = 0
        cfg.max_img_per_gpu = args.max_img_per_gpu
        cfg.logging.log_dir = str(Path(args.output_json).expanduser().resolve().parent / "_tmp_grad_audit_logs")
        cfg.logging.log_visuals = False
        cfg.logging.log_freq = 999999
        cfg.checkpoint.enabled = False
        if args.checkpoint:
            cfg.checkpoint.resume_checkpoint_path = _as_posix_path(args.checkpoint)
        else:
            cfg.checkpoint.resume_checkpoint_path = None
        cfg.checkpoint.save_dir = str(Path(cfg.logging.log_dir) / "ckpts")
        cfg.optim.amp.enabled = not args.no_amp
        cfg.data.train.num_workers = 0
        cfg.data.train.max_img_per_gpu = args.max_img_per_gpu
        cfg.data.train.common_config.max_img_per_gpu = args.max_img_per_gpu
        cfg.data.train.common_config.fix_img_num = args.img_num
        cfg.data.train.common_config.img_nums = [args.img_num, args.img_num]
        cfg.data.train.common_config.allow_duplicate_img = False
        cfg.data.train.common_config.repeat_batch = False
        cfg.data.train.dataset.dataset_configs[0].len_train = 1
        cfg.data.train.dataset.dataset_configs[0].len_test = 1
        if case_roots:
            cfg.data.train.dataset.dataset_configs[0].case_roots = case_roots
        if "val" in cfg.data and cfg.data.val is not None:
            cfg.data.val.num_workers = 0
            cfg.data.val.max_img_per_gpu = args.max_img_per_gpu
            cfg.data.val.common_config.max_img_per_gpu = args.max_img_per_gpu
            cfg.data.val.common_config.fix_img_num = args.img_num
            cfg.data.val.common_config.img_nums = [args.img_num, args.img_num]
            cfg.data.val.common_config.allow_duplicate_img = False
            cfg.data.val.common_config.repeat_batch = False
            cfg.data.val.dataset.dataset_configs[0].len_train = 1
            cfg.data.val.dataset.dataset_configs[0].len_test = 1
            if case_roots:
                cfg.data.val.dataset.dataset_configs[0].case_roots = case_roots
        if not args.keep_base_loss:
            cfg.loss.camera = None
            cfg.loss.depth = None
            cfg.loss.point = None
    return cfg


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _grad_stats_from_tensor(tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None:
        return {"has_grad": False, "norm": 0.0, "max_abs": 0.0, "mean_abs": 0.0}
    grad = tensor.grad
    if grad is None:
        return {"has_grad": False, "norm": 0.0, "max_abs": 0.0, "mean_abs": 0.0}
    grad = grad.detach()
    finite = torch.isfinite(grad)
    if not finite.any():
        return {"has_grad": True, "norm": None, "max_abs": None, "mean_abs": None}
    vals = grad[finite].float()
    return {
        "has_grad": True,
        "norm": float(torch.linalg.vector_norm(vals).item()),
        "max_abs": float(vals.abs().max().item()),
        "mean_abs": float(vals.abs().mean().item()),
    }


def _zero_model_grads(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.grad = None


def _param_group_stats(model: torch.nn.Module) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for group, prefixes in PARAM_GROUP_PREFIXES.items():
        sum_sq = 0.0
        max_abs = 0.0
        mean_abs_num = 0.0
        mean_abs_den = 0
        trainable = 0
        with_grad = 0
        for name, param in model.named_parameters():
            if not any(name.startswith(prefix) for prefix in prefixes):
                continue
            if param.requires_grad:
                trainable += param.numel()
            if param.grad is None:
                continue
            grad = param.grad.detach()
            finite = torch.isfinite(grad)
            if not finite.any():
                continue
            vals = grad[finite].float()
            with_grad += int(vals.numel())
            sum_sq += float(torch.sum(vals * vals).item())
            max_abs = max(max_abs, float(vals.abs().max().item()))
            mean_abs_num += float(vals.abs().sum().item())
            mean_abs_den += int(vals.numel())
        stats[group] = {
            "trainable_params": trainable,
            "grad_values": with_grad,
            "has_grad": with_grad > 0,
            "norm": math.sqrt(sum_sq) if with_grad > 0 else 0.0,
            "max_abs": max_abs,
            "mean_abs": mean_abs_num / mean_abs_den if mean_abs_den > 0 else 0.0,
        }
    return stats


def _tensor_loss_value(value: Any) -> float | None:
    if not torch.is_tensor(value):
        return _float_or_none(value)
    return _float_or_none(value.detach().float().cpu().item())


def _run_component(
    trainer: Trainer,
    batch: dict[str, Any],
    component: str,
) -> dict[str, Any]:
    model = trainer.model
    model.train()
    _zero_model_grads(model)

    amp_type = trainer.optim_conf.amp.amp_dtype
    if amp_type == "bfloat16":
        amp_dtype = torch.bfloat16
    elif amp_type == "float16":
        amp_dtype = torch.float16
    else:
        raise ValueError(f"Invalid AMP dtype: {amp_type}")

    with torch.cuda.amp.autocast(enabled=trainer.optim_conf.amp.enabled, dtype=amp_dtype):
        predictions = model(
            images=batch["images"],
            human_prior_feature_maps=batch.get("smpl_vertex_feature_maps", batch.get("prior_maps")),
            human_prior_summary_tokens=batch.get("smpl_summary_tokens", batch.get("prior_summary_tokens")),
        )
        retained_predictions = {}
        for key in PREDICTION_KEYS:
            tensor = predictions.get(key)
            if torch.is_tensor(tensor) and tensor.requires_grad:
                tensor.retain_grad()
                retained_predictions[key] = tensor
        loss_dict = trainer.loss(predictions, batch)

    if component not in loss_dict:
        return {
            "status": "missing_loss_key",
            "component": component,
            "available_loss_keys": sorted(loss_dict.keys()),
        }
    loss = loss_dict[component]
    if not torch.is_tensor(loss) or not loss.requires_grad:
        return {
            "status": "no_grad_loss",
            "component": component,
            "loss_value": _tensor_loss_value(loss),
        }

    loss.backward()

    return {
        "status": "ok",
        "component": component,
        "loss_value": _tensor_loss_value(loss),
        "prediction_grads": {
            key: _grad_stats_from_tensor(tensor)
            for key, tensor in retained_predictions.items()
        },
        "param_group_grads": _param_group_stats(model),
        "all_loss_values": {
            key: _tensor_loss_value(value)
            for key, value in loss_dict.items()
            if key == component or key in DEFAULT_COMPONENTS
        },
    }


def _format_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Self-Geometry Loss Gradient Audit",
        "",
        f"- Config: `{payload['config_name']}`",
        f"- Checkpoint: `{payload.get('checkpoint') or 'none'}`",
        f"- Case roots: `{payload.get('case_roots')}`",
        f"- Device: `{payload['device']}`",
        f"- Base camera/depth/point losses disabled: `{payload['base_losses_disabled']}`",
        "",
        "## Components",
        "",
        "| component | status | loss | world_points grad | point_head grad | depth grad | depth_head grad | normal grad | normal_head grad |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["components"]:
        pred = item.get("prediction_grads", {})
        params = item.get("param_group_grads", {})

        def norm_text(container: dict[str, Any], key: str) -> str:
            stats = container.get(key, {})
            value = stats.get("norm", 0.0)
            if value is None:
                return "nan"
            return f"{float(value):.4e}"

        lines.append(
            "| {component} | {status} | {loss} | {world} | {point_head} | {depth} | {depth_head} | {normal} | {normal_head} |".format(
                component=item.get("component"),
                status=item.get("status"),
                loss="n/a" if item.get("loss_value") is None else f"{float(item['loss_value']):.6g}",
                world=norm_text(pred, "world_points"),
                point_head=norm_text(params, "point_head"),
                depth=norm_text(pred, "depth"),
                depth_head=norm_text(params, "depth_head"),
                normal=norm_text(pred, "normal"),
                normal_head=norm_text(params, "normal_head"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A self-geometry component only helps final point cloud geometry if `world_points` and/or `point_head` receive non-zero gradients.",
            "- If normal/depth consistency mainly moves `normal_head` while `world_points` stays near zero, it can improve 2D normal metrics without changing Open3D geometry.",
            "- This audit is local evidence only; it does not claim visual pass.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md = Path(args.output_md).expanduser().resolve() if args.output_md else output_json.with_suffix(".md")

    cfg = _compose_cfg(args)
    trainer = Trainer(**cfg)
    loader = trainer.train_dataset.get_loader(epoch=0)
    batch = next(iter(loader))
    with torch.cuda.amp.autocast(enabled=False):
        batch = trainer._process_batch(batch)
    batch = copy_data_to_device(batch, trainer.device, non_blocking=True)

    components = [_run_component(trainer, batch, component) for component in args.components]
    payload = {
        "config_name": args.config,
        "checkpoint": args.checkpoint,
        "case_roots": args.case_root,
        "device": args.device,
        "img_num": args.img_num,
        "max_img_per_gpu": args.max_img_per_gpu,
        "base_losses_disabled": not args.keep_base_loss,
        "components": components,
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(_format_md(payload), encoding="utf-8")
    print(f"[selfgeom-grad-audit] wrote {output_json}")
    print(f"[selfgeom-grad-audit] wrote {output_md}")


if __name__ == "__main__":
    main()
