from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
LOCAL_ROOT = REPO_ROOT / "output" / "surface_research_preflight_local"
DEFAULT_OUTPUT_DIR = LOCAL_ROOT / "V16_human_prior_adapter_probe"
DEFAULT_JSON = REPORTS / "20260508_v16_human_prior_adapter_probe.json"
DEFAULT_MD = REPORTS / "20260508_v16_human_prior_adapter_probe.md"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.models.aggregator import Aggregator  # noqa: E402
from vggt.models.human_prior import HumanPriorAdapter  # noqa: E402


INSPECTED_FILES = [
    "vggt/models/human_prior.py",
    "vggt/models/aggregator.py",
    "vggt/models/vggt.py",
    "training/loss_smplx_native_prior.py",
    "training/loss.py",
]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return json_ready(value.detach().cpu().item())
        return json_ready(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def safe_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    lower = resolved.as_posix().lower()
    if "surface_research_preflight_local" not in lower or "v16_human_prior_adapter_probe" not in lower:
        raise ValueError(f"Refusing output path outside the V16 adapter probe sandbox: {resolved}")
    for token in ("strict_pass", "strict_gate_registry", "candidate_export", "teacher_export", "predictions"):
        if token in lower:
            raise ValueError(f"Refusing forbidden output token {token!r}: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def tensor_rms_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a.detach() - b.detach()) ** 2)).cpu())


def tensor_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach()).cpu())


def scalar(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def cuda_probe() -> dict[str, Any]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            available = bool(torch.cuda.is_available())
            device_name = torch.cuda.get_device_name(0) if available else None
        except Exception as exc:  # pragma: no cover - environment dependent
            return {
                "torch_version": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": False,
                "device_name": None,
                "error": str(exc),
                "warnings": [str(item.message) for item in caught],
            }
    return {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": available,
        "device_name": device_name,
        "warnings": [str(item.message) for item in caught],
    }


def make_images(batch_size: int, seq_len: int, height: int, width: int) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, height)
    x = torch.linspace(0.0, 1.0, width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    channels = torch.stack([xx, yy, 0.5 + 0.25 * torch.sin(2.0 * math.pi * xx)], dim=0)
    images = channels.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1, 1)
    for view_idx in range(seq_len):
        images[:, view_idx] = (images[:, view_idx] + 0.05 * view_idx).clamp(0.0, 1.0)
    return images.float()


def make_prior_maps(batch_size: int, seq_len: int, channels: int, height: int, width: int) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height)
    x = torch.linspace(-1.0, 1.0, width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius = torch.sqrt(xx * xx + yy * yy).clamp(max=1.5)
    patterns = [
        xx,
        yy,
        torch.sin(math.pi * xx),
        torch.cos(math.pi * yy),
        torch.exp(-3.0 * radius),
        (xx * yy),
        torch.sin(2.0 * math.pi * (xx + yy)),
        (radius < 0.65).float(),
    ]
    selected = []
    for channel_idx in range(channels):
        selected.append(patterns[channel_idx % len(patterns)] + 0.03 * channel_idx)
    prior = torch.stack(selected, dim=0).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1, 1)
    for view_idx in range(seq_len):
        prior[:, view_idx] = prior[:, view_idx].roll(shifts=view_idx, dims=-1) + 0.07 * view_idx
    return prior.float()


def pad_last_dim(value: torch.Tensor, channels: int) -> torch.Tensor:
    if value.shape[-1] >= channels:
        return value[..., :channels]
    pad_shape = (*value.shape[:-1], channels - value.shape[-1])
    return torch.cat([value, torch.zeros(pad_shape, dtype=value.dtype, device=value.device)], dim=-1)


def make_summary_tokens(prior_maps: torch.Tensor, token_count: int, summary_channels: int) -> torch.Tensor:
    mean = prior_maps.mean(dim=(-2, -1))
    std = prior_maps.std(dim=(-2, -1))
    maxv = prior_maps.amax(dim=(-2, -1))
    minv = prior_maps.amin(dim=(-2, -1))
    stats = [mean, std, maxv, minv]
    tokens = []
    for token_idx in range(token_count):
        row = pad_last_dim(stats[token_idx % len(stats)], summary_channels).clone()
        row = row + 0.01 * token_idx
        tokens.append(row)
    return torch.stack(tokens, dim=2).float()


def make_fixed_target_delta(prior_maps: torch.Tensor, target_hw: tuple[int, int], embed_dim: int) -> torch.Tensor:
    batch_size, seq_len, channels = prior_maps.shape[:3]
    pooled = F.interpolate(
        prior_maps.reshape(batch_size * seq_len, channels, prior_maps.shape[-2], prior_maps.shape[-1]),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    pooled_tokens = pooled.flatten(2).transpose(1, 2)
    basis = torch.linspace(-0.7, 0.7, channels * embed_dim, dtype=prior_maps.dtype).reshape(channels, embed_dim)
    return torch.tanh(pooled_tokens @ basis / math.sqrt(float(channels)))


def adapter_no_train_probe(
    real_prior: torch.Tensor,
    zero_prior: torch.Tensor,
    shuffle_prior: torch.Tensor,
    summary_real: torch.Tensor,
    summary_zero: torch.Tensor,
    summary_shuffle: torch.Tensor,
    target_hw: tuple[int, int],
    embed_dim: int,
    depth: int,
    hidden_dim: int,
) -> dict[str, Any]:
    batch_size, seq_len, channels = real_prior.shape[:3]
    summary_channels = summary_real.shape[-1]
    patch_tokens = torch.randn(batch_size * seq_len, target_hw[0] * target_hw[1], embed_dim)

    torch.manual_seed(20260508)
    default_adapter = HumanPriorAdapter(
        in_channels=channels,
        embed_dim=embed_dim,
        depth=depth,
        summary_in_channels=summary_channels,
        hidden_dim=hidden_dim,
        gate_init=0.0,
        multi_scale_factors=(1, 2),
    ).eval()
    torch.manual_seed(20260508)
    gated_adapter = HumanPriorAdapter(
        in_channels=channels,
        embed_dim=embed_dim,
        depth=depth,
        summary_in_channels=summary_channels,
        hidden_dim=hidden_dim,
        gate_init=0.25,
        multi_scale_factors=(1, 2),
    ).eval()

    with torch.no_grad():
        default_tokens = {
            "real": default_adapter.project_prior_maps(real_prior, target_hw),
            "zero": default_adapter.project_prior_maps(zero_prior, target_hw),
            "shuffle": default_adapter.project_prior_maps(shuffle_prior, target_hw),
        }
        default_fused = {
            name: default_adapter.fuse_input_tokens(patch_tokens, tokens)
            for name, tokens in default_tokens.items()
        }
        gated_tokens = {
            "real": gated_adapter.project_prior_maps(real_prior, target_hw),
            "zero": gated_adapter.project_prior_maps(zero_prior, target_hw),
            "shuffle": gated_adapter.project_prior_maps(shuffle_prior, target_hw),
        }
        gated_fused = {
            name: gated_adapter.fuse_input_tokens(patch_tokens, tokens)
            for name, tokens in gated_tokens.items()
        }
        summary_proj = {
            "real": gated_adapter.project_summary_tokens(summary_real),
            "zero": gated_adapter.project_summary_tokens(summary_zero),
            "shuffle": gated_adapter.project_summary_tokens(summary_shuffle),
        }
        summary_fused = {
            name: gated_adapter.fuse_global_summary_tokens(patch_tokens, tokens, layer_idx=0)
            for name, tokens in summary_proj.items()
        }

    return {
        "default_gate_init": 0.0,
        "gated_probe_gate_init": 0.25,
        "projection_norms": {name: tensor_norm(value) for name, value in default_tokens.items()},
        "projection_rms_delta_real_zero": tensor_rms_delta(default_tokens["real"], default_tokens["zero"]),
        "projection_rms_delta_real_shuffle": tensor_rms_delta(default_tokens["real"], default_tokens["shuffle"]),
        "default_gate_input_fusion_rms_delta_real_zero": tensor_rms_delta(default_fused["real"], default_fused["zero"]),
        "default_gate_input_fusion_rms_delta_real_shuffle": tensor_rms_delta(default_fused["real"], default_fused["shuffle"]),
        "gated_input_fusion_rms_delta_real_zero": tensor_rms_delta(gated_fused["real"], gated_fused["zero"]),
        "gated_input_fusion_rms_delta_real_shuffle": tensor_rms_delta(gated_fused["real"], gated_fused["shuffle"]),
        "summary_projection_norms": {name: tensor_norm(value) for name, value in summary_proj.items()},
        "gated_summary_fusion_rms_delta_real_zero": tensor_rms_delta(summary_fused["real"], summary_fused["zero"]),
        "gated_summary_fusion_rms_delta_real_shuffle": tensor_rms_delta(summary_fused["real"], summary_fused["shuffle"]),
    }


def tiny_aggregator_probe(
    images: torch.Tensor,
    real_prior: torch.Tensor,
    zero_prior: torch.Tensor,
    shuffle_prior: torch.Tensor,
    summary_real: torch.Tensor,
    summary_zero: torch.Tensor,
    summary_shuffle: torch.Tensor,
    embed_dim: int,
    depth: int,
    hidden_dim: int,
) -> dict[str, Any]:
    batch_size, seq_len, _, height, width = images.shape
    channels = real_prior.shape[2]
    summary_channels = summary_real.shape[-1]

    def build(gate_init: float) -> Aggregator:
        torch.manual_seed(20260508)
        model = Aggregator(
            img_size=height,
            patch_size=8,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=4,
            mlp_ratio=2.0,
            num_register_tokens=1,
            patch_embed="conv",
            aa_order=["frame", "global"],
            aa_block_size=1,
            qk_norm=False,
            rope_freq=-1,
            human_prior_channels=channels,
            human_prior_summary_channels=summary_channels,
            human_prior_hidden_dim=hidden_dim,
            human_prior_gate_init=gate_init,
            human_prior_multi_scale_factors=(1, 2),
        )
        model.eval()
        return model

    variants = {
        "real": (real_prior, summary_real),
        "zero": (zero_prior, summary_zero),
        "shuffle": (shuffle_prior, summary_shuffle),
    }
    results: dict[str, Any] = {}
    for gate_label, gate_init in (("default_gate", 0.0), ("nonzero_gate", 0.25)):
        model = build(gate_init)
        outputs = {}
        with torch.no_grad():
            for name, (prior, summary) in variants.items():
                out_list, patch_start_idx = model(images, prior_maps=prior, prior_summary_tokens=summary)
                outputs[name] = out_list[-1]
        results[gate_label] = {
            "gate_init": gate_init,
            "patch_start_idx": int(patch_start_idx),
            "last_output_shape": list(outputs["real"].shape),
            "rms_delta_real_zero": tensor_rms_delta(outputs["real"], outputs["zero"]),
            "rms_delta_real_shuffle": tensor_rms_delta(outputs["real"], outputs["shuffle"]),
            "output_norms": {name: tensor_norm(value) for name, value in outputs.items()},
            "input_gate": scalar(model.human_prior_adapter.input_fusion.gate),
            "frame_gate_0": scalar(model.human_prior_adapter.frame_fusions[0].gate),
            "global_gate_0": scalar(model.human_prior_adapter.global_fusions[0].gate),
            "summary_gate_0": scalar(model.human_prior_adapter.global_summary_fusions[0].gate),
        }
    results["note"] = (
        "This is a real Aggregator.forward route on CPU with patch_embed='conv' and reduced dimensions; "
        "it avoids the default DINOv2-large VGGT configuration."
    )
    results["input_shape"] = {"images": list(images.shape), "prior_maps": list(real_prior.shape), "summary": list(summary_real.shape)}
    results["batch_sequence"] = {"batch_size": batch_size, "seq_len": seq_len}
    return results


def grad_summary(model: torch.nn.Module) -> dict[str, Any]:
    rows = {}
    total_sq = 0.0
    nonzero = 0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        norm_value = float(torch.linalg.vector_norm(param.grad.detach()).cpu())
        rows[name] = norm_value
        total_sq += norm_value * norm_value
        if norm_value > 1e-12:
            nonzero += 1
    return {
        "total_grad_norm": math.sqrt(total_sq),
        "nonzero_grad_param_count": nonzero,
        "sample_grad_norms": {key: rows[key] for key in sorted(rows)[:12]},
    }


def adapter_train_sanity(
    real_prior: torch.Tensor,
    zero_prior: torch.Tensor,
    shuffle_prior: torch.Tensor,
    target_hw: tuple[int, int],
    embed_dim: int,
    depth: int,
    hidden_dim: int,
    steps: int,
    lr: float,
) -> dict[str, Any]:
    batch_size, seq_len, channels = real_prior.shape[:3]
    patch_tokens = torch.randn(batch_size * seq_len, target_hw[0] * target_hw[1], embed_dim)
    target_delta = make_fixed_target_delta(real_prior, target_hw, embed_dim)
    target_tokens = (patch_tokens + 0.25 * target_delta).detach()

    torch.manual_seed(20260509)
    adapter = HumanPriorAdapter(
        in_channels=channels,
        embed_dim=embed_dim,
        depth=depth,
        summary_in_channels=0,
        hidden_dim=hidden_dim,
        gate_init=0.0,
        multi_scale_factors=(1, 2),
        enable_frame_fusion=False,
        enable_global_fusion=False,
        enable_summary_fusion=False,
    )
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
    history = []
    first_grad = None
    last_grad = None
    initial_loss = None

    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        prior_tokens = adapter.project_prior_maps(real_prior, target_hw)
        fused = adapter.fuse_input_tokens(patch_tokens, prior_tokens)
        loss = F.mse_loss(fused, target_tokens)
        loss.backward()
        if step == 0:
            initial_loss = scalar(loss)
            first_grad = grad_summary(adapter)
        if step == int(steps) - 1:
            last_grad = grad_summary(adapter)
        optimizer.step()
        if step in {0, 1, 2, 4, 9, int(steps) - 1}:
            history.append(
                {
                    "step": int(step),
                    "loss": scalar(loss),
                    "input_gate": scalar(adapter.input_fusion.gate),
                }
            )

    with torch.no_grad():
        real_out = adapter.fuse_input_tokens(patch_tokens, adapter.project_prior_maps(real_prior, target_hw))
        zero_out = adapter.fuse_input_tokens(patch_tokens, adapter.project_prior_maps(zero_prior, target_hw))
        shuffle_out = adapter.fuse_input_tokens(patch_tokens, adapter.project_prior_maps(shuffle_prior, target_hw))
        final_loss = F.mse_loss(real_out, target_tokens)

    return {
        "steps": int(steps),
        "lr": float(lr),
        "initial_loss": initial_loss,
        "final_loss": scalar(final_loss),
        "loss_delta": float(initial_loss - scalar(final_loss)) if initial_loss is not None else None,
        "history": history,
        "input_gate_final": scalar(adapter.input_fusion.gate),
        "real_zero_output_rms_delta_after_train": tensor_rms_delta(real_out, zero_out),
        "real_shuffle_output_rms_delta_after_train": tensor_rms_delta(real_out, shuffle_out),
        "first_backward": first_grad,
        "last_backward": last_grad,
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V16 Human Prior Adapter Probe",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only local probe. No strict pass, package, registry, teacher export, candidate export, or prediction bundle is written.",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Full VGGT Forward",
        "",
        f"- skipped: `{summary['full_vggt_forward']['skipped']}`",
        f"- reason: {summary['full_vggt_forward']['reason']}",
        "",
        "## Key Metrics",
        "",
    ]
    metrics = summary["metrics"]
    lines.extend(
        [
            f"- no_train_projection_rms_real_zero: `{metrics['adapter_no_train']['projection_rms_delta_real_zero']}`",
            f"- no_train_default_gate_input_rms_real_zero: `{metrics['adapter_no_train']['default_gate_input_fusion_rms_delta_real_zero']}`",
            f"- no_train_gated_input_rms_real_zero: `{metrics['adapter_no_train']['gated_input_fusion_rms_delta_real_zero']}`",
            f"- routed_aggregator_nonzero_gate_rms_real_zero: `{metrics['tiny_aggregator']['nonzero_gate']['rms_delta_real_zero']}`",
            f"- train_initial_loss: `{metrics['adapter_train_sanity']['initial_loss']}`",
            f"- train_final_loss: `{metrics['adapter_train_sanity']['final_loss']}`",
            f"- train_input_gate_final: `{metrics['adapter_train_sanity']['input_gate_final']}`",
            f"- train_real_zero_output_rms: `{metrics['adapter_train_sanity']['real_zero_output_rms_delta_after_train']}`",
        ]
    )
    lines.extend(["", "## Inspected Files", ""])
    for row in summary["inspected_files"]:
        lines.append(f"- {row['path']}: exists=`{row['exists']}`")
    lines.extend(["", "## Outputs", ""])
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Blockers", ""])
    blockers = summary.get("blockers") or []
    lines.extend([f"- {item}" for item in blockers] if blockers else ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    torch.set_num_threads(max(1, int(args.threads)))
    out_dir = safe_output_dir(args.output_dir)

    images = make_images(args.batch_size, args.seq_len, args.height, args.width)
    real_prior = make_prior_maps(args.batch_size, args.seq_len, args.prior_channels, args.height, args.width)
    zero_prior = torch.zeros_like(real_prior)
    shuffle_prior = real_prior.roll(shifts=1, dims=1).flip(dims=(-1,))
    summary_real = make_summary_tokens(real_prior, args.summary_tokens, args.summary_channels)
    summary_zero = make_summary_tokens(zero_prior, args.summary_tokens, args.summary_channels)
    summary_shuffle = make_summary_tokens(shuffle_prior, args.summary_tokens, args.summary_channels)
    target_hw = (args.height // 8, args.width // 8)

    adapter_no_train = adapter_no_train_probe(
        real_prior,
        zero_prior,
        shuffle_prior,
        summary_real,
        summary_zero,
        summary_shuffle,
        target_hw,
        args.embed_dim,
        args.depth,
        args.hidden_dim,
    )
    tiny_aggregator = tiny_aggregator_probe(
        images,
        real_prior,
        zero_prior,
        shuffle_prior,
        summary_real,
        summary_zero,
        summary_shuffle,
        args.embed_dim,
        args.depth,
        args.hidden_dim,
    )
    train_sanity = adapter_train_sanity(
        real_prior,
        zero_prior,
        shuffle_prior,
        target_hw,
        args.embed_dim,
        args.depth,
        args.hidden_dim,
        args.train_steps,
        args.lr,
    )

    nonzero_prior_path = (
        adapter_no_train["projection_rms_delta_real_zero"] > 1e-6
        and tiny_aggregator["nonzero_gate"]["rms_delta_real_zero"] > 1e-8
        and train_sanity["real_zero_output_rms_delta_after_train"] > 1e-8
        and abs(train_sanity["input_gate_final"]) > 1e-8
        and train_sanity["final_loss"] < train_sanity["initial_loss"]
    )
    blockers = []
    if not nonzero_prior_path:
        blockers.append("Adapter path remained inert under the CPU train sanity; route to V16-3-F for prior-routing investigation.")

    summary = {
        "task": "V16-DLINE-ROUTED-SMPLX Agent C human prior adapter probe",
        "created_utc": utc_now(),
        "status": "v16_human_prior_adapter_nonzero_trainable" if nonzero_prior_path else "route_to_v16_3_f_prior_path_inert",
        "research_only": True,
        "no_strict_pass": True,
        "no_package": True,
        "device": "cpu",
        "seed": int(args.seed),
        "inspected_files": [
            {"path": str((REPO_ROOT / file_path).resolve()), "exists": (REPO_ROOT / file_path).is_file()}
            for file_path in INSPECTED_FILES
        ],
        "full_vggt_forward": {
            "skipped": True,
            "reason": (
                "Skipped full VGGT.forward for this local probe: the local PyTorch CUDA build reports an RTX 5080 "
                "with unsupported sm_120, and default VGGT uses the heavy DINOv2-large/depth-24 Aggregator. "
                "VGGT.__init__ does not expose the tiny patch_embed/depth controls needed for a CPU-safe full VGGT run "
                "without editing model code outside Agent C scope. A real CPU Aggregator.forward route with patch_embed='conv' "
                "was run instead."
            ),
            "cuda_probe": cuda_probe(),
        },
        "probe_shapes": {
            "images": list(images.shape),
            "prior_maps": list(real_prior.shape),
            "prior_summary_tokens": list(summary_real.shape),
            "target_hw": list(target_hw),
        },
        "metrics": {
            "adapter_no_train": adapter_no_train,
            "tiny_aggregator": tiny_aggregator,
            "adapter_train_sanity": train_sanity,
        },
        "decision": (
            "HumanPriorAdapter projects real priors differently from zero/shuffled controls, the routed tiny Aggregator "
            "changes output when gates are nonzero, and adapter-only training from gate zero learns a nonzero prior effect."
            if nonzero_prior_path
            else "The CPU-safe probes did not produce a nonzero trainable prior effect; route this to V16-3-F."
        ),
        "blockers": blockers,
        "outputs": {
            "output_dir": str(out_dir),
            "summary_json": str((out_dir / "summary.json").resolve()),
            "report_json": str(args.output_json.resolve()),
            "report_md": str(args.output_md.resolve()),
        },
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="V16 CPU-safe HumanPriorAdapter real/zero/shuffle probe.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--prior-channels", type=int, default=6)
    parser.add_argument("--summary-tokens", type=int, default=2)
    parser.add_argument("--summary-channels", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.03)
    args = parser.parse_args()

    summary = build_summary(args)
    write_json(args.output_json, summary)
    write_markdown(args.output_md, summary)
    print(
        json.dumps(
            json_ready(
                {
                    "status": summary["status"],
                    "report_json": args.output_json,
                    "report_md": args.output_md,
                    "summary_json": Path(summary["outputs"]["summary_json"]),
                    "train_final_loss": summary["metrics"]["adapter_train_sanity"]["final_loss"],
                    "train_real_zero_rms": summary["metrics"]["adapter_train_sanity"]["real_zero_output_rms_delta_after_train"],
                }
            ),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
