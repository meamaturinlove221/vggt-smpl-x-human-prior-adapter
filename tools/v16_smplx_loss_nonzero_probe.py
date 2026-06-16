from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
LOCAL_ROOT = REPO_ROOT / "output" / "surface_research_preflight_local"
DEFAULT_OUTPUT_DIR = LOCAL_ROOT / "V16_smplx_loss_nonzero_probe"
DEFAULT_JSON = REPORTS / "20260508_v16_smplx_loss_nonzero_probe.json"
DEFAULT_MD = REPORTS / "20260508_v16_smplx_loss_nonzero_probe.md"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "training") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "training"))

from training.loss_smplx_native_prior import DEFAULT_NATIVE_HUMAN_PRIOR, SMPLXNativePriorLoss  # noqa: E402


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
    if "surface_research_preflight_local" not in lower or "v16_smplx_loss_nonzero_probe" not in lower:
        raise ValueError(f"Refusing output path outside the V16 SMPL-X loss probe sandbox: {resolved}")
    for token in ("strict_pass", "strict_gate_registry", "candidate_export", "teacher_export", "predictions"):
        if token in lower:
            raise ValueError(f"Refusing forbidden output token {token!r}: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def scalar(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def grad_norm(value: torch.Tensor | None) -> float:
    if value is None or value.grad is None:
        return 0.0
    return float(torch.linalg.vector_norm(value.grad.detach()).cpu())


def tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    flat = value.detach().reshape(-1).float()
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return {"count": int(flat.numel()), "finite": 0}
    return {
        "count": int(flat.numel()),
        "finite": int(finite.numel()),
        "min": float(finite.min().cpu()),
        "mean": float(finite.mean().cpu()),
        "max": float(finite.max().cpu()),
    }


def make_grid(batch_size: int, seq_len: int, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.linspace(-1.0, 1.0, height)
    x = torch.linspace(-1.0, 1.0, width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = xx.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
    yy = yy.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
    for view_idx in range(seq_len):
        xx[:, view_idx] = xx[:, view_idx] + 0.03 * view_idx
        yy[:, view_idx] = yy[:, view_idx] - 0.02 * view_idx
    return xx.float(), yy.float()


def make_prediction_tensors(batch_size: int, seq_len: int, height: int, width: int) -> dict[str, torch.Tensor]:
    xx, yy = make_grid(batch_size, seq_len, height, width)
    depth = (1.8 + 0.15 * xx + 0.08 * yy).unsqueeze(-1).clone().requires_grad_(True)
    world_points = torch.stack(
        [
            0.45 * xx + 0.05 * torch.sin(math.pi * yy),
            0.35 * yy + 0.04 * torch.cos(math.pi * xx),
            depth.detach()[..., 0] + 0.10 * xx * yy,
        ],
        dim=-1,
    ).clone().requires_grad_(True)
    depth_conf = torch.full((batch_size, seq_len, height, width), 0.85, dtype=torch.float32, requires_grad=True)
    points_conf = torch.full((batch_size, seq_len, height, width), 0.80, dtype=torch.float32, requires_grad=True)
    return {
        "depth": depth,
        "world_points": world_points,
        "depth_conf": depth_conf,
        "world_points_conf": points_conf,
    }


def make_masks(batch_size: int, seq_len: int, height: int, width: int) -> dict[str, torch.Tensor]:
    y_idx = torch.arange(height).view(1, 1, height, 1)
    x_idx = torch.arange(width).view(1, 1, 1, width)
    body = (y_idx >= int(0.20 * height)) & (y_idx < int(0.78 * height)) & (x_idx >= int(0.25 * width)) & (x_idx < int(0.75 * width))
    left = (y_idx >= int(0.42 * height)) & (y_idx < int(0.66 * height)) & (x_idx >= int(0.08 * width)) & (x_idx < int(0.24 * width))
    right = (y_idx >= int(0.42 * height)) & (y_idx < int(0.66 * height)) & (x_idx >= int(0.76 * width)) & (x_idx < int(0.92 * width))
    masks = {
        "smplx_body_anchor_mask": body.repeat(batch_size, seq_len, 1, 1),
        "smplx_left_hand_anchor_mask": left.repeat(batch_size, seq_len, 1, 1),
        "smplx_right_hand_anchor_mask": right.repeat(batch_size, seq_len, 1, 1),
    }
    masks["smplx_hand_anchor_mask"] = masks["smplx_left_hand_anchor_mask"] | masks["smplx_right_hand_anchor_mask"]
    masks["prior_mask"] = masks["smplx_body_anchor_mask"] | masks["smplx_hand_anchor_mask"]
    return {key: value.bool() for key, value in masks.items()}


def make_targets(predictions: dict[str, torch.Tensor], variant: str) -> tuple[torch.Tensor, torch.Tensor]:
    base_depth = predictions["depth"].detach()
    base_points = predictions["world_points"].detach()
    batch_size, seq_len, height, width, _ = base_depth.shape
    xx, yy = make_grid(batch_size, seq_len, height, width)
    offset_depth = 0.22 + 0.06 * torch.sin(math.pi * xx) + 0.04 * torch.cos(math.pi * yy)
    offset_points = torch.stack(
        [
            0.18 + 0.03 * yy,
            -0.11 + 0.02 * xx,
            0.16 + 0.04 * torch.sin(math.pi * (xx + yy)),
        ],
        dim=-1,
    )
    real_depth = base_depth + offset_depth.unsqueeze(-1)
    real_points = base_points + offset_points

    if variant == "real":
        return real_depth, real_points
    if variant == "zero":
        return torch.zeros_like(real_depth), torch.zeros_like(real_points)
    if variant == "shuffle":
        return real_depth.roll(shifts=1, dims=1).flip(dims=(3,)), real_points.roll(shifts=1, dims=1).flip(dims=(3,))
    raise ValueError(f"Unknown target variant: {variant}")


def make_batch(predictions: dict[str, torch.Tensor], variant: str, include_prior_targets: bool = True) -> dict[str, torch.Tensor]:
    batch_size, seq_len, height, width, _ = predictions["depth"].shape
    batch = make_masks(batch_size, seq_len, height, width)
    if include_prior_targets:
        prior_depths, prior_points = make_targets(predictions, variant)
        batch["prior_depths"] = prior_depths
        batch["prior_points"] = prior_points
    return batch


def make_loss() -> SMPLXNativePriorLoss:
    human_prior = {
        "weight": 1.0,
        "smplx_weak_anchor": {
            "body_weight": 0.25,
            "hand_weight": 0.50,
            "depth_loss_weight": 0.25,
            "point_loss_weight": 0.50,
            "gamma": 1.0,
            "alpha": 0.0,
            "valid_range": -1,
            "min_pixels": 8,
            "supervise_conf": False,
            "use_separate_hand_masks": True,
            "exclude_head_roi": False,
            "exclude_face_roi": False,
            "exclude_hairline_roi": False,
            "exclude_ear_band_roi": False,
        },
    }
    native_prior = {
        "use_default_human_prior": False,
        "strict_required_keys": False,
        "required_batch_keys": [
            "prior_depths",
            "prior_points",
            "smplx_body_anchor_mask",
            "smplx_left_hand_anchor_mask",
            "smplx_right_hand_anchor_mask",
        ],
        "required_prediction_keys": ["depth", "world_points"],
    }
    return SMPLXNativePriorLoss(human_prior=human_prior, native_prior=native_prior)


def reset_grads(predictions: dict[str, torch.Tensor]) -> None:
    for value in predictions.values():
        if torch.is_tensor(value) and value.grad is not None:
            value.grad.zero_()


def loss_values(loss_dict: dict[str, Any]) -> dict[str, float]:
    return {
        key: scalar(value)
        for key, value in sorted(loss_dict.items())
        if torch.is_tensor(value) and value.numel() == 1
    }


def run_case(loss_fn: SMPLXNativePriorLoss, variant: str, batch: dict[str, torch.Tensor], predictions: dict[str, torch.Tensor]) -> dict[str, Any]:
    reset_grads(predictions)
    loss_dict = loss_fn(predictions, batch)
    objective = loss_dict["objective"]
    objective.backward()
    return {
        "variant": variant,
        "losses": loss_values(loss_dict),
        "gradient_norms": {
            "depth": grad_norm(predictions["depth"]),
            "world_points": grad_norm(predictions["world_points"]),
            "depth_conf": grad_norm(predictions["depth_conf"]),
            "world_points_conf": grad_norm(predictions["world_points_conf"]),
        },
        "mask_pixels": {
            "body": int(batch.get("smplx_body_anchor_mask", torch.zeros(())).sum().item()) if "smplx_body_anchor_mask" in batch else 0,
            "left_hand": int(batch.get("smplx_left_hand_anchor_mask", torch.zeros(())).sum().item()) if "smplx_left_hand_anchor_mask" in batch else 0,
            "right_hand": int(batch.get("smplx_right_hand_anchor_mask", torch.zeros(())).sum().item()) if "smplx_right_hand_anchor_mask" in batch else 0,
            "prior": int(batch.get("prior_mask", torch.zeros(())).sum().item()) if "prior_mask" in batch else 0,
        },
    }


def run_missing_key_cases(loss_fn: SMPLXNativePriorLoss, predictions: dict[str, torch.Tensor]) -> dict[str, Any]:
    dummy_batch = make_batch(predictions, "real", include_prior_targets=False)
    dummy_batch.pop("smplx_left_hand_anchor_mask", None)
    dummy_batch.pop("smplx_right_hand_anchor_mask", None)
    reset_grads(predictions)
    loss_dict = loss_fn(predictions, dummy_batch)
    loss_dict["objective"].backward()
    missing = loss_fn._find_missing_keys(predictions, dummy_batch)

    strict_loss = SMPLXNativePriorLoss(
        human_prior=loss_fn.human_prior,
        native_prior={
            "strict_required_keys": True,
            "required_batch_keys": list(loss_fn.required_batch_keys),
            "required_prediction_keys": list(loss_fn.required_prediction_keys),
        },
    )
    strict_error = None
    try:
        strict_loss(predictions, dummy_batch)
    except KeyError as exc:
        strict_error = str(exc)
    return {
        "missing_keys": missing,
        "expected_missing_key_count": len(missing),
        "reported_missing_key_count": scalar(loss_dict["loss_smplx_native_missing_key_count"]),
        "dummy_zero_losses": loss_values(loss_dict),
        "gradient_norms": {
            "depth": grad_norm(predictions["depth"]),
            "world_points": grad_norm(predictions["world_points"]),
            "depth_conf": grad_norm(predictions["depth_conf"]),
            "world_points_conf": grad_norm(predictions["world_points_conf"]),
        },
        "strict_error_contains_missing": bool(strict_error and "missing required keys" in strict_error),
        "strict_error": strict_error,
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V16 SMPL-X Prior Loss Nonzero Probe",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only local probe. No strict pass, package, registry, teacher export, candidate export, or prediction bundle is written.",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Key Metrics",
        "",
    ]
    for name, row in summary["cases"].items():
        lines.append(
            f"- {name}: objective=`{row['losses'].get('objective')}` "
            f"weak_anchor=`{row['losses'].get('loss_prior_smplx_weak_anchor')}` "
            f"grad_depth=`{row['gradient_norms']['depth']}` "
            f"grad_points=`{row['gradient_norms']['world_points']}`"
        )
    missing = summary["missing_key_probe"]
    lines.extend(
        [
            f"- missing_key_count: reported=`{missing['reported_missing_key_count']}` expected=`{missing['expected_missing_key_count']}`",
            f"- dummy_zero_objective: `{missing['dummy_zero_losses'].get('objective')}`",
            "",
            "## Inspected Files",
            "",
        ]
    )
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
    predictions = make_prediction_tensors(args.batch_size, args.seq_len, args.height, args.width)
    loss_fn = make_loss()

    cases = {}
    for variant in ("real", "zero", "shuffle"):
        batch = make_batch(predictions, variant, include_prior_targets=True)
        cases[variant] = run_case(loss_fn, variant, batch, predictions)

    missing_key_probe = run_missing_key_cases(loss_fn, predictions)

    real_loss = cases["real"]["losses"].get("loss_prior_smplx_weak_anchor", 0.0)
    zero_loss = cases["zero"]["losses"].get("loss_prior_smplx_weak_anchor", 0.0)
    shuffle_loss = cases["shuffle"]["losses"].get("loss_prior_smplx_weak_anchor", 0.0)
    nonzero_grads = (
        cases["real"]["gradient_norms"]["depth"] > 1e-8
        and cases["real"]["gradient_norms"]["world_points"] > 1e-8
    )
    losses_distinct = len({round(real_loss, 7), round(zero_loss, 7), round(shuffle_loss, 7)}) >= 2
    dummy_zero_ok = abs(missing_key_probe["dummy_zero_losses"].get("objective", 1.0)) < 1e-12
    missing_count_ok = (
        int(missing_key_probe["reported_missing_key_count"])
        == int(missing_key_probe["expected_missing_key_count"])
        and int(missing_key_probe["expected_missing_key_count"]) > 0
    )

    blockers = []
    if real_loss <= 0.0:
        blockers.append("Real SMPL-X weak-anchor prior loss was not positive.")
    if not losses_distinct:
        blockers.append("Real/zero/shuffle weak-anchor losses were not distinguishable.")
    if not nonzero_grads:
        blockers.append("Relevant prediction tensors did not receive nonzero gradients.")
    if not dummy_zero_ok:
        blockers.append("Missing prior target path did not remain dummy zero.")
    if not missing_count_ok:
        blockers.append("Native prior missing-key count did not match expected missing keys.")

    summary = {
        "task": "V16-DLINE-ROUTED-SMPLX Agent C SMPL-X native prior loss nonzero/gradient probe",
        "created_utc": utc_now(),
        "status": "v16_smplx_prior_loss_nonzero_with_gradients" if not blockers else "v16_smplx_prior_loss_probe_blocked",
        "research_only": True,
        "no_strict_pass": True,
        "no_package": True,
        "device": "cpu",
        "seed": int(args.seed),
        "inspected_files": [
            {"path": str((REPO_ROOT / file_path).resolve()), "exists": (REPO_ROOT / file_path).is_file()}
            for file_path in INSPECTED_FILES
        ],
        "loss_config": {
            "wrapper": "training.loss_smplx_native_prior.SMPLXNativePriorLoss",
            "default_native_human_prior_keys": sorted(DEFAULT_NATIVE_HUMAN_PRIOR.keys()),
            "active_branch": "compute_prior_smplx_weak_anchor_loss",
            "min_pixels": 8,
            "valid_range": -1,
            "supervise_conf": False,
        },
        "prediction_stats": {
            "depth": tensor_stats(predictions["depth"]),
            "world_points": tensor_stats(predictions["world_points"]),
            "depth_conf": tensor_stats(predictions["depth_conf"]),
            "world_points_conf": tensor_stats(predictions["world_points_conf"]),
        },
        "cases": cases,
        "missing_key_probe": missing_key_probe,
        "decision": (
            "SMPL-X weak-anchor prior loss is positive for real synthetic priors, changes under zero/shuffle controls, "
            "reports dummy-zero when required target keys are missing, and backpropagates nonzero gradients to depth/world_points."
            if not blockers
            else "One or more SMPL-X prior loss nonzero/gradient checks failed; inspect blockers before routing this as a live training signal."
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
    parser = argparse.ArgumentParser(description="V16 CPU-safe SMPL-X native prior loss nonzero/gradient probe.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--width", type=int, default=48)
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
                    "real_weak_anchor": summary["cases"]["real"]["losses"]["loss_prior_smplx_weak_anchor"],
                    "real_depth_grad": summary["cases"]["real"]["gradient_norms"]["depth"],
                    "real_points_grad": summary["cases"]["real"]["gradient_norms"]["world_points"],
                    "missing_key_count": summary["missing_key_probe"]["reported_missing_key_count"],
                }
            ),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
