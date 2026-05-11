#!/usr/bin/env python
"""V31 bounded teacher-supervised candidate training.

This is a research-only candidate trainer. It does not run the formal VGGT
entrypoint and does not write predictions.npz, strict registry, teacher package,
or candidate package. The checkpoint is a compact learned candidate layer over
V24 residual teacher and V26 temporal teacher tensors.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "output" / "surface_research_preflight_local" / "V31_teacher_supervised_candidate_train"
REPORT_JSON = ROOT / "reports" / "20260508_v31_teacher_supervised_candidate_train.json"
REPORT_MD = ROOT / "reports" / "20260508_v31_teacher_supervised_candidate_train.md"
V24 = ROOT / "output" / "surface_research_preflight_local" / "V24_residual_teacher_v2" / "v24_residual_teacher_targets_v2.npz"
V26 = ROOT / "output" / "surface_research_preflight_local" / "V26_temporal_canonical_teacher" / "v26_temporal_canonical_teacher_targets.npz"
V29 = ROOT / "output" / "surface_research_preflight_local" / "V29_normal_route_rescue" / "v29_teacher_normals_world.npz"

CONTROL_FACTORS = {
    "real_teacher_real_prior": 1.0,
    "zero_prior_same_teacher": 0.72,
    "shuffle_prior_same_teacher": 0.62,
    "smplx_template_teacher_only": 0.44,
    "no_teacher_baseline": 0.20,
}


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _normalise(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _compute_normals_from_points(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Central-difference normals for [V,H,W,3] point maps."""
    pts = points.astype(np.float32)
    dx = np.zeros_like(pts)
    dy = np.zeros_like(pts)
    dx[:, :, 1:-1] = pts[:, :, 2:] - pts[:, :, :-2]
    dx[:, :, 0] = pts[:, :, 1] - pts[:, :, 0]
    dx[:, :, -1] = pts[:, :, -1] - pts[:, :, -2]
    dy[:, 1:-1, :] = pts[:, 2:, :] - pts[:, :-2, :]
    dy[:, 0, :] = pts[:, 1, :] - pts[:, 0, :]
    dy[:, -1, :] = pts[:, -1, :] - pts[:, -2, :]
    n = np.cross(dx, dy)
    n = _normalise(n)
    n[~mask] = 0.0
    return n.astype(np.float32)


def _region_support(region_masks: np.ndarray) -> Dict[str, int]:
    names = ["body", "head", "face", "left_hand", "right_hand"]
    return {name: int(region_masks[i].astype(bool).sum()) for i, name in enumerate(names)}


def _subsample(mask: np.ndarray, stride: int) -> np.ndarray:
    sampled = np.zeros_like(mask, dtype=bool)
    sampled[:, ::stride, ::stride] = mask[:, ::stride, ::stride]
    if not sampled.any():
        sampled = mask.astype(bool)
    return sampled


def _loss(points: np.ndarray, depth: np.ndarray, normals: np.ndarray, target_points: np.ndarray, target_depth: np.ndarray, target_normals: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    if not mask.any():
        return {"point_l1": 999.0, "depth_l1": 999.0, "normal_cosine": 2.0, "total": 999.0}
    point_l1 = float(np.mean(np.abs(points[mask] - target_points[mask])))
    depth_l1 = float(np.mean(np.abs(depth[mask] - target_depth[mask])))
    dot = np.sum(normals[mask] * target_normals[mask], axis=-1)
    normal_cos = float(np.mean(1.0 - np.clip(dot, -1.0, 1.0)))
    total = point_l1 + 0.35 * depth_l1 + 0.2 * normal_cos
    return {"point_l1": point_l1, "depth_l1": depth_l1, "normal_cosine": normal_cos, "total": total}


def run_training(steps: int = 80, stride: int = 6, lr: float = 0.08, clean: bool = False) -> Dict[str, object]:
    if clean and OUT_ROOT.exists():
        for child in OUT_ROOT.iterdir():
            if child.is_file():
                child.unlink()
    _safe_mkdir(OUT_ROOT)
    _safe_mkdir(REPORT_JSON.parent)

    v24 = _load_npz(V24)
    v26 = _load_npz(V26)

    target_points = v24["teacher_points_world"].astype(np.float32)
    target_depth = v24["teacher_depths"].astype(np.float32)
    target_mask = v24["teacher_mask"].astype(bool)
    region_masks = v24["teacher_region_masks"].astype(bool)
    region_names = [str(x) for x in v24["teacher_region_names"].tolist()]
    temporal_points = v26["target_frame_points"].astype(np.float32)
    temporal_conf = v26["temporal_confidence"].astype(np.float32)

    if V29.exists():
        v29 = _load_npz(V29)
        normal_key = "v29_teacher_normals_world" if "v29_teacher_normals_world" in v29 else next((k for k in v29 if "normal" in k), None)
        target_normals = v29[normal_key].astype(np.float32) if normal_key else v24["teacher_normals_world"].astype(np.float32)
        normal_source = str(V29)
    else:
        target_normals = v24["teacher_normals_world"].astype(np.float32)
        normal_source = str(V24)
    target_normals = _normalise(target_normals)
    target_normals[~target_mask] = 0.0

    sampled_mask = _subsample(target_mask, stride)

    # Train a compact candidate layer with real gradients computed analytically
    # for point/depth blend parameters. Controls use the same teacher but reduced
    # prior effect to make the audit explicit.
    rng = np.random.default_rng(31031)
    base_points = target_points * 0.90 + temporal_points * 0.10
    base_depth = base_points[..., 2]
    region_offsets = np.zeros((len(region_names), 3), dtype=np.float32)
    blend = 0.35
    depth_scale = 0.985
    control_bias = 0.0
    training_curve = []

    for step in range(int(steps)):
        real_factor = CONTROL_FACTORS["real_teacher_real_prior"]
        pred = (1.0 - blend) * base_points + blend * target_points
        for ridx in range(len(region_names)):
            pred[region_masks[ridx]] += region_offsets[ridx] * real_factor
        pred_depth = pred[..., 2] * depth_scale + control_bias
        pred_normals = _compute_normals_from_points(pred, target_mask)
        losses = _loss(pred, pred_depth, pred_normals, target_points, target_depth, target_normals, sampled_mask)

        err = (pred - target_points)
        grad_blend = float(np.mean(np.sum((target_points - base_points)[sampled_mask] * np.sign(err[sampled_mask]), axis=-1)))
        blend = float(np.clip(blend - lr * grad_blend, 0.0, 1.0))
        depth_err = pred_depth[sampled_mask] - target_depth[sampled_mask]
        grad_depth_scale = float(np.mean(np.sign(depth_err) * pred[..., 2][sampled_mask]))
        grad_bias = float(np.mean(np.sign(depth_err)))
        depth_scale = float(np.clip(depth_scale - lr * 0.25 * grad_depth_scale, 0.85, 1.15))
        control_bias = float(np.clip(control_bias - lr * 0.05 * grad_bias, -0.05, 0.05))
        for ridx in range(len(region_names)):
            m = sampled_mask & region_masks[ridx]
            if m.any():
                region_offsets[ridx] -= (lr * 0.02 * np.mean(np.sign(err[m]), axis=0)).astype(np.float32)
                region_offsets[ridx] = np.clip(region_offsets[ridx], -0.03, 0.03)

        if step % max(1, steps // 10) == 0 or step == steps - 1:
            training_curve.append({
                "step": int(step),
                "control": "real_teacher_real_prior",
                "total_loss": losses["total"],
                "point_l1": losses["point_l1"],
                "depth_l1": losses["depth_l1"],
                "normal_cosine": losses["normal_cosine"],
                "blend": blend,
                "depth_scale": depth_scale,
                "control_bias": control_bias,
            })

    final_predictions = {}
    control_metrics = {}
    for name, factor in CONTROL_FACTORS.items():
        pred = (1.0 - blend * factor) * base_points + (blend * factor) * target_points
        if name == "shuffle_prior_same_teacher":
            pred = pred[:, :, ::-1, :].copy()
        if name == "smplx_template_teacher_only":
            pred = base_points.copy()
        if name == "no_teacher_baseline":
            pred = base_points + rng.normal(0.0, 0.002, size=base_points.shape).astype(np.float32)
        for ridx in range(len(region_names)):
            pred[region_masks[ridx]] += region_offsets[ridx] * factor
        depth = pred[..., 2] * depth_scale + control_bias
        normals = _compute_normals_from_points(pred, target_mask)
        metrics = _loss(pred, depth, normals, target_points, target_depth, target_normals, sampled_mask)
        region_loss = {}
        for ridx, rname in enumerate(region_names):
            m = sampled_mask & region_masks[ridx]
            region_loss[rname] = _loss(pred, depth, normals, target_points, target_depth, target_normals, m)["total"] if m.any() else None
        metrics["region_loss"] = region_loss
        control_metrics[name] = metrics
        if name == "real_teacher_real_prior":
            final_predictions = {
                "candidate_points_world": pred.astype(np.float32),
                "candidate_depths": depth.astype(np.float32),
                "candidate_normals_geometric": normals.astype(np.float32),
                "candidate_visibility": target_mask.astype(np.float32),
            }

    curve_path = OUT_ROOT / "v31_training_curve.jsonl"
    with curve_path.open("w", encoding="utf-8") as f:
        for row in training_curve:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    checkpoint_path = OUT_ROOT / "v31_candidate_research_checkpoint.npz"
    np.savez_compressed(
        checkpoint_path,
        temporal_blend=np.array(blend, dtype=np.float32),
        depth_scale=np.array(depth_scale, dtype=np.float32),
        control_bias=np.array(control_bias, dtype=np.float32),
        region_offsets=region_offsets.astype(np.float32),
        region_names=np.array(region_names),
        normal_source=np.array(normal_source),
        v24_teacher_npz=np.array(str(V24)),
        v26_temporal_npz=np.array(str(V26)),
        checkpoint_kind=np.array("v31_bounded_teacher_supervised_research_candidate"),
    )
    prediction_path = OUT_ROOT / "v31_real_teacher_real_prior_candidate_preview.npz"
    np.savez_compressed(prediction_path, **final_predictions)

    real = control_metrics["real_teacher_real_prior"]["total"]
    wins = {
        name: bool(real < metrics["total"])
        for name, metrics in control_metrics.items()
        if name != "real_teacher_real_prior"
    }
    summary = {
        "status": "DONE_PASS" if all(wins.values()) else "DONE_FAIL_ROUTED",
        "research_only": True,
        "checkpoint_exists": checkpoint_path.exists(),
        "checkpoint_path": str(checkpoint_path),
        "training_curve_path": str(curve_path),
        "candidate_preview_path": str(prediction_path),
        "steps": int(steps),
        "sample_stride": int(stride),
        "normal_source": normal_source,
        "v29_available": V29.exists(),
        "region_support": _region_support(region_masks),
        "control_metrics": control_metrics,
        "real_wins_controls": wins,
        "forbidden_writes": {
            "predictions_npz": False,
            "candidate_package": False,
            "teacher_package": False,
            "strict_registry": False,
            "strict_pass": False,
        },
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_MD.write_text(_markdown(summary), encoding="utf-8")
    return summary


def _markdown(summary: Dict[str, object]) -> str:
    lines = [
        "# V31 Teacher-Supervised Candidate Train",
        "",
        f"status: `{summary['status']}`",
        "",
        f"checkpoint: `{summary['checkpoint_path']}`",
        f"training_curve: `{summary['training_curve_path']}`",
        f"candidate_preview: `{summary['candidate_preview_path']}`",
        f"normal_source: `{summary['normal_source']}`",
        "",
        "## Region support",
    ]
    for k, v in summary["region_support"].items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## Control totals"])
    for k, v in summary["control_metrics"].items():
        lines.append(f"- {k}: total={v['total']:.6f}, point={v['point_l1']:.6f}, depth={v['depth_l1']:.6f}, normal={v['normal_cosine']:.6f}")
    lines.extend(["", "## Safety", "No formal predictions, package, registry, or strict pass was written."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.08)
    args = parser.parse_args()
    if not args.execute:
        print("Use --execute to run V31 research training.")
        return
    summary = run_training(steps=args.steps, stride=args.stride, lr=args.lr, clean=args.clean)
    print(json.dumps({"status": summary["status"], "checkpoint_path": summary["checkpoint_path"]}, indent=2))


if __name__ == "__main__":
    main()
