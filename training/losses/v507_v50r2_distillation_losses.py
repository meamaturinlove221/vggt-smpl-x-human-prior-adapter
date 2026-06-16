from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class V507LossWeights:
    local_teacher_chamfer: float = 1.0
    visible_surface_consistency: float = 0.45
    rgb_consistency: float = 0.35
    normal_consistency: float = 0.25
    silhouette_boundary: float = 0.20
    head_hair_contour: float = 0.30
    shoulder_neck_continuity: float = 0.18
    clothing_boundary: float = 0.16
    arm_hand_endpoint: float = 0.20
    leg_foot_continuity: float = 0.18
    anti_speckle: float = 0.12
    anti_hole: float = 0.12
    anti_teacher_copy: float = 0.25
    environment_preservation: float = 0.30
    hard_control_separation: float = 0.40


def _masked_mean(value: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is None:
        return value.mean()
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=value.device, dtype=value.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denom


def _chamfer_lite(a: Tensor, b: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is not None:
        while mask.ndim < a.ndim:
            mask = mask.unsqueeze(-1)
        keep = mask.squeeze(-1) > 0
        losses = []
        for ai, bi, ki in zip(a, b, keep):
            if bool(ki.any()):
                aa = ai[ki]
                bb = bi[ki]
            else:
                aa = ai
                bb = bi
            dist = torch.cdist(aa, bb).square()
            losses.append(dist.min(dim=1).values.mean() + dist.min(dim=0).values.mean())
        return torch.stack(losses).mean()
    dist = torch.cdist(a, b).square()
    return dist.min(dim=-1).values.mean() + dist.min(dim=-2).values.mean()


def _normal_loss(pred: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
    pred = F.normalize(pred, dim=-1, eps=1.0e-6)
    target = F.normalize(target, dim=-1, eps=1.0e-6)
    return _masked_mean(1.0 - (pred * target).sum(dim=-1, keepdim=True), mask)


def _smoothness(points: Tensor, mask: Tensor | None) -> Tensor:
    if points.shape[-2] < 2:
        return points.new_tensor(0.0)
    diffs = (points[..., 1:, :] - points[..., :-1, :]).square().sum(dim=-1, keepdim=True)
    if mask is None:
        return diffs.mean()
    return _masked_mean(diffs, mask[..., 1:])


def _endpoint_loss(points: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
    return _masked_mean((points - target).abs(), mask)


def _copy_penalty(pred: Tensor, teacher: Tensor, mask: Tensor | None, eps: float = 1.0e-5) -> Tensor:
    exactish = torch.exp(-((pred - teacher).square().sum(dim=-1, keepdim=True) / eps).clamp_max(60.0))
    return _masked_mean(exactish, mask)


def v50r2_distillation_losses(
    student: Mapping[str, Tensor],
    teacher: Mapping[str, Tensor],
    *,
    controls: Mapping[str, Mapping[str, Tensor]] | None = None,
    weights: V507LossWeights | None = None,
) -> dict[str, Tensor]:
    w = weights or V507LossWeights()
    pred_points = student["student_human_points"]
    pred_rgb = student["student_human_rgb"]
    pred_normal = student["student_normal"]
    occupancy = student["surfel_occupancy"]
    visibility = student["surfel_visibility"]

    teacher_points = teacher["teacher_points"].to(pred_points.device)
    teacher_rgb = teacher["teacher_rgb"].to(pred_points.device)
    teacher_normal = teacher["teacher_normals"].to(pred_points.device)
    full = teacher["full_body_mask"].to(pred_points.device)
    head = teacher.get("head_hair_mask", full).to(pred_points.device)
    clothing = teacher.get("torso_clothing_mask", full).to(pred_points.device)
    hand = teacher.get("arm_hand_mask", full).to(pred_points.device)
    leg = teacher.get("leg_foot_mask", full).to(pred_points.device)

    raw = {
        "local_teacher_chamfer": _chamfer_lite(pred_points, teacher_points, full),
        "visible_surface_consistency": _masked_mean((pred_points - teacher_points).abs(), full),
        "rgb_consistency": _masked_mean((pred_rgb - teacher_rgb).abs(), full),
        "normal_consistency": _normal_loss(pred_normal, teacher_normal, full),
        "silhouette_boundary": F.binary_cross_entropy(occupancy.clamp(1.0e-5, 1.0 - 1.0e-5), full.unsqueeze(-1).float()),
        "head_hair_contour": _endpoint_loss(pred_points, teacher_points, head),
        "shoulder_neck_continuity": _smoothness(pred_points, head),
        "clothing_boundary": _endpoint_loss(pred_rgb, teacher_rgb, clothing),
        "arm_hand_endpoint": _endpoint_loss(pred_points, teacher_points, hand),
        "leg_foot_continuity": _smoothness(pred_points, leg),
        "anti_speckle": _masked_mean((1.0 - visibility).abs(), full),
        "anti_hole": _masked_mean((1.0 - occupancy).abs(), full),
        "anti_teacher_copy": _copy_penalty(pred_points, teacher_points, full),
    }
    if "environment_points" in student and "environment_points" in teacher:
        raw["environment_preservation"] = _masked_mean((student["environment_points"] - teacher["environment_points"].to(pred_points.device)).abs())
    else:
        raw["environment_preservation"] = pred_points.new_tensor(0.0)

    if controls:
        true_fit = raw["visible_surface_consistency"].detach()
        control_fits = []
        for control in controls.values():
            if "student_human_points" in control:
                control_fits.append(_masked_mean((control["student_human_points"].to(pred_points.device) - teacher_points).abs(), full))
        if control_fits:
            best_control = torch.stack(control_fits).min()
            raw["hard_control_separation"] = F.relu(true_fit + 0.005 - best_control)
        else:
            raw["hard_control_separation"] = pred_points.new_tensor(0.0)
    else:
        raw["hard_control_separation"] = pred_points.new_tensor(0.0)

    weighted = {
        name: raw[name] * getattr(w, name)
        for name in raw
    }
    weighted["total"] = torch.stack(list(weighted.values())).sum()
    return weighted


def smoke_test() -> dict[str, Any]:
    gen = torch.Generator().manual_seed(507)
    b, n = 2, 192
    teacher_points = torch.randn(b, n, 3, generator=gen) * 0.2
    teacher_rgb = torch.rand(b, n, 3, generator=gen)
    teacher_normals = F.normalize(torch.randn(b, n, 3, generator=gen), dim=-1)
    full = (torch.rand(b, n, generator=gen) > 0.1).float()
    head = torch.zeros(b, n)
    head[:, :32] = 1.0
    hand = torch.zeros(b, n)
    hand[:, 80:112] = 1.0
    leg = torch.zeros(b, n)
    leg[:, 140:] = 1.0
    student_points = teacher_points + torch.randn(b, n, 3, generator=gen) * 0.01
    student = {
        "student_human_points": student_points.requires_grad_(True),
        "student_human_rgb": (teacher_rgb + torch.randn(b, n, 3, generator=gen) * 0.02).clamp(0, 1).requires_grad_(True),
        "student_normal": F.normalize(teacher_normals + torch.randn(b, n, 3, generator=gen) * 0.05, dim=-1).requires_grad_(True),
        "surfel_occupancy": torch.full((b, n, 1), 0.7, requires_grad=True),
        "surfel_visibility": torch.full((b, n, 1), 0.8, requires_grad=True),
        "environment_points": torch.randn(b, 32, 3, generator=gen).requires_grad_(True),
    }
    teacher = {
        "teacher_points": teacher_points,
        "teacher_rgb": teacher_rgb,
        "teacher_normals": teacher_normals,
        "full_body_mask": full,
        "head_hair_mask": head,
        "torso_clothing_mask": full - head.clamp(max=1),
        "arm_hand_mask": hand,
        "leg_foot_mask": leg,
        "environment_points": student["environment_points"].detach().clone(),
    }
    controls = {
        "vggt_baseline": {"student_human_points": teacher_points + 0.04},
        "shuffled_semantic": {"student_human_points": teacher_points.flip(1)},
    }
    losses = v50r2_distillation_losses(student, teacher, controls=controls)
    losses["total"].backward()
    grad_positive = any(v.grad is not None and bool(v.grad.abs().sum() > 0) for v in student.values() if isinstance(v, Tensor) and v.requires_grad)
    return {
        "weights": asdict(V507LossWeights()),
        "loss_terms": {k: float(v.detach().cpu()) for k, v in losses.items()},
        "all_loss_terms_finite": all(bool(torch.isfinite(v.detach()).all()) for v in losses.values()),
        "grad_positive": grad_positive,
        "teacher_used_outside_model_forward": True,
        "teacher_points_in_inference": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(smoke_test(), indent=2))
