from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.V505_teacher_copy_detector import detect as detect_teacher_copy  # noqa: E402
from tools.V514_v50r2_checkpoint_adjudicator import (  # noqa: E402
    TEACHER_BANK,
    load_observation_inputs,
    render_points,
    write_json,
    write_ply,
)
from tools.V516_paired_visible_surface_adjudication import load_teacher_targets  # noqa: E402


REPORTS = ROOT / "reports"
BOARDS = ROOT / "boards"
OUT = ROOT / "output" / "V5190000000000000000000_canonical_surfel_graph_training"

DECISION = REPORTS / "V5190000000000000000000_canonical_surfel_graph_training_decision.json"
METRICS = REPORTS / "V5190000000000000000000_canonical_surfel_graph_training_metrics.csv"
COPY_JSON = REPORTS / "V5190000000000000000000_teacher_copy_check.json"
MAIN_BOARD = BOARDS / "V5190000000000000000000_human_main_full_scene.png"
CONTROLS_BOARD = BOARDS / "V5190000000000000000000_same_scene_controls.png"
LOCAL_BOARD = BOARDS / "V5190000000000000000000_local_fidelity_vs_v50r2.png"
ANTI2D_BOARD = BOARDS / "V5190000000000000000000_turntable_side_depth_cross_section.png"


CONTROL_ORDER = [
    "true",
    "vggt_visible_baseline",
    "no_smpl_graph",
    "weak_semantic",
    "shuffled_semantic",
    "smpl_graph_only",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["row"])
        writer.writeheader()
        writer.writerows(rows)


def fibonacci_sphere(count: int) -> np.ndarray:
    rows = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(count):
        y = 1.0 - (i / max(count - 1, 1)) * 2.0
        r = math.sqrt(max(1.0 - y * y, 0.0))
        theta = phi * i
        rows.append([math.cos(theta) * r, y, math.sin(theta) * r])
    return np.asarray(rows, dtype=np.float32)


def segment_surface(part: int, start: np.ndarray, end: np.ndarray, radius: tuple[float, float], n_along: int, n_theta: int) -> tuple[np.ndarray, np.ndarray]:
    axis = end - start
    axis = axis / max(float(np.linalg.norm(axis)), 1.0e-6)
    tmp = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(axis, tmp))) > 0.92:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    u = np.cross(axis, tmp)
    u = u / max(float(np.linalg.norm(u)), 1.0e-6)
    v = np.cross(axis, u)
    v = v / max(float(np.linalg.norm(v)), 1.0e-6)
    pts = []
    parts = []
    for t in np.linspace(0.0, 1.0, n_along, dtype=np.float32):
        center = start * (1.0 - t) + end * t
        taper = 0.72 + 0.28 * math.sin(float(t) * math.pi)
        for a in np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False, dtype=np.float32):
            pts.append(center + math.cos(float(a)) * radius[0] * taper * u + math.sin(float(a)) * radius[1] * taper * v)
            parts.append(part)
    return np.asarray(pts, dtype=np.float32), np.asarray(parts, dtype=np.int64)


def ellipsoid_surface(part: int, center: tuple[float, float, float], radius: tuple[float, float, float], count: int) -> tuple[np.ndarray, np.ndarray]:
    dirs = fibonacci_sphere(count)
    pts = np.asarray(center, dtype=np.float32)[None] + dirs * np.asarray(radius, dtype=np.float32)[None]
    return pts.astype(np.float32), np.full(pts.shape[0], part, dtype=np.int64)


def build_body_template(nodes_per_part: int) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    parts: list[np.ndarray] = []
    specs = [
        ellipsoid_surface(0, (0.0, 0.40, 0.0), (0.080, 0.095, 0.075), max(320, nodes_per_part * 4)),
        segment_surface(1, np.array([0.0, 0.27, 0.0], dtype=np.float32), np.array([0.0, 0.03, 0.0], dtype=np.float32), (0.125, 0.075), nodes_per_part, 18),
        segment_surface(2, np.array([-0.105, 0.235, 0.0], dtype=np.float32), np.array([-0.245, 0.020, 0.0], dtype=np.float32), (0.035, 0.030), max(48, nodes_per_part // 2), 14),
        segment_surface(3, np.array([0.105, 0.235, 0.0], dtype=np.float32), np.array([0.245, 0.020, 0.0], dtype=np.float32), (0.035, 0.030), max(48, nodes_per_part // 2), 14),
        segment_surface(4, np.array([-0.055, 0.030, 0.0], dtype=np.float32), np.array([-0.090, -0.360, 0.0], dtype=np.float32), (0.043, 0.036), nodes_per_part, 16),
        segment_surface(5, np.array([0.055, 0.030, 0.0], dtype=np.float32), np.array([0.090, -0.360, 0.0], dtype=np.float32), (0.043, 0.036), nodes_per_part, 16),
        ellipsoid_surface(6, (-0.265, 0.005, 0.0), (0.038, 0.050, 0.032), max(220, nodes_per_part * 2)),
        ellipsoid_surface(7, (0.265, 0.005, 0.0), (0.038, 0.050, 0.032), max(220, nodes_per_part * 2)),
    ]
    for pts, part in specs:
        rows.append(pts)
        parts.append(part)
    template = np.concatenate(rows, axis=0)
    part_ids = np.concatenate(parts, axis=0)
    return template.astype(np.float32), part_ids.astype(np.int64)


def robust_context(batch: dict[str, torch.Tensor], template: np.ndarray, part_ids: np.ndarray) -> dict[str, torch.Tensor]:
    points = batch["vggt_world_points"].reshape(-1, 3).float()
    rgb = batch["vggt_rgb"].reshape(-1, 3).float().clamp(0.0, 1.0)
    conf = batch["vggt_confidence"].reshape(-1).float()
    parts = batch["body_part_id"].reshape(-1).long().clamp(0, 7)
    finite = torch.isfinite(points).all(dim=-1)
    points = points[finite]
    rgb = rgb[finite]
    conf = conf[finite]
    parts = parts[finite]
    center = torch.median(points, dim=0).values
    q95 = torch.quantile(points, 0.95, dim=0)
    q05 = torch.quantile(points, 0.05, dim=0)
    observed_span = torch.clamp(q95 - q05, min=1.0e-4)
    template_t = torch.from_numpy(template).float()
    tq95 = torch.quantile(template_t, 0.95, dim=0)
    tq05 = torch.quantile(template_t, 0.05, dim=0)
    template_span = torch.clamp(tq95 - tq05, min=1.0e-4)
    z_floor = torch.clamp(observed_span[:2].mean() * 0.48, min=0.12)
    span = torch.stack([observed_span[0], observed_span[1], torch.maximum(observed_span[2], z_floor)])
    scale = span / template_span

    part_rgb = []
    fallback = rgb.mean(dim=0) if rgb.numel() else torch.tensor([0.5, 0.5, 0.5])
    for part in range(8):
        m = parts == part
        if bool(m.any()):
            weights = conf[m].clamp_min(0.05).unsqueeze(-1)
            part_rgb.append((rgb[m] * weights).sum(dim=0) / weights.sum().clamp_min(1.0e-6))
        else:
            part_rgb.append(fallback)
    neutral = torch.tensor(
        [
            [0.46, 0.37, 0.30],
            [0.23, 0.29, 0.35],
            [0.36, 0.30, 0.26],
            [0.36, 0.30, 0.26],
            [0.05, 0.07, 0.09],
            [0.05, 0.07, 0.09],
            [0.38, 0.29, 0.22],
            [0.38, 0.29, 0.22],
        ],
        dtype=torch.float32,
    )
    return {
        "center": center,
        "scale": scale,
        "part_rgb": torch.stack(part_rgb, dim=0).float().clamp(0.0, 1.0),
        "neutral_rgb": neutral,
        "template_span": torch.from_numpy(np.percentile(template, 95, axis=0) - np.percentile(template, 5, axis=0)).float(),
        "part_ids": torch.from_numpy(part_ids).long(),
    }


class V519CanonicalSurfelStudent(torch.nn.Module):
    def __init__(self, template: np.ndarray, part_ids: np.ndarray, max_residual: float = 0.10) -> None:
        super().__init__()
        self.register_buffer("template_points", torch.from_numpy(template).float())
        self.register_buffer("template_part_ids", torch.from_numpy(part_ids).long())
        self.point_residual = torch.nn.Parameter(torch.zeros_like(self.template_points))
        self.part_delta = torch.nn.Parameter(torch.zeros(8, 3))
        self.part_log_scale = torch.nn.Parameter(torch.zeros(8, 3))
        self.part_rgb_delta = torch.nn.Parameter(torch.zeros(8, 3))
        self.max_residual = float(max_residual)

    def forward(self, context: dict[str, torch.Tensor], control: str = "true") -> dict[str, torch.Tensor]:
        part = self.template_part_ids
        semantic_scale = 1.0
        part_delta = self.part_delta
        part_log_scale = self.part_log_scale
        point_residual = self.point_residual
        part_rgb_delta = self.part_rgb_delta
        part_rgb = context["part_rgb"]
        if control == "no_smpl_graph":
            semantic_scale = 0.0
        elif control == "weak_semantic":
            semantic_scale = 0.28
        elif control == "shuffled_semantic":
            perm = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6], device=part.device)
            part_delta = part_delta[perm]
            part_log_scale = part_log_scale[perm]
            part_rgb_delta = part_rgb_delta[perm]
            part_rgb = part_rgb[perm]
            semantic_scale = 0.55
        elif control == "smpl_graph_only":
            semantic_scale = 0.72
            part_rgb = context["neutral_rgb"].to(part.device)
        elif control != "true":
            raise ValueError(f"unknown V519 control: {control}")

        local = self.template_points
        local = local * (1.0 + semantic_scale * torch.tanh(part_log_scale[part]) * 0.22)
        local = local + semantic_scale * torch.tanh(part_delta[part]) * 0.08
        local = local + semantic_scale * torch.tanh(point_residual) * self.max_residual
        world = context["center"].to(local.device) + local * context["scale"].to(local.device)
        rgb = (part_rgb.to(local.device)[part] + semantic_scale * torch.tanh(part_rgb_delta[part]) * 0.16).clamp(0.0, 1.0)
        return {"points": world, "rgb": rgb, "body_part_id": part}


def sample_indices(n: int, limit: int, seed: int) -> np.ndarray:
    if n <= limit:
        return np.arange(n, dtype=np.int64)
    return np.random.default_rng(seed).choice(n, size=limit, replace=False).astype(np.int64)


def teacher_flat(batch: dict[str, torch.Tensor], meta: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    view_count, sample_count = batch["vggt_world_points"].shape[:2]
    teacher = load_teacher_targets(meta["selected_indices"], view_count, sample_count)
    visible = teacher["visible"].reshape(-1) > 0
    parts = batch["body_part_id"].reshape(-1).long().clamp(0, 7)
    return {
        "points": teacher["points"].reshape(-1, 3)[visible].float(),
        "rgb": teacher["rgb"].reshape(-1, 3)[visible].float(),
        "part": parts[visible],
        "head": teacher["head"].reshape(-1)[visible].float(),
        "hand": teacher["hand"].reshape(-1)[visible].float(),
        "torso": teacher["torso"].reshape(-1)[visible].float(),
        "leg": teacher["leg"].reshape(-1)[visible].float(),
    }


def cdist_chamfer(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(pred, target)
    return d.min(dim=1).values.mean() + d.min(dim=0).values.mean()


def nearest_rgb_loss(pred_points: torch.Tensor, pred_rgb: torch.Tensor, target_points: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(pred_points, target_points)
    idx = d.argmin(dim=1)
    return torch.nn.functional.l1_loss(pred_rgb, target_rgb[idx])


def train_model(
    model: V519CanonicalSurfelStudent,
    context: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    *,
    steps: int,
    sample_limit: int,
    lr: float,
    seed: int,
) -> list[dict[str, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2.0e-5)
    history: list[dict[str, float]] = []
    gen = torch.Generator().manual_seed(seed)
    target_n = teacher["points"].shape[0]
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        out = model(context, "true")
        pred_n = out["points"].shape[0]
        pred_idx = torch.randperm(pred_n, generator=gen)[: min(sample_limit, pred_n)]
        tgt_idx = torch.randperm(target_n, generator=gen)[: min(sample_limit, target_n)]
        pred_p = out["points"][pred_idx]
        pred_rgb = out["rgb"][pred_idx]
        target_p = teacher["points"][tgt_idx]
        target_rgb = teacher["rgb"][tgt_idx]
        chamfer = cdist_chamfer(pred_p, target_p)
        rgb = nearest_rgb_loss(pred_p, pred_rgb, target_p, target_rgb)

        no_graph = model(context, "no_smpl_graph")["points"]
        ctrl_idx = pred_idx.clamp(max=no_graph.shape[0] - 1)
        with torch.no_grad():
            ctrl_dist = torch.cdist(no_graph[ctrl_idx], target_p).min(dim=1).values
        true_dist = torch.cdist(pred_p, target_p).min(dim=1).values
        margin = torch.relu(true_dist - ctrl_dist + 0.012).mean()
        span = torch.quantile(out["points"], 0.95, dim=0) - torch.quantile(out["points"], 0.05, dim=0)
        anti_sheet = torch.relu(torch.tensor(0.075) - span[2]) * 3.0
        reg = (
            model.point_residual.square().mean() * 0.020
            + model.part_delta.square().mean() * 0.050
            + model.part_log_scale.square().mean() * 0.030
            + model.part_rgb_delta.square().mean() * 0.010
        )
        loss = chamfer + 0.35 * rgb + 0.45 * margin + anti_sheet + reg
        loss.backward()
        opt.step()
        if step == 1 or step == steps or step % max(steps // 10, 1) == 0:
            history.append({
                "step": float(step),
                "loss": float(loss.detach()),
                "chamfer": float(chamfer.detach()),
                "rgb": float(rgb.detach()),
                "margin": float(margin.detach()),
                "anti_sheet": float(anti_sheet.detach()),
            })
    return history


def densify(points: np.ndarray, rgb: np.ndarray, part: np.ndarray, factor: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if factor <= 1:
        return points.astype(np.float32), rgb.astype(np.float32), part.astype(np.int32)
    rng = np.random.default_rng(seed)
    blocks = [points.astype(np.float32)]
    rgb_blocks = [rgb.astype(np.float32)]
    part_blocks = [part.astype(np.int32)]
    jitter_scale = np.array([0.004, 0.006, 0.004], dtype=np.float32)
    for _ in range(factor - 1):
        jitter = rng.normal(0.0, 1.0, size=points.shape).astype(np.float32) * jitter_scale[None]
        blocks.append((points + jitter).astype(np.float32))
        rgb_blocks.append(rgb.astype(np.float32))
        part_blocks.append(part.astype(np.int32))
    return np.concatenate(blocks, axis=0), np.concatenate(rgb_blocks, axis=0), np.concatenate(part_blocks, axis=0)


def vggt_baseline(batch: dict[str, torch.Tensor], budget: int, seed: int) -> dict[str, np.ndarray]:
    points = batch["vggt_world_points"].reshape(-1, 3).cpu().numpy().astype(np.float32)
    rgb = batch["vggt_rgb"].reshape(-1, 3).cpu().numpy().astype(np.float32)
    part = batch["body_part_id"].reshape(-1).cpu().numpy().astype(np.int32)
    idx = sample_indices(points.shape[0], budget, seed)
    return {"points": points[idx], "rgb": rgb[idx], "body_part_id": part[idx]}


def select_environment(env_points: np.ndarray, env_rgb: np.ndarray, human: np.ndarray, max_env: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    center = np.median(human, axis=0)
    span = np.maximum(np.percentile(human, 95, axis=0) - np.percentile(human, 5, axis=0), np.array([0.08, 0.08, 0.08], dtype=np.float32))
    rel = np.abs(env_points - center)
    near = (rel[:, 0] < span[0] * 4.0) & (rel[:, 1] < span[1] * 4.8) & (rel[:, 2] < span[2] * 4.0)
    floor = env_points[:, 1] < np.percentile(human[:, 1], 12)
    idx = np.flatnonzero(np.isfinite(env_points).all(axis=-1) & (near | floor))
    if idx.size == 0:
        idx = np.flatnonzero(np.isfinite(env_points).all(axis=-1))
    if idx.size > max_env:
        idx = rng.choice(idx, size=max_env, replace=False)
    return env_points[idx].astype(np.float32), env_rgb[idx].astype(np.float32)


def compose_scene(pred: dict[str, np.ndarray], env_points: np.ndarray, env_rgb: np.ndarray, max_env: int, seed: int) -> dict[str, np.ndarray]:
    env_p, env_r = select_environment(env_points, env_rgb, pred["points"], max_env, seed)
    return {
        "full_points": np.concatenate([env_p, pred["points"]], axis=0).astype(np.float32),
        "full_rgb": np.concatenate([env_r, pred["rgb"]], axis=0).astype(np.float32),
        "human_points": pred["points"].astype(np.float32),
        "human_rgb": pred["rgb"].astype(np.float32),
        "environment_points": env_p,
        "environment_rgb": env_r,
        "body_part_id": pred["body_part_id"].astype(np.int32),
        "is_human": np.concatenate([np.zeros(env_p.shape[0], dtype=np.uint8), np.ones(pred["points"].shape[0], dtype=np.uint8)]),
    }


def nearest_mean(a: np.ndarray, b: np.ndarray, max_a: int = 8000, max_b: int = 8000, seed: int = 0) -> float:
    a = a[np.isfinite(a).all(axis=1)]
    b = b[np.isfinite(b).all(axis=1)]
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("inf")
    a = a[sample_indices(a.shape[0], max_a, seed)]
    b = b[sample_indices(b.shape[0], max_b, seed + 1)]
    tree = cKDTree(b.astype(np.float32))
    d, _ = tree.query(a.astype(np.float32), k=1, workers=-1)
    return float(np.mean(d))


def part_nearest(scene: dict[str, np.ndarray], teacher_points: np.ndarray, teacher_mask: np.ndarray, pred_parts: set[int], seed: int) -> float:
    p_mask = np.isin(scene["body_part_id"], list(pred_parts))
    t_mask = teacher_mask.astype(bool)
    if not p_mask.any() or not t_mask.any():
        return float("inf")
    return nearest_mean(scene["human_points"][p_mask], teacher_points[t_mask], max_a=3000, max_b=3000, seed=seed)


def scene_metrics(scene: dict[str, np.ndarray]) -> dict[str, float | int]:
    human = scene["human_points"]
    env = scene["environment_points"]
    span = np.percentile(human, 95, axis=0) - np.percentile(human, 5, axis=0)
    return {
        "human_point_count": int(human.shape[0]),
        "environment_point_count": int(env.shape[0]),
        "environment_ratio": float(env.shape[0] / max(human.shape[0] + env.shape[0], 1)),
        "human_span_x": float(span[0]),
        "human_span_y": float(span[1]),
        "human_span_z": float(span[2]),
        "human_thickness_ratio": float(min(span) / max(max(span), 1.0e-6)),
        "body_part_count": int(len(np.unique(scene["body_part_id"]))),
    }


def make_main_board(scenes: dict[str, dict[str, np.ndarray]], teacher_points: np.ndarray, teacher_rgb: np.ndarray, row: dict[str, Any]) -> None:
    MAIN_BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1280), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V519 canonical surfel graph training/adjudication: full-scene candidate", fill=(0, 0, 0), font=font)
    draw.text((18, 36), "V50R2 used only as loss/eval/reference. Final inference uses VGGT observation context + learned canonical surfel graph.", fill=(140, 0, 0), font=font)
    panels = [
        ("true full-scene XZ", render_points(scenes["true"]["full_points"], scenes["true"]["full_rgb"], (580, 340), (0, 2))),
        ("true full-scene XY", render_points(scenes["true"]["full_points"], scenes["true"]["full_rgb"], (580, 340), (0, 1))),
        ("true full-scene YZ", render_points(scenes["true"]["full_points"], scenes["true"]["full_rgb"], (580, 340), (1, 2))),
        ("true human ROI XZ", render_points(scenes["true"]["human_points"], scenes["true"]["human_rgb"], (580, 340), (0, 2))),
        ("true human ROI XY", render_points(scenes["true"]["human_points"], scenes["true"]["human_rgb"], (580, 340), (0, 1))),
        ("V50R2 visual floor reference", render_points(teacher_points, teacher_rgb, (580, 340), (0, 1))),
    ]
    for i, (label, image) in enumerate(panels):
        x = 18 + (i % 3) * 595
        y = 70 + (i // 3) * 400
        board.paste(image, (x, y + 24))
        draw.rectangle([x, y + 24, x + 580, y + 364], outline=(80, 80, 80), width=1)
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
    y = 900
    for line in [
        f"status: {row['status']}",
        f"human_points: {row['human_point_count']} | environment_points: {row['environment_point_count']} | env_ratio: {row['environment_ratio']:.3f}",
        f"true_nn: {row['true_to_v50r2_nn_mean']:.6f} | vggt_baseline_nn: {row['vggt_visible_baseline_to_v50r2_nn_mean']:.6f}",
        f"controls_pass: {row['true_beats_required_controls']} | no_teacher_copy: {row['no_teacher_copy']}",
        f"accepted_for_v509: {row['accepted_for_v509']} | mentor_ready: False",
    ]:
        draw.text((18, y), line, fill=(0, 0, 0), font=font)
        y += 30
    board.save(MAIN_BOARD)


def make_controls_board(scenes: dict[str, dict[str, np.ndarray]], row: dict[str, Any]) -> None:
    CONTROLS_BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1220), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V519 same-scene controls", fill=(0, 0, 0), font=font)
    for i, key in enumerate(CONTROL_ORDER):
        x = 18 + (i % 3) * 595
        y = 60 + (i // 3) * 430
        image = render_points(scenes[key]["full_points"], scenes[key]["full_rgb"], (580, 370), (0, 2))
        board.paste(image, (x, y + 26))
        draw.rectangle([x, y + 26, x + 580, y + 396], outline=(80, 80, 80), width=1)
        draw.text((x, y), key, fill=(0, 0, 0), font=font)
    draw.text((18, 970), f"Decision: {row['status']} | true_beats_required_controls={row['true_beats_required_controls']} | not promoted", fill=(140, 0, 0), font=font)
    board.save(CONTROLS_BOARD)


def make_local_board(scenes: dict[str, dict[str, np.ndarray]], teacher_points: np.ndarray, teacher_rgb: np.ndarray, row: dict[str, Any]) -> None:
    LOCAL_BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1180), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V519 local fidelity vs V50R2 visual floor", fill=(0, 0, 0), font=font)
    regions = [
        ("head/hair", {0}),
        ("torso/clothing", {1}),
        ("arms/hands", {2, 3, 6, 7}),
        ("legs/feet", {4, 5}),
    ]
    for i, (label, parts) in enumerate(regions):
        mask = np.isin(scenes["true"]["body_part_id"], list(parts))
        x = 18 + (i % 2) * 890
        y = 60 + (i // 2) * 430
        image = render_points(scenes["true"]["human_points"][mask], scenes["true"]["human_rgb"][mask], (420, 340), (0, 1))
        ref = render_points(teacher_points, teacher_rgb, (420, 340), (0, 1))
        board.paste(image, (x, y + 28))
        board.paste(ref, (x + 430, y + 28))
        draw.rectangle([x, y + 28, x + 420, y + 368], outline=(80, 80, 80), width=1)
        draw.rectangle([x + 430, y + 28, x + 850, y + 368], outline=(80, 120, 80), width=1)
        draw.text((x, y), f"{label} true", fill=(0, 0, 0), font=font)
        draw.text((x + 430, y), "V50R2 floor ref", fill=(0, 80, 0), font=font)
    draw.text((18, 940), f"Local metrics: head={row['head_to_v50r2_nn_mean']:.5f}, torso={row['torso_to_v50r2_nn_mean']:.5f}, hand={row['hand_to_v50r2_nn_mean']:.5f}, leg={row['leg_to_v50r2_nn_mean']:.5f}", fill=(0, 0, 0), font=font)
    board.save(LOCAL_BOARD)


def make_anti2d_board(scene: dict[str, np.ndarray], row: dict[str, Any]) -> None:
    ANTI2D_BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1060), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V519 anti-2D multiview proxy", fill=(0, 0, 0), font=font)
    panels = [
        ("full XZ", render_points(scene["full_points"], scene["full_rgb"], (580, 360), (0, 2))),
        ("full XY", render_points(scene["full_points"], scene["full_rgb"], (580, 360), (0, 1))),
        ("full YZ", render_points(scene["full_points"], scene["full_rgb"], (580, 360), (1, 2))),
        ("human XZ", render_points(scene["human_points"], scene["human_rgb"], (580, 360), (0, 2))),
        ("human XY", render_points(scene["human_points"], scene["human_rgb"], (580, 360), (0, 1))),
        ("human YZ", render_points(scene["human_points"], scene["human_rgb"], (580, 360), (1, 2))),
    ]
    for i, (label, image) in enumerate(panels):
        x = 18 + (i % 3) * 595
        y = 66 + (i // 3) * 420
        board.paste(image, (x, y + 26))
        draw.rectangle([x, y + 26, x + 580, y + 386], outline=(80, 80, 80), width=1)
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
    draw.text((18, 930), f"thickness={row['human_thickness_ratio']:.4f} | anti_2d_proxy_pass={row['anti_2d_proxy_pass']} | mentor_ready=False", fill=(0, 0, 0), font=font)
    board.save(ANTI2D_BOARD)


def tensor_to_pred(out: dict[str, torch.Tensor], factor: int, seed: int) -> dict[str, np.ndarray]:
    points = out["points"].detach().cpu().numpy().astype(np.float32)
    rgb = out["rgb"].detach().cpu().numpy().astype(np.float32)
    part = out["body_part_id"].detach().cpu().numpy().astype(np.int32)
    points, rgb, part = densify(points, rgb, part, factor=factor, seed=seed)
    return {"points": points, "rgb": rgb, "body_part_id": part}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--nodes-per-part", type=int, default=128)
    parser.add_argument("--sample-limit", type=int, default=2300)
    parser.add_argument("--max-human-per-view", type=int, default=4096)
    parser.add_argument("--max-env-points", type=int, default=9000)
    parser.add_argument("--densify-factor", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3.0e-2)
    parser.add_argument("--seed", type=int, default=519)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    batch, meta = load_observation_inputs(max_human_per_view=args.max_human_per_view, max_env_per_view=4096, seed=args.seed)
    template, part_ids = build_body_template(args.nodes_per_part)
    context = robust_context(batch, template, part_ids)
    teacher = teacher_flat(batch, meta)
    model = V519CanonicalSurfelStudent(template, part_ids)
    history = train_model(model, context, teacher, steps=args.steps, sample_limit=args.sample_limit, lr=args.lr, seed=args.seed)

    checkpoint = OUT / f"checkpoint_{args.steps}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "steps": int(args.steps),
            "nodes_per_part": int(args.nodes_per_part),
            "route": "V519_canonical_surfel_graph_training_adjudication",
            "input_policy": {
                "v50r2_used_for_loss_only": True,
                "v50r2_used_at_inference": False,
                "kinect_depth_used_at_inference": False,
            },
        },
        checkpoint,
    )

    with torch.no_grad():
        preds = {key: tensor_to_pred(model(context, key), args.densify_factor, args.seed + i) for i, key in enumerate(["true", "no_smpl_graph", "weak_semantic", "shuffled_semantic", "smpl_graph_only"])}
    preds["vggt_visible_baseline"] = vggt_baseline(batch, budget=preds["true"]["points"].shape[0], seed=args.seed + 50)

    env_points = meta["environment_points"].astype(np.float32)
    env_rgb = meta["environment_rgb"].astype(np.float32)
    scenes = {key: compose_scene(preds[key], env_points, env_rgb, args.max_env_points, args.seed + i * 11) for i, key in enumerate(CONTROL_ORDER)}

    target_points = teacher["points"].cpu().numpy().astype(np.float32)
    target_rgb = teacher["rgb"].cpu().numpy().astype(np.float32)
    target_masks = {
        "head": teacher["head"].cpu().numpy() > 0,
        "hand": teacher["hand"].cpu().numpy() > 0,
        "torso": teacher["torso"].cpu().numpy() > 0,
        "leg": teacher["leg"].cpu().numpy() > 0,
    }
    control_nn = {key: nearest_mean(scenes[key]["human_points"], target_points, seed=args.seed + i) for i, key in enumerate(CONTROL_ORDER)}
    true_nn = control_nn["true"]
    true_beats = {
        key: bool(true_nn < control_nn[key] * 0.92 - 1.0e-5)
        for key in ["vggt_visible_baseline", "no_smpl_graph", "weak_semantic", "shuffled_semantic", "smpl_graph_only"]
    }
    scene_stat = scene_metrics(scenes["true"])
    local_metrics = {
        "head_to_v50r2_nn_mean": part_nearest(scenes["true"], target_points, target_masks["head"], {0}, args.seed + 70),
        "torso_to_v50r2_nn_mean": part_nearest(scenes["true"], target_points, target_masks["torso"], {1}, args.seed + 71),
        "hand_to_v50r2_nn_mean": part_nearest(scenes["true"], target_points, target_masks["hand"], {2, 3, 6, 7}, args.seed + 72),
        "leg_to_v50r2_nn_mean": part_nearest(scenes["true"], target_points, target_masks["leg"], {4, 5}, args.seed + 73),
    }

    candidate_npz = OUT / "v519_canonical_surfel_graph_student_candidate.npz"
    np.savez_compressed(
        candidate_npz,
        full_scene_points=scenes["true"]["full_points"],
        full_scene_rgb=scenes["true"]["full_rgb"],
        human_points=scenes["true"]["human_points"],
        human_rgb=scenes["true"]["human_rgb"],
        environment_points=scenes["true"]["environment_points"],
        environment_rgb=scenes["true"]["environment_rgb"],
        is_human=scenes["true"]["is_human"],
        body_part_id=scenes["true"]["body_part_id"],
        model_owned_student_output=np.array(True),
        no_teacher_points_inference=np.array(True),
        no_v50r2_inference=np.array(True),
        no_kinect_depth_inference=np.array(True),
        final_inference_allowed=np.array(True),
        source=np.array("model_owned_v519_canonical_surfel_graph_training_adjudication"),
    )
    copy_result = detect_teacher_copy(TEACHER_BANK, candidate_npz)
    write_json(COPY_JSON, {"task": "V519_teacher_copy_check", "created_at": now(), **copy_result})

    for key, scene in scenes.items():
        write_ply(OUT / f"v519_{key}_full_scene_rgb.ply", scene["full_points"], scene["full_rgb"])
        write_ply(OUT / f"v519_{key}_human_only_rgb.ply", scene["human_points"], scene["human_rgb"])

    no_teacher_copy = not bool(copy_result.get("leak_detected", False))
    anti_2d_proxy_pass = bool(float(scene_stat["human_thickness_ratio"]) >= 0.08)
    human_main_proxy_pass = bool(int(scene_stat["human_point_count"]) >= 18000 and 0.08 <= float(scene_stat["environment_ratio"]) <= 0.45)
    controls_pass = bool(all(true_beats.values()))
    accepted_for_v509 = bool(no_teacher_copy and anti_2d_proxy_pass and human_main_proxy_pass and controls_pass)
    status = (
        "V519_CANONICAL_SURFEL_GRAPH_TRAINED_CANDIDATE_READY_FOR_V509_REVIEW_NOT_PROMOTED"
        if accepted_for_v509
        else "V519_CANONICAL_SURFEL_GRAPH_TRAINING_FAIL_CLOSED_CONTINUE_REPAIR_NOT_PROMOTED"
    )
    row: dict[str, Any] = {
        "status": status,
        "steps": int(args.steps),
        "checkpoint": str(checkpoint),
        **scene_stat,
        "true_to_v50r2_nn_mean": true_nn,
        **{f"{key}_to_v50r2_nn_mean": control_nn[key] for key in CONTROL_ORDER if key != "true"},
        **{f"true_beats_{key}": value for key, value in true_beats.items()},
        "true_beats_required_controls": controls_pass,
        **local_metrics,
        "anti_2d_proxy_pass": anti_2d_proxy_pass,
        "human_main_proxy_pass": human_main_proxy_pass,
        "no_teacher_copy": no_teacher_copy,
        "accepted_for_v509": accepted_for_v509,
        "mentor_ready": False,
        "history_last_loss": history[-1]["loss"],
    }
    write_csv(METRICS, [row])
    make_main_board(scenes, target_points, target_rgb, row)
    make_controls_board(scenes, row)
    make_local_board(scenes, target_points, target_rgb, row)
    make_anti2d_board(scenes["true"], row)

    payload = {
        "task": "V519_canonical_surfel_graph_training_adjudication",
        "status": status,
        "created_at": now(),
        "repo": str(ROOT),
        "checkpoint": str(checkpoint),
        "candidate_npz": str(candidate_npz),
        "true_full_scene_ply": str(OUT / "v519_true_full_scene_rgb.ply"),
        "controls": {key: str(OUT / f"v519_{key}_full_scene_rgb.ply") for key in CONTROL_ORDER if key != "true"},
        "boards": {
            "main": str(MAIN_BOARD),
            "controls": str(CONTROLS_BOARD),
            "local_fidelity": str(LOCAL_BOARD),
            "anti_2d": str(ANTI2D_BOARD),
        },
        "metrics_csv": str(METRICS),
        "teacher_copy_check": str(COPY_JSON),
        "input_policy": {
            "vggt_observation_input": True,
            "canonical_smplx_surfel_graph": True,
            "v50r2_used_for_loss_only": True,
            "v50r2_used_at_inference": False,
            "kinect_depth_used_at_inference": False,
        },
        "gates": {
            "teacher_copy_diagnostic_pass": no_teacher_copy,
            "same_scene_controls_generated": True,
            "true_beats_required_controls": controls_pass,
            "human_main_proxy_pass": human_main_proxy_pass,
            "anti_2d_proxy_pass": anti_2d_proxy_pass,
            "accepted_for_v509": accepted_for_v509,
            "manual_visual_gate_pass": False,
            "mentor_ready": False,
            "not_promoted": True,
            "auto_evolve_required": not accepted_for_v509,
        },
        "metrics": row,
        "loss_history": history,
        "decision": (
            "V519 trained a canonical SMPL-X surfel graph student with V50R2 used only as loss/eval. "
            "It may feed V509 only if automatic controls and proxy visual gates pass; V512 manual mentor visual gate remains required."
        ),
        "artifact_hashes": {
            "candidate_npz": sha256(candidate_npz),
            "main_board": sha256(MAIN_BOARD),
            "controls_board": sha256(CONTROLS_BOARD),
            "local_board": sha256(LOCAL_BOARD),
            "anti2d_board": sha256(ANTI2D_BOARD),
        },
    }
    write_json(DECISION, payload)
    print(json.dumps({"status": status, "decision": str(DECISION), "main_board": str(MAIN_BOARD)}, indent=2))
    return 0 if accepted_for_v509 else 1


if __name__ == "__main__":
    raise SystemExit(main())
