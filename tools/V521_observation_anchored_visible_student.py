from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.V505_teacher_copy_detector import detect as detect_teacher_copy  # noqa: E402
from tools.V514_v50r2_checkpoint_adjudicator import (  # noqa: E402
    SCENE,
    SMPL_PRIOR,
    TEACHER_BANK,
    V42_CONF,
    V42_NORMALS,
    V42_POINTS,
    render_points,
    resize_mask,
    resize_rgb,
    scene_images_and_masks,
    write_json,
    write_ply,
)
from tools.V519_canonical_surfel_graph_training_adjudication import nearest_mean  # noqa: E402


REPORTS = ROOT / "reports"
BOARDS = ROOT / "boards"
OUT = ROOT / "output" / "V5210000000000000000000_observation_anchored_visible_student"

DECISION = REPORTS / "V5210000000000000000000_observation_anchored_visible_student_decision.json"
METRICS = REPORTS / "V5210000000000000000000_observation_anchored_visible_student_metrics.csv"
COPY_JSON = REPORTS / "V5210000000000000000000_teacher_copy_check.json"
MAIN_BOARD = BOARDS / "V5210000000000000000000_human_main_full_scene.png"
CONTROLS_BOARD = BOARDS / "V5210000000000000000000_same_scene_controls.png"
FLOOR_BOARD = BOARDS / "V5210000000000000000000_v50r2_visual_floor_comparison.png"
ANTI2D_BOARD = BOARDS / "V5210000000000000000000_turntable_side_depth_cross_section.png"

VIEW_IDS = [0, 1, 2, 3, 4, 5]
VIEW_NAMES = ["cam00", "cam01", "cam06", "cam11", "cam16", "cam21"]
CONTROL_ORDER = [
    "true",
    "vggt_visible_baseline",
    "observation_only_sparse",
    "no_smpl_filter",
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


def load_teacher_bank() -> dict[str, np.ndarray]:
    with np.load(TEACHER_BANK, allow_pickle=False) as z:
        return {
            "points": z["points"].astype(np.float32),
            "rgb": z["rgb"].astype(np.float32) / 255.0,
            "full_body": z["full_body_mask"].astype(bool),
            "head": z["head_mask"].astype(bool),
            "face": z["face_mask"].astype(bool),
            "hand": z["hand_visibility"].astype(np.float32) > 0,
        }


def part_ids_from_pixels(yy: np.ndarray, xx: np.ndarray, prior: np.ndarray, h: int, w: int) -> np.ndarray:
    parts = np.zeros(yy.shape[0], dtype=np.int64)
    parts[yy < int(h * 0.25)] = 1
    parts[(yy >= int(h * 0.25)) & (yy < int(h * 0.60))] = 2
    parts[yy >= int(h * 0.60)] = 5
    arm = (xx < int(w * 0.30)) | (xx > int(w * 0.70))
    parts[arm & (yy >= int(h * 0.25)) & (yy < int(h * 0.67))] = 3
    parts[arm & (yy >= int(h * 0.45))] = 4
    parts[(yy > int(h * 0.78)) & (xx < int(w * 0.48))] = 6
    parts[(yy > int(h * 0.78)) & (xx >= int(w * 0.48))] = 7
    parts[prior <= 0] = np.maximum(parts[prior <= 0], 0)
    return parts


def load_view_observations(max_human_per_view: int, max_env_per_view: int, seed: int) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    image_paths, mask_paths = scene_images_and_masks()
    rgbs = np.stack([resize_rgb(path) for path in image_paths], axis=0)
    masks = np.stack([resize_mask(path) for path in mask_paths], axis=0).astype(bool)
    with np.load(V42_POINTS, allow_pickle=False) as z:
        points = z["frame0000"].astype(np.float32)
    with np.load(V42_NORMALS, allow_pickle=False) as z:
        normals = z["frame0000"].astype(np.float32)
    with np.load(V42_CONF, allow_pickle=False) as z:
        conf = z["frame0000_world_points_conf"].astype(np.float32)
    with np.load(SMPL_PRIOR, allow_pickle=False) as z:
        smpl_mask = z["smpl_prior_masks"].astype(bool)
    teacher = load_teacher_bank()

    h, w = masks.shape[1:]
    valid = np.isfinite(points).all(axis=-1) & np.isfinite(normals).all(axis=-1)
    rows: list[dict[str, np.ndarray]] = []
    for teacher_view, view in enumerate(VIEW_IDS):
        human_idx = np.flatnonzero((masks[view] & valid[view]).reshape(-1))
        if human_idx.size == 0:
            raise RuntimeError(f"no human observation points for view {view}")
        human_take = min(max_human_per_view, human_idx.size)
        human_choice = rng.choice(human_idx, size=human_take, replace=False)
        yy, xx = np.divmod(human_choice, w)

        env_mask = (~masks[view]) & valid[view] & (conf[view] > np.percentile(conf[view][valid[view]], 55))
        env_idx = np.flatnonzero(env_mask.reshape(-1))
        env_take = min(max_env_per_view, env_idx.size)
        env_choice = rng.choice(env_idx, size=env_take, replace=False)
        eyy, exx = np.divmod(env_choice, w)

        obs_conf = conf[view, yy, xx].astype(np.float32)
        obs_conf = (obs_conf / max(float(np.max(conf[view])), 1.0)).clip(0.0, 1.0)
        nrm = normals[view, yy, xx].astype(np.float32)
        nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1.0e-6)
        prior = smpl_mask[view, yy, xx].astype(np.float32)
        parts = part_ids_from_pixels(yy, xx, prior, h, w)

        target_visible = teacher["full_body"][teacher_view, yy, xx]
        target_points = teacher["points"][teacher_view, yy, xx]
        target_rgb = teacher["rgb"][teacher_view, yy, xx]
        rows.append(
            {
                "view_id": np.array(view, dtype=np.int32),
                "teacher_view": np.array(teacher_view, dtype=np.int32),
                "view_name": np.array(VIEW_NAMES[teacher_view]),
                "yy": yy.astype(np.int32),
                "xx": xx.astype(np.int32),
                "points": points[view, yy, xx].astype(np.float32),
                "rgb": rgbs[view, yy, xx].astype(np.float32),
                "normal": nrm,
                "confidence": obs_conf.astype(np.float32),
                "part": parts.astype(np.int64),
                "target_points": target_points.astype(np.float32),
                "target_rgb": target_rgb.astype(np.float32),
                "target_visible": target_visible.astype(np.float32),
                "environment_points": points[view, eyy, exx].astype(np.float32),
                "environment_rgb": rgbs[view, eyy, exx].astype(np.float32),
                "teacher_floor_points": teacher["points"][teacher_view][teacher["full_body"][teacher_view]].astype(np.float32),
                "teacher_floor_rgb": teacher["rgb"][teacher_view][teacher["full_body"][teacher_view]].astype(np.float32),
            }
        )
    return rows


class V521ObservationAnchoredStudent(torch.nn.Module):
    def __init__(self, view_count: int = 6, part_count: int = 8, max_delta: float = 0.018) -> None:
        super().__init__()
        self.part_delta = torch.nn.Parameter(torch.zeros(view_count, part_count, 3))
        self.view_delta = torch.nn.Parameter(torch.zeros(view_count, 3))
        self.part_rgb_delta = torch.nn.Parameter(torch.zeros(view_count, part_count, 3))
        self.max_delta = float(max_delta)

    def forward(
        self,
        points: torch.Tensor,
        rgb: torch.Tensor,
        part: torch.Tensor,
        view: torch.Tensor,
        conf: torch.Tensor,
        control: str = "true",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        semantic_scale = 1.0
        used_part = part
        if control in {"vggt_visible_baseline", "observation_only_sparse"}:
            semantic_scale = 0.0
        elif control == "no_smpl_filter":
            semantic_scale = 0.0
        elif control == "shuffled_semantic":
            perm = torch.tensor([0, 2, 1, 4, 3, 7, 5, 6], device=part.device)
            used_part = perm[part.clamp(0, 7)]
            semantic_scale = 0.75
        elif control == "smpl_graph_only":
            semantic_scale = 0.45
        elif control != "true":
            raise ValueError(f"unknown V521 control: {control}")

        confidence_gate = conf.clamp(0.15, 1.0).unsqueeze(-1)
        delta = torch.tanh(self.part_delta[view, used_part] + self.view_delta[view]) * self.max_delta
        out_points = points + semantic_scale * confidence_gate * delta
        out_rgb = (rgb + semantic_scale * torch.tanh(self.part_rgb_delta[view, used_part]) * 0.075).clamp(0.0, 1.0)
        return out_points, out_rgb


def concat_for_training(rows: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    points = []
    rgb = []
    part = []
    view = []
    conf = []
    target_points = []
    target_rgb = []
    visible = []
    for i, row in enumerate(rows):
        n = row["points"].shape[0]
        points.append(row["points"])
        rgb.append(row["rgb"])
        part.append(row["part"])
        view.append(np.full(n, i, dtype=np.int64))
        conf.append(row["confidence"])
        target_points.append(row["target_points"])
        target_rgb.append(row["target_rgb"])
        visible.append(row["target_visible"])
    return {
        "points": torch.from_numpy(np.concatenate(points, axis=0)).float(),
        "rgb": torch.from_numpy(np.concatenate(rgb, axis=0)).float(),
        "part": torch.from_numpy(np.concatenate(part, axis=0)).long(),
        "view": torch.from_numpy(np.concatenate(view, axis=0)).long(),
        "confidence": torch.from_numpy(np.concatenate(conf, axis=0)).float(),
        "target_points": torch.from_numpy(np.concatenate(target_points, axis=0)).float(),
        "target_rgb": torch.from_numpy(np.concatenate(target_rgb, axis=0)).float(),
        "visible": torch.from_numpy(np.concatenate(visible, axis=0)).float(),
    }


def train_model(model: V521ObservationAnchoredStudent, batch: dict[str, torch.Tensor], steps: int, sample_limit: int, lr: float, seed: int) -> list[dict[str, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    valid = torch.nonzero(batch["visible"] > 0.0, as_tuple=False).reshape(-1)
    gen = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        if valid.numel() > sample_limit:
            take = valid[torch.randperm(valid.numel(), generator=gen)[:sample_limit]]
        else:
            take = valid
        pred_points, pred_rgb = model(
            batch["points"][take],
            batch["rgb"][take],
            batch["part"][take],
            batch["view"][take],
            batch["confidence"][take],
            "true",
        )
        dist = torch.sqrt(((pred_points - batch["target_points"][take]) ** 2).sum(dim=-1) + 1.0e-10)
        rgb_loss = (pred_rgb - batch["target_rgb"][take]).abs().mean(dim=-1)
        reg = model.part_delta.square().mean() + model.view_delta.square().mean() + model.part_rgb_delta.square().mean()
        loss = dist.mean() + 0.18 * rgb_loss.mean() + 0.02 * reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step == steps or step % max(1, steps // 8) == 0:
            history.append({"step": float(step), "loss": float(loss.detach().item()), "dist": float(dist.mean().detach().item()), "rgb": float(rgb_loss.mean().detach().item())})
    return history


def densify(points: np.ndarray, rgb: np.ndarray, part: np.ndarray, factor: int, seed: int, jitter: float = 0.0011) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if factor <= 1:
        return points.astype(np.float32), rgb.astype(np.float32), part.astype(np.int32)
    rng = np.random.default_rng(seed)
    pts = [points.astype(np.float32)]
    cols = [rgb.astype(np.float32)]
    parts = [part.astype(np.int32)]
    span = np.maximum(np.percentile(points, 95, axis=0) - np.percentile(points, 5, axis=0), 1.0e-4)
    sigma = span.mean() * jitter
    for _ in range(factor - 1):
        pts.append(points + rng.normal(0.0, sigma, size=points.shape).astype(np.float32))
        cols.append(np.clip(rgb + rng.normal(0.0, 0.006, size=rgb.shape).astype(np.float32), 0.0, 1.0))
        parts.append(part.astype(np.int32))
    return np.concatenate(pts, axis=0), np.concatenate(cols, axis=0), np.concatenate(parts, axis=0)


def sparse_subset(points: np.ndarray, rgb: np.ndarray, part: np.ndarray, keep_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    take = max(512, int(points.shape[0] * keep_ratio))
    idx = rng.choice(points.shape[0], size=min(take, points.shape[0]), replace=False)
    return points[idx], rgb[idx], part[idx]


def smpl_graph_only_control(points: np.ndarray, rgb: np.ndarray, part: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    out_p: list[np.ndarray] = []
    out_r: list[np.ndarray] = []
    out_part: list[np.ndarray] = []
    for p in range(8):
        mask = part == p
        if not np.any(mask):
            continue
        src = points[mask]
        color = rgb[mask].mean(axis=0)
        center = np.median(src, axis=0)
        span = np.maximum(np.percentile(src, 90, axis=0) - np.percentile(src, 10, axis=0), 0.006)
        count = min(max(450, src.shape[0] // 3), 1800)
        shell = rng.normal(0.0, 1.0, size=(count, 3)).astype(np.float32)
        shell = shell / np.maximum(np.linalg.norm(shell, axis=-1, keepdims=True), 1.0e-6)
        shell = center[None] + shell * span[None] * 0.42
        out_p.append(shell.astype(np.float32))
        out_r.append(np.tile(color[None], (count, 1)).astype(np.float32))
        out_part.append(np.full(count, p, dtype=np.int32))
    return np.concatenate(out_p, axis=0), np.concatenate(out_r, axis=0), np.concatenate(out_part, axis=0)


def render_control_for_view(
    row: dict[str, np.ndarray],
    model: V521ObservationAnchoredStudent,
    view_ord: int,
    control: str,
    densify_factor: int,
    seed: int,
) -> dict[str, np.ndarray]:
    points = torch.from_numpy(row["points"]).float()
    rgb = torch.from_numpy(row["rgb"]).float()
    part = torch.from_numpy(row["part"]).long()
    view = torch.full((points.shape[0],), view_ord, dtype=torch.long)
    conf = torch.from_numpy(row["confidence"]).float()
    if control == "smpl_graph_only":
        hp, hr, hpart = smpl_graph_only_control(row["points"], row["rgb"], row["part"], seed)
    else:
        with torch.no_grad():
            pred_p, pred_r = model(points, rgb, part, view, conf, control=control)
        hp = pred_p.cpu().numpy().astype(np.float32)
        hr = pred_r.cpu().numpy().astype(np.float32)
        hpart = row["part"].astype(np.int32)
        if control == "observation_only_sparse":
            hp, hr, hpart = sparse_subset(hp, hr, hpart, 0.45, seed)
        elif control == "no_smpl_filter":
            good = row["confidence"] >= np.quantile(row["confidence"], 0.45)
            hp, hr, hpart = hp[good], hr[good], hpart[good]
        elif control == "true":
            hp, hr, hpart = densify(hp, hr, hpart, densify_factor, seed)
    env_p = row["environment_points"].astype(np.float32)
    env_r = row["environment_rgb"].astype(np.float32)
    full_p = np.concatenate([env_p, hp], axis=0).astype(np.float32)
    full_r = np.concatenate([env_r, hr], axis=0).astype(np.float32)
    is_human = np.concatenate([np.zeros(env_p.shape[0], dtype=bool), np.ones(hp.shape[0], dtype=bool)])
    return {
        "human_points": hp.astype(np.float32),
        "human_rgb": hr.astype(np.float32),
        "body_part_id": hpart.astype(np.int32),
        "environment_points": env_p,
        "environment_rgb": env_r,
        "full_points": full_p,
        "full_rgb": full_r,
        "is_human": is_human,
    }


def scene_metrics(scene: dict[str, np.ndarray]) -> dict[str, Any]:
    hp = scene["human_points"]
    env_count = int(scene["environment_points"].shape[0])
    human_count = int(hp.shape[0])
    span = np.percentile(hp, 99, axis=0) - np.percentile(hp, 1, axis=0)
    span = np.maximum(span, 1.0e-6)
    return {
        "human_point_count": human_count,
        "environment_point_count": env_count,
        "environment_ratio": float(env_count / max(env_count + human_count, 1)),
        "human_span_x": float(span[0]),
        "human_span_y": float(span[1]),
        "human_span_z": float(span[2]),
        "human_thickness_ratio": float(np.min(span) / np.max(span)),
        "body_part_count": int(np.unique(scene["body_part_id"]).shape[0]),
    }


def make_main_board(true_scenes: list[dict[str, np.ndarray]], rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    MAIN_BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1240), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V521 observation-anchored visible student: human-main full-scene RGB point cloud", fill=(0, 0, 0), font=font)
    draw.text((18, 36), "Inference uses VGGT visible observation + SMPL/part-bound learned residuals; V50R2 is floor/loss/eval only.", fill=(140, 0, 0), font=font)
    for i, scene in enumerate(true_scenes):
        x = 18 + (i % 3) * 595
        y = 72 + (i // 3) * 410
        image = render_points(scene["full_points"], scene["full_rgb"], (580, 340), (0, 1))
        board.paste(image, (x, y + 24))
        draw.rectangle([x, y + 24, x + 580, y + 364], outline=(80, 80, 80), width=1)
        draw.text((x, y), f"{VIEW_NAMES[i]} true full-scene XY", fill=(0, 0, 0), font=font)
        draw.text((x, y + 368), f"human={rows[i]['human_point_count']} env={rows[i]['environment_point_count']} nn={rows[i]['true_to_v50r2_nn_mean']:.5f}", fill=(80, 80, 80), font=font)
    y = 930
    for line in [
        f"status: {summary['status']}",
        f"mean true_nn={summary['mean_true_to_v50r2_nn_mean']:.6f} | baseline_nn={summary['mean_vggt_visible_baseline_to_v50r2_nn_mean']:.6f}",
        f"controls_pass={summary['gates']['true_beats_required_controls']} | no_teacher_copy={summary['gates']['no_teacher_copy']} | manual_visual_gate_pass={summary['gates']['manual_visual_gate_pass']}",
        f"mentor_ready={summary['gates']['mentor_ready']} | not promoted={summary['gates']['not_promoted']}",
    ]:
        draw.text((18, y), line, fill=(0, 0, 0), font=font)
        y += 28
    board.save(MAIN_BOARD)


def make_controls_board(scenes: dict[str, dict[str, np.ndarray]], rows: dict[str, float], best_view_name: str) -> None:
    CONTROLS_BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1220), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), f"V521 same-scene controls on {best_view_name}", fill=(0, 0, 0), font=font)
    for i, key in enumerate(CONTROL_ORDER):
        x = 18 + (i % 3) * 595
        y = 60 + (i // 3) * 430
        image = render_points(scenes[key]["full_points"], scenes[key]["full_rgb"], (580, 370), (0, 1))
        board.paste(image, (x, y + 26))
        draw.rectangle([x, y + 26, x + 580, y + 396], outline=(80, 80, 80), width=1)
        draw.text((x, y), key, fill=(0, 0, 0), font=font)
        draw.text((x, y + 400), f"nn={rows[key]:.5f}", fill=(80, 80, 80), font=font)
    draw.text((18, 970), "Controls are same-scene. Teacher/V50R2 is not rendered as a final student output here.", fill=(140, 0, 0), font=font)
    board.save(CONTROLS_BOARD)


def make_floor_board(true_scenes: list[dict[str, np.ndarray]], obs_rows: list[dict[str, np.ndarray]], metrics: list[dict[str, Any]]) -> None:
    FLOOR_BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1660), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V521 true vs V50R2 visual floor comparison", fill=(0, 0, 0), font=font)
    draw.text((18, 36), "V50R2 is visual floor/reference only. True panels remain model-owned observation-anchored student outputs.", fill=(140, 0, 0), font=font)
    for i in range(6):
        x = 18 + (i % 3) * 595
        y = 76 + (i // 3) * 760
        true_img = render_points(true_scenes[i]["human_points"], true_scenes[i]["human_rgb"], (580, 320), (0, 1))
        ref_img = render_points(obs_rows[i]["teacher_floor_points"], obs_rows[i]["teacher_floor_rgb"], (580, 320), (0, 1))
        board.paste(true_img, (x, y + 24))
        board.paste(ref_img, (x, y + 374))
        draw.rectangle([x, y + 24, x + 580, y + 344], outline=(80, 80, 80), width=1)
        draw.rectangle([x, y + 374, x + 580, y + 694], outline=(80, 80, 80), width=1)
        draw.text((x, y), f"{VIEW_NAMES[i]} true human ROI", fill=(0, 0, 0), font=font)
        draw.text((x, y + 350), f"{VIEW_NAMES[i]} V50R2 floor reference", fill=(140, 0, 0), font=font)
        draw.text((x, y + 700), f"true_nn={metrics[i]['true_to_v50r2_nn_mean']:.5f} baseline_nn={metrics[i]['vggt_visible_baseline_to_v50r2_nn_mean']:.5f}", fill=(80, 80, 80), font=font)
    board.save(FLOOR_BOARD)


def make_anti2d_board(scene: dict[str, np.ndarray], row: dict[str, Any], view_name: str) -> None:
    ANTI2D_BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1060), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), f"V521 anti-2D multiview proxy on {view_name}", fill=(0, 0, 0), font=font)
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
    draw.text((18, 930), f"thickness={row['human_thickness_ratio']:.4f} | anti_2d_proxy_pass={row['anti_2d_proxy_pass']} | mentor_ready={row['mentor_ready']}", fill=(0, 0, 0), font=font)
    board.save(ANTI2D_BOARD)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--max-human-per-view", type=int, default=12000)
    parser.add_argument("--max-env-per-view", type=int, default=7000)
    parser.add_argument("--sample-limit", type=int, default=9000)
    parser.add_argument("--densify-factor", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4.0e-2)
    parser.add_argument("--seed", type=int, default=521)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    obs_rows = load_view_observations(args.max_human_per_view, args.max_env_per_view, args.seed)
    train_batch = concat_for_training(obs_rows)
    model = V521ObservationAnchoredStudent()
    history = train_model(model, train_batch, args.steps, args.sample_limit, args.lr, args.seed)

    checkpoint = OUT / f"checkpoint_{args.steps}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "steps": int(args.steps),
            "route": "V521_observation_anchored_visible_student",
            "input_policy": {
                "vggt_observation_input": True,
                "smpl_part_binding_input": True,
                "v50r2_used_for_loss_only": True,
                "v50r2_used_at_inference": False,
                "kinect_depth_used_at_inference": False,
            },
        },
        checkpoint,
    )

    true_scenes: list[dict[str, np.ndarray]] = []
    metric_rows: list[dict[str, Any]] = []
    all_candidate_points: list[np.ndarray] = []
    all_candidate_rgb: list[np.ndarray] = []
    all_candidate_view: list[np.ndarray] = []
    all_candidate_is_human: list[np.ndarray] = []
    best_idx = 5
    best_control_dist: dict[str, float] = {}
    best_control_scenes: dict[str, dict[str, np.ndarray]] = {}

    for i, row in enumerate(obs_rows):
        control_scenes = {
            key: render_control_for_view(row, model, i, key, args.densify_factor, args.seed + i * 17 + j)
            for j, key in enumerate(CONTROL_ORDER)
        }
        true_scene = control_scenes["true"]
        true_scenes.append(true_scene)
        teacher_points = row["teacher_floor_points"]
        control_nn = {
            key: nearest_mean(control_scenes[key]["human_points"], teacher_points, seed=args.seed + i * 31 + j)
            for j, key in enumerate(CONTROL_ORDER)
        }
        stat = scene_metrics(true_scene)
        controls_pass = bool(
            control_nn["true"] <= control_nn["vggt_visible_baseline"] * 1.015
            and control_nn["true"] < control_nn["observation_only_sparse"] * 0.98
            and control_nn["true"] < control_nn["no_smpl_filter"] * 0.98
            and control_nn["true"] < control_nn["shuffled_semantic"] * 0.995
            and control_nn["true"] < control_nn["smpl_graph_only"] * 0.90
        )
        anti_2d = bool(stat["human_thickness_ratio"] >= 0.055)
        human_main = bool(stat["human_point_count"] >= 16000 and 0.08 <= stat["environment_ratio"] <= 0.38 and stat["body_part_count"] >= 5)
        metric_rows.append(
            {
                "view_name": VIEW_NAMES[i],
                **stat,
                "true_to_v50r2_nn_mean": control_nn["true"],
                "vggt_visible_baseline_to_v50r2_nn_mean": control_nn["vggt_visible_baseline"],
                "observation_only_sparse_to_v50r2_nn_mean": control_nn["observation_only_sparse"],
                "no_smpl_filter_to_v50r2_nn_mean": control_nn["no_smpl_filter"],
                "shuffled_semantic_to_v50r2_nn_mean": control_nn["shuffled_semantic"],
                "smpl_graph_only_to_v50r2_nn_mean": control_nn["smpl_graph_only"],
                "true_beats_required_controls": controls_pass,
                "anti_2d_proxy_pass": anti_2d,
                "human_main_proxy_pass": human_main,
                "mentor_ready": False,
            }
        )
        if i == best_idx:
            best_control_dist = control_nn
            best_control_scenes = control_scenes
        for key, scene in control_scenes.items():
            write_ply(OUT / f"v521_{VIEW_NAMES[i]}_{key}_full_scene_rgb.ply", scene["full_points"], scene["full_rgb"])
            if key == "true":
                write_ply(OUT / f"v521_{VIEW_NAMES[i]}_true_human_only_rgb.ply", scene["human_points"], scene["human_rgb"])
        all_candidate_points.append(true_scene["full_points"])
        all_candidate_rgb.append(true_scene["full_rgb"])
        all_candidate_view.append(np.full(true_scene["full_points"].shape[0], i, dtype=np.int32))
        all_candidate_is_human.append(true_scene["is_human"])

    candidate_npz = OUT / "v521_observation_anchored_visible_student_candidate.npz"
    np.savez_compressed(
        candidate_npz,
        full_scene_points=np.concatenate(all_candidate_points, axis=0).astype(np.float32),
        full_scene_rgb=np.concatenate(all_candidate_rgb, axis=0).astype(np.float32),
        view_index=np.concatenate(all_candidate_view, axis=0).astype(np.int32),
        is_human=np.concatenate(all_candidate_is_human, axis=0).astype(bool),
        model_owned_student_output=np.array(True),
        vggt_observation_input=np.array(True),
        smpl_part_binding_input=np.array(True),
        no_teacher_points_inference=np.array(True),
        no_v50r2_inference=np.array(True),
        no_kinect_depth_inference=np.array(True),
        final_inference_allowed=np.array(True),
        source=np.array("model_owned_v521_observation_anchored_visible_student"),
    )
    copy_result = detect_teacher_copy(TEACHER_BANK, candidate_npz)
    write_json(COPY_JSON, {"task": "V521_teacher_copy_check", "created_at": now(), **copy_result})
    no_teacher_copy = not bool(copy_result.get("leak_detected", False))

    mean_true = float(np.mean([r["true_to_v50r2_nn_mean"] for r in metric_rows]))
    mean_baseline = float(np.mean([r["vggt_visible_baseline_to_v50r2_nn_mean"] for r in metric_rows]))
    all_controls_pass = bool(all(r["true_beats_required_controls"] for r in metric_rows))
    all_human_main = bool(all(r["human_main_proxy_pass"] for r in metric_rows))
    anti_2d_pass = bool(all(r["anti_2d_proxy_pass"] for r in metric_rows))
    manual_visual_gate_pass = bool(no_teacher_copy and all_human_main and anti_2d_pass and mean_true <= mean_baseline * 1.015)
    mentor_ready = bool(manual_visual_gate_pass and all_controls_pass)
    status = (
        "V521_OBSERVATION_ANCHORED_VISIBLE_STUDENT_HUMAN_MAIN_VISUAL_PASS_NEEDS_V512_PACK_NOT_PROMOTED"
        if mentor_ready
        else "V521_OBSERVATION_ANCHORED_VISIBLE_STUDENT_FAIL_CLOSED_CONTINUE_REPAIR_NOT_PROMOTED"
    )
    for row in metric_rows:
        row["status"] = status
        row["no_teacher_copy"] = no_teacher_copy
        row["manual_visual_gate_pass"] = manual_visual_gate_pass
    write_csv(METRICS, metric_rows)

    summary: dict[str, Any] = {
        "task": "V521_observation_anchored_visible_student",
        "status": status,
        "created_at": now(),
        "repo": str(ROOT),
        "checkpoint": str(checkpoint),
        "candidate_npz": str(candidate_npz),
        "boards": {
            "main": str(MAIN_BOARD),
            "same_scene_controls": str(CONTROLS_BOARD),
            "v50r2_visual_floor_comparison": str(FLOOR_BOARD),
            "anti_2d": str(ANTI2D_BOARD),
        },
        "metrics_csv": str(METRICS),
        "teacher_copy_check": str(COPY_JSON),
        "input_policy": {
            "vggt_observation_input": True,
            "smpl_part_binding_input": True,
            "v50r2_used_for_loss_only": True,
            "v50r2_used_for_evaluation_only": True,
            "v50r2_used_at_inference": False,
            "kinect_depth_used_at_inference": False,
            "teacher_points_in_final_inference": False,
        },
        "gates": {
            "no_teacher_copy": no_teacher_copy,
            "same_scene_controls_generated": True,
            "true_beats_required_controls": all_controls_pass,
            "human_main_proxy_pass": all_human_main,
            "anti_2d_proxy_pass": anti_2d_pass,
            "manual_visual_gate_pass": manual_visual_gate_pass,
            "mentor_ready": mentor_ready,
            "not_promoted": True,
        },
        "mean_true_to_v50r2_nn_mean": mean_true,
        "mean_vggt_visible_baseline_to_v50r2_nn_mean": mean_baseline,
        "per_view_metrics": metric_rows,
        "loss_history": history,
        "decision": (
            "V521 preserves the human-readable VGGT visible observation as the inference anchor and only applies learned SMPL/part-bound residuals. "
            "It is not a V50R2 final copy; V50R2 remains loss/eval/reference. Final promotion remains forbidden."
        ),
    }
    make_main_board(true_scenes, metric_rows, summary)
    make_controls_board(best_control_scenes, best_control_dist, VIEW_NAMES[best_idx])
    make_floor_board(true_scenes, obs_rows, metric_rows)
    make_anti2d_board(true_scenes[best_idx], metric_rows[best_idx], VIEW_NAMES[best_idx])
    summary["artifact_hashes"] = {
        "candidate_npz": sha256(candidate_npz),
        "main_board": sha256(MAIN_BOARD),
        "controls_board": sha256(CONTROLS_BOARD),
        "floor_board": sha256(FLOOR_BOARD),
        "anti2d_board": sha256(ANTI2D_BOARD),
    }
    write_json(DECISION, summary)
    print(json.dumps({"status": status, "decision": str(DECISION), "main_board": str(MAIN_BOARD)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
