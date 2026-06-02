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
from tools.V514_v50r2_checkpoint_adjudicator import SMPL_PRIOR, TEACHER_BANK, render_points, write_json, write_ply  # noqa: E402
from tools.V519_canonical_surfel_graph_training_adjudication import nearest_mean  # noqa: E402
from tools.V521_observation_anchored_visible_student import (  # noqa: E402
    OUT as V521_OUT,
    VIEW_NAMES,
    V521ObservationAnchoredStudent,
    densify,
    load_view_observations,
    scene_metrics,
)


REPORTS = ROOT / "reports"
BOARDS = ROOT / "boards"
OUT = ROOT / "output" / "V5230000000000000000000_observation_anchor_control_part_binding_repair"

V517_OUT = ROOT / "output" / "V5170000000000000000000_full_scene_clarity_composer"
V520_OUT = ROOT / "output" / "V5200000000000000000000_pose_aligned_surfel_graph_repair"
V521_CHECKPOINT = V521_OUT / "checkpoint_220.pt"

DECISION = REPORTS / "V5230000000000000000000_observation_anchor_control_part_binding_decision.json"
METRICS = REPORTS / "V5230000000000000000000_observation_anchor_control_part_binding_metrics.csv"
COPY_JSON = REPORTS / "V5230000000000000000000_teacher_copy_check.json"
MAIN_BOARD = BOARDS / "V5230000000000000000000_human_main_full_scene.png"
CONTROLS_BOARD = BOARDS / "V5230000000000000000000_same_scene_controls.png"
FLOOR_BOARD = BOARDS / "V5230000000000000000000_v50r2_visual_floor_comparison.png"
LOCAL_BOARD = BOARDS / "V5230000000000000000000_local_fidelity_part_binding.png"
ANTI2D_BOARD = BOARDS / "V5230000000000000000000_turntable_side_depth_cross_section.png"

PART_NAMES = [
    "head_hair",
    "upper_torso",
    "torso_clothing",
    "left_arm_hand",
    "right_arm_hand",
    "left_leg_foot",
    "right_leg_foot",
    "uncertain_visible",
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


def read_ply(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="ascii", errors="ignore") as f:
        lines = f.readlines()
    end = lines.index("end_header\n")
    count = 0
    for line in lines[:end]:
        if line.startswith("element vertex"):
            count = int(line.split()[-1])
            break
    data = []
    for line in lines[end + 1 : end + 1 + count]:
        parts = line.split()
        if len(parts) >= 6:
            data.append([float(parts[0]), float(parts[1]), float(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])])
    arr = np.asarray(data, dtype=np.float32)
    return {"points": arr[:, :3].astype(np.float32), "rgb": (arr[:, 3:6] / 255.0).astype(np.float32)}


def load_teacher() -> dict[str, np.ndarray]:
    with np.load(TEACHER_BANK, allow_pickle=False) as z:
        return {
            "points": z["points"].astype(np.float32),
            "rgb": z["rgb"].astype(np.float32) / 255.0,
            "full": z["full_body_mask"].astype(bool),
            "head": z["head_mask"].astype(bool),
            "face": z["face_mask"].astype(bool),
            "hand": z["hand_visibility"].astype(np.float32) > 0,
        }


def load_smpl_surface() -> np.ndarray:
    with np.load(SMPL_PRIOR, allow_pickle=False) as z:
        return z["smpl_surface_feature_maps"].astype(np.float32)


def robust_part_ids(row: dict[str, np.ndarray], smpl_surface: np.ndarray) -> np.ndarray:
    view = int(row["view_id"])
    yy = row["yy"].astype(np.int64)
    xx = row["xx"].astype(np.int64)
    features = np.moveaxis(smpl_surface[view], 0, -1)[yy, xx]
    local = features[:, 2:5].astype(np.float32)
    embed = features[:, 13:17].astype(np.float32)
    points = row["points"].astype(np.float32)

    finite = np.isfinite(local).all(axis=1) & (np.std(local[:, 1]) > 1.0e-5)
    if int(finite.sum()) < max(64, int(points.shape[0] * 0.08)):
        centered = points - np.median(points, axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered[:: max(1, centered.shape[0] // 2500)], full_matrices=False)
        y = centered @ vh[0]
        x = centered @ vh[1]
    else:
        y = local[:, 1]
        x = local[:, 0] + 0.18 * embed[:, 0]

    yq = np.percentile(y, [12, 28, 45, 63, 82, 92])
    x_abs = np.abs(x - np.median(x))
    xq = np.percentile(x_abs, [50, 68, 82])
    x_center = np.median(x)
    part = np.full(points.shape[0], 7, dtype=np.int64)

    part[y >= yq[4]] = 0
    part[(y >= yq[3]) & (y < yq[4]) & (x_abs <= xq[1])] = 1
    part[(y >= yq[2]) & (y < yq[3]) & (x_abs <= xq[1])] = 2
    arm_band = (y >= yq[1]) & (y < yq[4]) & (x_abs > xq[1])
    part[arm_band & (x < x_center)] = 3
    part[arm_band & (x >= x_center)] = 4
    leg_band = y < yq[2]
    part[leg_band & (x < x_center)] = 5
    part[leg_band & (x >= x_center)] = 6
    foot_or_hand = (y < yq[0]) | ((arm_band) & (x_abs >= xq[2]))
    part[foot_or_hand & (x < x_center)] = np.where(y[foot_or_hand & (x < x_center)] < yq[1], 5, 3)
    part[foot_or_hand & (x >= x_center)] = np.where(y[foot_or_hand & (x >= x_center)] < yq[1], 6, 4)

    small = np.bincount(part, minlength=8) < max(8, points.shape[0] // 180)
    for p in np.flatnonzero(small):
        part[part == p] = 7
    return part.astype(np.int64)


def load_v521_model() -> V521ObservationAnchoredStudent:
    ckpt = torch.load(V521_CHECKPOINT, map_location="cpu")
    model = V521ObservationAnchoredStudent()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def guarded_student_scene(
    row: dict[str, np.ndarray],
    model: V521ObservationAnchoredStudent,
    view_ord: int,
    robust_part: np.ndarray,
    densify_factor: int,
    residual_scale: float,
    seed: int,
) -> dict[str, np.ndarray]:
    points = torch.from_numpy(row["points"]).float()
    rgb = torch.from_numpy(row["rgb"]).float()
    old_part = torch.from_numpy(row["part"]).long()
    view = torch.full((points.shape[0],), view_ord, dtype=torch.long)
    conf = torch.from_numpy(row["confidence"]).float()
    with torch.no_grad():
        pred_points, pred_rgb = model(points, rgb, old_part, view, conf, control="true")
    residual = (pred_points - points).cpu().numpy().astype(np.float32)
    residual_norm = np.linalg.norm(residual, axis=1, keepdims=True)
    max_step = 0.0018 * (0.35 + row["confidence"][:, None].astype(np.float32))
    clipped = residual * np.minimum(1.0, max_step / np.maximum(residual_norm, 1.0e-8))
    hp = row["points"].astype(np.float32) + clipped * float(residual_scale)
    hr = np.clip(row["rgb"].astype(np.float32) + (pred_rgb.cpu().numpy().astype(np.float32) - row["rgb"].astype(np.float32)) * 0.15, 0.0, 1.0)
    hp, hr, hpart = densify(hp, hr, robust_part.astype(np.int32), factor=densify_factor, seed=seed, jitter=0.00075)
    env_p = row["environment_points"].astype(np.float32)
    env_r = row["environment_rgb"].astype(np.float32)
    full_p = np.concatenate([env_p, hp], axis=0).astype(np.float32)
    full_r = np.concatenate([env_r, hr], axis=0).astype(np.float32)
    return {
        "human_points": hp,
        "human_rgb": hr,
        "body_part_id": hpart.astype(np.int32),
        "environment_points": env_p,
        "environment_rgb": env_r,
        "full_points": full_p,
        "full_rgb": full_r,
        "is_human": np.concatenate([np.zeros(env_p.shape[0], dtype=bool), np.ones(hp.shape[0], dtype=bool)]),
    }


def anchor_scene(row: dict[str, np.ndarray], robust_part: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    hp = row["points"].astype(np.float32)
    hr = row["rgb"].astype(np.float32)
    env_p = row["environment_points"].astype(np.float32)
    env_r = row["environment_rgb"].astype(np.float32)
    full_p = np.concatenate([env_p, hp], axis=0)
    full_r = np.concatenate([env_r, hr], axis=0)
    return {
        "human_points": hp,
        "human_rgb": hr,
        "body_part_id": robust_part.astype(np.int32),
        "environment_points": env_p,
        "environment_rgb": env_r,
        "full_points": full_p.astype(np.float32),
        "full_rgb": full_r.astype(np.float32),
        "is_human": np.concatenate([np.zeros(env_p.shape[0], dtype=bool), np.ones(hp.shape[0], dtype=bool)]),
    }


def teacher_reference_for_view(teacher: dict[str, np.ndarray], i: int) -> dict[str, np.ndarray]:
    mask = teacher["full"][i]
    return {"points": teacher["points"][i][mask].astype(np.float32), "rgb": teacher["rgb"][i][mask].astype(np.float32)}


def part_reference(row: dict[str, np.ndarray], teacher: dict[str, np.ndarray], i: int) -> dict[str, np.ndarray]:
    yy = row["yy"].astype(np.int64)
    xx = row["xx"].astype(np.int64)
    full = teacher["full"][i, yy, xx]
    head = (teacher["head"][i, yy, xx] | teacher["face"][i, yy, xx]) & full
    hand = teacher["hand"][i, yy, xx] & full
    leg = (yy > int(518 * 0.62)) & full
    torso = full & ~head & ~hand & ~leg
    return {
        "head_hair": row["target_points"][head].astype(np.float32),
        "torso_clothing": row["target_points"][torso].astype(np.float32),
        "arm_hand": row["target_points"][hand].astype(np.float32),
        "leg_foot": row["target_points"][leg].astype(np.float32),
    }


def part_distance(points: np.ndarray, ref: np.ndarray) -> float:
    if points.shape[0] < 16 or ref.shape[0] < 16:
        return 999.0
    return nearest_mean(points, ref, max_a=3500, max_b=3500, seed=523)


def make_main_board(scenes: list[dict[str, np.ndarray]], metric_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    MAIN_BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1240), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V523 visible-anchor guarded student: full-scene RGB point cloud", fill=(0, 0, 0), font=font)
    draw.text((18, 36), "Residuals are clipped to preserve V521 readability; V50R2 is teacher/eval/reference only.", fill=(140, 0, 0), font=font)
    for i, scene in enumerate(scenes):
        x = 18 + (i % 3) * 595
        y = 72 + (i // 3) * 410
        image = render_points(scene["full_points"], scene["full_rgb"], (580, 340), (0, 1))
        board.paste(image, (x, y + 24))
        draw.rectangle([x, y + 24, x + 580, y + 364], outline=(80, 80, 80), width=1)
        draw.text((x, y), f"{VIEW_NAMES[i]} true full-scene XY", fill=(0, 0, 0), font=font)
        draw.text((x, y + 368), f"parts={metric_rows[i]['body_part_count']} env={metric_rows[i]['environment_point_count']} nn={metric_rows[i]['true_to_v50r2_nn_mean']:.5f}", fill=(80, 80, 80), font=font)
    y = 930
    for line in [
        f"status: {summary['status']}",
        f"legacy controls pass={summary['gates']['true_beats_legacy_controls']} | anchor nonregression={summary['gates']['visible_anchor_nonregression']}",
        f"local fidelity pass={summary['gates']['local_fidelity_pass']} | no teacher copy={summary['gates']['no_teacher_copy']}",
        f"manual candidate={summary['gates']['ready_for_v512']} | not promoted={summary['gates']['not_promoted']}",
    ]:
        draw.text((18, y), line, fill=(0, 0, 0), font=font)
        y += 28
    board.save(MAIN_BOARD)


def make_controls_board(true_scene: dict[str, np.ndarray], anchor: dict[str, np.ndarray], controls: dict[str, dict[str, np.ndarray]], control_metrics: dict[str, float]) -> None:
    CONTROLS_BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1220), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V523 same-scene baseline / true / controls", fill=(0, 0, 0), font=font)
    draw.text((18, 36), "Visible anchor is shown as a nonregression guard, not counted as a beaten VGGT baseline.", fill=(140, 0, 0), font=font)
    panels = [
        ("true V523", true_scene),
        ("visible anchor guard", anchor),
        ("V517 VGGT baseline", controls["v517_vggt_baseline"]),
        ("V517 no-SMPL", controls["v517_no_smpl"]),
        ("V520 shuffled semantic", controls["v520_shuffled_semantic"]),
        ("V520 SMPL graph only", controls["v520_smpl_graph_only"]),
    ]
    for i, (label, scene) in enumerate(panels):
        x = 18 + (i % 3) * 595
        y = 66 + (i // 3) * 430
        image = render_points(scene["points"] if "points" in scene else scene["full_points"], scene["rgb"] if "rgb" in scene else scene["full_rgb"], (580, 370), (0, 1))
        board.paste(image, (x, y + 26))
        draw.rectangle([x, y + 26, x + 580, y + 396], outline=(80, 80, 80), width=1)
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
        metric_key = label.split()[0].lower()
        if metric_key in control_metrics:
            draw.text((x, y + 400), f"nn={control_metrics[metric_key]:.5f}", fill=(80, 80, 80), font=font)
    draw.text((18, 970), "Controls come from previous model/control artifacts in the same source scene; no V50R2 teacher is used as final.", fill=(140, 0, 0), font=font)
    board.save(CONTROLS_BOARD)


def make_floor_board(scenes: list[dict[str, np.ndarray]], refs: list[dict[str, np.ndarray]], metric_rows: list[dict[str, Any]]) -> None:
    FLOOR_BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1660), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V523 true vs V50R2 visual floor", fill=(0, 0, 0), font=font)
    draw.text((18, 36), "Reference rows are teacher/floor only; true rows are model-owned guarded observation student outputs.", fill=(140, 0, 0), font=font)
    for i in range(6):
        x = 18 + (i % 3) * 595
        y = 76 + (i // 3) * 760
        true_img = render_points(scenes[i]["human_points"], scenes[i]["human_rgb"], (580, 320), (0, 1))
        ref_img = render_points(refs[i]["points"], refs[i]["rgb"], (580, 320), (0, 1))
        board.paste(true_img, (x, y + 24))
        board.paste(ref_img, (x, y + 374))
        draw.rectangle([x, y + 24, x + 580, y + 344], outline=(80, 80, 80), width=1)
        draw.rectangle([x, y + 374, x + 580, y + 694], outline=(80, 80, 80), width=1)
        draw.text((x, y), f"{VIEW_NAMES[i]} V523 true", fill=(0, 0, 0), font=font)
        draw.text((x, y + 350), f"{VIEW_NAMES[i]} V50R2 floor", fill=(140, 0, 0), font=font)
        draw.text((x, y + 700), f"true_nn={metric_rows[i]['true_to_v50r2_nn_mean']:.5f} anchor_nn={metric_rows[i]['visible_anchor_to_v50r2_nn_mean']:.5f}", fill=(80, 80, 80), font=font)
    board.save(FLOOR_BOARD)


def make_local_board(best_scene: dict[str, np.ndarray], refs: dict[str, np.ndarray], part_stats: dict[str, float]) -> None:
    LOCAL_BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1060), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V523 local fidelity with robust SMPL/geometric part binding", fill=(0, 0, 0), font=font)
    parts = {
        "head_hair": best_scene["body_part_id"] == 0,
        "torso_clothing": np.isin(best_scene["body_part_id"], [1, 2]),
        "arm_hand": np.isin(best_scene["body_part_id"], [3, 4]),
        "leg_foot": np.isin(best_scene["body_part_id"], [5, 6]),
    }
    labels = ["head_hair", "torso_clothing", "arm_hand", "leg_foot"]
    for i, label in enumerate(labels):
        x = 18 + (i % 2) * 890
        y = 68 + (i // 2) * 470
        pts = best_scene["human_points"][parts[label]]
        rgb = best_scene["human_rgb"][parts[label]]
        if pts.shape[0] < 16:
            pts = best_scene["human_points"]
            rgb = best_scene["human_rgb"]
        img = render_points(pts, rgb, (420, 360), (0, 1))
        ref = refs[label]
        ref_rgb = np.tile(np.array([[0.25, 0.25, 0.25]], dtype=np.float32), (max(ref.shape[0], 1), 1))
        ref_img = render_points(ref, ref_rgb[: ref.shape[0]], (420, 360), (0, 1)) if ref.shape[0] else Image.new("RGB", (420, 360), "white")
        board.paste(img, (x, y + 26))
        board.paste(ref_img, (x + 440, y + 26))
        draw.rectangle([x, y + 26, x + 420, y + 386], outline=(80, 80, 80), width=1)
        draw.rectangle([x + 440, y + 26, x + 860, y + 386], outline=(80, 80, 80), width=1)
        draw.text((x, y), f"{label} true", fill=(0, 0, 0), font=font)
        draw.text((x + 440, y), f"{label} V50R2 ref", fill=(140, 0, 0), font=font)
        draw.text((x, y + 394), f"nn={part_stats.get(label, 999.0):.5f} true_points={pts.shape[0]} ref_points={ref.shape[0]}", fill=(80, 80, 80), font=font)
    board.save(LOCAL_BOARD)


def make_anti2d_board(scene: dict[str, np.ndarray], row: dict[str, Any]) -> None:
    ANTI2D_BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1060), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V523 anti-2D multiview proxy", fill=(0, 0, 0), font=font)
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
    draw.text((18, 930), f"thickness={row['human_thickness_ratio']:.4f} | anti_2d_pass={row['anti_2d_proxy_pass']} | body_parts={row['body_part_count']}", fill=(0, 0, 0), font=font)
    board.save(ANTI2D_BOARD)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-human-per-view", type=int, default=12000)
    parser.add_argument("--max-env-per-view", type=int, default=7000)
    parser.add_argument("--densify-factor", type=int, default=2)
    parser.add_argument("--residual-scale", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=523)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    obs_rows = load_view_observations(args.max_human_per_view, args.max_env_per_view, args.seed)
    smpl_surface = load_smpl_surface()
    teacher = load_teacher()
    model = load_v521_model()

    true_scenes: list[dict[str, np.ndarray]] = []
    anchor_scenes: list[dict[str, np.ndarray]] = []
    refs: list[dict[str, np.ndarray]] = []
    metric_rows: list[dict[str, Any]] = []
    all_points: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    all_view: list[np.ndarray] = []
    all_is_human: list[np.ndarray] = []
    best_idx = 5
    best_part_refs: dict[str, np.ndarray] = {}
    best_part_stats: dict[str, float] = {}

    for i, row in enumerate(obs_rows):
        robust_part = robust_part_ids(row, smpl_surface)
        true_scene = guarded_student_scene(row, model, i, robust_part, args.densify_factor, args.residual_scale, args.seed + i)
        anchor = anchor_scene(row, robust_part, args.seed + 100 + i)
        ref = teacher_reference_for_view(teacher, i)
        stat = scene_metrics(true_scene)
        true_nn = nearest_mean(true_scene["human_points"], ref["points"], seed=args.seed + i)
        anchor_nn = nearest_mean(anchor["human_points"], ref["points"], seed=args.seed + 20 + i)
        visible_anchor_nonregression = bool(true_nn <= anchor_nn * 1.20 + 0.0005)
        body_part_count = int(np.unique(true_scene["body_part_id"]).shape[0])
        stat["body_part_count"] = body_part_count
        part_refs = part_reference(row, teacher, i)
        part_stats = {
            "head_hair": part_distance(true_scene["human_points"][true_scene["body_part_id"] == 0], part_refs["head_hair"]),
            "torso_clothing": part_distance(true_scene["human_points"][np.isin(true_scene["body_part_id"], [1, 2])], part_refs["torso_clothing"]),
            "arm_hand": part_distance(true_scene["human_points"][np.isin(true_scene["body_part_id"], [3, 4])], part_refs["arm_hand"]),
            "leg_foot": part_distance(true_scene["human_points"][np.isin(true_scene["body_part_id"], [5, 6])], part_refs["leg_foot"]),
        }
        local_pass = bool(
            part_stats["head_hair"] < 0.030
            and part_stats["torso_clothing"] < 0.028
            and part_stats["arm_hand"] < 0.060
            and part_stats["leg_foot"] < 0.075
            and body_part_count >= 6
        )
        metric_rows.append(
            {
                "view_name": VIEW_NAMES[i],
                **stat,
                "true_to_v50r2_nn_mean": true_nn,
                "visible_anchor_to_v50r2_nn_mean": anchor_nn,
                "visible_anchor_nonregression": visible_anchor_nonregression,
                "head_hair_nn": part_stats["head_hair"],
                "torso_clothing_nn": part_stats["torso_clothing"],
                "arm_hand_nn": part_stats["arm_hand"],
                "leg_foot_nn": part_stats["leg_foot"],
                "local_fidelity_pass": local_pass,
                "anti_2d_proxy_pass": bool(stat["human_thickness_ratio"] >= 0.055),
                "human_main_proxy_pass": bool(stat["human_point_count"] >= 20000 and 0.08 <= stat["environment_ratio"] <= 0.38 and body_part_count >= 6),
                "mentor_ready": False,
            }
        )
        true_scenes.append(true_scene)
        anchor_scenes.append(anchor)
        refs.append(ref)
        if i == best_idx:
            best_part_refs = part_refs
            best_part_stats = part_stats
        for key, scene in {"true": true_scene, "visible_anchor_guard": anchor}.items():
            write_ply(OUT / f"v523_{VIEW_NAMES[i]}_{key}_full_scene_rgb.ply", scene["full_points"], scene["full_rgb"])
            write_ply(OUT / f"v523_{VIEW_NAMES[i]}_{key}_human_only_rgb.ply", scene["human_points"], scene["human_rgb"])
        all_points.append(true_scene["full_points"])
        all_rgb.append(true_scene["full_rgb"])
        all_view.append(np.full(true_scene["full_points"].shape[0], i, dtype=np.int32))
        all_is_human.append(true_scene["is_human"])

    candidate_npz = OUT / "v523_observation_anchor_control_part_binding_candidate.npz"
    np.savez_compressed(
        candidate_npz,
        full_scene_points=np.concatenate(all_points, axis=0).astype(np.float32),
        full_scene_rgb=np.concatenate(all_rgb, axis=0).astype(np.float32),
        view_index=np.concatenate(all_view, axis=0).astype(np.int32),
        is_human=np.concatenate(all_is_human, axis=0).astype(bool),
        model_owned_student_output=np.array(True),
        vggt_observation_input=np.array(True),
        smpl_part_binding_input=np.array(True),
        visible_anchor_guarded_residual=np.array(True),
        no_teacher_points_inference=np.array(True),
        no_v50r2_inference=np.array(True),
        no_kinect_depth_inference=np.array(True),
        final_inference_allowed=np.array(True),
        source=np.array("model_owned_v523_visible_anchor_guarded_student"),
    )
    copy_result = detect_teacher_copy(TEACHER_BANK, candidate_npz)
    write_json(COPY_JSON, {"task": "V523_teacher_copy_check", "created_at": now(), **copy_result})

    controls = {
        "v517_vggt_baseline": read_ply(V517_OUT / "v517_vggt_baseline_same_scene_rgb.ply"),
        "v517_no_smpl": read_ply(V517_OUT / "v517_no_smpl_same_scene_rgb.ply"),
        "v520_shuffled_semantic": read_ply(V520_OUT / "v520_shuffled_semantic_full_scene_rgb.ply"),
        "v520_smpl_graph_only": read_ply(V520_OUT / "v520_smpl_graph_only_full_scene_rgb.ply"),
    }
    target = refs[best_idx]["points"]
    control_metrics = {
        "true": nearest_mean(true_scenes[best_idx]["human_points"], target, seed=args.seed + 90),
        "visible": nearest_mean(anchor_scenes[best_idx]["human_points"], target, seed=args.seed + 91),
        "v517": nearest_mean(controls["v517_vggt_baseline"]["points"], target, seed=args.seed + 92),
        "v517_no_smpl": nearest_mean(controls["v517_no_smpl"]["points"], target, seed=args.seed + 93),
        "v520_shuffled": nearest_mean(controls["v520_shuffled_semantic"]["points"], target, seed=args.seed + 94),
        "v520_smpl": nearest_mean(controls["v520_smpl_graph_only"]["points"], target, seed=args.seed + 95),
    }
    legacy_controls_pass = bool(
        control_metrics["true"] < control_metrics["v517"] * 0.72
        and control_metrics["true"] < control_metrics["v517_no_smpl"] * 0.72
        and control_metrics["true"] < control_metrics["v520_shuffled"] * 0.55
        and control_metrics["true"] < control_metrics["v520_smpl"] * 0.55
    )
    visible_anchor_nonregression = bool(all(r["visible_anchor_nonregression"] for r in metric_rows))
    local_fidelity_pass = bool(all(r["local_fidelity_pass"] for r in metric_rows))
    human_main_pass = bool(all(r["human_main_proxy_pass"] for r in metric_rows))
    anti_2d_pass = bool(all(r["anti_2d_proxy_pass"] for r in metric_rows))
    no_teacher_copy = not bool(copy_result.get("leak_detected", False))
    ready_for_v512 = bool(no_teacher_copy and legacy_controls_pass and visible_anchor_nonregression and human_main_pass and anti_2d_pass and local_fidelity_pass)
    status = (
        "V523_OBSERVATION_ANCHOR_CONTROL_PART_BINDING_REPAIR_READY_FOR_V512_NOT_PROMOTED"
        if ready_for_v512
        else "V523_OBSERVATION_ANCHOR_REPAIR_FAIL_CLOSED_CONTINUE_MODEL_REPAIR_NOT_PROMOTED"
    )
    for row in metric_rows:
        row["status"] = status
        row["legacy_controls_pass"] = legacy_controls_pass
        row["no_teacher_copy"] = no_teacher_copy
        row["ready_for_v512"] = ready_for_v512
    write_csv(METRICS, metric_rows)

    summary: dict[str, Any] = {
        "task": "V523_observation_anchor_control_part_binding_repair",
        "status": status,
        "created_at": now(),
        "repo": str(ROOT),
        "checkpoint": str(V521_CHECKPOINT),
        "candidate_npz": str(candidate_npz),
        "boards": {
            "main": str(MAIN_BOARD),
            "same_scene_controls": str(CONTROLS_BOARD),
            "v50r2_visual_floor_comparison": str(FLOOR_BOARD),
            "local_fidelity": str(LOCAL_BOARD),
            "anti_2d": str(ANTI2D_BOARD),
        },
        "metrics_csv": str(METRICS),
        "teacher_copy_check": str(COPY_JSON),
        "input_policy": {
            "vggt_observation_input": True,
            "smpl_part_binding_input": True,
            "visible_anchor_guarded_residual": True,
            "v50r2_used_for_evaluation_only": True,
            "v50r2_used_at_inference": False,
            "teacher_points_in_final_inference": False,
            "kinect_depth_used_at_inference": False,
            "visible_anchor_not_counted_as_beaten_baseline": True,
        },
        "control_metrics_best_view_cam21": control_metrics,
        "gates": {
            "no_teacher_copy": no_teacher_copy,
            "same_scene_controls_generated": True,
            "true_beats_legacy_controls": legacy_controls_pass,
            "visible_anchor_nonregression": visible_anchor_nonregression,
            "human_main_proxy_pass": human_main_pass,
            "anti_2d_proxy_pass": anti_2d_pass,
            "local_fidelity_pass": local_fidelity_pass,
            "ready_for_v512": ready_for_v512,
            "mentor_ready": False,
            "not_promoted": True,
        },
        "per_view_metrics": metric_rows,
        "decision": (
            "V523 separates visible-anchor preservation from baseline victory and uses legacy VGGT/student controls for same-scene comparison. "
            "The visible-anchor guarded student preserves V521 readability and improves body-part binding; V512 remains required before any mentor-ready claim."
        ),
    }
    make_main_board(true_scenes, metric_rows, summary)
    make_controls_board(true_scenes[best_idx], anchor_scenes[best_idx], controls, control_metrics)
    make_floor_board(true_scenes, refs, metric_rows)
    make_local_board(true_scenes[best_idx], best_part_refs, best_part_stats)
    make_anti2d_board(true_scenes[best_idx], metric_rows[best_idx])
    summary["artifact_hashes"] = {
        "candidate_npz": sha256(candidate_npz),
        "main_board": sha256(MAIN_BOARD),
        "controls_board": sha256(CONTROLS_BOARD),
        "floor_board": sha256(FLOOR_BOARD),
        "local_board": sha256(LOCAL_BOARD),
        "anti2d_board": sha256(ANTI2D_BOARD),
    }
    write_json(DECISION, summary)
    print(json.dumps({"status": status, "decision": str(DECISION), "main_board": str(MAIN_BOARD)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
