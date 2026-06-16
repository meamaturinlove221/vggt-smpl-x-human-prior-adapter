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
    TEACHER_BANK,
    flatten_output,
    load_model,
    load_observation_inputs,
    render_points,
    write_json,
    write_ply,
)


REPORTS = ROOT / "reports"
BOARDS = ROOT / "boards"
OUT = ROOT / "output" / "V5160000000000000000000_paired_visible_surface_adjudication"
DEFAULT_CHECKPOINT = ROOT / "output" / "V5150000000000000000000_observation_bound_repair_training" / "checkpoint_240.pt"

DECISION = REPORTS / "V5160000000000000000000_paired_visible_surface_decision.json"
METRICS = REPORTS / "V5160000000000000000000_paired_visible_surface_metrics.csv"
COPY_JSON = REPORTS / "V5160000000000000000000_teacher_copy_check.json"
BOARD = BOARDS / "V5160000000000000000000_paired_visible_surface_board.png"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_teacher_targets(selected_indices: np.ndarray, view_count: int, sample_count: int) -> dict[str, torch.Tensor]:
    with np.load(TEACHER_BANK, allow_pickle=False) as z:
        teacher_points = z["points"].astype(np.float32)
        teacher_rgb = z["rgb"].astype(np.float32) / 255.0
        full = z["full_body_mask"].astype(bool)
        head = z["head_mask"].astype(bool)
        face = z["face_mask"].astype(bool)
        hand = z["hand_visibility"].astype(np.float32) > 0
    selected = selected_indices.reshape(view_count, sample_count, 3)
    pts = np.zeros((view_count, sample_count, 3), dtype=np.float32)
    rgb = np.zeros((view_count, sample_count, 3), dtype=np.float32)
    visible = np.zeros((view_count, sample_count), dtype=np.float32)
    head_mask = np.zeros((view_count, sample_count), dtype=np.float32)
    hand_mask = np.zeros((view_count, sample_count), dtype=np.float32)
    torso_mask = np.zeros((view_count, sample_count), dtype=np.float32)
    leg_mask = np.zeros((view_count, sample_count), dtype=np.float32)
    for view in range(min(6, view_count)):
        yy = selected[view, :, 1]
        xx = selected[view, :, 2]
        valid = full[view, yy, xx]
        pts[view] = teacher_points[view, yy, xx]
        rgb[view] = teacher_rgb[view, yy, xx]
        visible[view] = valid.astype(np.float32)
        head_valid = (head[view, yy, xx] | face[view, yy, xx]) & valid
        hand_valid = hand[view, yy, xx] & valid
        leg_valid = (yy > int(518 * 0.62)) & valid
        torso_valid = valid & ~head_valid & ~hand_valid & ~leg_valid
        head_mask[view] = head_valid.astype(np.float32)
        hand_mask[view] = hand_valid.astype(np.float32)
        torso_mask[view] = torso_valid.astype(np.float32)
        leg_mask[view] = leg_valid.astype(np.float32)
    return {
        "points": torch.from_numpy(pts),
        "rgb": torch.from_numpy(rgb),
        "visible": torch.from_numpy(visible),
        "head": torch.from_numpy(head_mask),
        "hand": torch.from_numpy(hand_mask),
        "torso": torch.from_numpy(torso_mask),
        "leg": torch.from_numpy(leg_mask),
    }


def paired_distance(points: np.ndarray, target: torch.Tensor, weight: torch.Tensor) -> float:
    pred = torch.from_numpy(points.reshape(target.shape)).float()
    w = weight.float().unsqueeze(-1)
    err = (pred - target.float()).square().sum(dim=-1, keepdim=True).sqrt()
    denom = w.sum().clamp_min(1.0)
    return float((err * w).sum().item() / denom.item())


def paired_rgb(rgb: np.ndarray, target: torch.Tensor, weight: torch.Tensor) -> float:
    pred = torch.from_numpy(rgb.reshape(target.shape)).float()
    w = weight.float().unsqueeze(-1)
    err = (pred - target.float()).abs().mean(dim=-1, keepdim=True)
    denom = w.sum().clamp_min(1.0)
    return float((err * w).sum().item() / denom.item())


def part_metrics(points: np.ndarray, target: torch.Tensor, parts: dict[str, torch.Tensor]) -> dict[str, float]:
    return {f"{name}_paired_dist": paired_distance(points, target, weight) for name, weight in parts.items()}


def better_than_control(true_value: float, control_value: float, *, abs_margin: float = 5.0e-5, rel_ratio: float = 0.85) -> bool:
    return bool(true_value < control_value - abs_margin or true_value <= control_value * rel_ratio)


def make_board(
    full_true_points: np.ndarray,
    full_true_rgb: np.ndarray,
    full_base_points: np.ndarray,
    full_base_rgb: np.ndarray,
    teacher_points: np.ndarray,
    teacher_rgb: np.ndarray,
    row: dict[str, Any],
) -> None:
    BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1240), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V516 paired visible-surface adjudication: model-owned candidate vs V50R2 visual floor", fill=(0, 0, 0), font=font)
    draw.text((18, 36), "Paired teacher metrics are auxiliary; mentor pass still requires V509/V512 full-scene visual gate.", fill=(140, 0, 0), font=font)
    panels = [
        ("true full-scene XZ", render_points(full_true_points, full_true_rgb, (580, 360), (0, 2))),
        ("baseline full-scene XZ", render_points(full_base_points, full_base_rgb, (580, 360), (0, 2))),
        ("V50R2 paired visible reference XZ", render_points(teacher_points, teacher_rgb, (580, 360), (0, 2))),
        ("true full-scene XY", render_points(full_true_points, full_true_rgb, (580, 360), (0, 1))),
        ("baseline full-scene XY", render_points(full_base_points, full_base_rgb, (580, 360), (0, 1))),
        ("V50R2 paired visible reference XY", render_points(teacher_points, teacher_rgb, (580, 360), (0, 1))),
    ]
    for i, (label, image) in enumerate(panels):
        x = 18 + (i % 3) * 595
        y = 70 + (i // 3) * 420
        board.paste(image, (x, y + 26))
        draw.rectangle([x, y + 26, x + 580, y + 386], outline=(80, 80, 80), width=1)
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
    y = 920
    for line in [
        f"status: {row['status']}",
        f"true_paired_dist: {row['true_paired_dist']:.6f}",
        f"baseline_paired_dist: {row['baseline_paired_dist']:.6f}",
        f"no_smpl_paired_dist: {row['no_smpl_paired_dist']:.6f}",
        f"shuffled_paired_dist: {row['shuffled_paired_dist']:.6f}",
        f"true_improves_baseline: {row['true_improves_baseline']}",
        f"true_improves_no_smpl: {row['true_improves_no_smpl']}",
        f"paired_surface_pass: {row['paired_surface_pass']}",
        f"mentor_ready: False",
    ]:
        draw.text((18, y), line, fill=(0, 0, 0), font=font)
        y += 26
    board.save(BOARD)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=516)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    batch, meta = load_observation_inputs(max_human_per_view=args.max_samples, max_env_per_view=2048, seed=args.seed)
    model, ckpt = load_model(Path(args.checkpoint))
    model.eval()
    with torch.no_grad():
        outputs = {control: flatten_output(model(batch, control=control)) for control in ["true", "vggt_baseline", "no_smpl", "shuffled_semantic"]}
    view_count, sample_count = batch["vggt_world_points"].shape[:2]
    teacher = load_teacher_targets(meta["selected_indices"], view_count, sample_count)
    visible = teacher["visible"]
    parts = {k: teacher[k] for k in ["head", "hand", "torso", "leg"]}

    true_dist = paired_distance(outputs["true"]["points"], teacher["points"], visible)
    base_dist = paired_distance(outputs["vggt_baseline"]["points"], teacher["points"], visible)
    no_smpl_dist = paired_distance(outputs["no_smpl"]["points"], teacher["points"], visible)
    shuffled_dist = paired_distance(outputs["shuffled_semantic"]["points"], teacher["points"], visible)
    true_rgb = paired_rgb(outputs["true"]["rgb"], teacher["rgb"], visible)
    base_rgb = paired_rgb(outputs["vggt_baseline"]["rgb"], teacher["rgb"], visible)
    true_part = part_metrics(outputs["true"]["points"], teacher["points"], parts)
    base_part = part_metrics(outputs["vggt_baseline"]["points"], teacher["points"], parts)

    env_points = meta["environment_points"].astype(np.float32)
    env_rgb = meta["environment_rgb"].astype(np.float32)
    full_true_points = np.concatenate([env_points, outputs["true"]["points"]], axis=0)
    full_true_rgb = np.concatenate([env_rgb, outputs["true"]["rgb"]], axis=0)
    full_base_points = np.concatenate([env_points, outputs["vggt_baseline"]["points"]], axis=0)
    full_base_rgb = np.concatenate([env_rgb, outputs["vggt_baseline"]["rgb"]], axis=0)
    teacher_points = teacher["points"].numpy().reshape(-1, 3)
    teacher_rgb = teacher["rgb"].numpy().reshape(-1, 3)
    teacher_visible = visible.numpy().reshape(-1) > 0
    teacher_points_visible = teacher_points[teacher_visible]
    teacher_rgb_visible = teacher_rgb[teacher_visible]

    candidate_npz = OUT / "v516_model_owned_student_candidate.npz"
    np.savez_compressed(
        candidate_npz,
        predicted_points=outputs["true"]["points"],
        predicted_rgb=outputs["true"]["rgb"],
        full_scene_points=full_true_points,
        full_scene_rgb=full_true_rgb,
        paired_teacher_visible_count=np.array(int(teacher_visible.sum())),
        model_owned_student_output=np.array(True),
        no_teacher_points_inference=np.array(True),
        no_v50r2_inference=np.array(True),
        no_kinect_depth_inference=np.array(True),
        final_inference_allowed=np.array(True),
        source=np.array("model_owned_v516_paired_visible_surface_candidate"),
    )
    copy_result = detect_teacher_copy(TEACHER_BANK, candidate_npz)
    write_json(COPY_JSON, {"task": "V516_teacher_copy_check", "created_at": now(), **copy_result})

    true_improves_baseline = better_than_control(true_dist, base_dist)
    true_improves_no_smpl = better_than_control(true_dist, no_smpl_dist)
    true_improves_shuffled = better_than_control(true_dist, shuffled_dist)
    rgb_not_worse = bool(true_rgb <= base_rgb + 0.015)
    part_pass = bool(
        true_part["head_paired_dist"] <= base_part["head_paired_dist"] + 0.002
        and true_part["torso_paired_dist"] <= base_part["torso_paired_dist"] + 0.002
        and true_part["leg_paired_dist"] <= base_part["leg_paired_dist"] + 0.002
    )
    paired_surface_pass = bool(
        not copy_result["leak_detected"]
        and true_improves_baseline
        and true_improves_no_smpl
        and true_improves_shuffled
        and rgb_not_worse
        and part_pass
    )
    status = (
        "V516_PAIRED_VISIBLE_SURFACE_CONTROL_SEPARATION_PASS_NEEDS_V509_V512_NOT_PROMOTED"
        if paired_surface_pass
        else "V516_PAIRED_VISIBLE_SURFACE_FAIL_CLOSED_CONTROL_SEPARATION_NOT_PROMOTED"
    )
    write_ply(OUT / "v516_true_full_scene_rgb.ply", full_true_points, full_true_rgb)
    write_ply(OUT / "v516_vggt_baseline_full_scene_rgb.ply", full_base_points, full_base_rgb)
    write_ply(OUT / "v516_v50r2_paired_visible_reference.ply", teacher_points_visible, teacher_rgb_visible)

    row = {
        "status": status,
        "checkpoint": str(args.checkpoint),
        "checkpoint_steps": int(ckpt.get("steps", 0)),
        "paired_visible_count": int(teacher_visible.sum()),
        "human_point_count": int(outputs["true"]["points"].shape[0]),
        "environment_point_count": int(env_points.shape[0]),
        "teacher_copy_leak_detected": copy_result["leak_detected"],
        "true_paired_dist": true_dist,
        "baseline_paired_dist": base_dist,
        "no_smpl_paired_dist": no_smpl_dist,
        "shuffled_paired_dist": shuffled_dist,
        "true_vs_baseline_ratio": true_dist / max(base_dist, 1.0e-8),
        "true_vs_no_smpl_ratio": true_dist / max(no_smpl_dist, 1.0e-8),
        "true_vs_shuffled_ratio": true_dist / max(shuffled_dist, 1.0e-8),
        "true_paired_rgb_l1": true_rgb,
        "baseline_paired_rgb_l1": base_rgb,
        "true_head_paired_dist": true_part["head_paired_dist"],
        "baseline_head_paired_dist": base_part["head_paired_dist"],
        "true_torso_paired_dist": true_part["torso_paired_dist"],
        "baseline_torso_paired_dist": base_part["torso_paired_dist"],
        "true_leg_paired_dist": true_part["leg_paired_dist"],
        "baseline_leg_paired_dist": base_part["leg_paired_dist"],
        "true_improves_baseline": true_improves_baseline,
        "true_improves_no_smpl": true_improves_no_smpl,
        "true_improves_shuffled": true_improves_shuffled,
        "rgb_not_worse_than_baseline": rgb_not_worse,
        "part_paired_surface_pass": part_pass,
        "paired_surface_pass": paired_surface_pass,
        "accepted_for_v509": False,
    }
    write_csv(METRICS, [row])
    make_board(full_true_points, full_true_rgb, full_base_points, full_base_rgb, teacher_points_visible, teacher_rgb_visible, row)
    payload = {
        "task": "V516_paired_visible_surface_adjudication",
        "status": status,
        "created_at": now(),
        "repo": str(ROOT),
        "checkpoint": str(args.checkpoint),
        "candidate_npz": str(candidate_npz),
        "true_full_scene_ply": str(OUT / "v516_true_full_scene_rgb.ply"),
        "baseline_full_scene_ply": str(OUT / "v516_vggt_baseline_full_scene_rgb.ply"),
        "paired_v50r2_reference_ply": str(OUT / "v516_v50r2_paired_visible_reference.ply"),
        "board": str(BOARD),
        "metrics_csv": str(METRICS),
        "teacher_copy_check": str(COPY_JSON),
        "input_policy": {
            "vggt_observation_input": True,
            "v50r2_used_for_paired_evaluation_only": True,
            "v50r2_used_at_inference": False,
            "candidate_points_used_at_inference": False,
            "kinect_depth_used_at_inference": False,
        },
        "gates": {
            "model_owned_forward_ran": True,
            "no_teacher_copy": not copy_result["leak_detected"],
            "paired_true_improves_vggt_baseline": true_improves_baseline,
            "paired_true_improves_no_smpl": true_improves_no_smpl,
            "paired_true_improves_shuffled_semantic": true_improves_shuffled,
            "paired_rgb_not_worse_than_baseline": rgb_not_worse,
            "part_paired_surface_pass": part_pass,
            "paired_surface_pass": paired_surface_pass,
            "paired_gate_threshold": "pass if true distance is at least 5e-5 lower than control or <= 0.85x control distance",
            "accepted_for_v509": False,
            "mentor_ready": False,
            "not_promoted": True,
        },
        "metrics": row,
        "decision": (
            "Paired visible-surface control separation passes as an auxiliary gate, but V509/V512 full-scene visual gates must still decide mentor readiness."
            if paired_surface_pass
            else "Fail closed: paired visible-surface evaluation still does not prove true beats controls sufficiently. Continue control-separation / SMPL binding repair."
        ),
        "artifact_hashes": {
            "candidate_npz": sha256(candidate_npz),
            "true_full_scene_ply": sha256(OUT / "v516_true_full_scene_rgb.ply"),
            "baseline_full_scene_ply": sha256(OUT / "v516_vggt_baseline_full_scene_rgb.ply"),
            "board": sha256(BOARD),
        },
    }
    write_json(DECISION, payload)
    print(json.dumps({"status": status, "decision": str(DECISION), "board": str(BOARD)}, indent=2))
    return 0 if paired_surface_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
