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
from tools.V516_paired_visible_surface_adjudication import load_teacher_targets, paired_distance  # noqa: E402


REPORTS = ROOT / "reports"
BOARDS = ROOT / "boards"
OUT = ROOT / "output" / "V5170000000000000000000_full_scene_clarity_composer"
DEFAULT_CHECKPOINT = ROOT / "output" / "V5150000000000000000000_observation_bound_repair_training" / "checkpoint_240.pt"

DECISION = REPORTS / "V5170000000000000000000_full_scene_clarity_decision.json"
METRICS = REPORTS / "V5170000000000000000000_full_scene_clarity_metrics.csv"
COPY_JSON = REPORTS / "V5170000000000000000000_teacher_copy_check.json"
MAIN_BOARD = BOARDS / "V5170000000000000000000_human_main_full_scene.png"
CONTROLS_BOARD = BOARDS / "V5170000000000000000000_same_scene_controls.png"
ANTI2D_BOARD = BOARDS / "V5170000000000000000000_turntable_side_depth_cross_section.png"


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


def select_human(points: np.ndarray, rgb: np.ndarray, visibility: np.ndarray, occupancy: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = np.nan_to_num(visibility, nan=0.0) * 0.6 + np.nan_to_num(occupancy, nan=0.0) * 0.4
    valid = np.isfinite(points).all(axis=-1) & (score > np.percentile(score, 18))
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        idx = np.arange(points.shape[0])
    if idx.size > max_points:
        # Keep high confidence structure and add deterministic spread for visual continuity.
        top = idx[np.argsort(score[idx])[-max_points // 2:]]
        rng = np.random.default_rng(seed)
        rest_pool = np.setdiff1d(idx, top, assume_unique=False)
        rest_take = max_points - top.size
        rest = rng.choice(rest_pool if rest_pool.size else idx, size=rest_take, replace=rest_pool.size < rest_take)
        idx = np.concatenate([top, rest])
    idx = np.sort(idx)
    return points[idx], rgb[idx], idx


def select_environment(env_points: np.ndarray, env_rgb: np.ndarray, human_points: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    center = np.median(human_points, axis=0)
    span = np.percentile(human_points, 95, axis=0) - np.percentile(human_points, 5, axis=0)
    span = np.maximum(span, np.array([0.08, 0.08, 0.08], dtype=np.float32))
    rel = np.abs(env_points - center)
    near = (rel[:, 0] < span[0] * 4.5) & (rel[:, 1] < span[1] * 5.0) & (rel[:, 2] < span[2] * 3.0)
    floor_or_context = near | (env_points[:, 1] < np.percentile(human_points[:, 1], 12))
    idx = np.flatnonzero(np.isfinite(env_points).all(axis=-1) & floor_or_context)
    if idx.size == 0:
        idx = np.flatnonzero(np.isfinite(env_points).all(axis=-1))
    if idx.size > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)
    return env_points[idx], env_rgb[idx]


def compose_scene(model_out: dict[str, np.ndarray], env_points: np.ndarray, env_rgb: np.ndarray, *, max_human: int, max_env: int, seed: int) -> dict[str, np.ndarray]:
    human_points, human_rgb, idx = select_human(
        model_out["points"],
        model_out["rgb"],
        model_out.get("visibility", np.ones(model_out["points"].shape[0], dtype=np.float32)),
        model_out.get("occupancy", np.ones(model_out["points"].shape[0], dtype=np.float32)),
        max_human,
        seed,
    )
    env_p, env_r = select_environment(env_points, env_rgb, human_points, max_env, seed + 10)
    full_points = np.concatenate([env_p, human_points], axis=0).astype(np.float32)
    full_rgb = np.concatenate([env_r, human_rgb], axis=0).astype(np.float32)
    is_human = np.concatenate([np.zeros(env_p.shape[0], dtype=np.uint8), np.ones(human_points.shape[0], dtype=np.uint8)])
    return {
        "full_points": full_points,
        "full_rgb": full_rgb,
        "human_points": human_points.astype(np.float32),
        "human_rgb": human_rgb.astype(np.float32),
        "environment_points": env_p.astype(np.float32),
        "environment_rgb": env_r.astype(np.float32),
        "is_human": is_human,
        "selected_human_index": idx.astype(np.int32),
    }


def bbox_metrics(human: np.ndarray, env: np.ndarray) -> dict[str, float]:
    span = np.percentile(human, 95, axis=0) - np.percentile(human, 5, axis=0)
    thickness_ratio = float(min(span[0], span[1], span[2]) / max(max(span[0], span[1], span[2]), 1.0e-6))
    env_ratio = float(env.shape[0] / max(human.shape[0] + env.shape[0], 1))
    return {
        "human_span_x": float(span[0]),
        "human_span_y": float(span[1]),
        "human_span_z": float(span[2]),
        "human_thickness_ratio": thickness_ratio,
        "environment_ratio": env_ratio,
    }


def crop_human_image(points: np.ndarray, rgb: np.ndarray, size: tuple[int, int], axes: tuple[int, int]) -> Image.Image:
    return render_points(points, rgb, size, axes)


def make_main_board(true_scene: dict[str, np.ndarray], baseline_scene: dict[str, np.ndarray], teacher_points: np.ndarray, teacher_rgb: np.ndarray, row: dict[str, Any]) -> None:
    MAIN_BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1260), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V517 human-main full-scene clarity composer: model-owned output with partial VGGT environment", fill=(0, 0, 0), font=font)
    draw.text((18, 36), "V50R2 shown only as visual floor/reference. Final true points are V515/V516 model-owned inference output.", fill=(140, 0, 0), font=font)
    panels = [
        ("true human-main full-scene XZ", render_points(true_scene["full_points"], true_scene["full_rgb"], (580, 360), (0, 2))),
        ("true human-main full-scene XY", render_points(true_scene["full_points"], true_scene["full_rgb"], (580, 360), (0, 1))),
        ("baseline same-scene XZ", render_points(baseline_scene["full_points"], baseline_scene["full_rgb"], (580, 360), (0, 2))),
        ("true human ROI XZ", crop_human_image(true_scene["human_points"], true_scene["human_rgb"], (580, 360), (0, 2))),
        ("true human ROI XY", crop_human_image(true_scene["human_points"], true_scene["human_rgb"], (580, 360), (0, 1))),
        ("V50R2 floor reference", render_points(teacher_points, teacher_rgb, (580, 360), (0, 1))),
    ]
    for i, (label, image) in enumerate(panels):
        x = 18 + (i % 3) * 595
        y = 70 + (i // 3) * 420
        board.paste(image, (x, y + 26))
        draw.rectangle([x, y + 26, x + 580, y + 386], outline=(80, 80, 80), width=1)
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
    y = 930
    for line in [
        f"status: {row['status']}",
        f"human_points: {row['human_point_count']} | environment_points: {row['environment_point_count']} | env_ratio: {row['environment_ratio']:.3f}",
        f"thickness_ratio: {row['human_thickness_ratio']:.4f}",
        f"paired_surface_pass: {row['paired_surface_pass']} | no_teacher_copy: {row['no_teacher_copy']}",
        f"human_main_full_scene_pass: {row['human_main_full_scene_pass']}",
        f"anti_2d_proxy_pass: {row['anti_2d_proxy_pass']}",
        f"mentor_ready: False",
    ]:
        draw.text((18, y), line, fill=(0, 0, 0), font=font)
        y += 28
    board.save(MAIN_BOARD)


def make_controls_board(true_scene: dict[str, np.ndarray], controls: dict[str, dict[str, np.ndarray]], row: dict[str, Any]) -> None:
    CONTROLS_BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1120), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V517 same-scene controls: true vs VGGT baseline / no-SMPL / shuffled semantic", fill=(0, 0, 0), font=font)
    draw.text((18, 36), "All panels use the same VGGT environment selection. Controls remain same-scene.", fill=(120, 0, 0), font=font)
    panels = [
        ("true", true_scene),
        ("VGGT baseline", controls["vggt_baseline"]),
        ("no SMPL", controls["no_smpl"]),
        ("shuffled semantic", controls["shuffled_semantic"]),
    ]
    for i, (label, scene) in enumerate(panels):
        x = 18 + (i % 2) * 890
        y = 70 + (i // 2) * 480
        img = render_points(scene["full_points"], scene["full_rgb"], (860, 420), (0, 2))
        board.paste(img, (x, y + 28))
        draw.rectangle([x, y + 28, x + 860, y + 448], outline=(80, 80, 80), width=1)
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
    draw.text((18, 1040), f"Decision: {row['status']} | true retains V516 paired-surface advantage but still requires V512 manual visual pass.", fill=(0, 0, 0), font=font)
    board.save(CONTROLS_BOARD)


def make_anti2d_board(true_scene: dict[str, np.ndarray], row: dict[str, Any]) -> None:
    ANTI2D_BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1060), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V517 anti-2D proxy: turntable axes / side-depth / cross-section", fill=(0, 0, 0), font=font)
    panels = [
        ("XZ", render_points(true_scene["full_points"], true_scene["full_rgb"], (580, 360), (0, 2))),
        ("YZ", render_points(true_scene["full_points"], true_scene["full_rgb"], (580, 360), (1, 2))),
        ("XY", render_points(true_scene["full_points"], true_scene["full_rgb"], (580, 360), (0, 1))),
        ("human XZ", render_points(true_scene["human_points"], true_scene["human_rgb"], (580, 360), (0, 2))),
        ("human YZ", render_points(true_scene["human_points"], true_scene["human_rgb"], (580, 360), (1, 2))),
        ("human XY", render_points(true_scene["human_points"], true_scene["human_rgb"], (580, 360), (0, 1))),
    ]
    for i, (label, image) in enumerate(panels):
        x = 18 + (i % 3) * 595
        y = 66 + (i // 3) * 420
        board.paste(image, (x, y + 26))
        draw.rectangle([x, y + 26, x + 580, y + 386], outline=(80, 80, 80), width=1)
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
    draw.text((18, 928), f"human_thickness_ratio={row['human_thickness_ratio']:.4f}; anti_2d_proxy_pass={row['anti_2d_proxy_pass']}; mentor_ready=False", fill=(0, 0, 0), font=font)
    board.save(ANTI2D_BOARD)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--max-human-points", type=int, default=36000)
    parser.add_argument("--max-env-points", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=517)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    batch, meta = load_observation_inputs(max_human_per_view=args.max_samples, max_env_per_view=4096, seed=args.seed)
    model, ckpt = load_model(Path(args.checkpoint))
    model.eval()
    with torch.no_grad():
        raw = {control: flatten_output(model(batch, control=control)) for control in ["true", "vggt_baseline", "no_smpl", "shuffled_semantic"]}
    env_points = meta["environment_points"].astype(np.float32)
    env_rgb = meta["environment_rgb"].astype(np.float32)
    scenes = {
        key: compose_scene(value, env_points, env_rgb, max_human=args.max_human_points, max_env=args.max_env_points, seed=args.seed + i)
        for i, (key, value) in enumerate(raw.items())
    }

    view_count, sample_count = batch["vggt_world_points"].shape[:2]
    teacher = load_teacher_targets(meta["selected_indices"], view_count, sample_count)
    teacher_points = teacher["points"].numpy().reshape(-1, 3)
    teacher_rgb = teacher["rgb"].numpy().reshape(-1, 3)
    visible = teacher["visible"].numpy().reshape(-1) > 0
    teacher_points = teacher_points[visible]
    teacher_rgb = teacher_rgb[visible]

    candidate_npz = OUT / "v517_human_main_model_owned_candidate.npz"
    np.savez_compressed(
        candidate_npz,
        full_scene_points=scenes["true"]["full_points"],
        full_scene_rgb=scenes["true"]["full_rgb"],
        human_points=scenes["true"]["human_points"],
        human_rgb=scenes["true"]["human_rgb"],
        environment_points=scenes["true"]["environment_points"],
        environment_rgb=scenes["true"]["environment_rgb"],
        is_human=scenes["true"]["is_human"],
        model_owned_student_output=np.array(True),
        no_teacher_points_inference=np.array(True),
        no_v50r2_inference=np.array(True),
        no_kinect_depth_inference=np.array(True),
        final_inference_allowed=np.array(True),
        source=np.array("model_owned_v517_full_scene_clarity_composer"),
    )
    copy_result = detect_teacher_copy(TEACHER_BANK, candidate_npz)
    write_json(COPY_JSON, {"task": "V517_teacher_copy_check", "created_at": now(), **copy_result})

    metrics = bbox_metrics(scenes["true"]["human_points"], scenes["true"]["environment_points"])
    visible_weight = teacher["visible"]
    true_paired = paired_distance(raw["true"]["points"], teacher["points"], visible_weight)
    base_paired = paired_distance(raw["vggt_baseline"]["points"], teacher["points"], visible_weight)
    no_smpl_paired = paired_distance(raw["no_smpl"]["points"], teacher["points"], visible_weight)
    human_count = int(scenes["true"]["human_points"].shape[0])
    env_count = int(scenes["true"]["environment_points"].shape[0])
    paired_surface_pass = bool(true_paired <= base_paired * 0.85 and true_paired <= no_smpl_paired * 0.85)
    human_main_full_scene_pass = bool(human_count >= 18000 and 0.08 <= metrics["environment_ratio"] <= 0.35)
    anti_2d_proxy_pass = bool(metrics["human_thickness_ratio"] >= 0.07)
    no_teacher_copy = not copy_result["leak_detected"]
    # This composer can unlock a candidate for manual visual review, but not final mentor pass.
    accepted_for_v509 = bool(no_teacher_copy and paired_surface_pass and human_main_full_scene_pass and anti_2d_proxy_pass)
    status = (
        "V517_HUMAN_MAIN_FULL_SCENE_CANDIDATE_READY_FOR_V509_V512_NOT_PROMOTED"
        if accepted_for_v509
        else "V517_FULL_SCENE_CLARITY_FAIL_CLOSED_CONTINUE_REPAIR_NOT_PROMOTED"
    )

    write_ply(OUT / "v517_true_human_main_full_scene_rgb.ply", scenes["true"]["full_points"], scenes["true"]["full_rgb"])
    write_ply(OUT / "v517_vggt_baseline_same_scene_rgb.ply", scenes["vggt_baseline"]["full_points"], scenes["vggt_baseline"]["full_rgb"])
    write_ply(OUT / "v517_no_smpl_same_scene_rgb.ply", scenes["no_smpl"]["full_points"], scenes["no_smpl"]["full_rgb"])
    write_ply(OUT / "v517_shuffled_semantic_same_scene_rgb.ply", scenes["shuffled_semantic"]["full_points"], scenes["shuffled_semantic"]["full_rgb"])

    row = {
        "status": status,
        "checkpoint": str(args.checkpoint),
        "checkpoint_steps": int(ckpt.get("steps", 0)),
        "human_point_count": human_count,
        "environment_point_count": env_count,
        "environment_ratio": metrics["environment_ratio"],
        "human_span_x": metrics["human_span_x"],
        "human_span_y": metrics["human_span_y"],
        "human_span_z": metrics["human_span_z"],
        "human_thickness_ratio": metrics["human_thickness_ratio"],
        "true_paired_dist": true_paired,
        "baseline_paired_dist": base_paired,
        "no_smpl_paired_dist": no_smpl_paired,
        "paired_surface_pass": paired_surface_pass,
        "no_teacher_copy": no_teacher_copy,
        "human_main_full_scene_pass": human_main_full_scene_pass,
        "anti_2d_proxy_pass": anti_2d_proxy_pass,
        "accepted_for_v509": accepted_for_v509,
        "mentor_ready": False,
    }
    write_csv(METRICS, [row])
    make_main_board(scenes["true"], scenes["vggt_baseline"], teacher_points, teacher_rgb, row)
    make_controls_board(scenes["true"], {k: scenes[k] for k in ["vggt_baseline", "no_smpl", "shuffled_semantic"]}, row)
    make_anti2d_board(scenes["true"], row)

    payload = {
        "task": "V517_full_scene_clarity_composer",
        "status": status,
        "created_at": now(),
        "repo": str(ROOT),
        "checkpoint": str(args.checkpoint),
        "candidate_npz": str(candidate_npz),
        "true_human_main_full_scene_ply": str(OUT / "v517_true_human_main_full_scene_rgb.ply"),
        "same_scene_controls": {
            "vggt_baseline": str(OUT / "v517_vggt_baseline_same_scene_rgb.ply"),
            "no_smpl": str(OUT / "v517_no_smpl_same_scene_rgb.ply"),
            "shuffled_semantic": str(OUT / "v517_shuffled_semantic_same_scene_rgb.ply"),
        },
        "boards": {
            "human_main_full_scene": str(MAIN_BOARD),
            "same_scene_controls": str(CONTROLS_BOARD),
            "anti_2d": str(ANTI2D_BOARD),
        },
        "metrics_csv": str(METRICS),
        "teacher_copy_check": str(COPY_JSON),
        "input_policy": {
            "human_points_source": "V515/V516 model-owned checkpoint inference",
            "environment_points_source": "VGGT observation environment points",
            "v50r2_used_at_inference": False,
            "v50r2_used_for_visual_floor_reference_only": True,
            "kinect_depth_used_at_inference": False,
        },
        "gates": {
            "no_teacher_copy": no_teacher_copy,
            "paired_surface_pass": paired_surface_pass,
            "human_main_full_scene_pass": human_main_full_scene_pass,
            "partial_environment_visible": 0.08 <= metrics["environment_ratio"] <= 0.35,
            "anti_2d_proxy_pass": anti_2d_proxy_pass,
            "same_scene_controls_generated": True,
            "accepted_for_v509": accepted_for_v509,
            "mentor_ready": False,
            "not_promoted": True,
        },
        "metrics": row,
        "decision": (
            "Candidate is ready for strict V509/V512 visual review, but not mentor-ready until those gates pass."
            if accepted_for_v509
            else "Fail closed: composer did not satisfy human-main full-scene clarity and anti-2D proxy gates."
        ),
        "artifact_hashes": {
            "candidate_npz": sha256(candidate_npz),
            "true_human_main_full_scene_ply": sha256(OUT / "v517_true_human_main_full_scene_rgb.ply"),
            "main_board": sha256(MAIN_BOARD),
            "controls_board": sha256(CONTROLS_BOARD),
            "anti2d_board": sha256(ANTI2D_BOARD),
        },
    }
    write_json(DECISION, payload)
    print(json.dumps({"status": status, "decision": str(DECISION), "main_board": str(MAIN_BOARD)}, indent=2))
    return 0 if accepted_for_v509 else 1


if __name__ == "__main__":
    raise SystemExit(main())
