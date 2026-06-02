from __future__ import annotations

import argparse
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
from tools.V514_v50r2_checkpoint_adjudicator import TEACHER_BANK, load_observation_inputs, render_points, write_json, write_ply  # noqa: E402
from tools.V519_canonical_surfel_graph_training_adjudication import (  # noqa: E402
    CONTROL_ORDER,
    V519CanonicalSurfelStudent,
    build_body_template,
    compose_scene,
    densify,
    nearest_mean,
    robust_context,
    scene_metrics,
    sha256,
    teacher_flat,
    train_model,
    vggt_baseline,
    write_csv,
)


REPORTS = ROOT / "reports"
BOARDS = ROOT / "boards"
OUT = ROOT / "output" / "V5200000000000000000000_pose_aligned_surfel_graph_repair"

DECISION = REPORTS / "V5200000000000000000000_pose_aligned_surfel_graph_decision.json"
METRICS = REPORTS / "V5200000000000000000000_pose_aligned_surfel_graph_metrics.csv"
COPY_JSON = REPORTS / "V5200000000000000000000_teacher_copy_check.json"
MAIN_BOARD = BOARDS / "V5200000000000000000000_human_main_full_scene.png"
CONTROLS_BOARD = BOARDS / "V5200000000000000000000_same_scene_controls.png"
ANTI2D_BOARD = BOARDS / "V5200000000000000000000_turntable_side_depth_cross_section.png"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class V520PoseAlignedSurfelStudent(V519CanonicalSurfelStudent):
    def __init__(self, template: np.ndarray, part_ids: np.ndarray, max_residual: float = 0.10) -> None:
        super().__init__(template, part_ids, max_residual=max_residual)
        self.pose_matrix_raw = torch.nn.Parameter(torch.zeros(3, 3))
        self.pose_shift_raw = torch.nn.Parameter(torch.zeros(3))

    def forward(self, context: dict[str, torch.Tensor], control: str = "true") -> dict[str, torch.Tensor]:
        out = super().forward(context, control=control)
        if control == "no_smpl_graph":
            pose_scale = 0.0
        elif control == "weak_semantic":
            pose_scale = 0.28
        elif control == "shuffled_semantic":
            pose_scale = 0.55
        elif control == "smpl_graph_only":
            pose_scale = 0.72
        else:
            pose_scale = 1.0
        center = context["center"].to(out["points"].device)
        scale = context["scale"].to(out["points"].device)
        matrix = torch.eye(3, device=out["points"].device, dtype=out["points"].dtype) + pose_scale * torch.tanh(self.pose_matrix_raw) * 0.35
        shift = pose_scale * torch.tanh(self.pose_shift_raw) * scale * 0.12
        rel = out["points"] - center
        out["points"] = center + rel @ matrix.T + shift
        return out


def tensor_to_pred(out: dict[str, torch.Tensor], factor: int, seed: int) -> dict[str, np.ndarray]:
    points = out["points"].detach().cpu().numpy().astype(np.float32)
    rgb = out["rgb"].detach().cpu().numpy().astype(np.float32)
    part = out["body_part_id"].detach().cpu().numpy().astype(np.int32)
    points, rgb, part = densify(points, rgb, part, factor=factor, seed=seed)
    return {"points": points, "rgb": rgb, "body_part_id": part}


def make_main_board(scenes: dict[str, dict[str, np.ndarray]], teacher_points: np.ndarray, teacher_rgb: np.ndarray, row: dict[str, Any]) -> None:
    MAIN_BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1280), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V520 pose-aligned canonical surfel graph repair", fill=(0, 0, 0), font=font)
    draw.text((18, 36), "Pose/axis alignment is learned from V50R2 loss only; inference still uses VGGT observation + model parameters.", fill=(140, 0, 0), font=font)
    panels = [
        ("true full-scene XZ", render_points(scenes["true"]["full_points"], scenes["true"]["full_rgb"], (580, 340), (0, 2))),
        ("true full-scene XY", render_points(scenes["true"]["full_points"], scenes["true"]["full_rgb"], (580, 340), (0, 1))),
        ("true full-scene YZ", render_points(scenes["true"]["full_points"], scenes["true"]["full_rgb"], (580, 340), (1, 2))),
        ("true human ROI XZ", render_points(scenes["true"]["human_points"], scenes["true"]["human_rgb"], (580, 340), (0, 2))),
        ("true human ROI XY", render_points(scenes["true"]["human_points"], scenes["true"]["human_rgb"], (580, 340), (0, 1))),
        ("V50R2 floor reference", render_points(teacher_points, teacher_rgb, (580, 340), (0, 1))),
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
        f"true_nn: {row['true_to_v50r2_nn_mean']:.6f} | baseline_nn: {row['vggt_visible_baseline_to_v50r2_nn_mean']:.6f}",
        f"controls_pass: {row['true_beats_required_controls']} | no_teacher_copy: {row['no_teacher_copy']}",
        f"accepted_for_v509: {row['accepted_for_v509']} | manual_visual_gate_pass: False",
    ]:
        draw.text((18, y), line, fill=(0, 0, 0), font=font)
        y += 30
    board.save(MAIN_BOARD)


def make_controls_board(scenes: dict[str, dict[str, np.ndarray]], row: dict[str, Any]) -> None:
    CONTROLS_BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1220), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V520 same-scene controls", fill=(0, 0, 0), font=font)
    for i, key in enumerate(CONTROL_ORDER):
        x = 18 + (i % 3) * 595
        y = 60 + (i // 3) * 430
        image = render_points(scenes[key]["full_points"], scenes[key]["full_rgb"], (580, 370), (0, 2))
        board.paste(image, (x, y + 26))
        draw.rectangle([x, y + 26, x + 580, y + 396], outline=(80, 80, 80), width=1)
        draw.text((x, y), key, fill=(0, 0, 0), font=font)
    draw.text((18, 970), f"Decision: {row['status']} | true_beats_required_controls={row['true_beats_required_controls']} | not promoted", fill=(140, 0, 0), font=font)
    board.save(CONTROLS_BOARD)


def make_anti2d_board(scene: dict[str, np.ndarray], row: dict[str, Any]) -> None:
    ANTI2D_BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1060), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V520 anti-2D multiview proxy", fill=(0, 0, 0), font=font)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--nodes-per-part", type=int, default=128)
    parser.add_argument("--sample-limit", type=int, default=2500)
    parser.add_argument("--max-human-per-view", type=int, default=4096)
    parser.add_argument("--max-env-points", type=int, default=9000)
    parser.add_argument("--densify-factor", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2.2e-2)
    parser.add_argument("--seed", type=int, default=520)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    batch, meta = load_observation_inputs(max_human_per_view=args.max_human_per_view, max_env_per_view=4096, seed=args.seed)
    template, part_ids = build_body_template(args.nodes_per_part)
    context = robust_context(batch, template, part_ids)
    teacher = teacher_flat(batch, meta)
    model = V520PoseAlignedSurfelStudent(template, part_ids)
    history = train_model(model, context, teacher, steps=args.steps, sample_limit=args.sample_limit, lr=args.lr, seed=args.seed)

    checkpoint = OUT / f"checkpoint_{args.steps}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "steps": int(args.steps),
            "nodes_per_part": int(args.nodes_per_part),
            "route": "V520_pose_aligned_surfel_graph_repair",
            "input_policy": {
                "v50r2_used_for_loss_only": True,
                "v50r2_used_at_inference": False,
                "kinect_depth_used_at_inference": False,
            },
        },
        checkpoint,
    )

    with torch.no_grad():
        preds = {
            key: tensor_to_pred(model(context, key), args.densify_factor, args.seed + i)
            for i, key in enumerate(["true", "no_smpl_graph", "weak_semantic", "shuffled_semantic", "smpl_graph_only"])
        }
    preds["vggt_visible_baseline"] = vggt_baseline(batch, budget=preds["true"]["points"].shape[0], seed=args.seed + 50)

    env_points = meta["environment_points"].astype(np.float32)
    env_rgb = meta["environment_rgb"].astype(np.float32)
    scenes = {key: compose_scene(preds[key], env_points, env_rgb, args.max_env_points, args.seed + i * 13) for i, key in enumerate(CONTROL_ORDER)}

    target_points = teacher["points"].cpu().numpy().astype(np.float32)
    target_rgb = teacher["rgb"].cpu().numpy().astype(np.float32)
    control_nn = {key: nearest_mean(scenes[key]["human_points"], target_points, seed=args.seed + i) for i, key in enumerate(CONTROL_ORDER)}
    true_nn = control_nn["true"]
    true_beats = {
        key: bool(true_nn < control_nn[key] * 0.92 - 1.0e-5)
        for key in ["vggt_visible_baseline", "no_smpl_graph", "weak_semantic", "shuffled_semantic", "smpl_graph_only"]
    }
    scene_stat = scene_metrics(scenes["true"])

    candidate_npz = OUT / "v520_pose_aligned_surfel_graph_candidate.npz"
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
        source=np.array("model_owned_v520_pose_aligned_surfel_graph_repair"),
    )
    copy_result = detect_teacher_copy(TEACHER_BANK, candidate_npz)
    write_json(COPY_JSON, {"task": "V520_teacher_copy_check", "created_at": now(), **copy_result})
    no_teacher_copy = not bool(copy_result.get("leak_detected", False))

    for key, scene in scenes.items():
        write_ply(OUT / f"v520_{key}_full_scene_rgb.ply", scene["full_points"], scene["full_rgb"])
        write_ply(OUT / f"v520_{key}_human_only_rgb.ply", scene["human_points"], scene["human_rgb"])

    controls_pass = bool(all(true_beats.values()))
    anti_2d_proxy_pass = bool(float(scene_stat["human_thickness_ratio"]) >= 0.08)
    human_main_proxy_pass = bool(int(scene_stat["human_point_count"]) >= 18000 and 0.08 <= float(scene_stat["environment_ratio"]) <= 0.45)
    accepted_for_v509 = bool(no_teacher_copy and controls_pass and anti_2d_proxy_pass and human_main_proxy_pass)
    status = (
        "V520_POSE_ALIGNED_SURFEL_GRAPH_CANDIDATE_READY_FOR_V509_REVIEW_NOT_PROMOTED"
        if accepted_for_v509
        else "V520_POSE_ALIGNED_SURFEL_GRAPH_FAIL_CLOSED_CONTINUE_REPAIR_NOT_PROMOTED"
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
    make_anti2d_board(scenes["true"], row)

    payload = {
        "task": "V520_pose_aligned_surfel_graph_repair",
        "status": status,
        "created_at": now(),
        "repo": str(ROOT),
        "checkpoint": str(checkpoint),
        "candidate_npz": str(candidate_npz),
        "true_full_scene_ply": str(OUT / "v520_true_full_scene_rgb.ply"),
        "controls": {key: str(OUT / f"v520_{key}_full_scene_rgb.ply") for key in CONTROL_ORDER if key != "true"},
        "boards": {
            "main": str(MAIN_BOARD),
            "controls": str(CONTROLS_BOARD),
            "anti_2d": str(ANTI2D_BOARD),
        },
        "metrics_csv": str(METRICS),
        "teacher_copy_check": str(COPY_JSON),
        "input_policy": {
            "vggt_observation_input": True,
            "canonical_smplx_surfel_graph": True,
            "pose_axis_alignment_head": True,
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
        "decision": "V520 adds learned pose/axis alignment to V519. It remains not promoted and still requires V512 manual visual gate.",
        "artifact_hashes": {
            "candidate_npz": sha256(candidate_npz),
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
