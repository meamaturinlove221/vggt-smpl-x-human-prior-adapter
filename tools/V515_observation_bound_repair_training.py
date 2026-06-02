from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.V514_v50r2_checkpoint_adjudicator import (  # noqa: E402
    CHECKPOINT as V508_CHECKPOINT,
    TEACHER_BANK,
    flatten_output,
    load_model,
    load_observation_inputs,
    load_teacher_points,
    nearest_mean,
    render_points,
    write_json,
    write_ply,
)
from PIL import Image, ImageDraw, ImageFont  # noqa: E402


REPORTS = ROOT / "reports"
BOARDS = ROOT / "boards"
OUT = ROOT / "output" / "V5150000000000000000000_observation_bound_repair_training"
DECISION = REPORTS / "V5150000000000000000000_observation_bound_repair_decision.json"
METRICS = REPORTS / "V5150000000000000000000_observation_bound_repair_metrics.csv"
BOARD = BOARDS / "V5150000000000000000000_observation_bound_repair_board.png"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_supervision(selected_indices: np.ndarray, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    with np.load(TEACHER_BANK, allow_pickle=False) as z:
        teacher_points = z["points"].astype(np.float32)
        teacher_rgb = z["rgb"].astype(np.float32) / 255.0
        teacher_mask = z["full_body_mask"].astype(bool)
    view_n, sample_n = batch["vggt_world_points"].shape[:2]
    selected = selected_indices.reshape(view_n, sample_n, 3)
    target_points = batch["vggt_world_points"].clone()
    target_rgb = batch["vggt_rgb"].clone()
    teacher_weight = torch.zeros(view_n, sample_n, 1)
    for view in range(min(6, view_n)):
        yy = selected[view, :, 1]
        xx = selected[view, :, 2]
        valid = teacher_mask[view, yy, xx]
        target_points[view, valid] = torch.from_numpy(teacher_points[view, yy[valid], xx[valid]])
        target_rgb[view, valid] = torch.from_numpy(teacher_rgb[view, yy[valid], xx[valid]])
        teacher_weight[view, valid, 0] = 1.0
    return {
        "target_points": target_points,
        "target_rgb": target_rgb,
        "teacher_weight": teacher_weight,
        "preserve_points": batch["vggt_world_points"].clone(),
        "preserve_rgb": batch["vggt_rgb"].clone(),
    }


def weighted_mse(value: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = (value - target).square().sum(dim=-1, keepdim=True)
    denom = weight.sum().clamp_min(1.0)
    return (loss * weight).sum() / denom


def make_repair_board(true_points: np.ndarray, true_rgb: np.ndarray, base_points: np.ndarray, base_rgb: np.ndarray, teacher_points: np.ndarray, metrics: dict[str, Any]) -> None:
    BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1180), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V515 observation-bound repair: train subset only, not mentor-ready", fill=(0, 0, 0), font=font)
    draw.text((18, 36), "V50R2 is loss/reference only. Full V514 adjudication is required before V509 insertion.", fill=(140, 0, 0), font=font)
    panels = [
        ("repair true full-scene XZ", render_points(true_points, true_rgb, (580, 360), (0, 2))),
        ("repair true full-scene XY", render_points(true_points, true_rgb, (580, 360), (0, 1))),
        ("repair baseline XZ", render_points(base_points, base_rgb, (580, 360), (0, 2))),
        ("V50R2 floor reference XZ", render_points(teacher_points, np.tile(np.array([[0.2, 0.2, 0.2]], dtype=np.float32), (teacher_points.shape[0], 1)), (580, 360), (0, 2))),
    ]
    for i, (label, img) in enumerate(panels):
        x = 18 + (i % 3) * 595
        y = 70 + (i // 3) * 430
        board.paste(img, (x, y + 26))
        draw.rectangle([x, y + 26, x + 580, y + 386], outline=(80, 80, 80), width=1)
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
    y = 930
    for line in [
        f"status: {metrics['status']}",
        f"true_to_v50r2_nn_mean: {metrics['true_to_v50r2_nn_mean']:.6f}",
        f"baseline_to_v50r2_nn_mean: {metrics['baseline_to_v50r2_nn_mean']:.6f}",
        f"no_smpl_to_v50r2_nn_mean: {metrics['no_smpl_to_v50r2_nn_mean']:.6f}",
        f"shuffled_to_v50r2_nn_mean: {metrics['shuffled_to_v50r2_nn_mean']:.6f}",
        f"true_improves_baseline: {metrics['true_improves_baseline']}",
        f"accepted_for_v509: False",
    ]:
        draw.text((18, y), line, fill=(0, 0, 0), font=font)
        y += 28
    board.save(BOARD)


def train(args: argparse.Namespace) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    batch, meta = load_observation_inputs(max_human_per_view=args.max_samples, max_env_per_view=512, seed=args.seed)
    model, source_ckpt = load_model(Path(args.checkpoint))
    model.train()
    target = load_supervision(meta["selected_indices"], batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    history = []
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        out_true = model(batch, control="true")
        out_base = model(batch, control="vggt_baseline")
        out_no_smpl = model(batch, control="no_smpl")
        w = target["teacher_weight"]
        true_target = weighted_mse(out_true["student_human_points"], target["target_points"], w)
        rgb_target = weighted_mse(out_true["student_human_rgb"], target["target_rgb"], w)
        preserve = F.mse_loss(out_true["student_human_points"], target["preserve_points"]) * 0.15
        residual = out_true["residual"].square().mean() * 0.15
        base_dist = ((out_base["student_human_points"] - target["target_points"]).square().sum(dim=-1, keepdim=True) * w).detach()
        no_smpl_dist = ((out_no_smpl["student_human_points"] - target["target_points"]).square().sum(dim=-1, keepdim=True) * w).detach()
        true_dist = ((out_true["student_human_points"] - target["target_points"]).square().sum(dim=-1, keepdim=True) * w)
        margin = torch.relu(true_dist - base_dist + args.margin).sum() / w.sum().clamp_min(1.0)
        margin = margin + torch.relu(true_dist - no_smpl_dist + args.margin).sum() / w.sum().clamp_min(1.0)
        loss = true_target + 0.35 * rgb_target + preserve + residual + 0.5 * margin
        loss.backward()
        opt.step()
        if step == 1 or step % max(args.steps // 10, 1) == 0 or step == args.steps:
            history.append({
                "step": step,
                "loss": float(loss.detach()),
                "true_target": float(true_target.detach()),
                "rgb_target": float(rgb_target.detach()),
                "margin": float(margin.detach()),
                "preserve": float(preserve.detach()),
            })

    model.eval()
    with torch.no_grad():
        outs = {control: flatten_output(model(batch, control=control)) for control in ["true", "vggt_baseline", "no_smpl", "shuffled_semantic"]}
    teacher_points = load_teacher_points()
    true_nn = nearest_mean(outs["true"]["points"], teacher_points, max_a=5000, max_b=5000, seed=args.seed)
    base_nn = nearest_mean(outs["vggt_baseline"]["points"], teacher_points, max_a=5000, max_b=5000, seed=args.seed + 1)
    no_smpl_nn = nearest_mean(outs["no_smpl"]["points"], teacher_points, max_a=5000, max_b=5000, seed=args.seed + 2)
    shuffled_nn = nearest_mean(outs["shuffled_semantic"]["points"], teacher_points, max_a=5000, max_b=5000, seed=args.seed + 3)
    true_improves_baseline = bool(true_nn < base_nn - 5.0e-4)
    true_improves_no_smpl = bool(true_nn < no_smpl_nn - 5.0e-4)
    true_improves_shuffled = bool(true_nn < shuffled_nn - 5.0e-4)
    checkpoint = OUT / f"checkpoint_{args.steps}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": source_ckpt.get("config", {}),
            "steps": int(args.steps),
            "source_checkpoint": str(args.checkpoint),
            "route": "V515_observation_bound_repair_training",
        },
        checkpoint,
    )
    env_points = batch["environment_points"].numpy()[0]
    env_rgb = batch["environment_rgb"].numpy()[0]
    full_true_points = np.concatenate([env_points, outs["true"]["points"]], axis=0)
    full_true_rgb = np.concatenate([env_rgb, outs["true"]["rgb"]], axis=0)
    full_base_points = np.concatenate([env_points, outs["vggt_baseline"]["points"]], axis=0)
    full_base_rgb = np.concatenate([env_rgb, outs["vggt_baseline"]["rgb"]], axis=0)
    write_ply(OUT / "v515_train_subset_true_full_scene_rgb.ply", full_true_points, full_true_rgb)
    write_ply(OUT / "v515_train_subset_baseline_full_scene_rgb.ply", full_base_points, full_base_rgb)
    status = (
        "V515_OBSERVATION_BOUND_REPAIR_TRAIN_SUBSET_CONTROL_SEPARATION_PASS_NEEDS_FULL_ADJUDICATION_NOT_PROMOTED"
        if (true_improves_baseline and true_improves_no_smpl and true_improves_shuffled)
        else "V515_OBSERVATION_BOUND_REPAIR_TRAIN_SUBSET_FAIL_CONTINUE_REPAIR_NOT_PROMOTED"
    )
    row = {
        "status": status,
        "steps": args.steps,
        "max_samples_per_view": args.max_samples,
        "source_checkpoint": str(args.checkpoint),
        "output_checkpoint": str(checkpoint),
        "true_to_v50r2_nn_mean": true_nn,
        "baseline_to_v50r2_nn_mean": base_nn,
        "no_smpl_to_v50r2_nn_mean": no_smpl_nn,
        "shuffled_to_v50r2_nn_mean": shuffled_nn,
        "true_improves_baseline": true_improves_baseline,
        "true_improves_no_smpl": true_improves_no_smpl,
        "true_improves_shuffled": true_improves_shuffled,
        "teacher_weight_count": int(target["teacher_weight"].sum().item()),
        "history_last_loss": history[-1]["loss"],
    }
    write_csv(METRICS, [row])
    make_repair_board(full_true_points, full_true_rgb, full_base_points, full_base_rgb, teacher_points, row)
    payload = {
        "task": "V515_observation_bound_repair_training",
        "status": status,
        "created_at": now(),
        "repo": str(ROOT),
        "source_checkpoint": str(args.checkpoint),
        "output_checkpoint": str(checkpoint),
        "metrics_csv": str(METRICS),
        "board": str(BOARD),
        "train_subset_true_full_scene_ply": str(OUT / "v515_train_subset_true_full_scene_rgb.ply"),
        "train_subset_baseline_full_scene_ply": str(OUT / "v515_train_subset_baseline_full_scene_rgb.ply"),
        "input_policy": {
            "vggt_observation_input": True,
            "v50r2_used_for_loss_only": True,
            "v50r2_used_at_inference": False,
            "kinect_depth_used_at_inference": False,
            "requires_v514_full_adjudication_before_v509": True,
        },
        "gates": {
            "train_subset_true_improves_vggt_baseline": true_improves_baseline,
            "train_subset_true_improves_no_smpl": true_improves_no_smpl,
            "train_subset_true_improves_shuffled_semantic": true_improves_shuffled,
            "accepted_for_v509": False,
            "mentor_ready": False,
            "not_promoted": True,
        },
        "loss_history": history,
        "metrics": row,
        "decision": "This is a repair checkpoint on observation-bound inputs. It may only feed V514 full adjudication; it is not a mentor-ready output and is not promoted.",
    }
    write_json(DECISION, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(V508_CHECKPOINT))
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--max-samples", type=int, default=768)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--margin", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=515)
    args = parser.parse_args()
    payload = train(args)
    print(json.dumps({"status": payload["status"], "decision": str(DECISION), "checkpoint": payload["output_checkpoint"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
