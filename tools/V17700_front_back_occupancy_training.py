from __future__ import annotations

import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO = Path(os.environ.get("VGGT_REPO_ROOT", r"D:\vggt\vggt-canonical-surfel-adapter"))
sys.path.insert(0, str(REPO))

from models.v135_anti_billboard_topology_volume_student import AntiBillboardTopologyVolumeStudent  # noqa: E402
from tools.V13300_anti_billboard_metric_v2 import anti_billboard_metric_v2, pca_frame  # noqa: E402
from tools.V17300_multishell_topology_decoder_training import (  # noqa: E402
    ALLOWED_FACE_CLAIM,
    BASE_MATRIX,
    BOARDS,
    CONTROL_CONFIGS,
    REPORTS,
    build_batch,
    compose,
    cross_panel,
    hard_control_scores,
    load_npz,
    read_manifest,
    render_panel,
    rotation_matrix,
    select_device,
    training_losses,
    write_csv,
    write_json,
    write_ply,
)


OUTPUT = REPO / "output"
OUT_ROOT = OUTPUT / "V17700000000000000000_front_back_occupancy_training"
TRUE_CONFIG = "front_back_occupancy_true"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def as_rgb(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype != np.uint8:
        if out.size and np.issubdtype(out.dtype, np.number) and float(np.nanmax(out)) <= 1.5:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out[:, :3]


def _section_extreme_quota(points: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _center, _vals, _axes, proj = pca_frame(points)
    ranges = np.maximum(np.ptp(proj, axis=0), 1e-9)
    a = np.clip(np.floor((proj[:, 0] - proj[:, 0].min()) / ranges[0] * 18).astype(int), 0, 17)
    b = np.clip(np.floor((proj[:, 1] - proj[:, 1].min()) / ranges[1] * 14).astype(int), 0, 13)
    t = np.clip(np.floor((proj[:, 2] - proj[:, 2].min()) / ranges[2] * 10).astype(int), 0, 9)
    front = []
    back = []
    for ia in range(18):
        for ib in range(14):
            m = np.where((a == ia) & (b == ib))[0]
            if len(m) < 4:
                continue
            lo = m[t[m] <= 2]
            hi = m[t[m] >= 7]
            if len(lo) and len(hi):
                front.extend(lo.tolist())
                back.extend(hi.tolist())
    front_arr = np.array(front, dtype=np.int64)
    back_arr = np.array(back, dtype=np.int64)
    if len(front_arr) == 0 or len(back_arr) == 0:
        order = np.argsort(proj[:, 2])
        k = max(1, min(len(order) // 5, budget // 2))
        front_arr = order[:k]
        back_arr = order[-k:]
    return front_arr, back_arr, proj[:, 2].astype(np.float32)


def front_back_decode(
    out: dict[str, torch.Tensor],
    aux: dict[str, np.ndarray],
    targets: dict[str, torch.Tensor],
    *,
    replace_fraction: float = 0.62,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_points = aux["human_points"]
    base_rgb = aux["human_rgb"]
    body = aux["body_part_id"]
    idx = aux["idx"]
    weak = targets["weak"][0, :, 0].detach().cpu().numpy()
    repair = aux["repair"][idx] | (weak > 0.32)
    if not np.any(repair):
        return base_points.copy(), base_rgb.copy(), body.astype(np.int16).copy()

    shell_map = {
        "front": out["front_shell"][0].detach().cpu().numpy().astype(np.float32),
        "back": out["back_shell"][0].detach().cpu().numpy().astype(np.float32),
        "left": out["left_shell"][0].detach().cpu().numpy().astype(np.float32),
        "right": out["right_shell"][0].detach().cpu().numpy().astype(np.float32),
    }
    candidate = np.where(repair)[0]
    candidate = candidate[np.argsort(-aux["priority"][idx[candidate]])]
    target_n = len(base_points)
    replace_n = int(round(target_n * replace_fraction))
    keep_n = target_n - replace_n
    keep_mask = np.ones(target_n, dtype=bool)
    keep_mask[idx[candidate]] = False
    keep_pool = np.where(keep_mask)[0]
    if len(keep_pool) > keep_n:
        keep_pool = keep_pool[np.linspace(0, len(keep_pool) - 1, keep_n).round().astype(int)]

    # Reserve half the decoded budget for section-extreme front/back layers.
    front_idx, back_idx, thin_coord = _section_extreme_quota(base_points[idx[candidate]], max(2, replace_n // 2))
    cand_front = candidate[front_idx % len(candidate)]
    cand_back = candidate[back_idx % len(candidate)]
    per = {
        "front": replace_n // 4,
        "back": replace_n // 4,
        "left": replace_n // 8,
        "right": replace_n // 8,
    }
    remainder = replace_n - sum(per.values())
    per["front"] += remainder // 2
    per["back"] += remainder - remainder // 2

    decoded_points = []
    decoded_rgb = []
    decoded_body = []
    for layer, quota in per.items():
        if layer == "front":
            source = cand_front
        elif layer == "back":
            source = cand_back
        else:
            source = candidate
        reps = int(np.ceil(max(1, quota) / max(1, len(source))))
        pick = np.tile(source, reps)[:quota]
        decoded_points.append(shell_map[layer][pick])
        decoded_rgb.append(base_rgb[idx[pick]])
        decoded_body.append(body[idx[pick]])

    human = np.concatenate([base_points[keep_pool], *decoded_points], axis=0)
    rgb = np.concatenate([base_rgb[keep_pool], *decoded_rgb], axis=0)
    body_out = np.concatenate([body[keep_pool], *decoded_body], axis=0).astype(np.int16)
    if len(human) > target_n:
        human = human[:target_n]
        rgb = rgb[:target_n]
        body_out = body_out[:target_n]
    elif len(human) < target_n:
        add = target_n - len(human)
        human = np.concatenate([human, base_points[:add]], axis=0)
        rgb = np.concatenate([rgb, base_rgb[:add]], axis=0)
        body_out = np.concatenate([body_out, body[:add]], axis=0).astype(np.int16)
    return human.astype(np.float32), as_rgb(rgb), body_out


def extra_front_back_loss(out: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
    front_back = (out["front_shell"] - out["back_shell"]).norm(dim=-1)
    left_right = (out["left_shell"] - out["right_shell"]).norm(dim=-1)
    weak = targets["weak"].squeeze(-1)
    # Force actual front/back split beyond side shell split in weak regions.
    fb_target = 0.082 + 0.025 * weak
    extreme = torch.relu(fb_target - front_back).mean()
    side_balance = torch.relu(left_right * 0.75 - front_back).mean()
    return extreme + 0.25 * side_balance


def train_case(row: dict[str, str], steps: int, max_points: int, device: torch.device) -> dict[str, Any]:
    model = AntiBillboardTopologyVolumeStudent().to(device)
    batch, targets, aux = build_batch(row, max_points=max_points, device=device)
    control_scores = hard_control_scores(row["case"])
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    history: list[dict[str, float | int]] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        losses = training_losses(out, targets, control_scores)
        fb_loss = extra_front_back_loss(out, targets)
        total = losses["total"] + 0.55 * fb_loss
        total.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in {0, steps - 1} or (step + 1) % 50 == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(total.detach().cpu()),
                    "base_loss": float(losses["total"].detach().cpu()),
                    "front_back_loss": float(fb_loss.detach().cpu()),
                    "grad_norm": float(grad),
                }
            )
    with torch.no_grad():
        out = model(batch)
    human, human_rgb, body = front_back_decode(out, aux, targets)
    full = np.concatenate([human, aux["environment_points"]], axis=0)
    full_rgb = np.concatenate([human_rgb, aux["environment_rgb"]], axis=0)
    out_dir = OUT_ROOT / row["case"] / TRUE_CONFIG
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "predictions.npz",
        human_points=human,
        human_rgb=human_rgb,
        environment_points=aux["environment_points"],
        environment_rgb=aux["environment_rgb"],
        full_scene_points=full,
        full_scene_rgb=full_rgb,
        body_part_id=body,
        model_owned_student_output=np.array(True),
        teacher_points_used_at_inference=np.array(False),
        raw_kinect_depth_used_at_inference=np.array(False),
        facial_detail_target_applicable=np.array(False),
        face_detail_claim_allowed=np.array(False),
        allowed_face_claim=np.array(ALLOWED_FACE_CLAIM),
        config=np.array(TRUE_CONFIG),
        case_id=np.array(row["case"]),
        front_back_occupancy_decoded=np.array(True),
    )
    write_ply(out_dir / "full_scene_rgb_pointcloud.ply", full, full_rgb)
    metric = anti_billboard_metric_v2(human, body)
    return {
        "case": row["case"],
        "config": TRUE_CONFIG,
        "prediction": str(out_dir / "predictions.npz"),
        "ply": str(out_dir / "full_scene_rgb_pointcloud.ply"),
        "steps": steps,
        "max_points": max_points,
        "device": str(device),
        "history_json": json.dumps(history),
        "front_back_occupancy_decoded": True,
        "model_owned_student_output": True,
        "teacher_points_used_at_inference": False,
        "raw_kinect_depth_used_at_inference": False,
        "facial_detail_target_applicable": False,
        "face_detail_claim_allowed": False,
        **{f"control_{k}_score": v for k, v in control_scores.items()},
        **metric,
    }


def load_config(case: str, config: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    path = OUT_ROOT / case / config / "predictions.npz" if config == TRUE_CONFIG else BASE_MATRIX / case / config / "predictions.npz"
    pred = load_npz(path)
    body = np.asarray(pred["body_part_id"]) if "body_part_id" in pred else None
    return np.asarray(pred["human_points"], dtype=np.float32), as_rgb(pred["human_rgb"]), body


def compare(rows: list[dict[str, str]], manifest: list[dict[str, Any]], created_at: str) -> None:
    metric_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in rows:
        case = row["case"]
        case_metrics: dict[str, dict[str, Any]] = {}
        for config in [TRUE_CONFIG, *CONTROL_CONFIGS]:
            pts, _rgb, body = load_config(case, config)
            met = anti_billboard_metric_v2(pts, body)
            case_metrics[config] = met
            metric_rows.append({"case": case, "config": config, **met})
        true = case_metrics[TRUE_CONFIG]
        ts = float(true["anti_billboard_score_v2"])
        if bool(true["billboard_fail_v2"]):
            failures.append({"case": case, "reason": "true_billboard_fail_v2", "true_score": ts})
        for cfg in CONTROL_CONFIGS:
            cs = float(case_metrics[cfg]["anti_billboard_score_v2"])
            if cs >= ts * 0.96:
                failures.append({"case": case, "reason": "control_close_or_better_v2", "control": cfg, "true_score": ts, "control_score": cs})
    write_csv(REPORTS / "V17700000000000000000_seed_metrics.csv", metric_rows)
    first = rows[0]["case"]
    panels = []
    for cfg in [TRUE_CONFIG, "real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control", "tiny_synthetic_token_control"]:
        pts, rgb, _body = load_config(first, cfg)
        panels.append(render_panel(pts, rgb, f"{first} {cfg.replace('_', ' ')}"))
    compose(panels, 3, BOARDS / "V17700000000000000000_front_back_occupancy_board.png")
    pts, rgb, _body = load_config(first, TRUE_CONFIG)
    turn = []
    for label, rot in [
        ("front", rotation_matrix(0, 0)),
        ("back", rotation_matrix(180, 0)),
        ("left", rotation_matrix(-90, 0)),
        ("right", rotation_matrix(90, 0)),
        ("oblique", rotation_matrix(-30, 61)),
    ]:
        turn.append(render_panel(pts, rgb, f"{first} true {label}", rot=rot))
    turn.append(cross_panel(pts, f"{first} true cross-section"))
    compose(turn, 3, BOARDS / "V17700000000000000000_turntable_cross_section.png")
    status = "V17700_FRONT_BACK_OCCUPANCY_FAIL_CLOSED_CONTINUE" if failures else "V17700_FRONT_BACK_OCCUPANCY_PRECHECK_PASS_REQUIRES_MENTOR_VISUAL"
    write_json(
        REPORTS / "V17700000000000000000_training_decision.json",
        {
            "created_at": created_at,
            "status": status,
            "mentor_ready": False,
            "external_hard_block": False,
            "cases": [r["case"] for r in rows],
            "training_manifest_rows": len(manifest),
            "failures": failures,
            "board": str(BOARDS / "V17700000000000000000_front_back_occupancy_board.png"),
            "turntable_cross_section": str(BOARDS / "V17700000000000000000_turntable_cross_section.png"),
            "face_detail_claim_allowed": False,
            "allowed_face_claim": ALLOWED_FACE_CLAIM,
            "summary": "V177 explicitly targets thin-axis front/back occupancy. It remains fail-closed unless front/back separation, hard controls, and visual boards pass.",
        },
    )


def main() -> int:
    created_at = now()
    rows = read_manifest()
    device, runtime = select_device()
    steps = int(os.environ.get("V17700_STEPS", "90"))
    max_points = int(os.environ.get("V17700_MAX_POINTS", "4096"))
    manifest = [train_case(row, steps=steps, max_points=max_points, device=device) for row in rows]
    write_csv(REPORTS / "V17700000000000000000_training_manifest.csv", manifest)
    compare(rows, manifest, created_at)
    write_json(REPORTS / "V17700000000000000000_runtime_environment.json", {"created_at": created_at, **runtime, "steps": steps, "max_points": max_points})
    print(json.dumps({"created_at": created_at, "status": "V17700_DONE", "device": str(device), "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
