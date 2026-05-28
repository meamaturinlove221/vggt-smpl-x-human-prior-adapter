from __future__ import annotations

import csv
import json
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
from tools.V13300_anti_billboard_metric_v2 import anti_billboard_metric_v2  # noqa: E402
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
OUT_ROOT = OUTPUT / "V17500000000000000000_layer_balanced_multishell_training"
TRUE_CONFIG = "layer_balanced_multishell_true"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def as_rgb(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype != np.uint8:
        if out.size and np.issubdtype(out.dtype, np.number) and float(np.nanmax(out)) <= 1.5:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out[:, :3]


def layer_balanced_decode(
    out: dict[str, torch.Tensor],
    aux: dict[str, np.ndarray],
    targets: dict[str, torch.Tensor],
    *,
    decoded_fraction: float = 0.58,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_points = aux["human_points"]
    base_rgb = aux["human_rgb"]
    body = aux["body_part_id"]
    idx = aux["idx"]
    weak = targets["weak"][0, :, 0].detach().cpu().numpy()
    repair_selected = aux["repair"][idx] | (weak > 0.35)
    if not np.any(repair_selected):
        return base_points.copy(), base_rgb.copy(), body.copy().astype(np.int16)

    # Prefer actual shell surfaces over the center student point. This repairs
    # V173's bug: budget trimming kept too many early student/front copies and
    # dropped back/side layers, which depressed front_back_separation_ratio.
    shell_tensors = [
        out["front_shell"],
        out["back_shell"],
        out["left_shell"],
        out["right_shell"],
        out["student_xyz"],
    ]
    shell_names = ["front", "back", "left", "right", "center"]
    shells = [t[0].detach().cpu().numpy().astype(np.float32) for t in shell_tensors]
    priority = aux["priority"][idx] + repair_selected.astype(np.float32) * 3.0 + weak.astype(np.float32)
    candidate = np.where(repair_selected)[0]
    candidate = candidate[np.argsort(-priority[candidate])]

    target_n = len(base_points)
    decoded_budget = int(np.clip(round(target_n * decoded_fraction), len(candidate), target_n - 8000))
    keep_budget = target_n - decoded_budget

    keep_mask = np.ones(target_n, dtype=bool)
    keep_mask[idx[candidate]] = False
    keep_pool = np.where(keep_mask)[0]
    # Keep a deterministic spread of unedited baseline anchors; this preserves
    # the original high-confidence zones without letting them consume all point
    # budget.
    if len(keep_pool) > keep_budget:
        keep_pick = np.linspace(0, len(keep_pool) - 1, keep_budget).round().astype(int)
        keep_pool = keep_pool[keep_pick]

    per_layer = [decoded_budget // len(shells)] * len(shells)
    for i in range(decoded_budget % len(shells)):
        per_layer[i] += 1
    decoded_points: list[np.ndarray] = []
    decoded_rgb: list[np.ndarray] = []
    decoded_body: list[np.ndarray] = []
    decoded_layer: list[str] = []
    for layer_name, layer_points, quota in zip(shell_names, shells, per_layer, strict=True):
        if quota <= 0:
            continue
        reps = int(np.ceil(quota / max(1, len(candidate))))
        tiled = np.tile(candidate, reps)[:quota]
        decoded_points.append(layer_points[tiled])
        decoded_rgb.append(base_rgb[idx[tiled]])
        decoded_body.append(body[idx[tiled]])
        decoded_layer.extend([layer_name] * len(tiled))

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


def train_case(row: dict[str, str], steps: int, max_points: int, device: torch.device) -> dict[str, Any]:
    model = AntiBillboardTopologyVolumeStudent().to(device)
    batch, targets, aux = build_batch(row, max_points=max_points, device=device)
    control_scores = hard_control_scores(row["case"])
    opt = torch.optim.AdamW(model.parameters(), lr=9e-4, weight_decay=1e-4)
    history: list[dict[str, float | int]] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        losses = training_losses(out, targets, control_scores)
        losses["total"].backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in {0, steps - 1} or (step + 1) % 50 == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(losses["total"].detach().cpu()),
                    "shell_width": float(losses["shell_width"].detach().cpu()),
                    "adjacency": float(losses["adjacency"].detach().cpu()),
                    "volume": float(losses["volume"].detach().cpu()),
                    "adv_margin": float(losses["adv_margin"].detach().cpu()),
                    "grad_norm": float(grad),
                }
            )
    with torch.no_grad():
        out = model(batch)
    human, human_rgb, body = layer_balanced_decode(out, aux, targets)
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
        layer_balanced_multishell_decoded=np.array(True),
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
        "layer_balanced_multishell_decoded": True,
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
        for config in CONTROL_CONFIGS:
            cs = float(case_metrics[config]["anti_billboard_score_v2"])
            if cs >= ts * 0.96:
                failures.append({"case": case, "reason": "control_close_or_better_v2", "control": config, "true_score": ts, "control_score": cs})
    write_csv(REPORTS / "V17500000000000000000_seed_metrics.csv", metric_rows)
    first = rows[0]["case"]
    panels = []
    for config in [TRUE_CONFIG, "real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control", "tiny_synthetic_token_control"]:
        pts, rgb, _body = load_config(first, config)
        panels.append(render_panel(pts, rgb, f"{first} {config.replace('_', ' ')}"))
    compose(panels, 3, BOARDS / "V17500000000000000000_layer_balanced_board.png")
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
    compose(turn, 3, BOARDS / "V17500000000000000000_turntable_cross_section.png")
    status = "V17500_LAYER_BALANCED_MULTISHELL_FAIL_CLOSED_CONTINUE" if failures else "V17500_LAYER_BALANCED_PRECHECK_PASS_REQUIRES_MENTOR_VISUAL"
    write_json(
        REPORTS / "V17500000000000000000_training_decision.json",
        {
            "created_at": created_at,
            "status": status,
            "mentor_ready": False,
            "external_hard_block": False,
            "cases": [r["case"] for r in rows],
            "training_manifest_rows": len(manifest),
            "failures": failures,
            "board": str(BOARDS / "V17500000000000000000_layer_balanced_board.png"),
            "turntable_cross_section": str(BOARDS / "V17500000000000000000_turntable_cross_section.png"),
            "face_detail_claim_allowed": False,
            "allowed_face_claim": ALLOWED_FACE_CLAIM,
            "summary": "V175 repairs V173 budget trimming by enforcing layer-balanced shell selection. It remains fail-closed unless true beats controls and billboard_fail_v2 is false.",
        },
    )


def main() -> int:
    created_at = now()
    rows = read_manifest()
    device, runtime = select_device()
    steps = int(os.environ.get("V17500_STEPS", "80"))
    max_points = int(os.environ.get("V17500_MAX_POINTS", "4096"))
    manifest = [train_case(row, steps=steps, max_points=max_points, device=device) for row in rows]
    write_csv(REPORTS / "V17500000000000000000_training_manifest.csv", manifest)
    compare(rows, manifest, created_at)
    write_json(REPORTS / "V17500000000000000000_runtime_environment.json", {"created_at": created_at, **runtime, "steps": steps, "max_points": max_points})
    print(json.dumps({"created_at": created_at, "status": "V17500_DONE", "device": str(device), "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
