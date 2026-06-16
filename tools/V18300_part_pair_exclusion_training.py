from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO = Path(os.environ.get("VGGT_REPO_ROOT", r"D:\vggt\vggt-canonical-surfel-adapter"))
sys.path.insert(0, str(REPO))
OUTPUT = REPO / "output"
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
OUT_ROOT = OUTPUT / "V18300000000000000000_part_pair_exclusion_training"
BASE_MATRIX = OUTPUT / "V10700000000000000000_volume_aware_training_matrix"
V173_ROOT = OUTPUT / "V17300000000000000000_multishell_topology_decoder_training"
V181_ROOT = OUTPUT / "V18100000000000000000_adjacency_topology_training"
TRUE_CONFIG = "part_pair_exclusion_true"
V173_CONFIG = "multishell_topology_decoder_true"
V181_CONFIG = "adjacency_topology_true"
CONTROL_CONFIGS = [
    "real_vggt_baseline_only",
    "same_topology_no_semantic",
    "shuffled_smpl_feature",
    "thickness_only_control",
    "posthoc_surfel_only",
    "tiny_synthetic_token_control",
]
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"

from models.v182_part_pair_exclusion_topology_student import (  # noqa: E402
    PART_PAIR_EXCLUSION_PAIRS,
    VALID_CONTACT_PAIRS,
    PartPairExclusionConfig,
    PartPairExclusionTopologyStudent,
)
from tools.V17300_multishell_topology_decoder_training import (  # noqa: E402
    as_rgb,
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
    write_ply,
)
from tools.V18000_adjacency_aware_collision_metric import adjacency_collision_metric_v4  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure(path.parent)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["case"])
        writer.writeheader()
        writer.writerows(rows)


def pair_center_losses(points: torch.Tensor, body_part: torch.Tensor) -> dict[str, torch.Tensor]:
    centers: dict[int, torch.Tensor] = {}
    spreads: dict[int, torch.Tensor] = {}
    for part in [int(x) for x in torch.unique(body_part).detach().cpu().tolist()]:
        mask = body_part == part
        if int(mask.sum()) < 20:
            continue
        pts = points[mask]
        centers[part] = pts.mean(dim=0)
        spreads[part] = torch.sqrt(torch.var(pts, dim=0, unbiased=False) + 1e-6)
    invalid_terms: list[torch.Tensor] = []
    contact_terms: list[torch.Tensor] = []
    order_terms: list[torch.Tensor] = []
    for a, b in PART_PAIR_EXCLUSION_PAIRS:
        if a not in centers or b not in centers:
            continue
        dist = torch.norm(centers[a] - centers[b])
        scale = 0.5 * (torch.norm(spreads[a]) + torch.norm(spreads[b])) + 1e-6
        invalid_terms.append(torch.relu(2.65 - dist / scale))
    for a, b in VALID_CONTACT_PAIRS:
        if a not in centers or b not in centers:
            continue
        dist = torch.norm(centers[a] - centers[b])
        scale = 0.5 * (torch.norm(spreads[a]) + torch.norm(spreads[b])) + 1e-6
        norm = dist / scale
        contact_terms.append(torch.relu(norm - 2.25) + torch.relu(0.55 - norm))
    for a, b in [(0, 1), (1, 4), (4, 6), (1, 5), (5, 7)]:
        if a in centers and b in centers:
            order_terms.append(torch.relu(0.015 - (centers[b][0] - centers[a][0])))
    zero = torch.zeros((), device=points.device)
    return {
        "invalid_pair_center_loss": torch.stack(invalid_terms).mean() if invalid_terms else zero,
        "valid_contact_center_loss": torch.stack(contact_terms).mean() if contact_terms else zero,
        "semantic_order_loss": torch.stack(order_terms).mean() if order_terms else zero,
    }


def part_pair_training_losses(
    out: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    control_scores: dict[str, float],
    body_part: torch.Tensor,
) -> dict[str, torch.Tensor]:
    # Reuse shell/projection-preservation losses where names match the previous student.
    losses = training_losses(out, targets, control_scores)
    body = body_part[0].long()
    shells = torch.stack(
        [out["student_xyz"], out["front_shell"], out["back_shell"], out["left_shell"], out["right_shell"]],
        dim=2,
    )[0]
    points = shells.reshape(-1, 3)
    shell_body = body.repeat_interleave(shells.shape[1])
    center_losses = pair_center_losses(points, shell_body)
    part_target = F.one_hot(body.clamp(0, 7), num_classes=8).float()
    part_gate_loss = F.binary_cross_entropy_with_logits(out["part_gate_logits"][0], part_target)
    exclusion_target = torch.ones_like(out["pair_exclusion"])
    contact_target = torch.ones_like(out["valid_contact"])
    exclusion_loss = F.binary_cross_entropy_with_logits(out["pair_exclusion_logits"], exclusion_target)
    contact_loss = F.binary_cross_entropy_with_logits(out["valid_contact_logits"], contact_target)
    offset_reg = out["selected_part_offset"].square().mean()
    losses.update(center_losses)
    losses.update(
        {
            "part_gate_loss": part_gate_loss,
            "pair_exclusion_head_loss": exclusion_loss,
            "valid_contact_head_loss": contact_loss,
            "part_offset_regularization": offset_reg,
        }
    )
    losses["total"] = (
        losses["total"]
        + 0.72 * center_losses["invalid_pair_center_loss"]
        + 0.32 * center_losses["valid_contact_center_loss"]
        + 0.30 * center_losses["semantic_order_loss"]
        + 0.18 * part_gate_loss
        + 0.10 * exclusion_loss
        + 0.08 * contact_loss
        + 0.35 * offset_reg
    )
    return losses


def decode_part_pair(out: dict[str, torch.Tensor], aux: dict[str, np.ndarray], targets: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_points = aux["human_points"]
    base_rgb = aux["human_rgb"]
    body = aux["body_part_id"]
    idx = aux["idx"]
    repair = aux["repair"][idx]
    weak = targets["weak"][0, :, 0].detach().cpu().numpy()
    shells = torch.stack(
        [out["student_xyz"], out["front_shell"], out["back_shell"], out["left_shell"], out["right_shell"]],
        dim=2,
    )[0].detach().cpu().numpy()
    part_gate = out["part_gate"][0].detach().cpu().numpy()
    selected_gate = part_gate[np.arange(len(idx)), np.clip(body[idx], 0, 7)]
    selected_replace = repair | (weak > 0.30) | (selected_gate > 0.58)
    keep = np.ones(len(base_points), dtype=bool)
    keep[idx[selected_replace]] = False
    shell_rgb = np.repeat(base_rgb[idx][:, None, :], shells.shape[1], axis=1)
    decoded_points = shells[selected_replace].reshape(-1, 3)
    decoded_rgb = shell_rgb[selected_replace].reshape(-1, 3)
    decoded_body = np.repeat(body[idx[selected_replace]], shells.shape[1])
    merged_points = np.concatenate([base_points[keep], decoded_points], axis=0)
    merged_rgb = np.concatenate([base_rgb[keep], decoded_rgb], axis=0)
    merged_body = np.concatenate([body[keep], decoded_body], axis=0)
    if len(merged_points) < len(base_points):
        add = len(base_points) - len(merged_points)
        order = np.argsort(-aux["priority"])[:add]
        merged_points = np.concatenate([merged_points, base_points[order]], axis=0)
        merged_rgb = np.concatenate([merged_rgb, base_rgb[order]], axis=0)
        merged_body = np.concatenate([merged_body, body[order]], axis=0)
    elif len(merged_points) > len(base_points):
        priority = np.concatenate(
            [
                np.zeros(int(keep.sum()), dtype=np.float32) + 0.45,
                np.repeat(aux["priority"][idx[selected_replace]] + selected_gate[selected_replace], shells.shape[1]),
            ],
            axis=0,
        )
        order = np.argsort(-priority)[: len(base_points)]
        merged_points = merged_points[order]
        merged_rgb = merged_rgb[order]
        merged_body = merged_body[order]
    return merged_points.astype(np.float32), as_rgb(merged_rgb), merged_body.astype(np.int16)


def train_case(row: dict[str, str], steps: int, max_points: int, device: torch.device) -> dict[str, Any]:
    model = PartPairExclusionTopologyStudent().to(device)
    batch, targets, aux = build_batch(row, max_points=max_points, device=device)
    control_scores = hard_control_scores(row["case"])
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
    history: list[dict[str, float | int]] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        losses = part_pair_training_losses(out, targets, control_scores, batch["body_part_id"])
        losses["total"].backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in {0, steps - 1} or (step + 1) % 50 == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(losses["total"].detach().cpu()),
                    "invalid_pair_center": float(losses["invalid_pair_center_loss"].detach().cpu()),
                    "valid_contact_center": float(losses["valid_contact_center_loss"].detach().cpu()),
                    "semantic_order": float(losses["semantic_order_loss"].detach().cpu()),
                    "part_gate": float(losses["part_gate_loss"].detach().cpu()),
                    "offset_reg": float(losses["part_offset_regularization"].detach().cpu()),
                    "grad_norm": float(grad),
                }
            )
    with torch.no_grad():
        out = model(batch)
    human, human_rgb, body = decode_part_pair(out, aux, targets)
    env = aux["environment_points"]
    env_rgb = aux["environment_rgb"]
    full = np.concatenate([human, env], axis=0)
    full_rgb = np.concatenate([human_rgb, env_rgb], axis=0)
    out_dir = ensure(OUT_ROOT / row["case"] / TRUE_CONFIG)
    np.savez_compressed(
        out_dir / "predictions.npz",
        human_points=human,
        human_rgb=human_rgb,
        environment_points=env,
        environment_rgb=env_rgb,
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
        part_pair_exclusion_decoder=np.array(True),
    )
    write_ply(out_dir / "full_scene_rgb_pointcloud.ply", full, full_rgb)
    metric, _pairs = adjacency_collision_metric_v4(human, body)
    return {
        "case": row["case"],
        "config": TRUE_CONFIG,
        "prediction": str(out_dir / "predictions.npz"),
        "ply": str(out_dir / "full_scene_rgb_pointcloud.ply"),
        "steps": steps,
        "max_points": max_points,
        "device": str(device),
        "history_json": json.dumps(history),
        "model_owned_student_output": True,
        "teacher_points_used_at_inference": False,
        "raw_kinect_depth_used_at_inference": False,
        "facial_detail_target_applicable": False,
        "face_detail_claim_allowed": False,
        **{f"control_{k}_score": v for k, v in control_scores.items()},
        **metric,
    }


def config_path(case: str, config: str) -> Path:
    if config == TRUE_CONFIG:
        return OUT_ROOT / case / config / "predictions.npz"
    if config == V173_CONFIG:
        return V173_ROOT / case / config / "predictions.npz"
    if config == V181_CONFIG:
        return V181_ROOT / case / config / "predictions.npz"
    return BASE_MATRIX / case / config / "predictions.npz"


def load_config(case: str, config: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    pred = load_npz(config_path(case, config))
    body = np.asarray(pred["body_part_id"]) if "body_part_id" in pred else None
    return np.asarray(pred["human_points"], dtype=np.float32), as_rgb(pred["human_rgb"]), body


def compare(rows: list[dict[str, str]], created_at: str) -> None:
    score_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    configs = [TRUE_CONFIG, V173_CONFIG, V181_CONFIG, *CONTROL_CONFIGS]
    for row in rows:
        case = row["case"]
        case_scores: dict[str, dict[str, Any]] = {}
        for config in configs:
            path = config_path(case, config)
            if not path.exists():
                continue
            pts, _rgb, body = load_config(case, config)
            metric, _pairs = adjacency_collision_metric_v4(pts, body)
            score_rows.append({"case": case, "config": config, **metric})
            case_scores[config] = metric
        true = case_scores.get(TRUE_CONFIG)
        if true is None:
            failures.append({"case": case, "reason": "missing_part_pair_true"})
            continue
        true_score = float(true["combined_topology_volume_score_v4"])
        if bool(true["combined_fail_v4"]):
            failures.append({"case": case, "reason": "true_combined_fail_v4", "true_score": true_score})
        if float(true["invalid_overlap_penalty"]) > 0.58:
            failures.append({"case": case, "reason": "invalid_distant_overlap_too_high", "penalty": float(true["invalid_overlap_penalty"])})
        for config in [V173_CONFIG, V181_CONFIG, "real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control"]:
            if config not in case_scores:
                continue
            score = float(case_scores[config]["combined_topology_volume_score_v4"])
            if score >= true_score * 0.96:
                failures.append({"case": case, "reason": "control_or_prior_close_or_better_v4", "control": config, "true_score": true_score, "control_score": score})
    write_csv(REPORTS / "V18300000000000000000_part_pair_scores.csv", score_rows)
    first = rows[0]["case"] if rows else ""
    panels: list[Image.Image] = []
    for config in [TRUE_CONFIG, V173_CONFIG, V181_CONFIG, "real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature"]:
        path = config_path(first, config)
        if path.exists():
            pts, rgb, _body = load_config(first, config)
            panels.append(render_panel(pts, rgb, f"{first} {config.replace('_', ' ')}"))
    if panels:
        compose(panels, 3, BOARDS / "V18300000000000000000_part_pair_exclusion_board.png")
    turn: list[Image.Image] = []
    if first and config_path(first, TRUE_CONFIG).exists():
        pts, rgb, _body = load_config(first, TRUE_CONFIG)
        for label, rot in [
            ("front", rotation_matrix(0, 0)),
            ("back", rotation_matrix(180, 0)),
            ("left", rotation_matrix(-90, 0)),
            ("right", rotation_matrix(90, 0)),
            ("oblique", rotation_matrix(-30, 61)),
        ]:
            turn.append(render_panel(pts, rgb, f"{first} part-pair true {label}", rot=rot))
        turn.append(cross_panel(pts, f"{first} part-pair true cross-section"))
        compose(turn, 3, BOARDS / "V18300000000000000000_part_pair_turntable_cross_section.png")
    decision = {
        "created_at": created_at,
        "status": "V18300_PART_PAIR_EXCLUSION_TRAINING_FAIL_CLOSED_CONTINUE" if failures else "V18300_PART_PAIR_PRECHECK_PASS_REQUIRES_MENTOR_VISUAL",
        "mentor_ready": False,
        "external_hard_block": False,
        "cases": [r["case"] for r in rows],
        "failures": failures,
        "score_csv": str(REPORTS / "V18300000000000000000_part_pair_scores.csv"),
        "board": str(BOARDS / "V18300000000000000000_part_pair_exclusion_board.png"),
        "turntable_cross_section": str(BOARDS / "V18300000000000000000_part_pair_turntable_cross_section.png"),
        "face_detail_claim_allowed": False,
        "allowed_face_claim": ALLOWED_FACE_CLAIM,
        "summary": "V183 uses a part-pair exclusion decoder with semantic graph occupancy heads. It remains fail-closed unless it beats V173/V181 and mentor visual gates.",
    }
    write_json(REPORTS / "V18300000000000000000_training_decision.json", decision)
    route = f"""# V183 Part-Pair Exclusion Decoder Route

Created: {created_at}

Status: `{decision["status"]}`

This is a trained representation attempt, not a viewer or thickness fix. It remains diagnostic until full-scene mentor visual gates pass.

Failure count: `{len(failures)}`

If this still fails, the next repair must move from point-anchor shell decoding to canonical SMPL-X graph/surfel occupancy with explicit body-part topology constraints across the full mesh support.
"""
    (REPORTS / "V18300000000000000000_route_state.md").write_text(route, encoding="utf-8")


def main() -> int:
    created_at = now()
    rows = read_manifest()
    device, runtime = select_device()
    steps = int(os.environ.get("V18300_STEPS", "40"))
    max_points = int(os.environ.get("V18300_MAX_POINTS", "4096"))
    manifest = [train_case(row, steps=steps, max_points=max_points, device=device) for row in rows]
    write_csv(REPORTS / "V18300000000000000000_training_manifest.csv", manifest)
    compare(rows, created_at)
    write_json(
        REPORTS / "V18300000000000000000_runtime_environment.json",
        {"created_at": created_at, **runtime, "steps": steps, "max_points": max_points},
    )
    print(json.dumps({"created_at": created_at, "status": "V18300_DONE", "device": str(device), "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
