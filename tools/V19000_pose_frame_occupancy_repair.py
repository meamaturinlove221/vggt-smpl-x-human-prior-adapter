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
OUT_ROOT = OUTPUT / "V19000000000000000000_pose_frame_occupancy_repair"
TRUE_CONFIG = "pose_frame_occupancy_true"
V187_CONFIG = "visible_anchor_canonical_surfel_true"
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"
PART_COUNT = 8

from models.v184_canonical_surfel_graph_occupancy_student import (  # noqa: E402
    CanonicalSurfelGraphOccupancyStudent,
)
from tools.V17300_multishell_topology_decoder_training import (  # noqa: E402
    as_rgb,
    compose,
    cross_panel,
    load_npz,
    read_manifest,
    render_panel,
    rotation_matrix,
    select_device,
    write_ply,
)
from tools.V18000_adjacency_aware_collision_metric import adjacency_collision_metric_v4  # noqa: E402
from tools.V18700_visible_anchor_canonical_surfel_training import (  # noqa: E402
    CONTROL_CONFIGS,
    V186_CONFIG,
    build_batch,
    graph_losses,
)


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


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-6)


def part_radius(body: torch.Tensor) -> torch.Tensor:
    values = torch.tensor(
        [0.035, 0.058, 0.032, 0.032, 0.046, 0.046, 0.026, 0.026],
        dtype=torch.float32,
        device=body.device,
    )
    return values[body.long().clamp(0, PART_COUNT - 1)].unsqueeze(-1)


def pose_frame_losses(out: dict[str, torch.Tensor], target: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    base = graph_losses(out, target)
    xyz = batch["surfel_xyz"].float()
    normal = _normalize(batch["surfel_normal"].float())
    tangent = _normalize(batch["surfel_tangent"].float())
    binormal = _normalize(torch.cross(normal, tangent, dim=-1))
    radius = part_radius(target["body_part"]) * (1.0 + 0.35 * target["weak"].clamp(0, 1))
    gate = torch.maximum(target["weak"].clamp(0, 1), target["anchor_affinity"].clamp(0, 1) * 0.55)
    target_front = xyz + normal * radius
    target_back = xyz - normal * radius
    target_left = xyz + binormal * radius * 0.62
    target_right = xyz - binormal * radius * 0.62
    front = ((out["front_shell"] - target_front).square() * gate).mean()
    back = ((out["back_shell"] - target_back).square() * gate).mean()
    width = (out["front_shell"] - out["back_shell"]).norm(dim=-1, keepdim=True)
    width_target = (target_front - target_back).norm(dim=-1, keepdim=True)
    width_loss = ((width - width_target).square() * gate).mean()
    side_width = (target_left - target_right).norm(dim=-1, keepdim=True)
    local_width_floor = F.relu(side_width * 0.85 - out["thickness"].sum(dim=-1, keepdim=True)) * gate
    normal_align = (1.0 - (out["normal"] * normal).sum(dim=-1, keepdim=True).clamp(-1, 1)).mul(gate).mean()
    occ = out["occupancy"]
    completeness = F.relu(0.48 - (occ * gate).mean())
    part_terms: list[torch.Tensor] = []
    body = target["body_part"].long().clamp(0, PART_COUNT - 1)
    for part in range(PART_COUNT):
        mask = body == part
        if bool(mask.any()):
            part_terms.append(F.relu(0.36 - occ[mask.unsqueeze(-1)].mean()))
    part_occupancy = torch.stack(part_terms).mean() if part_terms else occ.mean() * 0.0
    no_float = (occ * F.relu(target["anchor_distance"] - 1.05)).mean()
    pose_total = (
        1.65 * (front + back)
        + 0.90 * width_loss
        + 0.55 * local_width_floor.mean()
        + 0.30 * normal_align
        + 0.46 * completeness
        + 0.36 * part_occupancy
        + 0.34 * no_float
    )
    total = 0.45 * base["total"] + pose_total
    return {
        "total": total,
        "base_graph": base["total"],
        "pose_front": front,
        "pose_back": back,
        "pose_width": width_loss,
        "side_floor": local_width_floor.mean(),
        "normal_align": normal_align,
        "completeness": completeness,
        "part_occupancy": part_occupancy,
        "floating": no_float,
        "base_shell": base["shell"],
        "base_coverage": base["coverage"],
    }


def decode_pose_frame(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], target: dict[str, torch.Tensor], aux: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    occ = out["occupancy"][0, :, 0].detach().cpu().numpy()
    center = out["student_xyz"][0].detach().cpu().numpy()
    front = out["front_shell"][0].detach().cpu().numpy()
    back = out["back_shell"][0].detach().cpu().numpy()
    tangent = batch["surfel_tangent"][0].detach().cpu().numpy()
    body_sel = aux["body"][aux["idx"]]
    rgb_sel = aux["rgb"][aux["idx"]]
    radius = part_radius(target["body_part"])[0, :, 0].detach().cpu().numpy()
    left = center + tangent * radius[:, None] * 0.62
    right = center - tangent * radius[:, None] * 0.62
    anchor_sel = aux["anchor_affinity"][aux["idx"]]
    score = 0.62 * occ + 0.38 * anchor_sel
    keep = np.zeros(len(score), dtype=bool)
    per_part = max(256, int(np.ceil(60000 / (PART_COUNT * 5))))
    for part in range(PART_COUNT):
        ids = np.flatnonzero(body_sel == part)
        if len(ids) == 0:
            continue
        take = min(len(ids), per_part)
        chosen = ids[np.argsort(-score[ids])[:take]]
        keep[chosen] = True
    keep |= score >= np.quantile(score, 0.43)
    shells = np.stack([center, front, back, left, right], axis=1)
    points = shells[keep].reshape(-1, 3)
    colors = np.repeat(rgb_sel[keep][:, None, :], shells.shape[1], axis=1).reshape(-1, 3)
    body = np.repeat(body_sel[keep], shells.shape[1])
    target_n = 60000
    if len(points) == 0:
        order = np.argsort(-aux["priority"])[:target_n]
        return aux["surfel_xyz"][order].astype(np.float32), aux["rgb"][order], aux["body"][order].astype(np.int16)
    if len(points) < target_n:
        reps = int(np.ceil(target_n / len(points)))
        points = np.tile(points, (reps, 1))[:target_n]
        colors = np.tile(colors, (reps, 1))[:target_n]
        body = np.tile(body, reps)[:target_n]
    elif len(points) > target_n:
        order = np.linspace(0, len(points) - 1, target_n).astype(int)
        points, colors, body = points[order], colors[order], body[order]
    return points.astype(np.float32), as_rgb(colors), body.astype(np.int16)


def train_case(row: dict[str, str], steps: int, max_points: int, device: torch.device) -> dict[str, Any]:
    model = CanonicalSurfelGraphOccupancyStudent().to(device)
    batch, target, aux = build_batch(row, max_points=max_points, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
    history: list[dict[str, float | int]] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        losses = pose_frame_losses(out, target, batch)
        losses["total"].backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in {0, steps - 1} or (step + 1) % 50 == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(losses["total"].detach().cpu()),
                    "base_graph": float(losses["base_graph"].detach().cpu()),
                    "pose_front": float(losses["pose_front"].detach().cpu()),
                    "pose_back": float(losses["pose_back"].detach().cpu()),
                    "pose_width": float(losses["pose_width"].detach().cpu()),
                    "side_floor": float(losses["side_floor"].detach().cpu()),
                    "normal_align": float(losses["normal_align"].detach().cpu()),
                    "completeness": float(losses["completeness"].detach().cpu()),
                    "part_occupancy": float(losses["part_occupancy"].detach().cpu()),
                    "floating": float(losses["floating"].detach().cpu()),
                    "grad_norm": float(grad),
                }
            )
    with torch.no_grad():
        out = model(batch)
    human, human_rgb, body = decode_pose_frame(out, batch, target, aux)
    env, env_rgb = aux["environment_points"], aux["environment_rgb"]
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
        pose_frame_occupancy_repair=np.array(True),
        feature_bank_fallback_used=np.array(bool(aux["feature_bank_fallback_used"])),
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
        "feature_bank_fallback_used": bool(aux["feature_bank_fallback_used"]),
        **metric,
    }


def config_path(case: str, config: str) -> Path:
    if config == TRUE_CONFIG:
        return OUT_ROOT / case / config / "predictions.npz"
    if config == V187_CONFIG:
        return OUTPUT / "V18700000000000000000_visible_anchor_canonical_surfel_training" / case / config / "predictions.npz"
    if config == V186_CONFIG:
        return OUTPUT / "V18600000000000000000_part_coverage_canonical_surfel_training" / case / config / "predictions.npz"
    return OUTPUT / "V1400000000000000000_learned_residual_matrix" / case / config / "predictions.npz"


def load_config(case: str, config: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    pred = load_npz(config_path(case, config))
    body = np.asarray(pred["body_part_id"]) if "body_part_id" in pred else None
    return np.asarray(pred["human_points"], dtype=np.float32), as_rgb(pred["human_rgb"]), body


def compare(rows: list[dict[str, str]], created_at: str) -> None:
    score_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    configs = [TRUE_CONFIG, V187_CONFIG, V186_CONFIG, "real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control"]
    for row in rows:
        case = row["case"]
        case_scores: dict[str, dict[str, Any]] = {}
        for cfg in configs:
            path = config_path(case, cfg)
            if not path.exists():
                continue
            pts, _rgb, body = load_config(case, cfg)
            metric, _pairs = adjacency_collision_metric_v4(pts, body)
            score_rows.append({"case": case, "config": cfg, **metric})
            case_scores[cfg] = metric
        true = case_scores.get(TRUE_CONFIG)
        if true is None:
            failures.append({"case": case, "reason": "missing_pose_frame_true"})
            continue
        true_score = float(true["combined_topology_volume_score_v4"])
        if bool(true["combined_fail_v4"]):
            failures.append({"case": case, "reason": "true_combined_fail_v4", "true_score": true_score})
        for cfg in [V187_CONFIG, V186_CONFIG, "real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control"]:
            if cfg in case_scores and float(case_scores[cfg]["combined_topology_volume_score_v4"]) >= true_score * 0.96:
                failures.append(
                    {
                        "case": case,
                        "reason": "control_or_prior_close_or_better_v4",
                        "control": cfg,
                        "true_score": true_score,
                        "control_score": float(case_scores[cfg]["combined_topology_volume_score_v4"]),
                    }
                )
    write_csv(REPORTS / "V19000000000000000000_pose_frame_scores.csv", score_rows)
    first = rows[0]["case"] if rows else ""
    panels: list[Image.Image] = []
    for cfg in configs:
        path = config_path(first, cfg)
        if path.exists():
            pts, rgb, _body = load_config(first, cfg)
            panels.append(render_panel(pts, rgb, f"{first} {cfg.replace('_', ' ')}"))
    if panels:
        compose(panels, 3, BOARDS / "V19000000000000000000_pose_frame_board.png")
    if first and config_path(first, TRUE_CONFIG).exists():
        pts, rgb, _body = load_config(first, TRUE_CONFIG)
        panels = [
            render_panel(pts, rgb, f"{first} pose-frame {label}", rot=rot)
            for label, rot in [
                ("front", rotation_matrix(0, 0)),
                ("back", rotation_matrix(180, 0)),
                ("left", rotation_matrix(-90, 0)),
                ("right", rotation_matrix(90, 0)),
                ("oblique", rotation_matrix(-30, 61)),
            ]
        ]
        panels.append(cross_panel(pts, f"{first} pose-frame cross-section"))
        compose(panels, 3, BOARDS / "V19000000000000000000_pose_frame_turntable_cross_section.png")
    decision = {
        "created_at": created_at,
        "status": "V19000_POSE_FRAME_OCCUPANCY_FAIL_CLOSED_CONTINUE" if failures else "V19000_POSE_FRAME_PRECHECK_PASS_REQUIRES_MENTOR_VISUAL",
        "mentor_ready": False,
        "external_hard_block": False,
        "failures": failures,
        "score_csv": str(REPORTS / "V19000000000000000000_pose_frame_scores.csv"),
        "board": str(BOARDS / "V19000000000000000000_pose_frame_board.png"),
        "turntable_cross_section": str(BOARDS / "V19000000000000000000_pose_frame_turntable_cross_section.png"),
        "face_detail_claim_allowed": False,
        "allowed_face_claim": ALLOWED_FACE_CLAIM,
        "summary": "V190 makes pose-frame front/back shell supervision active and decodes center/front/back/side shells. It remains fail-closed unless full-scene mentor visuals and hard controls pass.",
    }
    write_json(REPORTS / "V19000000000000000000_pose_frame_decision.json", decision)


def main() -> int:
    created_at = now()
    rows = read_manifest()
    device, runtime = select_device()
    steps = int(os.environ.get("V19000_STEPS", "40"))
    max_points = int(os.environ.get("V19000_MAX_POINTS", "8192"))
    manifest = [train_case(row, steps=steps, max_points=max_points, device=device) for row in rows]
    write_csv(REPORTS / "V19000000000000000000_pose_frame_training_manifest.csv", manifest)
    compare(rows, created_at)
    write_json(
        REPORTS / "V19000000000000000000_runtime_environment.json",
        {"created_at": created_at, **runtime, "steps": steps, "max_points": max_points},
    )
    print(json.dumps({"created_at": created_at, "status": "V19000_DONE", "device": str(device), "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
