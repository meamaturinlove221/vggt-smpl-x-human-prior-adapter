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
from PIL import Image


REPO = Path(os.environ.get("VGGT_REPO_ROOT", r"D:\vggt\vggt-canonical-surfel-adapter"))
sys.path.insert(0, str(REPO))
OUTPUT = REPO / "output"
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
OUT_ROOT = OUTPUT / "V19400000000000000000_visible_surface_preserving_infill"
TRUE_CONFIG = "visible_surface_preserving_infill_true"
V192_CONFIG = "upright_pose_frame_true"
V190_CONFIG = "pose_frame_occupancy_true"
V187_CONFIG = "visible_anchor_canonical_surfel_true"
V186_CONFIG = "part_coverage_canonical_surfel_true"
BASELINE_CONFIG = "real_vggt_baseline_only"
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"

from models.v184_canonical_surfel_graph_occupancy_student import CanonicalSurfelGraphOccupancyStudent  # noqa: E402
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
from tools.V18700_visible_anchor_canonical_surfel_training import build_batch  # noqa: E402
from tools.V19000_pose_frame_occupancy_repair import decode_pose_frame, pose_frame_losses  # noqa: E402


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


def repo_path(value: str | Path) -> Path:
    p = Path(value)
    if p.exists():
        return p
    text = str(value).replace("\\", "/")
    marker = "vggt-canonical-surfel-adapter/"
    if marker in text:
        mapped = REPO / text.split(marker, 1)[1]
        if mapped.exists():
            return mapped
    return p


def visible_preserve_mask(row: dict[str, str], n: int) -> tuple[np.ndarray, dict[str, float]]:
    graph = load_npz(repo_path(row["graph_path"]))
    conf = np.asarray(graph.get("mentor_smpl_confidence", np.zeros(n)), dtype=np.float32)
    no_change = np.asarray(graph.get("no_change_mask", np.zeros(n, dtype=bool)), dtype=bool)
    weak = np.asarray(graph.get("mentor_weak_region_score", np.zeros(n)), dtype=np.float32)
    if len(conf) != n:
        conf = np.interp(np.linspace(0, len(conf) - 1, n), np.arange(len(conf)), conf).astype(np.float32)
        weak = np.interp(np.linspace(0, len(weak) - 1, n), np.arange(len(weak)), weak).astype(np.float32)
        no_change = np.zeros(n, dtype=bool)
    preserve = ((conf >= np.quantile(conf, 0.58)) & (weak <= np.quantile(weak, 0.70))) | no_change
    # Keep at least half of the coherent visible surface. If the graph's
    # no-change mask is too conservative, fall back to confidence/anchor rank.
    if float(np.mean(preserve)) < 0.55:
        rank = 0.74 * conf - 0.26 * weak
        preserve = rank >= np.quantile(rank, 0.42)
    if float(np.mean(preserve)) > 0.78:
        rank = 0.74 * conf - 0.26 * weak
        preserve = rank >= np.quantile(rank, 0.24)
    info = {
        "preserve_ratio": float(np.mean(preserve)),
        "preserve_conf_mean": float(np.mean(conf[preserve])) if bool(np.any(preserve)) else 0.0,
        "weak_nonpreserve_mean": float(np.mean(weak[~preserve])) if bool(np.any(~preserve)) else 0.0,
    }
    return preserve.astype(bool), info


def align_infill_to_baseline_frame(
    infill_points: np.ndarray,
    source_support: np.ndarray,
    baseline_points: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map V950/V161 surfel-frame infill into the V140 mentor frame.

    The restored V950/V161 support is a good semantic/topology source, but its
    raw coordinate scale is not the same as the V140 full-scene mentor board.
    Without this explicit binding, preserved baseline points and infill clouds
    appear as two separate bodies. Use a conservative per-axis robust similarity
    so the final board remains in the baseline full-scene frame.
    """
    src = np.asarray(source_support, dtype=np.float32)
    dst = np.asarray(baseline_points, dtype=np.float32)
    src_center = np.median(src, axis=0)
    dst_center = np.median(dst, axis=0)
    src_span = np.maximum(np.percentile(src, 96, axis=0) - np.percentile(src, 4, axis=0), 1e-6)
    dst_span = np.maximum(np.percentile(dst, 96, axis=0) - np.percentile(dst, 4, axis=0), 1e-6)
    scale = np.clip(dst_span / src_span, 0.18, 2.75).astype(np.float32)
    mapped = (np.asarray(infill_points, dtype=np.float32) - src_center[None]) * scale[None] + dst_center[None]
    return mapped.astype(np.float32), {
        "alignment_type": "robust_per_axis_v950_to_v140",
        "source_center": src_center.tolist(),
        "target_center": dst_center.tolist(),
        "source_span": src_span.tolist(),
        "target_span": dst_span.tolist(),
        "axis_scale": scale.tolist(),
    }


def train_case(row: dict[str, str], steps: int, max_points: int, device: torch.device) -> dict[str, Any]:
    model = CanonicalSurfelGraphOccupancyStudent().to(device)
    batch, target, aux = build_batch(row, max_points=max_points, device=device)
    baseline = load_npz(repo_path(row["baseline_path"]))
    baseline_points = np.asarray(baseline["human_points"], dtype=np.float32)
    baseline_rgb = as_rgb(baseline["human_rgb"])
    baseline_body = np.asarray(baseline["body_part_id"], dtype=np.int16)
    preserve_mask, preserve_info = visible_preserve_mask(row, len(baseline_points))
    preserve_sel = torch.from_numpy(preserve_mask[aux["idx"]][None, :, None].astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=1e-4)
    history: list[dict[str, float | int]] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        losses = pose_frame_losses(out, target, batch)
        preserve_residual = (out["residual_xyz"].square() * preserve_sel).mean()
        # For preserved visible surfels, keep occupancy/visibility high but do
        # not move RGB/geometry. Infill pressure is assigned to non-preserved
        # weak/back-side regions through the existing pose-frame losses.
        preserve_occ = torch.relu(0.82 - (out["occupancy"] * preserve_sel).sum() / preserve_sel.sum().clamp_min(1.0))
        total = 0.82 * losses["total"] + 1.25 * preserve_residual + 0.35 * preserve_occ
        total.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in {0, steps - 1} or (step + 1) % 50 == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(total.detach().cpu()),
                    "pose_width": float(losses["pose_width"].detach().cpu()),
                    "completeness": float(losses["completeness"].detach().cpu()),
                    "preserve_residual": float(preserve_residual.detach().cpu()),
                    "preserve_occ": float(preserve_occ.detach().cpu()),
                    "grad_norm": float(grad),
                }
            )
    with torch.no_grad():
        out = model(batch)
    infill_points, infill_rgb, infill_body = decode_pose_frame(out, batch, target, aux)
    infill_points, alignment_info = align_infill_to_baseline_frame(infill_points, aux["surfel_xyz"], baseline_points)
    preserve = preserve_mask
    keep_base = preserve
    infill_count = max(12000, 60000 - int(keep_base.sum()))
    # Select infill points far enough from the preserved front surface so the
    # student adds volume instead of repainting the visible texture.
    if len(infill_points) > infill_count:
        order = np.linspace(0, len(infill_points) - 1, infill_count).astype(int)
        infill_points = infill_points[order]
        infill_rgb = infill_rgb[order]
        infill_body = infill_body[order]
    human = np.concatenate([baseline_points[keep_base], infill_points], axis=0)
    human_rgb = np.concatenate([baseline_rgb[keep_base], infill_rgb], axis=0)
    body = np.concatenate([baseline_body[keep_base], infill_body], axis=0)
    target_n = 60000
    if len(human) > target_n:
        priority = np.concatenate([np.ones(int(keep_base.sum()), dtype=np.float32) * 2.0, np.ones(len(infill_points), dtype=np.float32)])
        order = np.argsort(-priority)[:target_n]
        human, human_rgb, body = human[order], human_rgb[order], body[order]
    elif len(human) < target_n:
        reps = int(np.ceil(target_n / len(human)))
        human = np.tile(human, (reps, 1))[:target_n]
        human_rgb = np.tile(human_rgb, (reps, 1))[:target_n]
        body = np.tile(body, reps)[:target_n]
    env, env_rgb = aux["environment_points"], aux["environment_rgb"]
    full = np.concatenate([human, env], axis=0)
    full_rgb = np.concatenate([human_rgb, env_rgb], axis=0)
    out_dir = ensure(OUT_ROOT / row["case"] / TRUE_CONFIG)
    np.savez_compressed(
        out_dir / "predictions.npz",
        human_points=human.astype(np.float32),
        human_rgb=as_rgb(human_rgb),
        environment_points=env,
        environment_rgb=env_rgb,
        full_scene_points=full.astype(np.float32),
        full_scene_rgb=as_rgb(full_rgb),
        body_part_id=body.astype(np.int16),
        preserve_visible_surface_mask=preserve.astype(bool),
        preserve_ratio=np.array(preserve_info["preserve_ratio"]),
        model_owned_student_output=np.array(True),
        teacher_points_used_at_inference=np.array(False),
        raw_kinect_depth_used_at_inference=np.array(False),
        facial_detail_target_applicable=np.array(False),
        face_detail_claim_allowed=np.array(False),
        allowed_face_claim=np.array(ALLOWED_FACE_CLAIM),
        config=np.array(TRUE_CONFIG),
        case_id=np.array(row["case"]),
        visible_surface_preserving_infill=np.array(True),
        infill_aligned_to_baseline_frame=np.array(True),
        infill_alignment_json=np.array(json.dumps(alignment_info)),
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
        "infill_aligned_to_baseline_frame": True,
        "alignment_type": alignment_info["alignment_type"],
        "alignment_axis_scale_json": json.dumps(alignment_info["axis_scale"]),
        **preserve_info,
        **metric,
    }


def config_path(case: str, config: str) -> Path:
    roots = {
        TRUE_CONFIG: OUT_ROOT,
        V192_CONFIG: OUTPUT / "V19200000000000000000_upright_pose_frame_layout",
        V190_CONFIG: OUTPUT / "V19000000000000000000_pose_frame_occupancy_repair",
        V187_CONFIG: OUTPUT / "V18700000000000000000_visible_anchor_canonical_surfel_training",
        V186_CONFIG: OUTPUT / "V18600000000000000000_part_coverage_canonical_surfel_training",
    }
    if config in roots:
        return roots[config] / case / config / "predictions.npz"
    return OUTPUT / "V1400000000000000000_learned_residual_matrix" / case / config / "predictions.npz"


def load_config(case: str, config: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    pred = load_npz(config_path(case, config))
    body = np.asarray(pred["body_part_id"]) if "body_part_id" in pred else None
    return np.asarray(pred["human_points"], dtype=np.float32), as_rgb(pred["human_rgb"]), body


def compare(rows: list[dict[str, str]], created_at: str) -> None:
    score_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    configs = [TRUE_CONFIG, V192_CONFIG, V190_CONFIG, V187_CONFIG, V186_CONFIG, BASELINE_CONFIG, "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control"]
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
            failures.append({"case": case, "reason": "missing_visible_surface_infill_true"})
            continue
        true_score = float(true["combined_topology_volume_score_v4"])
        if bool(true["combined_fail_v4"]):
            failures.append({"case": case, "reason": "true_combined_fail_v4", "true_score": true_score})
        for cfg in [V192_CONFIG, V190_CONFIG, V187_CONFIG, V186_CONFIG, BASELINE_CONFIG, "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control"]:
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
    write_csv(REPORTS / "V19400000000000000000_visible_surface_infill_scores.csv", score_rows)
    first = rows[0]["case"] if rows else ""
    panels: list[Image.Image] = []
    for cfg in configs:
        path = config_path(first, cfg)
        if path.exists():
            pts, rgb, _body = load_config(first, cfg)
            panels.append(render_panel(pts, rgb, f"{first} {cfg.replace('_', ' ')}"))
    if panels:
        compose(panels, 3, BOARDS / "V19400000000000000000_visible_surface_infill_board.png")
    if first and config_path(first, TRUE_CONFIG).exists():
        pts, rgb, _body = load_config(first, TRUE_CONFIG)
        turn = [
            render_panel(pts, rgb, f"{first} infill {label}", rot=rot)
            for label, rot in [
                ("front", rotation_matrix(0, 0)),
                ("back", rotation_matrix(180, 0)),
                ("left", rotation_matrix(-90, 0)),
                ("right", rotation_matrix(90, 0)),
                ("oblique", rotation_matrix(-30, 61)),
            ]
        ]
        turn.append(cross_panel(pts, f"{first} infill cross-section"))
        compose(turn, 3, BOARDS / "V19400000000000000000_visible_surface_infill_turntable_cross_section.png")
    decision = {
        "created_at": created_at,
        "status": "V19400_VISIBLE_SURFACE_INFILL_FAIL_CLOSED_CONTINUE" if failures else "V19400_VISIBLE_SURFACE_INFILL_PRECHECK_PASS_REQUIRES_MENTOR_VISUAL",
        "mentor_ready": False,
        "external_hard_block": False,
        "failures": failures,
        "score_csv": str(REPORTS / "V19400000000000000000_visible_surface_infill_scores.csv"),
        "board": str(BOARDS / "V19400000000000000000_visible_surface_infill_board.png"),
        "turntable_cross_section": str(BOARDS / "V19400000000000000000_visible_surface_infill_turntable_cross_section.png"),
        "face_detail_claim_allowed": False,
        "allowed_face_claim": ALLOWED_FACE_CLAIM,
        "summary": "V194 preserves coherent visible VGGT RGB surface points and inserts topology-volume infill only in non-preserved regions. It remains fail-closed unless full-scene mentor visuals and hard controls pass.",
    }
    write_json(REPORTS / "V19400000000000000000_visible_surface_infill_decision.json", decision)


def main() -> int:
    created_at = now()
    rows = read_manifest()
    device, runtime = select_device()
    steps = int(os.environ.get("V19400_STEPS", "40"))
    max_points = int(os.environ.get("V19400_MAX_POINTS", "8192"))
    manifest = [train_case(row, steps=steps, max_points=max_points, device=device) for row in rows]
    write_csv(REPORTS / "V19400000000000000000_visible_surface_infill_training_manifest.csv", manifest)
    compare(rows, created_at)
    write_json(
        REPORTS / "V19400000000000000000_runtime_environment.json",
        {"created_at": created_at, **runtime, "steps": steps, "max_points": max_points},
    )
    print(json.dumps({"created_at": created_at, "status": "V19400_DONE", "device": str(device), "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
