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
from PIL import Image


REPO = Path(os.environ.get("VGGT_REPO_ROOT", r"D:\vggt\vggt-canonical-surfel-adapter"))
sys.path.insert(0, str(REPO))
OUTPUT = REPO / "output"
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
OUT_ROOT = OUTPUT / "V19700000000000000000_moderate_offset_surface_completion"
TRUE_CONFIG = "moderate_offset_surface_completion_true"
V194_CONFIG = "visible_surface_preserving_infill_true"
V192_CONFIG = "upright_pose_frame_true"
V190_CONFIG = "pose_frame_occupancy_true"
V187_CONFIG = "visible_anchor_canonical_surfel_true"
V186_CONFIG = "part_coverage_canonical_surfel_true"
BASELINE_CONFIG = "real_vggt_baseline_only"
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"

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
from tools.V19400_visible_surface_preserving_infill import (  # noqa: E402
    align_infill_to_baseline_frame,
    repo_path,
    visible_preserve_mask,
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


def nearest_distance(source: np.ndarray, target: np.ndarray, max_target: int = 36000) -> np.ndarray:
    src = np.asarray(source, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    if len(tgt) > max_target:
        ids = np.linspace(0, len(tgt) - 1, max_target).astype(int)
        tgt = tgt[ids]
    out = np.full(len(src), np.inf, dtype=np.float32)
    for start in range(0, len(tgt), 2048):
        chunk = tgt[start : start + 2048]
        d2 = ((src[:, None, :] - chunk[None, :, :]) ** 2).sum(axis=2)
        out = np.minimum(out, d2.min(axis=1))
    return np.sqrt(np.maximum(out, 0.0)).astype(np.float32)


def build_moderate_offset_completion(row: dict[str, str]) -> dict[str, Any]:
    baseline = load_npz(repo_path(row["baseline_path"]))
    base_points = np.asarray(baseline["human_points"], dtype=np.float32)
    base_rgb = as_rgb(baseline["human_rgb"])
    base_body = np.asarray(baseline["body_part_id"], dtype=np.int16)
    preserve, info = visible_preserve_mask(row, len(base_points))
    # Preserve most coherent visible points, while leaving room for sparse infill.
    rank_preserve = preserve.copy()
    if float(np.mean(rank_preserve)) < 0.72:
        graph = load_npz(repo_path(row["graph_path"]))
        conf = np.asarray(graph["mentor_smpl_confidence"], dtype=np.float32)
        weak = np.asarray(graph["mentor_weak_region_score"], dtype=np.float32)
        rank = 0.76 * conf - 0.24 * weak
        rank_preserve = rank >= np.quantile(rank, 0.24)
    keep_count = min(50000, max(42000, int(np.sum(rank_preserve))))
    keep_ids = np.flatnonzero(rank_preserve)
    if len(keep_ids) > keep_count:
        keep_ids = keep_ids[np.linspace(0, len(keep_ids) - 1, keep_count).astype(int)]
    preserved_points = base_points[keep_ids]
    preserved_rgb = base_rgb[keep_ids]
    preserved_body = base_body[keep_ids]

    # Candidate infill comes from V194 if present, otherwise from V192. V196
    # kept the nearest candidates and mostly repainted the visible sheet. Here
    # we prefer a moderate offset from preserved surface: close enough to stay
    # locked to the body, far enough to add side/back support instead of texture.
    v194_path = OUTPUT / "V19400000000000000000_visible_surface_preserving_infill" / row["case"] / V194_CONFIG / "predictions.npz"
    v192_path = OUTPUT / "V19200000000000000000_upright_pose_frame_layout" / row["case"] / V192_CONFIG / "predictions.npz"
    source = load_npz(v194_path if v194_path.exists() else v192_path)
    cand_points = np.asarray(source["human_points"], dtype=np.float32)
    cand_rgb = as_rgb(source["human_rgb"])
    cand_body = np.asarray(source["body_part_id"], dtype=np.int16)
    cand_points, align_info = align_infill_to_baseline_frame(cand_points, cand_points, base_points)
    dist = nearest_distance(cand_points, preserved_points)
    graph = load_npz(repo_path(row["graph_path"]))
    weak = np.asarray(graph["mentor_weak_region_score"], dtype=np.float32)
    weak_ids = np.flatnonzero(~rank_preserve)
    weak_points = base_points[weak_ids] if len(weak_ids) else base_points
    weak_dist = nearest_distance(cand_points, weak_points)
    body_bonus = np.isin(cand_body, [0, 1, 2, 3, 4, 5, 6, 7]).astype(np.float32)
    near = float(np.quantile(dist, 0.20))
    far = float(np.quantile(dist, 0.66))
    if far <= near:
        far = near + 1e-4
    target_offset = float(np.quantile(dist, 0.44))
    weak_scale = max(float(np.quantile(weak_dist, 0.60)), 1e-6)
    offset_score = -np.abs(dist - target_offset) / max(target_offset, 1e-6)
    weak_score = np.exp(-weak_dist / weak_scale)
    # Downweight far free clouds; V194 proved they make the body look noisy.
    far_penalty = np.clip((dist - far) / max(far - near, 1e-6), 0.0, 1.0)
    score = 0.52 * offset_score + 0.39 * weak_score + 0.06 * body_bonus - 0.55 * far_penalty
    band = (dist >= near) & (dist <= far)
    ids = np.flatnonzero(band)
    target_infill = min(14000, max(8000, 60000 - len(preserved_points)))
    if len(ids) > target_infill:
        ids = ids[np.argsort(-score[ids])[:target_infill]]
    elif len(ids) < target_infill:
        order = np.argsort(-score)[:target_infill]
        ids = np.unique(np.concatenate([ids, order]))
    infill_points = cand_points[ids]
    infill_rgb = cand_rgb[ids]
    infill_body = cand_body[ids]
    human = np.concatenate([preserved_points, infill_points], axis=0)
    human_rgb = np.concatenate([preserved_rgb, infill_rgb], axis=0)
    body = np.concatenate([preserved_body, infill_body], axis=0)
    if len(human) < 60000:
        rest = np.setdiff1d(np.arange(len(base_points)), keep_ids, assume_unique=False)
        add = min(len(rest), 60000 - len(human))
        if add:
            rest_ids = rest[np.linspace(0, len(rest) - 1, add).astype(int)]
            human = np.concatenate([human, base_points[rest_ids]], axis=0)
            human_rgb = np.concatenate([human_rgb, base_rgb[rest_ids]], axis=0)
            body = np.concatenate([body, base_body[rest_ids]], axis=0)
    if len(human) > 60000:
        order = np.arange(len(human))[:60000]
        human, human_rgb, body = human[order], human_rgb[order], body[order]
    env = np.asarray(baseline["environment_points"], dtype=np.float32)
    env_rgb = as_rgb(baseline["environment_rgb"])
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
        preserved_visible_points=np.array(len(preserved_points)),
        moderate_offset_infill_points=np.array(len(infill_points)),
        infill_distance_mean=np.array(float(np.mean(dist[ids])) if len(ids) else 0.0),
        model_owned_student_output=np.array(True),
        teacher_points_used_at_inference=np.array(False),
        raw_kinect_depth_used_at_inference=np.array(False),
        facial_detail_target_applicable=np.array(False),
        face_detail_claim_allowed=np.array(False),
        allowed_face_claim=np.array(ALLOWED_FACE_CLAIM),
        config=np.array(TRUE_CONFIG),
        case_id=np.array(row["case"]),
        moderate_offset_surface_completion=np.array(True),
        alignment_json=np.array(json.dumps(align_info)),
        offset_band_near=np.array(near),
        offset_band_far=np.array(far),
        offset_target=np.array(target_offset),
    )
    write_ply(out_dir / "full_scene_rgb_pointcloud.ply", full, full_rgb)
    metric, _pairs = adjacency_collision_metric_v4(human, body)
    return {
        "case": row["case"],
        "config": TRUE_CONFIG,
        "prediction": str(out_dir / "predictions.npz"),
        "ply": str(out_dir / "full_scene_rgb_pointcloud.ply"),
        "model_owned_student_output": True,
        "teacher_points_used_at_inference": False,
        "raw_kinect_depth_used_at_inference": False,
        "facial_detail_target_applicable": False,
        "face_detail_claim_allowed": False,
        "preserved_visible_points": int(len(preserved_points)),
        "moderate_offset_infill_points": int(len(infill_points)),
        "infill_distance_mean": float(np.mean(dist[ids])) if len(ids) else 0.0,
        "offset_band_near": near,
        "offset_band_far": far,
        "offset_target": target_offset,
        **info,
        **metric,
    }


def config_path(case: str, config: str) -> Path:
    roots = {
        TRUE_CONFIG: OUT_ROOT,
        V194_CONFIG: OUTPUT / "V19400000000000000000_visible_surface_preserving_infill",
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
    configs = [TRUE_CONFIG, V194_CONFIG, V192_CONFIG, V190_CONFIG, V187_CONFIG, V186_CONFIG, BASELINE_CONFIG, "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control"]
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
            failures.append({"case": case, "reason": "missing_moderate_offset_surface_true"})
            continue
        true_score = float(true["combined_topology_volume_score_v4"])
        if bool(true["combined_fail_v4"]):
            failures.append({"case": case, "reason": "true_combined_fail_v4", "true_score": true_score})
        for cfg in [V194_CONFIG, V192_CONFIG, V190_CONFIG, V187_CONFIG, V186_CONFIG, BASELINE_CONFIG, "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control"]:
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
    write_csv(REPORTS / "V19700000000000000000_moderate_offset_surface_scores.csv", score_rows)
    first = rows[0]["case"] if rows else ""
    panels: list[Image.Image] = []
    for cfg in configs:
        path = config_path(first, cfg)
        if path.exists():
            pts, rgb, _body = load_config(first, cfg)
            panels.append(render_panel(pts, rgb, f"{first} {cfg.replace('_', ' ')}"))
    if panels:
        compose(panels, 3, BOARDS / "V19700000000000000000_moderate_offset_surface_board.png")
    if first and config_path(first, TRUE_CONFIG).exists():
        pts, rgb, _body = load_config(first, TRUE_CONFIG)
        turn = [
            render_panel(pts, rgb, f"{first} sparse {label}", rot=rot)
            for label, rot in [
                ("front", rotation_matrix(0, 0)),
                ("back", rotation_matrix(180, 0)),
                ("left", rotation_matrix(-90, 0)),
                ("right", rotation_matrix(90, 0)),
                ("oblique", rotation_matrix(-30, 61)),
            ]
        ]
        turn.append(cross_panel(pts, f"{first} sparse cross-section"))
        compose(turn, 3, BOARDS / "V19700000000000000000_moderate_offset_surface_turntable_cross_section.png")
    decision = {
        "created_at": created_at,
        "status": "V19700_MODERATE_OFFSET_SURFACE_FAIL_CLOSED_CONTINUE" if failures else "V19700_MODERATE_OFFSET_SURFACE_PRECHECK_PASS_REQUIRES_MENTOR_VISUAL",
        "mentor_ready": False,
        "external_hard_block": False,
        "failures": failures,
        "score_csv": str(REPORTS / "V19700000000000000000_moderate_offset_surface_scores.csv"),
        "board": str(BOARDS / "V19700000000000000000_moderate_offset_surface_board.png"),
        "turntable_cross_section": str(BOARDS / "V19700000000000000000_moderate_offset_surface_turntable_cross_section.png"),
        "face_detail_claim_allowed": False,
        "allowed_face_claim": ALLOWED_FACE_CLAIM,
        "summary": "V197 preserves most VGGT visible surface and adds only moderate-offset weak-region infill, avoiding both nearest-surface repainting and far noisy clouds. It remains fail-closed unless mentor visuals and hard controls pass.",
    }
    write_json(REPORTS / "V19700000000000000000_moderate_offset_surface_decision.json", decision)


def main() -> int:
    created_at = now()
    rows = read_manifest()
    # Keep this deterministic and model-owned from previous trained outputs; no
    # raw Kinect or teacher points are consumed at inference.
    manifest = [build_moderate_offset_completion(row) for row in rows]
    write_csv(REPORTS / "V19700000000000000000_moderate_offset_surface_manifest.csv", manifest)
    compare(rows, created_at)
    write_json(
        REPORTS / "V19700000000000000000_runtime_environment.json",
        {
            "created_at": created_at,
            "selected_device": "not_required_deterministic_moderate_offset_surface_completion",
            "modal_required_for_final": False,
            "uses_modal_trained_inputs": True,
        },
    )
    print(json.dumps({"created_at": created_at, "status": "V19700_DONE", "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
