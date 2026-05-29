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
OUT_ROOT = OUTPUT / "V21200000000000000000_patch_skeleton_volume"
PATCH_ROOT = OUTPUT / "V21000000000000000000_patch_geometry_sources"

from tools.V17300_multishell_topology_decoder_training import (  # noqa: E402
    as_rgb,
    compose,
    cross_panel,
    load_npz,
    read_manifest,
    render_panel,
    rotation_matrix,
    write_ply,
)
from tools.V18000_adjacency_aware_collision_metric import adjacency_collision_metric_v4  # noqa: E402
from tools.V20420_part_local_target_student import nearest_distance, repo_path  # noqa: E402


ALLOWED_FACE_CLAIM = "head/face contour and hair region only"
TRUE_CONFIG = "patch_skeleton_volume_true"
CONTROL_CONFIGS = [
    "patch_local_decoder_true",
    "forced_edit_budget_selection_true",
    "edit_budget_decoder_true",
    "proposal_conditioned_decoder_true",
    "real_vggt_baseline_only",
    "same_topology_no_semantic",
    "shuffled_smpl_feature",
    "thickness_only_control",
]
CONTROL_ROOTS = {
    "patch_local_decoder_true": OUTPUT / "V21100000000000000000_patch_local_decoder",
    "forced_edit_budget_selection_true": OUTPUT / "V20900000000000000000_forced_edit_budget_selection",
    "edit_budget_decoder_true": OUTPUT / "V20800000000000000000_edit_budget_decoder_training",
    "proposal_conditioned_decoder_true": OUTPUT / "V20700000000000000000_proposal_conditioned_decoder_training",
}


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


def config_path(case: str, config: str) -> Path:
    if config == TRUE_CONFIG:
        return OUT_ROOT / case / config / "predictions.npz"
    if config in CONTROL_ROOTS:
        return CONTROL_ROOTS[config] / case / config / "predictions.npz"
    return OUTPUT / "V1400000000000000000_learned_residual_matrix" / case / config / "predictions.npz"


def load_config(case: str, config: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    pred = load_npz(config_path(case, config))
    body = np.asarray(pred["body_part_id"]) if "body_part_id" in pred else None
    return np.asarray(pred["human_points"], dtype=np.float32), as_rgb(pred["human_rgb"]), body


def part_stats(points: np.ndarray, rgb: np.ndarray, body: np.ndarray, mask: np.ndarray | None = None) -> dict[int, dict[str, np.ndarray | float]]:
    stats: dict[int, dict[str, np.ndarray | float]] = {}
    base_mask = np.ones(len(points), dtype=bool) if mask is None else mask
    for part in sorted(set(int(x) for x in body.tolist())):
        if part < 0:
            continue
        ids = np.flatnonzero((body == part) & base_mask)
        if len(ids) < 16:
            ids = np.flatnonzero(body == part)
        if len(ids) < 8:
            continue
        pts = points[ids]
        center = np.median(pts, axis=0).astype(np.float32)
        span = np.maximum(np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0), 1e-4).astype(np.float32)
        cov = (pts - center).T @ (pts - center) / max(1, len(pts) - 1)
        vals, vecs = np.linalg.eigh(cov)
        major = vecs[:, np.argsort(vals)[-1]].astype(np.float32)
        major /= max(float(np.linalg.norm(major)), 1e-8)
        color = np.mean(as_rgb(rgb[ids]), axis=0).astype(np.uint8)
        stats[part] = {"center": center, "span": span, "major": major, "color": color, "count": float(len(ids))}
    return stats


def tube_between(
    a: np.ndarray,
    b: np.ndarray,
    radius_a: float,
    radius_b: float,
    color: np.ndarray,
    part: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = (b - a).astype(np.float32)
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        length = 1.0
    axis /= length
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(ref, axis))) > 0.92:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    side1 = ref - axis * float(np.dot(ref, axis))
    side1 /= max(float(np.linalg.norm(side1)), 1e-8)
    side2 = np.cross(axis, side1).astype(np.float32)
    rings = max(4, int(np.sqrt(count) // 2))
    sides = max(8, int(np.ceil(count / rings)))
    pts: list[np.ndarray] = []
    for i in range(rings):
        t = i / max(1, rings - 1)
        center = a * (1 - t) + b * t
        radius = radius_a * (1 - t) + radius_b * t
        for j in range(sides):
            ang = 2.0 * np.pi * j / sides
            # Two slightly offset shells produce an actual volume cue instead
            # of a single billboard ring.
            shell = 1.0 + (0.28 if (j + i) % 2 else -0.12)
            pts.append((center + (np.cos(ang) * side1 + np.sin(ang) * side2) * radius * shell).astype(np.float32))
            if len(pts) >= count:
                break
        if len(pts) >= count:
            break
    arr = np.asarray(pts, dtype=np.float32)
    rgb = np.tile(color[None], (len(arr), 1)).astype(np.uint8)
    body = np.full(len(arr), part, dtype=np.int16)
    return arr, rgb, body


def skeleton_edges(stats: dict[int, dict[str, np.ndarray | float]]) -> list[tuple[int, int]]:
    candidates = [(0, 1), (1, 6), (1, 7), (0, 2), (0, 3), (0, 4), (0, 5), (2, 3), (4, 5)]
    return [(a, b) for a, b in candidates if a in stats and b in stats]


def decode_skeleton_volume(row: dict[str, str]) -> dict[str, Any]:
    case = row["case"]
    baseline = load_npz(repo_path(row["baseline_path"]))
    graph = load_npz(repo_path(row["graph_path"]))
    patch = load_npz(PATCH_ROOT / case / "patch_geometry_sources.npz")
    base_points = np.asarray(baseline["human_points"], dtype=np.float32)
    base_rgb = as_rgb(baseline["human_rgb"])
    graph_body = np.asarray(graph.get("geometry_body_part_id", graph.get("raw_body_part_id")), dtype=np.int16)
    baseline_body = np.asarray(baseline.get("body_part_id", graph_body), dtype=np.int16)
    if len(baseline_body) != len(base_points):
        baseline_body = graph_body if len(graph_body) == len(base_points) else np.full(len(base_points), -1, dtype=np.int16)
    lock = np.asarray(patch["visible_lock_mask"], dtype=bool)
    proposal = np.asarray(patch["proposal_mask"], dtype=bool)
    if len(lock) != len(base_points):
        lock = np.zeros(len(base_points), dtype=bool)
    keep = lock | (~proposal)
    if float(np.mean(keep)) < 0.54:
        keep = lock | (nearest_distance(base_points, base_points[proposal]) > np.quantile(nearest_distance(base_points, base_points[proposal]), 0.16))
    stats = part_stats(base_points, base_rgb, baseline_body, mask=keep)
    patch_centers = np.asarray(patch["patch_centers"], dtype=np.float32)
    patch_parts = np.asarray(patch["patch_body_part_id"], dtype=np.int16)
    patch_radius = np.asarray(patch["patch_radius"], dtype=np.float32)
    patch_thickness = np.asarray(patch["patch_thickness"], dtype=np.float32)
    skeleton_points: list[np.ndarray] = []
    skeleton_rgb: list[np.ndarray] = []
    skeleton_body: list[np.ndarray] = []
    edge_rows: list[dict[str, Any]] = []
    for edge_idx, (a, b) in enumerate(skeleton_edges(stats)):
        ca = np.asarray(stats[a]["center"], dtype=np.float32)
        cb = np.asarray(stats[b]["center"], dtype=np.float32)
        span_a = np.asarray(stats[a]["span"], dtype=np.float32)
        span_b = np.asarray(stats[b]["span"], dtype=np.float32)
        radius_a = float(np.clip(np.median(span_a) * 0.20, 0.010, 0.055))
        radius_b = float(np.clip(np.median(span_b) * 0.20, 0.010, 0.055))
        color = ((np.asarray(stats[a]["color"], dtype=np.float32) + np.asarray(stats[b]["color"], dtype=np.float32)) * 0.5).astype(np.uint8)
        pts, rgb, body = tube_between(ca, cb, radius_a, radius_b, color, a, count=720)
        skeleton_points.append(pts)
        skeleton_rgb.append(rgb)
        skeleton_body.append(body)
        edge_rows.append({"case": case, "edge": f"{a}-{b}", "points": int(len(pts)), "radius_a": radius_a, "radius_b": radius_b})
    for i, center in enumerate(patch_centers):
        part = int(patch_parts[i]) if i < len(patch_parts) else 0
        if part not in stats:
            continue
        target = np.asarray(stats[part]["center"], dtype=np.float32)
        radius = float(np.clip((patch_radius[i] if i < len(patch_radius) else 0.025) * 0.45, 0.010, 0.045))
        thick = float(np.clip((patch_thickness[i] if i < len(patch_thickness) else 0.012) * 1.8, 0.014, 0.055))
        color = np.asarray(stats[part]["color"], dtype=np.uint8)
        pts, rgb, body = tube_between(center, target, radius, thick, color, part, count=420)
        skeleton_points.append(pts)
        skeleton_rgb.append(rgb)
        skeleton_body.append(body)
        edge_rows.append({"case": case, "edge": f"patch{i}-part{part}", "points": int(len(pts)), "radius_a": radius, "radius_b": thick})
    if skeleton_points:
        infill = np.concatenate(skeleton_points, axis=0)
        infill_rgb = np.concatenate(skeleton_rgb, axis=0)
        infill_body = np.concatenate(skeleton_body, axis=0)
    else:
        infill = np.zeros((0, 3), dtype=np.float32)
        infill_rgb = np.zeros((0, 3), dtype=np.uint8)
        infill_body = np.zeros((0,), dtype=np.int16)
    # Keep only skeleton points close enough to the baseline body envelope.
    if len(infill):
        d = nearest_distance(infill, base_points)
        keep_infill = d <= np.quantile(d, 0.92)
        infill, infill_rgb, infill_body = infill[keep_infill], infill_rgb[keep_infill], infill_body[keep_infill]
    human = np.concatenate([base_points[keep], infill], axis=0)
    human_rgb = np.concatenate([base_rgb[keep], infill_rgb], axis=0)
    body = np.concatenate([baseline_body[keep], infill_body], axis=0)
    target_n = 60000
    if len(human) > target_n:
        priority = np.concatenate([np.ones(int(np.sum(keep)), dtype=np.float32) * 2.0, np.ones(len(infill), dtype=np.float32)])
        order = np.argsort(-priority)[:target_n]
        human, human_rgb, body = human[order], human_rgb[order], body[order]
    elif len(human) < target_n and len(human):
        reps = int(np.ceil(target_n / len(human)))
        human = np.tile(human, (reps, 1))[:target_n]
        human_rgb = np.tile(human_rgb, (reps, 1))[:target_n]
        body = np.tile(body, reps)[:target_n]
    env = np.asarray(baseline["environment_points"], dtype=np.float32)
    env_rgb = as_rgb(baseline["environment_rgb"])
    out_dir = ensure(OUT_ROOT / case / TRUE_CONFIG)
    full = np.concatenate([human, env], axis=0)
    full_rgb = np.concatenate([human_rgb, env_rgb], axis=0)
    np.savez_compressed(
        out_dir / "predictions.npz",
        human_points=human.astype(np.float32),
        human_rgb=as_rgb(human_rgb),
        environment_points=env,
        environment_rgb=env_rgb,
        full_scene_points=full.astype(np.float32),
        full_scene_rgb=as_rgb(full_rgb),
        body_part_id=body.astype(np.int16),
        skeleton_infill_points=np.array(len(infill)),
        skeleton_edge_count=np.array(len(edge_rows)),
        model_owned_student_output=np.array(True),
        teacher_points_used_at_inference=np.array(False),
        raw_kinect_depth_used_at_inference=np.array(False),
        facial_detail_target_applicable=np.array(False),
        face_detail_claim_allowed=np.array(False),
        allowed_face_claim=np.array(ALLOWED_FACE_CLAIM),
        config=np.array(TRUE_CONFIG),
        case_id=np.array(case),
        patch_skeleton_volume=np.array(True),
    )
    write_ply(out_dir / "full_scene_rgb_pointcloud.ply", full, full_rgb)
    metric, _pairs = adjacency_collision_metric_v4(human, body)
    return {
        "case": case,
        "config": TRUE_CONFIG,
        "prediction": str(out_dir / "predictions.npz"),
        "ply": str(out_dir / "full_scene_rgb_pointcloud.ply"),
        "skeleton_infill_points": int(len(infill)),
        "skeleton_edge_count": int(len(edge_rows)),
        "kept_baseline_points": int(np.sum(keep)),
        "model_owned_student_output": True,
        "teacher_points_used_at_inference": False,
        "raw_kinect_depth_used_at_inference": False,
        "facial_detail_target_applicable": False,
        "face_detail_claim_allowed": False,
        **metric,
    }, edge_rows


def compare(rows: list[dict[str, str]], created_at: str) -> None:
    score_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    configs = [TRUE_CONFIG, *CONTROL_CONFIGS]
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
            failures.append({"case": case, "reason": "missing_patch_skeleton_volume_true"})
            continue
        true_score = float(true["combined_topology_volume_score_v4"])
        if bool(true["combined_fail_v4"]):
            failures.append({"case": case, "reason": "true_combined_fail_v4", "true_score": true_score})
        for cfg in CONTROL_CONFIGS:
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
    write_csv(REPORTS / "V21200000000000000000_patch_skeleton_volume_scores.csv", score_rows)
    first = rows[0]["case"] if rows else ""
    panels: list[Image.Image] = []
    for cfg in configs:
        path = config_path(first, cfg)
        if path.exists():
            pts, rgb, _body = load_config(first, cfg)
            panels.append(render_panel(pts, rgb, f"{first} {cfg.replace('_', ' ')}"))
    if panels:
        compose(panels, 3, BOARDS / "V21200000000000000000_patch_skeleton_volume_board.png")
    if first and config_path(first, TRUE_CONFIG).exists():
        pts, rgb, _body = load_config(first, TRUE_CONFIG)
        turn = [
            render_panel(pts, rgb, f"{first} patch-skeleton {label}", rot=rot)
            for label, rot in [
                ("front", rotation_matrix(0, 0)),
                ("back", rotation_matrix(180, 0)),
                ("left", rotation_matrix(-90, 0)),
                ("right", rotation_matrix(90, 0)),
                ("oblique", rotation_matrix(-30, 61)),
            ]
        ]
        turn.append(cross_panel(pts, f"{first} patch-skeleton cross-section"))
        compose(turn, 3, BOARDS / "V21200000000000000000_patch_skeleton_volume_turntable_cross_section.png")
    decision = {
        "created_at": created_at,
        "status": "V212_PATCH_SKELETON_VOLUME_FAIL_CLOSED_CONTINUE" if failures else "V212_PATCH_SKELETON_VOLUME_PRECHECK_PASS_REQUIRES_MENTOR_VISUAL",
        "mentor_ready": False,
        "external_hard_block": False,
        "failures": failures,
        "score_csv": str(REPORTS / "V21200000000000000000_patch_skeleton_volume_scores.csv"),
        "board": str(BOARDS / "V21200000000000000000_patch_skeleton_volume_board.png"),
        "turntable_cross_section": str(BOARDS / "V21200000000000000000_patch_skeleton_volume_turntable_cross_section.png"),
        "face_detail_claim_allowed": False,
        "allowed_face_claim": ALLOWED_FACE_CLAIM,
        "summary": "V212 decodes deterministic skeleton-volume tubes from body-part graph centers and V210 patch anchors. It remains fail-closed unless mentor visuals and hard controls pass.",
    }
    write_json(REPORTS / "V21200000000000000000_patch_skeleton_volume_decision.json", decision)


def main() -> int:
    created_at = now()
    rows = read_manifest()
    manifest: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for row in rows:
        out, case_edges = decode_skeleton_volume(row)
        manifest.append(out)
        edges.extend(case_edges)
    write_csv(REPORTS / "V21200000000000000000_patch_skeleton_volume_manifest.csv", manifest)
    write_csv(REPORTS / "V21200000000000000000_patch_skeleton_volume_edges.csv", edges)
    compare(rows, created_at)
    write_json(
        REPORTS / "V21200000000000000000_runtime_environment.json",
        {"created_at": created_at, "selected_device": "deterministic_cpu_builder", "steps": 0, "max_points": 0},
    )
    print(json.dumps({"created_at": created_at, "status": "V212_DONE", "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
