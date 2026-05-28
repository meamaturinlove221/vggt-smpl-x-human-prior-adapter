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
from PIL import Image, ImageDraw


REPO = Path(os.environ.get("VGGT_REPO_ROOT", r"D:\vggt\vggt-canonical-surfel-adapter"))
sys.path.insert(0, str(REPO))
OUTPUT = REPO / "output"
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"

CONFIG_PATHS = {
    "moderate_offset_surface_completion_true": OUTPUT / "V19700000000000000000_moderate_offset_surface_completion",
    "surface_locked_sparse_completion_true": OUTPUT / "V19600000000000000000_surface_locked_sparse_completion",
    "visible_surface_preserving_infill_true": OUTPUT / "V19400000000000000000_visible_surface_preserving_infill",
    "real_vggt_baseline_only": OUTPUT / "V1400000000000000000_learned_residual_matrix",
    "same_topology_no_semantic": OUTPUT / "V1400000000000000000_learned_residual_matrix",
    "shuffled_smpl_feature": OUTPUT / "V1400000000000000000_learned_residual_matrix",
}
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"

from tools.V17300_multishell_topology_decoder_training import (  # noqa: E402
    as_rgb,
    compose,
    cross_panel,
    load_npz,
    read_manifest,
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


def config_path(case: str, config: str) -> Path:
    return CONFIG_PATHS[config] / case / config / "predictions.npz"


def load_config(case: str, config: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    pred = load_npz(config_path(case, config))
    body = np.asarray(pred["body_part_id"]) if "body_part_id" in pred else None
    env = np.asarray(pred["environment_points"], dtype=np.float32) if "environment_points" in pred else np.empty((0, 3), dtype=np.float32)
    env_rgb = as_rgb(pred["environment_rgb"]) if "environment_rgb" in pred else np.empty((0, 3), dtype=np.uint8)
    return np.asarray(pred["human_points"], dtype=np.float32), as_rgb(pred["human_rgb"]), body, env, env_rgb


def source_upright_rotation(points: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Render with the dominant body axis vertical instead of sideways.

    The original oblique renderer is useful for depth cues, but these cases are
    back/side-back views whose longest axis lies close to world y. This rotation
    makes mentor boards readable without claiming that geometry itself improved.
    """
    pts = np.asarray(points, dtype=np.float64)
    centered = pts - np.median(pts, axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, len(pts) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    long_axis = vecs[:, order[0]]
    if long_axis[1] < 0:
        long_axis = -long_axis
    # A camera-like basis: screen-y is body long axis, screen-x is mostly world x,
    # depth is orthogonal with a small oblique tilt so the board keeps point-cloud cues.
    up = long_axis / max(np.linalg.norm(long_axis), 1e-9)
    x_seed = np.array([1.0, 0.0, 0.0])
    right = x_seed - up * float(np.dot(x_seed, up))
    if np.linalg.norm(right) < 1e-6:
        right = np.array([0.0, 0.0, 1.0]) - up * float(np.dot(np.array([0.0, 0.0, 1.0]), up))
    right = right / max(np.linalg.norm(right), 1e-9)
    depth = np.cross(right, up)
    depth = depth / max(np.linalg.norm(depth), 1e-9)
    # Add a restrained yaw into depth while preserving vertical body readability.
    right2 = 0.94 * right + 0.18 * depth
    right2 = right2 / max(np.linalg.norm(right2), 1e-9)
    depth2 = np.cross(right2, up)
    depth2 = depth2 / max(np.linalg.norm(depth2), 1e-9)
    rot = np.stack([right2, up, depth2], axis=0)
    return rot.astype(np.float32), {
        "long_axis": long_axis.tolist(),
        "right_axis": right2.tolist(),
        "up_axis": up.tolist(),
        "depth_axis": depth2.tolist(),
        "eigenvalues_desc": vals[order].tolist(),
    }


def render_source_upright(points: np.ndarray, colors: np.ndarray, title: str, *, full_env: tuple[np.ndarray, np.ndarray] | None = None) -> Image.Image:
    size = (430, 320)
    im = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(im)
    rot, _info = source_upright_rotation(points)
    if full_env is not None and len(full_env[0]):
        env_pts, env_rgb = full_env
        all_pts = np.concatenate([env_pts, points], axis=0)
        all_rgb = np.concatenate([env_rgb, colors], axis=0)
        env_mask = np.arange(len(all_pts)) < len(env_pts)
    else:
        all_pts = points
        all_rgb = colors
        env_mask = np.zeros(len(points), dtype=bool)
    pts = (all_pts - np.median(points, axis=0, keepdims=True)) @ rot.T
    lo = np.percentile(pts[:, :2], 1, axis=0)
    hi = np.percentile(pts[:, :2], 99, axis=0)
    pad = (hi - lo) * np.array([0.20, 0.13]) + 1e-6
    lo -= pad
    hi += pad
    q = (pts[:, :2] - lo[None]) / (hi[None] - lo[None] + 1e-9)
    q[:, 1] = 1.0 - q[:, 1]
    xy = np.clip(q * np.array([size[0] - 48, size[1] - 70]) + np.array([24, 44]), 0, [size[0] - 1, size[1] - 1]).astype(int)
    depth = pts[:, 2]
    cue = np.clip((depth - np.quantile(depth, 0.03)) / max(np.quantile(depth, 0.97) - np.quantile(depth, 0.03), 1e-9), 0, 1)
    rgb = np.clip(all_rgb.astype(np.float32) * (0.58 + 0.50 * cue[:, None]), 0, 255).astype(np.uint8)
    rgb[env_mask] = np.clip(rgb[env_mask].astype(np.float32) * np.array([0.62, 0.80, 0.66]), 0, 255).astype(np.uint8)
    order = np.argsort(depth)
    step = max(1, len(order) // 62000)
    for i in order[::step]:
        x, y = xy[i]
        c = tuple(rgb[i].tolist())
        im.putpixel((int(x), int(y)), c)
        if not env_mask[i] and 1 <= x < size[0] - 1 and 1 <= y < size[1] - 1:
            im.putpixel((int(x + 1), int(y)), c)
            im.putpixel((int(x), int(y + 1)), c)
    draw.text((8, 8), title[:76], fill=(10, 10, 10))
    draw.text((8, size[1] - 24), "source-upright render only; geometry gate remains fail-closed", fill=(55, 55, 55))
    return im


def main() -> int:
    created_at = now()
    rows = read_manifest()
    configs = [
        "real_vggt_baseline_only",
        "moderate_offset_surface_completion_true",
        "visible_surface_preserving_infill_true",
        "same_topology_no_semantic",
        "shuffled_smpl_feature",
    ]
    metric_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    first_case = rows[0]["case"]
    panels: list[Image.Image] = []
    transform_info: dict[str, Any] = {}
    for cfg in configs:
        pts, rgb, body, env, env_rgb = load_config(first_case, cfg)
        panels.append(render_source_upright(pts, rgb, f"{first_case} {cfg.replace('_', ' ')}", full_env=(env, env_rgb)))
        _rot, info = source_upright_rotation(pts)
        transform_info[cfg] = info
    compose(panels, 2, BOARDS / "V19800000000000000000_source_upright_full_scene_board.png")
    for row in rows:
        case_scores: dict[str, dict[str, Any]] = {}
        for cfg in configs:
            path = config_path(row["case"], cfg)
            if not path.exists():
                continue
            pts, _rgb, body, _env, _env_rgb = load_config(row["case"], cfg)
            metric, _pairs = adjacency_collision_metric_v4(pts, body)
            case_scores[cfg] = metric
            metric_rows.append({"case": row["case"], "config": cfg, **metric})
        true = case_scores.get("moderate_offset_surface_completion_true")
        if true is None:
            failures.append({"case": row["case"], "reason": "missing_v197_true"})
            continue
        true_score = float(true["combined_topology_volume_score_v4"])
        if bool(true["combined_fail_v4"]):
            failures.append({"case": row["case"], "reason": "v197_true_combined_fail_v4", "true_score": true_score})
        for cfg in ["real_vggt_baseline_only", "visible_surface_preserving_infill_true", "same_topology_no_semantic", "shuffled_smpl_feature"]:
            if cfg in case_scores and float(case_scores[cfg]["combined_topology_volume_score_v4"]) >= true_score * 0.96:
                failures.append(
                    {
                        "case": row["case"],
                        "reason": "control_or_prior_close_or_better_after_upright_render_audit",
                        "control": cfg,
                        "true_score": true_score,
                        "control_score": float(case_scores[cfg]["combined_topology_volume_score_v4"]),
                    }
                )
    write_csv(REPORTS / "V19800000000000000000_source_upright_metric_audit.csv", metric_rows)
    decision = {
        "created_at": created_at,
        "status": "V19800_SOURCE_UPRIGHT_RENDER_AUDIT_FAIL_CLOSED_CONTINUE",
        "mentor_ready": False,
        "external_hard_block": False,
        "source_upright_board": str(BOARDS / "V19800000000000000000_source_upright_full_scene_board.png"),
        "metric_csv": str(REPORTS / "V19800000000000000000_source_upright_metric_audit.csv"),
        "failures": failures,
        "transform_info_first_case": transform_info,
        "face_detail_claim_allowed": False,
        "allowed_face_claim": ALLOWED_FACE_CLAIM,
        "summary": "V198 fixes the mentor-board orientation for visual audit only. It does not override V196/V197 fail-closed geometry and hard-control decisions.",
    }
    write_json(REPORTS / "V19800000000000000000_source_upright_visual_decision.json", decision)
    print(json.dumps({"created_at": created_at, "status": decision["status"], "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
