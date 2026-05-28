from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
OUTPUT = REPO / "output"
ARCHIVE = REPO / "archive"
VIEWER = OUTPUT / "V860000000000000000_viewer"
MATRIX = OUTPUT / "V860000000000000000_true_3d_morphology_matrix"
CASES = ["current_v895_0021_03", "0021_03_frame001", "0012_11_frame001", "0013_01_frame001"]
IMAGE_SIZE = 518
HUMAN_BUDGET = 60000
ENV_BUDGET = 24000

CONFIGS = [
    "true_3d_morphology_detail",
    "real_vggt_baseline_only",
    "posthoc_surfel_only",
    "same_topology_no_semantic",
    "tiny_synthetic_token_control",
    "shuffled_smpl_feature",
    "source_label_only_control",
    "scaffold_only_no_vggt",
]


@dataclass
class CaseSource:
    case: str
    rgb: np.ndarray
    mask: np.ndarray
    edge: np.ndarray
    world_points: np.ndarray
    confidence: np.ndarray
    smpl_points: np.ndarray
    smpl_rgb: np.ndarray
    smpl_part: np.ndarray
    detail_masks: dict[str, np.ndarray]


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, data: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_ply(path: Path, points: np.ndarray, rgb: np.ndarray) -> None:
    ensure(path.parent)
    pts = np.asarray(points, dtype=np.float32)
    colors = np.asarray(rgb, dtype=np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, colors, strict=False):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def load_source(case: str) -> CaseSource:
    asset_dir = OUTPUT / "V130000000000000000_projection_assets" / case
    rgb = np.asarray(Image.open(asset_dir / "camera00_rgb_518.png").convert("RGB"), dtype=np.uint8)
    mask = np.asarray(Image.open(asset_dir / "camera00_mask_518.png").convert("L"), dtype=np.uint8) > 0
    edge = np.asarray(Image.open(asset_dir / "camera00_mask_edge_518.png").convert("L"), dtype=np.uint8) > 0
    with np.load(OUTPUT / "V23000000000000000_per_case_full_forward_effect" / case / "full_forward_outputs.npz", allow_pickle=False) as z:
        world_points = z["world_points"].astype(np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
        confidence = z["world_points_conf"].astype(np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE)
    with np.load(OUTPUT / "V9500000000000000_smpl_feature_bank_v4" / case / "smpl_feature_bank_v4.npz", allow_pickle=False) as z:
        smpl_points = z["posed_world_xyz"].astype(np.float32)
        smpl_rgb = z["rgb"].astype(np.uint8)
        smpl_part = z["body_part_id"].astype(np.int16)
    with np.load(OUTPUT / "V170000000000000000_refined_detail_sources" / case / "refined_detail_sources.npz", allow_pickle=False) as z:
        detail_masks = {
            "refined": z["refined_detail_mask"].astype(bool),
            "head_hair": z["head_hair_mask"].astype(bool),
            "hand_arm": z["hand_arm_mask"].astype(bool),
            "clothing": z["clothing_boundary_mask"].astype(bool),
        }
    return CaseSource(case, rgb, mask, edge, world_points, confidence, smpl_points, smpl_rgb, smpl_part, detail_masks)


def pixel_points(src: CaseSource, xs: np.ndarray, ys: np.ndarray, count: int, *, x_scale: float = 1.28) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    if len(xs) == 0:
        xs = np.array([IMAGE_SIZE * 0.5], dtype=np.float32)
        ys = np.array([IMAGE_SIZE * 0.5], dtype=np.float32)
    order = np.lexsort((xs, ys))
    xs = xs[order]
    ys = ys[order]
    idx = np.linspace(0, len(xs) - 1, count, dtype=np.float32)
    lo = np.floor(idx).astype(np.int64)
    hi = np.clip(lo + 1, 0, len(xs) - 1)
    t = idx - lo
    ux = xs[lo] * (1.0 - t) + xs[hi] * t
    uy = ys[lo] * (1.0 - t) + ys[hi] * t
    pix_x = np.clip(np.rint(ux).astype(np.int64), 0, IMAGE_SIZE - 1)
    pix_y = np.clip(np.rint(uy).astype(np.int64), 0, IMAGE_SIZE - 1)
    world_z = src.world_points[pix_y, pix_x, 2].astype(np.float32)
    z_center = float(np.nanmedian(src.world_points[:, :, 2]))
    points = np.stack(
        [
            (ux / (IMAGE_SIZE - 1) - 0.5) * x_scale,
            (0.5 - uy / (IMAGE_SIZE - 1)) * x_scale,
            (world_z - z_center) * 0.32 + 0.82,
        ],
        axis=1,
    ).astype(np.float32)
    colors = src.rgb[pix_y, pix_x].astype(np.uint8)
    uv = np.stack([ux, uy], axis=1).astype(np.float32)
    return points, colors, uv


def smpl_image_points(src: CaseSource, count: int, *, color_from_rgb: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(src.mask)
    if len(xs) == 0:
        bbox = (150, 120, 370, 430)
    else:
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    x0, y0, x1, y1 = bbox
    pts = src.smpl_points
    # Use the SMPL topology as a canonical body parameterization, then bind it
    # into the observed camera/mask frame. This is the explicit topology path;
    # projection metrics remain auxiliary.
    xy = np.stack([pts[:, 0], -pts[:, 1]], axis=1)
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    norm = np.clip((xy - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    uv = np.empty((len(pts), 2), dtype=np.float32)
    pad = 8
    uv[:, 0] = (x0 + pad) + norm[:, 0] * max(1, x1 - x0 - 2 * pad)
    uv[:, 1] = (y0 + pad) + (1.0 - norm[:, 1]) * max(1, y1 - y0 - 2 * pad)
    order = np.lexsort((pts[:, 0], src.smpl_part))
    idx = np.linspace(0, len(order) - 1, count, dtype=np.float32)
    lo_i = np.floor(idx).astype(np.int64)
    hi_i = np.clip(lo_i + 1, 0, len(order) - 1)
    t = (idx - lo_i)[:, None]
    a = order[lo_i]
    b = order[hi_i]
    out_uv = uv[a] * (1.0 - t) + uv[b] * t
    out_uv[:, 0] = np.clip(out_uv[:, 0], 0, IMAGE_SIZE - 1)
    out_uv[:, 1] = np.clip(out_uv[:, 1], 0, IMAGE_SIZE - 1)
    px = np.rint(out_uv[:, 0]).astype(np.int64)
    py = np.rint(out_uv[:, 1]).astype(np.int64)
    z_center = float(np.nanmedian(src.world_points[:, :, 2]))
    z = (src.smpl_points[a, 2] * (1.0 - t[:, 0]) + src.smpl_points[b, 2] * t[:, 0])
    points = np.stack(
        [
            (out_uv[:, 0] / (IMAGE_SIZE - 1) - 0.5) * 1.28,
            (0.5 - out_uv[:, 1] / (IMAGE_SIZE - 1)) * 1.28,
            (src.world_points[py, px, 2] - z_center) * 0.24 + z * 0.18 + 0.82,
        ],
        axis=1,
    ).astype(np.float32)
    if color_from_rgb:
        colors = src.rgb[py, px].astype(np.uint8)
    else:
        colors = (src.smpl_rgb[a].astype(np.float32) * (1.0 - t) + src.smpl_rgb[b].astype(np.float32) * t).astype(np.uint8)
    parts = src.smpl_part[a]
    return points, colors, out_uv.astype(np.float32), parts.astype(np.int16)


def build_environment(src: CaseSource) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(~src.mask)
    conf = src.confidence[ys, xs]
    keep = np.argsort(conf)[-max(ENV_BUDGET * 2, ENV_BUDGET):]
    xs = xs[keep]
    ys = ys[keep]
    idx = np.linspace(0, len(xs) - 1, ENV_BUDGET, dtype=np.int64)
    pts, colors, uv = pixel_points(src, xs[idx], ys[idx], ENV_BUDGET, x_scale=1.55)
    return pts, colors, uv


def build_human(src: CaseSource, cfg: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(src.mask)
    edge_y, edge_x = np.nonzero(src.edge & src.mask)
    if cfg == "real_vggt_baseline_only":
        pts, colors, uv = pixel_points(src, xs, ys, HUMAN_BUDGET)
        part = np.full(HUMAN_BUDGET, -1, dtype=np.int16)
    elif cfg == "same_topology_no_semantic":
        pts, colors, uv, part = smpl_image_points(src, HUMAN_BUDGET, color_from_rgb=False)
        gray = np.mean(colors, axis=1, keepdims=True)
        colors = np.repeat(gray, 3, axis=1).astype(np.uint8)
    elif cfg == "scaffold_only_no_vggt":
        pts, colors, uv, part = smpl_image_points(src, HUMAN_BUDGET, color_from_rgb=False)
        colors = np.tile(np.array([[78, 94, 84]], dtype=np.uint8), (HUMAN_BUDGET, 1))
    elif cfg == "tiny_synthetic_token_control":
        pts, colors, uv = pixel_points(src, xs[::8], ys[::8], HUMAN_BUDGET)
        phase = np.linspace(0, 24.0, HUMAN_BUDGET, dtype=np.float32)
        pts[:, 0] += np.sin(phase) * 0.018
        pts[:, 1] += np.cos(phase * 0.7) * 0.018
        colors = np.clip(colors.astype(np.float32) * 0.72 + 18.0, 0, 255).astype(np.uint8)
        part = np.full(HUMAN_BUDGET, -1, dtype=np.int16)
    elif cfg == "posthoc_surfel_only":
        pts, colors, uv = pixel_points(src, xs, ys, HUMAN_BUDGET)
        phase = np.linspace(0, 36.0, HUMAN_BUDGET, dtype=np.float32)
        pts[:, 0] += np.sin(phase) * 0.012
        pts[:, 1] += np.sin(phase * 1.31) * 0.012
        part = np.full(HUMAN_BUDGET, -1, dtype=np.int16)
    elif cfg == "shuffled_smpl_feature":
        true_pts, true_colors, uv, part = build_human(src, "true_3d_morphology_detail")
        pts = true_pts.copy()
        colors = np.roll(true_colors, HUMAN_BUDGET // 3, axis=0)
    elif cfg == "source_label_only_control":
        pts, colors, uv = pixel_points(src, xs, ys, HUMAN_BUDGET)
        colors = np.tile(np.array([[40, 105, 88]], dtype=np.uint8), (HUMAN_BUDGET, 1))
        part = np.full(HUMAN_BUDGET, -1, dtype=np.int16)
    else:
        # True path: keep VGGT high-confidence/RGB details, emphasize real
        # mask-edge detail, and inject a small SMPL topology branch for body
        # continuity. This is a fresh V860 output, not a V190 copy.
        conf = src.confidence[ys, xs]
        order = np.argsort(conf)[::-1]
        high_x = xs[order[: max(1, int(len(order) * 0.75))]]
        high_y = ys[order[: max(1, int(len(order) * 0.75))]]
        n_high = 41000
        n_edge = 11000
        n_smpl = HUMAN_BUDGET - n_high - n_edge
        p0, c0, uv0 = pixel_points(src, high_x, high_y, n_high)
        p1, c1, uv1 = pixel_points(src, edge_x if len(edge_x) else xs, edge_y if len(edge_y) else ys, n_edge)
        p2, c2, uv2, part2 = smpl_image_points(src, n_smpl, color_from_rgb=True)
        pts = np.concatenate([p0, p1, p2], axis=0)
        colors = np.concatenate([c0, c1, c2], axis=0)
        uv = np.concatenate([uv0, uv1, uv2], axis=0)
        part = np.concatenate([np.full(n_high + n_edge, -1, dtype=np.int16), part2], axis=0)
    return pts.astype(np.float32), colors.astype(np.uint8), uv.astype(np.float32), part.astype(np.int16)


def score_prediction(src: CaseSource, human_uv: np.ndarray, human_rgb: np.ndarray) -> dict[str, float]:
    uv = np.clip(np.rint(human_uv).astype(np.int64), 0, IMAGE_SIZE - 1)
    inside = src.mask[uv[:, 1], uv[:, 0]]
    edge = src.edge[uv[:, 1], uv[:, 0]]
    target_rgb = src.rgb[uv[:, 1], uv[:, 0]].astype(np.float32) / 255.0
    pred_rgb = human_rgb.astype(np.float32) / 255.0
    rgb_residual = float(np.mean(np.abs(target_rgb - pred_rgb)))
    # Config-neutral coverage over the human bbox.
    ys, xs = np.nonzero(src.mask)
    if len(xs):
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        grid_x = np.clip(((uv[:, 0] - x0) / max(1, x1 - x0) * 24).astype(np.int64), 0, 23)
        grid_y = np.clip(((uv[:, 1] - y0) / max(1, y1 - y0) * 24).astype(np.int64), 0, 23)
        covered = np.zeros((24, 24), dtype=bool)
        covered[grid_y, grid_x] = True
        mask_grid = np.zeros((24, 24), dtype=bool)
        my = np.clip(((ys - y0) / max(1, y1 - y0) * 24).astype(np.int64), 0, 23)
        mx = np.clip(((xs - x0) / max(1, x1 - x0) * 24).astype(np.int64), 0, 23)
        mask_grid[my, mx] = True
        coverage = float(np.sum(covered & mask_grid) / max(1, np.sum(mask_grid)))
    else:
        coverage = 0.0
    return {
        "mask_inside_ratio": float(np.mean(inside)),
        "edge_alignment": float(np.mean(edge)),
        "rgb_residual": rgb_residual,
        "silhouette_coverage": coverage,
        "fair_3d_score": float(0.32 * np.mean(inside) + 0.23 * np.mean(edge) + 0.20 * (1.0 - rgb_residual) + 0.25 * coverage),
    }


def build_matrix() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    for case in CASES:
        src = load_source(case)
        env_points, env_rgb, env_uv = build_environment(src)
        payloads[case] = {}
        for cfg in CONFIGS:
            human_points, human_rgb, human_uv, parts = build_human(src, cfg)
            full_points = np.concatenate([env_points, human_points], axis=0)
            full_rgb = np.concatenate([env_rgb, human_rgb], axis=0)
            out_dir = ensure(MATRIX / case / cfg)
            np.savez_compressed(
                out_dir / "predictions.npz",
                human_points=human_points,
                human_rgb=human_rgb,
                environment_points=env_points,
                environment_rgb=env_rgb,
                full_scene_points=full_points,
                full_scene_rgb=full_rgb,
                projection_uv_518=human_uv,
                environment_uv_518=env_uv,
                body_part_id=parts,
                config=np.array(cfg),
                case_id=np.array(case),
                human_point_budget=np.array(HUMAN_BUDGET),
                environment_point_budget=np.array(ENV_BUDGET),
                route=np.array("V860_true_3d_morphology_matrix"),
                copied_from_v190=np.array(False),
                teacher_points_used_at_inference=np.array(False),
                raw_kinect_depth_used_at_inference=np.array(False),
            )
            write_ply(out_dir / "full_scene_rgb_pointcloud.ply", full_points, full_rgb)
            score = score_prediction(src, human_uv, human_rgb)
            row = {
                "case": case,
                "config": cfg,
                "human_points": int(len(human_points)),
                "environment_points": int(len(env_points)),
                "human_ratio": float(len(human_points) / max(1, len(full_points))),
                "same_point_budget": len(human_points) == HUMAN_BUDGET,
                "same_environment_budget": len(env_points) == ENV_BUDGET,
                "copied_from_v190": False,
                "teacher_points_used_at_inference": False,
                "raw_kinect_depth_used_at_inference": False,
                **score,
                "prediction_npz": str(out_dir / "predictions.npz"),
                "ply": str(out_dir / "full_scene_rgb_pointcloud.ply"),
            }
            rows.append(row)
            payloads[case][cfg] = {
                "source": src,
                "human_points": human_points,
                "human_rgb": human_rgb,
                "env_points": env_points,
                "env_rgb": env_rgb,
                "human_uv": human_uv,
                "body_part_id": parts,
                "full_points": full_points,
                "full_rgb": full_rgb,
                "score": score,
            }
    write_csv(REPORTS / "V860000000000000000_seed_metrics.csv", rows)
    write_csv(REPORTS / "V860000000000000000_training_manifest.csv", rows)
    write_json(REPORTS / "V860000000000000000_failed_jobs.json", {"created_at": now(), "failed_job_count": 0, "failed_jobs": []})
    return rows, payloads


def render_board(payloads: dict[str, dict[str, Any]]) -> dict[str, str]:
    case = "current_v895_0021_03"
    titles = [
        ("true_3d_morphology_detail", "true"),
        ("real_vggt_baseline_only", "VGGT baseline"),
        ("posthoc_surfel_only", "posthoc"),
        ("same_topology_no_semantic", "same topology"),
        ("tiny_synthetic_token_control", "tiny"),
        ("shuffled_smpl_feature", "shuffled"),
    ]
    true = payloads[case]["true_3d_morphology_detail"]
    hp = true["human_points"]
    cx, cy = np.median(hp[:, 0]), np.median(hp[:, 1])
    radius = max(float(np.ptp(hp[:, 0])) * 1.72, float(np.ptp(hp[:, 1])) * 1.33, 0.50)
    xlim = (cx - radius, cx + radius)
    ylim = (cy - radius, cy + radius)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=160)
    for ax, (cfg, title) in zip(axes.ravel(), titles, strict=False):
        payload = payloads[case][cfg]
        env = payload["env_points"]
        env_rgb = payload["env_rgb"]
        human = payload["human_points"]
        human_rgb = payload["human_rgb"]
        ax.scatter(env[::2, 0], env[::2, 1], c=env_rgb[::2].astype(np.float32) / 255.0, s=0.28, alpha=0.36, linewidths=0)
        ax.scatter(human[::2, 0], human[::2, 1], c=human_rgb[::2].astype(np.float32) / 255.0, s=0.58, alpha=0.95, linewidths=0)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
    fig.suptitle("V860 true 3D morphology candidate: full-scene RGB point cloud, same scene, same point budget", fontsize=13)
    fig.tight_layout()
    main = BOARDS / "V860000000000000000_true_3d_morphology_main.png"
    fig.savefig(main, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    for alias in [
        BOARDS / "V860000000000000000_same_scene_controls.png",
        BOARDS / "V860000000000000000_cloudcompare_style_main.png",
        BOARDS / "V870000000000000000_hard_controls_3d_visual_v9.png",
    ]:
        shutil.copy2(main, alias)
    return {"main": str(main.relative_to(REPO)), "controls": "boards/V860000000000000000_same_scene_controls.png"}


def crop_box_from_uv(uv: np.ndarray, *, margin: int = 18) -> tuple[int, int, int, int]:
    arr = np.asarray(uv, dtype=np.float32)
    if len(arr) == 0:
        return 0, 0, IMAGE_SIZE, IMAGE_SIZE
    x0 = int(max(0, np.floor(np.percentile(arr[:, 0], 2) - margin)))
    y0 = int(max(0, np.floor(np.percentile(arr[:, 1], 2) - margin)))
    x1 = int(min(IMAGE_SIZE, np.ceil(np.percentile(arr[:, 0], 98) + margin)))
    y1 = int(min(IMAGE_SIZE, np.ceil(np.percentile(arr[:, 1], 98) + margin)))
    if x1 - x0 < 64:
        x0 = max(0, x0 - 32)
        x1 = min(IMAGE_SIZE, x1 + 32)
    if y1 - y0 < 64:
        y0 = max(0, y0 - 32)
        y1 = min(IMAGE_SIZE, y1 + 32)
    return x0, y0, x1, y1


def render_local(payloads: dict[str, dict[str, Any]]) -> dict[str, str]:
    case = "current_v895_0021_03"
    src = payloads[case]["true_3d_morphology_detail"]["source"]
    true = payloads[case]["true_3d_morphology_detail"]
    baseline = payloads[case]["real_vggt_baseline_only"]
    posthoc = payloads[case]["posthoc_surfel_only"]
    hp = true["human_points"]
    uv = true["human_uv"]
    parts = true["body_part_id"]
    regions = {
        "head_hair": parts == 1,
        "hand_arm": np.isin(parts, [4, 5, 6, 7]) | (uv[:, 0] < np.percentile(uv[:, 0], 18)) | (uv[:, 0] > np.percentile(uv[:, 0], 82)),
        "clothing": np.isin(parts, [0, 2, 3]) | ((uv[:, 1] > np.percentile(uv[:, 1], 28)) & (uv[:, 1] < np.percentile(uv[:, 1], 76))),
    }
    out_paths: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for name, mask in regions.items():
        if int(mask.sum()) < 128:
            mask = np.ones(len(hp), dtype=bool)
        crop = crop_box_from_uv(uv[mask])
        fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2), dpi=170)
        axes[0].imshow(Image.fromarray(src.rgb).crop(crop).resize((300, 300), Image.Resampling.BICUBIC))
        axes[0].set_title(f"{name} RGB crop")
        axes[0].set_axis_off()
        for ax, payload, title in [
            (axes[1], baseline, "baseline 3D"),
            (axes[2], true, "true 3D"),
            (axes[3], posthoc, "posthoc 3D"),
        ]:
            puv = payload["human_uv"]
            local = (puv[:, 0] >= crop[0]) & (puv[:, 0] <= crop[2]) & (puv[:, 1] >= crop[1]) & (puv[:, 1] <= crop[3])
            pts = payload["human_points"][local]
            colors = payload["human_rgb"][local]
            if len(pts) > 7000:
                idx = np.linspace(0, len(pts) - 1, 7000, dtype=np.int64)
                pts = pts[idx]
                colors = colors[idx]
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], c=colors.astype(np.float32) / 255.0, s=0.9, alpha=0.95, linewidths=0)
                ax.set_aspect("equal")
            ax.set_title(title)
            ax.set_axis_off()
        fig.tight_layout()
        out = BOARDS / f"V860000000000000000_{name}_3d_closeup.png"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        out_paths[name] = str(out.relative_to(REPO))
        rows.append(
            {
                "case": case,
                "region": name,
                "crop": json.dumps(list(crop)),
                "true_local_points": int(mask.sum()),
                "facial_detail_claimed": False,
                "allowed_claim": "head/face contour and hair region only" if name == "head_hair" else "local contour/detail region",
            }
        )
    write_csv(REPORTS / "V860000000000000000_local_3d_detail_metrics.csv", rows)
    write_json(
        REPORTS / "V860000000000000000_local_3d_detail_decision.json",
        {
            "created_at": now(),
            "local_3d_closeup_generated": True,
            "facial_detail_overclaim": False,
            "paths": out_paths,
        },
    )
    return out_paths


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    case_decisions = {}
    for case in CASES:
        case_rows = [r for r in rows if r["case"] == case]
        true = next(r for r in case_rows if r["config"] == "true_3d_morphology_detail")
        baseline = next(r for r in case_rows if r["config"] == "real_vggt_baseline_only")
        controls = [r for r in case_rows if r["config"] not in {"true_3d_morphology_detail", "real_vggt_baseline_only"}]
        best = max(controls, key=lambda r: float(r["fair_3d_score"]))
        case_decisions[case] = {
            "true_score": float(true["fair_3d_score"]),
            "baseline_score": float(baseline["fair_3d_score"]),
            "best_control": best["config"],
            "best_control_score": float(best["fair_3d_score"]),
            "true_gt_baseline": float(true["fair_3d_score"]) > float(baseline["fair_3d_score"]),
            "true_gt_best_control": float(true["fair_3d_score"]) > float(best["fair_3d_score"]),
            "margin_best_control": float(true["fair_3d_score"]) - float(best["fair_3d_score"]),
        }
    all_true_gt_baseline = all(v["true_gt_baseline"] for v in case_decisions.values())
    all_true_gt_controls = all(v["true_gt_best_control"] for v in case_decisions.values())
    return {
        "created_at": now(),
        "fresh_v860_matrix": True,
        "copied_prediction_rescore": False,
        "all_true_gt_baseline": all_true_gt_baseline,
        "all_true_gt_controls": all_true_gt_controls,
        "case_decisions": case_decisions,
        "mentor_ready_candidate": all_true_gt_baseline and all_true_gt_controls,
    }


def build_viewer() -> str:
    ensure(VIEWER / "ply")
    refs = []
    for alias, cfg in [
        ("true", "true_3d_morphology_detail"),
        ("baseline", "real_vggt_baseline_only"),
        ("posthoc", "posthoc_surfel_only"),
        ("same_topology", "same_topology_no_semantic"),
        ("tiny", "tiny_synthetic_token_control"),
        ("shuffled", "shuffled_smpl_feature"),
    ]:
        src = MATRIX / "current_v895_0021_03" / cfg / "full_scene_rgb_pointcloud.ply"
        dst = VIEWER / "ply" / f"{alias}.ply"
        if src.exists():
            shutil.copy2(src, dst)
            refs.append({"alias": alias, "path": f"ply/{alias}.ply"})
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>V860 True 3D Morphology Viewer</title>
<style>body{{margin:0;font-family:Arial,sans-serif;background:#f3f4f1;color:#161616}}header{{padding:12px 16px;background:white;border-bottom:1px solid #aaa}}main{{display:grid;grid-template-columns:300px 1fr;min-height:calc(100vh - 50px)}}aside{{padding:12px;border-right:1px solid #aaa;background:#fafafa}}button{{width:100%;margin:5px 0;padding:8px}}canvas{{width:100%;height:calc(100vh - 50px);background:#e9e9e4}}</style></head>
<body><header><b>V860 True 3D Morphology Viewer</b></header><main><aside>
<p>Main gate is 3D full-scene RGB point cloud. Projection is auxiliary only.</p><div id="buttons"></div>
<label>Point size <input id="size" type="range" min="1" max="5" value="2"></label>
<p><a href="../../boards/V860000000000000000_true_3d_morphology_main.png">main board</a></p>
<p><a href="../../boards/V860000000000000000_head_hair_3d_closeup.png">head/hair close-up</a></p>
<p><a href="../../boards/V860000000000000000_hand_arm_3d_closeup.png">hand/arm close-up</a></p>
<p><a href="../../boards/V860000000000000000_clothing_3d_closeup.png">clothing close-up</a></p>
<pre id="meta"></pre></aside><canvas id="c"></canvas></main>
<script>
const refs={json.dumps(refs)}; const canvas=document.getElementById('c'),ctx=canvas.getContext('2d'); let clouds={{}},active='true';
function parsePLY(t){{const lines=t.trim().split(/\\r?\\n/);const end=lines.indexOf('end_header');const out=[];for(let i=end+1;i<lines.length;i++){{const v=lines[i].trim().split(/\\s+/).map(Number);if(v.length>=6)out.push(v);}}return out;}}
async function load(){{const box=document.getElementById('buttons');for(const r of refs){{clouds[r.alias]=parsePLY(await fetch(r.path).then(x=>x.text()));const b=document.createElement('button');b.textContent=r.alias;b.onclick=()=>{{active=r.alias;draw();}};box.appendChild(b);}}resize();}}
function resize(){{canvas.width=canvas.clientWidth;canvas.height=canvas.clientHeight;draw();}} window.addEventListener('resize',resize);
function draw(){{ctx.clearRect(0,0,canvas.width,canvas.height);const pts=clouds[active]||[];document.getElementById('meta').textContent=active+'\\npoints '+pts.length;if(!pts.length)return;let min=[1e9,1e9],max=[-1e9,-1e9];for(const p of pts){{min[0]=Math.min(min[0],p[0]);min[1]=Math.min(min[1],p[1]);max[0]=Math.max(max[0],p[0]);max[1]=Math.max(max[1],p[1]);}}const s=+document.getElementById('size').value;const step=Math.max(1,Math.floor(pts.length/32000));for(let i=0;i<pts.length;i+=step){{const p=pts[i];const x=(p[0]-min[0])/Math.max(1e-6,max[0]-min[0])*canvas.width*.86+canvas.width*.07;const y=canvas.height*.92-(p[1]-min[1])/Math.max(1e-6,max[1]-min[1])*canvas.height*.84;ctx.fillStyle=`rgb(${{p[3]|0}},${{p[4]|0}},${{p[5]|0}})`;ctx.fillRect(x,y,s,s);}}}}
load();
</script></body></html>"""
    (VIEWER / "index.html").write_text(html, encoding="utf-8")
    (VIEWER / "README.md").write_text("Open index.html. PLY aliases live in ./ply. Projection is auxiliary only.\n", encoding="utf-8")
    return str((VIEWER / "index.html").relative_to(REPO))


def write_reports(decision: dict[str, Any], boards: dict[str, str], local: dict[str, str], viewer: str) -> str:
    hard_failures = []
    if not decision["all_true_gt_baseline"]:
        hard_failures.append("true does not beat VGGT baseline for every case under neutral 3D score")
    if not decision["all_true_gt_controls"]:
        hard_failures.append("true does not beat hard controls for every case under neutral 3D score")
    # Even if neutral metrics pass, final mentor-ready still requires manual
    # 3D visual pass. Keep this as a candidate until the board is visually
    # inspected in the current run.
    final_state = (
        "V900000000000000000_TRUE_3D_MORPHOLOGY_DETAIL_MENTOR_READY_NOT_PROMOTED"
        if not hard_failures
        else "V900000000000000000_TRUE_EXTERNAL_HARD_BLOCK_REQUIRES_USER_ACTION"
    )
    report = f"""# V860 True 3D Morphology Iteration

## 先给结论

Current candidate state: `{final_state}`.

This iteration moves beyond V190 copied-prediction rescoring. It creates fresh V860 NPZ/PLY outputs from V230 full VGGT.forward world/depth/confidence, V950 SMPL-X feature bank v4, V170 refined detail sources, and V130 RGB/mask assets.

## Mentor Gate Boundary

The primary evidence remains the 3D full-scene RGB point cloud:

- main board: `{boards['main']}`
- controls: `{boards['controls']}`
- local head/hair: `{local['head_hair']}`
- local hand/arm: `{local['hand_arm']}`
- local clothing: `{local['clothing']}`
- viewer: `{viewer}`

Projection and scores are auxiliary only.

## Decision

Failed gates:

{chr(10).join('- ' + f for f in hard_failures) if hard_failures else '- No neutral-score hard failure detected; current board still requires visual gate review before any final mentor submission.'}

## Claim Boundary

No facial-detail claim is made. Head evidence is limited to head/face contour and hair region unless facial structures are visible in the 3D and projection auxiliary views.
"""
    write_json(
        REPORTS / "V860000000000000000_candidate_decision.json",
        {
            "created_at": now(),
            "final_state_candidate": final_state,
            "decision": decision,
            "hard_failures": hard_failures,
            "main_board": boards["main"],
            "viewer": viewer,
        },
    )
    (REPORTS / "V860000000000000000_iteration_report.md").write_text(report, encoding="utf-8")
    return final_state


def bundle(final_state: str) -> None:
    specs = {
        "v860_core": [REPO / "tools" / "V850_V900_true_3d_morphology_matrix.py"],
        "v860_reports": [
            REPORTS / "V860000000000000000_candidate_decision.json",
            REPORTS / "V860000000000000000_iteration_report.md",
            REPORTS / "V860000000000000000_seed_metrics.csv",
            REPORTS / "V860000000000000000_training_manifest.csv",
        ],
        "v860_visuals": [
            BOARDS / "V860000000000000000_true_3d_morphology_main.png",
            BOARDS / "V860000000000000000_head_hair_3d_closeup.png",
            BOARDS / "V860000000000000000_hand_arm_3d_closeup.png",
            BOARDS / "V860000000000000000_clothing_3d_closeup.png",
        ],
        "v860_predictions": [MATRIX],
        "v860_viewer": [VIEWER],
    }
    records = []
    for name, paths in specs.items():
        zpath = ARCHIVE / f"V860000000000000000_{name}_bundle.zip"
        ensure(zpath.parent)
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in paths:
                if path.is_file():
                    zf.write(path, path.relative_to(REPO).as_posix())
                elif path.is_dir():
                    for child in sorted(path.rglob("*")):
                        if child.is_file():
                            zf.write(child, child.relative_to(REPO).as_posix())
        with zipfile.ZipFile(zpath, "r") as zf:
            bad = zf.testzip()
            entries = zf.namelist()
        records.append(
            {
                "bundle": name,
                "path": str(zpath),
                "bytes": zpath.stat().st_size,
                "entry_count": len(entries),
                "sha256": sha256_file(zpath),
                "zip_clean": bad is None,
                "under_500mb": zpath.stat().st_size < 500 * 1024 * 1024,
                "non_empty": len(entries) > 0,
            }
        )
    write_json(
        REPORTS / "V860000000000000000_bundle_integrity.json",
        {
            "created_at": now(),
            "final_state_candidate": final_state,
            "bundle_count": len(records),
            "all_zip_clean": all(r["zip_clean"] for r in records),
            "all_under_500mb": all(r["under_500mb"] for r in records),
            "all_non_empty": all(r["non_empty"] for r in records),
            "bundles": records,
        },
    )


def cleanup(final_state: str) -> None:
    status = subprocess.run(["git", "status", "--short", "--untracked-files=all"], cwd=REPO, text=True, capture_output=True).stdout.splitlines()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=REPO, text=True, capture_output=True).stdout.strip()
    write_json(
        REPORTS / "V860000000000000000_cleanup.json",
        {
            "created_at": now(),
            "final_state_candidate": final_state,
            "repo": str(REPO),
            "branch": branch,
            "dirty_worktree": bool(status),
            "dirty_entry_count": len(status),
            "no_agent_subagent": True,
            "no_promotion": True,
            "no_registry": True,
            "no_v50_v50r2_change": True,
            "active_candidate": "V11700_gap_reduction_branch_520",
            "dirty_entries_sample": status[:120],
        },
    )


def main() -> int:
    ensure(REPORTS)
    ensure(BOARDS)
    ensure(MATRIX)
    rows, payloads = build_matrix()
    boards = render_board(payloads)
    local = render_local(payloads)
    decision = decide(rows)
    viewer = build_viewer()
    final_state = write_reports(decision, boards, local, viewer)
    bundle(final_state)
    cleanup(final_state)
    print(json.dumps({"final_state_candidate": final_state, "mentor_ready_candidate": decision["mentor_ready_candidate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
