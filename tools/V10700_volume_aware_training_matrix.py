from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
OUTPUT = REPO / "output"
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
VIEWER = REPO / "viewer"
TRAINING_MANIFEST = REPORTS / "V10210000000000000000_training_asset_manifest.csv"
MATRIX_ROOT = OUTPUT / "V10700000000000000000_volume_aware_training_matrix"
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"

sys.path.insert(0, str(REPO))
from models.v105_volume_aware_visible_morphology_student import (  # noqa: E402
    VolumeAwareVisibleMorphologyConfig,
    VolumeAwareVisibleMorphologyStudent,
)


CONFIGS = [
    "volume_aware_true",
    "real_vggt_baseline_only",
    "posthoc_surfel_only",
    "same_topology_no_semantic",
    "tiny_synthetic_token_control",
    "shuffled_smpl_feature",
    "thickness_only_control",
    "baseline_preservation_only",
]


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


def read_manifest() -> list[dict[str, str]]:
    with TRAINING_MANIFEST.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def as_rgb(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype != np.uint8:
        if out.size and np.issubdtype(out.dtype, np.number) and float(np.nanmax(out)) <= 1.5:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    center = np.mean(pts, axis=0)
    x = pts - center[None]
    cov = (x.T @ x) / max(1, len(x) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    proj = x @ vecs
    return center, vals, vecs, proj


def geom(points: np.ndarray) -> dict[str, float | int]:
    _center, vals, _vecs, proj = pca_frame(points)
    ranges = np.ptp(proj, axis=0)
    bbox = np.ptp(points, axis=0)
    return {
        "point_count": int(len(points)),
        "bbox_x": float(bbox[0]),
        "bbox_y": float(bbox[1]),
        "bbox_z": float(bbox[2]),
        "z_range": float(bbox[2]),
        "pca_range_1": float(ranges[0]),
        "pca_range_2": float(ranges[1]),
        "pca_range_3": float(ranges[2]),
        "pca_thickness_ratio": float(ranges[2] / max(ranges[0], 1e-9)),
        "eigen_ratio_small_large": float(vals[2] / max(vals[0], 1e-12)),
    }


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    ry = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]])
    return rz @ rx @ ry


def project(points: np.ndarray, rot: np.ndarray, lo: np.ndarray, hi: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    pts = (points - np.mean(points, axis=0, keepdims=True)) @ rot.T
    q = (pts[:, :2] - lo[None]) / (hi[None] - lo[None] + 1e-9)
    q[:, 1] = 1.0 - q[:, 1]
    width, height = size
    xy = np.clip(q * np.array([width - 54, height - 82]) + np.array([27, 48]), 0, [width - 1, height - 1]).astype(np.int32)
    return xy, pts[:, 2]


def depth_tint(colors: np.ndarray, depth: np.ndarray) -> np.ndarray:
    rgb = as_rgb(colors).astype(np.float32)
    d = (depth - np.quantile(depth, 0.03)) / max(np.quantile(depth, 0.97) - np.quantile(depth, 0.03), 1e-9)
    d = np.clip(d, 0.0, 1.0)
    shade = 0.66 + 0.38 * d[:, None]
    cool = np.array([0.82, 0.92, 1.04], dtype=np.float32)
    warm = np.array([1.08, 1.00, 0.86], dtype=np.float32)
    return np.clip(rgb * shade * (cool[None] * (1 - d[:, None]) + warm[None] * d[:, None]), 0, 255).astype(np.uint8)


def render_panel(points: np.ndarray, colors: np.ndarray, title: str, rot: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Image.Image:
    size = (420, 330)
    im = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(im)
    xy, depth = project(points, rot, lo, hi, size)
    rgb = depth_tint(colors, depth)
    order = np.argsort(depth)
    step = max(1, len(order) // 70000)
    for i in order[::step]:
        x, y = xy[i]
        c = tuple(rgb[i].tolist())
        im.putpixel((int(x), int(y)), c)
        if 1 <= x < size[0] - 1 and 1 <= y < size[1] - 1:
            im.putpixel((int(x + 1), int(y)), c)
            im.putpixel((int(x), int(y + 1)), c)
    draw.text((10, 9), title, fill=(15, 15, 15))
    return im


def bounds(items: list[tuple[str, np.ndarray, np.ndarray]], rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = []
    for _title, pts, _rgb in items:
        xy.append(((pts - np.mean(pts, axis=0, keepdims=True)) @ rot.T)[:, :2])
    all_xy = np.concatenate(xy, axis=0)
    lo = np.percentile(all_xy, 1, axis=0)
    hi = np.percentile(all_xy, 99, axis=0)
    pad = (hi - lo) * 0.16 + 1e-6
    return lo - pad, hi + pad


def compose(panels: list[Image.Image], cols: int, path: Path) -> None:
    ensure(path.parent)
    w, h = panels[0].size
    canvas = Image.new("RGB", (w * cols, h * int(math.ceil(len(panels) / cols))), (255, 255, 255))
    for i, panel in enumerate(panels):
        canvas.paste(panel, ((i % cols) * w, (i // cols) * h))
    canvas.save(path)


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    ensure(path.parent)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, as_rgb(colors), strict=False):
            f.write(f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def control_path(row: dict[str, str], config: str) -> Path:
    if config == "real_vggt_baseline_only":
        return Path(row["baseline_path"])
    return Path(row[f"control_{config}_path"])


def build_batch(row: dict[str, str], points: int = 3072) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], np.ndarray]:
    base = load_npz(Path(row["baseline_path"]))
    graph = load_npz(Path(row["graph_path"]))
    weak_npz = load_npz(OUTPUT / "V10400000000000000000_weak_volume_regions" / row["case"] / "weak_volume_regions.npz")
    hp = np.asarray(base["human_points"], dtype=np.float32)
    rgb = np.asarray(base["human_rgb"], dtype=np.float32) / 255.0
    score = np.asarray(graph["mentor_weak_region_score"], dtype=np.float32)
    weak = np.asarray(weak_npz["weak_volume_region_mask"], dtype=bool).astype(np.float32)
    no_change = np.asarray(graph["no_change_mask"], dtype=bool)
    body = np.asarray(graph["geometry_body_part_id"], dtype=np.int64)
    conf = np.asarray(graph["mentor_smpl_confidence"], dtype=np.float32)
    order = np.argsort(-(weak + score * 0.15 + conf * 0.03))
    idx = order[:points]
    cfg = VolumeAwareVisibleMorphologyConfig()
    smpl = np.zeros((len(idx), cfg.smpl_feature_dim), dtype=np.float32)
    smpl[:, :8] = np.eye(8, dtype=np.float32)[np.clip(body[idx], 0, 7)]
    smpl[:, 8] = score[idx]
    smpl[:, 9] = conf[idx]
    batch = {
        "anchor_xyz": torch.from_numpy(hp[idx][None]),
        "anchor_rgb": torch.from_numpy(rgb[idx][None]),
        "confidence": torch.from_numpy(conf[idx][None]),
        "weak_region": torch.from_numpy(np.maximum(weak[idx], score[idx] * 0.25)[None]),
        "anchor_features": torch.zeros(1, len(idx), cfg.anchor_feature_dim),
        "smpl_features": torch.from_numpy(smpl[None]),
        "vggt_token_context": torch.zeros(1, cfg.token_dim),
    }
    normal = np.asarray(weak_npz["pca_axes"], dtype=np.float32)[:, 2]
    anchor = hp[idx]
    sign = np.sign((anchor - anchor.mean(axis=0, keepdims=True)) @ normal)
    sign[sign == 0] = 1.0
    target = anchor + normal[None] * sign[:, None] * (0.010 + 0.017 * np.maximum(weak[idx], score[idx])[:, None])
    targets = {
        "target_xyz": torch.from_numpy(target[None].astype(np.float32)),
        "weak": batch["weak_region"],
        "no_change": torch.from_numpy(no_change[idx][None]),
    }
    return batch, targets, idx


def train_case(row: dict[str, str], steps: int, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = VolumeAwareVisibleMorphologyConfig(max_residual=0.035, max_rgb_delta=0.035, max_shell_offset=0.032)
    model = VolumeAwareVisibleMorphologyStudent(cfg)
    batch, targets, idx = build_batch(row)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        weak = targets["weak"].unsqueeze(-1)
        no_change = targets["no_change"].unsqueeze(-1).float()
        fit = (torch.abs(out["student_xyz"] - targets["target_xyz"]) * (0.2 + weak)).mean()
        preserve = (torch.linalg.norm(out["residual_xyz"], dim=-1, keepdim=True) * no_change).mean()
        shell = -torch.linalg.norm(out["front_shell"] - out["back_shell"], dim=-1).mean() * 0.05
        normal_smooth = out["normal"].diff(dim=1).abs().mean() * 0.01
        rgb = out["rgb_delta"].abs().mean() * 0.05
        loss = fit + preserve * 3.0 + shell + normal_smooth + rgb
        loss.backward()
        optimizer.step()
        if step in {0, steps - 1}:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "fit": float(fit.detach()),
                    "preserve": float(preserve.detach()),
                    "shell_term": float(shell.detach()),
                }
            )
    out = model(batch)
    base = load_npz(Path(row["baseline_path"]))
    human = np.asarray(base["human_points"], dtype=np.float32).copy()
    rgb = as_rgb(base["human_rgb"]).copy()
    pred_points = out["student_xyz"].detach().numpy()[0].astype(np.float32)
    pred_rgb = np.clip(out["student_rgb"].detach().numpy()[0] * 255.0, 0, 255).astype(np.uint8)
    human[idx] = pred_points
    rgb[idx] = pred_rgb
    env = np.asarray(base["environment_points"], dtype=np.float32)
    env_rgb = as_rgb(base["environment_rgb"])
    full = np.concatenate([human, env], axis=0)
    full_rgb = np.concatenate([rgb, env_rgb], axis=0)
    case_dir = ensure(MATRIX_ROOT / row["case"] / "volume_aware_true")
    np.savez_compressed(
        case_dir / "predictions.npz",
        human_points=human,
        human_rgb=rgb,
        environment_points=env,
        environment_rgb=env_rgb,
        full_scene_points=full,
        full_scene_rgb=full_rgb,
        volume_residual_indices=idx.astype(np.int32),
        model_owned_student_output=np.array(True),
        teacher_points_used_at_inference=np.array(False),
        raw_kinect_depth_used_at_inference=np.array(False),
        facial_detail_target_applicable=np.array(False),
        face_detail_claim_allowed=np.array(False),
        allowed_face_claim=np.array(ALLOWED_FACE_CLAIM),
        config=np.array("volume_aware_true"),
        case_id=np.array(row["case"]),
    )
    write_ply(case_dir / "full_scene_rgb_pointcloud.ply", full, full_rgb)
    return {
        "case": row["case"],
        "config": "volume_aware_true",
        "steps": steps,
        "seed": seed,
        "prediction": str(case_dir / "predictions.npz"),
        "ply": str(case_dir / "full_scene_rgb_pointcloud.ply"),
        "loss_history": json.dumps(history),
        "human_pca_thickness_ratio": geom(human)["pca_thickness_ratio"],
        "human_z_range": geom(human)["z_range"],
        "model_owned_student_output": True,
        "teacher_points_used_at_inference": False,
        "raw_kinect_depth_used_at_inference": False,
        "mentor_ready": False,
    }


def copy_controls(row: dict[str, str]) -> list[dict[str, Any]]:
    out = []
    for config in CONFIGS[1:]:
        if config in {"thickness_only_control", "baseline_preservation_only"}:
            src = Path(row["baseline_path"])
        else:
            src = control_path(row, config)
        pred = load_npz(src)
        human = np.asarray(pred["human_points"], dtype=np.float32)
        rgb = as_rgb(pred["human_rgb"])
        env = np.asarray(pred["environment_points"], dtype=np.float32)
        env_rgb = as_rgb(pred["environment_rgb"])
        if config == "thickness_only_control":
            _center, _vals, vecs, proj = pca_frame(human)
            normal = vecs[:, 2]
            sign = np.sign(proj[:, 2])
            sign[sign == 0] = 1.0
            human = human + normal[None] * sign[:, None] * 0.018
        full = np.concatenate([human, env], axis=0)
        full_rgb = np.concatenate([rgb, env_rgb], axis=0)
        case_dir = ensure(MATRIX_ROOT / row["case"] / config)
        np.savez_compressed(
            case_dir / "predictions.npz",
            human_points=human,
            human_rgb=rgb,
            environment_points=env,
            environment_rgb=env_rgb,
            full_scene_points=full,
            full_scene_rgb=full_rgb,
            model_owned_student_output=np.array(config == "baseline_preservation_only"),
            teacher_points_used_at_inference=np.array(False),
            raw_kinect_depth_used_at_inference=np.array(False),
            facial_detail_target_applicable=np.array(False),
            face_detail_claim_allowed=np.array(False),
            allowed_face_claim=np.array(ALLOWED_FACE_CLAIM),
            config=np.array(config),
            case_id=np.array(row["case"]),
        )
        write_ply(case_dir / "full_scene_rgb_pointcloud.ply", full, full_rgb)
        met = geom(human)
        out.append(
            {
                "case": row["case"],
                "config": config,
                "prediction": str(case_dir / "predictions.npz"),
                "ply": str(case_dir / "full_scene_rgb_pointcloud.ply"),
                "human_pca_thickness_ratio": met["pca_thickness_ratio"],
                "human_z_range": met["z_range"],
                "teacher_points_used_at_inference": False,
                "raw_kinect_depth_used_at_inference": False,
                "mentor_ready": False,
            }
        )
    return out


def render_boards(rows: list[dict[str, str]]) -> None:
    row = next(r for r in rows if r["case"] == "0012_11_frame001")
    items = []
    for config in CONFIGS:
        pred = load_npz(MATRIX_ROOT / row["case"] / config / "predictions.npz")
        items.append((config.replace("_", " "), np.asarray(pred["full_scene_points"], dtype=np.float32), as_rgb(pred["full_scene_rgb"])))
    rot = rotation_matrix(-34, 58, 0)
    lo, hi = bounds(items, rot)
    panels = [render_panel(points, rgb, title, rot, lo, hi) for title, points, rgb in items]
    compose(panels, 4, BOARDS / "V10800000000000000000_same_scene_baseline_true_controls_oblique.png")
    compose([panels[0]], 1, BOARDS / "V10800000000000000000_advisor_human_main_full_scene_oblique.png")
    side_rot = rotation_matrix(88, 58, 0)
    slo, shi = bounds(items[:3], side_rot)
    side_panels = [render_panel(points, rgb, title + " side-depth", side_rot, slo, shi) for title, points, rgb in items[:3]]
    compose(side_panels, 3, BOARDS / "V10800000000000000000_turntable_side_depth_thickness.png")
    # Local visible-region board: crop by masks but keep 3D rendering, not projection-only.
    graph = load_npz(Path(row["graph_path"]))
    pred_true = load_npz(MATRIX_ROOT / row["case"] / "volume_aware_true" / "predictions.npz")
    pred_base = load_npz(MATRIX_ROOT / row["case"] / "real_vggt_baseline_only" / "predictions.npz")
    local_panels = []
    masks = [
        ("head hair contour", "head_hair_contour_mask"),
        ("shoulder neck", "shoulder_neck_mask"),
        ("hand arm endpoint", "hand_arm_endpoint_mask"),
        ("clothing boundary", "clothing_torso_boundary_mask"),
        ("leg foot", "leg_foot_morphology_mask"),
    ]
    local_rot = rotation_matrix(-26, 62, 0)
    for name, key in masks:
        mask = np.asarray(graph[key], dtype=bool)
        for label, pred in [("baseline", pred_base), ("true", pred_true)]:
            pts = np.asarray(pred["human_points"], dtype=np.float32)[mask]
            rgb = as_rgb(pred["human_rgb"])[mask]
            if len(pts) == 0:
                continue
            llo, lhi = bounds([(name, pts, rgb)], local_rot)
            local_panels.append(render_panel(pts, rgb, f"{name} {label}", local_rot, llo, lhi, ))
    compose(local_panels, 4, BOARDS / "V11000000000000000000_local_3d_morphology_grid.png")


def gate(metrics: list[dict[str, Any]], created_at: str) -> None:
    by_case: dict[str, dict[str, float]] = {}
    for r in metrics:
        by_case.setdefault(str(r["case"]), {})[str(r["config"])] = float(r["human_pca_thickness_ratio"])
    failures = []
    for case, vals in by_case.items():
        true = vals.get("volume_aware_true")
        if true is None:
            failures.append({"case": case, "reason": "missing true prediction"})
            continue
        for control in [
            "real_vggt_baseline_only",
            "posthoc_surfel_only",
            "same_topology_no_semantic",
            "tiny_synthetic_token_control",
            "shuffled_smpl_feature",
            "thickness_only_control",
        ]:
            if vals.get(control, -1) >= true:
                failures.append(
                    {
                        "case": case,
                        "reason": "control_thickness_not_weaker",
                        "control": control,
                        "control_thickness": vals.get(control),
                        "true_thickness": true,
                    }
                )
    decision = {
        "created_at": created_at,
        "status": "V10800_3D_VISUAL_GATE_FAIL_CLOSED_CONTINUE_TRAINING"
        if failures
        else "V10800_3D_VISUAL_GATE_METRIC_PRECHECK_PASS_REQUIRES_HUMAN_VISUAL",
        "failures": failures[:40],
        "mentor_ready": False,
        "external_hard_block": False,
        "face_detail_claim_allowed": False,
        "allowed_face_claim": ALLOWED_FACE_CLAIM,
        "boards": {
            "advisor_main": str(BOARDS / "V10800000000000000000_advisor_human_main_full_scene_oblique.png"),
            "controls": str(BOARDS / "V10800000000000000000_same_scene_baseline_true_controls_oblique.png"),
            "turntable": str(BOARDS / "V10800000000000000000_turntable_side_depth_thickness.png"),
            "local": str(BOARDS / "V11000000000000000000_local_3d_morphology_grid.png"),
        },
        "note": "This gate is not final; projection and thickness-only evidence cannot override mentor visual requirements.",
    }
    write_json(REPORTS / "V10800000000000000000_3d_visual_gate.json", decision)
    write_json(
        REPORTS / "V11400000000000000000_volume_causality_decision.json",
        {
            "created_at": created_at,
            "status": "V114_VOLUME_CAUSALITY_FAIL_CLOSED" if failures else "V114_VOLUME_CAUSALITY_PRECHECK_PASS_REQUIRES_VISUAL",
            "failures": failures[:40],
            "mentor_ready": False,
            "external_hard_block": False,
        },
    )


def main() -> int:
    created_at = now()
    ensure(MATRIX_ROOT)
    rows = [r for r in read_manifest() if r.get("eligible_for_training_payload") == "True"]
    metrics: list[dict[str, Any]] = []
    failed_jobs: list[dict[str, Any]] = []
    for row in rows:
        try:
            metrics.append(train_case(row, steps=80, seed=10700))
            metrics.extend(copy_controls(row))
        except Exception as exc:  # noqa: BLE001
            failed_jobs.append({"case": row.get("case"), "error": f"{type(exc).__name__}: {exc}"})
    write_csv(REPORTS / "V10700000000000000000_training_manifest.csv", metrics)
    write_csv(REPORTS / "V10700000000000000000_seed_metrics.csv", metrics)
    write_json(REPORTS / "V10700000000000000000_failed_jobs.json", {"created_at": created_at, "failed_jobs": failed_jobs})
    if not failed_jobs:
        render_boards(rows)
        gate(metrics, created_at)
    payload = {
        "created_at": created_at,
        "status": "V10700_LOCAL_TINY_VOLUME_AWARE_MATRIX_COMPLETE_INTERNAL_ONLY" if not failed_jobs else "V10700_LOCAL_TINY_VOLUME_AWARE_MATRIX_HAS_FAILURES",
        "case_count": len(rows),
        "failed_job_count": len(failed_jobs),
        "mentor_ready": False,
        "external_hard_block": False,
        "next": "If visual/hard-control gates fail, continue architecture/training repair rather than return.",
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
