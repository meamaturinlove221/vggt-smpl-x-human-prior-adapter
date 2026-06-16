from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
OUTPUT = REPO / "output"
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
GOALS = REPO / "docs" / "goals"

BASE_ROOT = OUTPUT / "V1400000000000000000_learned_residual_matrix"
V10140_ROOT = OUTPUT / "V10140000000000000000_full_scene_visual_objective_search"

CASE = "0012_11_frame001"
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def as_rgb(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype != np.uint8:
        if out.size and float(np.nanmax(out)) <= 1.5:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure(path.parent)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["name"])
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pred_path(config: str) -> Path:
    if config == "candidate":
        return V10140_ROOT / CASE / "full_scene_objective_best" / "predictions.npz"
    return BASE_ROOT / CASE / config / "predictions.npz"


def pca_metrics(points: np.ndarray) -> dict[str, float]:
    pts = np.asarray(points, dtype=np.float64)
    center = np.mean(pts, axis=0)
    x = pts - center[None]
    cov = (x.T @ x) / max(1, len(x) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    proj = x @ vecs
    ranges = np.ptp(proj, axis=0)
    bbox = np.ptp(pts, axis=0)
    return {
        "bbox_x": float(bbox[0]),
        "bbox_y": float(bbox[1]),
        "bbox_z": float(bbox[2]),
        "bbox_min_over_max": float(np.min(bbox) / max(np.max(bbox), 1e-9)),
        "pca_range_1": float(ranges[0]),
        "pca_range_2": float(ranges[1]),
        "pca_range_3": float(ranges[2]),
        "pca_thickness_ratio": float(ranges[2] / max(ranges[0], 1e-9)),
        "eigen_ratio_small_large": float(vals[2] / max(vals[0], 1e-12)),
        "z_iqr": float(np.quantile(pts[:, 2], 0.75) - np.quantile(pts[:, 2], 0.25)),
        "z_p95_p05": float(np.quantile(pts[:, 2], 0.95) - np.quantile(pts[:, 2], 0.05)),
        "point_count": int(len(pts)),
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
    xy = np.clip(q * np.array([width - 48, height - 74]) + np.array([24, 44]), 0, [width - 1, height - 1]).astype(np.int32)
    return xy, pts[:, 2]


def depth_tint(colors: np.ndarray, depth: np.ndarray, strength: float = 0.35) -> np.ndarray:
    rgb = as_rgb(colors).astype(np.float32)
    d = (depth - np.quantile(depth, 0.03)) / max(np.quantile(depth, 0.97) - np.quantile(depth, 0.03), 1e-9)
    d = np.clip(d, 0.0, 1.0)
    shade = 0.70 + strength * d[:, None]
    cool = np.array([0.88, 0.94, 1.0], dtype=np.float32)
    warm = np.array([1.06, 0.98, 0.90], dtype=np.float32)
    tint = cool[None] * (1.0 - d[:, None]) + warm[None] * d[:, None]
    return np.clip(rgb * shade * tint, 0, 255).astype(np.uint8)


def render_panel(points: np.ndarray, colors: np.ndarray, title: str, rot: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Image.Image:
    width, height = 390, 320
    im = Image.new("RGB", (width, height), (248, 248, 244))
    draw = ImageDraw.Draw(im)
    xy, depth = project(points, rot, lo, hi, (width, height))
    rgb = depth_tint(colors, depth)
    order = np.argsort(depth)
    step = max(1, len(order) // 62000)
    for i in order[::step]:
        x, y = xy[i]
        c = tuple(rgb[i].tolist())
        im.putpixel((int(x), int(y)), c)
        if 1 <= x < width - 1 and 1 <= y < height - 1:
            im.putpixel((int(x + 1), int(y)), c)
            im.putpixel((int(x), int(y + 1)), c)
    draw.text((10, 9), title, fill=(20, 20, 20))
    return im


def make_board(preds: list[tuple[str, dict[str, np.ndarray]]]) -> Path:
    # Oblique view: rotate around vertical and pitch downward, so depth cues are visible.
    rot = rotation_matrix(yaw_deg=-32.0, pitch_deg=58.0, roll_deg=0.0)
    projected = []
    for _, pred in preds:
        pts = np.asarray(pred["full_scene_points"], dtype=np.float32)
        centered = pts - np.mean(pts, axis=0, keepdims=True)
        projected.append(centered @ rot.T)
    all_xy = np.concatenate([p[:, :2] for p in projected], axis=0)
    lo = np.percentile(all_xy, 1, axis=0)
    hi = np.percentile(all_xy, 99, axis=0)
    pad = (hi - lo) * 0.14 + 1e-6
    lo -= pad
    hi += pad
    canvas = Image.new("RGB", (390 * 3, 320 * 2), (255, 255, 255))
    for i, (title, pred) in enumerate(preds):
        panel = render_panel(
            np.asarray(pred["full_scene_points"], dtype=np.float32),
            as_rgb(pred["full_scene_rgb"]),
            title,
            rot,
            lo,
            hi,
        )
        canvas.paste(panel, ((i % 3) * 390, (i // 3) * 320))
    path = BOARDS / f"V10170000000000000000_{CASE}_oblique_depth_pointcloud_audit.png"
    ensure(path.parent)
    canvas.save(path)
    return path


def main() -> int:
    created_at = now()
    configs = [
        ("baseline", "real_vggt_baseline_only"),
        ("candidate", "candidate"),
        ("posthoc", "posthoc_surfel_only"),
        ("same topology", "same_topology_no_semantic"),
        ("tiny", "tiny_synthetic_token_control"),
        ("shuffled", "shuffled_smpl_feature"),
    ]
    rows: list[dict[str, Any]] = []
    preds: list[tuple[str, dict[str, np.ndarray]]] = []
    missing: list[str] = []
    for label, config in configs:
        path = pred_path(config)
        if not path.exists():
            missing.append(str(path))
            continue
        pred = load_npz(path)
        preds.append((f"{CASE} {label}", pred))
        human = np.asarray(pred["human_points"], dtype=np.float32)
        full = np.asarray(pred["full_scene_points"], dtype=np.float32)
        human_metrics = {f"human_{k}": v for k, v in pca_metrics(human).items()}
        full_metrics = {f"full_{k}": v for k, v in pca_metrics(full).items()}
        flatness_flag = human_metrics["human_pca_thickness_ratio"] < 0.055 or human_metrics["human_eigen_ratio_small_large"] < 0.004
        rows.append(
            {
                "case": CASE,
                "label": label,
                "config": config,
                **human_metrics,
                **full_metrics,
                "flatness_flag": bool(flatness_flag),
            }
        )

    board = make_board(preds) if preds else ""
    audit_csv = REPORTS / "V10170000000000000000_flatness_geometry_audit.csv"
    write_csv(audit_csv, rows)

    candidate_row = next((r for r in rows if r["label"] == "candidate"), None)
    baseline_row = next((r for r in rows if r["label"] == "baseline"), None)
    rendering_is_2d = True
    geometry_flat = bool(candidate_row and candidate_row["flatness_flag"])
    candidate_not_more_3d_than_baseline = bool(
        candidate_row
        and baseline_row
        and candidate_row["human_pca_thickness_ratio"] <= baseline_row["human_pca_thickness_ratio"] * 1.05
    )
    status = (
        "V10170_GEOMETRY_FLATNESS_REPRESENTATION_REPAIR_REQUIRED"
        if geometry_flat or candidate_not_more_3d_than_baseline
        else "V10170_RENDERING_DEPTH_CUE_REPAIR_REQUIRED"
    )

    next_goal = GOALS / "V10180000000000000000_auto_evolved_depth_cue_or_geometry_repair_route.md"
    ensure(next_goal.parent)
    next_goal.write_text(
        f"""# V10180 Auto-Evolved Depth-Cue Or Geometry Repair Route

Created: {created_at}

V10170 audits the user's observation that the V10150 board looks too 2D.

Rules:
- If the geometry is flat or not more 3D than baseline, return to representation/geometry repair.
- If geometry is acceptable but the board is 2D because of renderer projection, rebuild mentor boards with oblique view, depth shading, and side/local 3D views.
- Do not claim mentor-ready from render-only repair.
- Face detail remains not applicable; allowed claim is head/back-head contour and hair region only.
""",
        encoding="utf-8",
    )

    decision = {
        "created_at": created_at,
        "status": status,
        "user_observation": "current board looks too 2D and lacks point-cloud depth feel",
        "audit_csv": str(audit_csv),
        "board": str(board),
        "missing_assets": missing,
        "rendering_is_2d": rendering_is_2d,
        "v10150_renderer_issue": "V10150 projected points[:, :2] directly, so the board is effectively orthographic XY with weak depth cues.",
        "candidate_geometry_flat": geometry_flat,
        "candidate_not_more_3d_than_baseline": candidate_not_more_3d_than_baseline,
        "mentor_ready": False,
        "mentor_visual_pass": False,
        "external_hard_block": False,
        "facial_detail_target_applicable": False,
        "face_detail_claim_allowed": False,
        "allowed_face_claim": ALLOWED_FACE_CLAIM,
        "next_goal": str(next_goal),
        "next_action": (
            "Route to representation/geometry repair; render repair alone cannot satisfy mentor visual gate."
            if geometry_flat or candidate_not_more_3d_than_baseline
            else "Rebuild boards with oblique/depth-cued rendering, then rerun mentor visual gate."
        ),
    }
    write_json(REPORTS / "V10170000000000000000_flatness_and_depth_render_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
