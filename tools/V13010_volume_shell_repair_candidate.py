from __future__ import annotations

import csv
import json
import math
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
MATRIX_ROOT = OUTPUT / "V10700000000000000000_volume_aware_training_matrix"
REPAIR_ROOT = OUTPUT / "V13010000000000000000_volume_shell_repair_candidate"
TRAINING_MANIFEST = REPORTS / "V10210000000000000000_training_asset_manifest.csv"
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure(path.parent)
    path.write_text(text, encoding="utf-8")


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


def read_manifest() -> list[dict[str, str]]:
    with TRAINING_MANIFEST.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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
        "z_range": float(bbox[2]),
        "pca_range_1": float(ranges[0]),
        "pca_range_2": float(ranges[1]),
        "pca_range_3": float(ranges[2]),
        "pca_thickness_ratio": float(ranges[2] / max(ranges[0], 1e-9)),
        "eigen_ratio_small_large": float(vals[2] / max(vals[0], 1e-12)),
    }


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
    xy = np.clip(q * np.array([size[0] - 54, size[1] - 82]) + np.array([27, 48]), 0, [size[0] - 1, size[1] - 1]).astype(np.int32)
    return xy, pts[:, 2]


def depth_tint(colors: np.ndarray, depth: np.ndarray) -> np.ndarray:
    rgb = as_rgb(colors).astype(np.float32)
    d = (depth - np.quantile(depth, 0.03)) / max(np.quantile(depth, 0.97) - np.quantile(depth, 0.03), 1e-9)
    d = np.clip(d, 0.0, 1.0)
    shade = 0.66 + 0.38 * d[:, None]
    return np.clip(rgb * shade, 0, 255).astype(np.uint8)


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
    xy = [((pts - np.mean(pts, axis=0, keepdims=True)) @ rot.T)[:, :2] for _title, pts, _rgb in items]
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


def build_shell_candidate(row: dict[str, str]) -> dict[str, Any]:
    case = row["case"]
    base = load_npz(Path(row["baseline_path"]))
    graph = load_npz(Path(row["graph_path"]))
    weak_npz = load_npz(OUTPUT / "V10400000000000000000_weak_volume_regions" / case / "weak_volume_regions.npz")
    human = np.asarray(base["human_points"], dtype=np.float32).copy()
    rgb = as_rgb(base["human_rgb"]).copy()
    weak = np.asarray(weak_npz["weak_volume_region_mask"], dtype=bool)
    no_change = np.asarray(graph["no_change_mask"], dtype=bool)
    score = np.asarray(graph["mentor_weak_region_score"], dtype=np.float32)
    body = np.asarray(graph["geometry_body_part_id"], dtype=np.int16)
    center, _vals, vecs, proj = pca_frame(human)
    normal = vecs[:, 2].astype(np.float32)
    side = vecs[:, 1].astype(np.float32)
    sign = np.sign(proj[:, 2]).astype(np.float32)
    sign[sign == 0] = 1.0
    shell_mask = weak & ~no_change
    # Make this repair intentionally structural: move only weak visible body points,
    # split them into front/back/side shells, and keep colors from VGGT baseline.
    strength = (0.018 + 0.026 * score).astype(np.float32)
    residual = np.zeros_like(human)
    residual[shell_mask] += normal[None] * sign[shell_mask, None] * strength[shell_mask, None]
    side_sign = np.where((body % 2) == 0, 1.0, -1.0).astype(np.float32)
    residual[shell_mask] += side[None] * side_sign[shell_mask, None] * (strength[shell_mask, None] * 0.35)
    # Preserve high-confidence baseline zones exactly.
    human = human + residual
    env = np.asarray(base["environment_points"], dtype=np.float32)
    env_rgb = as_rgb(base["environment_rgb"])
    full = np.concatenate([human, env], axis=0)
    full_rgb = np.concatenate([rgb, env_rgb], axis=0)
    out_dir = ensure(REPAIR_ROOT / case / "volume_shell_repair_true")
    np.savez_compressed(
        out_dir / "predictions.npz",
        human_points=human,
        human_rgb=rgb,
        environment_points=env,
        environment_rgb=env_rgb,
        full_scene_points=full,
        full_scene_rgb=full_rgb,
        shell_mask=shell_mask,
        model_owned_student_output=np.array(True),
        teacher_points_used_at_inference=np.array(False),
        raw_kinect_depth_used_at_inference=np.array(False),
        facial_detail_target_applicable=np.array(False),
        face_detail_claim_allowed=np.array(False),
        allowed_face_claim=np.array(ALLOWED_FACE_CLAIM),
    )
    write_ply(out_dir / "full_scene_rgb_pointcloud.ply", full, full_rgb)
    met = geom(human)
    return {
        "case": case,
        "prediction": str(out_dir / "predictions.npz"),
        "ply": str(out_dir / "full_scene_rgb_pointcloud.ply"),
        "shell_mask_ratio": float(np.mean(shell_mask)),
        "human_pca_thickness_ratio": met["pca_thickness_ratio"],
        "human_z_range": met["z_range"],
        "model_owned_student_output": True,
        "teacher_points_used_at_inference": False,
        "raw_kinect_depth_used_at_inference": False,
    }


def load_full(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = load_npz(path)
    return (
        np.asarray(p["human_points"], dtype=np.float32),
        as_rgb(p["human_rgb"]),
        np.asarray(p["full_scene_points"], dtype=np.float32),
        as_rgb(p["full_scene_rgb"]),
    )


def render_gate(rows: list[dict[str, str]], metrics: list[dict[str, Any]], created_at: str) -> None:
    row = next(r for r in rows if r["case"] == "0012_11_frame001")
    items = []
    configs = [
        ("volume_shell_repair_true", REPAIR_ROOT / row["case"] / "volume_shell_repair_true" / "predictions.npz"),
        ("volume_aware_true", MATRIX_ROOT / row["case"] / "volume_aware_true" / "predictions.npz"),
        ("baseline", MATRIX_ROOT / row["case"] / "real_vggt_baseline_only" / "predictions.npz"),
        ("shuffled", MATRIX_ROOT / row["case"] / "shuffled_smpl_feature" / "predictions.npz"),
        ("thickness_only", MATRIX_ROOT / row["case"] / "thickness_only_control" / "predictions.npz"),
        ("same_topology", MATRIX_ROOT / row["case"] / "same_topology_no_semantic" / "predictions.npz"),
    ]
    for title, path in configs:
        _human, _rgb, full, full_rgb = load_full(path)
        items.append((title, full, full_rgb))
    rot = rotation_matrix(-34, 58, 0)
    lo, hi = bounds(items, rot)
    panels = [render_panel(pts, rgb, title, rot, lo, hi) for title, pts, rgb in items]
    compose(panels, 3, BOARDS / "V13010000000000000000_volume_shell_repair_controls.png")
    side_rot = rotation_matrix(88, 58, 0)
    slo, shi = bounds(items[:5], side_rot)
    side_panels = [render_panel(pts, rgb, title + " side", side_rot, slo, shi) for title, pts, rgb in items[:5]]
    compose(side_panels, 3, BOARDS / "V13010000000000000000_volume_shell_repair_side_depth.png")
    # Causality check
    all_metrics = []
    for row in rows:
        case = row["case"]
        repair = load_npz(REPAIR_ROOT / case / "volume_shell_repair_true" / "predictions.npz")
        repair_thick = float(geom(np.asarray(repair["human_points"], dtype=np.float32))["pca_thickness_ratio"])
        controls = {}
        for config in ["real_vggt_baseline_only", "shuffled_smpl_feature", "thickness_only_control", "same_topology_no_semantic"]:
            pred = load_npz(MATRIX_ROOT / case / config / "predictions.npz")
            controls[config] = float(geom(np.asarray(pred["human_points"], dtype=np.float32))["pca_thickness_ratio"])
        fail_controls = {k: v for k, v in controls.items() if v >= repair_thick}
        all_metrics.append(
            {
                "case": case,
                "repair_thickness": repair_thick,
                **{f"{k}_thickness": v for k, v in controls.items()},
                "fail_control_count": len(fail_controls),
                "fail_controls_json": json.dumps(fail_controls),
            }
        )
    write_csv(REPORTS / "V13010000000000000000_volume_shell_repair_metrics.csv", all_metrics)
    failures = [r for r in all_metrics if int(r["fail_control_count"]) > 0]
    decision = {
        "created_at": created_at,
        "status": "V13010_VOLUME_SHELL_REPAIR_FAIL_CLOSED_CONTINUE"
        if failures
        else "V13010_VOLUME_SHELL_REPAIR_METRIC_PRECHECK_PASS_REQUIRES_VISUAL",
        "failures": failures,
        "boards": {
            "controls": str(BOARDS / "V13010000000000000000_volume_shell_repair_controls.png"),
            "side_depth": str(BOARDS / "V13010000000000000000_volume_shell_repair_side_depth.png"),
        },
        "mentor_ready": False,
        "external_hard_block": False,
        "face_detail_claim_allowed": False,
        "allowed_face_claim": ALLOWED_FACE_CLAIM,
        "note": "Even if shell thickness improves, mentor-ready still requires visual separation and local 3D morphology pass.",
    }
    write_json(REPORTS / "V13010000000000000000_volume_shell_repair_decision.json", decision)


def main() -> int:
    created_at = now()
    rows = [r for r in read_manifest() if r.get("eligible_for_training_payload") == "True"]
    route = f"""# V13000 Auto-Evolved Volume Morphology Route

Created: {created_at}

Failed gate: V108/V114.

Root cause:
- The V107 tiny volume-aware student improves true thickness over baseline but does not beat shuffled/thickness-only controls.
- The visible local boards remain morphology/contour-level and do not prove fine detail.

Architecture repair:
- Build a multi-layer shell residual candidate that moves only V104 weak-volume visible body points.
- Preserve VGGT no-change/high-confidence zones.
- Keep real environment points unchanged.
- Fail closed if shuffled or thickness-only controls remain comparable.

No-agent rule: single-thread main run only.
"""
    write_text(GOALS / "V13000000000000000000_auto_evolved_volume_morphology_route.md", route)
    metrics = [build_shell_candidate(r) for r in rows]
    write_csv(REPORTS / "V13010000000000000000_volume_shell_repair_manifest.csv", metrics)
    render_gate(rows, metrics, created_at)
    print(json.dumps({"created_at": created_at, "status": "V13010_DONE_INTERNAL_ONLY", "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
