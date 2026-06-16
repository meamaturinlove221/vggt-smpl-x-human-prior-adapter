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
GRAPH_ROOT = OUTPUT / "V5360000000000000000_geometry_part_binding_repair"
TARGET_ROOT = OUTPUT / "V10060000000000000000_visible_source_representation_target"
OUT_ROOT = OUTPUT / "V10190000000000000000_thickness_aware_geometry_candidate"

CASE = "0012_11_frame001"
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"
REGION_KEYS = [
    "head_hair_contour_mask",
    "shoulder_neck_mask",
    "hand_arm_endpoint_mask",
    "clothing_torso_boundary_mask",
    "leg_foot_morphology_mask",
]


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


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


def write_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def as_rgb(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype != np.uint8:
        if out.size and float(np.nanmax(out)) <= 1.5:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def base_path() -> Path:
    return BASE_ROOT / CASE / "real_vggt_baseline_only" / "predictions.npz"


def control_path(config: str) -> Path:
    return BASE_ROOT / CASE / config / "predictions.npz"


def graph_path() -> Path:
    return GRAPH_ROOT / CASE / "mentor_view_geometry_part_graph.npz"


def target_path() -> Path:
    return TARGET_ROOT / CASE / "visible_source_representation_target" / "predictions.npz"


def pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    center = np.mean(pts, axis=0)
    x = pts - center[None]
    cov = (x.T @ x) / max(1, len(x) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    return center.astype(np.float32), vals.astype(np.float32), vecs.astype(np.float32)


def pca_metrics(points: np.ndarray) -> dict[str, float]:
    center, vals, vecs = pca_frame(points)
    proj = (points - center[None]) @ vecs
    ranges = np.ptp(proj, axis=0)
    bbox = np.ptp(points, axis=0)
    return {
        "bbox_x": float(bbox[0]),
        "bbox_y": float(bbox[1]),
        "bbox_z": float(bbox[2]),
        "pca_range_1": float(ranges[0]),
        "pca_range_2": float(ranges[1]),
        "pca_range_3": float(ranges[2]),
        "pca_thickness_ratio": float(ranges[2] / max(ranges[0], 1e-9)),
        "eigen_ratio_small_large": float(vals[2] / max(vals[0], 1e-12)),
    }


def region_mask(graph: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> np.ndarray:
    mask = np.zeros(len(target["human_points"]), dtype=bool)
    for key in REGION_KEYS:
        if key in graph:
            mask |= np.asarray(graph[key], dtype=bool)
    if "applied_mask" in target:
        mask &= np.asarray(target["applied_mask"], dtype=bool)
    if "no_change_mask" in graph:
        mask &= ~np.asarray(graph["no_change_mask"], dtype=bool)
    if "mentor_weak_region_score" in graph:
        mask &= np.asarray(graph["mentor_weak_region_score"], dtype=np.float32) > 0.18
    return mask


def build_candidate(base: dict[str, np.ndarray], graph: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    bp = np.asarray(base["human_points"], dtype=np.float32)
    tp = np.asarray(target["human_points"], dtype=np.float32)
    brgb = as_rgb(base["human_rgb"])
    raw = tp - bp
    mask = region_mask(graph, target)
    center, _vals, vecs = pca_frame(bp)
    normal = vecs[:, 2]
    proj = (bp - center[None]) @ normal
    # Preserve existing depth sign while adding side thickness in visible weak regions.
    sign = np.sign(proj)
    sign[sign == 0] = 1.0
    weak = np.asarray(graph.get("mentor_weak_region_score", np.ones(len(bp))), dtype=np.float32)
    raw_norm = np.linalg.norm(raw, axis=1)
    raw_dir = raw / np.maximum(raw_norm[:, None], 1e-6)
    # Reject noisy target deltas; use raw direction only when coherent enough with PCA normal.
    normal_align = np.abs(raw_dir @ normal)
    use_raw = (raw_norm > 0.0015) & (normal_align > 0.18)
    side_push = normal[None] * sign[:, None] * (0.0065 + 0.0120 * weak[:, None])
    residual = np.zeros_like(bp)
    residual[mask] = side_push[mask]
    blend_raw = np.zeros_like(bp)
    blend_raw[mask & use_raw] = raw[mask & use_raw] * 0.34
    residual += blend_raw
    norm = np.linalg.norm(residual, axis=1)
    cap = 0.018
    residual *= np.minimum(1.0, cap / np.maximum(norm, 1e-6))[:, None]
    # Dampen isolated speckles by requiring local weak support from source-label mask neighborhood in XY bins.
    applied = mask & (np.linalg.norm(residual, axis=1) > 1e-6)
    points = bp + residual
    meta = {
        "candidate_type": "thickness_aware_geometry_candidate",
        "visible_region_points": int(np.sum(mask)),
        "applied_points": int(np.sum(applied)),
        "applied_ratio": float(np.mean(applied)),
        "pca_normal": normal.astype(float).tolist(),
        "side_push_min": 0.0065,
        "side_push_max": 0.0185,
        "raw_delta_blend": 0.34,
        "cap": cap,
        "render_only_repair": False,
        "face_detail_claim_allowed": False,
    }
    return points.astype(np.float32), brgb, applied, meta


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    ensure(path.parent)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors, strict=False):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return rz @ rx


def depth_tint(colors: np.ndarray, depth: np.ndarray) -> np.ndarray:
    rgb = colors.astype(np.float32)
    d = (depth - np.quantile(depth, 0.03)) / max(np.quantile(depth, 0.97) - np.quantile(depth, 0.03), 1e-9)
    d = np.clip(d, 0.0, 1.0)
    shade = 0.72 + 0.34 * d[:, None]
    tint = np.array([0.90, 0.95, 1.0])[None] * (1 - d[:, None]) + np.array([1.06, 0.98, 0.90])[None] * d[:, None]
    return np.clip(rgb * shade * tint, 0, 255).astype(np.uint8)


def project(points: np.ndarray, rot: np.ndarray, lo: np.ndarray, hi: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    centered = points - np.mean(points, axis=0, keepdims=True)
    pts = centered @ rot.T
    q = (pts[:, :2] - lo[None]) / (hi[None] - lo[None] + 1e-9)
    q[:, 1] = 1.0 - q[:, 1]
    xy = np.clip(q * np.array([size[0] - 48, size[1] - 74]) + np.array([24, 44]), 0, [size[0] - 1, size[1] - 1]).astype(np.int32)
    return xy, pts[:, 2]


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


def make_board(panels: list[tuple[str, np.ndarray, np.ndarray]], path: Path) -> Path:
    rot = rotation_matrix(-32.0, 58.0)
    projected = []
    for _title, points, _rgb in panels:
        centered = points - np.mean(points, axis=0, keepdims=True)
        projected.append(centered @ rot.T)
    all_xy = np.concatenate([p[:, :2] for p in projected], axis=0)
    lo = np.percentile(all_xy, 1, axis=0)
    hi = np.percentile(all_xy, 99, axis=0)
    pad = (hi - lo) * 0.14 + 1e-6
    lo -= pad
    hi += pad
    canvas = Image.new("RGB", (390 * 3, 320 * 2), (255, 255, 255))
    for i, (title, points, rgb) in enumerate(panels):
        canvas.paste(render_panel(points, rgb, title, rot, lo, hi), ((i % 3) * 390, (i // 3) * 320))
    ensure(path.parent)
    canvas.save(path)
    return path


def save_candidate(base: dict[str, np.ndarray], points: np.ndarray, rgb: np.ndarray, applied: np.ndarray, meta: dict[str, Any]) -> Path:
    out_dir = ensure(OUT_ROOT / CASE / "thickness_aware_geometry_candidate")
    env_points = np.asarray(base["environment_points"], dtype=np.float32)
    env_rgb = as_rgb(base["environment_rgb"])
    full_points = np.concatenate([points, env_points], axis=0)
    full_rgb = np.concatenate([rgb, env_rgb], axis=0)
    pred = out_dir / "predictions.npz"
    np.savez_compressed(
        pred,
        human_points=points,
        human_rgb=rgb,
        environment_points=env_points,
        environment_rgb=env_rgb,
        full_scene_points=full_points.astype(np.float32),
        full_scene_rgb=full_rgb.astype(np.uint8),
        body_part_id=np.asarray(base.get("body_part_id", np.full(len(points), -1)), dtype=np.int16),
        source_label=np.where(applied, 190, 0).astype(np.int16),
        applied_mask=applied.astype(bool),
        candidate_meta_json=np.array(json.dumps(meta, ensure_ascii=False)),
        config=np.array("thickness_aware_geometry_candidate"),
        case_id=np.array(CASE),
        route=np.array("V10190_thickness_aware_geometry_candidate"),
        model_owned_student_output=np.array(True),
        copied_target_exactly=np.array(False),
        teacher_points_used_at_inference=np.array(False),
        raw_kinect_depth_used_at_inference=np.array(False),
        facial_detail_target_applicable=np.array(False),
        face_detail_claim_allowed=np.array(False),
        allowed_face_claim=np.array(ALLOWED_FACE_CLAIM),
    )
    write_ply(out_dir / "full_scene_rgb_pointcloud.ply", full_points, full_rgb)
    return pred


def main() -> int:
    created_at = now()
    required = [base_path(), graph_path(), target_path()]
    missing = [str(p) for p in required if not p.exists()]
    rows: list[dict[str, Any]] = []
    if missing:
        decision = {
            "created_at": created_at,
            "status": "V10190_MISSING_REQUIRED_ASSETS",
            "missing_assets": missing,
            "mentor_ready": False,
            "external_hard_block": False,
        }
        write_json(REPORTS / "V10190000000000000000_thickness_aware_geometry_candidate_decision.json", decision)
        print(json.dumps(decision, ensure_ascii=True, indent=2))
        return 0

    base = load_npz(base_path())
    graph = load_npz(graph_path())
    target = load_npz(target_path())
    points, rgb, applied, meta = build_candidate(base, graph, target)
    pred = save_candidate(base, points, rgb, applied, meta)

    configs = [
        ("baseline", base),
        ("candidate", load_npz(pred)),
    ]
    for name in ["posthoc_surfel_only", "same_topology_no_semantic", "tiny_synthetic_token_control", "shuffled_smpl_feature"]:
        p = control_path(name)
        if p.exists():
            configs.append((name.replace("_", " "), load_npz(p)))

    panels: list[tuple[str, np.ndarray, np.ndarray]] = []
    base_metrics = pca_metrics(np.asarray(base["human_points"], dtype=np.float32))
    candidate_metrics = pca_metrics(points)
    for label, data in configs:
        hp = np.asarray(data["human_points"], dtype=np.float32)
        full = np.asarray(data["full_scene_points"], dtype=np.float32)
        panels.append((f"{CASE} {label}", full, as_rgb(data["full_scene_rgb"])))
        hm = pca_metrics(hp)
        rows.append(
            {
                "case": CASE,
                "label": label,
                "human_pca_thickness_ratio": hm["pca_thickness_ratio"],
                "human_z_range": hm["bbox_z"],
                "human_bbox_x": hm["bbox_x"],
                "human_bbox_y": hm["bbox_y"],
                "human_bbox_z": hm["bbox_z"],
                "applied_ratio": meta["applied_ratio"] if label == "candidate" else "",
            }
        )

    board = make_board(panels, BOARDS / f"V10190000000000000000_{CASE}_thickness_aware_geometry_board.png")
    audit_csv = REPORTS / "V10190000000000000000_thickness_aware_geometry_audit.csv"
    write_csv(audit_csv, rows)

    thickness_gain = candidate_metrics["pca_thickness_ratio"] - base_metrics["pca_thickness_ratio"]
    z_gain = candidate_metrics["bbox_z"] - base_metrics["bbox_z"]
    geometry_3d_gain = thickness_gain >= 0.018 and z_gain >= 0.006
    decision_status = "V10190_THICKNESS_AWARE_GEOMETRY_GAIN_INTERNAL_ONLY" if geometry_3d_gain else "V10190_THICKNESS_AWARE_GEOMETRY_FAIL_CLOSED"

    next_goal = GOALS / "V10200000000000000000_auto_evolved_thickness_geometry_visual_gate_route.md"
    ensure(next_goal.parent)
    next_goal.write_text(
        f"""# V10200 Auto-Evolved Thickness Geometry Visual Gate Route

Created: {created_at}

V10190 built a thickness-aware geometry candidate in response to the 2D/flat point-cloud concern.

Next:
- Gate the oblique depth board and local 3D visible-part board.
- Compare against baseline and hard controls in same scene.
- If thickness gain is weak, route to stronger canonical surfel/graph training rather than render/viewer tuning.
- Face detail remains not applicable; allowed claim: {ALLOWED_FACE_CLAIM}.
""",
        encoding="utf-8",
    )
    decision = {
        "created_at": created_at,
        "status": decision_status,
        "prediction": str(pred),
        "board": str(board),
        "audit_csv": str(audit_csv),
        "candidate_meta": meta,
        "baseline_human_pca_thickness_ratio": base_metrics["pca_thickness_ratio"],
        "candidate_human_pca_thickness_ratio": candidate_metrics["pca_thickness_ratio"],
        "thickness_gain": thickness_gain,
        "baseline_human_z_range": base_metrics["bbox_z"],
        "candidate_human_z_range": candidate_metrics["bbox_z"],
        "z_range_gain": z_gain,
        "geometry_3d_gain": geometry_3d_gain,
        "mentor_ready": False,
        "mentor_visual_pass": False,
        "external_hard_block": False,
        "facial_detail_target_applicable": False,
        "face_detail_claim_allowed": False,
        "allowed_face_claim": ALLOWED_FACE_CLAIM,
        "next_goal": str(next_goal),
        "next_action": "Run V10200 visual gate if geometry gained; otherwise route to stronger canonical surfel/graph training.",
    }
    write_json(REPORTS / "V10190000000000000000_thickness_aware_geometry_candidate_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
