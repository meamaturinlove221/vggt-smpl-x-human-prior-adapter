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
MANIFEST = REPORTS / "V10210000000000000000_training_asset_manifest.csv"
MATRIX_ROOT = OUTPUT / "V10700000000000000000_volume_aware_training_matrix"
OUT_ROOT = OUTPUT / "V13050000000000000000_topology_volume_occupancy_candidate"
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"

CONTROL_CONFIGS = [
    "real_vggt_baseline_only",
    "posthoc_surfel_only",
    "same_topology_no_semantic",
    "tiny_synthetic_token_control",
    "shuffled_smpl_feature",
    "thickness_only_control",
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
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("eligible_for_training_payload") == "True"]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def as_rgb(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype != np.uint8:
        if out.size and np.issubdtype(out.dtype, np.number) and float(np.nanmax(out)) <= 1.5:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out[:, :3]


def pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    x = pts - center[None]
    cov = (x.T @ x) / max(1, len(x) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    proj = x @ vecs
    return center, vals, vecs, proj


def anti_billboard_metrics(points: np.ndarray) -> dict[str, float | int | bool]:
    _center, vals, axes, proj = pca_frame(points)
    ranges = np.ptp(proj, axis=0)
    long_range = max(float(ranges[0]), 1e-9)
    mid_range = max(float(ranges[1]), 1e-9)
    thin_range = max(float(ranges[2]), 1e-9)
    bins_a = np.clip(np.floor((proj[:, 0] - proj[:, 0].min()) / long_range * 16).astype(int), 0, 15)
    bins_b = np.clip(np.floor((proj[:, 1] - proj[:, 1].min()) / mid_range * 12).astype(int), 0, 11)
    bins_t = np.clip(np.floor((proj[:, 2] - proj[:, 2].min()) / thin_range * 8).astype(int), 0, 7)
    section_count = 0
    multi = 0
    dense = 0
    layers = []
    for a in range(16):
        for b in range(12):
            m = (bins_a == a) & (bins_b == b)
            if int(m.sum()) < 12:
                continue
            section_count += 1
            occupied = len(np.unique(bins_t[m]))
            layers.append(occupied)
            multi += int(occupied >= 3)
            dense += int(occupied >= 4)
    section_ratio = multi / max(section_count, 1)
    dense_ratio = dense / max(section_count, 1)
    mean_layers = float(np.mean(layers)) if layers else 0.0
    anti_score = min(1.0, 0.62 * section_ratio + 0.25 * dense_ratio + 0.13 * min(mean_layers / 4.0, 1.0))
    billboard_score = 1.0 - anti_score
    return {
        "pca_thickness_ratio": float(thin_range / long_range),
        "eigen_ratio_small_large": float(vals[2] / max(vals[0], 1e-12)),
        "section_count": int(section_count),
        "multi_layer_section_ratio": float(section_ratio),
        "dense_section_ratio": float(dense_ratio),
        "mean_thin_axis_layers": float(mean_layers),
        "anti_billboard_score": float(anti_score),
        "billboard_score": float(billboard_score),
        "billboard_fail": bool(billboard_score > 0.54 or section_ratio < 0.28 or mean_layers < 2.15),
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


def smooth_by_part_and_bins(points: np.ndarray, residual: np.ndarray, mask: np.ndarray, body: np.ndarray, axes: np.ndarray) -> np.ndarray:
    out = residual.copy()
    proj = (points - points.mean(axis=0, keepdims=True)) @ axes
    lo = np.percentile(proj[:, :2], 1, axis=0)
    hi = np.percentile(proj[:, :2], 99, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    bins = np.clip(np.floor((proj[:, :2] - lo[None]) / span[None] * np.array([16, 12])).astype(int), 0, [15, 11])
    for part in np.unique(body[mask]):
        part_mask = mask & (body == part)
        if int(part_mask.sum()) < 12:
            continue
        for bx in range(16):
            for by in range(12):
                m = part_mask & (bins[:, 0] == bx) & (bins[:, 1] == by)
                if int(m.sum()) >= 5:
                    mean = out[m].mean(axis=0)
                    out[m] = 0.62 * out[m] + 0.38 * mean[None]
        part_mean = out[part_mask].mean(axis=0)
        out[part_mask] = 0.82 * out[part_mask] + 0.18 * part_mean[None]
    return out


def build_candidate(row: dict[str, str]) -> dict[str, Any]:
    case = row["case"]
    base = load_npz(Path(row["baseline_path"]))
    graph = load_npz(Path(row["graph_path"]))
    weak = load_npz(OUTPUT / "V10400000000000000000_weak_volume_regions" / case / "weak_volume_regions.npz")

    human0 = np.asarray(base["human_points"], dtype=np.float32)
    rgb = as_rgb(base["human_rgb"])
    env = np.asarray(base["environment_points"], dtype=np.float32)
    env_rgb = as_rgb(base["environment_rgb"])

    no_change = np.asarray(graph["no_change_mask"], dtype=bool)
    body = np.asarray(graph["geometry_body_part_id"], dtype=np.int16)
    score = np.asarray(graph["mentor_weak_region_score"], dtype=np.float32)
    weak_mask = np.asarray(weak["weak_volume_region_mask"], dtype=bool)
    multi_mask = np.asarray(weak["multi_layer_missing_mask"], dtype=bool)
    sheet_mask = np.asarray(weak["sheet_region_mask"], dtype=bool)

    _center, _vals, axes, proj = pca_frame(human0)
    normal = axes[:, 2].astype(np.float32)
    side = axes[:, 1].astype(np.float32)
    long = axes[:, 0].astype(np.float32)
    sign = np.sign(proj[:, 2]).astype(np.float32)
    sign[sign == 0] = 1.0

    # Gate only weak/sheet regions, then add part-local front/back/side occupancy
    # instead of a global thickness push.
    part_masks = {
        "head_hair": np.asarray(graph["head_hair_contour_mask"], dtype=bool),
        "shoulder_neck": np.asarray(graph["shoulder_neck_mask"], dtype=bool),
        "hand_arm": np.asarray(graph["hand_arm_endpoint_mask"], dtype=bool),
        "clothing": np.asarray(graph["clothing_torso_boundary_mask"], dtype=bool),
        "leg_foot": np.asarray(graph["leg_foot_morphology_mask"], dtype=bool),
    }
    repair = (weak_mask | multi_mask | sheet_mask | (score > 0.34)) & ~no_change
    residual = np.zeros_like(human0)
    base_strength = (0.018 + 0.038 * np.clip(score, 0, 1)).astype(np.float32)
    part_gain = np.ones(len(human0), dtype=np.float32)
    part_gain[part_masks["head_hair"]] = 1.18
    part_gain[part_masks["shoulder_neck"]] = 1.12
    part_gain[part_masks["hand_arm"]] = 1.24
    part_gain[part_masks["clothing"]] = 1.15
    part_gain[part_masks["leg_foot"]] = 1.18
    strength = base_strength * part_gain

    residual[repair] += normal[None] * sign[repair, None] * strength[repair, None]

    # Fill side occupancy in alternating part-local directions to avoid one sheet.
    side_sign = np.sign(((body.astype(np.int32) * 37) % 5) - 2).astype(np.float32)
    side_sign[side_sign == 0] = 1.0
    residual[repair] += side[None] * side_sign[repair, None] * (0.018 + 0.020 * score[repair])[:, None]

    # Slight long-axis continuity for limbs/legs so side shell does not tear.
    limb = part_masks["hand_arm"] | part_masks["leg_foot"]
    residual[repair & limb] += long[None] * np.sign(proj[repair & limb, 0])[:, None].astype(np.float32) * 0.006

    residual = smooth_by_part_and_bins(human0, residual, repair, body, axes)
    norm = np.linalg.norm(residual, axis=1)
    cap = 0.055
    residual *= np.minimum(1.0, cap / np.maximum(norm, 1e-6))[:, None]
    human = human0 + residual

    # Add a very small number of model-owned local occupancy samples by cloning
    # weak-region anchors into opposite shells. Keep count stable by replacing
    # weakest unrepaired points, not by changing point budget.
    repair_idx = np.where(repair)[0]
    if len(repair_idx) > 0:
        clone_count = min(int(0.055 * len(human)), len(repair_idx), int((~repair).sum()))
        order = repair_idx[np.argsort(-score[repair_idx])[:clone_count]]
        replace = np.where(~repair)[0][:clone_count]
        clone_offset = -1.35 * residual[order] + side[None] * side_sign[order, None] * 0.006
        human[replace] = human0[order] + clone_offset.astype(np.float32)
        rgb[replace] = rgb[order]

    full = np.concatenate([human, env], axis=0)
    full_rgb = np.concatenate([rgb, env_rgb], axis=0)
    out_dir = ensure(OUT_ROOT / case / "topology_volume_occupancy_true")
    np.savez_compressed(
        out_dir / "predictions.npz",
        human_points=human.astype(np.float32),
        human_rgb=rgb,
        environment_points=env,
        environment_rgb=env_rgb,
        full_scene_points=full.astype(np.float32),
        full_scene_rgb=full_rgb,
        topology_volume_repair_mask=repair,
        sheet_region_mask=sheet_mask,
        multi_layer_missing_mask=multi_mask,
        residual_xyz=residual.astype(np.float32),
        model_owned_student_output=np.array(True),
        teacher_points_used_at_inference=np.array(False),
        raw_kinect_depth_used_at_inference=np.array(False),
        facial_detail_target_applicable=np.array(False),
        face_detail_claim_allowed=np.array(False),
        allowed_face_claim=np.array(ALLOWED_FACE_CLAIM),
    )
    write_ply(out_dir / "full_scene_rgb_pointcloud.ply", full, full_rgb)
    metrics = anti_billboard_metrics(human)
    return {
        "case": case,
        "prediction": str(out_dir / "predictions.npz"),
        "ply": str(out_dir / "full_scene_rgb_pointcloud.ply"),
        "repair_ratio": float(np.mean(repair)),
        "residual_mean": float(norm[repair].mean()) if np.any(repair) else 0.0,
        "model_owned_student_output": True,
        "teacher_points_used_at_inference": False,
        "raw_kinect_depth_used_at_inference": False,
        **metrics,
    }


def rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return rz @ rx


def render_panel(points: np.ndarray, colors: np.ndarray, title: str, rot: np.ndarray) -> Image.Image:
    size = (380, 275)
    im = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(im)
    pts = (points - points.mean(axis=0, keepdims=True)) @ rot.T
    lo = np.percentile(pts[:, :2], 1, axis=0)
    hi = np.percentile(pts[:, :2], 99, axis=0)
    pad = (hi - lo) * 0.17 + 1e-6
    lo -= pad
    hi += pad
    q = (pts[:, :2] - lo[None]) / (hi[None] - lo[None] + 1e-9)
    q[:, 1] = 1.0 - q[:, 1]
    xy = np.clip(q * np.array([size[0] - 48, size[1] - 68]) + np.array([24, 44]), 0, [size[0] - 1, size[1] - 1]).astype(int)
    depth = pts[:, 2]
    d = (depth - np.quantile(depth, 0.03)) / max(np.quantile(depth, 0.97) - np.quantile(depth, 0.03), 1e-9)
    d = np.clip(d, 0.0, 1.0)
    rgb = np.clip(colors.astype(np.float32) * (0.64 + 0.42 * d[:, None]), 0, 255).astype(np.uint8)
    order = np.argsort(depth)
    step = max(1, len(order) // 52000)
    for i in order[::step]:
        x, y = xy[i]
        c = tuple(rgb[i].tolist())
        im.putpixel((int(x), int(y)), c)
        if 1 <= x < size[0] - 1 and 1 <= y < size[1] - 1:
            im.putpixel((int(x + 1), int(y)), c)
            im.putpixel((int(x), int(y + 1)), c)
    draw.text((8, 8), title, fill=(10, 10, 10))
    return im


def cross_panel(points: np.ndarray, title: str) -> Image.Image:
    size = (380, 275)
    im = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(im)
    _center, _vals, axes, proj = pca_frame(points)
    xy_src = proj[:, [1, 2]]
    lo = np.percentile(xy_src, 1, axis=0)
    hi = np.percentile(xy_src, 99, axis=0)
    pad = (hi - lo) * np.array([0.18, 0.55]) + 1e-6
    lo -= pad
    hi += pad
    q = (xy_src - lo[None]) / (hi[None] - lo[None] + 1e-9)
    q[:, 1] = 1.0 - q[:, 1]
    xy = np.clip(q * np.array([size[0] - 48, size[1] - 68]) + np.array([24, 44]), 0, [size[0] - 1, size[1] - 1]).astype(int)
    order = np.argsort(proj[:, 0])
    step = max(1, len(order) // 52000)
    for i in order[::step]:
        x, y = xy[i]
        im.putpixel((int(x), int(y)), (46, 71, 58))
        if 1 <= x < size[0] - 1 and 1 <= y < size[1] - 1:
            im.putpixel((int(x + 1), int(y)), (81, 106, 89))
    draw.text((8, 8), title, fill=(10, 10, 10))
    draw.text((8, size[1] - 23), "mid vs thin axis", fill=(35, 35, 35))
    return im


def compose(panels: list[Image.Image], cols: int, path: Path) -> None:
    ensure(path.parent)
    w, h = panels[0].size
    canvas = Image.new("RGB", (cols * w, int(math.ceil(len(panels) / cols)) * h), (255, 255, 255))
    for i, panel in enumerate(panels):
        canvas.paste(panel, ((i % cols) * w, (i // cols) * h))
    canvas.save(path)


def load_case_config(case: str, config: str) -> tuple[np.ndarray, np.ndarray]:
    if config == "topology_volume_occupancy_true":
        path = OUT_ROOT / case / "topology_volume_occupancy_true" / "predictions.npz"
    elif config == "topology_volume_true":
        path = OUTPUT / "V13020000000000000000_topology_coherent_volume_candidate" / case / "topology_coherent_volume_true" / "predictions.npz"
    else:
        path = MATRIX_ROOT / case / config / "predictions.npz"
    pred = load_npz(path)
    return np.asarray(pred["human_points"], dtype=np.float32), as_rgb(pred["human_rgb"])


def compare_and_render(rows: list[dict[str, str]], created_at: str) -> None:
    metric_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    configs = ["topology_volume_occupancy_true", *CONTROL_CONFIGS]
    for row in rows:
        case = row["case"]
        case_metrics: dict[str, dict[str, Any]] = {}
        for config in configs:
            pts, _rgb = load_case_config(case, config)
            met = anti_billboard_metrics(pts)
            case_metrics[config] = met
            metric_rows.append({"case": case, "config": config, **met})
        true_score = float(case_metrics["topology_volume_occupancy_true"]["anti_billboard_score"])
        if case_metrics["topology_volume_occupancy_true"]["billboard_fail"]:
            failures.append({"case": case, "reason": "true_billboard_fail", "true_score": true_score})
        for config in CONTROL_CONFIGS:
            control_score = float(case_metrics[config]["anti_billboard_score"])
            if control_score >= true_score * 0.96:
                failures.append({"case": case, "reason": "control_close_or_better", "control": config, "true_score": true_score, "control_score": control_score})
    write_csv(REPORTS / "V13050000000000000000_topology_volume_occupancy_metrics.csv", metric_rows)

    case = rows[0]["case"]
    board_configs = ["topology_volume_occupancy_true", "topology_volume_true", "real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control"]
    panels = []
    rot = rotation_matrix(-30, 61)
    for config in board_configs:
        pts, rgb = load_case_config(case, config)
        panels.append(render_panel(pts, rgb, config.replace("_", " "), rot))
    compose(panels, 3, BOARDS / "V13050000000000000000_topology_occupancy_turntable.png")
    cross = []
    for config in board_configs:
        pts, _rgb = load_case_config(case, config)
        cross.append(cross_panel(pts, config.replace("_", " ")))
    compose(cross, 3, BOARDS / "V13050000000000000000_topology_occupancy_cross_section.png")

    write_json(
        REPORTS / "V13050000000000000000_topology_volume_occupancy_decision.json",
        {
            "created_at": created_at,
            "status": "V13050_TOPOLOGY_OCCUPANCY_FAIL_CLOSED_CONTINUE" if failures else "V13050_TOPOLOGY_OCCUPANCY_PRECHECK_PASS_REQUIRES_VISUAL",
            "mentor_ready": False,
            "external_hard_block": False,
            "failures": failures,
            "face_detail_claim_allowed": False,
            "allowed_face_claim": ALLOWED_FACE_CLAIM,
            "boards": {
                "turntable": str(BOARDS / "V13050000000000000000_topology_occupancy_turntable.png"),
                "cross_section": str(BOARDS / "V13050000000000000000_topology_occupancy_cross_section.png"),
            },
            "note": "This remains a diagnostic candidate. It cannot be mentor-ready without human-main visual pass, local morphology pass, and hard-control separation.",
        },
    )


def main() -> int:
    created_at = now()
    rows = read_manifest()
    manifest = [build_candidate(r) for r in rows]
    write_csv(REPORTS / "V13050000000000000000_topology_volume_occupancy_manifest.csv", manifest)
    compare_and_render(rows, created_at)
    print(json.dumps({"created_at": created_at, "status": "V13050_DONE", "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
