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

TRUE_ROOT = OUTPUT / "V13020000000000000000_topology_coherent_volume_candidate"
MATRIX_ROOT = OUTPUT / "V10700000000000000000_volume_aware_training_matrix"
MANIFEST = REPORTS / "V10210000000000000000_training_asset_manifest.csv"

CASES = [
    "0012_11_frame001",
    "0013_01_frame001",
    "0021_03_frame001",
    "current_v895_0021_03",
]
CONFIGS = [
    "topology_volume_true",
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


def read_manifest() -> dict[str, dict[str, str]]:
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        return {r.get("case", ""): r for r in csv.DictReader(f)}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def prediction_path(case: str, config: str, manifest: dict[str, dict[str, str]]) -> Path | None:
    if config == "topology_volume_true":
        p = TRUE_ROOT / case / "topology_coherent_volume_true" / "predictions.npz"
    elif config == "real_vggt_baseline_only":
        p = Path(manifest.get(case, {}).get("baseline_path", ""))
    else:
        p = MATRIX_ROOT / case / config / "predictions.npz"
    return p if p and p.exists() else None


def human_points(pred: dict[str, np.ndarray]) -> np.ndarray:
    if "human_points" in pred:
        return np.asarray(pred["human_points"], dtype=np.float64)
    if "points" in pred:
        return np.asarray(pred["points"], dtype=np.float64)
    raise KeyError("No human_points or points in prediction")


def human_rgb(pred: dict[str, np.ndarray], n: int) -> np.ndarray:
    key = "human_rgb" if "human_rgb" in pred else "rgb"
    if key in pred:
        rgb = np.asarray(pred[key])
    else:
        rgb = np.full((n, 3), 120, dtype=np.uint8)
    if rgb.dtype != np.uint8:
        if rgb.size and np.issubdtype(rgb.dtype, np.number) and float(np.nanmax(rgb)) <= 1.5:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb[:, :3]


def pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    x = points - center[None]
    cov = (x.T @ x) / max(1, len(x) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    return center, vals[order], vecs[:, order]


def occupancy_metrics(points: np.ndarray) -> dict[str, float | int | bool]:
    _center, vals, axes = pca_frame(points)
    proj = (points - points.mean(axis=0, keepdims=True)) @ axes
    ranges = np.ptp(proj, axis=0)
    long_range = max(float(ranges[0]), 1e-9)
    mid_range = max(float(ranges[1]), 1e-9)
    thin_range = max(float(ranges[2]), 1e-9)

    # Use the long/mid plane as the visible body sheet. A real 3D body should
    # have multiple occupied bins along the thin axis inside many cross sections.
    bins_a = np.clip(np.floor((proj[:, 0] - proj[:, 0].min()) / long_range * 16).astype(int), 0, 15)
    bins_b = np.clip(np.floor((proj[:, 1] - proj[:, 1].min()) / mid_range * 12).astype(int), 0, 11)
    bins_t = np.clip(np.floor((proj[:, 2] - proj[:, 2].min()) / thin_range * 8).astype(int), 0, 7)

    section_count = 0
    multi_layer_sections = 0
    dense_sections = 0
    thin_bin_counts: list[int] = []
    for a in range(16):
        for b in range(12):
            m = (bins_a == a) & (bins_b == b)
            count = int(m.sum())
            if count < 12:
                continue
            section_count += 1
            occupied_t = len(np.unique(bins_t[m]))
            thin_bin_counts.append(occupied_t)
            if occupied_t >= 3:
                multi_layer_sections += 1
            if occupied_t >= 4:
                dense_sections += 1

    section_ratio = multi_layer_sections / max(section_count, 1)
    dense_ratio = dense_sections / max(section_count, 1)
    mean_layers = float(np.mean(thin_bin_counts)) if thin_bin_counts else 0.0
    billboard_score = 1.0 - min(1.0, 0.62 * section_ratio + 0.25 * dense_ratio + 0.13 * min(mean_layers / 4.0, 1.0))
    thin_ratio = thin_range / long_range
    eigen_ratio = float(vals[2] / max(vals[0], 1e-12))
    fail = billboard_score > 0.54 or section_ratio < 0.28 or mean_layers < 2.15
    return {
        "point_count": int(len(points)),
        "pca_thickness_ratio": thin_ratio,
        "eigen_ratio_small_large": eigen_ratio,
        "section_count": section_count,
        "multi_layer_section_ratio": section_ratio,
        "dense_section_ratio": dense_ratio,
        "mean_thin_axis_layers": mean_layers,
        "anti_billboard_score": 1.0 - billboard_score,
        "billboard_score": billboard_score,
        "billboard_fail": bool(fail),
    }


def rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return rz @ rx


def draw_point_panel(points: np.ndarray, colors: np.ndarray, title: str, rot: np.ndarray) -> Image.Image:
    size = (360, 260)
    im = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(im)
    pts = (points - points.mean(axis=0, keepdims=True)) @ rot.T
    lo = np.percentile(pts[:, :2], 1, axis=0)
    hi = np.percentile(pts[:, :2], 99, axis=0)
    pad = (hi - lo) * 0.18 + 1e-6
    lo -= pad
    hi += pad
    q = (pts[:, :2] - lo[None]) / (hi[None] - lo[None] + 1e-9)
    q[:, 1] = 1.0 - q[:, 1]
    xy = np.clip(q * np.array([size[0] - 44, size[1] - 64]) + np.array([22, 42]), 0, [size[0] - 1, size[1] - 1]).astype(int)
    depth = pts[:, 2]
    d = (depth - np.quantile(depth, 0.03)) / max(np.quantile(depth, 0.97) - np.quantile(depth, 0.03), 1e-9)
    d = np.clip(d, 0.0, 1.0)
    rgb = np.clip(colors.astype(np.float32) * (0.63 + 0.44 * d[:, None]), 0, 255).astype(np.uint8)
    order = np.argsort(depth)
    step = max(1, len(order) // 48000)
    for i in order[::step]:
        x, y = xy[i]
        c = tuple(rgb[i].tolist())
        im.putpixel((int(x), int(y)), c)
        if 1 <= x < size[0] - 1 and 1 <= y < size[1] - 1:
            im.putpixel((int(x + 1), int(y)), c)
            im.putpixel((int(x), int(y + 1)), c)
    draw.text((8, 8), title, fill=(10, 10, 10))
    return im


def draw_cross_section(points: np.ndarray, title: str) -> Image.Image:
    size = (360, 260)
    im = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(im)
    _center, _vals, axes = pca_frame(points)
    proj = (points - points.mean(axis=0, keepdims=True)) @ axes
    # Show mid-vs-thin axes to reveal whether the body is a sheet or has layers.
    xy_src = proj[:, [1, 2]]
    lo = np.percentile(xy_src, 1, axis=0)
    hi = np.percentile(xy_src, 99, axis=0)
    pad = (hi - lo) * np.array([0.18, 0.55]) + 1e-6
    lo -= pad
    hi += pad
    q = (xy_src - lo[None]) / (hi[None] - lo[None] + 1e-9)
    q[:, 1] = 1.0 - q[:, 1]
    xy = np.clip(q * np.array([size[0] - 44, size[1] - 64]) + np.array([22, 42]), 0, [size[0] - 1, size[1] - 1]).astype(int)
    order = np.argsort(proj[:, 0])
    step = max(1, len(order) // 48000)
    for i in order[::step]:
        x, y = xy[i]
        im.putpixel((int(x), int(y)), (50, 74, 61))
        if 1 <= x < size[0] - 1 and 1 <= y < size[1] - 1:
            im.putpixel((int(x + 1), int(y)), (78, 102, 86))
    draw.text((8, 8), title, fill=(10, 10, 10))
    draw.text((8, size[1] - 22), "cross-section: mid axis vs thin axis", fill=(35, 35, 35))
    return im


def compose(panels: list[Image.Image], cols: int, path: Path) -> None:
    ensure(path.parent)
    w, h = panels[0].size
    canvas = Image.new("RGB", (cols * w, int(math.ceil(len(panels) / cols)) * h), (255, 255, 255))
    for i, panel in enumerate(panels):
        canvas.paste(panel, ((i % cols) * w, (i // cols) * h))
    canvas.save(path)


def main() -> int:
    manifest = read_manifest()
    rows: list[dict[str, Any]] = []
    first_case_items: list[tuple[str, np.ndarray, np.ndarray]] = []
    for case in CASES:
        for config in CONFIGS:
            p = prediction_path(case, config, manifest)
            if p is None:
                rows.append({"case": case, "config": config, "missing": True, "billboard_fail": True})
                continue
            pred = load_npz(p)
            pts = human_points(pred)
            rgb = human_rgb(pred, len(pts))
            metrics = occupancy_metrics(pts)
            rows.append(
                {
                    "case": case,
                    "config": config,
                    "prediction_path": str(p),
                    "missing": False,
                    **metrics,
                }
            )
            if case == "0012_11_frame001":
                first_case_items.append((config, pts, rgb))

    true_rows = [r for r in rows if r.get("config") == "topology_volume_true" and not r.get("missing")]
    control_failures: list[dict[str, Any]] = []
    for case in CASES:
        true = next((r for r in rows if r.get("case") == case and r.get("config") == "topology_volume_true"), None)
        if not true or true.get("missing"):
            control_failures.append({"case": case, "reason": "missing_true"})
            continue
        if true.get("billboard_fail"):
            control_failures.append({"case": case, "reason": "true_billboard_fail", "billboard_score": true.get("billboard_score")})
        for r in rows:
            if r.get("case") != case or r.get("config") == "topology_volume_true" or r.get("missing"):
                continue
            if float(r.get("anti_billboard_score", 0.0)) >= float(true.get("anti_billboard_score", 0.0)) * 0.96:
                control_failures.append(
                    {
                        "case": case,
                        "reason": "control_anti_billboard_close_or_better",
                        "control": r.get("config"),
                        "true_score": true.get("anti_billboard_score"),
                        "control_score": r.get("anti_billboard_score"),
                    }
                )

    write_csv(REPORTS / "V13040000000000000000_anti_billboard_audit.csv", rows)
    decision = {
        "created_at": now(),
        "status": "V13040_ANTI_BILLBOARD_FAIL_CLOSED_CONTINUE",
        "mentor_ready": False,
        "external_hard_block": False,
        "user_correction_accepted": True,
        "problem": "Current point clouds still read as billboard/textured sprite in side-depth and turntable views.",
        "hard_gate": "A true route must show topology-connected 3D body volume, not only higher thickness.",
        "true_billboard_fail_cases": [r["case"] for r in true_rows if r.get("billboard_fail")],
        "control_failures": control_failures,
        "boards": {
            "turntable": str(BOARDS / "V13040000000000000000_anti_billboard_turntable.png"),
            "cross_section": str(BOARDS / "V13040000000000000000_anti_billboard_cross_section.png"),
        },
        "next_route": [
            "Do not continue thickness-only repair.",
            "Add anti-billboard topology-volume representation with front/back/side shell occupancy.",
            "Train with side cross-section occupancy and limb/torso continuity losses.",
            "Keep projection and renderer as auxiliary only.",
        ],
    }
    write_json(REPORTS / "V13040000000000000000_anti_billboard_decision.json", decision)
    (REPORTS / "V13040000000000000000_goal_correction.md").write_text(
        "# V13040 Goal Correction: Anti-Billboard Topology-Volume\n\n"
        "The user is correct: the current images still make the human read as a two-dimensional billboard. "
        "This is not solved by oblique rendering or by increasing a PCA thickness metric. "
        "The route is corrected from generic volume-aware morphology to anti-billboard topology-volume morphology.\n\n"
        "Hard gates added:\n\n"
        "- side-depth and turntable views must not read as a textured paper cutout;\n"
        "- true must show front/back/side shell separation and local cross-section occupancy;\n"
        "- torso, limbs, head/hair, shoulder/neck, clothing boundary, and leg/foot regions must remain topologically connected;\n"
        "- thickness-only and shuffled/random controls must not look equally 3D;\n"
        "- local close-ups must show 3D morphology, not only contour or texture.\n\n"
        "Current status remains fail-closed and not mentor-ready. This is not an external hard block.\n",
        encoding="utf-8",
    )

    if first_case_items:
        rot_oblique = rotation(-28, 62)
        panels = [draw_point_panel(pts, rgb, name.replace("_", " "), rot_oblique) for name, pts, rgb in first_case_items]
        compose(panels, 3, BOARDS / "V13040000000000000000_anti_billboard_turntable.png")
        cross = [draw_cross_section(pts, name.replace("_", " ")) for name, pts, _rgb in first_case_items]
        compose(cross, 3, BOARDS / "V13040000000000000000_anti_billboard_cross_section.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
