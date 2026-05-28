from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from PIL import Image, ImageDraw


REPO = Path(os.environ.get("VGGT_REPO_ROOT", r"D:\vggt\vggt-canonical-surfel-adapter"))
OUTPUT = REPO / "output"
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
TRUE_ROOT = OUTPUT / "V13700000000000000000_anti_billboard_training_matrix"
BASE_MATRIX = OUTPUT / "V10700000000000000000_volume_aware_training_matrix"
METRICS = REPORTS / "V13700000000000000000_seed_metrics.csv"
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"


CONTROL_CONFIGS = [
    "real_vggt_baseline_only",
    "same_topology_no_semantic",
    "shuffled_smpl_feature",
    "thickness_only_control",
    "posthoc_surfel_only",
    "tiny_synthetic_token_control",
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


def pca_numpy(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return rz @ rx


def project(points: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return (points - points.mean(axis=0, keepdims=True)) @ rot.T


def render_panel(points: np.ndarray, colors: np.ndarray, title: str, *, full_scene: bool = False, rot: np.ndarray | None = None) -> Image.Image:
    size = (420, 310) if full_scene else (390, 282)
    im = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(im)
    rot = rot if rot is not None else rotation_matrix(-30, 61)
    pts = project(points, rot)
    lo = np.percentile(pts[:, :2], 1, axis=0)
    hi = np.percentile(pts[:, :2], 99, axis=0)
    pad = (hi - lo) * 0.17 + 1e-6
    lo -= pad
    hi += pad
    q = (pts[:, :2] - lo[None]) / (hi[None] - lo[None] + 1e-9)
    q[:, 1] = 1.0 - q[:, 1]
    xy = np.clip(q * np.array([size[0] - 50, size[1] - 72]) + np.array([25, 46]), 0, [size[0] - 1, size[1] - 1]).astype(int)
    depth = pts[:, 2]
    d = (depth - np.quantile(depth, 0.03)) / max(np.quantile(depth, 0.97) - np.quantile(depth, 0.03), 1e-9)
    d = np.clip(d, 0.0, 1.0)
    rgb = np.clip(colors.astype(np.float32) * (0.62 + 0.45 * d[:, None]), 0, 255).astype(np.uint8)
    order = np.argsort(depth)
    step = max(1, len(order) // (76000 if full_scene else 52000))
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
    size = (390, 282)
    im = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(im)
    _center, _vals, _axes, proj = pca_numpy(points)
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


def prediction_path(case: str, config: str) -> Path:
    if config == "anti_billboard_topology_volume_true":
        return TRUE_ROOT / case / config / "predictions.npz"
    return BASE_MATRIX / case / config / "predictions.npz"


def load_points(case: str, config: str, *, full_scene: bool = False) -> tuple[np.ndarray, np.ndarray]:
    pred = load_npz(prediction_path(case, config))
    if full_scene:
        return np.asarray(pred["full_scene_points"], dtype=np.float32), as_rgb(pred["full_scene_rgb"])
    return np.asarray(pred["human_points"], dtype=np.float32), as_rgb(pred["human_rgb"])


def read_metrics() -> list[dict[str, str]]:
    with METRICS.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    created_at = now()
    metrics = read_metrics()
    cases = sorted({row["case"] for row in metrics if row.get("config") == "anti_billboard_topology_volume_true"})
    first = cases[0]
    true_full, true_rgb = load_points(first, "anti_billboard_topology_volume_true", full_scene=True)
    render_panel(true_full, true_rgb, f"{first} true full-scene", full_scene=True).save(
        BOARDS / "V13800000000000000000_advisor_human_main_full_scene.png"
    )
    control_panels: list[Image.Image] = []
    for config in ["real_vggt_baseline_only", "anti_billboard_topology_volume_true", "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control", "posthoc_surfel_only", "tiny_synthetic_token_control"]:
        pts, rgb = load_points(first, config, full_scene=(config in {"real_vggt_baseline_only", "anti_billboard_topology_volume_true"}))
        control_panels.append(render_panel(pts, rgb, f"{first} {config.replace('_', ' ')}", full_scene=False))
    compose(control_panels, 3, BOARDS / "V13800000000000000000_same_scene_baseline_true_controls.png")
    turntable: list[Image.Image] = []
    true_human, true_hrgb = load_points(first, "anti_billboard_topology_volume_true")
    for title, rot in [
        ("front", rotation_matrix(0, 0)),
        ("back", rotation_matrix(180, 0)),
        ("left", rotation_matrix(-90, 0)),
        ("right", rotation_matrix(90, 0)),
        ("oblique", rotation_matrix(-30, 61)),
    ]:
        turntable.append(render_panel(true_human, true_hrgb, f"{first} true {title}", rot=rot))
    turntable.append(cross_panel(true_human, f"{first} true cross-section"))
    compose(turntable, 3, BOARDS / "V13800000000000000000_turntable_side_depth_cross_section.png")

    causality_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in cases:
        case_rows = {row["config"]: row for row in metrics if row["case"] == case}
        true = case_rows.get("anti_billboard_topology_volume_true")
        if not true:
            failures.append({"case": case, "reason": "missing_true_metrics"})
            continue
        true_score = float(true["anti_billboard_score_v2"])
        true_fail = str(true["billboard_fail_v2"]).lower() == "true"
        if true_fail:
            failures.append({"case": case, "reason": "true_billboard_fail_v2", "true_score": true_score})
        for config in CONTROL_CONFIGS:
            control = case_rows.get(config)
            if not control:
                continue
            c_score = float(control["anti_billboard_score_v2"])
            close_or_better = c_score >= true_score * 0.96
            causality_rows.append(
                {
                    "case": case,
                    "control": config,
                    "true_score": true_score,
                    "control_score": c_score,
                    "true_billboard_fail": true_fail,
                    "control_close_or_better": close_or_better,
                }
            )
            if close_or_better:
                failures.append({"case": case, "reason": "control_close_or_better", "control": config, "true_score": true_score, "control_score": c_score})
    write_csv(REPORTS / "V14000000000000000000_anti_billboard_causality_gate.csv", causality_rows)
    write_json(
        REPORTS / "V13800000000000000000_visual_gate.json",
        {
            "created_at": created_at,
            "status": "V13800_VISUAL_GATE_FAIL_CLOSED_CONTINUE",
            "mentor_ready": False,
            "external_hard_block": False,
            "reason": "V137 Modal A10 training still has billboard_fail_v2 on all cases and controls close/better.",
            "required_next": "V17000 auto-evolution; do not package as mentor-ready.",
            "boards": {
                "advisor_human_main_full_scene": str(BOARDS / "V13800000000000000000_advisor_human_main_full_scene.png"),
                "same_scene_controls": str(BOARDS / "V13800000000000000000_same_scene_baseline_true_controls.png"),
                "turntable_side_depth_cross_section": str(BOARDS / "V13800000000000000000_turntable_side_depth_cross_section.png"),
            },
            "face_detail_claim_allowed": False,
            "allowed_face_claim": ALLOWED_FACE_CLAIM,
        },
    )
    write_json(
        REPORTS / "V14000000000000000000_anti_billboard_causality_decision.json",
        {
            "created_at": created_at,
            "status": "V14000_CAUSALITY_GATE_FAIL_CLOSED_CONTINUE",
            "mentor_ready": False,
            "external_hard_block": False,
            "failures": failures,
            "summary": "True does not beat anti-billboard hard controls across required cases; same-topology/shuffled/thickness-only remain close or stronger.",
        },
    )
    # Reuse the controls board as a compact causality board for this fail-closed stage.
    Image.open(BOARDS / "V13800000000000000000_same_scene_baseline_true_controls.png").save(
        BOARDS / "V14000000000000000000_anti_billboard_causality_board.png"
    )
    print(json.dumps({"created_at": created_at, "status": "V138_V140_FAIL_CLOSED", "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
