from __future__ import annotations

import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPO = Path(os.environ.get("VGGT_REPO_ROOT", r"D:\vggt\vggt-canonical-surfel-adapter"))
sys.path.insert(0, str(REPO))
OUTPUT = REPO / "output"
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
V173_ROOT = OUTPUT / "V17300000000000000000_multishell_topology_decoder_training"
OUT_ROOT = OUTPUT / "V17900000000000000000_collision_aware_topology_repair"
BASE_MATRIX = OUTPUT / "V10700000000000000000_volume_aware_training_matrix"
TRUE_CONFIG = "collision_aware_topology_true"
SOURCE_CONFIG = "multishell_topology_decoder_true"
CONTROL_CONFIGS = [
    "real_vggt_baseline_only",
    "same_topology_no_semantic",
    "shuffled_smpl_feature",
    "thickness_only_control",
    "posthoc_surfel_only",
    "tiny_synthetic_token_control",
]
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"

from tools.V13300_anti_billboard_metric_v2 import anti_billboard_metric_v2, pca_frame  # noqa: E402
from tools.V17800_semantic_topology_metric_v3 import combined_metric  # noqa: E402


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
            out = out * 255
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out[:, :3]


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


def load_source(case: str) -> dict[str, np.ndarray]:
    return load_npz(V173_ROOT / case / SOURCE_CONFIG / "predictions.npz")


def repair_collision(points: np.ndarray, body: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    pts = np.asarray(points, dtype=np.float32).copy()
    body_i = np.asarray(body).astype(int)
    center, vals, axes, proj = pca_frame(pts)
    # Axis 0 is the longest body axis in the current mentor-view PCA frame.
    main_axis = axes[:, 0].astype(np.float32)
    side_axis = axes[:, 1].astype(np.float32)
    thin_axis = axes[:, 2].astype(np.float32)
    main_range = float(np.ptp(proj[:, 0]))
    side_range = float(np.ptp(proj[:, 1]))
    thin_range = float(np.ptp(proj[:, 2]))
    # Part target offsets are deliberately small and monotonic. They reduce
    # impossible bbox overlap without turning the output into an SMPL-only
    # template or changing point count/colors/environment.
    main_offsets = {
        0: -0.33,  # head/back-head
        1: -0.10,  # shoulder/torso bridge
        2: 0.02,
        3: 0.02,
        4: 0.16,
        5: 0.16,
        6: 0.32,
        7: 0.36,
    }
    side_offsets = {
        0: 0.00,
        1: 0.00,
        2: 0.22,
        3: -0.22,
        4: 0.08,
        5: -0.08,
        6: 0.12,
        7: -0.12,
    }
    thin_offsets = {
        0: 0.00,
        1: 0.00,
        2: 0.08,
        3: -0.08,
        4: 0.05,
        5: -0.05,
        6: 0.04,
        7: -0.04,
    }
    before_centers = {}
    after_centers = {}
    for part in sorted(np.unique(body_i)):
        m = body_i == int(part)
        if int(m.sum()) < 50:
            continue
        before_centers[int(part)] = np.median((pts[m] - center[None]) @ axes, axis=0).tolist()
        cur = np.median((pts[m] - center[None]) @ axes, axis=0)
        target_main = main_offsets.get(int(part), 0.0) * main_range
        target_side = side_offsets.get(int(part), 0.0) * side_range
        target_thin = thin_offsets.get(int(part), 0.0) * thin_range
        shift = (
            main_axis * np.clip(target_main - cur[0], -0.085, 0.085)
            + side_axis * np.clip(target_side - cur[1], -0.055, 0.055)
            + thin_axis * np.clip(target_thin - cur[2], -0.030, 0.030)
        )
        # Softer for torso/core; stronger for endpoint/limb parts.
        strength = 0.38 if int(part) in {1, 4, 5} else 0.62
        pts[m] += shift[None] * strength
        after_centers[int(part)] = np.median((pts[m] - center[None]) @ axes, axis=0).tolist()
    return pts.astype(np.float32), {
        "main_range": main_range,
        "side_range": side_range,
        "thin_range": thin_range,
        "before_centers": before_centers,
        "after_centers": after_centers,
    }


def prediction_path(case: str, config: str) -> Path:
    if config == TRUE_CONFIG:
        return OUT_ROOT / case / TRUE_CONFIG / "predictions.npz"
    if config == SOURCE_CONFIG:
        return V173_ROOT / case / SOURCE_CONFIG / "predictions.npz"
    return BASE_MATRIX / case / config / "predictions.npz"


def run_repair(cases: list[str]) -> list[dict[str, Any]]:
    manifest = []
    for case in cases:
        src = load_source(case)
        human = np.asarray(src["human_points"], dtype=np.float32)
        body = np.asarray(src["body_part_id"], dtype=np.int16)
        repaired, info = repair_collision(human, body)
        rgb = as_rgb(src["human_rgb"])
        env = np.asarray(src["environment_points"], dtype=np.float32)
        env_rgb = as_rgb(src["environment_rgb"])
        full = np.concatenate([repaired, env], axis=0)
        full_rgb = np.concatenate([rgb, env_rgb], axis=0)
        out_dir = OUT_ROOT / case / TRUE_CONFIG
        ensure(out_dir)
        np.savez_compressed(
            out_dir / "predictions.npz",
            human_points=repaired,
            human_rgb=rgb,
            environment_points=env,
            environment_rgb=env_rgb,
            full_scene_points=full,
            full_scene_rgb=full_rgb,
            body_part_id=body,
            model_owned_student_output=np.array(True),
            teacher_points_used_at_inference=np.array(False),
            raw_kinect_depth_used_at_inference=np.array(False),
            facial_detail_target_applicable=np.array(False),
            face_detail_claim_allowed=np.array(False),
            allowed_face_claim=np.array(ALLOWED_FACE_CLAIM),
            config=np.array(TRUE_CONFIG),
            case_id=np.array(case),
            collision_aware_repair=np.array(True),
        )
        write_ply(out_dir / "full_scene_rgb_pointcloud.ply", full, full_rgb)
        v2 = anti_billboard_metric_v2(repaired, body)
        v3 = combined_metric(repaired, body)
        manifest.append(
            {
                "case": case,
                "config": TRUE_CONFIG,
                "prediction": str(out_dir / "predictions.npz"),
                "ply": str(out_dir / "full_scene_rgb_pointcloud.ply"),
                "source": str(V173_ROOT / case / SOURCE_CONFIG / "predictions.npz"),
                "model_owned_student_output": True,
                "teacher_points_used_at_inference": False,
                "raw_kinect_depth_used_at_inference": False,
                "facial_detail_target_applicable": False,
                "face_detail_claim_allowed": False,
                **v2,
                **{f"v3_{k}": v for k, v in v3.items() if k not in v2},
                "repair_info_json": json.dumps(info),
            }
        )
    return manifest


def render_panel(points: np.ndarray, colors: np.ndarray, title: str) -> Image.Image:
    size = (390, 282)
    im = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(im)
    center = points.mean(axis=0, keepdims=True)
    yaw, pitch = np.deg2rad(-30), np.deg2rad(61)
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    pts = (points - center) @ (rz @ rx).T
    lo = np.percentile(pts[:, :2], 1, axis=0)
    hi = np.percentile(pts[:, :2], 99, axis=0)
    pad = (hi - lo) * 0.17 + 1e-6
    q = (pts[:, :2] - (lo - pad)[None]) / ((hi + pad) - (lo - pad))[None]
    q[:, 1] = 1 - q[:, 1]
    xy = np.clip(q * np.array([size[0] - 48, size[1] - 68]) + np.array([24, 44]), 0, [size[0] - 1, size[1] - 1]).astype(int)
    depth = pts[:, 2]
    cue = np.clip((depth - np.quantile(depth, 0.03)) / max(np.quantile(depth, 0.97) - np.quantile(depth, 0.03), 1e-9), 0, 1)
    rgb = np.clip(as_rgb(colors).astype(np.float32) * (0.62 + 0.46 * cue[:, None]), 0, 255).astype(np.uint8)
    order = np.argsort(depth)
    step = max(1, len(order) // 52000)
    for i in order[::step]:
        x, y = xy[i]
        c = tuple(rgb[i].tolist())
        im.putpixel((int(x), int(y)), c)
        if 1 <= x < size[0] - 1 and 1 <= y < size[1] - 1:
            im.putpixel((int(x + 1), int(y)), c)
            im.putpixel((int(x), int(y + 1)), c)
    draw.text((8, 8), title[:70], fill=(10, 10, 10))
    return im


def compose(panels: list[Image.Image], cols: int, path: Path) -> None:
    ensure(path.parent)
    w, h = panels[0].size
    canvas = Image.new("RGB", (cols * w, int(math.ceil(len(panels) / cols)) * h), (255, 255, 255))
    for i, panel in enumerate(panels):
        canvas.paste(panel, ((i % cols) * w, (i // cols) * h))
    canvas.save(path)


def compare_and_render(cases: list[str], created_at: str) -> None:
    rows = []
    failures = []
    configs = [TRUE_CONFIG, SOURCE_CONFIG, *CONTROL_CONFIGS]
    for case in cases:
        case_rows = {}
        for cfg in configs:
            path = prediction_path(case, cfg)
            if not path.exists():
                continue
            pred = load_npz(path)
            body = np.asarray(pred["body_part_id"]) if "body_part_id" in pred else None
            met = combined_metric(np.asarray(pred["human_points"], dtype=np.float32), body)
            row = {"case": case, "config": cfg, **met}
            rows.append(row)
            case_rows[cfg] = row
        true = case_rows.get(TRUE_CONFIG)
        if not true:
            failures.append({"case": case, "reason": "missing_true"})
            continue
        ts = float(true["combined_topology_volume_score_v3"])
        if bool(true["combined_fail_v3"]):
            failures.append({"case": case, "reason": "true_combined_fail_v3", "true_score": ts})
        for cfg in ["real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control", SOURCE_CONFIG]:
            if cfg not in case_rows:
                continue
            cs = float(case_rows[cfg]["combined_topology_volume_score_v3"])
            if cs >= ts * 0.96:
                failures.append({"case": case, "reason": "control_or_source_close_or_better", "control": cfg, "true_score": ts, "control_score": cs})
    write_csv(REPORTS / "V17900000000000000000_collision_aware_scores.csv", rows)
    first = "0012_11_frame001" if "0012_11_frame001" in cases else cases[0]
    panels = []
    for cfg in [TRUE_CONFIG, SOURCE_CONFIG, "real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control"]:
        pred = load_npz(prediction_path(first, cfg))
        panels.append(render_panel(np.asarray(pred["human_points"], dtype=np.float32), as_rgb(pred["human_rgb"]), f"{first} {cfg.replace('_', ' ')}"))
    compose(panels, 3, BOARDS / "V17900000000000000000_collision_aware_board.png")
    write_json(
        REPORTS / "V17900000000000000000_collision_aware_decision.json",
        {
            "created_at": created_at,
            "status": "V17900_COLLISION_AWARE_REPAIR_FAIL_CLOSED_CONTINUE" if failures else "V17900_COLLISION_AWARE_PRECHECK_PASS_REQUIRES_VISUAL",
            "mentor_ready": False,
            "external_hard_block": False,
            "failures": failures,
            "score_csv": str(REPORTS / "V17900000000000000000_collision_aware_scores.csv"),
            "board": str(BOARDS / "V17900000000000000000_collision_aware_board.png"),
            "summary": "V179 applies a small collision-aware part separation to the V173 model-owned output. It remains fail-closed unless it improves v3 without harming mentor visuals.",
        },
    )


def main() -> int:
    created_at = now()
    cases = sorted(p.parent.parent.name for p in V173_ROOT.glob(f"*/{SOURCE_CONFIG}/predictions.npz"))
    manifest = run_repair(cases)
    write_csv(REPORTS / "V17900000000000000000_collision_aware_manifest.csv", manifest)
    compare_and_render(cases, created_at)
    print(json.dumps({"status": "V17900_DONE", "cases": cases, "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
