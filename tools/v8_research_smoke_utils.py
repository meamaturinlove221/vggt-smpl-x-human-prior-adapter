from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
    "teacher_export": "blocked",
    "candidate_export": "blocked",
    "predictions_export": "blocked",
}
RESEARCH_FLAGS = {
    "research_only": True,
    "no_export": True,
    "no_predictions_write": True,
    "no_registry_write": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_strict_pass_write": True,
    "not_teacher": True,
    "not_candidate": True,
}


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        val = float(value)
        return val if math.isfinite(val) else str(val)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_clean_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)


def make_human_points(n: int = 2600, seed: int = 7, *, clothing: bool = False, hair: bool = False, hands: bool = False) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    colors: list[np.ndarray] = []

    body_n = max(400, n // 2)
    theta = rng.uniform(0, 2 * np.pi, body_n)
    y = rng.uniform(-0.82, 0.72, body_n)
    radius = 0.16 * (1.0 - 0.30 * np.abs(y)) + rng.normal(0, 0.008, body_n)
    body = np.stack([radius * np.cos(theta), y, 0.11 * np.sin(theta)], axis=1)
    if clothing:
        bulge = ((y > -0.45) & (y < 0.2)).astype(np.float32) * 0.035
        body[:, 0] += bulge * np.sign(body[:, 0])
        body[:, 2] += 0.018 * np.sin(3 * theta)
    parts.append(body)
    colors.append(np.tile(np.asarray([160, 180, 210], dtype=np.uint8), (body_n, 1)))

    head_n = max(260, n // 8)
    phi = rng.uniform(0, np.pi, head_n)
    theta = rng.uniform(0, 2 * np.pi, head_n)
    head = np.stack([0.105 * np.sin(phi) * np.cos(theta), 0.84 + 0.13 * np.cos(phi), 0.085 * np.sin(phi) * np.sin(theta)], axis=1)
    parts.append(head)
    colors.append(np.tile(np.asarray([220, 178, 140], dtype=np.uint8), (head_n, 1)))

    if hair:
        hair_n = max(260, n // 7)
        theta = rng.uniform(-0.85 * np.pi, 0.85 * np.pi, hair_n)
        ring = rng.uniform(0.0, 1.0, hair_n)
        hair_pts = np.stack([
            0.118 * np.cos(theta) * (0.70 + 0.30 * ring),
            0.91 + 0.085 * ring + rng.normal(0, 0.008, hair_n),
            0.095 * np.sin(theta) * (0.75 + 0.25 * ring),
        ], axis=1)
        parts.append(hair_pts)
        colors.append(np.tile(np.asarray([35, 28, 24], dtype=np.uint8), (hair_n, 1)))

    if hands:
        hand_n = max(220, n // 9)
        for side, sx, color in (("left", -1.0, [50, 115, 255]), ("right", 1.0, [255, 135, 45])):
            root = np.asarray([sx * 0.25, 0.08, 0.0])
            hand_parts = []
            for finger in range(5):
                base = root + np.asarray([sx * 0.018 * (finger - 2), -0.015 * abs(finger - 2), 0.006 * (finger - 2)])
                length = 0.06 + 0.014 * (1.0 - abs(finger - 2) / 3.0)
                t = rng.uniform(0, 1, hand_n // 5)
                pts = base + np.stack([sx * length * t, -0.025 * t, 0.006 * np.sin(t * np.pi * (finger + 1))], axis=1)
                pts += rng.normal(0, 0.004, pts.shape)
                hand_parts.append(pts)
            hand = np.concatenate(hand_parts, axis=0)
            parts.append(hand)
            colors.append(np.tile(np.asarray(color, dtype=np.uint8), (hand.shape[0], 1)))

    points = np.concatenate(parts, axis=0).astype(np.float32)
    color_arr = np.concatenate(colors, axis=0).astype(np.uint8)
    return points, color_arr


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for p, c in zip(points, colors, strict=False):
            handle.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def projection_png(points: np.ndarray, colors: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 620, 720
    centered = points.astype(np.float64) - np.median(points, axis=0, keepdims=True)
    ax = 0.85 * centered[:, 0] + 0.25 * centered[:, 2]
    ay = -0.95 * centered[:, 1] + 0.12 * centered[:, 2]
    depth = centered[:, 2] - 0.15 * centered[:, 0]
    qx = np.quantile(ax, [0.01, 0.99])
    qy = np.quantile(ay, [0.01, 0.99])
    px = np.clip(((ax - qx[0]) / max(1e-6, qx[1] - qx[0]) * (width - 1)).round().astype(np.int32), 0, width - 1)
    py = np.clip(((1.0 - (ay - qy[0]) / max(1e-6, qy[1] - qy[0])) * (height - 1)).round().astype(np.int32), 0, height - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for idx in np.argsort(depth):
        x, y = int(px[idx]), int(py[idx])
        canvas[max(0, y - 1):min(height, y + 2), max(0, x - 1):min(width, x + 2)] = colors[idx]
    img = Image.fromarray(canvas, "RGB")
    ImageDraw.Draw(img).text((8, 8), title, fill=(0, 0, 0))
    img.save(path)


def contact_sheet(image_paths: list[Path], out_path: Path, title: str) -> None:
    thumbs = [Image.open(path).convert("RGB").resize((260, 300), Image.Resampling.BICUBIC) for path in image_paths]
    sheet = Image.new("RGB", (max(1, len(thumbs)) * 260, 336), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=(0, 0, 0))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, (idx * 260, 34))
        draw.text((idx * 260 + 6, 318), image_paths[idx].stem[:32], fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def simple_iou(points: np.ndarray, target: np.ndarray, radius: float = 0.045) -> float:
    sample = points[:: max(1, points.shape[0] // 600)]
    tgt = target[:: max(1, target.shape[0] // 600)]
    d = np.linalg.norm(sample[:, None, :] - tgt[None, :, :], axis=-1)
    near = (d.min(axis=1) < radius).mean()
    cover = (d.min(axis=0) < radius).mean()
    return float((near * cover) / max(1e-6, near + cover - near * cover))


def write_report(path: Path, title: str, summary: dict[str, Any]) -> None:
    lines = [
        f"# {title}",
        "",
        f"Status: `{summary.get('status')}`",
        "",
        "Research-only V8 cloud/preflight smoke. No predictions, teacher, candidate, registry, or strict pass export.",
        "",
        "## Decision",
        "",
        str(summary.get("decision")),
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False, sort_keys=True)[:24000],
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
