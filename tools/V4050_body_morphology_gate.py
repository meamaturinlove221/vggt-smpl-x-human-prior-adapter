from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
MATRIX = REPO / "output" / "V1400000000000000000_learned_residual_matrix"
CASE = "current_v895_0021_03"
CONFIGS = [
    ("real_vggt_baseline_only", "VGGT baseline"),
    ("smpl_conditioned_local_residual_true", "true residual"),
    ("posthoc_surfel_only", "posthoc"),
    ("same_topology_no_semantic", "same topology"),
    ("tiny_synthetic_token_control", "tiny"),
    ("shuffled_smpl_feature", "shuffled"),
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
        writer = csv.DictWriter(f, fieldnames=fields or ["config"])
        writer.writeheader()
        writer.writerows(rows)


def load_pred(config: str) -> dict[str, np.ndarray]:
    path = MATRIX / CASE / config / "predictions.npz"
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def stripe_artifact_score(points: np.ndarray) -> float:
    # Horizontal slice repetition shows up as concentrated y rows after quantization.
    y = points[:, 1]
    bins = np.histogram(y, bins=96)[0].astype(np.float32)
    if bins.sum() <= 0:
        return 1.0
    return float(bins.max() / max(1.0, bins.mean()))


def body_span(points: np.ndarray) -> tuple[float, float, float]:
    span = np.ptp(points, axis=0)
    return float(span[0]), float(span[1]), float(span[2])


def draw_panel(pred: dict[str, np.ndarray], title: str, size: tuple[int, int], lo: np.ndarray, hi: np.ndarray) -> Image.Image:
    width, height = size
    pts = np.asarray(pred["human_points"], dtype=np.float32)
    rgb = np.asarray(pred["human_rgb"], dtype=np.uint8)
    im = Image.new("RGB", size, (250, 250, 246))
    draw = ImageDraw.Draw(im)
    q = (pts[:, :2] - lo[None]) / (hi[None] - lo[None] + 1e-6)
    q[:, 1] = 1.0 - q[:, 1]
    xy = np.clip(q * np.array([width - 46, height - 78]) + np.array([23, 42]), 0, [width - 1, height - 1]).astype(np.int32)
    order = np.argsort(pts[:, 2])
    step = max(1, len(order) // 36000)
    for i in order[::step]:
        x, y = xy[i]
        c = tuple(rgb[i].tolist())
        im.putpixel((int(x), int(y)), c)
        if 1 <= x < width - 1 and 1 <= y < height - 1:
            im.putpixel((int(x + 1), int(y)), c)
    draw.text((12, 10), title, fill=(18, 18, 18))
    return im


def main() -> int:
    preds = [(cfg, title, load_pred(cfg)) for cfg, title in CONFIGS if (MATRIX / CASE / cfg / "predictions.npz").exists()]
    all_pts = np.concatenate([pred["human_points"][:, :2] for _, _, pred in preds], axis=0)
    lo = np.percentile(all_pts, 1, axis=0)
    hi = np.percentile(all_pts, 99, axis=0)
    pad = (hi - lo) * 0.18 + 1e-6
    lo -= pad
    hi += pad

    rows: list[dict[str, Any]] = []
    panels: list[Image.Image] = []
    for cfg, title, pred in preds:
        pts = np.asarray(pred["human_points"], dtype=np.float32)
        sx, sy, sz = body_span(pts)
        stripe = stripe_artifact_score(pts)
        rows.append(
            {
                "case": CASE,
                "config": cfg,
                "human_points": int(len(pts)),
                "span_x": sx,
                "span_y": sy,
                "span_z": sz,
                "stripe_artifact_score": stripe,
                "body_shape_readable_manual": cfg in {"real_vggt_baseline_only", "tiny_synthetic_token_control"},
                "mentor_visual_pass": False,
                "face_detail_claim_allowed": False,
                "allowed_face_claim": "head/face contour and hair region only",
            }
        )
        panels.append(draw_panel(pred, title, (460, 410), lo, hi))

    canvas = Image.new("RGB", (460 * 3, 410 * 2), (255, 255, 255))
    for idx, im in enumerate(panels[:6]):
        canvas.paste(im, ((idx % 3) * 460, (idx // 3) * 410))
    board = BOARDS / "V4050000000000000000_body_morphology_controls_board.png"
    ensure(board.parent)
    canvas.save(board)
    metrics = REPORTS / "V4050000000000000000_body_morphology_metrics.csv"
    write_csv(metrics, rows)

    true_row = next((row for row in rows if row["config"] == "smpl_conditioned_local_residual_true"), {})
    baseline_row = next((row for row in rows if row["config"] == "real_vggt_baseline_only"), {})
    visual_fail_reasons = [
        "true residual adds unnatural vertical/limb ghost structures above the body",
        "same-topology and shuffled controls remain visually close or worse in ways that undermine true-specific claim",
        "baseline/tiny preserve a more natural back-view human silhouette than true in current board",
        "face is not visible in source view; only head/hair contour claims are allowed",
    ]
    decision = {
        "created_at": now(),
        "status": "V405_BODY_MORPHOLOGY_FAIL_CLOSED",
        "board": str(board),
        "metrics": str(metrics),
        "metric_true_gt_baseline_from_previous_v160": True,
        "mentor_visual_pass": False,
        "true_visually_better_than_baseline": False,
        "true_visually_better_than_controls": False,
        "stripe_artifact_true": true_row.get("stripe_artifact_score"),
        "stripe_artifact_baseline": baseline_row.get("stripe_artifact_score"),
        "facial_detail_target_applicable": False,
        "face_detail_claim_allowed": False,
        "allowed_face_claim": "head/face contour and hair region only",
        "visual_fail_reasons": visual_fail_reasons,
        "next_action": (
            "Reject current residual output as mentor evidence. Repair representation/target to avoid ghost limbs and stripe artifacts; "
            "optimize body silhouette/head-hair contour/hand-arm/clothing boundary under same-scene controls."
        ),
    }
    write_json(REPORTS / "V4050000000000000000_body_morphology_gate.json", decision)
    print(json.dumps(decision, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
