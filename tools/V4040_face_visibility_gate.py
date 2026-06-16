from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
PAYLOAD = REPO / "output" / "V2500000000000000000_visual_residual_payload"
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
CASES = ["current_v895_0021_03", "0021_03_frame001", "0012_11_frame001", "0013_01_frame001"]


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
        return {key: z[key] for key in z.files}


def first_rgb(case: str) -> np.ndarray:
    data = load_npz(PAYLOAD / case / "real_vggt_tokens_and_predictions.npz")
    img = data["input_images"]
    while img.ndim > 3:
        img = img[0]
    if img.shape[0] == 3:
        img = np.moveaxis(img, 0, -1)
    img = np.asarray(img, dtype=np.float32)
    if img.max() <= 1.5:
        img *= 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def crop_stats(img: np.ndarray, uv: np.ndarray) -> tuple[float, float, tuple[int, int, int, int]]:
    if len(uv) == 0:
        return 0.0, 0.0, (0, 0, 0, 0)
    h, w = img.shape[:2]
    u = uv[:, 0] / 518.0 * w
    v = uv[:, 1] / 518.0 * h
    finite = np.isfinite(u) & np.isfinite(v)
    u = u[finite]
    v = v[finite]
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if len(u) == 0:
        return 0.0, 0.0, (0, 0, 0, 0)
    inside_ratio = float(np.mean(inside))
    if not np.any(inside):
        return inside_ratio, 0.0, (0, 0, 0, 0)
    x0, x1 = int(np.clip(np.percentile(u[inside], 2), 0, w - 1)), int(np.clip(np.percentile(u[inside], 98), 0, w - 1))
    y0, y1 = int(np.clip(np.percentile(v[inside], 2), 0, h - 1)), int(np.clip(np.percentile(v[inside], 98), 0, h - 1))
    if x1 <= x0 or y1 <= y0:
        return inside_ratio, 0.0, (x0, y0, x1, y1)
    crop = img[y0:y1, x0:x1].astype(np.float32) / 255.0
    contrast = float(crop.std(axis=(0, 1)).mean()) if crop.size else 0.0
    area_ratio = float(((x1 - x0) * (y1 - y0)) / max(1, w * h))
    return inside_ratio, contrast * area_ratio, (x0, y0, x1, y1)


def back_view_heuristic(img: np.ndarray) -> tuple[bool, str]:
    """Detect the obvious case here: person is viewed from the back, so facial landmarks are unavailable."""
    h, w = img.shape[:2]
    # The current payload images are low-res and back-facing. Dark head at top
    # center plus high-contrast shirt back is enough to block facial claims, not
    # enough to prove any positive face detail.
    top = img[: max(1, h // 3), w // 3 : 2 * w // 3].astype(np.float32)
    middle = img[h // 3 : 2 * h // 3, w // 4 : 3 * w // 4].astype(np.float32)
    top_dark = float(top.mean()) < 85.0
    middle_contrast = float(middle.std()) > 18.0
    return bool(top_dark and middle_contrast), f"top_dark={top_dark};middle_contrast={middle_contrast}"


def high_res_context_uv_stats(case: str, img: np.ndarray) -> tuple[float, float, tuple[int, int, int, int], int]:
    path = PAYLOAD / case / "high_res_scene_context.npz"
    if not path.exists():
        return 0.0, 0.0, (0, 0, 0, 0), 0
    ctx = load_npz(path)
    if "body_part_id" not in ctx or "projection_uv_518" not in ctx:
        return 0.0, 0.0, (0, 0, 0, 0), 0
    part = ctx["body_part_id"].reshape(-1)
    uv = ctx["projection_uv_518"].reshape(-1, 2)
    head = part == 1
    inside_ratio, contrast_area, bbox = crop_stats(img, uv[head])
    return inside_ratio, contrast_area, bbox, int(head.sum())


def main() -> int:
    rows: list[dict[str, Any]] = []
    panels: list[Image.Image] = []
    for case in CASES:
        img = first_rgb(case)
        back_view, back_reason = back_view_heuristic(img)
        smpl = load_npz(PAYLOAD / case / "smpl_feature_bank.npz")
        part = smpl["body_part_id"].reshape(-1)
        uv = smpl["projection_uv_camera00"].reshape(-1, 2)
        head = part == 1
        inside_ratio, contrast_area, bbox = crop_stats(img, uv[head])
        uv_source = "smpl_feature_bank_projection_uv_camera00"
        if inside_ratio <= 0.01:
            ctx_inside, ctx_contrast, ctx_bbox, ctx_count = high_res_context_uv_stats(case, img)
            if ctx_count > 0:
                inside_ratio, contrast_area, bbox = ctx_inside, ctx_contrast, ctx_bbox
                uv_source = "high_res_scene_context_projection_uv_518"
        # Conservative: source images here are low-res VGGT inputs. This gate can
        # identify clear absence, but not prove facial landmark detail.
        face_pursuit_allowed = bool((not back_view) and head.sum() > 128 and inside_ratio > 0.35 and contrast_area > 0.0025)
        facial_detail_claim_allowed = False
        rows.append(
            {
                "case": case,
                "head_projected_points": int(head.sum()),
                "uv_source": uv_source,
                "head_uv_inside_ratio": inside_ratio,
                "head_crop_contrast_area": contrast_area,
                "bbox": ",".join(map(str, bbox)),
                "source_view_face_visible": not back_view,
                "source_view_visibility_reason": back_reason,
                "face_pursuit_allowed": face_pursuit_allowed,
                "facial_detail_claim_allowed": facial_detail_claim_allowed,
                "allowed_claim": "head/face contour and hair region only",
            }
        )
        im = Image.fromarray(img).resize((360, 360))
        draw = ImageDraw.Draw(im)
        sx = 360 / img.shape[1]
        sy = 360 / img.shape[0]
        x0, y0, x1, y1 = bbox
        draw.rectangle((x0 * sx, y0 * sy, x1 * sx, y1 * sy), outline=(0, 255, 0), width=2)
        draw.text((8, 8), case, fill=(255, 40, 40))
        draw.text((8, 334), f"pursue={face_pursuit_allowed}; facial_claim=false", fill=(255, 40, 40))
        panels.append(im)
    canvas = Image.new("RGB", (720, 720), (255, 255, 255))
    for idx, im in enumerate(panels):
        canvas.paste(im, ((idx % 2) * 360, (idx // 2) * 360))
    board = BOARDS / "V4040000000000000000_face_visibility_source_board.png"
    ensure(board.parent)
    canvas.save(board)
    metrics = REPORTS / "V4040000000000000000_face_visibility_metrics.csv"
    write_csv(metrics, rows)
    decision = {
        "created_at": now(),
        "status": "FACE_VISIBILITY_GATE_DONE_NO_FACIAL_DETAIL_CLAIM",
        "board": str(board),
        "metrics": str(metrics),
        "face_pursuit_case_count": sum(1 for row in rows if row["face_pursuit_allowed"]),
        "facial_detail_claim_allowed": False,
        "allowed_claim": "head/face contour and hair region only",
        "rows": rows,
        "next_action": "Only cases with face_pursuit_allowed may enter a face ROI target-building route; no case may claim facial detail without explicit 3D landmark evidence.",
    }
    write_json(REPORTS / "V4040000000000000000_face_visibility_gate.json", decision)
    print(json.dumps(decision, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
