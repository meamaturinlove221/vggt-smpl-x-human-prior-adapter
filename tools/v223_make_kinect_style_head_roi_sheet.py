from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "output" / "training_cases" / "0012_11_frame0000_6views_smplx_native_prior_v15"
PKG = ROOT / "output" / "frozen_candidates" / "V50R2_rebuilt_from_sessions_gdrive_modal" / "package_files"
OUT = ROOT / "output" / "mentor_report_v50r2" / "images"
REPORTS = ROOT / "reports"
CANONICAL_V50R2_SOURCE_SCRIPT = ROOT / "tools" / "v223_v50r2_view_consistent_sources.py"


def _run_v50r2_view_consistent_replacement() -> int:
    import runpy

    runpy.run_path(str(CANONICAL_V50R2_SOURCE_SCRIPT), run_name="__main__")
    return 0


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def font(size: int):
    try:
        return ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", size)
    except Exception:
        return ImageFont.load_default()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def human_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    inv = ~mask.astype(bool)
    H, W = inv.shape
    seen = np.zeros_like(inv, dtype=bool)
    stack: list[tuple[int, int]] = []
    for x in range(W):
        if inv[0, x]:
            stack.append((0, x))
        if inv[H - 1, x]:
            stack.append((H - 1, x))
    for y in range(H):
        if inv[y, 0]:
            stack.append((y, 0))
        if inv[y, W - 1]:
            stack.append((y, W - 1))
    while stack:
        y, x = stack.pop()
        if y < 0 or y >= H or x < 0 or x >= W or seen[y, x] or not inv[y, x]:
            continue
        seen[y, x] = True
        stack.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
    return mask.astype(bool) | (inv & ~seen)


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    for sy, sx in zip(*np.where(mask & ~seen)):
        stack = [(int(sy), int(sx))]
        comp: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            if y < 0 or y >= H or x < 0 or x >= W or seen[y, x] or not mask[y, x]:
                continue
            seen[y, x] = True
            comp.append((y, x))
            stack.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
        if len(comp) > len(best):
            best = comp
    out = np.zeros_like(mask)
    if best:
        yy, xx = zip(*best)
        out[np.asarray(yy), np.asarray(xx)] = True
    return out


def clean_mask(mask: np.ndarray) -> np.ndarray:
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    im = im.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(7))
    return fill_internal_holes(largest_component(np.asarray(im) > 0))


def roi_mask(full_mask: np.ndarray, mode: str) -> np.ndarray:
    mask = clean_mask(full_mask)
    x0, y0, x1, y1 = human_bbox(mask)
    h = max(y1 - y0, 1)
    w = max(x1 - x0, 1)
    yy, xx = np.indices(mask.shape)
    if mode == "face2d":
        return mask & (yy <= y0 + int(h * 0.38)) & (xx >= x0 + int(w * 0.10)) & (xx <= x1 - int(w * 0.10))
    if mode == "head2d":
        return mask & (yy <= y0 + int(h * 0.62))
    return mask


def rgba_crop(image: np.ndarray, full_mask: np.ndarray, points: np.ndarray, mode: str) -> tuple[Image.Image, int]:
    valid = np.isfinite(points).all(axis=-1) & (np.linalg.norm(points, axis=-1) > 1e-6)
    mask = roi_mask(full_mask & valid, mode)
    if int(mask.sum()) < 100:
        mask = roi_mask(full_mask, mode)
    x0, y0, x1, y1 = human_bbox(mask)
    pad = int(max(x1 - x0, y1 - y0) * 0.10)
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(mask.shape[1], x1 + pad)
    y1 = min(mask.shape[0], y1 + pad)
    rgba = Image.fromarray(image[y0:y1, x0:x1]).convert("RGBA")
    rgba.putalpha(Image.fromarray((mask[y0:y1, x0:x1].astype(np.uint8) * 255)))
    return rgba, int(mask.sum())


def paste_center(sheet: Image.Image, rgba: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    max_w = x1 - x0
    max_h = y1 - y0
    scale = min(max_w / max(rgba.width, 1), max_h / max(rgba.height, 1))
    scaled = rgba.resize((max(1, int(rgba.width * scale)), max(1, int(rgba.height * scale))), Image.LANCZOS)
    px = x0 + (max_w - scaled.width) // 2
    py = y0 + (max_h - scaled.height) // 2
    sheet.paste(scaled.convert("RGB"), (px, py), scaled)


def main() -> int:
    return _run_v50r2_view_consistent_replacement()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    inp = load_npz(CASE / "inputs.npz")
    cand = load_npz(PKG / "candidate_files__candidate_points.npz")
    camera_ids = [str(x) for x in inp["camera_ids"]]
    by_cam = {cam: i for i, cam in enumerate(camera_ids)}
    columns = [
        ("cam15_face2d", "15"),
        ("cam30_face2d", "30"),
        ("cam59_face2d", "59"),
        ("cam30_best_face2d", "30"),
    ]
    W, H = 1920, 1080
    margin_x = 48
    title_h = 50
    cell_w = (W - margin_x * 2) // 4
    cell_h = 410
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), "V50R2 RGB camera-view head/face ROI sheet, hard-mask point-map projection", fill=(0, 0, 0), font=font(14))
    records: list[dict[str, object]] = []

    for col, (label, cam) in enumerate(columns):
        idx = by_cam[cam]
        for row, mode in enumerate(["face2d", "head2d"]):
            rgba, count = rgba_crop(inp["images"][idx], inp["point_masks"][idx].astype(bool), cand["candidate_points_world"][idx], mode)
            x0 = margin_x + col * cell_w
            y0 = title_h + row * 500
            paste_center(sheet, rgba, (x0 + 18, y0 + 12, x0 + cell_w - 18, y0 + cell_h))
            text_y = y0 + cell_h + 16
            row_label = label.replace("face2d", mode)
            draw.text((x0 + 4, text_y), row_label, fill=(0, 0, 0), font=font(15))
            draw.text((x0 + 4, text_y + 26), f"roi={count} cam={cam} source=V50R2 point-map", fill=(0, 0, 0), font=font(13))
            records.append({"label": row_label, "camera": cam, "mode": mode, "roi_pixels": count})

    out = OUT / "09_kinect_style_head_face_roi_sheet.png"
    sheet.save(out)
    report = {
        "task": "v223_make_kinect_style_head_roi_sheet",
        "created_utc": now(),
        "output": str(out.resolve()),
        "records": records,
        "note": "Kinect-reference-style 2x4 sheet: top row face2d, bottom row head2d, generated from V50R2 hard-mask RGB camera-view point-map projection.",
    }
    json_path = REPORTS / "20260509_v50r2_kinect_style_head_face_roi_sheet.json"
    md_path = REPORTS / "20260509_v50r2_kinect_style_head_face_roi_sheet.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_lines = [
        "# V50R2 Kinect-Style Head/Face ROI Sheet",
        "",
        f"- image: `{out.resolve()}`",
        "- layout: top row face2d, bottom row head2d",
        "- source: V50R2 RGB camera-view hard-mask point-map projection",
        "",
    ]
    for rec in records:
        md_lines.append(f"- {rec['label']}: cam={rec['camera']}, roi={rec['roi_pixels']}")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
