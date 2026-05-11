from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "output" / "training_cases" / "0012_11_frame0000_6views_smplx_native_prior_v15"
PKG = ROOT / "output" / "frozen_candidates" / "V50R2_rebuilt_from_sessions_gdrive_modal" / "package_files"
OUT = ROOT / "output" / "mentor_report_v50r2" / "rgb_dot_pointcloud"
IMG = OUT / "images"
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


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    for sy, sx in zip(*np.where(mask & ~seen)):
        comp: list[tuple[int, int]] = []
        stack = [(int(sy), int(sx))]
        while stack:
            y, x = stack.pop()
            if y < 0 or y >= h or x < 0 or x >= w or seen[y, x] or not mask[y, x]:
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


def erode_binary(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    out = np.ones_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out &= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return out


def green_spill_pixels(rgb: np.ndarray) -> np.ndarray:
    """Detect green-screen spill in source RGB colors.

    This is intentionally conservative: it targets saturated green edge pixels
    from the capture background, not ordinary dark clothing shadows.
    """
    arr = rgb.astype(np.int16)
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    saturated = (g > 72) & (g > r + 22) & (g > b + 18) & (g * 100 > (r + b + 1) * 62)
    low_saturation_edge = (g > 32) & (g > r + 5) & (g > b + 5) & (r < 180) & (b < 180)
    return saturated | low_saturation_edge


def despill_rgb(rgb: np.ndarray) -> tuple[np.ndarray, int]:
    out = rgb.astype(np.int16).copy()
    spill = green_spill_pixels(out.astype(np.uint8))
    if spill.any():
        r = out[spill, 0]
        b = out[spill, 2]
        # Clamp the green channel to a neutral value instead of deleting the
        # point. This keeps true point-cloud density while removing capture
        # background spill from the colorization.
        out[spill, 1] = np.minimum(out[spill, 1], np.maximum(r, b) + 2)
    return np.clip(out, 0, 255).astype(np.uint8), int(np.count_nonzero(spill))


def region_mask(mask: np.ndarray, region: str) -> np.ndarray:
    mask = largest_component(mask)
    x0, y0, x1, y1 = bbox(mask)
    h = max(y1 - y0, 1)
    w = max(x1 - x0, 1)
    yy, xx = np.indices(mask.shape)
    if region == "full":
        return mask
    if region == "head_face":
        return mask & (yy <= y0 + int(h * 0.42)) & (xx >= x0 + int(w * 0.06)) & (xx <= x1 - int(w * 0.06))
    if region == "upper":
        return mask & (yy <= y0 + int(h * 0.64))
    if region == "left_hand":
        return mask & (xx <= x0 + int(w * 0.47)) & (yy >= y0 + int(h * 0.20)) & (yy <= y0 + int(h * 0.83))
    if region == "right_hand":
        return mask & (xx >= x0 + int(w * 0.53)) & (yy >= y0 + int(h * 0.20)) & (yy <= y0 + int(h * 0.83))
    return mask


def draw_dot_cloud(
    image: np.ndarray,
    points: np.ndarray,
    mask: np.ndarray,
    out_size: tuple[int, int],
    title: str,
    label: str,
    dot_radius: int,
    max_points: int | None = None,
) -> tuple[Image.Image, int, int]:
    valid = mask.astype(bool) & np.isfinite(points).all(axis=-1) & (np.linalg.norm(points, axis=-1) > 1e-6)
    if not valid.any():
        valid = mask.astype(bool)
    if int(valid.sum()) > 7000:
        eroded_valid = erode_binary(valid, radius=1)
        if int(eroded_valid.sum()) > int(valid.sum()) * 0.72:
            valid = eroded_valid
    eroded = erode_binary(valid, radius=2)
    boundary = valid & ~eroded
    spill = green_spill_pixels(image) & boundary
    removed_spill = int(np.count_nonzero(valid & spill))
    valid = valid & ~spill
    if not valid.any():
        valid = mask.astype(bool) & ~green_spill_pixels(image)
    x0, y0, x1, y1 = bbox(valid)
    pad = int(max(x1 - x0, y1 - y0) * 0.10)
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(valid.shape[1], x1 + pad)
    y1 = min(valid.shape[0], y1 + pad)
    yy, xx = np.where(valid[y0:y1, x0:x1])
    yy = yy + y0
    xx = xx + x0
    if max_points and len(xx) > max_points:
        # Deterministic decimation. Keeps the image point-cloud-like instead of
        # becoming a continuous RGB cutout.
        idx = np.linspace(0, len(xx) - 1, max_points).astype(np.int64)
        yy = yy[idx]
        xx = xx[idx]
    rgb, corrected_spill = despill_rgb(image[yy, xx])
    removed_spill += corrected_spill
    ow, oh = out_size
    canvas = Image.new("RGB", out_size, "white")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((14, 12), title, fill=(0, 0, 0, 255), font=font(17))
    draw.text((14, oh - 42), f"{label}, green-spill removed={removed_spill:,}", fill=(0, 0, 0, 230), font=font(14))
    roi_w = max(x1 - x0, 1)
    roi_h = max(y1 - y0, 1)
    scale = min((ow - 80) / roi_w, (oh - 110) / roi_h)
    x_center = (x0 + x1) / 2.0
    y_center = (y0 + y1) / 2.0
    px = (xx.astype(np.float64) - x_center) * scale + ow / 2.0
    py = (yy.astype(np.float64) - y_center) * scale + oh / 2.0 + 8
    # Draw far-to-near approximately by candidate z. This keeps face/hand dots
    # visually close to a point-cloud render instead of a flat painted mask.
    z = points[yy, xx, 2] if points.shape[:2] == valid.shape else np.zeros_like(px)
    order = np.argsort(z)
    for j in order:
        color = tuple(int(v) for v in rgb[j])
        x = float(px[j])
        y = float(py[j])
        r = dot_radius
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color + (235,))
    return canvas, int(len(xx)), removed_spill


def make_sheet(path: Path, items: list[tuple[str, Image.Image]], cols: int, thumb: tuple[int, int]) -> None:
    tw, th = thumb
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * tw, rows * (th + 40)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate(items):
        x = (i % cols) * tw
        y = (i // cols) * (th + 40)
        im2 = im.copy()
        im2.thumbnail((tw - 10, th - 10), Image.LANCZOS)
        sheet.paste(im2, (x + (tw - im2.width) // 2, y + 4))
        draw.text((x + 12, y + th + 10), label, fill=(0, 0, 0), font=font(18))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> int:
    return _run_v50r2_view_consistent_replacement()
    IMG.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    inp = load_npz(CASE / "inputs.npz")
    cand = load_npz(PKG / "candidate_files__candidate_points.npz")
    hand_patch = load_npz(PKG / "candidate_files__hand_patch.npz")
    head_patch = load_npz(PKG / "candidate_files__head_face_patch.npz")
    images = inp["images"]
    masks = inp["point_masks"].astype(bool)
    points = cand["candidate_points_world"]
    hand_points = hand_patch["hand_points_world"]
    hand_region = hand_patch["hand_region_id_map"].astype(np.uint8)
    head_points = head_patch["refined_points_world"]
    head_mask = head_patch["head_mask"].astype(bool)
    face_mask = head_patch["face_mask"].astype(bool)
    camera_ids = [str(x) for x in inp["camera_ids"]]

    full_items: list[tuple[str, Image.Image]] = []
    head_items: list[tuple[str, Image.Image]] = []
    hand_items: list[tuple[str, Image.Image]] = []
    records: list[dict[str, object]] = []

    for i, cam in enumerate(camera_ids):
        im, n, removed = draw_dot_cloud(
            images[i],
            points[i],
            region_mask(masks[i], "full"),
            (540, 430),
            f"V50R2 point cloud cam{cam} full",
            "hard-mask RGB point cloud",
            dot_radius=1,
            max_points=17000,
        )
        im, n, removed = draw_dot_cloud(images[i], points[i], region_mask(masks[i], "full"), (540, 430), f"V50R2 point cloud cam{cam} full", f"hard-mask RGB point cloud, points={n:,}", 1, 17000)
        full_items.append((f"cam{cam} full", im))
        im.save(IMG / f"dot_pointcloud_cam{cam}_full.png")
        records.append({"image": f"dot_pointcloud_cam{cam}_full.png", "camera": cam, "region": "full", "points": n, "green_spill_removed": removed})

    for i, cam in enumerate(camera_ids):
        if cam not in {"15", "30", "59"}:
            continue
        for region, label, radius, cap, pts, roi in [
            ("face", "face", 2, 7000, head_points, face_mask),
            ("head", "head", 2, 9000, head_points, head_mask),
        ]:
            im, n, removed = draw_dot_cloud(images[i], pts[i], roi[i] & masks[i], (540, 430), f"V50R2 point cloud cam{cam} {label}", "RGB point cloud ROI", radius, cap)
            im, n, removed = draw_dot_cloud(images[i], pts[i], roi[i] & masks[i], (540, 430), f"V50R2 point cloud cam{cam} {label}", f"RGB point cloud ROI, points={n:,}", radius, cap)
            head_items.append((f"cam{cam} {label}", im))
            im.save(IMG / f"dot_pointcloud_cam{cam}_{region}.png")
            records.append({"image": f"dot_pointcloud_cam{cam}_{region}.png", "camera": cam, "region": region, "points": n, "green_spill_removed": removed})

    for i, cam in enumerate(camera_ids):
        if cam not in {"15", "30", "59"}:
            continue
        for region_id, region, label in [(1, "left_hand", "left hand"), (2, "right_hand", "right hand")]:
            roi = (hand_region[i] == region_id) & masks[i]
            if int(roi.sum()) < 64:
                # Fall back to the broader candidate mask only when the native
                # hand patch has too little support in that view.
                roi = region_mask(masks[i], region)
                pts = points
                source = "candidate broader fallback"
            else:
                pts = hand_points
                source = "V50R2 hand_patch"
            im, n, removed = draw_dot_cloud(images[i], pts[i], roi, (540, 430), f"V50R2 point cloud cam{cam} {label}", "RGB hand point cloud", 2, 7000)
            im, n, removed = draw_dot_cloud(images[i], pts[i], roi, (540, 430), f"V50R2 point cloud cam{cam} {label}", f"RGB hand point cloud, points={n:,}", 2, 7000)
            hand_items.append((f"cam{cam} {label}", im))
            im.save(IMG / f"dot_pointcloud_cam{cam}_{region}.png")
            records.append({"image": f"dot_pointcloud_cam{cam}_{region}.png", "camera": cam, "region": region, "points": n, "source": source, "green_spill_removed": removed})

    full_sheet = IMG / "dot_pointcloud_full_body_sheet.png"
    head_sheet = IMG / "dot_pointcloud_head_face_sheet.png"
    hand_sheet = IMG / "dot_pointcloud_hand_sheet.png"
    make_sheet(full_sheet, full_items, 3, (540, 430))
    make_sheet(head_sheet, head_items, 3, (540, 430))
    make_sheet(hand_sheet, hand_items, 3, (540, 430))

    # Also publish as the current mentor-report figure set.
    (ROOT / "output" / "mentor_report_v50r2" / "images").mkdir(parents=True, exist_ok=True)
    for src, dst in [
        (full_sheet, ROOT / "output" / "mentor_report_v50r2" / "images" / "01_full_body.png"),
        (head_sheet, ROOT / "output" / "mentor_report_v50r2" / "images" / "02_head_face.png"),
        (hand_sheet, ROOT / "output" / "mentor_report_v50r2" / "images" / "04_left_hand.png"),
        (hand_sheet, ROOT / "output" / "mentor_report_v50r2" / "images" / "05_right_hand.png"),
    ]:
        Image.open(src).save(dst)

    report = {
        "task": "v223_make_rgb_dot_pointcloud_sheets",
        "created_utc": now(),
        "point_source": str((PKG / "candidate_files__candidate_points.npz").resolve()),
        "rgb_source": str((CASE / "inputs.npz").resolve()),
        "output_dir": str(IMG.resolve()),
        "sheets": {
            "full_body": str(full_sheet.resolve()),
            "head_face": str(head_sheet.resolve()),
            "hands": str(hand_sheet.resolve()),
        },
        "records": records,
        "render_policy": "actual RGB-colored dots sampled from valid V50R2 point-map pixels; full body uses candidate_points, head/face uses head_face_patch, hands use hand_patch when available; saturated green-screen spill pixels are removed from dot colors; no continuous RGB cutout, no filled surface, no SMPL-X template mesh.",
    }
    json_path = REPORTS / "20260509_v50r2_rgb_dot_pointcloud_sheets.json"
    md_path = REPORTS / "20260509_v50r2_rgb_dot_pointcloud_sheets.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(
        "\n".join([
            "# V50R2 RGB Dot Point-Cloud Sheets",
            "",
            f"- full body: `{full_sheet.resolve()}`",
            f"- head/face: `{head_sheet.resolve()}`",
            f"- hands: `{hand_sheet.resolve()}`",
            "",
            "These figures render the candidate as RGB-colored point dots, not as a continuous RGB crop.",
        ]),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
