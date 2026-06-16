from __future__ import annotations

import csv
import json
import math
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
V25 = ROOT / "output" / "surface_research_cloud_preflight" / "V25_research_vggt_predictions"
SCENE = ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_12views_tmf_v223_repaired"
OUT = ROOT / "output" / "surface_research_cloud_preflight" / "V25_base_vggt_pointmap_ply_export_for_advisor_20260603"

POINTS_NPZ = V25 / "research_points_world.npz"
DEPTH_NPZ = V25 / "research_depths.npz"
CONF_NPZ = V25 / "research_confidence.npz"
SUMMARY_JSON = V25 / "research_summary.json"
SCENE_MANIFEST = SCENE / "scene_manifest.json"

TARGET_SIZE = 518
SIX_CAMERAS = ["00", "01", "06", "11", "16", "21"]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jr(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jr(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jr(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
    if isinstance(value, np.ndarray):
        return jr(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jr(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_inputs() -> None:
    missing = [p for p in [POINTS_NPZ, DEPTH_NPZ, CONF_NPZ, SUMMARY_JSON, SCENE_MANIFEST] if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(str(p) for p in missing))


def preprocess_rgb(path: Path, target_size: int = TARGET_SIZE) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    width, height = img.size
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14
    resized = img.resize((int(new_width), int(new_height)), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    pad_left = (target_size - int(new_width)) // 2
    pad_top = (target_size - int(new_height)) // 2
    canvas.paste(resized, (pad_left, pad_top))
    valid = np.zeros((target_size, target_size), dtype=bool)
    valid[pad_top : pad_top + int(new_height), pad_left : pad_left + int(new_width)] = True
    info = {
        "original_size": [width, height],
        "resized_size": [int(new_width), int(new_height)],
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "mode": "VGGT load_and_preprocess_images pad equivalent",
    }
    return np.asarray(canvas, dtype=np.uint8), valid, info


def preprocess_mask(path: Path, target_size: int = TARGET_SIZE) -> np.ndarray:
    img = Image.open(path).convert("L")
    width, height = img.size
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14
    resized = img.resize((int(new_width), int(new_height)), Image.Resampling.NEAREST)
    canvas = Image.new("L", (target_size, target_size), 0)
    pad_left = (target_size - int(new_width)) // 2
    pad_top = (target_size - int(new_height)) // 2
    canvas.paste(resized, (pad_left, pad_top))
    return np.asarray(canvas, dtype=np.uint8) > 127


def write_binary_ply(path: Path, points: np.ndarray, rgb: np.ndarray, conf: np.ndarray, depth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    conf = np.asarray(conf, dtype=np.float32).reshape(-1)
    depth = np.asarray(depth, dtype=np.float32).reshape(-1)
    if not (len(points) == len(rgb) == len(conf) == len(depth)):
        raise ValueError(f"PLY field length mismatch for {path}")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property float confidence\n"
        "property float depth\n"
        "end_header\n"
    ).encode("ascii")
    arr = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("confidence", "<f4"),
            ("depth", "<f4"),
        ],
    )
    arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    arr["confidence"] = conf
    arr["depth"] = depth
    with path.open("wb") as f:
        f.write(header)
        arr.tofile(f)


def points_for_mask(
    points_map: np.ndarray,
    rgb_img: np.ndarray,
    conf_map: np.ndarray,
    depth_map: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = mask & np.isfinite(points_map).all(axis=-1) & np.isfinite(conf_map) & np.isfinite(depth_map)
    yy, xx = np.where(valid)
    pts = points_map[yy, xx]
    rgb = rgb_img[yy, xx]
    conf = conf_map[yy, xx]
    depth = depth_map[yy, xx]
    yx = np.stack([yy, xx], axis=1).astype(np.uint16) if len(yy) else np.zeros((0, 2), dtype=np.uint16)
    return pts, rgb, conf, depth, yx


def render_point_cloud(points: np.ndarray, rgb: np.ndarray, size: tuple[int, int] = (640, 520), max_points: int = 90000) -> Image.Image:
    w, h = size
    if len(points) == 0:
        return Image.new("RGB", size, "white")
    pts = np.asarray(points, dtype=np.float32)
    cols = np.asarray(rgb, dtype=np.uint8)
    if len(pts) > max_points:
        rng = np.random.default_rng(20260603)
        keep = np.sort(rng.choice(len(pts), max_points, replace=False))
        pts = pts[keep]
        cols = cols[keep]
    pts = pts - np.nanmedian(pts, axis=0, keepdims=True)
    yaw = math.radians(-20.0)
    pitch = math.radians(10.0)
    ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]], dtype=np.float32)
    p = pts @ (ry @ rx).T
    order = np.argsort(p[:, 2])
    p = p[order]
    cols = cols[order]
    x = p[:, 0]
    y = -p[:, 1]
    x_span = max(float(np.nanpercentile(x, 99) - np.nanpercentile(x, 1)), 1e-6)
    y_span = max(float(np.nanpercentile(y, 99) - np.nanpercentile(y, 1)), 1e-6)
    scale = min((w * 0.80) / x_span, (h * 0.78) / y_span)
    u = (x - np.nanmedian(x)) * scale + w * 0.50
    v = (y - np.nanmedian(y)) * scale + h * 0.52
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img, "RGBA")
    for px, py, c in zip(u, v, cols):
        if -2 <= px < w + 2 and -2 <= py < h + 2:
            draw.point((int(px), int(py)), fill=(int(c[0]), int(c[1]), int(c[2]), 210))
    return img


def colorize_map(arr: np.ndarray, mask: np.ndarray | None = None, label: str = "") -> Image.Image:
    x = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(x)
    if mask is not None:
        finite &= mask
    if finite.any():
        lo, hi = np.nanpercentile(x[finite], [2, 98])
    else:
        lo, hi = 0.0, 1.0
    norm = np.clip((x - lo) / max(float(hi - lo), 1e-6), 0, 1)
    r = np.clip(255 * norm, 0, 255).astype(np.uint8)
    g = np.clip(255 * (1 - np.abs(norm - 0.5) * 2), 0, 255).astype(np.uint8)
    b = np.clip(255 * (1 - norm), 0, 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[~finite] = 255
    img = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(img)
    if label:
        draw.rectangle((0, 0, 270, 24), fill=(255, 255, 255))
        draw.text((6, 5), f"{label} [{lo:.3f},{hi:.3f}]", fill=(0, 0, 0))
    return img


def make_contact_sheet(items: list[tuple[str, Image.Image]], out: Path, cols: int = 3, cell: tuple[int, int] = (640, 580)) -> None:
    cw, ch = cell
    rows = int(math.ceil(len(items) / cols))
    sheet = Image.new("RGB", (cols * cw, rows * ch), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (label, img) in enumerate(items):
        x0 = (idx % cols) * cw
        y0 = (idx // cols) * ch
        draw.text((x0 + 14, y0 + 12), label, fill=(0, 0, 0))
        panel = img.copy()
        panel.thumbnail((cw - 28, ch - 54), Image.Resampling.LANCZOS)
        sheet.paste(panel, (x0 + (cw - panel.width) // 2, y0 + 42 + (ch - 54 - panel.height) // 2))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def make_triptych(depth_img: Image.Image, conf_img: Image.Image, ply_img: Image.Image, out: Path, title: str) -> None:
    cell_w, cell_h = 520, 520
    canvas = Image.new("RGB", (cell_w * 3, cell_h + 42), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), title, fill=(0, 0, 0))
    for i, (label, img) in enumerate([("depth", depth_img), ("world_points_conf", conf_img), ("RGB PLY render", ply_img)]):
        x = i * cell_w
        draw.text((x + 12, 32), label, fill=(0, 0, 0))
        panel = img.copy()
        panel.thumbnail((cell_w - 22, cell_h - 68), Image.Resampling.LANCZOS)
        canvas.paste(panel, (x + (cell_w - panel.width) // 2, 62 + (cell_h - 68 - panel.height) // 2))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def zip_dir(src: Path, dst: Path) -> dict[str, Any]:
    if dst.exists():
        dst.unlink()
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in src.rglob("*"):
            if path.is_file() and path != dst:
                zf.write(path, path.relative_to(src))
    with zipfile.ZipFile(dst, "r") as zf:
        bad = zf.testzip()
        entries = len(zf.namelist())
    return {"path": dst, "testzip": bad, "entries": entries, "size_bytes": dst.stat().st_size}


def main() -> int:
    ensure_inputs()
    for sub in ["ply/full_scene", "ply/human_roi", "ply/confidence_filtered", "metadata", "boards", "reports", "archive"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    summary = read_json(SUMMARY_JSON)
    scene_manifest = read_json(SCENE_MANIFEST)
    views = scene_manifest["exported_views"][:12]

    with np.load(POINTS_NPZ, allow_pickle=False) as z:
        points12 = np.asarray(z["frame0000"], dtype=np.float32)
        extrinsic = np.asarray(z["frame0000_extrinsic"], dtype=np.float32)
        intrinsic = np.asarray(z["frame0000_intrinsic"], dtype=np.float32)
    with np.load(DEPTH_NPZ, allow_pickle=False) as z:
        depth12 = np.asarray(z["frame0000"], dtype=np.float32)
        if depth12.ndim == 4 and depth12.shape[-1] == 1:
            depth12 = depth12[..., 0]
    with np.load(CONF_NPZ, allow_pickle=False) as z:
        depth_conf12 = np.asarray(z["frame0000_depth_conf"], dtype=np.float32)
        world_conf12 = np.asarray(z["frame0000_world_points_conf"], dtype=np.float32)

    if points12.shape != (12, TARGET_SIZE, TARGET_SIZE, 3):
        raise ValueError(f"unexpected points shape: {points12.shape}")
    if depth12.shape != (12, TARGET_SIZE, TARGET_SIZE):
        raise ValueError(f"unexpected depth shape: {depth12.shape}")
    if world_conf12.shape != (12, TARGET_SIZE, TARGET_SIZE):
        raise ValueError(f"unexpected confidence shape: {world_conf12.shape}")

    manifest_rows: list[dict[str, Any]] = []
    rgb_stack: list[np.ndarray] = []
    valid_stack: list[np.ndarray] = []
    mask_stack: list[np.ndarray] = []
    camera_ids: list[str] = []
    preprocess_info: dict[str, Any] = {}
    preview_images: dict[str, dict[str, Image.Image]] = {"full_scene": {}, "human_roi": {}, "confidence_filtered": {}}
    triptych_items: list[tuple[str, Image.Image]] = []

    for idx, view in enumerate(views):
        cam = str(view["camera_id"])
        camera_ids.append(cam)
        rgb, valid_img, pp_info = preprocess_rgb(Path(view["image_path"]))
        human_mask = preprocess_mask(Path(view["mask_path"]))
        rgb_stack.append(rgb)
        valid_stack.append(valid_img)
        mask_stack.append(human_mask)
        preprocess_info[cam] = pp_info | {"image_path": view["image_path"], "mask_path": view["mask_path"]}

        points_map = points12[idx]
        conf_map = world_conf12[idx]
        depth_map = depth12[idx]
        finite = np.isfinite(points_map).all(axis=-1) & np.isfinite(conf_map) & np.isfinite(depth_map)
        full_mask = valid_img & finite
        conf_values = conf_map[full_mask]
        conf_threshold = float(np.percentile(conf_values, 70.0)) if len(conf_values) else float("nan")
        masks = {
            "full_scene": full_mask,
            "human_roi": full_mask & human_mask,
            "confidence_filtered": full_mask & (conf_map >= conf_threshold),
        }
        for kind, mask in masks.items():
            pts, cols, conf, depth, yx = points_for_mask(points_map, rgb, conf_map, depth_map, mask)
            ply_path = OUT / "ply" / kind / f"{kind}_rgb_world_points_cam{cam}.ply"
            write_binary_ply(ply_path, pts, cols, conf, depth)
            meta_npz = OUT / "metadata" / f"{kind}_cam{cam}_confidence_depth_indices.npz"
            np.savez_compressed(
                meta_npz,
                yx=yx,
                confidence=conf.astype(np.float32),
                depth=depth.astype(np.float32),
                camera_id=np.asarray(cam),
                view_index=np.asarray(idx, dtype=np.int32),
                confidence_threshold=np.asarray(conf_threshold, dtype=np.float32),
                mask_kind=np.asarray(kind),
            )
            manifest_rows.append(
                {
                    "frame_id": "frame0000",
                    "view_index": idx,
                    "camera_id": cam,
                    "ply_kind": kind,
                    "ply_path": str(ply_path.resolve()),
                    "metadata_npz": str(meta_npz.resolve()),
                    "point_count": int(len(pts)),
                    "rgb_source": str(Path(view["image_path"]).resolve()),
                    "rgb_preprocess": "VGGT pad mode, bicubic resize, white padding, 518x518",
                    "confidence_source": str(CONF_NPZ.resolve()) + "::frame0000_world_points_conf",
                    "depth_source": str(DEPTH_NPZ.resolve()) + "::frame0000",
                    "points_source": str(POINTS_NPZ.resolve()) + "::frame0000",
                    "confidence_threshold": conf_threshold if kind == "confidence_filtered" else "",
                    "mask_source": "valid_preprocessed_rgb_area" if kind == "full_scene" else ("existing_scene_manifest_mask_resized_with_vggt_pad_preprocess" if kind == "human_roi" else "valid_preprocessed_rgb_area_and_world_points_conf_p70"),
                    "contains_xyz": True,
                    "contains_rgb": True,
                    "contains_confidence_property": True,
                    "smpl_used": False,
                    "v50r2_used": False,
                    "final_output_allowed": False,
                    "notes": "V25 base VGGT research-only point map export; not promoted.",
                }
            )
            if cam in SIX_CAMERAS:
                preview_images[kind][cam] = render_point_cloud(pts, cols, max_points=120000 if kind == "full_scene" else 90000)
        if cam in SIX_CAMERAS:
            depth_img = colorize_map(depth_map, full_mask, f"cam{cam} depth")
            conf_img = colorize_map(conf_map, full_mask, f"cam{cam} conf")
            full_pts, full_cols, _, _, _ = points_for_mask(points_map, rgb, conf_map, depth_map, full_mask)
            ply_img = render_point_cloud(full_pts, full_cols, max_points=120000)
            triptych = OUT / "boards" / f"V25_frame0000_cam{cam}_depth_conf_rgb_ply_triptych.png"
            make_triptych(depth_img, conf_img, ply_img, triptych, f"V25 base VGGT frame0000 cam{cam}")
            triptych_items.append((f"cam{cam}", Image.open(triptych).convert("RGB")))

    # Save common metadata maps for traceability.
    np.savez_compressed(
        OUT / "metadata" / "V25_frame0000_maps_and_masks_518.npz",
        camera_ids=np.asarray(camera_ids),
        rgb_518=np.stack(rgb_stack).astype(np.uint8),
        valid_image_mask=np.stack(valid_stack).astype(np.uint8),
        human_roi_mask=np.stack(mask_stack).astype(np.uint8),
        depth=depth12.astype(np.float32),
        world_points_conf=world_conf12.astype(np.float32),
        depth_conf=depth_conf12.astype(np.float32),
        intrinsic=intrinsic.astype(np.float32),
        extrinsic=extrinsic.astype(np.float32),
    )

    # Contact sheets for the six mentor-priority views.
    for kind in ["full_scene", "human_roi", "confidence_filtered"]:
        items = [(f"cam{cam} {kind}", preview_images[kind][cam]) for cam in SIX_CAMERAS if cam in preview_images[kind]]
        make_contact_sheet(items, OUT / "boards" / f"V25_frame0000_six_view_{kind}_rgb_world_points_contact_sheet.png", cols=3)
    make_contact_sheet(triptych_items, OUT / "boards" / "V25_frame0000_six_view_depth_conf_rgb_ply_triptych_sheet.png", cols=1, cell=(1600, 610))

    fields = [
        "frame_id",
        "view_index",
        "camera_id",
        "ply_kind",
        "ply_path",
        "metadata_npz",
        "point_count",
        "rgb_source",
        "rgb_preprocess",
        "confidence_source",
        "depth_source",
        "points_source",
        "confidence_threshold",
        "mask_source",
        "contains_xyz",
        "contains_rgb",
        "contains_confidence_property",
        "smpl_used",
        "v50r2_used",
        "final_output_allowed",
        "notes",
    ]
    write_csv(OUT / "reports" / "V25_frame0000_base_vggt_ply_manifest.csv", manifest_rows, fields)
    write_json(
        OUT / "reports" / "V25_frame0000_base_vggt_ply_manifest.json",
        {
            "created_at": now_utc(),
            "status": "V25_BASE_VGGT_POINTMAP_PER_VIEW_PLY_EXPORTED_NOT_PROMOTED",
            "input_role": "base VGGT research-only output",
            "frame_id": "frame0000",
            "camera_ids": camera_ids,
            "six_panel_camera_ids": SIX_CAMERAS,
            "points_npz": POINTS_NPZ,
            "depth_npz": DEPTH_NPZ,
            "confidence_npz": CONF_NPZ,
            "scene_manifest": SCENE_MANIFEST,
            "preprocess_info": preprocess_info,
            "v25_summary": {
                "prior_used_frame0000": summary["frame_summaries"]["frame0000"].get("prior_used"),
                "model_prior_channels": summary.get("model_prior_channels"),
                "model_prior_summary_channels": summary.get("model_prior_summary_channels"),
                "research_only": summary.get("research_only"),
                "no_predictions_write": summary.get("no_predictions_write"),
                "decision": summary.get("decision"),
            },
            "forbidden_mixing": {
                "v50r2_used": False,
                "smpl_mesh_used": False,
                "merged_or_synthesis_ply_used": False,
            },
            "final_output_allowed": False,
            "rows": manifest_rows,
        },
    )
    readme = f"""# V25 Base VGGT Point Map PLY Export

Created: `{now_utc()}`

This package exports `frame0000` from the base VGGT research output:

- points: `{POINTS_NPZ}`
- depth: `{DEPTH_NPZ}`
- confidence: `{CONF_NPZ}`
- RGB/mask source: `{SCENE_MANIFEST}`

Identity boundary:

- V25 base VGGT only.
- `prior_used=false`, `model_prior_channels=0` in the V25 summary.
- No V50R2 PLY, no SMPL mesh, no fused/merged/synthesis PLY was used.
- `final_output_allowed=false`; this is for human review before any promotion.

PLY groups:

- `ply/full_scene/`: valid padded RGB area with environment and human.
- `ply/human_roi/`: existing scene human masks, resized with the same VGGT pad preprocessing.
- `ply/confidence_filtered/`: valid full-scene area with per-view world point confidence >= p70.

Each PLY contains `x y z red green blue confidence depth` as vertex properties. Per-PLY confidence/depth/index arrays are also saved under `metadata/`.
"""
    (OUT / "reports" / "V25_base_vggt_pointmap_export_readme.md").write_text(readme, encoding="utf-8")
    archive_info = zip_dir(OUT, OUT / "archive" / "V25_base_vggt_pointmap_ply_export_for_advisor.zip")
    write_json(
        OUT / "reports" / "V25_frame0000_export_final_status.json",
        {
            "created_at": now_utc(),
            "status": "V25_BASE_VGGT_POINTMAP_PER_VIEW_PLY_EXPORTED_NOT_PROMOTED",
            "out_dir": OUT,
            "archive": archive_info,
            "ply_count": len(manifest_rows),
            "zip_testzip_clean": archive_info["testzip"] is None,
            "final_output_allowed": False,
        },
    )
    print(json.dumps(jr({"out_dir": OUT, "archive": archive_info, "ply_count": len(manifest_rows)}), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
