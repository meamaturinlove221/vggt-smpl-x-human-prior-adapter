from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "tools" / "v223_make_kinect_style_3d_pointcloud_sheets.py"
OUT = ROOT / "output" / "mentor_report_v50r2" / "reference_angle_pointcloud"
IMG = OUT / "images"
REPORTS = ROOT / "reports"
CANONICAL_V50R2_SOURCE_SCRIPT = ROOT / "tools" / "v223_v50r2_view_consistent_sources.py"


def _run_v50r2_view_consistent_replacement() -> int:
    import runpy

    runpy.run_path(str(CANONICAL_V50R2_SOURCE_SCRIPT), run_name="__main__")
    return 0


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_base_module():
    spec = importlib.util.spec_from_file_location("v223pc", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def upper_roi(mask: np.ndarray, top: float, bottom: float, x_pad_frac: float = 0.10, keep_side: str = "all") -> np.ndarray:
    yy, xx = np.where(mask.astype(bool))
    if len(xx) == 0:
        return mask.astype(bool)
    y0, y1 = int(yy.min()), int(yy.max())
    x0, x1 = int(xx.min()), int(xx.max())
    h = max(y1 - y0 + 1, 1)
    w = max(x1 - x0 + 1, 1)
    ya = int(round(y0 + top * h))
    yb = int(round(y0 + bottom * h))
    xa = max(0, int(round(x0 - x_pad_frac * w)))
    xb = min(mask.shape[1] - 1, int(round(x1 + x_pad_frac * w)))
    grid_y, grid_x = np.indices(mask.shape)
    roi = mask.astype(bool) & (grid_y >= ya) & (grid_y <= yb) & (grid_x >= xa) & (grid_x <= xb)
    if keep_side == "left":
        roi &= grid_x <= int(round(x0 + 0.72 * w))
    elif keep_side == "right":
        roi &= grid_x >= int(round(x0 + 0.28 * w))
    return roi


def main() -> int:
    return _run_v50r2_view_consistent_replacement()
    mod = load_base_module()
    IMG.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    inp = mod.load_npz(mod.CASE / "inputs.npz")
    targets = mod.load_npz(mod.CASE / "targets.npz")
    depths = mod.load_npz(mod.V32 / "candidate_depths_research.npz")["candidate_depths"]

    images = inp["images"]
    masks = inp["point_masks"].astype(bool)
    intrinsics = targets["intrinsics"]
    cams = [str(x) for x in inp["camera_ids"]]

    preferred = "30" if "30" in cams else cams[len(cams) // 2]
    i = cams.index(preferred)
    face_upper = upper_roi(masks[i], top=0.00, bottom=0.43, x_pad_frac=0.02, keep_side="left")
    head_shoulder = upper_roi(masks[i], top=0.00, bottom=0.64, x_pad_frac=0.03, keep_side="left")

    yaws = [-26.0, -16.0, -6.0, 8.0]
    rgb_items: list[tuple[str, Image.Image]] = []
    depth_items: list[tuple[str, Image.Image]] = []
    records: list[dict[str, object]] = []
    for row_name, roi, max_points in [("face_upper", face_upper, 1800), ("head_shoulder", head_shoulder, 2600)]:
        for yaw in yaws:
            im, n = mod.draw_depth_cloud(
                images[i],
                depths[i],
                roi,
                intrinsics[i],
                f"V50R2 reference-angle point cloud cam{preferred} {row_name}",
                "depth-unprojected ROI point cloud",
                (430, 330),
                max_points,
                1.72,
                5000 + int((yaw + 50) * 10) + (0 if row_name == "face_upper" else 1000),
                yaw_deg=yaw,
                pitch=0.12,
                color_mode="rgb",
            )
            rgb_items.append((f"{row_name} yaw {yaw:g}", im))
            records.append({"camera": preferred, "region": row_name, "yaw": yaw, "points": n, "mode": "rgb"})
            im_depth, nd = mod.draw_depth_cloud(
                images[i],
                depths[i],
                roi,
                intrinsics[i],
                f"V50R2 depth-colored point cloud cam{preferred} {row_name}",
                "same ROI points, depth-color proof",
                (430, 330),
                max_points,
                1.72,
                7000 + int((yaw + 50) * 10) + (0 if row_name == "face_upper" else 1000),
                yaw_deg=yaw,
                pitch=0.12,
                color_mode="depth",
            )
            depth_items.append((f"{row_name} yaw {yaw:g}", im_depth))
            records.append({"camera": preferred, "region": row_name, "yaw": yaw, "points": nd, "mode": "depth"})

    rgb_sheet = IMG / "v50r2_reference_angle_head_shoulder_pointcloud.png"
    depth_sheet = IMG / "v50r2_reference_angle_head_shoulder_depthcolor.png"
    mod.make_sheet(rgb_sheet, rgb_items, cols=4, thumb=(430, 330))
    mod.make_sheet(depth_sheet, depth_items, cols=4, thumb=(430, 330))

    main_img = ROOT / "output" / "mentor_report_v50r2" / "images"
    main_img.mkdir(parents=True, exist_ok=True)
    Image.open(rgb_sheet).save(main_img / "07_reference_angle_head_shoulder_pointcloud.png")
    Image.open(depth_sheet).save(main_img / "08_reference_angle_head_shoulder_depthcolor.png")

    report = {
        "task": "v223_make_reference_angle_pointcloud_sheet",
        "created_utc": now(),
        "camera": preferred,
        "rgb_sheet": str(rgb_sheet.resolve()),
        "depth_sheet": str(depth_sheet.resolve()),
        "published_rgb": str((main_img / "07_reference_angle_head_shoulder_pointcloud.png").resolve()),
        "published_depth": str((main_img / "08_reference_angle_head_shoulder_depthcolor.png").resolve()),
        "policy": "Kinect-reference-like 2x4 head/shoulder ROI sheet. Uses candidate depth unprojected by protocol intrinsics; rows are face/upper and head/shoulder ROI; columns are mild oblique yaw variants. No continuous crop or mesh surface.",
        "records": records,
    }
    (REPORTS / "20260509_v50r2_reference_angle_pointcloud_sheet.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (REPORTS / "20260509_v50r2_reference_angle_pointcloud_sheet.md").write_text(
        "\n".join(
            [
                "# V50R2 Reference-Angle Head/Shoulder Point Cloud Sheet",
                "",
                f"- RGB point cloud: `{rgb_sheet.resolve()}`",
                f"- depth-colored proof: `{depth_sheet.resolve()}`",
                "",
                "The sheet is generated from depth-unprojected ROI points rather than an RGB crop.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
