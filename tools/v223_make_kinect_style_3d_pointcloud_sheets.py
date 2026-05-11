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
V32 = ROOT / "output" / "surface_research_preflight_local" / "V32_candidate_inference_research"
OUT = ROOT / "output" / "mentor_report_v50r2" / "kinect_style_3d_pointcloud"
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


def despill_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.int16).copy()
    r = arr[:, 0]
    g = arr[:, 1]
    b = arr[:, 2]
    spill = (g > 38) & (g > r + 5) & (g > b + 5) & (r < 190) & (b < 190)
    if spill.any():
        arr[spill, 1] = np.minimum(arr[spill, 1], np.maximum(r[spill], b[spill]) + 2)
    return np.clip(arr, 0, 255).astype(np.uint8)


def collect_points(
    image: np.ndarray,
    points: np.ndarray,
    mask: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = mask.astype(bool) & np.isfinite(points).all(axis=-1) & (np.linalg.norm(points, axis=-1) > 1e-6)
    yy, xx = np.where(valid)
    if len(xx) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), yy, xx
    if len(xx) > max_points:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(xx), size=max_points, replace=False))
        yy = yy[keep]
        xx = xx[keep]
    xyz = points[yy, xx].astype(np.float64)
    rgb = despill_rgb(image[yy, xx])
    return xyz, rgb, yy, xx


def collect_depth_unprojected_points(
    image: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = mask.astype(bool) & np.isfinite(depth) & (depth > 1e-4)
    yy, xx = np.where(valid)
    if len(xx) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), yy, xx
    rng = np.random.default_rng(seed)
    if len(xx) > max_points:
        keep = rng.choice(len(xx), size=max_points, replace=False)
        yy = yy[keep]
        xx = xx[keep]
    z = depth[yy, xx].astype(np.float64)
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    x = (xx.astype(np.float64) - cx) * z / max(fx, 1e-8)
    y = (yy.astype(np.float64) - cy) * z / max(fy, 1e-8)
    xyz = np.stack([x, y, z], axis=1)
    rgb = despill_rgb(image[yy, xx])
    return xyz, rgb, yy, xx


def camera_to_world(points_cam: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    R = extrinsic[:3, :3].astype(np.float64)
    t = extrinsic[:3, 3].astype(np.float64)
    return (points_cam - t[None, :]) @ R


def collect_world_from_depth(
    image: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    xyz_cam, rgb, _yy, _xx = collect_depth_unprojected_points(
        image,
        depth,
        mask,
        intrinsic,
        max_points=max_points,
        seed=seed,
    )
    if len(xyz_cam) == 0:
        return xyz_cam, rgb
    xyz_world = camera_to_world(xyz_cam, extrinsic)
    return xyz_world, rgb


def fit_camera_affine(candidate_points: np.ndarray, target_cam_points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = (
        mask.astype(bool)
        & np.isfinite(candidate_points).all(axis=-1)
        & np.isfinite(target_cam_points).all(axis=-1)
        & (np.linalg.norm(candidate_points, axis=-1) > 1e-6)
        & (np.linalg.norm(target_cam_points, axis=-1) > 1e-6)
    )
    src = candidate_points[valid].astype(np.float64)
    dst = target_cam_points[valid].astype(np.float64)
    if len(src) < 16:
        return np.eye(4, 3, dtype=np.float64)
    keep = np.ones(len(src), dtype=bool)
    for axis in range(3):
        lo, hi = np.percentile(src[:, axis], [2, 98])
        keep &= (src[:, axis] >= lo) & (src[:, axis] <= hi)
    src = src[keep]
    dst = dst[keep]
    A = np.concatenate([src, np.ones((len(src), 1), dtype=np.float64)], axis=1)
    M, *_ = np.linalg.lstsq(A, dst, rcond=None)
    return M


def fit_camera_similarity(candidate_points: np.ndarray, target_cam_points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fit a shape-preserving similarity transform for visualization only.

    The older free affine fit could shear candidate points into slanted or
    torn-looking sheets. A similarity transform preserves the candidate point
    cloud's structure while putting it into the protocol camera-space scale.
    """
    valid = (
        mask.astype(bool)
        & np.isfinite(candidate_points).all(axis=-1)
        & np.isfinite(target_cam_points).all(axis=-1)
        & (np.linalg.norm(candidate_points, axis=-1) > 1e-6)
        & (np.linalg.norm(target_cam_points, axis=-1) > 1e-6)
    )
    src = candidate_points[valid].astype(np.float64)
    dst = target_cam_points[valid].astype(np.float64)
    if len(src) < 16:
        return np.eye(4, 3, dtype=np.float64)

    keep = np.ones(len(src), dtype=bool)
    for arr in (src, dst):
        for axis in range(3):
            lo, hi = np.percentile(arr[:, axis], [2, 98])
            keep &= (arr[:, axis] >= lo) & (arr[:, axis] <= hi)
    src = src[keep]
    dst = dst[keep]
    if len(src) < 16:
        return np.eye(4, 3, dtype=np.float64)

    def solve(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        ma = a.mean(axis=0)
        mb = b.mean(axis=0)
        ac = a - ma
        bc = b - mb
        cov = (ac.T @ bc) / len(a)
        U, S, Vt = np.linalg.svd(cov)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        var = np.mean(np.sum(ac * ac, axis=1))
        scale = float(np.sum(S) / max(var, 1e-12))
        t = mb - scale * (ma @ R.T)
        return scale, R, t

    scale, R, t = solve(src, dst)
    pred = scale * (src @ R.T) + t
    resid = np.linalg.norm(pred - dst, axis=1)
    cutoff = np.percentile(resid, 90)
    robust = resid <= cutoff
    if int(robust.sum()) >= 32:
        scale, R, t = solve(src[robust], dst[robust])
    M = np.zeros((4, 3), dtype=np.float64)
    M[:3, :] = scale * R.T
    M[3, :] = t
    return M


def apply_camera_affine(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    flat = points.reshape(-1, 3).astype(np.float64)
    A = np.concatenate([flat, np.ones((len(flat), 1), dtype=np.float64)], axis=1)
    out = A @ M
    return out.reshape(points.shape).astype(np.float32)


def camera_oblique_project(xyz: np.ndarray, angle_deg: float, pitch: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = xyz.astype(np.float64)
    center = np.median(pts, axis=0)
    q = pts - center
    theta = np.deg2rad(angle_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    x = c * q[:, 0] + s * q[:, 2]
    z = -s * q[:, 0] + c * q[:, 2]
    y = q[:, 1] + pitch * z
    return x, y, z


def camera_depth_point_project(
    xyz: np.ndarray,
    out_size: tuple[int, int],
    seed: int,
    yaw_deg: float,
    pitch: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    W, H = out_size
    pts = xyz.astype(np.float64)
    center = np.median(pts, axis=0)
    q = pts - center
    theta = np.deg2rad(yaw_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    x = c * q[:, 0] + s * q[:, 2]
    z = -s * q[:, 0] + c * q[:, 2]
    y = q[:, 1] + pitch * z
    keep0 = np.ones(len(x), dtype=bool)
    for arr in (x, y, z):
        lo, hi = np.percentile(arr, [0.8, 99.2])
        keep0 &= (arr >= lo) & (arr <= hi)
    if int(keep0.sum()) >= 32:
        x, y, z = x[keep0], y[keep0], z[keep0]
    xlo, xhi = np.percentile(x, [0.8, 99.2])
    ylo, yhi = np.percentile(y, [0.8, 99.2])
    scale = min((W - 78) / max(xhi - xlo, 1e-8), (H - 96) / max(yhi - ylo, 1e-8))
    px = (x - (xlo + xhi) * 0.5) * scale + W * 0.5
    py = (y - (ylo + yhi) * 0.5) * scale + H * 0.50
    zlo, zhi = np.percentile(z, [2, 98])
    zn = np.clip((z - zlo) / max(zhi - zlo, 1e-8), 0.0, 1.0)
    rng = np.random.default_rng(seed)
    px = px + rng.normal(0.0, 0.36, size=len(px))
    py = py + rng.normal(0.0, 0.36, size=len(py))
    if int(keep0.sum()) >= 32:
        full_px = np.zeros(len(keep0), dtype=np.float64)
        full_py = np.zeros(len(keep0), dtype=np.float64)
        full_zn = np.zeros(len(keep0), dtype=np.float64)
        full_px[keep0] = px
        full_py[keep0] = py
        full_zn[keep0] = zn
        return full_px, full_py, full_zn
    return px, py, zn


def depth_aware_project(
    xx: np.ndarray,
    yy: np.ndarray,
    xyz: np.ndarray,
    out_size: tuple[int, int],
    seed: int,
    depth_px: float,
    parallax_sign: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render an upright camera view with point-cloud depth cues.

    A true oblique 3D rotation makes side cameras collapse into slanted strips.
    For mentor figures we keep the camera projection readable, then use the
    candidate camera-space depth for draw order, point size, shading, jitter and
    a small parallax shift. This preserves actual point samples without turning
    the plot into a continuous RGB crop.
    """
    W, H = out_size
    x = xx.astype(np.float64)
    y = yy.astype(np.float64)
    z = xyz[:, 2].astype(np.float64)
    xlo, xhi = np.percentile(x, [0.5, 99.5])
    ylo, yhi = np.percentile(y, [0.5, 99.5])
    usable_w = W - 78
    usable_h = H - 96
    scale = min(usable_w / max(xhi - xlo, 1e-8), usable_h / max(yhi - ylo, 1e-8))
    px = (x - (xlo + xhi) * 0.5) * scale + W * 0.5
    py = (y - (ylo + yhi) * 0.5) * scale + H * 0.50

    zlo, zhi = np.percentile(z, [3, 97])
    zn = np.clip((z - zlo) / max(zhi - zlo, 1e-8), 0.0, 1.0)
    centered = zn - 0.5
    px = px + parallax_sign * depth_px * centered
    py = py - 0.16 * depth_px * centered

    rng = np.random.default_rng(seed)
    px = px + rng.normal(0.0, 0.38, size=len(px))
    py = py + rng.normal(0.0, 0.38, size=len(py))
    return px, py, zn


def draw_depth_cloud(
    image: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    title: str,
    subtitle: str,
    out_size: tuple[int, int],
    max_points: int,
    radius: float,
    seed: int,
    yaw_deg: float,
    pitch: float = 0.055,
    color_mode: str = "rgb",
) -> tuple[Image.Image, int]:
    xyz, rgb, _yy, _xx = collect_depth_unprojected_points(
        image,
        depth,
        mask,
        intrinsic,
        max_points=max_points,
        seed=seed,
    )
    W, H = out_size
    canvas = Image.new("RGB", out_size, "white")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((12, 10), title, fill=(0, 0, 0, 255), font=font(14))
    if len(xyz) == 0:
        draw.text((12, H - 34), subtitle + " | points=0", fill=(0, 0, 0), font=font(12))
        return canvas, 0

    px, py, zn = camera_depth_point_project(xyz, out_size, seed=seed, yaw_deg=yaw_deg, pitch=pitch)
    keep = (px >= 6) & (px < W - 6) & (py >= 28) & (py < H - 42)
    if int(keep.sum()) >= 32:
        px, py, zn, rgb = px[keep], py[keep], zn[keep], rgb[keep]

    order = np.argsort(zn)[::-1]
    for j in order:
        near = 1.0 - float(zn[j])
        if color_mode == "depth":
            # Blue -> tan depth ramp, deliberately not photographic.
            c = (
                int(40 + 170 * near),
                int(70 + 110 * near),
                int(105 + 50 * (1.0 - near)),
            )
        elif color_mode == "gray":
            v = int(42 + 160 * near)
            c = (v, v, v)
        else:
            shade = 0.68 + 0.38 * near
            c = tuple(int(v) for v in np.clip(rgb[j].astype(np.float32) * shade, 0, 255))
        rr = max(0.92, radius * (0.88 + 0.42 * near))
        ox = np.sign(yaw_deg or 1.0) * (1.15 + 0.85 * near)
        oy = 0.72 + 0.36 * near
        # Stronger shadow/outline keeps the result visibly point-cloud-like
        # instead of a resampled photograph.
        draw.ellipse((px[j] + ox - rr, py[j] + oy - rr, px[j] + ox + rr, py[j] + oy + rr), fill=(0, 0, 0, 76))
        draw.ellipse((px[j] - rr - 0.35, py[j] - rr - 0.35, px[j] + rr + 0.35, py[j] + rr + 0.35), outline=(0, 0, 0, 150), width=1)
        draw.ellipse((px[j] - rr, py[j] - rr, px[j] + rr, py[j] + rr), fill=c + (218,))
    draw.text((12, H - 34), f"{subtitle} | points={len(px):,}", fill=(0, 0, 0, 235), font=font(12))
    return canvas, int(len(px))


def fuse_world_points(
    images: np.ndarray,
    depths: np.ndarray,
    masks: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    per_view_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vids: list[np.ndarray] = []
    for i in range(len(images)):
        p, c = collect_world_from_depth(
            images[i],
            depths[i],
            masks[i],
            intrinsics[i],
            extrinsics[i],
            max_points=per_view_points,
            seed=seed + i * 997,
        )
        if len(p) == 0:
            continue
        pts.append(p)
        cols.append(c)
        vids.append(np.full(len(p), i, dtype=np.int16))
    if not pts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8), np.zeros((0,), dtype=np.int16)
    return np.concatenate(pts, axis=0), np.concatenate(cols, axis=0), np.concatenate(vids, axis=0)


def rotation_matrix_z(angle_deg: float) -> np.ndarray:
    t = np.deg2rad(angle_deg)
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_matrix_x(angle_deg: float) -> np.ndarray:
    t = np.deg2rad(angle_deg)
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def draw_fused_world_cloud(
    points: np.ndarray,
    rgb: np.ndarray,
    title: str,
    subtitle: str,
    out_size: tuple[int, int],
    max_points: int,
    radius: float,
    seed: int,
    yaw_deg: float,
    pitch_deg: float = 5.0,
    center_quantiles: tuple[float, float] = (2.0, 98.0),
) -> tuple[Image.Image, int]:
    W, H = out_size
    canvas = Image.new("RGB", out_size, "white")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((12, 10), title, fill=(0, 0, 0, 255), font=font(14))
    valid = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1e-8)
    pts = points[valid].astype(np.float64)
    cols = rgb[valid]
    if len(pts) == 0:
        draw.text((12, H - 34), subtitle + " | points=0", fill=(0, 0, 0), font=font(12))
        return canvas, 0
    rng = np.random.default_rng(seed)
    if len(pts) > max_points:
        keep = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[keep]
        cols = cols[keep]

    lo_q, hi_q = center_quantiles
    keep = np.ones(len(pts), dtype=bool)
    for axis in range(3):
        lo, hi = np.percentile(pts[:, axis], [lo_q, hi_q])
        keep &= (pts[:, axis] >= lo) & (pts[:, axis] <= hi)
    if int(keep.sum()) >= 32:
        pts = pts[keep]
        cols = cols[keep]

    center = np.median(pts, axis=0)
    q = pts - center
    R = rotation_matrix_x(pitch_deg) @ rotation_matrix_z(yaw_deg)
    v = q @ R.T
    x = v[:, 0]
    y = -v[:, 2] + 0.10 * v[:, 1]
    z = v[:, 1]
    xlo, xhi = np.percentile(x, [1, 99])
    ylo, yhi = np.percentile(y, [1, 99])
    scale = min((W - 78) / max(xhi - xlo, 1e-8), (H - 96) / max(yhi - ylo, 1e-8))
    px = (x - (xlo + xhi) * 0.5) * scale + W * 0.5
    py = (y - (ylo + yhi) * 0.5) * scale + H * 0.50
    zlo, zhi = np.percentile(z, [2, 98])
    zn = np.clip((z - zlo) / max(zhi - zlo, 1e-8), 0.0, 1.0)
    px = px + rng.normal(0.0, 0.34, size=len(px))
    py = py + rng.normal(0.0, 0.34, size=len(py))
    keep2 = (px >= 6) & (px < W - 6) & (py >= 28) & (py < H - 42)
    if int(keep2.sum()) >= 32:
        px, py, zn, cols = px[keep2], py[keep2], zn[keep2], cols[keep2]

    order = np.argsort(zn)
    for j in order:
        near = float(zn[j])
        shade = 0.76 + 0.32 * near
        c = tuple(int(vv) for vv in np.clip(cols[j].astype(np.float32) * shade, 0, 255))
        rr = max(0.70, radius * (0.82 + 0.34 * near))
        ox = 0.46 + 0.32 * near
        oy = 0.38 + 0.24 * near
        draw.ellipse((px[j] + ox - rr, py[j] + oy - rr, px[j] + ox + rr, py[j] + oy + rr), fill=(0, 0, 0, 52))
        draw.ellipse((px[j] - rr, py[j] - rr, px[j] + rr, py[j] + rr), fill=c + (224,))
    draw.text((12, H - 34), f"{subtitle} | points={len(px):,}", fill=(0, 0, 0, 235), font=font(12))
    return canvas, int(len(px))


def draw_cloud(
    image: np.ndarray,
    points: np.ndarray,
    mask: np.ndarray,
    title: str,
    subtitle: str,
    out_size: tuple[int, int],
    max_points: int,
    radius: float,
    seed: int,
    depth_px: float = 16.0,
    parallax_sign: float = 1.0,
    mode: str = "upright",
) -> tuple[Image.Image, int]:
    xyz, rgb, yy, xx = collect_points(image, points, mask, max_points=max_points, seed=seed)
    W, H = out_size
    canvas = Image.new("RGB", out_size, "white")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((12, 10), title, fill=(0, 0, 0, 255), font=font(14))
    if len(xyz) == 0:
        draw.text((12, H - 34), subtitle + " | points=0", fill=(0, 0, 0), font=font(12))
        return canvas, 0
    if mode == "oblique":
        x, y, z = camera_oblique_project(xyz, angle_deg=parallax_sign * depth_px, pitch=0.06)
        keep0 = np.ones(len(x), dtype=bool)
        for arr in (x, y, z):
            lo, hi = np.percentile(arr, [1, 99])
            keep0 &= (arr >= lo) & (arr <= hi)
        if int(keep0.sum()) >= 32:
            x, y, z, rgb = x[keep0], y[keep0], z[keep0], rgb[keep0]
        xlo, xhi = np.percentile(x, [1, 99])
        ylo, yhi = np.percentile(y, [1, 99])
        scale = min((W - 78) / max(xhi - xlo, 1e-8), (H - 96) / max(yhi - ylo, 1e-8))
        px = (x - (xlo + xhi) * 0.5) * scale + W * 0.5
        py = (y - (ylo + yhi) * 0.5) * scale + H * 0.50
        zlo, zhi = np.percentile(z, [2, 98])
        zn = np.clip((z - zlo) / max(zhi - zlo, 1e-8), 0.0, 1.0)
        rng = np.random.default_rng(seed)
        px = px + rng.normal(0.0, 0.32, size=len(px))
        py = py + rng.normal(0.0, 0.32, size=len(py))
    else:
        px, py, zn = depth_aware_project(xx, yy, xyz, out_size, seed=seed, depth_px=depth_px, parallax_sign=parallax_sign)
    keep = (px >= 6) & (px < W - 6) & (py >= 28) & (py < H - 42)
    if int(keep.sum()) >= 32:
        px, py, zn, rgb = px[keep], py[keep], zn[keep], rgb[keep]
    order = np.argsort(zn)[::-1]
    for j in order:
        # Larger and brighter near-camera points give a visible point-cloud
        # depth ordering while keeping the subject upright and readable.
        near = 1.0 - float(zn[j])
        shade = 0.78 + 0.30 * near
        c = tuple(int(v) for v in np.clip(rgb[j].astype(np.float32) * shade, 0, 255))
        rr = max(0.72, radius * (0.82 + 0.34 * near))
        ox = parallax_sign * (0.60 + 0.45 * near)
        oy = 0.45 + 0.25 * near
        draw.ellipse((px[j] + ox - rr, py[j] + oy - rr, px[j] + ox + rr, py[j] + oy + rr), fill=(0, 0, 0, 52))
        draw.ellipse((px[j] - rr, py[j] - rr, px[j] + rr, py[j] + rr), fill=c + (228,))
    draw.text((12, H - 34), f"{subtitle} | points={len(xyz):,}", fill=(0, 0, 0, 235), font=font(12))
    return canvas, int(len(xyz))


def make_sheet(path: Path, items: list[tuple[str, Image.Image]], cols: int, thumb: tuple[int, int]) -> None:
    tw, th = thumb
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * tw, rows * (th + 38)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate(items):
        x = (i % cols) * tw
        y = (i // cols) * (th + 38)
        sheet.paste(im, (x, y))
        draw.text((x + 10, y + th + 8), label, fill=(0, 0, 0), font=font(14))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> int:
    return _run_v50r2_view_consistent_replacement()
    IMG.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    inp = load_npz(CASE / "inputs.npz")
    targets = load_npz(CASE / "targets.npz")
    cand = load_npz(PKG / "candidate_files__candidate_points.npz")
    head = load_npz(PKG / "candidate_files__head_face_patch.npz")
    hand = load_npz(PKG / "candidate_files__hand_patch.npz")
    candidate_depths = load_npz(V32 / "candidate_depths_research.npz")["candidate_depths"]

    images = inp["images"]
    masks = inp["point_masks"].astype(bool)
    intrinsics = targets["intrinsics"]
    extrinsics = targets["extrinsics"]
    cams = [str(x) for x in inp["camera_ids"]]
    full_points = cand["candidate_points_world"]
    head_points = head["refined_points_world"]
    hand_points = hand["hand_points_world"]
    face_mask = head["face_mask"].astype(bool)
    head_mask = head["head_mask"].astype(bool)
    hand_region = hand["hand_region_id_map"].astype(np.uint8)
    target_cam_points = targets["cam_points"]
    transforms = [fit_camera_similarity(full_points[i], target_cam_points[i], masks[i]) for i in range(len(cams))]
    full_points_view = np.stack([apply_camera_affine(full_points[i], transforms[i]) for i in range(len(cams))], axis=0)
    head_points_view = np.stack([apply_camera_affine(head_points[i], transforms[i]) for i in range(len(cams))], axis=0)
    hand_points_view = np.stack([apply_camera_affine(hand_points[i], transforms[i]) for i in range(len(cams))], axis=0)

    records: list[dict[str, object]] = []
    full_items: list[tuple[str, Image.Image]] = []
    full_depth_items: list[tuple[str, Image.Image]] = []
    for i, cam in enumerate(cams):
        yaw = -5.0 if cam in {"15", "30"} else 5.0
        im, n = draw_depth_cloud(
            images[i],
            candidate_depths[i],
            masks[i],
            intrinsics[i],
            f"V50R2 sparse RGB point cloud cam{cam} full",
            "depth-unprojected camera point cloud",
            (520, 390),
            2200,
            1.62,
            10 + i,
            yaw_deg=yaw,
            pitch=0.09,
            color_mode="rgb",
        )
        full_items.append((f"cam{cam} full", im))
        im.save(IMG / f"kinect_style_cam{cam}_full.png")
        records.append({"camera": cam, "region": "full", "points": n, "image": f"kinect_style_cam{cam}_full.png"})
        im_depth, n_depth = draw_depth_cloud(
            images[i],
            candidate_depths[i],
            masks[i],
            intrinsics[i],
            f"V50R2 depth-colored point cloud cam{cam} full",
            "same points, depth-color proof",
            (520, 390),
            2200,
            1.62,
            900 + i,
            yaw_deg=yaw,
            pitch=0.09,
            color_mode="depth",
        )
        full_depth_items.append((f"cam{cam} depth", im_depth))
        im_depth.save(IMG / f"kinect_style_cam{cam}_full_depthcolor.png")
        records.append({"camera": cam, "region": "full_depthcolor", "points": n_depth, "image": f"kinect_style_cam{cam}_full_depthcolor.png"})

    head_items: list[tuple[str, Image.Image]] = []
    head_cams = ["15", "30", "45", "59"]
    cam_to_i = {c: i for i, c in enumerate(cams)}
    for cam in head_cams:
        if cam not in cam_to_i:
            continue
        i = cam_to_i[cam]
        for region, roi, title in [("face2d", face_mask[i] & masks[i], "face"), ("head2d", head_mask[i] & masks[i], "head")]:
            yaw = -8.0 if cam in {"15", "30"} else 8.0
            im, n = draw_depth_cloud(
                images[i],
                candidate_depths[i],
                roi,
                intrinsics[i],
                f"V50R2 depth-unprojected RGB point cloud cam{cam} {title}",
                "head/face ROI, camera-space point cloud",
                (480, 360),
                1500,
                1.68,
                200 + i,
                yaw_deg=yaw,
                pitch=0.10,
                color_mode="rgb",
            )
            head_items.append((f"cam{cam}_{region}", im))
            im.save(IMG / f"kinect_style_cam{cam}_{region}.png")
            records.append({"camera": cam, "region": region, "points": n, "image": f"kinect_style_cam{cam}_{region}.png"})

    # Reorder to match the Kinect reference: top row face2d, bottom row head2d.
    ordered_head: list[tuple[str, Image.Image]] = []
    for region in ["face2d", "head2d"]:
        for cam in head_cams:
            p = IMG / f"kinect_style_cam{cam}_{region}.png"
            if p.exists():
                ordered_head.append((f"cam{cam}_{region}", Image.open(p).convert("RGB")))

    hand_items: list[tuple[str, Image.Image]] = []
    for cam in ["15", "30", "59"]:
        if cam not in cam_to_i:
            continue
        i = cam_to_i[cam]
        for rid, region in [(1, "left_hand"), (2, "right_hand")]:
            roi = (hand_region[i] == rid) & masks[i]
            yaw = -5.0 if cam in {"15", "30"} else 5.0
            im, n = draw_depth_cloud(
                images[i],
                candidate_depths[i],
                roi,
                intrinsics[i],
                f"V50R2 sparse RGB point cloud cam{cam} {region}",
                "hand ROI camera point cloud",
                (520, 390),
                1500,
                1.70,
                300 + i * 10 + rid,
                yaw_deg=yaw,
                pitch=0.10,
                color_mode="rgb",
            )
            hand_items.append((f"cam{cam} {region}", im))
            im.save(IMG / f"kinect_style_cam{cam}_{region}.png")
            records.append({"camera": cam, "region": region, "points": n, "image": f"kinect_style_cam{cam}_{region}.png"})

    full_sheet = IMG / "kinect_style_full_body_3d_pointcloud_sheet.png"
    full_depth_sheet = IMG / "kinect_style_full_body_depthcolor_pointcloud_sheet.png"
    head_sheet = IMG / "kinect_style_head_face_3d_pointcloud_sheet.png"
    hand_sheet = IMG / "kinect_style_hands_3d_pointcloud_sheet.png"
    make_sheet(full_sheet, full_items, cols=3, thumb=(520, 390))
    make_sheet(full_depth_sheet, full_depth_items, cols=3, thumb=(520, 390))
    make_sheet(head_sheet, ordered_head, cols=4, thumb=(480, 360))
    make_sheet(hand_sheet, hand_items, cols=3, thumb=(520, 390))

    # Publish these as the current mentor-report point-cloud figures.
    main_img_dir = ROOT / "output" / "mentor_report_v50r2" / "images"
    main_img_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in [
        (full_sheet, main_img_dir / "01_full_body.png"),
        (head_sheet, main_img_dir / "02_head_face.png"),
        (IMG / "kinect_style_cam30_head2d.png", main_img_dir / "03_hairline.png"),
        (hand_sheet, main_img_dir / "04_left_hand.png"),
        (hand_sheet, main_img_dir / "05_right_hand.png"),
        (full_depth_sheet, main_img_dir / "06_full_body_depthcolor_pointcloud.png"),
    ]:
        if src.exists():
            Image.open(src).save(dst)

    report = {
        "task": "v223_make_kinect_style_3d_pointcloud_sheets",
        "created_utc": now(),
        "output_dir": str(IMG.resolve()),
        "sheets": {
            "full_body": str(full_sheet.resolve()),
            "full_body_depthcolor": str(full_depth_sheet.resolve()),
            "head_face": str(head_sheet.resolve()),
            "hands": str(hand_sheet.resolve()),
        },
        "records": records,
        "render_policy": "Sparse depth-unprojected per-camera point-cloud rendering. Candidate_depths_research is unprojected with protocol intrinsics; full/head/face/hand masks select points; small yaw/pitch is used only for point-cloud depth cues. This intentionally avoids unsafe 6-view world fusion because the V50R2 candidate depth is per-view visible-surface evidence, not a closed fused object. Points are RGB-colored sparse dots with depth ordering and shadow dots, no continuous RGB crop and no mesh surface. V50 frozen candidate artifacts are read-only.",
    }
    json_path = REPORTS / "20260509_v50r2_kinect_style_3d_pointcloud_sheets.json"
    md_path = REPORTS / "20260509_v50r2_kinect_style_3d_pointcloud_sheets.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(
        "\n".join([
            "# V50R2 Kinect-Style 3D RGB Point Cloud Sheets",
            "",
            f"- full body: `{full_sheet.resolve()}`",
            f"- head/face: `{head_sheet.resolve()}`",
            f"- hands: `{hand_sheet.resolve()}`",
            "",
            "These are depth-unprojected camera-space point-cloud renderings, not camera-view RGB crops.",
        ]),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
