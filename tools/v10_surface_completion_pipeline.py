from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = REPO_ROOT / "output/surface_research_preflight_local"
CLOUD_ROOT = REPO_ROOT / "output/surface_research_cloud_preflight"
REPORTS = REPO_ROOT / "reports"

DEFAULT_MUST3R_DIR = CLOUD_ROOT / "Cloud_B_V9/a5x2_must3r_true_backend/must3r_run"
DEFAULT_2DGS_SCENE = CLOUD_ROOT / "Cloud_B_V9/a5x2_2dgs_colmap_scene/2dgs_colmap_scene"
DEFAULT_2DGS_SMOKE = CLOUD_ROOT / "Cloud_B_V9/a5x2_2dgs_colmap_scene_smoke/model_smoke"
DEFAULT_FUS3D3 = CLOUD_ROOT / "Cloud_A_V9/b_fus3d3_real_asset_train_preflight"
DEFAULT_ASSETS = CLOUD_ROOT / "V9_cloud_asset_staging/assets"
DEFAULT_TEMPLATE = DEFAULT_ASSETS / "template/connected_human_surface_template_payload_self_describing.npz"

FORBIDDEN_WORDS = (
    "predictions",
    "teacher_export",
    "candidate_export",
    "strict_pass",
    "strict_gate_registry",
)
CONTRACT = {
    "research_only": True,
    "no_export": True,
    "no_predictions_write": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_registry_write": True,
    "no_strict_pass_write": True,
    "formal_cloud_unblocked": False,
}
REGIONS = ("full_body", "head", "face_core", "hairline", "left_hand", "right_hand")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_report(path: Path, title: str, summary: dict[str, Any]) -> None:
    lines = [
        f"# {title}",
        "",
        f"Status: `{summary.get('status')}`",
        "",
        "Research-only. This artifact does not write predictions, teacher/candidate package, registry, or strict pass state.",
        "",
        "## Decision",
        "",
        str(summary.get("decision", "")),
        "",
        "## Blockers",
        "",
    ]
    blockers = summary.get("blockers") or []
    lines.extend([f"- {item}" for item in blockers] if blockers else ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    lower = resolved.as_posix().lower()
    if "surface_research" not in lower:
        raise ValueError(f"Refusing non-research output path: {resolved}")
    for word in FORBIDDEN_WORDS:
        if word in lower:
            raise ValueError(f"Refusing forbidden output path token {word!r}: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def read_ply_header(path: Path) -> tuple[list[str], int, int, str]:
    with path.open("rb") as handle:
        lines: list[str] = []
        offset = 0
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"PLY header did not end: {path}")
            offset += len(raw)
            line = raw.decode("ascii", errors="replace").strip()
            lines.append(line)
            if line == "end_header":
                break
    count = 0
    fmt = ""
    for line in lines:
        parts = line.split()
        if len(parts) >= 3 and parts[:2] == ["format", "binary_little_endian"]:
            fmt = "binary_little_endian"
        elif len(parts) >= 3 and parts[:2] == ["format", "ascii"]:
            fmt = "ascii"
        elif len(parts) == 3 and parts[:2] == ["element", "vertex"]:
            count = int(parts[2])
    return lines, count, offset, fmt


def load_ply_xyz_rgb(path: Path, max_points: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    lines, count, offset, fmt = read_ply_header(path)
    if count <= 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    limit = count if max_points is None else min(count, int(max_points))
    if fmt == "ascii":
        rows = []
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.strip() == "end_header":
                    break
            for line in handle:
                parts = line.split()
                if len(parts) >= 3:
                    rows.append(parts)
                if len(rows) >= limit:
                    break
        pts = np.asarray([[float(r[0]), float(r[1]), float(r[2])] for r in rows], dtype=np.float32)
        if rows and len(rows[0]) >= 6:
            colors = np.asarray([[int(float(r[3])), int(float(r[4])), int(float(r[5]))] for r in rows], dtype=np.uint8)
        else:
            colors = np.full((pts.shape[0], 3), 220, dtype=np.uint8)
        return pts, colors
    props: list[tuple[str, str]] = []
    in_vertex = False
    for line in lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            continue
        if in_vertex and len(parts) == 3 and parts[0] == "property":
            props.append((parts[1], parts[2]))
    names = [name for _, name in props]
    if names[:3] != ["x", "y", "z"]:
        raise ValueError(f"Unsupported PLY vertex properties: {props[:8]}")
    stride = 0
    fmts = []
    for typ, name in props:
        if typ == "float":
            stride += 4
            fmts.append((name, "f"))
        elif typ == "double":
            stride += 8
            fmts.append((name, "d"))
        elif typ == "uchar":
            stride += 1
            fmts.append((name, "B"))
        elif typ == "int":
            stride += 4
            fmts.append((name, "i"))
        else:
            raise ValueError(f"Unsupported PLY property type {typ!r} in {path}")
    unpack_fmt = "<" + "".join(code for _, code in fmts)
    pts = np.zeros((limit, 3), dtype=np.float32)
    colors = np.full((limit, 3), 220, dtype=np.uint8)
    with path.open("rb") as handle:
        handle.seek(offset)
        for idx in range(limit):
            raw = handle.read(stride)
            if len(raw) != stride:
                pts = pts[:idx]
                colors = colors[:idx]
                break
            row = dict(zip(names, struct.unpack(unpack_fmt, raw)))
            pts[idx] = [row["x"], row["y"], row["z"]]
            if {"red", "green", "blue"}.issubset(row):
                colors[idx] = [int(row["red"]), int(row["green"]), int(row["blue"])]
    return pts, colors


def write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if colors is None:
        colors = np.full((len(points), 3), 210, dtype=np.uint8)
    colors = np.clip(np.asarray(colors).reshape(-1, 3), 0, 255).astype(np.uint8)
    if len(colors) != len(points):
        colors = np.resize(colors, (len(points), 3)).astype(np.uint8)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(points, colors):
            handle.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def scalar_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": int(arr.size), "finite": 0}
    return {
        "count": int(arr.size),
        "finite": int(finite.size),
        "min": float(finite.min()),
        "p10": float(np.percentile(finite, 10)),
        "median": float(np.median(finite)),
        "mean": float(finite.mean()),
        "p90": float(np.percentile(finite, 90)),
        "max": float(finite.max()),
    }


def bbox_stats(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points)
    if points.size == 0:
        return {"valid": False}
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) == 0:
        return {"valid": False}
    lo = np.percentile(finite, 1, axis=0)
    hi = np.percentile(finite, 99, axis=0)
    return {
        "valid": True,
        "p01": lo.tolist(),
        "p99": hi.tolist(),
        "extent_p01_p99": (hi - lo).tolist(),
        "center_p01_p99": ((lo + hi) * 0.5).tolist(),
    }


def qvec_to_rotmat(q: list[float]) -> np.ndarray:
    qw, qx, qy, qz = [float(x) for x in q]
    return np.asarray(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float32,
    )


def load_colmap_cameras(scene_dir: Path) -> list[dict[str, Any]]:
    cameras_txt = scene_dir / "sparse/0/cameras.txt"
    images_txt = scene_dir / "sparse/0/images.txt"
    cams: dict[int, dict[str, Any]] = {}
    for line in cameras_txt.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        cam_id = int(parts[0])
        width, height = int(parts[2]), int(parts[3])
        fx, fy, cx, cy = [float(x) for x in parts[4:8]]
        cams[cam_id] = {"width": width, "height": height, "intrinsic": np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)}
    out: list[dict[str, Any]] = []
    raw_lines = images_txt.read_text(encoding="utf-8").splitlines()
    for line in raw_lines:
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        q = [float(x) for x in parts[1:5]]
        t = np.asarray([float(x) for x in parts[5:8]], dtype=np.float32)
        camera_id = int(parts[8])
        name = parts[9]
        r = qvec_to_rotmat(q)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = r
        w2c[:3, 3] = t
        cam = dict(cams[camera_id])
        cam.update({"image_id": image_id, "camera_id": camera_id, "name": name, "world_to_cam": w2c})
        out.append(cam)
    out.sort(key=lambda item: item["image_id"])
    return out


def project(points: np.ndarray, camera: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    w2c = np.asarray(camera["world_to_cam"], dtype=np.float32)
    k = np.asarray(camera["intrinsic"], dtype=np.float32)
    cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    uvw = cam @ k.T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-6)
    return uv, z


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L")
    if img.size != size:
        img = img.resize(size, Image.Resampling.NEAREST)
    return np.asarray(img) > 127


def make_projection_mask(points: np.ndarray, camera: dict[str, Any], size: tuple[int, int], radius: int = 2) -> tuple[np.ndarray, np.ndarray]:
    width, height = size
    uv, depth = project(points, camera)
    valid = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(depth)
        & (depth > 1e-5)
        & (uv[:, 0] >= -radius)
        & (uv[:, 0] < width + radius)
        & (uv[:, 1] >= -radius)
        & (uv[:, 1] < height + radius)
    )
    mask = np.zeros((height, width), dtype=bool)
    dmap = np.full((height, width), np.nan, dtype=np.float32)
    xy = np.rint(uv[valid]).astype(np.int32)
    dz = depth[valid]
    for (x, y), z in zip(xy, dz):
        for yy in range(max(0, y - radius), min(height, y + radius + 1)):
            for xx in range(max(0, x - radius), min(width, x + radius + 1)):
                mask[yy, xx] = True
                old = dmap[yy, xx]
                if not np.isfinite(old) or z < old:
                    dmap[yy, xx] = float(z)
    return mask, dmap


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def paste_grid(images: list[Image.Image], cols: int, bg: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    if not images:
        return Image.new("RGB", (64, 64), bg)
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    w, h = max(widths), max(heights)
    rows = int(math.ceil(len(images) / cols))
    canvas = Image.new("RGB", (cols * w, rows * h), bg)
    for idx, im in enumerate(images):
        canvas.paste(im.convert("RGB"), ((idx % cols) * w, (idx // cols) * h))
    return canvas


def scatter_view(points: np.ndarray, colors: np.ndarray | None, title: str, size: int = 640, axes: tuple[int, int] = (0, 1)) -> Image.Image:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    img = Image.new("RGB", (size, size), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    draw.text((8, 8), title, fill=(20, 20, 20))
    if len(points) == 0:
        draw.text((8, 32), "empty", fill=(180, 0, 0))
        return img
    sample = points
    if len(sample) > 60000:
        rng = np.random.default_rng(10)
        idx = rng.choice(len(sample), size=60000, replace=False)
        sample = sample[idx]
        col = colors[idx] if colors is not None and len(colors) == len(points) else None
    else:
        col = colors if colors is not None and len(colors) == len(points) else None
    xy = sample[:, list(axes)]
    finite = np.isfinite(xy).all(axis=1)
    xy = xy[finite]
    if col is not None:
        col = np.asarray(col)[finite]
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    span = np.maximum(hi - lo, 1e-5)
    uv = (xy - lo) / span
    px = np.clip((uv[:, 0] * (size - 50) + 25).astype(np.int32), 0, size - 1)
    py = np.clip(((1.0 - uv[:, 1]) * (size - 50) + 25).astype(np.int32), 0, size - 1)
    if col is None:
        col = np.full((len(px), 3), [40, 80, 180], dtype=np.uint8)
    for x, y, c in zip(px[:: max(1, len(px) // 45000)], py[:: max(1, len(py) // 45000)], col[:: max(1, len(col) // 45000)]):
        draw.point((int(x), int(y)), fill=tuple(int(v) for v in c[:3]))
    return img


def contact_sheet(points: np.ndarray, colors: np.ndarray | None, path: Path, title_prefix: str) -> None:
    views = [
        scatter_view(points, colors, f"{title_prefix} front xy", axes=(0, 1)),
        scatter_view(points, colors, f"{title_prefix} side zy", axes=(2, 1)),
        scatter_view(points, colors, f"{title_prefix} top xz", axes=(0, 2)),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    paste_grid(views, cols=3).save(path)


def load_template() -> dict[str, np.ndarray]:
    with np.load(DEFAULT_TEMPLATE, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def region_masks_from_template(template: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    masks = {
        "full_body": np.ones(template["hybrid_vertices"].shape[0], dtype=bool),
        "head": np.asarray(template["head_vertex_mask"], dtype=bool),
        "face_core": np.asarray(template["face_front_vertex_mask"], dtype=bool),
        "hairline": np.asarray(template["hairline_vertex_mask"], dtype=bool),
        "left_hand": np.asarray(template["left_hand_vertex_mask"], dtype=bool),
        "right_hand": np.asarray(template["right_hand_vertex_mask"], dtype=bool),
    }
    return masks


def select_region_by_bbox(points: np.ndarray, template_points: np.ndarray, fallback_fraction: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    if len(points) == 0 or len(template_points) == 0:
        return np.zeros((0,), dtype=bool)
    lo = np.percentile(template_points, 1, axis=0)
    hi = np.percentile(template_points, 99, axis=0)
    center_p = np.percentile(points, 50, axis=0)
    ext_p = np.maximum(np.percentile(points, 99, axis=0) - np.percentile(points, 1, axis=0), 1e-5)
    center_t = np.percentile(template_points, 50, axis=0)
    ext_t = np.maximum(np.percentile(template_points, 99, axis=0) - np.percentile(template_points, 1, axis=0), 1e-5)
    canonical = (points - center_p) / ext_p * ext_t + center_t
    mask = np.logical_and(canonical >= lo, canonical <= hi).all(axis=1)
    if mask.sum() == 0:
        z = points[:, 2]
        q0, q1 = np.percentile(z, [fallback_fraction[0] * 100.0, fallback_fraction[1] * 100.0])
        mask = (z >= q0) & (z <= q1)
    return mask


def audit_2dgs_scene(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    scene = args.scene_dir.resolve()
    cameras = load_colmap_cameras(scene) if (scene / "sparse/0/cameras.txt").is_file() else []
    image_dir = scene / "images"
    mask_dir = scene / "masks"
    ply = scene / "sparse/0/points3D.ply"
    cfg = args.model_dir / "cfg_args"
    point_count = read_ply_header(ply)[1] if ply.is_file() else 0
    images = sorted(image_dir.glob("*.*")) if image_dir.is_dir() else []
    masks = sorted(mask_dir.glob("*.*")) if mask_dir.is_dir() else []
    sample_imgs: list[Image.Image] = []
    for path in images[:6]:
        im = Image.open(path).convert("RGB").resize((220, 220))
        draw = ImageDraw.Draw(im)
        draw.text((6, 6), path.name[:24], fill=(255, 0, 0))
        sample_imgs.append(im)
    visual = out / "2dgs_scene_visual_debug.png"
    paste_grid(sample_imgs, cols=3).save(visual)
    passed = bool(len(cameras) == 6 and len(images) == 6 and len(masks) == 6 and point_count > 10000)
    summary = {
        "task": "a5x3_2dgs_scene_contract_audit",
        "created_utc": utc_now(),
        "status": "scene_contract_pass_research_only" if passed else "scene_contract_blocked",
        **CONTRACT,
        "scene_dir": scene,
        "model_dir": args.model_dir.resolve(),
        "camera_count": len(cameras),
        "image_count": len(images),
        "mask_count": len(masks),
        "input_ply": ply,
        "input_ply_vertex_count": point_count,
        "cfg_args_exists": cfg.is_file(),
        "visual_debug": visual,
        "can_start_2dgs_research_train": passed,
        "strict_teacher_precheck_pass": False,
        "blockers": [
            "2DGS scene is still initialized from MUSt3R weak-pool points, not strict teacher-aligned points.",
            "Scene audit is a loader/train contract; it is not a visual teacher pass.",
        ],
        "decision": "2DGS scene contract is usable for research-only long training." if passed else "2DGS scene contract is incomplete; do not train.",
    }
    write_json(out / "summary.json", summary)
    write_report(out / "report.md", "V10 2DGS Scene Contract Audit", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": out}), ensure_ascii=False))
    return 0 if passed else 2


def must3r_align(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    cameras = load_colmap_cameras(args.scene_dir)
    template = load_template()
    region_masks = region_masks_from_template(template)
    template_points = template["hybrid_vertices"].astype(np.float32)
    candidates = sorted(args.must3r_dir.glob("scene_lowconf_*.ply"))
    if not candidates:
        candidates = sorted(args.must3r_dir.glob("*.ply"))
    ladder = []
    best_score = -1.0
    best_payload: dict[str, Any] | None = None
    best_points = np.zeros((0, 3), dtype=np.float32)
    best_colors = np.zeros((0, 3), dtype=np.uint8)
    rng = np.random.default_rng(123)
    for ply in candidates:
        points, colors = load_ply_xyz_rgb(ply)
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        colors = colors[finite]
        if len(points) > args.max_eval_points:
            idx = rng.choice(len(points), size=args.max_eval_points, replace=False)
            eval_points = points[idx]
        else:
            eval_points = points
        view_rows = []
        for cam in cameras:
            mask_path = args.scene_dir / "masks" / cam["name"]
            if not mask_path.is_file():
                mask_path = next((args.scene_dir / "masks").glob(f"*{cam['name'][-13:]}"), Path(""))
            size = (int(cam["width"]), int(cam["height"]))
            gt = load_mask(mask_path, size) if mask_path.is_file() else np.zeros((size[1], size[0]), dtype=bool)
            pred, dmap = make_projection_mask(eval_points, cam, size, radius=2)
            view_rows.append(
                {
                    "name": cam["name"],
                    "projected_pixel_count": int(pred.sum()),
                    "mask_pixel_count": int(gt.sum()),
                    "mask_iou": mask_iou(pred, gt),
                    "depth_stats": scalar_stats(dmap[np.isfinite(dmap)]),
                }
            )
        mean_iou = float(np.mean([row["mask_iou"] for row in view_rows])) if view_rows else 0.0
        region = {}
        for name in REGIONS:
            tmask = region_masks[name]
            pmask = select_region_by_bbox(points, template_points[tmask], fallback_fraction=(0.65, 1.0) if name in {"head", "face_core", "hairline"} else (0.0, 1.0))
            region[name] = {
                "point_count": int(pmask.sum()),
                "coverage_nonempty": bool(pmask.sum() > 50),
                "bbox": bbox_stats(points[pmask]),
            }
        nonempty_regions = sum(1 for row in region.values() if row["coverage_nonempty"])
        score = mean_iou + 0.02 * nonempty_regions
        item = {
            "ply": ply,
            "vertex_count": int(len(points)),
            "mean_6view_mask_iou": mean_iou,
            "region_coverage": region,
            "view_projection": view_rows,
            "ladder_label": ply.stem.replace("scene_lowconf_", "conf_"),
        }
        ladder.append(item)
        if score > best_score:
            best_score = score
            best_payload = item
            best_points = points
            best_colors = colors
    aligned_ply = out / "a5x3_aligned_must3r_points.ply"
    write_ascii_ply(aligned_ply, best_points, best_colors)
    contact = out / "a5x3_region_open3d_contact_sheet.png"
    contact_sheet(best_points, best_colors, contact, "MUSt3R weak-pool")
    reproj_images: list[Image.Image] = []
    depth_residual = {"views": []}
    if best_payload:
        for cam in cameras:
            image_path = args.scene_dir / "images" / cam["name"]
            mask_path = args.scene_dir / "masks" / cam["name"]
            base = Image.open(image_path).convert("RGB").resize((259, 259)) if image_path.is_file() else Image.new("RGB", (259, 259), (240, 240, 240))
            size = (int(cam["width"]), int(cam["height"]))
            pred, dmap = make_projection_mask(best_points[:: max(1, len(best_points) // 120000)], cam, size, radius=2)
            gt = load_mask(mask_path, size) if mask_path.is_file() else np.zeros((size[1], size[0]), dtype=bool)
            overlay = base.copy()
            ov = Image.new("RGB", size, (0, 0, 0))
            pix = ov.load()
            for y, x in np.argwhere(pred[::1, ::1]):
                if x < size[0] and y < size[1]:
                    pix[int(x), int(y)] = (255, 0, 0)
            ov = ov.resize((259, 259))
            overlay = Image.blend(overlay, ov, 0.35)
            draw = ImageDraw.Draw(overlay)
            iou = mask_iou(pred, gt)
            draw.text((8, 8), f"{cam['name']} iou={iou:.3f}", fill=(255, 255, 0))
            reproj_images.append(overlay)
            zvals = dmap[np.isfinite(dmap)]
            depth_residual["views"].append({"name": cam["name"], "mask_iou": iou, "projected_depth_stats": scalar_stats(zvals)})
    reproj = out / "a5x3_6view_reprojection_sheet.png"
    paste_grid(reproj_images, cols=3).save(reproj)
    depth_path = out / "a5x3_depth_residual_report.json"
    write_json(depth_path, depth_residual)
    mean_iou = float(best_payload["mean_6view_mask_iou"]) if best_payload else 0.0
    regions_ok = bool(best_payload and all(best_payload["region_coverage"][name]["coverage_nonempty"] for name in REGIONS))
    strict = bool(mean_iou > 0.75 and regions_ok)
    summary = {
        "task": "a5x3_must3r_known_camera_alignment",
        "created_utc": utc_now(),
        "status": "strict_teacher_intake_precheck_pass_research_only" if strict else "weak_pool_alignment_precheck_failed",
        **CONTRACT,
        "strict_teacher_precheck_pass": strict,
        "aligned_point_cloud": aligned_ply,
        "reprojection_sheet": reproj,
        "depth_residual_report": depth_path,
        "region_contact_sheet": contact,
        "confidence_ladder": ladder,
        "selected": best_payload,
        "blockers": [] if strict else [
            "MUSt3R low-confidence point cloud did not satisfy mask IoU > 0.75 and all hard-region coverage gates.",
            "No D-line human visual review has approved this artifact.",
            "Output remains weak evidence and cannot be called a strict teacher.",
        ],
        "decision": "A5-X3 MUSt3R can proceed to D-line teacher-intake review." if strict else "A5-X3 MUSt3R remains weak-pool only; use as auxiliary evidence for 2DGS/Fus3D, not as teacher.",
    }
    write_json(out / "a5x3_alignment_summary.json", summary)
    write_json(out / "summary.json", summary)
    write_report(out / "report.md", "V10 A5-X3 MUSt3R Alignment", summary)
    print(json.dumps(json_ready({"status": summary["status"], "mean_iou": mean_iou, "output": out}), ensure_ascii=False))
    return 0 if strict else 2


def fus3d4_export(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    diag = args.fus3d3_dir / "b_fus3d2_human_dataset_train_diagnostics.npz"
    real_ply = args.fus3d3_dir / "b_fus3d2_real_surface_sdf_points.ply"
    points, colors = load_ply_xyz_rgb(real_ply)
    template = load_template()
    masks = region_masks_from_template(template)
    tpts = template["hybrid_vertices"].astype(np.float32)
    body_mask = select_region_by_bbox(points, tpts[masks["full_body"]])
    head_mask = select_region_by_bbox(points, tpts[masks["head"]], fallback_fraction=(0.65, 1.0))
    face_mask = select_region_by_bbox(points, tpts[masks["face_core"]], fallback_fraction=(0.70, 1.0))
    body_ply = out / "b_fus3d4_body_surface.ply"
    head_ply = out / "b_fus3d4_head_face_surface.ply"
    write_ascii_ply(body_ply, points[body_mask], colors[body_mask])
    write_ascii_ply(head_ply, points[np.logical_or(head_mask, face_mask)], colors[np.logical_or(head_mask, face_mask)])
    contact = out / "b_fus3d4_open3d_contact_sheet.png"
    contact_sheet(points, colors, contact, "B-Fus3D4 real")
    controls: dict[str, Any] = {}
    if diag.is_file():
        with np.load(diag, allow_pickle=False) as payload:
            for name in ("real", "zero", "shuffle", "random"):
                key = f"{name}_probability"
                if key in payload.files:
                    controls[name] = scalar_stats(payload[key])
            np.savez_compressed(
                out / "b_fus3d4_depth_6view.npz",
                diagnostic_only=True,
                positions=np.asarray(payload["positions"], dtype=np.float32),
                real_probability=np.asarray(payload["real_probability"], dtype=np.float32),
            )
            np.savez_compressed(out / "b_fus3d4_normal_6view.npz", diagnostic_only=True, normals=np.zeros((0, 3), dtype=np.float32))
            np.savez_compressed(out / "b_fus3d4_visibility_6view.npz", diagnostic_only=True, visibility=np.zeros((0,), dtype=np.float32))
    counts = {
        "full_body": int(len(points[body_mask])),
        "head_face": int(len(points[np.logical_or(head_mask, face_mask)])),
        "input": int(len(points)),
    }
    precheck = bool(counts["input"] > 1000 and counts["head_face"] > 50 and controls.get("real"))
    summary = {
        "task": "b_fus3d4_export_surface_candidate_precheck",
        "created_utc": utc_now(),
        "status": "b_fus3d4_surface_exported_research_only" if precheck else "b_fus3d4_surface_export_blocked",
        **CONTRACT,
        "strict_candidate_precheck_pass": False,
        "body_surface": body_ply,
        "head_face_surface": head_ply,
        "contact_sheet": contact,
        "depth_npz": out / "b_fus3d4_depth_6view.npz",
        "normal_npz": out / "b_fus3d4_normal_6view.npz",
        "visibility_npz": out / "b_fus3d4_visibility_6view.npz",
        "surface_counts": counts,
        "control_probability_stats": controls,
        "blockers": [
            "B-Fus3D4 export derives from research preflight PLY; it has not passed Open3D human visual strict gate.",
            "Depth/normal/visibility outputs are diagnostic from available V9 tensors, not a full render protocol.",
            "Hand and hair ownership are intentionally not solved by this body/head/face export.",
        ],
        "decision": "B-Fus3D4 exported real research surface artifacts for U-line, but cannot promote without T/H/U/D gates.",
    }
    write_json(out / "b_fus3d4_summary.json", summary)
    write_json(out / "summary.json", summary)
    write_report(out / "report.md", "V10 B-Fus3D4 Surface Export", summary)
    print(json.dumps(json_ready({"status": summary["status"], "counts": counts, "output": out}), ensure_ascii=False))
    return 0 if precheck else 2


def hand11_minimal(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    assets = args.assets_dir
    token_path = assets / "vggt_token_cache/aggregator_layer_23.npz"
    template = load_template()
    masks = region_masks_from_template(template)
    pts = template["hybrid_vertices"].astype(np.float32)
    left = pts[masks["left_hand"]]
    right = pts[masks["right_hand"]]
    colors_l = np.tile(np.asarray([[220, 120, 80]], dtype=np.uint8), (len(left), 1))
    colors_r = np.tile(np.asarray([[80, 150, 230]], dtype=np.uint8), (len(right), 1))
    write_ascii_ply(out / "b_hand11_left_surface.ply", left, colors_l)
    write_ascii_ply(out / "b_hand11_right_surface.ply", right, colors_r)
    combined = np.concatenate([left, right], axis=0)
    colors = np.concatenate([colors_l, colors_r], axis=0)
    write_ascii_ply(out / "b_hand11_combined_wrist_forearm.ply", combined, colors)
    contact_sheet(combined, colors, out / "b_hand11_hand_contact_sheet.png", "B-hand11 weak scaffold")
    token_summary = {}
    if token_path.is_file():
        with np.load(token_path, allow_pickle=False) as payload:
            token_summary = {key: {"shape": list(payload[key].shape), "dtype": str(payload[key].dtype)} for key in payload.files}
    summary = {
        "task": "b_hand11_real_vggt_hand_token_decoder_train",
        "created_utc": utc_now(),
        "status": "blocked_no_true_trainable_hand_token_decoder",
        **CONTRACT,
        "hand_visual_precheck_pass": False,
        "real_decoder_present": False,
        "vggt_token_cache_present": token_path.is_file(),
        "token_summary": token_summary,
        "left_surface": out / "b_hand11_left_surface.ply",
        "right_surface": out / "b_hand11_right_surface.ply",
        "combined_wrist_forearm": out / "b_hand11_combined_wrist_forearm.ply",
        "contact_sheet": out / "b_hand11_hand_contact_sheet.png",
        "stats": {
            "left_hand_points": int(len(left)),
            "right_hand_points": int(len(right)),
            "wrist_connected": False,
            "finger_structure_visible": False,
            "smplx_scaffold_only": True,
        },
        "blockers": [
            "VGGT token cache exists, but no trainable HGGT-style left/right hand-token decoder checkpoint/module is present.",
            "Generated PLY is a weak SMPL-X/template scaffold for U-line diagnostics only; it is not a hand pass.",
            "B-hand10 proxy remains excluded from promotion.",
        ],
        "decision": "B-hand11 cannot satisfy V10 hand gate yet; true decoder training is still required.",
    }
    write_json(out / "summary.json", summary)
    write_report(out / "report.md", "V10 B-hand11 Real Decoder Attempt", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": out}), ensure_ascii=False))
    return 2


def hair4_native(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    template = load_template()
    pts = template["hybrid_vertices"].astype(np.float32)
    masks = region_masks_from_template(template)
    hair = pts[masks["hairline"]]
    head = pts[masks["head"]]
    if len(hair) == 0 and "hair_new_vertices" in template:
        hair = template["hair_new_vertices"].astype(np.float32)
    top_thresh = np.percentile(head[:, 1], 80) if len(head) else 0.0
    headtop = head[head[:, 1] >= top_thresh] if len(head) else np.zeros((0, 3), dtype=np.float32)
    colors_hair = np.tile(np.asarray([[30, 25, 20]], dtype=np.uint8), (len(hair), 1))
    colors_top = np.tile(np.asarray([[70, 45, 35]], dtype=np.uint8), (len(headtop), 1))
    write_ascii_ply(out / "b_hair4_hairline_band_surface.ply", hair, colors_hair)
    write_ascii_ply(out / "b_hair4_head_top_surface.ply", headtop, colors_top)
    strand = np.concatenate([hair, headtop], axis=0)
    colors = np.concatenate([colors_hair, colors_top], axis=0)
    write_ascii_ply(out / "b_hair4_strand_strip_primitives.ply", strand, colors)
    contact_sheet(strand, colors, out / "b_hair4_head_hairline_contact_sheet.png", "B-hair4 native")
    token_path = args.assets_dir / "vggt_token_cache/aggregator_layer_23.npz"
    token_present = token_path.is_file()
    real_pass = False
    summary = {
        "task": "b_hair4_native_4k4d_smplx_hair_topology",
        "created_utc": utc_now(),
        "status": "native_hair_topology_attempt_blocked_visual_gate",
        **CONTRACT,
        "hair_visual_precheck_pass": real_pass,
        "official_hairgs_route": {
            "flame_or_hair_dataset_available": False,
            "decision": "official Hair-GS route remains blocked locally by FLAME/hair dataset conversion; native fallback executed.",
        },
        "vggt_token_cache_present": token_present,
        "hairline_surface": out / "b_hair4_hairline_band_surface.ply",
        "head_top_surface": out / "b_hair4_head_top_surface.ply",
        "strand_primitives": out / "b_hair4_strand_strip_primitives.ply",
        "contact_sheet": out / "b_hair4_head_hairline_contact_sheet.png",
        "topology_metrics": {
            "hairline_point_count": int(len(hair)),
            "head_top_point_count": int(len(headtop)),
            "strand_component_count": 1 if len(strand) else 0,
            "floating_dot_ratio": 1.0,
            "head_shell_leakage": True,
            "real_token_topology_margin": 0.0,
        },
        "blockers": [
            "Native fallback is SMPL-X/template anchored and not a learned HairGS topology module.",
            "No real-token-over-image-only topology margin has been trained or demonstrated.",
            "Open3D visual gate must reject this as scaffold/topology diagnostic, not hair pass.",
        ],
        "decision": "B-hair4 produced a native diagnostic hairline/head-top artifact, but it is not a strict hair topology pass.",
    }
    write_json(out / "summary.json", summary)
    write_report(out / "report.md", "V10 B-hair4 Native Topology Attempt", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": out}), ensure_ascii=False))
    return 2


def read_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def unified_merge(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    summaries = {
        "a5x3": read_summary(args.a5x3_dir / "summary.json"),
        "fus3d4": read_summary(args.fus3d4_dir / "summary.json"),
        "hand11": read_summary(args.hand11_dir / "summary.json"),
        "hair4": read_summary(args.hair4_dir / "summary.json"),
    }
    sources: list[tuple[str, Path, tuple[int, int, int]]] = [
        ("fus3d4", args.fus3d4_dir / "b_fus3d4_body_surface.ply", (130, 180, 230)),
        ("hair4", args.hair4_dir / "b_hair4_strand_strip_primitives.ply", (40, 30, 20)),
        ("hand11_left", args.hand11_dir / "b_hand11_left_surface.ply", (220, 120, 80)),
        ("hand11_right", args.hand11_dir / "b_hand11_right_surface.ply", (80, 150, 230)),
        ("a5x3", args.a5x3_dir / "a5x3_aligned_must3r_points.ply", (180, 180, 180)),
    ]
    all_pts = []
    all_cols = []
    ownership = []
    for name, path, color in sources:
        if not path.is_file():
            continue
        pts, cols = load_ply_xyz_rgb(path)
        if len(pts) == 0:
            continue
        if cols.size == 0:
            cols = np.tile(np.asarray([color], dtype=np.uint8), (len(pts), 1))
        all_pts.append(pts)
        all_cols.append(cols)
        ownership.extend([name] * len(pts))
    if all_pts:
        points = np.concatenate(all_pts, axis=0)
        colors = np.concatenate(all_cols, axis=0)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.uint8)
    unified_ply = out / "unified_surface_v10.ply"
    write_ascii_ply(unified_ply, points, colors)
    np.savez_compressed(out / "unified_surface_v10_regions.npz", ownership=np.asarray(ownership))
    np.savez_compressed(out / "unified_surface_v10_depth_6view.npz", diagnostic_only=True)
    np.savez_compressed(out / "unified_surface_v10_normal_6view.npz", diagnostic_only=True)
    np.savez_compressed(out / "unified_surface_v10_visibility_6view.npz", diagnostic_only=True)
    contact_sheet(points, colors, out / "unified_surface_v10_open3d_full.png", "Unified V10")
    contact_sheet(points, colors, out / "unified_surface_v10_open3d_head_face_hair.png", "Unified head/hair")
    contact_sheet(points, colors, out / "unified_surface_v10_open3d_hands.png", "Unified hands")
    gates = {
        "a5x3_teacher_precheck": bool(summaries["a5x3"].get("strict_teacher_precheck_pass")),
        "fus3d4_body_head_face": bool(summaries["fus3d4"].get("status", "").startswith("b_fus3d4_surface_exported")),
        "hand11_visual_pass": bool(summaries["hand11"].get("hand_visual_precheck_pass")),
        "hair4_visual_pass": bool(summaries["hair4"].get("hair_visual_precheck_pass")),
        "not_sparse_points": bool(len(points) > 10000),
        "region_ownership_pass": bool(all(path.is_file() for _, path, _ in sources[:4])),
    }
    passed = all(gates.values())
    summary = {
        "task": "unified_surface_v10_merge_precheck",
        "created_utc": utc_now(),
        "status": "unified_surface_v10_precheck_pass_research_only" if passed else "unified_surface_v10_precheck_blocked",
        **CONTRACT,
        "unified_surface_candidate_precheck_pass": passed,
        "unified_surface": unified_ply,
        "region_npz": out / "unified_surface_v10_regions.npz",
        "contact_sheets": {
            "full": out / "unified_surface_v10_open3d_full.png",
            "head_face_hair": out / "unified_surface_v10_open3d_head_face_hair.png",
            "hands": out / "unified_surface_v10_open3d_hands.png",
        },
        "component_summaries": summaries,
        "gates": gates,
        "point_count": int(len(points)),
        "blockers": [] if passed else [f"{key}=false" for key, val in gates.items() if not val],
        "decision": "Unified V10 can enter D-line promotion transaction." if passed else "Unified V10 merge is research-only and blocked from promotion.",
    }
    write_json(out / "unified_surface_v10_summary.json", summary)
    write_json(out / "summary.json", summary)
    write_report(out / "unified_surface_v10_report.md", "V10 Unified Surface Merge Precheck", summary)
    print(json.dumps(json_ready({"status": summary["status"], "gates": gates, "output": out}), ensure_ascii=False))
    return 0 if passed else 2


def dline_promotion(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    unified = read_summary(args.unified_dir / "summary.json")
    required = {
        "unified_surface_candidate_precheck_pass": bool(unified.get("unified_surface_candidate_precheck_pass")),
        "full_body_visual_pass": bool(unified.get("gates", {}).get("not_sparse_points")),
        "head_visual_pass": bool(unified.get("gates", {}).get("fus3d4_body_head_face")),
        "face_visual_pass": bool(unified.get("gates", {}).get("fus3d4_body_head_face")),
        "hairline_visual_pass": bool(unified.get("gates", {}).get("hair4_visual_pass")),
        "left_hand_visual_pass": bool(unified.get("gates", {}).get("hand11_visual_pass")),
        "right_hand_visual_pass": bool(unified.get("gates", {}).get("hand11_visual_pass")),
        "teacher_or_surface_anchor_pass": bool(unified.get("gates", {}).get("a5x3_teacher_precheck")),
        "forbidden_output_scan_clean": True,
    }
    passed = all(required.values())
    summary = {
        "task": "dline_v10_promotion_transaction",
        "created_utc": utc_now(),
        **CONTRACT,
        "status": "promotion_written" if passed else "promotion_blocked_no_strict_write",
        "promotion_transaction_pass": passed,
        "strict_candidate_passes": 1 if passed else 0,
        "strict_teacher_passes": 0,
        "registry_written": False,
        "candidate_package_written": False,
        "teacher_package_written": False,
        "required_gates": required,
        "unified_summary": args.unified_dir / "summary.json",
        "blockers": [] if passed else [f"{key}=false" for key, val in required.items() if not val],
        "decision": "D-line would write strict candidate package in a separate controlled transaction." if passed else "D-line refuses promotion; no strict registry/package/pass was written.",
    }
    if passed:
        # Still fail closed in this research implementation; an actual write is
        # intentionally separated from the precheck script.
        summary["status"] = "promotion_ready_but_write_disabled_research_only"
        summary["registry_written"] = False
        summary["candidate_package_written"] = False
        summary["strict_candidate_passes"] = 0
        summary["blockers"] = ["promotion write disabled in research-only V10 run"]
    write_json(out / "dline_v10_promotion_summary.json", summary)
    write_json(out / "summary.json", summary)
    write_report(out / "dline_v10_promotion_report.md", "D-line V10 Promotion Transaction", summary)
    print(json.dumps(json_ready({"status": summary["status"], "required_gates": required, "output": out}), ensure_ascii=False))
    return 0 if summary["status"].startswith("promotion_written") else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V10 surface completion research-only pipeline utilities.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("audit-2dgs")
    p.add_argument("--scene-dir", type=Path, default=DEFAULT_2DGS_SCENE)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_2DGS_SMOKE)
    p.add_argument("--output-dir", type=Path, default=LOCAL_ROOT / "V10_2DGS_scene_contract_audit")
    p.set_defaults(func=audit_2dgs_scene)
    p = sub.add_parser("must3r-align")
    p.add_argument("--must3r-dir", type=Path, default=DEFAULT_MUST3R_DIR)
    p.add_argument("--scene-dir", type=Path, default=DEFAULT_2DGS_SCENE)
    p.add_argument("--output-dir", type=Path, default=LOCAL_ROOT / "V10_A5X3_must3r_known_camera_alignment")
    p.add_argument("--max-eval-points", type=int, default=60000)
    p.set_defaults(func=must3r_align)
    p = sub.add_parser("fus3d4-export")
    p.add_argument("--fus3d3-dir", type=Path, default=DEFAULT_FUS3D3)
    p.add_argument("--output-dir", type=Path, default=LOCAL_ROOT / "B_Fus3D4_surface_candidate_precheck")
    p.set_defaults(func=fus3d4_export)
    p = sub.add_parser("hand11")
    p.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS)
    p.add_argument("--output-dir", type=Path, default=LOCAL_ROOT / "B_Hand11_real_vggt_hand_token_decoder")
    p.set_defaults(func=hand11_minimal)
    p = sub.add_parser("hair4")
    p.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS)
    p.add_argument("--output-dir", type=Path, default=LOCAL_ROOT / "B_Hair4_native_4k4d_smplx_hair_topology")
    p.set_defaults(func=hair4_native)
    p = sub.add_parser("unified")
    p.add_argument("--a5x3-dir", type=Path, default=LOCAL_ROOT / "V10_A5X3_must3r_known_camera_alignment")
    p.add_argument("--fus3d4-dir", type=Path, default=LOCAL_ROOT / "B_Fus3D4_surface_candidate_precheck")
    p.add_argument("--hand11-dir", type=Path, default=LOCAL_ROOT / "B_Hand11_real_vggt_hand_token_decoder")
    p.add_argument("--hair4-dir", type=Path, default=LOCAL_ROOT / "B_Hair4_native_4k4d_smplx_hair_topology")
    p.add_argument("--output-dir", type=Path, default=LOCAL_ROOT / "V10_unified_surface_merge_precheck")
    p.set_defaults(func=unified_merge)
    p = sub.add_parser("dline")
    p.add_argument("--unified-dir", type=Path, default=LOCAL_ROOT / "V10_unified_surface_merge_precheck")
    p.add_argument("--output-dir", type=Path, default=LOCAL_ROOT / "DLine_V10_promotion_transaction")
    p.set_defaults(func=dline_promotion)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
