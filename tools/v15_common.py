from __future__ import annotations

import json
import math
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
LOCAL_ROOT = REPO_ROOT / "output/surface_research_preflight_local"
CLOUD_ROOT = REPO_ROOT / "output/surface_research_cloud_preflight"

DEFAULT_2DGS_SCENE = CLOUD_ROOT / "Cloud_B_V9/a5x2_2dgs_colmap_scene/2dgs_colmap_scene"
DEFAULT_2DGS_30K_PLY = (
    CLOUD_ROOT / "Cloud_G_V10/a5x3_2dgs_colmap_scene_30k/model_smoke/point_cloud/iteration_30000/point_cloud.ply"
)
DEFAULT_G3_DIR = LOCAL_ROOT / "V11_G3_2DGS_surface_anchor"
DEFAULT_TMF_SCENE = REPO_ROOT / "output/4k4d_scenes/0012_11_frame0000_12views_tmf"
DEFAULT_SAPIENS_NORMAL = CLOUD_ROOT / "V13_Sapiens_Normal/sapiens_normals.npz"
DEFAULT_SAPIENS_DEPTH = CLOUD_ROOT / "V13_Sapiens_Depth/sapiens_depths.npz"


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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
    ]
    metrics = summary.get("metrics")
    if isinstance(metrics, dict) and metrics:
        lines.extend(["## Metrics", ""])
        for key, value in metrics.items():
            lines.append(f"- {key}: `{json_ready(value)}`")
        lines.append("")
    blockers = summary.get("blockers") or []
    lines.extend(["## Blockers", ""])
    lines.extend([f"- {item}" for item in blockers] if blockers else ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_v15_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    lower = resolved.as_posix().lower()
    if "surface_research" not in lower or "v15" not in lower:
        raise ValueError(f"Refusing non-V15 research output path: {resolved}")
    for token in ("predictions", "teacher_export", "candidate_export", "strict_gate_registry", "strict_pass"):
        if token in lower:
            raise ValueError(f"Refusing forbidden output path token {token!r}: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


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


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(vectors, dtype=np.float32)
    length = np.linalg.norm(arr, axis=-1)
    valid = np.isfinite(arr).all(axis=-1) & (length > eps)
    out = np.zeros_like(arr, dtype=np.float32)
    out[valid] = arr[valid] / length[valid, None]
    return out, valid


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


def ply_vertex_properties(lines: list[str]) -> list[tuple[str, str]]:
    props: list[tuple[str, str]] = []
    in_vertex = False
    for line in lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            continue
        if in_vertex and len(parts) == 3 and parts[0] == "property":
            props.append((parts[1], parts[2]))
    return props


def load_binary_ply_vertices(path: Path) -> tuple[np.ndarray, list[str]]:
    lines, count, offset, fmt = read_ply_header(path)
    if fmt != "binary_little_endian":
        raise ValueError(f"Expected binary_little_endian PLY for raw Gaussian attributes: {path}")
    props = ply_vertex_properties(lines)
    type_map = {"float": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4"}
    dtype = np.dtype([(name, type_map[typ]) for typ, name in props])
    with path.open("rb") as handle:
        handle.seek(offset)
        arr = np.fromfile(handle, dtype=dtype, count=count)
    return arr, [name for _, name in props]


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
    if not cameras_txt.is_file() or not images_txt.is_file():
        return []
    cams: dict[int, dict[str, Any]] = {}
    for line in cameras_txt.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        cam_id = int(parts[0])
        model = parts[1]
        width, height = int(parts[2]), int(parts[3])
        params = [float(x) for x in parts[4:]]
        if model not in {"PINHOLE", "OPENCV", "SIMPLE_PINHOLE"}:
            continue
        if model == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1:3]
        else:
            fx, fy, cx, cy = params[:4]
        cams[cam_id] = {
            "camera_id": cam_id,
            "model": model,
            "width": width,
            "height": height,
            "intrinsic": np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }
    out: list[dict[str, Any]] = []
    for line in images_txt.read_text(encoding="utf-8").splitlines():
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
        if camera_id not in cams:
            continue
        r = qvec_to_rotmat(q)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = r
        w2c[:3, 3] = t
        cam = dict(cams[camera_id])
        cam.update(
            {
                "image_id": image_id,
                "name": name,
                "stem": Path(name).stem,
                "world_to_cam": w2c,
                "rotation_w2c": r,
                "translation_w2c": t,
                "camera_center_world": (-r.T @ t).astype(np.float32),
            }
        )
        out.append(cam)
    out.sort(key=lambda item: int(item["image_id"]))
    return out


def load_tmf_manifest(scene_dir: Path) -> dict[str, Any]:
    return read_json(scene_dir / "scene_manifest.json")


def tmf_view_rows(scene_dir: Path) -> list[dict[str, Any]]:
    manifest = load_tmf_manifest(scene_dir)
    rows: list[dict[str, Any]] = []
    for idx, view in enumerate(manifest.get("exported_views", []) or []):
        image_path = Path(view.get("image_path", ""))
        mask_path = Path(view.get("mask_path", ""))
        image_size = view.get("image_size") or [None, None]
        rows.append(
            {
                "index": idx,
                "camera_id": str(view.get("camera_id")),
                "role": view.get("role"),
                "image_name": image_path.name,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "image_exists": image_path.is_file(),
                "mask_exists": mask_path.is_file(),
                "image_size": image_size,
                "mask_coverage": view.get("mask_coverage"),
            }
        )
    return rows


def parse_camera_id_from_name(name: str) -> str:
    stem = Path(name).stem
    if "cam" in stem:
        suffix = stem.rsplit("cam", 1)[-1]
        digits = "".join(ch for ch in suffix if ch.isdigit())
        if digits:
            return digits.zfill(2)
    if "view" in stem:
        suffix = stem.rsplit("view", 1)[-1]
        digits = "".join(ch for ch in suffix if ch.isdigit())
        if digits:
            return digits.zfill(2)
    return ""


def resize_nearest(arr: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = size_hw
    src = np.asarray(arr)
    if src.shape[:2] == (target_h, target_w):
        return src.copy()
    pil_mode = None
    if src.dtype == np.bool_:
        img = Image.fromarray(src.astype(np.uint8) * 255, mode="L")
        out = np.asarray(img.resize((target_w, target_h), Image.Resampling.NEAREST)) > 127
        return out
    if src.ndim == 2:
        img = Image.fromarray(src.astype(np.float32), mode="F")
        return np.asarray(img.resize((target_w, target_h), Image.Resampling.NEAREST)).astype(src.dtype)
    chans = []
    for channel in range(src.shape[2]):
        img = Image.fromarray(src[..., channel].astype(np.float32), mode="F")
        chans.append(np.asarray(img.resize((target_w, target_h), Image.Resampling.NEAREST)))
    out = np.stack(chans, axis=-1)
    return out.astype(src.dtype)


def resize_bilinear_float(arr: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = size_hw
    src = np.asarray(arr)
    if src.shape[:2] == (target_h, target_w):
        return src.copy()
    if src.ndim == 2:
        img = Image.fromarray(src.astype(np.float32), mode="F")
        return np.asarray(img.resize((target_w, target_h), Image.Resampling.BILINEAR)).astype(np.float32)
    chans = []
    for channel in range(src.shape[2]):
        img = Image.fromarray(src[..., channel].astype(np.float32), mode="F")
        chans.append(np.asarray(img.resize((target_w, target_h), Image.Resampling.BILINEAR)))
    return np.stack(chans, axis=-1).astype(np.float32)


def normal_angle_metrics(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(valid, dtype=bool)
    if not np.any(mask):
        return {"valid_pixels": 0}
    aa, va = normalize_vectors(np.asarray(a, dtype=np.float32))
    bb, vb = normalize_vectors(np.asarray(b, dtype=np.float32))
    mask = mask & va & vb
    if not np.any(mask):
        return {"valid_pixels": 0}
    dot = np.clip(np.sum(aa[mask] * bb[mask], axis=-1), -1.0, 1.0)
    abs_dot = np.abs(dot)
    signed_angle = np.degrees(np.arccos(dot))
    abs_angle = np.degrees(np.arccos(abs_dot))
    return {
        "valid_pixels": int(dot.size),
        "signed_cos_mean": float(dot.mean()),
        "signed_cos_median": float(np.median(dot)),
        "negative_dot_frac": float(np.mean(dot < 0.0)),
        "signed_angle_mean_deg": float(signed_angle.mean()),
        "signed_angle_median_deg": float(np.median(signed_angle)),
        "abs_angle_mean_deg": float(abs_angle.mean()),
        "abs_angle_median_deg": float(np.median(abs_angle)),
        "abs_angle_p90_deg": float(np.percentile(abs_angle, 90)),
    }


def project_points(points: np.ndarray, camera: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    w2c = np.asarray(camera["world_to_cam"], dtype=np.float32)
    k = np.asarray(camera["intrinsic"], dtype=np.float32)
    cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    uvw = cam @ k.T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-8)
    return uv, z, cam


def sigmoid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -60.0, 60.0)))


def quaternions_to_mats(q: np.ndarray) -> np.ndarray:
    quat = np.asarray(q, dtype=np.float32)
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    quat = quat / np.maximum(norms, 1e-8)
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    mats = np.empty((quat.shape[0], 3, 3), dtype=np.float32)
    mats[:, 0, 0] = 1 - 2 * qy * qy - 2 * qz * qz
    mats[:, 0, 1] = 2 * qx * qy - 2 * qz * qw
    mats[:, 0, 2] = 2 * qx * qz + 2 * qy * qw
    mats[:, 1, 0] = 2 * qx * qy + 2 * qz * qw
    mats[:, 1, 1] = 1 - 2 * qx * qx - 2 * qz * qz
    mats[:, 1, 2] = 2 * qy * qz - 2 * qx * qw
    mats[:, 2, 0] = 2 * qx * qz - 2 * qy * qw
    mats[:, 2, 1] = 2 * qy * qz + 2 * qx * qw
    mats[:, 2, 2] = 1 - 2 * qx * qx - 2 * qy * qy
    return mats


def derive_2dgs_world_normals(vertices: np.ndarray, props: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    has_nxyz = all(name in props for name in ("nx", "ny", "nz"))
    nxyz = np.zeros_like(xyz)
    if has_nxyz:
        nxyz = np.stack([vertices["nx"], vertices["ny"], vertices["nz"]], axis=1).astype(np.float32)
    nxyz_norm = np.linalg.norm(nxyz, axis=1)
    qnames = ("rot_0", "rot_1", "rot_2", "rot_3")
    has_quat = all(name in props for name in qnames)
    normals = nxyz.copy()
    source = "nx_ny_nz"
    if int((nxyz_norm > 1e-8).sum()) == 0 and has_quat:
        quat = np.stack([vertices[name] for name in qnames], axis=1).astype(np.float32)
        mats = quaternions_to_mats(quat)
        normals = mats[:, :, 2]
        source = "rot_quaternion_local_z"
    normals, valid = normalize_vectors(normals)
    centers = xyz
    if valid.any():
        view_vec = centers - np.median(centers[valid], axis=0, keepdims=True)
        flip = np.sum(normals * view_vec, axis=1) < 0.0
        normals[flip] *= -1.0
    meta = {
        "normal_source": source,
        "has_nx_ny_nz_fields": has_nxyz,
        "nx_ny_nz_nonzero_count": int((nxyz_norm > 1e-8).sum()),
        "has_rot_quaternion_fields": has_quat,
        "valid_normal_count": int(valid.sum()),
        "normal_length": scalar_stats(np.linalg.norm(normals, axis=1)),
    }
    return normals.astype(np.float32), meta


def camera_id_overlap(g3_names: list[str], sapiens_names: list[str]) -> list[dict[str, Any]]:
    sapiens_by_cam: dict[str, list[tuple[int, str]]] = {}
    for idx, name in enumerate(sapiens_names):
        sapiens_by_cam.setdefault(parse_camera_id_from_name(name), []).append((idx, name))
    rows: list[dict[str, Any]] = []
    for g_idx, name in enumerate(g3_names):
        cam_id = parse_camera_id_from_name(name)
        matches = sapiens_by_cam.get(cam_id, [])
        rows.append(
            {
                "g3_index": int(g_idx),
                "g3_name": name,
                "camera_id": cam_id,
                "sapiens_matches": [{"index": int(i), "name": n} for i, n in matches],
                "match_count": len(matches),
            }
        )
    return rows


def fit_affine_depth(relative: np.ndarray, metric: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(valid, dtype=bool) & np.isfinite(relative) & np.isfinite(metric)
    if int(mask.sum()) < 10:
        return {"fit_valid": False, "sample_count": int(mask.sum())}
    x = np.asarray(relative, dtype=np.float64)[mask].reshape(-1)
    y = np.asarray(metric, dtype=np.float64)[mask].reshape(-1)
    a = np.vstack([x, np.ones_like(x)]).T
    scale, bias = np.linalg.lstsq(a, y, rcond=None)[0]
    pred = scale * x + bias
    residual = pred - y
    denom = np.maximum(np.abs(y), 1e-6)
    corr = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 and np.std(x) > 1e-8 and np.std(y) > 1e-8 else 0.0
    return {
        "fit_valid": True,
        "sample_count": int(x.size),
        "scale": float(scale),
        "bias": float(bias),
        "corr": corr,
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "median_abs_error": float(np.median(np.abs(residual))),
        "median_relative_abs_error": float(np.median(np.abs(residual) / denom)),
        "metric_depth": scalar_stats(y),
        "relative_depth": scalar_stats(x),
    }
