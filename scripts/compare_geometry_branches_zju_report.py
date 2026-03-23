import argparse
import contextlib
import csv
import json
import math
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


DEFAULT_MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
CONF_EPS = 1e-6
DEPTH_EPS = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare point-map vs depth-unprojection branches on a ZJU case defined by an old report.json."
    )
    parser.add_argument("--report_json", type=str, required=True, help="Old report.json that contains seq/frame/source/target metadata.")
    parser.add_argument(
        "--local_zju_root",
        type=str,
        default="",
        help="Local ZJU_MoCap root that corresponds to /mnt/data/zju_mocap. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory. Defaults to output/geometry_baseline_zju/<case>_<timestamp>.",
    )
    parser.add_argument("--checkpoint", type=str, default="", help="Optional local checkpoint path.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--preprocess_mode", type=str, default="crop", choices=["crop", "pad"])
    parser.add_argument("--conf_percentile", type=float, default=25.0)
    parser.add_argument("--export_max_points", type=int, default=250000)
    parser.add_argument("--render_max_points", type=int, default=750000)
    parser.add_argument("--render_size", nargs=2, type=int, default=[518, 518])
    parser.add_argument("--z_tolerance", type=float, default=0.02)
    parser.add_argument("--min_conf", type=float, default=1e-6)
    parser.add_argument(
        "--primary_branch",
        type=str,
        default="depth_unproject",
        choices=["depth_unproject", "point_map", "auto"],
        help="Branch to materialize as the primary geometry output. 'auto' uses the branch decision winner and falls back to depth_unproject on ties.",
    )
    parser.add_argument("--skip_save_predictions", action="store_true")
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def detect_local_zju_root():
    candidates = [
        Path("G:/\u6570\u636e\u96c6/datasets/ZJU_MoCap/data/zju_mocap"),
        Path("G:/\u9879\u76ee\u5907\u4efd/Redo_viewpoints_at_60\u00b0_intervals_add_random_perturbations_vggt/datasets/ZJU_MoCap/data/zju_mocap"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Could not auto-detect a local ZJU root. Please pass --local_zju_root explicitly.")


def resolve_output_dir(report_json, output_dir, meta):
    if output_dir:
        return ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile = infer_view_profile(Path(report_json), meta)
    case_name = "{seq}_frame_{frame:06d}_{tgt}_{profile}".format(
        seq=meta["seq_name"],
        frame=int(meta["frame_id"]),
        tgt=meta["tgt_camera"],
        profile=profile,
    )
    return ensure_dir(Path("output") / "geometry_baseline_zju" / f"{case_name}_{timestamp}")


def infer_view_profile(report_json, meta):
    profile = str(meta.get("view_profile", "")).strip()
    if profile:
        return profile
    parts = report_json.parts
    if "infer_out" in parts:
        idx = parts.index("infer_out")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown_profile"


def resolve_device(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device_name == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    ensure_supported_cuda_runtime()
    return torch.device("cuda")


def ensure_supported_cuda_runtime():
    capability = torch.cuda.get_device_capability(0)
    sm_tag = f"sm_{capability[0]}{capability[1]}"
    arch_list = set(torch.cuda.get_arch_list())
    if sm_tag in arch_list:
        return
    device_name = torch.cuda.get_device_name(0)
    supported = ", ".join(sorted(arch_list))
    raise RuntimeError(
        "The current PyTorch build cannot run on the detected CUDA GPU.\n"
        f"Detected device: {device_name} ({sm_tag})\n"
        f"PyTorch CUDA arch list: {supported}\n"
        "Install a newer PyTorch build that supports this GPU, or rerun with --device cpu."
    )


def resolve_dtype(dtype_name, device):
    if device.type != "cuda":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    capability_major = torch.cuda.get_device_capability(device=device)[0]
    return torch.bfloat16 if capability_major >= 8 else torch.float16


def load_model(checkpoint_path, device):
    model = VGGT()
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        if isinstance(state_dict, dict):
            keys = [str(key) for key in state_dict.keys()]
            if keys and all(key.startswith("module.") for key in keys):
                state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}
    else:
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


def branch_threshold(confidence_map, finite_mask, percentile):
    conf = np.asarray(confidence_map, dtype=np.float32).reshape(-1)
    valid_conf = conf[np.asarray(finite_mask).reshape(-1) & np.isfinite(conf) & (conf > CONF_EPS)]
    if valid_conf.size == 0:
        return CONF_EPS
    if percentile <= 0:
        return CONF_EPS
    return max(float(np.percentile(valid_conf, percentile)), CONF_EPS)


def summarize_branch(points, conf):
    finite = np.isfinite(points).all(axis=-1)
    valid = finite & np.isfinite(conf) & (conf > CONF_EPS)
    valid_ratio = float(valid.mean())
    conf_values = conf[valid]
    summary = {
        "valid_ratio": valid_ratio,
        "valid_points": int(valid.sum()),
        "total_points": int(valid.size),
        "confidence_percentiles": {},
    }
    if conf_values.size == 0:
        return summary
    conf_percentiles = np.percentile(conf_values, [5, 25, 50, 75, 95]).tolist()
    summary["confidence_percentiles"] = {
        "p05": float(conf_percentiles[0]),
        "p25": float(conf_percentiles[1]),
        "p50": float(conf_percentiles[2]),
        "p75": float(conf_percentiles[3]),
        "p95": float(conf_percentiles[4]),
    }
    return summary


def summarize_branch_delta(point_map, depth_points, point_conf, depth_conf):
    finite_mask = np.isfinite(point_map).all(axis=-1) & np.isfinite(depth_points).all(axis=-1)
    valid_mask = finite_mask & (point_conf > CONF_EPS) & (depth_conf > CONF_EPS)
    if not valid_mask.any():
        return {"shared_valid_ratio": 0.0, "distance_percentiles": {}}
    distances = np.linalg.norm(point_map[valid_mask] - depth_points[valid_mask], axis=-1)
    percentiles = np.percentile(distances, [50, 75, 90, 95, 99]).tolist()
    return {
        "shared_valid_ratio": float(valid_mask.mean()),
        "distance_percentiles": {
            "p50": float(percentiles[0]),
            "p75": float(percentiles[1]),
            "p90": float(percentiles[2]),
            "p95": float(percentiles[3]),
            "p99": float(percentiles[4]),
        },
    }


def build_export_mask(points, conf, percentile, max_points):
    finite = np.isfinite(points).all(axis=-1)
    threshold = branch_threshold(conf, finite, percentile)
    flat_finite = finite.reshape(-1)
    flat_conf = conf.reshape(-1)
    mask = flat_finite & np.isfinite(flat_conf) & (flat_conf >= threshold)
    selected_indices = np.flatnonzero(mask)
    if selected_indices.size > max_points:
        selected_conf = flat_conf[selected_indices]
        top_local = np.argpartition(selected_conf, -max_points)[-max_points:]
        selected_indices = selected_indices[top_local]
        selected_indices.sort()
    final_mask = np.zeros(flat_conf.shape[0], dtype=bool)
    final_mask[selected_indices] = True
    return final_mask.reshape(conf.shape), threshold


def write_binary_ply(path, points, colors):
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    vertex_data = np.empty(
        points.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex_data["x"] = points[:, 0]
    vertex_data["y"] = points[:, 1]
    vertex_data["z"] = points[:, 2]
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        vertex_data.tofile(handle)


def export_branch_ply(output_path, points, conf, colors, percentile, max_points):
    export_mask, threshold = build_export_mask(points, conf, percentile, max_points)
    flat_points = points.reshape(-1, 3)[export_mask.reshape(-1)]
    flat_colors = colors.reshape(-1, 3)[export_mask.reshape(-1)]
    write_binary_ply(output_path, flat_points, flat_colors)
    return {
        "path": str(output_path),
        "exported_points": int(flat_points.shape[0]),
        "confidence_threshold": float(threshold),
    }


def camera_centers(extrinsic_3x4):
    rotation = extrinsic_3x4[:, :3, :3]
    translation = extrinsic_3x4[:, :3, 3]
    return -(np.transpose(rotation, (0, 2, 1)) @ translation[..., None])[..., 0]


def rmse(a, b):
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))


def umeyama_similarity(src_xyz, dst_xyz):
    src = np.asarray(src_xyz, dtype=np.float64)
    dst = np.asarray(dst_xyz, dtype=np.float64)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src0 = src - mu_src
    dst0 = dst - mu_dst
    cov = (dst0.T @ src0) / float(src.shape[0])
    u, s, vt = np.linalg.svd(cov)
    d = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        d[-1, -1] = -1.0
    rotation = u @ d @ vt
    var_src = float(np.sum(src0 * src0) / float(src.shape[0]))
    scale = float(np.trace(np.diag(s) @ d) / var_src)
    translation = mu_dst - scale * (rotation @ mu_src)
    return scale, rotation, translation


def apply_sim3_points(points_xyz, scale, rotation, translation):
    points = np.asarray(points_xyz, dtype=np.float64)
    return scale * (points @ rotation.T) + translation


def read_opencv_matrix(fs, key):
    node = fs.getNode(key)
    if node.empty():
        raise KeyError(f"missing node in yaml: {key}")
    matrix = node.mat()
    if matrix is None:
        raise KeyError(f"node has no matrix data: {key}")
    return np.asarray(matrix, dtype=np.float64)


def load_zju_cameras(seq_dir, camera_names):
    intri_path = seq_dir / "intri.yml"
    extri_path = seq_dir / "extri.yml"
    temp_paths = []
    use_temp_copy = (not str(intri_path).isascii()) or (not str(extri_path).isascii())
    if use_temp_copy:
        temp_root = Path(tempfile.gettempdir()) / "vggt_zju_yaml"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_intri = temp_root / "intri.yml"
        temp_extri = temp_root / "extri.yml"
        shutil.copyfile(intri_path, temp_intri)
        shutil.copyfile(extri_path, temp_extri)
        temp_paths = [temp_intri, temp_extri]
        intri_fs = cv2.FileStorage(str(temp_intri), cv2.FILE_STORAGE_READ)
        extri_fs = cv2.FileStorage(str(temp_extri), cv2.FILE_STORAGE_READ)
    else:
        intri_fs = cv2.FileStorage(str(intri_path), cv2.FILE_STORAGE_READ)
        extri_fs = cv2.FileStorage(str(extri_path), cv2.FILE_STORAGE_READ)
        if not intri_fs.isOpened() or not extri_fs.isOpened():
            intri_fs.release()
            extri_fs.release()
            temp_root = Path(tempfile.gettempdir()) / "vggt_zju_yaml"
            temp_root.mkdir(parents=True, exist_ok=True)
            temp_intri = temp_root / "intri.yml"
            temp_extri = temp_root / "extri.yml"
            shutil.copyfile(intri_path, temp_intri)
            shutil.copyfile(extri_path, temp_extri)
            temp_paths = [temp_intri, temp_extri]
            intri_fs = cv2.FileStorage(str(temp_intri), cv2.FILE_STORAGE_READ)
            extri_fs = cv2.FileStorage(str(temp_extri), cv2.FILE_STORAGE_READ)
    if not intri_fs.isOpened() or not extri_fs.isOpened():
        raise RuntimeError("failed to open zju camera yaml files")
    cameras = {}
    try:
        for cam in camera_names:
            intrinsic = read_opencv_matrix(intri_fs, f"K_{cam}")
            rotation = read_opencv_matrix(extri_fs, f"Rot_{cam}")
            translation = read_opencv_matrix(extri_fs, f"T_{cam}").reshape(3)
            extrinsic = np.concatenate([rotation, translation[:, None]], axis=1)
            cameras[cam] = {"intrinsic": intrinsic, "extrinsic": extrinsic}
    finally:
        intri_fs.release()
        extri_fs.release()
        for temp_path in temp_paths:
            try:
                temp_path.unlink()
            except OSError:
                pass
    return cameras


def resolve_frame_image_path(seq_dir, camera_name, frame_id):
    stem = f"{int(frame_id):06d}"
    for ext in (".jpg", ".png", ".jpeg"):
        path = seq_dir / camera_name / f"{stem}{ext}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"frame image not found for {camera_name} frame={frame_id}")


def scale_intrinsic(intrinsic_3x3, src_hw, dst_hw):
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    out = np.asarray(intrinsic_3x3, dtype=np.float64).copy()
    out[0, 0] *= sx
    out[1, 1] *= sy
    out[0, 2] *= sx
    out[1, 2] *= sy
    return out


def render_forward_splat(
    *,
    world_points_s_hw3,
    world_conf_s_hw,
    src_rgb_s_hw3,
    tgt_extrinsic_3x4,
    tgt_intrinsic_3x3,
    out_hw,
    z_eps,
    min_conf,
    z_tolerance,
    max_points,
):
    height, width = int(out_hw[0]), int(out_hw[1])
    points = np.asarray(world_points_s_hw3, dtype=np.float64).reshape(-1, 3)
    conf = np.asarray(world_conf_s_hw, dtype=np.float64).reshape(-1)
    rgb = np.asarray(src_rgb_s_hw3, dtype=np.float64).reshape(-1, 3)

    finite = np.isfinite(points).all(axis=-1) & np.isfinite(conf) & (conf >= float(min_conf))
    candidate_indices = np.flatnonzero(finite)
    if candidate_indices.size == 0:
        return {
            "image": np.zeros((height, width, 3), dtype=np.float32),
            "weight": np.zeros((height, width), dtype=np.float32),
            "stats": {
                "coverage_ratio": 0.0,
                "valid_contrib": 0,
                "mean_conf": 0.0,
                "z_eps": float(z_eps),
                "min_conf": float(min_conf),
                "z_tolerance": float(z_tolerance),
                "input_points": 0,
                "rendered_points": 0,
            },
        }

    if candidate_indices.size > max_points:
        candidate_conf = conf[candidate_indices]
        top_local = np.argpartition(candidate_conf, -max_points)[-max_points:]
        candidate_indices = candidate_indices[top_local]

    points = points[candidate_indices]
    conf = conf[candidate_indices]
    rgb = rgb[candidate_indices]

    rotation = np.asarray(tgt_extrinsic_3x4[:3, :3], dtype=np.float64)
    translation = np.asarray(tgt_extrinsic_3x4[:3, 3], dtype=np.float64)
    cam = points @ rotation.T + translation[None, :]
    z = cam[:, 2]
    x = cam[:, 0] / np.maximum(z, 1e-12)
    y = cam[:, 1] / np.maximum(z, 1e-12)
    fx = float(tgt_intrinsic_3x3[0, 0])
    fy = float(tgt_intrinsic_3x3[1, 1])
    cx = float(tgt_intrinsic_3x3[0, 2])
    cy = float(tgt_intrinsic_3x3[1, 2])
    u = fx * x + cx
    v = fy * y + cy
    xi = np.rint(u).astype(np.int64)
    yi = np.rint(v).astype(np.int64)

    valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(z)
    valid &= z > float(z_eps)
    valid &= xi >= 0
    valid &= yi >= 0
    valid &= xi < width
    valid &= yi < height
    if not np.any(valid):
        return {
            "image": np.zeros((height, width, 3), dtype=np.float32),
            "weight": np.zeros((height, width), dtype=np.float32),
            "stats": {
                "coverage_ratio": 0.0,
                "valid_contrib": 0,
                "mean_conf": 0.0,
                "z_eps": float(z_eps),
                "min_conf": float(min_conf),
                "z_tolerance": float(z_tolerance),
                "input_points": int(candidate_indices.size),
                "rendered_points": 0,
            },
        }

    pix = yi[valid] * width + xi[valid]
    z_valid = z[valid]
    conf_valid = conf[valid]
    rgb_valid = rgb[valid]

    order = np.lexsort((z_valid, pix))
    pix_sorted = pix[order]
    z_sorted = z_valid[order]
    conf_sorted = conf_valid[order]
    rgb_sorted = rgb_valid[order]

    _, start_idx = np.unique(pix_sorted, return_index=True)
    counts = np.diff(np.concatenate([start_idx, np.array([pix_sorted.size])]))
    z_min = z_sorted[start_idx]
    z_min_rep = np.repeat(z_min, counts)
    keep = z_sorted <= (z_min_rep + float(z_tolerance))

    pix_kept = pix_sorted[keep]
    conf_kept = conf_sorted[keep]
    rgb_kept = rgb_sorted[keep]

    pixel_count = int(height * width)
    sum_w = np.bincount(pix_kept, weights=conf_kept, minlength=pixel_count).astype(np.float64)
    sum_r = np.bincount(pix_kept, weights=conf_kept * rgb_kept[:, 0], minlength=pixel_count).astype(np.float64)
    sum_g = np.bincount(pix_kept, weights=conf_kept * rgb_kept[:, 1], minlength=pixel_count).astype(np.float64)
    sum_b = np.bincount(pix_kept, weights=conf_kept * rgb_kept[:, 2], minlength=pixel_count).astype(np.float64)
    hit = sum_w > 0.0

    pred = np.zeros((pixel_count, 3), dtype=np.float64)
    pred[hit, 0] = sum_r[hit] / np.maximum(sum_w[hit], 1e-12)
    pred[hit, 1] = sum_g[hit] / np.maximum(sum_w[hit], 1e-12)
    pred[hit, 2] = sum_b[hit] / np.maximum(sum_w[hit], 1e-12)
    pred = np.clip(pred.reshape(height, width, 3), 0.0, 1.0).astype(np.float32)

    weight_map = sum_w.reshape(height, width)
    if np.any(hit):
        p99 = float(np.percentile(weight_map[hit.reshape(height, width)], 99.0))
        denom = max(p99, 1e-8)
        weight01 = np.clip(weight_map / denom, 0.0, 1.0).astype(np.float32)
    else:
        weight01 = np.zeros((height, width), dtype=np.float32)

    return {
        "image": pred,
        "weight": weight01,
        "stats": {
            "coverage_ratio": float(hit.mean()),
            "valid_contrib": int(conf_kept.size),
            "mean_conf": float(conf_kept.mean()) if conf_kept.size > 0 else 0.0,
            "z_eps": float(z_eps),
            "min_conf": float(min_conf),
            "z_tolerance": float(z_tolerance),
            "input_points": int(candidate_indices.size),
            "rendered_points": int(hit.sum()),
        },
    }


def gaussian_kernel(channels, device, dtype, ksize=11, sigma=1.5):
    coords = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    g2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    return g2d.repeat(channels, 1, 1, 1)


def metrics(pred01, tgt01):
    pred = torch.from_numpy(pred01.transpose(2, 0, 1)).unsqueeze(0).float()
    tgt = torch.from_numpy(tgt01.transpose(2, 0, 1)).unsqueeze(0).float()
    mae = float(F.l1_loss(pred, tgt, reduction="mean").item())
    mse = F.mse_loss(pred, tgt, reduction="mean").clamp_min(1e-8)
    psnr = float((-10.0 * torch.log10(mse)).item())
    c1 = 0.01**2
    c2 = 0.03**2
    kernel = gaussian_kernel(3, pred.device, pred.dtype)

    def filt(x):
        return F.conv2d(x, kernel, padding=5, groups=3)

    mu_x = filt(pred)
    mu_y = filt(tgt)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = filt(pred * pred) - mu_x2
    sigma_y2 = filt(tgt * tgt) - mu_y2
    sigma_xy = filt(pred * tgt) - mu_xy
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-8
    )
    return {"mae": mae, "psnr": psnr, "ssim": float(ssim_map.mean().item())}


def to_uint8(image01):
    return np.clip(image01 * 255.0, 0.0, 255.0).round().astype(np.uint8)


def save_rgb_png(path, image01):
    Image.fromarray(to_uint8(image01)).save(path)


def save_gray_png(path, image01):
    gray = np.clip(image01 * 255.0, 0.0, 255.0).round().astype(np.uint8)
    Image.fromarray(gray, mode="L").save(path)


def resolve_primary_branch(requested_branch, decision):
    if requested_branch in ("point_map", "depth_unproject"):
        return requested_branch, f"explicitly requested `{requested_branch}`"
    winner = str(decision.get("decision", "")).strip()
    if winner in ("point_map", "depth_unproject"):
        return winner, f"auto-selected decision winner `{winner}`"
    return "depth_unproject", "decision was tied or inconclusive, so fell back to geometry-first default `depth_unproject`"


def write_primary_markdown_report(path, summary):
    primary = summary["primary"]
    branch = summary["branches"][primary["selected_branch"]]
    lines = [
        "# Geometry-First Primary Output",
        "",
        f"- selected_branch: `{primary['selected_branch']}`",
        f"- selected_label: `{primary['selected_label']}`",
        f"- selection_reason: {primary['selection_reason']}",
        f"- target_case: `{summary['case']['seq_name']} / frame {summary['case']['frame_id']} / {summary['case']['target_camera']}`",
        f"- view_profile: `{summary['case']['view_profile']}`",
        f"- source_count: `{summary['case']['source_count']}`",
        "",
        "## Primary Branch Metrics",
        "",
        f"- coverage_ratio: `{branch['render']['coverage_ratio']:.4f}`",
        f"- mae: `{branch['metrics']['mae']:.4f}`",
        f"- psnr: `{branch['metrics']['psnr']:.4f}`",
        f"- ssim: `{branch['metrics']['ssim']:.4f}`",
        f"- exported_points: `{branch['export']['exported_points']}`",
        "",
        "## Files",
        "",
        f"- primary_render_png: `{summary['primary']['render_png']}`",
        f"- primary_weight_png: `{summary['primary']['weight_png']}`",
        f"- primary_point_cloud_ply: `{summary['primary']['point_cloud_ply']}`",
        "",
        "## Note",
        "",
        "- This keeps the comparison artifacts, but also materializes one canonical geometry output for downstream use.",
        "- Under the current mentor direction, `depth + camera` remains the default primary branch unless a stronger reason overrides it.",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def make_tiled_report(output_path, target_image, point_render, depth_render):
    target_u8 = to_uint8(target_image)
    point_u8 = to_uint8(point_render)
    depth_u8 = to_uint8(depth_render)
    point_diff = np.abs(point_u8.astype(np.int16) - target_u8.astype(np.int16)).astype(np.uint8)
    depth_diff = np.abs(depth_u8.astype(np.int16) - target_u8.astype(np.int16)).astype(np.uint8)
    branch_diff = np.abs(point_u8.astype(np.int16) - depth_u8.astype(np.int16)).astype(np.uint8)
    panels = [
        ("Target", target_u8),
        ("Point Map", point_u8),
        ("Depth+Camera", depth_u8),
        ("Point vs Target", point_diff),
        ("Depth vs Target", depth_diff),
        ("Point vs Depth", branch_diff),
    ]
    font = ImageFont.load_default()
    panel_images = []
    panel_width = max(image.shape[1] for _, image in panels)
    panel_height = max(image.shape[0] for _, image in panels)
    label_height = 20
    for label, image in panels:
        canvas = Image.new("RGB", (panel_width, panel_height + label_height), color=(20, 20, 20))
        pil_image = Image.fromarray(image)
        canvas.paste(pil_image, ((panel_width - pil_image.width) // 2, label_height))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 3), label, fill=(255, 255, 255), font=font)
        panel_images.append(canvas)
    rows = []
    for start in range(0, len(panel_images), 3):
        row = panel_images[start : start + 3]
        row_width = sum(image.width for image in row)
        row_height = max(image.height for image in row)
        row_canvas = Image.new("RGB", (row_width, row_height), color=(12, 12, 12))
        cursor = 0
        for image in row:
            row_canvas.paste(image, (cursor, 0))
            cursor += image.width
        rows.append(row_canvas)
    total_width = max(row.width for row in rows)
    total_height = sum(row.height for row in rows)
    mosaic = Image.new("RGB", (total_width, total_height), color=(12, 12, 12))
    cursor_y = 0
    for row in rows:
        mosaic.paste(row, (0, cursor_y))
        cursor_y += row.height
    mosaic.save(output_path)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_markdown_report(path, summary):
    legacy_metrics = summary.get("legacy_report_metrics") or {}
    lines = [
        "# Geometry Branch Baseline For ZJU Report Case",
        "",
        f"- seq_name: `{summary['case']['seq_name']}`",
        f"- frame_id: `{summary['case']['frame_id']}`",
        f"- target_camera: `{summary['case']['target_camera']}`",
        f"- view_profile: `{summary['case']['view_profile']}`",
        f"- source_count: `{summary['case']['source_count']}`",
        f"- local_zju_root: `{summary['run_config']['local_zju_root']}`",
        f"- device: `{summary['environment']['device']}`",
        f"- dtype: `{summary['environment']['dtype']}`",
        "",
        "## Camera Alignment",
        "",
        f"- src_center_rmse_before: `{summary['alignment']['src_center_rmse_before']:.6f}`",
        f"- src_center_rmse_after: `{summary['alignment']['src_center_rmse_after']:.6f}`",
        "",
        "## Branch Stats",
        "",
        "| Branch | Valid Ratio | Valid Points | Conf P50 | Conf P95 | Exported PLY Points | Coverage | MAE | PSNR | SSIM |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for branch_key in ("point_map", "depth_unproject"):
        branch = summary["branches"][branch_key]
        conf = branch["summary"]["confidence_percentiles"]
        render = branch["render"]
        metric = branch["metrics"]
        lines.append(
            "| {name} | {valid_ratio:.4f} | {valid_points} | {p50:.4f} | {p95:.4f} | {exported} | {cov:.4f} | {mae:.4f} | {psnr:.4f} | {ssim:.4f} |".format(
                name=branch["label"],
                valid_ratio=branch["summary"]["valid_ratio"],
                valid_points=branch["summary"]["valid_points"],
                p50=conf.get("p50", float("nan")),
                p95=conf.get("p95", float("nan")),
                exported=branch["export"]["exported_points"],
                cov=render["coverage_ratio"],
                mae=metric["mae"],
                psnr=metric["psnr"],
                ssim=metric["ssim"],
            )
        )
    lines.extend(
        [
            "",
            "## Branch Delta",
            "",
            f"- shared_valid_ratio: `{summary['branch_delta']['shared_valid_ratio']:.4f}`",
            f"- distance_p50: `{summary['branch_delta']['distance_percentiles'].get('p50', float('nan')):.6f}`",
            f"- distance_p90: `{summary['branch_delta']['distance_percentiles'].get('p90', float('nan')):.6f}`",
            "",
            "## Decision",
            "",
            f"- decision: `{summary['decision']['decision']}`",
            f"- mae_winner: `{summary['decision']['mae_winner']}`",
            f"- coverage_winner: `{summary['decision']['coverage_winner']}`",
            "",
            "## Primary Output",
            "",
            f"- selected_branch: `{summary['primary']['selected_branch']}`",
            f"- selected_label: `{summary['primary']['selected_label']}`",
            f"- selection_reason: {summary['primary']['selection_reason']}",
            f"- primary_render_png: `{summary['primary']['render_png']}`",
            f"- primary_point_cloud_ply: `{summary['primary']['point_cloud_ply']}`",
        ]
    )
    if legacy_metrics:
        lines.extend(
            [
                "",
                "## Legacy Reference",
                "",
                f"- old_native_mae: `{legacy_metrics.get('mae', float('nan'))}`",
                f"- old_native_psnr: `{legacy_metrics.get('psnr', float('nan'))}`",
                f"- old_native_ssim: `{legacy_metrics.get('ssim', float('nan'))}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This run uses only the source views from the old ZJU case and re-renders both branches into the real target camera.",
            "- If `Depth+Camera` stays competitive or better here, the geometry-first direction is supported on the human-domain case too.",
            "- Do not restore the legacy ghost stack before these source-only geometry baselines are understood.",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def write_render_csv(path, summary):
    fieldnames = [
        "branch",
        "coverage_ratio",
        "mae",
        "psnr",
        "ssim",
        "rendered_points",
        "input_points",
        "confidence_threshold",
        "exported_points",
    ]
    rows = []
    for branch_key in ("point_map", "depth_unproject"):
        branch = summary["branches"][branch_key]
        rows.append(
            {
                "branch": branch["label"],
                "coverage_ratio": branch["render"]["coverage_ratio"],
                "mae": branch["metrics"]["mae"],
                "psnr": branch["metrics"]["psnr"],
                "ssim": branch["metrics"]["ssim"],
                "rendered_points": branch["render"]["rendered_points"],
                "input_points": branch["render"]["input_points"],
                "confidence_threshold": branch["render"]["confidence_threshold"],
                "exported_points": branch["export"]["exported_points"],
            }
        )
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def decide_winner(point_metrics, depth_metrics, point_render, depth_render):
    mae_winner = "tie"
    coverage_winner = "tie"
    decision = "tie"
    if depth_metrics["mae"] < point_metrics["mae"]:
        mae_winner = "depth_unproject"
    elif point_metrics["mae"] < depth_metrics["mae"]:
        mae_winner = "point_map"
    if depth_render["coverage_ratio"] > point_render["coverage_ratio"]:
        coverage_winner = "depth_unproject"
    elif point_render["coverage_ratio"] > depth_render["coverage_ratio"]:
        coverage_winner = "point_map"
    if (mae_winner == "depth_unproject") and (depth_render["coverage_ratio"] >= point_render["coverage_ratio"]):
        decision = "depth_unproject"
    elif (mae_winner == "point_map") and (point_render["coverage_ratio"] >= depth_render["coverage_ratio"]):
        decision = "point_map"
    return {
        "decision": decision,
        "mae_winner": mae_winner,
        "coverage_winner": coverage_winner,
    }


def main():
    args = parse_args()
    report_json = Path(args.report_json).resolve()
    report_payload = json.loads(report_json.read_text(encoding="utf-8"))
    meta = report_payload["meta"]
    view_profile = infer_view_profile(report_json, meta)

    local_zju_root = Path(args.local_zju_root).resolve() if args.local_zju_root else detect_local_zju_root().resolve()
    seq_dir = local_zju_root / meta["seq_name"]
    output_dir = resolve_output_dir(report_json, args.output_dir, meta)
    ply_dir = ensure_dir(output_dir / "ply")
    render_dir = ensure_dir(output_dir / "renders")

    source_cameras = list(meta["src_cameras"])
    target_camera = str(meta["tgt_camera"])
    frame_id = int(meta["frame_id"])
    source_image_paths = [resolve_frame_image_path(seq_dir, cam, frame_id) for cam in source_cameras]
    target_image_path = resolve_frame_image_path(seq_dir, target_camera, frame_id)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    print(f"[info] output_dir={output_dir}")
    print(f"[info] seq={meta['seq_name']} frame={frame_id} target={target_camera} profile={view_profile}")
    print(f"[info] source_count={len(source_image_paths)} device={device} dtype={dtype}")

    model = load_model(args.checkpoint, device)
    images = load_and_preprocess_images([str(path) for path in source_image_paths], mode=args.preprocess_mode).to(device)
    source_colors = np.clip(images.detach().cpu().numpy().transpose(0, 2, 3, 1), 0.0, 1.0).astype(np.float32)

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=dtype)
        if device.type == "cuda" and dtype != torch.float32
        else contextlib.nullcontext()
    )
    with torch.no_grad():
        with autocast_ctx:
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    outputs = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            outputs[key] = value.detach().cpu().numpy().squeeze(0)
    outputs["extrinsic"] = extrinsic.detach().cpu().numpy().squeeze(0)
    outputs["intrinsic"] = intrinsic.detach().cpu().numpy().squeeze(0)
    outputs["world_points_from_depth"] = unproject_depth_map_to_point_map(
        outputs["depth"], outputs["extrinsic"], outputs["intrinsic"]
    )

    point_map = np.asarray(outputs["world_points"], dtype=np.float32)
    point_conf = np.asarray(outputs["world_points_conf"], dtype=np.float32)
    depth_points = np.asarray(outputs["world_points_from_depth"], dtype=np.float32)
    depth_conf = np.asarray(outputs["depth_conf"], dtype=np.float32)

    gt_cameras = load_zju_cameras(seq_dir, source_cameras + [target_camera])
    centers_pred = camera_centers(np.asarray(outputs["extrinsic"], dtype=np.float64))
    centers_gt = camera_centers(np.stack([gt_cameras[cam]["extrinsic"] for cam in source_cameras], axis=0))
    rmse_before = rmse(centers_pred, centers_gt)
    scale, sim_rotation, sim_translation = umeyama_similarity(centers_pred, centers_gt)
    centers_after = apply_sim3_points(centers_pred, scale, sim_rotation, sim_translation)
    rmse_after = rmse(centers_after, centers_gt)

    point_map_aligned = apply_sim3_points(point_map.reshape(-1, 3), scale, sim_rotation, sim_translation).reshape(point_map.shape)
    depth_points_aligned = apply_sim3_points(depth_points.reshape(-1, 3), scale, sim_rotation, sim_translation).reshape(depth_points.shape)

    target_image_full = np.asarray(Image.open(str(target_image_path)).convert("RGB"), dtype=np.float32) / 255.0
    render_hw = (int(args.render_size[0]), int(args.render_size[1]))
    target_image = np.asarray(
        Image.fromarray(np.clip(target_image_full * 255.0, 0.0, 255.0).astype(np.uint8)).resize(
            (render_hw[1], render_hw[0]), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0
    target_intrinsic = scale_intrinsic(gt_cameras[target_camera]["intrinsic"], target_image_full.shape[:2], render_hw)
    target_extrinsic = gt_cameras[target_camera]["extrinsic"]

    point_render = render_forward_splat(
        world_points_s_hw3=point_map_aligned,
        world_conf_s_hw=point_conf,
        src_rgb_s_hw3=source_colors,
        tgt_extrinsic_3x4=target_extrinsic,
        tgt_intrinsic_3x3=target_intrinsic,
        out_hw=render_hw,
        z_eps=DEPTH_EPS,
        min_conf=args.min_conf,
        z_tolerance=args.z_tolerance,
        max_points=args.render_max_points,
    )
    depth_render = render_forward_splat(
        world_points_s_hw3=depth_points_aligned,
        world_conf_s_hw=depth_conf,
        src_rgb_s_hw3=source_colors,
        tgt_extrinsic_3x4=target_extrinsic,
        tgt_intrinsic_3x3=target_intrinsic,
        out_hw=render_hw,
        z_eps=DEPTH_EPS,
        min_conf=args.min_conf,
        z_tolerance=args.z_tolerance,
        max_points=args.render_max_points,
    )

    point_export = export_branch_ply(
        ply_dir / "point_map_aligned.ply",
        point_map_aligned,
        point_conf,
        to_uint8(source_colors),
        args.conf_percentile,
        args.export_max_points,
    )
    depth_export = export_branch_ply(
        ply_dir / "depth_unproject_aligned.ply",
        depth_points_aligned,
        depth_conf,
        to_uint8(source_colors),
        args.conf_percentile,
        args.export_max_points,
    )

    point_summary = summarize_branch(point_map_aligned, point_conf)
    depth_summary = summarize_branch(depth_points_aligned, depth_conf)
    branch_delta = summarize_branch_delta(point_map_aligned, depth_points_aligned, point_conf, depth_conf)
    point_metrics = metrics(point_render["image"], target_image)
    depth_metrics = metrics(depth_render["image"], target_image)
    decision = decide_winner(point_metrics, depth_metrics, point_render["stats"], depth_render["stats"])

    target_png = render_dir / "target.png"
    point_render_png = render_dir / "point_map.png"
    depth_render_png = render_dir / "depth_unproject.png"
    point_weight_png = render_dir / "point_map_weight.png"
    depth_weight_png = render_dir / "depth_unproject_weight.png"
    comparison_png = render_dir / "target_compare.png"
    save_rgb_png(target_png, target_image)
    save_rgb_png(point_render_png, point_render["image"])
    save_rgb_png(depth_render_png, depth_render["image"])
    save_gray_png(point_weight_png, point_render["weight"])
    save_gray_png(depth_weight_png, depth_render["weight"])
    make_tiled_report(comparison_png, target_image, point_render["image"], depth_render["image"])

    primary_branch, primary_reason = resolve_primary_branch(args.primary_branch, decision)
    primary_render_alias = render_dir / "primary_render.png"
    primary_weight_alias = render_dir / "primary_weight.png"
    primary_ply_alias = ply_dir / "primary_geometry.ply"
    if primary_branch == "point_map":
        shutil.copyfile(point_render_png, primary_render_alias)
        shutil.copyfile(point_weight_png, primary_weight_alias)
        shutil.copyfile(ply_dir / "point_map_aligned.ply", primary_ply_alias)
    else:
        shutil.copyfile(depth_render_png, primary_render_alias)
        shutil.copyfile(depth_weight_png, primary_weight_alias)
        shutil.copyfile(ply_dir / "depth_unproject_aligned.ply", primary_ply_alias)

    if not args.skip_save_predictions:
        np.savez_compressed(
            output_dir / "predictions.npz",
            point_map=point_map,
            point_conf=point_conf,
            depth_points=depth_points,
            depth_conf=depth_conf,
            point_map_aligned=point_map_aligned,
            depth_points_aligned=depth_points_aligned,
            pred_extrinsic=outputs["extrinsic"],
            pred_intrinsic=outputs["intrinsic"],
            source_image_names=np.array([path.name for path in source_image_paths], dtype=object),
            target_image_name=np.array([target_image_path.name], dtype=object),
        )

    summary = {
        "run_config": {
            "report_json": str(report_json),
            "local_zju_root": str(local_zju_root),
            "output_dir": str(output_dir.resolve()),
            "checkpoint": args.checkpoint or DEFAULT_MODEL_URL,
            "preprocess_mode": args.preprocess_mode,
            "conf_percentile": float(args.conf_percentile),
            "export_max_points": int(args.export_max_points),
            "render_max_points": int(args.render_max_points),
            "render_size": list(render_hw),
            "z_tolerance": float(args.z_tolerance),
            "min_conf": float(args.min_conf),
            "save_predictions": not bool(args.skip_save_predictions),
        },
        "environment": {
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        },
        "case": {
            "seq_name": meta["seq_name"],
            "frame_id": frame_id,
            "view_profile": view_profile,
            "source_cameras": source_cameras,
            "target_camera": target_camera,
            "source_count": len(source_cameras),
            "source_image_paths": [str(path) for path in source_image_paths],
            "target_image_path": str(target_image_path),
        },
        "alignment": {
            "scale": float(scale),
            "src_center_rmse_before": float(rmse_before),
            "src_center_rmse_after": float(rmse_after),
            "sim3_rotation": sim_rotation.tolist(),
            "sim3_translation": sim_translation.tolist(),
        },
        "branches": {
            "point_map": {
                "label": "Point Map",
                "summary": point_summary,
                "export": point_export,
                "render": {**point_render["stats"], "confidence_threshold": float(point_export["confidence_threshold"])},
                "metrics": point_metrics,
            },
            "depth_unproject": {
                "label": "Depth+Camera",
                "summary": depth_summary,
                "export": depth_export,
                "render": {**depth_render["stats"], "confidence_threshold": float(depth_export["confidence_threshold"])},
                "metrics": depth_metrics,
            },
        },
        "branch_delta": branch_delta,
        "decision": decision,
        "primary": {
            "requested_branch": args.primary_branch,
            "selected_branch": primary_branch,
            "selected_label": "Point Map" if primary_branch == "point_map" else "Depth+Camera",
            "selection_reason": primary_reason,
            "render_png": str(primary_render_alias.relative_to(output_dir)),
            "weight_png": str(primary_weight_alias.relative_to(output_dir)),
            "point_cloud_ply": str(primary_ply_alias.relative_to(output_dir)),
        },
        "files": {
            "target_png": str(target_png.relative_to(output_dir)),
            "point_map_render_png": str(point_render_png.relative_to(output_dir)),
            "depth_unproject_render_png": str(depth_render_png.relative_to(output_dir)),
            "point_map_weight_png": str(point_weight_png.relative_to(output_dir)),
            "depth_unproject_weight_png": str(depth_weight_png.relative_to(output_dir)),
            "comparison_png": str(comparison_png.relative_to(output_dir)),
            "point_map_ply": str((ply_dir / "point_map_aligned.ply").relative_to(output_dir)),
            "depth_unproject_ply": str((ply_dir / "depth_unproject_aligned.ply").relative_to(output_dir)),
        },
        "legacy_report_metrics": report_payload.get("metrics", {}).get("native", {}),
    }

    save_json(output_dir / "summary.json", summary)
    write_markdown_report(output_dir / "summary.md", summary)
    save_json(output_dir / "primary_summary.json", summary["primary"])
    write_primary_markdown_report(output_dir / "primary_summary.md", summary)
    write_render_csv(output_dir / "render_metrics.csv", summary)

    print("[done] ZJU geometry branch baseline complete.")
    print(f"[done] Summary: {output_dir / 'summary.md'}")
    print(f"[done] Compare PNG: {comparison_png}")


if __name__ == "__main__":
    main()
