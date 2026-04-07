import argparse
import json
import math
import sys
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

from scripts.compare_geometry_branches_zju_report import (  # noqa: E402
    DEPTH_EPS,
    detect_local_zju_root,
    load_model,
    load_zju_cameras,
    metrics as full_frame_metrics,
    render_forward_splat,
    resolve_device,
    resolve_dtype,
    resolve_frame_image_path,
    save_json,
    scale_intrinsic,
    to_uint8,
    umeyama_similarity,
    apply_sim3_points,
    camera_centers,
    rmse,
)
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402
from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402


DEFAULT_VARIANTS = [
    "guided_blur",
    "mask_hole_fill",
    "mask_hole_fill_plus_guided",
]
SIDE_VARIANTS = [
    "mask_hole_fill_plus_guided_strong",
    "mask_hole_fill_plus_guided_soft",
]
LEGACY_PEAK_COLLAPSE_PROXY_VARIANTS = [
    "soft_top1_inside_fg",
    "soft_top1_margin_inside_fg",
    "soft_top1_margin_plus_bottom_suppress",
    "soft_top1_margin_plus_fg_lcc_proxy",
]
CORRESPONDENCE_PROXY_VARIANTS = [
    "consensus_medoid_inside_fg",
    "consensus_margin_inside_fg",
    "consensus_label_smooth_inside_fg",
    "consensus_margin_plus_coverage_floor",
]
PROXY_VARIANTS = LEGACY_PEAK_COLLAPSE_PROXY_VARIANTS + CORRESPONDENCE_PROXY_VARIANTS
DEFAULT_PROXY_CONFIG = {
    "render_mode": "rehydrated",
    "alpha_floor": 0.15,
    "support_gate_pow": 1.0,
    "baseline_blend": 0.30,
    "gaussian_sigma": 0.0,
    "morph_close_k": 0,
    "label_majority_k": 5,
    "label_smooth_mix": 0.45,
    "consensus_margin_floor": 0.05,
    "source_subset": [],
    "coverage_floor_ratio": 0.78,
    "coverage_floor_mix": 0.35,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen-teacher visual-lift variants on explicit ZJU case manifests."
    )
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case-set", default="cases", choices=["cases", "hero_cases", "benchmark_cases", "all"])
    parser.add_argument("--variants", default="default")
    parser.add_argument("--local-zju-root", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--preprocess-mode", default="crop", choices=["crop", "pad"])
    parser.add_argument("--render-size", nargs=2, type=int, default=[518, 518])
    parser.add_argument("--render-max-points", type=int, default=750000)
    parser.add_argument("--z-tolerance", type=float, default=0.02)
    parser.add_argument("--min-conf", type=float, default=1e-6)
    parser.add_argument("--target-mask-source", default="mask", choices=["none", "mask", "mask_cihp"])
    parser.add_argument("--support-threshold", type=float, default=0.25)
    parser.add_argument("--bottom-band-ratio", type=float, default=0.2)
    parser.add_argument("--proxy-config-json", default="")
    return parser.parse_args()


def _load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict manifest: {path}")
    return payload


def _load_proxy_config(path_text: str) -> dict:
    payload = dict(DEFAULT_PROXY_CONFIG)
    text = str(path_text or "").strip()
    if not text:
        return payload
    raw = json.loads(Path(text).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"proxy config must be a dict: {text}")
    payload.update(raw)
    payload["render_mode"] = str(payload.get("render_mode", "alpha_only")).strip() or "alpha_only"
    payload["alpha_floor"] = float(payload.get("alpha_floor", 0.0))
    payload["support_gate_pow"] = float(payload.get("support_gate_pow", 1.0))
    payload["baseline_blend"] = float(payload.get("baseline_blend", 0.0))
    payload["gaussian_sigma"] = float(payload.get("gaussian_sigma", 0.0))
    payload["morph_close_k"] = int(payload.get("morph_close_k", 0))
    payload["label_majority_k"] = max(int(payload.get("label_majority_k", 5)), 1)
    payload["label_smooth_mix"] = float(payload.get("label_smooth_mix", 0.45))
    payload["consensus_margin_floor"] = float(payload.get("consensus_margin_floor", 0.05))
    payload["coverage_floor_ratio"] = float(payload.get("coverage_floor_ratio", 0.78))
    payload["coverage_floor_mix"] = float(payload.get("coverage_floor_mix", 0.35))
    raw_subset = payload.get("source_subset", [])
    payload["source_subset"] = [int(item) for item in raw_subset] if isinstance(raw_subset, list) else []
    return payload


def _select_cases(manifest: dict, case_set: str) -> list[dict]:
    if case_set == "all":
        rows = []
        for key in ("hero_cases", "benchmark_cases", "cases"):
            rows.extend(list(manifest.get(key, [])))
        return rows
    return list(manifest.get(case_set, []))


def _resolve_variants(raw_value: str) -> list[str]:
    text = str(raw_value or "").strip()
    if not text or text == "default":
        return list(DEFAULT_VARIANTS)
    if text == "none":
        return []
    if text == "default_plus_side":
        return list(DEFAULT_VARIANTS) + list(SIDE_VARIANTS)
    return [item.strip() for item in text.split(",") if item.strip()]


def _load_target_mask(seq_dir: Path, camera_name: str, frame_id: int, out_hw: tuple[int, int], mask_source: str) -> np.ndarray:
    if mask_source == "none":
        return np.ones(out_hw, dtype=bool)
    mask_path = seq_dir / mask_source / camera_name / f"{int(frame_id):06d}.png"
    mask = Image.open(mask_path).convert("L").resize((int(out_hw[1]), int(out_hw[0])), Image.Resampling.NEAREST)
    return np.asarray(mask, dtype=np.uint8) > 0


def _gaussian_kernel(channels: int, device: torch.device, dtype: torch.dtype, ksize: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    values = torch.exp(-(coords**2) / (2 * sigma * sigma))
    values = values / values.sum()
    kernel_2d = (values[:, None] * values[None, :]).unsqueeze(0).unsqueeze(0)
    return kernel_2d.repeat(channels, 1, 1, 1)


def _ssim_rgb(pred01: np.ndarray, tgt01: np.ndarray) -> float:
    pred = torch.from_numpy(pred01.transpose(2, 0, 1)).unsqueeze(0).float()
    tgt = torch.from_numpy(tgt01.transpose(2, 0, 1)).unsqueeze(0).float()
    c1 = 0.01**2
    c2 = 0.03**2
    kernel = _gaussian_kernel(3, pred.device, pred.dtype)

    def filt(x: torch.Tensor) -> torch.Tensor:
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
    return float(ssim_map.mean().item())


def _masked_metrics(pred01: np.ndarray, tgt01: np.ndarray, mask_hw: np.ndarray) -> dict:
    mask = np.asarray(mask_hw, dtype=np.float32)[..., None]
    denom = max(float(mask.sum() * 3.0), 1.0)
    abs_err = np.abs(pred01 - tgt01) * mask
    mse = float((((pred01 - tgt01) ** 2) * mask).sum() / denom)
    pred_masked = pred01 * mask
    tgt_masked = tgt01 * mask
    return {
        "l1": float(abs_err.sum() / denom),
        "psnr": float(-10.0 * math.log10(max(mse, 1e-8))),
        "ssim": _ssim_rgb(pred_masked, tgt_masked),
        "pixel_count": int(mask_hw.sum()),
    }


def _clamp_image(image01: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(image01, dtype=np.float32), 0.0, 1.0)


def _apply_variant(depth_render: np.ndarray, weight01: np.ndarray, fg_mask: np.ndarray, variant_name: str) -> np.ndarray:
    depth_render = _clamp_image(depth_render)
    weight01 = np.asarray(weight01, dtype=np.float32)
    fg_mask = np.asarray(fg_mask, dtype=bool)
    if variant_name == "guided_blur":
        blur = cv2.GaussianBlur(to_uint8(depth_render), (0, 0), 1.2).astype(np.float32) / 255.0
        alpha = np.clip((weight01 - 0.08) / 0.25, 0.0, 1.0)[..., None]
        return _clamp_image(depth_render * alpha + blur * (1.0 - alpha))

    hole = (weight01 < 0.03).astype(np.uint8) * 255
    inpaint = cv2.inpaint(to_uint8(depth_render), hole, 3, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
    if variant_name == "mask_hole_fill":
        mask_hole = ((weight01 < 0.05) & fg_mask).astype(np.float32)[..., None]
        return _clamp_image(depth_render * (1.0 - mask_hole) + inpaint * mask_hole)

    if variant_name == "mask_hole_fill_plus_guided":
        blur = cv2.GaussianBlur(to_uint8(depth_render), (0, 0), 1.2).astype(np.float32) / 255.0
        alpha = np.clip((weight01 - 0.08) / 0.25, 0.0, 1.0)[..., None]
        guided = depth_render * alpha + blur * (1.0 - alpha)
        mask_hole = ((weight01 < 0.05) & fg_mask).astype(np.float32)[..., None]
        return _clamp_image(guided * (1.0 - mask_hole) + inpaint * mask_hole)

    if variant_name == "mask_hole_fill_plus_guided_strong":
        blur = cv2.GaussianBlur(to_uint8(depth_render), (0, 0), 1.6).astype(np.float32) / 255.0
        alpha = np.clip((weight01 - 0.1) / 0.28, 0.0, 1.0)[..., None]
        guided = depth_render * alpha + blur * (1.0 - alpha)
        mask_hole = ((weight01 < 0.07) & fg_mask).astype(np.float32)[..., None]
        return _clamp_image(guided * (1.0 - mask_hole) + inpaint * mask_hole)

    if variant_name == "mask_hole_fill_plus_guided_soft":
        mask_hole = ((np.clip(0.12 - weight01, 0.0, 0.12) / 0.12) * fg_mask.astype(np.float32))[..., None]
        blur = cv2.GaussianBlur(to_uint8(inpaint), (0, 0), 0.8).astype(np.float32) / 255.0
        soft_inpaint = inpaint * 0.7 + blur * 0.3
        return _clamp_image(depth_render * (1.0 - mask_hole) + soft_inpaint * mask_hole)

    raise ValueError(f"Unsupported variant: {variant_name}")


def _save_rgb(path: Path, image01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(image01)).save(path)


def _save_gray(path: Path, image01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gray = np.clip(np.asarray(image01, dtype=np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(gray, mode="L").save(path)


def _build_bottom_band_mask(mask_hw: np.ndarray, bottom_band_ratio: float) -> np.ndarray:
    bottom_band_ratio = float(max(0.0, min(0.95, bottom_band_ratio)))
    bottom = np.zeros_like(mask_hw, dtype=bool)
    if bottom_band_ratio <= 0.0:
        return bottom
    cutoff = int(mask_hw.shape[0] * (1.0 - bottom_band_ratio))
    if cutoff < mask_hw.shape[0]:
        bottom[cutoff:, :] = True
    return bottom


def _apply_heatmap(values01: np.ndarray) -> np.ndarray:
    gray = np.clip(np.asarray(values01, dtype=np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def _visible_rgb_mask(image01: np.ndarray, *, threshold: float = 0.05) -> np.ndarray:
    image01 = np.asarray(image01, dtype=np.float32)
    if image01.ndim != 3 or image01.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got shape={image01.shape}")
    return np.max(image01, axis=2) > float(threshold)


def _normalize_support_weight(raw_weight: np.ndarray) -> np.ndarray:
    raw_weight = np.clip(np.asarray(raw_weight, dtype=np.float32), 0.0, None)
    hit = raw_weight > 0.0
    if not np.any(hit):
        return np.zeros_like(raw_weight, dtype=np.float32)
    p99 = float(np.percentile(raw_weight[hit], 99.0))
    denom = max(p99, 1e-8)
    return np.clip(raw_weight / denom, 0.0, 1.0).astype(np.float32)


def _render_forward_splat_with_raw(
    *,
    world_points_s_hw3: np.ndarray,
    world_conf_s_hw: np.ndarray,
    src_rgb_s_hw3: np.ndarray,
    tgt_extrinsic_3x4: np.ndarray,
    tgt_intrinsic_3x3: np.ndarray,
    out_hw: tuple[int, int],
    z_eps: float,
    min_conf: float,
    z_tolerance: float,
    max_points: int,
) -> dict:
    render = render_forward_splat(
        world_points_s_hw3=world_points_s_hw3,
        world_conf_s_hw=world_conf_s_hw,
        src_rgb_s_hw3=src_rgb_s_hw3,
        tgt_extrinsic_3x4=tgt_extrinsic_3x4,
        tgt_intrinsic_3x3=tgt_intrinsic_3x3,
        out_hw=out_hw,
        z_eps=z_eps,
        min_conf=min_conf,
        z_tolerance=z_tolerance,
        max_points=max_points,
    )

    height, width = int(out_hw[0]), int(out_hw[1])
    points = np.asarray(world_points_s_hw3, dtype=np.float64).reshape(-1, 3)
    conf = np.asarray(world_conf_s_hw, dtype=np.float64).reshape(-1)
    finite = np.isfinite(points).all(axis=-1) & np.isfinite(conf) & (conf >= float(min_conf))
    candidate_indices = np.flatnonzero(finite)
    if candidate_indices.size == 0:
        render["raw_weight"] = np.zeros((height, width), dtype=np.float32)
        return render

    if candidate_indices.size > int(max_points):
        candidate_conf = conf[candidate_indices]
        top_local = np.argpartition(candidate_conf, -int(max_points))[-int(max_points):]
        candidate_indices = candidate_indices[top_local]

    points = points[candidate_indices]
    conf = conf[candidate_indices]

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
        render["raw_weight"] = np.zeros((height, width), dtype=np.float32)
        return render

    pix = yi[valid] * width + xi[valid]
    z_valid = z[valid]
    conf_valid = conf[valid]

    order = np.lexsort((z_valid, pix))
    pix_sorted = pix[order]
    z_sorted = z_valid[order]
    conf_sorted = conf_valid[order]

    _, start_idx = np.unique(pix_sorted, return_index=True)
    counts = np.diff(np.concatenate([start_idx, np.array([pix_sorted.size])]))
    z_min = z_sorted[start_idx]
    z_min_rep = np.repeat(z_min, counts)
    keep = z_sorted <= (z_min_rep + float(z_tolerance))

    pix_kept = pix_sorted[keep]
    conf_kept = conf_sorted[keep]
    pixel_count = int(height * width)
    sum_w = np.bincount(pix_kept, weights=conf_kept, minlength=pixel_count).astype(np.float64)
    render["raw_weight"] = sum_w.reshape(height, width).astype(np.float32)
    return render


def _count_profile_peaks(profile: np.ndarray, *, min_rel_height: float = 0.35) -> int:
    profile = np.asarray(profile, dtype=np.float32).reshape(-1)
    if profile.size < 3:
        return 0
    peak_threshold = float(profile.max()) * float(min_rel_height)
    if peak_threshold <= 0.0:
        return 0
    peak_count = 0
    for idx in range(1, profile.size - 1):
        center = float(profile[idx])
        if center < peak_threshold:
            continue
        if center >= float(profile[idx - 1]) and center >= float(profile[idx + 1]):
            peak_count += 1
    return int(peak_count)


def _connected_component_ratio(binary_mask: np.ndarray) -> tuple[int, float]:
    binary_mask = np.asarray(binary_mask, dtype=np.uint8)
    component_count, component_labels = cv2.connectedComponents(binary_mask, connectivity=8)
    fg_connected_components = max(int(component_count) - 1, 0)
    if fg_connected_components <= 0:
        return 0, 0.0
    component_areas = [int((component_labels == label).sum()) for label in range(1, component_count)]
    largest_area = max(component_areas) if component_areas else 0
    ratio = float(largest_area / max(int(binary_mask.sum()), 1))
    return fg_connected_components, ratio


def _smooth_source_label_map(source_label_map: np.ndarray, fg_mask: np.ndarray, num_sources: int, *, kernel_size: int = 5) -> np.ndarray:
    fg_mask = np.asarray(fg_mask, dtype=bool)
    source_label_map = np.asarray(source_label_map, dtype=np.int32)
    probs = []
    kernel = (int(kernel_size), int(kernel_size))
    for source_idx in range(int(num_sources)):
        label_mask = ((source_label_map == source_idx) & fg_mask).astype(np.float32)
        probs.append(cv2.blur(label_mask, kernel))
    probs = np.stack(probs, axis=0).astype(np.float32)
    denom = np.clip(probs.sum(axis=0, keepdims=True), 1e-8, None)
    return probs / denom


def _smooth_gate_map(gate_map: np.ndarray, *, gaussian_sigma: float, morph_close_k: int) -> np.ndarray:
    gate_map = np.clip(np.asarray(gate_map, dtype=np.float32), 0.0, 1.0)
    result = gate_map
    if float(gaussian_sigma) > 0.0:
        result = cv2.GaussianBlur(result, (0, 0), float(gaussian_sigma))
    kernel_size = int(max(morph_close_k, 0))
    if kernel_size > 1:
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        result_u8 = np.clip(result * 255.0, 0.0, 255.0).astype(np.uint8)
        result_u8 = cv2.morphologyEx(result_u8, cv2.MORPH_CLOSE, kernel)
        result = result_u8.astype(np.float32) / 255.0
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _label_smoothness_map(source_label_map: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    fg_mask = np.asarray(fg_mask, dtype=bool)
    labels = np.asarray(source_label_map, dtype=np.int32)
    smooth = np.zeros_like(labels, dtype=np.float32)
    count = np.zeros_like(labels, dtype=np.float32)
    for axis, head, tail in ((0, slice(1, None), slice(None, -1)), (1, slice(1, None), slice(None, -1))):
        if axis == 0:
            valid = fg_mask[head, :] & fg_mask[tail, :]
            same = labels[head, :] == labels[tail, :]
            smooth[head, :] += (same & valid).astype(np.float32)
            smooth[tail, :] += (same & valid).astype(np.float32)
            count[head, :] += valid.astype(np.float32)
            count[tail, :] += valid.astype(np.float32)
        else:
            valid = fg_mask[:, head] & fg_mask[:, tail]
            same = labels[:, head] == labels[:, tail]
            smooth[:, head] += (same & valid).astype(np.float32)
            smooth[:, tail] += (same & valid).astype(np.float32)
            count[:, head] += valid.astype(np.float32)
            count[:, tail] += valid.astype(np.float32)
    result = np.zeros_like(smooth, dtype=np.float32)
    valid = count > 0
    result[valid] = smooth[valid] / count[valid]
    return np.where(fg_mask, result, 0.0).astype(np.float32)


def _source_label_switch_count(source_label_map: np.ndarray, fg_mask: np.ndarray) -> float:
    fg_mask = np.asarray(fg_mask, dtype=bool)
    labels = np.asarray(source_label_map, dtype=np.int32)
    switch_count = 0
    for axis, head, tail in ((0, slice(1, None), slice(None, -1)), (1, slice(1, None), slice(None, -1))):
        if axis == 0:
            valid = fg_mask[head, :] & fg_mask[tail, :]
            switch_count += int(((labels[head, :] != labels[tail, :]) & valid).sum())
        else:
            valid = fg_mask[:, head] & fg_mask[:, tail]
            switch_count += int(((labels[:, head] != labels[:, tail]) & valid).sum())
    return float(switch_count)


def _source_label_fragmentation(source_label_map: np.ndarray, fg_mask: np.ndarray, num_sources: int) -> float:
    total_components = 0
    for source_idx in range(int(num_sources)):
        binary_mask = ((source_label_map == source_idx) & fg_mask).astype(np.uint8)
        component_count, _labels = cv2.connectedComponents(binary_mask, connectivity=8)
        total_components += max(int(component_count) - 1, 0)
    return float(total_components)


def _source_agreement_maps(source_support: np.ndarray, source_images: np.ndarray, fg_mask: np.ndarray) -> dict:
    source_support = np.clip(np.asarray(source_support, dtype=np.float32), 0.0, None)
    source_images = np.clip(np.asarray(source_images, dtype=np.float32), 0.0, 1.0)
    fg_mask = np.asarray(fg_mask, dtype=bool)
    if source_support.ndim != 3:
        raise ValueError(f"source_support must be [S,H,W], got shape={source_support.shape}")
    if source_images.ndim != 4:
        raise ValueError(f"source_images must be [S,H,W,3], got shape={source_images.shape}")

    total_support = source_support.sum(axis=0)
    denom = np.clip(total_support[None, ...], 1e-8, None)
    probs = source_support / denom
    sorted_probs = np.sort(probs, axis=0)
    top1 = sorted_probs[-1]
    top2 = sorted_probs[-2] if probs.shape[0] > 1 else np.zeros_like(top1)
    top1_idx = np.argmax(probs, axis=0).astype(np.float32)
    if probs.shape[0] > 1:
        entropy = -(probs * np.log(np.clip(probs, 1e-8, None))).sum(axis=0) / float(np.log(float(probs.shape[0])))
    else:
        entropy = np.zeros_like(top1, dtype=np.float32)

    support01_stack = np.stack([_normalize_support_weight(source_support[idx]) for idx in range(source_support.shape[0])], axis=0)
    agreement_strength = np.zeros_like(source_support, dtype=np.float32)
    for src_idx in range(source_support.shape[0]):
        src_score = np.zeros_like(total_support, dtype=np.float32)
        for other_idx in range(source_support.shape[0]):
            color_delta = np.abs(source_images[src_idx] - source_images[other_idx]).mean(axis=2)
            support_delta = np.abs(support01_stack[src_idx] - support01_stack[other_idx])
            similarity = np.exp(-color_delta / 0.15) * np.exp(-support_delta / 0.25)
            src_score += probs[other_idx] * similarity.astype(np.float32)
        agreement_strength[src_idx] = src_score

    medoid_scores = probs * agreement_strength
    medoid_sum = np.clip(medoid_scores.sum(axis=0, keepdims=True), 1e-8, None)
    medoid_probs = medoid_scores / medoid_sum
    sorted_medoid_probs = np.sort(medoid_probs, axis=0)
    medoid_top1 = sorted_medoid_probs[-1]
    medoid_top2 = sorted_medoid_probs[-2] if medoid_probs.shape[0] > 1 else np.zeros_like(medoid_top1)
    medoid_label_map = np.argmax(medoid_probs, axis=0).astype(np.int32)
    medoid_consensus_map = np.max(agreement_strength, axis=0).astype(np.float32)
    medoid_label_smoothness_map = _label_smoothness_map(medoid_label_map, fg_mask)
    smoothed_label_prior = _smooth_source_label_map(medoid_label_map, fg_mask, medoid_probs.shape[0], kernel_size=5)

    return {
        "probs": probs.astype(np.float32),
        "entropy_map": np.where(fg_mask, entropy, 0.0).astype(np.float32),
        "top1_mass_map": np.where(fg_mask, top1, 0.0).astype(np.float32),
        "top1_margin_map": np.where(fg_mask, top1 - top2, 0.0).astype(np.float32),
        "top1_index_map": np.where(fg_mask, top1_idx + 1.0, 0.0).astype(np.float32),
        "medoid_probs": medoid_probs.astype(np.float32),
        "medoid_support_map": np.where(fg_mask, medoid_top1, 0.0).astype(np.float32),
        "medoid_margin_map": np.where(fg_mask, medoid_top1 - medoid_top2, 0.0).astype(np.float32),
        "medoid_label_map": np.where(fg_mask, medoid_label_map + 1, 0).astype(np.float32),
        "consensus_ratio_map": np.where(fg_mask, medoid_consensus_map, 0.0).astype(np.float32),
        "label_smoothness_map": medoid_label_smoothness_map.astype(np.float32),
        "smoothed_label_prior": smoothed_label_prior.astype(np.float32),
    }


def _source_dominance_statistics(
    source_support: np.ndarray,
    source_images: np.ndarray,
    fg_mask: np.ndarray,
    *,
    support_threshold: float,
) -> dict:
    source_support = np.clip(np.asarray(source_support, dtype=np.float32), 0.0, None)
    fg_mask = np.asarray(fg_mask, dtype=bool)
    total_support = source_support.sum(axis=0)
    inside_fg = fg_mask & (total_support > 1e-8)
    zero_map = np.zeros_like(total_support, dtype=np.float32)
    if not np.any(inside_fg):
        return {
            "source_entropy_inside_fg": 1.0,
            "source_top1_mass_ratio_inside_fg": 0.0,
            "source_top1_top2_margin_inside_fg": 0.0,
            "source_medoid_support_ratio_inside_fg": 0.0,
            "correspondence_consensus_ratio_inside_fg": 0.0,
            "source_label_smoothness_inside_fg": 0.0,
            "source_id_switch_count_inside_fg": 0.0,
            "source_top1_spatial_fragmentation": 0.0,
            "source_entropy_map": zero_map,
            "source_top1_mass_map": zero_map,
            "source_top1_top2_margin_map": zero_map,
            "source_top1_index_map": zero_map,
            "source_medoid_support_map": zero_map,
            "source_medoid_top1_top2_margin_map": zero_map,
            "source_medoid_index_map": zero_map,
            "correspondence_consensus_map": zero_map,
            "source_label_smoothness_map": zero_map,
            "fg_largest_component_ratio": 0.0,
        }

    agreement_maps = _source_agreement_maps(source_support, source_images, fg_mask)
    fg_weights = total_support[inside_fg]
    fg_weights = fg_weights / max(float(fg_weights.sum()), 1e-8)
    high_support_fg = inside_fg & (_normalize_support_weight(total_support) >= float(support_threshold))
    _, largest_ratio = _connected_component_ratio(high_support_fg.astype(np.uint8))
    medoid_label_zero_based = np.clip(np.asarray(agreement_maps["medoid_label_map"], dtype=np.int32) - 1, 0, None)
    return {
        "source_entropy_inside_fg": float((agreement_maps["entropy_map"][inside_fg] * fg_weights).sum()),
        "source_top1_mass_ratio_inside_fg": float((agreement_maps["top1_mass_map"][inside_fg] * fg_weights).sum()),
        "source_top1_top2_margin_inside_fg": float((agreement_maps["top1_margin_map"][inside_fg] * fg_weights).sum()),
        "source_medoid_support_ratio_inside_fg": float((agreement_maps["medoid_support_map"][inside_fg] * fg_weights).sum()),
        "correspondence_consensus_ratio_inside_fg": float((agreement_maps["consensus_ratio_map"][inside_fg] * fg_weights).sum()),
        "source_label_smoothness_inside_fg": float((agreement_maps["label_smoothness_map"][inside_fg] * fg_weights).sum()),
        "source_id_switch_count_inside_fg": _source_label_switch_count(medoid_label_zero_based, fg_mask),
        "source_top1_spatial_fragmentation": _source_label_fragmentation(
            medoid_label_zero_based,
            fg_mask,
            source_support.shape[0],
        ),
        "source_entropy_map": agreement_maps["entropy_map"],
        "source_top1_mass_map": agreement_maps["top1_mass_map"],
        "source_top1_top2_margin_map": agreement_maps["top1_margin_map"],
        "source_top1_index_map": agreement_maps["top1_index_map"],
        "source_medoid_support_map": agreement_maps["medoid_support_map"],
        "source_medoid_top1_top2_margin_map": agreement_maps["medoid_margin_map"],
        "source_medoid_index_map": agreement_maps["medoid_label_map"],
        "correspondence_consensus_map": agreement_maps["consensus_ratio_map"],
        "source_label_smoothness_map": agreement_maps["label_smoothness_map"],
        "fg_largest_component_ratio": float(largest_ratio),
    }


def _build_proxy_variant_artifacts(
    *,
    source_images: np.ndarray,
    source_raw_support: np.ndarray,
    fg_mask: np.ndarray,
    baseline_image: np.ndarray,
    bottom_band_ratio: float,
    proxy_config: dict | None = None,
) -> dict[str, dict]:
    proxy_config = {**DEFAULT_PROXY_CONFIG, **(proxy_config or {})}
    fg_mask = np.asarray(fg_mask, dtype=bool)
    source_images = np.asarray(source_images, dtype=np.float32)
    source_raw_support = np.clip(np.asarray(source_raw_support, dtype=np.float32), 0.0, None)

    source_subset = [int(idx) for idx in proxy_config.get("source_subset", []) if 0 <= int(idx) < int(source_raw_support.shape[0])]
    if source_subset:
        source_images = source_images[source_subset]
        source_raw_support = source_raw_support[source_subset]

    gaussian_sigma = max(float(proxy_config.get("gaussian_sigma", 0.0)), 0.0)
    morph_close_k = int(proxy_config.get("morph_close_k", 0))
    label_majority_k = max(int(proxy_config.get("label_majority_k", 5)), 1)
    label_smooth_mix = float(proxy_config.get("label_smooth_mix", 0.45))
    consensus_margin_floor = float(proxy_config.get("consensus_margin_floor", 0.05))
    coverage_floor_ratio = float(proxy_config.get("coverage_floor_ratio", 0.70))
    coverage_floor_mix = float(proxy_config.get("coverage_floor_mix", 0.35))
    alpha_floor = float(proxy_config.get("alpha_floor", 0.15))
    support_gate_pow = float(proxy_config.get("support_gate_pow", 1.0))
    baseline_blend = float(proxy_config.get("baseline_blend", 0.30))
    render_mode = str(proxy_config.get("render_mode", "rehydrated")).strip().lower() or "rehydrated"

    total_support = source_raw_support.sum(axis=0)
    denom = np.clip(total_support[None, ...], 1e-8, None)
    probs = source_raw_support / denom
    sharp_probs = np.power(probs, 4.0)
    sharp_probs = sharp_probs / np.clip(sharp_probs.sum(axis=0, keepdims=True), 1e-8, None)
    sorted_probs = np.sort(probs, axis=0)
    top1 = sorted_probs[-1]
    top2 = sorted_probs[-2] if probs.shape[0] > 1 else np.zeros_like(top1)
    margin = np.clip(top1 - top2, 0.0, 1.0)
    margin_gate = np.clip((margin - consensus_margin_floor) / 0.25, 0.0, 1.0)
    margin_gate = _smooth_gate_map(margin_gate, gaussian_sigma=gaussian_sigma, morph_close_k=morph_close_k)
    bottom_band = _build_bottom_band_mask(fg_mask, bottom_band_ratio).astype(np.float32)

    agreement_maps = _source_agreement_maps(source_raw_support, source_images, fg_mask)
    medoid_probs = agreement_maps["medoid_probs"]
    medoid_support_map = np.asarray(agreement_maps["medoid_support_map"], dtype=np.float32)
    medoid_margin_map = np.asarray(agreement_maps["medoid_margin_map"], dtype=np.float32)
    medoid_label_smoothness = np.asarray(agreement_maps["label_smoothness_map"], dtype=np.float32)
    smoothed_label_prior = _smooth_source_label_map(
        np.asarray(agreement_maps["medoid_label_map"], dtype=np.int32),
        fg_mask,
        source_raw_support.shape[0],
        kernel_size=label_majority_k,
    ).astype(np.float32)

    def _compose_variant(support_gate: np.ndarray, source_gate: np.ndarray) -> dict:
        support_gate = np.clip(np.asarray(support_gate, dtype=np.float32), 0.0, 1.0) * fg_mask.astype(np.float32)
        support_gate = _smooth_gate_map(
            support_gate,
            gaussian_sigma=gaussian_sigma,
            morph_close_k=morph_close_k,
        )
        support_gate = np.power(np.clip(support_gate, 0.0, 1.0), max(support_gate_pow, 1e-6)).astype(np.float32)
        source_gate = np.clip(np.asarray(source_gate, dtype=np.float32), 0.0, None)
        if source_gate.ndim == 2:
            source_gate = source_gate[None, ...]
        alpha_gate = np.clip(alpha_floor + (1.0 - alpha_floor) * support_gate, 0.0, 1.0).astype(np.float32)
        alpha_gate = np.where(fg_mask, alpha_gate, 0.0).astype(np.float32)
        gated_source_support = source_raw_support * source_gate * support_gate[None, ...]
        proxy_total_support = gated_source_support.sum(axis=0)
        proxy_support01 = _normalize_support_weight(proxy_total_support)
        proxy_probs = gated_source_support / np.clip(proxy_total_support[None, ...], 1e-8, None)
        proxy_rgb = np.clip((proxy_probs[..., None] * source_images).sum(axis=0), 0.0, 1.0).astype(np.float32)
        if render_mode == "alpha_only":
            proxy_image = np.clip(proxy_rgb * alpha_gate[..., None], 0.0, 1.0).astype(np.float32)
        else:
            effective_alpha = np.clip(alpha_gate * (1.0 - baseline_blend), 0.0, 1.0).astype(np.float32)
            proxy_image = np.clip(
                baseline_image * (1.0 - effective_alpha[..., None]) + proxy_rgb * effective_alpha[..., None],
                0.0,
                1.0,
            ).astype(np.float32)
        coverage_mask = fg_mask & _visible_rgb_mask(proxy_image)
        return {
            "image": proxy_image,
            "support_weight": proxy_support01,
            "raw_support_weight": proxy_total_support.astype(np.float32),
            "source_support": gated_source_support.astype(np.float32),
            "source_subset": list(source_subset),
            "alpha_map": alpha_gate.astype(np.float32),
            "proxy_rgb": proxy_rgb.astype(np.float32),
            "coverage_mask": coverage_mask.astype(bool),
            "render_mode": render_mode,
        }

    legacy_variants = {
        "soft_top1_inside_fg": _compose_variant(
            support_gate=top1,
            source_gate=sharp_probs,
        ),
        "soft_top1_margin_inside_fg": _compose_variant(
            support_gate=top1 * margin_gate,
            source_gate=sharp_probs * margin_gate[None, ...],
        ),
        "soft_top1_margin_plus_bottom_suppress": _compose_variant(
            support_gate=top1 * margin_gate * (1.0 - 0.25 * bottom_band),
            source_gate=sharp_probs * (margin_gate * (1.0 - 0.25 * bottom_band))[None, ...],
        ),
    }
    legacy_fg_lcc = legacy_variants["soft_top1_margin_inside_fg"]
    legacy_support_mask = (legacy_fg_lcc["support_weight"] >= 0.35) & fg_mask
    component_count, labels = cv2.connectedComponents(legacy_support_mask.astype(np.uint8), connectivity=8)
    if component_count > 1:
        areas = [int((labels == label).sum()) for label in range(1, component_count)]
        keep_mask = (labels == (1 + int(np.argmax(areas)))).astype(np.float32)
    else:
        keep_mask = np.zeros_like(legacy_fg_lcc["support_weight"], dtype=np.float32)
    legacy_variants["soft_top1_margin_plus_fg_lcc_proxy"] = _compose_variant(
        support_gate=top1 * margin_gate * keep_mask,
        source_gate=sharp_probs * (margin_gate * keep_mask)[None, ...],
    )

    consensus_gate = np.clip(0.35 + 0.65 * medoid_support_map, 0.0, 1.0) * fg_mask.astype(np.float32)
    consensus_gate = _smooth_gate_map(
        consensus_gate,
        gaussian_sigma=gaussian_sigma,
        morph_close_k=morph_close_k,
    )
    medoid_margin_gate = np.clip(
        0.55 + 0.45 * np.clip((medoid_margin_map - consensus_margin_floor) / 0.25, 0.0, 1.0),
        0.0,
        1.0,
    )
    medoid_margin_gate = _smooth_gate_map(
        medoid_margin_gate,
        gaussian_sigma=gaussian_sigma,
        morph_close_k=morph_close_k,
    )

    consensus_medoid = _compose_variant(
        support_gate=consensus_gate,
        source_gate=medoid_probs,
    )
    consensus_margin = _compose_variant(
        support_gate=consensus_gate * medoid_margin_gate,
        source_gate=medoid_probs * medoid_margin_gate[None, ...],
    )
    consensus_smooth = _compose_variant(
        support_gate=consensus_gate * np.clip(0.45 + 0.55 * medoid_label_smoothness, 0.0, 1.0),
        source_gate=((1.0 - label_smooth_mix) * medoid_probs + label_smooth_mix * smoothed_label_prior),
    )

    current_coverage = float(
        ((consensus_margin["support_weight"] >= 0.12) & fg_mask).sum() / max(int(fg_mask.sum()), 1)
    )
    if current_coverage < coverage_floor_ratio:
        deficit = float(np.clip((coverage_floor_ratio - current_coverage) / max(coverage_floor_ratio, 1e-6), 0.0, 1.0))
        mixed_gate = np.clip((1.0 - deficit * coverage_floor_mix) * (consensus_gate * medoid_margin_gate) + deficit * coverage_floor_mix * consensus_gate, 0.0, 1.0)
        mixed_source = np.clip((1.0 - deficit * coverage_floor_mix) * (medoid_probs * medoid_margin_gate[None, ...]) + deficit * coverage_floor_mix * medoid_probs, 0.0, None)
    else:
        mixed_gate = consensus_gate * medoid_margin_gate
        mixed_source = medoid_probs * medoid_margin_gate[None, ...]
    consensus_floor = _compose_variant(
        support_gate=mixed_gate,
        source_gate=mixed_source,
    )

    return {
        **legacy_variants,
        "consensus_medoid_inside_fg": consensus_medoid,
        "consensus_margin_inside_fg": consensus_margin,
        "consensus_label_smooth_inside_fg": consensus_smooth,
        "consensus_margin_plus_coverage_floor": consensus_floor,
    }


def _compute_support_metrics(
    weight01: np.ndarray,
    fg_mask: np.ndarray,
    image01: np.ndarray,
    *,
    support_threshold: float,
    bottom_band_ratio: float,
    raw_support_weight: np.ndarray | None = None,
    alpha_map: np.ndarray | None = None,
    coverage_mask: np.ndarray | None = None,
    source_support: np.ndarray | None = None,
    source_images: np.ndarray | None = None,
    baseline_support_weight: np.ndarray | None = None,
    baseline_raw_support_weight: np.ndarray | None = None,
    baseline_alpha_map: np.ndarray | None = None,
    baseline_coverage_mask: np.ndarray | None = None,
    baseline_image01: np.ndarray | None = None,
) -> dict:
    weight01 = np.clip(np.asarray(weight01, dtype=np.float32), 0.0, 1.0)
    fg_mask = np.asarray(fg_mask, dtype=bool)
    raw_support_weight = np.clip(np.asarray(raw_support_weight if raw_support_weight is not None else weight01, dtype=np.float32), 0.0, None)
    alpha_map = np.clip(np.asarray(alpha_map if alpha_map is not None else weight01, dtype=np.float32), 0.0, 1.0)
    outside_fg = ~fg_mask
    bottom_band = _build_bottom_band_mask(fg_mask, bottom_band_ratio)
    bg_bottom = outside_fg & bottom_band
    total_weight = float(weight01.sum())
    denom = max(total_weight, 1e-8)
    high_support = weight01 >= float(support_threshold)
    mean_rgb = np.asarray(image01, dtype=np.float32).mean(axis=2)
    visible_rgb_mask = _visible_rgb_mask(image01)
    nonblack_mask = visible_rgb_mask
    render_coverage_mask = fg_mask & nonblack_mask
    rehydrated_coverage_mask = np.asarray(
        coverage_mask if coverage_mask is not None else render_coverage_mask,
        dtype=bool,
    )
    outside_pixels = int(outside_fg.sum())
    bg_bottom_pixels = int(bg_bottom.sum())
    fg_support_binary = (high_support & fg_mask).astype(np.uint8)
    fg_connected_components, compactness = _connected_component_ratio(fg_support_binary)
    fg_support_mass = weight01 * fg_mask.astype(np.float32)
    row_profile = fg_support_mass.sum(axis=1)
    col_profile = fg_support_mass.sum(axis=0)
    fg_peak_count_y = _count_profile_peaks(row_profile)
    fg_peak_count_x = _count_profile_peaks(col_profile)
    fg_mask_coverage_ratio = float(render_coverage_mask.sum() / max(int(fg_mask.sum()), 1))
    fg_alpha_coverage_ratio = float((fg_mask & (alpha_map >= 0.05)).sum() / max(int(fg_mask.sum()), 1))
    fg_rehydrated_coverage_ratio = float(rehydrated_coverage_mask.sum() / max(int(fg_mask.sum()), 1))
    fg_visible_rgb_coverage_ratio = fg_rehydrated_coverage_ratio
    fg_bbox_cover_ratio = 0.0
    fg_bbox_visible_cover_ratio = 0.0
    if int(fg_mask.sum()) > 0:
        ys, xs = np.where(fg_mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        fg_bbox = np.zeros_like(fg_mask, dtype=bool)
        fg_bbox[y0:y1, x0:x1] = True
        fg_bbox_cover_ratio = float((fg_bbox & rehydrated_coverage_mask).sum() / max(int(fg_bbox.sum()), 1))
        fg_bbox_visible_cover_ratio = fg_bbox_cover_ratio
    _cover_components, largest_fg_component_cover_ratio = _connected_component_ratio(rehydrated_coverage_mask.astype(np.uint8))
    largest_fg_visible_component_ratio = float(largest_fg_component_cover_ratio)
    fg_support_mass_coverage_ratio = float(
        raw_support_weight[fg_mask].sum() / max(float(raw_support_weight.max()) * max(int(fg_mask.sum()), 1), 1e-8)
    ) if int(fg_mask.sum()) > 0 else 0.0
    fg_visible_mass_ratio = float(
        np.asarray(image01, dtype=np.float32)[fg_mask].sum() / max(float(np.asarray(image01, dtype=np.float32).max()) * 3.0 * max(int(fg_mask.sum()), 1), 1e-8)
    ) if int(fg_mask.sum()) > 0 else 0.0
    fg_support_visible_overlap_ratio = float(
        ((high_support & rehydrated_coverage_mask).sum()) / max(int((high_support & fg_mask).sum()), 1)
    )
    if baseline_support_weight is None:
        fg_retained_support_area_ratio = 1.0
        fg_retained_mass_ratio = 1.0
    else:
        baseline_support_weight = np.asarray(baseline_support_weight, dtype=np.float32)
        baseline_raw_support_weight = np.clip(
            np.asarray(baseline_raw_support_weight if baseline_raw_support_weight is not None else baseline_support_weight, dtype=np.float32),
            0.0,
            None,
        )
        baseline_high_support_fg = (baseline_support_weight >= float(support_threshold)) & fg_mask
        fg_retained_support_area_ratio = float(
            (high_support & fg_mask).sum() / max(int(baseline_high_support_fg.sum()), 1)
        )
        fg_retained_mass_ratio = float(
            raw_support_weight[fg_mask].sum() / max(float(baseline_raw_support_weight[fg_mask].sum()), 1e-8)
        )
    if baseline_image01 is None:
        fg_retained_area_ratio = 1.0
    else:
        baseline_mean_rgb = np.asarray(baseline_image01, dtype=np.float32).mean(axis=2)
        baseline_nonblack_fg = fg_mask & _visible_rgb_mask(baseline_image01)
        baseline_alpha_map = np.clip(
            np.asarray(baseline_alpha_map if baseline_alpha_map is not None else baseline_support_weight, dtype=np.float32),
            0.0,
            1.0,
        )
        baseline_rehydrated_coverage_mask = np.asarray(
            baseline_coverage_mask if baseline_coverage_mask is not None else baseline_nonblack_fg,
            dtype=bool,
        )
        fg_retained_area_ratio = float(rehydrated_coverage_mask.sum() / max(int(baseline_rehydrated_coverage_mask.sum()), 1))
    metric_truth_bug = any(
        value > 1.001
        for value in [
            fg_mask_coverage_ratio,
            fg_alpha_coverage_ratio,
            fg_rehydrated_coverage_ratio,
            fg_bbox_cover_ratio,
            fg_retained_area_ratio,
            fg_retained_support_area_ratio,
            fg_retained_mass_ratio,
            fg_support_visible_overlap_ratio,
            fg_visible_mass_ratio,
            largest_fg_visible_component_ratio,
        ]
    )
    fg_mask_coverage_ratio = float(np.clip(fg_mask_coverage_ratio, 0.0, 1.0))
    fg_alpha_coverage_ratio = float(np.clip(fg_alpha_coverage_ratio, 0.0, 1.0))
    fg_rehydrated_coverage_ratio = float(np.clip(fg_rehydrated_coverage_ratio, 0.0, 1.0))
    fg_visible_rgb_coverage_ratio = float(np.clip(fg_visible_rgb_coverage_ratio, 0.0, 1.0))
    fg_bbox_cover_ratio = float(np.clip(fg_bbox_cover_ratio, 0.0, 1.0))
    fg_bbox_visible_cover_ratio = float(np.clip(fg_bbox_visible_cover_ratio, 0.0, 1.0))
    fg_retained_area_ratio = float(np.clip(fg_retained_area_ratio, 0.0, 1.0))
    fg_retained_support_area_ratio = float(np.clip(fg_retained_support_area_ratio, 0.0, 1.0))
    fg_retained_mass_ratio = float(np.clip(fg_retained_mass_ratio, 0.0, 1.0))
    fg_support_mass_coverage_ratio = float(np.clip(fg_support_mass_coverage_ratio, 0.0, 1.0))
    fg_visible_mass_ratio = float(np.clip(fg_visible_mass_ratio, 0.0, 1.0))
    fg_support_visible_overlap_ratio = float(np.clip(fg_support_visible_overlap_ratio, 0.0, 1.0))
    largest_fg_visible_component_ratio = float(np.clip(largest_fg_visible_component_ratio, 0.0, 1.0))
    human_erasure_penalty = float(
        max(0.0, 0.72 - fg_visible_rgb_coverage_ratio)
        + max(0.0, 0.70 - fg_bbox_visible_cover_ratio)
        + max(0.0, 0.55 - largest_fg_visible_component_ratio)
        + max(0.0, 0.60 - fg_visible_mass_ratio)
    )
    payload = {
        "support_threshold": float(support_threshold),
        "bottom_band_ratio": float(bottom_band_ratio),
        "support_inside_fg_ratio": float(weight01[fg_mask].sum() / denom),
        "off_body_support_ratio": float(weight01[outside_fg].sum() / denom),
        "off_body_nonblack_ratio": float((outside_fg & nonblack_mask).sum() / max(outside_pixels, 1)),
        "bg_bottom_support_ratio": float(weight01[bg_bottom].sum() / denom),
        "high_support_inside_fg_ratio": float(high_support[fg_mask].sum() / max(int(high_support.sum()), 1)),
        "high_support_outside_fg_ratio": float(high_support[outside_fg].sum() / max(int(high_support.sum()), 1)),
        "bg_nonblack_intensity": float(mean_rgb[outside_fg].mean()) if outside_pixels > 0 else 0.0,
        "bg_bottom_nonblack_intensity": float(mean_rgb[bg_bottom].mean()) if bg_bottom_pixels > 0 else 0.0,
        "fg_peak_count_x": fg_peak_count_x,
        "fg_peak_count_y": fg_peak_count_y,
        "fg_peak_count": int(max(fg_peak_count_x, fg_peak_count_y)),
        "fg_compactness": compactness,
        "fg_connected_components": fg_connected_components,
        "fg_mask_coverage_ratio": fg_mask_coverage_ratio,
        "fg_alpha_coverage_ratio": fg_alpha_coverage_ratio,
        "fg_rehydrated_coverage_ratio": fg_rehydrated_coverage_ratio,
        "fg_support_mass_coverage_ratio": fg_support_mass_coverage_ratio,
        "fg_visible_rgb_coverage_ratio": fg_visible_rgb_coverage_ratio,
        "fg_support_visible_overlap_ratio": fg_support_visible_overlap_ratio,
        "fg_bbox_visible_cover_ratio": fg_bbox_visible_cover_ratio,
        "largest_fg_visible_component_ratio": largest_fg_visible_component_ratio,
        "fg_visible_mass_ratio": fg_visible_mass_ratio,
        "fg_bbox_cover_ratio": fg_bbox_cover_ratio,
        "fg_retained_area_ratio": fg_retained_area_ratio,
        "fg_retained_support_area_ratio": fg_retained_support_area_ratio,
        "fg_retained_mass_ratio": fg_retained_mass_ratio,
        "human_erasure_penalty": human_erasure_penalty,
        "metric_truth_bug": bool(metric_truth_bug),
        "total_support_mass": total_weight,
    }
    if source_support is not None and source_images is not None:
        source_stats = _source_dominance_statistics(
            source_support,
            source_images,
            fg_mask,
            support_threshold=float(support_threshold),
        )
        payload.update(
            {
                "source_entropy_inside_fg": float(source_stats["source_entropy_inside_fg"]),
                "source_top1_mass_ratio_inside_fg": float(source_stats["source_top1_mass_ratio_inside_fg"]),
                "source_top1_top2_margin_inside_fg": float(source_stats["source_top1_top2_margin_inside_fg"]),
                "source_medoid_support_ratio_inside_fg": float(source_stats["source_medoid_support_ratio_inside_fg"]),
                "correspondence_consensus_ratio_inside_fg": float(source_stats["correspondence_consensus_ratio_inside_fg"]),
                "source_label_smoothness_inside_fg": float(source_stats["source_label_smoothness_inside_fg"]),
                "source_id_switch_count_inside_fg": float(source_stats["source_id_switch_count_inside_fg"]),
                "source_top1_spatial_fragmentation": float(source_stats["source_top1_spatial_fragmentation"]),
                "fg_largest_component_ratio": float(source_stats["fg_largest_component_ratio"]),
                "largest_fg_component_cover_ratio": float(largest_fg_component_cover_ratio),
            }
        )
    else:
        payload.update(
            {
                "source_entropy_inside_fg": 1.0,
                "source_top1_mass_ratio_inside_fg": 0.0,
                "source_top1_top2_margin_inside_fg": 0.0,
                "source_medoid_support_ratio_inside_fg": 0.0,
                "correspondence_consensus_ratio_inside_fg": 0.0,
                "source_label_smoothness_inside_fg": 0.0,
                "source_id_switch_count_inside_fg": 0.0,
                "source_top1_spatial_fragmentation": 0.0,
                "fg_largest_component_ratio": compactness,
                "largest_fg_component_cover_ratio": float(largest_fg_component_cover_ratio),
            }
        )
    return payload


def _save_support_visuals(
    renders_dir: Path,
    target01: np.ndarray,
    fg_mask: np.ndarray,
    variant_artifacts: dict[str, dict],
    *,
    support_threshold: float,
    bottom_band_ratio: float,
) -> dict:
    fg_mask = np.asarray(fg_mask, dtype=bool)
    outside_fg = ~fg_mask
    bottom_band = _build_bottom_band_mask(fg_mask, bottom_band_ratio)
    bg_bottom = outside_fg & bottom_band
    target_u8 = to_uint8(target01)

    def _support_overlay(support_weight: np.ndarray) -> np.ndarray:
        high_support = support_weight >= float(support_threshold)
        overlay = target_u8.astype(np.float32)
        overlay[fg_mask] = 0.65 * overlay[fg_mask] + 0.35 * np.array([64, 224, 96], dtype=np.float32)
        overlay[outside_fg & high_support] = 0.45 * overlay[outside_fg & high_support] + 0.55 * np.array([255, 72, 72], dtype=np.float32)
        overlay[bg_bottom & high_support] = 0.35 * overlay[bg_bottom & high_support] + 0.65 * np.array([255, 160, 48], dtype=np.float32)
        return np.clip(overlay, 0.0, 255.0).astype(np.uint8)

    def _coverage_overlay(coverage_mask: np.ndarray) -> np.ndarray:
        covered_fg = fg_mask & np.asarray(coverage_mask, dtype=bool)
        missing_fg = fg_mask & ~covered_fg
        overlay = target_u8.astype(np.float32)
        overlay[covered_fg] = 0.55 * overlay[covered_fg] + 0.45 * np.array([48, 224, 96], dtype=np.float32)
        overlay[missing_fg] = 0.45 * overlay[missing_fg] + 0.55 * np.array([255, 96, 64], dtype=np.float32)
        overlay[outside_fg] = 0.65 * overlay[outside_fg]
        return np.clip(overlay, 0.0, 255.0).astype(np.uint8)

    def _label_map_rgb(label_map: np.ndarray) -> np.ndarray:
        label_map = np.asarray(label_map, dtype=np.float32)
        return _apply_heatmap(label_map / max(float(label_map.max()), 1.0)).astype(np.float32) / 255.0

    baseline_artifact = variant_artifacts["baseline_depth_unproject"]
    baseline_weight01 = np.asarray(baseline_artifact["support_weight"], dtype=np.float32)
    baseline_inside = np.where(fg_mask, baseline_weight01, 0.0)
    baseline_outside = np.where(outside_fg, baseline_weight01, 0.0)
    baseline_bottom = np.where(bg_bottom, baseline_weight01, 0.0)
    _save_rgb(renders_dir / "support_inside_fg.png", _apply_heatmap(baseline_inside).astype(np.float32) / 255.0)
    _save_rgb(renders_dir / "support_outside_fg.png", _apply_heatmap(baseline_outside).astype(np.float32) / 255.0)
    _save_rgb(renders_dir / "support_overlay_on_fg.png", _support_overlay(baseline_weight01).astype(np.float32) / 255.0)
    _save_rgb(renders_dir / "bg_bottom_support.png", _apply_heatmap(baseline_bottom).astype(np.float32) / 255.0)
    _save_rgb(
        renders_dir / "fg_coverage_overlay.png",
        _coverage_overlay(np.asarray(baseline_artifact.get("coverage_mask", fg_mask & ((baseline_artifact["support_weight"] >= 0.05) | (np.asarray(baseline_artifact["image"], dtype=np.float32).mean(axis=2) > 0.05))), dtype=bool)).astype(np.float32) / 255.0,
    )
    _save_rgb(
        renders_dir / "source_label_map.png",
        _label_map_rgb(baseline_artifact.get("source_medoid_index_map", baseline_artifact.get("source_top1_index_map", np.zeros_like(baseline_weight01)))),
    )

    per_variant = {}
    for variant_name, artifact in variant_artifacts.items():
        image01 = np.asarray(artifact["image"], dtype=np.float32)
        support_weight = np.asarray(artifact["support_weight"], dtype=np.float32)
        alpha_map = np.asarray(artifact.get("alpha_map", support_weight), dtype=np.float32)
        coverage_mask = np.asarray(
            artifact.get("coverage_mask", fg_mask & _visible_rgb_mask(image01)),
            dtype=bool,
        )
        inside_support = np.where(fg_mask, support_weight, 0.0)
        outside_support = np.where(outside_fg, support_weight, 0.0)
        bottom_support = np.where(bg_bottom, support_weight, 0.0)

        inside_path = renders_dir / f"support_inside_fg_{variant_name}.png"
        outside_path = renders_dir / f"support_outside_fg_{variant_name}.png"
        overlay_path = renders_dir / f"support_overlay_on_fg_{variant_name}.png"
        support_bottom_path = renders_dir / f"bg_bottom_support_{variant_name}.png"
        coverage_path = renders_dir / f"fg_coverage_overlay_{variant_name}.png"
        alpha_path = renders_dir / f"alpha_map_{variant_name}.png"
        _save_rgb(inside_path, _apply_heatmap(inside_support).astype(np.float32) / 255.0)
        _save_rgb(outside_path, _apply_heatmap(outside_support).astype(np.float32) / 255.0)
        _save_rgb(overlay_path, _support_overlay(support_weight).astype(np.float32) / 255.0)
        _save_rgb(support_bottom_path, _apply_heatmap(bottom_support).astype(np.float32) / 255.0)
        _save_rgb(coverage_path, _coverage_overlay(coverage_mask).astype(np.float32) / 255.0)
        _save_rgb(alpha_path, _apply_heatmap(alpha_map).astype(np.float32) / 255.0)

        mean_rgb = np.asarray(image01, dtype=np.float32).mean(axis=2)
        bg_nonblack = np.where(outside_fg, mean_rgb, 0.0)
        bg_bottom_nonblack = np.where(bg_bottom, mean_rgb, 0.0)
        bg_heat = _apply_heatmap(bg_nonblack / max(float(bg_nonblack.max()), 1e-8))
        bottom_heat = _apply_heatmap(bg_bottom_nonblack / max(float(bg_bottom_nonblack.max()), 1e-8))
        bg_path = renders_dir / f"bg_nonblack_heatmap_{variant_name}.png"
        bottom_path = renders_dir / f"bg_bottom_nonblack_heatmap_{variant_name}.png"
        _save_rgb(bg_path, bg_heat.astype(np.float32) / 255.0)
        _save_rgb(bottom_path, bottom_heat.astype(np.float32) / 255.0)

        source_top1_mass_path = renders_dir / f"source_top1_mass_{variant_name}.png"
        source_margin_path = renders_dir / f"source_top1_margin_{variant_name}.png"
        source_entropy_path = renders_dir / f"source_entropy_{variant_name}.png"
        source_index_path = renders_dir / f"source_top1_index_{variant_name}.png"
        source_medoid_path = renders_dir / f"source_medoid_support_{variant_name}.png"
        consensus_path = renders_dir / f"correspondence_consensus_{variant_name}.png"
        source_smooth_path = renders_dir / f"source_label_smoothness_{variant_name}.png"
        source_label_map_path = renders_dir / f"source_label_map_{variant_name}.png"
        top1_mass_map = artifact.get("source_top1_mass_map", np.zeros_like(support_weight, dtype=np.float32))
        source_margin_map = artifact.get("source_top1_top2_margin_map", np.zeros_like(support_weight, dtype=np.float32))
        source_entropy_map = artifact.get("source_entropy_map", np.zeros_like(support_weight, dtype=np.float32))
        source_index_map = artifact.get("source_top1_index_map", np.zeros_like(support_weight, dtype=np.float32))
        medoid_map = artifact.get("source_medoid_support_map", np.zeros_like(support_weight, dtype=np.float32))
        consensus_map = artifact.get("correspondence_consensus_map", np.zeros_like(support_weight, dtype=np.float32))
        source_smooth_map = artifact.get("source_label_smoothness_map", np.zeros_like(support_weight, dtype=np.float32))
        source_label_map = artifact.get("source_medoid_index_map", source_index_map)
        _save_rgb(source_top1_mass_path, _apply_heatmap(top1_mass_map).astype(np.float32) / 255.0)
        _save_rgb(source_margin_path, _apply_heatmap(source_margin_map).astype(np.float32) / 255.0)
        _save_rgb(source_entropy_path, _apply_heatmap(source_entropy_map).astype(np.float32) / 255.0)
        _save_rgb(source_index_path, _apply_heatmap(source_index_map / max(float(np.max(source_index_map)), 1.0)).astype(np.float32) / 255.0)
        _save_rgb(source_medoid_path, _apply_heatmap(medoid_map).astype(np.float32) / 255.0)
        _save_rgb(consensus_path, _apply_heatmap(consensus_map).astype(np.float32) / 255.0)
        _save_rgb(source_smooth_path, _apply_heatmap(source_smooth_map).astype(np.float32) / 255.0)
        _save_rgb(source_label_map_path, _label_map_rgb(source_label_map))
        per_variant[variant_name] = {
            "support_inside_fg_png": inside_path.name,
            "support_outside_fg_png": outside_path.name,
            "support_overlay_on_fg_png": overlay_path.name,
            "bg_bottom_support_png": support_bottom_path.name,
            "bg_nonblack_heatmap_png": bg_path.name,
            "bg_bottom_nonblack_heatmap_png": bottom_path.name,
            "fg_coverage_overlay_png": coverage_path.name,
            "alpha_map_png": alpha_path.name,
            "source_top1_mass_png": source_top1_mass_path.name,
            "source_top1_top2_margin_png": source_margin_path.name,
            "source_entropy_png": source_entropy_path.name,
            "source_top1_index_png": source_index_path.name,
            "source_medoid_support_png": source_medoid_path.name,
            "correspondence_consensus_png": consensus_path.name,
            "source_label_smoothness_png": source_smooth_path.name,
            "source_label_map_png": source_label_map_path.name,
        }

    return {
        "support_inside_fg_png": "support_inside_fg.png",
        "support_outside_fg_png": "support_outside_fg.png",
        "support_overlay_on_fg_png": "support_overlay_on_fg.png",
        "bg_bottom_support_png": "bg_bottom_support.png",
        "fg_coverage_overlay_png": "fg_coverage_overlay.png",
        "source_label_map_png": "source_label_map.png",
        "per_variant": per_variant,
    }


def _make_case_panel(output_path: Path, target01: np.ndarray, point01: np.ndarray, depth01: np.ndarray, weight01: np.ndarray, fg_mask: np.ndarray, variant_images: dict[str, np.ndarray]) -> None:
    panels = [
        ("Target", to_uint8(target01)),
        ("Point Map", to_uint8(point01)),
        ("Depth+Camera", to_uint8(depth01)),
        ("Depth Weight", np.repeat(np.expand_dims(np.clip(weight01 * 255.0, 0.0, 255.0).astype(np.uint8), -1), 3, axis=2)),
        ("FG Mask", np.repeat(np.expand_dims((fg_mask.astype(np.uint8) * 255), -1), 3, axis=2)),
    ]
    for name, image01 in variant_images.items():
        diff = np.abs(to_uint8(image01).astype(np.int16) - to_uint8(target01).astype(np.int16)).astype(np.uint8)
        panels.append((name, to_uint8(image01)))
        panels.append((f"{name} diff", diff))

    panel_width = max(image.shape[1] for _label, image in panels)
    panel_height = max(image.shape[0] for _label, image in panels)
    label_height = 22
    font = ImageFont.load_default()
    tiles = []
    for label, image in panels:
        canvas = Image.new("RGB", (panel_width, panel_height + label_height), color=(18, 18, 18))
        pil_image = Image.fromarray(image)
        canvas.paste(pil_image, ((panel_width - pil_image.width) // 2, label_height))
        ImageDraw.Draw(canvas).text((6, 4), label, fill=(255, 255, 255), font=font)
        tiles.append(canvas)

    columns = 3
    rows = []
    for index in range(0, len(tiles), columns):
        row_tiles = tiles[index : index + columns]
        row_width = sum(tile.width for tile in row_tiles)
        row_height = max(tile.height for tile in row_tiles)
        row_canvas = Image.new("RGB", (row_width, row_height), color=(12, 12, 12))
        cursor_x = 0
        for tile in row_tiles:
            row_canvas.paste(tile, (cursor_x, 0))
            cursor_x += tile.width
        rows.append(row_canvas)

    mosaic = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), color=(12, 12, 12))
    cursor_y = 0
    for row in rows:
        mosaic.paste(row, (0, cursor_y))
        cursor_y += row.height
    mosaic.save(output_path)


def _rank_variants(case_results: list[dict]) -> list[dict]:
    baseline_rows = {row["case_id"]: row for row in case_results if row["variant"] == "baseline_depth_unproject"}
    grouped: dict[str, list[dict]] = {}
    for row in case_results:
        grouped.setdefault(row["variant"], []).append(row)

    ranking = []
    for variant, rows in grouped.items():
        if variant == "baseline_depth_unproject":
            continue
        improvements = []
        for row in rows:
            baseline = baseline_rows[row["case_id"]]
            improvements.append(
                {
                    "case_id": row["case_id"],
                    "full_l1_delta": row["metrics"]["full"]["mae"] - baseline["metrics"]["full"]["mae"],
                    "full_ssim_delta": row["metrics"]["full"]["ssim"] - baseline["metrics"]["full"]["ssim"],
                    "masked_l1_delta": row["metrics"]["fg_masked"]["l1"] - baseline["metrics"]["fg_masked"]["l1"],
                    "masked_ssim_delta": row["metrics"]["fg_masked"]["ssim"] - baseline["metrics"]["fg_masked"]["ssim"],
                    "off_body_support_ratio_delta": row["support_metrics"]["off_body_support_ratio"] - baseline["support_metrics"]["off_body_support_ratio"],
                    "off_body_nonblack_ratio_delta": row["support_metrics"]["off_body_nonblack_ratio"] - baseline["support_metrics"]["off_body_nonblack_ratio"],
                    "bg_bottom_support_ratio_delta": row["support_metrics"]["bg_bottom_support_ratio"] - baseline["support_metrics"]["bg_bottom_support_ratio"],
                    "support_inside_fg_ratio_delta": row["support_metrics"]["support_inside_fg_ratio"] - baseline["support_metrics"]["support_inside_fg_ratio"],
                    "fg_peak_count_delta": row["support_metrics"]["fg_peak_count"] - baseline["support_metrics"]["fg_peak_count"],
                    "fg_connected_components_delta": row["support_metrics"]["fg_connected_components"] - baseline["support_metrics"]["fg_connected_components"],
                    "fg_compactness_delta": row["support_metrics"]["fg_compactness"] - baseline["support_metrics"]["fg_compactness"],
                    "fg_largest_component_ratio_delta": row["support_metrics"]["fg_largest_component_ratio"] - baseline["support_metrics"]["fg_largest_component_ratio"],
                    "fg_mask_coverage_ratio_delta": row["support_metrics"]["fg_mask_coverage_ratio"] - baseline["support_metrics"]["fg_mask_coverage_ratio"],
                    "fg_bbox_cover_ratio_delta": row["support_metrics"]["fg_bbox_cover_ratio"] - baseline["support_metrics"]["fg_bbox_cover_ratio"],
                    "fg_retained_area_ratio_delta": row["support_metrics"]["fg_retained_area_ratio"] - baseline["support_metrics"]["fg_retained_area_ratio"],
                    "fg_retained_support_area_ratio_delta": row["support_metrics"]["fg_retained_support_area_ratio"] - baseline["support_metrics"]["fg_retained_support_area_ratio"],
                    "fg_retained_mass_ratio_delta": row["support_metrics"]["fg_retained_mass_ratio"] - baseline["support_metrics"]["fg_retained_mass_ratio"],
                    "human_erasure_penalty_delta": row["support_metrics"]["human_erasure_penalty"] - baseline["support_metrics"]["human_erasure_penalty"],
                    "correspondence_consensus_ratio_inside_fg_delta": row["support_metrics"]["correspondence_consensus_ratio_inside_fg"] - baseline["support_metrics"]["correspondence_consensus_ratio_inside_fg"],
                    "source_medoid_support_ratio_inside_fg_delta": row["support_metrics"]["source_medoid_support_ratio_inside_fg"] - baseline["support_metrics"]["source_medoid_support_ratio_inside_fg"],
                    "source_label_smoothness_inside_fg_delta": row["support_metrics"]["source_label_smoothness_inside_fg"] - baseline["support_metrics"]["source_label_smoothness_inside_fg"],
                    "source_id_switch_count_inside_fg_delta": row["support_metrics"]["source_id_switch_count_inside_fg"] - baseline["support_metrics"]["source_id_switch_count_inside_fg"],
                    "source_top1_spatial_fragmentation_delta": row["support_metrics"]["source_top1_spatial_fragmentation"] - baseline["support_metrics"]["source_top1_spatial_fragmentation"],
                    "source_entropy_inside_fg_delta": row["support_metrics"]["source_entropy_inside_fg"] - baseline["support_metrics"]["source_entropy_inside_fg"],
                    "source_top1_mass_ratio_inside_fg_delta": row["support_metrics"]["source_top1_mass_ratio_inside_fg"] - baseline["support_metrics"]["source_top1_mass_ratio_inside_fg"],
                    "source_top1_top2_margin_inside_fg_delta": row["support_metrics"]["source_top1_top2_margin_inside_fg"] - baseline["support_metrics"]["source_top1_top2_margin_inside_fg"],
                }
            )
        ranking.append(
            {
                "variant": variant,
                "case_count": len(rows),
                "mean_full_l1_delta": float(np.mean([item["full_l1_delta"] for item in improvements])),
                "mean_full_ssim_delta": float(np.mean([item["full_ssim_delta"] for item in improvements])),
                "mean_masked_l1_delta": float(np.mean([item["masked_l1_delta"] for item in improvements])),
                "mean_masked_ssim_delta": float(np.mean([item["masked_ssim_delta"] for item in improvements])),
                "mean_off_body_support_ratio_delta": float(np.mean([item["off_body_support_ratio_delta"] for item in improvements])),
                "mean_off_body_nonblack_ratio_delta": float(np.mean([item["off_body_nonblack_ratio_delta"] for item in improvements])),
                "mean_bg_bottom_support_ratio_delta": float(np.mean([item["bg_bottom_support_ratio_delta"] for item in improvements])),
                "mean_support_inside_fg_ratio_delta": float(np.mean([item["support_inside_fg_ratio_delta"] for item in improvements])),
                "mean_fg_peak_count_delta": float(np.mean([item["fg_peak_count_delta"] for item in improvements])),
                "mean_fg_connected_components_delta": float(np.mean([item["fg_connected_components_delta"] for item in improvements])),
                "mean_fg_compactness_delta": float(np.mean([item["fg_compactness_delta"] for item in improvements])),
                "mean_fg_largest_component_ratio_delta": float(np.mean([item["fg_largest_component_ratio_delta"] for item in improvements])),
                "mean_fg_mask_coverage_ratio_delta": float(np.mean([item["fg_mask_coverage_ratio_delta"] for item in improvements])),
                "mean_fg_bbox_cover_ratio_delta": float(np.mean([item["fg_bbox_cover_ratio_delta"] for item in improvements])),
                "mean_fg_retained_area_ratio_delta": float(np.mean([item["fg_retained_area_ratio_delta"] for item in improvements])),
                "mean_fg_retained_support_area_ratio_delta": float(np.mean([item["fg_retained_support_area_ratio_delta"] for item in improvements])),
                "mean_fg_retained_mass_ratio_delta": float(np.mean([item["fg_retained_mass_ratio_delta"] for item in improvements])),
                "mean_human_erasure_penalty_delta": float(np.mean([item["human_erasure_penalty_delta"] for item in improvements])),
                "mean_correspondence_consensus_ratio_inside_fg_delta": float(np.mean([item["correspondence_consensus_ratio_inside_fg_delta"] for item in improvements])),
                "mean_source_medoid_support_ratio_inside_fg_delta": float(np.mean([item["source_medoid_support_ratio_inside_fg_delta"] for item in improvements])),
                "mean_source_label_smoothness_inside_fg_delta": float(np.mean([item["source_label_smoothness_inside_fg_delta"] for item in improvements])),
                "mean_source_id_switch_count_inside_fg_delta": float(np.mean([item["source_id_switch_count_inside_fg_delta"] for item in improvements])),
                "mean_source_top1_spatial_fragmentation_delta": float(np.mean([item["source_top1_spatial_fragmentation_delta"] for item in improvements])),
                "mean_source_entropy_inside_fg_delta": float(np.mean([item["source_entropy_inside_fg_delta"] for item in improvements])),
                "mean_source_top1_mass_ratio_inside_fg_delta": float(np.mean([item["source_top1_mass_ratio_inside_fg_delta"] for item in improvements])),
                "mean_source_top1_top2_margin_inside_fg_delta": float(np.mean([item["source_top1_top2_margin_inside_fg_delta"] for item in improvements])),
                "improved_full_count": int(sum(1 for item in improvements if item["full_l1_delta"] < 0.0 and item["full_ssim_delta"] > 0.0)),
                "improved_masked_count": int(sum(1 for item in improvements if item["masked_l1_delta"] < 0.0 and item["masked_ssim_delta"] > 0.0)),
                "case_improvements": improvements,
            }
        )
    ranking.sort(
        key=lambda row: (
            -row["improved_masked_count"],
            row["mean_masked_l1_delta"],
            -row["mean_masked_ssim_delta"],
            row["mean_full_l1_delta"],
            -row["mean_full_ssim_delta"],
        )
    )
    return ranking


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Teacher-Fixed Visual Lift Evaluation",
        "",
        f"- checked_at: `{payload['checked_at']}`",
        f"- case_set: `{payload['case_set']}`",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- case_count: `{payload['case_count']}`",
        f"- variants: `{', '.join(payload['variants'])}`",
        "",
        "## Variant Ranking",
        "",
        "| Variant | Cases | Improved Masked | Mean dL1(masked) | Mean dSSIM(masked) | Improved Full | Mean dL1(full) | Mean dSSIM(full) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["variant_ranking"]:
        lines.append(
            "| {variant} | {case_count} | {improved_masked_count} | {mean_masked_l1_delta:.6f} | {mean_masked_ssim_delta:.6f} | {improved_full_count} | {mean_full_l1_delta:.6f} | {mean_full_ssim_delta:.6f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _infer_case_id(case: dict) -> str:
    return "{seq}_frame_{frame:06d}_{target}".format(
        seq=str(case["seq_name"]),
        frame=int(case["frame_id"]),
        target=str(case["target_camera"]),
    )


def _evaluate_case(
    *,
    case: dict,
    model,
    device: torch.device,
    dtype: torch.dtype,
    zju_root: Path,
    preprocess_mode: str,
    render_hw: tuple[int, int],
    render_max_points: int,
    z_tolerance: float,
    min_conf: float,
    target_mask_source: str,
    variants: list[str],
    output_dir: Path,
    support_threshold: float,
    bottom_band_ratio: float,
    proxy_config: dict,
) -> list[dict]:
    seq_dir = zju_root / str(case["seq_name"])
    source_cameras = [str(camera) for camera in case["source_cameras"]]
    target_camera = str(case["target_camera"])
    frame_id = int(case["frame_id"])
    case_id = _infer_case_id(case)
    case_dir = output_dir / case_id
    renders_dir = case_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    source_image_paths = [resolve_frame_image_path(seq_dir, camera, frame_id) for camera in source_cameras]
    target_image_path = resolve_frame_image_path(seq_dir, target_camera, frame_id)
    images = load_and_preprocess_images([str(path) for path in source_image_paths], mode=preprocess_mode).to(device)
    source_colors = np.clip(images.detach().cpu().numpy().transpose(0, 2, 3, 1), 0.0, 1.0).astype(np.float32)
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=dtype)
        if device.type == "cuda" and dtype != torch.float32
        else torch.no_grad()
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
    centers_gt = camera_centers(np.stack([gt_cameras[camera]["extrinsic"] for camera in source_cameras], axis=0))
    scale, sim_rotation, sim_translation = umeyama_similarity(centers_pred, centers_gt)
    point_map_aligned = apply_sim3_points(point_map.reshape(-1, 3), scale, sim_rotation, sim_translation).reshape(point_map.shape)
    depth_points_aligned = apply_sim3_points(depth_points.reshape(-1, 3), scale, sim_rotation, sim_translation).reshape(depth_points.shape)

    target_image_full = np.asarray(Image.open(target_image_path).convert("RGB"), dtype=np.float32) / 255.0
    target_image = np.asarray(
        Image.fromarray(to_uint8(target_image_full)).resize((render_hw[1], render_hw[0]), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    target_intrinsic = scale_intrinsic(gt_cameras[target_camera]["intrinsic"], target_image_full.shape[:2], render_hw)
    target_extrinsic = gt_cameras[target_camera]["extrinsic"]
    fg_mask = _load_target_mask(seq_dir, target_camera, frame_id, render_hw, target_mask_source)

    point_render = _render_forward_splat_with_raw(
        world_points_s_hw3=point_map_aligned,
        world_conf_s_hw=point_conf,
        src_rgb_s_hw3=source_colors,
        tgt_extrinsic_3x4=target_extrinsic,
        tgt_intrinsic_3x3=target_intrinsic,
        out_hw=render_hw,
        z_eps=DEPTH_EPS,
        min_conf=min_conf,
        z_tolerance=z_tolerance,
        max_points=render_max_points,
    )
    depth_render = _render_forward_splat_with_raw(
        world_points_s_hw3=depth_points_aligned,
        world_conf_s_hw=depth_conf,
        src_rgb_s_hw3=source_colors,
        tgt_extrinsic_3x4=target_extrinsic,
        tgt_intrinsic_3x3=target_intrinsic,
        out_hw=render_hw,
        z_eps=DEPTH_EPS,
        min_conf=min_conf,
        z_tolerance=z_tolerance,
        max_points=render_max_points,
    )

    baseline_depth = _clamp_image(depth_render["image"])
    point_image = _clamp_image(point_render["image"])
    weight01 = np.asarray(depth_render["weight"], dtype=np.float32)
    depth_raw_weight = np.asarray(depth_render["raw_weight"], dtype=np.float32)

    source_render_images = []
    source_render_raw_weights = []
    for source_index in range(len(source_cameras)):
        source_render = _render_forward_splat_with_raw(
            world_points_s_hw3=depth_points_aligned[source_index : source_index + 1],
            world_conf_s_hw=depth_conf[source_index : source_index + 1],
            src_rgb_s_hw3=source_colors[source_index : source_index + 1],
            tgt_extrinsic_3x4=target_extrinsic,
            tgt_intrinsic_3x3=target_intrinsic,
            out_hw=render_hw,
            z_eps=DEPTH_EPS,
            min_conf=min_conf,
            z_tolerance=z_tolerance,
            max_points=render_max_points,
        )
        source_render_images.append(np.asarray(source_render["image"], dtype=np.float32))
        source_render_raw_weights.append(np.asarray(source_render["raw_weight"], dtype=np.float32))
    source_render_images = np.stack(source_render_images, axis=0).astype(np.float32)
    source_render_raw_weights = np.stack(source_render_raw_weights, axis=0).astype(np.float32)

    variant_artifacts = {
        "baseline_depth_unproject": {
            "image": baseline_depth,
            "support_weight": weight01,
            "raw_support_weight": depth_raw_weight,
            "source_support": source_render_raw_weights,
            "source_subset": list(range(len(source_cameras))),
            "alpha_map": weight01.astype(np.float32),
            "coverage_mask": fg_mask & _visible_rgb_mask(baseline_depth),
            "render_mode": "baseline",
        }
    }
    for variant_name in variants:
        if variant_name in PROXY_VARIANTS:
            continue
        candidate_image = _apply_variant(baseline_depth, weight01, fg_mask, variant_name)
        variant_artifacts[variant_name] = {
            "image": candidate_image,
            "support_weight": weight01,
            "raw_support_weight": depth_raw_weight,
            "source_support": source_render_raw_weights,
            "source_subset": list(range(len(source_cameras))),
            "alpha_map": weight01.astype(np.float32),
            "coverage_mask": fg_mask & _visible_rgb_mask(candidate_image),
            "render_mode": "visual_lift",
        }
    proxy_artifacts = _build_proxy_variant_artifacts(
        source_images=source_render_images,
        source_raw_support=source_render_raw_weights,
        fg_mask=fg_mask,
        baseline_image=baseline_depth,
        bottom_band_ratio=bottom_band_ratio,
        proxy_config=proxy_config,
    )
    for variant_name in variants:
        if variant_name in PROXY_VARIANTS:
            variant_artifacts[variant_name] = proxy_artifacts[variant_name]

    for artifact in variant_artifacts.values():
        subset = artifact.get("source_subset", [])
        if subset:
            stats_source_images = source_render_images[subset]
        else:
            stats_source_images = source_render_images
        source_stats = _source_dominance_statistics(
            artifact["source_support"],
            stats_source_images,
            fg_mask,
            support_threshold=float(support_threshold),
        )
        artifact.update(source_stats)

    _save_rgb(renders_dir / "target.png", target_image)
    _save_rgb(renders_dir / "point_map.png", point_image)
    _save_rgb(renders_dir / "depth_unproject.png", baseline_depth)
    _save_gray(renders_dir / "depth_weight.png", weight01)
    _save_gray(renders_dir / "fg_mask.png", fg_mask.astype(np.float32))
    for variant_name, artifact in variant_artifacts.items():
        _save_rgb(renders_dir / f"{variant_name}.png", artifact["image"])

    support_files = _save_support_visuals(
        renders_dir,
        target_image,
        fg_mask,
        variant_artifacts,
        support_threshold=support_threshold,
        bottom_band_ratio=bottom_band_ratio,
    )

    _make_case_panel(
        renders_dir / "comparison_panel.png",
        target_image,
        point_image,
        baseline_depth,
        weight01,
        fg_mask,
        {name: artifact["image"] for name, artifact in variant_artifacts.items()},
    )

    rows = []
    for variant_name, artifact in variant_artifacts.items():
        image01 = artifact["image"]
        subset = artifact.get("source_subset", [])
        if subset:
            metric_source_images = source_render_images[subset]
        else:
            metric_source_images = source_render_images
        support_metrics = _compute_support_metrics(
            artifact["support_weight"],
            fg_mask,
            image01,
            support_threshold=support_threshold,
            bottom_band_ratio=bottom_band_ratio,
            raw_support_weight=artifact.get("raw_support_weight"),
            alpha_map=artifact.get("alpha_map"),
            coverage_mask=artifact.get("coverage_mask"),
            source_support=artifact["source_support"],
            source_images=metric_source_images,
            baseline_support_weight=weight01,
            baseline_raw_support_weight=depth_raw_weight,
            baseline_alpha_map=variant_artifacts["baseline_depth_unproject"].get("alpha_map"),
            baseline_coverage_mask=variant_artifacts["baseline_depth_unproject"].get("coverage_mask"),
            baseline_image01=baseline_depth,
        )
        rows.append(
            {
                "case_id": case_id,
                "variant": variant_name,
                "case": {
                    "seq_name": str(case["seq_name"]),
                    "frame_id": frame_id,
                    "target_camera": target_camera,
                    "source_cameras": source_cameras,
                },
                "metrics": {
                    "full": full_frame_metrics(image01, target_image),
                    "fg_masked": _masked_metrics(image01, target_image, fg_mask),
                },
                "support_metrics": support_metrics,
                "files": {
                    "target_png": str((renders_dir / "target.png").relative_to(output_dir)),
                    "point_map_png": str((renders_dir / "point_map.png").relative_to(output_dir)),
                    "depth_unproject_png": str((renders_dir / "depth_unproject.png").relative_to(output_dir)),
                    "weight_png": str((renders_dir / "depth_weight.png").relative_to(output_dir)),
                    "fg_mask_png": str((renders_dir / "fg_mask.png").relative_to(output_dir)),
                    "support_inside_fg_png": str((renders_dir / support_files["per_variant"][variant_name]["support_inside_fg_png"]).relative_to(output_dir)),
                    "support_outside_fg_png": str((renders_dir / support_files["per_variant"][variant_name]["support_outside_fg_png"]).relative_to(output_dir)),
                    "support_overlay_on_fg_png": str((renders_dir / support_files["per_variant"][variant_name]["support_overlay_on_fg_png"]).relative_to(output_dir)),
                    "bg_bottom_support_png": str((renders_dir / support_files["per_variant"][variant_name]["bg_bottom_support_png"]).relative_to(output_dir)),
                    "bg_nonblack_heatmap_png": str((renders_dir / support_files["per_variant"][variant_name]["bg_nonblack_heatmap_png"]).relative_to(output_dir)),
                    "bg_bottom_nonblack_heatmap_png": str((renders_dir / support_files["per_variant"][variant_name]["bg_bottom_nonblack_heatmap_png"]).relative_to(output_dir)),
                    "fg_coverage_overlay_png": str((renders_dir / support_files["per_variant"][variant_name]["fg_coverage_overlay_png"]).relative_to(output_dir)),
                    "alpha_map_png": str((renders_dir / support_files["per_variant"][variant_name]["alpha_map_png"]).relative_to(output_dir)),
                    "source_top1_mass_png": str((renders_dir / support_files["per_variant"][variant_name]["source_top1_mass_png"]).relative_to(output_dir)),
                    "source_top1_top2_margin_png": str((renders_dir / support_files["per_variant"][variant_name]["source_top1_top2_margin_png"]).relative_to(output_dir)),
                    "source_entropy_png": str((renders_dir / support_files["per_variant"][variant_name]["source_entropy_png"]).relative_to(output_dir)),
                    "source_top1_index_png": str((renders_dir / support_files["per_variant"][variant_name]["source_top1_index_png"]).relative_to(output_dir)),
                    "source_medoid_support_png": str((renders_dir / support_files["per_variant"][variant_name]["source_medoid_support_png"]).relative_to(output_dir)),
                    "correspondence_consensus_png": str((renders_dir / support_files["per_variant"][variant_name]["correspondence_consensus_png"]).relative_to(output_dir)),
                    "source_label_smoothness_png": str((renders_dir / support_files["per_variant"][variant_name]["source_label_smoothness_png"]).relative_to(output_dir)),
                    "source_label_map_png": str((renders_dir / support_files["per_variant"][variant_name]["source_label_map_png"]).relative_to(output_dir)),
                    "variant_png": str((renders_dir / f"{variant_name}.png").relative_to(output_dir)),
                    "comparison_panel_png": str((renders_dir / "comparison_panel.png").relative_to(output_dir)),
                },
                "alignment": {
                    "scale": float(scale),
                    "src_center_rmse_after": float(rmse(apply_sim3_points(centers_pred, scale, sim_rotation, sim_translation), centers_gt)),
                },
                "baseline_render_stats": depth_render["stats"],
                "proxy_config": proxy_config,
            }
        )

    save_json(case_dir / "case_metrics.json", {"rows": rows})
    return rows


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)
    cases = _select_cases(manifest, args.case_set)
    if not cases:
        raise ValueError(f"No cases found for case_set={args.case_set}")

    variants = _resolve_variants(args.variants)
    proxy_config = _load_proxy_config(args.proxy_config_json)
    zju_root = Path(args.local_zju_root).resolve() if args.local_zju_root else detect_local_zju_root().resolve()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    render_hw = (int(args.render_size[0]), int(args.render_size[1]))
    model = load_model(args.checkpoint, device)

    all_rows = []
    for case_index, case in enumerate(cases, start=1):
        all_rows.extend(
            _evaluate_case(
                case=case,
                model=model,
                device=device,
                dtype=dtype,
                zju_root=zju_root,
                preprocess_mode=args.preprocess_mode,
                render_hw=render_hw,
                render_max_points=int(args.render_max_points),
                z_tolerance=float(args.z_tolerance),
                min_conf=float(args.min_conf),
                target_mask_source=args.target_mask_source,
                variants=variants,
                output_dir=output_dir,
                support_threshold=float(args.support_threshold),
                bottom_band_ratio=float(args.bottom_band_ratio),
                proxy_config=proxy_config,
            )
        )
        partial_payload = {
            "checked_at": datetime.now().astimezone().isoformat(),
            "manifest_path": str(manifest_path),
            "checkpoint": str(Path(args.checkpoint)),
            "case_set": args.case_set,
            "case_count": len(cases),
            "completed_case_count": int(case_index),
            "latest_case_id": _infer_case_id(case),
            "variants": ["baseline_depth_unproject"] + variants,
            "proxy_config": proxy_config,
            "rows": all_rows,
            "variant_ranking": _rank_variants(all_rows),
        }
        save_json(
            output_dir / "progress.json",
            {
                "checked_at": partial_payload["checked_at"],
                "case_set": args.case_set,
                "case_count": len(cases),
                "completed_case_count": int(case_index),
                "latest_case_id": _infer_case_id(case),
                "variants": partial_payload["variants"],
                "proxy_config": proxy_config,
            },
        )
        save_json(output_dir / "summary.partial.json", partial_payload)

    payload = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "manifest_path": str(manifest_path),
        "checkpoint": str(Path(args.checkpoint)),
        "case_set": args.case_set,
        "case_count": len(cases),
        "variants": ["baseline_depth_unproject"] + variants,
        "proxy_config": proxy_config,
        "rows": all_rows,
        "variant_ranking": _rank_variants(all_rows),
    }
    save_json(output_dir / "summary.json", payload)
    _write_markdown(output_dir / "summary.md", payload)
    print(output_dir / "summary.json")
    print(output_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
