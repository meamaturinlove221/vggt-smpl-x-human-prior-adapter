import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


CONF_EPS = 1e-6


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def to_uint8(image01):
    return np.clip(np.asarray(image01, dtype=np.float32) * 255.0, 0.0, 255.0).round().astype(np.uint8)


def load_target_mask(seq_dir, camera_name, frame_id, target_hw, mask_source="mask"):
    height, width = int(target_hw[0]), int(target_hw[1])
    if mask_source == "none":
        return np.ones((height, width), dtype=bool)
    mask_path = Path(seq_dir) / mask_source / camera_name / f"{int(frame_id):06d}.png"
    if not mask_path.is_file():
        raise FileNotFoundError(f"target mask not found: {mask_path}")
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def build_region_masks(fg_mask, edge_px=5, bottom_band_ratio=0.2):
    fg_mask = np.asarray(fg_mask, dtype=bool)
    height, width = fg_mask.shape
    edge_px = max(int(edge_px), 0)
    bottom_band_ratio = float(np.clip(bottom_band_ratio, 0.0, 1.0))

    if edge_px > 0:
        kernel = np.ones((edge_px * 2 + 1, edge_px * 2 + 1), dtype=np.uint8)
        fg_u8 = fg_mask.astype(np.uint8)
        eroded = cv2.erode(fg_u8, kernel, iterations=1) > 0
        dilated = cv2.dilate(fg_u8, kernel, iterations=1) > 0
    else:
        eroded = fg_mask.copy()
        dilated = fg_mask.copy()

    bottom_start = int(math.floor(height * (1.0 - bottom_band_ratio)))
    bottom_start = max(0, min(height, bottom_start))
    bottom_mask = np.zeros((height, width), dtype=bool)
    bottom_mask[bottom_start:, :] = True

    return {
        "fg_human": fg_mask,
        "fg_edge": dilated & (~eroded),
        "bg_far": ~dilated,
        "bg_bottom_band": (~dilated) & bottom_mask,
    }


def make_target_mask_overlay(output_path, target_image01, region_masks):
    base = to_uint8(target_image01).astype(np.float32)
    overlay = base.copy()
    palette = {
        "fg_human": (48, 196, 92),
        "fg_edge": (255, 214, 10),
        "bg_far": (60, 128, 255),
        "bg_bottom_band": (255, 86, 56),
    }
    alpha_map = {
        "fg_human": 0.18,
        "fg_edge": 0.55,
        "bg_far": 0.12,
        "bg_bottom_band": 0.38,
    }
    for region_name, region_mask in region_masks.items():
        mask = np.asarray(region_mask, dtype=bool)
        color = np.asarray(palette[region_name], dtype=np.float32)
        alpha = float(alpha_map[region_name])
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha

    canvas = Image.fromarray(np.clip(overlay, 0.0, 255.0).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    cursor_y = 8
    for region_name in ("fg_human", "fg_edge", "bg_far", "bg_bottom_band"):
        color = palette[region_name]
        draw.rectangle((8, cursor_y, 22, cursor_y + 12), fill=color)
        draw.text((28, cursor_y - 1), region_name, fill=(255, 255, 255), font=font)
        cursor_y += 16
    canvas.save(output_path)


def gaussian_kernel(channels, device, dtype, ksize=11, sigma=1.5):
    coords = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    g2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    return g2d.repeat(channels, 1, 1, 1)


def masked_metrics(pred01, tgt01, mask_hw=None):
    pred = torch.from_numpy(np.asarray(pred01, dtype=np.float32).transpose(2, 0, 1)).unsqueeze(0).float()
    tgt = torch.from_numpy(np.asarray(tgt01, dtype=np.float32).transpose(2, 0, 1)).unsqueeze(0).float()

    if mask_hw is None:
        mask = torch.ones((1, 1, pred.shape[-2], pred.shape[-1]), dtype=torch.bool)
    else:
        mask_np = np.asarray(mask_hw, dtype=bool)
        if mask_np.shape != tuple(pred.shape[-2:]):
            raise ValueError(f"Mask shape {mask_np.shape} does not match image shape {tuple(pred.shape[-2:])}")
        mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).bool()

    pixel_count = int(mask.sum().item())
    if pixel_count == 0:
        return {"mae": None, "psnr": None, "ssim": None, "pixel_count": 0}

    abs_diff = (pred - tgt).abs().mean(dim=1, keepdim=True)
    mae = float(abs_diff[mask].mean().item())

    sq_diff = ((pred - tgt) ** 2).mean(dim=1, keepdim=True)
    mse_value = float(sq_diff[mask].mean().item())
    psnr = float(-10.0 * math.log10(max(mse_value, 1e-8)))

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
    ssim_scalar = float(ssim_map.mean(dim=1, keepdim=True)[mask].mean().item())

    return {
        "mae": mae,
        "psnr": psnr,
        "ssim": ssim_scalar,
        "pixel_count": pixel_count,
    }


def write_binary_ply(path, points, colors):
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2:
        points = points.reshape(-1, 3)
    if colors.ndim != 2:
        colors = colors.reshape(-1, 3)
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
    if points.shape[0] > 0:
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
        if points.shape[0] > 0:
            vertex_data.tofile(handle)


def project_points_to_target(
    *,
    world_points_s_hw3,
    world_conf_s_hw,
    src_rgb_s_hw3,
    tgt_extrinsic_3x4,
    tgt_intrinsic_3x3,
    out_hw,
    min_conf,
):
    height, width = int(out_hw[0]), int(out_hw[1])
    points = np.asarray(world_points_s_hw3, dtype=np.float64).reshape(-1, 3)
    conf = np.asarray(world_conf_s_hw, dtype=np.float64).reshape(-1)
    colors = np.asarray(src_rgb_s_hw3, dtype=np.float32).reshape(-1, 3)

    finite = np.isfinite(points).all(axis=-1) & np.isfinite(conf) & (conf >= float(min_conf))
    candidate_indices = np.flatnonzero(finite)
    candidate_points = int(candidate_indices.size)
    if candidate_points == 0:
        return {
            "total_points": int(points.shape[0]),
            "candidate_points": 0,
            "projected_points": 0,
            "points": np.zeros((0, 3), dtype=np.float32),
            "conf": np.zeros((0,), dtype=np.float32),
            "colors": np.zeros((0, 3), dtype=np.float32),
            "xi": np.zeros((0,), dtype=np.int64),
            "yi": np.zeros((0,), dtype=np.int64),
            "z": np.zeros((0,), dtype=np.float32),
        }

    points = points[candidate_indices]
    conf = conf[candidate_indices]
    colors = colors[candidate_indices]

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
    valid &= z > CONF_EPS
    valid &= xi >= 0
    valid &= yi >= 0
    valid &= xi < width
    valid &= yi < height

    return {
        "total_points": int(np.asarray(world_points_s_hw3).reshape(-1, 3).shape[0]),
        "candidate_points": candidate_points,
        "projected_points": int(valid.sum()),
        "points": np.asarray(points[valid], dtype=np.float32),
        "conf": np.asarray(conf[valid], dtype=np.float32),
        "colors": np.asarray(colors[valid], dtype=np.float32),
        "xi": xi[valid],
        "yi": yi[valid],
        "z": np.asarray(z[valid], dtype=np.float32),
    }


def export_projected_region_ply(output_path, projected, region_mask, max_points=250000):
    mask = np.asarray(region_mask, dtype=bool)
    if projected["projected_points"] == 0:
        write_binary_ply(output_path, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8))
        return {
            "path": str(output_path),
            "selected_points": 0,
            "max_points": int(max_points),
        }

    point_region_mask = mask[projected["yi"], projected["xi"]]
    selected_indices = np.flatnonzero(point_region_mask)
    if selected_indices.size > max_points:
        selected_conf = projected["conf"][selected_indices]
        top_local = np.argpartition(selected_conf, -max_points)[-max_points:]
        selected_indices = selected_indices[top_local]
        selected_indices.sort()

    points = projected["points"][selected_indices]
    colors = to_uint8(projected["colors"][selected_indices])
    write_binary_ply(output_path, points, colors)
    return {
        "path": str(output_path),
        "selected_points": int(points.shape[0]),
        "max_points": int(max_points),
    }


def summarize_confidence(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "p50": None, "p95": None}
    p50, p95 = np.percentile(values, [50, 95]).tolist()
    return {
        "mean": float(values.mean()),
        "p50": float(p50),
        "p95": float(p95),
    }


def compute_branch_region_diagnostics(
    *,
    branch_key,
    branch_label,
    points_hw3,
    conf_hw,
    source_colors_hw3,
    target_extrinsic,
    target_intrinsic,
    out_hw,
    min_conf,
    render_image,
    render_weight,
    target_image,
    region_masks,
    ply_dir,
    export_max_points,
):
    projected = project_points_to_target(
        world_points_s_hw3=points_hw3,
        world_conf_s_hw=conf_hw,
        src_rgb_s_hw3=source_colors_hw3,
        tgt_extrinsic_3x4=target_extrinsic,
        tgt_intrinsic_3x3=target_intrinsic,
        out_hw=out_hw,
        min_conf=min_conf,
    )
    hit_mask = np.asarray(render_weight, dtype=np.float32) > 0.0
    regions_payload = {}
    files = {}
    export_region_names = ("fg_human", "bg_bottom_band")
    projected_total = max(int(projected["projected_points"]), 1)

    for region_name, region_mask in region_masks.items():
        region_mask = np.asarray(region_mask, dtype=bool)
        pixel_count = int(region_mask.sum())
        if projected["projected_points"] > 0:
            point_region_mask = region_mask[projected["yi"], projected["xi"]]
            point_count = int(point_region_mask.sum())
            conf_values = projected["conf"][point_region_mask]
        else:
            point_region_mask = np.zeros((0,), dtype=bool)
            point_count = 0
            conf_values = np.zeros((0,), dtype=np.float32)

        entry = {
            "pixel_count": pixel_count,
            "pixel_ratio": None if region_mask.size == 0 else float(pixel_count / max(1, region_mask.size)),
            "projected_point_count": point_count,
            "projected_point_ratio": float(point_count / projected_total) if projected_total > 0 else None,
            "rendered_pixel_count": int(hit_mask[region_mask].sum()) if pixel_count > 0 else 0,
            "coverage_ratio": float(hit_mask[region_mask].mean()) if pixel_count > 0 else None,
            "confidence": summarize_confidence(conf_values),
            "render_metrics": masked_metrics(render_image, target_image, region_mask),
        }
        if region_name in export_region_names:
            ply_path = Path(ply_dir) / f"{branch_key}_{region_name}.ply"
            export_payload = export_projected_region_ply(
                output_path=ply_path,
                projected=projected,
                region_mask=region_mask,
                max_points=export_max_points,
            )
            entry["region_ply"] = str(ply_path)
            entry["export"] = export_payload
            files[f"{branch_key}_{region_name}_ply"] = str(ply_path)
        regions_payload[region_name] = entry

    return {
        "label": branch_label,
        "total_points": int(projected["total_points"]),
        "candidate_points": int(projected["candidate_points"]),
        "projected_points": int(projected["projected_points"]),
        "regions": regions_payload,
    }, files


def summarize_region_comparison(point_branch_payload, depth_branch_payload):
    comparison = {}
    for region_name in point_branch_payload["regions"].keys():
        point_region = point_branch_payload["regions"][region_name]
        depth_region = depth_branch_payload["regions"][region_name]
        point_mae = point_region["render_metrics"]["mae"]
        depth_mae = depth_region["render_metrics"]["mae"]
        point_cov = point_region["coverage_ratio"]
        depth_cov = depth_region["coverage_ratio"]
        mae_winner = "tie"
        coverage_winner = "tie"
        if point_mae is not None and depth_mae is not None:
            if depth_mae < point_mae:
                mae_winner = "depth_unproject"
            elif point_mae < depth_mae:
                mae_winner = "point_map"
        if point_cov is not None and depth_cov is not None:
            if depth_cov > point_cov:
                coverage_winner = "depth_unproject"
            elif point_cov > depth_cov:
                coverage_winner = "point_map"
        comparison[region_name] = {
            "mae_winner": mae_winner,
            "coverage_winner": coverage_winner,
        }
    return comparison


def relativize_payload_paths(payload, output_dir):
    output_dir = Path(output_dir).resolve()
    if isinstance(payload, dict):
        converted = {}
        for key, value in payload.items():
            converted[key] = relativize_payload_paths(value, output_dir)
        return converted
    if isinstance(payload, list):
        return [relativize_payload_paths(item, output_dir) for item in payload]
    if isinstance(payload, str):
        try:
            path = Path(payload)
        except (TypeError, ValueError):
            return payload
        if path.is_absolute():
            try:
                return str(path.resolve().relative_to(output_dir))
            except ValueError:
                return payload
    return payload


def compute_region_diagnostics(
    *,
    output_dir,
    seq_dir,
    frame_id,
    target_camera,
    target_image,
    target_extrinsic,
    target_intrinsic,
    source_colors,
    point_map_aligned,
    point_conf,
    depth_points_aligned,
    depth_conf,
    point_render,
    depth_render,
    target_mask_source="mask",
    region_edge_px=5,
    bottom_band_ratio=0.2,
    min_conf=1e-6,
    export_max_points=250000,
    case_meta=None,
    legacy_reference_metrics=None,
):
    output_dir = ensure_dir(output_dir)
    ply_dir = ensure_dir(output_dir / "ply")
    fg_mask = load_target_mask(
        seq_dir=seq_dir,
        camera_name=target_camera,
        frame_id=frame_id,
        target_hw=target_image.shape[:2],
        mask_source=target_mask_source,
    )
    region_masks = build_region_masks(
        fg_mask=fg_mask,
        edge_px=region_edge_px,
        bottom_band_ratio=bottom_band_ratio,
    )
    overlay_path = output_dir / "target_mask_overlay.png"
    make_target_mask_overlay(overlay_path, target_image, region_masks)

    point_branch, point_files = compute_branch_region_diagnostics(
        branch_key="point_map",
        branch_label="Point Map",
        points_hw3=point_map_aligned,
        conf_hw=point_conf,
        source_colors_hw3=source_colors,
        target_extrinsic=target_extrinsic,
        target_intrinsic=target_intrinsic,
        out_hw=target_image.shape[:2],
        min_conf=min_conf,
        render_image=point_render["image"],
        render_weight=point_render["weight"],
        target_image=target_image,
        region_masks=region_masks,
        ply_dir=ply_dir,
        export_max_points=export_max_points,
    )
    depth_branch, depth_files = compute_branch_region_diagnostics(
        branch_key="depth_unproject",
        branch_label="Depth+Camera",
        points_hw3=depth_points_aligned,
        conf_hw=depth_conf,
        source_colors_hw3=source_colors,
        target_extrinsic=target_extrinsic,
        target_intrinsic=target_intrinsic,
        out_hw=target_image.shape[:2],
        min_conf=min_conf,
        render_image=depth_render["image"],
        render_weight=depth_render["weight"],
        target_image=target_image,
        region_masks=region_masks,
        ply_dir=ply_dir,
        export_max_points=export_max_points,
    )

    payload = {
        "case": case_meta or {},
        "config": {
            "target_mask_source": target_mask_source,
            "region_edge_px": int(region_edge_px),
            "bottom_band_ratio": float(bottom_band_ratio),
            "min_conf": float(min_conf),
            "export_max_points": int(export_max_points),
        },
        "regions": {
            region_name: {
                "pixel_count": int(mask.sum()),
                "pixel_ratio": float(mask.mean()),
            }
            for region_name, mask in region_masks.items()
        },
        "branches": {
            "point_map": point_branch,
            "depth_unproject": depth_branch,
        },
        "comparison": summarize_region_comparison(point_branch, depth_branch),
        "legacy_reference_metrics": legacy_reference_metrics or {},
        "files": {
            "target_mask_overlay_png": str(overlay_path),
            **point_files,
            **depth_files,
        },
    }
    return relativize_payload_paths(payload, output_dir)


def _fmt_metric(value):
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def write_region_markdown_report(path, payload):
    lines = [
        "# Geometry Region Diagnostics",
        "",
    ]
    case = payload.get("case") or {}
    if case:
        lines.extend(
            [
                f"- seq_name: `{case.get('seq_name', '')}`",
                f"- frame_id: `{case.get('frame_id', '')}`",
                f"- target_camera: `{case.get('target_camera', '')}`",
                f"- view_profile: `{case.get('view_profile', '')}`",
                "",
            ]
        )
    lines.extend(
        [
            f"- target_mask_source: `{payload['config']['target_mask_source']}`",
            f"- region_edge_px: `{payload['config']['region_edge_px']}`",
            f"- bottom_band_ratio: `{payload['config']['bottom_band_ratio']}`",
            "",
            "## Region Pixels",
            "",
            "| Region | Pixel Count | Pixel Ratio |",
            "| --- | ---: | ---: |",
        ]
    )
    for region_name, region_payload in payload["regions"].items():
        lines.append(
            f"| `{region_name}` | {region_payload['pixel_count']} | {region_payload['pixel_ratio']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Branch-Region Metrics",
            "",
            "| Region | Branch | Projected Points | Projected Ratio | Coverage | MAE | PSNR | SSIM | Conf Mean | Conf P50 | Conf P95 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for region_name in payload["regions"].keys():
        for branch_key in ("point_map", "depth_unproject"):
            branch_region = payload["branches"][branch_key]["regions"][region_name]
            conf = branch_region["confidence"]
            metrics = branch_region["render_metrics"]
            lines.append(
                "| `{region}` | `{branch}` | {proj} | {proj_ratio} | {cov} | {mae} | {psnr} | {ssim} | {cmean} | {cp50} | {cp95} |".format(
                    region=region_name,
                    branch=branch_key,
                    proj=branch_region["projected_point_count"],
                    proj_ratio=_fmt_metric(branch_region["projected_point_ratio"]),
                    cov=_fmt_metric(branch_region["coverage_ratio"]),
                    mae=_fmt_metric(metrics["mae"]),
                    psnr=_fmt_metric(metrics["psnr"]),
                    ssim=_fmt_metric(metrics["ssim"]),
                    cmean=_fmt_metric(conf["mean"]),
                    cp50=_fmt_metric(conf["p50"]),
                    cp95=_fmt_metric(conf["p95"]),
                )
            )

    lines.extend(
        [
            "",
            "## Region Winners",
            "",
            "| Region | MAE Winner | Coverage Winner |",
            "| --- | --- | --- |",
        ]
    )
    for region_name, comparison in payload["comparison"].items():
        lines.append(
            f"| `{region_name}` | `{comparison['mae_winner']}` | `{comparison['coverage_winner']}` |"
        )

    legacy_metrics = payload.get("legacy_reference_metrics") or {}
    if legacy_metrics:
        lines.extend(
            [
                "",
                "## Legacy Reference",
                "",
                f"- full_frame_mae: `{legacy_metrics.get('mae', 'n/a')}`",
                f"- full_frame_psnr: `{legacy_metrics.get('psnr', 'n/a')}`",
                f"- full_frame_ssim: `{legacy_metrics.get('ssim', 'n/a')}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- target_mask_overlay_png: `{payload['files']['target_mask_overlay_png']}`",
        ]
    )
    for key in (
        "point_map_fg_human_ply",
        "depth_unproject_fg_human_ply",
        "point_map_bg_bottom_band_ply",
        "depth_unproject_bg_bottom_band_ply",
    ):
        if key in payload["files"]:
            lines.append(f"- {key}: `{payload['files'][key]}`")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
