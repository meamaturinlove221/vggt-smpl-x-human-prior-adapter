import argparse
import contextlib
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
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
        description="Run one VGGT inference pass and compare point-map vs depth-unprojection branches."
    )
    parser.add_argument("--image_folder", type=str, required=True, help="Folder containing input images.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory for reports and exported artifacts. Defaults to output/geometry_baseline/<scene>_<timestamp>.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional local checkpoint path. Defaults to the official VGGT-1B checkpoint URL.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--preprocess_mode", type=str, default="crop", choices=["crop", "pad"])
    parser.add_argument("--max_images", type=int, default=0, help="Limit the number of sorted input images. 0 means all.")
    parser.add_argument(
        "--conf_percentile",
        type=float,
        default=25.0,
        help="Filter out the lowest X percent of confidence values before export and rendering.",
    )
    parser.add_argument(
        "--export_max_points",
        type=int,
        default=250000,
        help="Maximum number of points written to each exported PLY.",
    )
    parser.add_argument(
        "--render_max_points",
        type=int,
        default=750000,
        help="Maximum number of points used when rendering the branch comparison views.",
    )
    parser.add_argument(
        "--target_frames",
        type=str,
        default="0",
        help='Comma-separated target frame indices, or "all". Default: 0.',
    )
    parser.add_argument(
        "--include_target_frame",
        action="store_true",
        help="Include the target frame's own points in the re-rendered comparison view.",
    )
    parser.add_argument(
        "--skip_save_predictions",
        action="store_true",
        help="Skip saving the compressed predictions.npz artifact.",
    )
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def resolve_output_dir(image_folder, output_dir):
    if output_dir:
        return ensure_dir(output_dir)
    folder_name = Path(image_folder).resolve().name or "scene"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path("output") / "geometry_baseline" / f"{folder_name}_{timestamp}")


def collect_image_paths(image_folder, max_images):
    image_dir = Path(image_folder)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    image_paths = [path for path in sorted(image_dir.iterdir()) if path.is_file()]
    if not image_paths:
        raise ValueError(f"No images found in {image_folder}")
    if max_images > 0:
        image_paths = image_paths[:max_images]
    return image_paths


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
        "Install a newer PyTorch build that supports this GPU, or rerun with --device cpu for a slow smoke test.\n"
        "Torch already suggests using a CUDA 12.8 or 13.0 build for this machine."
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
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
    else:
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


def run_inference(model, image_paths, device, dtype, preprocess_mode):
    images = load_and_preprocess_images([str(path) for path in image_paths], mode=preprocess_mode).to(device)
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=dtype)
        if device.type == "cuda" and dtype != torch.float32
        else contextlib.nullcontext()
    )

    with torch.no_grad():
        with autocast_ctx:
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    outputs = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            outputs[key] = value.detach().cpu().numpy().squeeze(0)

    outputs["world_points_from_depth"] = unproject_depth_map_to_point_map(
        outputs["depth"], outputs["extrinsic"], outputs["intrinsic"]
    )
    outputs["image_names"] = np.array([path.name for path in image_paths], dtype=object)
    return outputs


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
    point_values = points[valid]

    summary = {
        "valid_ratio": valid_ratio,
        "valid_points": int(valid.sum()),
        "total_points": int(valid.size),
        "confidence_percentiles": {},
        "xyz_abs_percentiles": {},
    }

    if conf_values.size == 0:
        return summary

    conf_percentiles = np.percentile(conf_values, [5, 25, 50, 75, 95]).tolist()
    xyz_abs = np.abs(point_values.reshape(-1, 3))
    xyz_percentiles = np.percentile(xyz_abs, [50, 90, 95, 99], axis=0)

    summary["confidence_percentiles"] = {
        "p05": float(conf_percentiles[0]),
        "p25": float(conf_percentiles[1]),
        "p50": float(conf_percentiles[2]),
        "p75": float(conf_percentiles[3]),
        "p95": float(conf_percentiles[4]),
    }
    summary["xyz_abs_percentiles"] = {
        "x": [float(v) for v in xyz_percentiles[:, 0]],
        "y": [float(v) for v in xyz_percentiles[:, 1]],
        "z": [float(v) for v in xyz_percentiles[:, 2]],
    }
    return summary


def summarize_branch_delta(point_map, depth_points, point_conf, depth_conf):
    finite_mask = np.isfinite(point_map).all(axis=-1) & np.isfinite(depth_points).all(axis=-1)
    valid_mask = finite_mask & (point_conf > CONF_EPS) & (depth_conf > CONF_EPS)
    if not valid_mask.any():
        return {
            "shared_valid_ratio": 0.0,
            "distance_percentiles": {},
        }

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


def to_uint8_images(images_nchw):
    images = np.asarray(images_nchw, dtype=np.float32)
    images = np.clip(images, 0.0, 1.0)
    return (images.transpose(0, 2, 3, 1) * 255.0).round().astype(np.uint8)


def write_binary_ply(path, points, colors):
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.shape[0] != colors.shape[0]:
        raise ValueError("Points and colors must contain the same number of rows.")

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


def parse_target_frames(raw_value, frame_count):
    text = str(raw_value).strip().lower()
    if text == "all":
        return list(range(frame_count))
    values = []
    for part in str(raw_value).split(","):
        piece = part.strip()
        if not piece:
            continue
        index = int(piece)
        if index < 0 or index >= frame_count:
            raise ValueError(f"Target frame index out of range: {index}")
        values.append(index)
    if not values:
        raise ValueError("No valid target frame indices were provided.")
    return sorted(set(values))


def flatten_branch(points, conf, colors):
    seq_len, height, width, _ = points.shape
    frame_ids = np.repeat(np.arange(seq_len, dtype=np.int32), height * width)
    return (
        points.reshape(-1, 3),
        conf.reshape(-1),
        colors.reshape(-1, 3),
        frame_ids,
        height,
        width,
    )


def render_branch_to_view(
    points,
    conf,
    colors,
    target_extrinsic,
    target_intrinsic,
    target_frame_idx,
    percentile,
    max_points,
    include_target_frame,
):
    flat_points, flat_conf, flat_colors, frame_ids, height, width = flatten_branch(points, conf, colors)
    if points.shape[0] == 1:
        include_target_frame = True
    finite = np.isfinite(flat_points).all(axis=-1)
    threshold = branch_threshold(conf, finite.reshape(points.shape[:3]), percentile)

    keep = finite & np.isfinite(flat_conf) & (flat_conf >= threshold)
    if not include_target_frame:
        keep &= frame_ids != target_frame_idx

    candidate_indices = np.flatnonzero(keep)
    if candidate_indices.size == 0:
        return {
            "image": np.zeros((height, width, 3), dtype=np.uint8),
            "mask": np.zeros((height, width), dtype=bool),
            "depth": np.full((height, width), np.inf, dtype=np.float32),
            "confidence_threshold": float(threshold),
            "input_points": 0,
            "rendered_points": 0,
        }

    if candidate_indices.size > max_points:
        candidate_conf = flat_conf[candidate_indices]
        top_local = np.argpartition(candidate_conf, -max_points)[-max_points:]
        candidate_indices = candidate_indices[top_local]

    candidate_points = flat_points[candidate_indices]
    candidate_colors = flat_colors[candidate_indices]

    rotation = target_extrinsic[:, :3]
    translation = target_extrinsic[:, 3]
    camera_points = candidate_points @ rotation.T + translation
    depth_values = camera_points[:, 2]

    positive_depth = np.isfinite(depth_values) & (depth_values > DEPTH_EPS)
    camera_points = camera_points[positive_depth]
    candidate_colors = candidate_colors[positive_depth]
    depth_values = depth_values[positive_depth]
    if depth_values.size == 0:
        return {
            "image": np.zeros((height, width, 3), dtype=np.uint8),
            "mask": np.zeros((height, width), dtype=bool),
            "depth": np.full((height, width), np.inf, dtype=np.float32),
            "confidence_threshold": float(threshold),
            "input_points": int(candidate_indices.size),
            "rendered_points": 0,
        }

    fx = float(target_intrinsic[0, 0])
    fy = float(target_intrinsic[1, 1])
    cx = float(target_intrinsic[0, 2])
    cy = float(target_intrinsic[1, 2])

    projected_u = fx * (camera_points[:, 0] / depth_values) + cx
    projected_v = fy * (camera_points[:, 1] / depth_values) + cy
    finite_projection = np.isfinite(projected_u) & np.isfinite(projected_v)
    projected_u = projected_u[finite_projection]
    projected_v = projected_v[finite_projection]
    candidate_colors = candidate_colors[finite_projection]
    depth_values = depth_values[finite_projection]

    projected_u = np.rint(projected_u).astype(np.int32)
    projected_v = np.rint(projected_v).astype(np.int32)

    in_bounds = (
        (projected_u >= 0)
        & (projected_u < width)
        & (projected_v >= 0)
        & (projected_v < height)
    )

    if not in_bounds.any():
        return {
            "image": np.zeros((height, width, 3), dtype=np.uint8),
            "mask": np.zeros((height, width), dtype=bool),
            "depth": np.full((height, width), np.inf, dtype=np.float32),
            "confidence_threshold": float(threshold),
            "input_points": int(candidate_indices.size),
            "rendered_points": 0,
        }

    projected_u = projected_u[in_bounds]
    projected_v = projected_v[in_bounds]
    candidate_colors = candidate_colors[in_bounds]
    depth_values = depth_values[in_bounds]

    sort_order = np.argsort(depth_values, kind="stable")
    sorted_u = projected_u[sort_order]
    sorted_v = projected_v[sort_order]
    sorted_depth = depth_values[sort_order]
    sorted_colors = candidate_colors[sort_order]
    pixel_ids = sorted_v * width + sorted_u
    _, first_indices = np.unique(pixel_ids, return_index=True)

    render = np.zeros((height, width, 3), dtype=np.uint8)
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)

    chosen_u = sorted_u[first_indices]
    chosen_v = sorted_v[first_indices]
    chosen_depth = sorted_depth[first_indices]
    chosen_colors = sorted_colors[first_indices]

    render[chosen_v, chosen_u] = chosen_colors
    depth_map[chosen_v, chosen_u] = chosen_depth
    mask[chosen_v, chosen_u] = True

    return {
        "image": render,
        "mask": mask,
        "depth": depth_map,
        "confidence_threshold": float(threshold),
        "input_points": int(candidate_indices.size),
        "rendered_points": int(first_indices.size),
    }


def masked_mae(rendered, target, mask):
    if not mask.any():
        return math.nan
    diff = np.abs(rendered.astype(np.float32) - target.astype(np.float32))
    return float(diff[mask].mean() / 255.0)


def make_tiled_report(output_path, target_image, point_render, depth_render):
    point_diff = np.abs(point_render.astype(np.int16) - target_image.astype(np.int16)).astype(np.uint8)
    depth_diff = np.abs(depth_render.astype(np.int16) - target_image.astype(np.int16)).astype(np.uint8)
    branch_diff = np.abs(point_render.astype(np.int16) - depth_render.astype(np.int16)).astype(np.uint8)

    panels = [
        ("Target", target_image),
        ("Point Map", point_render),
        ("Depth+Camera", depth_render),
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
    lines = [
        "# Geometry Branch Baseline",
        "",
        f"- scene: `{summary['inputs']['scene_name']}`",
        f"- image_count: `{summary['inputs']['image_count']}`",
        f"- image_hw: `{summary['inputs']['image_hw']}`",
        f"- preprocess_mode: `{summary['run_config']['preprocess_mode']}`",
        f"- device: `{summary['environment']['device']}`",
        f"- dtype: `{summary['environment']['dtype']}`",
        "",
        "## Branch Stats",
        "",
        "| Branch | Valid Ratio | Valid Points | Conf P50 | Conf P95 | Exported PLY Points |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for branch_key in ("point_map", "depth_unproject"):
        branch = summary["branches"][branch_key]
        conf = branch["summary"]["confidence_percentiles"]
        export = branch["export"]
        lines.append(
            "| {name} | {valid_ratio:.4f} | {valid_points} | {p50:.4f} | {p95:.4f} | {exported} |".format(
                name=branch["label"],
                valid_ratio=branch["summary"]["valid_ratio"],
                valid_points=branch["summary"]["valid_points"],
                p50=conf.get("p50", float("nan")),
                p95=conf.get("p95", float("nan")),
                exported=export["exported_points"],
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
            "## Re-render Checks",
            "",
            "| Target Frame | Image | Point Coverage | Depth Coverage | Point MAE | Depth MAE | Report |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    for render in summary["renders"]:
        lines.append(
            "| {frame} | {name} | {point_cov:.4f} | {depth_cov:.4f} | {point_mae:.4f} | {depth_mae:.4f} | `{report}` |".format(
                frame=render["target_frame_idx"],
                name=render["image_name"],
                point_cov=render["point_map"]["coverage_ratio"],
                depth_cov=render["depth_unproject"]["coverage_ratio"],
                point_mae=render["point_map"]["mae_to_target"],
                depth_mae=render["depth_unproject"]["mae_to_target"],
                report=render["files"]["comparison_png"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- First inspect the branch comparison PNGs under `renders/`.",
            "- If `Depth+Camera` gives cleaner structure or better target-frame reprojection coverage/MAE, keep the next step on the geometry chain.",
            "- Do not reintroduce the legacy ghost stack until this baseline is clearly understood.",
        ]
    )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def write_render_csv(path, rows):
    fieldnames = [
        "target_frame_idx",
        "image_name",
        "point_map_coverage_ratio",
        "depth_unproject_coverage_ratio",
        "point_map_mae_to_target",
        "depth_unproject_mae_to_target",
        "point_map_rendered_points",
        "depth_unproject_rendered_points",
        "comparison_png",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    output_dir = resolve_output_dir(args.image_folder, args.output_dir)
    ply_dir = ensure_dir(output_dir / "ply")
    render_dir = ensure_dir(output_dir / "renders")

    image_paths = collect_image_paths(args.image_folder, args.max_images)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print(f"[info] output_dir={output_dir}")
    print(f"[info] using {len(image_paths)} images from {args.image_folder}")
    print(f"[info] device={device} dtype={dtype}")

    model = load_model(args.checkpoint, device)
    outputs = run_inference(model, image_paths, device, dtype, args.preprocess_mode)

    colors = to_uint8_images(outputs["images"])
    point_map = np.asarray(outputs["world_points"], dtype=np.float32)
    point_conf = np.asarray(outputs["world_points_conf"], dtype=np.float32)
    depth_points = np.asarray(outputs["world_points_from_depth"], dtype=np.float32)
    depth_conf = np.asarray(outputs["depth_conf"], dtype=np.float32)

    point_summary = summarize_branch(point_map, point_conf)
    depth_summary = summarize_branch(depth_points, depth_conf)
    branch_delta = summarize_branch_delta(point_map, depth_points, point_conf, depth_conf)

    point_export = export_branch_ply(
        ply_dir / "point_map.ply",
        point_map,
        point_conf,
        colors,
        args.conf_percentile,
        args.export_max_points,
    )
    depth_export = export_branch_ply(
        ply_dir / "depth_unproject.ply",
        depth_points,
        depth_conf,
        colors,
        args.conf_percentile,
        args.export_max_points,
    )

    target_indices = parse_target_frames(args.target_frames, colors.shape[0])
    render_rows = []
    render_summaries = []
    for target_frame_idx in target_indices:
        target_image = colors[target_frame_idx]
        point_render = render_branch_to_view(
            point_map,
            point_conf,
            colors,
            outputs["extrinsic"][target_frame_idx],
            outputs["intrinsic"][target_frame_idx],
            target_frame_idx,
            args.conf_percentile,
            args.render_max_points,
            args.include_target_frame,
        )
        depth_render = render_branch_to_view(
            depth_points,
            depth_conf,
            colors,
            outputs["extrinsic"][target_frame_idx],
            outputs["intrinsic"][target_frame_idx],
            target_frame_idx,
            args.conf_percentile,
            args.render_max_points,
            args.include_target_frame,
        )

        comparison_name = f"target_{target_frame_idx:03d}_compare.png"
        make_tiled_report(
            render_dir / comparison_name,
            target_image,
            point_render["image"],
            depth_render["image"],
        )

        render_summary = {
            "target_frame_idx": int(target_frame_idx),
            "image_name": str(outputs["image_names"][target_frame_idx]),
            "point_map": {
                "coverage_ratio": float(point_render["mask"].mean()),
                "mae_to_target": masked_mae(point_render["image"], target_image, point_render["mask"]),
                "rendered_points": int(point_render["rendered_points"]),
                "input_points": int(point_render["input_points"]),
                "confidence_threshold": float(point_render["confidence_threshold"]),
            },
            "depth_unproject": {
                "coverage_ratio": float(depth_render["mask"].mean()),
                "mae_to_target": masked_mae(depth_render["image"], target_image, depth_render["mask"]),
                "rendered_points": int(depth_render["rendered_points"]),
                "input_points": int(depth_render["input_points"]),
                "confidence_threshold": float(depth_render["confidence_threshold"]),
            },
            "files": {
                "comparison_png": str(Path("renders") / comparison_name),
            },
        }
        render_summaries.append(render_summary)
        render_rows.append(
            {
                "target_frame_idx": render_summary["target_frame_idx"],
                "image_name": render_summary["image_name"],
                "point_map_coverage_ratio": render_summary["point_map"]["coverage_ratio"],
                "depth_unproject_coverage_ratio": render_summary["depth_unproject"]["coverage_ratio"],
                "point_map_mae_to_target": render_summary["point_map"]["mae_to_target"],
                "depth_unproject_mae_to_target": render_summary["depth_unproject"]["mae_to_target"],
                "point_map_rendered_points": render_summary["point_map"]["rendered_points"],
                "depth_unproject_rendered_points": render_summary["depth_unproject"]["rendered_points"],
                "comparison_png": render_summary["files"]["comparison_png"],
            }
        )

    if not args.skip_save_predictions:
        np.savez_compressed(
            output_dir / "predictions.npz",
            images=outputs["images"],
            depth=outputs["depth"],
            depth_conf=outputs["depth_conf"],
            world_points=outputs["world_points"],
            world_points_conf=outputs["world_points_conf"],
            world_points_from_depth=outputs["world_points_from_depth"],
            extrinsic=outputs["extrinsic"],
            intrinsic=outputs["intrinsic"],
            pose_enc=outputs["pose_enc"],
            image_names=outputs["image_names"],
        )

    summary = {
        "run_config": {
            "image_folder": str(Path(args.image_folder).resolve()),
            "output_dir": str(output_dir.resolve()),
            "checkpoint": args.checkpoint or DEFAULT_MODEL_URL,
            "preprocess_mode": args.preprocess_mode,
            "conf_percentile": float(args.conf_percentile),
            "export_max_points": int(args.export_max_points),
            "render_max_points": int(args.render_max_points),
            "target_frames": target_indices,
            "include_target_frame": bool(args.include_target_frame),
            "save_predictions": not bool(args.skip_save_predictions),
        },
        "environment": {
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        },
        "inputs": {
            "scene_name": Path(args.image_folder).resolve().name,
            "image_count": len(image_paths),
            "image_hw": list(colors.shape[1:3]),
            "image_names": [path.name for path in image_paths],
        },
        "branches": {
            "point_map": {
                "label": "Point Map",
                "summary": point_summary,
                "export": point_export,
            },
            "depth_unproject": {
                "label": "Depth+Camera",
                "summary": depth_summary,
                "export": depth_export,
            },
        },
        "branch_delta": branch_delta,
        "renders": render_summaries,
    }

    save_json(output_dir / "summary.json", summary)
    write_markdown_report(output_dir / "summary.md", summary)
    write_render_csv(output_dir / "render_metrics.csv", render_rows)

    print("[done] Geometry branch baseline complete.")
    print(f"[done] Summary: {output_dir / 'summary.md'}")
    print(f"[done] Point-map PLY: {ply_dir / 'point_map.ply'}")
    print(f"[done] Depth+camera PLY: {ply_dir / 'depth_unproject.ply'}")


if __name__ == "__main__":
    main()
