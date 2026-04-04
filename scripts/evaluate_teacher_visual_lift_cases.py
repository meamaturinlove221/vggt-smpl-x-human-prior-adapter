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
    return parser.parse_args()


def _load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict manifest: {path}")
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

    point_render = render_forward_splat(
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
    depth_render = render_forward_splat(
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

    variant_images = {
        "baseline_depth_unproject": baseline_depth,
    }
    for variant_name in variants:
        variant_images[variant_name] = _apply_variant(baseline_depth, weight01, fg_mask, variant_name)

    _save_rgb(renders_dir / "target.png", target_image)
    _save_rgb(renders_dir / "point_map.png", point_image)
    _save_rgb(renders_dir / "depth_unproject.png", baseline_depth)
    _save_gray(renders_dir / "depth_weight.png", weight01)
    _save_gray(renders_dir / "fg_mask.png", fg_mask.astype(np.float32))
    for variant_name, image01 in variant_images.items():
        _save_rgb(renders_dir / f"{variant_name}.png", image01)

    _make_case_panel(
        renders_dir / "comparison_panel.png",
        target_image,
        point_image,
        baseline_depth,
        weight01,
        fg_mask,
        variant_images,
    )

    rows = []
    for variant_name, image01 in variant_images.items():
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
                "files": {
                    "target_png": str((renders_dir / "target.png").relative_to(output_dir)),
                    "point_map_png": str((renders_dir / "point_map.png").relative_to(output_dir)),
                    "depth_unproject_png": str((renders_dir / "depth_unproject.png").relative_to(output_dir)),
                    "weight_png": str((renders_dir / "depth_weight.png").relative_to(output_dir)),
                    "fg_mask_png": str((renders_dir / "fg_mask.png").relative_to(output_dir)),
                    "variant_png": str((renders_dir / f"{variant_name}.png").relative_to(output_dir)),
                    "comparison_panel_png": str((renders_dir / "comparison_panel.png").relative_to(output_dir)),
                },
                "alignment": {
                    "scale": float(scale),
                    "src_center_rmse_after": float(rmse(apply_sim3_points(centers_pred, scale, sim_rotation, sim_translation), centers_gt)),
                },
                "baseline_render_stats": depth_render["stats"],
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
