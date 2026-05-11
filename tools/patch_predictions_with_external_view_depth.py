from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from normal_line_multiview_eval import load_scene_view  # noqa: E402
from render_open3d_pointcloud import unproject_depth_map_to_point_map_numpy  # noqa: E402
from vggt.utils.normal_refiner import face_box_from_mask, head_box_from_mask, shoulder_box_from_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch VGGT prediction depth/world-points with independently projected "
            "per-view external depth. This is a local diagnostic candidate builder; "
            "it must still pass the hardened mentor gate before being treated as useful."
        )
    )
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--external-depth-npz", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--align-roi-kind", choices=("head", "face", "face_core", "head_face", "shoulder", "all"), default="head_face")
    parser.add_argument("--apply-roi-kind", choices=("head", "face", "face_core", "head_face", "shoulder", "all"), default="head_face")
    parser.add_argument("--fit-mode", choices=("affine", "shift", "identity"), default="affine")
    parser.add_argument("--max-depth-delta", type=float, default=0.035)
    parser.add_argument("--min-fit-pixels", type=int, default=128)
    parser.add_argument("--min-fit-corr", type=float, default=-1.0)
    parser.add_argument("--conf-boost-percentile", type=float, default=70.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _roi_mask(mask: np.ndarray, roi_kind: str) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if roi_kind == "all":
        return mask.copy()
    boxes: list[tuple[int, int, int, int] | None] = []
    if roi_kind in {"head", "head_face"}:
        boxes.append(head_box_from_mask(mask))
    if roi_kind in {"face", "head_face"}:
        boxes.append(face_box_from_mask(mask))
    if roi_kind == "face_core":
        face_box = face_box_from_mask(mask)
        if face_box is not None:
            x0, y0, x1, y1 = face_box
            width = x1 - x0
            height = y1 - y0
            core_w = max(16, int(round(width * 0.70)))
            core_h = max(16, int(round(height * 0.68)))
            cx = int(round((x0 + x1) * 0.5))
            cy = y0 + int(round(height * 0.48))
            boxes.append((cx - core_w // 2, cy - core_h // 2, cx + core_w // 2, cy + core_h // 2))
    if roi_kind == "shoulder":
        boxes.append(shoulder_box_from_mask(mask))

    out = np.zeros(mask.shape, dtype=bool)
    for box in boxes:
        if box is None:
            continue
        x0, y0, x1, y1 = box
        x0 = max(0, min(mask.shape[1], int(x0)))
        x1 = max(0, min(mask.shape[1], int(x1)))
        y0 = max(0, min(mask.shape[0], int(y0)))
        y1 = max(0, min(mask.shape[0], int(y1)))
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] |= mask[y0:y1, x0:x1]
    return out


def _fit_depth(
    external_depth: np.ndarray,
    anchor_depth: np.ndarray,
    mask: np.ndarray,
    *,
    fit_mode: str,
    min_fit_pixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    valid = mask & np.isfinite(external_depth) & np.isfinite(anchor_depth) & (external_depth > 0.0)
    if int(valid.sum()) < int(min_fit_pixels):
        median_anchor = float(np.nanmedian(anchor_depth[np.isfinite(anchor_depth)]))
        aligned = np.full_like(anchor_depth, median_anchor, dtype=np.float32)
        return aligned, {"valid": int(valid.sum()), "fallback": "too_few_points", "bias": median_anchor, "scale": 0.0}

    x = external_depth[valid].astype(np.float64)
    y = anchor_depth[valid].astype(np.float64)
    corr = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 and np.std(x) > 1e-8 and np.std(y) > 1e-8 else 0.0
    if fit_mode == "identity":
        scale = 1.0
        bias = 0.0
    elif fit_mode == "shift":
        scale = 1.0
        bias = float(np.median(y - x))
    else:
        design = np.stack([x, np.ones_like(x)], axis=1)
        scale, bias = np.linalg.lstsq(design, y, rcond=None)[0]
        scale = float(scale)
        bias = float(bias)
        if not np.isfinite(scale) or not np.isfinite(bias) or abs(scale) > 10.0:
            scale = 1.0
            bias = float(np.median(y - x))
            fit_mode = "shift_fallback"

    aligned = (scale * external_depth + bias).astype(np.float32)
    err = scale * x + bias - y
    stats = {
        "valid": int(valid.sum()),
        "fit_mode": fit_mode,
        "scale": float(scale),
        "bias": float(bias),
        "corr": corr,
        "mae": float(np.mean(np.abs(err))),
        "p50_abs": float(np.percentile(np.abs(err), 50)),
        "p95_abs": float(np.percentile(np.abs(err), 95)),
    }
    return aligned, stats


def _load_scene_masks(scene_dir: Path, view_count: int, hw: tuple[int, int]) -> np.ndarray:
    return np.stack([load_scene_view(scene_dir, idx, hw).mask.astype(bool) for idx in range(view_count)], axis=0)


def _depth_vis(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = depth[mask & np.isfinite(depth)]
    if values.size < 16:
        values = depth[np.isfinite(depth)]
    lo, hi = np.percentile(values, [2, 98]) if values.size else (0.0, 1.0)
    gray = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    gray[~np.isfinite(gray)] = 1.0
    return np.repeat((gray[..., None] * 255.0).astype(np.uint8), 3, axis=-1)


def _make_preview(
    *,
    rgb: np.ndarray,
    external_depth: np.ndarray,
    aligned_depth: np.ndarray,
    anchor_depth: np.ndarray,
    patched_depth: np.ndarray,
    apply_mask: np.ndarray,
    out_path: Path,
) -> None:
    delta = patched_depth - anchor_depth
    diff = np.ones((*anchor_depth.shape, 3), dtype=np.float32)
    scaled = np.clip(delta / 0.05, -1.0, 1.0)
    diff[..., 0] = np.clip(scaled, 0.0, 1.0)
    diff[..., 2] = np.clip(-scaled, 0.0, 1.0)
    diff[..., 1] = 0.0
    diff[~apply_mask] = 1.0

    overlay = rgb.astype(np.float32).copy()
    overlay[apply_mask] = 0.55 * overlay[apply_mask] + np.array([0.0, 220.0, 40.0], dtype=np.float32) * 0.45
    tiles = [
        rgb.astype(np.uint8),
        _depth_vis(external_depth, apply_mask),
        _depth_vis(aligned_depth, apply_mask),
        _depth_vis(anchor_depth, apply_mask),
        _depth_vis(patched_depth, apply_mask),
        (diff * 255.0).astype(np.uint8),
        overlay.astype(np.uint8),
    ]
    labels = ["RGB", "external", "aligned", "base", "patched", "delta", "apply mask"]
    height, width = rgb.shape[:2]
    canvas = Image.new("RGB", (width * len(tiles), height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (tile, label) in enumerate(zip(tiles, labels)):
        canvas.paste(Image.fromarray(tile), (idx * width, 24))
        draw.text((idx * width + 4, 4), label, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(args.base_predictions)
    scene_dir = Path(args.scene_dir)
    external_path = Path(args.external_depth_npz)
    base = np.load(base_path, allow_pickle=False)
    external = np.load(external_path, allow_pickle=False)
    data = {key: np.array(base[key]) for key in base.files}

    base_depth = np.asarray(data["depth"], dtype=np.float32)
    if base_depth.ndim == 4 and base_depth.shape[-1] == 1:
        base_depth_2d = base_depth[..., 0]
    else:
        base_depth_2d = base_depth
    external_depth = np.asarray(external["depth"], dtype=np.float32)
    external_mask = np.asarray(external["mask"], dtype=bool) if "mask" in external.files else external_depth > 0.0
    if external_depth.shape != base_depth_2d.shape:
        raise ValueError(f"depth shape mismatch: external {external_depth.shape}, base {base_depth_2d.shape}")

    view_count, height, width = base_depth_2d.shape
    masks = _load_scene_masks(scene_dir, view_count, (height, width))
    align_masks = np.stack([_roi_mask(masks[idx], args.align_roi_kind) for idx in range(view_count)], axis=0)
    apply_masks = np.stack([_roi_mask(masks[idx], args.apply_roi_kind) for idx in range(view_count)], axis=0)
    align_masks &= masks & external_mask & np.isfinite(base_depth_2d)
    apply_masks &= masks & external_mask & np.isfinite(base_depth_2d)

    patched_depth = base_depth_2d.copy()
    aligned_depth = np.zeros_like(base_depth_2d, dtype=np.float32)
    records: list[dict[str, Any]] = []
    for view_idx in range(view_count):
        aligned, fit_stats = _fit_depth(
            external_depth[view_idx],
            base_depth_2d[view_idx],
            align_masks[view_idx],
            fit_mode=args.fit_mode,
            min_fit_pixels=int(args.min_fit_pixels),
        )
        aligned_depth[view_idx] = aligned
        corr_ok = float(fit_stats.get("corr", 0.0)) >= float(args.min_fit_corr)
        delta = np.clip(aligned - base_depth_2d[view_idx], -float(args.max_depth_delta), float(args.max_depth_delta))
        use_mask = apply_masks[view_idx] & corr_ok
        patched_depth[view_idx][use_mask] = base_depth_2d[view_idx][use_mask] + delta[use_mask]

        rgb = load_scene_view(scene_dir, view_idx, (height, width)).image
        _make_preview(
            rgb=rgb,
            external_depth=external_depth[view_idx],
            aligned_depth=aligned_depth[view_idx],
            anchor_depth=base_depth_2d[view_idx],
            patched_depth=patched_depth[view_idx],
            apply_mask=use_mask,
            out_path=output_dir / "previews" / f"{view_idx:02d}_external_depth_patch_preview.png",
        )
        changed = patched_depth[view_idx] - base_depth_2d[view_idx]
        records.append(
            {
                "view_index": int(view_idx),
                "align_pixels": int(align_masks[view_idx].sum()),
                "apply_pixels": int(apply_masks[view_idx].sum()),
                "used_pixels": int(use_mask.sum()),
                "corr_gate_passed": bool(corr_ok),
                "fit": fit_stats,
                "delta_mean_abs": float(np.mean(np.abs(changed[use_mask]))) if use_mask.any() else 0.0,
                "delta_p95_abs": float(np.percentile(np.abs(changed[use_mask]), 95)) if use_mask.any() else 0.0,
                "delta_max_abs": float(np.max(np.abs(changed[use_mask]))) if use_mask.any() else 0.0,
            }
        )

    data["depth"] = patched_depth[..., None].astype(np.float32)
    depth_points = unproject_depth_map_to_point_map_numpy(
        data["depth"],
        np.asarray(data["extrinsic"], dtype=np.float32),
        np.asarray(data["intrinsic"], dtype=np.float32),
    )
    if "world_points" in data:
        data["world_points"] = np.asarray(data["world_points"], dtype=np.float32)
        data["world_points"][apply_masks] = depth_points[apply_masks]

    for conf_key in ("depth_conf", "world_points_conf"):
        if conf_key in data:
            conf = np.asarray(data[conf_key], dtype=np.float32)
            valid_values = conf[masks & np.isfinite(conf) & (conf > 0)]
            boost = float(np.percentile(valid_values, float(args.conf_boost_percentile))) if valid_values.size else float(args.conf_boost_percentile)
            conf[apply_masks] = np.maximum(conf[apply_masks], boost)
            data[conf_key] = conf

    out_npz = output_dir / "predictions.npz"
    np.savez_compressed(out_npz, **data)
    summary = {
        "task": "patch_predictions_with_external_view_depth",
        "truthful_status": "diagnostic_candidate_not_final_pass",
        "base_predictions": str(base_path.resolve()),
        "external_depth_npz": str(external_path.resolve()),
        "scene_dir": str(scene_dir.resolve()),
        "output_predictions": str(out_npz.resolve()),
        "align_roi_kind": args.align_roi_kind,
        "apply_roi_kind": args.apply_roi_kind,
        "fit_mode": args.fit_mode,
        "max_depth_delta": float(args.max_depth_delta),
        "min_fit_corr": float(args.min_fit_corr),
        "records": records,
        "notes": [
            "External depth is per-view projected before this patch; this avoids fitting Kinect 3D to the VGGT shell.",
            "This patch changes prediction tensors for local diagnosis only and must pass package_normal_candidate_gate plus visual review before any success claim.",
        ],
    }
    (output_dir / "patch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
