from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from build_mesh_raycast_training_case import _world_to_cam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse two dense head/face teacher target NPZ files. The primary "
            "teacher is kept wherever it is depth-compatible with the same "
            "protocol reference prediction; the fallback teacher fills only "
            "the remaining depth-compatible holes."
        )
    )
    parser.add_argument("--primary-npz", required=True)
    parser.add_argument("--fallback-npz", required=True)
    parser.add_argument("--predictions-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--depth-tolerance", type=float, default=0.06)
    parser.add_argument("--primary-name", default="primary")
    parser.add_argument("--fallback-name", default="fallback")
    parser.add_argument("--mask-key", default="teacher_mask")
    return parser.parse_args()


def load_teacher(path: Path, *, mask_key: str, extrinsic: np.ndarray) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    if "world_points" not in payload.files:
        raise KeyError(f"{path} has no world_points")
    world = np.asarray(payload["world_points"], dtype=np.float32)
    if mask_key in payload.files:
        mask = np.asarray(payload[mask_key], dtype=bool)
    elif "teacher_mask" in payload.files:
        mask = np.asarray(payload["teacher_mask"], dtype=bool)
    elif "roi_mask" in payload.files:
        mask = np.asarray(payload["roi_mask"], dtype=bool)
    else:
        mask = np.isfinite(world).all(axis=-1)
    if "depths" in payload.files:
        depth = np.asarray(payload["depths"], dtype=np.float32)
    elif "depth" in payload.files:
        depth = np.asarray(payload["depth"], dtype=np.float32)
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
    else:
        depths = []
        for view_idx in range(world.shape[0]):
            cam = _world_to_cam(world[view_idx].reshape(-1, 3), extrinsic[view_idx])
            depths.append(cam[:, 2].reshape(world.shape[1:3]))
        depth = np.stack(depths, axis=0).astype(np.float32)
    roi = np.asarray(payload["roi_mask"], dtype=bool) if "roi_mask" in payload.files else mask.copy()
    finite = np.isfinite(world).all(axis=-1) & np.isfinite(depth) & (depth > 0.05)
    return {"world": world, "depth": depth, "mask": mask & finite, "roi": roi}


def depth_compatible_mask(teacher: dict[str, np.ndarray], anchor_depth: np.ndarray, tolerance: float) -> np.ndarray:
    residual = np.abs(teacher["depth"] - anchor_depth)
    return teacher["mask"] & np.isfinite(residual) & (residual <= float(tolerance))


def summarize_selection(
    *,
    primary_ok: np.ndarray,
    fallback_fill: np.ndarray,
    anchor_depth: np.ndarray,
    fused_depth: np.ndarray,
    source_label: np.ndarray,
) -> dict[str, Any]:
    entries = []
    for view_idx in range(source_label.shape[0]):
        selected = source_label[view_idx] > 0
        residual = np.abs(fused_depth[view_idx][selected] - anchor_depth[view_idx][selected])
        residual = residual[np.isfinite(residual)]
        stats = {"p50": None, "p90": None, "p95": None}
        if residual.size:
            p50, p90, p95 = np.percentile(residual, [50, 90, 95])
            stats = {"p50": float(p50), "p90": float(p90), "p95": float(p95)}
        entries.append(
            {
                "view_index": int(view_idx),
                "primary_pixels": int(primary_ok[view_idx].sum()),
                "fallback_fill_pixels": int(fallback_fill[view_idx].sum()),
                "selected_pixels": int(selected.sum()),
                "fallback_fraction": float(fallback_fill[view_idx].sum() / max(int(selected.sum()), 1)),
                "depth_residual": stats,
            }
        )
    return {
        "entries": entries,
        "total_primary_pixels": int(primary_ok.sum()),
        "total_fallback_fill_pixels": int(fallback_fill.sum()),
        "total_selected_pixels": int((source_label > 0).sum()),
        "fallback_fraction": float(fallback_fill.sum() / max(int((source_label > 0).sum()), 1)),
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = Path(args.predictions_npz).resolve()
    with np.load(predictions_path, allow_pickle=False) as pred:
        extrinsic = np.asarray(pred["extrinsic"], dtype=np.float32)
        anchor_depth = np.asarray(pred["depth"], dtype=np.float32)
        if anchor_depth.ndim == 4 and anchor_depth.shape[-1] == 1:
            anchor_depth = anchor_depth[..., 0]
    primary = load_teacher(Path(args.primary_npz).resolve(), mask_key=str(args.mask_key), extrinsic=extrinsic)
    fallback = load_teacher(Path(args.fallback_npz).resolve(), mask_key=str(args.mask_key), extrinsic=extrinsic)
    if primary["world"].shape != fallback["world"].shape or primary["world"].shape[:3] != anchor_depth.shape:
        raise ValueError(
            "teacher/reference shapes do not match: "
            f"primary={primary['world'].shape}, fallback={fallback['world'].shape}, depth={anchor_depth.shape}"
        )
    primary_ok = depth_compatible_mask(primary, anchor_depth, float(args.depth_tolerance))
    fallback_ok = depth_compatible_mask(fallback, anchor_depth, float(args.depth_tolerance))
    fallback_fill = (~primary_ok) & fallback_ok
    fused_world = np.zeros_like(primary["world"], dtype=np.float32)
    fused_depth = np.zeros_like(primary["depth"], dtype=np.float32)
    fused_mask = primary_ok | fallback_fill
    source_label = np.zeros(fused_mask.shape, dtype=np.uint8)
    fused_world[primary_ok] = primary["world"][primary_ok]
    fused_depth[primary_ok] = primary["depth"][primary_ok]
    source_label[primary_ok] = 1
    fused_world[fallback_fill] = fallback["world"][fallback_fill]
    fused_depth[fallback_fill] = fallback["depth"][fallback_fill]
    source_label[fallback_fill] = 2
    roi_mask = primary["roi"] | fallback["roi"] | fused_mask
    out_path = output_dir / "teacher_targets.npz"
    np.savez_compressed(
        out_path,
        world_points=fused_world.astype(np.float32),
        depths=fused_depth.astype(np.float32),
        teacher_mask=fused_mask.astype(bool),
        roi_mask=roi_mask.astype(bool),
        source_label=source_label,
    )
    summary = {
        "primary_npz": str(Path(args.primary_npz).resolve()),
        "fallback_npz": str(Path(args.fallback_npz).resolve()),
        "predictions_npz": str(predictions_path),
        "teacher_targets": str(out_path),
        "depth_tolerance": float(args.depth_tolerance),
        "primary_name": str(args.primary_name),
        "fallback_name": str(args.fallback_name),
        "source_label": {"0": "none", "1": str(args.primary_name), "2": str(args.fallback_name)},
        "selection": summarize_selection(
            primary_ok=primary_ok,
            fallback_fill=fallback_fill,
            anchor_depth=anchor_depth,
            fused_depth=fused_depth,
            source_label=source_label,
        ),
        "truthful_note": (
            "Hybrid teacher construction only. It must pass audit_headface_teacher_surface.py "
            "before any local one-frame overfit smoke."
        ),
    }
    (output_dir / "hybrid_teacher_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
