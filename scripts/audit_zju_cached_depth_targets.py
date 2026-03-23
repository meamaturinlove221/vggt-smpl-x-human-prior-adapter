import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"

for root in (str(REPO_ROOT), str(TRAINING_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from scripts.compare_geometry_branches_zju_report import detect_local_zju_root
from scripts.zju_geometry_region_utils import load_target_mask, save_json


BUCKET_ORDER = ("fg_human", "fg_edge", "fg_bottom_20pct", "fg_nonbottom")
PERCENTILES = (5, 25, 40, 50, 60, 75, 95)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit cached pseudo depth/depth_conf targets for ZJU and recommend a zju_min_depth_conf threshold."
    )
    parser.add_argument("--config", default="zju_vggt_geom_minimal")
    parser.add_argument("--zju_dir", default="")
    parser.add_argument("--seq_name", default="CoreView_390")
    parser.add_argument("--geom_subdir", default="vggt_geom")
    parser.add_argument("--mask_source", default="mask", choices=["none", "mask", "mask_cihp"])
    parser.add_argument("--edge_px", type=int, default=5)
    parser.add_argument("--bottom_band_ratio", type=float, default=0.2)
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--holdout_stride", type=int, default=10)
    parser.add_argument("--num_batches", type=int, default=8)
    parser.add_argument("--min_valid_pixels", type=int, default=100)
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def resolve_output_dir(output_dir):
    if output_dir:
        return ensure_dir(output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path("output") / f"zju_cached_depth_target_audit_{stamp}")


def resolve_zju_dir(zju_dir):
    if zju_dir:
        return Path(zju_dir).resolve()
    return detect_local_zju_root().resolve()


def build_audit_bucket_masks(fg_mask, edge_px, bottom_band_ratio):
    fg_mask = np.asarray(fg_mask, dtype=bool)
    edge_px = max(int(edge_px), 0)
    if edge_px > 0:
        kernel = np.ones((edge_px * 2 + 1, edge_px * 2 + 1), dtype=np.uint8)
        eroded = cv2.erode(fg_mask.astype(np.uint8), kernel, iterations=1) > 0
    else:
        eroded = fg_mask.copy()

    height = fg_mask.shape[0]
    bottom_start = int(math.floor(height * (1.0 - float(np.clip(bottom_band_ratio, 0.0, 1.0)))))
    bottom_start = max(0, min(height, bottom_start))
    bottom_mask = np.zeros_like(fg_mask, dtype=bool)
    bottom_mask[bottom_start:, :] = True

    return {
        "fg_human": fg_mask,
        "fg_edge": fg_mask & (~eroded),
        "fg_bottom_20pct": fg_mask & bottom_mask,
        "fg_nonbottom": fg_mask & (~bottom_mask),
    }


def init_bucket_stats():
    return {
        bucket_name: {
            "pixel_count": 0,
            "sum_conf": 0.0,
            "sum_sq_conf": 0.0,
            "max_conf": 0.0,
        }
        for bucket_name in BUCKET_ORDER
    }


def init_bucket_histograms(num_bins):
    return {
        bucket_name: np.zeros((num_bins,), dtype=np.int64)
        for bucket_name in BUCKET_ORDER
    }


def summarize_percentiles_from_hist(hist, bin_edges):
    total = int(hist.sum())
    if total == 0:
        return {f"p{percentile}": None for percentile in PERCENTILES}
    cdf = np.cumsum(hist)
    summary = {}
    for percentile in PERCENTILES:
        target = (percentile / 100.0) * max(total - 1, 0)
        index = int(np.searchsorted(cdf, target, side="right"))
        index = max(0, min(index, len(bin_edges) - 2))
        summary[f"p{percentile}"] = float((bin_edges[index] + bin_edges[index + 1]) * 0.5)
    return summary


def keep_count_from_hist(hist, bin_edges, threshold):
    index = int(np.searchsorted(bin_edges, threshold, side="right") - 1)
    index = max(0, min(index, len(hist) - 1))
    return int(hist[index:].sum())


def scan_cached_targets(seq_dir, geom_subdir, mask_source, edge_px, bottom_band_ratio, num_bins=4096):
    geom_dir = Path(seq_dir) / geom_subdir
    geom_paths = sorted(geom_dir.glob("frame_*.npz"))
    if not geom_paths:
        raise FileNotFoundError(f"No cached geometry frames found under {geom_dir}")

    stats = init_bucket_stats()
    frame_count = 0
    view_count = 0
    total_pixels = 0

    for geom_path in geom_paths:
        frame_id = int(geom_path.stem.split("_")[-1])
        geom = np.load(geom_path, allow_pickle=True)
        depth_conf = np.asarray(geom["depth_conf"], dtype=np.float32)
        cam_names = [str(name) for name in geom["cam_names"]]
        target_hw = depth_conf.shape[1:3]
        frame_count += 1
        for local_idx, cam_name in enumerate(cam_names):
            fg_mask = load_target_mask(
                seq_dir=seq_dir,
                camera_name=cam_name,
                frame_id=frame_id,
                target_hw=target_hw,
                mask_source=mask_source,
            )
            bucket_masks = build_audit_bucket_masks(fg_mask, edge_px=edge_px, bottom_band_ratio=bottom_band_ratio)
            conf_map = depth_conf[local_idx]
            total_pixels += int(conf_map.size)
            view_count += 1
            for bucket_name, bucket_mask in bucket_masks.items():
                values = conf_map[bucket_mask]
                if values.size == 0:
                    continue
                stats[bucket_name]["pixel_count"] += int(values.size)
                stats[bucket_name]["sum_conf"] += float(values.sum(dtype=np.float64))
                stats[bucket_name]["sum_sq_conf"] += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
                stats[bucket_name]["max_conf"] = max(stats[bucket_name]["max_conf"], float(values.max()))

    hist_max = max(1.0, max(bucket["max_conf"] for bucket in stats.values()))
    histograms = init_bucket_histograms(num_bins)
    bin_edges = np.linspace(0.0, hist_max, num_bins + 1, dtype=np.float64)

    for geom_path in geom_paths:
        frame_id = int(geom_path.stem.split("_")[-1])
        geom = np.load(geom_path, allow_pickle=True)
        depth_conf = np.asarray(geom["depth_conf"], dtype=np.float32)
        cam_names = [str(name) for name in geom["cam_names"]]
        target_hw = depth_conf.shape[1:3]
        for local_idx, cam_name in enumerate(cam_names):
            fg_mask = load_target_mask(
                seq_dir=seq_dir,
                camera_name=cam_name,
                frame_id=frame_id,
                target_hw=target_hw,
                mask_source=mask_source,
            )
            bucket_masks = build_audit_bucket_masks(fg_mask, edge_px=edge_px, bottom_band_ratio=bottom_band_ratio)
            conf_map = depth_conf[local_idx]
            for bucket_name, bucket_mask in bucket_masks.items():
                values = conf_map[bucket_mask]
                if values.size == 0:
                    continue
                histograms[bucket_name] += np.histogram(values, bins=bin_edges)[0]

    bucket_payload = {}
    for bucket_name in BUCKET_ORDER:
        bucket_stats = stats[bucket_name]
        pixel_count = int(bucket_stats["pixel_count"])
        mean_conf = None
        std_conf = None
        if pixel_count > 0:
            mean_conf = float(bucket_stats["sum_conf"] / pixel_count)
            variance = max(bucket_stats["sum_sq_conf"] / pixel_count - mean_conf * mean_conf, 0.0)
            std_conf = float(math.sqrt(variance))
        bucket_payload[bucket_name] = {
            "pixel_count": pixel_count,
            "pixel_ratio_vs_total_pixels": float(pixel_count / max(total_pixels, 1)),
            "mean_conf": mean_conf,
            "std_conf": std_conf,
            "max_conf": float(bucket_stats["max_conf"]),
            "percentiles": summarize_percentiles_from_hist(histograms[bucket_name], bin_edges),
        }
    return {
        "geom_dir": str(geom_dir.resolve()),
        "frame_count": frame_count,
        "view_count": view_count,
        "total_pixels": int(total_pixels),
        "histogram_bins": int(num_bins),
        "histogram_max": float(hist_max),
        "bucket_stats": bucket_payload,
        "histograms": histograms,
        "bin_edges": bin_edges,
    }


def build_cfg(args, zju_dir, min_depth_conf):
    overrides = [
        f"zju_dir='{str(zju_dir).replace(chr(92), '/')}'",
        f"zju_seq_names='{args.seq_name}'",
        f"zju_geom_subdir='{args.geom_subdir}'",
        f"zju_mask_source='{args.mask_source}'",
        f"zju_min_depth_conf={float(min_depth_conf)}",
        f"zju_holdout_stride={int(args.holdout_stride)}",
        f"data.train.common_config.fix_img_num={int(args.num_images)}",
        f"data.val.common_config.fix_img_num={int(args.num_images)}",
        "data.train.common_config.fix_aspect_ratio=1.0",
        "data.val.common_config.fix_aspect_ratio=1.0",
        "data.train.common_config.allow_duplicate_img=False",
        "data.val.common_config.allow_duplicate_img=False",
        "data.train.common_config.load_depth=True",
        "data.val.common_config.load_depth=True",
        "data.train.num_workers=0",
        "data.val.num_workers=0",
        "num_workers=0",
        "model.enable_camera=True",
        "model.enable_depth=True",
        "model.enable_point=False",
        "model.enable_track=False",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(TRAINING_ROOT / "config")):
        return compose(config_name=args.config, overrides=overrides)


def collect_valid_pixel_counts(cfg, split, num_batches):
    dataset = instantiate(cfg.data.val if split == "val" else cfg.data.train, _recursive_=False)
    dataset.seed = cfg.seed_value
    loader = dataset.get_loader(epoch=0)
    counts = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break
        point_masks = batch["point_masks"]
        count = int(point_masks.sum().item()) if hasattr(point_masks, "sum") else int(np.asarray(point_masks).sum())
        counts.append(count)
    return counts


def summarize_valid_pixel_counts(counts, min_valid_pixels):
    pass_count = sum(1 for count in counts if count >= min_valid_pixels)
    return {
        "batch_count": len(counts),
        "counts": counts,
        "min_valid_pixels": int(min_valid_pixels),
        "mean_valid_pixels": float(sum(counts) / max(len(counts), 1)),
        "min_valid_pixels_observed": int(min(counts) if counts else 0),
        "max_valid_pixels_observed": int(max(counts) if counts else 0),
        "pass_batches": int(pass_count),
        "pass_fraction": float(pass_count / max(len(counts), 1)),
        "passes_all_batches": bool(counts) and all(count >= min_valid_pixels for count in counts),
    }


def choose_threshold(candidate_payload):
    sorted_rows = sorted(candidate_payload, key=lambda row: float(row["value"]), reverse=True)
    for row in sorted_rows:
        if row["train"]["passes_all_batches"] and row["val"]["passes_all_batches"]:
            return {
                "label": row["label"],
                "value": float(row["value"]),
            }
    return None


def main():
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    zju_dir = resolve_zju_dir(args.zju_dir)
    seq_dir = zju_dir / args.seq_name

    scan_payload = scan_cached_targets(
        seq_dir=seq_dir,
        geom_subdir=args.geom_subdir,
        mask_source=args.mask_source,
        edge_px=args.edge_px,
        bottom_band_ratio=args.bottom_band_ratio,
    )

    fg_human_percentiles = scan_payload["bucket_stats"]["fg_human"]["percentiles"]
    candidate_labels = ("p40", "p50", "p60")
    candidates = []
    for label in candidate_labels:
        threshold_value = fg_human_percentiles[label]
        cfg = build_cfg(args, zju_dir=zju_dir, min_depth_conf=threshold_value)
        train_counts = collect_valid_pixel_counts(cfg, "train", args.num_batches)
        val_counts = collect_valid_pixel_counts(cfg, "val", args.num_batches)
        candidates.append(
            {
                "label": label,
                "value": float(threshold_value),
                "train": summarize_valid_pixel_counts(train_counts, args.min_valid_pixels),
                "val": summarize_valid_pixel_counts(val_counts, args.min_valid_pixels),
            }
        )

    selected_threshold = choose_threshold(candidates)
    if selected_threshold is None:
        conclusion = (
            "No fg_human percentile candidate kept >=100 valid pixels in every sampled train/val batch. "
            "Stop at diagnosis; do not launch target-threshold training yet."
        )
    else:
        conclusion = (
            f"Selected {selected_threshold['label']}={selected_threshold['value']:.4f} as the highest threshold "
            "that preserved >=100 valid pixels in every sampled train/val batch."
        )

    histograms_serialized = {
        bucket_name: scan_payload["histograms"][bucket_name].tolist()
        for bucket_name in BUCKET_ORDER
    }
    payload = {
        "config": args.config,
        "zju_dir": str(zju_dir.resolve()),
        "seq_name": args.seq_name,
        "geom_subdir": args.geom_subdir,
        "mask_source": args.mask_source,
        "edge_px": int(args.edge_px),
        "bottom_band_ratio": float(args.bottom_band_ratio),
        "num_images": int(args.num_images),
        "holdout_stride": int(args.holdout_stride),
        "num_batches": int(args.num_batches),
        "min_valid_pixels": int(args.min_valid_pixels),
        "scan": {
            "geom_dir": scan_payload["geom_dir"],
            "frame_count": int(scan_payload["frame_count"]),
            "view_count": int(scan_payload["view_count"]),
            "total_pixels": int(scan_payload["total_pixels"]),
            "histogram_bins": int(scan_payload["histogram_bins"]),
            "histogram_max": float(scan_payload["histogram_max"]),
            "bucket_stats": scan_payload["bucket_stats"],
            "bin_edges": scan_payload["bin_edges"].tolist(),
            "histograms": histograms_serialized,
        },
        "threshold_candidates": candidates,
        "selected_threshold": selected_threshold,
        "conclusion": conclusion,
    }

    save_json(output_dir / "cached_depth_target_audit.json", payload)

    lines = [
        "# ZJU Cached Depth Target Audit",
        "",
        f"- zju_dir: `{payload['zju_dir']}`",
        f"- seq_name: `{payload['seq_name']}`",
        f"- geom_subdir: `{payload['geom_subdir']}`",
        f"- mask_source: `{payload['mask_source']}`",
        f"- edge_px: `{payload['edge_px']}`",
        f"- bottom_band_ratio: `{payload['bottom_band_ratio']}`",
        f"- frame_count: `{payload['scan']['frame_count']}`",
        f"- view_count: `{payload['scan']['view_count']}`",
        f"- total_pixels: `{payload['scan']['total_pixels']}`",
        f"- num_batches: `{payload['num_batches']}`",
        f"- min_valid_pixels: `{payload['min_valid_pixels']}`",
        "",
        "## Buckets",
        "",
        "| Bucket | Pixels | Ratio vs Total | Mean Conf | Std Conf | P40 | P50 | P60 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for bucket_name in BUCKET_ORDER:
        bucket = payload["scan"]["bucket_stats"][bucket_name]
        percentiles = bucket["percentiles"]
        lines.append(
            "| `{bucket}` | {pixels} | {ratio:.4f} | {mean} | {std} | {p40} | {p50} | {p60} |".format(
                bucket=bucket_name,
                pixels=bucket["pixel_count"],
                ratio=bucket["pixel_ratio_vs_total_pixels"],
                mean="n/a" if bucket["mean_conf"] is None else f"{bucket['mean_conf']:.4f}",
                std="n/a" if bucket["std_conf"] is None else f"{bucket['std_conf']:.4f}",
                p40="n/a" if percentiles["p40"] is None else f"{percentiles['p40']:.4f}",
                p50="n/a" if percentiles["p50"] is None else f"{percentiles['p50']:.4f}",
                p60="n/a" if percentiles["p60"] is None else f"{percentiles['p60']:.4f}",
            )
        )

    lines.extend(
        [
            "",
            "## Threshold Candidates",
            "",
            "| Label | Value | Train Min Valid | Train Pass All | Val Min Valid | Val Pass All |",
            "| --- | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for row in candidates:
        lines.append(
            "| `{label}` | {value:.4f} | {train_min} | `{train_pass}` | {val_min} | `{val_pass}` |".format(
                label=row["label"],
                value=row["value"],
                train_min=row["train"]["min_valid_pixels_observed"],
                train_pass=row["train"]["passes_all_batches"],
                val_min=row["val"]["min_valid_pixels_observed"],
                val_pass=row["val"]["passes_all_batches"],
            )
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- selected_threshold: `{payload['selected_threshold']}`",
            f"- conclusion: {payload['conclusion']}",
        ]
    )
    (output_dir / "cached_depth_target_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] Wrote {output_dir / 'cached_depth_target_audit.md'}")


if __name__ == "__main__":
    main()
