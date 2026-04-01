import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"

for root in (str(REPO_ROOT), str(TRAINING_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from training.data.datasets.zju_vggt_geom import ZjuVggtGeomDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Probe the ZJU VGGT-geom pseudo-supervision dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional training config name or path. When set, inherit ZJU dataset/source-policy settings from it.",
    )
    parser.add_argument(
        "--zju_dir",
        type=str,
        default="",
        help="Optional local ZJU root. Auto-detected when omitted.",
    )
    parser.add_argument("--seq_names", nargs="+", default=["CoreView_390"])
    parser.add_argument("--geom_subdir", type=str, default="vggt_geom")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument(
        "--aggregate_samples",
        type=int,
        default=1,
        help="Number of deterministic sample indices to probe starting from --sample_index.",
    )
    parser.add_argument(
        "--aggregate_stride",
        type=int,
        default=1,
        help="Stride between probed sample indices when --aggregate_samples > 1.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed both random and numpy for reproducible source-view sampling.")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--camera_source", type=str, default="gt", choices=["gt", "geom"])
    parser.add_argument("--mask_source", type=str, default="mask", choices=["none", "mask", "mask_cihp"])
    parser.add_argument(
        "--source_policy",
        type=str,
        default="random",
        choices=["random", "nearest_ring", "uniform_ring", "nearest_plus_uniform_tail"],
    )
    parser.add_argument("--source_view_pool", type=str, default="cached_only", choices=["cached_only", "geom_plus_raw"])
    parser.add_argument("--source_view_pool_train_probability", type=float, default=1.0)
    parser.add_argument("--source_anchor_policy", type=str, default="random", choices=["random", "max_depth_conf"])
    parser.add_argument("--min_supervised_views", type=int, default=1)
    parser.add_argument(
        "--supervised_view_quality_filter",
        type=str,
        default="none",
        choices=["none", "drop_worst_by_depth_conf_if_multi_supervised"],
    )
    parser.add_argument(
        "--conf_depth_view_quality_filter",
        type=str,
        default="none",
        choices=["none", "drop_worst_by_depth_conf_if_multi_supervised"],
    )
    parser.add_argument("--allow_duplicate_img", action="store_true", help="Allow duplicate view sampling in the probe.")
    parser.add_argument("--min_depth_conf", type=float, default=0.0)
    parser.add_argument("--holdout_stride", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="output/zju_vggt_geom_probe")
    return parser.parse_args()


def resolve_config_path(config_ref: str) -> Path:
    raw = str(config_ref or "").strip()
    if not raw:
        raise ValueError("config_ref must be non-empty")
    candidate = Path(raw)
    if candidate.is_file():
        return candidate.resolve()
    config_dir = TRAINING_ROOT / "config"
    named = config_dir / raw
    if named.is_file():
        return named.resolve()
    if not named.suffix:
        named_yaml = named.with_suffix(".yaml")
        if named_yaml.is_file():
            return named_yaml.resolve()
    raise FileNotFoundError(f"Unable to resolve config '{config_ref}' from {config_dir}")


def load_config_with_defaults(config_ref: str):
    config_dir = TRAINING_ROOT / "config"

    def _resolve_default_path(base_dir: Path, item) -> Path:
        if isinstance(item, str):
            rel = Path(item)
        elif isinstance(item, dict):
            key, value = next(iter(item.items()))
            rel = Path(str(key)) / str(value)
        else:
            raise TypeError(f"Unsupported defaults item: {item!r}")
        candidate = base_dir / rel
        if candidate.is_file():
            return candidate.resolve()
        if not candidate.suffix:
            candidate_yaml = candidate.with_suffix(".yaml")
            if candidate_yaml.is_file():
                return candidate_yaml.resolve()
        candidate = config_dir / rel
        if candidate.is_file():
            return candidate.resolve()
        if not candidate.suffix:
            candidate_yaml = candidate.with_suffix(".yaml")
            if candidate_yaml.is_file():
                return candidate_yaml.resolve()
        raise FileNotFoundError(f"Unable to resolve defaults item {item!r} from {base_dir}")

    def _load(path: Path, seen):
        resolved = path.resolve()
        if resolved in seen:
            raise RuntimeError(f"Recursive config defaults detected at {resolved}")
        seen = set(seen)
        seen.add(resolved)
        current = OmegaConf.load(resolved)
        defaults = list(current.get("defaults", []))
        merged = OmegaConf.create()
        include_self = False
        for item in defaults:
            if item == "_self_":
                include_self = True
                continue
            if isinstance(item, dict) and "_self_" in item:
                include_self = True
                continue
            child_path = _resolve_default_path(resolved.parent, item)
            merged = OmegaConf.merge(merged, _load(child_path, seen))
        current_without_defaults = OmegaConf.create({k: v for k, v in current.items() if k != "defaults"})
        if include_self or not defaults:
            merged = OmegaConf.merge(merged, current_without_defaults)
        return merged

    config_path = resolve_config_path(config_ref)
    return config_path, _load(config_path, set())


def apply_config_defaults(args):
    config_ref = str(getattr(args, "config", "") or "").strip()
    if not config_ref:
        return None

    config_path, cfg = load_config_with_defaults(config_ref)

    def _resolved(key, default=None):
        if key not in cfg:
            return default
        value = cfg[key]
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        return value

    field_map = {
        "zju_source_policy": "source_policy",
        "zju_source_view_pool": "source_view_pool",
        "zju_source_view_pool_train_probability": "source_view_pool_train_probability",
        "zju_source_anchor_policy": "source_anchor_policy",
        "zju_min_supervised_views": "min_supervised_views",
        "zju_supervised_view_quality_filter": "supervised_view_quality_filter",
        "zju_conf_depth_view_quality_filter": "conf_depth_view_quality_filter",
        "zju_min_depth_conf": "min_depth_conf",
        "zju_holdout_stride": "holdout_stride",
        "zju_geom_subdir": "geom_subdir",
        "zju_camera_source": "camera_source",
        "zju_mask_source": "mask_source",
    }
    for config_key, arg_key in field_map.items():
        value = _resolved(config_key, None)
        if value is not None:
            setattr(args, arg_key, value)

    seq_names = _resolved("zju_seq_names", None)
    if seq_names is not None:
        if isinstance(seq_names, str):
            args.seq_names = [item.strip() for item in seq_names.split(",") if item.strip()]
        else:
            args.seq_names = [str(item) for item in seq_names]

    zju_dir = _resolved("zju_dir", None)
    if zju_dir not in (None, "", "/YOUR/PATH/TO/ZJU"):
        args.zju_dir = str(zju_dir)

    return config_path


def resolve_zju_dir(requested, seq_names, geom_subdir):
    candidates = []
    requested = str(requested).strip()
    if requested:
        candidates.append(Path(requested))
    g_datasets = "G:\\" + chr(0x6570) + chr(0x636E) + chr(0x96C6)
    candidates.extend(
        [
            Path(r"F:\datasets\ZJU_MoCap\data\zju_mocap"),
            Path(g_datasets) / "datasets" / "ZJU_MoCap" / "data" / "zju_mocap",
        ]
    )

    geom_subdirs = [item.strip() for item in str(geom_subdir).split(",") if item.strip()]
    seen = set()
    best_candidate = None
    best_score = None
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            candidate = candidate.absolute()
        candidate_key = str(candidate).lower()
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if not candidate.is_dir():
            continue
        valid_subdir_count = 0
        total_frame_count = 0
        for seq_name in seq_names:
            for geom_subdir_name in geom_subdirs:
                geom_dir = candidate / str(seq_name) / geom_subdir_name
                if not geom_dir.is_dir():
                    continue
                frame_count = sum(1 for _ in geom_dir.glob("frame_*.npz"))
                if frame_count > 0:
                    valid_subdir_count += 1
                    total_frame_count += frame_count
        if valid_subdir_count <= 0:
            continue
        score = (int(valid_subdir_count), int(total_frame_count))
        if best_candidate is None or score > best_score:
            best_candidate = candidate
            best_score = score
    if best_candidate is not None:
        return best_candidate
    raise FileNotFoundError(
        f"Unable to resolve a ZJU root with {seq_names[0]}/{geom_subdir}/frame_*.npz from candidates: {candidates}"
    )


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


def save_mosaic(path, images_hwc):
    labeled = [Image.fromarray(np.asarray(img, dtype=np.uint8)) for img in images_hwc]
    width = max(im.width for im in labeled)
    height = max(im.height for im in labeled)
    cols = min(4, len(labeled))
    rows = (len(labeled) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * width, rows * height), color=(16, 16, 16))
    for idx, image in enumerate(labeled):
        row = idx // cols
        col = idx % cols
        canvas.paste(image, (col * width, row * height))
    canvas.save(path)


def summarize_point_cloud(points, written_vertex_count):
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return {
            "point_count_full": 0,
            "point_count_written": int(written_vertex_count),
            "downsampled_for_ply": False,
            "bbox_min": [0.0, 0.0, 0.0],
            "bbox_max": [0.0, 0.0, 0.0],
            "extent": [0.0, 0.0, 0.0],
            "centroid": [0.0, 0.0, 0.0],
            "radius_percentiles": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
            "axis_percentiles": {
                "x": {"p05": 0.0, "p50": 0.0, "p95": 0.0},
                "y": {"p05": 0.0, "p50": 0.0, "p95": 0.0},
                "z": {"p05": 0.0, "p50": 0.0, "p95": 0.0},
            },
        }

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    extent = bbox_max - bbox_min
    centroid = points.mean(axis=0)
    centered = points - centroid[None, :]
    radii = np.linalg.norm(centered, axis=1)

    def _axis_stats(axis_values):
        p05, p50, p95 = np.percentile(axis_values, [5.0, 50.0, 95.0])
        return {"p05": float(p05), "p50": float(p50), "p95": float(p95)}

    p50, p95, p99 = np.percentile(radii, [50.0, 95.0, 99.0])
    return {
        "point_count_full": int(points.shape[0]),
        "point_count_written": int(written_vertex_count),
        "downsampled_for_ply": bool(points.shape[0] != int(written_vertex_count)),
        "bbox_min": [float(x) for x in bbox_min.tolist()],
        "bbox_max": [float(x) for x in bbox_max.tolist()],
        "extent": [float(x) for x in extent.tolist()],
        "centroid": [float(x) for x in centroid.tolist()],
        "radius_percentiles": {"p50": float(p50), "p95": float(p95), "p99": float(p99)},
        "axis_percentiles": {
            "x": _axis_stats(points[:, 0]),
            "y": _axis_stats(points[:, 1]),
            "z": _axis_stats(points[:, 2]),
        },
    }


def export_sample_artifacts(output_dir, images, depths, flat_points, flat_colors):
    write_binary_ply(output_dir / "sample_world_points.ply", flat_points, flat_colors)
    save_mosaic(output_dir / "sample_images.png", images)

    depth_vis = []
    for depth in depths:
        valid = depth > 0
        if valid.any():
            p95 = np.percentile(depth[valid], 95.0)
            scaled = np.clip(depth / max(p95, 1e-6), 0.0, 1.0)
        else:
            scaled = np.zeros_like(depth)
        depth_vis.append((scaled * 255.0).astype(np.uint8))
    save_mosaic(output_dir / "sample_depths.png", [np.stack([d, d, d], axis=-1) for d in depth_vis])


def build_sample_payload(args, dataset, sample, masks, pointcloud_stats, zju_dir, config_path, sample_index):
    return {
        "config_path": str(config_path) if config_path is not None else "",
        "seed": int(args.seed),
        "probe_sample_index": int(sample_index),
        "dataset_len": len(dataset),
        "sequence_list_len": dataset.sequence_list_len,
        "sample_seq_name": sample["seq_name"],
        "ids": sample["ids"].tolist(),
        "num_images": len(sample["images"]),
        "available_view_count": int(sample.get("available_view_count", len(sample["images"]))),
        "image_shape": list(np.asarray(sample["images"][0]).shape),
        "depth_shape": list(np.asarray(sample["depths"][0]).shape),
        "valid_ratio": float(masks.mean()),
        "valid_points": int(masks.sum()),
        "pointcloud_stats": pointcloud_stats,
        "camera_source": args.camera_source,
        "mask_source": args.mask_source,
        "source_policy": args.source_policy,
        "requested_source_view_pool": str(sample.get("selection_requested_source_view_pool", args.source_view_pool)),
        "effective_source_view_pool": str(sample.get("selection_source_view_pool", args.source_view_pool)),
        "source_view_pool_train_probability": float(
            sample.get("selection_source_view_pool_train_probability", args.source_view_pool_train_probability)
        ),
        "rawpool_candidate_pool_used": bool(
            sample.get(
                "selection_rawpool_candidate_pool_used",
                sample.get("selection_source_view_pool", args.source_view_pool) == "geom_plus_raw",
            )
        ),
        "source_anchor_policy": str(sample.get("selection_source_anchor_policy", args.source_anchor_policy)),
        "min_supervised_views": int(sample.get("selection_min_supervised_views", args.min_supervised_views)),
        "supervised_view_quality_filter": str(
            sample.get("selection_supervised_view_quality_filter", args.supervised_view_quality_filter)
        ),
        "camera_names": list(sample.get("camera_names", [])),
        "candidate_supervised_camera_names": list(sample.get("candidate_supervised_camera_names", [])),
        "supervised_camera_names": list(sample.get("supervised_camera_names", [])),
        "source_only_camera_names": list(sample.get("source_only_camera_names", [])),
        "source_only_view_count": int(len(sample.get("source_only_camera_names", []))),
        "supervised_view_count": int(len(sample.get("supervised_camera_names", []))),
        "dropped_supervised_camera_names": list(sample.get("dropped_supervised_camera_names", [])),
        "conf_depth_camera_names": list(sample.get("conf_depth_camera_names", [])),
        "conf_depth_dropped_camera_names": list(sample.get("conf_depth_dropped_camera_names", [])),
        "supervised_view_quality_scores": dict(sample.get("supervised_view_quality_scores", {})),
        "geom_subdirs_present": list(sample.get("geom_subdirs_present", [])),
        "selection_anchor_camera": sample.get("selection_anchor_camera"),
        "available_candidate_view_count": int(
            sample.get("available_candidate_view_count", sample.get("available_view_count", len(sample["images"])))
        ),
        "available_candidate_camera_names": list(sample.get("available_candidate_camera_names", [])),
        "zju_dir": str(zju_dir),
        "geom_subdir": args.geom_subdir,
        "seq_names": args.seq_names,
    }


def write_sample_summary(output_dir, payload):
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# ZJU VGGT-Geom Dataset Probe",
                "",
                f"- dataset_len: `{payload['dataset_len']}`",
                f"- sequence_list_len: `{payload['sequence_list_len']}`",
                f"- probe_sample_index: `{payload['probe_sample_index']}`",
                f"- sample_seq_name: `{payload['sample_seq_name']}`",
                f"- ids: `{payload['ids']}`",
                f"- num_images: `{payload['num_images']}`",
                f"- available_view_count: `{payload['available_view_count']}`",
                f"- image_shape: `{payload['image_shape']}`",
                f"- depth_shape: `{payload['depth_shape']}`",
                f"- valid_ratio: `{payload['valid_ratio']:.4f}`",
                f"- valid_points: `{payload['valid_points']}`",
                f"- pointcloud_point_count_full: `{payload['pointcloud_stats']['point_count_full']}`",
                f"- pointcloud_point_count_written: `{payload['pointcloud_stats']['point_count_written']}`",
                f"- pointcloud_downsampled_for_ply: `{payload['pointcloud_stats']['downsampled_for_ply']}`",
                f"- pointcloud_centroid: `{payload['pointcloud_stats']['centroid']}`",
                f"- pointcloud_extent: `{payload['pointcloud_stats']['extent']}`",
                f"- pointcloud_radius_percentiles: `{payload['pointcloud_stats']['radius_percentiles']}`",
                f"- pointcloud_axis_percentiles: `{payload['pointcloud_stats']['axis_percentiles']}`",
                f"- camera_source: `{payload['camera_source']}`",
                f"- mask_source: `{payload['mask_source']}`",
                f"- source_policy: `{payload['source_policy']}`",
                f"- requested_source_view_pool: `{payload['requested_source_view_pool']}`",
                f"- effective_source_view_pool: `{payload['effective_source_view_pool']}`",
                f"- source_view_pool_train_probability: `{payload['source_view_pool_train_probability']}`",
                f"- rawpool_candidate_pool_used: `{payload['rawpool_candidate_pool_used']}`",
                f"- source_anchor_policy: `{payload['source_anchor_policy']}`",
                f"- min_supervised_views: `{payload['min_supervised_views']}`",
                f"- supervised_view_quality_filter: `{payload['supervised_view_quality_filter']}`",
                f"- camera_names: `{payload['camera_names']}`",
                f"- candidate_supervised_camera_names: `{payload['candidate_supervised_camera_names']}`",
                f"- supervised_camera_names: `{payload['supervised_camera_names']}`",
                f"- source_only_camera_names: `{payload['source_only_camera_names']}`",
                f"- supervised_view_count: `{payload['supervised_view_count']}`",
                f"- source_only_view_count: `{payload['source_only_view_count']}`",
                f"- dropped_supervised_camera_names: `{payload['dropped_supervised_camera_names']}`",
                f"- conf_depth_camera_names: `{payload['conf_depth_camera_names']}`",
                f"- conf_depth_dropped_camera_names: `{payload['conf_depth_dropped_camera_names']}`",
                f"- supervised_view_quality_scores: `{payload['supervised_view_quality_scores']}`",
                f"- geom_subdirs_present: `{payload['geom_subdirs_present']}`",
                f"- selection_anchor_camera: `{payload['selection_anchor_camera']}`",
                f"- available_candidate_view_count: `{payload['available_candidate_view_count']}`",
                f"- available_candidate_camera_names: `{payload['available_candidate_camera_names']}`",
                "",
                "## Files",
                "",
                "- `sample_images.png`",
                "- `sample_depths.png`",
                "- `sample_world_points.ply`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_aggregate_summary(args, config_path, zju_dir, payloads):
    valid_ratios = np.asarray([item["valid_ratio"] for item in payloads], dtype=np.float64)
    valid_points = np.asarray([item["valid_points"] for item in payloads], dtype=np.float64)
    supervised_counts = np.asarray([item["supervised_view_count"] for item in payloads], dtype=np.float64)
    source_only_counts = np.asarray([item["source_only_view_count"] for item in payloads], dtype=np.float64)
    extents = np.asarray([item["pointcloud_stats"]["extent"] for item in payloads], dtype=np.float64)
    radius_p95 = np.asarray(
        [item["pointcloud_stats"]["radius_percentiles"]["p95"] for item in payloads], dtype=np.float64
    )

    anchor_histogram = {}
    for item in payloads:
        anchor = str(item.get("selection_anchor_camera") or "")
        anchor_histogram[anchor] = anchor_histogram.get(anchor, 0) + 1

    return {
        "config_path": str(config_path) if config_path is not None else "",
        "seed": int(args.seed),
        "aggregate_samples": int(len(payloads)),
        "aggregate_stride": int(args.aggregate_stride),
        "sample_indices": [int(item["probe_sample_index"]) for item in payloads],
        "source_policy": args.source_policy,
        "camera_source": args.camera_source,
        "mask_source": args.mask_source,
        "zju_dir": str(zju_dir),
        "geom_subdir": args.geom_subdir,
        "seq_names": args.seq_names,
        "valid_ratio": {
            "mean": float(valid_ratios.mean()),
            "median": float(np.median(valid_ratios)),
            "min": float(valid_ratios.min()),
            "max": float(valid_ratios.max()),
        },
        "valid_points": {
            "mean": float(valid_points.mean()),
            "median": float(np.median(valid_points)),
            "min": int(valid_points.min()),
            "max": int(valid_points.max()),
        },
        "supervised_view_count": {
            "mean": float(supervised_counts.mean()),
            "min": int(supervised_counts.min()),
            "max": int(supervised_counts.max()),
        },
        "source_only_view_count": {
            "mean": float(source_only_counts.mean()),
            "min": int(source_only_counts.min()),
            "max": int(source_only_counts.max()),
        },
        "pointcloud_extent": {
            "mean": [float(x) for x in extents.mean(axis=0).tolist()],
            "median": [float(x) for x in np.median(extents, axis=0).tolist()],
            "max": [float(x) for x in extents.max(axis=0).tolist()],
        },
        "pointcloud_radius_p95": {
            "mean": float(radius_p95.mean()),
            "median": float(np.median(radius_p95)),
            "min": float(radius_p95.min()),
            "max": float(radius_p95.max()),
        },
        "selection_anchor_camera_histogram": anchor_histogram,
        "sample_seq_names": [str(item["sample_seq_name"]) for item in payloads],
    }


def write_aggregate_summary(output_dir, aggregate):
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "aggregate_summary.md").write_text(
        "\n".join(
            [
                "# ZJU VGGT-Geom Dataset Aggregate Probe",
                "",
                f"- aggregate_samples: `{aggregate['aggregate_samples']}`",
                f"- aggregate_stride: `{aggregate['aggregate_stride']}`",
                f"- sample_indices: `{aggregate['sample_indices']}`",
                f"- source_policy: `{aggregate['source_policy']}`",
                f"- valid_ratio_mean: `{aggregate['valid_ratio']['mean']:.6f}`",
                f"- valid_ratio_median: `{aggregate['valid_ratio']['median']:.6f}`",
                f"- valid_ratio_minmax: `[{aggregate['valid_ratio']['min']:.6f}, {aggregate['valid_ratio']['max']:.6f}]`",
                f"- valid_points_mean: `{aggregate['valid_points']['mean']:.2f}`",
                f"- supervised_view_count_mean: `{aggregate['supervised_view_count']['mean']:.2f}`",
                f"- source_only_view_count_mean: `{aggregate['source_only_view_count']['mean']:.2f}`",
                f"- pointcloud_extent_mean: `{aggregate['pointcloud_extent']['mean']}`",
                f"- pointcloud_extent_median: `{aggregate['pointcloud_extent']['median']}`",
                f"- pointcloud_radius_p95_mean: `{aggregate['pointcloud_radius_p95']['mean']:.6f}`",
                f"- pointcloud_radius_p95_median: `{aggregate['pointcloud_radius_p95']['median']:.6f}`",
                f"- selection_anchor_camera_histogram: `{aggregate['selection_anchor_camera_histogram']}`",
                "",
                "## Files",
                "",
                "- `summary.json` / `summary.md` still describe the first sampled example",
                "- `per_sample_summaries.json` stores the deterministic sample basket",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    config_path = apply_config_defaults(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    zju_dir = resolve_zju_dir(args.zju_dir, args.seq_names, args.geom_subdir)

    common_conf = OmegaConf.create(
        {
            "debug": False,
            "training": args.split == "train",
            "inside_random": False,
            "allow_duplicate_img": bool(args.allow_duplicate_img),
            "load_depth": True,
            "img_size": 518,
            "patch_size": 14,
            "rescale": True,
            "rescale_aug": False,
            "landscape_check": False,
            "augs": {
                "scales": None,
            },
        }
    )

    dataset = ZjuVggtGeomDataset(
        common_conf=common_conf,
        split=args.split,
        ZJU_DIR=str(zju_dir),
        seq_names=args.seq_names,
        geom_subdir=args.geom_subdir,
        holdout_stride=args.holdout_stride,
        camera_source=args.camera_source,
        mask_source=args.mask_source,
        source_policy=args.source_policy,
        source_view_pool=args.source_view_pool,
        source_view_pool_train_probability=args.source_view_pool_train_probability,
        source_anchor_policy=args.source_anchor_policy,
        min_supervised_views=args.min_supervised_views,
        supervised_view_quality_filter=args.supervised_view_quality_filter,
        conf_depth_view_quality_filter=args.conf_depth_view_quality_filter,
        min_depth_conf=args.min_depth_conf,
        len_train=-1,
        len_test=-1,
    )
    sample_count = max(1, int(args.aggregate_samples))
    sample_stride = max(1, int(args.aggregate_stride))
    sample_indices = []
    for offset in range(sample_count):
        idx = int(args.sample_index) + offset * sample_stride
        if idx >= len(dataset):
            break
        sample_indices.append(idx)
    if not sample_indices:
        raise IndexError(f"No valid sample indices available from start={args.sample_index} with stride={sample_stride}")

    payloads = []
    first_sample_summary_path = output_dir / "summary.md"
    for loop_idx, sample_index in enumerate(sample_indices):
        sample = dataset.get_data(seq_index=sample_index, img_per_seq=args.num_images)
        images = np.stack(sample["images"]).astype(np.uint8)
        depths = np.stack(sample["depths"]).astype(np.float32)
        world_points = np.stack(sample["world_points"]).astype(np.float32)
        masks = np.stack(sample["point_masks"]).astype(bool)

        flat_points_full = world_points.reshape(-1, 3)[masks.reshape(-1)]
        flat_colors_full = images.reshape(-1, 3)[masks.reshape(-1)]
        flat_points = flat_points_full
        flat_colors = flat_colors_full
        if flat_points.shape[0] > 250000:
            keep = np.linspace(0, flat_points.shape[0] - 1, 250000, dtype=np.int64)
            flat_points = flat_points[keep]
            flat_colors = flat_colors[keep]

        pointcloud_stats = summarize_point_cloud(flat_points_full, written_vertex_count=flat_points.shape[0])
        payload = build_sample_payload(
            args=args,
            dataset=dataset,
            sample=sample,
            masks=masks,
            pointcloud_stats=pointcloud_stats,
            zju_dir=zju_dir,
            config_path=config_path,
            sample_index=sample_index,
        )
        payloads.append(payload)

        if loop_idx == 0:
            export_sample_artifacts(output_dir, images, depths, flat_points, flat_colors)
            write_sample_summary(output_dir, payload)

    (output_dir / "per_sample_summaries.json").write_text(
        json.dumps(payloads, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    aggregate = build_aggregate_summary(args=args, config_path=config_path, zju_dir=zju_dir, payloads=payloads)
    write_aggregate_summary(output_dir, aggregate)

    print(first_sample_summary_path)


if __name__ == "__main__":
    main()
