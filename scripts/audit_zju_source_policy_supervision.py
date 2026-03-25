import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"

for root in (str(REPO_ROOT), str(TRAINING_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from training.data.datasets.zju_vggt_geom import ZjuVggtGeomDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit supervision eligibility for a ZJU source-policy dataset config."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Training config path to audit.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to audit.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=32,
        help="How many sequential dataset samples to audit.",
    )
    parser.add_argument(
        "--zju_dir",
        type=str,
        default="",
        help="Optional explicit ZJU root. Auto-detected when omitted or placeholder.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for summary artifacts.",
    )
    return parser.parse_args()


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def resolve_zju_dir(requested, seq_names, geom_subdir):
    requested = str(requested).strip()
    if requested and "YOUR/PATH/TO/ZJU" not in requested:
        candidate = Path(requested)
        if candidate.is_dir():
            return candidate.resolve()

    candidates = []
    g_datasets = "G:\\" + chr(0x6570) + chr(0x636E) + chr(0x96C6)
    candidates.extend(
        [
            Path(r"F:\datasets\ZJU_MoCap\data\zju_mocap"),
            Path(g_datasets) / "datasets" / "ZJU_MoCap" / "data" / "zju_mocap",
        ]
    )

    geom_subdirs = ensure_list(geom_subdir)
    best_candidate = None
    best_score = None
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate.absolute()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        if not resolved.is_dir():
            continue
        valid_subdir_count = 0
        total_frame_count = 0
        for seq_name in seq_names:
            for geom_subdir_name in geom_subdirs:
                geom_dir = resolved / str(seq_name) / geom_subdir_name
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
            best_candidate = resolved
            best_score = score

    if best_candidate is None:
        raise FileNotFoundError(
            f"Unable to resolve ZJU root for seq_names={seq_names} geom_subdir={geom_subdir}."
        )
    return best_candidate


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def load_config_with_local_defaults(config_path: Path):
    cfg = OmegaConf.load(config_path)
    if "data" in cfg:
        return cfg

    merged = OmegaConf.create()
    for entry in cfg.get("defaults", []):
        if entry == "_self_":
            continue
        if isinstance(entry, str):
            parent_name = entry
        elif isinstance(entry, dict):
            if len(entry) != 1:
                continue
            parent_name = str(next(iter(entry.values())))
        else:
            continue
        if parent_name == "_self_":
            continue
        parent_path = (config_path.parent / f"{parent_name}.yaml").resolve()
        if not parent_path.is_file():
            raise FileNotFoundError(f"Unable to resolve default config {parent_name} from {config_path}")
        merged = OmegaConf.merge(merged, load_config_with_local_defaults(parent_path))
    merged = OmegaConf.merge(merged, cfg)
    return merged


def load_dataset_from_config(config_path: Path, split: str, zju_dir_override: str):
    cfg = load_config_with_local_defaults(config_path)
    split_node = cfg.data.train if split == "train" else cfg.data.val
    dataset_cfg = OmegaConf.to_container(
        split_node.dataset.dataset_configs[0],
        resolve=True,
    )
    dataset_cfg.pop("_target_", None)
    dataset_cfg["split"] = split
    dataset_cfg["len_train"] = -1
    dataset_cfg["len_test"] = -1
    seq_names = ensure_list(dataset_cfg.get("seq_names"))
    geom_subdir = dataset_cfg.get("geom_subdir", "vggt_geom")
    resolved_zju_dir = resolve_zju_dir(
        zju_dir_override or dataset_cfg.get("ZJU_DIR", ""),
        seq_names,
        geom_subdir,
    )
    dataset_cfg["ZJU_DIR"] = str(resolved_zju_dir)

    common_conf_dict = OmegaConf.to_container(split_node.common_config, resolve=True)
    common_conf_dict["training"] = split == "train"
    common_conf_dict.setdefault("debug", False)
    common_conf_dict.setdefault("inside_random", False)
    common_conf_dict.setdefault("allow_duplicate_img", False)
    common_conf_dict.setdefault("load_depth", True)
    common_conf_dict.setdefault("rescale", True)
    common_conf_dict.setdefault("rescale_aug", False)
    common_conf_dict.setdefault("landscape_check", False)
    common_conf_dict.setdefault("augs", {"scales": None})
    common_conf = OmegaConf.create(common_conf_dict)

    dataset = ZjuVggtGeomDataset(
        common_conf=common_conf,
        **dataset_cfg,
    )
    num_images = int(common_conf_dict.get("fix_img_num", 0) or 0)
    if num_images <= 0:
        raise ValueError(f"Config {config_path} has invalid fix_img_num={num_images}.")
    return cfg, dataset, num_images, resolved_zju_dir


def build_initial_role_stats():
    return {
        "view_count": 0,
        "point_pixels_total": 0,
        "foreground_pixels_total": 0,
        "camera_loss_eligible_views": 0,
        "depth_loss_eligible_views": 0,
        "unproject_loss_eligible_views": 0,
        "samples_seen": 0,
    }


def main():
    args = parse_args()
    config_path = (REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg, dataset, num_images, resolved_zju_dir = load_dataset_from_config(
        config_path=config_path,
        split=args.split,
        zju_dir_override=args.zju_dir,
    )

    sample_count = min(int(args.num_samples), len(dataset))
    role_stats = defaultdict(build_initial_role_stats)
    camera_counter = Counter()
    anchor_counter = Counter()
    effective_source_view_pool_counts = Counter()
    source_only_view_count_histogram = Counter()
    rawpool_candidate_pool_used_samples = 0
    source_only_views_total = 0
    source_only_nonzero_cases = []
    sample_rows = []

    for sample_index in range(sample_count):
        sample = dataset.get_data(seq_index=sample_index, img_per_seq=num_images)
        supervised = set(sample.get("supervised_camera_names", []))
        source_only = set(sample.get("source_only_camera_names", []))
        camera_names = list(sample.get("camera_names", []))
        point_masks = sample["point_masks"]
        foreground_masks = sample["foreground_masks"]
        effective_source_view_pool = str(
            sample.get("selection_source_view_pool", cfg.get("zju_source_view_pool", ""))
        )
        requested_source_view_pool = str(
            sample.get("selection_requested_source_view_pool", effective_source_view_pool)
        )
        source_view_pool_train_probability = float(
            sample.get(
                "selection_source_view_pool_train_probability",
                cfg.get("zju_source_view_pool_train_probability", 1.0),
            )
        )
        rawpool_candidate_pool_used = bool(
            sample.get(
                "selection_rawpool_candidate_pool_used",
                effective_source_view_pool == "geom_plus_raw",
            )
        )
        source_only_view_count = int(len(source_only))
        effective_source_view_pool_counts[effective_source_view_pool] += 1
        source_only_view_count_histogram[source_only_view_count] += 1
        rawpool_candidate_pool_used_samples += int(rawpool_candidate_pool_used)
        source_only_views_total += source_only_view_count

        sample_row = {
            "sample_index": int(sample_index),
            "seq_name": str(sample.get("seq_name")),
            "camera_names": camera_names,
            "supervised_camera_names": list(supervised),
            "source_only_camera_names": list(source_only),
            "requested_source_view_pool": requested_source_view_pool,
            "effective_source_view_pool": effective_source_view_pool,
            "source_view_pool_train_probability": float(source_view_pool_train_probability),
            "rawpool_candidate_pool_used": bool(rawpool_candidate_pool_used),
            "source_only_view_count": int(source_only_view_count),
            "selection_anchor_camera": sample.get("selection_anchor_camera"),
            "view_rows": [],
        }
        if sample.get("selection_anchor_camera"):
            anchor_counter[str(sample["selection_anchor_camera"])] += 1

        for view_idx, camera_name in enumerate(camera_names):
            role = "supervised" if camera_name in supervised else "source_only"
            if camera_name not in supervised and camera_name not in source_only:
                role = "unknown"
            point_pixels = int(np.asarray(point_masks[view_idx], dtype=bool).sum())
            foreground_pixels = int(np.asarray(foreground_masks[view_idx], dtype=bool).sum())
            camera_counter[(role, str(camera_name))] += 1

            stats = role_stats[role]
            stats["view_count"] += 1
            stats["point_pixels_total"] += point_pixels
            stats["foreground_pixels_total"] += foreground_pixels
            stats["camera_loss_eligible_views"] += int(point_pixels > 100)
            stats["depth_loss_eligible_views"] += int(point_pixels > 0)
            stats["unproject_loss_eligible_views"] += int(point_pixels > 0)
            stats["samples_seen"] += 1

            if role == "source_only" and point_pixels > 0:
                source_only_nonzero_cases.append(
                    {
                        "sample_index": int(sample_index),
                        "seq_name": str(sample.get("seq_name")),
                        "camera_name": str(camera_name),
                        "point_pixels": int(point_pixels),
                    }
                )

            sample_row["view_rows"].append(
                {
                    "camera_name": str(camera_name),
                    "role": role,
                    "point_pixels": int(point_pixels),
                    "foreground_pixels": int(foreground_pixels),
                    "camera_loss_eligible": bool(point_pixels > 100),
                    "depth_loss_eligible": bool(point_pixels > 0),
                    "unproject_loss_eligible": bool(point_pixels > 0),
                }
            )
        sample_rows.append(sample_row)

    role_summary = {}
    for role, stats in role_stats.items():
        view_count = max(1, int(stats["view_count"]))
        role_summary[role] = {
            **stats,
            "avg_point_pixels_per_view": float(stats["point_pixels_total"]) / float(view_count),
            "avg_foreground_pixels_per_view": float(stats["foreground_pixels_total"]) / float(view_count),
        }

    top_cameras = {
        role: [
            {"camera_name": camera_name, "count": int(count)}
            for (camera_role, camera_name), count in camera_counter.most_common()
            if camera_role == role
        ][:8]
        for role in role_summary.keys()
    }

    payload = {
        "config_path": repo_rel(config_path),
        "split": args.split,
        "zju_dir": str(resolved_zju_dir),
        "num_samples": int(sample_count),
        "num_images": int(num_images),
        "source_policy": str(cfg.get("zju_source_policy", "")),
        "requested_source_view_pool": str(cfg.get("zju_source_view_pool", "")),
        "source_view_pool_train_probability": float(cfg.get("zju_source_view_pool_train_probability", 1.0)),
        "source_anchor_policy": str(cfg.get("zju_source_anchor_policy", "")),
        "min_supervised_views": int(cfg.get("zju_min_supervised_views", 1)),
        "effective_source_view_pool_counts": {
            str(pool_name): int(count)
            for pool_name, count in effective_source_view_pool_counts.items()
        },
        "rawpool_candidate_pool_used_samples": int(rawpool_candidate_pool_used_samples),
        "cached_only_fallback_samples": int(sample_count - rawpool_candidate_pool_used_samples),
        "avg_source_only_views_per_sample": float(source_only_views_total) / float(max(1, sample_count)),
        "source_only_view_count_histogram": {
            str(view_count): int(count)
            for view_count, count in sorted(source_only_view_count_histogram.items())
        },
        "role_summary": role_summary,
        "top_cameras_by_role": top_cameras,
        "anchor_counts": {str(key): int(value) for key, value in anchor_counter.most_common()},
        "source_only_nonzero_point_pixel_cases": source_only_nonzero_cases,
        "rawpool_geometry_leak_detected": bool(source_only_nonzero_cases),
        "sample_rows": sample_rows,
    }

    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# ZJU Source-Policy Supervision Audit",
        "",
        f"- config_path: `{payload['config_path']}`",
        f"- split: `{payload['split']}`",
        f"- zju_dir: `{payload['zju_dir']}`",
        f"- num_samples: `{payload['num_samples']}`",
        f"- num_images: `{payload['num_images']}`",
        f"- source_policy: `{payload['source_policy']}`",
        f"- requested_source_view_pool: `{payload['requested_source_view_pool']}`",
        f"- source_view_pool_train_probability: `{payload['source_view_pool_train_probability']}`",
        f"- source_anchor_policy: `{payload['source_anchor_policy']}`",
        f"- min_supervised_views: `{payload['min_supervised_views']}`",
        f"- rawpool_geometry_leak_detected: `{payload['rawpool_geometry_leak_detected']}`",
        "",
        "## Exposure Summary",
        "",
        f"- effective_source_view_pool_counts: `{payload['effective_source_view_pool_counts']}`",
        f"- rawpool_candidate_pool_used_samples: `{payload['rawpool_candidate_pool_used_samples']}`",
        f"- cached_only_fallback_samples: `{payload['cached_only_fallback_samples']}`",
        f"- avg_source_only_views_per_sample: `{payload['avg_source_only_views_per_sample']:.2f}`",
        f"- source_only_view_count_histogram: `{payload['source_only_view_count_histogram']}`",
        "",
        "## Role Summary",
        "",
    ]
    for role in sorted(role_summary):
        stats = role_summary[role]
        lines.extend(
            [
                f"### {role}",
                "",
                f"- view_count: `{stats['view_count']}`",
                f"- point_pixels_total: `{stats['point_pixels_total']}`",
                f"- foreground_pixels_total: `{stats['foreground_pixels_total']}`",
                f"- avg_point_pixels_per_view: `{stats['avg_point_pixels_per_view']:.2f}`",
                f"- avg_foreground_pixels_per_view: `{stats['avg_foreground_pixels_per_view']:.2f}`",
                f"- camera_loss_eligible_views: `{stats['camera_loss_eligible_views']}`",
                f"- depth_loss_eligible_views: `{stats['depth_loss_eligible_views']}`",
                f"- unproject_loss_eligible_views: `{stats['unproject_loss_eligible_views']}`",
                "",
            ]
        )
        top_rows = top_cameras.get(role, [])
        if top_rows:
            lines.append("- top_cameras:")
            for row in top_rows:
                lines.append(f"  - `{row['camera_name']}`: `{row['count']}`")
            lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- `requested_source_view_pool` is the config-level pool; `effective_source_view_pool_counts` shows the per-sample pool after any train-only rawpool mixing has been applied.",
            "- `camera_loss_eligible_views` uses the same `point_masks.sum() > 100` gate as `compute_camera_loss()`.",
            "- `depth_loss_eligible_views` and `unproject_loss_eligible_views` use the same GT-side `point_masks > 0` support assumption as the current loss path.",
            "- Any non-empty `source_only_nonzero_point_pixel_cases` would indicate a real supervision leak for rawpool-only views.",
            "",
        ]
    )
    if source_only_nonzero_cases:
        lines.append("## Source-only Nonzero Point Cases")
        lines.append("")
        for row in source_only_nonzero_cases[:16]:
            lines.append(
                f"- sample_index={row['sample_index']} seq={row['seq_name']} camera={row['camera_name']} point_pixels={row['point_pixels']}"
            )
        lines.append("")

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_md)


if __name__ == "__main__":
    main()
