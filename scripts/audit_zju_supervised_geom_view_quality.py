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

from training.data.datasets.zju_vggt_geom import ZjuVggtGeomDataset, _merge_geom_payloads


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit supervised geom-view quality on a ZJU source-policy config."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--zju_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
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


def load_dataset_from_config(config_path: Path, split: str, zju_dir_override: str):
    cfg = OmegaConf.load(config_path)
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

    dataset = ZjuVggtGeomDataset(common_conf=common_conf, **dataset_cfg)
    num_images = int(common_conf_dict.get("fix_img_num", 0) or 0)
    if num_images <= 0:
        raise ValueError(f"Config {config_path} has invalid fix_img_num={num_images}.")
    return cfg, dataset, num_images, resolved_zju_dir


def safe_mean_depth_conf(geom, camera_name: str) -> float:
    cam_names = [str(name) for name in geom["cam_names"]]
    local_idx = cam_names.index(str(camera_name))
    depth_conf = np.asarray(geom["depth_conf"][local_idx], dtype=np.float32)
    return float(depth_conf.mean())


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
    supervised_hist = Counter()
    worst_camera_counter = Counter()
    best_camera_counter = Counter()
    quality_gap_values = []
    per_camera_quality = defaultdict(list)
    multi_supervised_rows = []

    for sample_index in range(sample_count):
        sample = dataset.get_data(seq_index=sample_index, img_per_seq=num_images)
        entry = dataset.sequence_list[sample_index]
        geom = _merge_geom_payloads(entry.get("geom_paths") or [entry["geom_path"]])
        camera_names = list(sample.get("camera_names", []))
        supervised_camera_names = list(sample.get("supervised_camera_names", []))
        point_masks = sample["point_masks"]

        supervised_hist[len(supervised_camera_names)] += 1
        if len(supervised_camera_names) < 2:
            continue

        rows = []
        for camera_name in supervised_camera_names:
            selected_idx = camera_names.index(camera_name)
            point_pixels = int(np.asarray(point_masks[selected_idx], dtype=bool).sum())
            mean_depth_conf = safe_mean_depth_conf(geom, camera_name)
            per_camera_quality[str(camera_name)].append(mean_depth_conf)
            rows.append(
                {
                    "camera_name": str(camera_name),
                    "mean_depth_conf": float(mean_depth_conf),
                    "point_pixels": int(point_pixels),
                }
            )

        rows.sort(key=lambda row: row["mean_depth_conf"])
        worst = rows[0]
        best = rows[-1]
        worst_camera_counter[worst["camera_name"]] += 1
        best_camera_counter[best["camera_name"]] += 1
        quality_gap_values.append(float(best["mean_depth_conf"] - worst["mean_depth_conf"]))
        multi_supervised_rows.append(
            {
                "sample_index": int(sample_index),
                "seq_name": str(sample.get("seq_name")),
                "selection_anchor_camera": sample.get("selection_anchor_camera"),
                "supervised_count": int(len(supervised_camera_names)),
                "worst_supervised_camera": worst["camera_name"],
                "worst_mean_depth_conf": float(worst["mean_depth_conf"]),
                "best_supervised_camera": best["camera_name"],
                "best_mean_depth_conf": float(best["mean_depth_conf"]),
                "best_minus_worst_depth_conf": float(best["mean_depth_conf"] - worst["mean_depth_conf"]),
                "supervised_views": rows,
            }
        )

    multi_count = len(multi_supervised_rows)
    top_worst_camera, top_worst_count = (None, 0)
    if worst_camera_counter:
        top_worst_camera, top_worst_count = worst_camera_counter.most_common(1)[0]

    payload = {
        "config_path": repo_rel(config_path),
        "split": args.split,
        "zju_dir": str(resolved_zju_dir),
        "num_samples": int(sample_count),
        "num_images": int(num_images),
        "source_policy": str(cfg.get("zju_source_policy", "")),
        "source_view_pool": str(cfg.get("zju_source_view_pool", "")),
        "source_anchor_policy": str(cfg.get("zju_source_anchor_policy", "")),
        "min_supervised_views": int(cfg.get("zju_min_supervised_views", 1)),
        "supervised_count_histogram": {str(key): int(value) for key, value in sorted(supervised_hist.items())},
        "multi_supervised_sample_count": int(multi_count),
        "multi_supervised_ratio": float(multi_count) / float(max(1, sample_count)),
        "worst_supervised_camera_counts": {str(key): int(value) for key, value in worst_camera_counter.most_common()},
        "best_supervised_camera_counts": {str(key): int(value) for key, value in best_camera_counter.most_common()},
        "top_worst_supervised_camera": top_worst_camera,
        "top_worst_supervised_camera_ratio": float(top_worst_count) / float(max(1, multi_count)),
        "avg_best_minus_worst_depth_conf": float(np.mean(quality_gap_values)) if quality_gap_values else 0.0,
        "median_best_minus_worst_depth_conf": float(np.median(quality_gap_values)) if quality_gap_values else 0.0,
        "per_camera_mean_depth_conf": {
            camera_name: {
                "count": int(len(values)),
                "mean_depth_conf": float(np.mean(values)),
                "median_depth_conf": float(np.median(values)),
            }
            for camera_name, values in sorted(per_camera_quality.items())
        },
        "supports_conditioned_filter_experiment": bool(
            multi_count >= 8 and top_worst_count >= 6 and (float(np.mean(quality_gap_values)) if quality_gap_values else 0.0) >= 1.0
        ),
        "multi_supervised_rows": multi_supervised_rows,
    }

    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# ZJU Supervised Geom View Quality Audit",
        "",
        f"- config_path: `{payload['config_path']}`",
        f"- split: `{payload['split']}`",
        f"- num_samples: `{payload['num_samples']}`",
        f"- num_images: `{payload['num_images']}`",
        f"- source_policy: `{payload['source_policy']}`",
        f"- source_view_pool: `{payload['source_view_pool']}`",
        f"- source_anchor_policy: `{payload['source_anchor_policy']}`",
        f"- min_supervised_views: `{payload['min_supervised_views']}`",
        f"- supervised_count_histogram: `{payload['supervised_count_histogram']}`",
        f"- multi_supervised_sample_count: `{payload['multi_supervised_sample_count']}`",
        f"- multi_supervised_ratio: `{payload['multi_supervised_ratio']:.4f}`",
        f"- top_worst_supervised_camera: `{payload['top_worst_supervised_camera']}`",
        f"- top_worst_supervised_camera_ratio: `{payload['top_worst_supervised_camera_ratio']:.4f}`",
        f"- avg_best_minus_worst_depth_conf: `{payload['avg_best_minus_worst_depth_conf']:.4f}`",
        f"- median_best_minus_worst_depth_conf: `{payload['median_best_minus_worst_depth_conf']:.4f}`",
        f"- supports_conditioned_filter_experiment: `{payload['supports_conditioned_filter_experiment']}`",
        "",
        "## Worst Supervised Camera Counts",
        "",
    ]
    for camera_name, count in payload["worst_supervised_camera_counts"].items():
        lines.append(f"- `{camera_name}`: `{count}`")
    lines.extend(["", "## Per-Camera Mean Depth Conf", ""])
    for camera_name, row in payload["per_camera_mean_depth_conf"].items():
        lines.append(
            f"- `{camera_name}`: count=`{row['count']}` mean=`{row['mean_depth_conf']:.4f}` median=`{row['median_depth_conf']:.4f}`"
        )

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_md)


if __name__ == "__main__":
    main()
