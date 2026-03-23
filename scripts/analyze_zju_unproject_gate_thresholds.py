import argparse
import json
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"

for root in (str(REPO_ROOT), str(TRAINING_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from train_utils.general import copy_data_to_device
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from loss import unproject_depth_and_pose_to_world_points


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze how depth-confidence thresholds affect unproject-geometry valid-point counts."
    )
    parser.add_argument("--config", default="zju_vggt_geom_minimal")
    parser.add_argument("--zju-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seq-names", default="CoreView_390")
    parser.add_argument("--geom-subdir", default="vggt_geom")
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-valid-points", type=int, default=100)
    parser.add_argument(
        "--threshold-percentiles",
        type=float,
        nargs="*",
        default=[50.0, 60.0, 65.0, 70.0, 75.0],
    )
    parser.add_argument(
        "--explicit-thresholds",
        type=float,
        nargs="*",
        default=[],
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def process_batch(batch: dict) -> dict:
    normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = (
        normalize_camera_extrinsics_and_points_batch(
            extrinsics=batch["extrinsics"],
            cam_points=batch["cam_points"],
            world_points=batch["world_points"],
            depths=batch["depths"],
            point_masks=batch["point_masks"],
        )
    )
    batch["extrinsics"] = normalized_extrinsics
    batch["cam_points"] = normalized_cam_points
    batch["world_points"] = normalized_world_points
    batch["depths"] = normalized_depths
    return batch


def build_cfg(args):
    overrides = [
        f"zju_dir='{args.zju_dir.replace(chr(92), '/')}'",
        f"zju_seq_names='{args.seq_names}'",
        f"zju_geom_subdir='{args.geom_subdir}'",
        f"data.train.common_config.fix_img_num={args.num_images}",
        f"data.val.common_config.fix_img_num={args.num_images}",
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


def load_model(cfg, checkpoint_path: str, device: str):
    model = instantiate(cfg.model, _recursive_=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(model_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def collect_split_records(cfg, model, split: str, args):
    dataset = instantiate(cfg.data.val if split == "val" else cfg.data.train, _recursive_=False)
    dataset.seed = cfg.seed_value
    loader = dataset.get_loader(epoch=0)
    records = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.num_batches:
                break

            batch = process_batch(batch)
            batch = copy_data_to_device(batch, torch.device(args.device), non_blocking=True)

            with torch.amp.autocast(device_type=torch.device(args.device).type, enabled=True, dtype=torch.bfloat16):
                predictions = model(batch["images"])

            pred_depth = predictions["depth"][..., 0]
            pred_conf = predictions["depth_conf"]
            image_hw = batch["images"].shape[-2:]
            pred_world = unproject_depth_and_pose_to_world_points(
                predictions["depth"],
                predictions["pose_enc"],
                image_size_hw=image_hw,
            )

            base_valid_mask = (
                batch["point_masks"]
                & (pred_depth > 1e-8)
                & torch.isfinite(pred_depth)
                & torch.isfinite(pred_conf)
                & torch.isfinite(pred_world).all(dim=-1)
            )

            base_conf = pred_conf[base_valid_mask].detach().float().cpu()
            records.append(
                {
                    "batch_index": batch_idx,
                    "base_valid_count": int(base_conf.numel()),
                    "conf_values": base_conf.tolist(),
                }
            )

    return records


def summarize_threshold(batch_records, threshold: float, min_valid_points: int):
    counts = []
    ratios = []
    for row in batch_records:
        conf_values = torch.tensor(row["conf_values"], dtype=torch.float32)
        base_count = int(row["base_valid_count"])
        if threshold <= 0:
            count = base_count
        elif conf_values.numel() == 0:
            count = 0
        else:
            count = int((conf_values >= threshold).sum().item())
        counts.append(count)
        ratios.append(0.0 if base_count == 0 else float(count / base_count))

    batch_count = len(batch_records)
    pass_count = sum(1 for count in counts if count >= min_valid_points)
    return {
        "threshold": float(threshold),
        "batch_count": batch_count,
        "mean_valid_count": float(sum(counts) / max(1, batch_count)),
        "min_valid_count": int(min(counts) if counts else 0),
        "max_valid_count": int(max(counts) if counts else 0),
        "mean_keep_ratio": float(sum(ratios) / max(1, batch_count)),
        "min_keep_ratio": float(min(ratios) if ratios else 0.0),
        "max_keep_ratio": float(max(ratios) if ratios else 0.0),
        "batches_ge_min_valid_points": int(pass_count),
        "fraction_ge_min_valid_points": float(pass_count / max(1, batch_count)),
        "per_batch_valid_counts": counts,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args)
    model = load_model(cfg, args.checkpoint, args.device)

    split_records = {
        "train": collect_split_records(cfg, model, "train", args),
        "val": collect_split_records(cfg, model, "val", args),
    }

    combined_conf = []
    for split_name in ("train", "val"):
        for row in split_records[split_name]:
            combined_conf.extend(row["conf_values"])

    if not combined_conf:
        raise RuntimeError("No valid confidence values were collected.")

    conf_tensor = torch.tensor(combined_conf, dtype=torch.float32)
    percentile_thresholds = {
        f"p{int(percentile)}": float(torch.quantile(conf_tensor, percentile / 100.0).item())
        for percentile in args.threshold_percentiles
    }

    threshold_items = [("raw", 0.0)]
    threshold_items.extend((label, value) for label, value in percentile_thresholds.items())
    threshold_items.extend(
        (f"explicit_{index}", float(value))
        for index, value in enumerate(args.explicit_thresholds, start=1)
    )

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "num_images": args.num_images,
        "num_batches": args.num_batches,
        "min_valid_points": args.min_valid_points,
        "threshold_percentiles": percentile_thresholds,
        "splits": {},
    }

    for split_name, batch_records in split_records.items():
        split_summary = {
            "batch_records": [
                {
                    "batch_index": row["batch_index"],
                    "base_valid_count": row["base_valid_count"],
                }
                for row in batch_records
            ],
            "thresholds": {},
        }
        for label, threshold in threshold_items:
            split_summary["thresholds"][label] = summarize_threshold(
                batch_records=batch_records,
                threshold=threshold,
                min_valid_points=args.min_valid_points,
            )
        summary["splits"][split_name] = split_summary

    json_path = output_dir / "unproject_gate_thresholds.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Unproject Gate Threshold Analysis",
        "",
        f"- config: `{args.config}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- num_images: `{args.num_images}`",
        f"- num_batches_per_split: `{args.num_batches}`",
        f"- min_valid_points: `{args.min_valid_points}`",
        "",
        "## Threshold Percentiles",
        "",
    ]
    for label, value in percentile_thresholds.items():
        md_lines.append(f"- {label}: `{value:.4f}`")

    for split_name in ("train", "val"):
        split_summary = summary["splits"][split_name]
        md_lines.extend(
            [
                "",
                f"## {split_name.capitalize()}",
                "",
                "| threshold | value | mean valid | min valid | max valid | mean keep ratio | min keep ratio | pass batches | pass fraction |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, threshold in threshold_items:
            row = split_summary["thresholds"][label]
            md_lines.append(
                "| {label} | {value:.4f} | {mean_valid_count:.1f} | {min_valid_count} | {max_valid_count} | {mean_keep_ratio:.4f} | {min_keep_ratio:.4f} | {batches_ge_min_valid_points} | {fraction_ge_min_valid_points:.4f} |".format(
                    label=label,
                    value=threshold,
                    **row,
                )
            )

    md_path = output_dir / "unproject_gate_thresholds.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[done] Wrote {json_path}")
    print(f"[done] Wrote {md_path}")


if __name__ == "__main__":
    main()
