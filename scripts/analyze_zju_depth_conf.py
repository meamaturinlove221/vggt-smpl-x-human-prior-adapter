import argparse
import json
import math
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


def maybe_sample(tensor: torch.Tensor, max_elements: int = 200_000) -> torch.Tensor:
    flat = tensor.reshape(-1)
    if flat.numel() <= max_elements:
        return flat
    indices = torch.randperm(flat.numel(), device=flat.device)[:max_elements]
    return flat[indices]


def summarize_percentiles(values: torch.Tensor, percentiles: list[float]) -> dict[str, float]:
    quantiles = torch.tensor(percentiles, device=values.device, dtype=values.dtype) / 100.0
    result = torch.quantile(values, quantiles)
    return {
        f"p{int(p)}": float(v.item())
        for p, v in zip(percentiles, result)
    }


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    count = int(mask.sum().item())
    if count == 0:
        return None
    return float(values[mask].mean().item())


def safe_corrcoef(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2 or y.numel() < 2:
        return None
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.std(unbiased=False) * y.std(unbiased=False)
    if denom.item() == 0:
        return None
    return float(((x * y).mean() / denom).item())


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


def main():
    parser = argparse.ArgumentParser(description="Analyze depth confidence vs geometry/depth error on ZJU VGGT geometry data.")
    parser.add_argument("--config", default="zju_vggt_geom_minimal")
    parser.add_argument("--zju-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seq-names", default="CoreView_390")
    parser.add_argument("--geom-subdir", default="vggt_geom")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        cfg = compose(config_name=args.config, overrides=overrides)

    dataset = instantiate(cfg.data.val if args.split == "val" else cfg.data.train, _recursive_=False)
    dataset.seed = cfg.seed_value
    loader = dataset.get_loader(epoch=0)

    model = instantiate(cfg.model, _recursive_=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(model_state_dict, strict=False)
    model = model.to(args.device)
    model.eval()

    conf_values = []
    depth_errors = []
    world_errors = []
    topk_rows = []

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

            valid_mask = (
                batch["point_masks"]
                & torch.isfinite(pred_depth)
                & torch.isfinite(pred_conf)
                & torch.isfinite(pred_world).all(dim=-1)
            )

            conf = pred_conf[valid_mask]
            depth_err = (pred_depth - batch["depths"]).abs()[valid_mask]
            world_err = (pred_world - batch["world_points"]).norm(dim=-1)[valid_mask]

            conf_values.append(conf.detach().cpu())
            depth_errors.append(depth_err.detach().cpu())
            world_errors.append(world_err.detach().cpu())

    conf_values = torch.cat(conf_values)
    depth_errors = torch.cat(depth_errors)
    world_errors = torch.cat(world_errors)

    conf_sample = maybe_sample(conf_values)
    depth_sample = maybe_sample(depth_errors)
    world_sample = maybe_sample(world_errors)
    sample_count = min(conf_sample.numel(), depth_sample.numel(), world_sample.numel())
    conf_sample = conf_sample[:sample_count]
    depth_sample = depth_sample[:sample_count]
    world_sample = world_sample[:sample_count]

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    conf_percentiles = summarize_percentiles(conf_values, percentiles)

    top_ratios = [0.1, 0.25, 0.5]
    bottom_ratios = [0.1, 0.25]
    rows = []
    for ratio in top_ratios:
        thresh = torch.quantile(conf_values, 1.0 - ratio)
        mask = conf_values >= thresh
        rows.append(
            {
                "bucket": f"top_{int(ratio * 100)}pct_conf",
                "count": int(mask.sum().item()),
                "conf_threshold": float(thresh.item()),
                "mean_depth_err": masked_mean(depth_errors, mask),
                "mean_world_err": masked_mean(world_errors, mask),
            }
        )
    for ratio in bottom_ratios:
        thresh = torch.quantile(conf_values, ratio)
        mask = conf_values <= thresh
        rows.append(
            {
                "bucket": f"bottom_{int(ratio * 100)}pct_conf",
                "count": int(mask.sum().item()),
                "conf_threshold": float(thresh.item()),
                "mean_depth_err": masked_mean(depth_errors, mask),
                "mean_world_err": masked_mean(world_errors, mask),
            }
        )

    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "num_batches": args.num_batches,
        "num_images": args.num_images,
        "total_valid_points": int(conf_values.numel()),
        "depth_conf_percentiles": conf_percentiles,
        "depth_conf_mean": float(conf_values.mean().item()),
        "depth_conf_std": float(conf_values.std(unbiased=False).item()),
        "depth_err_mean": float(depth_errors.mean().item()),
        "world_err_mean": float(world_errors.mean().item()),
        "corr_conf_depth_err": safe_corrcoef(conf_sample, depth_sample),
        "corr_conf_world_err": safe_corrcoef(conf_sample, world_sample),
        "buckets": rows,
    }

    json_path = output_dir / "depth_conf_analysis.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Depth Confidence Analysis",
        "",
        f"- config: `{args.config}`",
        f"- split: `{args.split}`",
        f"- num_batches: `{args.num_batches}`",
        f"- num_images: `{args.num_images}`",
        f"- total_valid_points: `{payload['total_valid_points']}`",
        f"- depth_conf_mean: `{payload['depth_conf_mean']:.4f}`",
        f"- depth_conf_std: `{payload['depth_conf_std']:.4f}`",
        f"- depth_err_mean: `{payload['depth_err_mean']:.4f}`",
        f"- world_err_mean: `{payload['world_err_mean']:.4f}`",
        f"- corr(conf, depth_err): `{payload['corr_conf_depth_err']}`",
        f"- corr(conf, world_err): `{payload['corr_conf_world_err']}`",
        "",
        "## Percentiles",
        "",
    ]
    for key, value in conf_percentiles.items():
        md_lines.append(f"- {key}: `{value:.4f}`")
    md_lines.extend(
        [
            "",
            "## Buckets",
            "",
            "| bucket | count | conf threshold | mean depth err | mean world err |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        md_lines.append(
            "| {bucket} | {count} | {conf_threshold:.4f} | {mean_depth_err} | {mean_world_err} |".format(
                bucket=row["bucket"],
                count=row["count"],
                conf_threshold=row["conf_threshold"],
                mean_depth_err="n/a" if row["mean_depth_err"] is None else f"{row['mean_depth_err']:.4f}",
                mean_world_err="n/a" if row["mean_world_err"] is None else f"{row['mean_world_err']:.4f}",
            )
        )

    md_path = output_dir / "depth_conf_analysis.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[done] Wrote {json_path}")
    print(f"[done] Wrote {md_path}")


if __name__ == "__main__":
    main()
