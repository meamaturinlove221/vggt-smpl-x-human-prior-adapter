import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"

for root in (str(REPO_ROOT), str(TRAINING_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from loss import check_and_fix_inf_nan
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from training.data.datasets.zju_vggt_geom import ZjuVggtGeomDataset


FRAME_PATTERN = re.compile(r"_frame_(\d+)$")
METRIC_KEYS = (
    "conf_depth_mean",
    "reg_depth_mean",
    "pred_depth_conf_mean",
    "fg_human_conf_depth_mean",
    "bg_bottom_band_conf_depth_mean",
    "bg_far_conf_depth_mean",
    "fg_human_reg_depth_mean",
    "bg_bottom_band_reg_depth_mean",
    "bg_far_reg_depth_mean",
    "fg_bottom_conf_depth_mean",
    "fg_nonbottom_conf_depth_mean",
    "fg_bottom_reg_depth_mean",
    "fg_nonbottom_reg_depth_mean",
)
REGION_KEYS = ("fg_human", "bg_bottom_band", "bg_far")
REGION_DISPLAY_NAMES = {
    "fg_human": "fg_human",
    "bg_bottom_band": "bg_bottom_band",
    "bg_far": "bg_far",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit conf_depth attribution on ZJU nearest-rawpool style configs."
    )
    parser.add_argument("--config", type=str, required=True, help="Training config name or yaml path.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Primary checkpoint path.")
    parser.add_argument("--label", type=str, default="candidate", help="Primary checkpoint label.")
    parser.add_argument("--reference-checkpoint", type=str, default="", help="Optional reference checkpoint.")
    parser.add_argument("--reference-label", type=str, default="reference", help="Reference checkpoint label.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to audit.")
    parser.add_argument("--num-samples", type=int, default=32, help="How many sequential samples to audit.")
    parser.add_argument("--num-images", type=int, default=4, help="fix_img_num override.")
    parser.add_argument("--zju-dir", type=str, default="", help="Optional explicit ZJU root.")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for audit artifacts.")
    return parser.parse_args()


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def resolve_config_path(raw_value: str) -> Path:
    candidate = Path(raw_value)
    if candidate.suffix.lower() == ".yaml":
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        return candidate
    return (TRAINING_ROOT / "config" / f"{raw_value}.yaml").resolve()


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


def load_dataset_and_cfg(config_path: Path, split: str, zju_dir_override: str, num_images: int):
    cfg = load_config_with_local_defaults(config_path)
    split_node = cfg.data.train if split == "train" else cfg.data.val
    dataset_cfg = OmegaConf.to_container(split_node.dataset.dataset_configs[0], resolve=True)
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
    common_conf_dict["fix_img_num"] = int(num_images)
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
    return cfg, dataset, resolved_zju_dir


def sample_to_batch(sample: dict) -> dict:
    batch = {
        "seq_name": [str(sample["seq_name"])],
        "ids": torch.from_numpy(np.asarray(sample["ids"], dtype=np.int64)).unsqueeze(0),
        "images": torch.from_numpy(np.stack(sample["images"]).astype(np.float32)).permute(0, 3, 1, 2).to(torch.get_default_dtype()).div(255).unsqueeze(0),
        "depths": torch.from_numpy(np.stack(sample["depths"]).astype(np.float32)).unsqueeze(0),
        "extrinsics": torch.from_numpy(np.stack(sample["extrinsics"]).astype(np.float32)).unsqueeze(0),
        "intrinsics": torch.from_numpy(np.stack(sample["intrinsics"]).astype(np.float32)).unsqueeze(0),
        "cam_points": torch.from_numpy(np.stack(sample["cam_points"]).astype(np.float32)).unsqueeze(0),
        "world_points": torch.from_numpy(np.stack(sample["world_points"]).astype(np.float32)).unsqueeze(0),
        "point_masks": torch.from_numpy(np.stack(sample["point_masks"]).astype(bool)).unsqueeze(0),
        "foreground_masks": torch.from_numpy(np.stack(sample["foreground_masks"]).astype(bool)).unsqueeze(0),
        "conf_depth_point_masks": torch.from_numpy(np.stack(sample["conf_depth_point_masks"]).astype(bool)).unsqueeze(0),
    }
    return batch


def normalize_batch(batch: dict) -> dict:
    normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = normalize_camera_extrinsics_and_points_batch(
        extrinsics=batch["extrinsics"],
        cam_points=batch["cam_points"],
        world_points=batch["world_points"],
        depths=batch["depths"],
        point_masks=batch["point_masks"],
    )
    batch["extrinsics"] = normalized_extrinsics
    batch["cam_points"] = normalized_cam_points
    batch["world_points"] = normalized_world_points
    batch["depths"] = normalized_depths
    return batch


def load_model(model_cfg, checkpoint_path: Path, device: torch.device):
    model = instantiate(model_cfg, _recursive_=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(model_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def build_bottom_mask(height: int, width: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(height, device=device)
    cutoff = int(height * 0.8)
    return (ys[:, None] >= cutoff).expand(height, width)


def safe_mean(values: torch.Tensor, mask: torch.Tensor):
    if int(mask.sum().item()) <= 0:
        return None
    return float(values[mask].mean().item())


def parse_frame_id(seq_name: str):
    match = FRAME_PATTERN.search(str(seq_name))
    if match is None:
        return None
    return int(match.group(1))


def determine_view_role(camera_name: str, sample: dict):
    supervised = set(sample.get("supervised_camera_names", []))
    anchor = sample.get("selection_anchor_camera")
    if camera_name == anchor and camera_name in supervised:
        return "anchor_supervised"
    if camera_name in supervised:
        return "extra_supervised"
    return "source_only"


def make_frame_key(frame_id):
    if frame_id is None:
        return "frame_unknown"
    return f"frame_{int(frame_id):06d}"


def build_region_masks(view_valid, view_conf_valid, foreground, bottom_mask):
    background = ~foreground
    fg_human_valid = view_valid & foreground
    fg_human_conf_valid = view_conf_valid & foreground
    bg_bottom_band_valid = view_valid & background & bottom_mask
    bg_bottom_band_conf_valid = view_conf_valid & background & bottom_mask
    bg_far_valid = view_valid & background & (~bottom_mask)
    bg_far_conf_valid = view_conf_valid & background & (~bottom_mask)
    return {
        "fg_human_valid": fg_human_valid,
        "fg_human_conf_valid": fg_human_conf_valid,
        "bg_bottom_band_valid": bg_bottom_band_valid,
        "bg_bottom_band_conf_valid": bg_bottom_band_conf_valid,
        "bg_far_valid": bg_far_valid,
        "bg_far_conf_valid": bg_far_conf_valid,
        "fg_bottom_valid": fg_human_valid & bottom_mask,
        "fg_bottom_conf_valid": fg_human_conf_valid & bottom_mask,
        "fg_nonbottom_valid": fg_human_valid & (~bottom_mask),
        "fg_nonbottom_conf_valid": fg_human_conf_valid & (~bottom_mask),
    }


def run_model_rows(model, label: str, batch: dict, sample: dict, gamma: float, alpha: float, device: torch.device):
    with torch.no_grad():
        batch_gpu = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        predictions = model(images=batch_gpu["images"])

    pred_depth = check_and_fix_inf_nan(predictions["depth"], f"{label}_pred_depth")
    pred_conf = check_and_fix_inf_nan(predictions["depth_conf"], f"{label}_pred_conf").clamp_min(1e-8)
    gt_depth = batch_gpu["depths"][..., None]
    point_mask = batch_gpu["point_masks"]
    conf_depth_mask = batch_gpu["conf_depth_point_masks"] & point_mask
    valid_mask = point_mask & torch.isfinite(pred_depth[..., 0]) & torch.isfinite(pred_conf)
    conf_valid_mask = conf_depth_mask & torch.isfinite(pred_depth[..., 0]) & torch.isfinite(pred_conf)

    reg_map = torch.norm(gt_depth - pred_depth, dim=-1)
    conf_map = gamma * reg_map * pred_conf - alpha * torch.log(pred_conf)

    foreground_masks = batch_gpu["foreground_masks"]
    _, _, hh, ww = point_mask.shape
    bottom_mask = build_bottom_mask(hh, ww, device)

    frame_id = parse_frame_id(sample["seq_name"])
    rows = []
    for view_idx, camera_name in enumerate(sample["camera_names"]):
        view_valid = valid_mask[0, view_idx]
        view_conf_valid = conf_valid_mask[0, view_idx]
        foreground = foreground_masks[0, view_idx]
        role = determine_view_role(camera_name, sample)
        region_masks = build_region_masks(view_valid, view_conf_valid, foreground, bottom_mask)
        quality_score = sample.get("supervised_view_quality_scores", {}).get(str(camera_name))

        rows.append(
            {
                "label": label,
                "sample_index": int(sample["audit_sample_index"]),
                "seq_name": str(sample["seq_name"]),
                "frame_id": frame_id,
                "view_idx": int(view_idx),
                "camera_name": str(camera_name),
                "view_role": role,
                "selection_anchor_camera": sample.get("selection_anchor_camera"),
                "quality_score": None if quality_score is None else float(quality_score),
                "valid_pixels": int(view_valid.sum().item()),
                "conf_valid_pixels": int(view_conf_valid.sum().item()),
                "fg_human_valid_pixels": int(region_masks["fg_human_valid"].sum().item()),
                "bg_bottom_band_valid_pixels": int(region_masks["bg_bottom_band_valid"].sum().item()),
                "bg_far_valid_pixels": int(region_masks["bg_far_valid"].sum().item()),
                "fg_human_conf_valid_pixels": int(region_masks["fg_human_conf_valid"].sum().item()),
                "bg_bottom_band_conf_valid_pixels": int(region_masks["bg_bottom_band_conf_valid"].sum().item()),
                "bg_far_conf_valid_pixels": int(region_masks["bg_far_conf_valid"].sum().item()),
                "conf_depth_mean": safe_mean(conf_map[0, view_idx], view_conf_valid),
                "reg_depth_mean": safe_mean(reg_map[0, view_idx], view_valid),
                "pred_depth_conf_mean": safe_mean(pred_conf[0, view_idx], view_conf_valid),
                "fg_human_conf_depth_mean": safe_mean(conf_map[0, view_idx], region_masks["fg_human_conf_valid"]),
                "bg_bottom_band_conf_depth_mean": safe_mean(conf_map[0, view_idx], region_masks["bg_bottom_band_conf_valid"]),
                "bg_far_conf_depth_mean": safe_mean(conf_map[0, view_idx], region_masks["bg_far_conf_valid"]),
                "fg_human_reg_depth_mean": safe_mean(reg_map[0, view_idx], region_masks["fg_human_valid"]),
                "bg_bottom_band_reg_depth_mean": safe_mean(reg_map[0, view_idx], region_masks["bg_bottom_band_valid"]),
                "bg_far_reg_depth_mean": safe_mean(reg_map[0, view_idx], region_masks["bg_far_valid"]),
                "fg_bottom_conf_depth_mean": safe_mean(conf_map[0, view_idx], region_masks["fg_bottom_conf_valid"]),
                "fg_nonbottom_conf_depth_mean": safe_mean(conf_map[0, view_idx], region_masks["fg_nonbottom_conf_valid"]),
                "fg_bottom_reg_depth_mean": safe_mean(reg_map[0, view_idx], region_masks["fg_bottom_valid"]),
                "fg_nonbottom_reg_depth_mean": safe_mean(reg_map[0, view_idx], region_masks["fg_nonbottom_valid"]),
            }
        )
    return rows


def add_summary_bucket(bucket, row):
    bucket["view_count"] += 1
    bucket["valid_pixels"] += int(row["valid_pixels"])
    bucket["conf_valid_pixels"] += int(row["conf_valid_pixels"])
    for key in METRIC_KEYS:
        value = row.get(key)
        if value is not None:
            bucket[f"{key}_sum"] += float(value)
            bucket[f"{key}_count"] += 1


def finalize_summary(grouped_rows: dict):
    payload = {}
    for key, bucket in grouped_rows.items():
        row = {
            "view_count": int(bucket["view_count"]),
            "valid_pixels": int(bucket["valid_pixels"]),
            "conf_valid_pixels": int(bucket["conf_valid_pixels"]),
        }
        for metric in METRIC_KEYS:
            count = int(bucket[f"{metric}_count"])
            row[metric] = None if count <= 0 else float(bucket[f"{metric}_sum"] / count)
        payload[str(key)] = row
    return payload


def build_empty_bucket():
    bucket = {
        "view_count": 0,
        "valid_pixels": 0,
        "conf_valid_pixels": 0,
    }
    for metric in METRIC_KEYS:
        bucket[f"{metric}_sum"] = 0.0
        bucket[f"{metric}_count"] = 0
    return bucket


def build_region_summary(rows, metric_prefix: str = ""):
    by_region = defaultdict(build_empty_bucket)
    for row in rows:
        for region_key in REGION_KEYS:
            translated = {
                "valid_pixels": int(row.get(f"{region_key}_valid_pixels", 0)),
                "conf_valid_pixels": int(row.get(f"{region_key}_conf_valid_pixels", 0)),
                "conf_depth_mean": row.get(f"{metric_prefix}{region_key}_conf_depth_mean"),
                "reg_depth_mean": row.get(f"{metric_prefix}{region_key}_reg_depth_mean"),
                "pred_depth_conf_mean": None,
            }
            add_summary_bucket(by_region[region_key], translated)
    return finalize_summary(by_region)


def build_aggregate(rows):
    by_role = defaultdict(build_empty_bucket)
    by_camera = defaultdict(build_empty_bucket)
    by_frame = defaultdict(build_empty_bucket)
    for row in rows:
        add_summary_bucket(by_role[row["view_role"]], row)
        add_summary_bucket(by_camera[row["camera_name"]], row)
        add_summary_bucket(by_frame[make_frame_key(row["frame_id"])], row)
    return {
        "by_role": finalize_summary(by_role),
        "by_camera": finalize_summary(by_camera),
        "by_frame": finalize_summary(by_frame),
        "by_region": build_region_summary(rows),
    }


def build_delta_rows(primary_rows, reference_rows):
    ref_index = {
        (row["sample_index"], row["view_idx"], row["camera_name"]): row
        for row in reference_rows
    }
    delta_rows = []
    for row in primary_rows:
        key = (row["sample_index"], row["view_idx"], row["camera_name"])
        ref = ref_index.get(key)
        if ref is None:
            continue
        merged = {
            "sample_index": row["sample_index"],
            "seq_name": row["seq_name"],
            "frame_id": row["frame_id"],
            "view_idx": row["view_idx"],
            "camera_name": row["camera_name"],
            "view_role": row["view_role"],
            "selection_anchor_camera": row["selection_anchor_camera"],
            "quality_score": row["quality_score"],
            "valid_pixels": row["valid_pixels"],
            "conf_valid_pixels": row["conf_valid_pixels"],
        }
        for region_key in REGION_KEYS:
            merged[f"{region_key}_valid_pixels"] = row.get(f"{region_key}_valid_pixels", 0)
            merged[f"{region_key}_conf_valid_pixels"] = row.get(f"{region_key}_conf_valid_pixels", 0)
        for metric in METRIC_KEYS:
            merged[f"primary_{metric}"] = row[metric]
            merged[f"reference_{metric}"] = ref[metric]
            if row[metric] is None or ref[metric] is None:
                merged[f"delta_{metric}"] = None
            else:
                merged[f"delta_{metric}"] = float(row[metric] - ref[metric])
        delta_rows.append(merged)
    return delta_rows


def summarize_delta_rows(delta_rows):
    by_role = defaultdict(build_empty_bucket)
    by_camera = defaultdict(build_empty_bucket)
    by_frame = defaultdict(build_empty_bucket)
    for row in delta_rows:
        translated = {
            "view_role": row["view_role"],
            "camera_name": row["camera_name"],
            "frame_id": row["frame_id"],
            "valid_pixels": row["valid_pixels"],
            "conf_valid_pixels": row["conf_valid_pixels"],
            "conf_depth_mean": row["delta_conf_depth_mean"],
            "reg_depth_mean": row["delta_reg_depth_mean"],
            "pred_depth_conf_mean": row["delta_pred_depth_conf_mean"],
            "fg_human_conf_depth_mean": row["delta_fg_human_conf_depth_mean"],
            "bg_bottom_band_conf_depth_mean": row["delta_bg_bottom_band_conf_depth_mean"],
            "bg_far_conf_depth_mean": row["delta_bg_far_conf_depth_mean"],
            "fg_human_reg_depth_mean": row["delta_fg_human_reg_depth_mean"],
            "bg_bottom_band_reg_depth_mean": row["delta_bg_bottom_band_reg_depth_mean"],
            "bg_far_reg_depth_mean": row["delta_bg_far_reg_depth_mean"],
            "fg_bottom_conf_depth_mean": row["delta_fg_bottom_conf_depth_mean"],
            "fg_nonbottom_conf_depth_mean": row["delta_fg_nonbottom_conf_depth_mean"],
            "fg_bottom_reg_depth_mean": row["delta_fg_bottom_reg_depth_mean"],
            "fg_nonbottom_reg_depth_mean": row["delta_fg_nonbottom_reg_depth_mean"],
        }
        add_summary_bucket(by_role[row["view_role"]], translated)
        add_summary_bucket(by_camera[row["camera_name"]], translated)
        add_summary_bucket(by_frame[make_frame_key(row["frame_id"])], translated)
    return {
        "by_role": finalize_summary(by_role),
        "by_camera": finalize_summary(by_camera),
        "by_frame": finalize_summary(by_frame),
        "by_region": build_region_summary(delta_rows, metric_prefix="delta_"),
    }


def top_rows_by_metric(rows, metric, descending=True, limit=12):
    filtered = [row for row in rows if row.get(metric) is not None]
    filtered.sort(key=lambda row: row[metric], reverse=descending)
    return filtered[:limit]


def top_wrapped_rows_by_metric(rows, metric, descending=True, limit=12):
    filtered = [row for row in rows if row["row"].get(metric) is not None]
    filtered.sort(key=lambda row: row["row"][metric], reverse=descending)
    return filtered[:limit]


def format_metric(value):
    return "n/a" if value is None else f"{value:.4f}"


def first_positive_item(items, metric):
    for item in items:
        value = item["row"].get(metric)
        if value is not None and value > 0:
            return item
    return None


def nth_metric(items, metric, index):
    filtered = [item for item in items if item["row"].get(metric) is not None]
    if index >= len(filtered):
        return None
    return filtered[index]["row"].get(metric)


def build_candidate_recommendation(top_delta_cameras, top_delta_frames, top_delta_regions):
    top_camera = first_positive_item(top_delta_cameras, "conf_depth_mean")
    top_frame = first_positive_item(top_delta_frames, "conf_depth_mean")
    top_region = first_positive_item(top_delta_regions, "conf_depth_mean")

    top_camera_conf = None if top_camera is None else top_camera["row"].get("conf_depth_mean")
    top_frame_conf = None if top_frame is None else top_frame["row"].get("conf_depth_mean")
    top_region_conf = None if top_region is None else top_region["row"].get("conf_depth_mean")
    second_camera_conf = nth_metric(top_delta_cameras, "conf_depth_mean", 1)

    if (
        top_camera_conf is not None
        and top_camera_conf >= 0.05
        and (
            second_camera_conf is None
            or second_camera_conf <= 0.0
            or top_camera_conf - second_camera_conf >= 0.05
        )
    ):
        dominant_failure_shape = "anchor_conditioned"
        candidate_family = "anchor_conditioned_conf_target_normalization"
        reason = (
            f"The only clearly positive camera-level conf_depth delta is still concentrated in anchor {top_camera['key']}, "
            "so the next candidate should normalize or clip conf targets by anchor before reopening any broader family."
        )
    elif top_frame_conf is not None and top_frame_conf >= 0.05 and top_frame_conf > max(top_camera_conf or 0.0, top_region_conf or 0.0):
        dominant_failure_shape = "frame_concentrated"
        candidate_family = "frame_conditioned_conf_target_normalization"
        reason = (
            f"The worst remaining conf_depth delta is concentrated in {top_frame['key']}, "
            "so the next candidate should stay local and test a frame-conditioned conf target rule rather than another sampler axis."
        )
    elif top_region_conf is not None and top_region_conf >= 0.05:
        dominant_failure_shape = "region_conditioned"
        candidate_family = "region_conditioned_conf_target_normalization"
        reason = (
            f"The largest positive conf_depth delta is concentrated in {REGION_DISPLAY_NAMES.get(top_region['key'], top_region['key'])}, "
            "so the next candidate should normalize or weight conf targets by region."
        )
    else:
        dominant_failure_shape = "tail_distribution"
        candidate_family = "robust_conf_target_aggregation"
        reason = (
            "No single anchor, frame, or region dominates the remaining gap strongly enough, "
            "so the next candidate should target a robust conf-target aggregation rule that is materially different from active_view_mean."
        )

    return {
        "dominant_failure_shape": dominant_failure_shape,
        "recommended_candidate_family": candidate_family,
        "reason": reason,
        "top_anchor": None if top_camera is None else {
            "camera_name": top_camera["key"],
            "delta_conf_depth_mean": top_camera["row"].get("conf_depth_mean"),
            "delta_reg_depth_mean": top_camera["row"].get("reg_depth_mean"),
        },
        "top_frame": None if top_frame is None else {
            "frame_key": top_frame["key"],
            "delta_conf_depth_mean": top_frame["row"].get("conf_depth_mean"),
            "delta_reg_depth_mean": top_frame["row"].get("reg_depth_mean"),
        },
        "top_region": None if top_region is None else {
            "region": top_region["key"],
            "delta_conf_depth_mean": top_region["row"].get("conf_depth_mean"),
            "delta_reg_depth_mean": top_region["row"].get("reg_depth_mean"),
        },
    }


def write_markdown(output_path: Path, payload: dict):
    lines = [
        "# ZJU Conf-Depth Attribution Audit",
        "",
        f"- config_path: `{payload['config_path']}`",
        f"- split: `{payload['split']}`",
        f"- num_samples: `{payload['num_samples']}`",
        f"- num_images: `{payload['num_images']}`",
        f"- zju_dir: `{payload['zju_dir']}`",
        f"- primary_label: `{payload['primary_label']}`",
        f"- reference_label: `{payload['reference_label']}`",
        f"- dominant_failure_shape: `{payload['candidate_recommendation']['dominant_failure_shape']}`",
        f"- recommended_candidate_family: `{payload['candidate_recommendation']['recommended_candidate_family']}`",
        "",
        "## Candidate Recommendation",
        "",
        f"- reason: {payload['candidate_recommendation']['reason']}",
    ]
    top_anchor = payload["candidate_recommendation"].get("top_anchor")
    if top_anchor is not None:
        lines.append(
            "- top_anchor: `{camera}` conf={conf} reg={reg}".format(
                camera=top_anchor["camera_name"],
                conf=format_metric(top_anchor.get("delta_conf_depth_mean")),
                reg=format_metric(top_anchor.get("delta_reg_depth_mean")),
            )
        )
    top_frame = payload["candidate_recommendation"].get("top_frame")
    if top_frame is not None:
        lines.append(
            "- top_frame: `{frame}` conf={conf} reg={reg}".format(
                frame=top_frame["frame_key"],
                conf=format_metric(top_frame.get("delta_conf_depth_mean")),
                reg=format_metric(top_frame.get("delta_reg_depth_mean")),
            )
        )
    top_region = payload["candidate_recommendation"].get("top_region")
    if top_region is not None:
        lines.append(
            "- top_region: `{region}` conf={conf} reg={reg}".format(
                region=top_region["region"],
                conf=format_metric(top_region.get("delta_conf_depth_mean")),
                reg=format_metric(top_region.get("delta_reg_depth_mean")),
            )
        )
    lines.extend(
        [
            "",
            "## Primary By Role",
            "",
            "| role | conf_depth_mean | reg_depth_mean | pred_depth_conf_mean | fg_bottom_conf_depth_mean | fg_nonbottom_conf_depth_mean | conf_valid_pixels |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for role, row in payload["primary_summary"]["by_role"].items():
        lines.append(
            "| {role} | {conf} | {reg} | {pred_conf} | {bottom} | {nonbottom} | {pixels} |".format(
                role=role,
                conf=format_metric(row["conf_depth_mean"]),
                reg=format_metric(row["reg_depth_mean"]),
                pred_conf=format_metric(row["pred_depth_conf_mean"]),
                bottom=format_metric(row["fg_bottom_conf_depth_mean"]),
                nonbottom=format_metric(row["fg_nonbottom_conf_depth_mean"]),
                pixels=row["conf_valid_pixels"],
            )
        )

    if payload["delta_summary"] is not None:
        lines.extend(
            [
                "",
                "## Delta By Role",
                "",
                f"| role | delta conf_depth_mean ({payload['primary_label']} - {payload['reference_label']}) | delta reg_depth_mean | delta pred_depth_conf_mean | delta fg_bottom_conf_depth_mean | delta fg_nonbottom_conf_depth_mean |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for role, row in payload["delta_summary"]["by_role"].items():
            lines.append(
                "| {role} | {conf} | {reg} | {pred_conf} | {bottom} | {nonbottom} |".format(
                    role=role,
                    conf=format_metric(row["conf_depth_mean"]),
                    reg=format_metric(row["reg_depth_mean"]),
                    pred_conf=format_metric(row["pred_depth_conf_mean"]),
                    bottom=format_metric(row["fg_bottom_conf_depth_mean"]),
                    nonbottom=format_metric(row["fg_nonbottom_conf_depth_mean"]),
                )
            )

    lines.extend(
        [
            "",
            "## Anchor Summary",
            "",
            "| camera | conf_depth_mean | reg_depth_mean | pred_depth_conf_mean | fg_bottom_conf_depth_mean | view_count |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in payload["anchor_summary"]["top_primary"]:
        lines.append(
            "| {camera} | {conf} | {reg} | {pred_conf} | {bottom} | {count} |".format(
                camera=item["key"],
                conf=format_metric(item["row"]["conf_depth_mean"]),
                reg=format_metric(item["row"]["reg_depth_mean"]),
                pred_conf=format_metric(item["row"]["pred_depth_conf_mean"]),
                bottom=format_metric(item["row"]["fg_bottom_conf_depth_mean"]),
                count=item["row"]["view_count"],
            )
        )

    if payload["anchor_summary"]["top_delta"]:
        lines.extend(
            [
                "",
                f"### Top Worsened Anchors ({payload['primary_label']} - {payload['reference_label']})",
                "",
                "| camera | delta conf_depth_mean | delta reg_depth_mean | conf_valid_pixels |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for item in payload["anchor_summary"]["top_delta"]:
            lines.append(
                "| {camera} | {conf} | {reg} | {pixels} |".format(
                    camera=item["key"],
                    conf=format_metric(item["row"]["conf_depth_mean"]),
                    reg=format_metric(item["row"]["reg_depth_mean"]),
                    pixels=item["row"]["conf_valid_pixels"],
                )
            )

    lines.extend(
        [
            "",
            "## Frame Summary",
            "",
            "| frame | conf_depth_mean | reg_depth_mean | pred_depth_conf_mean | conf_valid_pixels |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in payload["frame_summary"]["top_primary"]:
        lines.append(
            "| {frame} | {conf} | {reg} | {pred_conf} | {pixels} |".format(
                frame=item["key"],
                conf=format_metric(item["row"]["conf_depth_mean"]),
                reg=format_metric(item["row"]["reg_depth_mean"]),
                pred_conf=format_metric(item["row"]["pred_depth_conf_mean"]),
                pixels=item["row"]["conf_valid_pixels"],
            )
        )

    if payload["frame_summary"]["top_delta"]:
        lines.extend(
            [
                "",
                f"### Top Worsened Frames ({payload['primary_label']} - {payload['reference_label']})",
                "",
                "| frame | delta conf_depth_mean | delta reg_depth_mean | conf_valid_pixels |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for item in payload["frame_summary"]["top_delta"]:
            lines.append(
                "| {frame} | {conf} | {reg} | {pixels} |".format(
                    frame=item["key"],
                    conf=format_metric(item["row"]["conf_depth_mean"]),
                    reg=format_metric(item["row"]["reg_depth_mean"]),
                    pixels=item["row"]["conf_valid_pixels"],
                )
            )

    lines.extend(
        [
            "",
            "## Region Summary",
            "",
            "| region | conf_depth_mean | reg_depth_mean | conf_valid_pixels | valid_pixels |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in payload["region_summary"]["top_primary"]:
        lines.append(
            "| {region} | {conf} | {reg} | {conf_pixels} | {valid_pixels} |".format(
                region=item["key"],
                conf=format_metric(item["row"]["conf_depth_mean"]),
                reg=format_metric(item["row"]["reg_depth_mean"]),
                conf_pixels=item["row"]["conf_valid_pixels"],
                valid_pixels=item["row"]["valid_pixels"],
            )
        )

    if payload["region_summary"]["top_delta"]:
        lines.extend(
            [
                "",
                f"### Top Worsened Regions ({payload['primary_label']} - {payload['reference_label']})",
                "",
                "| region | delta conf_depth_mean | delta reg_depth_mean | conf_valid_pixels | valid_pixels |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in payload["region_summary"]["top_delta"]:
            lines.append(
                "| {region} | {conf} | {reg} | {conf_pixels} | {valid_pixels} |".format(
                    region=item["key"],
                    conf=format_metric(item["row"]["conf_depth_mean"]),
                    reg=format_metric(item["row"]["reg_depth_mean"]),
                    conf_pixels=item["row"]["conf_valid_pixels"],
                    valid_pixels=item["row"]["valid_pixels"],
                )
            )

    if payload["top_delta_views"]:
        lines.extend(
            [
                "",
                f"## Top Worsened Views ({payload['primary_label']} - {payload['reference_label']})",
                "",
                "| seq | camera | role | quality_score | delta conf_depth_mean | delta reg_depth_mean | delta fg_bottom_conf_depth_mean |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in payload["top_delta_views"]:
            lines.append(
                "| {seq} | {camera} | {role} | {score} | {conf} | {reg} | {bottom} |".format(
                    seq=row["seq_name"],
                    camera=row["camera_name"],
                    role=row["view_role"],
                    score=format_metric(row["quality_score"]),
                    conf=format_metric(row["delta_conf_depth_mean"]),
                    reg=format_metric(row["delta_reg_depth_mean"]),
                    bottom=format_metric(row["delta_fg_bottom_conf_depth_mean"]),
                )
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = resolve_config_path(args.config)
    cfg, dataset, resolved_zju_dir = load_dataset_and_cfg(
        config_path=config_path,
        split=args.split,
        zju_dir_override=args.zju_dir,
        num_images=args.num_images,
    )
    device = torch.device(args.device)
    primary_checkpoint = Path(args.checkpoint).resolve()
    primary_model = load_model(cfg.model, primary_checkpoint, device)
    reference_model = None
    reference_checkpoint = None
    if args.reference_checkpoint:
        reference_checkpoint = Path(args.reference_checkpoint).resolve()
        reference_model = load_model(cfg.model, reference_checkpoint, device)

    gamma = float(cfg.loss.depth.gamma)
    alpha = float(cfg.loss.depth.alpha)
    sample_count = min(int(args.num_samples), len(dataset))

    primary_rows = []
    reference_rows = []
    sample_debug_rows = []
    for sample_index in range(sample_count):
        sample = dataset.get_data(seq_index=sample_index, img_per_seq=int(args.num_images))
        sample["audit_sample_index"] = int(sample_index)
        batch = normalize_batch(sample_to_batch(sample))

        primary_rows.extend(
            run_model_rows(primary_model, args.label, batch, sample, gamma=gamma, alpha=alpha, device=device)
        )
        if reference_model is not None:
            reference_rows.extend(
                run_model_rows(reference_model, args.reference_label, batch, sample, gamma=gamma, alpha=alpha, device=device)
            )
        sample_debug_rows.append(
            {
                "sample_index": int(sample_index),
                "seq_name": str(sample["seq_name"]),
                "selection_anchor_camera": sample.get("selection_anchor_camera"),
                "camera_names": list(sample.get("camera_names", [])),
                "candidate_supervised_camera_names": list(sample.get("candidate_supervised_camera_names", [])),
                "supervised_camera_names": list(sample.get("supervised_camera_names", [])),
                "conf_depth_camera_names": list(sample.get("conf_depth_camera_names", [])),
                "source_only_camera_names": list(sample.get("source_only_camera_names", [])),
                "dropped_supervised_camera_names": list(sample.get("dropped_supervised_camera_names", [])),
                "conf_depth_dropped_camera_names": list(sample.get("conf_depth_dropped_camera_names", [])),
                "supervised_view_quality_scores": dict(sample.get("supervised_view_quality_scores", {})),
            }
        )

    primary_summary = build_aggregate(primary_rows)
    reference_summary = None
    delta_rows = None
    delta_summary = None
    top_delta_views = []
    top_delta_cameras = []
    top_delta_frames = []
    top_delta_regions = []
    if reference_rows:
        reference_summary = build_aggregate(reference_rows)
        delta_rows = build_delta_rows(primary_rows, reference_rows)
        delta_summary = summarize_delta_rows(delta_rows)
        top_delta_views = top_rows_by_metric(delta_rows, "delta_conf_depth_mean", descending=True, limit=16)
        top_delta_cameras = top_wrapped_rows_by_metric(
            [{"key": key, "row": row} for key, row in delta_summary["by_camera"].items()],
            "conf_depth_mean",
            descending=True,
            limit=12,
        )
        top_delta_frames = top_wrapped_rows_by_metric(
            [{"key": key, "row": row} for key, row in delta_summary["by_frame"].items()],
            "conf_depth_mean",
            descending=True,
            limit=12,
        )
        top_delta_regions = top_wrapped_rows_by_metric(
            [{"key": key, "row": row} for key, row in delta_summary["by_region"].items()],
            "conf_depth_mean",
            descending=True,
            limit=12,
        )

    top_primary_cameras = top_wrapped_rows_by_metric(
        [{"key": key, "row": row} for key, row in primary_summary["by_camera"].items()],
        "conf_depth_mean",
        descending=True,
        limit=12,
    )
    top_primary_frames = top_wrapped_rows_by_metric(
        [{"key": key, "row": row} for key, row in primary_summary["by_frame"].items()],
        "conf_depth_mean",
        descending=True,
        limit=12,
    )
    top_primary_regions = top_wrapped_rows_by_metric(
        [{"key": key, "row": row} for key, row in primary_summary["by_region"].items()],
        "conf_depth_mean",
        descending=True,
        limit=12,
    )
    candidate_recommendation = build_candidate_recommendation(
        top_delta_cameras=top_delta_cameras,
        top_delta_frames=top_delta_frames,
        top_delta_regions=top_delta_regions,
    )

    payload = {
        "config_path": repo_rel(config_path),
        "split": args.split,
        "num_samples": int(sample_count),
        "num_images": int(args.num_images),
        "zju_dir": str(resolved_zju_dir),
        "primary_label": str(args.label),
        "primary_checkpoint": repo_rel(primary_checkpoint),
        "reference_label": str(args.reference_label),
        "reference_checkpoint": None if reference_checkpoint is None else repo_rel(reference_checkpoint),
        "gamma": gamma,
        "alpha": alpha,
        "sample_debug_rows": sample_debug_rows,
        "primary_summary": primary_summary,
        "reference_summary": reference_summary,
        "delta_summary": delta_summary,
        "anchor_summary": {
            "primary": primary_summary["by_camera"],
            "delta": None if delta_summary is None else delta_summary["by_camera"],
            "top_primary": top_primary_cameras,
            "top_delta": top_delta_cameras,
        },
        "frame_summary": {
            "primary": primary_summary["by_frame"],
            "delta": None if delta_summary is None else delta_summary["by_frame"],
            "top_primary": top_primary_frames,
            "top_delta": top_delta_frames,
        },
        "region_summary": {
            "primary": primary_summary["by_region"],
            "delta": None if delta_summary is None else delta_summary["by_region"],
            "top_primary": top_primary_regions,
            "top_delta": top_delta_regions,
        },
        "candidate_recommendation": candidate_recommendation,
        "top_primary_cameras": top_primary_cameras,
        "top_primary_frames": top_primary_frames,
        "top_primary_regions": top_primary_regions,
        "top_delta_views": top_delta_views,
        "top_delta_cameras": top_delta_cameras,
        "top_delta_frames": top_delta_frames,
        "top_delta_regions": top_delta_regions,
        "primary_rows": primary_rows,
        "reference_rows": reference_rows,
        "delta_rows": delta_rows,
    }

    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, payload)
    print(md_path)


if __name__ == "__main__":
    main()
