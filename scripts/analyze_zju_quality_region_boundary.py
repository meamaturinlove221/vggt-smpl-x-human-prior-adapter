import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"

for root in (str(REPO_ROOT), str(TRAINING_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from loss import build_reliable_foreground_region_mask
from training.data.datasets.zju_vggt_geom import ZjuVggtGeomDataset


DEFAULT_CONFIG = (
    "zju_vggt_geom_unproject_source_policy_nearest_rawpool_"
    "confdepth_dropworst_gradconfmask_minimal"
)
DEFAULT_QUALITY_SUMMARY = (
    REPO_ROOT
    / "output"
    / "zju_conf_depth_quality_signal_confdepth_dropworst_gradconfmask_20260326_v1"
    / "summary.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "output"
    / "zju_quality_region_boundary_confdepth_dropworst_gradconfmask_20260326_v1"
)
DEFAULT_QUALITY_THRESHOLDS = "2.75,3.0,3.25,3.5"
DEFAULT_BOTTOM_RATIOS = "0.1,0.2,0.3,0.4"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Separate quality-threshold selectivity from region selectivity for the "
            "remaining Camera_B1 conf-depth residual on the stable gradconfmask lead."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--quality-summary", type=Path, default=DEFAULT_QUALITY_SUMMARY)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--zju-dir", type=str, default="")
    parser.add_argument("--quality-thresholds", type=str, default=DEFAULT_QUALITY_THRESHOLDS)
    parser.add_argument("--bottom-ratios", type=str, default=DEFAULT_BOTTOM_RATIOS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def parse_float_list(raw_value: str) -> list[float]:
    values = []
    for item in ensure_list(raw_value):
        values.append(float(item))
    return values


def mean_or_none(values):
    if not values:
        return None
    return float(sum(values) / len(values))


def format_float(value):
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def threshold_label(threshold):
    if threshold is None:
        return "none"
    return f"q>={float(threshold):.2f}"


def region_label(bottom_ratio):
    if bottom_ratio is None:
        return "whole_foreground"
    return f"bottom_{float(bottom_ratio):.2f}"


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
    return OmegaConf.merge(merged, cfg)


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


def load_dataset(config_path: Path, split: str, zju_dir_override: str, num_images: int):
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
    return dataset, resolved_zju_dir


def summarize_rows(rows: list[dict], total_count: int) -> dict:
    values = {
        "quality_score": [float(row["quality_score"]) for row in rows if row.get("quality_score") is not None],
        "delta_conf_depth_mean": [float(row["delta_conf_depth_mean"]) for row in rows if row.get("delta_conf_depth_mean") is not None],
        "delta_reg_depth_mean": [float(row["delta_reg_depth_mean"]) for row in rows if row.get("delta_reg_depth_mean") is not None],
        "delta_fg_bottom_conf_depth_mean": [
            float(row["delta_fg_bottom_conf_depth_mean"])
            for row in rows
            if row.get("delta_fg_bottom_conf_depth_mean") is not None
        ],
        "delta_fg_nonbottom_conf_depth_mean": [
            float(row["delta_fg_nonbottom_conf_depth_mean"])
            for row in rows
            if row.get("delta_fg_nonbottom_conf_depth_mean") is not None
        ],
        "delta_fg_bottom_reg_depth_mean": [
            float(row["delta_fg_bottom_reg_depth_mean"])
            for row in rows
            if row.get("delta_fg_bottom_reg_depth_mean") is not None
        ],
        "delta_fg_nonbottom_reg_depth_mean": [
            float(row["delta_fg_nonbottom_reg_depth_mean"])
            for row in rows
            if row.get("delta_fg_nonbottom_reg_depth_mean") is not None
        ],
    }
    bottom_minus_nonbottom_conf = []
    bottom_minus_nonbottom_reg = []
    positive_nonbottom_conf_rows = 0
    bottom_gt_nonbottom_conf_rows = 0
    for row in rows:
        bottom_conf = row.get("delta_fg_bottom_conf_depth_mean")
        nonbottom_conf = row.get("delta_fg_nonbottom_conf_depth_mean")
        bottom_reg = row.get("delta_fg_bottom_reg_depth_mean")
        nonbottom_reg = row.get("delta_fg_nonbottom_reg_depth_mean")
        if nonbottom_conf is not None and float(nonbottom_conf) > 0.0:
            positive_nonbottom_conf_rows += 1
        if bottom_conf is not None and nonbottom_conf is not None:
            bottom_conf = float(bottom_conf)
            nonbottom_conf = float(nonbottom_conf)
            bottom_minus_nonbottom_conf.append(bottom_conf - nonbottom_conf)
            if bottom_conf > nonbottom_conf:
                bottom_gt_nonbottom_conf_rows += 1
        if bottom_reg is not None and nonbottom_reg is not None:
            bottom_minus_nonbottom_reg.append(float(bottom_reg) - float(nonbottom_reg))

    return {
        "count": len(rows),
        "selected_fraction": (len(rows) / total_count) if total_count else None,
        "quality_score_mean": mean_or_none(values["quality_score"]),
        "delta_conf_depth_mean": mean_or_none(values["delta_conf_depth_mean"]),
        "delta_reg_depth_mean": mean_or_none(values["delta_reg_depth_mean"]),
        "delta_fg_bottom_conf_depth_mean": mean_or_none(values["delta_fg_bottom_conf_depth_mean"]),
        "delta_fg_nonbottom_conf_depth_mean": mean_or_none(values["delta_fg_nonbottom_conf_depth_mean"]),
        "delta_fg_bottom_reg_depth_mean": mean_or_none(values["delta_fg_bottom_reg_depth_mean"]),
        "delta_fg_nonbottom_reg_depth_mean": mean_or_none(values["delta_fg_nonbottom_reg_depth_mean"]),
        "bottom_minus_nonbottom_conf_depth_mean": mean_or_none(bottom_minus_nonbottom_conf),
        "bottom_minus_nonbottom_reg_depth_mean": mean_or_none(bottom_minus_nonbottom_reg),
        "positive_nonbottom_conf_rows": int(positive_nonbottom_conf_rows),
        "bottom_gt_nonbottom_conf_rows": int(bottom_gt_nonbottom_conf_rows),
    }


def build_quality_threshold_sweep(anchor_rows: list[dict], thresholds: list[float]) -> list[dict]:
    sweep = []
    total_count = len(anchor_rows)
    for threshold in [None, *thresholds]:
        if threshold is None:
            selected_rows = list(anchor_rows)
        else:
            selected_rows = [
                row
                for row in anchor_rows
                if row.get("quality_score") is not None and float(row["quality_score"]) >= float(threshold)
            ]
        summary = summarize_rows(selected_rows, total_count=total_count)
        sweep.append(
            {
                "quality_threshold": threshold,
                "threshold_label": threshold_label(threshold),
                **summary,
            }
        )
    return sweep


def build_region_mask(foreground_masks: np.ndarray, bottom_ratio):
    if bottom_ratio is None:
        return foreground_masks
    foreground_tensor = torch.from_numpy(foreground_masks.astype(bool)).unsqueeze(0)
    nonbottom_mask = build_reliable_foreground_region_mask(
        foreground_tensor,
        erode_px=0,
        drop_bottom_ratio=float(bottom_ratio),
    )[0].cpu().numpy().astype(bool)
    return foreground_masks & (~nonbottom_mask)


def compute_train_slice_coverage(
    dataset,
    top_anchor_camera: str,
    thresholds: list[float],
    bottom_ratios: list[float],
    num_samples: int,
    num_images: int,
) -> dict:
    threshold_options = [None, *thresholds]
    region_options = [None, *bottom_ratios]
    coverage = {
        (threshold_label(threshold), region_label(bottom_ratio)): {
            "quality_threshold": threshold,
            "threshold_label": threshold_label(threshold),
            "region": region_label(bottom_ratio),
            "bottom_ratio": bottom_ratio,
            "eligible_samples": 0,
            "eligible_conf_depth_pixels": 0,
            "top_anchor_eligible_samples": 0,
            "top_anchor_quality_scores": [],
        }
        for threshold in threshold_options
        for bottom_ratio in region_options
    }

    total_samples = 0
    total_conf_depth_pixels = 0
    top_anchor_sample_count = 0
    max_samples = min(int(num_samples), len(dataset))

    for sample_idx in range(max_samples):
        sample = dataset.get_data(seq_index=sample_idx, img_per_seq=num_images)
        total_samples += 1

        anchor_camera = sample.get("selection_anchor_camera")
        anchor_camera = None if anchor_camera is None else str(anchor_camera)
        quality_scores = sample.get("supervised_view_quality_scores", {})
        anchor_quality = None
        if anchor_camera == top_anchor_camera:
            raw_quality = quality_scores.get(anchor_camera)
            if raw_quality is not None:
                anchor_quality = float(raw_quality)
            top_anchor_sample_count += 1

        conf_mask = np.stack(sample["conf_depth_point_masks"]).astype(bool)
        foreground_masks = np.stack(sample["foreground_masks"]).astype(bool)
        total_conf_depth_pixels += int(conf_mask.sum())
        region_masks = {
            region_label(bottom_ratio): build_region_mask(foreground_masks, bottom_ratio)
            for bottom_ratio in region_options
        }

        for threshold in threshold_options:
            quality_ok = anchor_camera == top_anchor_camera
            if threshold is not None:
                quality_ok = quality_ok and anchor_quality is not None and anchor_quality >= float(threshold)
            if not quality_ok:
                continue

            for bottom_ratio in region_options:
                entry = coverage[(threshold_label(threshold), region_label(bottom_ratio))]
                entry["eligible_samples"] += 1
                if anchor_camera == top_anchor_camera:
                    entry["top_anchor_eligible_samples"] += 1
                    if anchor_quality is not None:
                        entry["top_anchor_quality_scores"].append(anchor_quality)
                entry["eligible_conf_depth_pixels"] += int(
                    (conf_mask & region_masks[region_label(bottom_ratio)]).sum()
                )

    rows = []
    for (_, _), entry in coverage.items():
        quality_scores = sorted(entry.pop("top_anchor_quality_scores"))
        quality_median = None
        if quality_scores:
            quality_median = float(np.median(np.asarray(quality_scores, dtype=np.float32)))
        rows.append(
            {
                **entry,
                "num_samples": total_samples,
                "num_images": num_images,
                "top_anchor_sample_count": top_anchor_sample_count,
                "eligible_sample_fraction_total": (entry["eligible_samples"] / total_samples) if total_samples else None,
                "eligible_sample_fraction_top_anchor": (
                    entry["top_anchor_eligible_samples"] / top_anchor_sample_count
                    if top_anchor_sample_count
                    else None
                ),
                "eligible_conf_depth_pixel_fraction_total": (
                    entry["eligible_conf_depth_pixels"] / total_conf_depth_pixels
                    if total_conf_depth_pixels
                    else None
                ),
                "top_anchor_quality_median": quality_median,
                "top_anchor_quality_min": (float(quality_scores[0]) if quality_scores else None),
                "top_anchor_quality_max": (float(quality_scores[-1]) if quality_scores else None),
            }
        )

    rows.sort(key=lambda item: (item["threshold_label"], item["region"]))
    return {
        "num_samples": total_samples,
        "num_images": num_images,
        "top_anchor_camera": top_anchor_camera,
        "top_anchor_sample_count": top_anchor_sample_count,
        "total_conf_depth_pixels": total_conf_depth_pixels,
        "coverage_rows": rows,
    }


def build_recommendation(anchor_rows: list[dict], audit_sweep: list[dict], coverage_payload: dict) -> dict:
    total_rows = len(anchor_rows)
    full_audit = audit_sweep[0] if audit_sweep else {}
    current_rule = None
    q3_whole = None
    no_threshold_bottom20 = None
    no_threshold_whole = None
    for row in coverage_payload["coverage_rows"]:
        if row["threshold_label"] == "q>=3.00" and row["region"] == "bottom_0.20":
            current_rule = row
        elif row["threshold_label"] == "q>=3.00" and row["region"] == "whole_foreground":
            q3_whole = row
        elif row["threshold_label"] == "none" and row["region"] == "bottom_0.20":
            no_threshold_bottom20 = row
        elif row["threshold_label"] == "none" and row["region"] == "whole_foreground":
            no_threshold_whole = row

    nonbottom_positive_all = (
        full_audit.get("positive_nonbottom_conf_rows") == total_rows and total_rows > 0
    )
    bottom_dominates_all = (
        full_audit.get("bottom_gt_nonbottom_conf_rows") == total_rows and total_rows > 0
    )
    current_pixel_fraction = (
        None if current_rule is None else current_rule.get("eligible_conf_depth_pixel_fraction_total")
    )
    whole_q3_pixel_fraction = (
        None if q3_whole is None else q3_whole.get("eligible_conf_depth_pixel_fraction_total")
    )
    no_threshold_bottom20_fraction = (
        None
        if no_threshold_bottom20 is None
        else no_threshold_bottom20.get("eligible_conf_depth_pixel_fraction_total")
    )
    no_threshold_whole_fraction = (
        None
        if no_threshold_whole is None
        else no_threshold_whole.get("eligible_conf_depth_pixel_fraction_total")
    )

    reason_parts = []
    if total_rows > 0:
        reason_parts.append(
            f"All {total_rows} audited bad-anchor rows keep positive non-bottom conf_depth deltas, "
            f"and the bottom band is worse on {full_audit.get('bottom_gt_nonbottom_conf_rows', 0)} / {total_rows} rows."
        )
    if current_pixel_fraction is not None:
        reason_parts.append(
            f"The rejected hard q>=3.0 + bottom20 rule touches only "
            f"{format_float(current_pixel_fraction * 100.0)}% of conf-depth-supervised pixels."
        )
    if whole_q3_pixel_fraction is not None and current_pixel_fraction is not None:
        reason_parts.append(
            f"Keeping q>=3.0 but widening the region to whole foreground would raise coverage to "
            f"{format_float(whole_q3_pixel_fraction * 100.0)}%."
        )
    if no_threshold_bottom20_fraction is not None and current_pixel_fraction is not None:
        reason_parts.append(
            f"Dropping the hard quality threshold while keeping bottom20 would raise coverage to "
            f"{format_float(no_threshold_bottom20_fraction * 100.0)}%."
        )
    if no_threshold_whole_fraction is not None:
        reason_parts.append(
            f"Camera_B1 whole-foreground coverage without a hard threshold is "
            f"{format_float(no_threshold_whole_fraction * 100.0)}%."
        )

    if nonbottom_positive_all and current_pixel_fraction is not None and current_pixel_fraction < 0.01:
        selectivity_blocker = "quality_threshold_and_bottom20_both_over_selective"
        next_question_shape = "manual_broader_or_continuous_quality_conditioned_foreground_rule"
    elif nonbottom_positive_all:
        selectivity_blocker = "bottom20_region_over_selective"
        next_question_shape = "manual_broader_region_conditioned_quality_rule"
    else:
        selectivity_blocker = "quality_threshold_primary_selectivity_blocker"
        next_question_shape = "manual_softer_quality_conditioned_rule"

    return {
        "status": "boundary_check_complete",
        "selectivity_blocker": selectivity_blocker,
        "next_question_shape": next_question_shape,
        "reason": " ".join(reason_parts),
        "nonbottom_positive_all_rows": nonbottom_positive_all,
        "bottom_dominates_all_rows": bottom_dominates_all,
        "current_qge3_bottom20_pixel_fraction_total": current_pixel_fraction,
        "qge3_whole_foreground_pixel_fraction_total": whole_q3_pixel_fraction,
        "no_threshold_bottom20_pixel_fraction_total": no_threshold_bottom20_fraction,
        "no_threshold_whole_foreground_pixel_fraction_total": no_threshold_whole_fraction,
    }


def main():
    args = parse_args()
    config_path = resolve_config_path(args.config)
    quality_payload = load_json(args.quality_summary)
    attribution_summary_path = Path(quality_payload["attribution_summary"])
    attribution_payload = load_json(attribution_summary_path)

    thresholds = parse_float_list(args.quality_thresholds)
    bottom_ratios = parse_float_list(args.bottom_ratios)
    top_anchor_camera = str(quality_payload["top_anchor_camera"])

    anchor_rows = [
        row
        for row in attribution_payload.get("delta_rows", [])
        if row.get("view_role") == "anchor_supervised"
        and str(row.get("camera_name")) == top_anchor_camera
        and row.get("quality_score") is not None
        and row.get("delta_conf_depth_mean") is not None
        and row.get("delta_reg_depth_mean") is not None
        and row.get("delta_fg_bottom_conf_depth_mean") is not None
        and row.get("delta_fg_nonbottom_conf_depth_mean") is not None
        and row.get("delta_fg_bottom_reg_depth_mean") is not None
        and row.get("delta_fg_nonbottom_reg_depth_mean") is not None
    ]

    audit_summary = summarize_rows(anchor_rows, total_count=len(anchor_rows))
    audit_threshold_sweep = build_quality_threshold_sweep(anchor_rows, thresholds)
    dataset, resolved_zju_dir = load_dataset(
        config_path=config_path,
        split=args.split,
        zju_dir_override=args.zju_dir,
        num_images=args.num_images,
    )
    coverage_payload = compute_train_slice_coverage(
        dataset=dataset,
        top_anchor_camera=top_anchor_camera,
        thresholds=thresholds,
        bottom_ratios=bottom_ratios,
        num_samples=args.num_samples,
        num_images=args.num_images,
    )
    recommendation = build_recommendation(anchor_rows, audit_threshold_sweep, coverage_payload)

    result = {
        "config": str(config_path.resolve()),
        "quality_summary": str(args.quality_summary.resolve()),
        "attribution_summary": str(attribution_summary_path.resolve()),
        "split": args.split,
        "num_samples": coverage_payload["num_samples"],
        "num_images": coverage_payload["num_images"],
        "zju_dir": str(resolved_zju_dir),
        "top_anchor_camera": top_anchor_camera,
        "quality_thresholds": thresholds,
        "bottom_ratios": bottom_ratios,
        "audit_summary": {
            "anchor_row_count": len(anchor_rows),
            **audit_summary,
            "quality_threshold_sweep": audit_threshold_sweep,
        },
        "train_slice_coverage": coverage_payload,
        "recommendation": recommendation,
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", result)

    lines = [
        "# ZJU Quality-Region Boundary Check",
        "",
        f"- config: `{config_path}`",
        f"- quality_summary: `{args.quality_summary}`",
        f"- attribution_summary: `{attribution_summary_path}`",
        f"- split: `{args.split}`",
        f"- num_samples: `{coverage_payload['num_samples']}`",
        f"- top_anchor_camera: `{top_anchor_camera}`",
        "",
        "## Audit Summary",
        "",
        f"- anchor_row_count: `{len(anchor_rows)}`",
        f"- delta_conf_depth_mean: `{format_float(audit_summary['delta_conf_depth_mean'])}`",
        f"- delta_reg_depth_mean: `{format_float(audit_summary['delta_reg_depth_mean'])}`",
        f"- delta_fg_bottom_conf_depth_mean: `{format_float(audit_summary['delta_fg_bottom_conf_depth_mean'])}`",
        f"- delta_fg_nonbottom_conf_depth_mean: `{format_float(audit_summary['delta_fg_nonbottom_conf_depth_mean'])}`",
        f"- bottom_minus_nonbottom_conf_depth_mean: `{format_float(audit_summary['bottom_minus_nonbottom_conf_depth_mean'])}`",
        f"- positive_nonbottom_conf_rows: `{audit_summary['positive_nonbottom_conf_rows']}`",
        f"- bottom_gt_nonbottom_conf_rows: `{audit_summary['bottom_gt_nonbottom_conf_rows']}`",
        "",
        "## Audit Threshold Sweep",
        "",
        "| threshold | count | selected_fraction_pct | quality_score_mean | delta_conf_depth_mean | delta_reg_depth_mean | delta_fg_bottom_conf_depth_mean | delta_fg_nonbottom_conf_depth_mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in audit_threshold_sweep:
        lines.append(
            "| {threshold} | {count} | {selected_pct} | {quality_mean} | {delta_conf} | {delta_reg} | {delta_bottom} | {delta_nonbottom} |".format(
                threshold=row["threshold_label"],
                count=row["count"],
                selected_pct=format_float((row["selected_fraction"] or 0.0) * 100.0),
                quality_mean=format_float(row["quality_score_mean"]),
                delta_conf=format_float(row["delta_conf_depth_mean"]),
                delta_reg=format_float(row["delta_reg_depth_mean"]),
                delta_bottom=format_float(row["delta_fg_bottom_conf_depth_mean"]),
                delta_nonbottom=format_float(row["delta_fg_nonbottom_conf_depth_mean"]),
            )
        )

    lines.extend(
        [
            "",
            "## Train-Slice Coverage",
            "",
            "| threshold | region | eligible_samples | eligible_sample_fraction_total_pct | eligible_conf_depth_pixel_fraction_total_pct |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in coverage_payload["coverage_rows"]:
        lines.append(
            "| {threshold} | {region} | {samples} | {sample_pct} | {pixel_pct} |".format(
                threshold=row["threshold_label"],
                region=row["region"],
                samples=row["eligible_samples"],
                sample_pct=format_float((row["eligible_sample_fraction_total"] or 0.0) * 100.0),
                pixel_pct=format_float((row["eligible_conf_depth_pixel_fraction_total"] or 0.0) * 100.0),
            )
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- status: `{recommendation['status']}`",
            f"- selectivity_blocker: `{recommendation['selectivity_blocker']}`",
            f"- next_question_shape: `{recommendation['next_question_shape']}`",
            f"- reason: {recommendation['reason']}",
            "",
        ]
    )

    write_text(output_dir / "summary.md", "\n".join(lines))
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
