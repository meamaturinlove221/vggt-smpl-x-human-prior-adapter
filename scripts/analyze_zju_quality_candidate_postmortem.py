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
    "confdepth_dropworst_gradconfmask_anchorb1qge3bottom20confscale05_minimal"
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
    / "zju_quality_conditioned_candidate_postmortem_anchorb1qge3bottom20_20260326_v1"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Postmortem a rejected quality-conditioned conf-target candidate by combining "
            "quality-audit rows with train-slice activation coverage."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--quality-summary", type=Path, default=DEFAULT_QUALITY_SUMMARY)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--zju-dir", type=str, default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def format_float(value):
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def mean_or_none(values):
    if not values:
        return None
    return float(sum(values) / len(values))


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


def load_dataset_and_loss_cfg(config_path: Path, split: str, zju_dir_override: str, num_images: int):
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
    loss_cfg = OmegaConf.to_container(cfg.loss.depth, resolve=True)
    return dataset, resolved_zju_dir, loss_cfg


def summarize_threshold_group(rows, threshold):
    below_rows = [row for row in rows if float(row["quality_score"]) < threshold]
    above_rows = [row for row in rows if float(row["quality_score"]) >= threshold]

    def summarize(group_rows):
        return {
            "count": len(group_rows),
            "quality_score_mean": mean_or_none([float(row["quality_score"]) for row in group_rows]),
            "delta_conf_depth_mean": mean_or_none([float(row["delta_conf_depth_mean"]) for row in group_rows]),
            "delta_reg_depth_mean": mean_or_none([float(row["delta_reg_depth_mean"]) for row in group_rows]),
        }

    return {"below": summarize(below_rows), "above": summarize(above_rows)}


def main():
    args = parse_args()
    config_path = resolve_config_path(args.config)
    quality_payload = load_json(args.quality_summary)
    attribution_summary = Path(quality_payload["attribution_summary"])
    attribution_payload = load_json(attribution_summary)

    dataset, resolved_zju_dir, loss_cfg = load_dataset_and_loss_cfg(
        config_path,
        args.split,
        args.zju_dir,
        args.num_images,
    )

    target_cameras = ensure_list(loss_cfg.get("anchor_conditioned_conf_target_cameras"))
    quality_min = loss_cfg.get("anchor_conditioned_conf_target_quality_min")
    quality_max = loss_cfg.get("anchor_conditioned_conf_target_quality_max")
    bottom_ratio = float(loss_cfg.get("anchor_conditioned_conf_target_foreground_bottom_ratio", 0.0))
    scale_value = float(loss_cfg.get("anchor_conditioned_conf_target_scale", 1.0))

    anchor_rows = [
        row
        for row in attribution_payload.get("delta_rows", [])
        if row.get("view_role") == "anchor_supervised"
        and row.get("camera_name") is not None
        and row.get("quality_score") is not None
        and row.get("delta_conf_depth_mean") is not None
        and row.get("delta_reg_depth_mean") is not None
    ]

    top_anchor_camera = str(quality_payload["top_anchor_camera"])
    top_anchor_rows = [row for row in anchor_rows if str(row["camera_name"]) == top_anchor_camera]
    threshold_summary = None
    if quality_min is not None:
        threshold_summary = summarize_threshold_group(top_anchor_rows, float(quality_min))

    total_samples = 0
    anchor_match_samples = 0
    quality_match_samples = 0
    eligible_samples = 0
    total_conf_pixels = 0
    eligible_conf_pixels = 0
    eligible_bottom_pixels = 0
    b1_quality_scores = []
    b1_sample_count = 0
    b1_eligible_count = 0

    max_samples = min(int(args.num_samples), len(dataset))
    for sample_idx in range(max_samples):
        sample = dataset.get_data(seq_index=sample_idx, img_per_seq=args.num_images)
        total_samples += 1

        anchor_camera = sample.get("selection_anchor_camera")
        anchor_camera = None if anchor_camera is None else str(anchor_camera)
        quality_scores = sample.get("supervised_view_quality_scores", {})
        anchor_quality_score = None
        if anchor_camera is not None:
            raw_score = quality_scores.get(anchor_camera)
            if raw_score is not None:
                anchor_quality_score = float(raw_score)

        if anchor_camera == top_anchor_camera:
            b1_sample_count += 1
            if anchor_quality_score is not None:
                b1_quality_scores.append(anchor_quality_score)

        camera_match = anchor_camera in target_cameras
        quality_match = True
        if quality_min is not None:
            quality_match = anchor_quality_score is not None and anchor_quality_score >= float(quality_min)
        if quality_match and quality_max is not None:
            quality_match = anchor_quality_score is not None and anchor_quality_score <= float(quality_max)

        if camera_match:
            anchor_match_samples += 1
        if camera_match and quality_match:
            quality_match_samples += 1

        conf_mask = np.stack(sample["conf_depth_point_masks"]).astype(bool)
        total_conf_pixels += int(conf_mask.sum())

        if not (camera_match and quality_match):
            continue

        eligible_samples += 1
        if anchor_camera == top_anchor_camera:
            b1_eligible_count += 1

        foreground_masks = torch.from_numpy(np.stack(sample["foreground_masks"]).astype(bool)).unsqueeze(0)
        nonbottom_mask = build_reliable_foreground_region_mask(
            foreground_masks,
            erode_px=0,
            drop_bottom_ratio=bottom_ratio,
        )[0].cpu().numpy().astype(bool)
        bottom_mask = np.stack(sample["foreground_masks"]).astype(bool) & (~nonbottom_mask)
        eligible_bottom_pixels += int(bottom_mask.sum())
        eligible_conf_pixels += int((bottom_mask & conf_mask).sum())

    b1_quality_scores = sorted(b1_quality_scores)
    b1_quality_p50 = None
    if b1_quality_scores:
        b1_quality_p50 = float(np.median(np.asarray(b1_quality_scores, dtype=np.float32)))

    result = {
        "config": str(config_path.resolve()),
        "zju_dir": str(resolved_zju_dir),
        "split": args.split,
        "num_samples": total_samples,
        "num_images": args.num_images,
        "candidate_rule": {
            "target_cameras": target_cameras,
            "scale": scale_value,
            "quality_min": quality_min,
            "quality_max": quality_max,
            "foreground_bottom_ratio": bottom_ratio,
        },
        "audit_threshold_summary": threshold_summary,
        "train_slice_summary": {
            "anchor_match_samples": anchor_match_samples,
            "eligible_samples": eligible_samples,
            "eligible_sample_fraction_total": (eligible_samples / total_samples) if total_samples else None,
            "eligible_sample_fraction_anchor_match": (eligible_samples / anchor_match_samples) if anchor_match_samples else None,
            "total_conf_depth_pixels": total_conf_pixels,
            "eligible_conf_depth_pixels": eligible_conf_pixels,
            "eligible_conf_depth_pixel_fraction_total": (eligible_conf_pixels / total_conf_pixels) if total_conf_pixels else None,
            "eligible_bottom_pixels": eligible_bottom_pixels,
        },
        "top_anchor_train_slice": {
            "camera_name": top_anchor_camera,
            "sample_count": b1_sample_count,
            "eligible_count": b1_eligible_count,
            "eligible_fraction": (b1_eligible_count / b1_sample_count) if b1_sample_count else None,
            "quality_score_median": b1_quality_p50,
            "quality_score_min": (float(b1_quality_scores[0]) if b1_quality_scores else None),
            "quality_score_max": (float(b1_quality_scores[-1]) if b1_quality_scores else None),
        },
    }

    below = (threshold_summary or {}).get("below", {})
    above = (threshold_summary or {}).get("above", {})
    eligible_fraction_total = result["train_slice_summary"]["eligible_sample_fraction_total"]
    eligible_pixel_fraction = result["train_slice_summary"]["eligible_conf_depth_pixel_fraction_total"]
    recommendation_lines = []
    if quality_min is not None and below.get("count", 0) > 0:
        recommendation_lines.append(
            f"On the audited {top_anchor_camera} rows, {below['count']} / {len(top_anchor_rows)} bad-anchor rows sit below quality_min={float(quality_min):.1f}, "
            f"yet their mean delta_conf_depth is still {format_float(below.get('delta_conf_depth_mean'))} and mean delta_reg_depth is {format_float(below.get('delta_reg_depth_mean'))}."
        )
    if eligible_fraction_total is not None and eligible_pixel_fraction is not None:
        recommendation_lines.append(
            f"On the first {total_samples} {args.split} samples, the rule only activates on {eligible_samples} samples "
            f"({format_float(eligible_fraction_total * 100.0)}% of the slice) and {eligible_conf_pixels} conf-depth pixels "
            f"({format_float(eligible_pixel_fraction * 100.0)}% of conf-depth-supervised pixels)."
        )
    recommendation_lines.append(
        "That combination is consistent with an over-selective hard threshold: it narrows the already-bad anchor further, "
        "produces only a tiny conf_depth gain, and leaves reg_depth unchanged."
    )
    recommendation_lines.append(
        "Do not auto-open another sibling. If a fresh manual question is approved, it should justify a broader or continuous "
        "quality-conditioned rule rather than reusing the same hard q>=3.0 gate."
    )
    result["recommendation"] = {
        "status": "rejected_candidate_postmortem_complete",
        "next_question_shape": "manual_quality_conditioned_rule_redefinition_required",
        "reason": " ".join(recommendation_lines),
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", result)

    lines = [
        "# ZJU Quality-Conditioned Candidate Postmortem",
        "",
        f"- config: `{config_path}`",
        f"- quality_summary: `{args.quality_summary}`",
        f"- attribution_summary: `{attribution_summary}`",
        f"- split: `{args.split}`",
        f"- num_samples: `{total_samples}`",
        "",
        "## Candidate Rule",
        "",
        f"- target_cameras: `{target_cameras}`",
        f"- quality_min: `{quality_min}`",
        f"- quality_max: `{quality_max}`",
        f"- foreground_bottom_ratio: `{bottom_ratio}`",
        f"- scale: `{scale_value}`",
        "",
        f"## {top_anchor_camera} Audit Threshold Split",
        "",
        "| group | count | quality_score_mean | delta_conf_depth_mean | delta_reg_depth_mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    if threshold_summary is not None:
        for label in ("below", "above"):
            summary = threshold_summary[label]
            lines.append(
                f"| `{label}` | {summary['count']} | {format_float(summary['quality_score_mean'])} | "
                f"{format_float(summary['delta_conf_depth_mean'])} | {format_float(summary['delta_reg_depth_mean'])} |"
            )
    else:
        lines.append("| `n/a` | 0 | n/a | n/a | n/a |")

    lines.extend(
        [
            "",
            "## Train-Slice Coverage",
            "",
            "| metric | value |",
            "| --- | ---: |",
            f"| `anchor_match_samples` | {anchor_match_samples} |",
            f"| `eligible_samples` | {eligible_samples} |",
            f"| `eligible_sample_fraction_total_pct` | {format_float((eligible_fraction_total or 0.0) * 100.0)} |",
            f"| `eligible_sample_fraction_anchor_match_pct` | {format_float((result['train_slice_summary']['eligible_sample_fraction_anchor_match'] or 0.0) * 100.0)} |",
            f"| `eligible_conf_depth_pixel_fraction_total_pct` | {format_float((eligible_pixel_fraction or 0.0) * 100.0)} |",
            f"| `{top_anchor_camera}_sample_count` | {b1_sample_count} |",
            f"| `{top_anchor_camera}_eligible_count` | {b1_eligible_count} |",
            f"| `{top_anchor_camera}_eligible_fraction_pct` | {format_float((result['top_anchor_train_slice']['eligible_fraction'] or 0.0) * 100.0)} |",
            "",
            "## Recommendation",
            "",
            f"- status: `{result['recommendation']['status']}`",
            f"- next_question_shape: `{result['recommendation']['next_question_shape']}`",
            f"- reason: {result['recommendation']['reason']}",
            "",
        ]
    )
    write_text(output_dir / "summary.md", "\n".join(lines))
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
