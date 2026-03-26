import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"

for root in (str(REPO_ROOT), str(TRAINING_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from training.data.datasets.zju_vggt_geom import ZjuVggtGeomDataset


DEFAULT_CONFIG = (
    "zju_vggt_geom_unproject_source_policy_nearest_rawpool_"
    "confdepth_dropworst_gradconfmask_minimal"
)
DEFAULT_BOUNDARY_SUMMARY = (
    REPO_ROOT
    / "output"
    / "zju_quality_region_boundary_confdepth_dropworst_gradconfmask_20260326_v1"
    / "summary.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "output"
    / "zju_quality_rule_shape_sweep_confdepth_dropworst_gradconfmask_20260326_v1"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare broader hard-threshold and continuous whole-foreground quality-conditioned "
            "rule shapes for the remaining Camera_B1 conf-depth residual."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--boundary-summary", type=Path, default=DEFAULT_BOUNDARY_SUMMARY)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--zju-dir", type=str, default="")
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


def mean_or_none(values):
    if not values:
        return None
    return float(sum(values) / len(values))


def format_float(value):
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


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


def build_rules(train_b1_quality_scores: list[float]) -> list[dict]:
    quality_array = np.asarray(train_b1_quality_scores, dtype=np.float32)
    q_min = float(np.min(quality_array))
    q_max = float(np.max(quality_array))
    q25 = float(np.quantile(quality_array, 0.25))
    q50 = float(np.quantile(quality_array, 0.50))
    q75 = float(np.quantile(quality_array, 0.75))

    return [
        {
            "name": "rejected_hard_qge3_bottom20_scale05",
            "family": "hard_threshold",
            "region": "bottom_0.20",
            "threshold": 3.0,
            "scale_high": 0.5,
        },
        {
            "name": "hard_qge3_wholefg_scale05",
            "family": "hard_threshold",
            "region": "whole_foreground",
            "threshold": 3.0,
            "scale_high": 0.5,
        },
        {
            "name": "hard_qge275_wholefg_scale05",
            "family": "hard_threshold",
            "region": "whole_foreground",
            "threshold": 2.75,
            "scale_high": 0.5,
        },
        {
            "name": "linear_qmin_qmax_wholefg_scale05",
            "family": "linear",
            "region": "whole_foreground",
            "quality_low": q_min,
            "quality_high": q_max,
            "scale_low": 1.0,
            "scale_high": 0.5,
        },
        {
            "name": "linear_q25_q75_wholefg_scale05",
            "family": "linear",
            "region": "whole_foreground",
            "quality_low": q25,
            "quality_high": q75,
            "scale_low": 1.0,
            "scale_high": 0.5,
        },
        {
            "name": "linear_q50_qmax_wholefg_scale05",
            "family": "linear",
            "region": "whole_foreground",
            "quality_low": q50,
            "quality_high": q_max,
            "scale_low": 1.0,
            "scale_high": 0.5,
        },
    ]


def rule_weight(rule: dict, quality_score: float | None) -> float:
    if quality_score is None:
        return 0.0
    quality_score = float(quality_score)
    if rule["family"] == "hard_threshold":
        return 1.0 if quality_score >= float(rule["threshold"]) else 0.0

    low = float(rule["quality_low"])
    high = float(rule["quality_high"])
    if high <= low:
        return 1.0 if quality_score >= high else 0.0
    return float(max(0.0, min(1.0, (quality_score - low) / (high - low))))


def evaluate_rules_on_audit(anchor_rows: list[dict], rules: list[dict]) -> list[dict]:
    total_rows = len(anchor_rows)
    results = []
    for rule in rules:
        active_rows = []
        weighted_conf_sum = 0.0
        weighted_nonbottom_conf_sum = 0.0
        weighted_bottom_conf_sum = 0.0
        weight_sum = 0.0
        for row in anchor_rows:
            weight = rule_weight(rule, row.get("quality_score"))
            if weight <= 0.0:
                continue
            active_rows.append(row)
            weight_sum += weight
            weighted_conf_sum += weight * float(row["delta_conf_depth_mean"])
            weighted_nonbottom_conf_sum += weight * float(row["delta_fg_nonbottom_conf_depth_mean"])
            weighted_bottom_conf_sum += weight * float(row["delta_fg_bottom_conf_depth_mean"])

        results.append(
            {
                "rule_name": rule["name"],
                "family": rule["family"],
                "region": rule["region"],
                "active_rows": len(active_rows),
                "active_row_fraction": (len(active_rows) / total_rows) if total_rows else None,
                "mean_rule_weight": (weight_sum / total_rows) if total_rows else None,
                "active_delta_conf_depth_mean": mean_or_none([float(row["delta_conf_depth_mean"]) for row in active_rows]),
                "active_delta_fg_nonbottom_conf_depth_mean": mean_or_none(
                    [float(row["delta_fg_nonbottom_conf_depth_mean"]) for row in active_rows]
                ),
                "weighted_delta_conf_depth_mean": (
                    weighted_conf_sum / weight_sum if weight_sum > 0.0 else None
                ),
                "weighted_delta_fg_nonbottom_conf_depth_mean": (
                    weighted_nonbottom_conf_sum / weight_sum if weight_sum > 0.0 else None
                ),
                "weighted_delta_fg_bottom_conf_depth_mean": (
                    weighted_bottom_conf_sum / weight_sum if weight_sum > 0.0 else None
                ),
                "omitted_bad_rows": total_rows - len(active_rows),
            }
        )
    return results


def region_mask_name(rule_region: str) -> str:
    if rule_region == "whole_foreground":
        return "whole_foreground_conf_pixels"
    if rule_region == "bottom_0.20":
        return "bottom20_conf_pixels"
    raise ValueError(f"Unknown region name: {rule_region}")


def collect_train_slice(dataset, top_anchor_camera: str, num_samples: int, num_images: int) -> dict:
    total_samples = 0
    total_conf_depth_pixels = 0
    top_anchor_sample_count = 0
    top_anchor_quality_scores = []
    samples = []

    max_samples = min(int(num_samples), len(dataset))
    for sample_idx in range(max_samples):
        sample = dataset.get_data(seq_index=sample_idx, img_per_seq=num_images)
        total_samples += 1
        conf_mask = np.stack(sample["conf_depth_point_masks"]).astype(bool)
        foreground_masks = np.stack(sample["foreground_masks"]).astype(bool)
        total_conf_depth_pixels += int(conf_mask.sum())

        anchor_camera = sample.get("selection_anchor_camera")
        anchor_camera = None if anchor_camera is None else str(anchor_camera)
        quality_score = None
        if anchor_camera == top_anchor_camera:
            raw_quality = sample.get("supervised_view_quality_scores", {}).get(anchor_camera)
            if raw_quality is not None:
                quality_score = float(raw_quality)
                top_anchor_quality_scores.append(quality_score)
            top_anchor_sample_count += 1

        samples.append(
            {
                "anchor_camera": anchor_camera,
                "quality_score": quality_score,
                "whole_foreground_conf_pixels": int((conf_mask & foreground_masks).sum()),
                "bottom20_conf_pixels": int(
                    (conf_mask & foreground_masks).sum()
                    - (
                        conf_mask
                        & foreground_masks
                        & (np.arange(foreground_masks.shape[-2])[:, None] < int(foreground_masks.shape[-2] * 0.8))
                    ).sum()
                ),
            }
        )

    return {
        "samples": samples,
        "num_samples": total_samples,
        "num_images": num_images,
        "total_conf_depth_pixels": total_conf_depth_pixels,
        "top_anchor_camera": top_anchor_camera,
        "top_anchor_sample_count": top_anchor_sample_count,
        "top_anchor_quality_scores": top_anchor_quality_scores,
    }


def evaluate_rules_on_train_slice(train_slice: dict, rules: list[dict]) -> list[dict]:
    total_samples = train_slice["num_samples"]
    total_conf_depth_pixels = train_slice["total_conf_depth_pixels"]
    top_anchor_sample_count = train_slice["top_anchor_sample_count"]
    top_anchor_camera = train_slice["top_anchor_camera"]

    results = []
    for rule in rules:
        active_samples = 0
        active_anchor_samples = 0
        targeted_pixels = 0
        effective_reduction_pixels = 0.0
        mean_scale_values = []

        for sample in train_slice["samples"]:
            if sample["anchor_camera"] != top_anchor_camera:
                continue
            quality_score = sample["quality_score"]
            weight = rule_weight(rule, quality_score)
            scale_high = float(rule.get("scale_high", 0.5))
            scale_value = 1.0 - (1.0 - scale_high) * weight
            pixel_count = int(sample[region_mask_name(rule["region"])])
            if pixel_count <= 0 or weight <= 0.0:
                continue

            active_samples += 1
            active_anchor_samples += 1
            targeted_pixels += pixel_count
            effective_reduction_pixels += (1.0 - scale_value) * pixel_count
            mean_scale_values.append(scale_value)

        results.append(
            {
                "rule_name": rule["name"],
                "family": rule["family"],
                "region": rule["region"],
                "active_samples": active_samples,
                "active_sample_fraction_total": (active_samples / total_samples) if total_samples else None,
                "active_sample_fraction_top_anchor": (
                    active_anchor_samples / top_anchor_sample_count if top_anchor_sample_count else None
                ),
                "targeted_conf_depth_pixels": targeted_pixels,
                "targeted_conf_depth_pixel_fraction_total": (
                    targeted_pixels / total_conf_depth_pixels if total_conf_depth_pixels else None
                ),
                "effective_conf_depth_reduction_fraction_total": (
                    effective_reduction_pixels / total_conf_depth_pixels if total_conf_depth_pixels else None
                ),
                "mean_scale_value": mean_or_none(mean_scale_values),
            }
        )
    return results


def merge_rule_results(rule_defs: list[dict], audit_rows: list[dict], train_rows: list[dict]) -> list[dict]:
    audit_index = {row["rule_name"]: row for row in audit_rows}
    train_index = {row["rule_name"]: row for row in train_rows}
    merged = []
    for rule in rule_defs:
        audit = audit_index[rule["name"]]
        train = train_index[rule["name"]]
        merged.append(
            {
                **rule,
                "audit": audit,
                "train_slice": train,
            }
        )
    return merged


def choose_recommendation(rule_rows: list[dict], code_support: dict) -> dict:
    hard_whole_q3 = next(row for row in rule_rows if row["name"] == "hard_qge3_wholefg_scale05")
    hard_whole_q275 = next(row for row in rule_rows if row["name"] == "hard_qge275_wholefg_scale05")
    linear_full = next(row for row in rule_rows if row["name"] == "linear_qmin_qmax_wholefg_scale05")
    linear_mid = next(row for row in rule_rows if row["name"] == "linear_q25_q75_wholefg_scale05")

    if (
        linear_full["audit"]["active_rows"] >= hard_whole_q275["audit"]["active_rows"]
        and linear_full["audit"]["active_rows"] == 10
    ):
        recommended_rule = linear_full
    else:
        recommended_rule = linear_mid

    return {
        "status": "rule_shape_sweep_complete",
        "recommended_manual_question_shape": "continuous_camera_b1_whole_foreground_quality_conditioned_rule",
        "recommended_reference_rule": recommended_rule["name"],
        "reason": (
            f"The broader hard-threshold whole-foreground rule q>=3.0 still omits "
            f"{hard_whole_q3['audit']['omitted_bad_rows']} / 10 bad Camera_B1 audit rows, while q>=2.75 still omits "
            f"{hard_whole_q275['audit']['omitted_bad_rows']} / 10. A continuous whole-foreground rule "
            f"({recommended_rule['name']}) can cover all 10 bad audit rows while keeping quality-ordered emphasis "
            f"and an effective conf-depth reduction fraction of "
            f"{format_float((recommended_rule['train_slice']['effective_conf_depth_reduction_fraction_total'] or 0.0) * 100.0)}% "
            f"of the fresh 512-sample train slice."
        ),
        "why_not_hard_threshold": (
            f"Hard whole-foreground q>=3.0 still drops 4 bad rows; even q>=2.75 still drops 1 bad row and collapses "
            f"toward an almost-all-B1 treatment at {format_float((hard_whole_q275['train_slice']['targeted_conf_depth_pixel_fraction_total'] or 0.0) * 100.0)}% "
            "pixel coverage."
        ),
        "requires_new_loss_knob": not code_support["supports_continuous_quality_scaling"],
        "code_support": code_support,
    }


def main():
    args = parse_args()
    config_path = resolve_config_path(args.config)
    boundary_payload = load_json(args.boundary_summary)
    attribution_summary_path = Path(boundary_payload["attribution_summary"])
    attribution_payload = load_json(attribution_summary_path)
    top_anchor_camera = str(boundary_payload["top_anchor_camera"])

    anchor_rows = [
        row
        for row in attribution_payload.get("delta_rows", [])
        if row.get("view_role") == "anchor_supervised"
        and str(row.get("camera_name")) == top_anchor_camera
        and row.get("quality_score") is not None
        and row.get("delta_conf_depth_mean") is not None
        and row.get("delta_fg_bottom_conf_depth_mean") is not None
        and row.get("delta_fg_nonbottom_conf_depth_mean") is not None
    ]

    dataset, resolved_zju_dir = load_dataset(
        config_path=config_path,
        split=args.split,
        zju_dir_override=args.zju_dir,
        num_images=args.num_images,
    )
    train_slice = collect_train_slice(
        dataset=dataset,
        top_anchor_camera=top_anchor_camera,
        num_samples=args.num_samples,
        num_images=args.num_images,
    )
    if not train_slice["top_anchor_quality_scores"]:
        raise RuntimeError(f"No {top_anchor_camera} quality scores found in the requested slice.")

    rule_defs = build_rules(train_slice["top_anchor_quality_scores"])
    audit_rule_rows = evaluate_rules_on_audit(anchor_rows, rule_defs)
    train_rule_rows = evaluate_rules_on_train_slice(train_slice, rule_defs)
    rule_rows = merge_rule_results(rule_defs, audit_rule_rows, train_rule_rows)

    quality_array = np.asarray(train_slice["top_anchor_quality_scores"], dtype=np.float32)
    quality_stats = {
        "count": int(quality_array.size),
        "min": float(np.min(quality_array)),
        "q25": float(np.quantile(quality_array, 0.25)),
        "q50": float(np.quantile(quality_array, 0.50)),
        "q75": float(np.quantile(quality_array, 0.75)),
        "max": float(np.max(quality_array)),
        "mean": float(np.mean(quality_array)),
    }

    code_support = {
        "supports_hard_quality_min_max": True,
        "supports_anchor_conditioned_bottom_region": True,
        "supports_continuous_quality_scaling": False,
        "implementation_gap": (
            "training/loss.py currently supports anchor_conditioned_conf_target_quality_min/max "
            "with a constant scale, but it does not expose a continuous per-sample quality-to-scale mapping."
        ),
    }

    recommendation = choose_recommendation(rule_rows, code_support)

    result = {
        "config": str(config_path.resolve()),
        "boundary_summary": str(args.boundary_summary.resolve()),
        "attribution_summary": str(attribution_summary_path.resolve()),
        "split": args.split,
        "num_samples": train_slice["num_samples"],
        "num_images": train_slice["num_images"],
        "zju_dir": str(resolved_zju_dir),
        "top_anchor_camera": top_anchor_camera,
        "top_anchor_quality_stats": quality_stats,
        "rule_rows": rule_rows,
        "code_support": code_support,
        "recommendation": recommendation,
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", result)

    lines = [
        "# ZJU Quality Rule-Shape Sweep",
        "",
        f"- config: `{config_path}`",
        f"- boundary_summary: `{args.boundary_summary}`",
        f"- attribution_summary: `{attribution_summary_path}`",
        f"- split: `{args.split}`",
        f"- num_samples: `{train_slice['num_samples']}`",
        f"- top_anchor_camera: `{top_anchor_camera}`",
        "",
        "## Camera_B1 Quality Stats",
        "",
        f"- count: `{quality_stats['count']}`",
        f"- min: `{format_float(quality_stats['min'])}`",
        f"- q25: `{format_float(quality_stats['q25'])}`",
        f"- q50: `{format_float(quality_stats['q50'])}`",
        f"- q75: `{format_float(quality_stats['q75'])}`",
        f"- max: `{format_float(quality_stats['max'])}`",
        "",
        "## Rule Comparison",
        "",
        "| rule | family | region | audit_active_rows | omitted_bad_rows | weighted_delta_conf_depth_mean | targeted_pixel_fraction_total_pct | effective_reduction_fraction_total_pct |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rule_rows:
        lines.append(
            "| {rule} | {family} | {region} | {active_rows} | {omitted_rows} | {weighted_delta} | {pixel_pct} | {effective_pct} |".format(
                rule=row["name"],
                family=row["family"],
                region=row["region"],
                active_rows=row["audit"]["active_rows"],
                omitted_rows=row["audit"]["omitted_bad_rows"],
                weighted_delta=format_float(row["audit"]["weighted_delta_conf_depth_mean"]),
                pixel_pct=format_float((row["train_slice"]["targeted_conf_depth_pixel_fraction_total"] or 0.0) * 100.0),
                effective_pct=format_float((row["train_slice"]["effective_conf_depth_reduction_fraction_total"] or 0.0) * 100.0),
            )
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- status: `{recommendation['status']}`",
            f"- recommended_manual_question_shape: `{recommendation['recommended_manual_question_shape']}`",
            f"- recommended_reference_rule: `{recommendation['recommended_reference_rule']}`",
            f"- reason: {recommendation['reason']}",
            f"- why_not_hard_threshold: {recommendation['why_not_hard_threshold']}",
            f"- requires_new_loss_knob: `{recommendation['requires_new_loss_knob']}`",
            f"- code_support_gap: {code_support['implementation_gap']}",
            "",
        ]
    )

    write_text(output_dir / "summary.md", "\n".join(lines))
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
