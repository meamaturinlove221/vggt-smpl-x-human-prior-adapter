import json
from pathlib import Path


def rows_by_variant(summary: dict, variant: str) -> dict[str, dict]:
    return {str(row["case_id"]): row for row in summary.get("rows", []) if str(row.get("variant")) == variant}


def compare_variant(summary: dict, variant: str, case_ids: list[str]) -> dict:
    baseline = rows_by_variant(summary, "baseline_depth_unproject")
    candidate = rows_by_variant(summary, variant)
    rows = []
    for case_id in case_ids:
        base_row = baseline[case_id]
        cand_row = candidate[case_id]
        base_support = base_row["support_metrics"]
        cand_support = cand_row["support_metrics"]
        row = {
            "case_id": case_id,
            "delta_fg_connected_components": cand_support["fg_connected_components"] - base_support["fg_connected_components"],
            "delta_fg_peak_count": cand_support["fg_peak_count"] - base_support["fg_peak_count"],
            "delta_masked_l1": cand_row["metrics"]["fg_masked"]["l1"] - base_row["metrics"]["fg_masked"]["l1"],
            "delta_masked_ssim": cand_row["metrics"]["fg_masked"]["ssim"] - base_row["metrics"]["fg_masked"]["ssim"],
            "delta_off_body_support_ratio": cand_support["off_body_support_ratio"] - base_support["off_body_support_ratio"],
            "delta_off_body_nonblack_ratio": cand_support["off_body_nonblack_ratio"] - base_support["off_body_nonblack_ratio"],
            "delta_bg_bottom_support_ratio": cand_support["bg_bottom_support_ratio"] - base_support["bg_bottom_support_ratio"],
            "delta_source_entropy_inside_fg": cand_support["source_entropy_inside_fg"] - base_support["source_entropy_inside_fg"],
            "delta_source_top1_mass_ratio_inside_fg": cand_support["source_top1_mass_ratio_inside_fg"] - base_support["source_top1_mass_ratio_inside_fg"],
            "delta_source_top1_top2_margin_inside_fg": cand_support["source_top1_top2_margin_inside_fg"] - base_support["source_top1_top2_margin_inside_fg"],
            "delta_source_label_smoothness_inside_fg": cand_support["source_label_smoothness_inside_fg"] - base_support["source_label_smoothness_inside_fg"],
            "delta_source_id_switch_count_inside_fg": float(cand_support.get("source_id_switch_count_inside_fg", 0.0)) - float(base_support.get("source_id_switch_count_inside_fg", 0.0)),
            "delta_source_top1_spatial_fragmentation": float(cand_support.get("source_top1_spatial_fragmentation", 0.0)) - float(base_support.get("source_top1_spatial_fragmentation", 0.0)),
            "delta_correspondence_consensus_ratio_inside_fg": cand_support["correspondence_consensus_ratio_inside_fg"] - base_support["correspondence_consensus_ratio_inside_fg"],
            "delta_support_inside_fg_ratio": cand_support["support_inside_fg_ratio"] - base_support["support_inside_fg_ratio"],
            "delta_fg_bbox_cover_ratio": cand_support["fg_bbox_cover_ratio"] - base_support["fg_bbox_cover_ratio"],
            "delta_fg_visible_rgb_coverage_ratio": float(cand_support.get("fg_visible_rgb_coverage_ratio", cand_support.get("fg_rehydrated_coverage_ratio", 0.0))) - float(base_support.get("fg_visible_rgb_coverage_ratio", base_support.get("fg_rehydrated_coverage_ratio", 0.0))),
            "delta_fg_support_visible_overlap_ratio": float(cand_support.get("fg_support_visible_overlap_ratio", 0.0)) - float(base_support.get("fg_support_visible_overlap_ratio", 0.0)),
            "delta_fg_bbox_visible_cover_ratio": float(cand_support.get("fg_bbox_visible_cover_ratio", cand_support.get("fg_bbox_cover_ratio", 0.0))) - float(base_support.get("fg_bbox_visible_cover_ratio", base_support.get("fg_bbox_cover_ratio", 0.0))),
            "delta_largest_fg_visible_component_ratio": float(cand_support.get("largest_fg_visible_component_ratio", cand_support.get("largest_fg_component_cover_ratio", 0.0))) - float(base_support.get("largest_fg_visible_component_ratio", base_support.get("largest_fg_component_cover_ratio", 0.0))),
            "delta_fg_visible_mass_ratio": float(cand_support.get("fg_visible_mass_ratio", 0.0)) - float(base_support.get("fg_visible_mass_ratio", 0.0)),
            "fg_mask_coverage_ratio": float(cand_support.get("fg_mask_coverage_ratio", 0.0)),
            "fg_alpha_coverage_ratio": float(cand_support.get("fg_alpha_coverage_ratio", 0.0)),
            "fg_rehydrated_coverage_ratio": float(cand_support.get("fg_rehydrated_coverage_ratio", cand_support.get("fg_mask_coverage_ratio", 0.0))),
            "fg_visible_rgb_coverage_ratio": float(cand_support.get("fg_visible_rgb_coverage_ratio", cand_support.get("fg_rehydrated_coverage_ratio", cand_support.get("fg_mask_coverage_ratio", 0.0)))),
            "fg_support_visible_overlap_ratio": float(cand_support.get("fg_support_visible_overlap_ratio", 0.0)),
            "fg_bbox_visible_cover_ratio": float(cand_support.get("fg_bbox_visible_cover_ratio", cand_support.get("fg_bbox_cover_ratio", 0.0))),
            "largest_fg_visible_component_ratio": float(cand_support.get("largest_fg_visible_component_ratio", cand_support.get("largest_fg_component_cover_ratio", 0.0))),
            "fg_visible_mass_ratio": float(cand_support.get("fg_visible_mass_ratio", 0.0)),
            "fg_visible_coverage_retained_ratio": float(cand_support.get("fg_visible_coverage_retained_ratio", 1.0)),
            "fg_visible_bbox_retained_ratio": float(cand_support.get("fg_visible_bbox_retained_ratio", 1.0)),
            "largest_fg_visible_component_retained_ratio": float(cand_support.get("largest_fg_visible_component_retained_ratio", 1.0)),
            "fg_visible_mass_retained_ratio": float(cand_support.get("fg_visible_mass_retained_ratio", 1.0)),
            "fg_support_mass_coverage_ratio": float(cand_support.get("fg_support_mass_coverage_ratio", cand_support.get("fg_retained_mass_ratio", 0.0))),
            "fg_retained_area_ratio": float(cand_support.get("fg_retained_area_ratio", 1.0)),
            "fg_retained_mass_ratio": float(cand_support.get("fg_retained_mass_ratio", 1.0)),
            "fg_retained_support_area_ratio": float(cand_support.get("fg_retained_support_area_ratio", 1.0)),
            "largest_fg_component_cover_ratio": float(cand_support.get("largest_fg_component_cover_ratio", cand_support.get("fg_largest_component_ratio", 0.0))),
            "human_erasure_penalty": float(cand_support.get("human_erasure_penalty", 0.0)),
            "source_id_switch_count_inside_fg": float(cand_support.get("source_id_switch_count_inside_fg", 0.0)),
            "source_top1_spatial_fragmentation": float(cand_support.get("source_top1_spatial_fragmentation", 0.0)),
            "metric_truth_bug": bool(cand_support.get("metric_truth_bug", False)),
        }
        row["smoke_accept"] = smoke_accept(row)
        rows.append(row)

    count = max(len(rows), 1)
    return {
        "variant": variant,
        "case_count": len(rows),
        "improved_all_primary_count": int(sum(1 for row in rows if row["smoke_accept"])),
        "mean_delta_fg_connected_components": float(sum(row["delta_fg_connected_components"] for row in rows) / count),
        "mean_delta_fg_peak_count": float(sum(row["delta_fg_peak_count"] for row in rows) / count),
        "mean_delta_masked_l1": float(sum(row["delta_masked_l1"] for row in rows) / count),
        "mean_delta_masked_ssim": float(sum(row["delta_masked_ssim"] for row in rows) / count),
        "mean_delta_off_body_support_ratio": float(sum(row["delta_off_body_support_ratio"] for row in rows) / count),
        "mean_delta_off_body_nonblack_ratio": float(sum(row["delta_off_body_nonblack_ratio"] for row in rows) / count),
        "mean_delta_bg_bottom_support_ratio": float(sum(row["delta_bg_bottom_support_ratio"] for row in rows) / count),
        "mean_delta_source_entropy_inside_fg": float(sum(row["delta_source_entropy_inside_fg"] for row in rows) / count),
        "mean_delta_source_top1_mass_ratio_inside_fg": float(sum(row["delta_source_top1_mass_ratio_inside_fg"] for row in rows) / count),
        "mean_delta_source_top1_top2_margin_inside_fg": float(sum(row["delta_source_top1_top2_margin_inside_fg"] for row in rows) / count),
        "mean_delta_source_label_smoothness_inside_fg": float(sum(row["delta_source_label_smoothness_inside_fg"] for row in rows) / count),
        "mean_delta_source_id_switch_count_inside_fg": float(sum(row["delta_source_id_switch_count_inside_fg"] for row in rows) / count),
        "mean_delta_source_top1_spatial_fragmentation": float(sum(row["delta_source_top1_spatial_fragmentation"] for row in rows) / count),
        "mean_delta_correspondence_consensus_ratio_inside_fg": float(sum(row["delta_correspondence_consensus_ratio_inside_fg"] for row in rows) / count),
        "mean_delta_support_inside_fg_ratio": float(sum(row["delta_support_inside_fg_ratio"] for row in rows) / count),
        "mean_delta_fg_bbox_cover_ratio": float(sum(row["delta_fg_bbox_cover_ratio"] for row in rows) / count),
        "mean_delta_fg_visible_rgb_coverage_ratio": float(sum(row["delta_fg_visible_rgb_coverage_ratio"] for row in rows) / count),
        "mean_delta_fg_support_visible_overlap_ratio": float(sum(row["delta_fg_support_visible_overlap_ratio"] for row in rows) / count),
        "mean_delta_fg_bbox_visible_cover_ratio": float(sum(row["delta_fg_bbox_visible_cover_ratio"] for row in rows) / count),
        "mean_delta_largest_fg_visible_component_ratio": float(sum(row["delta_largest_fg_visible_component_ratio"] for row in rows) / count),
        "mean_delta_fg_visible_mass_ratio": float(sum(row["delta_fg_visible_mass_ratio"] for row in rows) / count),
        "mean_fg_mask_coverage_ratio": float(sum(row["fg_mask_coverage_ratio"] for row in rows) / count),
        "mean_fg_alpha_coverage_ratio": float(sum(row["fg_alpha_coverage_ratio"] for row in rows) / count),
        "mean_fg_rehydrated_coverage_ratio": float(sum(row["fg_rehydrated_coverage_ratio"] for row in rows) / count),
        "mean_fg_visible_rgb_coverage_ratio": float(sum(row["fg_visible_rgb_coverage_ratio"] for row in rows) / count),
        "mean_fg_support_visible_overlap_ratio": float(sum(row["fg_support_visible_overlap_ratio"] for row in rows) / count),
        "mean_fg_bbox_visible_cover_ratio": float(sum(row["fg_bbox_visible_cover_ratio"] for row in rows) / count),
        "mean_largest_fg_visible_component_ratio": float(sum(row["largest_fg_visible_component_ratio"] for row in rows) / count),
        "mean_fg_visible_mass_ratio": float(sum(row["fg_visible_mass_ratio"] for row in rows) / count),
        "mean_fg_visible_coverage_retained_ratio": float(sum(row["fg_visible_coverage_retained_ratio"] for row in rows) / count),
        "mean_fg_visible_bbox_retained_ratio": float(sum(row["fg_visible_bbox_retained_ratio"] for row in rows) / count),
        "mean_largest_fg_visible_component_retained_ratio": float(sum(row["largest_fg_visible_component_retained_ratio"] for row in rows) / count),
        "mean_fg_visible_mass_retained_ratio": float(sum(row["fg_visible_mass_retained_ratio"] for row in rows) / count),
        "mean_fg_support_mass_coverage_ratio": float(sum(row["fg_support_mass_coverage_ratio"] for row in rows) / count),
        "mean_fg_retained_area_ratio": float(sum(row["fg_retained_area_ratio"] for row in rows) / count),
        "mean_fg_retained_mass_ratio": float(sum(row["fg_retained_mass_ratio"] for row in rows) / count),
        "mean_fg_retained_support_area_ratio": float(sum(row["fg_retained_support_area_ratio"] for row in rows) / count),
        "mean_largest_fg_component_cover_ratio": float(sum(row["largest_fg_component_cover_ratio"] for row in rows) / count),
        "mean_human_erasure_penalty": float(sum(row["human_erasure_penalty"] for row in rows) / count),
        "mean_source_id_switch_count_inside_fg": float(sum(row["source_id_switch_count_inside_fg"] for row in rows) / count),
        "mean_source_top1_spatial_fragmentation": float(sum(row["source_top1_spatial_fragmentation"] for row in rows) / count),
        "metric_truth_bug": bool(any(row["metric_truth_bug"] for row in rows)),
        "rows": rows,
    }


def smoke_accept(row: dict) -> bool:
    return (
        (not row["metric_truth_bug"])
        and
        row["delta_fg_connected_components"] <= -20
        and row["delta_fg_peak_count"] <= -2
        and row["delta_masked_l1"] <= -0.005
        and row["delta_masked_ssim"] >= 0.002
        and row["delta_off_body_support_ratio"] <= 0.0
        and row["delta_bg_bottom_support_ratio"] <= 0.0
        and row["fg_visible_coverage_retained_ratio"] >= 0.98
        and row["fg_visible_bbox_retained_ratio"] >= 0.95
        and row["fg_visible_mass_retained_ratio"] >= 0.95
        and row["largest_fg_visible_component_retained_ratio"] >= 0.95
        and row["largest_fg_visible_component_ratio"] >= 0.55
        and row["human_erasure_penalty"] <= 0.05
    )


def hero_accept(compare: dict) -> bool:
    return (
        (not compare["metric_truth_bug"])
        and
        compare["improved_all_primary_count"] >= 3
        and compare["mean_delta_fg_connected_components"] < 0.0
        and compare["mean_delta_fg_peak_count"] < 0.0
        and compare["mean_delta_masked_l1"] <= 0.0
        and compare["mean_delta_masked_ssim"] >= 0.0
        and compare["mean_fg_visible_coverage_retained_ratio"] >= 0.99
        and compare["mean_fg_visible_bbox_retained_ratio"] >= 0.96
        and compare["mean_fg_visible_mass_retained_ratio"] >= 0.96
        and compare["mean_largest_fg_visible_component_retained_ratio"] >= 0.96
        and compare["mean_largest_fg_visible_component_ratio"] >= 0.55
        and compare["mean_human_erasure_penalty"] <= 0.04
        and compare["mean_delta_off_body_support_ratio"] <= 0.0
        and compare["mean_delta_bg_bottom_support_ratio"] <= 0.0
    )


def local20_accept(compare: dict) -> bool:
    return (
        (not compare["metric_truth_bug"])
        and
        compare["improved_all_primary_count"] >= 14
        and compare["mean_delta_fg_connected_components"] < 0.0
        and compare["mean_delta_fg_peak_count"] < 0.0
        and compare["mean_delta_masked_l1"] <= 0.0
        and compare["mean_delta_masked_ssim"] >= 0.0
        and compare["mean_fg_visible_coverage_retained_ratio"] >= 0.99
        and compare["mean_fg_visible_bbox_retained_ratio"] >= 0.97
        and compare["mean_fg_visible_mass_retained_ratio"] >= 0.97
        and compare["mean_largest_fg_visible_component_retained_ratio"] >= 0.97
        and compare["mean_largest_fg_visible_component_ratio"] >= 0.60
        and compare["mean_human_erasure_penalty"] <= 0.03
        and compare["mean_delta_off_body_support_ratio"] <= 0.0
        and compare["mean_delta_bg_bottom_support_ratio"] <= 0.0
    )


def classify_failure(compare: dict) -> str:
    if compare.get("metric_truth_bug"):
        return "metric_truth_bug"
    if (
        compare["mean_human_erasure_penalty"] > 0.05
        or compare["mean_fg_visible_coverage_retained_ratio"] < 0.98
        or compare["mean_fg_visible_bbox_retained_ratio"] < 0.95
        or compare["mean_fg_visible_mass_retained_ratio"] < 0.95
        or compare["mean_largest_fg_visible_component_retained_ratio"] < 0.95
    ):
        return "erasure_win"
    no_move = (
        abs(compare["mean_delta_fg_connected_components"]) < 1.0
        and abs(compare["mean_delta_fg_peak_count"]) < 1.0
        and abs(compare["mean_delta_masked_l1"]) < 0.002
        and abs(compare["mean_delta_masked_ssim"]) < 0.002
        and abs(compare["mean_delta_off_body_support_ratio"]) < 0.01
    )
    if no_move:
        return "no_movement"
    if (
        compare["mean_delta_off_body_support_ratio"] <= 0.0
        and compare["mean_delta_bg_bottom_support_ratio"] <= 0.0
        and (compare["mean_delta_fg_connected_components"] >= 0.0 or compare["mean_delta_fg_peak_count"] >= 0.0)
    ):
        return "background_only_win"
    if compare["mean_delta_fg_connected_components"] >= 0.0 or compare["mean_delta_fg_peak_count"] >= 0.0:
        return "fragmentation_win"
    return "no_movement"


def progress_key(compare: dict) -> tuple:
    return (
        0 if not compare.get("metric_truth_bug", False) else 1,
        compare["mean_human_erasure_penalty"],
        -compare["mean_fg_visible_coverage_retained_ratio"],
        -compare["mean_fg_visible_bbox_retained_ratio"],
        -compare["mean_fg_visible_mass_retained_ratio"],
        -compare["mean_largest_fg_visible_component_retained_ratio"],
        -compare["mean_largest_fg_visible_component_ratio"],
        compare["mean_delta_fg_connected_components"],
        compare["mean_delta_fg_peak_count"],
        compare["mean_delta_masked_l1"],
        -compare["mean_delta_masked_ssim"],
        compare["mean_delta_off_body_support_ratio"],
        compare["mean_delta_bg_bottom_support_ratio"],
    )


def effective_progress(compare: dict, best_compare: dict | None) -> bool:
    if best_compare is None:
        return compare["improved_all_primary_count"] > 0
    if compare.get("metric_truth_bug", False):
        return False
    better_count = 0
    if compare["mean_delta_fg_connected_components"] < best_compare["mean_delta_fg_connected_components"]:
        better_count += 1
    if compare["mean_delta_fg_peak_count"] < best_compare["mean_delta_fg_peak_count"]:
        better_count += 1
    if compare["mean_delta_masked_l1"] < best_compare["mean_delta_masked_l1"]:
        better_count += 1
    if compare["mean_delta_masked_ssim"] > best_compare["mean_delta_masked_ssim"]:
        better_count += 1
    if compare["mean_delta_off_body_support_ratio"] < best_compare["mean_delta_off_body_support_ratio"]:
        better_count += 1
    if compare["mean_delta_bg_bottom_support_ratio"] < best_compare["mean_delta_bg_bottom_support_ratio"]:
        better_count += 1
    if compare["mean_fg_visible_coverage_retained_ratio"] > best_compare.get("mean_fg_visible_coverage_retained_ratio", 0.0):
        better_count += 1
    if compare["mean_fg_visible_bbox_retained_ratio"] > best_compare.get("mean_fg_visible_bbox_retained_ratio", 0.0):
        better_count += 1
    if compare["mean_fg_visible_mass_retained_ratio"] > best_compare.get("mean_fg_visible_mass_retained_ratio", 0.0):
        better_count += 1
    if compare["mean_largest_fg_visible_component_ratio"] > best_compare.get("mean_largest_fg_visible_component_ratio", 0.0):
        better_count += 1
    if compare["mean_largest_fg_visible_component_retained_ratio"] > best_compare.get("mean_largest_fg_visible_component_retained_ratio", 0.0):
        better_count += 1
    if compare["mean_human_erasure_penalty"] < best_compare["mean_human_erasure_penalty"]:
        better_count += 1
    return (
        better_count >= 4
        and compare["mean_fg_visible_coverage_retained_ratio"] >= max(0.95, best_compare.get("mean_fg_visible_coverage_retained_ratio", 0.95) - 0.02)
        and compare["mean_fg_visible_mass_retained_ratio"] >= max(0.92, best_compare.get("mean_fg_visible_mass_retained_ratio", 0.92) - 0.02)
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Score correspondence/source-selection progress from a summary.json.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--case-ids", required=True, help="Comma-separated case ids")
    args = parser.parse_args()
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    compare = compare_variant(summary, args.variant, [item.strip() for item in args.case_ids.split(",") if item.strip()])
    compare["failure_class"] = classify_failure(compare)
    compare["hero_accept"] = hero_accept(compare)
    compare["local20_accept"] = local20_accept(compare)
    print(json.dumps(compare, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
