import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate geometry diagnostics from multiple ZJU geometry sweep roots."
    )
    parser.add_argument(
        "--sweep_roots",
        nargs="+",
        required=True,
        help="Sweep root directories that each contain sweep_manifest.json and summary.json.",
    )
    parser.add_argument(
        "--local_zju_root",
        type=str,
        required=True,
        help="Local ZJU root used to derive ring-order diagnostics.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for markdown/json/csv outputs.",
    )
    return parser.parse_args()


def load_ring_order(seq_dir):
    annots = np.load(Path(seq_dir) / "annots.npy", allow_pickle=True).item()
    rotations = annots["cams"]["R"]
    translations = annots["cams"]["T"]
    rows = []
    for index, (rotation, translation) in enumerate(zip(rotations, translations), start=1):
        rotation = np.asarray(rotation, dtype=np.float64)
        translation = np.asarray(translation, dtype=np.float64).reshape(3, 1)
        center = (-rotation.T @ translation).reshape(3)
        azimuth = float(np.degrees(np.arctan2(center[0], center[2])))
        rows.append((f"Camera_B{index}", azimuth))
    rows.sort(key=lambda item: item[1])
    return [name for name, _ in rows]


def circular_shortest_distance(index_a, index_b, ring_length):
    raw = abs(index_a - index_b)
    return min(raw, ring_length - raw)


def source_geometry_features(source_cameras, target_camera, ring_order):
    ring_length = len(ring_order)
    camera_to_index = {camera: idx for idx, camera in enumerate(ring_order)}
    source_indices = sorted(camera_to_index[camera] for camera in source_cameras)
    target_idx = camera_to_index[target_camera]
    circular_gaps = []
    for pos, index in enumerate(source_indices):
        nxt = source_indices[(pos + 1) % len(source_indices)]
        gap = (nxt - index) % ring_length
        if gap == 0:
            gap = ring_length
        circular_gaps.append(gap)
    shortest_target_distances = [
        circular_shortest_distance(camera_to_index[camera], target_idx, ring_length)
        for camera in source_cameras
    ]
    signed_offsets = []
    for camera in source_cameras:
        src_idx = camera_to_index[camera]
        delta = (src_idx - target_idx) % ring_length
        if delta > ring_length / 2:
            delta -= ring_length
        signed_offsets.append(delta)
    left_count = sum(1 for delta in signed_offsets if delta < 0)
    right_count = sum(1 for delta in signed_offsets if delta > 0)
    largest_gap = max(circular_gaps)
    covered_arc = ring_length - largest_gap
    return {
        "ring_length": ring_length,
        "source_gap_mean": float(np.mean(circular_gaps)),
        "source_gap_std": float(np.std(circular_gaps)),
        "source_gap_min": int(min(circular_gaps)),
        "source_gap_max": int(largest_gap),
        "source_coverage_arc": int(covered_arc),
        "source_coverage_ratio": float(covered_arc / max(1, ring_length - 1)),
        "target_ring_dist_mean": float(np.mean(shortest_target_distances)),
        "target_ring_dist_min": int(min(shortest_target_distances)),
        "target_ring_dist_max": int(max(shortest_target_distances)),
        "target_side_balance": float(abs(left_count - right_count) / max(1, len(source_cameras))),
    }


def normalize_source_policy(root_name, manifest_policy, case_profile):
    if manifest_policy:
        return str(manifest_policy)
    root_name = str(root_name).lower()
    if case_profile == "23cam_fullset":
        return "full_rig_excluding_target"
    if "nearest" in root_name:
        return "nearest_ring"
    if "uniform" in root_name:
        return "uniform_ring"
    if "targetaware" in root_name or "rotate" in root_name:
        return "rotate_template_offsets"
    return "fixed_template"


def load_case_records(sweep_root, local_zju_root, ring_cache):
    sweep_root = Path(sweep_root)
    root_label = sweep_root.name
    manifest = json.loads((sweep_root / "sweep_manifest.json").read_text(encoding="utf-8"))
    summary_payload = json.loads((sweep_root / "summary.json").read_text(encoding="utf-8"))
    rows = summary_payload.get("rows", [])
    records = []
    for row in rows:
        case_dir = Path(row["case_dir"])
        case_summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
        seq_name = case_summary["case"]["seq_name"]
        if seq_name not in ring_cache:
            ring_cache[seq_name] = load_ring_order(Path(local_zju_root) / seq_name)
        ring_order = ring_cache[seq_name]
        source_cameras = list(case_summary["case"]["source_cameras"])
        target_camera = str(case_summary["case"]["target_camera"])
        source_policy = normalize_source_policy(
            root_name=root_label,
            manifest_policy=manifest.get("source_policy"),
            case_profile=case_summary["case"]["view_profile"],
        )
        geom = source_geometry_features(source_cameras, target_camera, ring_order)
        point_branch = case_summary["branches"]["point_map"]
        depth_branch = case_summary["branches"]["depth_unproject"]
        record = {
            "root_label": root_label,
            "source_policy": source_policy,
            "seq_name": seq_name,
            "view_profile": case_summary["case"]["view_profile"],
            "frame_id": int(case_summary["case"]["frame_id"]),
            "target_camera": target_camera,
            "source_count": int(case_summary["case"]["source_count"]),
            "decision": case_summary["decision"]["decision"],
            "mae_winner": case_summary["decision"]["mae_winner"],
            "coverage_winner": case_summary["decision"]["coverage_winner"],
            "geometry_gain": float(point_branch["metrics"]["mae"] - depth_branch["metrics"]["mae"]),
            "coverage_gain": float(depth_branch["render"]["coverage_ratio"] - point_branch["render"]["coverage_ratio"]),
            "point_mae": float(point_branch["metrics"]["mae"]),
            "depth_mae": float(depth_branch["metrics"]["mae"]),
            "point_cov": float(point_branch["render"]["coverage_ratio"]),
            "depth_cov": float(depth_branch["render"]["coverage_ratio"]),
            "point_ssim": float(point_branch["metrics"]["ssim"]),
            "depth_ssim": float(depth_branch["metrics"]["ssim"]),
            "point_conf_p50": float(point_branch["summary"]["confidence_percentiles"]["p50"]),
            "depth_conf_p50": float(depth_branch["summary"]["confidence_percentiles"]["p50"]),
            "point_conf_p95": float(point_branch["summary"]["confidence_percentiles"]["p95"]),
            "depth_conf_p95": float(depth_branch["summary"]["confidence_percentiles"]["p95"]),
            "point_rendered_points": int(point_branch["render"]["rendered_points"]),
            "depth_rendered_points": int(depth_branch["render"]["rendered_points"]),
            "point_valid_contrib": int(point_branch["render"]["valid_contrib"]),
            "depth_valid_contrib": int(depth_branch["render"]["valid_contrib"]),
            "point_mean_conf": float(point_branch["render"]["mean_conf"]),
            "depth_mean_conf": float(depth_branch["render"]["mean_conf"]),
            "alignment_rmse_before": float(case_summary["alignment"]["src_center_rmse_before"]),
            "alignment_rmse_after": float(case_summary["alignment"]["src_center_rmse_after"]),
            "branch_dist_p50": float(case_summary["branch_delta"]["distance_percentiles"]["p50"]),
            "branch_dist_p90": float(case_summary["branch_delta"]["distance_percentiles"]["p90"]),
        }
        record.update(geom)
        records.append(record)
    return records


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def count_decisions(records):
    return {
        "depth_unproject": sum(1 for record in records if record["decision"] == "depth_unproject"),
        "point_map": sum(1 for record in records if record["decision"] == "point_map"),
        "tie": sum(1 for record in records if record["decision"] == "tie"),
    }


def summarize_records(records):
    counts = count_decisions(records)
    return {
        "runs": len(records),
        "depth_wins": counts["depth_unproject"],
        "point_wins": counts["point_map"],
        "ties": counts["tie"],
        "avg_geometry_gain": mean(record["geometry_gain"] for record in records),
        "avg_coverage_gain": mean(record["coverage_gain"] for record in records),
        "avg_alignment_rmse_after": mean(record["alignment_rmse_after"] for record in records),
        "avg_branch_dist_p90": mean(record["branch_dist_p90"] for record in records),
        "avg_source_coverage_ratio": mean(record["source_coverage_ratio"] for record in records),
        "avg_source_gap_std": mean(record["source_gap_std"] for record in records),
        "avg_target_ring_dist_mean": mean(record["target_ring_dist_mean"] for record in records),
        "avg_point_conf_p50": mean(record["point_conf_p50"] for record in records),
        "avg_depth_conf_p50": mean(record["depth_conf_p50"] for record in records),
        "avg_point_rendered_points": mean(record["point_rendered_points"] for record in records),
        "avg_depth_rendered_points": mean(record["depth_rendered_points"] for record in records),
    }


def write_case_csv(path, records):
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def top_camera_table(records, limit=10):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["root_label"], record["view_profile"], record["target_camera"])].append(record)
    rows = []
    for (root_label, profile, target_camera), subset in grouped.items():
        stats = summarize_records(subset)
        rows.append(
            {
                "root_label": root_label,
                "view_profile": profile,
                "target_camera": target_camera,
                "runs": len(subset),
                "depth_wins": stats["depth_wins"],
                "point_wins": stats["point_wins"],
                "avg_geometry_gain": stats["avg_geometry_gain"],
                "avg_coverage_gain": stats["avg_coverage_gain"],
                "avg_alignment_rmse_after": stats["avg_alignment_rmse_after"],
                "avg_branch_dist_p90": stats["avg_branch_dist_p90"],
            }
        )
    rows.sort(key=lambda row: (row["avg_geometry_gain"], row["avg_coverage_gain"], -row["point_wins"]))
    return rows[:limit]


def frame_table(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["root_label"], record["view_profile"], record["frame_id"])].append(record)
    rows = []
    for (root_label, profile, frame_id), subset in sorted(grouped.items()):
        stats = summarize_records(subset)
        rows.append(
            {
                "root_label": root_label,
                "view_profile": profile,
                "frame_id": frame_id,
                **stats,
            }
        )
    return rows


def write_markdown(path, records, grouped_stats, selected_groups):
    lines = [
        "# ZJU Geometry Diagnostics",
        "",
        f"- cases: `{len(records)}`",
        f"- groups: `{len(grouped_stats)}`",
        "",
        "## Group Overview",
        "",
        "| Label | Policy | Profile | Runs | Depth Wins | Point Wins | Ties | Avg Geometry Gain | Avg Coverage Gain | Avg Align RMSE After | Avg Branch Dist P90 | Avg Source Coverage | Avg Gap Std | Avg Target Ring Dist |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for group_key, stats in grouped_stats.items():
        label, policy, profile = group_key
        lines.append(
            "| {label} | {policy} | {profile} | {runs} | {depth_wins} | {point_wins} | {ties} | {avg_geometry_gain:.6f} | {avg_coverage_gain:.6f} | {avg_alignment_rmse_after:.6f} | {avg_branch_dist_p90:.6f} | {avg_source_coverage_ratio:.4f} | {avg_source_gap_std:.4f} | {avg_target_ring_dist_mean:.4f} |".format(
                label=label,
                policy=policy,
                profile=profile,
                **stats,
            )
        )

    lines.extend(["", "## Decision Diagnostics", ""])
    for title, subset in selected_groups:
        lines.append(f"### {title}")
        lines.append("")
        decision_groups = {
            "depth_unproject": [record for record in subset if record["decision"] == "depth_unproject"],
            "point_map": [record for record in subset if record["decision"] == "point_map"],
            "tie": [record for record in subset if record["decision"] == "tie"],
        }
        lines.extend(
            [
                "| Decision | Runs | Avg Geometry Gain | Avg Coverage Gain | Avg Align RMSE After | Avg Branch Dist P90 | Avg Point Conf P50 | Avg Depth Conf P50 | Avg Source Coverage | Avg Gap Std | Avg Target Ring Dist |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for decision_name, decision_subset in decision_groups.items():
            stats = summarize_records(decision_subset)
            lines.append(
                "| {decision} | {runs} | {avg_geometry_gain:.6f} | {avg_coverage_gain:.6f} | {avg_alignment_rmse_after:.6f} | {avg_branch_dist_p90:.6f} | {avg_point_conf_p50:.4f} | {avg_depth_conf_p50:.4f} | {avg_source_coverage_ratio:.4f} | {avg_source_gap_std:.4f} | {avg_target_ring_dist_mean:.4f} |".format(
                    decision=decision_name,
                    **stats,
                )
            )
        lines.append("")

    lines.extend(["## Hard Cameras", ""])
    for row in top_camera_table(records, limit=12):
        lines.append(
            "- `{root_label} / {view_profile} / {target_camera}`: point_wins=`{point_wins}`, depth_wins=`{depth_wins}`, avg_geometry_gain=`{avg_geometry_gain:.6f}`, avg_coverage_gain=`{avg_coverage_gain:.6f}`, avg_align_rmse_after=`{avg_alignment_rmse_after:.6f}`".format(
                **row
            )
        )

    lines.extend(["", "## Readout", ""])
    lines.extend(
        [
            "- `avg_alignment_rmse_after` helps check whether bad sparse cases are mainly explained by worse Sim(3) alignment.",
            "- `avg_branch_dist_p90` helps check whether the two point sources diverge more strongly in hard cases.",
            "- `avg_source_coverage_ratio`, `avg_gap_std`, and `avg_target_ring_dist_mean` summarize how the chosen source cameras sit on the real ZJU camera ring.",
            "- This report is intended to decide whether the next step should be more source-policy work or whether the geometry path is already clean enough to justify longer training.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ring_cache = {}
    records = []
    for sweep_root in args.sweep_roots:
        records.extend(load_case_records(sweep_root, args.local_zju_root, ring_cache))

    grouped = defaultdict(list)
    for record in records:
        grouped[(record["root_label"], record["source_policy"], record["view_profile"])].append(record)

    grouped_stats = {group_key: summarize_records(group_records) for group_key, group_records in sorted(grouped.items())}
    selected_groups = []
    preferred_order = [
        ("round1_coreview390_v1", "full_rig_excluding_target", "23cam_fullset"),
        ("round2_coreview390_targetaware_v1", "rotate_template_offsets", "6src_hist"),
        ("round2_coreview390_targetaware_v1", "rotate_template_offsets", "12src_nested"),
        ("round3_12src_uniform_v1", "uniform_ring", "12src_nested"),
    ]
    for group_key in preferred_order:
        if group_key in grouped:
            selected_groups.append((f"{group_key[0]} / {group_key[2]} / {group_key[1]}", grouped[group_key]))

    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "records": records,
                "grouped_stats": {
                    "|".join(map(str, group_key)): stats for group_key, stats in grouped_stats.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_case_csv(output_dir / "case_diagnostics.csv", records)
    write_case_csv(output_dir / "frame_diagnostics.csv", frame_table(records))
    write_markdown(output_dir / "summary.md", records, grouped_stats, selected_groups)
    print(output_dir / "summary.md")


if __name__ == "__main__":
    main()
