import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two ZJU geometry view-sweep summary.json files."
    )
    parser.add_argument("--round_a", type=str, required=True, help="Baseline sweep summary.json")
    parser.add_argument("--round_b", type=str, required=True, help="Candidate sweep summary.json")
    parser.add_argument("--label_a", type=str, default="round_a")
    parser.add_argument("--label_b", type=str, default="round_b")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for markdown/json/csv comparison outputs.",
    )
    return parser.parse_args()


def load_summary(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    manifest = payload.get("manifest", {})
    keyed = {}
    for row in rows:
        key = (str(row["view_profile"]), int(row["frame_id"]), str(row["target_camera"]))
        keyed[key] = row
    return {
        "path": str(Path(path).resolve()),
        "payload": payload,
        "rows": rows,
        "manifest": manifest,
        "keyed": keyed,
    }


def count_decisions(rows, decision_key="decision"):
    return {
        "depth_unproject": sum(1 for row in rows if row[decision_key] == "depth_unproject"),
        "point_map": sum(1 for row in rows if row[decision_key] == "point_map"),
        "tie": sum(1 for row in rows if row[decision_key] == "tie"),
    }


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def geometry_gain(row):
    return float(row["point_mae"]) - float(row["depth_mae"])


def coverage_gain(row):
    return float(row["depth_cov"]) - float(row["point_cov"])


def build_common_rows(summary_a, summary_b):
    common_keys = sorted(set(summary_a["keyed"]).intersection(summary_b["keyed"]))
    rows = []
    for key in common_keys:
        row_a = summary_a["keyed"][key]
        row_b = summary_b["keyed"][key]
        rows.append(
            {
                "view_profile": key[0],
                "frame_id": key[1],
                "target_camera": key[2],
                "decision_a": row_a["decision"],
                "decision_b": row_b["decision"],
                "source_count_a": row_a["source_count"],
                "source_count_b": row_b["source_count"],
                "geometry_gain_a": geometry_gain(row_a),
                "geometry_gain_b": geometry_gain(row_b),
                "coverage_gain_a": coverage_gain(row_a),
                "coverage_gain_b": coverage_gain(row_b),
                "geometry_gain_delta": geometry_gain(row_b) - geometry_gain(row_a),
                "coverage_gain_delta": coverage_gain(row_b) - coverage_gain(row_a),
                "point_mae_a": row_a["point_mae"],
                "depth_mae_a": row_a["depth_mae"],
                "point_mae_b": row_b["point_mae"],
                "depth_mae_b": row_b["depth_mae"],
                "point_cov_a": row_a["point_cov"],
                "depth_cov_a": row_a["depth_cov"],
                "point_cov_b": row_b["point_cov"],
                "depth_cov_b": row_b["depth_cov"],
            }
        )
    return rows


def compare_transition_counts(common_rows):
    improved = 0
    regressed = 0
    unchanged = 0
    changes = {"point_to_depth": 0, "depth_to_point": 0, "tie_to_depth": 0, "depth_to_tie": 0, "point_to_tie": 0, "tie_to_point": 0}
    for row in common_rows:
        a = row["decision_a"]
        b = row["decision_b"]
        if a == b:
            unchanged += 1
            continue
        if b == "depth_unproject" and a != "depth_unproject":
            improved += 1
        elif a == "depth_unproject" and b != "depth_unproject":
            regressed += 1
        else:
            unchanged += 1

        key = None
        if a == "point_map" and b == "depth_unproject":
            key = "point_to_depth"
        elif a == "depth_unproject" and b == "point_map":
            key = "depth_to_point"
        elif a == "tie" and b == "depth_unproject":
            key = "tie_to_depth"
        elif a == "depth_unproject" and b == "tie":
            key = "depth_to_tie"
        elif a == "point_map" and b == "tie":
            key = "point_to_tie"
        elif a == "tie" and b == "point_map":
            key = "tie_to_point"
        if key:
            changes[key] += 1

    return {
        "improved_to_depth": improved,
        "regressed_from_depth": regressed,
        "unchanged_or_sidegrade": unchanged,
        "transitions": changes,
    }


def summarize_subset(rows):
    return {
        "runs": len(rows),
        "avg_geometry_gain_delta": mean(row["geometry_gain_delta"] for row in rows),
        "avg_coverage_gain_delta": mean(row["coverage_gain_delta"] for row in rows),
        "avg_geometry_gain_a": mean(row["geometry_gain_a"] for row in rows),
        "avg_geometry_gain_b": mean(row["geometry_gain_b"] for row in rows),
        "avg_coverage_gain_a": mean(row["coverage_gain_a"] for row in rows),
        "avg_coverage_gain_b": mean(row["coverage_gain_b"] for row in rows),
        "transition_counts": compare_transition_counts(rows),
    }


def write_csv(path, rows):
    fieldnames = [
        "view_profile",
        "frame_id",
        "target_camera",
        "decision_a",
        "decision_b",
        "source_count_a",
        "source_count_b",
        "geometry_gain_a",
        "geometry_gain_b",
        "geometry_gain_delta",
        "coverage_gain_a",
        "coverage_gain_b",
        "coverage_gain_delta",
        "point_mae_a",
        "depth_mae_a",
        "point_mae_b",
        "depth_mae_b",
        "point_cov_a",
        "depth_cov_a",
        "point_cov_b",
        "depth_cov_b",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path, comparison, common_rows):
    label_a = comparison["label_a"]
    label_b = comparison["label_b"]
    summary_a = comparison["summary_a"]
    summary_b = comparison["summary_b"]
    lines = [
        "# ZJU Geometry Sweep Comparison",
        "",
        f"- {label_a}: `{summary_a['path']}`",
        f"- {label_b}: `{summary_b['path']}`",
        f"- {label_a} source_policy: `{summary_a['manifest'].get('source_policy', 'n/a')}`",
        f"- {label_b} source_policy: `{summary_b['manifest'].get('source_policy', 'n/a')}`",
        f"- {label_a} runs: `{summary_a['row_count']}`",
        f"- {label_b} runs: `{summary_b['row_count']}`",
        f"- common cases: `{len(common_rows)}`",
        f"- only in {label_a}: `{comparison['only_a_count']}`",
        f"- only in {label_b}: `{comparison['only_b_count']}`",
        "",
        "## Common-Case Overall",
        "",
    ]

    overall = comparison["overall_common"]
    counts_a = overall["decision_counts_a"]
    counts_b = overall["decision_counts_b"]
    transitions = overall["transition_counts"]["transitions"]
    lines.extend(
        [
            f"- {label_a} decisions on common cases: depth `{counts_a['depth_unproject']}`, point `{counts_a['point_map']}`, tie `{counts_a['tie']}`",
            f"- {label_b} decisions on common cases: depth `{counts_b['depth_unproject']}`, point `{counts_b['point_map']}`, tie `{counts_b['tie']}`",
            f"- avg geometry gain ({label_a}): `{overall['avg_geometry_gain_a']:.6f}`",
            f"- avg geometry gain ({label_b}): `{overall['avg_geometry_gain_b']:.6f}`",
            f"- avg geometry gain delta ({label_b} - {label_a}): `{overall['avg_geometry_gain_delta']:.6f}`",
            f"- avg coverage gain ({label_a}): `{overall['avg_coverage_gain_a']:.6f}`",
            f"- avg coverage gain ({label_b}): `{overall['avg_coverage_gain_b']:.6f}`",
            f"- avg coverage gain delta ({label_b} - {label_a}): `{overall['avg_coverage_gain_delta']:.6f}`",
            f"- improved_to_depth: `{overall['transition_counts']['improved_to_depth']}`",
            f"- regressed_from_depth: `{overall['transition_counts']['regressed_from_depth']}`",
            f"- point_to_depth: `{transitions['point_to_depth']}`",
            f"- depth_to_point: `{transitions['depth_to_point']}`",
            f"- tie_to_depth: `{transitions['tie_to_depth']}`",
            f"- depth_to_tie: `{transitions['depth_to_tie']}`",
            f"- point_to_tie: `{transitions['point_to_tie']}`",
            f"- tie_to_point: `{transitions['tie_to_point']}`",
            "",
            "## By Profile",
            "",
            "| Profile | Common Runs | "
            + f"{label_a} Depth Wins | {label_b} Depth Wins | "
            + "Avg Geometry Gain Delta | Avg Coverage Gain Delta | Improved To Depth | Regressed From Depth |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for profile_name, stats in comparison["by_profile"].items():
        counts_a = stats["decision_counts_a"]
        counts_b = stats["decision_counts_b"]
        lines.append(
            "| {profile} | {runs} | {depth_a} | {depth_b} | {geom_delta:.6f} | {cov_delta:.6f} | {improved} | {regressed} |".format(
                profile=profile_name,
                runs=stats["runs"],
                depth_a=counts_a["depth_unproject"],
                depth_b=counts_b["depth_unproject"],
                geom_delta=stats["avg_geometry_gain_delta"],
                cov_delta=stats["avg_coverage_gain_delta"],
                improved=stats["transition_counts"]["improved_to_depth"],
                regressed=stats["transition_counts"]["regressed_from_depth"],
            )
        )

    only_b_rows = comparison["only_b_rows"]
    if only_b_rows:
        counts_only_b = count_decisions(only_b_rows)
        lines.extend(
            [
                "",
                f"## Cases Only In {label_b}",
                "",
                f"- runs: `{len(only_b_rows)}`",
                f"- depth_unproject wins: `{counts_only_b['depth_unproject']}`",
                f"- point_map wins: `{counts_only_b['point_map']}`",
                f"- ties: `{counts_only_b['tie']}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- This compares the same VGGT geometry branch evaluation pipeline under two source-selection policies: `{summary_a['manifest'].get('source_policy', 'n/a')}` and `{summary_b['manifest'].get('source_policy', 'n/a')}`.",
            f"- Positive geometry gain delta means `{label_b}` made `depth + camera` more favorable relative to `point_map`.",
            "- This report is intended to answer whether sparse-view failures are mostly caused by fixed source subsets rather than by the geometry branch itself.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    summary_a = load_summary(args.round_a)
    summary_b = load_summary(args.round_b)

    common_rows = build_common_rows(summary_a, summary_b)
    only_a_keys = sorted(set(summary_a["keyed"]) - set(summary_b["keyed"]))
    only_b_keys = sorted(set(summary_b["keyed"]) - set(summary_a["keyed"]))
    only_b_rows = [summary_b["keyed"][key] for key in only_b_keys]

    overall_common = summarize_subset(common_rows)
    overall_common["decision_counts_a"] = count_decisions([summary_a["keyed"][(row["view_profile"], row["frame_id"], row["target_camera"])] for row in common_rows])
    overall_common["decision_counts_b"] = count_decisions([summary_b["keyed"][(row["view_profile"], row["frame_id"], row["target_camera"])] for row in common_rows])

    profiles = sorted({row["view_profile"] for row in common_rows})
    by_profile = {}
    for profile_name in profiles:
        subset = [row for row in common_rows if row["view_profile"] == profile_name]
        stats = summarize_subset(subset)
        stats["decision_counts_a"] = count_decisions(
            [summary_a["keyed"][(row["view_profile"], row["frame_id"], row["target_camera"])] for row in subset]
        )
        stats["decision_counts_b"] = count_decisions(
            [summary_b["keyed"][(row["view_profile"], row["frame_id"], row["target_camera"])] for row in subset]
        )
        by_profile[profile_name] = stats

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "summary_a": {"path": summary_a["path"], "manifest": summary_a["manifest"], "row_count": len(summary_a["rows"])},
        "summary_b": {"path": summary_b["path"], "manifest": summary_b["manifest"], "row_count": len(summary_b["rows"])},
        "only_a_count": len(only_a_keys),
        "only_b_count": len(only_b_keys),
        "overall_common": overall_common,
        "by_profile": by_profile,
        "only_b_rows": only_b_rows,
    }

    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "common_case_comparison.csv", common_rows)
    write_markdown(output_dir / "comparison.md", comparison, common_rows)
    print(output_dir / "comparison.md")


if __name__ == "__main__":
    main()
