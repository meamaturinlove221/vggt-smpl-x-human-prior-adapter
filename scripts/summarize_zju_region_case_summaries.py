import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REGION_ORDER = ("fg_human", "fg_edge", "bg_far", "bg_bottom_band")
BRANCH_ORDER = ("point_map", "depth_unproject")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize region diagnostics from one or more ZJU geometry summary.json files."
    )
    parser.add_argument(
        "--summary_json",
        nargs="+",
        required=True,
        help="One or more summary.json files produced by compare_geometry_branches_zju_report.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write summary.json / summary.csv / summary.md.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="zju_region_summary",
        help="Human-readable label for the batch.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def mean_or_none(values):
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def fmt(value):
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def extract_case_row(summary_path: Path, payload: dict) -> dict:
    case = payload.get("case", {})
    branches = payload.get("branches", {})
    region_payload = payload.get("region_diagnostics", {})
    region_branches = region_payload.get("branches", {})
    region_compare = region_payload.get("comparison", {})
    decision = payload.get("decision", {})

    row = {
        "summary_json": str(summary_path.resolve()),
        "seq_name": case.get("seq_name", ""),
        "frame_id": int(case.get("frame_id", 0)),
        "view_profile": case.get("view_profile", ""),
        "target_camera": case.get("target_camera", ""),
        "full_decision": decision.get("decision", ""),
        "full_mae_point": branches.get("point_map", {}).get("metrics", {}).get("mae"),
        "full_mae_depth": branches.get("depth_unproject", {}).get("metrics", {}).get("mae"),
        "full_cov_point": branches.get("point_map", {}).get("render", {}).get("coverage_ratio"),
        "full_cov_depth": branches.get("depth_unproject", {}).get("render", {}).get("coverage_ratio"),
    }
    if row["full_mae_point"] is not None and row["full_mae_depth"] is not None:
        row["full_depth_minus_point_mae"] = float(row["full_mae_depth"]) - float(row["full_mae_point"])
    else:
        row["full_depth_minus_point_mae"] = None
    if row["full_cov_point"] is not None and row["full_cov_depth"] is not None:
        row["full_depth_minus_point_cov"] = float(row["full_cov_depth"]) - float(row["full_cov_point"])
    else:
        row["full_depth_minus_point_cov"] = None

    for region_name in REGION_ORDER:
        row[f"{region_name}_mae_winner"] = region_compare.get(region_name, {}).get("mae_winner", "n/a")
        row[f"{region_name}_coverage_winner"] = region_compare.get(region_name, {}).get("coverage_winner", "n/a")
        for branch_name in BRANCH_ORDER:
            branch_region = region_branches.get(branch_name, {}).get("regions", {}).get(region_name, {})
            metrics = branch_region.get("render_metrics", {})
            row[f"{region_name}_{branch_name}_mae"] = metrics.get("mae")
            row[f"{region_name}_{branch_name}_psnr"] = metrics.get("psnr")
            row[f"{region_name}_{branch_name}_ssim"] = metrics.get("ssim")
            row[f"{region_name}_{branch_name}_coverage"] = branch_region.get("coverage_ratio")
            row[f"{region_name}_{branch_name}_projected_points"] = branch_region.get("projected_point_count")
        point_mae = row[f"{region_name}_point_map_mae"]
        depth_mae = row[f"{region_name}_depth_unproject_mae"]
        point_cov = row[f"{region_name}_point_map_coverage"]
        depth_cov = row[f"{region_name}_depth_unproject_coverage"]
        row[f"{region_name}_depth_minus_point_mae"] = (
            None if point_mae is None or depth_mae is None else float(depth_mae) - float(point_mae)
        )
        row[f"{region_name}_depth_minus_point_cov"] = (
            None if point_cov is None or depth_cov is None else float(depth_cov) - float(point_cov)
        )
    return row


def build_aggregate(rows: list[dict]) -> dict:
    aggregate = {
        "case_count": len(rows),
        "full_decision_counts": dict(Counter(row["full_decision"] for row in rows)),
        "full_avg_point_mae": mean_or_none(row["full_mae_point"] for row in rows),
        "full_avg_depth_mae": mean_or_none(row["full_mae_depth"] for row in rows),
        "full_avg_depth_minus_point_mae": mean_or_none(row["full_depth_minus_point_mae"] for row in rows),
        "full_avg_point_coverage": mean_or_none(row["full_cov_point"] for row in rows),
        "full_avg_depth_coverage": mean_or_none(row["full_cov_depth"] for row in rows),
        "full_avg_depth_minus_point_cov": mean_or_none(row["full_depth_minus_point_cov"] for row in rows),
        "regions": {},
    }

    for region_name in REGION_ORDER:
        mae_counter = Counter(row[f"{region_name}_mae_winner"] for row in rows)
        cov_counter = Counter(row[f"{region_name}_coverage_winner"] for row in rows)
        region_entry = {
            "mae_winner_counts": dict(mae_counter),
            "coverage_winner_counts": dict(cov_counter),
            "avg_depth_minus_point_mae": mean_or_none(row[f"{region_name}_depth_minus_point_mae"] for row in rows),
            "avg_depth_minus_point_cov": mean_or_none(row[f"{region_name}_depth_minus_point_cov"] for row in rows),
            "branches": {},
        }
        for branch_name in BRANCH_ORDER:
            region_entry["branches"][branch_name] = {
                "avg_mae": mean_or_none(row[f"{region_name}_{branch_name}_mae"] for row in rows),
                "avg_psnr": mean_or_none(row[f"{region_name}_{branch_name}_psnr"] for row in rows),
                "avg_ssim": mean_or_none(row[f"{region_name}_{branch_name}_ssim"] for row in rows),
                "avg_coverage": mean_or_none(row[f"{region_name}_{branch_name}_coverage"] for row in rows),
                "avg_projected_points": mean_or_none(row[f"{region_name}_{branch_name}_projected_points"] for row in rows),
            }
        aggregate["regions"][region_name] = region_entry
    return aggregate


def write_csv(path: Path, rows: list[dict]):
    fieldnames = [
        "seq_name",
        "frame_id",
        "view_profile",
        "target_camera",
        "full_decision",
        "full_mae_point",
        "full_mae_depth",
        "full_depth_minus_point_mae",
        "full_cov_point",
        "full_cov_depth",
        "full_depth_minus_point_cov",
    ]
    for region_name in REGION_ORDER:
        fieldnames.extend(
            [
                f"{region_name}_mae_winner",
                f"{region_name}_coverage_winner",
                f"{region_name}_point_map_mae",
                f"{region_name}_depth_unproject_mae",
                f"{region_name}_depth_minus_point_mae",
                f"{region_name}_point_map_coverage",
                f"{region_name}_depth_unproject_coverage",
                f"{region_name}_depth_minus_point_cov",
            ]
        )
    fieldnames.append("summary_json")

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_markdown(path: Path, label: str, rows: list[dict], aggregate: dict):
    lines = [
        f"# {label}",
        "",
        f"- case_count: `{aggregate['case_count']}`",
        f"- full_decision_counts: `{json.dumps(aggregate['full_decision_counts'], ensure_ascii=False)}`",
        f"- full_avg_point_mae: `{fmt(aggregate['full_avg_point_mae'])}`",
        f"- full_avg_depth_mae: `{fmt(aggregate['full_avg_depth_mae'])}`",
        f"- full_avg_depth_minus_point_mae: `{fmt(aggregate['full_avg_depth_minus_point_mae'])}`",
        f"- full_avg_point_coverage: `{fmt(aggregate['full_avg_point_coverage'])}`",
        f"- full_avg_depth_coverage: `{fmt(aggregate['full_avg_depth_coverage'])}`",
        f"- full_avg_depth_minus_point_cov: `{fmt(aggregate['full_avg_depth_minus_point_cov'])}`",
        "",
        "## Region Aggregate",
        "",
        "| Region | MAE Winner Counts | Coverage Winner Counts | Avg Depth-Point MAE | Avg Depth-Point Cov |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for region_name in REGION_ORDER:
        entry = aggregate["regions"][region_name]
        lines.append(
            "| `{region}` | `{mae}` | `{cov}` | {mae_delta} | {cov_delta} |".format(
                region=region_name,
                mae=json.dumps(entry["mae_winner_counts"], ensure_ascii=False),
                cov=json.dumps(entry["coverage_winner_counts"], ensure_ascii=False),
                mae_delta=fmt(entry["avg_depth_minus_point_mae"]),
                cov_delta=fmt(entry["avg_depth_minus_point_cov"]),
            )
        )

    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Target | Full Decision | Full Depth-Point MAE | fg_human Winner | bg_far Winner | bg_bottom Winner | Summary |",
            "| --- | --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| `{target}` | `{decision}` | {full_delta} | `{fg}` | `{bg_far}` | `{bg_bottom}` | `{summary}` |".format(
                target=row["target_camera"],
                decision=row["full_decision"],
                full_delta=fmt(row["full_depth_minus_point_mae"]),
                fg=row["fg_human_mae_winner"],
                bg_far=row["bg_far_mae_winner"],
                bg_bottom=row["bg_bottom_band_mae_winner"],
                summary=row["summary_json"],
            )
        )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output_dir.resolve())

    rows = []
    for raw_path in args.summary_json:
        summary_path = Path(raw_path).resolve()
        payload = load_json(summary_path)
        rows.append(extract_case_row(summary_path, payload))

    rows.sort(key=lambda row: (row["seq_name"], row["frame_id"], row["view_profile"], row["target_camera"]))
    aggregate = build_aggregate(rows)

    payload = {
        "label": args.label,
        "output_dir": str(output_dir),
        "aggregate": aggregate,
        "rows": rows,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    write_csv(output_dir / "summary.csv", rows)
    write_markdown(output_dir / "summary.md", args.label, rows, aggregate)
    print(f"[done] Wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
