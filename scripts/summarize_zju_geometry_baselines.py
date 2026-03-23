import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize ZJU geometry baseline runs.")
    parser.add_argument(
        "--input_root",
        type=str,
        default="output/geometry_baseline_zju",
        help="Root directory that contains per-run summary.json files.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="output/geometry_baseline_zju/coreview390_batch_summary",
        help="Output prefix without extension.",
    )
    return parser.parse_args()


def collect_rows(input_root):
    rows = []
    for summary_path in sorted(Path(input_root).glob("*/summary.json")):
        if "smoke" in summary_path.parent.name.lower():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_dir": str(summary_path.parent),
                "seq_name": payload["case"]["seq_name"],
                "frame_id": payload["case"]["frame_id"],
                "target_camera": payload["case"]["target_camera"],
                "view_profile": payload["case"]["view_profile"],
                "source_count": payload["case"]["source_count"],
                "decision": payload["decision"]["decision"],
                "mae_winner": payload["decision"]["mae_winner"],
                "coverage_winner": payload["decision"]["coverage_winner"],
                "point_mae": payload["branches"]["point_map"]["metrics"]["mae"],
                "depth_mae": payload["branches"]["depth_unproject"]["metrics"]["mae"],
                "point_cov": payload["branches"]["point_map"]["render"]["coverage_ratio"],
                "depth_cov": payload["branches"]["depth_unproject"]["render"]["coverage_ratio"],
                "point_psnr": payload["branches"]["point_map"]["metrics"]["psnr"],
                "depth_psnr": payload["branches"]["depth_unproject"]["metrics"]["psnr"],
                "summary_md": str(summary_path.with_suffix(".md")),
            }
        )
    return rows


def write_markdown(path, rows):
    depth_wins = [row for row in rows if row["decision"] == "depth_unproject"]
    point_wins = [row for row in rows if row["decision"] == "point_map"]
    ties = [row for row in rows if row["decision"] == "tie"]
    lines = [
        "# ZJU Geometry Baseline Summary",
        "",
        f"- runs: `{len(rows)}`",
        f"- depth_unproject_wins: `{len(depth_wins)}`",
        f"- point_map_wins: `{len(point_wins)}`",
        f"- ties: `{len(ties)}`",
        "",
        "| Profile | Sources | Decision | MAE Winner | Cov Winner | Point MAE | Depth MAE | Point Cov | Depth Cov | Summary |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {view_profile} | {source_count} | {decision} | {mae_winner} | {coverage_winner} | {point_mae:.4f} | {depth_mae:.4f} | {point_cov:.4f} | {depth_cov:.4f} | `{summary_md}` |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- `depth + camera` is the decision winner when it has lower MAE and no worse coverage.",
            "- A `tie` means one branch won MAE while the other won coverage.",
            "- For the CoreView_390 human-domain case, if `depth + camera` wins or stays competitive across profiles, the geometry-first direction remains justified.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    rows = collect_rows(args.input_root)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(output_prefix.with_suffix(".md"), rows)
    print(output_prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
