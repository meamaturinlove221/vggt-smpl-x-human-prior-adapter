import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize matched-source legacy-native backfill cases against current geometry summaries."
    )
    parser.add_argument("--backfill_manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_run_dir(frame_case_root: Path) -> Path:
    runs = sorted(frame_case_root.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No run_* directory under {frame_case_root}")
    return runs[-1]


def mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict) -> None:
    agg = payload["aggregate"]
    tag = str(payload["legacy_view_profile_tag"])
    lines = [
        "# Legacy Backfill Summary",
        "",
        f"- batch_root: `{payload['batch_root']}`",
        f"- current_sweep_root: `{payload['current_sweep_root']}`",
        f"- view_profile_tag: `{payload['legacy_view_profile_tag']}`",
        f"- cases: `{agg['cases']}`",
        f"- avg legacy native mae: `{agg['avg_legacy_native_mae']:.6f}`",
        f"- avg current point mae: `{agg['avg_current_point_mae']:.6f}`",
        f"- avg current depth mae: `{agg['avg_current_depth_mae']:.6f}`",
        f"- avg legacy gap point: `{agg['avg_legacy_gap_point']:.6f}`",
        f"- avg legacy gap depth: `{agg['avg_legacy_gap_depth']:.6f}`",
        f"- depth better than point: `{agg['depth_better_than_point']}`",
        "",
        "## Cases",
        "",
        "| Target | Legacy MAE | Point MAE | Depth MAE | Depth Gain | Gap Point | Gap Depth | Decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {target_camera} | {legacy_native_mae:.4f} | {current_point_mae:.4f} | {current_depth_mae:.4f} | {depth_vs_point_gain:.4f} | {legacy_gap_point:.4f} | {legacy_gap_depth:.4f} | {current_decision} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- This is the strict same-source comparison for the `{tag}` backfill batch.",
            "- `legacy native` and current branch results now share the same target camera and the same extracted source-camera list.",
            "- A positive `Depth Gain` means current `depth_unproject` beats current `point_map` on MAE.",
            "- `Gap Point` and `Gap Depth` are current-branch MAE minus legacy-native MAE, so smaller is better.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    manifest_path = Path(args.backfill_manifest)
    manifest = load_json(manifest_path)
    batch_root = Path(manifest["out_root"])
    legacy_tag = str(manifest["legacy_view_profile_tag"])
    seq_name = str(manifest["seq_name"])
    frame_id = int(manifest["frame_id"])

    rows = []
    for item in manifest["rows"]:
        target_camera = str(item["target_camera"])
        current_summary_path = Path(item["current_summary_json"])
        current_summary = load_json(current_summary_path)
        case_root = batch_root / legacy_tag / seq_name / f"frame_{frame_id:06d}_{target_camera}"
        run_dir = latest_run_dir(case_root)
        legacy_report_path = run_dir / "report.json"
        legacy_report = load_json(legacy_report_path)

        point_mae = float(current_summary["branches"]["point_map"]["metrics"]["mae"])
        depth_mae = float(current_summary["branches"]["depth_unproject"]["metrics"]["mae"])
        legacy_mae = float(legacy_report["metrics"]["native"]["mae"])
        rows.append(
            {
                "target_camera": target_camera,
                "current_summary_json": str(current_summary_path.resolve()),
                "legacy_report_json": str(legacy_report_path.resolve()),
                "legacy_native_mae": legacy_mae,
                "current_point_mae": point_mae,
                "current_depth_mae": depth_mae,
                "depth_vs_point_gain": point_mae - depth_mae,
                "legacy_gap_point": point_mae - legacy_mae,
                "legacy_gap_depth": depth_mae - legacy_mae,
                "current_decision": str(current_summary["decision"]["decision"]),
                "source_cameras": list(current_summary["case"]["source_cameras"]),
            }
        )

    aggregate = {
        "cases": len(rows),
        "avg_legacy_native_mae": mean(row["legacy_native_mae"] for row in rows),
        "avg_current_point_mae": mean(row["current_point_mae"] for row in rows),
        "avg_current_depth_mae": mean(row["current_depth_mae"] for row in rows),
        "avg_legacy_gap_point": mean(row["legacy_gap_point"] for row in rows),
        "avg_legacy_gap_depth": mean(row["legacy_gap_depth"] for row in rows),
        "depth_better_than_point": sum(1 for row in rows if row["depth_vs_point_gain"] > 0.0),
    }

    payload = {
        "batch_root": str(batch_root.resolve()),
        "current_sweep_root": str(manifest["current_sweep_root"]),
        "legacy_view_profile_tag": legacy_tag,
        "aggregate": aggregate,
        "rows": rows,
    }
    output_dir = Path(args.output_dir)
    write_json(output_dir / "summary.json", payload)
    write_markdown(output_dir / "summary.md", payload)


if __name__ == "__main__":
    main()
