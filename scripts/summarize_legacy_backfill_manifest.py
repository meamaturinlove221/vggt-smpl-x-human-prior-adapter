import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize matched-source legacy backfill cases against current geometry summaries."
    )
    parser.add_argument("--backfill_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--label", type=str, default="legacy_backfill_summary")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "seq_name",
        "frame_id",
        "view_profile",
        "target_camera",
        "current_decision",
        "legacy_native_mae",
        "current_point_mae",
        "current_depth_mae",
        "depth_vs_point_gain",
        "legacy_gap_point",
        "legacy_gap_depth",
        "gap_depth_minus_point",
        "gap_delta_abs",
        "current_summary_json",
        "legacy_report_json",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_markdown(path: Path, payload: dict) -> None:
    agg = payload["aggregate"]
    lines = [
        f"# {payload['label']}",
        "",
        f"- backfill_manifest: `{payload['backfill_manifest']}`",
        f"- cases: `{agg['cases']}`",
        f"- avg legacy native mae: `{agg['avg_legacy_native_mae']:.6f}`",
        f"- avg current point mae: `{agg['avg_current_point_mae']:.6f}`",
        f"- avg current depth mae: `{agg['avg_current_depth_mae']:.6f}`",
        f"- avg legacy gap point: `{agg['avg_legacy_gap_point']:.6f}`",
        f"- avg legacy gap depth: `{agg['avg_legacy_gap_depth']:.6f}`",
        f"- avg gap depth-minus-point: `{agg['avg_gap_depth_minus_point']:.6f}`",
        f"- depth better than point: `{agg['depth_better_than_point']}`",
        "",
        "## Cases",
        "",
        "| Case | Decision | Legacy MAE | Point MAE | Depth MAE | Gap Point | Gap Depth | |GapDepth-GapPoint| |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| `{case_id}` | `{current_decision}` | {legacy_native_mae:.4f} | {current_point_mae:.4f} | {current_depth_mae:.4f} | {legacy_gap_point:.4f} | {legacy_gap_depth:.4f} | {gap_delta_abs:.4f} |".format(
                **row
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    manifest = load_json(args.backfill_manifest)
    rows = []
    for item in manifest.get("rows", []):
        current_summary_path = Path(item["current_summary_json"]).resolve()
        legacy_report_path = Path(item["legacy_report_json"]).resolve()
        current_summary = load_json(current_summary_path)
        legacy_report = load_json(legacy_report_path)

        point_mae = float(current_summary["branches"]["point_map"]["metrics"]["mae"])
        depth_mae = float(current_summary["branches"]["depth_unproject"]["metrics"]["mae"])
        legacy_mae = float(legacy_report["metrics"]["native"]["mae"])
        gap_point = point_mae - legacy_mae
        gap_depth = depth_mae - legacy_mae
        row = {
            "case_id": "{seq_name}_frame_{frame_id:06d}_{target_camera}".format(
                seq_name=str(item["seq_name"]),
                frame_id=int(item["frame_id"]),
                target_camera=str(item["target_camera"]),
            ),
            "seq_name": str(item["seq_name"]),
            "frame_id": int(item["frame_id"]),
            "view_profile": str(item.get("view_profile", "")),
            "target_camera": str(item["target_camera"]),
            "current_summary_json": str(current_summary_path),
            "legacy_report_json": str(legacy_report_path),
            "legacy_native_mae": legacy_mae,
            "current_point_mae": point_mae,
            "current_depth_mae": depth_mae,
            "depth_vs_point_gain": point_mae - depth_mae,
            "legacy_gap_point": gap_point,
            "legacy_gap_depth": gap_depth,
            "gap_depth_minus_point": gap_depth - gap_point,
            "gap_delta_abs": abs(gap_depth - gap_point),
            "current_decision": str(current_summary["decision"]["decision"]),
            "source_cameras": list(item.get("source_cameras", [])),
        }
        rows.append(row)

    rows.sort(key=lambda item: (item["seq_name"], item["frame_id"], item["target_camera"]))
    aggregate = {
        "cases": len(rows),
        "avg_legacy_native_mae": mean(row["legacy_native_mae"] for row in rows),
        "avg_current_point_mae": mean(row["current_point_mae"] for row in rows),
        "avg_current_depth_mae": mean(row["current_depth_mae"] for row in rows),
        "avg_legacy_gap_point": mean(row["legacy_gap_point"] for row in rows),
        "avg_legacy_gap_depth": mean(row["legacy_gap_depth"] for row in rows),
        "avg_gap_depth_minus_point": mean(row["gap_depth_minus_point"] for row in rows),
        "depth_better_than_point": sum(1 for row in rows if row["depth_vs_point_gain"] > 0.0),
    }
    payload = {
        "label": args.label,
        "backfill_manifest": str(args.backfill_manifest.resolve()),
        "aggregate": aggregate,
        "rows": rows,
    }
    output_dir = ensure_dir(args.output_dir.resolve())
    write_json(output_dir / "summary.json", payload)
    write_csv(output_dir / "summary.csv", rows)
    write_markdown(output_dir / "summary.md", payload)
    print(f"[done] Wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
