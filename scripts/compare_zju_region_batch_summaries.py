import argparse
import json
from pathlib import Path


REGION_ORDER = ("fg_human", "fg_edge", "bg_far", "bg_bottom_band")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two aggregated ZJU geometry region batch summaries."
    )
    parser.add_argument("--summary_a", type=Path, required=True, help="Baseline region batch summary.json")
    parser.add_argument("--summary_b", type=Path, required=True, help="Candidate region batch summary.json")
    parser.add_argument("--label_a", type=str, default="summary_a")
    parser.add_argument("--label_b", type=str, default="summary_b")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--title", type=str, default="ZJU Region Batch Comparison")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_float(value):
    if value is None:
        return None
    return float(value)


def fmt(value):
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def subtract(a, b):
    if a is None or b is None:
        return None
    return float(b) - float(a)


def keyed_rows(rows: list[dict]) -> dict:
    keyed = {}
    for row in rows:
        key = (
            str(row.get("seq_name", "")),
            int(row.get("frame_id", 0)),
            str(row.get("view_profile", "")),
            str(row.get("target_camera", "")),
        )
        keyed[key] = row
    return keyed


def build_case_rows(rows_a: list[dict], rows_b: list[dict]) -> list[dict]:
    keyed_a = keyed_rows(rows_a)
    keyed_b = keyed_rows(rows_b)
    keys = sorted(set(keyed_a.keys()) & set(keyed_b.keys()))
    cases = []
    for key in keys:
        row_a = keyed_a[key]
        row_b = keyed_b[key]
        cases.append(
            {
                "seq_name": key[0],
                "frame_id": key[1],
                "view_profile": key[2],
                "target_camera": key[3],
                "full_decision_a": row_a.get("full_decision", "n/a"),
                "full_decision_b": row_b.get("full_decision", "n/a"),
                "full_depth_minus_point_mae_a": maybe_float(row_a.get("full_depth_minus_point_mae")),
                "full_depth_minus_point_mae_b": maybe_float(row_b.get("full_depth_minus_point_mae")),
                "full_depth_minus_point_mae_delta": subtract(
                    row_a.get("full_depth_minus_point_mae"),
                    row_b.get("full_depth_minus_point_mae"),
                ),
                "full_depth_minus_point_cov_a": maybe_float(row_a.get("full_depth_minus_point_cov")),
                "full_depth_minus_point_cov_b": maybe_float(row_b.get("full_depth_minus_point_cov")),
                "full_depth_minus_point_cov_delta": subtract(
                    row_a.get("full_depth_minus_point_cov"),
                    row_b.get("full_depth_minus_point_cov"),
                ),
            }
        )
        for region_name in REGION_ORDER:
            cases[-1][f"{region_name}_winner_a"] = row_a.get(f"{region_name}_mae_winner", "n/a")
            cases[-1][f"{region_name}_winner_b"] = row_b.get(f"{region_name}_mae_winner", "n/a")
            cases[-1][f"{region_name}_depth_minus_point_mae_a"] = maybe_float(
                row_a.get(f"{region_name}_depth_minus_point_mae")
            )
            cases[-1][f"{region_name}_depth_minus_point_mae_b"] = maybe_float(
                row_b.get(f"{region_name}_depth_minus_point_mae")
            )
            cases[-1][f"{region_name}_depth_minus_point_mae_delta"] = subtract(
                row_a.get(f"{region_name}_depth_minus_point_mae"),
                row_b.get(f"{region_name}_depth_minus_point_mae"),
            )
            cases[-1][f"{region_name}_depth_minus_point_cov_a"] = maybe_float(
                row_a.get(f"{region_name}_depth_minus_point_cov")
            )
            cases[-1][f"{region_name}_depth_minus_point_cov_b"] = maybe_float(
                row_b.get(f"{region_name}_depth_minus_point_cov")
            )
            cases[-1][f"{region_name}_depth_minus_point_cov_delta"] = subtract(
                row_a.get(f"{region_name}_depth_minus_point_cov"),
                row_b.get(f"{region_name}_depth_minus_point_cov"),
            )
    return cases


def build_payload(payload_a: dict, payload_b: dict, label_a: str, label_b: str, title: str) -> dict:
    agg_a = payload_a["aggregate"]
    agg_b = payload_b["aggregate"]
    rows = build_case_rows(payload_a.get("rows", []), payload_b.get("rows", []))

    full = {
        "case_count_a": int(agg_a.get("case_count", 0)),
        "case_count_b": int(agg_b.get("case_count", 0)),
        "matched_case_count": len(rows),
        "decision_counts_a": agg_a.get("full_decision_counts", {}),
        "decision_counts_b": agg_b.get("full_decision_counts", {}),
        "avg_depth_minus_point_mae_a": maybe_float(agg_a.get("full_avg_depth_minus_point_mae")),
        "avg_depth_minus_point_mae_b": maybe_float(agg_b.get("full_avg_depth_minus_point_mae")),
        "avg_depth_minus_point_mae_delta": subtract(
            agg_a.get("full_avg_depth_minus_point_mae"),
            agg_b.get("full_avg_depth_minus_point_mae"),
        ),
        "avg_depth_minus_point_cov_a": maybe_float(agg_a.get("full_avg_depth_minus_point_cov")),
        "avg_depth_minus_point_cov_b": maybe_float(agg_b.get("full_avg_depth_minus_point_cov")),
        "avg_depth_minus_point_cov_delta": subtract(
            agg_a.get("full_avg_depth_minus_point_cov"),
            agg_b.get("full_avg_depth_minus_point_cov"),
        ),
    }

    regions = {}
    for region_name in REGION_ORDER:
        entry_a = agg_a["regions"][region_name]
        entry_b = agg_b["regions"][region_name]
        regions[region_name] = {
            "mae_winner_counts_a": entry_a.get("mae_winner_counts", {}),
            "mae_winner_counts_b": entry_b.get("mae_winner_counts", {}),
            "coverage_winner_counts_a": entry_a.get("coverage_winner_counts", {}),
            "coverage_winner_counts_b": entry_b.get("coverage_winner_counts", {}),
            "avg_depth_minus_point_mae_a": maybe_float(entry_a.get("avg_depth_minus_point_mae")),
            "avg_depth_minus_point_mae_b": maybe_float(entry_b.get("avg_depth_minus_point_mae")),
            "avg_depth_minus_point_mae_delta": subtract(
                entry_a.get("avg_depth_minus_point_mae"),
                entry_b.get("avg_depth_minus_point_mae"),
            ),
            "avg_depth_minus_point_cov_a": maybe_float(entry_a.get("avg_depth_minus_point_cov")),
            "avg_depth_minus_point_cov_b": maybe_float(entry_b.get("avg_depth_minus_point_cov")),
            "avg_depth_minus_point_cov_delta": subtract(
                entry_a.get("avg_depth_minus_point_cov"),
                entry_b.get("avg_depth_minus_point_cov"),
            ),
        }

    return {
        "title": title,
        "label_a": label_a,
        "label_b": label_b,
        "summary_a": str(payload_a.get("output_dir", "")),
        "summary_b": str(payload_b.get("output_dir", "")),
        "full": full,
        "regions": regions,
        "matched_cases": rows,
    }


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict):
    lines = [
        f"# {payload['title']}",
        "",
        f"- {payload['label_a']}: `{payload['summary_a']}`",
        f"- {payload['label_b']}: `{payload['summary_b']}`",
        "",
        "## Full Frame",
        "",
        f"- matched_case_count: `{payload['full']['matched_case_count']}`",
        f"- decision_counts {payload['label_a']}: `{json.dumps(payload['full']['decision_counts_a'], ensure_ascii=False)}`",
        f"- decision_counts {payload['label_b']}: `{json.dumps(payload['full']['decision_counts_b'], ensure_ascii=False)}`",
        f"- avg depth-point MAE {payload['label_a']}: `{fmt(payload['full']['avg_depth_minus_point_mae_a'])}`",
        f"- avg depth-point MAE {payload['label_b']}: `{fmt(payload['full']['avg_depth_minus_point_mae_b'])}`",
        f"- avg depth-point MAE delta ({payload['label_b']} - {payload['label_a']}): `{fmt(payload['full']['avg_depth_minus_point_mae_delta'])}`",
        f"- avg depth-point coverage {payload['label_a']}: `{fmt(payload['full']['avg_depth_minus_point_cov_a'])}`",
        f"- avg depth-point coverage {payload['label_b']}: `{fmt(payload['full']['avg_depth_minus_point_cov_b'])}`",
        f"- avg depth-point coverage delta ({payload['label_b']} - {payload['label_a']}): `{fmt(payload['full']['avg_depth_minus_point_cov_delta'])}`",
        "",
        "## Region Aggregate",
        "",
        "| Region | Depth-Point MAE A | Depth-Point MAE B | Delta | Depth-Point Cov A | Depth-Point Cov B | Cov Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for region_name in REGION_ORDER:
        entry = payload["regions"][region_name]
        lines.append(
            "| `{region}` | {mae_a} | {mae_b} | {mae_delta} | {cov_a} | {cov_b} | {cov_delta} |".format(
                region=region_name,
                mae_a=fmt(entry["avg_depth_minus_point_mae_a"]),
                mae_b=fmt(entry["avg_depth_minus_point_mae_b"]),
                mae_delta=fmt(entry["avg_depth_minus_point_mae_delta"]),
                cov_a=fmt(entry["avg_depth_minus_point_cov_a"]),
                cov_b=fmt(entry["avg_depth_minus_point_cov_b"]),
                cov_delta=fmt(entry["avg_depth_minus_point_cov_delta"]),
            )
        )

    lines.extend(
        [
            "",
            "## Matched Cases",
            "",
            "| Target | Decision A | Decision B | Full Delta A | Full Delta B | Full Delta Change | bg_far A | bg_far B | bg_far Change | bg_bottom A | bg_bottom B | bg_bottom Change |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["matched_cases"]:
        lines.append(
            "| `{target}` | `{decision_a}` | `{decision_b}` | {full_a} | {full_b} | {full_delta} | {bg_far_a} | {bg_far_b} | {bg_far_delta} | {bg_bottom_a} | {bg_bottom_b} | {bg_bottom_delta} |".format(
                target=row["target_camera"],
                decision_a=row["full_decision_a"],
                decision_b=row["full_decision_b"],
                full_a=fmt(row["full_depth_minus_point_mae_a"]),
                full_b=fmt(row["full_depth_minus_point_mae_b"]),
                full_delta=fmt(row["full_depth_minus_point_mae_delta"]),
                bg_far_a=fmt(row["bg_far_depth_minus_point_mae_a"]),
                bg_far_b=fmt(row["bg_far_depth_minus_point_mae_b"]),
                bg_far_delta=fmt(row["bg_far_depth_minus_point_mae_delta"]),
                bg_bottom_a=fmt(row["bg_bottom_band_depth_minus_point_mae_a"]),
                bg_bottom_b=fmt(row["bg_bottom_band_depth_minus_point_mae_b"]),
                bg_bottom_delta=fmt(row["bg_bottom_band_depth_minus_point_mae_delta"]),
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    payload_a = load_json(args.summary_a)
    payload_b = load_json(args.summary_b)
    payload = build_payload(payload_a, payload_b, args.label_a, args.label_b, args.title)
    write_json(args.output_dir / "summary.json", payload)
    write_markdown(args.output_dir / "summary.md", payload)
    print(f"[done] Wrote {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
