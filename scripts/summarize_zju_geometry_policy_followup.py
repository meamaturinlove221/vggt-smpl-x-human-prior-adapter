import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize a matched subset comparison between two ZJU geometry sweep summaries."
    )
    parser.add_argument("--round_a", type=str, required=True, help="Baseline sweep summary.json")
    parser.add_argument("--round_b", type=str, required=True, help="Candidate sweep summary.json")
    parser.add_argument("--label_a", type=str, default="round_a")
    parser.add_argument("--label_b", type=str, default="round_b")
    parser.add_argument("--view_profile", type=str, required=True)
    parser.add_argument("--frame_id", type=int, required=True)
    parser.add_argument("--target_cameras", type=str, required=True, help="Comma-separated target cameras")
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def load_summary(path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    keyed = {}
    for row in payload.get("rows", []):
        key = (str(row["view_profile"]), int(row["frame_id"]), str(row["target_camera"]))
        keyed[key] = row
    return {
        "path": str(Path(path).resolve()),
        "manifest": payload.get("manifest", {}),
        "keyed": keyed,
    }


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def geometry_gain(row: dict) -> float:
    return float(row["point_mae"]) - float(row["depth_mae"])


def coverage_gain(row: dict) -> float:
    return float(row["depth_cov"]) - float(row["point_cov"])


def mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def count(rows: list[dict], key: str, value: str) -> int:
    return sum(1 for row in rows if row.get(key) == value)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict) -> None:
    rows = payload["rows"]
    aggregate = payload["aggregate"]
    lines = [
        "# ZJU Sparse Policy Follow-Up",
        "",
        f"- {payload['label_a']}: `{payload['round_a']}`",
        f"- {payload['label_b']}: `{payload['round_b']}`",
        f"- view_profile: `{payload['view_profile']}`",
        f"- frame_id: `{payload['frame_id']}`",
        f"- targets: `{','.join(payload['target_cameras'])}`",
        f"- source_policy {payload['label_a']}: `{payload['source_policy_a']}`",
        f"- source_policy {payload['label_b']}: `{payload['source_policy_b']}`",
        "",
        "## Aggregate",
        "",
        f"- cases: `{aggregate['cases']}`",
        f"- avg geometry gain {payload['label_a']}: `{aggregate['avg_geometry_gain_a']:.6f}`",
        f"- avg geometry gain {payload['label_b']}: `{aggregate['avg_geometry_gain_b']:.6f}`",
        f"- avg geometry gain delta ({payload['label_b']} - {payload['label_a']}): `{aggregate['avg_geometry_gain_delta']:.6f}`",
        f"- avg coverage gain {payload['label_a']}: `{aggregate['avg_coverage_gain_a']:.6f}`",
        f"- avg coverage gain {payload['label_b']}: `{aggregate['avg_coverage_gain_b']:.6f}`",
        f"- avg coverage gain delta ({payload['label_b']} - {payload['label_a']}): `{aggregate['avg_coverage_gain_delta']:.6f}`",
        f"- depth decisions {payload['label_a']}: `{aggregate['depth_decisions_a']}`",
        f"- depth decisions {payload['label_b']}: `{aggregate['depth_decisions_b']}`",
        f"- improved geometry cases: `{aggregate['improved_geometry_cases']}`",
        f"- worsened geometry cases: `{aggregate['worsened_geometry_cases']}`",
        "",
        "## Cases",
        "",
        "| Target | Decision A | Decision B | Gain A | Gain B | Delta | Cov A | Cov B | Cov Delta |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {target_camera} | {decision_a} | {decision_b} | {geometry_gain_a:.4f} | {geometry_gain_b:.4f} | {geometry_gain_delta:.4f} | {coverage_gain_a:.4f} | {coverage_gain_b:.4f} | {coverage_gain_delta:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- This subset keeps `view_profile={payload['view_profile']}` and `frame_id={payload['frame_id']}` fixed.",
            f"- The changed variable is the sparse source policy between `{payload['source_policy_a']}` and `{payload['source_policy_b']}`.",
            "- This is a current-current matched comparison; it is not yet a matched-source legacy-native comparison.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    summary_a = load_summary(args.round_a)
    summary_b = load_summary(args.round_b)
    targets = parse_csv(args.target_cameras)
    rows = []

    for target_camera in targets:
        key = (args.view_profile, int(args.frame_id), target_camera)
        if key not in summary_a["keyed"]:
            raise KeyError(f"Missing case in round_a: {key}")
        if key not in summary_b["keyed"]:
            raise KeyError(f"Missing case in round_b: {key}")
        row_a = summary_a["keyed"][key]
        row_b = summary_b["keyed"][key]
        rows.append(
            {
                "target_camera": target_camera,
                "decision_a": str(row_a["decision"]),
                "decision_b": str(row_b["decision"]),
                "geometry_gain_a": geometry_gain(row_a),
                "geometry_gain_b": geometry_gain(row_b),
                "geometry_gain_delta": geometry_gain(row_b) - geometry_gain(row_a),
                "coverage_gain_a": coverage_gain(row_a),
                "coverage_gain_b": coverage_gain(row_b),
                "coverage_gain_delta": coverage_gain(row_b) - coverage_gain(row_a),
                "point_mae_a": float(row_a["point_mae"]),
                "depth_mae_a": float(row_a["depth_mae"]),
                "point_mae_b": float(row_b["point_mae"]),
                "depth_mae_b": float(row_b["depth_mae"]),
                "point_cov_a": float(row_a["point_cov"]),
                "depth_cov_a": float(row_a["depth_cov"]),
                "point_cov_b": float(row_b["point_cov"]),
                "depth_cov_b": float(row_b["depth_cov"]),
                "case_dir_a": str(row_a.get("case_dir", "")),
                "case_dir_b": str(row_b.get("case_dir", "")),
            }
        )

    aggregate = {
        "cases": len(rows),
        "avg_geometry_gain_a": mean(row["geometry_gain_a"] for row in rows),
        "avg_geometry_gain_b": mean(row["geometry_gain_b"] for row in rows),
        "avg_geometry_gain_delta": mean(row["geometry_gain_delta"] for row in rows),
        "avg_coverage_gain_a": mean(row["coverage_gain_a"] for row in rows),
        "avg_coverage_gain_b": mean(row["coverage_gain_b"] for row in rows),
        "avg_coverage_gain_delta": mean(row["coverage_gain_delta"] for row in rows),
        "depth_decisions_a": count(rows, "decision_a", "depth_unproject"),
        "depth_decisions_b": count(rows, "decision_b", "depth_unproject"),
        "improved_geometry_cases": sum(1 for row in rows if row["geometry_gain_delta"] > 0.0),
        "worsened_geometry_cases": sum(1 for row in rows if row["geometry_gain_delta"] < 0.0),
    }

    payload = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "round_a": summary_a["path"],
        "round_b": summary_b["path"],
        "source_policy_a": str(summary_a["manifest"].get("source_policy", "n/a")),
        "source_policy_b": str(summary_b["manifest"].get("source_policy", "n/a")),
        "view_profile": args.view_profile,
        "frame_id": int(args.frame_id),
        "target_cameras": targets,
        "aggregate": aggregate,
        "rows": rows,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", payload)
    write_markdown(output_dir / "summary.md", payload)


if __name__ == "__main__":
    main()
