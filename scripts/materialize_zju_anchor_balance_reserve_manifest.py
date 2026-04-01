import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a counterbalance reserve manifest from refined-manifest-excluded near-tail rows, "
            "favoring non-dominant promoted anchor cohorts."
        )
    )
    parser.add_argument("--residual-jsonl", required=True)
    parser.add_argument("--exclude-manifest-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--target-count", type=int, default=85)
    parser.add_argument("--anchor-cap-b1", type=int, default=30)
    parser.add_argument("--anchor-cap-b9", type=int, default=30)
    parser.add_argument("--anchor-cap-b5", type=int, default=30)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def row_key(row: dict) -> tuple[str, int]:
    return str(row["seq_name"]), int(row["frame_id"])


def main():
    args = parse_args()
    residual_path = Path(args.residual_jsonl)
    exclude_manifest_path = Path(args.exclude_manifest_json)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    all_rows = load_jsonl(residual_path)
    exclude_manifest = load_json(exclude_manifest_path)
    excluded_keys = {row_key(row) for row in exclude_manifest.get("entries", [])}
    anchor_caps = {
        "Camera_B1": int(args.anchor_cap_b1),
        "Camera_B9": int(args.anchor_cap_b9),
        "Camera_B5": int(args.anchor_cap_b5),
    }
    ranked_rows = sorted(
        [
            row
            for row in all_rows
            if row_key(row) not in excluded_keys
            and str(row.get("promoted_anchor_camera") or "") in anchor_caps
        ],
        key=lambda row: (-float(row["joint_depth_geom_tail_score"]), str(row["seq_name"]), int(row["frame_id"])),
    )

    chosen = []
    anchor_counts = Counter()
    for row in ranked_rows:
        anchor = str(row.get("promoted_anchor_camera") or "")
        if anchor_counts[anchor] >= anchor_caps[anchor]:
            continue
        chosen.append(row)
        anchor_counts[anchor] += 1
        if len(chosen) >= int(args.target_count):
            break

    if len(chosen) != int(args.target_count):
        raise SystemExit(
            f"Unable to fill reserve manifest to target_count={args.target_count}; got {len(chosen)} entries."
        )

    source_set_counts = Counter(tuple(str(item) for item in row.get("selected_source_only_camera_names", [])) for row in chosen)
    payload = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "manifest_kind": "anchor_balance_reserve_manifest",
        "family": "tail_counterbalance_cohort_mixing",
        "source_residual_path": str(residual_path.resolve()),
        "excluded_manifest_path": str(exclude_manifest_path.resolve()),
        "target_count": int(args.target_count),
        "anchor_caps": anchor_caps,
        "anchor_counts": dict(anchor_counts),
        "source_only_set_counts": [
            {
                "selected_source_only_set": list(source_set),
                "count": int(count),
            }
            for source_set, count in source_set_counts.most_common()
        ],
        "score_min": float(min(float(row["joint_depth_geom_tail_score"]) for row in chosen)),
        "score_max": float(max(float(row["joint_depth_geom_tail_score"]) for row in chosen)),
        "score_mean": float(
            sum(float(row["joint_depth_geom_tail_score"]) for row in chosen) / float(len(chosen))
        ),
        "entries": chosen,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Anchor Balance Reserve Manifest",
        "",
        f"- target_count: `{payload['target_count']}`",
        f"- anchor_caps: `{payload['anchor_caps']}`",
        f"- anchor_counts: `{payload['anchor_counts']}`",
        f"- score_min: `{payload['score_min']}`",
        f"- score_max: `{payload['score_max']}`",
        f"- score_mean: `{payload['score_mean']}`",
        "",
        "## Source-Only Sets",
        "",
    ]
    for item in payload["source_only_set_counts"]:
        lines.append(
            "- source_only=`{source_only}` count=`{count}`".format(
                source_only=item["selected_source_only_set"],
                count=item["count"],
            )
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(output_json)
    print(output_md)


if __name__ == "__main__":
    main()
