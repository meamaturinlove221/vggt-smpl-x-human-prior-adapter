import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a refined hard-tail manifest that caps the dominant contract slice and dominant "
            "contiguous segment, then refills from high-score near-tail reserve rows."
        )
    )
    parser.add_argument("--residual-jsonl", required=True)
    parser.add_argument("--official-manifest-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--dominant-contract-cap-share", type=float, default=0.47)
    parser.add_argument("--dominant-segment-cap-share", type=float, default=0.21)
    parser.add_argument("--segment-gap-threshold", type=int, default=20)
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


def contract_key(row: dict) -> tuple[str, tuple[str, ...]]:
    return str(row.get("promoted_anchor_camera") or ""), tuple(
        str(item) for item in row.get("selected_source_only_camera_names", [])
    )


def build_seed_segments(selected_rows: list[dict], gap_threshold: int) -> dict[tuple[str, tuple[str, ...]], list[tuple[int, int]]]:
    grouped = defaultdict(list)
    for row in selected_rows:
        grouped[contract_key(row)].append(row)
    seed_segments = {}
    for key, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["frame_id"]))
        current = [ordered[0]]
        segments = []
        for row in ordered[1:]:
            if int(row["frame_id"]) - int(current[-1]["frame_id"]) <= gap_threshold:
                current.append(row)
            else:
                segments.append(current)
                current = [row]
        segments.append(current)
        seed_segments[key] = [
            (min(int(item["frame_id"]) for item in segment), max(int(item["frame_id"]) for item in segment))
            for segment in segments
        ]
    return seed_segments


def assign_seed_segment(seed_segments: list[tuple[int, int]], frame_id: int) -> int:
    if len(seed_segments) == 1:
        return 0
    boundaries = []
    for index in range(len(seed_segments) - 1):
        left_end = seed_segments[index][1]
        right_start = seed_segments[index + 1][0]
        boundaries.append((left_end + right_start) / 2.0)
    if frame_id <= boundaries[0]:
        return 0
    for index in range(1, len(boundaries)):
        if frame_id <= boundaries[index]:
            return index
    return len(seed_segments) - 1


def summarize_manifest(rows: list[dict], seed_segments: dict[tuple[str, tuple[str, ...]], list[tuple[int, int]]]) -> dict:
    contract_counts = Counter(contract_key(row) for row in rows)
    segment_counts = Counter()
    for row in rows:
        key = contract_key(row)
        segment_index = assign_seed_segment(seed_segments[key], int(row["frame_id"]))
        segment_counts[(key, segment_index)] += 1
    dominant_contract, dominant_contract_count = contract_counts.most_common(1)[0]
    dominant_segment, dominant_segment_count = segment_counts.most_common(1)[0]
    return {
        "contract_counts": [
            {
                "promoted_anchor_camera": key[0],
                "selected_source_only_set": list(key[1]),
                "count": int(count),
            }
            for key, count in contract_counts.most_common()
        ],
        "segment_counts": [
            {
                "promoted_anchor_camera": key[0][0],
                "selected_source_only_set": list(key[0][1]),
                "segment_index": int(key[1]),
                "segment_frame_start": int(seed_segments[key[0]][key[1]][0]),
                "segment_frame_end": int(seed_segments[key[0]][key[1]][1]),
                "count": int(count),
            }
            for key, count in segment_counts.most_common()
        ],
        "dominant_contract": {
            "promoted_anchor_camera": dominant_contract[0],
            "selected_source_only_set": list(dominant_contract[1]),
            "count": int(dominant_contract_count),
        },
        "dominant_segment": {
            "promoted_anchor_camera": dominant_segment[0][0],
            "selected_source_only_set": list(dominant_segment[0][1]),
            "segment_index": int(dominant_segment[1]),
            "segment_frame_start": int(seed_segments[dominant_segment[0]][dominant_segment[1]][0]),
            "segment_frame_end": int(seed_segments[dominant_segment[0]][dominant_segment[1]][1]),
            "count": int(dominant_segment_count),
        },
    }


def main():
    args = parse_args()
    residual_path = Path(args.residual_jsonl)
    official_manifest_path = Path(args.official_manifest_json)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    all_rows = load_jsonl(residual_path)
    official_manifest = load_json(official_manifest_path)
    selected_rows = list(official_manifest.get("entries", []))
    if not selected_rows:
        raise SystemExit(f"official manifest contains no entries: {official_manifest_path}")

    seed_segments = build_seed_segments(selected_rows, gap_threshold=args.segment_gap_threshold)
    official_summary = summarize_manifest(selected_rows, seed_segments)
    dominant_contract_key = (
        official_summary["dominant_contract"]["promoted_anchor_camera"],
        tuple(official_summary["dominant_contract"]["selected_source_only_set"]),
    )
    dominant_segment_index = int(official_summary["dominant_segment"]["segment_index"])

    target_count = int(len(selected_rows))
    dominant_contract_cap = max(1, int(round(target_count * float(args.dominant_contract_cap_share))))
    dominant_segment_cap = max(1, int(round(target_count * float(args.dominant_segment_cap_share))))

    chosen = {row_key(row): row for row in selected_rows}
    dominant_segment_rows = sorted(
        [
            row
            for row in chosen.values()
            if contract_key(row) == dominant_contract_key
            and assign_seed_segment(seed_segments[dominant_contract_key], int(row["frame_id"])) == dominant_segment_index
        ],
        key=lambda row: (float(row["joint_depth_geom_tail_score"]), int(row["frame_id"])),
    )
    for row in dominant_segment_rows[: max(0, len(dominant_segment_rows) - dominant_segment_cap)]:
        chosen.pop(row_key(row), None)

    dominant_contract_rows = sorted(
        [row for row in chosen.values() if contract_key(row) == dominant_contract_key],
        key=lambda row: (float(row["joint_depth_geom_tail_score"]), int(row["frame_id"])),
    )
    for row in dominant_contract_rows[: max(0, len(dominant_contract_rows) - dominant_contract_cap)]:
        chosen.pop(row_key(row), None)

    ranked_all_rows = sorted(
        all_rows,
        key=lambda row: (-float(row["joint_depth_geom_tail_score"]), str(row["seq_name"]), int(row["frame_id"])),
    )
    while len(chosen) < target_count:
        current_contract_counts = Counter(contract_key(row) for row in chosen.values())
        current_dominant_segment_count = sum(
            1
            for row in chosen.values()
            if contract_key(row) == dominant_contract_key
            and assign_seed_segment(seed_segments[dominant_contract_key], int(row["frame_id"])) == dominant_segment_index
        )
        added = False
        for row in ranked_all_rows:
            key = row_key(row)
            if key in chosen:
                continue
            contract = contract_key(row)
            if contract == dominant_contract_key and current_contract_counts[contract] >= dominant_contract_cap:
                continue
            if (
                contract == dominant_contract_key
                and assign_seed_segment(seed_segments[dominant_contract_key], int(row["frame_id"])) == dominant_segment_index
                and current_dominant_segment_count >= dominant_segment_cap
            ):
                continue
            chosen[key] = row
            added = True
            break
        if not added:
            break

    refined_rows = sorted(
        chosen.values(),
        key=lambda row: (-float(row["joint_depth_geom_tail_score"]), str(row["seq_name"]), int(row["frame_id"])),
    )
    refined_summary = summarize_manifest(refined_rows, seed_segments)
    official_keys = {row_key(row) for row in selected_rows}
    added_rows = [row for row in refined_rows if row_key(row) not in official_keys]

    payload = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "manifest_kind": "contract_segment_stratified_hardtail_bucket",
        "family": "hardtail_bucket_granularity_refinement",
        "source_manifest_path": str(official_manifest_path.resolve()),
        "source_manifest_status": official_manifest.get("status", ""),
        "target_entry_count": target_count,
        "dominant_contract_cap": dominant_contract_cap,
        "dominant_segment_cap": dominant_segment_cap,
        "segment_gap_threshold": int(args.segment_gap_threshold),
        "refinement_rule": {
            "mode": "cap_dominant_contract_and_segment_then_refill_from_near_tail_reserve",
            "dominant_contract_cap_share": float(args.dominant_contract_cap_share),
            "dominant_segment_cap_share": float(args.dominant_segment_cap_share),
        },
        "official_summary": official_summary,
        "refined_summary": refined_summary,
        "replacement_summary": {
            "replaced_entry_count": int(len(added_rows)),
            "replacement_contract_counts": [
                {
                    "promoted_anchor_camera": key[0],
                    "selected_source_only_set": list(key[1]),
                    "count": int(count),
                }
                for key, count in Counter(contract_key(row) for row in added_rows).most_common()
            ],
            "replacement_score_min": None
            if not added_rows
            else float(min(float(row["joint_depth_geom_tail_score"]) for row in added_rows)),
            "replacement_score_max": None
            if not added_rows
            else float(max(float(row["joint_depth_geom_tail_score"]) for row in added_rows)),
        },
        "entries": refined_rows,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Contract-Segment Stratified Hardtail Bucket",
        "",
        f"- target_entry_count: `{target_count}`",
        f"- dominant_contract_cap: `{dominant_contract_cap}`",
        f"- dominant_segment_cap: `{dominant_segment_cap}`",
        f"- replaced_entry_count: `{payload['replacement_summary']['replaced_entry_count']}`",
        f"- replacement_score_min: `{payload['replacement_summary']['replacement_score_min']}`",
        f"- replacement_score_max: `{payload['replacement_summary']['replacement_score_max']}`",
        "",
        "## Official Summary",
        "",
        f"- dominant_contract: `{official_summary['dominant_contract']}`",
        f"- dominant_segment: `{official_summary['dominant_segment']}`",
        "",
        "## Refined Summary",
        "",
        f"- dominant_contract: `{refined_summary['dominant_contract']}`",
        f"- dominant_segment: `{refined_summary['dominant_segment']}`",
        "",
        "## Replacement Contracts",
        "",
    ]
    for item in payload["replacement_summary"]["replacement_contract_counts"]:
        lines.append(
            "- anchor=`{anchor}` source_only=`{source_only}` count=`{count}`".format(
                anchor=item["promoted_anchor_camera"],
                source_only=item["selected_source_only_set"],
                count=item["count"],
            )
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(output_json)
    print(output_md)


if __name__ == "__main__":
    main()
