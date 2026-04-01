import argparse
import json
import math
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Freeze the official promoted hard-tail manifest from per-frame residual rows."
    )
    parser.add_argument("--frame-rows-jsonl", required=True)
    parser.add_argument("--hardcase-definition-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--promoted-config", required=True)
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


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_percentiles(rows, key: str):
    valid = [
        (index, float(row[key]))
        for index, row in enumerate(rows)
        if row.get(key) is not None
    ]
    if not valid:
        return {}
    valid.sort(key=lambda item: (item[1], item[0]))
    denom = max(len(valid) - 1, 1)
    percentiles = {}
    for rank, (index, _value) in enumerate(valid):
        if len(valid) == 1:
            percentiles[index] = 1.0
        else:
            percentiles[index] = float(rank / denom)
    return percentiles


def top_preview(entries, limit=12):
    return entries[:limit]


def main():
    args = parse_args()
    frame_rows_path = Path(args.frame_rows_jsonl)
    definition_path = Path(args.hardcase_definition_json)
    output_jsonl = Path(args.output_jsonl)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    for path in (output_jsonl, output_json, output_md):
        path.parent.mkdir(parents=True, exist_ok=True)

    definition = load_json(definition_path)
    rows = load_jsonl(frame_rows_path)
    rows.sort(key=lambda row: (str(row.get("seq_name") or ""), int(row.get("frame_id") or -1)))

    eligible_rows = [
        row
        for row in rows
        if row.get("eligible_primary_metrics")
        and row.get("seq_name")
        and row.get("frame_id") is not None
    ]

    conf_pct = compute_percentiles(eligible_rows, "conf_depth_mean")
    reg_pct = compute_percentiles(eligible_rows, "reg_depth_mean")
    unproject_pct = compute_percentiles(eligible_rows, "unproject_geometry_mean")

    score_weights = definition["bucket_definition"]["hard_case_metric"]["formula"]
    conf_weight = float(score_weights["conf_depth_percentile_weight"])
    reg_weight = float(score_weights["reg_depth_percentile_weight"])
    unproject_weight = float(score_weights["unproject_geometry_percentile_weight"])

    canonical_rows = []
    for index, row in enumerate(eligible_rows):
        canonical_row = {
            "seq_name": str(row["seq_name"]),
            "frame_id": int(row["frame_id"]),
            "sample_index": int(row["sample_index"]),
            "joint_depth_geom_tail_score": float(
                conf_weight * conf_pct[index]
                + reg_weight * reg_pct[index]
                + unproject_weight * unproject_pct[index]
            ),
            "conf_depth_mean": float(row["conf_depth_mean"]),
            "reg_depth_mean": float(row["reg_depth_mean"]),
            "unproject_geometry_mean": float(row["unproject_geometry_mean"]),
            "conf_depth_percentile": float(conf_pct[index]),
            "reg_depth_percentile": float(reg_pct[index]),
            "unproject_geometry_percentile": float(unproject_pct[index]),
            "promoted_anchor_camera": row.get("promoted_anchor_camera"),
            "selected_supervised_camera_names": list(row.get("selected_supervised_camera_names", [])),
            "selected_source_only_camera_names": list(row.get("selected_source_only_camera_names", [])),
            "candidate_supervised_camera_names": list(row.get("candidate_supervised_camera_names", [])),
            "dropped_supervised_camera_names": list(row.get("dropped_supervised_camera_names", [])),
            "selected_camera_names": list(row.get("selected_camera_names", [])),
            "supervised_view_count": int(row.get("supervised_view_count", 0)),
            "valid_pixels": int(row.get("valid_pixels", 0)),
            "conf_valid_pixels": int(row.get("conf_valid_pixels", 0)),
            "sample_seq_name": row.get("sample_seq_name"),
            "frame_key": row.get("frame_key"),
        }
        canonical_rows.append(canonical_row)

    canonical_rows.sort(
        key=lambda row: (
            -float(row["joint_depth_geom_tail_score"]),
            -float(row["conf_depth_percentile"]),
            -float(row["reg_depth_percentile"]),
            -float(row["unproject_geometry_percentile"]),
            str(row["seq_name"]),
            int(row["frame_id"]),
        )
    )

    threshold_rule = definition["bucket_definition"]["threshold_rule"]
    worst_percent = float(threshold_rule["worst_percent"])
    hard_tail_count = int(math.ceil(len(canonical_rows) * worst_percent / 100.0))
    hard_tail_count = max(hard_tail_count, 1) if canonical_rows else 0
    selected_entries = canonical_rows[:hard_tail_count]

    payload = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "family": definition.get("family"),
        "status": "frozen_from_promoted_residuals",
        "manifest_kind": "official_promoted_hard_tail_manifest",
        "promoted_local_lead_config": args.promoted_config,
        "definition_ref": str(definition_path),
        "train_split_only": True,
        "entries_frozen": True,
        "eligibility_split": threshold_rule["eligibility_split"],
        "tail_metric_name": definition["bucket_definition"]["hard_case_metric"]["name"],
        "tail_metric_formula": definition["bucket_definition"]["hard_case_metric"]["formula"],
        "threshold_rule": threshold_rule,
        "eligible_entry_count": int(len(canonical_rows)),
        "hard_tail_entry_count": int(hard_tail_count),
        "hard_tail_entry_share": 0.0 if len(canonical_rows) == 0 else float(hard_tail_count / len(canonical_rows)),
        "entries": selected_entries,
        "top_tail_preview": top_preview(selected_entries),
    }

    write_jsonl(output_jsonl, canonical_rows)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Promoted Hardcase Manifest",
        "",
        f"- promoted_local_lead_config: `{args.promoted_config}`",
        f"- tail_metric_name: `{payload['tail_metric_name']}`",
        f"- threshold_rule: `{payload['threshold_rule']}`",
        f"- eligible_entry_count: `{payload['eligible_entry_count']}`",
        f"- hard_tail_entry_count: `{payload['hard_tail_entry_count']}`",
        f"- hard_tail_entry_share: `{payload['hard_tail_entry_share']:.4f}`",
        "",
        "## Top Tail Preview",
        "",
    ]
    for row in top_preview(selected_entries):
        lines.append(
            "- `{seq}` frame `{frame}` score={score:.4f} conf_pct={conf:.4f} reg_pct={reg:.4f} unproject_pct={unproj:.4f} anchor={anchor} supervised={supervised} source_only={source_only}".format(
                seq=row["seq_name"],
                frame=int(row["frame_id"]),
                score=float(row["joint_depth_geom_tail_score"]),
                conf=float(row["conf_depth_percentile"]),
                reg=float(row["reg_depth_percentile"]),
                unproj=float(row["unproject_geometry_percentile"]),
                anchor=row.get("promoted_anchor_camera"),
                supervised=row.get("selected_supervised_camera_names", []),
                source_only=row.get("selected_source_only_camera_names", []),
            )
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(output_jsonl)
    print(output_json)
    print(output_md)


if __name__ == "__main__":
    main()
