import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two ZJU VGGT-geom probe directories.")
    parser.add_argument("--probe-a", required=True, help="Probe directory for candidate/promoted policy.")
    parser.add_argument("--probe-b", required=True, help="Probe directory for reference/previous policy.")
    parser.add_argument("--label-a", default="probe_a")
    parser.add_argument("--label-b", default="probe_b")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_probe_dir(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_dir():
        raise FileNotFoundError(f"Probe directory not found: {path}")
    return path


def jaccard(left, right):
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 1.0
    return float(len(left_set & right_set) / len(union))


def sorted_difference(left, right):
    return sorted(set(str(item) for item in left) - set(str(item) for item in right))


def parse_sample_identity(sample_seq_name):
    text = str(sample_seq_name)
    match = re.match(r"^(?:zju_)?(?P<seq_name>.+)_frame_(?P<frame_id>\d+)$", text)
    if not match:
        return {"sample_seq_name": text, "seq_name": "", "frame_id": None}
    return {
        "sample_seq_name": text,
        "seq_name": str(match.group("seq_name")),
        "frame_id": int(match.group("frame_id")),
    }


def histogram_from_payloads(payloads, key):
    counts = Counter()
    for payload in payloads:
        for item in payload.get(key, []):
            counts[str(item)] += 1
    return dict(sorted(counts.items()))


def role_for_camera(payload, camera_name):
    camera_name = str(camera_name)
    if camera_name in payload.get("supervised_camera_names", []):
        return "supervised"
    if camera_name in payload.get("source_only_camera_names", []):
        return "source_only"
    if camera_name in payload.get("camera_names", []):
        return "selected_other"
    return "absent"


def build_slot_transition_summary(payloads_a, payloads_b):
    max_slots = 0
    for payload in payloads_a + payloads_b:
        max_slots = max(max_slots, len(payload.get("camera_names", [])))

    slot_transition_histograms = {}
    slot_match_rates = {}
    for slot_idx in range(max_slots):
        counts = Counter()
        matches = 0
        total = 0
        for item_a, item_b in zip(payloads_a, payloads_b):
            cams_a = list(item_a.get("camera_names", []))
            cams_b = list(item_b.get("camera_names", []))
            cam_a = cams_a[slot_idx] if slot_idx < len(cams_a) else "<missing>"
            cam_b = cams_b[slot_idx] if slot_idx < len(cams_b) else "<missing>"
            counts[f"{cam_a} -> {cam_b}"] += 1
            matches += int(cam_a == cam_b)
            total += 1
        slot_key = f"slot_{slot_idx}"
        slot_transition_histograms[slot_key] = dict(sorted(counts.items()))
        slot_match_rates[slot_key] = float(matches / max(total, 1))

    return {
        "slot_match_rates": slot_match_rates,
        "slot_transition_histograms": slot_transition_histograms,
    }


def build_role_transition_summary(payloads_a, payloads_b):
    counts = Counter()
    for item_a, item_b in zip(payloads_a, payloads_b):
        camera_union = sorted(set(item_a.get("camera_names", [])) | set(item_b.get("camera_names", [])))
        for camera_name in camera_union:
            role_a = role_for_camera(item_a, camera_name)
            role_b = role_for_camera(item_b, camera_name)
            counts[f"{role_a} -> {role_b}"] += 1
    return dict(sorted(counts.items()))


def build_per_sample_diffs(payloads_a, payloads_b):
    if len(payloads_a) != len(payloads_b):
        raise ValueError(
            f"Per-sample probe lengths differ: {len(payloads_a)} vs {len(payloads_b)}. Use matching sample baskets."
        )

    rows = []
    for item_a, item_b in zip(payloads_a, payloads_b):
        if int(item_a["probe_sample_index"]) != int(item_b["probe_sample_index"]):
            raise ValueError(
                f"Probe sample index mismatch: {item_a['probe_sample_index']} vs {item_b['probe_sample_index']}"
            )
        identity = parse_sample_identity(item_a["sample_seq_name"])
        rows.append(
            {
                "probe_sample_index": int(item_a["probe_sample_index"]),
                "sample_seq_name": identity["sample_seq_name"],
                "seq_name": identity["seq_name"],
                "frame_id": identity["frame_id"],
                "anchor_camera_a": item_a.get("selection_anchor_camera"),
                "anchor_camera_b": item_b.get("selection_anchor_camera"),
                "camera_names_a": list(item_a.get("camera_names", [])),
                "camera_names_b": list(item_b.get("camera_names", [])),
                "camera_names_only_in_a": sorted_difference(item_a.get("camera_names", []), item_b.get("camera_names", [])),
                "camera_names_only_in_b": sorted_difference(item_b.get("camera_names", []), item_a.get("camera_names", [])),
                "camera_jaccard": jaccard(item_a.get("camera_names", []), item_b.get("camera_names", [])),
                "supervised_camera_names_a": list(item_a.get("supervised_camera_names", [])),
                "supervised_camera_names_b": list(item_b.get("supervised_camera_names", [])),
                "supervised_camera_names_only_in_a": sorted_difference(
                    item_a.get("supervised_camera_names", []), item_b.get("supervised_camera_names", [])
                ),
                "supervised_camera_names_only_in_b": sorted_difference(
                    item_b.get("supervised_camera_names", []), item_a.get("supervised_camera_names", [])
                ),
                "supervised_jaccard": jaccard(
                    item_a.get("supervised_camera_names", []), item_b.get("supervised_camera_names", [])
                ),
                "source_only_camera_names_a": list(item_a.get("source_only_camera_names", [])),
                "source_only_camera_names_b": list(item_b.get("source_only_camera_names", [])),
                "source_only_camera_names_only_in_a": sorted_difference(
                    item_a.get("source_only_camera_names", []), item_b.get("source_only_camera_names", [])
                ),
                "source_only_camera_names_only_in_b": sorted_difference(
                    item_b.get("source_only_camera_names", []), item_a.get("source_only_camera_names", [])
                ),
                "source_only_jaccard": jaccard(
                    item_a.get("source_only_camera_names", []), item_b.get("source_only_camera_names", [])
                ),
                "slot_3_camera_a": (list(item_a.get("camera_names", [])) + ["<missing>"] * 4)[3],
                "slot_3_camera_b": (list(item_b.get("camera_names", [])) + ["<missing>"] * 4)[3],
                "valid_ratio_delta_a_minus_b": float(item_a["valid_ratio"]) - float(item_b["valid_ratio"]),
                "valid_points_delta_a_minus_b": int(item_a["valid_points"]) - int(item_b["valid_points"]),
                "pointcloud_extent_delta_a_minus_b": [
                    float(a) - float(b)
                    for a, b in zip(item_a["pointcloud_stats"]["extent"], item_b["pointcloud_stats"]["extent"])
                ],
                "pointcloud_radius_p95_delta_a_minus_b": float(
                    item_a["pointcloud_stats"]["radius_percentiles"]["p95"]
                )
                - float(item_b["pointcloud_stats"]["radius_percentiles"]["p95"]),
            }
        )
    return rows


def build_summary(args, aggregate_a, aggregate_b, payloads_a, payloads_b):
    per_sample = build_per_sample_diffs(payloads_a, payloads_b)
    camera_jaccards = np.asarray([row["camera_jaccard"] for row in per_sample], dtype=np.float64)
    supervised_jaccards = np.asarray([row["supervised_jaccard"] for row in per_sample], dtype=np.float64)
    source_only_jaccards = np.asarray([row["source_only_jaccard"] for row in per_sample], dtype=np.float64)
    slot_transition_summary = build_slot_transition_summary(payloads_a, payloads_b)
    role_transition_histogram = build_role_transition_summary(payloads_a, payloads_b)

    return {
        "comparison_type": "zju_vggt_geom_probe_selection_contract_diff",
        "probe_a_label": args.label_a,
        "probe_b_label": args.label_b,
        "probe_a_dir": str(normalize_probe_dir(args.probe_a)).replace("\\", "/"),
        "probe_b_dir": str(normalize_probe_dir(args.probe_b)).replace("\\", "/"),
        "slot_transition_direction": "probe_a_to_probe_b",
        "role_transition_direction": "probe_a_to_probe_b",
        "aggregate_samples": int(aggregate_a["aggregate_samples"]),
        "aggregate_stride": int(aggregate_a["aggregate_stride"]),
        "sample_indices": list(aggregate_a["sample_indices"]),
        "probe_a": {
            "config_path": aggregate_a["config_path"].replace("\\", "/"),
            "source_policy": aggregate_a["source_policy"],
            "valid_ratio_mean": float(aggregate_a["valid_ratio"]["mean"]),
            "valid_points_mean": float(aggregate_a["valid_points"]["mean"]),
            "supervised_view_count_mean": float(aggregate_a["supervised_view_count"]["mean"]),
            "source_only_view_count_mean": float(aggregate_a["source_only_view_count"]["mean"]),
            "pointcloud_extent_mean": list(aggregate_a["pointcloud_extent"]["mean"]),
            "pointcloud_radius_p95_mean": float(aggregate_a["pointcloud_radius_p95"]["mean"]),
            "anchor_histogram": dict(aggregate_a["selection_anchor_camera_histogram"]),
            "supervised_camera_histogram": histogram_from_payloads(payloads_a, "supervised_camera_names"),
            "source_only_camera_histogram": histogram_from_payloads(payloads_a, "source_only_camera_names"),
        },
        "probe_b": {
            "config_path": aggregate_b["config_path"].replace("\\", "/"),
            "source_policy": aggregate_b["source_policy"],
            "valid_ratio_mean": float(aggregate_b["valid_ratio"]["mean"]),
            "valid_points_mean": float(aggregate_b["valid_points"]["mean"]),
            "supervised_view_count_mean": float(aggregate_b["supervised_view_count"]["mean"]),
            "source_only_view_count_mean": float(aggregate_b["source_only_view_count"]["mean"]),
            "pointcloud_extent_mean": list(aggregate_b["pointcloud_extent"]["mean"]),
            "pointcloud_radius_p95_mean": float(aggregate_b["pointcloud_radius_p95"]["mean"]),
            "anchor_histogram": dict(aggregate_b["selection_anchor_camera_histogram"]),
            "supervised_camera_histogram": histogram_from_payloads(payloads_b, "supervised_camera_names"),
            "source_only_camera_histogram": histogram_from_payloads(payloads_b, "source_only_camera_names"),
        },
        "delta_a_minus_b": {
            "valid_ratio_mean": float(aggregate_a["valid_ratio"]["mean"]) - float(aggregate_b["valid_ratio"]["mean"]),
            "valid_points_mean": float(aggregate_a["valid_points"]["mean"])
            - float(aggregate_b["valid_points"]["mean"]),
            "supervised_view_count_mean": float(aggregate_a["supervised_view_count"]["mean"])
            - float(aggregate_b["supervised_view_count"]["mean"]),
            "source_only_view_count_mean": float(aggregate_a["source_only_view_count"]["mean"])
            - float(aggregate_b["source_only_view_count"]["mean"]),
            "pointcloud_extent_mean": [
                float(a) - float(b)
                for a, b in zip(aggregate_a["pointcloud_extent"]["mean"], aggregate_b["pointcloud_extent"]["mean"])
            ],
            "pointcloud_radius_p95_mean": float(aggregate_a["pointcloud_radius_p95"]["mean"])
            - float(aggregate_b["pointcloud_radius_p95"]["mean"]),
            "camera_jaccard_mean": float(camera_jaccards.mean()),
            "supervised_jaccard_mean": float(supervised_jaccards.mean()),
            "source_only_jaccard_mean": float(source_only_jaccards.mean()),
            "anchor_match_rate": float(slot_transition_summary["slot_match_rates"].get("slot_0", 0.0)),
        },
        "slot_transition_summary": slot_transition_summary,
        "role_transition_histogram": role_transition_histogram,
        "per_sample_diffs": per_sample,
        "interpretation": [
            "Use this report as a selection-contract diff, not a direct long-gate metric proxy.",
            "High anchor-histogram overlap with lower camera/source-only overlap means the anchor stays stable while auxiliary source-slot allocation changes.",
            "Look at the slot transition histograms to see whether the policy change mostly swaps a late source slot rather than changing the anchor slot.",
            "Look at the role transition histogram to see whether cameras are being demoted from supervised to absent or re-routed into source-only roles.",
            "If probe_a is tighter in pointcloud extent yet still improves training, that indicates the win is likely due to camera-role assignment quality rather than broader pseudo-point-cloud coverage.",
        ],
    }


def write_markdown(path: Path, summary: dict):
    delta = summary["delta_a_minus_b"]
    lines = [
        "# ZJU VGGT-Geom Probe Selection Contract Diff",
        "",
        f"- probe_a_label: `{summary['probe_a_label']}`",
        f"- probe_b_label: `{summary['probe_b_label']}`",
        f"- slot_transition_direction: `{summary['slot_transition_direction']}`",
        f"- role_transition_direction: `{summary['role_transition_direction']}`",
        f"- aggregate_samples: `{summary['aggregate_samples']}`",
        f"- aggregate_stride: `{summary['aggregate_stride']}`",
        f"- sample_indices: `{summary['sample_indices']}`",
        "",
        "## Aggregate Delta",
        "",
        f"- delta valid_ratio_mean: `{delta['valid_ratio_mean']:.6f}`",
        f"- delta valid_points_mean: `{delta['valid_points_mean']:.2f}`",
        f"- delta supervised_view_count_mean: `{delta['supervised_view_count_mean']:.2f}`",
        f"- delta source_only_view_count_mean: `{delta['source_only_view_count_mean']:.2f}`",
        f"- delta pointcloud_extent_mean: `{[round(float(x), 6) for x in delta['pointcloud_extent_mean']]}`",
        f"- delta pointcloud_radius_p95_mean: `{delta['pointcloud_radius_p95_mean']:.6f}`",
        f"- camera_jaccard_mean: `{delta['camera_jaccard_mean']:.6f}`",
        f"- supervised_jaccard_mean: `{delta['supervised_jaccard_mean']:.6f}`",
        f"- source_only_jaccard_mean: `{delta['source_only_jaccard_mean']:.6f}`",
        f"- anchor_match_rate: `{delta['anchor_match_rate']:.6f}`",
        "",
        "## Histograms",
        "",
        f"- probe_a anchor_histogram: `{summary['probe_a']['anchor_histogram']}`",
        f"- probe_b anchor_histogram: `{summary['probe_b']['anchor_histogram']}`",
        f"- probe_a supervised_camera_histogram: `{summary['probe_a']['supervised_camera_histogram']}`",
        f"- probe_b supervised_camera_histogram: `{summary['probe_b']['supervised_camera_histogram']}`",
        f"- probe_a source_only_camera_histogram: `{summary['probe_a']['source_only_camera_histogram']}`",
        f"- probe_b source_only_camera_histogram: `{summary['probe_b']['source_only_camera_histogram']}`",
        f"- role_transition_histogram: `{summary['role_transition_histogram']}`",
        "",
        "## Slot Alignment",
        "",
        f"- slot_match_rates: `{summary['slot_transition_summary']['slot_match_rates']}`",
        f"- slot_0_transitions: `{summary['slot_transition_summary']['slot_transition_histograms'].get('slot_0', {})}`",
        f"- slot_1_transitions: `{summary['slot_transition_summary']['slot_transition_histograms'].get('slot_1', {})}`",
        f"- slot_2_transitions: `{summary['slot_transition_summary']['slot_transition_histograms'].get('slot_2', {})}`",
        f"- slot_3_transitions: `{summary['slot_transition_summary']['slot_transition_histograms'].get('slot_3', {})}`",
        "",
        "## Reading",
        "",
    ]
    lines.extend([f"- {item}" for item in summary["interpretation"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    probe_a_dir = normalize_probe_dir(args.probe_a)
    probe_b_dir = normalize_probe_dir(args.probe_b)
    aggregate_a = load_json(probe_a_dir / "aggregate_summary.json")
    aggregate_b = load_json(probe_b_dir / "aggregate_summary.json")
    payloads_a = load_json(probe_a_dir / "per_sample_summaries.json")
    payloads_b = load_json(probe_b_dir / "per_sample_summaries.json")
    summary = build_summary(args, aggregate_a, aggregate_b, payloads_a, payloads_b)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output_md, summary)
    print(output_json)
    print(output_md)


if __name__ == "__main__":
    main()
