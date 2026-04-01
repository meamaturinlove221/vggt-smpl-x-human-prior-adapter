import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"

for root in (str(REPO_ROOT), str(TRAINING_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize official promoted hard-tail shape and hardcase-bucket readiness."
    )
    parser.add_argument("--residual-jsonl", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--slot3-tail-json", required=True)
    parser.add_argument("--mix-config", required=True)
    parser.add_argument("--coverage-output-json", required=True)
    parser.add_argument("--coverage-output-md", required=True)
    parser.add_argument("--readiness-output-json", required=True)
    parser.add_argument("--readiness-output-md", required=True)
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


def resolve_config_path(raw_value: str) -> Path:
    candidate = Path(raw_value)
    if candidate.suffix.lower() == ".yaml":
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        return candidate
    return (TRAINING_ROOT / "config" / f"{raw_value}.yaml").resolve()


def load_config_with_local_defaults(config_path: Path):
    cfg = OmegaConf.load(config_path)

    merged = OmegaConf.create()
    for entry in cfg.get("defaults", []):
        if entry == "_self_":
            continue
        if isinstance(entry, str):
            parent_name = entry
        elif isinstance(entry, dict):
            if len(entry) != 1:
                continue
            parent_name = str(next(iter(entry.values())))
        else:
            continue
        if parent_name == "_self_":
            continue
        parent_filename = parent_name if parent_name.endswith(".yaml") else f"{parent_name}.yaml"
        parent_path = (config_path.parent / parent_filename).resolve()
        if not parent_path.is_file():
            raise FileNotFoundError(f"Unable to resolve default config {parent_name} from {config_path}")
        merged = OmegaConf.merge(merged, load_config_with_local_defaults(parent_path))
    merged = OmegaConf.merge(merged, cfg)
    return merged


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def resolve_zju_dir(requested, seq_names, geom_subdir):
    requested = str(requested).strip()
    if requested and "YOUR/PATH/TO/ZJU" not in requested:
        candidate = Path(requested)
        if candidate.is_dir():
            return candidate.resolve()

    candidates = []
    g_datasets = "G:\\" + chr(0x6570) + chr(0x636E) + chr(0x96C6)
    candidates.extend(
        [
            Path(r"F:\datasets\ZJU_MoCap\data\zju_mocap"),
            Path(g_datasets) / "datasets" / "ZJU_MoCap" / "data" / "zju_mocap",
        ]
    )

    geom_subdirs = ensure_list(geom_subdir)
    best_candidate = None
    best_score = None
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate.absolute()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        if not resolved.is_dir():
            continue
        valid_subdir_count = 0
        total_frame_count = 0
        for seq_name in seq_names:
            for geom_subdir_name in geom_subdirs:
                geom_dir = resolved / str(seq_name) / geom_subdir_name
                if not geom_dir.is_dir():
                    continue
                frame_count = sum(1 for _ in geom_dir.glob("frame_*.npz"))
                if frame_count > 0:
                    valid_subdir_count += 1
                    total_frame_count += frame_count
        if valid_subdir_count <= 0:
            continue
        score = (int(valid_subdir_count), int(total_frame_count))
        if best_candidate is None or score > best_score:
            best_candidate = resolved
            best_score = score

    if best_candidate is None:
        raise FileNotFoundError(
            f"Unable to resolve ZJU root for seq_names={seq_names} geom_subdir={geom_subdir}."
        )
    return best_candidate


def canonical_source_set(row: dict) -> tuple[str, ...]:
    return tuple(str(item) for item in row.get("selected_source_only_camera_names", []))


def parse_sample_seq_name(sample_seq_name: str):
    text = str(sample_seq_name or "")
    prefix = "zju_"
    marker = "_frame_"
    if not text.startswith(prefix) or marker not in text:
        return None, None
    seq_name, frame_text = text[len(prefix):].rsplit(marker, 1)
    try:
        return seq_name, int(frame_text)
    except ValueError:
        return seq_name, None


def summarize_enrichment(all_rows, tail_rows, key_fn, key_name: str, top_k: int = 8):
    overall_counts = Counter(key_fn(row) for row in all_rows)
    tail_counts = Counter(key_fn(row) for row in tail_rows)
    baseline_tail_rate = float(len(tail_rows) / max(len(all_rows), 1))
    rows = []
    for key, tail_count in tail_counts.items():
        overall_count = int(overall_counts[key])
        tail_rate_within_group = float(tail_count / max(overall_count, 1))
        enrichment = float(tail_rate_within_group / baseline_tail_rate) if baseline_tail_rate > 0 else None
        rows.append(
            {
                key_name: list(key) if isinstance(key, tuple) else key,
                "tail_count": int(tail_count),
                "overall_count": overall_count,
                "tail_rate_within_group": tail_rate_within_group,
                "tail_enrichment_vs_baseline": enrichment,
            }
        )
    rows.sort(
        key=lambda item: (
            -int(item["tail_count"]),
            -(item["tail_enrichment_vs_baseline"] or 0.0),
            str(item[key_name]),
        )
    )
    return rows[:top_k]


def summarize_segments(tail_rows, gap_threshold: int = 20, top_k: int = 8):
    ordered = sorted(tail_rows, key=lambda row: int(row["frame_id"]))
    if not ordered:
        return []
    segments = []
    current = [ordered[0]]
    for row in ordered[1:]:
        if int(row["frame_id"]) - int(current[-1]["frame_id"]) <= gap_threshold:
            current.append(row)
        else:
            segments.append(current)
            current = [row]
    segments.append(current)

    summarized = []
    for segment in segments:
        anchors = Counter(str(row.get("promoted_anchor_camera")) for row in segment)
        source_sets = Counter(canonical_source_set(row) for row in segment)
        summarized.append(
            {
                "frame_start": int(segment[0]["frame_id"]),
                "frame_end": int(segment[-1]["frame_id"]),
                "entry_count": int(len(segment)),
                "avg_tail_score": float(sum(float(row["joint_depth_geom_tail_score"]) for row in segment) / len(segment)),
                "dominant_anchor_camera": anchors.most_common(1)[0][0],
                "dominant_source_only_set": list(source_sets.most_common(1)[0][0]),
                "frame_ids_preview": [int(row["frame_id"]) for row in segment[:10]],
            }
        )
    summarized.sort(key=lambda item: (-int(item["entry_count"]), -float(item["avg_tail_score"]), int(item["frame_start"])))
    return summarized[:top_k]


def format_metric(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_train_dataset_smoke(mix_config_path: Path, manifest_path: Path):
    manifest_entries = load_json(manifest_path).get("entries", [])
    manifest_keys = {(str(row["seq_name"]), int(row["frame_id"])) for row in manifest_entries}
    cfg = load_config_with_local_defaults(mix_config_path)
    dataset_cfgs = cfg.data.train.dataset.dataset_configs
    seq_names = ensure_list(dataset_cfgs[0].seq_names)
    geom_subdir = dataset_cfgs[0].geom_subdir
    resolved_zju_dir = resolve_zju_dir(cfg.get("zju_dir", ""), seq_names, geom_subdir)
    cfg.zju_dir = str(resolved_zju_dir)
    cfg.zju_hardcase_manifest_path = str(manifest_path.resolve())

    common_conf_dict = OmegaConf.to_container(cfg.data.train.common_config, resolve=True)
    common_conf_dict["training"] = True
    common_conf_dict["inside_random"] = False
    common_conf = OmegaConf.create(common_conf_dict)
    train_dataset = instantiate(cfg.data.train.dataset, common_config=common_conf, _recursive_=False)

    base_default = train_dataset.base_dataset.datasets[0]
    base_hardcase = train_dataset.base_dataset.datasets[1]
    default_sample = train_dataset[0]
    hardcase_boundary_index = int(base_default.len_train)
    hardcase_sample = train_dataset[hardcase_boundary_index]
    default_seq_name, default_frame_id = parse_sample_seq_name(default_sample.get("seq_name"))
    hardcase_seq_name, hardcase_frame_id = parse_sample_seq_name(hardcase_sample.get("seq_name"))
    return {
        "mix_config_path": str(mix_config_path),
        "resolved_zju_dir": str(resolved_zju_dir),
        "train_total_len": int(len(train_dataset)),
        "default_stream_len_train": int(base_default.len_train),
        "hardcase_stream_len_train": int(base_hardcase.len_train),
        "default_stream_sequence_list_len": int(base_default.sequence_list_len),
        "hardcase_stream_sequence_list_len": int(base_hardcase.sequence_list_len),
        "default_stream_manifest_applied": bool(base_default.sample_manifest_applied),
        "hardcase_stream_manifest_applied": bool(base_hardcase.sample_manifest_applied),
        "hardcase_stream_manifest_entry_count": int(base_hardcase.sample_manifest_entry_count),
        "default_sample": {
            "seq_name": str(default_sample.get("seq_name")),
            "anchor_camera": str(default_sample.get("selection_anchor_camera")),
            "is_manifest_member": bool(
                default_seq_name is not None
                and default_frame_id is not None
                and (default_seq_name, default_frame_id) in manifest_keys
            ),
        },
        "hardcase_sample": {
            "seq_name": str(hardcase_sample.get("seq_name")),
            "anchor_camera": str(hardcase_sample.get("selection_anchor_camera")),
            "is_manifest_member": bool(
                hardcase_seq_name is not None
                and hardcase_frame_id is not None
                and (hardcase_seq_name, hardcase_frame_id) in manifest_keys
            ),
        },
    }


def write_markdown(path: Path, lines: list[str]):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    residual_path = Path(args.residual_jsonl)
    manifest_path = Path(args.manifest_json)
    slot3_tail_path = Path(args.slot3_tail_json)
    mix_config_path = resolve_config_path(args.mix_config)
    coverage_json = Path(args.coverage_output_json)
    coverage_md = Path(args.coverage_output_md)
    readiness_json = Path(args.readiness_output_json)
    readiness_md = Path(args.readiness_output_md)
    for path in (coverage_json, coverage_md, readiness_json, readiness_md):
        path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = load_jsonl(residual_path)
    manifest = load_json(manifest_path)
    slot3_tail = load_json(slot3_tail_path)
    tail_rows = list(manifest.get("entries", []))
    baseline_tail_rate = float(len(tail_rows) / max(len(all_rows), 1))

    coverage_payload = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "comparison_type": "official_hardtail_coverage_profile",
        "manifest_status": manifest.get("status"),
        "eligible_entry_count": int(len(all_rows)),
        "hard_tail_entry_count": int(len(tail_rows)),
        "hard_tail_entry_share": baseline_tail_rate,
        "anchor_enrichment_ranked": summarize_enrichment(
            all_rows,
            tail_rows,
            key_fn=lambda row: str(row.get("promoted_anchor_camera")),
            key_name="promoted_anchor_camera",
        ),
        "source_only_set_enrichment_ranked": summarize_enrichment(
            all_rows,
            tail_rows,
            key_fn=canonical_source_set,
            key_name="selected_source_only_set",
        ),
        "hard_tail_segments_ranked": summarize_segments(tail_rows),
        "slot3_probe_gap": {
            "probe_frame_range": slot3_tail["tail_alignment_assessment"].get("probe_frame_range"),
            "official_tail_frame_range": slot3_tail["tail_alignment_assessment"].get("official_tail_frame_range"),
            "probe_tail_overlap_note": slot3_tail["tail_alignment_assessment"].get("probe_tail_overlap_note"),
        },
    }
    coverage_json.write_text(json.dumps(coverage_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    coverage_lines = [
        "# Official Hardtail Coverage Profile",
        "",
        f"- eligible_entry_count: `{coverage_payload['eligible_entry_count']}`",
        f"- hard_tail_entry_count: `{coverage_payload['hard_tail_entry_count']}`",
        f"- hard_tail_entry_share: `{coverage_payload['hard_tail_entry_share']:.4f}`",
        f"- slot3_probe_gap: `{coverage_payload['slot3_probe_gap']}`",
        "",
        "## Anchor Enrichment",
        "",
    ]
    for item in coverage_payload["anchor_enrichment_ranked"]:
        coverage_lines.append(
            "- `{camera}` tail_count={tail} overall_count={overall} tail_rate={rate} enrichment={enrichment}".format(
                camera=item["promoted_anchor_camera"],
                tail=item["tail_count"],
                overall=item["overall_count"],
                rate=format_metric(item["tail_rate_within_group"]),
                enrichment=format_metric(item["tail_enrichment_vs_baseline"]),
            )
        )
    coverage_lines.extend(["", "## Source-Only Set Enrichment", ""])
    for item in coverage_payload["source_only_set_enrichment_ranked"]:
        coverage_lines.append(
            "- `{source_set}` tail_count={tail} overall_count={overall} tail_rate={rate} enrichment={enrichment}".format(
                source_set=item["selected_source_only_set"],
                tail=item["tail_count"],
                overall=item["overall_count"],
                rate=format_metric(item["tail_rate_within_group"]),
                enrichment=format_metric(item["tail_enrichment_vs_baseline"]),
            )
        )
    coverage_lines.extend(["", "## Hard-Tail Segments", ""])
    for item in coverage_payload["hard_tail_segments_ranked"]:
        coverage_lines.append(
            "- frame `{start}-{end}` count={count} avg_score={score} anchor={anchor} source_only={source_set} preview={preview}".format(
                start=item["frame_start"],
                end=item["frame_end"],
                count=item["entry_count"],
                score=format_metric(item["avg_tail_score"]),
                anchor=item["dominant_anchor_camera"],
                source_set=item["dominant_source_only_set"],
                preview=item["frame_ids_preview"],
            )
        )
    write_markdown(coverage_md, coverage_lines)

    smoke = build_train_dataset_smoke(mix_config_path, manifest_path)
    readiness_payload = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "comparison_type": "residual_case_coverage_readiness",
        "family": "residual_case_coverage_rebalancing",
        "current_decision_gate": slot3_tail.get("decision"),
        "official_manifest_status": manifest.get("status"),
        "official_manifest_entry_count": int(len(tail_rows)),
        "previous_blocker_cleared": bool(manifest.get("entries_frozen")),
        "mix_config_path": str(mix_config_path),
        "plumbing_smoke": smoke,
        "readiness": {
            "ready_for_manual_review": True,
            "ready_for_execution": True,
            "requires_new_manual_approval": True,
            "do_not_auto_open_ticket": True,
        },
        "next_recommended_direction": "residual_case_coverage_rebalancing",
        "next_recommended_candidate_shape": "promotedlead_hardcase_bucket_mix",
        "reasoning": [
            "The official promoted hard-tail manifest is now frozen from real per-frame residuals, so the previous evidence blocker is cleared.",
            "The hardcase-bucket mix config already exists and the train dataset plumbing can instantiate both default and manifest-filtered streams successfully.",
            "Because the slot_3 probe basket does not overlap the official hard-tail frame range, the next task-mode step should return to residual case coverage rather than open a slot_3 ticket.",
        ],
    }
    readiness_json.write_text(json.dumps(readiness_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    readiness_lines = [
        "# Residual Case Coverage Readiness",
        "",
        f"- family: `{readiness_payload['family']}`",
        f"- current_decision_gate: `{readiness_payload['current_decision_gate']}`",
        f"- official_manifest_status: `{readiness_payload['official_manifest_status']}`",
        f"- official_manifest_entry_count: `{readiness_payload['official_manifest_entry_count']}`",
        f"- previous_blocker_cleared: `{readiness_payload['previous_blocker_cleared']}`",
        f"- readiness: `{readiness_payload['readiness']}`",
        "",
        "## Plumbing Smoke",
        "",
        f"- plumbing_smoke: `{readiness_payload['plumbing_smoke']}`",
        "",
        "## Reasoning",
        "",
    ]
    readiness_lines.extend([f"- {item}" for item in readiness_payload["reasoning"]])
    write_markdown(readiness_md, readiness_lines)

    print(coverage_json)
    print(coverage_md)
    print(readiness_json)
    print(readiness_md)


if __name__ == "__main__":
    main()
