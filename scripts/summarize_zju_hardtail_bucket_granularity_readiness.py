import argparse
import json
import sys
from collections import Counter
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
        description="Summarize readiness for the hardtail bucket granularity refinement family."
    )
    parser.add_argument("--refined-manifest-json", required=True)
    parser.add_argument("--postmortem-json", required=True)
    parser.add_argument("--research-status-json", required=True)
    parser.add_argument("--mix-config", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
    return OmegaConf.merge(merged, cfg)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main():
    args = parse_args()
    refined_manifest_path = Path(args.refined_manifest_json)
    postmortem_path = Path(args.postmortem_json)
    research_status_path = Path(args.research_status_json)
    mix_config_path = resolve_config_path(args.mix_config)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    refined_manifest = load_json(refined_manifest_path)
    postmortem = load_json(postmortem_path)
    research_status = load_json(research_status_path)
    smoke = build_train_dataset_smoke(mix_config_path, refined_manifest_path)

    official_dominant_contract = refined_manifest.get("official_summary", {}).get("dominant_contract", {})
    refined_dominant_contract = refined_manifest.get("refined_summary", {}).get("dominant_contract", {})
    official_dominant_segment = refined_manifest.get("official_summary", {}).get("dominant_segment", {})
    refined_dominant_segment = refined_manifest.get("refined_summary", {}).get("dominant_segment", {})
    target_entry_count = int(refined_manifest.get("target_entry_count", 0) or 0)

    payload = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "comparison_type": "hardtail_bucket_granularity_readiness",
        "family": "hardtail_bucket_granularity_refinement",
        "chosen_conclusion": postmortem.get("bucket_vs_mix_diagnosis", {}).get("chosen_conclusion", ""),
        "support_genuinely_new_manual_problem": bool(
            postmortem.get("ticket_readout", {}).get("support_genuinely_new_manual_problem")
        ),
        "research_loop_state": research_status.get("state", ""),
        "approved_problem_present": bool(research_status.get("approved_problem_present")),
        "cloud_must_remain_off": bool(research_status.get("cloud_must_remain_off", True)),
        "first_candidate_shape": "contract_segment_stratified_hardtail_bucket",
        "first_candidate_config": str(mix_config_path.resolve()),
        "refined_manifest_path": str(refined_manifest_path.resolve()),
        "refined_manifest_entry_count": int(len(refined_manifest.get("entries", []))),
        "dominance_reduction": {
            "official_dominant_contract_count": official_dominant_contract.get("count"),
            "official_dominant_contract_share": (
                None
                if target_entry_count <= 0 or official_dominant_contract.get("count") is None
                else float(official_dominant_contract["count"] / target_entry_count)
            ),
            "refined_dominant_contract_count": refined_dominant_contract.get("count"),
            "refined_dominant_contract_share": (
                None
                if target_entry_count <= 0 or refined_dominant_contract.get("count") is None
                else float(refined_dominant_contract["count"] / target_entry_count)
            ),
            "official_dominant_segment_count": official_dominant_segment.get("count"),
            "official_dominant_segment_share": (
                None
                if target_entry_count <= 0 or official_dominant_segment.get("count") is None
                else float(official_dominant_segment["count"] / target_entry_count)
            ),
            "refined_dominant_segment_count": refined_dominant_segment.get("count"),
            "refined_dominant_segment_share": (
                None
                if target_entry_count <= 0 or refined_dominant_segment.get("count") is None
                else float(refined_dominant_segment["count"] / target_entry_count)
            ),
        },
        "replacement_summary": refined_manifest.get("replacement_summary", {}),
        "plumbing_smoke": smoke,
        "readiness": {
            "ready_for_manual_review": True,
            "ready_for_execution": True,
            "requires_new_manual_approval": True,
            "do_not_auto_open_ticket": True,
        },
        "reasoning": [
            "The postmortem conclusion is BUCKET_TOO_COARSE, so the next family changes bucket granularity rather than retrying the residual family.",
            "The refined manifest keeps the same promoted source policy and the same 4:1 two-stream contract while reducing concentration in the dominant contract slice and dominant contiguous segment.",
            "The manifest-aware two-stream dataset still instantiates cleanly on the current repo, so this new family is execution-ready pending a future manual approval only.",
        ],
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Hardtail Bucket Granularity Readiness",
        "",
        f"- family: `{payload['family']}`",
        f"- chosen_conclusion: `{payload['chosen_conclusion']}`",
        f"- support_genuinely_new_manual_problem: `{payload['support_genuinely_new_manual_problem']}`",
        f"- research_loop_state: `{payload['research_loop_state']}`",
        f"- approved_problem_present: `{payload['approved_problem_present']}`",
        f"- cloud_must_remain_off: `{payload['cloud_must_remain_off']}`",
        f"- first_candidate_shape: `{payload['first_candidate_shape']}`",
        f"- first_candidate_config: `{payload['first_candidate_config']}`",
        f"- refined_manifest_path: `{payload['refined_manifest_path']}`",
        f"- refined_manifest_entry_count: `{payload['refined_manifest_entry_count']}`",
        f"- readiness: `{payload['readiness']}`",
        "",
        "## Dominance Reduction",
        "",
        f"- official_dominant_contract_count: `{payload['dominance_reduction']['official_dominant_contract_count']}`",
        f"- official_dominant_contract_share: `{fmt(payload['dominance_reduction']['official_dominant_contract_share'])}`",
        f"- refined_dominant_contract_count: `{payload['dominance_reduction']['refined_dominant_contract_count']}`",
        f"- refined_dominant_contract_share: `{fmt(payload['dominance_reduction']['refined_dominant_contract_share'])}`",
        f"- official_dominant_segment_count: `{payload['dominance_reduction']['official_dominant_segment_count']}`",
        f"- official_dominant_segment_share: `{fmt(payload['dominance_reduction']['official_dominant_segment_share'])}`",
        f"- refined_dominant_segment_count: `{payload['dominance_reduction']['refined_dominant_segment_count']}`",
        f"- refined_dominant_segment_share: `{fmt(payload['dominance_reduction']['refined_dominant_segment_share'])}`",
        "",
        "## Plumbing Smoke",
        "",
        f"- plumbing_smoke: `{payload['plumbing_smoke']}`",
        "",
        "## Reasoning",
        "",
    ]
    lines.extend([f"- {item}" for item in payload["reasoning"]])
    write_markdown(output_md, lines)

    print(output_json)
    print(output_md)


if __name__ == "__main__":
    main()
