import argparse
import json
import sys
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
        description="Summarize readiness for the hybrid tail exposure balancing family."
    )
    parser.add_argument("--bucket-readiness-json", required=True)
    parser.add_argument("--soft-readiness-json", required=True)
    parser.add_argument("--research-status-json", required=True)
    parser.add_argument("--candidate-config", required=True)
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


def build_train_dataset_smoke(candidate_config_path: Path):
    cfg = load_config_with_local_defaults(candidate_config_path)
    dataset_cfg = cfg.data.train.dataset
    seq_names = ensure_list(dataset_cfg.dataset_configs[0].seq_names)
    geom_subdir = dataset_cfg.dataset_configs[0].geom_subdir
    resolved_zju_dir = resolve_zju_dir(cfg.get("zju_dir", ""), seq_names, geom_subdir)
    cfg.zju_dir = str(resolved_zju_dir)
    common_conf_dict = OmegaConf.to_container(cfg.data.train.common_config, resolve=True)
    common_conf_dict["training"] = True
    common_conf_dict["inside_random"] = False
    common_conf = OmegaConf.create(common_conf_dict)
    train_dataset = instantiate(cfg.data.train.dataset, common_config=common_conf, _recursive_=False)
    base_default = train_dataset.base_dataset.datasets[0]
    base_hardcase = train_dataset.base_dataset.datasets[1]
    hardcase_sample = train_dataset[int(base_default.len_train)]
    return {
        "candidate_config_path": str(candidate_config_path.resolve()),
        "resolved_zju_dir": str(resolved_zju_dir),
        "train_total_len": int(len(train_dataset)),
        "default_stream_len_train": int(base_default.len_train),
        "hardcase_stream_len_train": int(base_hardcase.len_train),
        "hardcase_stream_manifest_applied": bool(base_hardcase.sample_manifest_applied),
        "hardcase_stream_manifest_entry_count": int(base_hardcase.sample_manifest_entry_count),
        "hardcase_sample_anchor_camera": str(hardcase_sample.get("selection_anchor_camera")),
        "hardcase_sample_source_policy": str(hardcase_sample.get("selection_source_policy")),
    }


def write_markdown(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    bucket_readiness = load_json(Path(args.bucket_readiness_json))
    soft_readiness = load_json(Path(args.soft_readiness_json))
    research_status = load_json(Path(args.research_status_json))
    candidate_config_path = resolve_config_path(args.candidate_config)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    smoke = build_train_dataset_smoke(candidate_config_path)

    payload = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "comparison_type": "hybrid_tail_exposure_readiness",
        "family": "hybrid_tail_exposure_balancing",
        "research_loop_state": research_status.get("state", ""),
        "approved_problem_present": bool(research_status.get("approved_problem_present")),
        "cloud_must_remain_off": bool(research_status.get("cloud_must_remain_off", True)),
        "first_candidate_shape": "stratified_bucket_plus_soft_guard",
        "first_candidate_config": str(candidate_config_path.resolve()),
        "component_readout": {
            "bucket_candidate_short_delta_camera": (
                (bucket_readiness.get("replacement_summary", {}) and None)
            ),
            "bucket_dominance_reduction": bucket_readiness.get("dominance_reduction", {}),
            "soft_target_scales": soft_readiness.get("target_scales", {}),
            "soft_selected_quality_window": soft_readiness.get("selected_quality_window", {}),
            "soft_selected_depth_conf_window": soft_readiness.get("selected_depth_conf_window", {}),
        },
        "hypothesis": (
            "The refined bucket family already produced strong depth gains but paid a small camera tax, while "
            "the soft-tail family removed the camera tax but collapsed to near no-op. A hybrid candidate may "
            "preserve more of the bucket depth gain while using a mild soft guard to neutralize the dominant "
            "Camera_B13 contract slice."
        ),
        "plumbing_smoke": smoke,
        "readiness": {
            "ready_for_manual_review": True,
            "ready_for_execution": True,
            "requires_new_manual_approval": True,
            "do_not_auto_open_ticket": True,
        },
        "reasoning": [
            "This family is not bucket-only and not soft-only: it keeps the refined hardtail mix and adds a mild B13 soft guard using existing hooks only.",
            "The candidate remains config-only because both ingredients are already live in the repo: manifest-aware two-stream mixing and train-only anchor-conditioned smoothstep scaling.",
            "The next bounded question is whether the hybrid can keep the depth gain shape of the bucket family while removing its short-gate camera tax.",
        ],
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Hybrid Tail Exposure Readiness",
        "",
        f"- family: `{payload['family']}`",
        f"- research_loop_state: `{payload['research_loop_state']}`",
        f"- approved_problem_present: `{payload['approved_problem_present']}`",
        f"- cloud_must_remain_off: `{payload['cloud_must_remain_off']}`",
        f"- first_candidate_shape: `{payload['first_candidate_shape']}`",
        f"- first_candidate_config: `{payload['first_candidate_config']}`",
        "",
        "## Hypothesis",
        "",
        f"- {payload['hypothesis']}",
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
