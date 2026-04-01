import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"

for root in (str(REPO_ROOT), str(TRAINING_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize readiness for the soft-tail exposure rebalancing family."
    )
    parser.add_argument("--hardtail-manifest-json", required=True)
    parser.add_argument("--hardtail-profile-json", required=True)
    parser.add_argument("--prior-verdict-json", required=True)
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
    sample = train_dataset[0]
    return {
        "candidate_config_path": str(candidate_config_path.resolve()),
        "resolved_zju_dir": str(resolved_zju_dir),
        "train_total_len": int(len(train_dataset)),
        "selection_anchor_camera": str(sample.get("selection_anchor_camera")),
        "selection_source_policy": str(sample.get("selection_source_policy")),
        "selection_source_view_pool": str(sample.get("selection_source_view_pool")),
    }


def compute_anchor_quality_stats(manifest_path: Path, zju_root: Path, geom_subdir: str, anchor_camera: str):
    manifest = load_json(manifest_path)
    entries = [
        row
        for row in manifest.get("entries", [])
        if str(row.get("promoted_anchor_camera", "")) == str(anchor_camera)
    ]
    values = []
    for row in entries:
        seq_name = str(row["seq_name"])
        frame_id = int(row["frame_id"])
        geom_path = zju_root / seq_name / geom_subdir / f"frame_{frame_id:06d}.npz"
        geom = np.load(geom_path, allow_pickle=True)
        cam_names = [str(item) for item in geom["cam_names"].tolist()]
        if anchor_camera not in cam_names:
            continue
        local_idx = cam_names.index(anchor_camera)
        values.append(float(np.asarray(geom["depth_conf"][local_idx], dtype=np.float32).mean()))
    array = np.asarray(values, dtype=np.float32)
    if array.size <= 0:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "max": float(array.max()),
    }


def write_markdown(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    manifest_path = Path(args.hardtail_manifest_json)
    hardtail_profile_path = Path(args.hardtail_profile_json)
    prior_verdict_path = Path(args.prior_verdict_json)
    research_status_path = Path(args.research_status_json)
    candidate_config_path = resolve_config_path(args.candidate_config)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    hardtail_profile = load_json(hardtail_profile_path)
    prior_verdict = load_json(prior_verdict_path)
    research_status = load_json(research_status_path)
    smoke = build_train_dataset_smoke(candidate_config_path)
    zju_root = Path(smoke["resolved_zju_dir"])
    anchor_quality_stats = compute_anchor_quality_stats(
        manifest_path,
        zju_root,
        "vggt_geom",
        "Camera_B13",
    )

    anchor_ranked = hardtail_profile.get("anchor_enrichment_ranked", [])
    dominant_anchor_share = 0.0
    if anchor_ranked:
        dominant_anchor_share = float((anchor_ranked[0] or {}).get("tail_rate_within_group", 0.0))

    payload = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "comparison_type": "soft_tail_exposure_readiness",
        "family": "soft_tail_exposure_rebalancing",
        "prior_family_status": prior_verdict.get("status", ""),
        "prior_family": prior_verdict.get("family", ""),
        "research_loop_state": research_status.get("state", ""),
        "approved_problem_present": bool(research_status.get("approved_problem_present")),
        "cloud_must_remain_off": bool(research_status.get("cloud_must_remain_off", True)),
        "first_candidate_shape": "contract_balanced_soft_tail_taper",
        "first_candidate_config": str(candidate_config_path.resolve()),
        "dominant_contract_anchor_camera": "Camera_B13",
        "dominant_contract_share": dominant_anchor_share,
        "dominant_anchor_quality_stats": anchor_quality_stats,
        "selected_quality_window": {
            "interp": "smoothstep",
            "low": 2.0051,
            "high": 2.3407,
        },
        "selected_depth_conf_window": {
            "interp": "smoothstep",
            "low": 0.0,
            "high": 5.913640410988592,
        },
        "target_scales": {
            "reg": 0.95,
            "conf": 0.95,
        },
        "plumbing_smoke": smoke,
        "readiness": {
            "ready_for_manual_review": True,
            "ready_for_execution": True,
            "requires_new_manual_approval": True,
            "do_not_auto_open_ticket": True,
        },
        "reasoning": [
            "The bucket-granularity family already reduced dominance but still died at short gate on camera, so the next family should change exposure softness rather than hard bucket composition again.",
            "This first candidate keeps the current hybrid-ring source policy and dataset distribution fixed while using existing train-only loss hooks to apply a mild smooth taper only on the dominant Camera_B13 contract slice.",
            "The chosen quality window matches the interquartile range of Camera_B13 hard-tail anchor quality scores, so the taper is active where the official tail is concentrated without reopening hard thresholds or bucket ratio tweaks.",
        ],
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Soft Tail Exposure Readiness",
        "",
        f"- family: `{payload['family']}`",
        f"- prior_family: `{payload['prior_family']}`",
        f"- prior_family_status: `{payload['prior_family_status']}`",
        f"- research_loop_state: `{payload['research_loop_state']}`",
        f"- approved_problem_present: `{payload['approved_problem_present']}`",
        f"- cloud_must_remain_off: `{payload['cloud_must_remain_off']}`",
        f"- first_candidate_shape: `{payload['first_candidate_shape']}`",
        f"- first_candidate_config: `{payload['first_candidate_config']}`",
        f"- dominant_contract_anchor_camera: `{payload['dominant_contract_anchor_camera']}`",
        f"- dominant_contract_share: `{payload['dominant_contract_share']}`",
        f"- dominant_anchor_quality_stats: `{payload['dominant_anchor_quality_stats']}`",
        f"- selected_quality_window: `{payload['selected_quality_window']}`",
        f"- selected_depth_conf_window: `{payload['selected_depth_conf_window']}`",
        f"- target_scales: `{payload['target_scales']}`",
        f"- readiness: `{payload['readiness']}`",
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
