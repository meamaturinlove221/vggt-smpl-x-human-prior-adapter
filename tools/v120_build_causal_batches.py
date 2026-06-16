"""Build paired causal batches for the semantic-bottleneck route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_FEATURE_MAP = Path(
    r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild"
    r"\output\V8100000_V9000000_smplx_feature_encoding"
    r"\V8200000_smplx_feature_raster\feature_maps.npz"
)


SUPPORT_NAMES = [
    "smplx_visibility",
    "semantic_foreground",
    "signed_boundary",
    "phone_object_exclusion",
]
SEMANTIC_NAMES = [
    "canonical_x",
    "canonical_y",
    "canonical_z",
    "posed_x",
    "posed_y",
    "posed_z",
    "vertex_id_sin",
    "vertex_id_cos",
    "macro_part_scaled",
    "normal_x",
    "normal_y",
    "normal_z",
    "semantic_head_face",
    "semantic_hairline",
    "semantic_left_hand",
    "semantic_right_hand",
]
REQUIRED_SEMANTIC_FIELDS = [
    "canonical_x",
    "canonical_y",
    "canonical_z",
    "posed_x",
    "posed_y",
    "posed_z",
    "vertex_id_embedding",
    "body_part_id",
    "skinning_weights",
    "smpl_normal",
    "joint_distance",
    "nearest_vertex_index",
]


def _load_feature_maps(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=True) as data:
        return data["feature_maps"].astype(np.float32), [str(x) for x in data["channel_names"].tolist()]


def _indices(names: list[str], wanted: list[str]) -> list[int]:
    lookup = {name: i for i, name in enumerate(names)}
    return [lookup[name] for name in wanted if name in lookup]


def _sample_pixels(feature_maps: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    views, _, height, width = feature_maps.shape
    total = views * height * width
    count = min(count, total)
    flat_idx = rng.choice(total, size=count, replace=False)
    view = flat_idx // (height * width)
    rest = flat_idx % (height * width)
    y = rest // width
    x = rest % width
    sampled = feature_maps[view, :, y, x].transpose(0, 1).astype(np.float32)
    coords = np.stack(
        [
            view.astype(np.float32) / max(views - 1, 1),
            x.astype(np.float32) / max(width - 1, 1),
            y.astype(np.float32) / max(height - 1, 1),
        ],
        axis=1,
    )
    return sampled, coords


def build_causal_batch(feature_map_path: Path, output_dir: Path, *, sample_count: int = 4096, seed: int = 120000000) -> dict[str, Any]:
    feature_maps, names = _load_feature_maps(feature_map_path)
    sampled, coords = _sample_pixels(feature_maps, sample_count, seed)
    support_idx = _indices(names, SUPPORT_NAMES)
    semantic_idx = _indices(names, SEMANTIC_NAMES)
    rng = np.random.default_rng(seed + 17)

    support = np.concatenate([sampled[:, support_idx], coords], axis=1).astype(np.float32)
    semantic = sampled[:, semantic_idx].astype(np.float32)
    observation = np.concatenate(
        [
            sampled[:, _indices(names, ["posed_x", "posed_y", "posed_z"])],
            sampled[:, _indices(names, ["normal_x", "normal_y", "normal_z"])],
            sampled[:, _indices(names, ["smplx_visibility", "semantic_foreground"])],
        ],
        axis=1,
    ).astype(np.float32)

    random_semantic = rng.normal(0.0, 1.0, size=semantic.shape).astype(np.float32)
    shuffled_semantic = semantic[rng.permutation(semantic.shape[0])].astype(np.float32)
    zero_semantic = np.zeros_like(semantic)
    zero_observation = np.zeros_like(observation)
    zero_support = np.zeros_like(support)

    macro_idx = names.index("macro_part_scaled") if "macro_part_scaled" in names else None
    body_part = np.zeros((semantic.shape[0],), dtype=np.int64)
    if macro_idx is not None:
        body_part = np.clip(np.floor(sampled[:, macro_idx] * 15.0), 0, 15).astype(np.int64)
    vertex_bin = np.clip(
        np.floor(((sampled[:, names.index("vertex_id_sin")] + 1.0) * 0.5) * 2047.0),
        0,
        2047,
    ).astype(np.int64) if "vertex_id_sin" in names else np.zeros((semantic.shape[0],), dtype=np.int64)
    canonical_idx = _indices(names, ["canonical_x", "canonical_y", "canonical_z"])
    normal_idx = _indices(names, ["normal_x", "normal_y", "normal_z"])
    canonical_xyz = sampled[:, canonical_idx].astype(np.float32)
    normal = sampled[:, normal_idx].astype(np.float32)
    if normal.shape[1] != 3:
        normal = np.zeros_like(canonical_xyz)
        normal[:, 2] = 1.0

    batches = {
        "true": (support, semantic, observation, np.ones((semantic.shape[0],), dtype=np.float32)),
        "same_support_random_semantic": (support, random_semantic, observation, np.zeros((semantic.shape[0],), dtype=np.float32)),
        "same_support_shuffled_semantic": (support, shuffled_semantic, observation, np.zeros((semantic.shape[0],), dtype=np.float32)),
        "no_observation": (support, semantic, zero_observation, np.ones((semantic.shape[0],), dtype=np.float32)),
        "observation_only": (zero_support, zero_semantic, observation, np.zeros((semantic.shape[0],), dtype=np.float32)),
        "support_only": (support, zero_semantic, zero_observation, np.zeros((semantic.shape[0],), dtype=np.float32)),
        "no_teacher": (support, semantic, observation, np.ones((semantic.shape[0],), dtype=np.float32)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    batch_paths: dict[str, str] = {}
    for name, (sup, sem, obs, is_true) in batches.items():
        path = output_dir / f"{name}.npz"
        np.savez_compressed(
            path,
            support=sup,
            semantic=sem,
            observation=obs,
            target_delta_point=np.zeros((semantic.shape[0], 3), dtype=np.float32),
            target_normal=normal,
            target_occupancy=np.ones((semantic.shape[0], 1), dtype=np.float32),
            target_reliability=np.ones((semantic.shape[0], 1), dtype=np.float32),
            aux_canonical_xyz=canonical_xyz,
            aux_body_part=body_part,
            aux_nearest_vertex_bin=vertex_bin,
            aux_skinning_weights=np.zeros((semantic.shape[0], 24), dtype=np.float32),
            aux_is_true_semantic=is_true,
        )
        batch_paths[name] = str(path)

    available_fields = set(names)
    proxy_fields = {
        "vertex_id_embedding": ["vertex_id_sin", "vertex_id_cos"],
        "body_part_id": ["macro_part_scaled"],
        "smpl_normal": ["normal_x", "normal_y", "normal_z"],
    }
    missing_required = []
    for field in REQUIRED_SEMANTIC_FIELDS:
        if field in available_fields:
            continue
        if field in proxy_fields and all(x in available_fields for x in proxy_fields[field]):
            continue
        missing_required.append(field)

    schema = {
        "feature_map_path": str(feature_map_path),
        "feature_map_shape": list(feature_maps.shape),
        "channel_names": names,
        "support_channels": [names[i] for i in support_idx],
        "semantic_channels": [names[i] for i in semantic_idx],
        "observation_channels": ["posed_xyz_proxy", "normal_proxy", "visibility_foreground_proxy"],
        "sample_count": int(semantic.shape[0]),
        "batch_paths": batch_paths,
        "paired_causal_contract": {
            "same_support_random_semantic": "support and observation are bit-identical to true; semantic is randomized",
            "same_support_shuffled_semantic": "support and observation are bit-identical to true; semantic is shuffled",
        },
        "missing_required_semantic_fields": missing_required,
        "proxy_semantic_fields": proxy_fields,
    }
    return schema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-map", type=Path, default=DEFAULT_FEATURE_MAP)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--schema-out", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=120000000)
    args = parser.parse_args()

    schema = build_causal_batch(args.feature_map, args.output_dir, sample_count=args.sample_count, seed=args.seed)
    args.schema_out.parent.mkdir(parents=True, exist_ok=True)
    args.schema_out.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
