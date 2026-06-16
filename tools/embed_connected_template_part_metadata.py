from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


PART_NAMES = (
    "torso_limbs",
    "left_hand",
    "right_hand",
    "head_face",
    "head_top_hairline",
    "lower_clothing_proxy",
)

PART_FAMILIES = (
    "body",
    "hand",
    "hand",
    "face",
    "hair",
    "body",
)

REQUIRED_CARRIER_MASKS = (
    "face_front_vertex_mask",
    "hairline_vertex_mask",
    "head_vertex_mask",
    "left_hand_vertex_mask",
    "right_hand_vertex_mask",
    "lower_clothing_vertex_mask",
)

METADATA_KEYS = ("part_names", "part_families", "required_carrier_masks")
GEOMETRY_KEYS = (
    "vertices",
    "faces",
    "normals",
    "canonical_positions",
    "hybrid_vertices",
    "hybrid_faces",
    "hair_new_vertices",
    "hair_hybrid_faces",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a connected template payload NPZ and embed self-describing part "
            "metadata. Research-only utility: it never writes strict-pass state "
            "and does not alter geometry arrays."
        )
    )
    parser.add_argument("--input-payload", required=True, type=Path)
    parser.add_argument("--output-payload", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def metadata_arrays() -> dict[str, np.ndarray]:
    return {
        "part_names": np.asarray(PART_NAMES, dtype="U32"),
        "part_families": np.asarray(PART_FAMILIES, dtype="U16"),
        "required_carrier_masks": np.asarray(REQUIRED_CARRIER_MASKS, dtype="U40"),
    }


def validate_payload(arrays: dict[str, np.ndarray], input_path: Path) -> dict[str, Any]:
    if "part_ids" not in arrays:
        raise KeyError(f"{input_path} does not contain required array 'part_ids'")

    part_ids = np.asarray(arrays["part_ids"])
    if part_ids.ndim != 1:
        raise ValueError(f"part_ids must be 1D, got shape {part_ids.shape}")
    if not np.issubdtype(part_ids.dtype, np.integer):
        raise TypeError(f"part_ids must be integer dtype, got {part_ids.dtype}")

    if "hybrid_vertices" in arrays:
        hybrid_vertices = np.asarray(arrays["hybrid_vertices"])
        if hybrid_vertices.ndim != 2 or hybrid_vertices.shape[1] != 3:
            raise ValueError(f"hybrid_vertices must have shape (N, 3), got {hybrid_vertices.shape}")
        if int(part_ids.shape[0]) != int(hybrid_vertices.shape[0]):
            raise ValueError(
                "part_ids must align with hybrid_vertices when hybrid_vertices is present: "
                f"part_ids={part_ids.shape[0]} hybrid_vertices={hybrid_vertices.shape[0]}"
            )

    unknown_ids = sorted(set(int(value) for value in np.unique(part_ids)) - set(range(len(PART_NAMES))))
    if unknown_ids:
        raise ValueError(
            "part_ids contains ids without embedded fallback metadata: "
            f"{unknown_ids}; supported ids are 0..{len(PART_NAMES) - 1}"
        )

    missing_masks = [key for key in REQUIRED_CARRIER_MASKS if key not in arrays]
    misaligned_masks: list[dict[str, Any]] = []
    for key in REQUIRED_CARRIER_MASKS:
        if key not in arrays:
            continue
        mask = np.asarray(arrays[key])
        if mask.ndim != 1:
            misaligned_masks.append({"key": key, "shape": list(mask.shape), "reason": "not_1d"})
        elif int(mask.shape[0]) != int(part_ids.shape[0]):
            misaligned_masks.append(
                {
                    "key": key,
                    "shape": list(mask.shape),
                    "reason": "length_mismatch_with_part_ids",
                    "part_ids": int(part_ids.shape[0]),
                }
            )

    if misaligned_masks:
        raise ValueError(f"required carrier masks are not aligned: {misaligned_masks}")

    counts = {
        str(part_id): int(np.count_nonzero(part_ids == part_id))
        for part_id in range(len(PART_NAMES))
    }
    mask_counts = {
        key: int(np.count_nonzero(np.asarray(arrays[key], dtype=bool)))
        for key in REQUIRED_CARRIER_MASKS
        if key in arrays
    }
    return {
        "part_id_count": int(part_ids.shape[0]),
        "unique_part_ids": [int(value) for value in np.unique(part_ids).tolist()],
        "part_counts": counts,
        "required_carrier_masks_missing": missing_masks,
        "required_carrier_mask_counts": mask_counts,
    }


def validate_existing_metadata(arrays: dict[str, np.ndarray], additions: dict[str, np.ndarray]) -> None:
    for key, expected in additions.items():
        if key not in arrays:
            continue
        existing = np.asarray(arrays[key]).astype(str)
        if existing.shape != expected.shape or not np.array_equal(existing, expected.astype(str)):
            raise ValueError(
                f"{key} already exists with different values; refusing to rewrite existing metadata"
            )


def assert_geometry_unchanged(input_arrays: dict[str, np.ndarray], output_arrays: dict[str, np.ndarray]) -> None:
    for key in GEOMETRY_KEYS:
        if key not in input_arrays:
            continue
        if key not in output_arrays:
            raise AssertionError(f"output is missing geometry array {key}")
        before = np.asarray(input_arrays[key])
        after = np.asarray(output_arrays[key])
        if before.shape != after.shape or before.dtype != after.dtype or not np.array_equal(before, after):
            raise AssertionError(f"geometry array changed during metadata embedding: {key}")


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = args.input_payload.expanduser().resolve()
    output_path = args.output_payload.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite")
    if input_path == output_path:
        raise ValueError("input and output payload paths must be different")

    arrays = load_npz_arrays(input_path)
    additions = metadata_arrays()
    validate_existing_metadata(arrays, additions)
    validation = validate_payload(arrays, input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_arrays = dict(arrays)
    output_arrays.update(additions)
    np.savez_compressed(output_path, **output_arrays)

    written = load_npz_arrays(output_path)
    assert_geometry_unchanged(arrays, written)
    for key, expected in additions.items():
        actual = np.asarray(written[key]).astype(str)
        if actual.shape != expected.shape or not np.array_equal(actual, expected.astype(str)):
            raise AssertionError(f"metadata array was not written correctly: {key}")

    summary_path = output_path.with_suffix(".summary.json")
    summary = {
        "research_only": True,
        "strict_pass_write": False,
        "input_payload": input_path,
        "output_payload": output_path,
        "sidecar_summary": summary_path,
        "preserved_array_count": len(arrays),
        "added_metadata_keys": list(METADATA_KEYS),
        "all_output_keys": list(written.keys()),
        "part_names": {idx: name for idx, name in enumerate(PART_NAMES)},
        "part_families": {idx: family for idx, family in enumerate(PART_FAMILIES)},
        "required_carrier_masks": list(REQUIRED_CARRIER_MASKS),
        "geometry_keys_checked": [key for key in GEOMETRY_KEYS if key in arrays],
        **validation,
    }
    write_summary(summary_path, summary)

    print(f"wrote payload: {output_path}")
    print(f"wrote summary: {summary_path}")
    print("research_only=True strict_pass_write=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
