from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


KNOWN_PART_NAMES = {
    0: "torso_limbs",
    1: "left_hand",
    2: "right_hand",
    3: "head_face",
    4: "head_top_hairline",
    5: "lower_clothing_proxy",
}

KNOWN_PART_FAMILIES = {
    0: "body",
    1: "hand",
    2: "hand",
    3: "face",
    4: "hair",
    5: "body",
}

REQUIRED_MASKS = (
    "face_front_vertex_mask",
    "hairline_vertex_mask",
    "head_vertex_mask",
    "left_hand_vertex_mask",
    "right_hand_vertex_mask",
    "lower_clothing_vertex_mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a self-describing JSON sidecar contract for a connected human "
            "surface template payload. This is a hygiene utility only: it does "
            "not optimize geometry, export a teacher/candidate, or write strict pass state."
        )
    )
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def bbox(vertices: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if not np.any(mask):
        return {"count": 0, "bbox_min": None, "bbox_max": None}
    pts = vertices[mask]
    return {
        "count": int(pts.shape[0]),
        "bbox_min": pts.min(axis=0).astype(float).tolist(),
        "bbox_max": pts.max(axis=0).astype(float).tolist(),
    }


def main() -> int:
    args = parse_args()
    payload_path = args.payload.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite")
    if not payload_path.is_file():
        raise FileNotFoundError(payload_path)

    with np.load(payload_path, allow_pickle=False) as data:
        files = set(data.files)
        if "hybrid_vertices" in files and "hybrid_faces" in files:
            vertex_key = "hybrid_vertices"
            face_key = "hybrid_faces"
        elif "vertices" in files and "faces" in files:
            vertex_key = "vertices"
            face_key = "faces"
        else:
            raise KeyError("payload must contain hybrid_vertices/hybrid_faces or vertices/faces")
        vertices = np.asarray(data[vertex_key], dtype=np.float32)
        faces = np.asarray(data[face_key], dtype=np.int64)
        if "part_ids" in files and data["part_ids"].shape[0] == vertices.shape[0]:
            part_ids = np.asarray(data["part_ids"], dtype=np.int64)
        else:
            raise ValueError("part_ids must exist and align with selected vertices")
        mask_summary = {}
        for key in REQUIRED_MASKS:
            if key in files and data[key].shape[0] == vertices.shape[0]:
                mask_summary[key] = bbox(vertices, np.asarray(data[key], dtype=bool))
            else:
                mask_summary[key] = {"count": 0, "bbox_min": None, "bbox_max": None, "missing_or_unaligned": True}

    unique_parts = sorted(int(v) for v in np.unique(part_ids))
    part_summary = {}
    for part in unique_parts:
        mask = part_ids == int(part)
        part_summary[str(part)] = {
            "name": KNOWN_PART_NAMES.get(int(part), str(part)),
            "family": KNOWN_PART_FAMILIES.get(int(part), "unknown"),
            **bbox(vertices, mask),
        }

    contract = {
        "research_only": True,
        "strict_pass_write": False,
        "payload": str(payload_path),
        "contract_version": "connected_template_contract_v1",
        "mesh": {
            "vertex_key": vertex_key,
            "face_key": face_key,
            "vertices": int(vertices.shape[0]),
            "faces": int(faces.shape[0]),
            "part_ids_key": "part_ids",
            "part_ids_aligned": True,
        },
        "part_names": {str(k): v for k, v in KNOWN_PART_NAMES.items()},
        "part_families": {str(k): v for k, v in KNOWN_PART_FAMILIES.items()},
        "required_masks": mask_summary,
        "parts": part_summary,
        "downstream_rules": {
            "default_mesh": "hybrid mesh when part_ids aligns with hybrid_vertices",
            "never_mix": "Do not use base vertices/faces with hybrid part_ids.",
            "face_front": "Use face_front_vertex_mask for front-face critical support when available.",
            "hair": "Treat head_top_hairline as hairline carrier, not full independent hair volume.",
            "hands": "Split left_hand and right_hand before local support accounting.",
            "strict_gate": "This contract never implies teacher/candidate pass.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_ready(contract), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
