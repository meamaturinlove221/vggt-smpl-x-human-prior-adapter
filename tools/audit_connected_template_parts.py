from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
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

REQUIRED_CARRIERS = {
    "head_face": ("part", ("head_face",)),
    "head_top_hairline": ("part", ("head_top_hairline", "head_top_hairline_proxy")),
    "left_hand": ("part", ("left_hand",)),
    "right_hand": ("part", ("right_hand",)),
    "lower_clothing": ("part", ("lower_clothing", "lower_clothing_proxy")),
    "face_front_proxy": ("mask", ("face_front_vertex_mask",)),
}


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, value: int) -> int:
        parent = self.parent
        root = int(value)
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[value]) != value:
            nxt = int(parent[value])
            parent[value] = root
            value = nxt
        return root

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        rank_left = int(self.rank[root_left])
        rank_right = int(self.rank[root_right])
        if rank_left < rank_right:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if rank_left == rank_right:
            self.rank[root_left] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only connected human template part audit. It inspects a template "
            "payload NPZ, reports mesh/part/mask/component/bbox support, and writes "
            "markdown/json diagnostics for A4.1/B2 carrier repair planning. It does "
            "not run optimization and does not write strict-pass state."
        )
    )
    parser.add_argument("--payload", required=True, type=Path, help="connected_human_surface_template_payload.npz")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name", default=None, help="Report stem. Defaults to payload parent name.")
    parser.add_argument(
        "--mesh",
        choices=("auto", "hybrid", "base"),
        default="auto",
        help="Mesh to audit. Auto chooses the mesh whose vertex count matches part_ids.",
    )
    parser.add_argument("--top-components", type=int, default=6)
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe or "connected_template_parts_audit"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def load_npz(path: Path) -> dict[str, np.ndarray]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def array_inventory(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(arrays):
        arr = arrays[key]
        rows.append(
            {
                "key": key,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "size": int(arr.size),
            }
        )
    return rows


def is_vertices(value: Any) -> bool:
    arr = np.asarray(value)
    return arr.ndim == 2 and arr.shape[1] == 3 and np.issubdtype(arr.dtype, np.number)


def is_faces(value: Any) -> bool:
    arr = np.asarray(value)
    return arr.ndim == 2 and arr.shape[1] == 3 and np.issubdtype(arr.dtype, np.integer)


def coerce_faces(value: np.ndarray, vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    faces = np.asarray(value, dtype=np.int64)
    valid = (
        (faces.ndim == 2)
        and (faces.shape[1] == 3)
        and np.all(faces >= 0, axis=1)
        and np.all(faces < int(vertex_count), axis=1)
    )
    if isinstance(valid, np.ndarray):
        return faces, valid
    return faces, np.zeros((0,), dtype=bool)


def pick_mesh(arrays: dict[str, np.ndarray], requested: str) -> dict[str, Any]:
    part_ids = np.asarray(arrays["part_ids"], dtype=np.int64) if "part_ids" in arrays else None
    candidates: dict[str, dict[str, Any]] = {}
    if "vertices" in arrays and "faces" in arrays and is_vertices(arrays["vertices"]) and is_faces(arrays["faces"]):
        vertices = np.asarray(arrays["vertices"], dtype=np.float64)
        faces, valid = coerce_faces(arrays["faces"], vertices.shape[0])
        candidates["base"] = {
            "name": "base",
            "vertices_key": "vertices",
            "faces_key": "faces",
            "vertices": vertices,
            "faces": faces,
            "valid_face_mask": valid,
        }
    if (
        "hybrid_vertices" in arrays
        and "hybrid_faces" in arrays
        and is_vertices(arrays["hybrid_vertices"])
        and is_faces(arrays["hybrid_faces"])
    ):
        vertices = np.asarray(arrays["hybrid_vertices"], dtype=np.float64)
        faces, valid = coerce_faces(arrays["hybrid_faces"], vertices.shape[0])
        candidates["hybrid"] = {
            "name": "hybrid",
            "vertices_key": "hybrid_vertices",
            "faces_key": "hybrid_faces",
            "vertices": vertices,
            "faces": faces,
            "valid_face_mask": valid,
        }
    if requested != "auto":
        if requested not in candidates:
            raise KeyError(f"Requested {requested!r} mesh is not available in payload")
        chosen = candidates[requested]
    elif part_ids is not None:
        matching = [
            candidate
            for candidate in candidates.values()
            if int(candidate["vertices"].shape[0]) == int(part_ids.shape[0])
        ]
        if matching:
            chosen = sorted(matching, key=lambda item: 0 if item["name"] == "hybrid" else 1)[0]
        elif "hybrid" in candidates:
            chosen = candidates["hybrid"]
        elif "base" in candidates:
            chosen = candidates["base"]
        else:
            raise KeyError("Payload has no valid vertices/faces mesh pair")
    elif "hybrid" in candidates:
        chosen = candidates["hybrid"]
    elif "base" in candidates:
        chosen = candidates["base"]
    else:
        raise KeyError("Payload has no valid vertices/faces mesh pair")

    out = dict(chosen)
    out["available_meshes"] = {
        name: {
            "vertices_key": candidate["vertices_key"],
            "faces_key": candidate["faces_key"],
            "vertices": int(candidate["vertices"].shape[0]),
            "faces": int(candidate["faces"].shape[0]),
            "valid_faces": int(np.asarray(candidate["valid_face_mask"], dtype=bool).sum()),
        }
        for name, candidate in candidates.items()
    }
    out["part_ids"] = part_ids
    out["part_ids_aligned"] = part_ids is not None and int(part_ids.shape[0]) == int(out["vertices"].shape[0])
    return out


def decode_part_names(arrays: dict[str, np.ndarray], unique_part_ids: list[int]) -> tuple[dict[int, str], str, bool]:
    for key in ("part_names", "part_name_map", "part_labels"):
        if key not in arrays:
            continue
        value = arrays[key]
        try:
            decoded = value.item() if value.shape == () else value.tolist()
        except Exception:
            decoded = value
        mapping: dict[int, str] = {}
        if isinstance(decoded, dict):
            for raw_key, raw_value in decoded.items():
                try:
                    mapping[int(raw_key)] = str(raw_value)
                except (TypeError, ValueError):
                    continue
        elif isinstance(decoded, list):
            if all(isinstance(item, str) for item in decoded):
                mapping = {idx: str(name) for idx, name in enumerate(decoded)}
            else:
                for item in decoded:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        try:
                            mapping[int(item[0])] = str(item[1])
                        except (TypeError, ValueError):
                            continue
        if mapping:
            for part_id in unique_part_ids:
                mapping.setdefault(part_id, f"part_{part_id}")
            return mapping, key, True

    mapping = {part_id: KNOWN_PART_NAMES.get(part_id, f"part_{part_id}") for part_id in unique_part_ids}
    return mapping, "known_connected_template_fallback", False


def bbox_stats(vertices: np.ndarray) -> dict[str, Any] | None:
    if vertices.size == 0:
        return None
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    span = bbox_max - bbox_min
    return {
        "min": [float(v) for v in bbox_min],
        "max": [float(v) for v in bbox_max],
        "center": [float(v) for v in 0.5 * (bbox_min + bbox_max)],
        "span": [float(v) for v in span],
        "volume": float(np.prod(np.maximum(span, 0.0))),
    }


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return np.zeros((0,), dtype=np.float64)
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def component_summary(
    vertex_count: int,
    faces: np.ndarray,
    selector: np.ndarray | None = None,
    top_components: int = 6,
) -> dict[str, Any]:
    if selector is None:
        selected = np.arange(vertex_count, dtype=np.int64)
        local_index = np.arange(vertex_count, dtype=np.int64)
        local_faces = faces
    else:
        selector = np.asarray(selector, dtype=bool)
        selected = np.flatnonzero(selector).astype(np.int64)
        local_index = np.full((vertex_count,), -1, dtype=np.int64)
        local_index[selected] = np.arange(selected.shape[0], dtype=np.int64)
        if faces.size == 0:
            local_faces = np.zeros((0, 3), dtype=np.int64)
        else:
            face_local = local_index[faces]
            local_faces = face_local[np.all(face_local >= 0, axis=1)]

    if selected.shape[0] == 0:
        return {
            "component_count": 0,
            "selected_vertices": 0,
            "internal_faces": 0,
            "largest_vertices": 0,
            "top_components": [],
        }

    dsu = DisjointSet(int(selected.shape[0]))
    for face in local_faces:
        left, mid, right = (int(face[0]), int(face[1]), int(face[2]))
        dsu.union(left, mid)
        dsu.union(mid, right)

    root_vertex_counts: Counter[int] = Counter()
    for local_id in range(int(selected.shape[0])):
        root_vertex_counts[dsu.find(local_id)] += 1

    root_face_counts: Counter[int] = Counter()
    for face in local_faces:
        root_face_counts[dsu.find(int(face[0]))] += 1

    components = [
        {
            "component_index": idx,
            "vertices": int(vertex_num),
            "faces": int(root_face_counts.get(root, 0)),
        }
        for idx, (root, vertex_num) in enumerate(root_vertex_counts.most_common())
    ]
    return {
        "component_count": int(len(components)),
        "selected_vertices": int(selected.shape[0]),
        "internal_faces": int(local_faces.shape[0]),
        "largest_vertices": int(components[0]["vertices"]) if components else 0,
        "small_components_le_8_vertices": int(sum(1 for item in components if int(item["vertices"]) <= 8)),
        "top_components": components[: int(top_components)],
    }


def face_majority_part_ids(face_part_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    majority = np.zeros((face_part_ids.shape[0],), dtype=np.int64)
    mixed = np.zeros((face_part_ids.shape[0],), dtype=bool)
    for idx, row in enumerate(face_part_ids):
        values, counts = np.unique(row, return_counts=True)
        majority[idx] = int(values[int(np.argmax(counts))])
        mixed[idx] = bool(values.shape[0] > 1)
    return majority, mixed


def summarize_parts(
    vertices: np.ndarray,
    faces: np.ndarray,
    valid_face_mask: np.ndarray,
    part_ids: np.ndarray,
    part_names: dict[int, str],
    top_components: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    valid_faces = faces[np.asarray(valid_face_mask, dtype=bool)]
    areas = triangle_areas(vertices, valid_faces)
    face_part_ids = part_ids[valid_faces] if valid_faces.size else np.zeros((0, 3), dtype=np.int64)
    majority, mixed = face_majority_part_ids(face_part_ids) if face_part_ids.size else (
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=bool),
    )
    unique_part_ids = sorted(int(part_id) for part_id in np.unique(part_ids).tolist())
    parts: dict[str, Any] = {}
    for part_id in unique_part_ids:
        selector = part_ids == part_id
        touching = np.any(face_part_ids == part_id, axis=1) if face_part_ids.size else np.zeros((0,), dtype=bool)
        internal = np.all(face_part_ids == part_id, axis=1) if face_part_ids.size else np.zeros((0,), dtype=bool)
        majority_mask = majority == part_id
        part_key = str(part_id)
        parts[part_key] = {
            "part_id": part_id,
            "part_name": part_names.get(part_id, f"part_{part_id}"),
            "family": KNOWN_PART_FAMILIES.get(part_id, "unknown"),
            "vertex_count": int(selector.sum()),
            "vertex_fraction": float(selector.sum() / max(1, part_ids.shape[0])),
            "touching_face_count": int(touching.sum()),
            "internal_face_count": int(internal.sum()),
            "mixed_touching_face_count": int((touching & ~internal).sum()),
            "majority_face_count": int(majority_mask.sum()),
            "majority_area": float(areas[majority_mask].sum()) if areas.size else 0.0,
            "internal_area": float(areas[internal].sum()) if areas.size else 0.0,
            "bbox": bbox_stats(vertices[selector]),
            "components": component_summary(
                vertex_count=vertices.shape[0],
                faces=valid_faces,
                selector=selector,
                top_components=top_components,
            ),
        }

    face_stats = {
        "valid_face_count": int(valid_faces.shape[0]),
        "invalid_face_count": int(faces.shape[0] - valid_faces.shape[0]),
        "mixed_part_face_count": int(mixed.sum()),
        "mixed_part_face_fraction": float(mixed.sum() / max(1, valid_faces.shape[0])),
        "majority_face_counts_by_part": {
            str(part_id): int((majority == part_id).sum()) for part_id in unique_part_ids
        },
    }
    return parts, face_stats


def summarize_masks(
    arrays: dict[str, np.ndarray],
    vertices: np.ndarray,
    faces: np.ndarray,
    valid_face_mask: np.ndarray,
    part_ids: np.ndarray | None,
    part_names: dict[int, str],
    top_components: int,
) -> dict[str, Any]:
    valid_faces = faces[np.asarray(valid_face_mask, dtype=bool)]
    masks: dict[str, Any] = {}
    for key in sorted(arrays):
        if not key.endswith("_vertex_mask"):
            continue
        raw = np.asarray(arrays[key])
        if raw.shape != (vertices.shape[0],):
            masks[key] = {
                "usable": False,
                "shape": list(raw.shape),
                "expected_shape": [int(vertices.shape[0])],
            }
            continue
        selector = raw.astype(bool)
        overlap: dict[str, int] = {}
        if part_ids is not None and part_ids.shape[0] == vertices.shape[0]:
            for part_id in sorted(int(value) for value in np.unique(part_ids[selector]).tolist()):
                name = part_names.get(part_id, f"part_{part_id}")
                overlap[f"{part_id}:{name}"] = int(((part_ids == part_id) & selector).sum())
        masks[key] = {
            "usable": True,
            "vertex_count": int(selector.sum()),
            "vertex_fraction": float(selector.sum() / max(1, vertices.shape[0])),
            "bbox": bbox_stats(vertices[selector]),
            "part_overlap": overlap,
            "components": component_summary(
                vertex_count=vertices.shape[0],
                faces=valid_faces,
                selector=selector,
                top_components=top_components,
            ),
        }
    return masks


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def carrier_presence(
    part_summaries: dict[str, Any],
    masks: dict[str, Any],
    part_names_embedded: bool,
) -> dict[str, Any]:
    name_to_ids: dict[str, list[int]] = {}
    for part in part_summaries.values():
        name_to_ids.setdefault(normalized(str(part["part_name"])), []).append(int(part["part_id"]))

    presence: dict[str, Any] = {}
    for feature, (kind, aliases) in REQUIRED_CARRIERS.items():
        if kind == "part":
            matching_ids: list[int] = []
            for alias in aliases:
                alias_norm = normalized(alias)
                matching_ids.extend(name_to_ids.get(alias_norm, []))
            unique_ids = sorted(set(matching_ids))
            vertex_count = int(
                sum(int(part_summaries[str(part_id)]["vertex_count"]) for part_id in unique_ids if str(part_id) in part_summaries)
            )
            presence[feature] = {
                "present": bool(unique_ids),
                "source": "part_ids",
                "part_ids": unique_ids,
                "vertex_count": vertex_count,
            }
        else:
            matching_keys = [alias for alias in aliases if alias in masks and masks[alias].get("usable")]
            presence[feature] = {
                "present": bool(matching_keys),
                "source": "vertex_mask",
                "mask_keys": matching_keys,
                "vertex_count": int(sum(int(masks[key].get("vertex_count", 0)) for key in matching_keys)),
            }

    left_present = bool(presence.get("left_hand", {}).get("present"))
    right_present = bool(presence.get("right_hand", {}).get("present"))
    presence["hands_pair"] = {
        "present": left_present and right_present,
        "source": "part_ids",
        "part_ids": sorted(
            set(presence.get("left_hand", {}).get("part_ids", []) + presence.get("right_hand", {}).get("part_ids", []))
        ),
        "vertex_count": int(presence.get("left_hand", {}).get("vertex_count", 0))
        + int(presence.get("right_hand", {}).get("vertex_count", 0)),
    }
    presence["payload_part_names"] = {
        "present": bool(part_names_embedded),
        "source": "payload_key",
        "note": "part_names were embedded in the NPZ" if part_names_embedded else "part names were inferred by known template fallback",
    }
    return presence


def build_recommendations(audit: dict[str, Any]) -> dict[str, list[str]]:
    mesh = audit["mesh"]
    presence = audit["carrier_presence"]
    part_ids_ok = bool(mesh.get("part_ids_aligned"))

    payload_recs: list[str] = []
    a41_recs: list[str] = []
    b2_recs: list[str] = []

    if part_ids_ok:
        a41_recs.append(
            "Use the aligned hybrid part_ids and vertex masks as ROI carriers instead of global x/y body heuristics; compute per-view projected bboxes from selected part vertices."
        )
        a41_recs.append(
            "Split hand ROIs by left_hand/right_hand part ids, then clip each ROI to its projected mask/bbox so hand support cannot expand over the torso or lower clothing."
        )
    else:
        a41_recs.append(
            "Repair payload alignment first: A4.1 cannot build trustworthy local ROIs until part_ids match the audited vertex array."
        )

    if presence.get("head_face", {}).get("present") and presence.get("face_front_proxy", {}).get("present"):
        a41_recs.append(
            "For face ROI, prefer face_front_vertex_mask for front-face fitting and head_face for broader head support; keep head_top_hairline separate from face residuals."
        )
    if presence.get("head_top_hairline", {}).get("present"):
        a41_recs.append(
            "Treat head_top_hairline as the current hair/hairline carrier; if A4.1 needs full hair volume, add an explicit full_hair mask/family rather than merging it into head_face."
        )
    if presence.get("lower_clothing", {}).get("present"):
        a41_recs.append(
            "Exclude lower_clothing_proxy from head/hand ROI seeds; it is a useful body/lower-clothing carrier but should not be used as a generic body heuristic fallback."
        )

    if part_ids_ok:
        b2_recs.append(
            "Feed hybrid part_ids into B2 token/raster support accounting and reserve minimum raster-visible tokens for face, each hand, and hairline before global body tokens are accepted."
        )
        b2_recs.append(
            "Report projected-vs-raster support per family and per part; block local-carrier claims when face/hand tokens have projected support but no raster pixels."
        )
        b2_recs.append(
            "Use part-local bboxes/masks for token sampling so face and hands cannot be washed out by torso/lower-clothing majority faces."
        )
    else:
        b2_recs.append(
            "Do not consume this payload in B2 until part_ids align with the mesh used by the renderer/tokenizer."
        )

    if not presence.get("payload_part_names", {}).get("present"):
        payload_recs.append(
            "Embed part_names or part_name_map in the NPZ. The ids are usable here only because the audit knows the connected-template mapping."
        )
    payload_recs.append(
        "Embed part_families and canonical required masks (face_front, head_face, head_top_hairline, left_hand, right_hand, lower_clothing) so downstream tools do not hardcode ids."
    )
    payload_recs.append(
        "Keep hybrid_vertices/hybrid_faces as the default carrier mesh when part_ids length matches hybrid_vertices; base vertices/faces are not part-id aligned in this payload."
    )

    missing = [name for name, item in presence.items() if name != "payload_part_names" and not item.get("present")]
    if missing:
        payload_recs.append(f"Add missing required carrier features before claiming local carrier readiness: {', '.join(sorted(missing))}.")

    return {
        "a4_1_carrier_repair": a41_recs,
        "b2_carrier_repair": b2_recs,
        "payload_hygiene": payload_recs,
    }


def fmt_int(value: Any) -> str:
    return f"{int(value):,}"


def fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def fmt_vec(values: Any, digits: int = 4) -> str:
    if values is None:
        return "n/a"
    return "[" + ", ".join(f"{float(value):.{digits}f}" for value in values) + "]"


def render_md(audit: dict[str, Any]) -> str:
    mesh = audit["mesh"]
    lines: list[str] = [
        "# Connected Template Parts Audit",
        "",
        "Status: `read_only_diagnostic_no_optimization_no_strict_pass`",
        "",
        f"- Payload: `{audit['payload']}`",
        f"- Mesh audited: `{mesh['selected_mesh']}` using `{mesh['vertices_key']}` / `{mesh['faces_key']}`",
        f"- Vertices/faces: `{fmt_int(mesh['vertex_count'])}` / `{fmt_int(mesh['face_count'])}`",
        f"- Valid faces: `{fmt_int(mesh['valid_face_count'])}`; invalid faces: `{fmt_int(mesh['invalid_face_count'])}`",
        f"- part_ids aligned with audited vertices: `{mesh['part_ids_aligned']}`",
        f"- part_names embedded in payload: `{audit['part_names']['embedded_in_payload']}` (`{audit['part_names']['source']}`)",
        "",
        "## Payload Inventory",
        "",
        "| key | shape | dtype |",
        "| --- | ---: | --- |",
    ]
    for item in audit["payload_inventory"]:
        lines.append(f"| `{item['key']}` | `{item['shape']}` | `{item['dtype']}` |")

    lines.extend(
        [
            "",
            "## Available Meshes",
            "",
            "| mesh | vertices key | faces key | vertices | faces | valid faces |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for name, item in audit["available_meshes"].items():
        lines.append(
            f"| `{name}` | `{item['vertices_key']}` | `{item['faces_key']}` | "
            f"{fmt_int(item['vertices'])} | {fmt_int(item['faces'])} | {fmt_int(item['valid_faces'])} |"
        )

    face_stats = audit["face_part_stats"]
    lines.extend(
        [
            "",
            "## Part Distribution And Components",
            "",
            f"Mixed-part faces: `{fmt_int(face_stats['mixed_part_face_count'])}` / `{fmt_int(face_stats['valid_face_count'])}` ({fmt_pct(face_stats['mixed_part_face_fraction'])}).",
            "",
            "| id | name | family | vertices | majority faces | internal faces | mixed touching faces | components | largest | bbox min | bbox max |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for part_id in sorted((int(key) for key in audit["parts"].keys())):
        part = audit["parts"][str(part_id)]
        comp = part["components"]
        bbox = part.get("bbox") or {}
        lines.append(
            f"| {part_id} | `{part['part_name']}` | `{part['family']}` | "
            f"{fmt_int(part['vertex_count'])} ({fmt_pct(part['vertex_fraction'])}) | "
            f"{fmt_int(part['majority_face_count'])} | {fmt_int(part['internal_face_count'])} | "
            f"{fmt_int(part['mixed_touching_face_count'])} | {fmt_int(comp['component_count'])} | "
            f"{fmt_int(comp['largest_vertices'])} | `{fmt_vec(bbox.get('min'))}` | `{fmt_vec(bbox.get('max'))}` |"
        )

    lines.extend(
        [
            "",
            "## Required Carrier Presence",
            "",
            "| carrier | present | source | ids/keys | vertices |",
            "| --- | --- | --- | --- | ---: |",
        ]
    )
    for name in sorted(audit["carrier_presence"]):
        item = audit["carrier_presence"][name]
        ids_or_keys = item.get("part_ids") or item.get("mask_keys") or item.get("note") or ""
        if isinstance(ids_or_keys, list):
            ids_or_keys = ", ".join(str(value) for value in ids_or_keys)
        lines.append(
            f"| `{name}` | `{item.get('present')}` | `{item.get('source', '')}` | `{ids_or_keys}` | "
            f"{fmt_int(item.get('vertex_count', 0)) if 'vertex_count' in item else 'n/a'} |"
        )

    if audit["masks"]:
        lines.extend(
            [
                "",
                "## Vertex Masks",
                "",
                "| mask | usable | vertices | components | overlap | bbox min | bbox max |",
                "| --- | --- | ---: | ---: | --- | --- | --- |",
            ]
        )
        for key, mask in audit["masks"].items():
            if not mask.get("usable"):
                lines.append(f"| `{key}` | `False` | n/a | n/a | shape `{mask.get('shape')}` | n/a | n/a |")
                continue
            overlap = ", ".join(f"{name}={count}" for name, count in mask.get("part_overlap", {}).items())
            bbox = mask.get("bbox") or {}
            lines.append(
                f"| `{key}` | `True` | {fmt_int(mask['vertex_count'])} ({fmt_pct(mask['vertex_fraction'])}) | "
                f"{fmt_int(mask['components']['component_count'])} | `{overlap}` | "
                f"`{fmt_vec(bbox.get('min'))}` | `{fmt_vec(bbox.get('max'))}` |"
            )

    lines.extend(
        [
            "",
            "## Assessment",
            "",
            f"- Payload has usable aligned part ids for the audited `{mesh['selected_mesh']}` mesh: `{mesh['part_ids_aligned']}`.",
            "- The payload is strong enough to diagnose local carrier support, but it is not self-describing because `part_names`/`part_families` are not embedded.",
            "- For downstream A4.1/B2 repair, this audit treats `head_top_hairline` as a hairline carrier, not proof of a full independent hair component.",
            "",
            "## A4.1 Carrier Repair Suggestions",
            "",
        ]
    )
    for item in audit["recommendations"]["a4_1_carrier_repair"]:
        lines.append(f"- {item}")
    lines.extend(["", "## B2 Carrier Repair Suggestions", ""])
    for item in audit["recommendations"]["b2_carrier_repair"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Payload Hygiene Suggestions", ""])
    for item in audit["recommendations"]["payload_hygiene"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- JSON: `{audit['outputs']['json']}`",
            f"- Markdown: `{audit['outputs']['markdown']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    payload_path = args.payload.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    name = sanitize_name(args.name or f"{payload_path.parent.name}_parts_audit")
    json_path = output_dir / f"{name}.json"
    md_path = output_dir / f"{name}.md"

    arrays = load_npz(payload_path)
    mesh = pick_mesh(arrays, args.mesh)
    vertices = np.asarray(mesh["vertices"], dtype=np.float64)
    faces = np.asarray(mesh["faces"], dtype=np.int64)
    valid_face_mask = np.asarray(mesh["valid_face_mask"], dtype=bool)
    part_ids = mesh.get("part_ids")
    part_ids_aligned = bool(mesh.get("part_ids_aligned"))

    if part_ids is not None and part_ids_aligned:
        part_ids = np.asarray(part_ids, dtype=np.int64)
        unique_part_ids = sorted(int(part_id) for part_id in np.unique(part_ids).tolist())
        part_names, part_name_source, part_names_embedded = decode_part_names(arrays, unique_part_ids)
        part_summaries, face_part_stats = summarize_parts(
            vertices=vertices,
            faces=faces,
            valid_face_mask=valid_face_mask,
            part_ids=part_ids,
            part_names=part_names,
            top_components=args.top_components,
        )
    else:
        unique_part_ids = []
        part_names = {}
        part_name_source = "unavailable"
        part_names_embedded = False
        part_summaries = {}
        face_part_stats = {
            "valid_face_count": int(valid_face_mask.sum()),
            "invalid_face_count": int(faces.shape[0] - valid_face_mask.sum()),
            "mixed_part_face_count": None,
            "mixed_part_face_fraction": None,
            "majority_face_counts_by_part": {},
        }

    masks = summarize_masks(
        arrays=arrays,
        vertices=vertices,
        faces=faces,
        valid_face_mask=valid_face_mask,
        part_ids=part_ids if part_ids_aligned else None,
        part_names=part_names,
        top_components=args.top_components,
    )
    presence = carrier_presence(part_summaries, masks, part_names_embedded)

    audit: dict[str, Any] = {
        "status": "read_only_diagnostic_no_optimization_no_strict_pass",
        "payload": str(payload_path),
        "payload_inventory": array_inventory(arrays),
        "available_meshes": mesh["available_meshes"],
        "mesh": {
            "selected_mesh": mesh["name"],
            "vertices_key": mesh["vertices_key"],
            "faces_key": mesh["faces_key"],
            "vertex_count": int(vertices.shape[0]),
            "face_count": int(faces.shape[0]),
            "valid_face_count": int(valid_face_mask.sum()),
            "invalid_face_count": int(faces.shape[0] - valid_face_mask.sum()),
            "bbox": bbox_stats(vertices),
            "part_ids_present": "part_ids" in arrays,
            "part_ids_length": int(arrays["part_ids"].shape[0]) if "part_ids" in arrays else None,
            "part_ids_aligned": part_ids_aligned,
            "global_components": component_summary(
                vertex_count=vertices.shape[0],
                faces=faces[valid_face_mask],
                selector=None,
                top_components=args.top_components,
            ),
        },
        "part_names": {
            "embedded_in_payload": bool(part_names_embedded),
            "source": part_name_source,
            "mapping": part_names,
        },
        "parts": part_summaries,
        "face_part_stats": face_part_stats,
        "masks": masks,
        "carrier_presence": presence,
        "outputs": {
            "json": str(json_path),
            "markdown": str(md_path),
        },
    }
    audit["recommendations"] = build_recommendations(audit)

    write_json(json_path, audit)
    write_text(md_path, render_md(audit))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
