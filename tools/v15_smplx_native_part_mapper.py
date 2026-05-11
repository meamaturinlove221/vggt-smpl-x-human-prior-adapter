from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from tools.smplx_numpy import (  # noqa: E402
    build_smplx_vertex_features,
    load_smplx_model,
    resolve_smplx_model_path,
)
from v15_common import LOCAL_ROOT, REPORTS, json_ready, scalar_stats, safe_v15_output_dir, utc_now, write_json  # noqa: E402


DEFAULT_SMPLX_ROOT = Path("G:/\u6570\u636e\u96c6/datasets/smplx")
DEFAULT_OUT = LOCAL_ROOT / "V15_SMPLX_native_part_mapper"
DEFAULT_JSON = REPORTS / "20260508_v15_smplx_native_part_mapper.json"
DEFAULT_MD = REPORTS / "20260508_v15_smplx_native_part_mapper.md"

MACRO_GROUPS: dict[str, tuple[str, ...]] = {
    "root_torso": ("Global", "Pelvis", "Spine", "Spine1", "Spine2", "Spine3", "Neck"),
    "head_face": ("Head", "Jaw", "L_Eye", "R_Eye"),
    "left_arm": ("L_Shoulder", "L_Collar", "L_UpperArm", "L_ForeArm", "L_Elbow"),
    "right_arm": ("R_Shoulder", "R_Collar", "R_UpperArm", "R_ForeArm", "R_Elbow"),
    "left_hand": (
        "L_Hand",
        "L_Wrist",
        "L_Index1",
        "L_Index2",
        "L_Index3",
        "L_Middle1",
        "L_Middle2",
        "L_Middle3",
        "L_Pinky1",
        "L_Pinky2",
        "L_Pinky3",
        "L_Ring1",
        "L_Ring2",
        "L_Ring3",
        "L_Thumb1",
        "L_Thumb2",
        "L_Thumb3",
    ),
    "right_hand": (
        "R_Hand",
        "R_Wrist",
        "R_Index1",
        "R_Index2",
        "R_Index3",
        "R_Middle1",
        "R_Middle2",
        "R_Middle3",
        "R_Pinky1",
        "R_Pinky2",
        "R_Pinky3",
        "R_Ring1",
        "R_Ring2",
        "R_Ring3",
        "R_Thumb1",
        "R_Thumb2",
        "R_Thumb3",
    ),
    "left_leg": ("L_Hip", "L_Thigh", "L_Knee", "L_Calf", "L_Ankle", "L_Foot", "L_Toes"),
    "right_leg": ("R_Hip", "R_Thigh", "R_Knee", "R_Calf", "R_Ankle", "R_Foot", "R_Toes"),
}


PART_COLORS = np.asarray(
    [
        [160, 160, 160],
        [82, 128, 255],
        [255, 117, 84],
        [238, 199, 64],
        [152, 100, 230],
        [57, 196, 118],
        [54, 184, 202],
        [225, 94, 160],
    ],
    dtype=np.uint8,
)


def load_object_dict(model_path: Path, key: str) -> dict[str, int]:
    with np.load(model_path, allow_pickle=True) as payload:
        if key not in payload:
            return {}
        try:
            value = payload[key].item()
        except Exception:
            return {}
    return {str(k): int(v) for k, v in value.items()}


def invert_mapping(mapping: dict[str, int]) -> dict[int, str]:
    return {int(value): str(key) for key, value in mapping.items()}


def macro_ids_from_joint_names(joint2num: dict[str, int]) -> tuple[dict[str, list[int]], dict[int, str]]:
    macro_joint_ids: dict[str, list[int]] = {}
    joint_to_macro: dict[int, str] = {}
    for macro_name, names in MACRO_GROUPS.items():
        ids = sorted({int(joint2num[name]) for name in names if name in joint2num})
        macro_joint_ids[macro_name] = ids
        for joint_id in ids:
            joint_to_macro[joint_id] = macro_name
    return macro_joint_ids, joint_to_macro


def face_counts_for_vertex_labels(faces: np.ndarray, labels: np.ndarray, label_names: dict[int, str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    face_labels = labels[np.asarray(faces, dtype=np.int32)]
    for label_id, name in sorted(label_names.items()):
        any_count = int(np.any(face_labels == int(label_id), axis=1).sum())
        all_count = int(np.all(face_labels == int(label_id), axis=1).sum())
        rows[name] = {"face_any_vertex_count": any_count, "face_all_vertices_count": all_count}
    return rows


def save_colored_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = PART_COLORS[np.mod(labels, PART_COLORS.shape[0])]
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {vertices.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {faces.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for vertex, color in zip(vertices, colors):
            handle.write(f"{vertex[0]:.7f} {vertex[1]:.7f} {vertex[2]:.7f} {int(color[0])} {int(color[1])} {int(color[2])}\n")
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    out = safe_v15_output_dir(args.output_dir)
    model_path = resolve_smplx_model_path(args.smplx_root, args.gender)
    model = load_smplx_model(model_path)
    features = build_smplx_vertex_features(
        model_path=model_path,
        betas=np.zeros((int(args.num_betas),), dtype=np.float32),
        expression=np.zeros((int(args.num_expression),), dtype=np.float32),
        body_part_count=int(args.body_part_count),
    )
    weights = np.asarray(model["weights"], dtype=np.float32)
    faces = np.asarray(model["faces"], dtype=np.int32)
    rest_vertices = np.asarray(features["rest_vertices"], dtype=np.float32)
    dominant_joint = weights.argmax(axis=1).astype(np.int64)
    top_weight = weights.max(axis=1)
    sorted_weights = np.sort(weights, axis=1)
    top2_weight = sorted_weights[:, -2] if sorted_weights.shape[1] > 1 else np.zeros_like(top_weight)
    weight_entropy = np.asarray(features["weight_entropy"], dtype=np.float32)

    joint2num = load_object_dict(model_path, "joint2num")
    part2num = load_object_dict(model_path, "part2num")
    num2joint = invert_mapping(joint2num)
    macro_joint_ids, joint_to_macro = macro_ids_from_joint_names(joint2num)
    macro_names = sorted(MACRO_GROUPS)
    macro_name_to_id = {name: idx for idx, name in enumerate(macro_names)}
    vertex_macro = np.full((dominant_joint.shape[0],), macro_name_to_id["root_torso"], dtype=np.int64)
    for idx, joint_id in enumerate(dominant_joint):
        macro = joint_to_macro.get(int(joint_id), "root_torso")
        vertex_macro[idx] = macro_name_to_id[macro]
    vertex_native_cluster = np.asarray(features["vertex_body_part_ids"], dtype=np.int64)

    per_joint: dict[str, Any] = {}
    for joint_id in range(weights.shape[1]):
        mask = dominant_joint == joint_id
        per_joint[str(joint_id)] = {
            "name": num2joint.get(joint_id, f"joint_{joint_id}"),
            "vertex_count": int(mask.sum()),
            "top_weight": scalar_stats(top_weight[mask]) if np.any(mask) else {"count": 0, "finite": 0},
        }
    per_macro: dict[str, Any] = {}
    for name, macro_id in macro_name_to_id.items():
        mask = vertex_macro == macro_id
        per_macro[name] = {
            "macro_id": int(macro_id),
            "joint_ids": macro_joint_ids.get(name, []),
            "vertex_count": int(mask.sum()),
            "vertex_ratio": float(mask.sum() / max(vertex_macro.shape[0], 1)),
            "top_weight": scalar_stats(top_weight[mask]) if np.any(mask) else {"count": 0, "finite": 0},
            "entropy": scalar_stats(weight_entropy[mask]) if np.any(mask) else {"count": 0, "finite": 0},
        }
    macro_face_counts = face_counts_for_vertex_labels(
        faces,
        vertex_macro,
        {idx: name for name, idx in macro_name_to_id.items()},
    )
    native_cluster_names = {int(i): f"native_cluster_{int(i):02d}" for i in np.unique(vertex_native_cluster)}
    native_face_counts = face_counts_for_vertex_labels(faces, vertex_native_cluster, native_cluster_names)

    mapping_npz = out / "v15_smplx_native_part_mapping.npz"
    ply_path = out / "v15_smplx_native_macro_parts.ply"
    np.savez_compressed(
        mapping_npz,
        vertex_macro_part_ids=vertex_macro,
        vertex_native_cluster_ids=vertex_native_cluster,
        dominant_joint_ids=dominant_joint,
        top1_weight=top_weight.astype(np.float32),
        top2_weight=top2_weight.astype(np.float32),
        weight_entropy=weight_entropy.astype(np.float32),
        rest_vertices=rest_vertices,
        faces=faces,
        macro_names=np.asarray(macro_names),
        channel_names=np.asarray(features["channel_names"]),
        research_only=np.asarray(True),
    )
    save_colored_ply(ply_path, rest_vertices, faces, vertex_macro)

    macro_nonempty = sum(1 for row in per_macro.values() if int(row["vertex_count"]) > 0)
    blockers = []
    if int(dominant_joint.shape[0]) != int(rest_vertices.shape[0]):
        blockers.append("Dominant joint IDs are not aligned to the SMPL-X vertex count.")
    if macro_nonempty < 6:
        blockers.append("Too few nonempty native macro body parts were produced.")
    if not joint2num:
        blockers.append("SMPL-X joint2num metadata was not readable from the NPZ.")

    summary = {
        "task": "v15_smplx_native_part_mapper",
        "created_utc": utc_now(),
        "status": "v15_smplx_native_part_map_ready" if not blockers else "v15_smplx_native_part_map_blocked",
        "research_only": True,
        "smplx_native_only": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_strict_pass_claim": True,
        "inputs": {"smplx_root": str(Path(args.smplx_root).resolve()), "model_path": str(model_path), "gender": args.gender},
        "metrics": {
            "vertex_count": int(rest_vertices.shape[0]),
            "face_count": int(faces.shape[0]),
            "joint_count": int(weights.shape[1]),
            "macro_part_count": int(len(macro_names)),
            "macro_nonempty_count": int(macro_nonempty),
            "native_cluster_count": int(len(native_cluster_names)),
            "top1_weight": scalar_stats(top_weight),
            "top2_weight": scalar_stats(top2_weight),
            "weight_entropy": scalar_stats(weight_entropy),
            "static_feature_channel_count": int(np.asarray(features["vertex_features"]).shape[1]),
        },
        "joint2num": joint2num,
        "part2num": part2num,
        "macro_joint_ids": macro_joint_ids,
        "per_joint": per_joint,
        "per_macro": per_macro,
        "macro_face_counts": macro_face_counts,
        "native_cluster_face_counts": native_face_counts,
        "outputs": {"mapping_npz": str(mapping_npz.resolve()), "macro_ply": str(ply_path.resolve()), "summary": str((out / "summary.json").resolve())},
        "decision": (
            "SMPL-X native skinning weights and bundled joint metadata provide a usable body/hand/head part map without MANO, FLAME, SMPL, or HairGS assets."
            if not blockers
            else "SMPL-X native part map did not satisfy alignment/nonempty metadata checks."
        ),
        "blockers": blockers,
    }
    write_json(out / "summary.json", summary)
    return summary


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# V15 SMPL-X Native Part Mapper",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only native part map from SMPL-X skinning weights and joint metadata.",
        "",
        "## Metrics",
        "",
    ]
    for key, value in summary["metrics"].items():
        lines.append(f"- {key}: `{json_ready(value)}`")
    lines.extend(["", "## Macro Parts", "", "| Part | Vertices | Ratio | Joints |", "|---|---:|---:|---|"])
    for name, row in summary["per_macro"].items():
        lines.append(f"| {name} | {row['vertex_count']} | {row['vertex_ratio']:.4f} | {row['joint_ids']} |")
    lines.extend(["", "## Outputs", ""])
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- {item}" for item in summary["blockers"]] if summary["blockers"] else ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="V15 SMPL-X native part mapper from skinning weights.")
    parser.add_argument("--smplx-root", type=Path, default=DEFAULT_SMPLX_ROOT)
    parser.add_argument("--gender", choices=("neutral", "female", "male"), default="neutral")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--num-betas", type=int, default=10)
    parser.add_argument("--num-expression", type=int, default=10)
    parser.add_argument("--body-part-count", type=int, default=12)
    args = parser.parse_args()

    summary = build_summary(args)
    write_json(args.output_json, summary)
    write_markdown(summary, args.output_md)
    print(json.dumps(json_ready({"status": summary["status"], "metrics": summary["metrics"], "output": args.output_dir}), ensure_ascii=False))
    return 0 if not summary["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
