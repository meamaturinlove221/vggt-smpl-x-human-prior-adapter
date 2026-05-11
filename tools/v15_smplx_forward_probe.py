from __future__ import annotations

import argparse
import json
import math
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
    compute_vertex_normals,
    forward_smplx_mesh,
    load_smplx_model,
    resolve_smplx_model_path,
)
from v15_common import LOCAL_ROOT, REPORTS, json_ready, scalar_stats, safe_v15_output_dir, utc_now, write_json  # noqa: E402


DEFAULT_SMPLX_ROOT = Path("G:/\u6570\u636e\u96c6/datasets/smplx")
DEFAULT_OUT = LOCAL_ROOT / "V15_SMPLX_native_forward_probe"
DEFAULT_JSON = REPORTS / "20260508_v15_smplx_forward_probe.json"
DEFAULT_MD = REPORTS / "20260508_v15_smplx_forward_probe.md"


def axis_angle_probe_pose(num_joints: int) -> np.ndarray:
    pose = np.zeros((num_joints, 3), dtype=np.float32)
    if num_joints > 3:
        pose[3, 0] = 0.08
    if num_joints > 6:
        pose[6, 0] = -0.06
    if num_joints > 12:
        pose[12, 1] = 0.05
    if num_joints > 16:
        pose[16, 2] = -0.18
    if num_joints > 17:
        pose[17, 2] = 0.18
    if num_joints > 20:
        pose[20, 1] = -0.22
    if num_joints > 21:
        pose[21, 1] = 0.22
    return pose


def bbox(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "extent": (arr.max(axis=0) - arr.min(axis=0)).tolist(),
        "center": ((arr.max(axis=0) + arr.min(axis=0)) * 0.5).tolist(),
    }


def save_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    normals = np.asarray(normals, dtype=np.float32)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {vertices.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property float nx\nproperty float ny\nproperty float nz\n")
        handle.write(f"element face {faces.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex, normal in zip(vertices, normals):
            handle.write(
                f"{vertex[0]:.7f} {vertex[1]:.7f} {vertex[2]:.7f} {normal[0]:.7f} {normal[1]:.7f} {normal[2]:.7f}\n"
            )
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    out = safe_v15_output_dir(args.output_dir)
    model_path = resolve_smplx_model_path(args.smplx_root, args.gender)
    model = load_smplx_model(model_path)
    num_joints = int(model["parents"].shape[0])
    betas = np.zeros((int(args.num_betas),), dtype=np.float32)
    expression = np.zeros((int(args.num_expression),), dtype=np.float32)
    if expression.size:
        expression[0] = 0.15
    fullpose = axis_angle_probe_pose(num_joints)
    transl = np.asarray([0.0, 0.0, float(args.translation_z)], dtype=np.float32)

    rest_mesh = forward_smplx_mesh(
        model_path=model_path,
        betas=betas,
        expression=expression,
        fullpose=np.zeros_like(fullpose),
        transl=np.zeros(3, dtype=np.float32),
        scale=1.0,
    )
    pose_mesh = forward_smplx_mesh(
        model_path=model_path,
        betas=betas,
        expression=expression,
        fullpose=fullpose,
        transl=transl,
        scale=float(args.scale),
    )

    vertices = np.asarray(pose_mesh["vertices"], dtype=np.float32)
    faces = np.asarray(pose_mesh["faces"], dtype=np.int32)
    joints = np.asarray(pose_mesh["joints"], dtype=np.float32)
    normals = compute_vertex_normals(vertices, faces)
    displacement = vertices - np.asarray(rest_mesh["vertices"], dtype=np.float32)
    normal_lengths = np.linalg.norm(normals, axis=1)
    edge_a = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edge_b = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    face_area = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    ply_path = out / "v15_smplx_forward_probe_neutral.ply"
    npz_path = out / "v15_smplx_forward_probe_mesh.npz"
    save_ply(ply_path, vertices, faces, normals)
    np.savez_compressed(
        npz_path,
        vertices=vertices,
        faces=faces,
        joints=joints,
        normals=normals,
        betas=betas,
        expression=expression,
        fullpose=fullpose,
        transl=transl,
        model_path=np.asarray(str(model_path)),
        research_only=np.asarray(True),
    )

    finite_vertices = bool(np.isfinite(vertices).all())
    finite_joints = bool(np.isfinite(joints).all())
    nondegenerate = int((face_area > 1e-10).sum())
    blockers = []
    if not finite_vertices or not finite_joints:
        blockers.append("Forward output contains non-finite vertices or joints.")
    if nondegenerate <= int(0.95 * faces.shape[0]):
        blockers.append("Too many degenerate SMPL-X faces in forward output.")

    metrics = {
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "joint_count": int(joints.shape[0]),
        "finite_vertices": finite_vertices,
        "finite_joints": finite_joints,
        "nondegenerate_face_count": nondegenerate,
        "pose_nonzero_joint_count": int((np.linalg.norm(fullpose, axis=1) > 0).sum()),
        "bbox_extent": bbox(vertices)["extent"],
        "normal_length": scalar_stats(normal_lengths),
        "face_area": scalar_stats(face_area),
        "pose_displacement_norm": scalar_stats(np.linalg.norm(displacement, axis=1)),
    }
    summary = {
        "task": "v15_smplx_forward_probe",
        "created_utc": utc_now(),
        "status": "v15_smplx_forward_ready" if not blockers else "v15_smplx_forward_blocked",
        "research_only": True,
        "smplx_native_only": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_strict_pass_claim": True,
        "inputs": {"smplx_root": str(Path(args.smplx_root).resolve()), "model_path": str(model_path), "gender": args.gender},
        "parameters": {
            "num_betas": int(args.num_betas),
            "num_expression": int(args.num_expression),
            "scale": float(args.scale),
            "translation_z": float(args.translation_z),
        },
        "model_shapes": {
            "v_template": list(model["v_template"].shape),
            "faces": list(model["faces"].shape),
            "weights": list(model["weights"].shape),
            "J_regressor": list(model["J_regressor"].shape),
            "shapedirs": list(model["shapedirs"].shape),
            "posedirs": list(model["posedirs"].shape),
            "parents": list(model["parents"].shape),
        },
        "metrics": metrics,
        "bbox": bbox(vertices),
        "outputs": {"mesh_npz": str(npz_path.resolve()), "mesh_ply": str(ply_path.resolve()), "summary": str((out / "summary.json").resolve())},
        "decision": (
            "SMPL-X native NumPy forward produced finite vertices, joints, normals, and nondegenerate faces from local licensed SMPL-X assets."
            if not blockers
            else "SMPL-X native forward did not satisfy finite/nondegenerate sanity checks."
        ),
        "blockers": blockers,
    }
    write_json(out / "summary.json", summary)
    return summary


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# V15 SMPL-X Forward Probe",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only native SMPL-X forward probe. No package, cloud job, registry, teacher, candidate, or strict pass is produced.",
        "",
        "## Metrics",
        "",
    ]
    for key, value in summary["metrics"].items():
        lines.append(f"- {key}: `{json_ready(value)}`")
    lines.extend(["", "## Outputs", ""])
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- {item}" for item in summary["blockers"]] if summary["blockers"] else ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="V15 SMPL-X native forward sanity probe.")
    parser.add_argument("--smplx-root", type=Path, default=DEFAULT_SMPLX_ROOT)
    parser.add_argument("--gender", choices=("neutral", "female", "male"), default="neutral")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--num-betas", type=int, default=10)
    parser.add_argument("--num-expression", type=int, default=10)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--translation-z", type=float, default=3.0)
    args = parser.parse_args()

    summary = build_summary(args)
    write_json(args.output_json, summary)
    write_markdown(summary, args.output_md)
    print(json.dumps(json_ready({"status": summary["status"], "metrics": summary["metrics"], "output": args.output_dir}), ensure_ascii=False))
    return 0 if not summary["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
