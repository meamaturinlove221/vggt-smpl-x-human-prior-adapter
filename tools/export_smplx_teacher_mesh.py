from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.smplx_numpy import forward_smplx_mesh, resolve_smplx_model_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a posed SMPL-X mesh as a teacher-construction diagnostic. "
            "SMPL-X is only a weak body/head connector; it is not a face-detail teacher."
        )
    )
    parser.add_argument("--smplx-params", required=True, help="NPZ containing betas/fullpose/transl/scale/expression.")
    parser.add_argument("--smplx-model-dir", required=True, help="Directory containing SMPLX_NEUTRAL.npz.")
    parser.add_argument("--smplx-gender", choices=("neutral", "female", "male"), default="neutral")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="smplx_teacher_mesh.ply")
    parser.add_argument(
        "--transform-summary",
        default="",
        help="Optional facelandmark_teacher_summary.json containing camera_meta.export_transform_to_vggt.",
    )
    parser.add_argument(
        "--region",
        choices=("full", "upper", "head"),
        default="full",
        help="Geometry subset exported before optional similarity transform.",
    )
    parser.add_argument("--y-min", type=float, default=None, help="Override real-SMPL-X y lower bound for region crop.")
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
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_params(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        params = {name: payload[name] for name in payload.files}
    required = {"betas", "fullpose"}
    missing = sorted(required - set(params))
    if missing:
        raise ValueError(f"Missing SMPL-X params in {path}: {missing}")
    return params


def load_transform(summary_path: Path) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    camera_meta = summary.get("camera_meta") or {}
    transform = camera_meta.get("export_transform_to_vggt") or camera_meta.get("similarity_real_to_vggt")
    if not isinstance(transform, dict):
        raise ValueError(f"No export_transform_to_vggt found in {summary_path}")
    scale = float(transform["scale"])
    rotation = np.asarray(transform["rotation"], dtype=np.float64)
    translation = np.asarray(transform["translation"], dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(f"Bad transform shapes in {summary_path}: {rotation.shape}, {translation.shape}")
    return scale, rotation, translation, {
        "summary_path": str(summary_path.resolve()),
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
    }


def transform_points(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    points64 = np.asarray(points, dtype=np.float64)
    return (scale * (points64 @ rotation.T) + translation[None, :]).astype(np.float32)


def region_y_min(vertices: np.ndarray, region: str, override: float | None) -> float | None:
    if override is not None:
        return float(override)
    if region == "full":
        return None
    if region == "upper":
        return float(np.percentile(vertices[:, 1], 70.0))
    if region == "head":
        return float(np.percentile(vertices[:, 1], 90.0))
    raise ValueError(region)


def crop_mesh(vertices: np.ndarray, faces: np.ndarray, *, region: str, y_min_override: float | None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y_min = region_y_min(vertices, region, y_min_override)
    if y_min is None:
        return vertices.astype(np.float32), faces.astype(np.int32), {
            "region": region,
            "y_min": None,
            "kept_vertices": int(vertices.shape[0]),
            "kept_faces": int(faces.shape[0]),
        }
    keep_vertex = vertices[:, 1] >= float(y_min)
    keep_face = keep_vertex[faces].any(axis=1)
    used = np.zeros(vertices.shape[0], dtype=bool)
    used[faces[keep_face].reshape(-1)] = True
    old_to_new = np.full(vertices.shape[0], -1, dtype=np.int64)
    old_to_new[used] = np.arange(int(used.sum()), dtype=np.int64)
    new_vertices = vertices[used].astype(np.float32)
    new_faces = old_to_new[faces[keep_face]].astype(np.int32)
    valid_faces = (new_faces >= 0).all(axis=1)
    new_faces = new_faces[valid_faces]
    return new_vertices, new_faces, {
        "region": region,
        "y_min": float(y_min),
        "kept_vertices": int(new_vertices.shape[0]),
        "kept_faces": int(new_faces.shape[0]),
    }


def write_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False, compressed=False)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / str(args.output_name)
    if output_path.exists() and not bool(args.overwrite):
        raise FileExistsError(output_path)

    params_path = Path(args.smplx_params).expanduser().resolve()
    model_path = resolve_smplx_model_path(Path(args.smplx_model_dir).expanduser(), gender=str(args.smplx_gender))
    params = load_params(params_path)
    mesh = forward_smplx_mesh(
        model_path,
        betas=params["betas"],
        expression=params.get("expression"),
        fullpose=params["fullpose"],
        transl=params.get("transl"),
        scale=params.get("scale", 1.0),
    )
    vertices = np.asarray(mesh["vertices"], dtype=np.float32)
    faces = np.asarray(mesh["faces"], dtype=np.int32)
    vertices, faces, crop_meta = crop_mesh(
        vertices,
        faces,
        region=str(args.region),
        y_min_override=args.y_min,
    )
    transform_meta: dict[str, Any] | None = None
    if str(args.transform_summary).strip():
        scale, rotation, translation, transform_meta = load_transform(Path(args.transform_summary).expanduser().resolve())
        vertices = transform_points(vertices, scale, rotation, translation)

    write_mesh(output_path, vertices, faces)
    summary = {
        "task": "export_smplx_teacher_mesh",
        "truthful_status": "weak_connector_only_not_face_teacher",
        "smplx_params": str(params_path),
        "smplx_model_path": str(model_path),
        "output_mesh": str(output_path.resolve()),
        "crop": crop_meta,
        "transform_to_vggt": transform_meta,
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "bounds": {
            "min": vertices.min(axis=0).tolist() if vertices.size else [],
            "max": vertices.max(axis=0).tolist() if vertices.size else [],
        },
        "warning": (
            "SMPL-X can only provide topology/continuity for body, hands, head outline, and neck. "
            "It must not be counted as a real face/hair/detail teacher by itself."
        ),
    }
    (output_dir / "smplx_teacher_mesh_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
