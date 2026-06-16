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

from b_fus3d_decoder_skeleton_smoke import load_surface_features  # noqa: E402
from preflight_differentiable_renderer_backend import load_connected_mesh  # noqa: E402
from render_open3d_pointcloud import (  # noqa: E402
    _save_open3d_camera_renders,
    _save_open3d_renders,
    _save_projection_fallback,
)


DEFAULT_SURFACE_TOKEN_FEATURES = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D1_surface_token_smoke_hybrid6_layer23/surface_token_features.npz"
)
DEFAULT_TEMPLATE_PAYLOAD = Path(
    "output/normal_line_multiview_20260506/"
    "connected_surface_template_v28_semantic_detail_mouth_nose_fingers/"
    "connected_human_surface_template_payload.npz"
)
DEFAULT_PREDICTIONS = Path("output/local_inference_results/r34_raw518_r27_on6v_fullbody/predictions.npz")
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D3_open3d_precheck_hybrid6_layer23"
)

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
}

CONTRACT = {
    "research_only": True,
    "open3d_precheck_only": True,
    "local_only": True,
    "no_train": True,
    "no_cloud": True,
    "no_candidate_export": True,
    "no_teacher_export": True,
    "no_strict_pass_write": True,
    "writes_predictions_npz": False,
    "writes_prediction_arrays": False,
    "writes_strict_registry": False,
    "writes_candidate": False,
    "writes_teacher": False,
    "writes_checkpoint": False,
}

PART_TO_FAMILY = {
    0: "full_body",
    1: "left_hand",
    2: "right_hand",
    3: "face_core",
    4: "hairline",
    5: "full_body",
}
FALLBACK_FAMILIES = {
    "face_core": ("face_core", "face", "head", "full_body"),
    "hairline": ("hairline", "head", "full_body"),
    "left_hand": ("left_hand", "full_body"),
    "right_hand": ("right_hand", "full_body"),
    "full_body": ("full_body",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only B-Fus3D Open3D proxy precheck. It maps pooled "
            "surface-token evidence onto an existing connected carrier and "
            "renders diagnostic Open3D views. It never writes predictions, "
            "exports a teacher/candidate, writes a strict registry, or calls cloud."
        )
    )
    parser.add_argument("--surface-token-features", type=Path, default=DEFAULT_SURFACE_TOKEN_FEATURES)
    parser.add_argument("--template-payload", type=Path, default=DEFAULT_TEMPLATE_PAYLOAD)
    parser.add_argument("--predictions-npz", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--roi", choices=("full", "head", "face", "hands"), default="full")
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--camera-indices", default="0,1,2,3,4,5")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for piece in str(text).split(","):
        piece = piece.strip()
        if piece:
            out.append(int(piece))
    return out


def write_pointcloud_ply(path: Path, vertices: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float32)
    colors_u8 = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 255.0).astype(np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {vertices.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for vertex, color in zip(vertices, colors_u8, strict=False):
            handle.write(
                f"{float(vertex[0]):.7f} {float(vertex[1]):.7f} {float(vertex[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def write_mesh_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    colors_u8 = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 255.0).astype(np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {vertices.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {faces.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex, color in zip(vertices, colors_u8, strict=False):
            handle.write(
                f"{float(vertex[0]):.7f} {float(vertex[1]):.7f} {float(vertex[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def choose_feature(features: dict[str, np.ndarray], family: str) -> tuple[str, np.ndarray | None]:
    for candidate in FALLBACK_FAMILIES.get(family, (family, "full_body")):
        if candidate in features:
            return candidate, features[candidate]
    return family, None


def feature_palette(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not features:
        return {}
    stats: dict[str, np.ndarray] = {}
    for name, feature in features.items():
        arr = np.asarray(feature, dtype=np.float32).reshape(-1)
        stats[name] = np.asarray(
            [
                float(np.mean(arr)),
                float(np.std(arr)),
                float(np.linalg.norm(arr) / max(np.sqrt(arr.size), 1.0)),
            ],
            dtype=np.float32,
        )
    stacked = np.stack(list(stats.values()), axis=0)
    lo = np.percentile(stacked, 5, axis=0)
    hi = np.percentile(stacked, 95, axis=0)
    palette: dict[str, np.ndarray] = {}
    for name, stat in stats.items():
        rgb = (stat - lo) / np.maximum(hi - lo, 1e-6)
        rgb = 0.20 + 0.75 * np.clip(rgb, 0.0, 1.0)
        palette[name] = (rgb * 255.0).astype(np.float32)
    return palette


def map_vertex_colors(
    part_ids: np.ndarray,
    features: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    part_ids = np.asarray(part_ids, dtype=np.int64).reshape(-1)
    palette = feature_palette(features)
    base = np.asarray([174.0, 174.0, 174.0], dtype=np.float32)
    colors = np.tile(base[None, :], (part_ids.shape[0], 1))
    provenance: dict[str, Any] = {}
    for part_id in sorted(set(int(v) for v in part_ids.tolist())):
        requested_family = PART_TO_FAMILY.get(part_id, "full_body")
        used_family, feature = choose_feature(features, requested_family)
        mask = part_ids == part_id
        if feature is None or used_family not in palette:
            color = base
            status = "missing_feature_using_gray"
        else:
            color = palette[used_family]
            status = "feature_color_mapped"
        colors[mask] = color[None, :]
        provenance[str(part_id)] = {
            "requested_family": requested_family,
            "used_family": used_family,
            "status": status,
            "vertex_count": int(mask.sum()),
        }
    return colors, provenance


def load_prediction_cameras(path: Path | None) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    if path is None:
        return None, None, None
    path = path.expanduser().resolve()
    if not path.is_file():
        return None, None, f"predictions npz not found: {path}"
    with np.load(path, allow_pickle=False) as payload:
        if "extrinsic" not in payload.files or "intrinsic" not in payload.files:
            return None, None, f"predictions npz lacks extrinsic/intrinsic: {path}"
        extrinsic = np.asarray(payload["extrinsic"], dtype=np.float32)
        intrinsic = np.asarray(payload["intrinsic"], dtype=np.float32)
    return extrinsic, intrinsic, None


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Fus3D Open3D Proxy Precheck",
        "",
        "This is a research-only visualization proxy. It is not a B-Fus3D decoder,",
        "not a teacher, not a candidate, and not a strict pass.",
        "",
        "## Strict Truth",
        "",
        "```text",
        f"strict_candidate_passes = {summary['strict_candidate_passes']}",
        f"strict_teacher_passes = {summary['strict_teacher_passes']}",
        f"formal_cloud_train_infer_export = {summary['formal_cloud_train_infer_export']}",
        "```",
        "",
        "## Inputs",
        "",
        f"- surface_token_features: `{summary['inputs']['surface_token_features']}`",
        f"- template_payload: `{summary['inputs']['template_payload']}`",
        f"- predictions_npz: `{summary['inputs'].get('predictions_npz')}`",
        "",
        "## Carrier",
        "",
        "```text",
        f"vertices = {summary['carrier']['vertices']}",
        f"faces = {summary['carrier']['faces']}",
        f"part_ids = {summary['carrier']['part_ids']}",
        "```",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "```json",
            json.dumps(summary["contract"], indent=2, ensure_ascii=False),
            "```",
            "",
            "## Decision",
            "",
            summary["decision"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")

    feature_payload = load_surface_features(args.surface_token_features, [])
    features = feature_payload["features"]
    mesh = load_connected_mesh(args.template_payload)
    vertices = np.asarray(mesh["vertices"], dtype=np.float32)
    faces = np.asarray(mesh["faces"], dtype=np.int32)
    part_ids = np.asarray(mesh["part_ids"], dtype=np.int64)
    colors, provenance = map_vertex_colors(part_ids, features)

    pointcloud_ply = output_dir / "b_fus3d_open3d_precheck_pointcloud.ply"
    mesh_ply = output_dir / "b_fus3d_open3d_precheck_carrier_mesh.ply"
    write_pointcloud_ply(pointcloud_ply, vertices, colors)
    write_mesh_ply(mesh_ply, vertices, faces, colors)

    render_paths: list[str] = []
    fallback_paths: list[str] = []
    camera_render_paths: list[str] = []
    render_error = None
    try:
        render_paths = _save_open3d_renders(
            vertices,
            colors.astype(np.uint8),
            output_dir,
            args.roi,
            int(args.width),
            int(args.height),
            float(args.point_size),
            False,
        )
    except Exception as exc:  # pragma: no cover - Open3D display environment dependent.
        render_error = repr(exc)
        fallback_paths = _save_projection_fallback(
            vertices,
            colors.astype(np.uint8),
            output_dir,
            args.roi,
            int(args.width),
            int(args.height),
        )

    camera_error = None
    extrinsic, intrinsic, camera_load_error = load_prediction_cameras(args.predictions_npz)
    if extrinsic is not None and intrinsic is not None:
        try:
            camera_render_paths = _save_open3d_camera_renders(
                vertices,
                colors.astype(np.uint8),
                extrinsic,
                intrinsic,
                output_dir,
                parse_int_list(args.camera_indices),
                float(args.point_size),
                int(max(args.width, args.height)),
            )
        except Exception as exc:  # pragma: no cover - Open3D camera render can be environment dependent.
            camera_error = repr(exc)
    else:
        camera_error = camera_load_error

    summary: dict[str, Any] = {
        **STRICT_FACTS,
        "task": "b_fus3d_open3d_proxy_precheck",
        "truthful_status": "research_proxy_only_not_candidate_not_teacher",
        "contract": CONTRACT,
        "inputs": {
            "surface_token_features": str(args.surface_token_features.resolve()),
            "template_payload": str(args.template_payload.resolve()),
            "predictions_npz": str(args.predictions_npz.resolve()) if args.predictions_npz else None,
        },
        "feature_payload": {
            "family_names": feature_payload.get("family_names", []),
            "selected_view_indices": feature_payload.get("selected_view_indices", []),
            "patch_start_idx": feature_payload.get("patch_start_idx"),
            "feature_dim": feature_payload.get("feature_dim"),
            "missing_parts": feature_payload.get("missing_parts", []),
        },
        "carrier": {
            "vertices": int(vertices.shape[0]),
            "faces": int(faces.shape[0]),
            "part_ids": sorted(int(v) for v in set(part_ids.tolist())),
            "part_feature_provenance": provenance,
        },
        "outputs": {
            "pointcloud_ply": str(pointcloud_ply),
            "carrier_mesh_ply": str(mesh_ply),
            "open3d_renders": render_paths,
            "camera_renders": camera_render_paths,
            "fallback_renders": fallback_paths,
        },
        "errors": {
            "open3d_render_error": render_error,
            "camera_render_error": camera_error,
        },
        "decision": (
            "B-Fus3D token evidence can be visualized on the connected carrier. "
            "This is only a proxy precheck for coverage/provenance; it does not "
            "decode SDF/surface geometry, does not improve the carrier, and cannot "
            "satisfy mentor full/head/face/hairline/hand Open3D gates."
        ),
    }
    summary = json_ready(summary)
    (output_dir / "b_fus3d_open3d_precheck_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "b_fus3d_open3d_precheck_summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
