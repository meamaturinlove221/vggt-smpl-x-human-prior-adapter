from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from prepare_4k4d_prior_training_case import (  # noqa: E402
    align_intrinsics_for_scene_view,
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
)
from research_scene_assets import load_camera_params_sidecar, localize_scene_manifest_paths  # noqa: E402


STRICT_CANDIDATE_PASSES = 0
STRICT_TEACHER_PASSES = 0

METHODS: dict[str, dict[str, Any]] = {
    "A1_neural_sdf": {
        "label": "A1 known-camera neural SDF / NeuS-like research wrapper",
        "input_contract": ["raw_rgb", "mask", "known_camera", "same_frame_views", "optional_A3_visual_hull_seed"],
        "actual_runtime_required": ["torch", "skimage"],
        "actual_runtime_optional": ["tinycudann", "trimesh", "open3d", "nerfstudio"],
        "seed_use": "Use the A3 visual hull mesh only to initialize/bound the SDF volume.",
        "research_recipe": [
            "Load same-frame RGB, masks, and aligned known cameras from the scene manifest.",
            "Initialize the SDF bounding volume from the A3 visual-hull mesh bbox and camera-covered masks.",
            "Future research job may optimize silhouette and photometric consistency internally.",
            "Future mesh extraction must stay under a research artifact directory until separate review.",
        ],
        "blocked_exports": ["teacher mesh", "candidate predictions", "strict pass record", "training case mutation"],
    },
    "A2_gaussian_surface": {
        "label": "A2 known-camera Gaussian/2DGS surface research wrapper",
        "input_contract": ["raw_rgb", "mask", "known_camera", "same_frame_views", "optional_A3_visual_hull_seed"],
        "actual_runtime_required": ["torch"],
        "actual_runtime_alternatives": [["gsplat", "diff_gaussian_rasterization"]],
        "actual_runtime_optional": ["trimesh", "open3d"],
        "seed_use": "Use the A3 visual hull mesh only to place initial surfels/Gaussians inside the mask-consistent volume.",
        "research_recipe": [
            "Load same-frame RGB, masks, and aligned known cameras from the scene manifest.",
            "Sample initial research-only surfel/Gaussian carriers from the A3 visual-hull seed.",
            "Future research job may optimize mask coverage, color consistency, and surface regularization.",
            "Future surface extraction must remain a research artifact until separate review.",
        ],
        "blocked_exports": ["teacher mesh", "candidate predictions", "strict pass record", "training case mutation"],
    },
}

FORBIDDEN_ROUTES = [
    "r-number hyperparameter tuning",
    "VGGT depth/point/normal shell recovery",
    "Kinect/TSDF/signfix teacher",
    "threshold/support iteration loop",
    "formal VGGT cloud train/infer/export",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "A-line A1/A2 dense reconstruction research-preflight stub. "
            "It audits inputs/dependencies and writes a dry research recipe only."
        )
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a3-seed-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--view-indices", default="0,10,20,30,40,50")
    parser.add_argument("--target-size", type=int, default=96)
    parser.add_argument("--methods", default="A1_neural_sdf,A2_gaussian_surface")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_view_indices(spec: str, view_count: int) -> list[int]:
    out: list[int] = []
    for raw in str(spec).split(","):
        item = raw.strip()
        if not item:
            continue
        value = int(item)
        if value < 0:
            value = view_count + value
        if value < 0 or value >= view_count:
            raise IndexError(f"view index {raw} resolved to {value}, outside [0, {view_count})")
        out.append(value)
    if not out:
        out = list(range(min(6, view_count)))
    return sorted(dict.fromkeys(out))


def align_intrinsics_for_loaded_scene_view(intrinsic: np.ndarray, view: dict[str, Any], target_size: int) -> np.ndarray:
    image_size = view.get("image_size") or [target_size, target_size]
    native_size = int(image_size[0]) if len(image_size) >= 1 else int(target_size)
    meta = view.get("preprocess_meta") or {}
    if meta.get("transform") == "crop_pad_to_square" and native_size != int(target_size):
        native = align_intrinsics_for_scene_view(intrinsic, view, target_size=native_size)
        scale = float(target_size) / float(max(1, native_size))
        out = native.astype(np.float32).copy()
        out[0, :] *= scale
        out[1, :] *= scale
        return out
    return align_intrinsics_for_scene_view(intrinsic, view, target_size=target_size)


def find_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def dependency_probe(methods: list[str]) -> dict[str, Any]:
    modules = {
        "numpy",
        "PIL",
        "torch",
        "skimage",
        "tinycudann",
        "trimesh",
        "open3d",
        "nerfstudio",
        "gsplat",
        "diff_gaussian_rasterization",
        "nvdiffrast",
    }
    present = {module: find_module(module) for module in sorted(modules)}
    method_rows: list[dict[str, Any]] = []
    for method in methods:
        meta = METHODS[method]
        required = [str(item) for item in meta.get("actual_runtime_required", [])]
        alternatives = [[str(item) for item in group] for group in meta.get("actual_runtime_alternatives", [])]
        optional = [str(item) for item in meta.get("actual_runtime_optional", [])]
        required_ok = all(bool(present.get(item, False)) for item in required)
        alternatives_ok = all(any(bool(present.get(item, False)) for item in group) for group in alternatives)
        method_rows.append(
            {
                "method": method,
                "required_modules": {item: bool(present.get(item, False)) for item in required},
                "alternative_module_groups": [
                    {item: bool(present.get(item, False)) for item in group} for group in alternatives
                ],
                "optional_modules": {item: bool(present.get(item, False)) for item in optional},
                "actual_runtime_dependency_ready": bool(required_ok and alternatives_ok),
                "execution_state": "not_executed_research_stub_only",
            }
        )
    return {"module_presence": present, "methods": method_rows}


def mask_bbox(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return {"valid": False, "x0": 0, "y0": 0, "x1": 0, "y1": 0, "area": 0, "coverage": 0.0}
    return {
        "valid": True,
        "x0": int(xs.min()),
        "y0": int(ys.min()),
        "x1": int(xs.max()) + 1,
        "y1": int(ys.max()) + 1,
        "area": int(mask.sum()),
        "coverage": float(mask.mean()),
    }


def load_rgb_mask_probe(view: dict[str, Any], target_size: int) -> dict[str, Any]:
    image_path = Path(str(view["image_path"]))
    mask_path = Path(str(view["mask_path"]))
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    native_mask = np.asarray(mask, dtype=np.uint8) > 127
    if image.size != (target_size, target_size):
        image = image.resize((target_size, target_size), Image.Resampling.BICUBIC)
    if mask.size != (target_size, target_size):
        mask = mask.resize((target_size, target_size), Image.Resampling.NEAREST)
    target_mask = np.asarray(mask, dtype=np.uint8) > 127
    return {
        "image_path": str(image_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "native_image_size": list(Image.open(image_path).size),
        "native_mask_size": list(native_mask.shape[::-1]),
        "target_rgb_shape": [int(target_size), int(target_size), 3],
        "target_mask_bbox": mask_bbox(target_mask),
        "native_mask_bbox": mask_bbox(native_mask),
    }


def camera_sanity(intrinsic: np.ndarray, world_to_cam: np.ndarray, target_size: int) -> dict[str, Any]:
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    world_to_cam = np.asarray(world_to_cam, dtype=np.float32)
    rot = world_to_cam[:3, :3]
    trans = world_to_cam[:3, 3]
    det = float(np.linalg.det(rot))
    ortho_err = float(np.linalg.norm(rot.T @ rot - np.eye(3, dtype=np.float32)))
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "rotation_det": det,
        "rotation_orthogonality_error": ortho_err,
        "translation_norm": float(np.linalg.norm(trans)),
        "principal_point_inside_target": bool(0 <= cx <= target_size and 0 <= cy <= target_size),
        "positive_focal": bool(fx > 0 and fy > 0),
        "rotation_ok": bool(abs(abs(det) - 1.0) < 0.08 and ortho_err < 0.15),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_a3_seed(seed_dir: Path | None) -> dict[str, Any]:
    if seed_dir is None:
        return {
            "provided": False,
            "seed_ready_for_initialization": False,
            "note": "No A3 seed dir provided; A1/A2 recipes remain asset-only.",
        }
    seed_dir = seed_dir.resolve()
    summary_path = seed_dir / "visual_hull_init_summary.json"
    mesh_npz_path = seed_dir / "visual_hull_init_mesh.npz"
    mesh_ply_path = seed_dir / "visual_hull_init_mesh.ply"
    points_npz_path = seed_dir / "visual_hull_init_points.npz"
    out: dict[str, Any] = {
        "provided": True,
        "seed_dir": str(seed_dir),
        "summary_path": str(summary_path) if summary_path.is_file() else None,
        "mesh_npz_path": str(mesh_npz_path) if mesh_npz_path.is_file() else None,
        "mesh_ply_path": str(mesh_ply_path) if mesh_ply_path.is_file() else None,
        "points_npz_path": str(points_npz_path) if points_npz_path.is_file() else None,
        "research_only": True,
        "not_teacher": True,
    }
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        out["summary_sha256"] = sha256_file(summary_path)
        out["summary_status"] = summary.get("status")
        out["summary_core"] = {
            key: summary.get("summary", {}).get(key)
            for key in [
                "target_size",
                "grid_resolution",
                "support_threshold",
                "occupied_points",
                "occupied_fraction",
                "mesh_status",
                "mesh_vertices",
                "mesh_faces",
                "selected_views",
            ]
        }
    if mesh_npz_path.is_file():
        out["mesh_npz_sha256"] = sha256_file(mesh_npz_path)
        with np.load(mesh_npz_path, allow_pickle=False) as data:
            vertices = np.asarray(data["vertices"], dtype=np.float32)
            faces = np.asarray(data["faces"], dtype=np.int64)
            out["mesh_npz"] = {
                "vertices": int(vertices.shape[0]),
                "faces": int(faces.shape[0]),
                "bbox_min": vertices.min(axis=0).astype(float).tolist() if vertices.size else [0.0, 0.0, 0.0],
                "bbox_max": vertices.max(axis=0).astype(float).tolist() if vertices.size else [0.0, 0.0, 0.0],
                "grid_resolution": int(np.asarray(data["grid_resolution"]).item())
                if "grid_resolution" in data.files
                else None,
                "view_indices": np.asarray(data["view_indices"]).astype(int).tolist()
                if "view_indices" in data.files
                else [],
                "threshold_recorded_from_seed": int(np.asarray(data["threshold"]).item())
                if "threshold" in data.files
                else None,
            }
    if points_npz_path.is_file():
        out["points_npz_sha256"] = sha256_file(points_npz_path)
    mesh = out.get("mesh_npz", {})
    out["seed_ready_for_initialization"] = bool(int(mesh.get("vertices", 0)) > 0 and int(mesh.get("faces", 0)) > 0)
    out["decision"] = (
        "A3 seed may initialize A1/A2 research reconstruction only; it is not a teacher, "
        "not a candidate, and not a pass signal."
    )
    return out


def method_plan(method: str, assets_ready: bool, seed_ready: bool, deps: dict[str, Any]) -> dict[str, Any]:
    meta = METHODS[method]
    dep_row = next(row for row in deps["methods"] if row["method"] == method)
    return {
        "method": method,
        "label": meta["label"],
        "research_only": True,
        "input_contract": meta["input_contract"],
        "asset_ready_for_wrapper_research": bool(assets_ready),
        "a3_seed_bound": bool(seed_ready),
        "actual_runtime_dependency_ready": bool(dep_row["actual_runtime_dependency_ready"]),
        "current_execution": "dry_recipe_written_only",
        "no_surface_reconstruction_run": True,
        "seed_use": meta["seed_use"],
        "recipe": meta["research_recipe"],
        "blocked_exports": meta["blocked_exports"],
        "stop_before_teacher_conditions": [
            "Open3D full/head/face/hairline/hands review is missing",
            "strict_teacher_passes remains zero",
            "strict_candidate_passes remains zero",
            "no separate teacher gate has accepted the artifact",
        ],
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# A-Line Dense Reconstruction Preflight Stub",
        "",
        f"Status: `{payload['status']}`",
        "",
        "This is a research-preflight artifact only. It did not run A1/A2 reconstruction.",
        "",
        "## Gate Truth",
        "",
        "```text",
        f"strict_candidate_passes = {STRICT_CANDIDATE_PASSES}",
        f"strict_teacher_passes = {STRICT_TEACHER_PASSES}",
        "no_teacher_export = true",
        "no_candidate_export = true",
        "formal_vggt_cloud_train_infer_export = not_called",
        "```",
        "",
        "## Scene Audit",
        "",
        "```json",
        json.dumps(summary["scene"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## A3 Seed",
        "",
        "```json",
        json.dumps(summary["a3_seed"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Dependency Probe",
        "",
        "```json",
        json.dumps(summary["dependency_probe"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## A1/A2 Dry Recipe",
        "",
        "```json",
        json.dumps(payload["method_plans"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Forbidden Routes",
        "",
        "```json",
        json.dumps(FORBIDDEN_ROUTES, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Decision",
        "",
        "```text",
        payload["decision"],
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_methods = [item.strip() for item in str(args.methods).split(",") if item.strip()]
    unknown = [item for item in requested_methods if item not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}; choices={sorted(METHODS)}")

    scene_dir = args.scene_dir.resolve()
    manifest = localize_scene_manifest_paths(recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir)), scene_dir)
    views = manifest["exported_views"]
    view_indices = parse_view_indices(args.view_indices, len(views))
    dataset_root = args.dataset_root or Path(str(manifest.get("dataset_root", "")))
    camera_override = load_camera_params_sidecar(scene_dir)
    cameras, camera_source = resolve_scene_camera_params(manifest, dataset_root, args.subset_name, camera_override)

    target_size = int(args.target_size)
    rows: list[dict[str, Any]] = []
    mask_coverages: list[float] = []
    camera_rows: list[dict[str, Any]] = []
    selected_camera_ids: list[str] = []
    for view_index in view_indices:
        view = views[view_index]
        camera_id = str(view["camera_id"])
        selected_camera_ids.append(camera_id)
        probe = load_rgb_mask_probe(view, target_size)
        mask_coverages.append(float(probe["target_mask_bbox"]["coverage"]))
        params = cameras.get(camera_id)
        if params is None:
            cam = {"available": False}
        else:
            intrinsic = align_intrinsics_for_loaded_scene_view(
                np.asarray(params["intrinsic"], dtype=np.float32),
                view,
                target_size,
            )
            cam = {"available": True, **camera_sanity(intrinsic, np.asarray(params["world_to_cam"], dtype=np.float32), target_size)}
        camera_rows.append(cam)
        rows.append({"view_index": int(view_index), "camera_id": camera_id, "rgb_mask": probe, "camera": cam})

    cameras_available = all(bool(row.get("available")) for row in camera_rows)
    camera_numeric_ok = cameras_available and all(
        bool(row.get("positive_focal")) and bool(row.get("rotation_ok")) for row in camera_rows
    )
    masks_nonempty = bool(mask_coverages) and min(mask_coverages) > 0.02
    same_frame = manifest.get("seq_id") is not None and manifest.get("frame_id") is not None
    assets_ready = bool(len(view_indices) >= 3 and masks_nonempty and camera_numeric_ok and same_frame)
    a3_seed = load_a3_seed(args.a3_seed_dir)
    deps = dependency_probe(requested_methods)
    method_plans = [
        method_plan(method, assets_ready, bool(a3_seed.get("seed_ready_for_initialization")), deps)
        for method in requested_methods
    ]

    payload = {
        "status": "a_line_dense_reconstruction_preflight_stub_complete_not_reconstruction",
        "summary": {
            "research_only": True,
            "no_teacher_export": True,
            "no_candidate_export": True,
            "strict_candidate_passes": STRICT_CANDIDATE_PASSES,
            "strict_teacher_passes": STRICT_TEACHER_PASSES,
            "formal_vggt_cloud_train_infer_export_called": False,
            "scene": {
                "scene_dir": str(scene_dir),
                "seq_id": str(manifest.get("seq_id")),
                "frame_id": str(manifest.get("frame_id")),
                "same_frame_basis": "scene_manifest.seq_id/frame_id",
                "same_frame": bool(same_frame),
                "view_count_total": int(len(views)),
                "selected_views": view_indices,
                "selected_camera_ids": selected_camera_ids,
                "target_size": target_size,
                "camera_source": camera_source,
                "camera_numeric_ok": bool(camera_numeric_ok),
                "masks_nonempty": bool(masks_nonempty),
                "mask_coverage_min": float(min(mask_coverages) if mask_coverages else 0.0),
                "mask_coverage_max": float(max(mask_coverages) if mask_coverages else 0.0),
                "mask_coverage_mean": float(np.mean(mask_coverages)) if mask_coverages else 0.0,
                "asset_ready_for_a1_a2_research_wrapper": bool(assets_ready),
                "views": rows,
            },
            "a3_seed": a3_seed,
            "dependency_probe": deps,
        },
        "method_plans": method_plans,
        "hard_guards": {
            "forbidden_routes_not_used": FORBIDDEN_ROUTES,
            "strict_candidate_passes": STRICT_CANDIDATE_PASSES,
            "strict_teacher_passes": STRICT_TEACHER_PASSES,
            "output_kind": "research_json_markdown_only",
        },
        "command": " ".join(sys.argv),
        "decision": (
            "A1/A2 are ready for a dry research wrapper plan when scene assets are ready. "
            "This run wrote only JSON/Markdown research artifacts. It did not reconstruct, export, "
            "train, infer, or alter strict pass accounting."
        ),
    }

    json_path = output_dir / "a_line_dense_reconstruction_preflight_stub.json"
    md_path = output_dir / "a_line_dense_reconstruction_preflight_stub.md"
    recipe_path = output_dir / "a_line_dense_reconstruction_recipe.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    recipe_path.write_text(
        json.dumps(
            {
                "status": "dry_recipe_only",
                "research_only": True,
                "scene": payload["summary"]["scene"],
                "a3_seed": payload["summary"]["a3_seed"],
                "method_plans": method_plans,
                "hard_guards": payload["hard_guards"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(md_path, payload)
    print(json.dumps({"status": payload["status"], "output_dir": str(output_dir), "assets_ready": assets_ready}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
