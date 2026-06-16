from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from tools.research_cloud_common import json_ready, repo_root, write_json
from training.data.datasets.human_surface_sdf_dataset import HumanSurfaceSDFDataset, scalar_stats


REPO_ROOT = repo_root()
DEFAULT_STAGING_ROOT = REPO_ROOT / "output" / "surface_research_cloud_preflight" / "V9_cloud_asset_staging"
DEFAULT_MANIFEST = DEFAULT_STAGING_ROOT / "v9_cloud_a_asset_manifest.json"
DEFAULT_REPORT_MD = REPO_ROOT / "reports" / "20260507_v9_cloud_asset_staging_status.md"
DEFAULT_REPORT_JSON = REPO_ROOT / "reports" / "20260507_v9_cloud_asset_staging_status.json"
REQUIRED_QUERY_KEYS = (
    "query_indices",
    "query_positions",
    "query_families",
    "support",
    "token_ids",
    "uv",
    "depth",
    "mean_features",
    "variance_features",
    "selected_view_indices",
    "patch_start_idx",
    "patch_grid",
)
REQUIRED_TEMPLATE_KEYS = (
    "hybrid_vertices",
    "hybrid_faces",
    "part_ids",
    "face_front_vertex_mask",
    "hairline_vertex_mask",
    "head_vertex_mask",
    "left_hand_vertex_mask",
    "right_hand_vertex_mask",
    "lower_clothing_vertex_mask",
)
REQUIRED_CAMERA_KEYS = ("camera_ids", "intrinsics", "cam_to_world", "world_to_cam")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the staged V9 Cloud-A research asset bundle without launching cloud."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--feature-bins", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_path(path: Path) -> Path:
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.expanduser().resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} did not contain a JSON object")
    return payload


def file_check(record: dict[str, Any]) -> dict[str, Any]:
    path_text = record.get("staged_path") or record.get("source_path")
    path = resolve_path(Path(str(path_text)))
    out = {
        "role": record.get("role"),
        "path": str(path),
        "exists": path.is_file(),
        "expected_bytes": record.get("bytes"),
        "expected_sha256": record.get("sha256"),
    }
    if path.is_file():
        out["bytes"] = int(path.stat().st_size)
        if record.get("sha256"):
            out["sha256"] = sha256_file(path)
            out["sha256_ok"] = out["sha256"] == record.get("sha256")
        if record.get("bytes") is not None:
            out["bytes_ok"] = int(record["bytes"]) == int(out["bytes"])
    return out


def npz_summary(path: Path, required: tuple[str, ...]) -> tuple[dict[str, Any], list[str], dict[str, np.ndarray]]:
    path = resolve_path(path)
    reasons: list[str] = []
    summary: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "arrays": {}}
    arrays: dict[str, np.ndarray] = {}
    if not path.is_file():
        return summary, [f"missing npz: {path}"], arrays
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in required if key not in payload.files]
        if missing:
            reasons.append(f"{path.name} missing arrays: {missing}")
        for key in payload.files:
            array = np.asarray(payload[key])
            arrays[key] = array
            summary["arrays"][key] = {"shape": [int(v) for v in array.shape], "dtype": str(array.dtype)}
    summary["array_count"] = len(arrays)
    return summary, reasons, arrays


def verify_query_cache(path: Path) -> tuple[dict[str, Any], list[str]]:
    summary, reasons, arrays = npz_summary(path, REQUIRED_QUERY_KEYS)
    if reasons:
        return summary, reasons
    query_count = int(arrays["query_indices"].reshape(-1).shape[0])
    feature_dim = int(arrays["mean_features"].shape[1]) if arrays["mean_features"].ndim == 2 else 0
    view_count = int(arrays["token_ids"].shape[1]) if arrays["token_ids"].ndim == 2 else 0
    checks = {
        "query_count": query_count,
        "feature_dim": feature_dim,
        "view_count": view_count,
        "selected_view_count": int(arrays["selected_view_indices"].reshape(-1).shape[0]),
        "support_ge_1": int(np.count_nonzero(np.asarray(arrays["support"]).reshape(-1) >= 1)),
        "support_ge_2": int(np.count_nonzero(np.asarray(arrays["support"]).reshape(-1) >= 2)),
        "support_ge_3": int(np.count_nonzero(np.asarray(arrays["support"]).reshape(-1) >= 3)),
        "families": sorted(str(v) for v in np.unique(arrays["query_families"].astype(str))),
        "patch_start_idx": int(arrays["patch_start_idx"].reshape(-1)[0]),
        "patch_grid": int(arrays["patch_grid"].reshape(-1)[0]),
    }
    summary["checks"] = checks
    expected_shapes = {
        "query_positions": (query_count, 3),
        "support": (query_count,),
        "token_ids": (query_count, view_count),
        "uv": (query_count, view_count, 2),
        "depth": (query_count, view_count),
        "mean_features": (query_count, feature_dim),
        "variance_features": (query_count, feature_dim),
    }
    for key, expected in expected_shapes.items():
        if tuple(arrays[key].shape) != expected:
            reasons.append(f"query cache {key} shape {arrays[key].shape} != {expected}")
    if query_count <= 0:
        reasons.append("query cache has no queries")
    if feature_dim <= 0:
        reasons.append("query cache has no feature dimension")
    if checks["support_ge_1"] <= 0:
        reasons.append("query cache has no supported queries")
    return summary, reasons


def verify_template(path: Path, query_cache_path: Path) -> tuple[dict[str, Any], list[str]]:
    summary, reasons, arrays = npz_summary(path, REQUIRED_TEMPLATE_KEYS)
    if reasons:
        return summary, reasons
    vertices = np.asarray(arrays["hybrid_vertices"])
    faces = np.asarray(arrays["hybrid_faces"])
    part_ids = np.asarray(arrays["part_ids"])
    summary["checks"] = {
        "hybrid_vertices": int(vertices.shape[0]),
        "hybrid_faces": int(faces.shape[0]),
        "part_id_count": int(part_ids.shape[0]),
        "unique_part_ids": [int(v) for v in np.unique(part_ids)],
        "mask_counts": {key: int(np.count_nonzero(arrays[key])) for key in REQUIRED_TEMPLATE_KEYS if key.endswith("_mask")},
    }
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        reasons.append(f"template hybrid_vertices must be [N,3], got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        reasons.append(f"template hybrid_faces must be [F,3], got {faces.shape}")
    if part_ids.shape[0] != vertices.shape[0]:
        reasons.append("template part_ids count does not match hybrid_vertices")
    for key in REQUIRED_TEMPLATE_KEYS:
        if key.endswith("_mask") and arrays[key].shape[0] != vertices.shape[0]:
            reasons.append(f"template {key} count does not match hybrid_vertices")
    if query_cache_path.is_file():
        with np.load(query_cache_path, allow_pickle=False) as query:
            query_indices = np.asarray(query["query_indices"], dtype=np.int64).reshape(-1)
        if query_indices.size and int(query_indices.max()) >= vertices.shape[0]:
            reasons.append("query cache references template vertex index beyond hybrid_vertices")
    return summary, reasons


def image_headers(paths: list[Path], max_headers: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths[:max_headers]:
        with Image.open(path) as image:
            rows.append({"name": path.name, "size": [int(image.size[0]), int(image.size[1])], "mode": image.mode})
    return rows


def verify_scene(scene_dir: Path) -> tuple[dict[str, Any], list[str]]:
    scene_dir = resolve_path(scene_dir)
    reasons: list[str] = []
    manifest_path = scene_dir / "scene_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    exported_views = manifest.get("exported_views") if isinstance(manifest.get("exported_views"), list) else []
    images = sorted(path for path in (scene_dir / "images").iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES) if (scene_dir / "images").is_dir() else []
    masks = sorted(path for path in (scene_dir / "masks").iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES) if (scene_dir / "masks").is_dir() else []
    camera_summary, camera_reasons, camera_arrays = npz_summary(scene_dir / "camera_params_sidecar.npz", REQUIRED_CAMERA_KEYS)
    prior_summary, prior_reasons, _prior_arrays = npz_summary(scene_dir / "prior_maps.npz", ())
    reasons.extend(camera_reasons)
    if not (scene_dir / "prior_maps.npz").is_file():
        reasons.append(f"missing optional-but-expected 4k4d prior_maps.npz: {scene_dir / 'prior_maps.npz'}")
    elif prior_reasons:
        reasons.extend(prior_reasons)
    camera_count = int(camera_arrays.get("camera_ids", np.zeros((0,))).shape[0]) if camera_arrays else 0
    summary = {
        "path": str(scene_dir),
        "manifest_exists": manifest_path.is_file(),
        "exported_view_count": len(exported_views),
        "rgb_count": len(images),
        "mask_count": len(masks),
        "camera_count": camera_count,
        "rgb_headers": image_headers(images),
        "mask_headers": image_headers(masks),
        "camera_params_sidecar": camera_summary,
        "prior_maps": prior_summary,
        "target_camera": manifest.get("target_camera"),
        "source_camera_count": len(manifest.get("source_cameras") or []),
    }
    if not manifest_path.is_file():
        reasons.append(f"missing scene manifest: {manifest_path}")
    if len(exported_views) <= 0:
        reasons.append("scene manifest has no exported_views")
    if not images:
        reasons.append("4K4D RGB images are missing")
    if not masks:
        reasons.append("4K4D masks are missing")
    if images and masks and len(images) != len(masks):
        reasons.append(f"RGB/mask count mismatch: {len(images)} vs {len(masks)}")
    if exported_views and len(images) != len(exported_views):
        reasons.append(f"RGB count {len(images)} != exported_view_count {len(exported_views)}")
    if exported_views and len(masks) != len(exported_views):
        reasons.append(f"mask count {len(masks)} != exported_view_count {len(exported_views)}")
    if camera_count and exported_views and camera_count != len(exported_views):
        reasons.append(f"camera count {camera_count} != exported_view_count {len(exported_views)}")
    for row in summary["rgb_headers"]:
        if row["size"] != [518, 518]:
            reasons.append(f"RGB image {row['name']} is not 518x518: {row['size']}")
    for row in summary["mask_headers"]:
        if row["size"] != [518, 518]:
            reasons.append(f"mask image {row['name']} is not 518x518: {row['size']}")
    return summary, reasons


def verify_token_cache(path: Path, query_cache_path: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    path = resolve_path(path)
    warnings: list[str] = []
    if not path.is_file():
        return {"path": str(path), "exists": False}, [], [f"VGGT token cache not present: {path}"]
    summary, reasons, arrays = npz_summary(path, ("tokens", "patch_start_idx", "selected_view_indices"))
    if reasons:
        return summary, reasons, warnings
    tokens = arrays["tokens"]
    selected = arrays["selected_view_indices"].reshape(-1)
    patch_start_idx = int(arrays["patch_start_idx"].reshape(-1)[0])
    summary["checks"] = {
        "tokens_shape": [int(v) for v in tokens.shape],
        "patch_start_idx": patch_start_idx,
        "selected_view_indices": [int(v) for v in selected],
    }
    if tokens.ndim != 4 or tokens.shape[0] != 1:
        reasons.append(f"token cache tokens must be [1,S,P,C], got {tokens.shape}")
    if query_cache_path.is_file():
        with np.load(query_cache_path, allow_pickle=False) as query:
            q_selected = np.asarray(query["selected_view_indices"]).reshape(-1)
            q_patch_start = int(np.asarray(query["patch_start_idx"]).reshape(-1)[0])
            q_feature_dim = int(np.asarray(query["mean_features"]).shape[1])
            q_view_count = int(np.asarray(query["token_ids"]).shape[1])
        if not np.array_equal(selected.astype(q_selected.dtype), q_selected):
            reasons.append("token cache selected_view_indices do not match query cache")
        if patch_start_idx != q_patch_start:
            reasons.append("token cache patch_start_idx does not match query cache")
        if tokens.ndim == 4 and int(tokens.shape[1]) != q_view_count:
            reasons.append("token cache view count does not match query cache token_ids")
        if tokens.ndim == 4 and int(tokens.shape[3]) != q_feature_dim:
            reasons.append("token feature dim does not match query mean_features")
    return summary, reasons, warnings


def verify_dataset(query_cache: Path, template_payload: Path, *, max_cases: int, feature_bins: int, seed: int) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    try:
        dataset = HumanSurfaceSDFDataset(
            query_cache=query_cache,
            template_payload=template_payload,
            external_case_roots=[],
            max_cases=max_cases,
            shell_offsets=(-0.008, -0.003, 0.0, 0.003, 0.008),
            feature_bins=feature_bins,
            seed=seed,
        )
        arrays = dataset.as_arrays()
        features = arrays["features"]
        labels = arrays["labels"]
        families = arrays["families"].astype(str)
        summary = {
            "case_count": len(dataset),
            "sample_count": int(labels.shape[0]),
            "feature_dim": int(features.shape[1]),
            "label_stats": scalar_stats(labels),
            "sdf_stats": scalar_stats(arrays["sdf"]),
            "families": {family: int(np.count_nonzero(families == family)) for family in sorted(set(families.tolist()))},
            "finite_features": int(np.isfinite(features).sum()),
            "total_features": int(features.size),
            "genealogy": dataset.genealogy(),
        }
        if len(dataset) <= 0 or labels.shape[0] <= 0:
            reasons.append("HumanSurfaceSDFDataset produced no samples")
        if features.ndim != 2 or features.shape[1] <= 0:
            reasons.append("HumanSurfaceSDFDataset produced invalid features")
        if not np.isfinite(features).all():
            reasons.append("HumanSurfaceSDFDataset features contain non-finite values")
        if not {0.0, 1.0}.intersection(set(np.unique(labels).astype(float).tolist())):
            reasons.append("HumanSurfaceSDFDataset labels do not contain binary values")
        return summary, reasons
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}, [f"HumanSurfaceSDFDataset construction failed: {exc}"]


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    counts = summary["asset_counts"]
    missing = summary["missing_assets"]
    lines = [
        "# V9 Cloud-A Asset Staging Verification",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This verifies the staged real assets for Cloud-A only. No cloud job was launched.",
        "",
        "## Asset Counts",
        "",
        "```json",
        json.dumps(counts, indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Missing Assets",
        "",
        "```json",
        json.dumps(missing, indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Verification Reasons",
        "",
        "```json",
        json.dumps(summary["verification_reasons"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Dataset Smoke",
        "",
        "```json",
        json.dumps(summary["checks"]["dataset_smoke"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## Staged Assets",
        "",
        "```json",
        json.dumps(summary["staged_assets"], indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest_path = resolve_path(args.manifest)
    staging_root = resolve_path(args.staging_root)
    reasons: list[str] = []
    warnings: list[str] = []
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = load_json(manifest_path)
    staged_assets = manifest.get("staged_assets") if isinstance(manifest.get("staged_assets"), dict) else {}
    query_cache = resolve_path(Path(str(staged_assets.get("query_cache", staging_root / "assets" / "query_cache" / "b_fus3d_query_evidence_cache.npz"))))
    template_payload = resolve_path(Path(str(staged_assets.get("template_payload", staging_root / "assets" / "template" / "connected_human_surface_template_payload_self_describing.npz"))))
    scene_dir = resolve_path(Path(str(staged_assets.get("scene_dir", staging_root / "assets" / "4k4d_scene"))))
    token_value = staged_assets.get("token_cache")
    token_cache = resolve_path(Path(str(token_value))) if token_value else staging_root / "assets" / "vggt_token_cache" / "aggregator_layer_23.npz"

    file_checks = [file_check(record) for record in manifest.get("files", []) if isinstance(record, dict)]
    for check in file_checks:
        if not check["exists"]:
            role = check.get("role") or "asset"
            expected_required = role in {"query_cache", "template_payload", "4k4d_scene_manifest", "4k4d_camera_params_sidecar"} or str(role).startswith("4k4d_rgb") or str(role).startswith("4k4d_mask")
            if expected_required:
                reasons.append(f"missing staged file for {role}: {check['path']}")
            else:
                warnings.append(f"missing optional staged file for {role}: {check['path']}")
        if check.get("sha256_ok") is False:
            reasons.append(f"sha256 mismatch for {check.get('role')}: {check['path']}")
        if check.get("bytes_ok") is False:
            reasons.append(f"size mismatch for {check.get('role')}: {check['path']}")

    query_summary, query_reasons = verify_query_cache(query_cache)
    template_summary, template_reasons = verify_template(template_payload, query_cache)
    scene_summary, scene_reasons = verify_scene(scene_dir)
    token_summary, token_reasons, token_warnings = verify_token_cache(token_cache, query_cache)
    dataset_summary, dataset_reasons = verify_dataset(
        query_cache,
        template_payload,
        max_cases=args.max_cases,
        feature_bins=args.feature_bins,
        seed=args.seed,
    )
    reasons.extend(query_reasons)
    reasons.extend(template_reasons)
    reasons.extend(scene_reasons)
    reasons.extend(token_reasons)
    warnings.extend(token_warnings)

    missing_assets = list(manifest.get("missing_assets") or [])
    blocking_missing = [item for item in missing_assets if str(item).startswith("missing required")]
    reasons.extend(blocking_missing)

    asset_counts = dict(manifest.get("asset_counts") or {})
    asset_counts.update(
        {
            "verified_file_records": len(file_checks),
            "rgb_images_verified": int(scene_summary.get("rgb_count", 0)),
            "masks_verified": int(scene_summary.get("mask_count", 0)),
            "exported_views_verified": int(scene_summary.get("exported_view_count", 0)),
            "camera_count_verified": int(scene_summary.get("camera_count", 0)),
            "query_count_verified": int(query_summary.get("checks", {}).get("query_count", 0)),
            "query_feature_dim_verified": int(query_summary.get("checks", {}).get("feature_dim", 0)),
            "dataset_samples_verified": int(dataset_summary.get("sample_count", 0)),
            "token_cache_verified": int(bool(token_summary.get("exists"))),
        }
    )
    success = not reasons
    status = "verified" if success else "blocked"
    summary = {
        "schema_version": "20260507_v9_cloud_a_asset_verification_v1",
        "created_utc": now_utc(),
        "created_by": "tools/verify_research_cloud_assets.py",
        "status": status,
        "success": success,
        "task": "Task 2 Cloud-A asset staging",
        "cloud_lane": "Cloud-A",
        "lane": "b_fus3d2_human_dataset_train",
        "research_only": True,
        "no_cloud_launch": True,
        "no_train": True,
        "no_predictions_write": True,
        "no_registry_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_strict_pass_write": True,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_train_infer_export": "blocked",
        "manifest": str(manifest_path),
        "staging_root": str(staging_root),
        "staged_assets": {
            "query_cache": str(query_cache),
            "template_payload": str(template_payload),
            "scene_dir": str(scene_dir),
            "token_cache": str(token_cache) if token_cache.is_file() else None,
        },
        "asset_counts": asset_counts,
        "missing_assets": missing_assets,
        "verification_reasons": reasons,
        "warnings": warnings,
        "checks": {
            "files": file_checks,
            "query_cache": query_summary,
            "template_payload": template_summary,
            "scene": scene_summary,
            "vggt_token_cache": token_summary,
            "dataset_smoke": dataset_summary,
        },
        "outputs": {
            "status_report_md": str(resolve_path(args.report_md)),
            "status_report_json": str(resolve_path(args.report_json)),
        },
    }
    write_json(resolve_path(args.report_json), summary)
    write_markdown(resolve_path(args.report_md), summary)
    if args.json:
        print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps(json_ready({"status": status, "asset_counts": asset_counts, "missing_assets": missing_assets, "verification_reasons": reasons, "warnings": warnings}), indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
