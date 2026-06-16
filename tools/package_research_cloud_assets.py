from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.research_cloud_common import json_ready, repo_root, write_json


REPO_ROOT = repo_root()
DEFAULT_CONFIG = REPO_ROOT / "training" / "config" / "b_fus3d2_human_dataset.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "surface_research_cloud_preflight" / "V9_cloud_asset_staging"
DEFAULT_SCENE_DIR = REPO_ROOT / "output" / "4k4d_preprocessed_scene_variants" / "0012_11_frame0000_60views_human_crop"
DEFAULT_QUERY_CACHE = (
    REPO_ROOT
    / "output"
    / "surface_research_preflight_local"
    / "B_Fus3D6_query_evidence_cache_hybrid6_layer23"
    / "b_fus3d_query_evidence_cache.npz"
)
DEFAULT_QUERY_SUMMARY = DEFAULT_QUERY_CACHE.with_name("b_fus3d_query_evidence_cache_summary.json")
DEFAULT_TEMPLATE_PAYLOAD = (
    REPO_ROOT
    / "output"
    / "surface_research_preflight_local"
    / "connected_payload_self_describing"
    / "connected_human_surface_template_payload_self_describing.npz"
)
DEFAULT_TEMPLATE_SUMMARY = DEFAULT_TEMPLATE_PAYLOAD.with_suffix(".summary.json")
DEFAULT_TOKEN_CACHE = (
    REPO_ROOT
    / "output"
    / "surface_research_preflight_local"
    / "B_Fus3D0_token_cache_extract_hybrid6_518_roi_withhands_arrays_v2"
    / "token_cache"
    / "aggregator_layer_23.npz"
)
DEFAULT_TOKEN_SUMMARY_DIR = DEFAULT_TOKEN_CACHE.parents[1]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage the real local assets needed by the research-only Cloud-A B-Fus3D2 smoke. "
            "This copies query cache, template payload, 4K4D RGB/masks/cameras, and the optional "
            "VGGT token cache into output/surface_research_cloud_preflight/V9_cloud_asset_staging."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--query-cache", type=Path, default=DEFAULT_QUERY_CACHE)
    parser.add_argument("--query-summary", type=Path, default=DEFAULT_QUERY_SUMMARY)
    parser.add_argument("--template-payload", type=Path, default=DEFAULT_TEMPLATE_PAYLOAD)
    parser.add_argument("--template-summary", type=Path, default=DEFAULT_TEMPLATE_SUMMARY)
    parser.add_argument("--token-cache", type=Path, default=DEFAULT_TOKEN_CACHE)
    parser.add_argument(
        "--include-token-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy VGGT token cache if present. Missing token cache is recorded but does not block packaging.",
    )
    parser.add_argument("--manifest-name", default="v9_cloud_a_asset_manifest.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_path(path: Path) -> Path:
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.expanduser().resolve()


def rel_repo(path: Path) -> str:
    path = path.expanduser().resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, role: str, staged_path: Path | None = None, sha256: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {
        "role": role,
        "source_path": str(path.expanduser().resolve()),
        "source_rel": rel_repo(path),
        "exists": path.is_file(),
    }
    if staged_path is not None:
        record["staged_path"] = str(staged_path.expanduser().resolve())
        record["staged_rel"] = rel_repo(staged_path)
    if path.is_file():
        stat = path.stat()
        record.update(
            {
                "bytes": int(stat.st_size),
                "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
                "sha256": sha256_file(path) if sha256 else None,
            }
        )
    return record


def copy_file(src: Path, dst: Path, *, role: str, required: bool, sha256: bool = True) -> tuple[dict[str, Any], str | None]:
    src = resolve_path(src)
    if not src.is_file():
        record = file_record(src, role=role, staged_path=dst, sha256=False)
        return record, f"missing required {role}: {src}" if required else f"missing optional {role}: {src}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return file_record(src, role=role, staged_path=dst, sha256=sha256), None


def copy_tree_files(src_dir: Path, dst_dir: Path, *, role: str, suffixes: set[str] | None = None) -> list[dict[str, Any]]:
    src_dir = resolve_path(src_dir)
    if not src_dir.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for src in sorted(item for item in src_dir.rglob("*") if item.is_file()):
        if suffixes is not None and src.suffix.lower() not in suffixes:
            continue
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        records.append(file_record(src, role=role, staged_path=dst))
    return records


def npz_headers(path: Path, *, max_keys: int = 80) -> dict[str, Any]:
    path = resolve_path(path)
    if not path.is_file():
        return {"exists": False, "path": str(path), "arrays": {}}
    arrays: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as payload:
        for key in payload.files[:max_keys]:
            array = payload[key]
            arrays[key] = {
                "shape": [int(v) for v in array.shape],
                "dtype": str(array.dtype),
            }
    return {"exists": True, "path": str(path), "array_count": len(arrays), "arrays": arrays}


def scan_images(paths: list[Path], *, max_headers: int = 8) -> dict[str, Any]:
    headers: list[dict[str, Any]] = []
    suffixes: dict[str, int] = {}
    for idx, path in enumerate(paths):
        suffixes[path.suffix.lower()] = suffixes.get(path.suffix.lower(), 0) + 1
        if idx >= max_headers:
            continue
        with Image.open(path) as image:
            headers.append(
                {
                    "name": path.name,
                    "width": int(image.size[0]),
                    "height": int(image.size[1]),
                    "mode": image.mode,
                }
            )
    return {"count": len(paths), "suffix_histogram": suffixes, "sample_headers": headers}


def load_json_if_present(path: Path) -> dict[str, Any]:
    path = resolve_path(path)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"payload_type": type(payload).__name__}


def scene_summary(scene_dir: Path) -> dict[str, Any]:
    scene_dir = resolve_path(scene_dir)
    manifest_path = scene_dir / "scene_manifest.json"
    manifest = load_json_if_present(manifest_path)
    exported_views = manifest.get("exported_views") if isinstance(manifest.get("exported_views"), list) else []
    images = sorted(path for path in (scene_dir / "images").iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    masks = sorted(path for path in (scene_dir / "masks").iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    camera_sidecar = scene_dir / "camera_params_sidecar.npz"
    prior_maps = scene_dir / "prior_maps.npz"
    camera_ids = [str(view.get("camera_id", "")) for view in exported_views]
    return {
        "path": str(scene_dir),
        "manifest_exists": manifest_path.is_file(),
        "exported_view_count": len(exported_views),
        "target_camera": manifest.get("target_camera"),
        "source_camera_count": len(manifest.get("source_cameras") or []),
        "camera_ids_sample": camera_ids[:12],
        "images": scan_images(images),
        "masks": scan_images(masks),
        "camera_params_sidecar": npz_headers(camera_sidecar),
        "prior_maps": npz_headers(prior_maps, max_keys=20),
        "contact_sheets": {
            "rgb": file_record(scene_dir / "rgb_contact_sheet.png", role="4k4d_rgb_contact_sheet", sha256=False),
            "mask": file_record(scene_dir / "mask_contact_sheet.png", role="4k4d_mask_contact_sheet", sha256=False),
        },
    }


def clean_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir = resolve_path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    counts = manifest["asset_counts"]
    missing = manifest["missing_assets"]
    lines = [
        "# V9 Cloud-A Asset Staging",
        "",
        f"Status: `{manifest['status']}`",
        "",
        "This is a research-only asset package for Cloud-A B-Fus3D2. It stages inputs only;",
        "it does not launch cloud, train, write predictions, export teachers/candidates, or update strict pass state.",
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
        "## Staged Root",
        "",
        f"`{manifest['staging_root']}`",
        "",
        "## Manifest",
        "",
        f"`{manifest['outputs']['manifest_json']}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    clean_output_dir(output_dir, args.overwrite)

    assets_dir = output_dir / "assets"
    missing: list[str] = []
    files: list[dict[str, Any]] = []

    for src, dst_rel, role, required in (
        (args.config, Path("config/b_fus3d2_human_dataset.yaml"), "cloud_a_config", True),
        (args.query_cache, Path("query_cache/b_fus3d_query_evidence_cache.npz"), "query_cache", True),
        (args.query_summary, Path("query_cache/b_fus3d_query_evidence_cache_summary.json"), "query_cache_summary", False),
        (args.template_payload, Path("template/connected_human_surface_template_payload_self_describing.npz"), "template_payload", True),
        (args.template_summary, Path("template/connected_human_surface_template_payload_self_describing.summary.json"), "template_summary", False),
    ):
        record, miss = copy_file(src, assets_dir / dst_rel, role=role, required=required)
        files.append(record)
        if miss:
            missing.append(miss)

    scene_dir = resolve_path(args.scene_dir)
    scene_stage = assets_dir / "4k4d_scene"
    for name, required in (
        ("scene_manifest.json", True),
        ("camera_params_sidecar.npz", True),
        ("prior_maps.npz", False),
        ("rgb_contact_sheet.png", False),
        ("mask_contact_sheet.png", False),
    ):
        record, miss = copy_file(scene_dir / name, scene_stage / name, role=f"4k4d_{Path(name).stem}", required=required)
        files.append(record)
        if miss:
            missing.append(miss)

    image_records = copy_tree_files(scene_dir / "images", scene_stage / "images", role="4k4d_rgb", suffixes=IMAGE_SUFFIXES)
    mask_records = copy_tree_files(scene_dir / "masks", scene_stage / "masks", role="4k4d_mask", suffixes=IMAGE_SUFFIXES)
    files.extend(image_records)
    files.extend(mask_records)
    if not image_records:
        missing.append(f"missing required 4k4d RGB images: {scene_dir / 'images'}")
    if not mask_records:
        missing.append(f"missing required 4k4d masks: {scene_dir / 'masks'}")

    token_records: list[dict[str, Any]] = []
    token_cache = resolve_path(args.token_cache)
    if args.include_token_cache and token_cache.is_file():
        record, miss = copy_file(token_cache, assets_dir / "vggt_token_cache" / "aggregator_layer_23.npz", role="vggt_token_cache", required=False)
        token_records.append(record)
        if miss:
            missing.append(miss)
        for sidecar_name in ("b_fus3d_token_cache_summary.json", "token_layer_stats.json", "roi_token_coverage.json"):
            sidecar = DEFAULT_TOKEN_SUMMARY_DIR / sidecar_name
            if sidecar.is_file():
                record, _ = copy_file(
                    sidecar,
                    assets_dir / "vggt_token_cache" / sidecar_name,
                    role=f"vggt_token_cache_{Path(sidecar_name).stem}",
                    required=False,
                    sha256=sidecar_name != "roi_token_coverage.json",
                )
                token_records.append(record)
    elif args.include_token_cache:
        missing.append(f"missing optional vggt token cache: {token_cache}")
        token_records.append(file_record(token_cache, role="vggt_token_cache", staged_path=assets_dir / "vggt_token_cache" / "aggregator_layer_23.npz", sha256=False))
    files.extend(token_records)

    manifest_path = output_dir / args.manifest_name
    report_path = output_dir / "v9_cloud_a_asset_staging_report.md"
    query_cache_stage = assets_dir / "query_cache" / "b_fus3d_query_evidence_cache.npz"
    template_stage = assets_dir / "template" / "connected_human_surface_template_payload_self_describing.npz"
    token_stage = assets_dir / "vggt_token_cache" / "aggregator_layer_23.npz"
    staged_scene = assets_dir / "4k4d_scene"

    manifest = {
        "schema_version": "20260507_v9_cloud_a_asset_staging_v1",
        "created_utc": now_utc(),
        "created_by": "tools/package_research_cloud_assets.py",
        "status": "staged_with_missing_optional_assets" if missing else "staged",
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
        "staging_root": str(output_dir),
        "assets_root": str(assets_dir),
        "source_assets": {
            "config": str(resolve_path(args.config)),
            "query_cache": str(resolve_path(args.query_cache)),
            "template_payload": str(resolve_path(args.template_payload)),
            "scene_dir": str(scene_dir),
            "token_cache": str(token_cache),
        },
        "staged_assets": {
            "config": str(assets_dir / "config" / "b_fus3d2_human_dataset.yaml"),
            "query_cache": str(query_cache_stage),
            "template_payload": str(template_stage),
            "scene_dir": str(staged_scene),
            "token_cache": str(token_stage) if token_stage.is_file() else None,
        },
        "headers": {
            "query_cache": npz_headers(query_cache_stage),
            "template_payload": npz_headers(template_stage),
            "vggt_token_cache": npz_headers(token_stage) if token_stage.is_file() else {"exists": False, "path": str(token_stage)},
            "scene": scene_summary(staged_scene),
        },
        "asset_counts": {
            "files_total": len([item for item in files if item.get("exists")]),
            "bytes_total": int(sum(int(item.get("bytes") or 0) for item in files if item.get("exists"))),
            "query_cache_npz": int(query_cache_stage.is_file()),
            "template_payload_npz": int(template_stage.is_file()),
            "rgb_images": len(image_records),
            "masks": len(mask_records),
            "camera_npz": int((staged_scene / "camera_params_sidecar.npz").is_file()),
            "scene_manifest": int((staged_scene / "scene_manifest.json").is_file()),
            "prior_maps_npz": int((staged_scene / "prior_maps.npz").is_file()),
            "vggt_token_cache_npz": int(token_stage.is_file()),
            "token_sidecars": max(0, len(token_records) - int(token_stage.is_file())),
        },
        "missing_assets": missing,
        "files": files,
        "outputs": {
            "manifest_json": str(manifest_path),
            "report_md": str(report_path),
        },
    }
    write_json(manifest_path, manifest)
    write_markdown(report_path, manifest)
    print(json.dumps(json_ready({"status": manifest["status"], "manifest": manifest_path, "asset_counts": manifest["asset_counts"], "missing_assets": missing}), indent=2, ensure_ascii=False))
    return 0 if not any(item.startswith("missing required") for item in missing) else 2


if __name__ == "__main__":
    raise SystemExit(main())
