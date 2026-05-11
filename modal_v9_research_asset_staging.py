from __future__ import annotations

import json
import os
import time
from pathlib import Path, PurePosixPath

import modal


REPO_ROOT = Path(__file__).resolve().parent
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
APP_NAME = os.environ.get("VGGT_MODAL_V9_ASSET_APP_NAME", "vggt-v9-research-asset-staging")
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_V9_ASSET_TIMEOUT_SEC", str(30 * 60)))

FORBIDDEN_OUTPUT_TOKENS = ("strict_pass", "teacher_export", "candidate_export", "predictions", "formal_candidate", "strict_gate_registry")
FORBIDDEN_FALLBACK_TOKENS = (
    "make_procedural_sdf_case",
    "missing_query_cache_for_fallback",
    "missing_template_for_fallback",
    "Fallback used when query/template assets are absent",
    "procedural_human_surface_sdf_case",
)


IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install("numpy==1.26.1")
app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


def _normalize_subpath(value: str) -> str:
    cleaned = (value or "").strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("Expected a non-empty remote subpath.")
    parts = Path(cleaned).parts
    if ".." in parts:
        raise ValueError(f"Parent traversal is not allowed: {value!r}")
    lower = cleaned.lower()
    if "surface_research_cloud_preflight" not in lower:
        raise ValueError("V9 research cloud asset path must include surface_research_cloud_preflight.")
    if any(word in lower for word in FORBIDDEN_OUTPUT_TOKENS):
        raise ValueError(f"V9 research cloud asset path contains a forbidden token: {value!r}")
    return cleaned


def _file_entry(path: Path) -> dict:
    return {"path": str(path), "exists": path.is_file(), "bytes": int(path.stat().st_size) if path.is_file() else 0}


def _load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


@app.function(image=IMAGE, timeout=TIMEOUT_SEC, volumes={REMOTE_OUTPUT_DIR.as_posix(): output_volume})
def verify_remote_assets(remote_asset_subdir: str, remote_output_subdir: str) -> dict:
    import numpy as np

    asset_subdir = _normalize_subpath(remote_asset_subdir)
    output_subdir = _normalize_subpath(remote_output_subdir)
    asset_dir = Path(str(REMOTE_OUTPUT_DIR / asset_subdir))
    output_dir = Path(str(REMOTE_OUTPUT_DIR / output_subdir))
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = asset_dir / "asset_manifest.json"
    query_cache = asset_dir / "query_cache" / "b_fus3d_query_evidence_cache.npz"
    template_payload = asset_dir / "template" / "connected_human_surface_template_payload_self_describing.npz"
    token_dir = asset_dir / "token_cache"
    images_dir = asset_dir / "scene" / "images"
    masks_dir = asset_dir / "scene" / "masks"
    camera_sidecar = asset_dir / "scene" / "camera_params_sidecar.npz"
    prior_maps = asset_dir / "scene" / "prior_maps.npz"

    token_files = sorted(token_dir.glob("*.npz")) if token_dir.is_dir() else []
    image_files = sorted([p for p in images_dir.glob("*") if p.is_file()]) if images_dir.is_dir() else []
    mask_files = sorted([p for p in masks_dir.glob("*") if p.is_file()]) if masks_dir.is_dir() else []
    manifest_text = _load_text(manifest_path) if manifest_path.is_file() else ""
    forbidden_hits = [token for token in FORBIDDEN_FALLBACK_TOKENS if token in manifest_text]

    npz_shapes = {}
    for label, path in (("query_cache", query_cache), ("template_payload", template_payload), ("camera_sidecar", camera_sidecar), ("prior_maps", prior_maps)):
        if not path.is_file():
            continue
        try:
            with np.load(path, allow_pickle=False) as payload:
                npz_shapes[label] = {key: list(payload[key].shape) for key in payload.files[:12]}
        except Exception as exc:
            npz_shapes[label] = {"error": repr(exc)}
    if token_files:
        try:
            with np.load(token_files[0], allow_pickle=False) as payload:
                npz_shapes["token_cache_first"] = {"file": str(token_files[0]), "arrays": {key: list(payload[key].shape) for key in payload.files[:12]}}
        except Exception as exc:
            npz_shapes["token_cache_first"] = {"file": str(token_files[0]), "error": repr(exc)}

    missing = []
    if not manifest_path.is_file():
        missing.append("asset_manifest_json")
    if not query_cache.is_file():
        missing.append("query_cache")
    if not template_payload.is_file():
        missing.append("template_payload")
    if len(token_files) < 1:
        missing.append("token_cache_npz")
    if len(image_files) < 6:
        missing.append("images_ge_6")
    if len(mask_files) < 6:
        missing.append("masks_ge_6")
    if not camera_sidecar.is_file():
        missing.append("camera_sidecar")
    if not prior_maps.is_file():
        missing.append("prior_maps")

    verification = {
        "asset_dir": str(asset_dir),
        "manifest": _file_entry(manifest_path),
        "query_cache": _file_entry(query_cache),
        "template_payload": _file_entry(template_payload),
        "token_cache_npz_count": len(token_files),
        "image_count": len(image_files),
        "mask_count": len(mask_files),
        "camera_sidecar": _file_entry(camera_sidecar),
        "prior_maps": _file_entry(prior_maps),
        "npz_shapes": npz_shapes,
        "missing": missing,
        "forbidden_hits": forbidden_hits,
        "complete": not missing and not forbidden_hits,
    }
    summary = {
        "task": "v9_remote_cloud_asset_verification",
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "remote_v9_assets_verified" if verification["complete"] else "remote_v9_assets_blocked",
        "contract": {
            "research_only": True,
            "no_export": True,
            "no_predictions_write": True,
            "no_teacher_export": True,
            "no_candidate_export": True,
            "no_registry_write": True,
            "no_strict_pass_write": True,
            "formal_cloud_unblocked": False,
        },
        "remote_asset_subdir": asset_subdir,
        "remote_output_subdir": output_subdir,
        "remote_verification": verification,
        "decision": (
            "REMOTE_V9_ASSETS_READY: Modal volume can see real query/template/token/4K4D assets; no procedural fallback markers were found."
            if verification["complete"]
            else "REMOTE_V9_ASSETS_BLOCKED: missing assets or fallback markers remain; do not run Cloud-A training."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# V9 Remote Cloud Asset Verification",
                "",
                f"Status: `{summary['status']}`",
                "",
                f"- query_cache: `{verification['query_cache']['exists']}`",
                f"- template_payload: `{verification['template_payload']['exists']}`",
                f"- token_cache_npz_count: `{verification['token_cache_npz_count']}`",
                f"- images: `{verification['image_count']}`",
                f"- masks: `{verification['mask_count']}`",
                f"- camera_sidecar: `{verification['camera_sidecar']['exists']}`",
                f"- prior_maps: `{verification['prior_maps']['exists']}`",
                f"- forbidden_hits: `{verification['forbidden_hits']}`",
                "",
                "## Decision",
                "",
                summary["decision"],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_volume.commit()
    if not verification["complete"]:
        raise RuntimeError(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def _upload_tree(local_asset_dir: Path, remote_asset_subdir: str) -> int:
    local_asset_dir = local_asset_dir.expanduser().resolve()
    if not local_asset_dir.is_dir():
        raise FileNotFoundError(local_asset_dir)
    remote_asset_subdir = _normalize_subpath(remote_asset_subdir)
    files = [path for path in sorted(local_asset_dir.rglob("*")) if path.is_file()]
    with output_volume.batch_upload(force=True) as batch:
        for path in files:
            rel = path.relative_to(local_asset_dir).as_posix()
            batch.put_file(str(path), f"{remote_asset_subdir}/{rel}")
    output_volume.commit()
    return len(files)


def _download_volume_dir(remote_subdir: str, local_dir: Path) -> None:
    remote_subdir = _normalize_subpath(remote_subdir)
    local_dir = local_dir.expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_prefix = Path(remote_subdir)
    for entry in output_volume.listdir(remote_subdir, recursive=True):
        rel_path = Path(entry.path)
        try:
            rel_path = rel_path.relative_to(remote_prefix)
        except ValueError:
            pass
        dest_path = local_dir / rel_path
        if entry.type == modal.volume.FileEntryType.DIRECTORY:
            dest_path.mkdir(parents=True, exist_ok=True)
            continue
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with dest_path.open("wb") as handle:
            output_volume.read_file_into_fileobj(entry.path, handle)


@app.local_entrypoint()
def stage_and_verify(
    local_asset_dir: str,
    remote_asset_subdir: str = "surface_research_cloud_preflight/V9_cloud_asset_staging/assets",
    remote_output_subdir: str = "surface_research_cloud_preflight/V9_cloud_asset_staging_remote_verify",
    download_local_dir: str = "",
) -> None:
    remote_asset_subdir = _normalize_subpath(remote_asset_subdir)
    remote_output_subdir = _normalize_subpath(remote_output_subdir)
    uploaded = _upload_tree(Path(local_asset_dir), remote_asset_subdir)
    print(f"[v9-asset-staging] uploaded {uploaded} files to {remote_asset_subdir}")
    summary = verify_remote_assets.remote(remote_asset_subdir, remote_output_subdir)
    print("[v9-asset-staging] remote summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    local_dir = Path(download_local_dir).expanduser().resolve() if download_local_dir.strip() else REPO_ROOT / "output" / remote_output_subdir
    _download_volume_dir(remote_output_subdir, local_dir)
    print(f"[v9-asset-staging] downloaded remote verification to {local_dir}")


@app.local_entrypoint()
def download_run(remote_output_subdir: str, local_output_dir: str) -> None:
    _download_volume_dir(remote_output_subdir, Path(local_output_dir))
    print(f"[v9-asset-staging] downloaded artifacts to {Path(local_output_dir).resolve()}")
