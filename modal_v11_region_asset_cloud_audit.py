from __future__ import annotations

import json
import os
import time
from pathlib import Path, PurePosixPath
from typing import Any

import modal


REPO_ROOT = Path(__file__).resolve().parent
REMOTE_CODE_DIR = PurePosixPath("/workspace/vggt")
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
APP_NAME = os.environ.get("VGGT_MODAL_V11_AUDIT_APP_NAME", "vggt-v11-region-asset-cloud-audit")
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_V11_AUDIT_TIMEOUT_SEC", "3600"))

CODE_SYNC_IGNORE = [
    ".git",
    ".git/**",
    "__pycache__",
    "__pycache__/**",
    "output",
    "output/**",
    "reports",
    "reports/**",
]

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("numpy==1.26.4", "torch==2.3.1")
    .add_local_dir(str(REPO_ROOT / "tools"), remote_path=(REMOTE_CODE_DIR / "tools").as_posix(), ignore=CODE_SYNC_IGNORE)
    .add_local_dir(str(REPO_ROOT / "vggt"), remote_path=(REMOTE_CODE_DIR / "vggt").as_posix(), ignore=CODE_SYNC_IGNORE)
    .add_local_dir(str(REPO_ROOT / "external"), remote_path=(REMOTE_CODE_DIR / "external").as_posix(), ignore=CODE_SYNC_IGNORE)
)

app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


def _normalize_subpath(value: str) -> str:
    cleaned = (value or "").strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("output_subdir is empty")
    if ".." in Path(cleaned).parts:
        raise ValueError(f"Parent traversal is forbidden: {value!r}")
    lower = cleaned.lower()
    if "surface_research_cloud_preflight" not in lower:
        raise ValueError("V11 cloud audit output must be under surface_research_cloud_preflight")
    forbidden = ("strict_pass", "teacher_export", "candidate_export", "predictions", "formal_candidate")
    if any(word in lower for word in forbidden):
        raise ValueError(f"Forbidden output token in {value!r}")
    return cleaned


def _inventory(root: Path, patterns: tuple[str, ...]) -> list[str]:
    if not root.exists():
        return []
    out: list[str] = []
    for pattern in patterns:
        out.extend(str(path) for path in root.rglob(pattern))
    return sorted(set(out))


def _file_info(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "is_file": path.is_file(), "size": path.stat().st_size if path.exists() and path.is_file() else None}


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V11 Region Asset Cloud Audit",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only audit. No predictions, teacher/candidate package, registry, or strict pass write.",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Blocking Facts",
        "",
    ]
    lines.extend(f"- {item}" for item in summary.get("blocking_facts", []))
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


@app.function(image=IMAGE, cpu=2.0, memory=16 * 1024, timeout=TIMEOUT_SEC, volumes={REMOTE_OUTPUT_DIR.as_posix(): output_volume})
def run_v11_region_asset_cloud_audit(output_subdir: str) -> dict[str, Any]:
    subdir = _normalize_subpath(output_subdir)
    out = Path(str(REMOTE_OUTPUT_DIR / subdir))
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    code_root = Path(str(REMOTE_CODE_DIR))
    cloud_g = Path(str(REMOTE_OUTPUT_DIR / "surface_research_cloud_preflight/Cloud_G_V10"))
    cloud_g_tiers = {}
    for name, iteration in (("1k", "1000"), ("10k", "10000"), ("30k", "30000")):
        root = cloud_g / f"a5x3_2dgs_colmap_scene_{name}"
        ply = root / f"model_smoke/point_cloud/iteration_{iteration}/point_cloud.ply"
        cloud_g_tiers[name] = {"summary": _file_info(root / "summary.json"), "point_cloud": _file_info(ply)}
    hggt = code_root / "external/HGGT-main"
    hairgs = code_root / "external/hair-gs-master"
    hand_model = code_root / "vggt/models/human_hand_decoder.py"
    hair_model = code_root / "vggt/models/human_hair_strand_gaussian.py"
    import_results: dict[str, Any] = {}
    try:
        import sys

        sys.path.insert(0, str(code_root))
        from vggt.models.human_hand_decoder import HumanHandTokenResidualDecoder
        from vggt.models.human_hair_strand_gaussian import HumanHairStrandGaussian

        hand = HumanHandTokenResidualDecoder()
        hair = HumanHairStrandGaussian()
        import_results = {
            "hand_decoder_import": True,
            "hair_module_import": True,
            "hand_param_count": int(sum(p.numel() for p in hand.parameters())),
            "hair_param_count": int(sum(p.numel() for p in hair.parameters())),
        }
    except Exception as exc:
        import_results = {"hand_decoder_import": False, "hair_module_import": False, "error": repr(exc)}
    hggt_checkpoints = _inventory(hggt, ("*.pt", "*.pth", "*.ckpt", "*.npz"))
    hairgs_checkpoints = _inventory(hairgs, ("*.pt", "*.pth", "*.ckpt", "*.npz"))
    hairgs_dataset_files = _inventory(hairgs / "dataset", ("*.png", "*.jpg", "*.jpeg", "*.npy", "*.npz", "*.pkl", "*.obj"))
    blocking = []
    if not all(item["point_cloud"]["exists"] for item in cloud_g_tiers.values()):
        blocking.append("Cloud_G_V10 2DGS tier point clouds are incomplete on Modal volume.")
    if not hggt.exists() or not hggt_checkpoints:
        blocking.append("HGGT/hand route has no runnable checkpoint or MANO/HaMeR prior assets on cloud.")
    if not hairgs.exists() or not hairgs_dataset_files:
        blocking.append("HairGS route has source only; FLAME/hair dataset conversion assets are absent on cloud.")
    if not import_results.get("hand_decoder_import") or not import_results.get("hair_module_import"):
        blocking.append("V11 native hand/hair modules did not import on cloud.")
    summary = {
        "task": "v11_region_asset_cloud_audit",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - started, 3),
        "status": "research_cloud_asset_audit_complete_fail_closed" if blocking else "research_cloud_asset_audit_ready",
        "research_only": True,
        "no_export": True,
        "no_predictions_write": True,
        "no_registry_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_strict_pass_write": True,
        "formal_cloud_unblocked": False,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "cloud_g_tiers": cloud_g_tiers,
        "hggt": {"root": str(hggt), "exists": hggt.exists(), "checkpoint_like_files": hggt_checkpoints[:40], "checkpoint_like_count": len(hggt_checkpoints)},
        "hairgs": {
            "root": str(hairgs),
            "exists": hairgs.exists(),
            "checkpoint_like_files": hairgs_checkpoints[:40],
            "checkpoint_like_count": len(hairgs_checkpoints),
            "dataset_like_files": hairgs_dataset_files[:40],
            "dataset_like_count": len(hairgs_dataset_files),
        },
        "native_modules": {"hand_model": _file_info(hand_model), "hair_model": _file_info(hair_model), "imports": import_results},
        "blocking_facts": blocking,
        "decision": "Cloud confirms V11 hand/hair cannot be promoted without real external checkpoint/dataset or trained native checkpoints." if blocking else "Cloud assets are ready for next bounded research training.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(out / "report.md", summary)
    output_volume.commit()
    return summary


def _download_volume_dir(remote_subdir: str, local_dir: Path) -> None:
    subdir = _normalize_subpath(remote_subdir)
    local_dir = local_dir.expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_prefix = Path(subdir)
    for entry in output_volume.listdir(subdir, recursive=True):
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
def run(output_subdir: str = "surface_research_cloud_preflight/V11_region_asset_cloud_audit", download_local_dir: str = "") -> None:
    subdir = _normalize_subpath(output_subdir)
    print("[v11-cloud-audit] launching research-only cloud audit")
    print(json.dumps({"output_subdir": subdir}, indent=2, ensure_ascii=False))
    summary = run_v11_region_asset_cloud_audit.remote(subdir)
    print("[v11-cloud-audit] remote summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    local = Path(download_local_dir).expanduser().resolve() if download_local_dir.strip() else REPO_ROOT / "output" / subdir
    _download_volume_dir(subdir, local)
    print(f"[v11-cloud-audit] downloaded artifacts to {local}")
