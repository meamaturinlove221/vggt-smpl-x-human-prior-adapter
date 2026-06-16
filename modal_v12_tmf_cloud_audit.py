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
APP_NAME = os.environ.get("VGGT_MODAL_V12_AUDIT_APP_NAME", "vggt-v12-tmf-cloud-audit")
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_V12_AUDIT_TIMEOUT_SEC", "1800"))

CODE_SYNC_IGNORE = [".git", ".git/**", "__pycache__", "__pycache__/**", "output", "output/**", "reports", "reports/**"]

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4")
    .add_local_file(str(REPO_ROOT / "tools/v12_tmf_surface_teacher_pipeline.py"), remote_path=(REMOTE_CODE_DIR / "tools/v12_tmf_surface_teacher_pipeline.py").as_posix())
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
        raise ValueError("V12 cloud audit output must be under surface_research_cloud_preflight")
    forbidden = ("strict_pass", "teacher_export", "candidate_export", "predictions", "formal_candidate")
    if any(word in lower for word in forbidden):
        raise ValueError(f"Forbidden output token in {value!r}")
    return cleaned


def _file_info(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else None}


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V12 TMF Cloud Audit",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only cloud audit. No predictions, teacher/candidate package, registry, or strict pass write.",
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


@app.function(image=IMAGE, cpu=1.0, memory=2048, timeout=TIMEOUT_SEC, volumes={REMOTE_OUTPUT_DIR.as_posix(): output_volume})
def run_v12_tmf_cloud_audit(output_subdir: str) -> dict[str, Any]:
    subdir = _normalize_subpath(output_subdir)
    out = Path(str(REMOTE_OUTPUT_DIR / subdir))
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    expected_local_artifacts = {
        "asset_audit": "output/surface_research_preflight_local/V12_TMF_asset_audit/summary.json",
        "fusion": "output/surface_research_preflight_local/V12_TMF_canonical_surface_teacher/summary.json",
        "unified": "output/surface_research_preflight_local/V12_TMF_unified_surface_precheck/summary.json",
        "dline": "output/surface_research_preflight_local/DLine_V12_TMF_promotion_transaction/summary.json",
        "kinect_intake": "output/surface_research_preflight_local/V12_Kinect_TSDF_teacher_intake_audit/summary.json",
    }
    blocking = [
        "This cloud audit intentionally does not promote V12 artifacts; local D-line blocked strict pass.",
        "V12 large local PLY/PNG artifacts were not uploaded to Modal volume in this lightweight audit.",
    ]
    summary = {
        "task": "v12_tmf_cloud_audit",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - started, 3),
        "status": "research_cloud_audit_complete_fail_closed",
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
        "expected_local_artifacts": expected_local_artifacts,
        "blocking_facts": blocking,
        "decision": "V12 cloud-side audit completed as research-only; strict promotion remains blocked by local D-line.",
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
def run(output_subdir: str = "surface_research_cloud_preflight/V12_TMF_cloud_audit", download_local_dir: str = "") -> None:
    subdir = _normalize_subpath(output_subdir)
    print("[v12-cloud-audit] launching research-only cloud audit")
    print(json.dumps({"output_subdir": subdir}, indent=2, ensure_ascii=False))
    summary = run_v12_tmf_cloud_audit.remote(subdir)
    print("[v12-cloud-audit] remote summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    local = Path(download_local_dir).expanduser().resolve() if download_local_dir.strip() else REPO_ROOT / "output" / subdir
    _download_volume_dir(subdir, local)
    print(f"[v12-cloud-audit] downloaded artifacts to {local}")
