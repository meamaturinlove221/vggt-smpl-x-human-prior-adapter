from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

import modal


REPO_ROOT = Path(__file__).resolve().parent
REMOTE_CODE_DIR = PurePosixPath("/workspace/vggt")
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
APP_NAME = os.environ.get("VGGT_MODAL_V9_FUS3D3_APP_NAME", "vggt-v9-b-fus3d3-real-asset-train")
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_V9_FUS3D3_TIMEOUT_SEC", str(2 * 60 * 60)))

FORBIDDEN_OUTPUT_TOKENS = ("strict_pass", "teacher_export", "candidate_export", "predictions", "formal_candidate", "strict_gate_registry")

CODE_SYNC_IGNORE = [
    ".git",
    ".git/**",
    "__pycache__",
    "__pycache__/**",
    ".venv*",
    ".venv*/**",
    "output",
    "output/**",
    "reports",
    "reports/**",
]

TORCH_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install("numpy==1.26.1", "Pillow", "scipy", "scikit-image", "PyYAML", "torch==2.3.1", "torchvision==0.18.1")
    .add_local_dir(str(REPO_ROOT / "tools"), remote_path=(REMOTE_CODE_DIR / "tools").as_posix(), ignore=CODE_SYNC_IGNORE)
    .add_local_dir(str(REPO_ROOT / "training"), remote_path=(REMOTE_CODE_DIR / "training").as_posix(), ignore=CODE_SYNC_IGNORE)
)

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
        raise ValueError("V9 research output must include surface_research_cloud_preflight.")
    if any(word in lower for word in FORBIDDEN_OUTPUT_TOKENS):
        raise ValueError(f"V9 research output contains forbidden token: {value!r}")
    return cleaned


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@app.function(
    image=TORCH_IMAGE,
    memory=32 * 1024,
    timeout=TIMEOUT_SEC,
    volumes={REMOTE_OUTPUT_DIR.as_posix(): output_volume},
)
def run_real_asset_train(
    remote_asset_subdir: str,
    remote_output_subdir: str,
    max_steps: int,
    max_cases: int,
    max_hours: float,
) -> dict:
    asset_subdir = _normalize_subpath(remote_asset_subdir)
    output_subdir = _normalize_subpath(remote_output_subdir)
    asset_dir = Path(str(REMOTE_OUTPUT_DIR / asset_subdir))
    output_dir = Path(str(REMOTE_OUTPUT_DIR / output_subdir))
    output_dir.mkdir(parents=True, exist_ok=True)

    query_cache = asset_dir / "query_cache" / "b_fus3d_query_evidence_cache.npz"
    template_payload = asset_dir / "template" / "connected_human_surface_template_payload_self_describing.npz"
    wrapper_summary_path = output_dir / "v9_cloud_a_real_asset_wrapper_summary.json"
    status_md = output_dir / "v9_cloud_a_real_asset_status.md"
    status_json = output_dir / "v9_cloud_a_real_asset_status.json"

    guard = {
        "task": "v9_cloud_a_b_fus3d3_real_asset_train_preflight",
        "research_only": True,
        "no_export": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_strict_pass_write": True,
        "formal_cloud_unblocked": False,
        "remote_asset_subdir": asset_subdir,
        "remote_output_subdir": output_subdir,
        "query_cache_exists": query_cache.is_file(),
        "template_payload_exists": template_payload.is_file(),
        "allow_procedural_fallback": False,
    }
    cmd = [
        sys.executable,
        str(Path(str(REMOTE_CODE_DIR)) / "tools" / "b_fus3d2_human_dataset_train.py"),
        "--query-cache",
        str(query_cache),
        "--template-payload",
        str(template_payload),
        "--output-dir",
        str(output_dir),
        "--status-report",
        str(status_md),
        "--status-json",
        str(status_json),
        "--max-steps",
        str(int(max_steps)),
        "--max-cases",
        str(int(max_cases)),
        "--max-hours",
        str(float(max_hours)),
        "--overwrite",
    ]
    started = time.time()
    result = subprocess.run(cmd, cwd=str(REMOTE_CODE_DIR), check=False, capture_output=True, text=True)
    elapsed_sec = time.time() - started
    train_summary = _load_json(output_dir / "summary.json")
    procedural_fallback_used = bool(
        train_summary.get("procedural_fallback_used")
        or train_summary.get("genealogy", {}).get("procedural_fallback_used")
        or "procedural_human_surface_sdf_case" in json.dumps(train_summary, ensure_ascii=False)
    )
    summary = {
        **guard,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "completed" if result.returncode == 0 and not procedural_fallback_used else "failed",
        "returncode": int(result.returncode),
        "elapsed_sec": float(elapsed_sec),
        "command": cmd,
        "stdout_tail": result.stdout[-8000:],
        "stderr_tail": result.stderr[-8000:],
        "train_summary_present": bool(train_summary),
        "procedural_fallback_used": bool(procedural_fallback_used),
        "train_success": bool(train_summary.get("success", False)),
        "research_progress": bool(train_summary.get("research_progress", False)),
        "comparison": train_summary.get("comparison", {}),
        "decision": (
            "CLOUD_A_REAL_ASSET_PREFLIGHT_COMPLETE_NO_FALLBACK: remote training consumed staged query/template assets. This is still research-only and not a candidate/teacher."
            if result.returncode == 0 and not procedural_fallback_used
            else "CLOUD_A_REAL_ASSET_PREFLIGHT_FAILED: do not use this output for B-Fus3D3 claims."
        ),
    }
    wrapper_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "v9_cloud_a_real_asset_wrapper_report.md").write_text(
        "\n".join(
            [
                "# V9 Cloud-A Real Asset Train Preflight",
                "",
                f"Status: `{summary['status']}`",
                "",
                f"- query_cache_exists: `{summary['query_cache_exists']}`",
                f"- template_payload_exists: `{summary['template_payload_exists']}`",
                f"- allow_procedural_fallback: `{summary['allow_procedural_fallback']}`",
                f"- procedural_fallback_used: `{summary['procedural_fallback_used']}`",
                f"- train_success: `{summary['train_success']}`",
                f"- research_progress: `{summary['research_progress']}`",
                f"- returncode: `{summary['returncode']}`",
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
    if result.returncode != 0 or procedural_fallback_used:
        raise RuntimeError(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


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
def run_train(
    remote_asset_subdir: str = "surface_research_cloud_preflight/V9_cloud_asset_staging/assets",
    remote_output_subdir: str = "surface_research_cloud_preflight/Cloud_A_V9/b_fus3d3_real_asset_train_preflight",
    max_steps: int = 5000,
    max_cases: int = 8,
    max_hours: float = 1.0,
    download_local_dir: str = "",
) -> None:
    remote_asset_subdir = _normalize_subpath(remote_asset_subdir)
    remote_output_subdir = _normalize_subpath(remote_output_subdir)
    summary = run_real_asset_train.remote(remote_asset_subdir, remote_output_subdir, int(max_steps), int(max_cases), float(max_hours))
    print("[v9-b-fus3d3] remote summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    local_dir = Path(download_local_dir).expanduser().resolve() if download_local_dir.strip() else REPO_ROOT / "output" / remote_output_subdir
    _download_volume_dir(remote_output_subdir, local_dir)
    print(f"[v9-b-fus3d3] downloaded remote artifacts to {local_dir}")


@app.local_entrypoint()
def download_run(remote_output_subdir: str, local_output_dir: str) -> None:
    _download_volume_dir(remote_output_subdir, Path(local_output_dir))
    print(f"[v9-b-fus3d3] downloaded artifacts to {Path(local_output_dir).resolve()}")
