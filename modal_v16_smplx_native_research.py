from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

import modal


REPO_ROOT = Path(__file__).resolve().parent
REMOTE_CODE_DIR = PurePosixPath("/workspace/vggt")
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
APP_NAME = os.environ.get("VGGT_MODAL_V16_SMPLX_RESEARCH_APP_NAME", "vggt-v16-smplx-native-research")
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_V16_SMPLX_RESEARCH_TIMEOUT_SEC", str(2 * 60 * 60)))

FORBIDDEN_OUTPUT_TOKENS = (
    "strict_pass",
    "teacher_export",
    "candidate_export",
    "formal_candidate",
    "strict_gate_registry",
    "candidate_gate",
    "predictions",
)

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

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install("numpy==1.26.4", "torch==2.3.1", "torchvision==0.18.1")
    .add_local_file(
        str(REPO_ROOT / "tools" / "v16_vggt_smplx_microfit_runner.py"),
        remote_path=(REMOTE_CODE_DIR / "tools" / "v16_vggt_smplx_microfit_runner.py").as_posix(),
    )
    .add_local_dir(str(REPO_ROOT / "vggt"), remote_path=(REMOTE_CODE_DIR / "vggt").as_posix(), ignore=CODE_SYNC_IGNORE)
)

app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


def _normalize_subpath(value: str, *, require_research: bool = True) -> str:
    cleaned = (value or "").strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("Expected a non-empty remote subpath.")
    if ".." in Path(cleaned).parts:
        raise ValueError(f"Parent traversal is forbidden: {value!r}")
    lower = cleaned.lower()
    if require_research and "surface_research_cloud_preflight" not in lower:
        raise ValueError("V16 Modal output must stay under surface_research_cloud_preflight.")
    if any(token in lower for token in FORBIDDEN_OUTPUT_TOKENS):
        raise ValueError(f"V16 Modal path contains a forbidden formal-output token: {value!r}")
    return cleaned


def _file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "size": path.stat().st_size if path.is_file() else None,
    }


def _write_guard(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


@app.function(
    image=IMAGE,
    cpu=4.0,
    memory=24 * 1024,
    timeout=TIMEOUT_SEC,
    volumes={REMOTE_OUTPUT_DIR.as_posix(): output_volume},
)
def run_v16_research_microfit(
    remote_case_subdir: str,
    remote_output_subdir: str,
    max_views: int,
    target_size: int,
    steps: int,
) -> dict[str, Any]:
    output_subdir = _normalize_subpath(remote_output_subdir, require_research=True)
    case_subdir = _normalize_subpath(remote_case_subdir, require_research=False)
    output_dir = Path(str(REMOTE_OUTPUT_DIR / output_subdir))
    case_root = Path(str(REMOTE_OUTPUT_DIR / case_subdir))
    output_dir.mkdir(parents=True, exist_ok=True)

    guard = {
        "task": "v16_smplx_native_modal_research_microfit",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "research_only": True,
        "formal_candidate_train_infer_export": "blocked",
        "formal_cloud_unblocked": False,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "no_predictions_npz_formal_path": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_package_write": True,
        "no_strict_pass_write": True,
        "remote_case_subdir": case_subdir,
        "remote_output_subdir": output_subdir,
        "case_root": _file_info(case_root),
    }
    _write_guard(output_dir / "v16_modal_research_guard.json", guard)

    if not (case_root / "inputs.npz").is_file() or not (case_root / "targets.npz").is_file():
        summary = {
            **guard,
            "status": "blocked_missing_remote_case_research_only",
            "decision": "V16 Modal research skeleton is valid, but the SMPL-X prior case was not staged on the output volume.",
        }
        _write_guard(output_dir / "summary.json", summary)
        (output_dir / "summary.md").write_text(
            "\n".join(
                [
                    "# V16 Modal SMPL-X Native Research",
                    "",
                    f"Status: `{summary['status']}`",
                    "",
                    summary["decision"],
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        output_volume.commit()
        return summary

    cmd = [
        sys.executable,
        str(Path(str(REMOTE_CODE_DIR)) / "tools" / "v16_vggt_smplx_microfit_runner.py"),
        "--case-root",
        str(case_root),
        "--output-dir",
        str(output_dir / "V16_vggt_smplx_microfit_runner"),
        "--output-json",
        str(output_dir / "v16_vggt_smplx_microfit_runner.json"),
        "--output-md",
        str(output_dir / "v16_vggt_smplx_microfit_runner.md"),
        "--rollup-json",
        str(output_dir / "v16_execution_rollup.json"),
        "--rollup-md",
        str(output_dir / "v16_execution_rollup.md"),
        "--max-views",
        str(int(max_views)),
        "--target-size",
        str(int(target_size)),
        "--steps",
        str(int(steps)),
        "--device",
        "cpu",
        "--execute",
    ]
    started = time.time()
    result = subprocess.run(cmd, cwd=str(REMOTE_CODE_DIR), check=False, capture_output=True, text=True)
    summary = {
        **guard,
        "status": "completed_research_only" if result.returncode == 0 else "failed_research_only",
        "returncode": int(result.returncode),
        "elapsed_sec": round(time.time() - started, 3),
        "command": cmd,
        "stdout_tail": result.stdout[-8000:],
        "stderr_tail": result.stderr[-8000:],
        "decision": "V16 Modal lane ran only the research microfit guard path; no formal candidate, teacher, registry, package, pass, or formal prediction artifact was written.",
    }
    _write_guard(output_dir / "v16_modal_research_summary.json", summary)
    output_volume.commit()
    if result.returncode != 0:
        raise RuntimeError(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def _download_volume_dir(remote_subdir: str, local_dir: Path) -> None:
    remote_subdir = _normalize_subpath(remote_subdir, require_research=True)
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
def run(
    remote_case_subdir: str = "surface_research_cloud_preflight/V16_smplx_native_prior_case",
    remote_output_subdir: str = "surface_research_cloud_preflight/V16_vggt_smplx_microfit_runner",
    max_views: int = 2,
    target_size: int = 56,
    steps: int = 2,
    download_local_dir: str = "",
) -> None:
    output_subdir = _normalize_subpath(remote_output_subdir, require_research=True)
    case_subdir = _normalize_subpath(remote_case_subdir, require_research=False)
    print("[v16-smplx-research] launching research-only Modal microfit skeleton")
    print(json.dumps({"case": case_subdir, "output": output_subdir}, indent=2, ensure_ascii=False))
    summary = run_v16_research_microfit.remote(case_subdir, output_subdir, int(max_views), int(target_size), int(steps))
    print("[v16-smplx-research] remote summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    local_dir = (
        Path(download_local_dir).expanduser().resolve()
        if download_local_dir.strip()
        else REPO_ROOT / "output" / output_subdir
    )
    _download_volume_dir(output_subdir, local_dir)
    print(f"[v16-smplx-research] downloaded research artifacts to {local_dir}")
