from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

import modal


REPO_ROOT = Path(__file__).resolve().parent
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
APP_NAME = os.environ.get("VGGT_MODAL_V9_HAIRGS_FIX_APP_NAME", "vggt-v9-hairgs-dependency-fix-smoke")
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_V9_HAIRGS_FIX_TIMEOUT_SEC", str(3 * 60 * 60)))
DEFAULT_REMOTE_SUBDIR = "surface_research_cloud_preflight/V9_backend_dependency_smokes/hair_gs_fix2"
REPORT_MD = REPO_ROOT / "reports" / "20260507_v9_hairgs_dependency_fix_status.md"
REPORT_JSON = REPO_ROOT / "reports" / "20260507_v9_hairgs_dependency_fix_status.json"
FORBIDDEN_OUTPUT_TOKENS = (
    "strict_pass",
    "teacher_export",
    "candidate_export",
    "predictions",
    "formal_candidate",
    "strict_gate_registry",
)


IMAGE_HAIRGS_FIX = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git",
        "build-essential",
        "gcc-11",
        "g++-11",
        "cmake",
        "ninja-build",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libgl1",
        "libegl1",
        "wget",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "FORCE_CUDA": "1",
            "CC": "/usr/bin/gcc-11",
            "CXX": "/usr/bin/g++-11",
            "CUDAHOSTCXX": "/usr/bin/g++-11",
            "TORCH_CUDA_ARCH_LIST": "8.6",
            "MAX_JOBS": "2",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    .run_commands("python -m pip install --upgrade pip wheel 'setuptools<70'")
)

app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


def _normalize_subpath(value: str) -> str:
    cleaned = (value or "").strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("Expected a non-empty remote subpath.")
    if ".." in Path(cleaned).parts:
        raise ValueError(f"Parent traversal is not allowed: {value!r}")
    lower = cleaned.lower()
    if lower != DEFAULT_REMOTE_SUBDIR.lower():
        raise ValueError(f"This smoke is locked to {DEFAULT_REMOTE_SUBDIR!r}.")
    if any(word in lower for word in FORBIDDEN_OUTPUT_TOKENS):
        raise ValueError(f"Research output contains forbidden token: {value!r}")
    return cleaned


def _classify_failure(text: str) -> str:
    lower = text.lower()
    if "no matching distribution found for pytorch3d" in lower:
        return "pytorch3d_no_pypi_distribution"
    if "not a supported wheel on this platform" in lower:
        return "pytorch3d_wheel_platform_mismatch"
    if "detected cuda version" in lower and "mismatches" in lower:
        return "cuda_torch_version_mismatch"
    if "__match_any_sync" in text and "undefined" in lower:
        return "cuda_arch_too_low_for_match_any_sync"
    if "unsupported gnu version" in lower or "gcc versions later than" in lower:
        return "gcc_cuda_unsupported"
    if "no module named 'pytorch3d'" in lower or 'no module named "pytorch3d"' in lower:
        return "pytorch3d_missing_at_import"
    if "no module named 'diff_gaussian_rasterization'" in lower:
        return "diff_gaussian_missing_at_import"
    if "killed" in lower and "building wheel" in lower:
        return "build_killed_probable_memory"
    if "runtimeerror: error compiling objects for extension" in lower:
        return "torch_extension_compile_failed"
    if "failed building wheel" in lower:
        return "wheel_build_failed"
    return "failed_unclassified"


def _interesting_excerpt(text: str, limit: int = 6000) -> str:
    markers = (
        "error:",
        "RuntimeError:",
        "__match_any_sync",
        "mismatches",
        "No matching distribution",
        "ModuleNotFoundError",
        "FAILED:",
        "unsupported GNU",
        "Failed building wheel",
    )
    lines = text.splitlines()
    hits: list[str] = []
    for index, line in enumerate(lines):
        if any(marker.lower() in line.lower() for marker in markers):
            start = max(0, index - 3)
            end = min(len(lines), index + 5)
            hits.extend(lines[start:end])
            hits.append("")
    excerpt = "\n".join(hits).strip()
    if not excerpt:
        excerpt = text[-limit:]
    return excerpt[-limit:]


def _run(
    name: str,
    cmd: list[str],
    cwd: Path | None,
    output_dir: Path,
    timeout: int = 900,
    extra_env: dict[str, str] | None = None,
) -> dict:
    started = time.time()
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    try:
        print(f"[hairgs-fix2] RUN {name} cwd={cwd or os.getcwd()} cmd={' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        lines: list[str] = []
        deadline = time.time() + timeout
        assert proc.stdout is not None
        with log_path.open("w", encoding="utf-8", errors="replace") as handle:
            while True:
                if time.time() > deadline:
                    proc.kill()
                    line = f"[timeout after {timeout}s]\n"
                    lines.append(line)
                    handle.write(line)
                    break
                line = proc.stdout.readline()
                if line:
                    lines.append(line)
                    handle.write(line)
                    print(f"[hairgs-fix2] {line.rstrip()}", flush=True)
                    continue
                if proc.poll() is not None:
                    break
                time.sleep(0.2)
        returncode = proc.wait(timeout=10)
        combined = "".join(lines)
        return {
            "command": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": int(returncode),
            "log_path": str(log_path.relative_to(output_dir)),
            "elapsed_sec": float(time.time() - started),
            "failure_class": "" if returncode == 0 else _classify_failure(combined),
            "error_excerpt": "" if returncode == 0 else _interesting_excerpt(combined),
            "stdout_tail": combined[-12000:],
        }
    except Exception as exc:
        message = repr(exc)
        log_path.write_text(message + "\n", encoding="utf-8")
        return {
            "command": cmd,
            "cwd": str(cwd) if cwd else None,
            "error": message,
            "returncode": -1,
            "log_path": str(log_path.relative_to(output_dir)),
            "elapsed_sec": float(time.time() - started),
            "failure_class": _classify_failure(message),
            "error_excerpt": message,
            "stdout_tail": message,
        }


def _write_remote_result(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hair_gs_fix2_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# V9 Hair-GS Dependency Fix2 Smoke",
        "",
        f"Status: `{payload['status']}`",
        "",
        payload["decision"],
        "",
        "## Version Strategy",
        "",
        "- D:/2d-gaussian-splatting-main uses Python 3.8, PyTorch 2.0.0, torchvision 0.15.0, and local CUDA extension installs.",
        "- Earlier Hair-GS cu124 smoke failed both PyPI PyTorch3D resolution and diff-gaussian CUDA compilation.",
        "- This fix2 smoke uses CUDA 11.8, Python 3.10, torch 2.0.0+cu118, gcc/g++ 11, and TORCH_CUDA_ARCH_LIST=8.6.",
        "",
        "## Step Results",
        "",
        "| Step | RC | Failure class | Log |",
        "| --- | ---: | --- | --- |",
    ]
    for name, step in payload["steps"].items():
        if step.get("skipped"):
            lines.append(f"| {name} | skip | {step.get('reason', '')} | |")
            continue
        rc = step.get("returncode", "")
        klass = step.get("failure_class", "")
        log = step.get("log_path", "")
        lines.append(f"| {name} | {rc} | {klass} | `{log}` |")
    if payload.get("primary_failure"):
        lines.extend(["", "## Primary Failure", "", payload["primary_failure"]])
    lines.extend(["", "No pass/export/predictions artifacts were written."])
    (output_dir / "hair_gs_fix2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _first_failed_step(steps: dict[str, dict]) -> tuple[str, dict] | None:
    for name, step in steps.items():
        if step.get("skipped"):
            continue
        if step.get("returncode", 0) != 0:
            return name, step
    return None


@app.function(
    image=IMAGE_HAIRGS_FIX,
    gpu=os.environ.get("VGGT_MODAL_V9_HAIRGS_FIX_GPU", "A10G"),
    memory=48 * 1024,
    timeout=TIMEOUT_SEC,
    volumes={REMOTE_OUTPUT_DIR.as_posix(): output_volume},
)
def run_hairgs_dependency_fix(remote_output_subdir: str) -> dict:
    out_subdir = _normalize_subpath(remote_output_subdir)
    output_dir = Path(str(REMOTE_OUTPUT_DIR / out_subdir))
    output_dir.mkdir(parents=True, exist_ok=True)
    repo = output_dir / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    logs_dir = output_dir / "logs"
    if logs_dir.exists():
        shutil.rmtree(logs_dir)

    started = time.time()
    steps: dict[str, dict] = {}
    common = {
        "task": "v9_hairgs_dependency_fix2_smoke",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": "hair-gs",
        "research_only": True,
        "no_export": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_strict_pass_write": True,
        "formal_cloud_unblocked": False,
        "remote_output_subdir": out_subdir,
        "version_strategy": {
            "cuda_image": "nvidia/cuda:11.8.0-devel-ubuntu22.04",
            "python": "3.10",
            "torch": "2.0.0+cu118",
            "torchvision": "0.15.1+cu118",
            "compiler": "gcc-11/g++-11",
            "torch_cuda_arch_list": "8.6",
            "reason": "Match the working 2DGS cu118/torch2.0 line and force sm_86 so __match_any_sync is available.",
        },
    }

    steps["clone"] = _run(
        "clone",
        ["git", "clone", "--depth", "1", "--recursive", "https://github.com/yimin-pan/hair-gs.git", str(repo)],
        None,
        output_dir,
        timeout=1200,
    )
    if repo.is_dir():
        steps["git_revision"] = _run("git_revision", ["git", "rev-parse", "HEAD"], repo, output_dir, timeout=60)
        steps["git_submodules"] = _run("git_submodules", ["git", "submodule", "status"], repo, output_dir, timeout=120)
    else:
        payload = {
            **common,
            "status": "blocked_clone_failed",
            "steps": steps,
            "elapsed_sec": float(time.time() - started),
            "primary_failure": steps["clone"].get("error_excerpt") or steps["clone"].get("stdout_tail", ""),
            "decision": "Hair-GS clone failed, so no dependency fix could be attempted.",
        }
        _write_remote_result(output_dir, payload)
        output_volume.commit()
        return payload

    steps["system_cuda"] = _run("system_cuda", ["bash", "-lc", "nvcc --version && gcc-11 --version && g++-11 --version"], None, output_dir, timeout=120)
    steps["torch_install"] = _run(
        "torch_install",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch==2.0.0",
            "torchvision==0.15.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu118",
        ],
        None,
        output_dir,
        timeout=1800,
    )
    steps["numpy_pin"] = _run("numpy_pin", [sys.executable, "-m", "pip", "install", "numpy<2"], None, output_dir, timeout=300)
    steps["setuptools_pin"] = _run("setuptools_pin", [sys.executable, "-m", "pip", "install", "setuptools<70"], None, output_dir, timeout=300)
    steps["torch_probe"] = _run(
        "torch_probe",
        [
            sys.executable,
            "-c",
            "import json, os, torch; print(json.dumps({'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'arch': os.environ.get('TORCH_CUDA_ARCH_LIST')}))",
        ],
        None,
        output_dir,
        timeout=120,
    )
    steps["base_deps"] = _run(
        "base_deps",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "Cython",
            "plyfile",
            "pyrr",
            "scipy",
            "opencv-python",
            "pyvista",
            "dreifus",
            "glfw",
            "smplx",
            "chumpy-fix",
            "tqdm",
            "tensorboard",
        ],
        None,
        output_dir,
        timeout=1200,
    )
    steps["c_utils_install"] = _run(
        "c_utils_install",
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-cache-dir", "./c_utils"],
        repo,
        output_dir,
        timeout=900,
    )
    steps["diff_gaussian_install"] = _run(
        "diff_gaussian_install",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-cache-dir",
            "submodules/diff-gaussian-rasterization",
        ],
        repo,
        output_dir,
        timeout=1800,
    )
    steps["simple_knn_install"] = _run(
        "simple_knn_install",
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-cache-dir", "submodules/simple-knn"],
        repo,
        output_dir,
        timeout=1200,
    )
    steps["self_install"] = _run(
        "self_install",
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-cache-dir", "-e", "."],
        repo,
        output_dir,
        timeout=900,
    )

    steps["pytorch3d_wheel_install"] = _run(
        "pytorch3d_wheel_install",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "pytorch3d",
            "-f",
            "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt200/download.html",
        ],
        None,
        output_dir,
        timeout=1200,
    )
    steps["pytorch3d_probe_after_wheel"] = _run(
        "pytorch3d_probe_after_wheel",
        [
            sys.executable,
            "-c",
            "import torch; from pytorch3d.ops import knn_points; from pytorch3d import transforms; x=torch.rand(1,4,3,device='cuda'); y=torch.rand(1,5,3,device='cuda'); out=knn_points(x,y,K=1); print('pytorch3d ok', out.dists.shape, torch.__version__, torch.version.cuda)",
        ],
        None,
        output_dir,
        timeout=180,
    )
    if steps["pytorch3d_probe_after_wheel"].get("returncode") != 0:
        steps["pytorch3d_source_deps"] = _run(
            "pytorch3d_source_deps",
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "fvcore", "iopath", "ninja"],
            None,
            output_dir,
            timeout=600,
        )
        steps["pytorch3d_source_install"] = _run(
            "pytorch3d_source_install",
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "--no-cache-dir",
                "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.4",
            ],
            None,
            output_dir,
            timeout=3600,
        )
        steps["pytorch3d_probe_after_source"] = _run(
            "pytorch3d_probe_after_source",
            [
                sys.executable,
                "-c",
                "import torch; from pytorch3d.ops import knn_points; from pytorch3d import transforms; x=torch.rand(1,4,3,device='cuda'); y=torch.rand(1,5,3,device='cuda'); out=knn_points(x,y,K=1); print('pytorch3d ok', out.dists.shape, torch.__version__, torch.version.cuda)",
            ],
            None,
            output_dir,
            timeout=180,
        )
        pytorch3d_ready = steps["pytorch3d_probe_after_source"].get("returncode") == 0
    else:
        steps["pytorch3d_source_deps"] = {"skipped": True, "reason": "wheel/index install imported successfully"}
        steps["pytorch3d_source_install"] = {"skipped": True, "reason": "wheel/index install imported successfully"}
        steps["pytorch3d_probe_after_source"] = {"skipped": True, "reason": "wheel/index install imported successfully"}
        pytorch3d_ready = True

    steps["hairgs_import_probe"] = _run(
        "hairgs_import_probe",
        [
            sys.executable,
            "-c",
            "import json, torch; import diff_gaussian_rasterization; from simple_knn._C import distCUDA2; from pytorch3d.ops import knn_points; from scene import HairGaussianModel; print(json.dumps({'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'hair_model': str(HairGaussianModel)}))",
        ],
        repo,
        output_dir,
        timeout=180,
    )

    diff_ready = steps["diff_gaussian_install"].get("returncode") == 0
    simple_ready = steps["simple_knn_install"].get("returncode") == 0
    self_ready = steps["self_install"].get("returncode") == 0
    import_ready = steps["hairgs_import_probe"].get("returncode") == 0
    if import_ready:
        status = "dependency_import_ready_missing_flame_and_hair_dataset"
        decision = "Hair-GS dependencies import on Modal with cu118/torch2.0/gcc11/sm86; next blocker is FLAME and hair dataset conversion, not CUDA dependency build."
        primary_failure = ""
    else:
        status = "blocked_dependency_build_or_import_failed"
        failed = _first_failed_step(steps)
        if failed:
            failed_name, failed_step = failed
            primary_failure = f"{failed_name}: {failed_step.get('failure_class', 'failed')}.\n\n{failed_step.get('error_excerpt', '')}"
        else:
            primary_failure = "Import probe failed without an earlier failed install step."
        decision = (
            "Hair-GS dependency smoke still failed. "
            f"diff_gaussian_ready={diff_ready}, simple_knn_ready={simple_ready}, self_ready={self_ready}, pytorch3d_ready={pytorch3d_ready}. "
            "See primary_failure and per-step logs for the narrowed compiler/version error."
        )

    payload = {
        **common,
        "status": status,
        "dependency_flags": {
            "diff_gaussian_ready": diff_ready,
            "simple_knn_ready": simple_ready,
            "self_ready": self_ready,
            "pytorch3d_ready": pytorch3d_ready,
            "hairgs_import_ready": import_ready,
        },
        "steps": steps,
        "elapsed_sec": float(time.time() - started),
        "primary_failure": primary_failure,
        "decision": decision,
    }
    _write_remote_result(output_dir, payload)
    output_volume.commit()
    return payload


def _download_volume_dir(remote_subdir: str, local_dir: Path) -> None:
    remote_subdir = _normalize_subpath(remote_subdir)
    local_dir = local_dir.expanduser().resolve()
    allowed_root = (REPO_ROOT / "output" / DEFAULT_REMOTE_SUBDIR).resolve()
    if local_dir != allowed_root:
        raise ValueError(f"Local download dir is locked to {allowed_root}")
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_prefix = Path(remote_subdir)
    for entry in output_volume.listdir(remote_subdir, recursive=True):
        rel_path = Path(entry.path)
        try:
            rel_path = rel_path.relative_to(remote_prefix)
        except ValueError:
            pass
        if rel_path.parts and rel_path.parts[0] == "repo":
            continue
        dest_path = local_dir / rel_path
        if entry.type == modal.volume.FileEntryType.DIRECTORY:
            dest_path.mkdir(parents=True, exist_ok=True)
            continue
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with dest_path.open("wb") as handle:
            output_volume.read_file_into_fileobj(entry.path, handle)


def _read_text_if_exists(path: Path, limit: int = 20000) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:limit]


def _local_reference_findings() -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    env_2dgs = Path(r"D:\2d-gaussian-splatting-main\environment.yml")
    mast3r_pyproject = Path(r"D:\MASt3R-SLAM-main\pyproject.toml")
    old_hair_summary = REPO_ROOT / "output" / "surface_research_cloud_preflight" / "V9_backend_dependency_smokes" / "hair_gs_fixed" / "hair_gs_summary.json"
    old_hair_log = REPO_ROOT / "tools" / "v9_backend_cloud_logs" / "hairgs_fixed_20260507_231119.out.log"

    env_text = _read_text_if_exists(env_2dgs)
    findings.append(
        {
            "path": str(env_2dgs),
            "finding": "2DGS local environment pins Python 3.8, PyTorch 2.0.0, torchvision 0.15.0, and installs diff-surfel/simple-knn as local CUDA extensions."
            if "pytorch=2.0.0" in env_text
            else "2DGS environment file not available or did not expose expected torch pin.",
        }
    )
    mast3r_text = _read_text_if_exists(mast3r_pyproject)
    findings.append(
        {
            "path": str(mast3r_pyproject),
            "finding": "MASt3R-SLAM pyproject build-system requires torch and setuptools==70.0.0; previous smoke failure was a build isolation/CUDA mismatch class, not directly reusable for Hair-GS."
            if "setuptools==70.0.0" in mast3r_text
            else "MASt3R-SLAM pyproject not available or did not expose expected build pin.",
        }
    )
    if old_hair_summary.is_file():
        summary = json.loads(old_hair_summary.read_text(encoding="utf-8"))
        failures = []
        for name, step in summary.get("steps", {}).items():
            if step.get("returncode", 0) != 0:
                tail = step.get("stdout_tail") or step.get("stderr_tail") or ""
                failures.append({"step": name, "failure_class": _classify_failure(tail), "excerpt": _interesting_excerpt(tail, 1200)})
        findings.append({"path": str(old_hair_summary), "finding": json.dumps(failures, ensure_ascii=False)})
    if old_hair_log.is_file():
        log_text = _read_text_if_exists(old_hair_log, limit=80000)
        match_any_hits = re.findall(r".{0,120}__match_any_sync.{0,180}", log_text)
        findings.append(
            {
                "path": str(old_hair_log),
                "finding": "Old diff-gaussian compile log contains __match_any_sync undefined, consistent with missing sm_70+ gencode/default low CUDA architecture."
                if match_any_hits
                else "Old Hair-GS full log was present but did not expose __match_any_sync in the scanned prefix.",
            }
        )
    return findings


def _write_local_reports(summary: dict, local_dir: Path, remote_output_subdir: str) -> None:
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    references = _local_reference_findings()
    payload = {
        "task": "v9_hairgs_dependency_fix_status",
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "remote_output_subdir": remote_output_subdir,
        "local_output_dir": str(local_dir),
        "summary_path": str(local_dir / "hair_gs_fix2_summary.json"),
        "report_path": str(local_dir / "hair_gs_fix2_report.md"),
        "status": summary.get("status"),
        "decision": summary.get("decision"),
        "dependency_flags": summary.get("dependency_flags", {}),
        "primary_failure": summary.get("primary_failure", ""),
        "local_reference_findings": references,
        "no_export": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_strict_pass_write": True,
        "formal_cloud_unblocked": False,
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# V9 Hair-GS Dependency Fix Status",
        "",
        "Research-only dependency smoke. No pass/export/predictions artifacts were written.",
        "",
        f"- status: `{payload['status']}`",
        f"- decision: {payload['decision']}",
        f"- local summary: `{payload['summary_path']}`",
        f"- local report: `{payload['report_path']}`",
        "",
        "## Local Reference Checks",
    ]
    for item in references:
        lines.append(f"- `{item['path']}`: {item['finding']}")
    lines.extend(["", "## Dependency Flags"])
    for key, value in payload["dependency_flags"].items():
        lines.append(f"- `{key}`: `{value}`")
    if payload["primary_failure"]:
        lines.extend(["", "## Primary Failure", "", payload["primary_failure"]])
    lines.extend(["", "## Changed Files", "", "- `modal_v9_hairgs_dependency_fix_smoke.py`", "- `reports/20260507_v9_hairgs_dependency_fix_status.json`", "- `reports/20260507_v9_hairgs_dependency_fix_status.md`", "- `output/surface_research_cloud_preflight/V9_backend_dependency_smokes/hair_gs_fix2/**`"])
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


@app.local_entrypoint()
def run_smoke(
    remote_output_subdir: str = DEFAULT_REMOTE_SUBDIR,
    download_local_dir: str = "",
) -> None:
    remote_output_subdir = _normalize_subpath(remote_output_subdir)
    local_dir = (REPO_ROOT / "output" / DEFAULT_REMOTE_SUBDIR).resolve()
    if download_local_dir.strip():
        requested = Path(download_local_dir).expanduser().resolve()
        if requested != local_dir:
            raise ValueError(f"download_local_dir is locked to {local_dir}")
    summary = run_hairgs_dependency_fix.remote(remote_output_subdir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    _download_volume_dir(remote_output_subdir, local_dir)
    _write_local_reports(summary, local_dir, remote_output_subdir)
    print(f"[hairgs-fix2] downloaded artifacts to {local_dir}")
    print(f"[hairgs-fix2] wrote reports to {REPORT_JSON} and {REPORT_MD}")
