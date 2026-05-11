from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = REPO_ROOT / "reports" / "V14_H14_R14"
OUTPUT_ROOT = REPO_ROOT / "output" / "V14_H14_R14"
REPO_STAGE = OUTPUT_ROOT / "public_repos"
DOWNLOAD_STAGE = OUTPUT_ROOT / "downloads"
LOG_PATH = OUTPUT_ROOT / "acquisition.log"

FORBIDDEN_OUTPUT_TOKENS = (
    "predictions",
    "teacher_export",
    "candidate_export",
    "strict_pass",
    "strict_gate_registry",
)

LICENSED_ASSET_SETS = {
    "mano_right": {
        "manual_source": "MANO website/account",
        "files_any": ("MANO_RIGHT.pkl",),
    },
    "mano_both": {
        "manual_source": "MANO website/account",
        "files_all": ("MANO_LEFT.pkl", "MANO_RIGHT.pkl"),
    },
    "smpl_basic": {
        "manual_source": "SMPL website/account",
        "files_any": (
            "SMPL_NEUTRAL.pkl",
            "SMPL_MALE.pkl",
            "SMPL_FEMALE.pkl",
            "basicmodel_neutral_lbs_10_207_0_v1.0.0.pkl",
        ),
    },
    "smplx": {
        "manual_source": "SMPL-X website/account",
        "files_any": (
            "SMPLX_NEUTRAL.pkl",
            "SMPLX_NEUTRAL.npz",
            "SMPLX_MALE.npz",
            "SMPLX_FEMALE.npz",
        ),
    },
    "smplx_helpers": {
        "manual_source": "SMPL-X/ExPose helper file links referenced by the upstream projects",
        "files_all": ("SMPLX_to_J14.pkl", "MANO_SMPLX_vertex_ids.pkl", "SMPL-X__FLAME_vertex_ids.npy"),
    },
    "flame_osx": {
        "manual_source": "FLAME website/account",
        "files_any": ("FLAME_NEUTRAL.pkl", "generic_model.pkl", "flame2023.pkl"),
    },
    "flame_hairgs": {
        "manual_source": "FLAME website/account plus HairGS dataset preparation",
        "files_all": (
            "flame2023.pkl",
            "flame_static_embedding.pkl",
            "flame_dynamic_embedding.npy",
            "FLAME_masks.pkl",
        ),
    },
}

ROUTES = {
    "WiLoR": {
        "line": "H",
        "kind": "hand",
        "repo_url": "https://github.com/rolpotamias/WiLoR.git",
        "repo_dir": "WiLoR",
        "upstream": "https://github.com/rolpotamias/WiLoR",
        "required_licensed_sets": ("mano_right",),
        "public_downloads": (
            {
                "name": "wilor_detector.pt",
                "url": "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt",
                "target": "WiLoR/pretrained_models/detector.pt",
                "required_for_runtime": True,
                "size_hint_gb": 0.06,
                "policy": "small",
            },
            {
                "name": "wilor_final.ckpt",
                "url": "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt",
                "target": "WiLoR/pretrained_models/wilor_final.ckpt",
                "required_for_runtime": True,
                "size_hint_gb": 2.6,
                "policy": "large",
            },
        ),
        "manual_public_items": (),
        "readiness_notes": (
            "Official demo still requires licensed MANO_RIGHT.pkl in mano_data/.",
        ),
    },
    "HaMeR": {
        "line": "H",
        "kind": "hand",
        "repo_url": "https://github.com/geopavlakos/hamer.git",
        "repo_dir": "HaMeR",
        "upstream": "https://github.com/geopavlakos/hamer",
        "required_licensed_sets": ("mano_right",),
        "public_downloads": (
            {
                "name": "hamer_demo_data.tar.gz",
                "url": "https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz",
                "target": "HaMeR/hamer_demo_data.tar.gz",
                "required_for_runtime": True,
                "size_hint_gb": 1.0,
                "policy": "large",
            },
        ),
        "manual_public_items": (
            "If the direct tarball fails, upstream fetch_demo_data.sh uses Google Drive/gdown.",
        ),
        "readiness_notes": (
            "Official setup requires MANO_RIGHT.pkl under _DATA/data/mano.",
            "Submodule/install readiness was not treated as a hand pass.",
        ),
    },
    "OSX": {
        "line": "H/R",
        "kind": "whole_body",
        "repo_url": "https://github.com/IDEA-Research/OSX.git",
        "repo_dir": "OSX",
        "upstream": "https://github.com/IDEA-Research/OSX",
        "required_licensed_sets": ("smpl_basic", "smplx", "smplx_helpers", "mano_both", "flame_osx"),
        "public_downloads": (),
        "manual_public_items": (
            "Pretrained OSX snapshots are linked from upstream README via Google Drive.",
            "ViTPose pretrained encoder links are upstream-managed.",
        ),
        "readiness_notes": (
            "Requires common/utils/human_model_files populated with SMPL, SMPL-X, MANO, and FLAME files.",
        ),
    },
    "SMPLer-X": {
        "line": "H/R",
        "kind": "whole_body",
        "repo_url": "https://github.com/MotrixLab/SMPLer-X.git",
        "repo_dir": "SMPLer-X",
        "upstream": "https://github.com/MotrixLab/SMPLer-X",
        "required_licensed_sets": ("smpl_basic", "smplx", "smplx_helpers"),
        "public_downloads": (
            {
                "name": "smpler_x_s32.pth.tar",
                "url": "https://huggingface.co/caizhongang/SMPLer-X/resolve/main/smpler_x_s32.pth.tar?download=true",
                "target": "SMPLer-X/pretrained_models/smpler_x_s32.pth.tar",
                "required_for_runtime": True,
                "size_hint_gb": 0.20,
                "policy": "small",
            },
            {
                "name": "mmdet_faster_rcnn_r50_fpn_1x_coco.pth",
                "url": "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
                "target": "SMPLer-X/pretrained_models/mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
                "required_for_runtime": True,
                "size_hint_gb": 0.16,
                "policy": "small",
            },
        ),
        "manual_public_items": (
            "Larger SMPLer-X B/L/H checkpoints are public but intentionally size-gated by this manager.",
        ),
        "readiness_notes": (
            "Upstream inference also needs an mmdet config file and legacy mmcv/mmpose environment.",
        ),
    },
    "SMPLest-X": {
        "line": "H/R",
        "kind": "whole_body",
        "repo_url": "https://github.com/MotrixLab/SMPLest-X.git",
        "repo_dir": "SMPLest-X",
        "upstream": "https://github.com/MotrixLab/SMPLest-X",
        "required_licensed_sets": ("smpl_basic", "smplx"),
        "public_downloads": (
            {
                "name": "smplest_x_h_config_base.py",
                "url": "https://huggingface.co/waanqii/SMPLest-X/resolve/main/config_base.py",
                "target": "SMPLest-X/pretrained_models/smplest_x_h/config_base.py",
                "required_for_runtime": True,
                "size_hint_gb": 0.001,
                "policy": "small",
            },
            {
                "name": "smplest_x_h.pth.tar",
                "url": "https://huggingface.co/waanqii/SMPLest-X/resolve/main/smplest_x_h.pth.tar?download=true",
                "target": "SMPLest-X/pretrained_models/smplest_x_h/smplest_x_h.pth.tar",
                "required_for_runtime": True,
                "size_hint_gb": 8.25,
                "policy": "large",
            },
        ),
        "manual_public_items": (
            "The public SMPLest-X-H weight is about 8.25 GB and is skipped unless the size cap allows it.",
        ),
        "readiness_notes": (
            "YOLOv8x is expected to auto-download during first inference, but no inference was run.",
        ),
    },
    "HairGS": {
        "line": "R",
        "kind": "hair",
        "repo_url": "https://github.com/yimin-pan/hair-gs.git",
        "repo_dir": "HairGS",
        "upstream": "https://github.com/yimin-pan/hair-gs",
        "required_licensed_sets": ("flame_hairgs",),
        "public_downloads": (),
        "manual_public_items": (
            "HairGS datasets require external/manual raw data and FLAME files before parsing.",
        ),
        "readiness_notes": (
            "Existing external/hair-gs-master was inspected, but V14 acquisition stages an official clone under output only.",
        ),
    },
    "GaussianHaircut": {
        "line": "R",
        "kind": "hair",
        "repo_url": "https://github.com/eth-ait/GaussianHaircut.git",
        "repo_dir": "GaussianHaircut",
        "upstream": "https://github.com/eth-ait/GaussianHaircut",
        "required_licensed_sets": ("flame_osx",),
        "public_downloads": (),
        "manual_public_items": (
            "Requires a user scene/raw.mp4 and the upstream FLAME fitting/reconstruction prerequisites.",
        ),
        "readiness_notes": (
            "Upstream lists CUDA 11.8 and Blender 3.6 for the full pipeline.",
        ),
    },
}

KNOWN_EXISTING_PATHS = {
    "HairGS": (REPO_ROOT / "external" / "hair-gs-master",),
    "HGGT_context_only": (REPO_ROOT / "external" / "HGGT-main",),
    "MUSt3R_context_only": (REPO_ROOT / "external" / "must3r",),
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def append_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8") if not LOG_PATH.exists() else None
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] {message}\n")


def ensure_output_scope() -> None:
    for path in (OUTPUT_ROOT, REPORT_ROOT, REPO_STAGE, DOWNLOAD_STAGE):
        resolved = path.resolve()
        allowed = (REPO_ROOT / "output" / "V14_H14_R14").resolve()
        allowed_report = (REPO_ROOT / "reports" / "V14_H14_R14").resolve()
        if not (resolved == allowed or resolved == allowed_report or allowed in resolved.parents or allowed_report in resolved.parents):
            raise RuntimeError(f"Refusing path outside V14_H14_R14 scope: {resolved}")
        lower = resolved.as_posix().lower()
        for token in FORBIDDEN_OUTPUT_TOKENS:
            if token in lower:
                raise RuntimeError(f"Refusing forbidden output token {token!r}: {resolved}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)


def run_command(cmd: list[str], *, cwd: Path | None = None, timeout: int = 120) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "cwd": cwd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip()[-4000:],
            "stderr": proc.stderr.strip()[-4000:],
            "elapsed_sec": round(time.time() - started, 3),
        }
    except FileNotFoundError as exc:
        return {
            "cmd": cmd,
            "cwd": cwd,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "elapsed_sec": round(time.time() - started, 3),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "cwd": cwd,
            "returncode": "timeout",
            "stdout": (exc.stdout or "").strip()[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "").strip()[-4000:] if isinstance(exc.stderr, str) else "",
            "elapsed_sec": round(time.time() - started, 3),
        }


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def count_files(root: Path, patterns: tuple[str, ...]) -> int:
    if not root.exists():
        return 0
    total = 0
    for pattern in patterns:
        total += sum(1 for _ in root.rglob(pattern))
    return total


def hash_file(path: Path, limit_bytes: int = 32 * 1024 * 1024) -> dict[str, Any]:
    if not path.is_file():
        return {"sha256_first_bytes": None, "hashed_bytes": 0}
    digest = hashlib.sha256()
    read = 0
    with path.open("rb") as handle:
        while read < limit_bytes:
            chunk = handle.read(min(1024 * 1024, limit_bytes - read))
            if not chunk:
                break
            digest.update(chunk)
            read += len(chunk)
    return {"sha256_first_bytes": digest.hexdigest(), "hashed_bytes": read}


def inspect_git_repo(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": path,
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "is_git_repo": False,
        "remote": None,
        "head": None,
        "branch": None,
        "file_count": 0,
        "checkpoint_like_count": 0,
    }
    if not path.is_dir():
        return info
    info["file_count"] = sum(1 for item in path.rglob("*") if item.is_file())
    info["checkpoint_like_count"] = count_files(path, ("*.pt", "*.pth", "*.ckpt", "*.pth.tar", "*.safetensors", "*.npz", "*.pkl"))
    inside = run_command(["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"], timeout=30)
    info["is_git_repo"] = inside["returncode"] == 0 and inside["stdout"].strip() == "true"
    if not info["is_git_repo"]:
        return info
    remote = run_command(["git", "-C", str(path), "remote", "-v"], timeout=30)
    head = run_command(["git", "-C", str(path), "rev-parse", "HEAD"], timeout=30)
    branch = run_command(["git", "-C", str(path), "branch", "--show-current"], timeout=30)
    status = run_command(["git", "-C", str(path), "status", "--short"], timeout=30)
    info.update(
        {
            "remote": remote["stdout"],
            "head": head["stdout"],
            "branch": branch["stdout"],
            "dirty_short": status["stdout"],
        }
    )
    return info


def acquire_repo(route_name: str, route: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    target = REPO_STAGE / route["repo_dir"]
    status: dict[str, Any] = {
        "route": route_name,
        "target": target,
        "repo_url": route["repo_url"],
        "acquire_requested": args.acquire,
        "clone_or_update": None,
    }
    if args.acquire:
        if target.exists() and (target / ".git").exists():
            append_log(f"{route_name}: existing staged git repo, fetching")
            fetch = run_command(["git", "-C", str(target), "fetch", "--depth", "1", "origin"], timeout=args.git_timeout)
            status["clone_or_update"] = {"operation": "fetch_existing", "result": fetch}
            if fetch["returncode"] == 0 and args.update_existing:
                pull = run_command(["git", "-C", str(target), "pull", "--ff-only"], timeout=args.git_timeout)
                status["clone_or_update"]["pull"] = pull
        elif target.exists():
            status["clone_or_update"] = {
                "operation": "blocked_existing_non_git",
                "reason": "Target exists but is not a git repository; not deleting anything automatically.",
            }
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            cmd = ["git", "clone", "--depth", "1"]
            if args.recurse_submodules:
                cmd.extend(["--recurse-submodules", "--shallow-submodules"])
            cmd.extend([route["repo_url"], str(target)])
            append_log(f"{route_name}: cloning {route['repo_url']} to {target}")
            result = run_command(cmd, timeout=args.git_timeout)
            status["clone_or_update"] = {"operation": "clone", "result": result}
    status["repo"] = inspect_git_repo(target)
    return status


def should_download(entry: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    if args.download_policy == "none":
        return False, "skipped_by_policy_none"
    if args.download_policy == "small" and entry.get("policy") != "small":
        return False, "skipped_by_policy_small_only"
    hint = float(entry.get("size_hint_gb") or 0.0)
    if hint > args.max_download_gb:
        return False, f"skipped_by_size_hint_{hint:g}gb_gt_cap_{args.max_download_gb:g}gb"
    return True, "allowed"


def head_content_length(url: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "v14-readiness-manager/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            return {
                "ok": True,
                "status": getattr(response, "status", None),
                "content_length": int(length) if length and length.isdigit() else None,
                "final_url": response.geturl(),
            }
    except Exception as exc:  # noqa: BLE001 - report exact upstream failure
        return {"ok": False, "error": repr(exc), "content_length": None, "final_url": None}


def download_file(entry: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    url = entry["url"]
    dest = DOWNLOAD_STAGE / entry["target"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    allowed, policy_reason = should_download(entry, args)
    status: dict[str, Any] = {
        "name": entry["name"],
        "url": url,
        "target": dest,
        "required_for_runtime": bool(entry.get("required_for_runtime")),
        "policy_reason": policy_reason,
        "status": None,
        "bytes": file_size(dest),
        "partial_bytes": file_size(partial),
        "head": None,
    }
    if dest.exists() and dest.stat().st_size > 0:
        status.update({"status": "present", "bytes": dest.stat().st_size, **hash_file(dest)})
        return status
    if not allowed:
        status["status"] = "skipped"
        return status
    head = head_content_length(url, timeout=args.head_timeout)
    status["head"] = head
    max_bytes = int(args.max_download_gb * 1024**3)
    if head.get("content_length") and int(head["content_length"]) > max_bytes:
        status["status"] = "skipped_size_from_head"
        status["reason"] = f"HEAD content length {head['content_length']} exceeds cap {max_bytes}."
        return status

    resume_at = file_size(partial)
    request_headers = {"User-Agent": "v14-readiness-manager/1.0"}
    mode = "wb"
    if resume_at:
        request_headers["Range"] = f"bytes={resume_at}-"
        mode = "ab"
    request = urllib.request.Request(url, headers=request_headers)
    append_log(f"download: {entry['name']} -> {dest}")
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=args.download_timeout) as response:
            with partial.open(mode + "") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    if partial.stat().st_size > max_bytes:
                        raise RuntimeError(f"download exceeded cap {max_bytes} bytes")
        partial.replace(dest)
        status.update(
            {
                "status": "downloaded",
                "bytes": dest.stat().st_size,
                "partial_bytes": 0,
                "elapsed_sec": round(time.time() - started, 3),
                **hash_file(dest),
            }
        )
        return status
    except Exception as exc:  # noqa: BLE001 - keep exact blocker text
        status.update(
            {
                "status": "download_failed",
                "error": repr(exc),
                "bytes": file_size(dest),
                "partial_bytes": file_size(partial),
                "elapsed_sec": round(time.time() - started, 3),
            }
        )
        return status


def acquire_downloads(route: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    return [download_file(entry, args) for entry in route.get("public_downloads", ())]


def find_named_files(roots: list[Path], names: tuple[str, ...], max_hits: int = 12) -> dict[str, list[Path]]:
    result = {name: [] for name in names}
    remaining = set(names)
    for root in roots:
        if not root.exists() or not remaining:
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            name = path.name
            if name in remaining:
                result[name].append(path)
                if sum(len(v) for v in result.values()) >= max_hits:
                    break
        remaining = {name for name in names if not result[name]}
    return result


def search_roots() -> list[Path]:
    roots: list[Path] = [
        REPO_ROOT / "external_models",
        REPO_ROOT / "external",
        OUTPUT_ROOT,
    ]
    for key in (
        "MANO_ROOT",
        "SMPL_ROOT",
        "SMPLX_ROOT",
        "FLAME_ROOT",
        "HUMAN_MODEL_FILES",
        "SMPL_MODEL_DIR",
        "SMPLX_MODEL_DIR",
        "MANO_MODEL_DIR",
        "FLAME_MODEL_DIR",
    ):
        value = os.environ.get(key)
        if value:
            roots.append(Path(value).expanduser())
    seen: set[str] = set()
    unique = []
    for root in roots:
        try:
            resolved = str(root.resolve())
        except OSError:
            resolved = str(root)
        if resolved not in seen:
            unique.append(root)
            seen.add(resolved)
    return unique


def inspect_licensed_assets() -> dict[str, Any]:
    roots = search_roots()
    all_names: set[str] = set()
    for spec in LICENSED_ASSET_SETS.values():
        all_names.update(spec.get("files_any", ()))
        all_names.update(spec.get("files_all", ()))
    hits = find_named_files(roots, tuple(sorted(all_names)))
    sets: dict[str, Any] = {}
    for set_name, spec in LICENSED_ASSET_SETS.items():
        files_any = tuple(spec.get("files_any", ()))
        files_all = tuple(spec.get("files_all", ()))
        if files_any:
            present = [name for name in files_any if hits.get(name)]
            ok = bool(present)
            missing = [] if ok else list(files_any)
        else:
            present = [name for name in files_all if hits.get(name)]
            missing = [name for name in files_all if not hits.get(name)]
            ok = not missing
        sets[set_name] = {
            "ok": ok,
            "manual_source": spec["manual_source"],
            "present": present,
            "missing": missing,
            "hits": {name: hits.get(name, []) for name in set(files_any + files_all)},
        }
    return {"search_roots": roots, "sets": sets}


def inspect_known_existing_paths() -> dict[str, Any]:
    return {
        name: [inspect_git_repo(path) for path in paths]
        for name, paths in KNOWN_EXISTING_PATHS.items()
    }


def system_probe() -> dict[str, Any]:
    probes = {
        "python": [sys.executable, "--version"],
        "git": ["git", "--version"],
        "conda": ["conda", "--version"],
        "nvidia_smi": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        "nvcc": ["nvcc", "--version"],
        "blender": ["blender", "--version"],
    }
    result = {
        "platform": platform.platform(),
        "cwd": Path.cwd(),
        "env": {
            key: os.environ.get(key)
            for key in (
                "CUDA_PATH",
                "CUDA_HOME",
                "HF_HOME",
                "MANO_ROOT",
                "SMPL_ROOT",
                "SMPLX_ROOT",
                "FLAME_ROOT",
            )
            if os.environ.get(key)
        },
        "commands": {},
    }
    for name, cmd in probes.items():
        result["commands"][name] = run_command(cmd, timeout=30)
    return result


def route_readiness(
    route_name: str,
    route: dict[str, Any],
    repo_status: dict[str, Any],
    download_status: list[dict[str, Any]],
    licensed_assets: dict[str, Any],
    system: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    repo_ok = bool(repo_status.get("repo", {}).get("exists"))
    if not repo_ok:
        blockers.append(f"{route_name}: public source repo is not staged under {REPO_STAGE / route['repo_dir']}.")
    required_downloads = [item for item in download_status if item.get("required_for_runtime")]
    missing_downloads = [
        item
        for item in required_downloads
        if item.get("status") not in {"present", "downloaded"}
    ]
    for item in missing_downloads:
        blockers.append(f"{route_name}: required public download {item['name']} is {item.get('status')} ({item.get('policy_reason') or item.get('reason') or item.get('error')}).")
    for set_name in route.get("required_licensed_sets", ()):
        spec = licensed_assets["sets"][set_name]
        if not spec["ok"]:
            missing = ", ".join(spec["missing"])
            blockers.append(f"{route_name}: missing licensed/manual asset set {set_name}: {missing}; source={spec['manual_source']}.")
    if route_name in {"HairGS", "GaussianHaircut"}:
        blockers.append(f"{route_name}: no V14 user-provided hair scene package with RGB views/masks/COLMAP or raw.mp4 was validated.")
    if route_name == "GaussianHaircut":
        blender_ok = system["commands"]["blender"]["returncode"] == 0
        if not blender_ok:
            blockers.append("GaussianHaircut: Blender 3.6 executable was not found on PATH.")
    if route_name in {"HairGS", "GaussianHaircut"}:
        nvcc = system["commands"]["nvcc"]
        if nvcc["returncode"] != 0:
            blockers.append(f"{route_name}: CUDA compiler nvcc is not callable in the current shell.")
    if route_name == "SMPLer-X":
        warnings.append("SMPLer-X: mmdet config file still needs to be staged or generated from upstream before inference.")
    if route_name == "HaMeR" and repo_status.get("repo", {}).get("is_git_repo") and not (REPO_STAGE / route["repo_dir"] / "third-party").exists():
        warnings.append("HaMeR: third-party submodules were not verified as initialized.")
    return {
        "route": route_name,
        "line": route["line"],
        "kind": route["kind"],
        "repo_ok": repo_ok,
        "download_ok_count": sum(1 for item in download_status if item.get("status") in {"present", "downloaded"}),
        "required_download_count": len(required_downloads),
        "licensed_sets": {
            set_name: licensed_assets["sets"][set_name]["ok"]
            for set_name in route.get("required_licensed_sets", ())
        },
        "runnable": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "notes": route.get("readiness_notes", ()),
    }


def build_blocker_checklist(route_results: dict[str, Any]) -> list[dict[str, Any]]:
    checklist: list[dict[str, Any]] = []
    for route_name, payload in route_results.items():
        readiness = payload["readiness"]
        for blocker in readiness["blockers"]:
            checklist.append(
                {
                    "route": route_name,
                    "status": "open",
                    "blocker": blocker,
                    "required_action": required_action_for_blocker(blocker),
                    "license_guard": "manual_only" if "licensed/manual" in blocker or "licensed" in blocker else "public_or_environment",
                }
            )
    checklist.append(
        {
            "route": "ALL",
            "status": "open",
            "blocker": "No hand/hair strict pass, teacher export, candidate export, predictions write, or registry promotion was produced by V14.",
            "required_action": "Run real route-specific inference only after code, public checkpoints, licensed assets, dependencies, and input data are complete.",
            "license_guard": "promotion_guard",
        }
    )
    return checklist


def required_action_for_blocker(blocker: str) -> str:
    if "licensed/manual asset" in blocker:
        return "User must download the named MANO/SMPL/SMPL-X/FLAME files from the official licensed source and place them in the upstream-required folder or set an env var."
    if "required public download" in blocker:
        return "Re-run this manager with a larger --max-download-gb or fix the upstream download URL/network issue."
    if "public source repo" in blocker:
        return "Re-run with --acquire after network/git access is working, or manually clone the official repo under output/V14_H14_R14/public_repos."
    if "hair scene package" in blocker:
        return "Provide real hair input data in the upstream-required format; do not substitute synthetic/proxy hair artifacts."
    if "Blender" in blocker:
        return "Install Blender 3.6 or add it to PATH for GaussianHaircut readiness."
    if "CUDA compiler" in blocker:
        return "Expose a supported CUDA toolkit/nvcc for the hair CUDA extension build."
    return "Resolve the route-specific prerequisite and re-run the manager."


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# V14 H14/R14 External Hand/Hair Asset Manager",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only readiness/acquisition artifact. It does not write predictions, teacher/candidate packages, registry state, or any strict pass claim.",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Scope Guard",
        "",
        f"- Tool owner file: `{REPO_ROOT / 'tools' / 'v14_external_hand_hair_asset_manager.py'}`",
        f"- Report root: `{REPORT_ROOT}`",
        f"- Output root: `{OUTPUT_ROOT}`",
        "- Licensed MANO/SMPL/SMPL-X/FLAME assets were only searched for; none were fabricated or downloaded.",
        "",
        "## Successful Repo/Checkpoint/Download Status",
        "",
        "| Route | Repo staged | Downloads present/downloaded | Runnable |",
        "|---|---:|---:|---:|",
    ]
    for route_name, payload in summary["routes"].items():
        readiness = payload["readiness"]
        lines.append(
            f"| {route_name} | {str(readiness['repo_ok'])} | "
            f"{readiness['download_ok_count']}/{readiness['required_download_count']} | {str(readiness['runnable'])} |"
        )
    lines.extend(["", "## Exact Blocker Checklist", ""])
    for item in summary["blocker_checklist"]:
        lines.append(f"- [ ] {item['route']}: {item['blocker']}")
        lines.append(f"  Required action: {item['required_action']}")
    lines.extend(["", "## Route Details", ""])
    for route_name, payload in summary["routes"].items():
        readiness = payload["readiness"]
        lines.append(f"### {route_name}")
        lines.append(f"- repo: `{payload['repo_status']['repo'].get('path')}`")
        lines.append(f"- repo head: `{payload['repo_status']['repo'].get('head')}`")
        lines.append(f"- downloads: `{readiness['download_ok_count']}/{readiness['required_download_count']}` required present/downloaded")
        lines.append(f"- runnable: `{readiness['runnable']}`")
        if readiness["warnings"]:
            for warning in readiness["warnings"]:
                lines.append(f"- warning: {warning}")
        for note in readiness["notes"]:
            lines.append(f"- note: {note}")
        lines.append("")
    lines.extend(["## Sources Used For Route Rules", ""])
    for route_name, route in ROUTES.items():
        lines.append(f"- {route_name}: {route['upstream']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="V14 H14/R14 external hand/hair public asset acquisition and readiness manager.")
    parser.add_argument("--acquire", action=argparse.BooleanOptionalAction, default=True, help="Clone/fetch official public repos into output/V14_H14_R14.")
    parser.add_argument("--update-existing", action=argparse.BooleanOptionalAction, default=False, help="Fast-forward staged repos that already exist.")
    parser.add_argument("--recurse-submodules", action=argparse.BooleanOptionalAction, default=False, help="Clone submodules for new repos. Off by default to keep V14 bounded.")
    parser.add_argument("--download-policy", choices=("none", "small", "all"), default="small", help="Which public downloads to attempt.")
    parser.add_argument("--max-download-gb", type=float, default=1.0, help="Per-file download cap.")
    parser.add_argument("--git-timeout", type=int, default=900)
    parser.add_argument("--head-timeout", type=int, default=30)
    parser.add_argument("--download-timeout", type=int, default=900)
    parser.add_argument("--output-json", type=Path, default=REPORT_ROOT / "readiness.json")
    parser.add_argument("--output-md", type=Path, default=REPORT_ROOT / "readiness.md")
    args = parser.parse_args()

    ensure_output_scope()
    append_log("V14 manager start")

    system = system_probe()
    licensed_assets = inspect_licensed_assets()
    existing_paths = inspect_known_existing_paths()
    routes: dict[str, Any] = {}
    for route_name, route in ROUTES.items():
        repo_status = acquire_repo(route_name, route, args)
        download_status = acquire_downloads(route, args)
        readiness = route_readiness(route_name, route, repo_status, download_status, licensed_assets, system)
        routes[route_name] = {
            "route_config": route,
            "repo_status": repo_status,
            "downloads": download_status,
            "manual_public_items": route.get("manual_public_items", ()),
            "readiness": readiness,
        }
    blocker_checklist = build_blocker_checklist(routes)
    runnable_count = sum(1 for payload in routes.values() if payload["readiness"]["runnable"])
    summary = {
        "task": "V14_H14_R14_external_hand_hair_asset_manager",
        "created_utc": utc_now(),
        "status": "blocked_no_hand_hair_pass" if blocker_checklist else "ready_no_pass_claim",
        "decision": (
            "V14 began bounded public acquisition/readiness, but no hand/hair route is cleared for a pass. "
            "Manual licensed MANO/SMPL/SMPL-X/FLAME assets and route-specific inputs/dependencies remain blockers."
        ),
        "scope_guard": {
            "owned_tool": REPO_ROOT / "tools" / "v14_external_hand_hair_asset_manager.py",
            "report_root": REPORT_ROOT,
            "output_root": OUTPUT_ROOT,
            "no_predictions_write": True,
            "no_teacher_export": True,
            "no_candidate_export": True,
            "no_registry_write": True,
            "no_strict_pass_claim": True,
            "licensed_assets_fabricated": False,
        },
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "hand_hair_pass_claimed": False,
        "runnable_route_count": runnable_count,
        "routes": routes,
        "blocker_checklist": blocker_checklist,
        "licensed_asset_audit": licensed_assets,
        "known_existing_paths": existing_paths,
        "system_probe": system,
        "log": LOG_PATH,
    }
    write_json(args.output_json, summary)
    write_markdown(summary, args.output_md)
    append_log(f"V14 manager wrote {args.output_json} and {args.output_md}")
    print(json.dumps(json_ready({"status": summary["status"], "json": args.output_json, "md": args.output_md}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
