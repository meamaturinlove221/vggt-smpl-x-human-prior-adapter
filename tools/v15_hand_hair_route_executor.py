from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import v15_required_licensed_assets as licensed


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
V14_STAGE = REPO_ROOT / "output" / "V14_H14_R14"
D_ROOT = Path("D:/")

DEFAULT_ASSET_JSON = REPORTS / "20260508_v15_required_licensed_assets.json"
DEFAULT_ASSET_MD = REPORTS / "20260508_v15_required_licensed_assets.md"
DEFAULT_JSON = REPORTS / "20260508_v15_hair_hand_readiness.json"
DEFAULT_MD = REPORTS / "20260508_v15_hair_hand_readiness.md"

FORBIDDEN_OUTPUT_TOKENS = (
    "predictions",
    "teacher_export",
    "candidate_export",
    "strict_pass",
    "strict_gate_registry",
)

ROUTES: dict[str, dict[str, Any]] = {
    "WiLoR": {
        "line": "H",
        "kind": "hand",
        "repo_url": "https://github.com/rolpotamias/WiLoR.git",
        "aliases": ("WiLoR", "wilor"),
        "source_markers": ("demo.py", "wilor", "mano_data"),
        "required_asset_sets": ("mano_right",),
        "required_file_groups": {
            "detector": ("detector.pt",),
            "model_checkpoint": ("wilor_final.ckpt", "*.ckpt"),
        },
        "input_groups": {},
        "env_commands": (),
        "notes": ("Official route still needs licensed MANO_RIGHT.pkl.",),
    },
    "HaMeR": {
        "line": "H",
        "kind": "hand",
        "repo_url": "https://github.com/geopavlakos/hamer.git",
        "aliases": ("HaMeR", "hamer"),
        "source_markers": ("demo.py", "hamer", "fetch_demo_data.sh"),
        "required_asset_sets": ("mano_right",),
        "required_file_groups": {
            "hamer_checkpoint_or_demo_bundle": ("hamer.ckpt", "*hamer*.ckpt", "hamer_demo_data.tar.gz"),
        },
        "input_groups": {},
        "env_commands": (),
        "notes": ("Source alone is not a runnable hand route without the official checkpoint bundle and MANO.",),
    },
    "OSX": {
        "line": "H/R",
        "kind": "whole_body",
        "repo_url": "https://github.com/IDEA-Research/OSX.git",
        "aliases": ("OSX", "osx"),
        "source_markers": ("main", "common", "README.md"),
        "required_asset_sets": ("smpl_basic", "smplx", "smplx_helpers", "mano_both", "flame_core"),
        "required_file_groups": {
            "osx_checkpoint": ("*.pth.tar", "*.pth", "*.pt"),
        },
        "input_groups": {},
        "env_commands": (),
        "notes": ("Whole-body route is hand-capable only after human_model_files and checkpoints are populated.",),
    },
    "SMPLer-X": {
        "line": "H/R",
        "kind": "whole_body",
        "repo_url": "https://github.com/MotrixLab/SMPLer-X.git",
        "aliases": ("SMPLer-X", "SMPLerX", "smpler-x", "smplerx"),
        "source_markers": ("main", "common", "README.md"),
        "required_asset_sets": ("smpl_basic", "smplx", "smplx_helpers"),
        "required_file_groups": {
            "smpler_x_checkpoint": ("smpler_x_s32.pth.tar", "smpler_x_*.pth.tar", "*.pth.tar"),
            "detector_checkpoint": ("faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth", "*faster_rcnn*.pth"),
        },
        "input_groups": {},
        "env_commands": (),
        "notes": ("Legacy mmcv/mmdet environment is not validated by this asset worker.",),
    },
    "HairGS": {
        "line": "R",
        "kind": "hair",
        "repo_url": "https://github.com/yimin-pan/hair-gs.git",
        "aliases": ("HairGS", "hair-gs", "hair-gs-master", "hairgs"),
        "source_markers": ("train.py", "scene", "gaussian_renderer"),
        "required_asset_sets": ("flame_hairgs",),
        "required_file_groups": {},
        "input_groups": {
            "hair_scene_or_dataset": ("transforms_train.json", "cameras.json", "*.mp4", "*.ply", "*.obj", "*.npy", "*.npz"),
        },
        "env_commands": ("nvcc",),
        "notes": ("HairGS source without FLAME embeddings and a real hair scene is not an ownership asset.",),
    },
    "GaussianHaircut": {
        "line": "R",
        "kind": "hair",
        "repo_url": "https://github.com/eth-ait/GaussianHaircut.git",
        "aliases": ("GaussianHaircut", "gaussianhaircut"),
        "source_markers": ("README.md", "src", "scripts"),
        "required_asset_sets": ("flame_core",),
        "required_file_groups": {},
        "input_groups": {
            "raw_scene_video_or_reconstruction": ("raw.mp4", "*.mp4", "cameras.json", "transforms_train.json"),
        },
        "env_commands": ("nvcc", "blender"),
        "notes": ("GaussianHaircut also needs the upstream scene/reconstruction prerequisites.",),
    },
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_ready(value: Any) -> Any:
    return licensed.json_ready(value)


def write_json(path: Path, payload: Any) -> None:
    licensed.write_json(path, payload)


def run_command(cmd: list[str], *, cwd: Path | None = None, timeout: int = 20) -> dict[str, Any]:
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
            "stdout": (proc.stdout or "").strip()[-4000:],
            "stderr": (proc.stderr or "").strip()[-4000:],
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
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "cmd": cmd,
            "cwd": cwd,
            "returncode": "timeout",
            "stdout": stdout.strip()[-4000:],
            "stderr": stderr.strip()[-4000:],
            "elapsed_sec": round(time.time() - started, 3),
        }


def safe_stat(path: Path) -> dict[str, Any]:
    return licensed.safe_stat(path)


def d_root_route_candidates(route: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    if D_ROOT.exists():
        for alias in route["aliases"]:
            candidates.append(D_ROOT / alias)
    return candidates


def workspace_route_candidates(route_name: str, route: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for alias in route["aliases"]:
        candidates.append(REPO_ROOT / "external" / alias)
    if route_name == "HairGS":
        candidates.append(REPO_ROOT / "external" / "hair-gs-master")
    candidates.append(V14_STAGE / "public_repos" / route_name)
    if route_name == "HairGS":
        candidates.append(V14_STAGE / "public_repos" / "HairGS")
    return candidates


def route_candidate_paths(route_name: str, route: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    paths.extend(d_root_route_candidates(route))
    paths.extend(workspace_route_candidates(route_name, route))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            resolved = str(path.resolve())
        except OSError:
            resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def route_search_roots(route_name: str, route: dict[str, Any]) -> list[Path]:
    roots = route_candidate_paths(route_name, route)
    roots.append(V14_STAGE / "downloads" / route_name)
    for alias in route["aliases"]:
        roots.append(V14_STAGE / "downloads" / alias)
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = str(root.resolve())
        except OSError:
            resolved = str(root)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(root)
    return unique


def git_lock_files(path: Path) -> list[Path]:
    git_dir = path / ".git"
    if not git_dir.exists():
        return []
    try:
        return list(git_dir.rglob("*.lock"))
    except OSError:
        return []


def marker_hits(path: Path, markers: tuple[str, ...]) -> list[str]:
    hits: list[str] = []
    for marker in markers:
        if (path / marker).exists():
            hits.append(marker)
    return hits


def inspect_repo_candidate(path: Path, route: dict[str, Any], git_timeout: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        **safe_stat(path),
        "git_lock_files": [],
        "is_git_repo": False,
        "remote": None,
        "head": None,
        "branch": None,
        "source_marker_hits": [],
        "source_ready": False,
        "blocked_reason": None,
    }
    if not path.exists() or not path.is_dir():
        return row
    row["source_marker_hits"] = marker_hits(path, tuple(route.get("source_markers", ())))
    locks = git_lock_files(path)
    row["git_lock_files"] = locks
    if locks:
        row["blocked_reason"] = "git_lock_or_interrupted_clone"
        row["source_ready"] = False
        return row

    inside = run_command(["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"], timeout=git_timeout)
    row["git_probe"] = inside
    row["is_git_repo"] = inside["returncode"] == 0 and inside["stdout"].strip() == "true"
    if row["is_git_repo"]:
        row["remote"] = run_command(["git", "-C", str(path), "remote", "-v"], timeout=git_timeout).get("stdout")
        row["head"] = run_command(["git", "-C", str(path), "rev-parse", "HEAD"], timeout=git_timeout).get("stdout")
        row["branch"] = run_command(["git", "-C", str(path), "branch", "--show-current"], timeout=git_timeout).get("stdout")

    has_source_markers = bool(row["source_marker_hits"])
    if row["is_git_repo"] and has_source_markers:
        row["source_ready"] = True
    elif has_source_markers and path.name.lower() in {"hair-gs-master", "hair-gs", "hairgs"}:
        row["source_ready"] = True
        row["source_ready_note"] = "non_git_extracted_source_with_markers"
    else:
        row["blocked_reason"] = "missing_source_markers"
    return row


def iter_files_bounded(root: Path, *, max_files: int = 30000, max_seconds: float = 10.0) -> tuple[list[Path], dict[str, Any]]:
    started = time.time()
    files: list[Path] = []
    meta: dict[str, Any] = {
        "root": root,
        "exists": root.exists(),
        "scanned_files": 0,
        "scanned_dirs": 0,
        "truncated": False,
        "truncation_reason": None,
        "errors": [],
    }
    if not root.exists():
        return files, meta
    stack = [root]
    while stack:
        if time.time() - started > max_seconds:
            meta["truncated"] = True
            meta["truncation_reason"] = f"max_seconds_{max_seconds:g}"
            break
        current = stack.pop()
        if licensed.should_skip_dir(current):
            continue
        try:
            with current.iterdir() as iterator:
                for path in iterator:
                    try:
                        if path.is_dir():
                            if not licensed.should_skip_dir(path):
                                stack.append(path)
                                meta["scanned_dirs"] += 1
                            continue
                        if not path.is_file():
                            continue
                    except OSError as exc:
                        meta["errors"].append({"path": path, "error": repr(exc)})
                        continue
                    files.append(path)
                    meta["scanned_files"] += 1
                    if meta["scanned_files"] >= max_files:
                        meta["truncated"] = True
                        meta["truncation_reason"] = f"max_files_{max_files}"
                        stack.clear()
                        break
        except OSError as exc:
            meta["errors"].append({"path": current, "error": repr(exc)})
    meta["elapsed_sec"] = round(time.time() - started, 3)
    return files, meta


def find_matching_files(roots: list[Path], patterns: tuple[str, ...]) -> dict[str, Any]:
    group_hits: dict[str, list[dict[str, Any]]] = {pattern: [] for pattern in patterns}
    scan_meta: list[dict[str, Any]] = []
    for root in roots:
        files, meta = iter_files_bounded(root)
        scan_meta.append(meta)
        for path in files:
            name_lower = path.name.lower()
            for pattern in patterns:
                if fnmatch.fnmatch(name_lower, pattern.lower()):
                    group_hits[pattern].append(safe_stat(path))
    return {
        "patterns": patterns,
        "ok": any(group_hits.values()),
        "hits": {pattern: hits for pattern, hits in group_hits.items() if hits},
        "scan_meta": scan_meta,
    }


def evaluate_file_groups(roots: list[Path], groups: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for group_name, patterns in groups.items():
        rows[group_name] = find_matching_files(roots, patterns)
    missing = [name for name, row in rows.items() if not row["ok"]]
    return {"groups": rows, "ok": not missing, "missing_groups": missing}


def command_status(commands: tuple[str, ...]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for command in commands:
        rows[command] = {"path": shutil.which(command), "ok": shutil.which(command) is not None}
    return {"commands": rows, "ok": all(row["ok"] for row in rows.values())}


def safe_remote_probe(route_name: str, route: dict[str, Any], timeout: int, requested: bool) -> dict[str, Any]:
    if not requested:
        return {"requested": False, "status": "skipped"}
    result = run_command(["git", "ls-remote", "--heads", "--refs", route["repo_url"]], timeout=timeout)
    return {
        "requested": True,
        "status": "ok" if result["returncode"] == 0 else "failed_or_timeout",
        "route": route_name,
        "result": result,
    }


def ownership_artifact_candidates(route_name: str) -> list[Path]:
    roots = [
        REPO_ROOT / "output" / "surface_research_preflight_local",
        REPO_ROOT / "output" / "surface_research_cloud_preflight",
        V14_STAGE,
    ]
    route_tokens = {
        "WiLoR": ("wilor", "hand"),
        "HaMeR": ("hamer", "hand"),
        "OSX": ("osx", "hand", "hair"),
        "SMPLer-X": ("smpler", "hand"),
        "HairGS": ("hairgs", "hair"),
        "GaussianHaircut": ("gaussianhaircut", "hair"),
    }[route_name]
    candidates: list[Path] = []
    suffixes = (".ply", ".npz", ".json", ".pt", ".pth", ".ckpt")
    for root in roots:
        files, _ = iter_files_bounded(root, max_files=20000, max_seconds=5.0)
        for path in files:
            lower = path.as_posix().lower()
            if path.suffix.lower() not in suffixes:
                continue
            if any(token in lower for token in route_tokens) and "ownership" in lower:
                candidates.append(path)
    return candidates[:24]


def route_readiness(
    route_name: str,
    route: dict[str, Any],
    asset_summary: dict[str, Any],
    *,
    git_timeout: int,
    probe_remotes: bool,
    remote_timeout: int,
) -> dict[str, Any]:
    candidates = route_candidate_paths(route_name, route)
    repo_rows = [inspect_repo_candidate(path, route, git_timeout) for path in candidates]
    search_roots = route_search_roots(route_name, route)
    source_ready = any(row.get("source_ready") for row in repo_rows)
    locked_candidates = [row for row in repo_rows if row.get("git_lock_files")]

    required_sets = tuple(route["required_asset_sets"])
    licensed_rows = {
        set_name: asset_summary["asset_sets"][set_name]
        for set_name in required_sets
    }
    missing_asset_sets = [set_name for set_name, row in licensed_rows.items() if not row["ok"]]
    file_groups = evaluate_file_groups(search_roots, route.get("required_file_groups", {}))
    input_groups = evaluate_file_groups(search_roots, route.get("input_groups", {}))
    env = command_status(tuple(route.get("env_commands", ())))
    remote_probe = safe_remote_probe(route_name, route, remote_timeout, probe_remotes)
    ownership_candidates = ownership_artifact_candidates(route_name)

    blockers: list[str] = []
    warnings: list[str] = []
    if not source_ready:
        if locked_candidates:
            blockers.append(f"{route_name}: source staging includes locked/interrupted git directory; not repaired automatically.")
        else:
            blockers.append(f"{route_name}: no complete source checkout found in D:/, workspace external, or output/V14_H14_R14 staging.")
    for set_name in missing_asset_sets:
        row = licensed_rows[set_name]
        missing = ", ".join(row.get("missing") or [])
        blockers.append(f"{route_name}: missing licensed/manual asset set {set_name}: {missing}; source={row['manual_source']}.")
    for group_name in file_groups["missing_groups"]:
        blockers.append(f"{route_name}: missing required checkpoint/download group {group_name}.")
    for group_name in input_groups["missing_groups"]:
        blockers.append(f"{route_name}: missing required real input/dataset group {group_name}.")
    for command, row in env["commands"].items():
        if not row["ok"]:
            blockers.append(f"{route_name}: required executable {command} was not found on PATH.")
    if route_name in {"HairGS", "GaussianHaircut"}:
        warnings.append(f"{route_name}: hair topology ownership needs a real subject scene; source/demo assets are not enough.")
    if route_name in {"HaMeR", "OSX"} and source_ready and not file_groups["ok"]:
        warnings.append(f"{route_name}: local source exists, but model weights are not present.")

    runnable = (
        source_ready
        and not missing_asset_sets
        and file_groups["ok"]
        and input_groups["ok"]
        and env["ok"]
    )
    ownership_ready = bool(runnable and ownership_candidates)
    if runnable and not ownership_ready:
        blockers.append(f"{route_name}: prerequisites look runnable, but no V15 ownership output artifact was found or generated.")
    return {
        "route": route_name,
        "line": route["line"],
        "kind": route["kind"],
        "repo_url": route["repo_url"],
        "candidate_paths": candidates,
        "repo_candidates": repo_rows,
        "search_roots": search_roots,
        "source_ready": source_ready,
        "licensed_ready": not missing_asset_sets,
        "licensed_sets": {
            set_name: {
                "ok": row["ok"],
                "present": row["present"],
                "missing": row["missing"],
                "manual_source": row["manual_source"],
            }
            for set_name, row in licensed_rows.items()
        },
        "file_group_status": file_groups,
        "input_group_status": input_groups,
        "env_status": env,
        "remote_probe": remote_probe,
        "runnable": runnable,
        "ownership_ready": ownership_ready,
        "ownership_artifact_candidates": ownership_candidates,
        "warnings": warnings,
        "blockers": blockers,
        "notes": route["notes"],
    }


def build_summary(
    asset_summary: dict[str, Any],
    *,
    git_timeout: int = 20,
    probe_remotes: bool = False,
    remote_timeout: int = 20,
) -> dict[str, Any]:
    route_rows = {
        route_name: route_readiness(
            route_name,
            route,
            asset_summary,
            git_timeout=git_timeout,
            probe_remotes=probe_remotes,
            remote_timeout=remote_timeout,
        )
        for route_name, route in ROUTES.items()
    }
    runnable_routes = [name for name, row in route_rows.items() if row["runnable"]]
    not_runnable_routes = [name for name, row in route_rows.items() if not row["runnable"]]
    hand_ready = any(row["ownership_ready"] for row in route_rows.values() if row["kind"] in {"hand", "whole_body"})
    hair_ready = any(row["ownership_ready"] for row in route_rows.values() if row["kind"] == "hair")
    blockers: list[str] = []
    for name, row in route_rows.items():
        blockers.extend(row["blockers"])
    blockers.append("V15: no fake ownership pass was written; hand_ownership_ready and hair_ownership_ready require real route outputs.")
    status = (
        "v15_hand_hair_ownership_ready"
        if hand_ready and hair_ready
        else "v15_hand_hair_readiness_blocked_no_ownership_pass"
    )
    return {
        "task": "v15_hand_hair_route_executor",
        "created_utc": utc_now(),
        "status": status,
        "scope_guard": {
            "owned_tools": [
                REPO_ROOT / "tools" / "v15_required_licensed_assets.py",
                REPO_ROOT / "tools" / "v15_hand_hair_route_executor.py",
            ],
            "report_json": DEFAULT_JSON,
            "report_md": DEFAULT_MD,
            "read_only_route_audit": True,
            "no_predictions_write": True,
            "no_teacher_export": True,
            "no_candidate_export": True,
            "no_registry_write": True,
            "no_strict_pass_claim": True,
            "forbidden_output_tokens": FORBIDDEN_OUTPUT_TOKENS,
        },
        "asset_report_status": asset_summary.get("status"),
        "asset_report_path": DEFAULT_ASSET_JSON,
        "routes": route_rows,
        "routes_runnable": runnable_routes,
        "routes_not_runnable": not_runnable_routes,
        "hand_ownership_ready": hand_ready,
        "hair_ownership_ready": hair_ready,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "hand_hair_pass_claimed": False,
        "blockers": blockers,
        "decision": (
            "V15 hand/hair asset readiness remains blocked. Some source trees exist, but no route has the full combination "
            "of source, checkpoint/downloads, licensed MANO/SMPL-X/FLAME assets, runtime prerequisites, and real ownership outputs."
        ),
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# V15 Hand/Hair Readiness",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Read-only route readiness report. It does not write predictions, teacher/candidate packages, registries, or strict pass state.",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Ownership",
        "",
        f"- hand_ownership_ready: `{summary['hand_ownership_ready']}`",
        f"- hair_ownership_ready: `{summary['hair_ownership_ready']}`",
        f"- strict_candidate_passes: `{summary['strict_candidate_passes']}`",
        f"- strict_teacher_passes: `{summary['strict_teacher_passes']}`",
        "",
        "## Route Matrix",
        "",
        "| Route | Source | Licensed assets | Checkpoints/downloads | Inputs/env | Runnable | Ownership ready |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for route_name, row in summary["routes"].items():
        files_ok = row["file_group_status"]["ok"]
        inputs_ok = row["input_group_status"]["ok"] and row["env_status"]["ok"]
        lines.append(
            f"| {route_name} | {str(row['source_ready'])} | {str(row['licensed_ready'])} | "
            f"{str(files_ok)} | {str(inputs_ok)} | {str(row['runnable'])} | {str(row['ownership_ready'])} |"
        )
    lines.extend(["", "## Runnable Routes", ""])
    if summary["routes_runnable"]:
        lines.extend(f"- {item}" for item in summary["routes_runnable"])
    else:
        lines.append("- none")
    lines.extend(["", "## Not Runnable Routes", ""])
    lines.extend(f"- {item}" for item in summary["routes_not_runnable"])
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- {item}" for item in summary["blockers"])
    lines.extend(["", "## Route Details", ""])
    for route_name, row in summary["routes"].items():
        lines.append(f"### {route_name}")
        lines.append(f"- repo_url: `{row['repo_url']}`")
        lines.append(f"- source_ready: `{row['source_ready']}`")
        for repo_row in row["repo_candidates"]:
            if repo_row["exists"]:
                lock_count = len(repo_row.get("git_lock_files") or [])
                lines.append(
                    f"- candidate: `{repo_row['path']}` source_ready=`{repo_row.get('source_ready')}` "
                    f"is_git_repo=`{repo_row.get('is_git_repo')}` locks=`{lock_count}`"
                )
        for warning in row["warnings"]:
            lines.append(f"- warning: {warning}")
        for note in row["notes"]:
            lines.append(f"- note: {note}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def load_asset_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="V15 hand/hair route readiness executor/report writer.")
    parser.add_argument("--asset-json", type=Path, default=DEFAULT_ASSET_JSON)
    parser.add_argument("--asset-md", type=Path, default=DEFAULT_ASSET_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--refresh-assets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--probe-remotes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--git-timeout", type=int, default=20)
    parser.add_argument("--remote-timeout", type=int, default=20)
    parser.add_argument("--max-files-per-root", type=int, default=75000)
    parser.add_argument("--max-seconds-per-root", type=float, default=20.0)
    args = parser.parse_args()

    if args.refresh_assets or not args.asset_json.exists():
        asset_summary = licensed.build_summary(
            max_files_per_root=args.max_files_per_root,
            max_seconds_per_root=args.max_seconds_per_root,
        )
        licensed.write_json(args.asset_json, asset_summary)
        licensed.write_markdown(asset_summary, args.asset_md)
    else:
        asset_summary = load_asset_summary(args.asset_json)

    summary = build_summary(
        asset_summary,
        git_timeout=args.git_timeout,
        probe_remotes=args.probe_remotes,
        remote_timeout=args.remote_timeout,
    )
    summary["asset_report_path"] = args.asset_json
    write_json(args.output_json, summary)
    write_markdown(summary, args.output_md)
    print(
        json.dumps(
            json_ready(
                {
                    "status": summary["status"],
                    "json": args.output_json,
                    "md": args.output_md,
                    "routes_runnable": summary["routes_runnable"],
                    "routes_not_runnable": summary["routes_not_runnable"],
                }
            ),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
