from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/surface_research_cloud_preflight/V9_A5X2_true_backend_precheck"
DEFAULT_STATUS_JSON = REPO_ROOT / "reports/20260507_v9_a5x2_true_backend_status.json"
DEFAULT_STATUS_MD = REPO_ROOT / "reports/20260507_v9_a5x2_true_backend_status.md"

BACKEND_FAMILIES = ("MUSt3R", "MASt3R", "2DGS", "NeuS2")
CHECKPOINT_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".onnx",
    ".msgpack",
    ".npz",
}


@dataclass(frozen=True)
class BackendSpec:
    name: str
    commands: tuple[str, ...]
    python_modules: tuple[str, ...]
    repo_tokens: tuple[str, ...]
    checkpoint_tokens: tuple[str, ...]
    smoke_command: tuple[str, ...] | None = None


SPECS: tuple[BackendSpec, ...] = (
    BackendSpec(
        name="MUSt3R",
        commands=("must3r", "must3r.exe"),
        python_modules=("must3r",),
        repo_tokens=("must3r",),
        checkpoint_tokens=("must3r",),
    ),
    BackendSpec(
        name="MASt3R",
        commands=("mast3r", "mast3r.exe", "mast3r_slam", "mast3r_slam.exe"),
        python_modules=("mast3r", "dust3r", "mast3r_slam"),
        repo_tokens=("mast3r", "dust3r"),
        checkpoint_tokens=("mast3r", "dust3r"),
    ),
    BackendSpec(
        name="2DGS",
        commands=("2dgs", "2dgs.exe"),
        python_modules=(
            "gaussian_renderer",
            "gsplat",
            "diff_gaussian_rasterization",
            "simple_knn",
        ),
        repo_tokens=("2dgs", "gaussian-splatting", "gaussian_splatting", "gsplat"),
        checkpoint_tokens=("2dgs", "gaussian", "splat", "gsplat"),
    ),
    BackendSpec(
        name="NeuS2",
        commands=("neus2", "neus2.exe", "instant-ngp", "instant-ngp.exe"),
        python_modules=("neus2", "pyngp", "tinycudann"),
        repo_tokens=("neus2", "instant-ngp", "instant_ngp"),
        checkpoint_tokens=("neus2", "neus", "instant-ngp", "instant_ngp", "ngp"),
    ),
)

KNOWN_SIMULATED_OUTPUT_PARTS = (
    "output/surface_research_cloud_preflight/cloud_b_a5x_external_dense_teacher_intake_smoke",
    "output/surface_research_cloud_preflight/cloud_b/a5x_external_dense_teacher_intake",
    "output/surface_research_preflight",
    "output/surface_research_preflight_local",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V9 A5-X2 fail-fast precheck for true MUSt3R/MASt3R/2DGS/NeuS2 backend installs. "
            "It writes evidence reports only. It does not synthesize smoke data and does not write teacher, "
            "candidate, export, prediction, registry, or pass artifacts."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-json", type=Path, default=DEFAULT_STATUS_JSON)
    parser.add_argument("--status-md", type=Path, default=DEFAULT_STATUS_MD)
    parser.add_argument("--max-scan-depth", type=int, default=6)
    parser.add_argument("--max-files-per-root", type=int, default=4000)
    parser.add_argument("--max-evidence-per-backend", type=int, default=16)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-true-smoke",
        action="store_true",
        help=(
            "If a backend executable/import and checkpoint are both found, run one bounded version/help probe. "
            "The probe is never run when availability is missing."
        ),
    )
    return parser.parse_args()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(val) for val in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def path_for_report(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def normalize_path(path: Path) -> str:
    return path_for_report(path).replace("\\", "/").lower()


def is_known_simulated_path(path: Path) -> bool:
    normalized = normalize_path(path)
    return any(part in normalized for part in KNOWN_SIMULATED_OUTPUT_PARTS)


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists and is not empty; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)


def safe_which(name: str) -> dict[str, Any]:
    source = shutil.which(name)
    return {
        "name": name,
        "found": bool(source),
        "source": source or "",
    }


def probe_module(module_name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(module_name)
        return {
            "module": module_name,
            "found": spec is not None,
            "origin": str(spec.origin) if spec and spec.origin else "",
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - evidence collection should not fail the precheck.
        return {
            "module": module_name,
            "found": False,
            "origin": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def default_scan_roots(repo_root: Path) -> list[Path]:
    userprofile = Path(os.environ.get("USERPROFILE") or Path.home())
    roots = [
        repo_root,
        repo_root / "external_models",
        repo_root.parent,
        Path("D:/vggt"),
        Path("F:/vggt"),
        Path("D:/vggt_xiaogandulost"),
        userprofile / ".cache" / "huggingface",
        userprofile / ".cache" / "torch",
        userprofile / ".cache",
        userprofile / ".nerfstudio",
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        key = str(resolved).lower()
        if key not in seen and resolved.exists():
            seen.add(key)
            unique.append(resolved)
    return unique


def iter_files(root: Path, *, max_depth: int, max_files: int, deadline: float) -> list[Path]:
    files: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack and len(files) < max_files and time.perf_counter() < deadline:
        current, depth = stack.pop()
        try:
            entries = list(current.iterdir())
        except (OSError, PermissionError):
            continue
        for entry in entries:
            if len(files) >= max_files or time.perf_counter() >= deadline:
                break
            try:
                if entry.is_dir():
                    if depth < max_depth and entry.name not in {".git", "__pycache__", ".venv", "node_modules"}:
                        stack.append((entry, depth + 1))
                elif entry.is_file():
                    files.append(entry)
            except (OSError, PermissionError):
                continue
    return files


def token_match(path: Path, tokens: tuple[str, ...]) -> bool:
    lowered = normalize_path(path)
    return any(token.lower() in lowered for token in tokens)


def collect_filesystem_evidence(
    specs: tuple[BackendSpec, ...],
    *,
    roots: list[Path],
    max_depth: int,
    max_files_per_root: int,
    max_evidence: int,
    deadline: float,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    evidence = {
        spec.name: {"repo_like_paths": [], "checkpoint_like_paths": [], "simulated_or_research_artifacts": []}
        for spec in specs
    }
    root_summaries: list[dict[str, Any]] = []
    for root in roots:
        if time.perf_counter() >= deadline:
            root_summaries.append({"root": str(root), "scanned": False, "reason": "deadline_reached"})
            break
        root_start = time.perf_counter()
        files = iter_files(root, max_depth=max_depth, max_files=max_files_per_root, deadline=deadline)
        root_summaries.append(
            {
                "root": str(root),
                "scanned": True,
                "files_seen": len(files),
                "elapsed_seconds": round(time.perf_counter() - root_start, 4),
            }
        )
        parent_dirs = {path.parent for path in files}
        candidate_paths = files + list(parent_dirs)
        for spec in specs:
            repo_hits = evidence[spec.name]["repo_like_paths"]
            ckpt_hits = evidence[spec.name]["checkpoint_like_paths"]
            simulated_hits = evidence[spec.name]["simulated_or_research_artifacts"]
            for candidate in candidate_paths:
                if token_match(candidate, spec.repo_tokens):
                    row = {
                        "path": str(candidate),
                        "kind": "directory" if candidate.is_dir() else "file",
                        "simulated_or_research_output": is_known_simulated_path(candidate),
                    }
                    target = simulated_hits if row["simulated_or_research_output"] else repo_hits
                    if len(target) < max_evidence and row not in target:
                        target.append(row)
                if candidate.is_file() and candidate.suffix.lower() in CHECKPOINT_SUFFIXES and token_match(candidate, spec.checkpoint_tokens):
                    row = {
                        "path": str(candidate),
                        "bytes": candidate.stat().st_size if candidate.exists() else 0,
                        "simulated_or_research_output": is_known_simulated_path(candidate),
                    }
                    target = simulated_hits if row["simulated_or_research_output"] else ckpt_hits
                    if len(target) < max_evidence and row not in target:
                        target.append(row)
    return evidence, root_summaries


def run_bounded_smoke(spec: BackendSpec, executable: str | None, timeout_seconds: float) -> dict[str, Any]:
    if not executable:
        return {"attempted": False, "reason": "no_executable"}
    commands: list[tuple[str, ...]] = []
    if spec.smoke_command:
        commands.append(spec.smoke_command)
    commands.extend(((executable, "--version"), (executable, "-h")))
    for command in commands:
        try:
            result = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except FileNotFoundError:
            return {"attempted": True, "command": list(command), "returncode": None, "error": "FileNotFoundError"}
        except subprocess.TimeoutExpired as exc:
            return {
                "attempted": True,
                "command": list(command),
                "returncode": None,
                "error": f"TimeoutExpired after {exc.timeout}s",
                "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
            }
        if result.returncode in {0, 1, 2} or result.stdout or result.stderr:
            return {
                "attempted": True,
                "command": list(command),
                "returncode": int(result.returncode),
                "stdout_tail": result.stdout[-2000:],
                "stderr_tail": result.stderr[-2000:],
            }
    return {"attempted": False, "reason": "no_probe_command_produced_output"}


def backend_rows(
    specs: tuple[BackendSpec, ...],
    filesystem_evidence: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    allow_true_smoke: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    rows: dict[str, dict[str, Any]] = {}
    chosen_smoke: dict[str, Any] | None = None
    for spec in specs:
        commands = [safe_which(command) for command in spec.commands]
        modules = [probe_module(module) for module in spec.python_modules]
        fs = filesystem_evidence[spec.name]
        command_found = next((row["source"] for row in commands if row["found"]), "")
        module_found = next((row["origin"] for row in modules if row["found"]), "")
        checkpoint_hits = [row for row in fs["checkpoint_like_paths"] if not row.get("simulated_or_research_output")]
        repo_hits = [row for row in fs["repo_like_paths"] if not row.get("simulated_or_research_output")]
        runnable_backend_found = bool(command_found or module_found or repo_hits)
        checkpoint_found = bool(checkpoint_hits)
        true_backend_available = bool(runnable_backend_found and checkpoint_found)
        smoke = {"attempted": False, "reason": "not_available_or_not_requested"}
        if allow_true_smoke and true_backend_available and chosen_smoke is None:
            smoke = run_bounded_smoke(spec, command_found, timeout_seconds=20.0)
            chosen_smoke = {"backend": spec.name, **smoke}
        rows[spec.name] = {
            "true_backend_available": true_backend_available,
            "runnable_backend_found": runnable_backend_found,
            "checkpoint_found": checkpoint_found,
            "commands": commands,
            "python_modules": modules,
            "repo_like_paths": repo_hits,
            "checkpoint_like_paths": checkpoint_hits,
            "simulated_or_research_artifacts": fs["simulated_or_research_artifacts"],
            "bounded_true_smoke": smoke,
            "reason": (
                "backend_and_checkpoint_found"
                if true_backend_available
                else "missing_real_checkpoint"
                if runnable_backend_found and not checkpoint_found
                else "missing_runnable_backend"
                if checkpoint_found and not runnable_backend_found
                else "missing_runnable_backend_and_checkpoint"
            ),
        }
    return rows, chosen_smoke


def make_summary(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    deadline = start + max(10.0, float(args.max_seconds))
    roots = default_scan_roots(REPO_ROOT)
    fs_evidence, root_summaries = collect_filesystem_evidence(
        SPECS,
        roots=roots,
        max_depth=max(0, int(args.max_scan_depth)),
        max_files_per_root=max(200, int(args.max_files_per_root)),
        max_evidence=max(1, int(args.max_evidence_per_backend)),
        deadline=deadline,
    )
    rows, chosen_smoke = backend_rows(SPECS, fs_evidence, allow_true_smoke=bool(args.allow_true_smoke))
    available = [name for name, row in rows.items() if row["true_backend_available"]]
    if not available:
        status = "blocked_no_true_backend_or_checkpoint_installed"
        verdict = (
            "FAIL_FAST_NO_TRUE_BACKEND: no local MUSt3R/MASt3R/2DGS/NeuS2 installation was found with both a "
            "runnable backend and a matching checkpoint. V8 synthetic/research artifacts are explicitly excluded."
        )
    elif chosen_smoke and chosen_smoke.get("attempted"):
        status = "true_backend_precheck_smoke_attempted_no_export"
        verdict = (
            "TRUE_BACKEND_PRESENT_RESEARCH_ONLY: at least one backend family has runnable evidence and a checkpoint; "
            "one bounded command probe was attempted. No teacher/candidate/export/pass was written."
        )
    else:
        status = "true_backend_available_smoke_not_requested_no_export"
        verdict = (
            "TRUE_BACKEND_PRESENT_NO_SMOKE: at least one backend family has runnable evidence and a checkpoint, but "
            "--allow-true-smoke was not provided. No teacher/candidate/export/pass was written."
        )

    return {
        "status": status,
        "created_utc": now_utc(),
        "repo": str(REPO_ROOT),
        "success": bool(available),
        "pass": False,
        "research_only": True,
        "fail_fast_precheck": not bool(available),
        "no_synthetic_smoke": True,
        "no_predictions_write": True,
        "no_registry_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_strict_pass_write": True,
        "teacher_export": "blocked",
        "candidate_export": "blocked",
        "predictions_export": "blocked",
        "formal_cloud_train_infer_export": "blocked",
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "backend_families_checked": list(BACKEND_FAMILIES),
        "true_backend_available": bool(available),
        "available_backend_families": available,
        "backend_availability": rows,
        "bounded_true_backend_decision_smoke": chosen_smoke
        or {"attempted": False, "reason": "no_available_backend" if not available else "not_requested"},
        "scan": {
            "roots": [str(root) for root in roots],
            "root_summaries": root_summaries,
            "max_depth": int(args.max_scan_depth),
            "max_files_per_root": int(args.max_files_per_root),
            "elapsed_seconds": round(time.perf_counter() - start, 4),
            "python": sys.executable,
            "platform": platform.platform(),
        },
        "outputs": {
            "output_dir": str(args.output_dir.resolve()),
            "summary_json": str((args.output_dir / "summary.json").resolve()),
            "report_md": str((args.output_dir / "report.md").resolve()),
            "status_json": str(args.status_json.resolve()),
            "status_md": str(args.status_md.resolve()),
        },
        "decision": verdict,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V9 A5-X2 True Backend Precheck",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is a fail-fast backend availability precheck. It does not synthesize dense candidates and does not write teacher, candidate, export, prediction, registry, or strict pass artifacts.",
        "",
        "## Verdict",
        "",
        summary["decision"],
        "",
        "## Backend Availability",
        "",
        "| Backend | Runnable backend | Checkpoint | True available | Reason |",
        "|---|---:|---:|---:|---|",
    ]
    for name, row in summary["backend_availability"].items():
        lines.append(
            "| {name} | {runnable} | {checkpoint} | {available} | `{reason}` |".format(
                name=name,
                runnable="yes" if row["runnable_backend_found"] else "no",
                checkpoint="yes" if row["checkpoint_found"] else "no",
                available="yes" if row["true_backend_available"] else "no",
                reason=row["reason"],
            )
        )
    lines.extend(
        [
            "",
            "## Command Evidence",
            "",
        ]
    )
    for name, row in summary["backend_availability"].items():
        found = [cmd for cmd in row["commands"] if cmd["found"]]
        missing = [cmd["name"] for cmd in row["commands"] if not cmd["found"]]
        lines.append(f"### {name}")
        lines.append("")
        if found:
            for cmd in found:
                lines.append(f"- command `{cmd['name']}`: `{cmd['source']}`")
        else:
            lines.append(f"- commands missing: `{', '.join(missing)}`")
        module_found = [mod for mod in row["python_modules"] if mod["found"]]
        if module_found:
            for mod in module_found:
                lines.append(f"- python module `{mod['module']}`: `{mod['origin']}`")
        else:
            lines.append("- python modules missing or unavailable")
        if row["checkpoint_like_paths"]:
            lines.append("- checkpoint-like evidence:")
            for item in row["checkpoint_like_paths"][:8]:
                lines.append(f"  - `{item['path']}`")
        else:
            lines.append("- checkpoint-like evidence: none")
        if row["simulated_or_research_artifacts"]:
            lines.append("- excluded simulated/research artifact hits:")
            for item in row["simulated_or_research_artifacts"][:8]:
                lines.append(f"  - `{item['path']}`")
        lines.append("")
    lines.extend(
        [
            "## Guard Truth",
            "",
            "```text",
            "strict_candidate_passes = 0",
            "strict_teacher_passes = 0",
            "teacher_export = blocked",
            "candidate_export = blocked",
            "predictions_export = blocked",
            "synthetic_smoke = not run",
            "```",
            "",
            "## Outputs",
            "",
        ]
    )
    for value in summary["outputs"].values():
        lines.append(f"- `{value}`")
    lines.extend(
        [
            "",
            "## Summary JSON",
            "",
            "```json",
            json.dumps(json_ready(summary), indent=2, ensure_ascii=False, sort_keys=True)[:30000],
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.status_json = args.status_json.resolve()
    args.status_md = args.status_md.resolve()
    ensure_output_dir(args.output_dir, args.overwrite)
    summary = make_summary(args)
    summary_json = args.output_dir / "summary.json"
    report_md = args.output_dir / "report.md"
    write_json(summary_json, summary)
    write_report(report_md, summary)
    write_json(args.status_json, summary)
    write_report(args.status_md, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "true_backend_available": summary["true_backend_available"],
                "available_backend_families": summary["available_backend_families"],
                "decision": summary["decision"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] != "blocked_no_true_backend_or_checkpoint_installed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
