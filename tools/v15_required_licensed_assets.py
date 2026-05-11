from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
V14_STAGE = REPO_ROOT / "output" / "V14_H14_R14"
D_ROOT = Path("D:/")

DEFAULT_JSON = REPORTS / "20260508_v15_required_licensed_assets.json"
DEFAULT_MD = REPORTS / "20260508_v15_required_licensed_assets.md"

EXCLUDED_DIR_NAMES = {
    "$recycle.bin",
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "node_modules",
    "site-packages",
    "system volume information",
    "windowsapps",
    "wpsystem",
}

LICENSED_ASSET_SETS: dict[str, dict[str, Any]] = {
    "mano_right": {
        "family": "MANO",
        "manual_source": "MANO website/account",
        "files_any": ("MANO_RIGHT.pkl",),
        "why": "Required by WiLoR and HaMeR official hand inference routes.",
    },
    "mano_both": {
        "family": "MANO",
        "manual_source": "MANO website/account",
        "files_all": ("MANO_LEFT.pkl", "MANO_RIGHT.pkl"),
        "why": "Required by whole-body hand-capable routes such as OSX.",
    },
    "smpl_basic": {
        "family": "SMPL",
        "manual_source": "SMPL website/account",
        "files_any": (
            "SMPL_NEUTRAL.pkl",
            "SMPL_MALE.pkl",
            "SMPL_FEMALE.pkl",
            "basicmodel_neutral_lbs_10_207_0_v1.0.0.pkl",
        ),
        "why": "Required by whole-body routes that still load SMPL assets or regressors.",
    },
    "smplx": {
        "family": "SMPL-X",
        "manual_source": "SMPL-X website/account",
        "files_any": (
            "SMPLX_NEUTRAL.pkl",
            "SMPLX_NEUTRAL.npz",
            "SMPLX_MALE.pkl",
            "SMPLX_MALE.npz",
            "SMPLX_FEMALE.pkl",
            "SMPLX_FEMALE.npz",
        ),
        "why": "Required by OSX and SMPLer-X body/hand/face routes.",
    },
    "smplx_helpers": {
        "family": "SMPL-X",
        "manual_source": "SMPL-X/ExPose helper file links referenced by upstream projects",
        "files_all": (
            "SMPLX_to_J14.pkl",
            "MANO_SMPLX_vertex_ids.pkl",
            "SMPL-X__FLAME_vertex_ids.npy",
        ),
        "why": "Required by several whole-body pipelines for joint and vertex mapping.",
    },
    "flame_core": {
        "family": "FLAME",
        "manual_source": "FLAME website/account",
        "files_any": ("FLAME_NEUTRAL.pkl", "generic_model.pkl", "flame2023.pkl"),
        "why": "Required by face/hair/head routes that fit or initialize FLAME.",
    },
    "flame_hairgs": {
        "family": "FLAME",
        "manual_source": "FLAME website/account plus HairGS dataset preparation",
        "files_all": (
            "flame2023.pkl",
            "flame_static_embedding.pkl",
            "flame_dynamic_embedding.npy",
            "FLAME_masks.pkl",
        ),
        "why": "Required by HairGS before it can parse and train a real hair scene.",
    },
}

ROUTE_LICENSE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "WiLoR": ("mano_right",),
    "HaMeR": ("mano_right",),
    "OSX": ("smpl_basic", "smplx", "smplx_helpers", "mano_both", "flame_core"),
    "SMPLer-X": ("smpl_basic", "smplx", "smplx_helpers"),
    "HairGS": ("flame_hairgs",),
    "GaussianHaircut": ("flame_core",),
}

FOCUS_ENV_VARS = (
    "MANO_ROOT",
    "MANO_MODEL_DIR",
    "SMPL_ROOT",
    "SMPL_MODEL_DIR",
    "SMPLX_ROOT",
    "SMPLX_MODEL_DIR",
    "FLAME_ROOT",
    "FLAME_MODEL_DIR",
    "HUMAN_MODEL_FILES",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def safe_stat(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"path": path, "exists": path.exists()}
    try:
        stat = path.stat()
    except OSError as exc:
        row.update({"stat_error": repr(exc), "is_file": False, "is_dir": False, "size": 0})
        return row
    row.update(
        {
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
            "size": stat.st_size if path.is_file() else 0,
            "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
        }
    )
    return row


def all_required_filenames() -> tuple[str, ...]:
    names: set[str] = set()
    for spec in LICENSED_ASSET_SETS.values():
        names.update(spec.get("files_any", ()))
        names.update(spec.get("files_all", ()))
    return tuple(sorted(names))


def d_root_inventory() -> list[dict[str, Any]]:
    if not D_ROOT.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        entries = sorted(D_ROOT.iterdir(), key=lambda item: item.name.lower())
    except OSError:
        return rows
    for path in entries:
        rows.append(safe_stat(path))
    return rows


def selected_d_root_scan_dirs() -> list[Path]:
    if not D_ROOT.exists():
        return []
    focus_exact = {
        "wilor",
        "hamer",
        "osx",
        "smpler-x",
        "smplerx",
        "hairgs",
        "hair-gs",
        "hair-gs-master",
        "gaussianhaircut",
        "model",
        "models",
        "external_models",
        "data_used_in_4k4d",
    }
    focus_tokens = ("mano", "smpl", "smplx", "flame", "hair", "hamer", "wilor", "osx", "gaussian")
    roots: list[Path] = []
    try:
        entries = list(D_ROOT.iterdir())
    except OSError:
        return roots
    for path in entries:
        if not path.is_dir():
            continue
        lower = path.name.lower()
        if lower in EXCLUDED_DIR_NAMES:
            continue
        if lower in focus_exact or any(token in lower for token in focus_tokens):
            roots.append(path)
    return sorted(roots, key=lambda item: str(item).lower())


def default_scan_roots(include_env_roots: bool = True) -> list[Path]:
    roots: list[Path] = []
    roots.extend(selected_d_root_scan_dirs())
    roots.extend(
        [
            REPO_ROOT / "external",
            REPO_ROOT / "external_models",
            V14_STAGE,
            V14_STAGE / "public_repos",
            V14_STAGE / "downloads",
        ]
    )
    if include_env_roots:
        for key in FOCUS_ENV_VARS:
            value = os.environ.get(key)
            if value:
                roots.append(Path(value).expanduser())

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = str(root.resolve())
        except OSError:
            resolved = str(root)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(root)
    return unique


def should_skip_dir(path: Path) -> bool:
    lower = path.name.lower()
    if lower in EXCLUDED_DIR_NAMES:
        return True
    if lower.endswith(".egg-info"):
        return True
    return False


def scan_root(
    root: Path,
    target_names: tuple[str, ...],
    *,
    max_files: int,
    max_seconds: float,
    max_hits_per_name: int,
) -> dict[str, Any]:
    started = time.time()
    targets = {name.lower(): name for name in target_names}
    hits: dict[str, list[dict[str, Any]]] = {name: [] for name in target_names}
    total_hits: dict[str, int] = {name: 0 for name in target_names}
    row: dict[str, Any] = {
        "root": root,
        "exists": root.exists(),
        "scanned_files": 0,
        "scanned_dirs": 0,
        "truncated": False,
        "truncation_reason": None,
        "errors": [],
        "hits": hits,
        "hit_counts": total_hits,
    }
    if not root.exists():
        return row
    stack = [root]
    while stack:
        if time.time() - started > max_seconds:
            row["truncated"] = True
            row["truncation_reason"] = f"max_seconds_{max_seconds:g}"
            break
        current = stack.pop()
        if should_skip_dir(current):
            continue
        try:
            with os.scandir(current) as iterator:
                for entry in iterator:
                    path = Path(entry.path)
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if not should_skip_dir(path):
                                stack.append(path)
                                row["scanned_dirs"] += 1
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                    except OSError as exc:
                        row["errors"].append({"path": path, "error": repr(exc)})
                        continue
                    row["scanned_files"] += 1
                    if row["scanned_files"] >= max_files:
                        row["truncated"] = True
                        row["truncation_reason"] = f"max_files_{max_files}"
                        stack.clear()
                        break
                    canonical_name = targets.get(entry.name.lower())
                    if not canonical_name:
                        continue
                    total_hits[canonical_name] += 1
                    if len(hits[canonical_name]) < max_hits_per_name:
                        hits[canonical_name].append(safe_stat(path))
        except OSError as exc:
            row["errors"].append({"path": current, "error": repr(exc)})
    row["elapsed_sec"] = round(time.time() - started, 3)
    row["hit_counts"] = {name: count for name, count in total_hits.items() if count}
    row["hits"] = {name: values for name, values in hits.items() if values}
    return row


def merge_hits(scan_rows: list[dict[str, Any]], target_names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {
        name: {"count": 0, "hits": []}
        for name in target_names
    }
    seen_paths: set[str] = set()
    for row in scan_rows:
        counts = row.get("hit_counts") or {}
        for name, count in counts.items():
            merged.setdefault(name, {"count": 0, "hits": []})
            merged[name]["count"] += int(count)
        for name, hits in (row.get("hits") or {}).items():
            merged.setdefault(name, {"count": 0, "hits": []})
            for hit in hits:
                key = str(hit.get("path"))
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                merged[name]["hits"].append(hit)
    return {name: row for name, row in merged.items() if row["count"] or row["hits"]}


def evaluate_set(set_name: str, spec: dict[str, Any], merged_hits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    files_any = tuple(spec.get("files_any", ()))
    files_all = tuple(spec.get("files_all", ()))
    if files_any:
        present = [name for name in files_any if merged_hits.get(name, {}).get("count", 0) > 0]
        ok = bool(present)
        missing = [] if ok else list(files_any)
    else:
        present = [name for name in files_all if merged_hits.get(name, {}).get("count", 0) > 0]
        missing = [name for name in files_all if merged_hits.get(name, {}).get("count", 0) <= 0]
        ok = not missing
    names = files_any + files_all
    return {
        "ok": ok,
        "family": spec["family"],
        "manual_source": spec["manual_source"],
        "why": spec["why"],
        "files_any": files_any,
        "files_all": files_all,
        "present": present,
        "missing": missing,
        "hits": {name: merged_hits.get(name, {"count": 0, "hits": []}) for name in names},
    }


def route_license_status(asset_sets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    routes: dict[str, Any] = {}
    for route_name, required_sets in ROUTE_LICENSE_REQUIREMENTS.items():
        missing_sets = [set_name for set_name in required_sets if not asset_sets[set_name]["ok"]]
        routes[route_name] = {
            "required_sets": required_sets,
            "ok": not missing_sets,
            "missing_sets": missing_sets,
        }
    return routes


def build_blockers(asset_sets: dict[str, dict[str, Any]], routes: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for route_name, route in routes.items():
        for set_name in route["missing_sets"]:
            spec = asset_sets[set_name]
            missing = ", ".join(spec.get("missing") or [])
            blockers.append(
                f"{route_name}: missing licensed {spec['family']} set {set_name}: {missing}; source={spec['manual_source']}."
            )
    return blockers


def build_summary(
    *,
    max_files_per_root: int = 75000,
    max_seconds_per_root: float = 20.0,
    max_hits_per_name: int = 16,
    include_env_roots: bool = True,
) -> dict[str, Any]:
    target_names = all_required_filenames()
    scan_roots = default_scan_roots(include_env_roots=include_env_roots)
    scan_rows = [
        scan_root(
            root,
            target_names,
            max_files=max_files_per_root,
            max_seconds=max_seconds_per_root,
            max_hits_per_name=max_hits_per_name,
        )
        for root in scan_roots
    ]
    merged_hits = merge_hits(scan_rows, target_names)
    asset_sets = {
        set_name: evaluate_set(set_name, spec, merged_hits)
        for set_name, spec in LICENSED_ASSET_SETS.items()
    }
    routes = route_license_status(asset_sets)
    blockers = build_blockers(asset_sets, routes)
    found_families = sorted({row["family"] for row in asset_sets.values() if row["ok"]})
    missing_families = sorted({row["family"] for row in asset_sets.values() if not row["ok"]})
    return {
        "task": "v15_required_licensed_assets",
        "created_utc": utc_now(),
        "status": "v15_required_licensed_assets_blocked" if blockers else "v15_required_licensed_assets_ready",
        "scan_policy": {
            "d_root": str(D_ROOT),
            "d_root_top_level_inventory": True,
            "full_drive_recursive_scan": False,
            "bounded_selected_roots_only": True,
            "max_files_per_root": max_files_per_root,
            "max_seconds_per_root": max_seconds_per_root,
            "max_hits_per_name": max_hits_per_name,
            "include_env_roots": include_env_roots,
            "v14_stage": V14_STAGE,
        },
        "d_root_inventory": d_root_inventory(),
        "scan_roots": scan_roots,
        "scan_rows": scan_rows,
        "merged_hits": merged_hits,
        "asset_sets": asset_sets,
        "route_license_status": routes,
        "found_families": found_families,
        "missing_families": missing_families,
        "blockers": blockers,
        "decision": (
            "V15 did not find every licensed MANO/SMPL/SMPL-X/FLAME asset needed by the hand/hair routes "
            "inside the bounded D:/ and output/V14_H14_R14 search scope. Missing licensed assets remain manual-only."
            if blockers
            else "All required licensed asset sets were found in the bounded V15 search scope."
        ),
        "no_licensed_asset_fabrication": True,
        "no_downloads": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_strict_pass_claim": True,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# V15 Required Licensed Assets",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Read-only inventory. No licensed MANO/SMPL/SMPL-X/FLAME asset was downloaded, fabricated, copied, or promoted.",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Scan Scope",
        "",
        f"- D root inventory: `{D_ROOT}`",
        f"- V14 staging: `{V14_STAGE}`",
        f"- Full-drive recursive scan: `{summary['scan_policy']['full_drive_recursive_scan']}`",
        f"- Bounded roots: `{len(summary['scan_roots'])}`",
        "",
        "## Asset Sets",
        "",
        "| Set | Family | Ready | Present | Missing |",
        "|---|---|---:|---|---|",
    ]
    for set_name, row in summary["asset_sets"].items():
        present = ", ".join(row["present"]) if row["present"] else "-"
        missing = ", ".join(row["missing"]) if row["missing"] else "-"
        lines.append(f"| {set_name} | {row['family']} | {str(row['ok'])} | {present} | {missing} |")
    lines.extend(["", "## Route License Readiness", "", "| Route | Ready | Missing Sets |", "|---|---:|---|"])
    for route_name, row in summary["route_license_status"].items():
        missing = ", ".join(row["missing_sets"]) if row["missing_sets"] else "-"
        lines.append(f"| {route_name} | {str(row['ok'])} | {missing} |")
    lines.extend(["", "## Open Blockers", ""])
    if summary["blockers"]:
        lines.extend(f"- {item}" for item in summary["blockers"])
    else:
        lines.append("- none")
    lines.extend(["", "## Hits", ""])
    for set_name, row in summary["asset_sets"].items():
        lines.append(f"### {set_name}")
        any_hit = False
        for file_name, hit_row in row["hits"].items():
            hits = hit_row.get("hits") or []
            if not hits:
                continue
            any_hit = True
            lines.append(f"- {file_name}: {hit_row.get('count', len(hits))} hit(s)")
            for hit in hits[:5]:
                lines.append(f"  - `{hit['path']}` ({hit.get('size', 0)} bytes)")
        if not any_hit:
            lines.append("- no hits")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="V15 required licensed MANO/SMPL-X/FLAME asset scanner.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--max-files-per-root", type=int, default=75000)
    parser.add_argument("--max-seconds-per-root", type=float, default=20.0)
    parser.add_argument("--max-hits-per-name", type=int, default=16)
    parser.add_argument("--include-env-roots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    summary = build_summary(
        max_files_per_root=args.max_files_per_root,
        max_seconds_per_root=args.max_seconds_per_root,
        max_hits_per_name=args.max_hits_per_name,
        include_env_roots=args.include_env_roots,
    )
    write_json(args.output_json, summary)
    write_markdown(summary, args.output_md)
    print(json.dumps(json_ready({"status": summary["status"], "json": args.output_json, "md": args.output_md}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
