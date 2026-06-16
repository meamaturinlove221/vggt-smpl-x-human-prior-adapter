from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from tools.smplx_numpy import MODEL_FILENAMES, resolve_smplx_model_path  # noqa: E402
from v15_common import REPORTS, json_ready, utc_now, write_json  # noqa: E402


DEFAULT_SMPLX_ROOT = Path("G:/\u6570\u636e\u96c6/datasets/smplx")
DEFAULT_JSON = REPORTS / "20260508_v15_smplx_asset_scope_reset.json"
DEFAULT_MD = REPORTS / "20260508_v15_smplx_asset_scope_reset.md"

OPTIONAL_NONFATAL_ASSETS: dict[str, tuple[str, ...]] = {
    "MANO": ("MANO_LEFT.pkl", "MANO_RIGHT.pkl"),
    "FLAME": ("FLAME_NEUTRAL.pkl", "generic_model.pkl", "flame2023.pkl"),
    "SMPL": ("SMPL_NEUTRAL.pkl", "SMPL_MALE.pkl", "SMPL_FEMALE.pkl"),
    "HairGS": ("flame_static_embedding.pkl", "flame_dynamic_embedding.npy", "FLAME_masks.pkl"),
}


def _json_ready_local(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready_local(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready_local(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready_local(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def safe_stat(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"path": path, "exists": path.exists()}
    try:
        stat = path.stat()
    except OSError as exc:
        row.update({"error": repr(exc), "is_file": False, "size": 0})
        return row
    row.update(
        {
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
            "size": int(stat.st_size) if path.is_file() else 0,
            "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
        }
    )
    return row


def inspect_npz(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = safe_stat(path)
    if not path.is_file():
        return row
    try:
        with np.load(path, allow_pickle=True) as payload:
            row["keys"] = sorted(str(name) for name in payload.files)
            row["shape_summary"] = {}
            for key in ("v_template", "f", "weights", "J_regressor", "shapedirs", "posedirs", "kintree_table"):
                if key not in payload:
                    continue
                arr = payload[key]
                row["shape_summary"][key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            for key in ("part2num", "joint2num"):
                if key in payload:
                    try:
                        value = payload[key].item()
                        row[key] = {str(k): int(v) for k, v in sorted(value.items(), key=lambda item: int(item[1]))}
                    except Exception as exc:  # noqa: BLE001 - report exact malformed helper payload
                        row[f"{key}_error"] = repr(exc)
    except Exception as exc:  # noqa: BLE001 - asset report should be nonfatal
        row["load_error"] = repr(exc)
    return row


def search_optional_assets(search_roots: list[Path]) -> dict[str, Any]:
    names = sorted({name for values in OPTIONAL_NONFATAL_ASSETS.values() for name in values})
    hits: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    for root in search_roots:
        if not root.exists():
            continue
        stack = [root]
        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as iterator:
                    for entry in iterator:
                        path = Path(entry.path)
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                if entry.name.lower() not in {".git", "__pycache__", "node_modules", ".venv"}:
                                    stack.append(path)
                                continue
                            if entry.is_file(follow_symlinks=False) and entry.name in hits and len(hits[entry.name]) < 8:
                                hits[entry.name].append(safe_stat(path))
                        except OSError:
                            continue
            except OSError:
                continue
    grouped: dict[str, Any] = {}
    for family, family_names in OPTIONAL_NONFATAL_ASSETS.items():
        present = [name for name in family_names if hits.get(name)]
        grouped[family] = {
            "present": present,
            "missing": [name for name in family_names if not hits.get(name)],
            "fatal_for_smplx_native": False,
            "hits": {name: hits.get(name, []) for name in family_names},
        }
    return grouped


def build_summary(smplx_root: Path, preferred_gender: str) -> dict[str, Any]:
    smplx_root = smplx_root.expanduser().resolve()
    direct_models: dict[str, Any] = {}
    compact_models: dict[str, Any] = {}
    resolved: dict[str, Any] = {}
    for gender, filename in MODEL_FILENAMES.items():
        direct = smplx_root / filename
        compact = smplx_root / "smplx_npz" / filename
        direct_models[gender] = inspect_npz(direct)
        compact_models[gender] = inspect_npz(compact)
        try:
            resolved_path = resolve_smplx_model_path(smplx_root, gender)
            resolved[gender] = {
                "ok": True,
                "path": resolved_path,
                "source": "direct_root" if resolved_path == direct.resolve() else "smplx_npz_fallback",
            }
        except Exception as exc:  # noqa: BLE001
            resolved[gender] = {"ok": False, "error": repr(exc)}

    preferred = resolved.get(preferred_gender, {})
    optional_search_roots = [smplx_root, REPO_ROOT / "external", REPO_ROOT / "external_models"]
    optional = search_optional_assets(optional_search_roots)
    required_ready = bool(preferred.get("ok"))
    direct_ready = sum(1 for row in direct_models.values() if row.get("is_file") and not row.get("load_error"))
    compact_ready = sum(1 for row in compact_models.values() if row.get("is_file") and not row.get("load_error"))
    blockers = []
    if not required_ready:
        blockers.append(f"Preferred SMPL-X {preferred_gender} model could not be resolved under {smplx_root}.")

    return {
        "task": "v15_smplx_asset_scope_reset",
        "created_utc": utc_now(),
        "status": "v15_smplx_native_assets_ready" if required_ready else "v15_smplx_native_assets_blocked",
        "research_only": True,
        "smplx_native_only": True,
        "asset_root": smplx_root,
        "preferred_gender": preferred_gender,
        "resolved_models": resolved,
        "direct_models": direct_models,
        "compact_fallback_models": compact_models,
        "optional_nonfatal_assets": optional,
        "metrics": {
            "direct_smplx_npz_count": direct_ready,
            "compact_smplx_npz_count": compact_ready,
            "preferred_model_resolved": required_ready,
            "required_smplx_native_gender_count": len([row for row in resolved.values() if row.get("ok")]),
        },
        "scope_reset": {
            "mano_missing_is_fatal_for_smplx_native": False,
            "flame_missing_is_fatal_for_smplx_native": False,
            "smpl_missing_is_fatal_for_smplx_native": False,
            "hairgs_missing_is_fatal_for_smplx_native": False,
            "smplx_helpers_missing_is_fatal_for_smplx_native": False,
            "no_downloads": True,
            "no_asset_fabrication": True,
            "no_predictions_write": True,
            "no_teacher_export": True,
            "no_candidate_export": True,
            "no_registry_write": True,
            "no_strict_pass_claim": True,
        },
        "decision": (
            "Reset V15 native scope to the local SMPL-X NPZ assets only. Missing MANO/FLAME/SMPL/HairGS/helper "
            "assets remain blockers for their own routes, but they are not fatal for SMPL-X native forward, part, or raster probes."
            if required_ready
            else "SMPL-X native scope is blocked because the preferred SMPL-X model was not resolved."
        ),
        "blockers": blockers,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# V15 SMPL-X Asset Scope Reset",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only asset scope reset. No package, cloud job, registry, teacher, candidate, or strict pass is produced.",
        "",
        "## Decision",
        "",
        str(summary["decision"]),
        "",
        "## SMPL-X Native Models",
        "",
        "| Gender | Resolved | Source | Path | v_template | shapedirs | faces |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for gender, row in summary["resolved_models"].items():
        model_row = summary["direct_models"].get(gender, {})
        if row.get("source") == "smplx_npz_fallback":
            model_row = summary["compact_fallback_models"].get(gender, {})
        shapes = model_row.get("shape_summary") or {}
        v_shape = shapes.get("v_template", {}).get("shape")
        s_shape = shapes.get("shapedirs", {}).get("shape")
        f_shape = shapes.get("f", {}).get("shape")
        lines.append(
            f"| {gender} | {str(bool(row.get('ok')))} | {row.get('source', '-')} | `{row.get('path', row.get('error', '-'))}` | "
            f"{v_shape or '-'} | {s_shape or '-'} | {f_shape or '-'} |"
        )
    lines.extend(["", "## Nonfatal Missing Assets", ""])
    for family, row in summary["optional_nonfatal_assets"].items():
        missing = ", ".join(row["missing"]) if row["missing"] else "-"
        present = ", ".join(row["present"]) if row["present"] else "-"
        lines.append(f"- {family}: present `{present}`, missing `{missing}`, fatal_for_smplx_native `False`")
    lines.extend(["", "## Blockers", ""])
    if summary["blockers"]:
        lines.extend(f"- {item}" for item in summary["blockers"])
    else:
        lines.append("- none for SMPL-X native worker A")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="V15 SMPL-X native asset resolver and scope reset report.")
    parser.add_argument("--smplx-root", type=Path, default=DEFAULT_SMPLX_ROOT)
    parser.add_argument("--gender", choices=tuple(MODEL_FILENAMES), default="neutral")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    summary = build_summary(args.smplx_root, args.gender)
    write_json(args.output_json, summary)
    write_markdown(summary, args.output_md)
    print(json.dumps(json_ready({"status": summary["status"], "metrics": summary["metrics"], "json": args.output_json}), ensure_ascii=False))
    return 0 if not summary["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
