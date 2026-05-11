from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = REPO_ROOT / "output/surface_research_preflight_local/V30_prior_enabled_predictions"
CLOUD_ROOT = REPO_ROOT / "output/surface_research_cloud_preflight/V30_prior_enabled_predictions"
REPORT_JSON = REPO_ROOT / "reports/20260508_v30_prior_enabled_vggt_predictions.json"
REPORT_MD = REPO_ROOT / "reports/20260508_v30_prior_enabled_vggt_predictions.md"

FRAMES = ("frame0000", "frame0001", "frame0002")
CONTROLS = ("real", "zero", "shuffle", "random-region", "prior-dropout")
REQUIRED_RESEARCH_FILES = (
    "research_depths.npz",
    "research_points_world.npz",
    "research_confidence.npz",
    "research_normals_geometric.npz",
    "research_prior_effect.json",
)
FORBIDDEN_FILENAMES = {"predictions.npz"}
FORBIDDEN_TOKENS = (
    "strict_pass",
    "strict_gate_registry",
    "formal_candidate",
    "candidate_package",
    "teacher_package",
    "teacher_export",
    "candidate_export",
    "registry_refresh",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else 0,
        "mtime": path.stat().st_mtime if path.is_file() else None,
    }


def dir_info(path: Path) -> dict[str, Any]:
    files = []
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file()]
    return {
        "path": str(path.resolve()),
        "exists": path.is_dir(),
        "file_count": len(files),
        "total_size": int(sum(p.stat().st_size for p in files)),
    }


def npz_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path.resolve()), "exists": False, "keys": [], "arrays": {}}
    arrays: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as payload:
        keys = list(payload.files)
        for key in keys:
            value = np.asarray(payload[key])
            arrays[key] = array_summary(value)
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": path.stat().st_size,
        "keys": keys,
        "arrays": arrays,
    }


def array_summary(value: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(value)
    summary = {"shape": list(arr.shape), "dtype": str(arr.dtype), "total_count": int(arr.size)}
    if arr.dtype.kind in {"U", "S", "O"}:
        if arr.size <= 32:
            summary["values"] = arr.tolist()
        return summary
    finite = np.isfinite(arr)
    summary["finite_count"] = int(finite.sum())
    if finite.any():
        vals = arr[finite]
        summary.update(
            {
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "mean": float(np.mean(vals)),
                "p50": float(np.percentile(vals, 50)),
            }
        )
    if arr.dtype.kind == "b":
        summary["true_count"] = int(np.count_nonzero(arr))
    return summary


def scan_forbidden(root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not root.exists():
        return findings
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix().lower()
        if path.is_file() and path.name.lower() in FORBIDDEN_FILENAMES:
            findings.append({"path": str(path.resolve()), "reason": "predictions.npz is forbidden for V30"})
        hits = [token for token in FORBIDDEN_TOKENS if token in rel]
        if hits:
            findings.append({"path": str(path.resolve()), "reason": f"formal-output token(s) in path: {', '.join(hits)}"})
    return findings


def scene_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in FRAMES:
        frame_number = int(frame.replace("frame", ""))
        root = REPO_ROOT / f"output/4k4d_scenes/0012_11_frame{frame_number:04d}_12views_tmf"
        images = sorted((root / "images").glob("*.png")) if (root / "images").is_dir() else []
        masks = sorted((root / "masks").glob("*.png")) if (root / "masks").is_dir() else []
        rows.append(
            {
                "frame": frame,
                "scene_root": str(root.resolve()),
                "exists": root.is_dir(),
                "image_count": len(images),
                "mask_count": len(masks),
                "manifest": file_info(root / "scene_manifest.json"),
                "prior_maps": file_info(root / "prior_maps.npz"),
                "ready": bool(root.is_dir() and len(images) >= 12 and len(masks) >= 12 and (root / "scene_manifest.json").is_file()),
            }
        )
    return rows


def prior_case_rows() -> list[dict[str, Any]]:
    candidates = [
        REPO_ROOT / "output/training_cases/0012_11_frame0000_6views_smplx_native_prior_v15",
        REPO_ROOT / "output/training_cases/0012_11_frame0000_12views_sparseproto_smplxsurfacepose_v2",
    ]
    rows: list[dict[str, Any]] = []
    for root in candidates:
        input_path = root / "inputs.npz"
        row = {
            "case_root": str(root.resolve()),
            "exists": root.is_dir(),
            "inputs": file_info(input_path),
            "targets": file_info(root / "targets.npz"),
            "case_manifest": file_info(root / "case_manifest.json"),
            "prior_channels": None,
            "prior_summary_channels": None,
        }
        if input_path.is_file():
            with np.load(input_path, allow_pickle=False) as payload:
                if "prior_maps" in payload:
                    prior_maps = np.asarray(payload["prior_maps"])
                    row["prior_channels"] = int(prior_maps.shape[2] if prior_maps.ndim == 5 else prior_maps.shape[1])
                    row["prior_map_shape"] = list(prior_maps.shape)
                if "prior_summary_tokens" in payload:
                    summary = np.asarray(payload["prior_summary_tokens"])
                    row["prior_summary_channels"] = int(summary.shape[-1])
                    row["prior_summary_shape"] = list(summary.shape)
        rows.append(row)
    return rows


def write_report_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V30 Prior-Enabled VGGT Predictions",
        "",
        f"Status: `{summary.get('status')}`",
        "",
        "Research-only. V30 did not write formal `predictions.npz`, candidate package, teacher package, strict registry, or strict pass state.",
        "",
        "## Decision",
        "",
        str(summary.get("decision", "")),
        "",
        "## Verification",
        "",
    ]
    verifier = summary.get("verifier", {})
    lines.append(f"- modified VGGT code supports prior adapter: `{verifier.get('code_supports_prior_adapter')}`")
    lines.append(f"- usable prior-enabled checkpoint exists: `{verifier.get('usable_prior_enabled_checkpoint_exists')}`")
    lines.append(f"- base HF model allowed for key predictions: `{verifier.get('base_hf_model_allowed_for_key_predictions')}`")
    lines.extend(["", "## Blockers", ""])
    blockers = summary.get("blockers") or []
    lines.extend([f"- {item}" for item in blockers] if blockers else ["- none"])
    lines.extend(["", "## Outputs", ""])
    outputs = summary.get("outputs", {})
    for key, value in outputs.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Controls", ""])
    control = summary.get("control_audit", {})
    for name in CONTROLS:
        item = control.get("controls", {}).get(name, {})
        lines.append(f"- `{name}`: status=`{item.get('status')}` reason=`{item.get('reason')}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
