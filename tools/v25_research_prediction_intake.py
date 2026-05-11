from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CLOUD_ROOT = REPO_ROOT / "output/surface_research_cloud_preflight/V25_research_vggt_predictions"
LOCAL_ROOT = REPO_ROOT / "output/surface_research_preflight_local/V25_research_vggt_predictions"
REPORT_JSON = REPO_ROOT / "reports/20260508_v25_research_predictions_3frames.json"
REPORT_MD = REPO_ROOT / "reports/20260508_v25_research_predictions_3frames.md"

FRAMES = ("frame0000", "frame0001", "frame0002")
REQUIRED_FILES = (
    "research_depths.npz",
    "research_points_world.npz",
    "research_normals.npz",
    "research_confidence.npz",
    "research_summary.json",
)
FORBIDDEN_NAMES = {"predictions.npz"}
FORBIDDEN_PATH_TOKENS = (
    "strict_pass",
    "teacher_export",
    "candidate_export",
    "formal_candidate",
    "strict_gate_registry",
    "candidate_gate",
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


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else 0,
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def array_stats(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array)
    finite = np.isfinite(arr)
    stats: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite_count": int(finite.sum()),
        "total_count": int(arr.size),
    }
    if finite.any():
        vals = arr[finite]
        stats.update(
            {
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "mean": float(np.mean(vals)),
                "p05": float(np.percentile(vals, 5)),
                "p50": float(np.percentile(vals, 50)),
                "p95": float(np.percentile(vals, 95)),
            }
        )
    return stats


def npz_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path.resolve()), "exists": False, "keys": [], "arrays": {}}
    arrays: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as payload:
        keys = list(payload.files)
        for key in keys:
            value = np.asarray(payload[key])
            if value.dtype.kind in {"U", "S", "O"}:
                arrays[key] = {"shape": list(value.shape), "dtype": str(value.dtype), "values": value.tolist()}
            else:
                arrays[key] = array_stats(value)
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": path.stat().st_size,
        "keys": keys,
        "arrays": arrays,
    }


def scene_row(frame: str) -> dict[str, Any]:
    frame_num = int(frame.replace("frame", ""))
    scene = REPO_ROOT / f"output/4k4d_scenes/0012_11_frame{frame_num:04d}_12views_tmf"
    images = sorted((scene / "images").glob("*.png")) if (scene / "images").is_dir() else []
    masks = sorted((scene / "masks").glob("*.png")) if (scene / "masks").is_dir() else []
    return {
        "frame": frame,
        "scene": str(scene.resolve()),
        "exists": scene.is_dir(),
        "image_count": len(images),
        "mask_count": len(masks),
        "manifest": file_info(scene / "scene_manifest.json"),
        "prior_maps": file_info(scene / "prior_maps.npz"),
        "ready_12view_tmf": bool(scene.is_dir() and len(images) >= 12 and len(masks) >= 12 and (scene / "scene_manifest.json").is_file()),
    }


def scan_forbidden(root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not root.exists():
        return findings
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        lower = rel.lower()
        if path.is_file() and path.name.lower() in FORBIDDEN_NAMES:
            findings.append({"path": str(path.resolve()), "reason": "predictions.npz is forbidden for V25"})
        hit_tokens = [token for token in FORBIDDEN_PATH_TOKENS if token in lower]
        if hit_tokens:
            findings.append({"path": str(path.resolve()), "reason": f"formal-output token(s) in path: {', '.join(hit_tokens)}"})
    return findings


def validate_payload(root: Path) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    files = {name: file_info(root / name) for name in REQUIRED_FILES}
    for name, info in files.items():
        if not info["exists"]:
            blockers.append(f"missing required V25 file: {name}")

    arrays = {
        "research_depths.npz": npz_summary(root / "research_depths.npz"),
        "research_points_world.npz": npz_summary(root / "research_points_world.npz"),
        "research_normals.npz": npz_summary(root / "research_normals.npz"),
        "research_confidence.npz": npz_summary(root / "research_confidence.npz"),
    }

    for file_name, payload in arrays.items():
        if not payload.get("exists"):
            continue
        keys = set(payload.get("keys") or [])
        frame_keys = set()
        if "frame_keys" in keys:
            frame_values = payload["arrays"]["frame_keys"].get("values", [])
            frame_keys = {str(item) for item in frame_values}
        expected = set(FRAMES)
        if file_name != "research_normals.npz" and not expected.issubset(keys):
            blockers.append(f"{file_name} does not contain all frame arrays: expected {sorted(expected)}, got {sorted(keys)}")
        if "frame_keys" in keys and frame_keys != expected:
            blockers.append(f"{file_name} frame_keys mismatch: expected {sorted(expected)}, got {sorted(frame_keys)}")

    if arrays["research_normals.npz"].get("exists"):
        normal_keys = set(arrays["research_normals.npz"].get("keys") or [])
        has_frame_normals = set(FRAMES).issubset(normal_keys)
        has_explicit_absence = "normal_available" in normal_keys and "normal_reason" in normal_keys
        if not has_frame_normals and not has_explicit_absence:
            blockers.append("research_normals.npz must contain either frame normal arrays or explicit normal_available/normal_reason evidence")

    forbidden = scan_forbidden(root)
    if forbidden:
        blockers.append("forbidden V25 output found under research root")

    return blockers, {"files": files, "arrays": arrays, "forbidden_findings": forbidden}


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V25 Research VGGT Predictions 3 Frames",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only intake. No formal package, registry, strict pass, teacher export, candidate export, or `predictions.npz` is accepted.",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Research Files",
        "",
    ]
    for name, info in summary["payload"].get("files", {}).items():
        lines.append(f"- `{name}`: exists={str(info['exists']).lower()} size={info['size']}")
    lines.extend(["", "## Scenes", ""])
    for row in summary.get("scenes", []):
        lines.append(f"- `{row['frame']}`: images={row['image_count']} masks={row['mask_count']} ready={str(row['ready_12view_tmf']).lower()}")
    blockers = summary.get("blockers") or []
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- {item}" for item in blockers] if blockers else ["- none"])
    if summary.get("hard_impossible_evidence"):
        lines.extend(["", "## Hard-Impossible Evidence", "", "```json"])
        lines.append(json.dumps(summary["hard_impossible_evidence"], indent=2, ensure_ascii=False, sort_keys=True))
        lines.append("```")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Intake V25 research-only VGGT prediction arrays.")
    parser.add_argument("--cloud-root", type=Path, default=CLOUD_ROOT)
    parser.add_argument("--local-root", type=Path, default=LOCAL_ROOT)
    parser.add_argument("--output-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--output-md", type=Path, default=REPORT_MD)
    args = parser.parse_args()

    cloud_root = args.cloud_root.expanduser().resolve()
    local_root = args.local_root.expanduser().resolve()
    scenes = [scene_row(frame) for frame in FRAMES]
    summary_json = load_json(cloud_root / "research_summary.json")
    blockers, payload = validate_payload(cloud_root)

    for key, expected in {
        "research_only": True,
        "no_predictions_write": True,
        "no_registry_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_strict_pass_write": True,
    }.items():
        if summary_json and summary_json.get(key) is not expected:
            blockers.append(f"research_summary.json flag mismatch: {key} is not {expected}")

    hard_impossible = bool(summary_json.get("hard_impossible")) if summary_json else False
    status = "intake_pass_research_only" if not blockers and not hard_impossible else (
        "hard_impossible_evidence_recorded" if hard_impossible else "intake_blocked"
    )
    decision = (
        "V25 research arrays for frame0000/frame0001/frame0002 are present and scoped to research-only outputs."
        if status == "intake_pass_research_only"
        else (
            "V25 produced explicit hard-impossible evidence instead of research arrays; no formal artifact was written."
            if status == "hard_impossible_evidence_recorded"
            else "V25 research array intake found blockers; no formal artifact was written."
        )
    )
    report = {
        "task": "v25_research_prediction_intake",
        "created_utc": utc_now(),
        "status": status,
        "research_only": True,
        "no_export": True,
        "no_predictions_write": True,
        "no_registry_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_strict_pass_write": True,
        "formal_cloud_unblocked": False,
        "cloud_root": cloud_root,
        "local_root": local_root,
        "scenes": scenes,
        "payload": payload,
        "remote_summary": summary_json,
        "hard_impossible_evidence": summary_json.get("hard_impossible_evidence") if summary_json else None,
        "blockers": blockers,
        "decision": decision,
    }
    local_root.mkdir(parents=True, exist_ok=True)
    write_json(local_root / "intake_summary.json", report)
    write_markdown(local_root / "intake_summary.md", report)
    write_json(args.output_json, report)
    write_markdown(args.output_md, report)
    print(json.dumps(json_ready({"status": status, "blockers": blockers}), ensure_ascii=False))
    return 0 if status in {"intake_pass_research_only", "hard_impossible_evidence_recorded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
