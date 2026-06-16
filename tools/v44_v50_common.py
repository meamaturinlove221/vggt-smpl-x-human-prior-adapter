from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
LOCAL_OUT = REPO_ROOT / "output" / "surface_research_preflight_local"
CLOUD_OUT = REPO_ROOT / "output" / "surface_research_cloud_preflight"

REQUIRED_REGIONS = ("body", "head", "face", "left_hand", "right_hand")

FORBIDDEN_NAMES = {
    "predictions.npz",
    "candidate_package",
    "teacher_package",
    "strict_gate_registry",
    "strict_registry_entry",
}
FORBIDDEN_SUBSTRINGS = (
    "formal_candidate",
    "strict_pass",
    "strict_registry",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jr(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
    if isinstance(value, dict):
        return {str(k): jr(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jr(v) for v in value]
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jr(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, payload: dict[str, Any], extra_lines: list[str] | None = None) -> None:
    lines = [f"# {title}", "", f"Status: `{payload.get('status')}`", ""]
    decision = payload.get("decision")
    if decision:
        lines.extend([str(decision), ""])
    blockers = payload.get("blockers") or payload.get("remaining_blockers") or []
    if blockers:
        lines.extend(["## Blockers", ""])
        lines.extend([f"- {item}" for item in blockers])
        lines.append("")
    if extra_lines:
        lines.extend(extra_lines)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def file_row(path: Path) -> dict[str, Any]:
    return {
        "path": path,
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else 0,
        "mtime": path.stat().st_mtime if path.is_file() else None,
    }


def region_metric_pass(metrics: dict[str, Any], min_pixels: int = 1) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    for region in REQUIRED_REGIONS:
        row = metrics.get(region) if isinstance(metrics.get(region), dict) else {}
        pixels = row.get("pixel_count", row.get("pixels", 0))
        try:
            pixels_i = int(pixels)
        except Exception:
            pixels_i = 0
        if pixels_i < min_pixels:
            blockers.append(f"{region}_support_empty")
    return not blockers, blockers


def scan_forbidden(roots: list[Path]) -> list[Path]:
    hits: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.as_posix().lower()
            name = path.name.lower()
            if name in FORBIDDEN_NAMES or any(token in rel for token in FORBIDDEN_SUBSTRINGS):
                # Historical D-line directories are not current V44-V50 writes,
                # but current roots passed by callers should be narrow anyway.
                hits.append(path)
    return hits


def v30_prior_prediction_ready() -> tuple[bool, list[str], dict[str, Any]]:
    data = read_json(REPORTS / "20260508_v30_prior_enabled_vggt_predictions.json")
    blockers: list[str] = []
    if data.get("status") != "DONE_PASS":
        blockers.append("V30 prior-enabled prediction payload is not DONE_PASS")
    verifier = data.get("verifier") if isinstance(data.get("verifier"), dict) else {}
    if not bool(verifier.get("usable_prior_enabled_checkpoint_exists")):
        blockers.append("No usable prior-enabled VGGT checkpoint exists for V30/V42-style predictions")
    intake = data.get("intake") if isinstance(data.get("intake"), dict) else {}
    if intake.get("status") != "DONE_PASS":
        blockers.append("V30 prediction intake did not find a complete prior-enabled payload")
    return not blockers, blockers, data


def base_stage_statuses() -> dict[str, str]:
    names = {
        "v29": "20260508_v29_normal_route_rescue.json",
        "v30": "20260508_v30_prior_enabled_vggt_predictions.json",
        "v31": "20260508_v31_teacher_supervised_candidate_train.json",
        "v32": "20260508_v32_candidate_inference_region_audit.json",
        "v33": "20260508_v33_head_face_detail_route.json",
        "v34": "20260508_v34_smplx_native_hand_route.json",
        "v35": "20260508_v35_60view_support_expansion.json",
        "v36": "20260508_v36_final_promotion_report.json",
    }
    statuses: dict[str, str] = {}
    for key, filename in names.items():
        statuses[key] = str(read_json(REPORTS / filename).get("status", "MISSING"))
    return statuses
