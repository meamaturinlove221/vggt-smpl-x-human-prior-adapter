from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any


RESEARCH_ROOT_TOKEN = "surface_research_cloud_preflight"
FORBIDDEN_PATH_WORDS = (
    "strict_pass",
    "teacher_export",
    "candidate_export",
    "predictions",
    "prediction_export",
    "formal_candidate",
)
REQUIRED_RESEARCH_FLAGS = {
    "research_only": True,
    "no_export": True,
    "no_predictions_write": True,
    "no_registry_write": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize_output_dir(value: str | Path) -> Path:
    path = Path(value)
    text = str(path).replace("\\", "/").lower()
    if RESEARCH_ROOT_TOKEN not in text:
        raise ValueError(f"research output_dir must include {RESEARCH_ROOT_TOKEN!r}: {value}")
    if any(word in text for word in FORBIDDEN_PATH_WORDS):
        raise ValueError(f"research output_dir contains a forbidden pass/export token: {value}")
    return path


def validate_research_metadata(metadata: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key, expected in REQUIRED_RESEARCH_FLAGS.items():
        if metadata.get(key) is not expected:
            reasons.append(f"{key} is not {expected}")
    output_dir = str(metadata.get("output_dir", ""))
    if not output_dir:
        reasons.append("output_dir is missing")
    else:
        try:
            normalize_output_dir(output_dir)
        except Exception as exc:  # noqa: BLE001
            reasons.append(str(exc))
    for key in ("max_steps", "max_cases", "max_hours"):
        value = metadata.get(key)
        if value is None:
            reasons.append(f"{key} is missing")
            continue
        try:
            number = float(value)
        except Exception:  # noqa: BLE001
            reasons.append(f"{key} is not numeric")
            continue
        if not math.isfinite(number) or number <= 0:
            reasons.append(f"{key} must be positive")
    return reasons


def default_research_metadata(*, job_id: str, job_name: str, output_dir: Path, max_steps: int, max_cases: int, max_hours: float) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "job_name": job_name,
        "research_only": True,
        "no_export": True,
        "no_predictions_write": True,
        "no_registry_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_strict_pass_write": True,
        "formal_cloud_train_infer_export": "blocked",
        "output_dir": str(output_dir),
        "max_steps": int(max_steps),
        "max_cases": int(max_cases),
        "max_hours": float(max_hours),
        "created_utc": now_utc(),
    }
