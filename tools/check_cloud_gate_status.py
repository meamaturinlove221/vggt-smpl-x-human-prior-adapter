from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "reports" / "20260504_strict_gate_registry.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether the local strict mentor gate currently permits any cloud "
            "upload/run. This is read-only and never contacts Modal."
        )
    )
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--max-age-hours", type=float, default=24.0)
    parser.add_argument(
        "--teacher-supervised",
        action="store_true",
        help="Also require strict_teacher_passes > 0 for a teacher-supervised route.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable status.")
    return parser.parse_args()


def load_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Strict gate registry not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    registry_path = Path(args.registry).expanduser().resolve()
    status: dict[str, Any] = {
        "registry": str(registry_path),
        "cloud_allowed": False,
        "teacher_supervised": bool(args.teacher_supervised),
        "reasons": [],
    }

    try:
        registry = load_registry(registry_path)
    except Exception as exc:  # noqa: BLE001 - this is a guard/report script.
        status["reasons"].append(str(exc))
        if args.json:
            print(json.dumps(status, indent=2, ensure_ascii=False))
        else:
            print(f"CLOUD BLOCKED: {status['reasons'][0]}")
        return 2

    age_hours = (datetime.now(timezone.utc) - datetime.fromtimestamp(registry_path.stat().st_mtime, tz=timezone.utc)).total_seconds() / 3600.0
    counts = registry.get("counts", {})
    candidate_passes = int(counts.get("strict_candidate_passes", 0) or 0)
    teacher_passes = int(counts.get("strict_teacher_passes", 0) or 0)

    status.update(
        {
            "generated_at": registry.get("generated_at"),
            "registry_age_hours": round(age_hours, 3),
            "strict_candidate_passes": candidate_passes,
            "strict_teacher_passes": teacher_passes,
            "counts": counts,
        }
    )

    if age_hours > float(args.max_age_hours):
        status["reasons"].append(
            f"strict gate registry is stale ({age_hours:.1f}h > {float(args.max_age_hours):.1f}h)"
        )
    if candidate_passes <= 0:
        status["reasons"].append("strict_candidate_passes is 0")
    if args.teacher_supervised and teacher_passes <= 0:
        status["reasons"].append("teacher-supervised route requested but strict_teacher_passes is 0")

    status["cloud_allowed"] = not status["reasons"]
    if args.json:
        print(json.dumps(status, indent=2, ensure_ascii=False))
    elif status["cloud_allowed"]:
        print("CLOUD ALLOWED: local strict candidate gate is green.")
    else:
        print("CLOUD BLOCKED:")
        for reason in status["reasons"]:
            print(f"- {reason}")
        print(f"- strict_candidate_passes={candidate_passes}, strict_teacher_passes={teacher_passes}")
    return 0 if status["cloud_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

