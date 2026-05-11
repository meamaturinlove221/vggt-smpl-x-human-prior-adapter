from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "tools" / "check_cloud_gate_status.py"
CURRENT_SCHEMA = "20260504_visual_fullbody_hands_v2"


def write_registry(path: Path, *, schema: str, candidate_passes: int, teacher_passes: int) -> None:
    payload = {
        "schema_version": schema,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "strict_candidate_passes": int(candidate_passes),
            "strict_teacher_passes": int(teacher_passes),
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_checker(path: Path, *, teacher_supervised: bool = False) -> dict:
    cmd = [sys.executable, str(CHECKER), "--registry", str(path), "--json"]
    if teacher_supervised:
        cmd.append("--teacher-supervised")
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"checker did not print JSON\nstdout={completed.stdout}\nstderr={completed.stderr}") from exc
    payload["_exit_code"] = int(completed.returncode)
    return payload


def assert_blocked(payload: dict, expected_reason_substring: str) -> None:
    assert payload["_exit_code"] != 0, payload
    assert payload["cloud_allowed"] is False, payload
    reasons = "\n".join(payload.get("reasons") or [])
    assert expected_reason_substring in reasons, payload


def assert_allowed(payload: dict) -> None:
    assert payload["_exit_code"] == 0, payload
    assert payload["cloud_allowed"] is True, payload
    assert not payload.get("reasons"), payload


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="vggt_cloud_gate_test_") as tmp:
        tmp_root = Path(tmp)

        old_schema = tmp_root / "old_schema.json"
        write_registry(old_schema, schema="old", candidate_passes=1, teacher_passes=1)
        assert_blocked(run_checker(old_schema), "schema is not current")

        red_candidate = tmp_root / "red_candidate.json"
        write_registry(red_candidate, schema=CURRENT_SCHEMA, candidate_passes=0, teacher_passes=1)
        assert_blocked(run_checker(red_candidate), "strict_candidate_passes is 0")

        red_teacher = tmp_root / "red_teacher.json"
        write_registry(red_teacher, schema=CURRENT_SCHEMA, candidate_passes=1, teacher_passes=0)
        assert_blocked(run_checker(red_teacher, teacher_supervised=True), "strict_teacher_passes is 0")
        assert_allowed(run_checker(red_teacher, teacher_supervised=False))

    print("cloud gate status guard tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
