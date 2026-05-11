from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FROZEN = ROOT / "output" / "frozen_candidates" / "V50R2_rebuilt_from_sessions_gdrive_modal"
if not FROZEN.exists():
    FROZEN = ROOT / "output" / "frozen_candidates" / "V50_smplx_native_candidate_pass"
PACKAGE_FILES = FROZEN / "package_files"
HASH_MANIFEST = FROZEN / "hash_manifest.json"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def iter_manifest_files(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # V50R2 hash manifests are stored as {relative_path: {sha256, size}}.
    if manifest and all(isinstance(value, dict) and "sha256" in value for value in manifest.values()):
        for rel, value in manifest.items():
            rows.append({"key": rel, "path": FROZEN / rel, "expected_sha256": value.get("sha256")})
        return rows
    for key, value in manifest.get("copied_files", {}).items():
        frozen = value.get("frozen") or value.get("path")
        expected = value.get("sha256")
        if frozen:
            rows.append({"key": key, "path": Path(frozen), "expected_sha256": expected})
    for key, value in (("frozen_manifest", manifest.get("frozen_manifest")), ("frozen_registry", manifest.get("frozen_registry"))):
        if isinstance(value, dict) and value.get("path"):
            rows.append({"key": key, "path": Path(value["path"]), "expected_sha256": value.get("sha256")})
    return rows


def monitor() -> dict[str, Any]:
    manifest = read_json(HASH_MANIFEST, {})
    checks = []
    mismatches = []
    missing = []
    for row in iter_manifest_files(manifest):
        path = row["path"]
        exists = path.exists()
        actual = sha_file(path) if exists and path.is_file() else None
        ok = bool(exists and actual == row.get("expected_sha256"))
        check = {
            "key": row["key"],
            "path": str(path),
            "exists": exists,
            "expected_sha256": row.get("expected_sha256"),
            "actual_sha256": actual,
            "hash_match": ok,
        }
        checks.append(check)
        if not exists:
            missing.append(check)
        elif not ok:
            mismatches.append(check)

    forbidden_patterns = [
        "teacher_package",
        "strict_teacher_registry",
        "strict_teacher_pass",
        "candidate_package_v67",
        "V50_overwrite",
    ]
    forbidden_hits = []
    for root in [FROZEN, REPORTS]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and any(pattern.lower() in path.name.lower() for pattern in forbidden_patterns):
                forbidden_hits.append(str(path))

    payload = {
        "task": "V223_candidate_immutability_monitor",
        "created_utc": now(),
        "frozen_candidate": str(FROZEN),
        "hash_manifest": str(HASH_MANIFEST),
        "hash_invariant_pass": not missing and not mismatches,
        "candidate_package_still_immutable": not missing and not mismatches and not forbidden_hits,
        "checked_file_count": len(checks),
        "missing_count": len(missing),
        "mismatch_count": len(mismatches),
        "forbidden_hit_count": len(forbidden_hits),
        "missing": missing,
        "mismatches": mismatches,
        "forbidden_hits": forbidden_hits,
        "checks": checks,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "V226_A3_candidate_immutability_monitor.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORTS / "V226_A3_candidate_immutability_monitor.md").write_text(
        "\n".join(
            [
                "# V226 A3 Candidate Immutability Monitor",
                "",
                f"- hash_invariant_pass: `{payload['hash_invariant_pass']}`",
                f"- candidate_package_still_immutable: `{payload['candidate_package_still_immutable']}`",
                f"- checked_file_count: `{payload['checked_file_count']}`",
                f"- forbidden_hit_count: `{payload['forbidden_hit_count']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(out), "pass": payload["candidate_package_still_immutable"]}, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    monitor()
