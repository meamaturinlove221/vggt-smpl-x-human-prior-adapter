from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v30_prior_common import (
    CLOUD_ROOT,
    CONTROLS,
    FRAMES,
    LOCAL_ROOT,
    REPORT_JSON,
    REPO_ROOT,
    REQUIRED_RESEARCH_FILES,
    file_info,
    load_json,
    npz_summary,
    scan_forbidden,
    utc_now,
    write_json,
)


REPORT_INTAKE_JSON = REPO_ROOT / "reports/20260508_v30_prior_enabled_prediction_intake.json"


def _root_payload(root: Path) -> dict[str, Any]:
    files = {name: file_info(root / name) for name in REQUIRED_RESEARCH_FILES}
    arrays = {
        name: npz_summary(root / name)
        for name in (
            "research_depths.npz",
            "research_points_world.npz",
            "research_confidence.npz",
            "research_normals_geometric.npz",
        )
    }
    return {
        "root": str(root.resolve()),
        "exists": root.is_dir(),
        "files": files,
        "arrays": arrays,
        "prior_effect": load_json(root / "research_prior_effect.json"),
        "forbidden_findings": scan_forbidden(root),
    }


def _validate_payload(payload: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not payload.get("exists"):
        blockers.append(f"prediction root missing: {payload.get('root')}")
        return blockers
    files = payload.get("files", {})
    for name in REQUIRED_RESEARCH_FILES:
        if not files.get(name, {}).get("exists"):
            blockers.append(f"missing required V30 research file: {name}")
    if payload.get("forbidden_findings"):
        blockers.append("forbidden output path/name found under V30 research root")

    prior_effect = payload.get("prior_effect") or {}
    if prior_effect:
        controls = prior_effect.get("controls") or {}
        missing = [name for name in CONTROLS if name not in controls]
        if missing:
            blockers.append(f"research_prior_effect.json missing controls: {missing}")
        if prior_effect.get("human_prior_channels", 0) <= 0:
            blockers.append("research_prior_effect.json does not prove human_prior_channels > 0")
        for name in ("research_depths.npz", "research_points_world.npz", "research_normals_geometric.npz"):
            keys = set((payload.get("arrays", {}).get(name, {}) or {}).get("keys") or [])
            expected_frame_keys = set(FRAMES)
            if not expected_frame_keys.issubset(keys):
                blockers.append(f"{name} missing frame arrays {sorted(expected_frame_keys - keys)}")
    return blockers


def run_intake() -> dict[str, Any]:
    verifier = load_json(REPO_ROOT / "reports/20260508_v30_prior_channel_verifier.json")
    local_payload = _root_payload(LOCAL_ROOT)
    cloud_payload = _root_payload(CLOUD_ROOT)
    local_blockers = _validate_payload(local_payload)
    cloud_blockers = _validate_payload(cloud_payload)
    any_valid_prediction_payload = not local_blockers or not cloud_blockers

    blockers: list[str] = []
    if not verifier.get("usable_prior_enabled_checkpoint_exists"):
        blockers.append("V30 verifier found no usable prior-enabled VGGT checkpoint, so prediction payload is expected to be absent.")
    if not any_valid_prediction_payload:
        blockers.append("No complete prior-enabled V30 research prediction payload was found in local or cloud research roots.")

    summary = {
        "task": "v30_prior_enabled_prediction_intake",
        "created_utc": utc_now(),
        "status": "DONE_PASS" if any_valid_prediction_payload else "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE",
        "decision": (
            "Prior-enabled prediction payload is complete and research-only."
            if any_valid_prediction_payload
            else "Prediction intake fail-closed: V30 cannot accept base VGGT outputs or random prior-enabled weights."
        ),
        "verifier_status": verifier.get("status"),
        "usable_prior_enabled_checkpoint_exists": bool(verifier.get("usable_prior_enabled_checkpoint_exists")),
        "local_payload": local_payload,
        "cloud_payload": cloud_payload,
        "local_blockers": local_blockers,
        "cloud_blockers": cloud_blockers,
        "blockers": blockers,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_unblocked": False,
        "no_predictions_npz": not any(
            finding.get("reason", "").startswith("predictions.npz")
            for finding in (local_payload.get("forbidden_findings") or []) + (cloud_payload.get("forbidden_findings") or [])
        ),
        "no_package_registry_or_strict_pass": not (local_payload.get("forbidden_findings") or cloud_payload.get("forbidden_findings")),
    }
    write_json(REPORT_INTAKE_JSON, summary)
    write_json(LOCAL_ROOT / "v30_prediction_intake.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=REPORT_INTAKE_JSON)
    args = parser.parse_args()
    summary = run_intake()
    if args.json_out != REPORT_INTAKE_JSON:
        write_json(args.json_out, summary)
    print(summary["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
