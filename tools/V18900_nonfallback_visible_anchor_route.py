from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
REPORTS = REPO / "reports"
GOALS = REPO / "docs" / "goals"
BOARDS = REPO / "boards"
OUTPUT = REPO / "output"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure(path.parent)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    created_at = now()
    runtime_path = REPORTS / "V18700000000000000000_runtime_environment.json"
    manifest_path = REPORTS / "V18700000000000000000_training_manifest.csv"
    scores_path = REPORTS / "V18700000000000000000_visible_anchor_scores.csv"
    decision_path = REPORTS / "V18700000000000000000_training_decision.json"
    restore_path = REPORTS / "V18800000000000000000_asset_restoration_decision.json"
    required = [runtime_path, manifest_path, scores_path, decision_path, restore_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        write_json(
            REPORTS / "V18900000000000000000_nonfallback_visible_anchor_decision.json",
            {
                "created_at": created_at,
                "status": "V18900_MISSING_AUDIT_INPUTS",
                "mentor_ready": False,
                "external_hard_block": False,
                "missing": missing,
            },
        )
        return 1

    runtime = read_json(runtime_path)
    restore = read_json(restore_path)
    decision = read_json(decision_path)
    manifest = read_csv(manifest_path)
    scores = read_csv(scores_path)
    true_scores = [r for r in scores if r.get("config") == "visible_anchor_canonical_surfel_true"]
    fallback_flags = {r.get("feature_bank_fallback_used", "").strip().lower() for r in manifest}
    nonfallback_ok = fallback_flags == {"false"}
    modal_a10_ok = runtime.get("selected_device") == "cuda" and "A10" in str(runtime.get("cuda_device_name", ""))
    four_case_restore_ok = bool(restore.get("four_case_v950_v536_ok"))
    true_failures = [r for r in true_scores if r.get("combined_fail_v4", "").strip().lower() == "true"]
    close_controls = [
        f"{f.get('case')}::{f.get('control')}"
        for f in decision.get("failures", [])
        if f.get("reason") == "control_or_prior_close_or_better_v4"
    ]
    shell_losses_zero = []
    for r in manifest:
        hist = json.loads(r.get("history_json") or "[]")
        final = hist[-1] if hist else {}
        if float(final.get("shell", 0.0)) <= 1e-8 and float(final.get("coverage", 0.0)) <= 1e-8:
            shell_losses_zero.append(r.get("case", ""))

    advisor_board = BOARDS / "V18700000000000000000_visible_anchor_board.png"
    turntable_board = BOARDS / "V18700000000000000000_visible_anchor_turntable_cross_section.png"
    output_dirs = sorted((OUTPUT / "V18700000000000000000_visible_anchor_canonical_surfel_training").glob("*/*/predictions.npz"))
    status = "V18900_NONFALLBACK_VISIBLE_ANCHOR_FAIL_CLOSED_CONTINUE"
    decision_payload = {
        "created_at": created_at,
        "status": status,
        "mentor_ready": False,
        "external_hard_block": False,
        "v188_assets_restored": four_case_restore_ok,
        "modal_a10_run_confirmed": modal_a10_ok,
        "feature_bank_fallback_used": not nonfallback_ok,
        "feature_bank_fallback_flags": sorted(fallback_flags),
        "true_combined_fail_v4_cases": [r.get("case") for r in true_failures],
        "close_or_better_controls": close_controls,
        "shell_coverage_loss_inactive_cases": shell_losses_zero,
        "advisor_board": str(advisor_board),
        "turntable_cross_section_board": str(turntable_board),
        "prediction_count": len(output_dirs),
        "route_decision": "Continue to V189/V190 anchored pose-frame occupancy repair. Do not tune viewer/thickness/fallback; assets are restored and failure is model/representation-level.",
        "summary": "Restored V950/V536/V161/V140 assets allow a non-fallback Modal A10 V187 run, but the output remains fail-closed: full-scene visuals are torn/tilted clouds and hard controls remain close.",
    }
    write_json(REPORTS / "V18900000000000000000_nonfallback_visible_anchor_decision.json", decision_payload)

    route = f"""# V18900000000000000000 Non-Fallback Visible Anchor Failure Route

Created: {created_at}

## Conclusion

V188 restored the missing V950/V536/V161/V140 assets and V187 was rerun on Modal A10 without feature-bank fallback.

This removes the previous asset-restoration uncertainty, but it does not create mentor-ready evidence.

## Current Evidence

- Modal A10 run confirmed: `{modal_a10_ok}`
- feature_bank_fallback_used: `{not nonfallback_ok}`
- restored four-case V950/V536 assets: `{four_case_restore_ok}`
- V187 true combined_fail_v4 cases: `{len(true_failures)}/4`
- close/better controls: `{", ".join(close_controls) if close_controls else "none"}`
- advisor board: `{advisor_board}`
- turntable/cross-section board: `{turntable_board}`

## Failure Interpretation

The current failure is no longer a missing-asset hard block. The non-fallback visible-anchor student still produces a torn / tilted volume cloud rather than a natural human-main full-scene RGB point cloud.

The training history also shows shell / coverage / occupancy-completeness terms are effectively inactive in the final checkpoints for these cases. This means the model is optimizing anchor proximity without learning a posed, body-frame topology-volume occupancy structure.

## Next Route

Continue with anchored pose-frame occupancy repair:

1. Decode occupancy in posed SMPL local frames rather than free visible anchors.
2. Make front/back/side shell, part continuity, and cross-section occupancy active training objectives.
3. Add a full-scene upright pose-frame alignment gate before mentor rendering.
4. Keep real VGGT environment insertion, same-scene controls, and face-detail overclaim guards.
5. Fail closed if the main board is still billboard / torn cloud / tilted shell.

## Forbidden Returns

- mentor-ready from V187;
- fallback/surrogate evidence;
- render-only pass;
- thickness-only pass;
- projection-only pass;
- route exhausted;
- visual failure as external hard block.
"""
    write_text(GOALS / "V18900000000000000000_auto_evolved_pose_frame_occupancy_route.md", route)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
