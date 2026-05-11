#!/usr/bin/env python
"""Audit V31 training curve and checkpoint artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "surface_research_preflight_local" / "V31_teacher_supervised_candidate_train"
REPORT_JSON = ROOT / "reports" / "20260508_v31_training_curve_audit.json"
REPORT_MD = ROOT / "reports" / "20260508_v31_training_curve_audit.md"


def main() -> None:
    summary_path = OUT / "summary.json"
    curve_path = OUT / "v31_training_curve.jsonl"
    ckpt_path = OUT / "v31_candidate_research_checkpoint.npz"
    preview_path = OUT / "v31_real_teacher_real_prior_candidate_preview.npz"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    curve = []
    if curve_path.exists():
        curve = [json.loads(line) for line in curve_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    blockers = []
    if not ckpt_path.exists():
        blockers.append("missing_checkpoint")
    if not preview_path.exists():
        blockers.append("missing_candidate_preview")
    if len(curve) < 2:
        blockers.append("training_curve_too_short")
    if curve and not (curve[-1]["total_loss"] <= curve[0]["total_loss"]):
        blockers.append("loss_not_nonincreasing")
    wins = summary.get("real_wins_controls", {})
    if wins and not all(bool(v) for v in wins.values()):
        blockers.append("real_does_not_win_all_controls")
    if ckpt_path.exists():
        data = np.load(ckpt_path, allow_pickle=True)
        required = ["temporal_blend", "depth_scale", "control_bias", "region_offsets", "region_names"]
        missing = [k for k in required if k not in data.files]
        if missing:
            blockers.append("checkpoint_missing_" + ",".join(missing))
    audit = {
        "status": "DONE_PASS" if not blockers else "DONE_FAIL_ROUTED",
        "blockers": blockers,
        "checkpoint_path": str(ckpt_path),
        "preview_path": str(preview_path),
        "curve_path": str(curve_path),
        "curve_rows": len(curve),
        "initial_loss": curve[0]["total_loss"] if curve else None,
        "final_loss": curve[-1]["total_loss"] if curve else None,
        "real_wins_controls": wins,
    }
    REPORT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = ["# V31 Training Curve Audit", "", f"status: `{audit['status']}`", "", f"blockers: `{blockers}`", f"curve_rows: `{len(curve)}`"]
    if curve:
        lines.append(f"initial_loss: `{curve[0]['total_loss']:.6f}`")
        lines.append(f"final_loss: `{curve[-1]['total_loss']:.6f}`")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "blockers": blockers}, indent=2))


if __name__ == "__main__":
    main()
