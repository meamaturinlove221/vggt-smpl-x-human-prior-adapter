from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from v10_surface_completion_pipeline import REPORTS, REPO_ROOT, json_ready, write_json, write_report


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    parser = argparse.ArgumentParser(description="V14 TMF prediction readiness audit.")
    parser.add_argument("--output-json", type=Path, default=REPORTS / "20260508_v14_tmf_prediction_readiness.json")
    parser.add_argument("--output-md", type=Path, default=REPORTS / "20260508_v14_tmf_prediction_readiness.md")
    args = parser.parse_args()
    scene_rows = []
    pred_rows = []
    for frame in (0, 1, 2):
        scene = REPO_ROOT / f"output/4k4d_scenes/0012_11_frame{frame:04d}_12views_tmf"
        pred = REPO_ROOT / f"output/modal_results/0012_11_frame{frame:04d}_60views/predictions.npz"
        scene_rows.append({"frame": frame, "scene": str(scene), "exists": scene.is_dir()})
        pred_rows.append({"frame": frame, "predictions": str(pred), "exists": pred.is_file(), "size": pred.stat().st_size if pred.is_file() else 0})
    all_scenes = all(row["exists"] for row in scene_rows)
    all_predictions = all(row["exists"] for row in pred_rows)
    summary = {
        "task": "v14_t14_tmf_prediction_readiness",
        "created_utc": utc_now(),
        "status": "t14_predictions_missing" if not all_predictions else "t14_predictions_ready",
        "scenes": scene_rows,
        "predictions": pred_rows,
        "canonical_teacher_ready": False,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "decision": (
            "TMF cannot become a real canonical teacher until frame0001/frame0002 VGGT predictions are generated."
            if not all_predictions
            else "Frame predictions exist; next step is real canonical 2DGS/SDF, not procedural TMF."
        ),
        "blockers": [] if all_predictions else ["Missing adjacent-frame VGGT predictions for frame0001/frame0002."],
    }
    write_json(args.output_json, summary)
    write_report(args.output_md, "V14 TMF Prediction Readiness", summary)
    print(json.dumps(json_ready({"status": summary["status"], "canonical_teacher_ready": False}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
