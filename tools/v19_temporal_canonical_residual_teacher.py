from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
LOCAL_ROOT = REPO_ROOT / "output" / "surface_research_preflight_local"
DEFAULT_OUT = LOCAL_ROOT / "V19_temporal_canonical_residual_teacher"
DEFAULT_JSON = REPORTS / "20260508_v19_temporal_canonical_residual_teacher.json"
DEFAULT_MD = REPORTS / "20260508_v19_temporal_canonical_residual_teacher.md"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jr(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jr(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jr(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jr(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def dir_info(path: Path) -> dict[str, Any]:
    return {
        "path": path,
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "file_count": sum(1 for p in path.rglob("*") if p.is_file()) if path.is_dir() else 0,
    }


def prediction_info(frame_id: int) -> dict[str, Any]:
    hits = list((REPO_ROOT / "output").rglob(f"*frame{frame_id:04d}*predictions.npz"))
    return {"frame": frame_id, "prediction_count": len(hits), "predictions": hits[:10]}


def main() -> int:
    parser = argparse.ArgumentParser(description="V19 temporal canonical residual teacher audit.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    frames = [0, 1, 2]
    scene_rows = []
    pred_rows = []
    for frame in frames:
        scene = REPO_ROOT / "output" / "4k4d_scenes" / f"0012_11_frame{frame:04d}_12views_tmf"
        scene_rows.append(dir_info(scene))
        pred_rows.append(prediction_info(frame))

    scene_ready = all(row["exists"] and row["is_dir"] for row in scene_rows)
    predictions_ready = all(row["prediction_count"] > 0 for row in pred_rows)
    support_csv = out / "temporal_support_table.csv"
    support_csv.write_text(
        "frame,scene_exists,scene_file_count,prediction_count\n"
        + "\n".join(
            f"{frame},{scene_rows[i]['exists']},{scene_rows[i]['file_count']},{pred_rows[i]['prediction_count']}"
            for i, frame in enumerate(frames)
        )
        + "\n",
        encoding="utf-8",
    )
    status = "v19_temporal_assets_ready_predictions_missing_research_only"
    if scene_ready and predictions_ready:
        status = "v19_temporal_assets_and_predictions_ready_research_only"
    elif not scene_ready:
        status = "v19_temporal_scene_assets_missing_research_only"

    summary = {
        "task": "v19_temporal_canonical_residual_teacher",
        "created_utc": utc_now(),
        "status": status,
        "research_only": True,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "scene_ready": scene_ready,
        "predictions_ready": predictions_ready,
        "frames": scene_rows,
        "predictions": pred_rows,
        "outputs": {"support_csv": support_csv, "summary_json": out / "summary.json"},
        "decision": (
            "V19 can use frame0000/0001/0002 TMF scene assets for temporal canonical planning, "
            "but adjacent VGGT predictions are not available for a real canonical teacher."
            if scene_ready and not predictions_ready
            else "V19 temporal canonical audit completed."
        ),
        "blockers": [] if predictions_ready else ["Adjacent frame VGGT predictions are missing; do not construct a procedural temporal teacher."],
    }
    write_json(args.output_json, summary)
    write_json(out / "summary.json", summary)
    lines = [
        "# V19 Temporal Canonical Residual Teacher",
        "",
        f"Status: `{status}`",
        "",
        summary["decision"],
        "",
        "## Frame Support",
        "",
    ]
    for row, pred in zip(scene_rows, pred_rows):
        lines.append(f"- frame{pred['frame']:04d}: scene_exists=`{row['exists']}`, files=`{row['file_count']}`, predictions=`{pred['prediction_count']}`")
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- {b}" for b in summary["blockers"]] or ["- none"])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(jr({"status": status, "json": args.output_json}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
