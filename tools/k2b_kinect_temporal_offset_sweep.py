from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from v10_surface_completion_pipeline import LOCAL_ROOT, REPORTS, json_ready, write_json, write_report


REPO_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def gate_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    gate = summary.get("teacher_gate", {})
    target = summary.get("target_alignment", {})
    residuals = target.get("residual_percentiles", []) or gate.get("residual_percentiles", [])
    distances = gate.get("distance_to_base_percentiles", []) or summary.get("distance_to_base_on_teacher_mask_percentiles", [])
    views = gate.get("checks", {}).get("per_view_visibility", {}).get("views", [])
    hit_ratios = [float(v.get("hit_ratio_in_roi", 0.0)) for v in views if isinstance(v, dict)]
    connected = [float(v.get("largest_hit_component_ratio", 0.0)) for v in views if isinstance(v, dict)]
    return {
        "teacher_gate_pass": bool(gate.get("pass")),
        "teacher_targets_written": bool(summary.get("teacher_targets_written")),
        "residual_p50": float(residuals[2]) if len(residuals) > 2 else None,
        "residual_p95": float(residuals[5]) if len(residuals) > 5 else None,
        "distance_p50": float(distances[2]) if len(distances) > 2 else None,
        "distance_p95": float(distances[5]) if len(distances) > 5 else None,
        "view_pass_ratio": gate.get("checks", {}).get("per_view_visibility", {}).get("view_pass_ratio"),
        "min_hit_ratio": min(hit_ratios) if hit_ratios else None,
        "mean_hit_ratio": sum(hit_ratios) / len(hit_ratios) if hit_ratios else None,
        "min_connected_ratio": min(connected) if connected else None,
    }


def score(metrics: dict[str, Any]) -> float:
    residual = metrics.get("residual_p50")
    distance = metrics.get("distance_p50")
    if residual is None and distance is None:
        return float("inf")
    if residual is None:
        residual = 999.0
    if distance is None:
        distance = 999.0
    return float(residual) + float(distance)


def main() -> int:
    parser = argparse.ArgumentParser(description="V13 K2b Kinect temporal offset sweep.")
    parser.add_argument("--scene-dir", type=Path, default=REPO_ROOT / "output/4k4d_scenes/0012_11_frame0000_12views_tmf")
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=REPO_ROOT / "output/modal_results/0012_11_frame0000_60views/predictions.npz",
    )
    parser.add_argument("--output-dir", type=Path, default=LOCAL_ROOT / "V13_K2b_kinect_temporal_offset_sweep")
    parser.add_argument("--frames", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--roi-kind", default="all")
    parser.add_argument("--alignment-source", default="camera_axes")
    parser.add_argument("--transform-mode", default="similarity")
    parser.add_argument("--overwrite", action="store_true", default=True)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for frame in args.frames:
        frame_out = out / f"kinect_frame{int(frame):04d}_to_target_frame0000"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "tools/build_kinect_depth_teacher_targets.py"),
            "--scene-dir",
            str(args.scene_dir),
            "--base-predictions",
            str(args.base_predictions),
            "--output-dir",
            str(frame_out),
            "--frame",
            str(int(frame)),
            "--roi-kind",
            str(args.roi_kind),
            "--alignment-source",
            str(args.alignment_source),
            "--transform-mode",
            str(args.transform_mode),
            "--require-teacher-gate",
            "--overwrite",
        ]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
        summary_path = frame_out / "kinect_teacher_summary.json"
        summary = read_json(summary_path)
        metrics = gate_metrics(summary)
        rows.append(
            {
                "frame": int(frame),
                "returncode": int(proc.returncode),
                "summary": str(summary_path) if summary_path.is_file() else None,
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-2000:],
                "metrics": metrics,
                "score": score(metrics),
            }
        )

    valid_rows = [row for row in rows if row["summary"]]
    best = min(valid_rows, key=lambda row: row["score"]) if valid_rows else None
    any_pass = any(bool(row["metrics"].get("teacher_gate_pass")) for row in rows)
    summary = {
        "task": "v13_k2b_kinect_temporal_offset_sweep",
        "created_utc": utc_now(),
        "status": "k2b_temporal_sweep_complete",
        "scene_dir": str(args.scene_dir.resolve()),
        "base_predictions": str(args.base_predictions.resolve()),
        "frames_tested": [int(v) for v in args.frames],
        "rows": rows,
        "best_frame": best.get("frame") if best else None,
        "best_metrics": best.get("metrics") if best else {},
        "teacher_gate_pass": bool(any_pass),
        "strict_teacher_passes": 0,
        "strict_candidate_passes": 0,
        "decision": (
            "Temporal offset produced a gate-pass Kinect teacher candidate."
            if any_pass
            else "Temporal offset sweep did not fix the official Kinect teacher gate; Kinect remains weak evidence only."
        ),
    }
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v13_k2b_kinect_temporal_offset_sweep.json", summary)
    write_report(REPORTS / "20260508_v13_k2b_kinect_temporal_offset_sweep.md", "V13 K2b Kinect Temporal Offset Sweep", summary)
    print(json.dumps(json_ready({"status": summary["status"], "best_frame": summary["best_frame"], "teacher_gate_pass": any_pass}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
