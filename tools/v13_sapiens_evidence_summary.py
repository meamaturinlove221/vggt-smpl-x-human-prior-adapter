from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from v10_surface_completion_pipeline import REPORTS, REPO_ROOT, json_ready, write_json, write_report


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def npz_shapes(path: Path) -> dict[str, list[int]]:
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {name: [int(v) for v in data[name].shape] for name in data.files if hasattr(data[name], "shape")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize V13 Sapiens 2D supervision outputs.")
    parser.add_argument("--normal-dir", type=Path, default=REPO_ROOT / "output/surface_research_cloud_preflight/V13_Sapiens_Normal")
    parser.add_argument("--depth-dir", type=Path, default=REPO_ROOT / "output/surface_research_cloud_preflight/V13_Sapiens_Depth")
    parser.add_argument("--output-json", type=Path, default=REPORTS / "20260508_v13_sapiens_evidence_summary.json")
    parser.add_argument("--output-md", type=Path, default=REPORTS / "20260508_v13_sapiens_evidence_summary.md")
    args = parser.parse_args()

    normal_summary = load_json(args.normal_dir / "external_sapiens_normal_teacher_summary.json")
    depth_summary = load_json(args.depth_dir / "external_sapiens_depth_teacher_summary.json")
    summary = {
        "task": "v13_sapiens_2d_supervision_evidence",
        "created_utc": utc_now(),
        "status": "sapiens_2d_supervision_ready" if normal_summary and depth_summary else "sapiens_2d_supervision_incomplete",
        "normal": {
            "dir": str(args.normal_dir.resolve()),
            "summary": str((args.normal_dir / "external_sapiens_normal_teacher_summary.json").resolve()),
            "npz": str((args.normal_dir / "sapiens_normals.npz").resolve()),
            "num_views": normal_summary.get("num_views"),
            "model_repo": normal_summary.get("model_repo"),
            "model_filename": normal_summary.get("model_filename"),
            "shapes": npz_shapes(args.normal_dir / "sapiens_normals.npz"),
        },
        "depth": {
            "dir": str(args.depth_dir.resolve()),
            "summary": str((args.depth_dir / "external_sapiens_depth_teacher_summary.json").resolve()),
            "npz": str((args.depth_dir / "sapiens_depths.npz").resolve()),
            "num_views": depth_summary.get("num_views"),
            "model_repo": depth_summary.get("model_repo"),
            "model_filename": depth_summary.get("model_filename"),
            "shapes": npz_shapes(args.depth_dir / "sapiens_depths.npz"),
        },
        "strict_teacher_passes": 0,
        "strict_candidate_passes": 0,
        "research_only": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "decision": (
            "Sapiens normal/depth outputs are valid 2D supervision evidence for later G/F-line training, "
            "but they are not a 3D surface teacher and cannot unlock promotion."
        ),
    }
    write_json(args.output_json, summary)
    write_report(args.output_md, "V13 Sapiens 2D Evidence Summary", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": args.output_json}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
