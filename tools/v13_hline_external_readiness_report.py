from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from v10_surface_completion_pipeline import REPORTS, REPO_ROOT, json_ready, write_json, write_report


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _exists(path: Path) -> dict[str, Any]:
    return {"path": path, "exists": path.exists(), "is_dir": path.is_dir(), "is_file": path.is_file()}


def _count_files(root: Path, patterns: tuple[str, ...]) -> int:
    if not root.exists():
        return 0
    total = 0
    for pattern in patterns:
        total += len(list(root.rglob(pattern)))
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="V13 H-line external hand/hair route readiness report.")
    parser.add_argument("--output-json", type=Path, default=REPORTS / "20260508_v13_hline_external_readiness.json")
    parser.add_argument("--output-md", type=Path, default=REPORTS / "20260508_v13_hline_external_readiness.md")
    args = parser.parse_args()
    external = REPO_ROOT / "external"
    external_models = REPO_ROOT / "external_models"
    roots = {
        "HGGT": external / "HGGT-main",
        "HairGS": external / "hair-gs-master",
        "MUSt3R": external / "must3r",
        "HaMeR": external / "HaMeR",
        "WiLoR": external / "WiLoR",
        "OSX": external / "OSX",
        "SMPLerX": external / "SMPLer-X",
        "GaussianHaircut": external / "GaussianHaircut",
    }
    rows = {}
    for name, root in roots.items():
        rows[name] = {
            **_exists(root),
            "checkpoint_like_count": _count_files(root, ("*.pt", "*.pth", "*.ckpt", "*.safetensors", "*.npz", "*.pkl")),
            "dataset_like_count": _count_files(root / "dataset", ("*.png", "*.jpg", "*.jpeg", "*.npy", "*.npz", "*.pkl", "*.obj")),
        }
    media = {
        "hand_landmarker": _exists(external_models / "hand_landmarker.task"),
        "face_landmarker": _exists(external_models / "face_landmarker.task"),
    }
    blockers = [
        "No HaMeR/WiLoR/OSX/SMPLer-X/GaussianHaircut repo+checkpoint route is locally runnable.",
        "HGGT public source lacks released inference/checkpoint/assets for strict hand ownership.",
        "Hair-GS source lacks FLAME/hair datasets and local dependency stack for strict hair topology.",
        "MediaPipe task files are weak evidence only and forbidden for promotion.",
    ]
    summary = {
        "task": "v13_hline_external_readiness",
        "created_utc": utc_now(),
        "status": "hline_external_routes_blocked",
        "routes": rows,
        "external_models": media,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "decision": "H-line cannot produce strict hand/hair ownership from currently available local external assets.",
        "blockers": blockers,
    }
    write_json(args.output_json, summary)
    write_report(args.output_md, "V13 H-line External Readiness", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": args.output_json}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
