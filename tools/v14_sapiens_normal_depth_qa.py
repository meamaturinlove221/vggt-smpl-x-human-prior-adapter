from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from v10_surface_completion_pipeline import REPORTS, REPO_ROOT, json_ready, scalar_stats, write_json, write_report


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def array_stats(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    arr = np.asarray(values)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool)]
    arr = arr[np.isfinite(arr)]
    return scalar_stats(arr) if arr.size else {"count": 0, "finite": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description="V14 Sapiens normal/depth QA and convention report.")
    parser.add_argument("--normal-dir", type=Path, default=REPO_ROOT / "output/surface_research_cloud_preflight/V13_Sapiens_Normal")
    parser.add_argument("--depth-dir", type=Path, default=REPO_ROOT / "output/surface_research_cloud_preflight/V13_Sapiens_Depth")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output/surface_research_preflight_local/V14_S14_sapiens_qa")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    normal_npz = args.normal_dir / "sapiens_normals.npz"
    depth_npz = args.depth_dir / "sapiens_depths.npz"
    normal_summary = load_json(args.normal_dir / "external_sapiens_normal_teacher_summary.json")
    depth_summary = load_json(args.depth_dir / "external_sapiens_depth_teacher_summary.json")

    n = np.load(normal_npz, allow_pickle=False)
    d = np.load(depth_npz, allow_pickle=False)
    normals = np.asarray(n["normal"], dtype=np.float32)
    normal_mask = np.asarray(n["mask"], dtype=bool)
    depths = np.asarray(d["depth"], dtype=np.float32)
    depth_mask = np.asarray(d["mask"], dtype=bool)

    norm_len = np.linalg.norm(normals, axis=-1)
    valid_normal = normal_mask & np.isfinite(normals).all(axis=-1) & (norm_len > 0.25)
    valid_depth = depth_mask & np.isfinite(depths)
    shared_mask = valid_normal & valid_depth
    # Image-space convention placeholders: Sapiens outputs are kept as image-space 2D supervision.
    camera_normals = normals.copy()
    camera_normals[..., 1] *= -1.0
    world_normals = camera_normals.copy()
    np.savez_compressed(out / "s14_sapiens_camera_normals.npz", normal=camera_normals, mask=valid_normal, image_names=n["image_names"])
    np.savez_compressed(out / "s14_sapiens_world_normals.npz", normal=world_normals, mask=valid_normal, image_names=n["image_names"])

    per_view = []
    for idx in range(normals.shape[0]):
        total = int(np.prod(normals.shape[1:3]))
        per_view.append(
            {
                "index": int(idx),
                "image_name": str(n["image_names"][idx]),
                "normal_valid_ratio": float(valid_normal[idx].sum() / max(total, 1)),
                "depth_valid_ratio": float(valid_depth[idx].sum() / max(total, 1)),
                "shared_valid_ratio": float(shared_mask[idx].sum() / max(total, 1)),
                "normal_length": array_stats(norm_len[idx], valid_normal[idx]),
                "depth": array_stats(depths[idx], valid_depth[idx]),
            }
        )
    gates = {
        "normal_npz_exists": normal_npz.is_file(),
        "depth_npz_exists": depth_npz.is_file(),
        "view_count_match": int(normals.shape[0]) == int(depths.shape[0]) == 12,
        "shape_match": list(normals.shape[:3]) == [int(depths.shape[0]), int(depths.shape[1]), int(depths.shape[2])],
        "normal_valid_ratio_ok": float(valid_normal.sum() / max(np.prod(valid_normal.shape), 1)) > 0.02,
        "depth_valid_ratio_ok": float(valid_depth.sum() / max(np.prod(valid_depth.shape), 1)) > 0.02,
    }
    summary = {
        "task": "v14_s14_sapiens_normal_depth_qa",
        "created_utc": utc_now(),
        "status": "s14_sapiens_qa_ready" if all(gates.values()) else "s14_sapiens_qa_blocked",
        "inputs": {
            "normal_npz": str(normal_npz.resolve()),
            "depth_npz": str(depth_npz.resolve()),
            "normal_summary": str((args.normal_dir / "external_sapiens_normal_teacher_summary.json").resolve()),
            "depth_summary": str((args.depth_dir / "external_sapiens_depth_teacher_summary.json").resolve()),
        },
        "normal_summary": {"num_views": normal_summary.get("num_views"), "model_repo": normal_summary.get("model_repo")},
        "depth_summary": {"num_views": depth_summary.get("num_views"), "model_repo": depth_summary.get("model_repo")},
        "shapes": {"normal": list(normals.shape), "depth": list(depths.shape)},
        "gates": gates,
        "per_view": per_view,
        "outputs": {
            "camera_normals": str((out / "s14_sapiens_camera_normals.npz").resolve()),
            "world_normals": str((out / "s14_sapiens_world_normals.npz").resolve()),
        },
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "decision": "Sapiens normal/depth QA passed for supervision use only; no 3D teacher/candidate pass is implied.",
    }
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v14_sapiens_normal_depth_qa.json", summary)
    write_report(REPORTS / "20260508_v14_sapiens_normal_depth_qa.md", "V14 Sapiens Normal/Depth QA", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": out}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
