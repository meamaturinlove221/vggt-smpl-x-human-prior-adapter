from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from v15_common import (
    DEFAULT_2DGS_SCENE,
    DEFAULT_G3_DIR,
    DEFAULT_SAPIENS_DEPTH,
    DEFAULT_SAPIENS_NORMAL,
    DEFAULT_TMF_SCENE,
    LOCAL_ROOT,
    REPORTS,
    camera_id_overlap,
    json_ready,
    load_colmap_cameras,
    parse_camera_id_from_name,
    read_json,
    safe_v15_output_dir,
    tmf_view_rows,
    utc_now,
    write_json,
    write_report,
)


def _npz_names(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with np.load(path, allow_pickle=False) as payload:
        if "image_names" in payload:
            return [str(x) for x in payload["image_names"]]
        if "view_names" in payload:
            return [str(x) for x in payload["view_names"]]
    return []


def _g3_names(g3_dir: Path) -> list[str]:
    summary = read_json(g3_dir / "summary.json")
    selected = str(summary.get("selected_tier") or "")
    tiers = summary.get("tiers") if isinstance(summary.get("tiers"), dict) else {}
    payload = tiers.get(selected) if selected in tiers else {}
    rows = payload.get("projection_audit") if isinstance(payload, dict) else None
    if rows:
        return [str(row.get("name")) for row in rows]
    depth_npz = Path(summary.get("depth_npz", "")) if summary.get("depth_npz") else g3_dir / "g3_2dgs_anchor_depth_6view.npz"
    return _npz_names(depth_npz)


def _camera_row(camera: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_id": int(camera["image_id"]),
        "name": camera["name"],
        "camera_id": parse_camera_id_from_name(camera["name"]),
        "width": int(camera["width"]),
        "height": int(camera["height"]),
        "fx": float(camera["fx"]),
        "fy": float(camera["fy"]),
        "cx": float(camera["cx"]),
        "cy": float(camera["cy"]),
        "camera_center_world": np.asarray(camera["camera_center_world"]).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V15 camera/view contract remap audit between G3/2DGS and Sapiens/TMF.")
    parser.add_argument("--g3-dir", type=Path, default=DEFAULT_G3_DIR)
    parser.add_argument("--two-dgs-scene", type=Path, default=DEFAULT_2DGS_SCENE)
    parser.add_argument("--tmf-scene", type=Path, default=DEFAULT_TMF_SCENE)
    parser.add_argument("--sapiens-normal-npz", type=Path, default=DEFAULT_SAPIENS_NORMAL)
    parser.add_argument("--sapiens-depth-npz", type=Path, default=DEFAULT_SAPIENS_DEPTH)
    parser.add_argument("--output-dir", type=Path, default=LOCAL_ROOT / "V15_GS_camera_view_contract_remap")
    args = parser.parse_args()

    out = safe_v15_output_dir(args.output_dir)
    g3_names = _g3_names(args.g3_dir)
    sapiens_normal_names = _npz_names(args.sapiens_normal_npz)
    sapiens_depth_names = _npz_names(args.sapiens_depth_npz)
    tmf_rows = tmf_view_rows(args.tmf_scene)
    colmap_cameras = load_colmap_cameras(args.two_dgs_scene)

    overlap_rows = camera_id_overlap(g3_names, sapiens_normal_names)
    overlap_count = sum(1 for row in overlap_rows if row["match_count"] == 1)
    g3_camera_ids = [parse_camera_id_from_name(name) for name in g3_names]
    sapiens_camera_ids = [parse_camera_id_from_name(name) for name in sapiens_normal_names]
    missing_sapiens_for_g3 = [row for row in overlap_rows if row["match_count"] == 0]
    extra_sapiens = sorted(set(sapiens_camera_ids) - set(g3_camera_ids))
    exact_name_matches = sorted(set(g3_names).intersection(set(sapiens_normal_names)))

    remap_entries: list[dict[str, Any]] = []
    sapiens_index_by_cam = {parse_camera_id_from_name(name): idx for idx, name in enumerate(sapiens_normal_names)}
    for g_idx, g_name in enumerate(g3_names):
        cam_id = parse_camera_id_from_name(g_name)
        if cam_id not in sapiens_index_by_cam:
            continue
        s_idx = sapiens_index_by_cam[cam_id]
        remap_entries.append(
            {
                "camera_id": cam_id,
                "g3_index": int(g_idx),
                "g3_name": g_name,
                "sapiens_index": int(s_idx),
                "sapiens_name": sapiens_normal_names[s_idx],
            }
        )

    shapes = {
        "g3_view_count": len(g3_names),
        "sapiens_normal_view_count": len(sapiens_normal_names),
        "sapiens_depth_view_count": len(sapiens_depth_names),
        "tmf_manifest_view_count": len(tmf_rows),
        "colmap_camera_count": len(colmap_cameras),
        "g3_resolution": [518, 518] if g3_names else None,
        "sapiens_tensor_resolution": [1024, 768] if sapiens_normal_names else None,
        "tmf_image_sizes": sorted({tuple(row.get("image_size") or []) for row in tmf_rows}),
    }
    gates = {
        "g3_has_6_views": len(g3_names) == 6,
        "sapiens_has_12_views": len(sapiens_normal_names) == 12 and len(sapiens_depth_names) == 12,
        "sapiens_normal_depth_names_match": sapiens_normal_names == sapiens_depth_names,
        "tmf_manifest_has_12_views": len(tmf_rows) == 12,
        "colmap_has_6_cameras": len(colmap_cameras) == 6,
        "all_g3_cameras_have_sapiens_overlap": overlap_count == len(g3_names) and len(g3_names) > 0,
        "exact_view_names_match": len(exact_name_matches) == len(g3_names) == len(sapiens_normal_names),
        "same_view_count_and_resolution": len(g3_names) == len(sapiens_normal_names) and shapes["g3_resolution"] == shapes["sapiens_tensor_resolution"],
    }
    merge_ready = bool(
        gates["all_g3_cameras_have_sapiens_overlap"]
        and gates["sapiens_normal_depth_names_match"]
        and not gates["same_view_count_and_resolution"]
    )
    blockers = []
    if not gates["same_view_count_and_resolution"]:
        blockers.append("G3/2DGS is a 6-view 518x518 COLMAP contract while Sapiens/TMF is a 12-view 1024x768 tensor contract.")
    if not gates["exact_view_names_match"]:
        blockers.append("View filenames are not exact matches; remap is camera-id only and must be treated as an explicit resampling bridge.")
    if missing_sapiens_for_g3:
        blockers.append(f"{len(missing_sapiens_for_g3)} G3 cameras have no Sapiens camera-id match.")
    if extra_sapiens:
        blockers.append(f"Sapiens has {len(extra_sapiens)} extra cameras outside the G3 contract: {extra_sapiens}.")

    summary = {
        "task": "v15_camera_view_contract_remap",
        "created_utc": utc_now(),
        "status": "v15_camera_remap_ready_research_only" if merge_ready else "v15_camera_remap_blocked",
        "research_only": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "strict_teacher_passes": 0,
        "strict_candidate_passes": 0,
        "inputs": {
            "g3_dir": str(args.g3_dir.resolve()),
            "two_dgs_scene": str(args.two_dgs_scene.resolve()),
            "tmf_scene": str(args.tmf_scene.resolve()),
            "sapiens_normal_npz": str(args.sapiens_normal_npz.resolve()),
            "sapiens_depth_npz": str(args.sapiens_depth_npz.resolve()),
        },
        "metrics": {
            "g3_view_count": len(g3_names),
            "sapiens_view_count": len(sapiens_normal_names),
            "camera_id_overlap_count": overlap_count,
            "extra_sapiens_camera_count": len(extra_sapiens),
            "exact_name_match_count": len(exact_name_matches),
        },
        "gates": gates,
        "shapes": shapes,
        "g3_camera_ids": g3_camera_ids,
        "sapiens_camera_ids": sapiens_camera_ids,
        "overlap_rows": overlap_rows,
        "remap_entries": remap_entries,
        "colmap_cameras": [_camera_row(camera) for camera in colmap_cameras],
        "tmf_views": tmf_rows,
        "outputs": {"remap_json": str((out / "v15_camera_view_remap.json").resolve())},
        "decision": "Camera-id overlap is sufficient for bounded 6-view research comparisons, but not for direct teacher/candidate merging because view count, names, and resolution differ.",
        "blockers": blockers,
    }
    write_json(out / "v15_camera_view_remap.json", {"remap_entries": remap_entries, "overlap_rows": overlap_rows})
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v15_camera_view_contract_remap.json", summary)
    write_report(REPORTS / "20260508_v15_camera_view_contract_remap.md", "V15 Camera View Contract Remap", summary)
    print(json.dumps(json_ready({"status": summary["status"], "metrics": summary["metrics"], "output": out}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
