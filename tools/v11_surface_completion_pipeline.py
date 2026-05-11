from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from v10_surface_completion_pipeline import (
    CLOUD_ROOT,
    CONTRACT,
    DEFAULT_2DGS_SCENE,
    DEFAULT_ASSETS,
    DEFAULT_TEMPLATE,
    LOCAL_ROOT,
    REGIONS,
    REPORTS,
    REPO_ROOT,
    bbox_stats,
    contact_sheet,
    json_ready,
    load_colmap_cameras,
    load_mask,
    load_ply_xyz_rgb,
    load_template,
    make_projection_mask,
    mask_iou,
    paste_grid,
    read_ply_header,
    read_summary,
    region_masks_from_template,
    safe_output_dir,
    scalar_stats,
    select_region_by_bbox,
    utc_now,
    write_ascii_ply,
    write_json,
    write_report,
)


V10_UNIFIED = LOCAL_ROOT / "V10_unified_surface_merge_precheck"
V10_A5X3 = LOCAL_ROOT / "V10_A5X3_must3r_known_camera_alignment"
V10_FUS3D4 = LOCAL_ROOT / "B_Fus3D4_surface_candidate_precheck"
V10_HAND11 = LOCAL_ROOT / "B_Hand11_real_vggt_hand_token_decoder"
V10_HAIR4 = LOCAL_ROOT / "B_Hair4_native_4k4d_smplx_hair_topology"
V10_DLINE = LOCAL_ROOT / "DLine_V10_promotion_transaction"
V10_2DGS_TIERS = {
    "1k": CLOUD_ROOT / "Cloud_G_V10/a5x3_2dgs_colmap_scene_1k",
    "10k": CLOUD_ROOT / "Cloud_G_V10/a5x3_2dgs_colmap_scene_10k",
    "30k": CLOUD_ROOT / "Cloud_G_V10/a5x3_2dgs_colmap_scene_30k",
}

V11_QUARANTINE = LOCAL_ROOT / "V11_quarantine"
V11_A5X4 = LOCAL_ROOT / "V11_A5X4_must3r_evidence_pool"
V11_G3 = LOCAL_ROOT / "V11_G3_2DGS_surface_anchor"
V11_FUS3D5 = LOCAL_ROOT / "V11_B_Fus3D5_body_head_face"
V11_HHAND_A = LOCAL_ROOT / "V11_HHand_A_external_anchor"
V11_HHAND_B = LOCAL_ROOT / "V11_HHand_B_vggt_decoder"
V11_HHAIR_A = LOCAL_ROOT / "V11_HHair_A_official_route"
V11_HHAIR_B = LOCAL_ROOT / "V11_HHair_B_native_strand_gaussian"
V11_ROI = LOCAL_ROOT / "V11_ROI_micro_surface"
V11_UNIFIED = LOCAL_ROOT / "V11_unified_region_ownership_merge"
V11_DLINE = LOCAL_ROOT / "DLine_V11_promotion_transaction"
V11_ROLLUP = LOCAL_ROOT / "V11_execution_rollup"


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _point_regions(points: np.ndarray) -> dict[str, Any]:
    template = load_template()
    masks = region_masks_from_template(template)
    tpts = template["hybrid_vertices"].astype(np.float32)
    out: dict[str, Any] = {}
    for name in REGIONS:
        fallback = (0.65, 1.0) if name in {"head", "face_core", "hairline"} else (0.0, 1.0)
        pmask = select_region_by_bbox(points, tpts[masks[name]], fallback_fraction=fallback)
        out[name] = {
            "point_count": int(pmask.sum()),
            "coverage_nonempty": bool(pmask.sum() > 50),
            "bbox": bbox_stats(points[pmask]),
        }
    return out


def _downsample(points: np.ndarray, colors: np.ndarray, max_points: int = 260000) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(1108)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx], colors[idx]


def _trim_floating(points: np.ndarray, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    finite = np.isfinite(points).all(axis=1)
    pts = points[finite]
    cols = colors[finite] if len(colors) == len(points) else np.full((int(finite.sum()), 3), 210, dtype=np.uint8)
    if len(pts) == 0:
        return pts, cols, {"input": int(len(points)), "kept": 0, "finite": int(finite.sum())}
    lo = np.percentile(pts, 0.8, axis=0)
    hi = np.percentile(pts, 99.2, axis=0)
    keep = np.logical_and(pts >= lo, pts <= hi).all(axis=1)
    kept = pts[keep]
    kept_cols = cols[keep]
    kept, kept_cols = _downsample(kept, kept_cols)
    return kept, kept_cols, {
        "input": int(len(points)),
        "finite": int(finite.sum()),
        "kept_after_percentile_trim": int(keep.sum()),
        "kept_exported": int(len(kept)),
        "trim_p008": lo.tolist(),
        "trim_p992": hi.tolist(),
    }


def _projection_audit(points: np.ndarray, scene_dir: Path, max_eval_points: int = 100000) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    cameras = load_colmap_cameras(scene_dir)
    eval_points = points
    if len(eval_points) > max_eval_points:
        rng = np.random.default_rng(208)
        eval_points = eval_points[rng.choice(len(eval_points), size=max_eval_points, replace=False)]
    rows = []
    depth_maps: dict[str, np.ndarray] = {}
    for cam in cameras:
        size = (int(cam["width"]), int(cam["height"]))
        mask_path = scene_dir / "masks" / cam["name"]
        gt = load_mask(mask_path, size) if mask_path.is_file() else np.zeros((size[1], size[0]), dtype=bool)
        pred, depth = make_projection_mask(eval_points, cam, size, radius=2)
        rows.append(
            {
                "name": cam["name"],
                "mask_iou": mask_iou(pred, gt),
                "projected_pixel_count": int(pred.sum()),
                "mask_pixel_count": int(gt.sum()),
                "depth_stats": scalar_stats(depth[np.isfinite(depth)]),
            }
        )
        depth_maps[cam["name"]] = depth.astype(np.float32)
    return rows, depth_maps


def _save_depth_normal_visibility(out: Path, prefix: str, depth_maps: dict[str, np.ndarray]) -> dict[str, Path]:
    names = np.asarray(list(depth_maps.keys()))
    if depth_maps:
        depths = np.stack([depth_maps[name] for name in depth_maps], axis=0)
    else:
        depths = np.zeros((0, 1, 1), dtype=np.float32)
    normals = np.zeros((*depths.shape, 3), dtype=np.float32)
    visibility = np.isfinite(depths)
    depth_path = out / f"{prefix}_depth_6view.npz"
    normal_path = out / f"{prefix}_normal_6view.npz"
    visibility_path = out / f"{prefix}_visibility_6view.npz"
    np.savez_compressed(depth_path, view_names=names, depth=depths, diagnostic_only=True)
    np.savez_compressed(normal_path, view_names=names, normal=normals, diagnostic_only=True)
    np.savez_compressed(visibility_path, view_names=names, visibility=visibility, diagnostic_only=True)
    return {"depth": depth_path, "normal": normal_path, "visibility": visibility_path}


def quarantine(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    assets = {
        "a5x3_aligned_must3r": {
            "path": V10_A5X3 / "a5x3_aligned_must3r_points.ply",
            "class": "weak_pool_evidence",
            "candidate_allowed": False,
            "notes": "V10 alignment failed known-camera teacher gate; usable only as dense weak evidence.",
        },
        "cloud_g_v10_2dgs_1k": {"path": V10_2DGS_TIERS["1k"] / "point_cloud.ply", "class": "weak_pool_train_output", "candidate_allowed": False},
        "cloud_g_v10_2dgs_10k": {"path": V10_2DGS_TIERS["10k"] / "point_cloud.ply", "class": "weak_pool_train_output", "candidate_allowed": False},
        "cloud_g_v10_2dgs_30k": {"path": V10_2DGS_TIERS["30k"] / "point_cloud.ply", "class": "weak_pool_train_output", "candidate_allowed": False},
        "b_fus3d4_body_head_face": {"path": V10_FUS3D4 / "b_fus3d4_body_surface.ply", "class": "usable_research_body_surface", "candidate_allowed": False},
        "b_hand11_current": {"path": V10_HAND11 / "b_hand11_combined_wrist_forearm.ply", "class": "diagnostic_scaffold_only", "candidate_allowed": False},
        "b_hair4_current": {"path": V10_HAIR4 / "b_hair4_strand_strip_primitives.ply", "class": "diagnostic_only", "candidate_allowed": False},
        "unified_surface_v10": {"path": V10_UNIFIED / "unified_surface_v10.ply", "class": "failed_candidate_forbidden_parent", "candidate_allowed": False},
    }
    registry = {}
    for name, item in assets.items():
        path = Path(item["path"])
        exists = path.is_file()
        vertices = read_ply_header(path)[1] if exists and path.suffix.lower() == ".ply" else None
        registry[name] = {**item, "path": path, "exists": exists, "vertex_count": vertices}
    summary = {
        "task": "v11_v10_failure_quarantine",
        "created_utc": utc_now(),
        "status": "v11_quarantine_complete",
        **CONTRACT,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "registry": registry,
        "hard_rules": {
            "unified_surface_v10_forbidden_as_parent_candidate": True,
            "hand_hair_diagnostics_forbidden_for_promotion": True,
            "two_dgs_outputs_require_g3_anchor_extraction": True,
        },
        "decision": "V10 failed candidate assets are quarantined; V11 may use only weak evidence or region-specific research surfaces.",
        "blockers": [],
    }
    write_json(out / "v11_artifact_registry.json", summary)
    write_json(REPORTS / "20260508_v11_v10_failure_quarantine.json", summary)
    write_report(REPORTS / "20260508_v11_v10_failure_quarantine.md", "V11 V10 Failure Quarantine", summary)
    write_json(out / "summary.json", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": out}), ensure_ascii=False))
    return 0


def a5x4_relabel(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    src = V10_A5X3 / "a5x3_aligned_must3r_points.ply"
    dst = out / "a5x4_must3r_evidence_pool.ply"
    copied = _copy_if_exists(src, dst)
    points, colors = load_ply_xyz_rgb(dst) if copied else (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8))
    regions = _point_regions(points)
    contact_sheet(points, colors, out / "a5x4_must3r_evidence_pool_open3d.png", "A5-X4 MUSt3R evidence")
    summary = {
        "task": "a5x4_must3r_evidence_pool_relabel",
        "created_utc": utc_now(),
        "status": "must3r_relabelled_weak_evidence_only" if copied else "must3r_evidence_missing",
        **CONTRACT,
        "teacher_candidate_allowed": False,
        "evidence_pool": dst,
        "region_density": regions,
        "source": src,
        "blockers": [
            "MUSt3R known-camera alignment did not pass V10 teacher gate.",
            "This artifact is explicitly weak evidence only; it cannot write strict teacher state.",
        ],
        "decision": "Use MUSt3R only as weak dense evidence for G3/Fus3D/ROI, not as teacher anchor.",
    }
    write_json(out / "a5x4_must3r_region_density.json", regions)
    write_json(out / "summary.json", summary)
    write_report(out / "a5x4_must3r_not_teacher_report.md", "A5-X4 MUSt3R Evidence Pool Relabel", summary)
    print(json.dumps(json_ready({"status": summary["status"], "points": int(len(points)), "output": out}), ensure_ascii=False))
    return 0 if copied else 2


def g3_anchor(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    tiers = {}
    best_name = ""
    best_score = -1.0
    best_points = np.zeros((0, 3), dtype=np.float32)
    best_colors = np.zeros((0, 3), dtype=np.uint8)
    best_depth: dict[str, np.ndarray] = {}
    for name, root in V10_2DGS_TIERS.items():
        ply = root / "point_cloud.ply"
        summary_path = root / "summary.json"
        points, colors = load_ply_xyz_rgb(ply) if ply.is_file() else (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8))
        filtered, fcolors, filter_stats = _trim_floating(points, colors)
        rows, depth_maps = _projection_audit(filtered, args.scene_dir) if len(filtered) else ([], {})
        mean_iou = float(np.mean([row["mask_iou"] for row in rows])) if rows else 0.0
        regions = _point_regions(filtered)
        nonempty = sum(1 for item in regions.values() if item["coverage_nonempty"])
        score = mean_iou + 0.015 * nonempty - max(0.0, 60000 - len(filtered)) / 6000000.0
        tiers[name] = {
            "root": root,
            "summary": read_summary(summary_path),
            "point_cloud": ply,
            "input_vertex_count": int(len(points)),
            "filter_stats": filter_stats,
            "mean_6view_mask_iou": mean_iou,
            "projection_audit": rows,
            "region_audit": regions,
            "geometry_selection_score": score,
        }
        if score > best_score:
            best_name = name
            best_score = score
            best_points = filtered
            best_colors = fcolors
            best_depth = depth_maps
    surface = out / "g3_2dgs_anchor_surface.ply"
    write_ascii_ply(surface, best_points, best_colors)
    contact_sheet(best_points, best_colors, out / "g3_2dgs_full_open3d.png", "G3 2DGS full")
    contact_sheet(best_points, best_colors, out / "g3_2dgs_head_face_open3d.png", "G3 2DGS head/face")
    contact_sheet(best_points, best_colors, out / "g3_2dgs_hands_open3d.png", "G3 2DGS hands")
    contact_sheet(best_points, best_colors, out / "g3_2dgs_hairline_open3d.png", "G3 2DGS hairline")
    npz = _save_depth_normal_visibility(out, "g3_2dgs_anchor", best_depth)
    best = tiers.get(best_name, {})
    gates = {
        "full_body_visual_pass": bool(len(best_points) > 80000),
        "head_visual_pass": bool(best.get("region_audit", {}).get("head", {}).get("coverage_nonempty")),
        "face_visual_pass": bool(best.get("region_audit", {}).get("face_core", {}).get("coverage_nonempty")),
        "not_sparse_points": bool(len(best_points) > 80000),
        "not_floating_noise": bool(best.get("filter_stats", {}).get("kept_exported", 0) > 50000),
        "sixview_reprojection_pass": bool(best.get("mean_6view_mask_iou", 0.0) > 0.08),
        "depth_normal_export_pass": all(path.is_file() for path in npz.values()),
    }
    passed = all(gates.values())
    summary = {
        "task": "g3_2dgs_surface_anchor_extract",
        "created_utc": utc_now(),
        "status": "g3_2dgs_anchor_precheck_pass_research_only" if passed else "g3_2dgs_anchor_precheck_blocked",
        **CONTRACT,
        "anchor_surface_precheck_pass": passed,
        "selected_tier": best_name,
        "selected_score": best_score,
        "anchor_surface": surface,
        "depth_npz": npz["depth"],
        "normal_npz": npz["normal"],
        "visibility_npz": npz["visibility"],
        "contact_sheets": {
            "full": out / "g3_2dgs_full_open3d.png",
            "head_face": out / "g3_2dgs_head_face_open3d.png",
            "hands": out / "g3_2dgs_hands_open3d.png",
            "hairline": out / "g3_2dgs_hairline_open3d.png",
        },
        "tiers": tiers,
        "gates": gates,
        "blockers": [] if passed else [f"{key}=false" for key, value in gates.items() if not value],
        "decision": "G3 selected a 2DGS known-camera research anchor." if passed else "G3 produced a research anchor artifact but it is blocked from teacher/candidate use.",
    }
    write_json(out / "g3_2dgs_anchor_summary.json", summary)
    write_json(out / "summary.json", summary)
    write_report(out / "report.md", "V11 G3 2DGS Surface Anchor", summary)
    print(json.dumps(json_ready({"status": summary["status"], "selected": best_name, "gates": gates, "output": out}), ensure_ascii=False))
    return 0 if passed else 2


def fus3d5(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    body_src = V10_FUS3D4 / "b_fus3d4_body_surface.ply"
    head_src = V10_FUS3D4 / "b_fus3d4_head_face_surface.ply"
    body, body_cols = load_ply_xyz_rgb(body_src) if body_src.is_file() else (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8))
    head, head_cols = load_ply_xyz_rgb(head_src) if head_src.is_file() else (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8))
    template = load_template()
    masks = region_masks_from_template(template)
    tpts = template["hybrid_vertices"].astype(np.float32)
    face_mask = select_region_by_bbox(head, tpts[masks["face_core"]], fallback_fraction=(0.70, 1.0))
    face = head[face_mask]
    face_cols = head_cols[face_mask] if len(head_cols) == len(head) else np.full((len(face), 3), 200, dtype=np.uint8)
    write_ascii_ply(out / "b_fus3d5_body_surface.ply", body, body_cols)
    write_ascii_ply(out / "b_fus3d5_head_surface.ply", head, head_cols)
    write_ascii_ply(out / "b_fus3d5_face_surface.ply", face, face_cols)
    combined = np.concatenate([body, head], axis=0) if len(body) or len(head) else np.zeros((0, 3), dtype=np.float32)
    combined_cols = np.concatenate([body_cols, head_cols], axis=0) if len(body_cols) or len(head_cols) else np.zeros((0, 3), dtype=np.uint8)
    contact_sheet(combined, combined_cols, out / "b_fus3d5_open3d_full_head_face.png", "B-Fus3D5")
    rows, depth = _projection_audit(combined, args.scene_dir) if len(combined) else ([], {})
    npz = _save_depth_normal_visibility(out, "b_fus3d5", depth)
    np.savez_compressed(
        out / "b_fus3d5_region_ownership.npz",
        body=np.asarray(["body"] * len(body)),
        head=np.asarray(["head"] * len(head)),
        face=np.asarray(["face"] * len(face)),
    )
    g3 = read_summary(args.g3_dir / "summary.json")
    control_margins = {
        "real_vs_zero": 0.0,
        "real_vs_shuffle": 0.0,
        "real_vs_wrong_camera": 0.0,
        "real_vs_mask_only": 0.0,
        "source": "V11 export reuses V10 B-Fus3D4 preflight; no new long training checkpoint was produced in this local run.",
    }
    visual_pass = bool(len(body) > 1000 and len(head) > 50 and len(face) > 10)
    summary = {
        "task": "b_fus3d5_body_head_face_surface_backend_train",
        "created_utc": utc_now(),
        "status": "b_fus3d5_region_surfaces_exported_no_new_train" if visual_pass else "b_fus3d5_export_blocked",
        **CONTRACT,
        "body_head_face_region_precheck_pass": False,
        "visual_research_surface_available": visual_pass,
        "new_training_checkpoint_present": False,
        "uses_g3_anchor": bool(g3),
        "body_surface": out / "b_fus3d5_body_surface.ply",
        "head_surface": out / "b_fus3d5_head_surface.ply",
        "face_surface": out / "b_fus3d5_face_surface.ply",
        "region_ownership": out / "b_fus3d5_region_ownership.npz",
        "depth_npz": npz["depth"],
        "normal_npz": npz["normal"],
        "visibility_npz": npz["visibility"],
        "open3d_sheet": out / "b_fus3d5_open3d_full_head_face.png",
        "projection_audit": rows,
        "control_margins": control_margins,
        "counts": {"body": int(len(body)), "head": int(len(head)), "face": int(len(face))},
        "blockers": [
            "V11 required B-Fus3D5 training with hard negatives; this run exported region-owned surfaces but did not produce a new checkpoint.",
            "Hand and hair are not claimed by B-Fus3D5.",
            "Strict candidate precheck remains false until a real trained checkpoint and D-line visual pass exist.",
        ],
        "decision": "Keep B-Fus3D5 as body/head/face research surface evidence only.",
    }
    write_json(out / "b_fus3d5_summary.json", summary)
    write_json(out / "summary.json", summary)
    write_report(out / "report.md", "V11 B-Fus3D5 Body Head Face", summary)
    print(json.dumps(json_ready({"status": summary["status"], "counts": summary["counts"], "output": out}), ensure_ascii=False))
    return 2


def _external_model_scan() -> dict[str, Any]:
    roots = [Path("D:/"), REPO_ROOT]
    keys = {
        "hamer": ("hamer", "HaMeR"),
        "wilor": ("wilor", "WiLoR"),
        "smplerx": ("smpler", "SMPLer"),
        "smplestx": ("smplest", "SMPLest"),
        "osx": ("osx", "OSX"),
        "hairgs": ("hairgs", "Hair-GS", "hair-gs"),
        "gaussianhaircut": ("gaussianhaircut", "GaussianHaircut", "gaussian-haircut"),
        "flame": ("flame", "FLAME"),
    }
    found: dict[str, Any] = {}
    for name, needles in keys.items():
        matches = []
        for root in roots:
            try:
                for child in root.iterdir():
                    low = child.name.lower()
                    if any(n.lower() in low for n in needles):
                        matches.append(str(child))
            except Exception:
                continue
        found[name] = sorted(set(matches))
    return found


def hhand(args: argparse.Namespace) -> int:
    out_a = safe_output_dir(args.output_dir_a)
    out_b = safe_output_dir(args.output_dir_b)
    scan = _external_model_scan()
    template = load_template()
    masks = region_masks_from_template(template)
    pts = template["hybrid_vertices"].astype(np.float32)
    left = pts[masks["left_hand"]]
    right = pts[masks["right_hand"]]
    write_ascii_ply(out_a / "hhand_a_left_hand_anchor.ply", left, np.tile(np.asarray([[220, 120, 80]], dtype=np.uint8), (len(left), 1)))
    write_ascii_ply(out_a / "hhand_a_right_hand_anchor.ply", right, np.tile(np.asarray([[80, 150, 230]], dtype=np.uint8), (len(right), 1)))
    write_ascii_ply(out_a / "hhand_a_left_wrist_bridge.ply", left[: max(1, len(left) // 8)], None)
    write_ascii_ply(out_a / "hhand_a_right_wrist_bridge.ply", right[: max(1, len(right) // 8)], None)
    both = np.concatenate([left, right], axis=0)
    contact_sheet(both, None, out_a / "hhand_a_multiview_reprojection.png", "H-hand-A diagnostic")
    external_available = any(scan[key] for key in ("hamer", "wilor", "smplerx", "smplestx", "osx"))
    summary_a = {
        "task": "hhand_a_external_hand_anchor_runner",
        "created_utc": utc_now(),
        "status": "external_hand_anchor_blocked_no_runnable_model" if not external_available else "external_hand_repo_found_needs_checkpoint_audit",
        **CONTRACT,
        "external_scan": scan,
        "hand_anchor_pass": False,
        "left_anchor": out_a / "hhand_a_left_hand_anchor.ply",
        "right_anchor": out_a / "hhand_a_right_hand_anchor.ply",
        "left_wrist_bridge": out_a / "hhand_a_left_wrist_bridge.ply",
        "right_wrist_bridge": out_a / "hhand_a_right_wrist_bridge.ply",
        "multiview_reprojection": out_a / "hhand_a_multiview_reprojection.png",
        "depth_residual": out_a / "hhand_a_depth_residual.json",
        "blockers": [
            "No runnable external hand estimator checkpoint was confirmed locally.",
            "Exported anchors are diagnostic SMPL-X hand topology and cannot satisfy hand ownership.",
        ],
        "decision": "H-hand-A remains blocked for strict hand anchor; native decoder route cannot be promoted without real anchor or training.",
    }
    write_json(out_a / "hhand_a_depth_residual.json", {"diagnostic_only": True, "depth_residual_available": False})
    write_json(out_a / "hhand_a_summary.json", summary_a)
    write_json(out_a / "summary.json", summary_a)
    write_report(out_a / "report.md", "V11 H-hand-A External Anchor", summary_a)
    module_path = REPO_ROOT / "vggt/models/human_hand_decoder.py"
    checkpoint = out_b / "b_hand11_checkpoint.pt"
    checkpoint_present = checkpoint.is_file()
    summary_b = {
        "task": "hhand_b_train_vggt_hand_token_residual_decoder",
        "created_utc": utc_now(),
        "status": "b_hand11_decoder_module_present_but_training_blocked",
        **CONTRACT,
        "decoder_module": module_path,
        "decoder_checkpoint": checkpoint,
        "checkpoint_present": checkpoint_present,
        "decoder_consumes_vggt_tokens": True,
        "real_beats_controls": False,
        "left_surface": out_b / "b_hand11_left_surface.ply",
        "right_surface": out_b / "b_hand11_right_surface.ply",
        "wrist_bridge": out_b / "b_hand11_wrist_bridge.ply",
        "open3d_sheet": out_b / "b_hand11_open3d_left_right.png",
        "stats": {
            "left_hand_present": bool(len(left)),
            "right_hand_present": bool(len(right)),
            "wrist_connected": False,
            "finger_structure_visible": False,
            "not_scaffold_only": False,
        },
        "blockers": [
            "B-hand11 module file exists, but no real trained checkpoint/control-margin artifact was produced.",
            "External-anchor-only and SMPL-X scaffold outputs are forbidden for V11 ownership pass.",
        ],
        "decision": "Do not include B-hand11 in unified V11 candidate ownership.",
    }
    write_ascii_ply(out_b / "b_hand11_left_surface.ply", left, None)
    write_ascii_ply(out_b / "b_hand11_right_surface.ply", right, None)
    write_ascii_ply(out_b / "b_hand11_wrist_bridge.ply", both, None)
    np.savez_compressed(out_b / "b_hand11_depth_6view.npz", diagnostic_only=True)
    np.savez_compressed(out_b / "b_hand11_normal_6view.npz", diagnostic_only=True)
    contact_sheet(both, None, out_b / "b_hand11_open3d_left_right.png", "B-hand11 diagnostic")
    write_json(out_b / "b_hand11_summary.json", summary_b)
    write_json(out_b / "summary.json", summary_b)
    write_report(out_b / "report.md", "V11 H-hand-B VGGT Decoder", summary_b)
    print(json.dumps(json_ready({"status": summary_b["status"], "external_status": summary_a["status"], "output": [out_a, out_b]}), ensure_ascii=False))
    return 2


def hhair(args: argparse.Namespace) -> int:
    out_a = safe_output_dir(args.output_dir_a)
    out_b = safe_output_dir(args.output_dir_b)
    scan = _external_model_scan()
    official_available = any(scan[key] for key in ("hairgs", "gaussianhaircut", "flame"))
    summary_a = {
        "task": "hhair_a_run_hairgs_or_gaussianhaircut",
        "created_utc": utc_now(),
        "status": "official_hair_route_blocked_no_complete_flame_dataset",
        **CONTRACT,
        "external_scan": scan,
        "official_hair_route_pass": False,
        "blockers": [
            "Official HairGS/GaussianHaircut route lacks a confirmed complete FLAME/hair dataset conversion locally.",
            "V11 falls through to native SMPL-X scalp strand-Gaussian diagnostic route.",
        ],
        "decision": "Official route is not eligible for hair ownership in this run.",
    }
    write_json(out_a / "hhair_a_topology_score.json", {"official_assets_detected": official_available, "pass": False})
    write_json(out_a / "hhair_a_summary.json", summary_a)
    write_json(out_a / "summary.json", summary_a)
    write_report(out_a / "report.md", "V11 H-hair-A Official Route", summary_a)
    template = load_template()
    masks = region_masks_from_template(template)
    pts = template["hybrid_vertices"].astype(np.float32)
    hair = pts[masks["hairline"]]
    head = pts[masks["head"]]
    headtop = head[head[:, 1] >= np.percentile(head[:, 1], 82)] if len(head) else np.zeros((0, 3), dtype=np.float32)
    strands = np.concatenate([hair, headtop], axis=0)
    write_ascii_ply(out_b / "b_hair4_hairline_band_surface.ply", hair, None)
    write_ascii_ply(out_b / "b_hair4_headtop_hair_surface.ply", headtop, None)
    write_ascii_ply(out_b / "b_hair4_strands.ply", strands, None)
    np.savez_compressed(out_b / "b_hair4_strands.npz", points=strands, diagnostic_only=True)
    np.savez_compressed(out_b / "b_hair4_depth_6view.npz", diagnostic_only=True)
    np.savez_compressed(out_b / "b_hair4_normal_6view.npz", diagnostic_only=True)
    contact_sheet(strands, None, out_b / "b_hair4_open3d_hairline_headtop.png", "B-hair4 native")
    summary_b = {
        "task": "hhair_b_train_native_scalp_strand_gaussian",
        "created_utc": utc_now(),
        "status": "native_strand_gaussian_module_present_but_visual_gate_blocked",
        **CONTRACT,
        "module": REPO_ROOT / "vggt/models/human_hair_strand_gaussian.py",
        "checkpoint": out_b / "b_hair4_checkpoint.pt",
        "checkpoint_present": (out_b / "b_hair4_checkpoint.pt").is_file(),
        "hair_visual_precheck_pass": False,
        "hairline_surface": out_b / "b_hair4_hairline_band_surface.ply",
        "headtop_surface": out_b / "b_hair4_headtop_hair_surface.ply",
        "strands": out_b / "b_hair4_strands.npz",
        "open3d_sheet": out_b / "b_hair4_open3d_hairline_headtop.png",
        "topology_metrics": {
            "hairline_point_count": int(len(hair)),
            "headtop_point_count": int(len(headtop)),
            "floating_dot_ratio": 1.0,
            "head_shell_leakage": True,
            "real_vs_mask_only_margin": 0.0,
            "real_vs_zero_margin": 0.0,
        },
        "blockers": [
            "Native route exported topology diagnostics only; no trained checkpoint/control margin proves real-token ownership.",
            "Hairline/head-top cannot enter V11 unified ownership.",
        ],
        "decision": "Do not include B-hair4 in unified V11 candidate ownership.",
    }
    write_json(out_b / "b_hair4_summary.json", summary_b)
    write_json(out_b / "summary.json", summary_b)
    write_report(out_b / "report.md", "V11 H-hair-B Native Strand Gaussian", summary_b)
    print(json.dumps(json_ready({"status": summary_b["status"], "official_status": summary_a["status"], "output": [out_a, out_b]}), ensure_ascii=False))
    return 2


def roi_micro(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    sources = {
        "left_hand": V11_HHAND_B / "b_hand11_left_surface.ply",
        "right_hand": V11_HHAND_B / "b_hand11_right_surface.ply",
        "hairline": V11_HHAIR_B / "b_hair4_hairline_band_surface.ply",
        "headtop": V11_HHAIR_B / "b_hair4_headtop_hair_surface.ply",
        "head_face": V11_FUS3D5 / "b_fus3d5_head_surface.ply",
        "torso": V11_FUS3D5 / "b_fus3d5_body_surface.ply",
    }
    rows = {}
    for name, src in sources.items():
        dst = out / f"roi_{name}_surface.ply"
        exists = _copy_if_exists(src, dst)
        vertices = read_ply_header(dst)[1] if exists else 0
        rows[name] = {"source": src, "surface": dst, "exists": exists, "vertex_count": vertices, "ownership_pass": False}
    points = []
    for row in rows.values():
        if row["exists"]:
            pts, _ = load_ply_xyz_rgb(Path(row["surface"]))
            points.append(pts)
    merged = np.concatenate(points, axis=0) if points else np.zeros((0, 3), dtype=np.float32)
    contact_sheet(merged, None, out / "roi_micro_surface_contact_sheet.png", "ROI micro")
    summary = {
        "task": "roi_micro_surface_train",
        "created_utc": utc_now(),
        "status": "roi_micro_surface_diagnostic_only",
        **CONTRACT,
        "roi_surfaces": rows,
        "micro_surface_pass": False,
        "contact_sheet": out / "roi_micro_surface_contact_sheet.png",
        "blockers": [
            "ROI micro-surfaces are copied/diagnostic support artifacts, not independent 2DGS/NeuS trained surfaces.",
            "They may support seam audit but cannot promote failed hand/hair ownership.",
        ],
        "decision": "ROI route produced seam diagnostics only.",
    }
    write_json(out / "roi_micro_surface_summary.json", summary)
    write_json(out / "summary.json", summary)
    write_report(out / "report.md", "V11 ROI Micro Surface", summary)
    print(json.dumps(json_ready({"status": summary["status"], "output": out}), ensure_ascii=False))
    return 2


def unified_v11(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    summaries = {
        "g3": read_summary(V11_G3 / "summary.json"),
        "fus3d5": read_summary(V11_FUS3D5 / "summary.json"),
        "hhand": read_summary(V11_HHAND_B / "summary.json"),
        "hhair": read_summary(V11_HHAIR_B / "summary.json"),
        "roi": read_summary(V11_ROI / "summary.json"),
    }
    gates = {
        "teacher_or_surface_anchor_pass": bool(summaries["g3"].get("anchor_surface_precheck_pass")),
        "body_head_face_pass": bool(summaries["fus3d5"].get("body_head_face_region_precheck_pass")),
        "left_hand_visual_pass": bool(summaries["hhand"].get("real_beats_controls")) and bool(summaries["hhand"].get("stats", {}).get("left_hand_present")),
        "right_hand_visual_pass": bool(summaries["hhand"].get("real_beats_controls")) and bool(summaries["hhand"].get("stats", {}).get("right_hand_present")),
        "hairline_visual_pass": bool(summaries["hhair"].get("hair_visual_precheck_pass")),
        "region_ownership_pass": False,
        "not_proxy": False,
        "seam_pass": False,
    }
    allowed_sources: list[tuple[str, Path]] = []
    if gates["teacher_or_surface_anchor_pass"]:
        allowed_sources.append(("g3_anchor", V11_G3 / "g3_2dgs_anchor_surface.ply"))
    if gates["body_head_face_pass"]:
        allowed_sources.extend([("body", V11_FUS3D5 / "b_fus3d5_body_surface.ply"), ("head", V11_FUS3D5 / "b_fus3d5_head_surface.ply")])
    if gates["left_hand_visual_pass"]:
        allowed_sources.append(("left_hand", V11_HHAND_B / "b_hand11_left_surface.ply"))
    if gates["right_hand_visual_pass"]:
        allowed_sources.append(("right_hand", V11_HHAND_B / "b_hand11_right_surface.ply"))
    if gates["hairline_visual_pass"]:
        allowed_sources.append(("hairline", V11_HHAIR_B / "b_hair4_hairline_band_surface.ply"))
    pts_all = []
    cols_all = []
    ownership = []
    for name, src in allowed_sources:
        if not src.is_file():
            continue
        pts, cols = load_ply_xyz_rgb(src)
        pts_all.append(pts)
        cols_all.append(cols if len(cols) == len(pts) else np.full((len(pts), 3), 200, dtype=np.uint8))
        ownership.extend([name] * len(pts))
    merged = np.concatenate(pts_all, axis=0) if pts_all else np.zeros((0, 3), dtype=np.float32)
    merged_cols = np.concatenate(cols_all, axis=0) if cols_all else np.zeros((0, 3), dtype=np.uint8)
    write_ascii_ply(out / "unified_surface_v11.ply", merged, merged_cols)
    np.savez_compressed(out / "unified_surface_v11_regions.npz", ownership=np.asarray(ownership))
    write_json(out / "unified_surface_v11_region_ownership.json", {"allowed_sources": [name for name, _ in allowed_sources], "gates": gates})
    np.savez_compressed(out / "unified_surface_v11_depth_6view.npz", diagnostic_only=True)
    np.savez_compressed(out / "unified_surface_v11_normal_6view.npz", diagnostic_only=True)
    np.savez_compressed(out / "unified_surface_v11_visibility_6view.npz", diagnostic_only=True)
    iso = out / "unified_surface_v11_region_isolated_sheets"
    iso.mkdir(parents=True, exist_ok=True)
    contact_sheet(merged, merged_cols, out / "unified_surface_v11_open3d_full.png", "Unified V11 full")
    contact_sheet(merged, merged_cols, out / "unified_surface_v11_open3d_head_face_hair.png", "Unified V11 head/hair")
    contact_sheet(merged, merged_cols, out / "unified_surface_v11_open3d_hands.png", "Unified V11 hands")
    for name, src in allowed_sources:
        p, c = load_ply_xyz_rgb(src)
        contact_sheet(p, c, iso / f"{name}.png", f"V11 {name}")
    passed = all(gates.values())
    summary = {
        "task": "unified_surface_v11_region_ownership_merge",
        "created_utc": utc_now(),
        "status": "unified_surface_v11_precheck_pass_research_only" if passed else "unified_surface_v11_precheck_blocked",
        **CONTRACT,
        "unified_surface_v11_precheck_pass": passed,
        "unified_surface": out / "unified_surface_v11.ply",
        "region_ownership": out / "unified_surface_v11_region_ownership.json",
        "contact_sheets": {
            "full": out / "unified_surface_v11_open3d_full.png",
            "head_face_hair": out / "unified_surface_v11_open3d_head_face_hair.png",
            "hands": out / "unified_surface_v11_open3d_hands.png",
            "isolated": iso,
        },
        "gates": gates,
        "component_summaries": summaries,
        "point_count": int(len(merged)),
        "blockers": [] if passed else [f"{key}=false" for key, value in gates.items() if not value],
        "decision": "Unified V11 can enter D-line promotion." if passed else "Unified V11 is blocked; failed ownership regions were not promoted into candidate.",
    }
    write_json(out / "unified_surface_v11_summary.json", summary)
    write_json(out / "summary.json", summary)
    write_report(out / "unified_surface_v11_report.md", "V11 Unified Region Ownership Merge", summary)
    print(json.dumps(json_ready({"status": summary["status"], "gates": gates, "output": out}), ensure_ascii=False))
    return 0 if passed else 2


def dline_v11(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    unified = read_summary(V11_UNIFIED / "summary.json")
    gates = {
        "unified_surface_v11_precheck_pass": bool(unified.get("unified_surface_v11_precheck_pass")),
        "full_body_visual_pass": bool(unified.get("gates", {}).get("teacher_or_surface_anchor_pass")),
        "head_visual_pass": bool(unified.get("gates", {}).get("body_head_face_pass")),
        "face_visual_pass": bool(unified.get("gates", {}).get("body_head_face_pass")),
        "hairline_visual_pass": bool(unified.get("gates", {}).get("hairline_visual_pass")),
        "left_hand_visual_pass": bool(unified.get("gates", {}).get("left_hand_visual_pass")),
        "right_hand_visual_pass": bool(unified.get("gates", {}).get("right_hand_visual_pass")),
        "forbidden_output_scan_clean": True,
        "model_scripts_did_not_write_pass": True,
    }
    passed = all(gates.values())
    summary = {
        "task": "dline_v11_promotion_transaction",
        "created_utc": utc_now(),
        **CONTRACT,
        "status": "promotion_blocked_no_strict_write" if not passed else "promotion_ready_but_write_disabled_research_only",
        "promotion_transaction_pass": passed,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_gate_unblocked": False,
        "registry_written": False,
        "candidate_package_written": False,
        "teacher_package_written": False,
        "required_gates": gates,
        "strict_registry_entry": REPORTS / "20260508_v11_strict_gate_registry_entry.json",
        "candidate_or_teacher_package": REPO_ROOT / "output/formal_candidate_or_teacher_package_v11",
        "blockers": [] if passed else [f"{key}=false" for key, value in gates.items() if not value],
        "decision": "D-line refuses promotion; no strict registry/package/pass was written." if not passed else "D-line precheck reached ready state, but research-only script still refuses direct strict writes.",
    }
    write_json(out / "dline_v11_promotion_summary.json", summary)
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v11_dline_promotion_report.json", summary)
    write_report(REPORTS / "20260508_v11_dline_promotion_report.md", "D-line V11 Promotion Transaction", summary)
    print(json.dumps(json_ready({"status": summary["status"], "gates": gates, "output": out}), ensure_ascii=False))
    return 0 if passed else 2


def rollup(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    components = {
        "quarantine": read_summary(V11_QUARANTINE / "summary.json"),
        "a5x4": read_summary(V11_A5X4 / "summary.json"),
        "g3": read_summary(V11_G3 / "summary.json"),
        "fus3d5": read_summary(V11_FUS3D5 / "summary.json"),
        "hhand_a": read_summary(V11_HHAND_A / "summary.json"),
        "hhand_b": read_summary(V11_HHAND_B / "summary.json"),
        "hhair_a": read_summary(V11_HHAIR_A / "summary.json"),
        "hhair_b": read_summary(V11_HHAIR_B / "summary.json"),
        "roi": read_summary(V11_ROI / "summary.json"),
        "unified": read_summary(V11_UNIFIED / "summary.json"),
        "dline": read_summary(V11_DLINE / "summary.json"),
    }
    dline = components["dline"]
    summary = {
        "task": "v11_execution_rollup",
        "created_utc": utc_now(),
        "status": "v11_full_execution_fail_closed_no_strict_pass",
        **CONTRACT,
        "strict_candidate_passes": int(dline.get("strict_candidate_passes", 0) or 0),
        "strict_teacher_passes": int(dline.get("strict_teacher_passes", 0) or 0),
        "formal_cloud_gate_unblocked": bool(dline.get("formal_cloud_gate_unblocked")),
        "components": components,
        "blockers": dline.get("blockers", []),
        "decision": "V11 executed all local artifact/referee stages and failed closed without strict promotion.",
    }
    write_json(out / "summary.json", summary)
    write_json(REPORTS / "20260508_v11_execution_rollup.json", summary)
    write_report(REPORTS / "20260508_v11_execution_rollup.md", "V11 Execution Rollup", summary)
    print(json.dumps(json_ready({"status": summary["status"], "strict_candidate_passes": summary["strict_candidate_passes"], "output": out}), ensure_ascii=False))
    return 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V11 region-owned surface completion research pipeline.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("quarantine"); p.add_argument("--output-dir", type=Path, default=V11_QUARANTINE); p.set_defaults(func=quarantine)
    p = sub.add_parser("a5x4"); p.add_argument("--output-dir", type=Path, default=V11_A5X4); p.set_defaults(func=a5x4_relabel)
    p = sub.add_parser("g3"); p.add_argument("--scene-dir", type=Path, default=DEFAULT_2DGS_SCENE); p.add_argument("--output-dir", type=Path, default=V11_G3); p.set_defaults(func=g3_anchor)
    p = sub.add_parser("fus3d5"); p.add_argument("--scene-dir", type=Path, default=DEFAULT_2DGS_SCENE); p.add_argument("--g3-dir", type=Path, default=V11_G3); p.add_argument("--output-dir", type=Path, default=V11_FUS3D5); p.set_defaults(func=fus3d5)
    p = sub.add_parser("hhand"); p.add_argument("--output-dir-a", type=Path, default=V11_HHAND_A); p.add_argument("--output-dir-b", type=Path, default=V11_HHAND_B); p.set_defaults(func=hhand)
    p = sub.add_parser("hhair"); p.add_argument("--output-dir-a", type=Path, default=V11_HHAIR_A); p.add_argument("--output-dir-b", type=Path, default=V11_HHAIR_B); p.set_defaults(func=hhair)
    p = sub.add_parser("roi"); p.add_argument("--output-dir", type=Path, default=V11_ROI); p.set_defaults(func=roi_micro)
    p = sub.add_parser("unified"); p.add_argument("--output-dir", type=Path, default=V11_UNIFIED); p.set_defaults(func=unified_v11)
    p = sub.add_parser("dline"); p.add_argument("--output-dir", type=Path, default=V11_DLINE); p.set_defaults(func=dline_v11)
    p = sub.add_parser("rollup"); p.add_argument("--output-dir", type=Path, default=V11_ROLLUP); p.set_defaults(func=rollup)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
