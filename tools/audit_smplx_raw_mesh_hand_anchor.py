from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from audit_fullbody_hand_integrity import (  # noqa: E402
    connected_component_stats_2d,
    create_hand_landmarker,
    hand_risk_mask,
    mediapipe_hand_mask,
    point_box_spatial_stats,
)
from audit_smplx_weak_anchor import save_overlay, vertical_band_stats  # noqa: E402
from normal_line_multiview_eval import build_roi_masks, load_scene_view  # noqa: E402
from prepare_4k4d_prior_training_case import (  # noqa: E402
    align_intrinsics_for_scene_view,
    load_optional_annotation_payload,
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
    resolve_smplx_model_dir,
)
from tools.smplx_numpy import (  # noqa: E402
    forward_smplx_mesh,
    load_smplx_model,
    rasterize_world_mesh,
    resolve_smplx_model_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit raw SMPL-X mesh rasterization as a weak hand/body anchor. "
            "Unlike scene-local dense prior maps, this skips silhouette KNN fill "
            "and separately rasterizes hand/wrist mesh faces."
        )
    )
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--smplx-model-dir", type=Path)
    parser.add_argument("--smplx-gender", choices=("neutral", "female", "male"), default="neutral")
    parser.add_argument("--hand-joints", default="20,21,25-54")
    parser.add_argument("--min-body-views", type=int, default=4)
    parser.add_argument("--min-body-visible-ratio", type=float, default=0.35)
    parser.add_argument("--min-body-band-visible-ratio", type=float, default=0.02)
    parser.add_argument("--max-body-components", type=int, default=6)
    parser.add_argument("--min-body-largest-component-ratio", type=float, default=0.80)
    parser.add_argument("--min-hand-components", type=int, default=2)
    parser.add_argument("--min-hand-visible-ratio", type=float, default=0.20)
    parser.add_argument("--min-hand-visible-pixels", type=int, default=24)
    parser.add_argument("--max-hand-box-3d-extent", type=float, default=0.30)
    parser.add_argument("--max-hand-box-depth-range", type=float, default=0.18)
    parser.add_argument(
        "--hand-landmarker-model",
        default=str(REPO_ROOT / "external_models" / "hand_landmarker.task"),
    )
    parser.add_argument("--disable-mediapipe-hands", action="store_true")
    parser.add_argument("--hand-box-pad", type=int, default=24)
    parser.add_argument("--vertical-bins", type=int, default=10)
    return parser.parse_args()


def parse_joint_ids(text: str) -> set[int]:
    out: set[int] = set()
    for piece in str(text).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            left, right = piece.split("-", 1)
            out.update(range(int(left), int(right) + 1))
        else:
            out.add(int(piece))
    return out


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def homogeneous(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = matrix
        return out
    raise ValueError(f"Expected 3x4 or 4x4 matrix, got {matrix.shape}")


def render_mask_and_points(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    world_to_cam: np.ndarray,
    intrinsic: np.ndarray,
    image_hw: tuple[int, int],
    silhouette: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    world_to_cam = homogeneous(world_to_cam)
    rotation = world_to_cam[:3, :3].astype(np.float32)
    translation = world_to_cam[:3, 3].astype(np.float32)
    vertices_cam = vertices_world.astype(np.float32) @ rotation.T + translation[None, :]
    _, point_map_cam, _, raster_mask, meta = rasterize_world_mesh(
        world_vertices=vertices_cam,
        faces=faces.astype(np.int32),
        world_to_cam=np.eye(4, dtype=np.float32),
        intrinsic=intrinsic,
        image_hw=image_hw,
        silhouette_mask=silhouette,
        fill_knn=0,
        return_raster_mask=True,
    )
    return raster_mask.astype(bool), point_map_cam.astype(np.float32), meta


def main() -> int:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_manifest = recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir))
    dataset_root = Path(args.dataset_root or scene_manifest["dataset_root"]).expanduser()
    smplx_model_dir = resolve_smplx_model_dir(None if args.smplx_model_dir is None else str(args.smplx_model_dir))
    if smplx_model_dir is None:
        raise FileNotFoundError("Could not resolve SMPL-X model dir; pass --smplx-model-dir.")
    model_path = resolve_smplx_model_path(smplx_model_dir, args.smplx_gender)
    smplx_params, _ = load_optional_annotation_payload(scene_manifest, dataset_root, args.subset_name)
    if not smplx_params:
        raise ValueError("Scene annotations do not provide SMPL-X parameters.")

    mesh = forward_smplx_mesh(
        model_path=model_path,
        betas=smplx_params["betas"],
        expression=smplx_params.get("expression"),
        fullpose=smplx_params["fullpose"],
        transl=smplx_params.get("transl"),
        scale=smplx_params.get("scale", 1.0),
    )
    vertices = mesh["vertices"].astype(np.float32)
    faces = mesh["faces"].astype(np.int32)
    model = load_smplx_model(model_path)
    dominant_joint = np.asarray(model["weights"], dtype=np.float32).argmax(axis=1)
    hand_joints = parse_joint_ids(args.hand_joints)
    hand_vertex_mask = np.isin(dominant_joint, np.array(sorted(hand_joints), dtype=np.int64))
    hand_faces = faces[np.any(hand_vertex_mask[faces], axis=1)]

    camera_params, camera_source = resolve_scene_camera_params(scene_manifest, dataset_root, args.subset_name)
    detector = create_hand_landmarker(Path(args.hand_landmarker_model), bool(args.disable_mediapipe_hands))

    per_view: dict[str, Any] = {}
    body_views_ok = 0
    hand_views_ok = 0
    hand_candidate_views = 0
    compact_hand_views = 0
    implausible_hand_boxes = 0
    view_count = len(scene_manifest["exported_views"])
    for view_idx, view in enumerate(scene_manifest["exported_views"]):
        scene = load_scene_view(scene_dir, view_idx, (518, 518))
        height, width = scene.mask.shape
        camera_id = str(view["camera_id"]).zfill(2)
        cam = camera_params[camera_id]
        intrinsic = align_intrinsics_for_scene_view(cam["intrinsic"], view, target_size=height)
        body_mask = scene.mask.astype(bool)

        body_raster, _, body_meta = render_mask_and_points(
            vertices,
            faces,
            cam["world_to_cam"],
            intrinsic,
            (height, width),
            body_mask,
        )
        hand_raster, hand_points_cam, hand_meta = render_mask_and_points(
            vertices,
            hand_faces,
            cam["world_to_cam"],
            intrinsic,
            (height, width),
            body_mask,
        )

        rois = build_roi_masks(body_mask)
        mp_hand = mediapipe_hand_mask(scene.rgb, body_mask, detector, pad=int(args.hand_box_pad))
        if mp_hand is None:
            hand_support, hand_summary = hand_risk_mask(scene.rgb, body_mask)
        else:
            hand_support, hand_summary = mp_hand
        hand_visible = hand_raster & hand_support
        hand_support_pixels = int(hand_support.sum())
        hand_visible_pixels = int(hand_visible.sum())
        hand_visible_ratio = float(hand_visible_pixels / max(hand_support_pixels, 1)) if hand_support_pixels else 0.0
        hand_support_body_ratio = float(hand_support_pixels / max(int(body_mask.sum()), 1)) if hand_support_pixels else 0.0
        hand_components = connected_component_stats_2d(hand_visible)
        blocked = rois["head"] | rois["face"]
        hand_box_stats: list[dict[str, Any]] = []
        for box in hand_summary.get("boxes_xyxy", []) or []:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            x0, y0, x1, y1 = [int(value) for value in box]
            x0 = max(0, min(width, x0))
            x1 = max(0, min(width, x1))
            y0 = max(0, min(height, y0))
            y1 = max(0, min(height, y1))
            box_mask = np.zeros_like(body_mask, dtype=bool)
            if x1 > x0 and y1 > y0:
                box_mask[y0:y1, x0:x1] = True
            box_visible = hand_raster & box_mask & body_mask & ~blocked
            spatial = point_box_spatial_stats(hand_points_cam[box_visible])
            box_ok = bool(
                int(spatial["points"]) >= int(args.min_hand_visible_pixels)
                and float(spatial["max_extent"]) <= float(args.max_hand_box_3d_extent)
                and float(spatial["depth_range"]) <= float(args.max_hand_box_depth_range)
            )
            hand_box_stats.append(
                {
                    "box_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                    "visible_pixels": int(box_visible.sum()),
                    "spatial": spatial,
                    "gate_ok": box_ok,
                }
            )
        hand_box_ok = bool(
            hand_summary.get("roi_source") == "mediapipe_hand_landmarker"
            and hand_box_stats
            and all(row.get("gate_ok") for row in hand_box_stats)
        )
        hand_ok = bool(
            hand_support_pixels > 0
            and hand_summary.get("roi_source") == "mediapipe_hand_landmarker"
            and hand_visible_ratio >= float(args.min_hand_visible_ratio)
            and hand_visible_pixels >= int(args.min_hand_visible_pixels)
            and hand_box_ok
        )
        hand_candidate_views += int(hand_support_pixels > 0)
        hand_views_ok += int(hand_ok)
        compact_hand_views += int(hand_box_ok)
        implausible_hand_boxes += int(sum(1 for row in hand_box_stats if not row.get("gate_ok")))

        body_components = connected_component_stats_2d(body_raster & body_mask)
        bands, min_band = vertical_band_stats(body_mask, body_raster & body_mask, int(args.vertical_bins))
        body_visible_ratio = float((body_raster & body_mask).sum() / max(int(body_mask.sum()), 1))
        body_ok = bool(
            body_visible_ratio >= float(args.min_body_visible_ratio)
            and min_band >= float(args.min_body_band_visible_ratio)
            and int(body_components.get("components", 0)) <= int(args.max_body_components)
            and float(body_components.get("largest_component_ratio", 0.0))
            >= float(args.min_body_largest_component_ratio)
        )
        body_views_ok += int(body_ok)

        save_overlay(output_dir / f"view_{view_idx:02d}_raw_smplx_hand_anchor_overlay.png", scene.rgb, body_mask, body_raster & body_mask, hand_support)
        per_view[str(view_idx)] = {
            "camera_id": camera_id,
            "body_pixels": int(body_mask.sum()),
            "raw_body_raster_pixels": int((body_raster & body_mask).sum()),
            "raw_body_visible_ratio": body_visible_ratio,
            "raw_body_components": body_components,
            "raw_body_min_vertical_band_ratio": min_band,
            "raw_body_gate_ok": body_ok,
            "raw_body_raster_meta": body_meta,
            "raw_hand_raster_pixels": int((hand_raster & body_mask).sum()),
            "raw_hand_raster_meta": hand_meta,
            "head_raw_body_visible_ratio": float((body_raster & rois["head"]).sum() / max(int(rois["head"].sum()), 1)),
            "face_raw_body_visible_ratio": float((body_raster & rois["face"]).sum() / max(int(rois["face"].sum()), 1)),
            "vertical_bands": bands,
            "hand_risk": {
                **hand_summary,
                "raw_hand_visible_pixels": hand_visible_pixels,
                "raw_hand_visible_ratio": hand_visible_ratio,
                "support_body_ratio": hand_support_body_ratio,
                "visible_components": hand_components,
                "hand_box_3d": {
                    "boxes": hand_box_stats,
                    "boxes_ok": hand_box_ok,
                    "implausible_boxes": int(sum(1 for row in hand_box_stats if not row.get("gate_ok"))),
                },
                "gate_ok": hand_ok,
            },
            "overlay": str(output_dir / f"view_{view_idx:02d}_raw_smplx_hand_anchor_overlay.png"),
        }

    if detector is not None:
        detector.close()

    body_gate = {
        "views_passing_body_anchor": int(body_views_ok),
        "min_body_views": int(args.min_body_views),
        "pass": bool(body_views_ok >= int(args.min_body_views)),
    }
    hand_gate = {
        "eligible_views_with_hand_candidates": int(hand_candidate_views),
        "views_passing_raw_hand_anchor": int(hand_views_ok),
        "views_with_compact_3d_hand_boxes": int(compact_hand_views),
        "implausible_hand_boxes": int(implausible_hand_boxes),
        "min_hand_components": int(args.min_hand_components),
        "pass": bool(hand_views_ok >= int(args.min_hand_components)),
    }
    summary = {
        "task": "raw_smplx_mesh_hand_anchor_preflight",
        "truthful_status": "preflight_only_not_candidate_not_face_teacher",
        "scene_dir": str(scene_dir),
        "output_dir": str(output_dir),
        "camera_source": camera_source,
        "smplx_model_path": str(model_path),
        "hand_joints": sorted(hand_joints),
        "hand_vertex_count": int(hand_vertex_mask.sum()),
        "hand_face_count": int(hand_faces.shape[0]),
        "body_gate": body_gate,
        "hand_gate": hand_gate,
        "pass": bool(body_gate["pass"] and hand_gate["pass"]),
        "per_view": per_view,
        "notes": [
            "Raw mesh rasterization skips silhouette KNN fill.",
            "SMPL-X remains a weak topology anchor only; it is not a face, hair, clothing, or skirt teacher.",
            "A pass here is only a precondition for possible weak body/hand rehearsal, not a candidate pass.",
        ],
    }
    (output_dir / "raw_smplx_mesh_hand_anchor_summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
