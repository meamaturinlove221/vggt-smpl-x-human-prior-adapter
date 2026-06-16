from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for root in (REPO_ROOT, TOOLS_DIR):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from audit_fullbody_hand_integrity import (  # noqa: E402
    create_hand_landmarker,
    hand_risk_mask,
    mediapipe_hand_mask,
    point_box_spatial_stats,
)
from audit_smplx_raw_mesh_hand_anchor import (  # noqa: E402
    parse_joint_ids,
    render_mask_and_points,
    select_faces_for_joints,
)
from normal_line_multiview_eval import build_roi_masks, load_scene_view  # noqa: E402
from prepare_4k4d_prior_training_case import (  # noqa: E402
    align_intrinsics_for_scene_view,
    load_optional_annotation_payload,
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
    resolve_smplx_model_dir,
)
from tools.dna_4k4d import normalize_camera_id  # noqa: E402
from tools.smplx_numpy import forward_smplx_mesh, load_smplx_model, resolve_smplx_model_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local 4K4D pseudo-training case whose prior_depths/prior_points "
            "come only from raw SMPL-X body and per-box best-side hand rasterization. "
            "Head, face, hairline, and ear-band masks are protected; the raw SMPL-X "
            "anchor is a weak full-body/hand stabilizer, not a face teacher."
        )
    )
    parser.add_argument("--template-case-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--smplx-model-dir", type=Path)
    parser.add_argument("--smplx-gender", choices=("neutral", "female", "male"), default="neutral")
    parser.add_argument("--hand-landmarker-model", default=str(REPO_ROOT / "external_models" / "hand_landmarker.task"))
    parser.add_argument("--disable-mediapipe-hands", action="store_true")
    parser.add_argument("--hand-box-pad", type=int, default=24)
    parser.add_argument("--hand-face-mode", choices=("any", "all"), default="any")
    parser.add_argument("--left-hand-joints", default="20,25-39")
    parser.add_argument("--right-hand-joints", default="21,40-54")
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--max-cam-delta", type=float, default=0.24)
    parser.add_argument("--min-align-pixels", type=int, default=512)
    parser.add_argument("--max-correspondences", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **payload)


def closed_form_inverse_se3(extrinsic: np.ndarray) -> np.ndarray:
    matrix = np.asarray(extrinsic, dtype=np.float32)
    if matrix.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = matrix
        matrix = out
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float32)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def camera_to_world(points_cam: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    cam_to_world = closed_form_inverse_se3(extrinsic)
    rotation = cam_to_world[:3, :3]
    translation = cam_to_world[:3, 3]
    return (points_cam @ rotation.T + translation[None, None, :]).astype(np.float32)


def estimate_umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if source.shape[0] < 16:
        raise RuntimeError(f"Need at least 16 correspondences, got {source.shape[0]}")
    source = source.astype(np.float64)
    target = target.astype(np.float64)
    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)
    src0 = source - mu_source
    tgt0 = target - mu_target
    cov = (tgt0.T @ src0) / float(source.shape[0])
    u_mat, singular_values, vt_mat = np.linalg.svd(cov)
    d = np.eye(3, dtype=np.float64)
    if np.linalg.det(u_mat @ vt_mat) < 0:
        d[-1, -1] = -1.0
    rotation = u_mat @ d @ vt_mat
    variance = float(np.sum(src0 * src0) / max(source.shape[0], 1))
    scale = float(np.trace(np.diag(singular_values) @ d) / max(variance, 1e-12))
    translation = mu_target - scale * (rotation @ mu_source)
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, transform: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = transform
    flat = points.reshape(-1, 3).astype(np.float64)
    out = scale * (flat @ rotation.T) + translation[None, :]
    return out.reshape(points.shape).astype(np.float32)


def robust_similarity(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_correspondences: int,
    seed: int,
) -> tuple[tuple[float, np.ndarray, np.ndarray], np.ndarray]:
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[valid]
    target = target[valid]
    if source.shape[0] > int(max_correspondences) > 0:
        rng = np.random.default_rng(seed)
        keep_idx = rng.choice(source.shape[0], size=int(max_correspondences), replace=False)
        source = source[keep_idx]
        target = target[keep_idx]
    transform = estimate_umeyama(source, target)
    residual = np.linalg.norm(apply_similarity(source.reshape(1, -1, 3), transform).reshape(-1, 3) - target, axis=1)
    inlier = residual <= float(np.percentile(residual, 80.0))
    if int(inlier.sum()) >= 16:
        transform = estimate_umeyama(source[inlier], target[inlier])
        residual = np.linalg.norm(
            apply_similarity(source.reshape(1, -1, 3), transform).reshape(-1, 3) - target,
            axis=1,
        )
    return transform, residual


def protect_mask_from_targets(targets: dict[str, np.ndarray], view_idx: int, scene_mask: np.ndarray) -> np.ndarray:
    protected = np.zeros_like(scene_mask, dtype=bool)
    for key in ("head_roi_mask", "face_roi_mask", "hairline_mask", "ear_band_mask"):
        if key in targets:
            protected |= np.asarray(targets[key][view_idx], dtype=bool)
    if not protected.any():
        rois = build_roi_masks(scene_mask.astype(bool))
        protected |= rois["head"] | rois["face"]
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(protected.astype(np.uint8), kernel, iterations=1).astype(bool) & scene_mask


def draw_overlay(path: Path, rgb: np.ndarray, body: np.ndarray, hand: np.ndarray, protected: np.ndarray, text: str) -> None:
    out = rgb.astype(np.float32).copy()
    out[protected] = out[protected] * 0.58 + np.asarray([255, 230, 0], dtype=np.float32) * 0.42
    out[body] = out[body] * 0.45 + np.asarray([40, 120, 255], dtype=np.float32) * 0.55
    out[hand] = out[hand] * 0.35 + np.asarray([255, 125, 20], dtype=np.float32) * 0.65
    image = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), text, fill=(0, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def build_box_mask(shape: tuple[int, int], box: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = [int(v) for v in box]
    out = np.zeros(shape, dtype=bool)
    out[max(0, y0) : min(shape[0], y1), max(0, x0) : min(shape[1], x1)] = True
    return out


def main() -> int:
    args = parse_args()
    template_root = args.template_case_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} is not empty; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    template_manifest = json.loads((template_root / "case_manifest.json").read_text(encoding="utf-8"))
    inputs = load_npz(template_root / template_manifest.get("inputs_npz", "inputs.npz"))
    targets = load_npz(template_root / template_manifest.get("targets_npz", "targets.npz"))

    scene_dir = Path(template_manifest["scene_dir"]).expanduser().resolve()
    scene_manifest = recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir))
    dataset_root = Path(args.dataset_root or template_manifest.get("dataset_root") or scene_manifest["dataset_root"]).expanduser()
    smplx_model_dir = resolve_smplx_model_dir(None if args.smplx_model_dir is None else str(args.smplx_model_dir))
    if smplx_model_dir is None:
        raise FileNotFoundError("Could not resolve SMPL-X model dir; pass --smplx-model-dir or set VGGT_SMPLX_MODEL_DIR.")
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
    left_hand_joints = parse_joint_ids(args.left_hand_joints)
    right_hand_joints = parse_joint_ids(args.right_hand_joints)
    _, left_hand_faces = select_faces_for_joints(faces, dominant_joint, left_hand_joints, args.hand_face_mode)
    _, right_hand_faces = select_faces_for_joints(faces, dominant_joint, right_hand_joints, args.hand_face_mode)

    camera_params, camera_source = resolve_scene_camera_params(scene_manifest, dataset_root, args.subset_name)
    detector = create_hand_landmarker(Path(args.hand_landmarker_model), bool(args.disable_mediapipe_hands))

    point_masks = np.asarray(inputs["point_masks"], dtype=bool)
    base_cam_points = np.asarray(targets["cam_points"], dtype=np.float32)
    base_world_points = np.asarray(targets["world_points"], dtype=np.float32)
    base_conf = np.asarray(targets["world_points_conf"], dtype=np.float32)
    extrinsics = np.asarray(targets["extrinsics"], dtype=np.float32)
    view_count, height, width = point_masks.shape

    prior_depths = np.zeros((view_count, height, width), dtype=np.float32)
    prior_points = np.zeros_like(base_world_points, dtype=np.float32)
    bodyhand_mask = np.zeros((view_count, height, width), dtype=bool)
    body_anchor_mask = np.zeros_like(bodyhand_mask)
    hand_anchor_mask = np.zeros_like(bodyhand_mask)
    summaries: list[dict[str, Any]] = []

    for view_idx, view in enumerate(scene_manifest["exported_views"]):
        scene = load_scene_view(scene_dir, view_idx, (height, width))
        camera_id = normalize_camera_id(view["camera_id"])
        cam = camera_params[camera_id]
        intrinsic = align_intrinsics_for_scene_view(cam["intrinsic"], view, target_size=height)
        scene_mask = scene.mask.astype(bool) & point_masks[view_idx]
        protected = protect_mask_from_targets(targets, view_idx, scene_mask)

        body_raster, body_points_cam, body_meta = render_mask_and_points(
            vertices,
            faces,
            cam["world_to_cam"],
            intrinsic,
            (height, width),
            scene_mask,
        )
        left_raster, left_points_cam, left_meta = render_mask_and_points(
            vertices,
            left_hand_faces,
            cam["world_to_cam"],
            intrinsic,
            (height, width),
            scene_mask,
        )
        right_raster, right_points_cam, right_meta = render_mask_and_points(
            vertices,
            right_hand_faces,
            cam["world_to_cam"],
            intrinsic,
            (height, width),
            scene_mask,
        )

        mp_hand = mediapipe_hand_mask(scene.rgb, scene_mask, detector, pad=int(args.hand_box_pad))
        if mp_hand is None:
            hand_support, hand_summary = hand_risk_mask(scene.rgb, scene_mask)
        else:
            hand_support, hand_summary = mp_hand

        selected_hand = np.zeros_like(scene_mask, dtype=bool)
        selected_hand_cam = np.zeros((height, width, 3), dtype=np.float32)
        selected_boxes: list[dict[str, Any]] = []
        boxes = list(hand_summary.get("boxes_xyxy") or [])
        for box in boxes:
            box_mask = build_box_mask((height, width), box) & scene_mask & ~protected
            best: dict[str, Any] | None = None
            for side, raster, points_cam in (
                ("left", left_raster, left_points_cam),
                ("right", right_raster, right_points_cam),
            ):
                visible = raster & box_mask
                if int(visible.sum()) <= 0:
                    continue
                stats = point_box_spatial_stats(points_cam[visible])
                score = int(visible.sum())
                gate_ok = (
                    float(stats.get("max_extent", float("inf"))) <= 0.30
                    and float(stats.get("depth_range", float("inf"))) <= 0.18
                )
                candidate = {
                    "side": side,
                    "visible_pixels": int(visible.sum()),
                    "gate_ok": bool(gate_ok),
                    "max_extent": float(stats.get("max_extent", 0.0)),
                    "depth_range": float(stats.get("depth_range", 0.0)),
                    "score": int(score + (100000 if gate_ok else 0)),
                    "mask": visible,
                    "points_cam": points_cam,
                }
                if best is None or candidate["score"] > int(best["score"]):
                    best = candidate
            if best is None:
                continue
            visible = np.asarray(best["mask"], dtype=bool)
            selected_hand |= visible
            selected_hand_cam[visible] = np.asarray(best["points_cam"], dtype=np.float32)[visible]
            selected_boxes.append(
                {
                    key: value
                    for key, value in best.items()
                    if key not in {"mask", "points_cam"}
                }
            )

        body_mask = body_raster & scene_mask & ~protected
        target_raw_mask = (body_mask | selected_hand) & ~protected
        raw_cam_map = body_points_cam.copy()
        raw_cam_map[selected_hand] = selected_hand_cam[selected_hand]

        align_support = (
            body_mask
            & np.isfinite(raw_cam_map).all(axis=-1)
            & np.isfinite(base_cam_points[view_idx]).all(axis=-1)
            & np.isfinite(base_conf[view_idx])
            & (base_conf[view_idx] > 0.0)
        )
        threshold = (
            float(np.percentile(base_conf[view_idx][align_support], float(args.conf_percentile)))
            if align_support.any()
            else 0.0
        )
        align_mask = align_support & (base_conf[view_idx] >= threshold)
        if int(align_mask.sum()) < int(args.min_align_pixels):
            align_mask = align_support

        if int(align_mask.sum()) < 16:
            summaries.append(
                {
                    "view_index": int(view_idx),
                    "camera_id": camera_id,
                    "skipped": True,
                    "reason": "too_few_alignment_pixels",
                    "align_pixels": int(align_mask.sum()),
                    "target_pixels_raw": int(target_raw_mask.sum()),
                    "hand_boxes": selected_boxes,
                }
            )
            draw_overlay(
                output_dir / "overlays" / f"view_{view_idx:02d}_raw_smplx_bodyhand_anchor.png",
                scene.rgb,
                body_mask & False,
                selected_hand & False,
                protected,
                f"skip align={int(align_mask.sum())}",
            )
            continue

        src = raw_cam_map[align_mask]
        dst = base_cam_points[view_idx][align_mask]
        transform, residual = robust_similarity(
            src,
            dst,
            max_correspondences=int(args.max_correspondences),
            seed=int(args.seed) + view_idx,
        )
        aligned_cam = apply_similarity(raw_cam_map, transform)
        delta = np.linalg.norm(aligned_cam - base_cam_points[view_idx], axis=-1)
        valid_target = (
            target_raw_mask
            & np.isfinite(aligned_cam).all(axis=-1)
            & (aligned_cam[..., 2] > 1e-6)
            & np.isfinite(delta)
            & (delta <= float(args.max_cam_delta))
        )
        prior_depths[view_idx][valid_target] = aligned_cam[..., 2][valid_target]
        prior_points[view_idx] = camera_to_world(aligned_cam, extrinsics[view_idx])
        bodyhand_mask[view_idx] = valid_target
        body_anchor_mask[view_idx] = valid_target & body_mask
        hand_anchor_mask[view_idx] = valid_target & selected_hand

        valid_delta = delta[valid_target]
        summaries.append(
            {
                "view_index": int(view_idx),
                "camera_id": camera_id,
                "skipped": False,
                "conf_threshold": threshold,
                "align_pixels": int(align_mask.sum()),
                "target_pixels_raw": int(target_raw_mask.sum()),
                "target_pixels_kept": int(valid_target.sum()),
                "body_pixels_kept": int(body_anchor_mask[view_idx].sum()),
                "hand_pixels_kept": int(hand_anchor_mask[view_idx].sum()),
                "fit_residual_percentiles": [float(v) for v in np.percentile(residual, [0, 25, 50, 75, 90, 95, 99])],
                "delta_percentiles_kept": (
                    [float(v) for v in np.percentile(valid_delta, [0, 25, 50, 75, 90, 95, 99])]
                    if valid_delta.size
                    else []
                ),
                "body_raster_meta": body_meta,
                "left_hand_raster_meta": left_meta,
                "right_hand_raster_meta": right_meta,
                "hand_summary": hand_summary,
                "hand_boxes": selected_boxes,
            }
        )
        draw_overlay(
            output_dir / "overlays" / f"view_{view_idx:02d}_raw_smplx_bodyhand_anchor.png",
            scene.rgb,
            body_anchor_mask[view_idx],
            hand_anchor_mask[view_idx],
            protected,
            f"body={int(body_anchor_mask[view_idx].sum())} hand={int(hand_anchor_mask[view_idx].sum())}",
        )

    input_payload = dict(inputs)
    input_payload["prior_mask"] = bodyhand_mask.astype(bool)
    target_payload = dict(targets)
    target_payload["prior_depths"] = prior_depths.astype(np.float32)
    target_payload["prior_points"] = prior_points.astype(np.float32)
    target_payload["teacher_mask"] = point_masks.astype(bool)
    target_payload["smplx_bodyhand_anchor_mask"] = bodyhand_mask.astype(bool)
    target_payload["smplx_body_anchor_mask"] = body_anchor_mask.astype(bool)
    target_payload["smplx_hand_anchor_mask"] = hand_anchor_mask.astype(bool)
    target_payload.pop("prior_normals", None)

    save_npz(output_dir / "inputs.npz", input_payload)
    save_npz(output_dir / "targets.npz", target_payload)

    copied = []
    for name in ("roi_mask_augmentation_summary.json", "body_rehearsal_prior_targets_summary.json"):
        source = template_root / name
        if source.is_file():
            shutil.copy2(source, output_dir / source.name)
            copied.append(source.name)

    manifest = dict(template_manifest)
    manifest.update(
        {
            "case_id": output_dir.name,
            "template_case_root": str(template_root),
            "inputs_npz": "inputs.npz",
            "targets_npz": "targets.npz",
            "prior_geometry_source": "raw_smplx_bodyhand_weak_anchor_per_box_best_side",
            "prior_geometry_meta": {
                "source": "raw_smplx_bodyhand_weak_anchor_per_box_best_side",
                "truthful_status": "weak_body_hand_anchor_not_face_teacher",
                "camera_source": camera_source,
                "smplx_model_path": str(model_path),
                "conf_percentile": float(args.conf_percentile),
                "max_cam_delta": float(args.max_cam_delta),
                "head_face_hairline_ear_protected": True,
                "teacher_mask_semantics": "full point_masks for self-geometry; prior_mask is body/hands only",
                "per_view": summaries,
            },
            "raw_smplx_bodyhand_anchor_case": {
                "bodyhand_mask_pixels": int(bodyhand_mask.sum()),
                "body_anchor_pixels": int(body_anchor_mask.sum()),
                "hand_anchor_pixels": int(hand_anchor_mask.sum()),
                "copied_auxiliary_files": copied,
            },
        }
    )
    (output_dir / "case_manifest.json").write_text(
        json.dumps(json_ready(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "task": "build_raw_smplx_bodyhand_anchor_training_case",
        "truthful_status": "local_training_case_weak_bodyhand_anchor_not_face_teacher",
        "template_case_root": str(template_root),
        "output_dir": str(output_dir),
        "bodyhand_mask_pixels": int(bodyhand_mask.sum()),
        "body_anchor_pixels": int(body_anchor_mask.sum()),
        "hand_anchor_pixels": int(hand_anchor_mask.sum()),
        "per_view": summaries,
    }
    (output_dir / "raw_smplx_bodyhand_anchor_case_summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
