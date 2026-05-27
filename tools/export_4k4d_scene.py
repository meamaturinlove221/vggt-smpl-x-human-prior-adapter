from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dna_4k4d as dna  # noqa: E402
from vggt.utils.human_prior import (  # noqa: E402
    DEFAULT_SUMMARY_BIN_NAMES,
    DEFAULT_SUMMARY_FEATURE_NAMES,
    DEFAULT_SURFACE_FEATURE_NAMES,
    build_4k4d_smplx_vertices,
    build_body_local_vertices_from_pose_params,
    build_human_summary_tokens,
    build_pose_aligned_surface_feature_maps,
    load_4k4d_smplx_frame,
    preprocess_feature_map,
    project_vertices_to_feature_maps,
    world_to_camera_extrinsic_from_4k4d,
)


def decode_encoded_image(buffer: np.ndarray) -> Image.Image:
    decoded = cv2.imdecode(np.asarray(buffer), cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("Failed to decode encoded image bytes.")
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_rgb_frame(main_smc: Path, camera_id: str, frame_id: str) -> Image.Image:
    cam_num = int(camera_id)
    group_name = "Camera_5mp" if cam_num < 48 else "Camera_12mp"
    cam_key = str(cam_num)
    frame_key = str(int(frame_id))
    with h5py.File(main_smc, "r") as handle:
        buffer = handle[group_name][cam_key]["color"][frame_key][()]
    return decode_encoded_image(buffer)


def load_mask_frame(ann_smc: Path, camera_id: str, frame_id: str) -> Image.Image:
    cam_key = str(int(camera_id))
    frame_key = str(int(frame_id))
    with h5py.File(ann_smc, "r") as handle:
        buffer = handle["Mask"][cam_key]["mask"][frame_key][()]
    rgb = decode_encoded_image(buffer)
    gray = np.max(np.asarray(rgb), axis=2).astype(np.uint8)
    return Image.fromarray(gray, mode="L")


def load_camera_parameters(rgb_cams_smc: Path, camera_id: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(rgb_cams_smc, "r") as handle:
        camera_group_root = handle["Camera_Parameter"]
        cam_key = str(camera_id)
        if cam_key not in camera_group_root:
            fallback_key = str(int(camera_id))
            cam_key = fallback_key if fallback_key in camera_group_root else cam_key
        camera_group = camera_group_root[cam_key]
        intrinsic = np.asarray(camera_group["K"][()], dtype=np.float32)
        rt = np.asarray(camera_group["RT"][()], dtype=np.float32)
    extrinsic = world_to_camera_extrinsic_from_4k4d(rt)
    return intrinsic, extrinsic


def resolve_smplx_model_root(path_like: str) -> Path:
    candidates = []
    if path_like.strip():
        candidates.append(Path(path_like))
    candidates.extend(
        [
            Path("Z:/smplx"),
            Path("G:/数据集/datasets/smplx"),
            Path("G:/datasets/smplx"),
        ]
    )
    for candidate in candidates:
        if (candidate / "SMPLX_NEUTRAL.npz").is_file():
            return candidate.resolve()
    raise FileNotFoundError("Could not locate the SMPL-X model root containing SMPLX_NEUTRAL.npz.")


def make_contact_sheet(images: list[Image.Image], labels: list[str], output_path: Path) -> None:
    cell_w = 320
    cols = 4
    rows = (len(images) + cols - 1) // cols
    resized = []
    heights = []
    for image in images:
        height = int(image.height * (cell_w / image.width))
        resized.append(image.resize((cell_w, height)))
        heights.append(height)
    cell_h = max(heights)
    label_h = 20
    canvas = Image.new("RGB", (cols * cell_w, rows * (cell_h + label_h)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, (image, label) in enumerate(zip(resized, labels)):
        row = idx // cols
        col = idx % cols
        x = col * cell_w
        y = row * (cell_h + label_h)
        canvas.paste(image, (x, y))
        draw.text((x + 6, y + cell_h + 2), label, fill=(240, 240, 240), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def resolve_smc(context: dna.DatasetContext, canonical_path: str, temp_dir: Path) -> Path:
    status = dna.describe_expected_file(context, canonical_path)
    if status["status"] == "extracted":
        return Path(status["path"])
    if status["status"] == "archived":
        materialized = dna.materialize_archived_member(context, canonical_path, temp_dir)
        if materialized is not None:
            return materialized
    raise FileNotFoundError(f"Could not resolve required file: {canonical_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one 4K4D frame as a plain image scene for official VGGT inference.")
    parser.add_argument("--dataset-root", required=True, help="Extracted data_used_in_4K4D root or its parent folder.")
    parser.add_argument("--seq", required=True, help="Sequence id such as 0012_11")
    parser.add_argument("--frame", default="0", help="Frame id. Default: 0")
    parser.add_argument("--target-camera", default="00", help="Target camera id. Default: 00")
    parser.add_argument("--source-cameras", nargs="*", default=[], help="Explicit source camera ids.")
    parser.add_argument("--auto-sources", type=int, default=6, help="Auto-pick N source cameras if none are provided.")
    parser.add_argument("--all-cameras", action="store_true", help="Use all available cameras except the target as sources.")
    parser.add_argument("--output-dir", required=True, help="Output scene directory.")
    parser.add_argument("--smplx-model-root", default="", help="Local SMPL-X model root. Auto-detected when omitted.")
    parser.add_argument("--prior-target-size", type=int, default=518, help="Target size for exported prior feature maps.")
    parser.add_argument("--prior-image-mode", default="pad", choices=["crop", "pad"], help="Image-aligned preprocessing mode for prior feature maps.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing exported images.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    context = dna.build_context(Path(args.dataset_root), dna.SUBSET_NAME)
    frame_id = str(int(args.frame))
    target_camera = dna.normalize_camera_id(args.target_camera)
    output_dir = Path(args.output_dir).resolve()
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    human_prior_dir = output_dir / "human_prior"
    human_prior_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dna_4k4d_export_") as temp_name:
        temp_dir = Path(temp_name)
        main_smc = resolve_smc(context, f"{dna.SUBSET_NAME}/main/{args.seq}.smc", temp_dir)
        ann_smc = resolve_smc(context, f"{dna.SUBSET_NAME}/annotations/{args.seq}_annots.smc", temp_dir)
        rgb_cams_smc, rgb_cams_source = dna.materialize_rgb_cams_smc(context, args.seq, temp_dir)
        if rgb_cams_smc is None:
            raise FileNotFoundError(f"Could not resolve rgb_cams SMC for {args.seq}")
        camera_summary = dna.load_camera_summary(rgb_cams_smc)
        available_cameras = list(camera_summary["camera_ids"])
        smplx_model_root = resolve_smplx_model_root(args.smplx_model_root)
        smplx_pose_params = load_4k4d_smplx_frame(ann_smc, int(frame_id))
        smplx_vertices_world = build_4k4d_smplx_vertices(
            smplx_model_root,
            smplx_pose_params,
            gender="neutral",
        )
        smplx_body_local_vertices = build_body_local_vertices_from_pose_params(
            smplx_vertices_world,
            smplx_pose_params,
        )
        smpl_summary_tokens = build_human_summary_tokens(smplx_body_local_vertices)

        source_cameras = [dna.normalize_camera_id(camera) for camera in args.source_cameras]
        if args.all_cameras:
            source_cameras = [camera for camera in available_cameras if camera != target_camera]
        elif not source_cameras:
            source_cameras = dna.auto_pick_sources(available_cameras, target_camera, args.auto_sources)
        selected_cameras = [target_camera, *source_cameras]

        labels = []
        rgb_images = []
        mask_images = []
        exported = []
        prior_masks = []
        prior_density_maps = []
        prior_surface_feature_maps = []
        prior_coverages = []
        for idx, camera_id in enumerate(selected_cameras):
            role = "tgt" if idx == 0 else "src"
            prefix = f"{idx:02d}_{role}_cam{camera_id}"
            rgb_image = load_rgb_frame(main_smc, camera_id, frame_id)
            mask_image = load_mask_frame(ann_smc, camera_id, frame_id)
            intrinsic, extrinsic = load_camera_parameters(rgb_cams_smc, camera_id)

            raw_hw = (rgb_image.height, rgb_image.width)
            prior_mask_raw, prior_density_raw, prior_vertex_map_raw, _, _, _ = project_vertices_to_feature_maps(
                smplx_vertices_world,
                extrinsic,
                intrinsic,
                raw_hw,
            )
            prior_mask = preprocess_feature_map(
                prior_mask_raw.astype(np.float32),
                mode=args.prior_image_mode,
                target_size=args.prior_target_size,
                interpolation="nearest",
                pad_value=0.0,
            ) > 0.5
            prior_density = preprocess_feature_map(
                prior_density_raw.astype(np.float32),
                mode=args.prior_image_mode,
                target_size=args.prior_target_size,
                interpolation="bilinear",
                pad_value=0.0,
            ).astype(np.float16)
            prior_vertex_map = preprocess_feature_map(
                prior_vertex_map_raw.astype(np.float32),
                mode=args.prior_image_mode,
                target_size=args.prior_target_size,
                interpolation="bilinear",
                pad_value=0.0,
            ).astype(np.float16)
            _, _, surface_feature_map_raw, _ = build_pose_aligned_surface_feature_maps(
                smplx_vertices_world,
                smplx_body_local_vertices,
                extrinsic,
                intrinsic,
                raw_hw,
            )
            surface_feature_map = preprocess_feature_map(
                surface_feature_map_raw.astype(np.float32),
                mode=args.prior_image_mode,
                target_size=args.prior_target_size,
                interpolation="bilinear",
                pad_value=0.0,
            ).astype(np.float16)

            rgb_path = image_dir / f"{prefix}.png"
            mask_path = mask_dir / f"{prefix}.png"
            if not args.overwrite and (rgb_path.exists() or mask_path.exists()):
                raise FileExistsError(f"{rgb_path} already exists. Re-run with --overwrite.")

            rgb_image.save(rgb_path)
            mask_image.save(mask_path)

            labels.append(f"{camera_id} ({role})")
            rgb_images.append(rgb_image)
            mask_images.append(mask_image.convert("RGB"))
            prior_masks.append(prior_mask.astype(bool))
            prior_density_maps.append(prior_density)
            prior_surface_feature_maps.append(surface_feature_map)
            prior_coverages.append(float(prior_mask.mean()))
            exported.append(
                {
                    "camera_id": camera_id,
                    "role": role,
                    "image_path": str(rgb_path),
                    "mask_path": str(mask_path),
                    "image_size": list(rgb_image.size),
                    "mask_coverage": float((np.asarray(mask_image) > 0).mean()),
                    "prior_coverage_518": float(prior_mask.mean()),
                }
            )

        make_contact_sheet(rgb_images, labels, output_dir / "rgb_contact_sheet.png")
        make_contact_sheet(mask_images, labels, output_dir / "mask_contact_sheet.png")

        prior_bundle_path = human_prior_dir / "smplx_vertex_feature_maps.npz"
        np.savez_compressed(
            prior_bundle_path,
            smpl_vertex_feature_maps=np.stack(prior_surface_feature_maps, axis=0).astype(np.float16),
            smpl_surface_feature_maps=np.stack(prior_surface_feature_maps, axis=0).astype(np.float16),
            smpl_prior_feature_maps=np.stack(prior_density_maps, axis=0).astype(np.float16),
            smpl_prior_masks=np.stack(prior_masks, axis=0).astype(bool),
            channel_names=np.asarray(DEFAULT_SURFACE_FEATURE_NAMES),
            surface_channel_names=np.asarray(DEFAULT_SURFACE_FEATURE_NAMES),
            camera_ids=np.asarray(selected_cameras),
            smpl_summary_tokens=smpl_summary_tokens.astype(np.float16),
            summary_feature_names=np.asarray(DEFAULT_SUMMARY_FEATURE_NAMES),
            summary_bin_names=np.asarray(DEFAULT_SUMMARY_BIN_NAMES),
            prior_target_size=np.asarray([int(args.prior_target_size)], dtype=np.int32),
            prior_image_mode=np.asarray([str(args.prior_image_mode)]),
        )

        scene_manifest = {
            "dataset_root": str(context.subset_roots[0] if context.subset_roots else context.dataset_path),
            "seq_id": args.seq,
            "frame_id": frame_id,
            "target_camera": target_camera,
            "source_cameras": source_cameras,
            "camera_summary": camera_summary,
            "rgb_cams_source": rgb_cams_source,
            "main_smc": str(main_smc),
            "annotations_smc": str(ann_smc),
            "smplx_model_root": str(smplx_model_root),
            "exported_views": exported,
            "human_prior": {
                "bundle_path": str(prior_bundle_path),
                "channel_names": list(DEFAULT_SURFACE_FEATURE_NAMES),
                "summary_feature_names": list(DEFAULT_SUMMARY_FEATURE_NAMES),
                "summary_bin_names": list(DEFAULT_SUMMARY_BIN_NAMES),
                "prior_target_size": int(args.prior_target_size),
                "prior_image_mode": str(args.prior_image_mode),
                "mean_prior_coverage_518": float(np.mean(prior_coverages)) if prior_coverages else 0.0,
                "min_prior_coverage_518": float(np.min(prior_coverages)) if prior_coverages else 0.0,
                "max_prior_coverage_518": float(np.max(prior_coverages)) if prior_coverages else 0.0,
            },
        }
        with (output_dir / "scene_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(scene_manifest, handle, indent=2, ensure_ascii=False)

    print(f"Exported {len(selected_cameras)} views to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
