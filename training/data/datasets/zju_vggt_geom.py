# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import os.path as osp
import random
import re
import tempfile
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from data.base_dataset import BaseDataset
from data.dataset_util import depth_to_world_coords_points, read_image_cv2

REPO_ROOT = Path(__file__).resolve().parents[3]


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


_SAMPLE_SEQ_PATTERN = re.compile(r"^zju_(?P<seq_name>.+)_frame_(?P<frame_id>\d+)$")


def _normalize_manifest_seq_name(raw_value):
    text = str(raw_value).strip()
    match = _SAMPLE_SEQ_PATTERN.match(text)
    if match:
        return match.group("seq_name")
    return text


def _load_sample_manifest_entries(manifest_path: Path):
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_entries = payload
    if isinstance(payload, dict):
        raw_entries = (
            payload.get("entries")
            or payload.get("frame_entries")
            or payload.get("bucket_entries")
            or payload.get("items")
            or []
        )
    if not isinstance(raw_entries, list):
        raise ValueError(f"sample manifest must contain a list of entries: {manifest_path}")

    keys = set()
    entry_by_key = {}
    for entry in raw_entries:
        if not isinstance(entry, dict):
            raise ValueError(f"sample manifest entry must be an object: {entry!r}")
        seq_name = entry.get("seq_name")
        frame_id = entry.get("frame_id")
        if seq_name is None or frame_id is None:
            raise ValueError(
                "sample manifest entry must contain seq_name and frame_id: "
                f"{entry!r}"
            )
        key = (_normalize_manifest_seq_name(seq_name), int(frame_id))
        keys.add(key)
        entry_by_key[key] = dict(entry)
    return keys, entry_by_key


def _resolve_sample_manifest_path(manifest_path_like: str) -> Path:
    manifest_path = Path(str(manifest_path_like).strip())
    if manifest_path.is_absolute() and manifest_path.is_file():
        return manifest_path
    if manifest_path.is_file():
        return manifest_path.resolve()
    repo_relative = (REPO_ROOT / manifest_path).resolve()
    if repo_relative.is_file():
        return repo_relative
    return manifest_path


def _read_opencv_matrix(fs, key):
    node = fs.getNode(key)
    if node.empty():
        raise KeyError(f"missing node in yaml: {key}")
    matrix = node.mat()
    if matrix is None:
        raise KeyError(f"node has no matrix data: {key}")
    return np.asarray(matrix, dtype=np.float64)


def _load_zju_cameras(seq_dir: Path):
    intri_path = seq_dir / "intri.yml"
    extri_path = seq_dir / "extri.yml"
    temp_paths = []
    temp_root = None
    use_temp_copy = (not str(intri_path).isascii()) or (not str(extri_path).isascii())
    if use_temp_copy:
        temp_root = Path(tempfile.mkdtemp(prefix="vggt_zju_yaml_train_"))
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_intri = temp_root / "intri.yml"
        temp_extri = temp_root / "extri.yml"
        shutil.copyfile(intri_path, temp_intri)
        shutil.copyfile(extri_path, temp_extri)
        temp_paths = [temp_intri, temp_extri]
        intri_fs = cv2.FileStorage(str(temp_intri), cv2.FILE_STORAGE_READ)
        extri_fs = cv2.FileStorage(str(temp_extri), cv2.FILE_STORAGE_READ)
    else:
        intri_fs = cv2.FileStorage(str(intri_path), cv2.FILE_STORAGE_READ)
        extri_fs = cv2.FileStorage(str(extri_path), cv2.FILE_STORAGE_READ)

    if not intri_fs.isOpened() or not extri_fs.isOpened():
        raise RuntimeError(f"failed to open camera yaml files for {seq_dir}")

    cameras = {}
    try:
        for cam_dir in sorted(seq_dir.glob("Camera_*")):
            cam_name = cam_dir.name
            intrinsic = _read_opencv_matrix(intri_fs, f"K_{cam_name}")
            rotation = _read_opencv_matrix(extri_fs, f"Rot_{cam_name}")
            translation = _read_opencv_matrix(extri_fs, f"T_{cam_name}").reshape(3)
            extrinsic = np.concatenate([rotation, translation[:, None]], axis=1)
            cameras[cam_name] = {
                "intrinsic": intrinsic.astype(np.float32),
                "extrinsic": extrinsic.astype(np.float32),
            }
    finally:
        intri_fs.release()
        extri_fs.release()
        for temp_path in temp_paths:
            try:
                temp_path.unlink()
            except OSError:
                pass
        if temp_root is not None:
            try:
                temp_root.rmdir()
            except OSError:
                pass
    return cameras


def _scale_intrinsic(intrinsic_3x3, src_hw, dst_hw):
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    out = np.asarray(intrinsic_3x3, dtype=np.float32).copy()
    out[0, 0] *= sx
    out[1, 1] *= sy
    out[0, 2] *= sx
    out[1, 2] *= sy
    return out


def _build_camera_ring_order(camera_map):
    rows = []
    for cam_name, camera in camera_map.items():
        extrinsic = np.asarray(camera["extrinsic"], dtype=np.float64)
        rotation = extrinsic[:, :3]
        translation = extrinsic[:, 3].reshape(3, 1)
        center = (-rotation.T @ translation).reshape(3)
        azimuth = float(np.degrees(np.arctan2(center[0], center[2])))
        rows.append((str(cam_name), azimuth))
    rows.sort(key=lambda item: item[1])
    return [name for name, _ in rows]


def _build_nearest_ring_sources(anchor_camera, source_count, ring_order, available_cameras, excluded_cameras=None):
    if source_count <= 0:
        return []
    excluded = set(excluded_cameras or [])
    camera_to_index = {camera: idx for idx, camera in enumerate(ring_order)}
    if anchor_camera not in camera_to_index:
        raise ValueError(f"Anchor camera missing from ring order: {anchor_camera}")
    anchor_idx = camera_to_index[anchor_camera]
    selected = []
    for ring_step in range(1, len(ring_order)):
        for offset in (ring_step, -ring_step):
            camera = ring_order[(anchor_idx + offset) % len(ring_order)]
            if (
                camera == anchor_camera
                or camera not in available_cameras
                or camera in selected
                or camera in excluded
            ):
                continue
            selected.append(camera)
            if len(selected) == source_count:
                return selected
    raise ValueError(f"Unable to build nearest_ring sources for {anchor_camera}")


def _build_uniform_ring_sources(anchor_camera, source_count, ring_order, available_cameras, excluded_cameras=None):
    if source_count <= 0:
        return []
    excluded = set(excluded_cameras or [])
    camera_to_index = {camera: idx for idx, camera in enumerate(ring_order)}
    if anchor_camera not in camera_to_index:
        raise ValueError(f"Anchor camera missing from ring order: {anchor_camera}")
    anchor_idx = camera_to_index[anchor_camera]
    total_offsets = len(ring_order) - 1
    raw_offsets = []
    for position in range(source_count):
        offset = int(np.floor((position + 0.5) * total_offsets / source_count)) + 1
        offset = max(1, min(total_offsets, offset))
        raw_offsets.append(offset)

    selected = []
    used = set()
    for offset in raw_offsets:
        camera = ring_order[(anchor_idx + offset) % len(ring_order)]
        if (
            camera == anchor_camera
            or camera not in available_cameras
            or camera in used
            or camera in excluded
        ):
            continue
        selected.append(camera)
        used.add(camera)

    if len(selected) < source_count:
        for offset in range(1, len(ring_order)):
            camera = ring_order[(anchor_idx + offset) % len(ring_order)]
            if (
                camera == anchor_camera
                or camera not in available_cameras
                or camera in used
                or camera in excluded
            ):
                continue
            selected.append(camera)
            used.add(camera)
            if len(selected) == source_count:
                break

    if len(selected) != source_count:
        raise ValueError(f"Unable to build uniform_ring sources for {anchor_camera}")
    return selected


def _build_nearest_plus_uniform_tail_sources(
    anchor_camera,
    source_count,
    ring_order,
    available_cameras,
    excluded_cameras=None,
):
    if source_count <= 0:
        return []
    if source_count == 1:
        return _build_nearest_ring_sources(
            anchor_camera,
            source_count,
            ring_order,
            available_cameras,
            excluded_cameras=excluded_cameras,
        )

    nearest_count = max(1, source_count - 1)
    uniform_count = max(0, source_count - nearest_count)
    selected = _build_nearest_ring_sources(
        anchor_camera,
        nearest_count,
        ring_order,
        available_cameras,
        excluded_cameras=excluded_cameras,
    )
    if uniform_count <= 0:
        return selected

    excluded = set(excluded_cameras or [])
    excluded.update(selected)
    selected.extend(
        _build_uniform_ring_sources(
            anchor_camera,
            uniform_count,
            ring_order,
            available_cameras,
            excluded_cameras=excluded,
        )
    )
    return selected


def _build_nearest_plus_uniform_tail_supervised_reserve_selection(
    anchor_camera,
    img_per_seq,
    ring_order,
    geom_camera_names,
    candidate_camera_names,
):
    geom_available_set = set(geom_camera_names)
    candidate_available_set = set(candidate_camera_names)
    if anchor_camera not in geom_available_set:
        raise ValueError(f"Anchor camera missing from geom cameras: {anchor_camera}")

    secondary_supervised = _build_nearest_ring_sources(
        anchor_camera,
        1,
        ring_order,
        geom_available_set,
        excluded_cameras={anchor_camera},
    )
    selected_supervised = [anchor_camera] + list(secondary_supervised)
    source_count = max(0, int(img_per_seq) - len(selected_supervised))
    source_cameras = _build_nearest_plus_uniform_tail_sources(
        anchor_camera,
        source_count,
        ring_order,
        candidate_available_set,
        excluded_cameras=set(selected_supervised),
    )
    return selected_supervised, source_cameras


def _resize_rgb_image(image_rgb, target_hw):
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    return np.asarray(
        Image.fromarray(image_rgb).resize((target_w, target_h), Image.Resampling.BICUBIC),
        dtype=np.uint8,
    )


def _load_mask(seq_dir: Path, camera_name: str, frame_id: int, mask_source: str, target_hw):
    if mask_source == "none":
        return np.ones((int(target_hw[0]), int(target_hw[1])), dtype=bool)
    mask_path = seq_dir / mask_source / camera_name / f"{int(frame_id):06d}.png"
    if not mask_path.is_file():
        raise FileNotFoundError(f"mask not found: {mask_path}")
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    mask = cv2.resize(mask, (int(target_hw[1]), int(target_hw[0])), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def _resolve_image_path(seq_dir: Path, frame_id: int, camera_name: str, stored_path: str):
    stored = Path(stored_path)
    if stored.is_file():
        return stored
    for ext in (".jpg", ".png", ".jpeg"):
        candidate = seq_dir / camera_name / f"{int(frame_id):06d}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"image not found for {camera_name} frame={frame_id}")


def _merge_geom_payloads(geom_paths):
    if not geom_paths:
        raise ValueError("geom_paths must contain at least one .npz path")

    merged_cam_names = []
    merged_img_paths = []
    merged_depth = []
    merged_depth_conf = []
    merged_pointmap = []
    merged_extrinsic = []
    merged_intrinsic = []
    seen_cameras = set()

    for geom_path in geom_paths:
        geom = np.load(geom_path, allow_pickle=True)
        cam_names = [str(cam_name) for cam_name in geom["cam_names"].tolist()]
        for local_idx, cam_name in enumerate(cam_names):
            if cam_name in seen_cameras:
                continue
            seen_cameras.add(cam_name)
            merged_cam_names.append(cam_name)
            merged_img_paths.append(str(geom["img_paths"][local_idx]))
            merged_depth.append(np.asarray(geom["depth"][local_idx], dtype=np.float32))
            merged_depth_conf.append(np.asarray(geom["depth_conf"][local_idx], dtype=np.float32))
            merged_pointmap.append(np.asarray(geom["pointmap"][local_idx]))
            merged_extrinsic.append(np.asarray(geom["extrinsic"][local_idx], dtype=np.float32))
            merged_intrinsic.append(np.asarray(geom["intrinsic"][local_idx], dtype=np.float32))

    if not merged_cam_names:
        raise ValueError(f"Failed to load any merged views from geom paths: {geom_paths}")

    return {
        "cam_names": np.asarray(merged_cam_names),
        "img_paths": np.asarray(merged_img_paths, dtype=object),
        "depth": np.stack(merged_depth, axis=0),
        "depth_conf": np.stack(merged_depth_conf, axis=0),
        "pointmap": np.stack(merged_pointmap, axis=0),
        "extrinsic": np.stack(merged_extrinsic, axis=0),
        "intrinsic": np.stack(merged_intrinsic, axis=0),
    }


def _build_source_only_geometry(target_hw):
    hh, ww = int(target_hw[0]), int(target_hw[1])
    depth_map = np.zeros((hh, ww), dtype=np.float32)
    zero_points = np.zeros((hh, ww, 3), dtype=np.float32)
    point_mask = np.zeros((hh, ww), dtype=bool)
    return depth_map, zero_points, zero_points.copy(), point_mask


def _morph_kernel(radius):
    radius = int(max(0, radius))
    if radius <= 0:
        return None
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _binary_dilate_mask(mask, radius):
    kernel = _morph_kernel(radius)
    if kernel is None:
        return np.asarray(mask, dtype=bool)
    return cv2.dilate(np.asarray(mask, dtype=np.uint8), kernel, iterations=1).astype(bool)


def _binary_erode_mask(mask, radius):
    kernel = _morph_kernel(radius)
    if kernel is None:
        return np.asarray(mask, dtype=bool)
    return cv2.erode(np.asarray(mask, dtype=np.uint8), kernel, iterations=1).astype(bool)


def _binary_close_mask(mask, radius):
    kernel = _morph_kernel(radius)
    if kernel is None:
        return np.asarray(mask, dtype=bool)
    return cv2.morphologyEx(np.asarray(mask, dtype=np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)


def _build_edge_band_mask(mask, radius):
    radius = int(max(0, radius))
    if radius <= 0:
        return np.zeros_like(np.asarray(mask, dtype=bool), dtype=bool)
    dilated = _binary_dilate_mask(mask, radius)
    eroded = _binary_erode_mask(mask, radius)
    return dilated & ~eroded


def _project_smpl_vertices_to_feature_map(
    vertices_world,
    extrinsic,
    intrinsic,
    target_hw,
    *,
    point_radius_px=2,
    close_px=4,
    gaussian_sigma=2.5,
    eps=1e-6,
):
    hh, ww = int(target_hw[0]), int(target_hw[1])
    feature_map = np.zeros((hh, ww), dtype=np.float32)
    empty_mask = np.zeros((hh, ww), dtype=bool)
    empty_depth_map = np.zeros((hh, ww), dtype=np.float32)
    empty_world_points = np.zeros((hh, ww, 3), dtype=np.float32)
    if vertices_world is None:
        return empty_mask, feature_map, empty_mask, empty_depth_map, empty_world_points

    world_points = np.asarray(vertices_world, dtype=np.float32).reshape(-1, 3)
    if world_points.size == 0:
        return empty_mask, feature_map, empty_mask, empty_depth_map, empty_world_points

    rotation = np.asarray(extrinsic[:, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:, 3], dtype=np.float32).reshape(1, 3)
    cam_points = world_points @ rotation.T + translation
    finite_mask = np.isfinite(cam_points).all(axis=1) & np.isfinite(world_points).all(axis=1)
    cam_points = cam_points[finite_mask]
    world_points = world_points[finite_mask]
    if cam_points.size == 0:
        return empty_mask, feature_map, empty_mask, empty_depth_map, empty_world_points

    valid_depth = cam_points[:, 2] > float(eps)
    cam_points = cam_points[valid_depth]
    world_points = world_points[valid_depth]
    if cam_points.size == 0:
        return empty_mask, feature_map, empty_mask, empty_depth_map, empty_world_points

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    z = np.clip(cam_points[:, 2], float(eps), None)
    u = fx * cam_points[:, 0] / z + cx
    v = fy * cam_points[:, 1] / z + cy
    in_view = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0.0)
        & (u <= float(max(ww - 1, 0)))
        & (v >= 0.0)
        & (v <= float(max(hh - 1, 0)))
    )
    if not np.any(in_view):
        return empty_mask, feature_map, empty_mask, empty_depth_map, empty_world_points

    xx = np.rint(u[in_view]).astype(np.int32)
    yy = np.rint(v[in_view]).astype(np.int32)
    depth_values = z[in_view].astype(np.float32)
    world_points = world_points[in_view]
    np.add.at(feature_map, (yy, xx), 1.0)

    sparse_depth_map = np.full((hh, ww), np.inf, dtype=np.float32)
    sparse_world_points = np.zeros((hh, ww, 3), dtype=np.float32)
    sparse_mask = np.zeros((hh, ww), dtype=bool)
    for point_idx in np.argsort(depth_values, kind="stable"):
        x = int(xx[point_idx])
        y = int(yy[point_idx])
        depth_value = float(depth_values[point_idx])
        if sparse_mask[y, x] and depth_value >= float(sparse_depth_map[y, x]):
            continue
        sparse_mask[y, x] = True
        sparse_depth_map[y, x] = depth_value
        sparse_world_points[y, x] = world_points[point_idx]

    sparse_depth_map = np.where(sparse_mask, sparse_depth_map, 0.0).astype(np.float32)

    prior_mask = feature_map > 0.0
    prior_mask = _binary_dilate_mask(prior_mask, point_radius_px)
    prior_mask = _binary_close_mask(prior_mask, close_px)

    if gaussian_sigma > 0.0 and np.any(feature_map > 0.0):
        feature_map = cv2.GaussianBlur(
            feature_map,
            (0, 0),
            sigmaX=float(gaussian_sigma),
            sigmaY=float(gaussian_sigma),
        )
    max_value = float(feature_map.max()) if feature_map.size > 0 else 0.0
    if max_value > 0.0:
        feature_map = feature_map / max_value
    feature_map = np.maximum(feature_map, prior_mask.astype(np.float32))
    return (
        prior_mask.astype(bool),
        feature_map.astype(np.float32),
        sparse_mask.astype(bool),
        sparse_depth_map,
        sparse_world_points.astype(np.float32),
    )


def _densify_smpl_prior_geometry(
    sparse_mask,
    sparse_depth_map,
    sparse_world_points,
    target_mask,
    *,
    max_fill_distance_px=24.0,
):
    target_mask = np.asarray(target_mask, dtype=bool)
    hh, ww = target_mask.shape
    dense_mask = np.zeros((hh, ww), dtype=bool)
    dense_depth_map = np.zeros((hh, ww), dtype=np.float32)
    dense_world_points = np.zeros((hh, ww, 3), dtype=np.float32)

    sparse_mask = np.asarray(sparse_mask, dtype=bool)
    if not np.any(target_mask) or not np.any(sparse_mask):
        return dense_mask, dense_depth_map, dense_world_points

    source = (~sparse_mask).astype(np.uint8)
    distance_map, labels = cv2.distanceTransformWithLabels(
        source,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    fill_mask = target_mask & (labels > 0)
    max_fill_distance_px = float(max_fill_distance_px)
    if np.isfinite(max_fill_distance_px):
        if max_fill_distance_px <= 0.0:
            fill_mask &= sparse_mask
        else:
            fill_mask &= distance_map <= max_fill_distance_px
    if not np.any(fill_mask):
        return dense_mask, dense_depth_map, dense_world_points

    seed_coords = np.argwhere(sparse_mask)
    if seed_coords.size == 0:
        return dense_mask, dense_depth_map, dense_world_points

    fill_coords = np.argwhere(fill_mask)
    label_indices = labels[fill_mask].astype(np.int64) - 1
    valid_label_mask = (label_indices >= 0) & (label_indices < len(seed_coords))
    if not np.any(valid_label_mask):
        return dense_mask, dense_depth_map, dense_world_points

    fill_coords = fill_coords[valid_label_mask]
    source_coords = seed_coords[label_indices[valid_label_mask]]
    fill_ys = fill_coords[:, 0]
    fill_xs = fill_coords[:, 1]
    source_ys = source_coords[:, 0]
    source_xs = source_coords[:, 1]

    dense_mask[fill_ys, fill_xs] = True
    dense_depth_map[fill_ys, fill_xs] = sparse_depth_map[source_ys, source_xs]
    dense_world_points[fill_ys, fill_xs] = sparse_world_points[source_ys, source_xs]
    return dense_mask, dense_depth_map, dense_world_points


def _build_human_prior_masks(
    raw_foreground_mask,
    smpl_prior_mask,
    *,
    completion_dilate_px=5,
    head_hair_top_ratio=0.30,
    head_hair_horizontal_expand_ratio=0.15,
    head_hair_top_expand_ratio=0.10,
    head_hair_dilate_px=7,
    head_hair_edge_band_px=5,
    min_region_pixels=32,
):
    base_mask = np.asarray(raw_foreground_mask, dtype=bool) | np.asarray(smpl_prior_mask, dtype=bool)
    completion_mask = _binary_dilate_mask(base_mask, completion_dilate_px).astype(bool)
    if int(base_mask.sum()) < int(max(1, min_region_pixels)):
        zeros = np.zeros_like(base_mask, dtype=bool)
        return completion_mask, zeros, zeros

    ys, xs = np.where(base_mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    box_h = max(1, y1 - y0)
    box_w = max(1, x1 - x0)
    top_height = max(1, int(round(box_h * float(head_hair_top_ratio))))
    top_pad = int(round(box_h * float(head_hair_top_expand_ratio)))
    side_pad = int(round(box_w * float(head_hair_horizontal_expand_ratio)))

    head_region = np.zeros_like(base_mask, dtype=bool)
    head_region[
        max(0, y0 - top_pad) : min(base_mask.shape[0], y0 + top_height),
        max(0, x0 - side_pad) : min(base_mask.shape[1], x1 + side_pad),
    ] = True
    support_mask = _binary_dilate_mask(base_mask, head_hair_dilate_px)
    head_region = (head_region & support_mask).astype(bool)
    detail_mask = (head_region & _build_edge_band_mask(support_mask, head_hair_edge_band_px)).astype(bool)
    if int(detail_mask.sum()) < int(max(1, min_region_pixels)):
        detail_mask = head_region.copy()
    return completion_mask.astype(bool), head_region.astype(bool), detail_mask.astype(bool)


class ZjuVggtGeomDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        ZJU_DIR: str = None,
        seq_names=None,
        geom_subdir: str = "vggt_geom",
        holdout_stride: int = 10,
        len_train: int = 100000,
        len_test: int = 10000,
        camera_source: str = "gt",
        mask_source: str = "mask",
        use_foreground_mask: bool = True,
        min_depth_conf: float = 0.0,
        min_num_images: int = 2,
        source_policy: str = "random",
        source_view_pool: str = "cached_only",
        source_view_pool_train_probability: float = 1.0,
        source_anchor_policy: str = "random",
        min_supervised_views: int = 1,
        supervised_view_quality_filter: str = "none",
        conf_depth_view_quality_filter: str = "none",
        sample_manifest_path: str = "",
        sample_manifest_label: str = "",
        sample_manifest_use_entry_anchor: bool = False,
        sample_manifest_anchor_field: str = "promoted_anchor_camera",
        sample_manifest_use_entry_camera_set: bool = False,
        sample_manifest_camera_list_field: str = "selected_camera_names",
        sample_manifest_supervised_camera_field: str = "selected_supervised_camera_names",
        use_smpl_vertices_prior: bool = True,
        smpl_vertices_subdir: str = "new_vertices,vertices",
        smpl_prior_point_radius_px: int = 2,
        smpl_prior_close_px: int = 4,
        smpl_prior_gaussian_sigma: float = 2.5,
        smpl_completion_dilate_px: int = 5,
        smpl_completion_max_fill_px: float = 24.0,
        head_hair_top_ratio: float = 0.30,
        head_hair_horizontal_expand_ratio: float = 0.15,
        head_hair_top_expand_ratio: float = 0.10,
        head_hair_dilate_px: int = 7,
        head_hair_edge_band_px: int = 5,
    ):
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.load_depth = common_conf.load_depth

        if ZJU_DIR is None:
            raise ValueError("ZJU_DIR must be specified.")
        if camera_source not in ("gt", "geom"):
            raise ValueError("camera_source must be either 'gt' or 'geom'.")
        if mask_source not in ("none", "mask", "mask_cihp"):
            raise ValueError("mask_source must be one of 'none', 'mask', or 'mask_cihp'.")
        if source_policy not in ("random", "nearest_ring", "uniform_ring", "nearest_plus_uniform_tail", "nearest_plus_uniform_tail_supervised_reserve"):
            raise ValueError(
                "source_policy must be one of 'random', 'nearest_ring', 'uniform_ring', "
                "'nearest_plus_uniform_tail', or 'nearest_plus_uniform_tail_supervised_reserve'."
            )
        if source_view_pool not in ("cached_only", "geom_plus_raw"):
            raise ValueError("source_view_pool must be one of 'cached_only' or 'geom_plus_raw'.")
        source_view_pool_train_probability = float(source_view_pool_train_probability)
        if source_view_pool_train_probability < 0.0 or source_view_pool_train_probability > 1.0:
            raise ValueError("source_view_pool_train_probability must be within [0.0, 1.0].")
        if source_anchor_policy not in ("random", "max_depth_conf"):
            raise ValueError("source_anchor_policy must be one of 'random' or 'max_depth_conf'.")
        if supervised_view_quality_filter not in ("none", "drop_worst_by_depth_conf_if_multi_supervised"):
            raise ValueError(
                "supervised_view_quality_filter must be one of 'none' or 'drop_worst_by_depth_conf_if_multi_supervised'."
            )
        if conf_depth_view_quality_filter not in ("none", "drop_worst_by_depth_conf_if_multi_supervised"):
            raise ValueError(
                "conf_depth_view_quality_filter must be one of 'none' or 'drop_worst_by_depth_conf_if_multi_supervised'."
            )
        if source_view_pool == "geom_plus_raw" and camera_source != "gt":
            raise ValueError("source_view_pool='geom_plus_raw' currently requires camera_source='gt'.")
        min_supervised_views = int(min_supervised_views)
        if min_supervised_views < 1:
            raise ValueError("min_supervised_views must be >= 1.")
        if split not in ("train", "test"):
            raise ValueError(f"Invalid split: {split}")

        self.zju_root = Path(ZJU_DIR)
        self.seq_names = _ensure_list(seq_names) or ["CoreView_390"]
        self.geom_subdirs = _ensure_list(geom_subdir) or ["vggt_geom"]
        self.geom_subdir = ",".join(self.geom_subdirs)
        self.holdout_stride = int(holdout_stride)
        self.camera_source = camera_source
        self.mask_source = mask_source
        self.use_foreground_mask = bool(use_foreground_mask)
        self.min_depth_conf = float(min_depth_conf)
        self.min_num_images = int(min_num_images)
        self.source_policy = str(source_policy)
        self.source_view_pool = str(source_view_pool)
        self.source_view_pool_train_probability = float(source_view_pool_train_probability)
        self.source_anchor_policy = str(source_anchor_policy)
        self.min_supervised_views = int(min_supervised_views)
        self.supervised_view_quality_filter = str(supervised_view_quality_filter)
        self.conf_depth_view_quality_filter = str(conf_depth_view_quality_filter)
        self.sample_manifest_path = str(sample_manifest_path or "").strip()
        self.sample_manifest_label = str(sample_manifest_label or "").strip()
        self.sample_manifest_use_entry_anchor = bool(sample_manifest_use_entry_anchor)
        self.sample_manifest_anchor_field = str(sample_manifest_anchor_field or "promoted_anchor_camera").strip()
        self.sample_manifest_use_entry_camera_set = bool(sample_manifest_use_entry_camera_set)
        self.sample_manifest_camera_list_field = str(sample_manifest_camera_list_field or "selected_camera_names").strip()
        self.sample_manifest_supervised_camera_field = str(
            sample_manifest_supervised_camera_field or "selected_supervised_camera_names"
        ).strip()
        self.use_smpl_vertices_prior = bool(use_smpl_vertices_prior)
        self.smpl_vertices_subdirs = _ensure_list(smpl_vertices_subdir) or ["new_vertices", "vertices"]
        self.smpl_prior_point_radius_px = int(max(0, smpl_prior_point_radius_px))
        self.smpl_prior_close_px = int(max(0, smpl_prior_close_px))
        self.smpl_prior_gaussian_sigma = float(max(0.0, smpl_prior_gaussian_sigma))
        self.smpl_completion_dilate_px = int(max(0, smpl_completion_dilate_px))
        self.smpl_completion_max_fill_px = float(max(0.0, smpl_completion_max_fill_px))
        self.head_hair_top_ratio = float(max(0.05, min(0.75, head_hair_top_ratio)))
        self.head_hair_horizontal_expand_ratio = float(max(0.0, min(0.75, head_hair_horizontal_expand_ratio)))
        self.head_hair_top_expand_ratio = float(max(0.0, min(0.5, head_hair_top_expand_ratio)))
        self.head_hair_dilate_px = int(max(0, head_hair_dilate_px))
        self.head_hair_edge_band_px = int(max(0, head_hair_edge_band_px))
        self.sample_manifest_applied = False
        self.sample_manifest_entry_count = 0
        self.sample_manifest_entry_by_key = {}

        self.camera_store = {}
        self.camera_ring_orders = {}
        self.raw_source_camera_pools = {}
        self.smpl_vertices_dirs = {}
        self.smpl_vertices_cache = {}
        self.sequence_list = []

        for seq_name in self.seq_names:
            seq_dir = self.zju_root / seq_name
            if not seq_dir.is_dir():
                raise FileNotFoundError(f"ZJU sequence directory not found: {seq_dir}")

            self.camera_store[seq_name] = _load_zju_cameras(seq_dir)
            self.camera_ring_orders[seq_name] = _build_camera_ring_order(self.camera_store[seq_name])
            self.raw_source_camera_pools[seq_name] = [
                camera_name
                for camera_name in self.camera_ring_orders[seq_name]
                if (seq_dir / camera_name).is_dir()
            ]
            if self.use_smpl_vertices_prior:
                self.smpl_vertices_dirs[seq_name] = [
                    subdir_name
                    for subdir_name in self.smpl_vertices_subdirs
                    if (seq_dir / subdir_name).is_dir()
                ]
                if not self.smpl_vertices_dirs[seq_name]:
                    logging.warning(
                        "ZJU VGGT-Geom could not find requested SMPL vertices dirs for seq=%s under %s: %s",
                        seq_name,
                        seq_dir,
                        self.smpl_vertices_subdirs,
                    )
            else:
                self.smpl_vertices_dirs[seq_name] = []
            frame_sources = {}
            valid_geom_subdirs = []
            for geom_subdir_name in self.geom_subdirs:
                geom_dir = seq_dir / geom_subdir_name
                if not geom_dir.is_dir():
                    continue
                geom_paths = sorted(geom_dir.glob("frame_*.npz"))
                if not geom_paths:
                    continue
                valid_geom_subdirs.append(geom_subdir_name)
                for geom_path in geom_paths:
                    frame_id = int(geom_path.stem.split("_")[-1])
                    frame_sources.setdefault(frame_id, []).append(
                        {
                            "geom_path": geom_path,
                            "geom_subdir": geom_subdir_name,
                        }
                    )

            if not frame_sources:
                raise FileNotFoundError(
                    f"geometry cache directory not found or empty for any requested geom_subdir under {seq_dir}: {self.geom_subdirs}"
                )

            if len(valid_geom_subdirs) < len(self.geom_subdirs):
                missing = [subdir for subdir in self.geom_subdirs if subdir not in valid_geom_subdirs]
                logging.warning(
                    "ZJU VGGT-Geom missing or empty subdirs for seq=%s under root=%s: %s",
                    seq_name,
                    self.zju_root,
                    missing,
                )

            for frame_id in sorted(frame_sources):
                if self.holdout_stride > 0:
                    is_val = (frame_id % self.holdout_stride) == 0
                    if split == "train" and is_val:
                        continue
                    if split == "test" and not is_val:
                        continue
                ordered_sources = sorted(
                    frame_sources[frame_id],
                    key=lambda row: self.geom_subdirs.index(row["geom_subdir"]),
                )
                self.sequence_list.append(
                    {
                        "seq_name": seq_name,
                        "frame_id": frame_id,
                        "geom_path": ordered_sources[0]["geom_path"],
                        "geom_paths": [row["geom_path"] for row in ordered_sources],
                        "geom_subdirs_present": [row["geom_subdir"] for row in ordered_sources],
                        "seq_dir": seq_dir,
                    }
                )

        if self.sample_manifest_path and split == "train":
            manifest_path = _resolve_sample_manifest_path(self.sample_manifest_path)
            if not manifest_path.is_file():
                raise FileNotFoundError(f"sample manifest not found: {manifest_path}")
            manifest_keys, manifest_entry_by_key = _load_sample_manifest_entries(manifest_path)
            self.sample_manifest_entry_count = len(manifest_keys)
            if not manifest_keys:
                raise ValueError(f"sample manifest contains no usable entries: {manifest_path}")
            before_count = len(self.sequence_list)
            self.sequence_list = [
                row
                for row in self.sequence_list
                if (str(row["seq_name"]), int(row["frame_id"])) in manifest_keys
            ]
            matched_keys = {
                (str(row["seq_name"]), int(row["frame_id"]))
                for row in self.sequence_list
            }
            missing_count = len(manifest_keys - matched_keys)
            logging.info(
                "ZJU VGGT-Geom applied sample manifest split=%s path=%s kept=%d/%d missing=%d",
                split,
                manifest_path,
                len(self.sequence_list),
                before_count,
                missing_count,
            )
            if not self.sequence_list:
                raise ValueError(
                    f"sample manifest filtered all training entries for split={split}: {manifest_path}"
                )
            self.sample_manifest_entry_by_key = {
                key: manifest_entry_by_key[key]
                for key in matched_keys
                if key in manifest_entry_by_key
            }
            self.sample_manifest_applied = True
        elif self.sample_manifest_path:
            logging.info(
                "ZJU VGGT-Geom ignoring sample manifest for split=%s path=%s",
                split,
                self.sample_manifest_path,
            )

        if self.debug:
            self.sequence_list = self.sequence_list[: min(32, len(self.sequence_list))]

        if not self.sequence_list:
            raise ValueError(f"No geometry-cache frames found for split={split} under {self.zju_root}")

        self.sequence_list_len = len(self.sequence_list)
        if split == "train":
            self.len_train = self.sequence_list_len if len_train <= 0 else int(len_train)
        else:
            self.len_train = self.sequence_list_len if len_test <= 0 else int(len_test)

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: ZJU VGGT-Geom sequence count: {self.sequence_list_len}")
        logging.info(f"{status}: ZJU VGGT-Geom dataset length: {len(self)}")

    def _load_smpl_vertices(self, entry):
        if not self.use_smpl_vertices_prior:
            return None

        cache_key = (str(entry["seq_name"]), int(entry["frame_id"]))
        if cache_key in self.smpl_vertices_cache:
            return self.smpl_vertices_cache[cache_key]

        seq_name = str(entry["seq_name"])
        seq_dir = Path(entry["seq_dir"])
        frame_id = int(entry["frame_id"])
        candidate_names = [f"{frame_id}.npy", f"{frame_id:06d}.npy"]
        for subdir_name in self.smpl_vertices_dirs.get(seq_name, []):
            subdir = seq_dir / subdir_name
            for file_name in candidate_names:
                path = subdir / file_name
                if not path.is_file():
                    continue
                vertices = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
                self.smpl_vertices_cache[cache_key] = vertices
                return vertices

        self.smpl_vertices_cache[cache_key] = None
        return None

    def _resolve_source_view_pool_meta(self):
        requested_source_view_pool = self.source_view_pool
        effective_source_view_pool = requested_source_view_pool
        rawpool_candidate_pool_used = requested_source_view_pool == "geom_plus_raw"

        if (
            self.training
            and requested_source_view_pool == "geom_plus_raw"
            and self.source_view_pool_train_probability < 1.0
        ):
            rawpool_candidate_pool_used = random.random() < self.source_view_pool_train_probability
            effective_source_view_pool = "geom_plus_raw" if rawpool_candidate_pool_used else "cached_only"

        return {
            "requested_source_view_pool": requested_source_view_pool,
            "effective_source_view_pool": effective_source_view_pool,
            "source_view_pool_train_probability": float(self.source_view_pool_train_probability),
            "rawpool_candidate_pool_used": bool(rawpool_candidate_pool_used),
        }

    def _build_candidate_camera_names(self, entry, geom_camera_names, source_view_pool):
        if source_view_pool == "cached_only":
            return list(geom_camera_names)
        candidate_set = set(self.raw_source_camera_pools[entry["seq_name"]]) | set(geom_camera_names)
        ring_order = self.camera_ring_orders[entry["seq_name"]]
        return [camera for camera in ring_order if camera in candidate_set]

    def _select_anchor_camera(self, entry, geom_camera_names, geom):
        if self.sample_manifest_use_entry_anchor:
            manifest_entry = self.sample_manifest_entry_by_key.get((str(entry["seq_name"]), int(entry["frame_id"])), {})
            prescribed_anchor = str(manifest_entry.get(self.sample_manifest_anchor_field, "") or "").strip()
            if prescribed_anchor and prescribed_anchor in geom_camera_names:
                return prescribed_anchor
        if self.source_anchor_policy == "random":
            return random.choice(list(geom_camera_names))
        if self.source_anchor_policy == "max_depth_conf":
            depth_conf = np.asarray(geom["depth_conf"], dtype=np.float32)
            score_by_camera = {
                str(geom_camera_names[idx]): float(depth_conf[idx].mean())
                for idx in range(len(geom_camera_names))
            }
            return max(score_by_camera, key=score_by_camera.get)
        raise ValueError(f"Unsupported source anchor policy: {self.source_anchor_policy}")

    def _select_manifest_camera_set(self, entry, geom_camera_names, candidate_camera_names, img_per_seq):
        if not self.sample_manifest_use_entry_camera_set:
            return None
        manifest_entry = self.sample_manifest_entry_by_key.get((str(entry["seq_name"]), int(entry["frame_id"])), {})
        raw_selected = manifest_entry.get(self.sample_manifest_camera_list_field)
        if not isinstance(raw_selected, list):
            return None
        selected_camera_names = []
        seen = set()
        for camera_name in raw_selected:
            camera_name = str(camera_name or "").strip()
            if not camera_name or camera_name in seen:
                continue
            selected_camera_names.append(camera_name)
            seen.add(camera_name)
        if not selected_camera_names or len(selected_camera_names) != int(img_per_seq):
            return None
        candidate_camera_set = set(candidate_camera_names)
        if any(camera_name not in candidate_camera_set for camera_name in selected_camera_names):
            return None

        raw_supervised = manifest_entry.get(self.sample_manifest_supervised_camera_field)
        if isinstance(raw_supervised, list):
            prescribed_supervised = []
            seen_supervised = set()
            for camera_name in raw_supervised:
                camera_name = str(camera_name or "").strip()
                if not camera_name or camera_name in seen_supervised:
                    continue
                prescribed_supervised.append(camera_name)
                seen_supervised.add(camera_name)
        else:
            geom_camera_set = set(geom_camera_names)
            prescribed_supervised = [
                camera_name
                for camera_name in selected_camera_names
                if camera_name in geom_camera_set
            ]
        prescribed_supervised = [
            camera_name
            for camera_name in prescribed_supervised
            if camera_name in selected_camera_names
        ]

        anchor_camera = None
        if self.sample_manifest_use_entry_anchor:
            prescribed_anchor = str(manifest_entry.get(self.sample_manifest_anchor_field, "") or "").strip()
            if prescribed_anchor:
                anchor_camera = prescribed_anchor
        if anchor_camera not in selected_camera_names:
            anchor_camera = next(
                (camera for camera in prescribed_supervised if camera in selected_camera_names),
                None,
            )
        if anchor_camera not in selected_camera_names:
            anchor_camera = selected_camera_names[0]
        return selected_camera_names, prescribed_supervised, anchor_camera

    def _select_camera_names_with_source_policy(self, entry, geom_camera_names, geom, img_per_seq, source_view_pool_meta):
        effective_source_view_pool = source_view_pool_meta["effective_source_view_pool"]
        candidate_camera_names = self._build_candidate_camera_names(
            entry,
            geom_camera_names,
            effective_source_view_pool,
        )
        geom_camera_set = set(geom_camera_names)
        manifest_camera_set = self._select_manifest_camera_set(
            entry,
            geom_camera_names,
            candidate_camera_names,
            img_per_seq,
        )
        if manifest_camera_set is not None:
            selected_camera_names, selected_supervised_camera_names, anchor_camera = manifest_camera_set
            return selected_camera_names, {
                "source_policy": self.source_policy,
                "source_view_pool": effective_source_view_pool,
                "requested_source_view_pool": source_view_pool_meta["requested_source_view_pool"],
                "source_view_pool_train_probability": float(source_view_pool_meta["source_view_pool_train_probability"]),
                "rawpool_candidate_pool_used": bool(source_view_pool_meta["rawpool_candidate_pool_used"]),
                "source_anchor_policy": self.source_anchor_policy,
                "min_supervised_views": self.min_supervised_views,
                "supervised_view_quality_filter": self.supervised_view_quality_filter,
                "conf_depth_view_quality_filter": self.conf_depth_view_quality_filter,
                "selection_anchor_camera": anchor_camera,
                "selected_camera_names": list(selected_camera_names),
                "selected_supervised_camera_names": list(selected_supervised_camera_names),
                "selected_source_only_camera_names": [
                    camera for camera in selected_camera_names if camera not in set(selected_supervised_camera_names)
                ],
                "available_candidate_view_count": int(len(candidate_camera_names)),
                "available_candidate_camera_names": list(candidate_camera_names),
            }
        if img_per_seq >= len(candidate_camera_names):
            selected_camera_names = list(candidate_camera_names)
            if selected_camera_names:
                anchor_camera = self._select_anchor_camera(entry, geom_camera_names, geom)
                if anchor_camera not in selected_camera_names:
                    anchor_camera = next((camera for camera in selected_camera_names if camera in geom_camera_set), None)
                if anchor_camera is not None and selected_camera_names[0] != anchor_camera:
                    selected_camera_names.remove(anchor_camera)
                    selected_camera_names.insert(0, anchor_camera)
            else:
                anchor_camera = None
            return selected_camera_names, {
                "source_policy": self.source_policy,
                "source_view_pool": effective_source_view_pool,
                "requested_source_view_pool": source_view_pool_meta["requested_source_view_pool"],
                "source_view_pool_train_probability": float(source_view_pool_meta["source_view_pool_train_probability"]),
                "rawpool_candidate_pool_used": bool(source_view_pool_meta["rawpool_candidate_pool_used"]),
                "source_anchor_policy": self.source_anchor_policy,
                "min_supervised_views": self.min_supervised_views,
                "supervised_view_quality_filter": self.supervised_view_quality_filter,
                "conf_depth_view_quality_filter": self.conf_depth_view_quality_filter,
                "selection_anchor_camera": anchor_camera,
                "selected_camera_names": selected_camera_names,
                "selected_supervised_camera_names": [camera for camera in selected_camera_names if camera in geom_camera_set],
                "selected_source_only_camera_names": [camera for camera in selected_camera_names if camera not in geom_camera_set],
                "available_candidate_view_count": int(len(candidate_camera_names)),
                "available_candidate_camera_names": list(candidate_camera_names),
            }

        anchor_camera = self._select_anchor_camera(entry, geom_camera_names, geom)
        supervised_target = min(int(img_per_seq), len(geom_camera_names), self.min_supervised_views)
        selected_supervised = [anchor_camera]
        ring_order = self.camera_ring_orders[entry["seq_name"]]
        if supervised_target > 1:
            geom_available_set = set(geom_camera_names)
            if self.source_policy == "nearest_ring":
                selected_supervised.extend(
                    _build_nearest_ring_sources(
                        anchor_camera,
                        supervised_target - 1,
                        ring_order,
                        geom_available_set,
                    )
                )
            elif self.source_policy == "uniform_ring":
                selected_supervised.extend(
                    _build_uniform_ring_sources(
                        anchor_camera,
                        supervised_target - 1,
                        ring_order,
                        geom_available_set,
                    )
                )
            elif self.source_policy == "nearest_plus_uniform_tail":
                selected_supervised.extend(
                    _build_nearest_plus_uniform_tail_sources(
                        anchor_camera,
                        supervised_target - 1,
                        ring_order,
                        geom_available_set,
                    )
                )
            elif self.source_policy == "nearest_plus_uniform_tail_supervised_reserve":
                selected_supervised.extend(
                    _build_nearest_ring_sources(
                        anchor_camera,
                        supervised_target - 1,
                        ring_order,
                        geom_available_set,
                    )
                )
            else:
                raise ValueError(f"Unsupported source policy for training dataset: {self.source_policy}")
        selected_set = set(selected_supervised)
        source_count = max(0, int(img_per_seq) - len(selected_supervised))
        available_set = set(candidate_camera_names)

        if self.source_policy == "nearest_ring":
            source_cameras = _build_nearest_ring_sources(
                anchor_camera,
                source_count,
                ring_order,
                available_set,
                excluded_cameras=selected_set,
            )
        elif self.source_policy == "uniform_ring":
            source_cameras = _build_uniform_ring_sources(
                anchor_camera,
                source_count,
                ring_order,
                available_set,
                excluded_cameras=selected_set,
            )
        elif self.source_policy == "nearest_plus_uniform_tail":
            source_cameras = _build_nearest_plus_uniform_tail_sources(
                anchor_camera,
                source_count,
                ring_order,
                available_set,
                excluded_cameras=selected_set,
            )
        elif self.source_policy == "nearest_plus_uniform_tail_supervised_reserve":
            selected_supervised, source_cameras = _build_nearest_plus_uniform_tail_supervised_reserve_selection(
                anchor_camera,
                img_per_seq,
                ring_order,
                geom_camera_names,
                candidate_camera_names,
            )
        else:
            raise ValueError(f"Unsupported source policy for training dataset: {self.source_policy}")

        selected_camera_names = list(selected_supervised) + list(source_cameras)
        return selected_camera_names, {
            "source_policy": self.source_policy,
            "source_view_pool": effective_source_view_pool,
            "requested_source_view_pool": source_view_pool_meta["requested_source_view_pool"],
            "source_view_pool_train_probability": float(source_view_pool_meta["source_view_pool_train_probability"]),
            "rawpool_candidate_pool_used": bool(source_view_pool_meta["rawpool_candidate_pool_used"]),
            "source_anchor_policy": self.source_anchor_policy,
            "min_supervised_views": self.min_supervised_views,
            "supervised_view_quality_filter": self.supervised_view_quality_filter,
            "conf_depth_view_quality_filter": self.conf_depth_view_quality_filter,
            "selection_anchor_camera": anchor_camera,
            "selected_camera_names": selected_camera_names,
            "selected_supervised_camera_names": [camera for camera in selected_camera_names if camera in geom_camera_set],
            "selected_source_only_camera_names": [camera for camera in selected_camera_names if camera not in geom_camera_set],
            "available_candidate_view_count": int(len(candidate_camera_names)),
            "available_candidate_camera_names": list(candidate_camera_names),
        }

    def _apply_view_quality_filter(self, candidate_supervised_camera_names, score_by_camera, filter_name):
        dropped_supervised_camera_names = []
        if (
            filter_name == "drop_worst_by_depth_conf_if_multi_supervised"
            and len(candidate_supervised_camera_names) >= 2
            and score_by_camera
        ):
            dropped_supervised_camera_names = [
                min(candidate_supervised_camera_names, key=lambda camera_name: score_by_camera.get(str(camera_name), float("inf")))
            ]
        active_supervised_camera_names = [
            str(camera_name)
            for camera_name in candidate_supervised_camera_names
            if camera_name not in dropped_supervised_camera_names
        ]
        return active_supervised_camera_names, [str(camera_name) for camera_name in dropped_supervised_camera_names]

    def _build_supervised_view_quality_meta(self, selected_camera_names, selected_supervised_camera_names, geom_camera_names, geom):
        depth_conf = np.asarray(geom["depth_conf"], dtype=np.float32)
        score_by_camera = {}
        for local_idx, camera_name in enumerate(geom_camera_names):
            if camera_name in selected_supervised_camera_names:
                score_by_camera[str(camera_name)] = float(depth_conf[local_idx].mean())

        active_supervised_camera_names, dropped_supervised_camera_names = self._apply_view_quality_filter(
            selected_supervised_camera_names,
            score_by_camera,
            self.supervised_view_quality_filter,
        )
        conf_depth_active_supervised_camera_names, conf_depth_dropped_supervised_camera_names = self._apply_view_quality_filter(
            selected_supervised_camera_names,
            score_by_camera,
            self.conf_depth_view_quality_filter,
        )
        active_supervised_camera_names = [
            str(camera_name)
            for camera_name in selected_camera_names
            if camera_name in active_supervised_camera_names
        ]
        conf_depth_active_supervised_camera_names = [
            str(camera_name)
            for camera_name in selected_camera_names
            if camera_name in conf_depth_active_supervised_camera_names
        ]
        active_source_only_camera_names = [
            str(camera_name)
            for camera_name in selected_camera_names
            if camera_name not in active_supervised_camera_names
        ]

        return {
            "supervised_view_quality_filter": self.supervised_view_quality_filter,
            "candidate_supervised_camera_names": [str(camera_name) for camera_name in selected_supervised_camera_names],
            "active_supervised_camera_names": active_supervised_camera_names,
            "active_source_only_camera_names": active_source_only_camera_names,
            "dropped_supervised_camera_names": [str(camera_name) for camera_name in dropped_supervised_camera_names],
            "conf_depth_view_quality_filter": self.conf_depth_view_quality_filter,
            "conf_depth_active_supervised_camera_names": conf_depth_active_supervised_camera_names,
            "conf_depth_dropped_supervised_camera_names": [str(camera_name) for camera_name in conf_depth_dropped_supervised_camera_names],
            "supervised_view_quality_scores": {
                str(camera_name): float(score)
                for camera_name, score in score_by_camera.items()
            },
        }

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        del aspect_ratio  # geometry cache is already square and aligned to the model resolution

        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        entry = self.sequence_list[seq_index]
        if seq_name is not None and seq_name != entry["seq_name"]:
            raise ValueError("Explicit seq_name override is not supported for ZjuVggtGeomDataset.")

        geom = _merge_geom_payloads(entry.get("geom_paths") or [entry["geom_path"]])
        geom_camera_names = [str(cam_name) for cam_name in geom["cam_names"]]
        camera_to_local_idx = {camera: idx for idx, camera in enumerate(geom_camera_names)}
        available_views = len(geom_camera_names)
        source_view_pool_meta = self._resolve_source_view_pool_meta()
        available_candidate_camera_names = self._build_candidate_camera_names(
            entry,
            geom_camera_names,
            source_view_pool_meta["effective_source_view_pool"],
        )
        available_candidate_views = len(available_candidate_camera_names)
        if available_views < self.min_num_images:
            raise ValueError(f"Not enough cached views in {entry['geom_path']}: {available_views}")
        if img_per_seq is None:
            img_per_seq = available_views

        replace = self.allow_duplicate_img or (img_per_seq > available_candidate_views)
        selection_meta = {
            "source_policy": self.source_policy,
            "source_view_pool": source_view_pool_meta["effective_source_view_pool"],
            "requested_source_view_pool": source_view_pool_meta["requested_source_view_pool"],
            "source_view_pool_train_probability": float(source_view_pool_meta["source_view_pool_train_probability"]),
            "rawpool_candidate_pool_used": bool(source_view_pool_meta["rawpool_candidate_pool_used"]),
            "source_anchor_policy": self.source_anchor_policy,
            "min_supervised_views": self.min_supervised_views,
            "supervised_view_quality_filter": self.supervised_view_quality_filter,
            "selection_anchor_camera": None,
            "selected_camera_names": [],
            "selected_supervised_camera_names": [],
            "selected_source_only_camera_names": [],
            "available_candidate_view_count": int(available_candidate_views),
            "available_candidate_camera_names": list(available_candidate_camera_names),
        }
        if ids is None:
            if self.source_policy == "random" or replace:
                ids = np.random.choice(available_views, img_per_seq, replace=replace)
                selection_meta["selected_camera_names"] = [str(geom["cam_names"][int(local_idx)]) for local_idx in ids]
                selection_meta["selected_supervised_camera_names"] = list(selection_meta["selected_camera_names"])
            else:
                selected_camera_names, selection_meta = self._select_camera_names_with_source_policy(
                    entry,
                    geom_camera_names,
                    geom,
                    img_per_seq,
                    source_view_pool_meta,
                )
                ids = np.asarray([camera_to_local_idx.get(camera_name, -1) for camera_name in selected_camera_names], dtype=np.int64)
        ids = np.asarray(ids, dtype=np.int64)
        selected_camera_names = list(selection_meta["selected_camera_names"]) if selection_meta["selected_camera_names"] else [
            str(geom["cam_names"][int(local_idx)]) for local_idx in ids
        ]
        quality_meta = self._build_supervised_view_quality_meta(
            selected_camera_names,
            selection_meta["selected_supervised_camera_names"],
            geom_camera_names,
            geom,
        )
        active_supervised_camera_names = set(quality_meta["active_supervised_camera_names"])

        target_hw = tuple(int(v) for v in geom["depth"].shape[1:3])
        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        foreground_masks = []
        conf_depth_point_masks = []
        depth_conf_maps = []
        smpl_prior_masks = []
        smpl_prior_feature_maps = []
        human_prior_completion_masks = []
        human_prior_completion_depths = []
        human_prior_completion_world_points = []
        human_prior_completion_point_masks = []
        head_hair_region_masks = []
        head_hair_detail_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []
        conf_depth_active_supervised_camera_names = set(quality_meta["conf_depth_active_supervised_camera_names"])
        smpl_vertices_world = self._load_smpl_vertices(entry)

        for item_idx, cam_name in enumerate(selected_camera_names):
            local_idx = int(ids[item_idx]) if item_idx < len(ids) else -1
            has_cached_geom = local_idx >= 0
            is_supervised_camera = has_cached_geom and (cam_name in active_supervised_camera_names)
            stored_img_path = str(geom["img_paths"][local_idx]) if is_supervised_camera else ""
            image_path = _resolve_image_path(entry["seq_dir"], entry["frame_id"], cam_name, stored_img_path)
            image = read_image_cv2(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Failed to read image: {image_path}")
            original_size = np.array(image.shape[:2], dtype=np.int32)
            image = _resize_rgb_image(image, target_hw)

            if self.load_depth and is_supervised_camera:
                depth_map = np.asarray(geom["depth"][local_idx, ..., 0], dtype=np.float32)
                depth_conf = np.asarray(geom["depth_conf"][local_idx], dtype=np.float32)
            elif self.load_depth and not is_supervised_camera:
                depth_map = None
                depth_conf = None
            else:
                raise ValueError("ZjuVggtGeomDataset currently requires common_config.load_depth=True.")

            if self.use_foreground_mask:
                fg_mask = _load_mask(entry["seq_dir"], cam_name, entry["frame_id"], self.mask_source, target_hw)
            else:
                fg_mask = np.ones(target_hw, dtype=bool)
            raw_fg_mask = fg_mask.copy()

            if is_supervised_camera and self.min_depth_conf > 0:
                fg_mask = fg_mask & (depth_conf >= self.min_depth_conf)

            if is_supervised_camera:
                depth_map = np.where(fg_mask, depth_map, 0.0)

            if self.camera_source == "gt":
                camera = self.camera_store[entry["seq_name"]][cam_name]
                extri_opencv = np.asarray(camera["extrinsic"], dtype=np.float32)
                intri_opencv = _scale_intrinsic(camera["intrinsic"], original_size, target_hw).astype(np.float32)
            else:
                extri_opencv = np.asarray(geom["extrinsic"][local_idx], dtype=np.float32)
                intri_opencv = np.asarray(geom["intrinsic"][local_idx], dtype=np.float32)

            if smpl_vertices_world is not None:
                (
                    smpl_prior_mask,
                    smpl_prior_feature_map,
                    smpl_prior_sparse_mask,
                    smpl_prior_sparse_depth_map,
                    smpl_prior_sparse_world_points,
                ) = _project_smpl_vertices_to_feature_map(
                    smpl_vertices_world,
                    extri_opencv,
                    intri_opencv,
                    target_hw,
                    point_radius_px=self.smpl_prior_point_radius_px,
                    close_px=self.smpl_prior_close_px,
                    gaussian_sigma=self.smpl_prior_gaussian_sigma,
                )
            else:
                smpl_prior_mask = np.zeros(target_hw, dtype=bool)
                smpl_prior_feature_map = np.zeros(target_hw, dtype=np.float32)
                smpl_prior_sparse_mask = np.zeros(target_hw, dtype=bool)
                smpl_prior_sparse_depth_map = np.zeros(target_hw, dtype=np.float32)
                smpl_prior_sparse_world_points = np.zeros((*target_hw, 3), dtype=np.float32)
            human_prior_completion_mask, head_hair_region_mask, head_hair_detail_mask = _build_human_prior_masks(
                raw_fg_mask,
                smpl_prior_mask,
                completion_dilate_px=self.smpl_completion_dilate_px,
                head_hair_top_ratio=self.head_hair_top_ratio,
                head_hair_horizontal_expand_ratio=self.head_hair_horizontal_expand_ratio,
                head_hair_top_expand_ratio=self.head_hair_top_expand_ratio,
                head_hair_dilate_px=self.head_hair_dilate_px,
                head_hair_edge_band_px=self.head_hair_edge_band_px,
            )
            (
                human_prior_completion_point_mask,
                human_prior_completion_depth_map,
                human_prior_completion_world_point_map,
            ) = _densify_smpl_prior_geometry(
                smpl_prior_sparse_mask,
                smpl_prior_sparse_depth_map,
                smpl_prior_sparse_world_points,
                human_prior_completion_mask,
                max_fill_distance_px=self.smpl_completion_max_fill_px,
            )

            if is_supervised_camera:
                world_coords_points, cam_coords_points, point_mask = depth_to_world_coords_points(
                    depth_map, extri_opencv, intri_opencv
                )
            else:
                depth_map, world_coords_points, cam_coords_points, point_mask = _build_source_only_geometry(target_hw)

            if is_supervised_camera and cam_name in conf_depth_active_supervised_camera_names:
                conf_depth_point_mask = point_mask.astype(bool)
            else:
                conf_depth_point_mask = np.zeros(target_hw, dtype=bool)
            if is_supervised_camera and depth_conf is not None:
                depth_conf_map = depth_conf.astype(np.float32)
            else:
                depth_conf_map = np.zeros(target_hw, dtype=np.float32)

            images.append(image)
            depths.append(depth_map.astype(np.float32))
            cam_points.append(cam_coords_points.astype(np.float32))
            world_points.append(world_coords_points.astype(np.float32))
            point_masks.append(point_mask.astype(bool))
            foreground_masks.append(raw_fg_mask.astype(bool))
            conf_depth_point_masks.append(conf_depth_point_mask)
            depth_conf_maps.append(depth_conf_map)
            smpl_prior_masks.append(smpl_prior_mask.astype(bool))
            smpl_prior_feature_maps.append(smpl_prior_feature_map.astype(np.float32))
            human_prior_completion_masks.append(human_prior_completion_mask.astype(bool))
            human_prior_completion_depths.append(human_prior_completion_depth_map.astype(np.float32))
            human_prior_completion_world_points.append(human_prior_completion_world_point_map.astype(np.float32))
            human_prior_completion_point_masks.append(human_prior_completion_point_mask.astype(bool))
            head_hair_region_masks.append(head_hair_region_mask.astype(bool))
            head_hair_detail_masks.append(head_hair_detail_mask.astype(bool))
            extrinsics.append(extri_opencv.astype(np.float32))
            intrinsics.append(intri_opencv.astype(np.float32))
            image_paths.append(str(image_path))
            original_sizes.append(original_size)

        batch = {
            "seq_name": f"zju_{entry['seq_name']}_frame_{entry['frame_id']:06d}",
            "entry_seq_name": str(entry["seq_name"]),
            "entry_frame_id": int(entry["frame_id"]),
            "ids": ids,
            "frame_num": len(extrinsics),
            "available_view_count": int(available_views),
            "available_candidate_view_count": int(available_candidate_views),
            "camera_names": selected_camera_names,
            "geom_subdirs_present": list(entry.get("geom_subdirs_present", [])),
            "selection_sample_manifest_path": self.sample_manifest_path,
            "selection_sample_manifest_label": self.sample_manifest_label,
            "selection_sample_manifest_applied": bool(self.sample_manifest_applied),
            "selection_sample_manifest_use_entry_anchor": bool(self.sample_manifest_use_entry_anchor),
            "selection_sample_manifest_use_entry_camera_set": bool(self.sample_manifest_use_entry_camera_set),
            "selection_anchor_camera": selection_meta["selection_anchor_camera"],
            "selection_source_policy": selection_meta["source_policy"],
            "selection_source_view_pool": selection_meta["source_view_pool"],
            "selection_requested_source_view_pool": selection_meta["requested_source_view_pool"],
            "selection_source_view_pool_train_probability": float(selection_meta["source_view_pool_train_probability"]),
            "selection_rawpool_candidate_pool_used": bool(selection_meta["rawpool_candidate_pool_used"]),
            "selection_source_anchor_policy": selection_meta["source_anchor_policy"],
            "selection_min_supervised_views": int(selection_meta["min_supervised_views"]),
            "selection_supervised_view_quality_filter": quality_meta["supervised_view_quality_filter"],
            "selection_conf_depth_view_quality_filter": quality_meta["conf_depth_view_quality_filter"],
            "candidate_supervised_camera_names": list(quality_meta["candidate_supervised_camera_names"]),
            "supervised_camera_names": list(quality_meta["active_supervised_camera_names"]),
            "source_only_camera_names": list(quality_meta["active_source_only_camera_names"]),
            "dropped_supervised_camera_names": list(quality_meta["dropped_supervised_camera_names"]),
            "conf_depth_camera_names": list(quality_meta["conf_depth_active_supervised_camera_names"]),
            "conf_depth_dropped_camera_names": list(quality_meta["conf_depth_dropped_supervised_camera_names"]),
            "supervised_view_quality_scores": dict(quality_meta["supervised_view_quality_scores"]),
            "available_candidate_camera_names": list(selection_meta["available_candidate_camera_names"]),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "conf_depth_point_masks": conf_depth_point_masks,
            "depth_conf_maps": depth_conf_maps,
            "foreground_masks": foreground_masks,
            "smpl_prior_masks": smpl_prior_masks,
            "smpl_prior_feature_maps": smpl_prior_feature_maps,
            "human_prior_completion_masks": human_prior_completion_masks,
            "human_prior_completion_depths": human_prior_completion_depths,
            "human_prior_completion_world_points": human_prior_completion_world_points,
            "human_prior_completion_point_masks": human_prior_completion_point_masks,
            "head_hair_region_masks": head_hair_region_masks,
            "head_hair_detail_masks": head_hair_detail_masks,
            "image_paths": image_paths,
            "original_sizes": original_sizes,
        }
        return batch
