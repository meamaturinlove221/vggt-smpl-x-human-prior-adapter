# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import os.path as osp
import random
import tempfile
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from data.base_dataset import BaseDataset
from data.dataset_util import depth_to_world_coords_points, read_image_cv2


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


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
    use_temp_copy = (not str(intri_path).isascii()) or (not str(extri_path).isascii())
    if use_temp_copy:
        temp_root = Path(tempfile.gettempdir()) / "vggt_zju_yaml_train"
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
        if source_policy not in ("random", "nearest_ring", "uniform_ring", "nearest_plus_uniform_tail"):
            raise ValueError(
                "source_policy must be one of 'random', 'nearest_ring', 'uniform_ring', "
                "or 'nearest_plus_uniform_tail'."
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

        self.camera_store = {}
        self.camera_ring_orders = {}
        self.raw_source_camera_pools = {}
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

    def _select_anchor_camera(self, geom_camera_names, geom):
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

    def _select_camera_names_with_source_policy(self, entry, geom_camera_names, geom, img_per_seq, source_view_pool_meta):
        effective_source_view_pool = source_view_pool_meta["effective_source_view_pool"]
        candidate_camera_names = self._build_candidate_camera_names(
            entry,
            geom_camera_names,
            effective_source_view_pool,
        )
        geom_camera_set = set(geom_camera_names)
        if img_per_seq >= len(candidate_camera_names):
            selected_camera_names = list(candidate_camera_names)
            if selected_camera_names:
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

        anchor_camera = self._select_anchor_camera(geom_camera_names, geom)
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
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []
        conf_depth_active_supervised_camera_names = set(quality_meta["conf_depth_active_supervised_camera_names"])

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
            extrinsics.append(extri_opencv.astype(np.float32))
            intrinsics.append(intri_opencv.astype(np.float32))
            image_paths.append(str(image_path))
            original_sizes.append(original_size)

        batch = {
            "seq_name": f"zju_{entry['seq_name']}_frame_{entry['frame_id']:06d}",
            "ids": ids,
            "frame_num": len(extrinsics),
            "available_view_count": int(available_views),
            "available_candidate_view_count": int(available_candidate_views),
            "camera_names": selected_camera_names,
            "geom_subdirs_present": list(entry.get("geom_subdirs_present", [])),
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
            "image_paths": image_paths,
            "original_sizes": original_sizes,
        }
        return batch
