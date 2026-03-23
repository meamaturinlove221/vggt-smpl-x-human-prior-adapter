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
        if split not in ("train", "test"):
            raise ValueError(f"Invalid split: {split}")

        self.zju_root = Path(ZJU_DIR)
        self.seq_names = _ensure_list(seq_names) or ["CoreView_390"]
        self.geom_subdir = geom_subdir
        self.holdout_stride = int(holdout_stride)
        self.camera_source = camera_source
        self.mask_source = mask_source
        self.use_foreground_mask = bool(use_foreground_mask)
        self.min_depth_conf = float(min_depth_conf)
        self.min_num_images = int(min_num_images)

        self.camera_store = {}
        self.sequence_list = []

        for seq_name in self.seq_names:
            seq_dir = self.zju_root / seq_name
            geom_dir = seq_dir / self.geom_subdir
            if not seq_dir.is_dir():
                raise FileNotFoundError(f"ZJU sequence directory not found: {seq_dir}")
            if not geom_dir.is_dir():
                raise FileNotFoundError(f"geometry cache directory not found: {geom_dir}")

            self.camera_store[seq_name] = _load_zju_cameras(seq_dir)
            geom_paths = sorted(geom_dir.glob("frame_*.npz"))
            for geom_path in geom_paths:
                frame_id = int(geom_path.stem.split("_")[-1])
                if self.holdout_stride > 0:
                    is_val = (frame_id % self.holdout_stride) == 0
                    if split == "train" and is_val:
                        continue
                    if split == "test" and not is_val:
                        continue
                self.sequence_list.append(
                    {
                        "seq_name": seq_name,
                        "frame_id": frame_id,
                        "geom_path": geom_path,
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

        geom = np.load(entry["geom_path"], allow_pickle=True)
        available_views = len(geom["cam_names"])
        if available_views < self.min_num_images:
            raise ValueError(f"Not enough cached views in {entry['geom_path']}: {available_views}")
        if img_per_seq is None:
            img_per_seq = available_views

        replace = self.allow_duplicate_img or (img_per_seq > available_views)
        if ids is None:
            ids = np.random.choice(available_views, img_per_seq, replace=replace)
        ids = np.asarray(ids, dtype=np.int64)

        target_hw = tuple(int(v) for v in geom["depth"].shape[1:3])
        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []

        for local_idx in ids:
            cam_name = str(geom["cam_names"][local_idx])
            stored_img_path = str(geom["img_paths"][local_idx])
            image_path = _resolve_image_path(entry["seq_dir"], entry["frame_id"], cam_name, stored_img_path)
            image = read_image_cv2(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Failed to read image: {image_path}")
            original_size = np.array(image.shape[:2], dtype=np.int32)
            image = _resize_rgb_image(image, target_hw)

            if self.load_depth:
                depth_map = np.asarray(geom["depth"][local_idx, ..., 0], dtype=np.float32)
                depth_conf = np.asarray(geom["depth_conf"][local_idx], dtype=np.float32)
            else:
                raise ValueError("ZjuVggtGeomDataset currently requires common_config.load_depth=True.")

            if self.use_foreground_mask:
                fg_mask = _load_mask(entry["seq_dir"], cam_name, entry["frame_id"], self.mask_source, target_hw)
            else:
                fg_mask = np.ones(target_hw, dtype=bool)

            if self.min_depth_conf > 0:
                fg_mask = fg_mask & (depth_conf >= self.min_depth_conf)

            depth_map = np.where(fg_mask, depth_map, 0.0)

            if self.camera_source == "gt":
                camera = self.camera_store[entry["seq_name"]][cam_name]
                extri_opencv = np.asarray(camera["extrinsic"], dtype=np.float32)
                intri_opencv = _scale_intrinsic(camera["intrinsic"], original_size, target_hw).astype(np.float32)
            else:
                extri_opencv = np.asarray(geom["extrinsic"][local_idx], dtype=np.float32)
                intri_opencv = np.asarray(geom["intrinsic"][local_idx], dtype=np.float32)

            world_coords_points, cam_coords_points, point_mask = depth_to_world_coords_points(
                depth_map, extri_opencv, intri_opencv
            )

            images.append(image)
            depths.append(depth_map.astype(np.float32))
            cam_points.append(cam_coords_points.astype(np.float32))
            world_points.append(world_coords_points.astype(np.float32))
            point_masks.append(point_mask.astype(bool))
            extrinsics.append(extri_opencv.astype(np.float32))
            intrinsics.append(intri_opencv.astype(np.float32))
            image_paths.append(str(image_path))
            original_sizes.append(original_size)

        batch = {
            "seq_name": f"zju_{entry['seq_name']}_frame_{entry['frame_id']:06d}",
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "image_paths": image_paths,
            "original_sizes": original_sizes,
        }
        return batch
