from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from data.base_dataset import BaseDataset


class DNA4K4DPseudoDataset(BaseDataset):
    """Read exported single-case 4K4D/VGGT pseudo training bundles.

    The historical SMPL-X prior configs reference this dataset, but the loader
    was missing in the current checkout.  It intentionally reads only local
    `inputs.npz`, `targets.npz`, and optional manifest files from prebuilt case
    directories; it does not create new annotations or depend on external hand
    / hair routes.
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        len_train: int = 1,
        len_test: int = 1,
        case_roots: list[str] | tuple[str, ...] | None = None,
        **_: Any,
    ) -> None:
        super().__init__(common_conf)
        self.split = str(split)
        self.case_roots = [Path(root).expanduser() for root in (case_roots or [])]
        if not self.case_roots:
            raise ValueError("DNA4K4DPseudoDataset requires at least one case_root.")
        missing = [str(root) for root in self.case_roots if not (root / "inputs.npz").is_file() or not (root / "targets.npz").is_file()]
        if missing:
            raise FileNotFoundError("Missing inputs.npz/targets.npz in case_roots: " + ", ".join(missing))
        self.len_train = int(max(1, len_train))
        self.len_test = int(max(1, len_test))
        self.training = bool(getattr(common_conf, "training", self.split == "train"))

    def __len__(self) -> int:
        return self.len_train if self.split == "train" else self.len_test

    @staticmethod
    def _load_npz(path: Path) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as payload:
            return {key: np.array(payload[key]) for key in payload.files}

    @staticmethod
    def _optional_stack(targets: dict[str, Any], names: tuple[str, ...]) -> np.ndarray | None:
        for name in names:
            if name in targets:
                return np.asarray(targets[name])
        return None

    @staticmethod
    def _camera_names(inputs: dict[str, Any], view_count: int) -> list[str]:
        if "camera_ids" in inputs:
            return [str(x) for x in np.asarray(inputs["camera_ids"]).tolist()]
        return [f"{idx:02d}" for idx in range(view_count)]

    def _select_case_root(self, seq_index: int | None) -> Path:
        if seq_index is None:
            seq_index = 0
        return self.case_roots[int(seq_index) % len(self.case_roots)]

    def _select_view_indices(self, view_count: int, img_per_seq: int) -> list[int]:
        img_per_seq = int(img_per_seq or view_count)
        img_per_seq = max(1, min(view_count, img_per_seq))
        if img_per_seq == view_count:
            return list(range(view_count))
        if bool(getattr(self.common_config, "allow_duplicate_img", False)):
            return [idx % view_count for idx in range(img_per_seq)]
        if self.training:
            return sorted(random.sample(range(view_count), img_per_seq))
        return list(range(img_per_seq))

    def get_data(self, seq_index=None, seq_name=None, ids=None, img_per_seq=None, aspect_ratio=1.0):
        del aspect_ratio
        case_root = self._select_case_root(seq_index)
        inputs = self._load_npz(case_root / "inputs.npz")
        targets = self._load_npz(case_root / "targets.npz")
        images = np.asarray(inputs["images"], dtype=np.uint8)
        view_count = int(images.shape[0])
        if ids is None:
            selected_img_per_seq = int(img_per_seq or getattr(self.common_config, "fix_img_num", -1))
            if selected_img_per_seq <= 0:
                img_nums = getattr(self.common_config, "img_nums", [view_count, view_count])
                selected_img_per_seq = int(img_nums[0])
            view_indices = self._select_view_indices(view_count, selected_img_per_seq)
        else:
            view_indices = [int(i) for i in ids]

        idx = np.asarray(view_indices, dtype=np.int64)
        camera_names_all = self._camera_names(inputs, view_count)
        camera_names = [camera_names_all[i] for i in idx.tolist()]
        point_masks = np.asarray(inputs.get("point_masks", targets.get("teacher_mask", np.isfinite(targets["depths"]))), dtype=bool)

        output: dict[str, Any] = {
            "seq_name": str(seq_name or case_root.name),
            "ids": idx,
            "images": [images[i] for i in idx],
            "depths": [np.asarray(targets["depths"][i], dtype=np.float32) for i in idx],
            "extrinsics": [np.asarray(targets["extrinsics"][i], dtype=np.float32) for i in idx],
            "intrinsics": [np.asarray(targets["intrinsics"][i], dtype=np.float32) for i in idx],
            "cam_points": [np.asarray(targets["cam_points"][i], dtype=np.float32) for i in idx],
            "world_points": [np.asarray(targets["world_points"][i], dtype=np.float32) for i in idx],
            "point_masks": [np.asarray(point_masks[i], dtype=bool) for i in idx],
            "tracks": None,
            "track_masks": None,
            "camera_names": camera_names,
            "selection_anchor_camera": camera_names[0] if camera_names else None,
        }

        soft_alpha = inputs.get("soft_alpha")
        prior_mask = inputs.get("prior_mask")
        teacher_mask = targets.get("teacher_mask")
        foreground = soft_alpha > 0.05 if soft_alpha is not None else (teacher_mask if teacher_mask is not None else prior_mask)
        if foreground is not None:
            output["foreground_masks"] = [np.asarray(foreground[i], dtype=bool) for i in idx]
        if prior_mask is not None:
            output["smpl_prior_masks"] = [np.asarray(prior_mask[i], dtype=bool) for i in idx]
            output["human_prior_completion_masks"] = [np.asarray(prior_mask[i], dtype=bool) for i in idx]
        if "prior_maps" in inputs:
            maps = np.asarray(inputs["prior_maps"], dtype=np.float32)
            output["smpl_vertex_feature_maps"] = [maps[i] for i in idx]
            output["smpl_prior_feature_maps"] = [maps[i] for i in idx]
        if "prior_summary_tokens" in inputs:
            summary = np.asarray(inputs["prior_summary_tokens"], dtype=np.float32)
            output["smpl_summary_tokens"] = summary[idx]

        optional_masks = {
            "conf_depth_point_masks": ("teacher_mask", "smplx_native_visible_mask"),
            "human_prior_completion_point_masks": ("smplx_native_visible_mask", "teacher_mask"),
            "head_hair_region_masks": ("head_hair_region_mask", "head_face_mask", "face_mask"),
            "head_hair_detail_masks": ("hairline_mask", "face_mask", "head_face_mask"),
        }
        for out_key, source_names in optional_masks.items():
            arr = self._optional_stack(targets, source_names)
            if arr is not None:
                output[out_key] = [np.asarray(arr[i], dtype=bool) for i in idx]

        if "depth_conf" in targets:
            output["depth_conf_maps"] = [np.asarray(targets["depth_conf"][i], dtype=np.float32) for i in idx]
        if "prior_depths" in targets:
            output["human_prior_completion_depths"] = [np.asarray(targets["prior_depths"][i], dtype=np.float32) for i in idx]
        if "prior_points" in targets:
            output["human_prior_completion_world_points"] = [np.asarray(targets["prior_points"][i], dtype=np.float32) for i in idx]

        # Preserve native SMPL-X supervision masks/targets for SMPLXNativePriorLoss.
        for key in (
            "prior_depths",
            "prior_points",
            "prior_normals",
            "teacher_mask",
            "smplx_bodyhand_anchor_mask",
            "smplx_body_anchor_mask",
            "smplx_hand_anchor_mask",
            "smplx_left_hand_anchor_mask",
            "smplx_right_hand_anchor_mask",
            "smplx_native_visible_mask",
        ):
            if key in targets:
                output[key] = [np.asarray(targets[key][i]) for i in idx]
        if "prior_mask" in inputs:
            output["prior_mask"] = [np.asarray(inputs["prior_mask"][i], dtype=bool) for i in idx]
        if "prior_maps" in inputs:
            output["prior_maps"] = [np.asarray(inputs["prior_maps"][i], dtype=np.float32) for i in idx]

        return output
