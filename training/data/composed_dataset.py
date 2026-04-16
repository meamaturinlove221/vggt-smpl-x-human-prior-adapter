# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC

from hydra.utils import instantiate
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import bisect
from .dataset_util import *
from .track_util import *
from .augmentation import get_image_augmentation


class ComposedDataset(Dataset, ABC):
    """
    Composes multiple base datasets and applies common configurations.

    This dataset provides a flexible way to combine multiple base datasets while
    applying shared augmentations, track generation, and other processing steps.
    It handles image normalization, tensor conversion, and other preparations
    needed for training computer vision models with sequences of images.
    """
    def __init__(self, dataset_configs: dict, common_config: dict, **kwargs):
        """
        Initializes the ComposedDataset.

        Args:
            dataset_configs (dict): List of Hydra configurations for base datasets.
            common_config (dict): Shared configurations (augs, tracks, mode, etc.).
            **kwargs: Additional arguments (unused).
        """
        base_dataset_list = []

        # Instantiate each base dataset with common configuration
        for baseset_dict in dataset_configs:
            baseset = instantiate(baseset_dict, common_conf=common_config)
            base_dataset_list.append(baseset)

        # Use custom concatenation class that supports tuple indexing
        self.base_dataset = TupleConcatDataset(base_dataset_list, common_config)

        # --- Augmentation Settings ---
        # Controls whether to apply identical color jittering across all frames in a sequence
        self.cojitter = common_config.augs.cojitter
        # Probability of using shared jitter vs. frame-specific jitter
        self.cojitter_ratio = common_config.augs.cojitter_ratio
        # Initialize image augmentations (color jitter, grayscale, gaussian blur)
        self.image_aug = get_image_augmentation(
            color_jitter=common_config.augs.color_jitter,
            gray_scale=common_config.augs.gray_scale,
            gau_blur=common_config.augs.gau_blur,
        )

        # --- Optional Fixed Settings (useful for debugging) ---
        # Force each sequence to have exactly this many images (if > 0)
        self.fixed_num_images = common_config.fix_img_num
        # Force a specific aspect ratio for all images
        self.fixed_aspect_ratio = common_config.fix_aspect_ratio

        # --- Track Settings ---
        # Whether to include point tracks in the output
        self.load_track = common_config.load_track
        # Number of point tracks to include per sequence
        self.track_num = common_config.track_num

        # --- Mode Settings ---
        # Whether the dataset is being used for training (affects augmentations)
        self.training = common_config.training
        self.common_config = common_config

        self.total_samples = len(self.base_dataset)

    def __len__(self):
        """Returns the total number of sequences in the dataset."""
        return self.total_samples


    def __getitem__(self, idx_tuple):
        """
        Retrieves a data sample (sequence) from the dataset.

        Loads raw data, converts to PyTorch tensors, applies augmentations,
        and prepares tracks if enabled.

        Args:
            idx_tuple (tuple): a tuple of (seq_idx, num_images, aspect_ratio)

        Returns:
            dict: A dictionary containing the sequence data (images, poses, tracks, etc.).
        """
        # If fixed settings are provided, override the tuple values
        if self.fixed_num_images > 0:
            seq_idx = idx_tuple[0] if isinstance(idx_tuple, tuple) else idx_tuple
            idx_tuple = (seq_idx, self.fixed_num_images, self.fixed_aspect_ratio)

        # Retrieve the raw data batch from the appropriate base dataset
        batch = self.base_dataset[idx_tuple]
        seq_name = batch["seq_name"]

        # --- Data Conversion and Preparation ---
        # Convert numpy arrays to tensors
        images = torch.from_numpy(np.stack(batch["images"]).astype(np.float32)).contiguous()
        # Normalize images from [0, 255] to [0, 1]
        images = images.permute(0,3,1,2).to(torch.get_default_dtype()).div(255)

        # Convert other data to tensors with appropriate types
        depths = torch.from_numpy(np.stack(batch["depths"]).astype(np.float32))
        extrinsics = torch.from_numpy(np.stack(batch["extrinsics"]).astype(np.float32))
        intrinsics = torch.from_numpy(np.stack(batch["intrinsics"]).astype(np.float32))
        cam_points = torch.from_numpy(np.stack(batch["cam_points"]).astype(np.float32))
        world_points = torch.from_numpy(np.stack(batch["world_points"]).astype(np.float32))
        point_masks = torch.from_numpy(np.stack(batch["point_masks"])) # Mask indicating valid depths / world points / cam points per frame
        conf_depth_point_masks = None
        if "conf_depth_point_masks" in batch and batch["conf_depth_point_masks"] is not None:
            conf_depth_point_masks = torch.from_numpy(np.stack(batch["conf_depth_point_masks"]).astype(bool))
        depth_conf_maps = None
        if "depth_conf_maps" in batch and batch["depth_conf_maps"] is not None:
            depth_conf_maps = torch.from_numpy(np.stack(batch["depth_conf_maps"]).astype(np.float32))
        ids = torch.from_numpy(batch["ids"])    # Frame indices sampled from the original sequence
        foreground_masks = None
        if "foreground_masks" in batch and batch["foreground_masks"] is not None:
            foreground_masks = torch.from_numpy(np.stack(batch["foreground_masks"]).astype(bool))
        smpl_prior_masks = None
        if "smpl_prior_masks" in batch and batch["smpl_prior_masks"] is not None:
            smpl_prior_masks = torch.from_numpy(np.stack(batch["smpl_prior_masks"]).astype(bool))
        smpl_prior_feature_maps = None
        if "smpl_prior_feature_maps" in batch and batch["smpl_prior_feature_maps"] is not None:
            smpl_prior_feature_maps = torch.from_numpy(np.stack(batch["smpl_prior_feature_maps"]).astype(np.float32))
        human_prior_completion_masks = None
        if "human_prior_completion_masks" in batch and batch["human_prior_completion_masks"] is not None:
            human_prior_completion_masks = torch.from_numpy(
                np.stack(batch["human_prior_completion_masks"]).astype(bool)
            )
        human_prior_completion_depths = None
        if "human_prior_completion_depths" in batch and batch["human_prior_completion_depths"] is not None:
            human_prior_completion_depths = torch.from_numpy(
                np.stack(batch["human_prior_completion_depths"]).astype(np.float32)
            )
        human_prior_completion_world_points = None
        if "human_prior_completion_world_points" in batch and batch["human_prior_completion_world_points"] is not None:
            human_prior_completion_world_points = torch.from_numpy(
                np.stack(batch["human_prior_completion_world_points"]).astype(np.float32)
            )
        human_prior_completion_point_masks = None
        if "human_prior_completion_point_masks" in batch and batch["human_prior_completion_point_masks"] is not None:
            human_prior_completion_point_masks = torch.from_numpy(
                np.stack(batch["human_prior_completion_point_masks"]).astype(bool)
            )
        head_hair_region_masks = None
        if "head_hair_region_masks" in batch and batch["head_hair_region_masks"] is not None:
            head_hair_region_masks = torch.from_numpy(np.stack(batch["head_hair_region_masks"]).astype(bool))
        head_hair_detail_masks = None
        if "head_hair_detail_masks" in batch and batch["head_hair_detail_masks"] is not None:
            head_hair_detail_masks = torch.from_numpy(np.stack(batch["head_hair_detail_masks"]).astype(bool))


        # --- Apply Color Augmentation (training mode only) ---
        if self.training and self.image_aug is not None:
            if self.cojitter and random.random() > self.cojitter_ratio:
                # Apply the same color jittering transformation to all frames
                images = self.image_aug(images)
            else:
                # Apply different color jittering to each frame individually
                for aug_img_idx in range(len(images)):
                    images[aug_img_idx] = self.image_aug(images[aug_img_idx])


        # --- Prepare Final Sample Dictionary ---
        sample = {
            "seq_name": seq_name,
            "ids": ids,
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
        }
        if conf_depth_point_masks is not None:
            sample["conf_depth_point_masks"] = conf_depth_point_masks
        if depth_conf_maps is not None:
            sample["depth_conf_maps"] = depth_conf_maps
        if foreground_masks is not None:
            sample["foreground_masks"] = foreground_masks
        if smpl_prior_masks is not None:
            sample["smpl_prior_masks"] = smpl_prior_masks
        if smpl_prior_feature_maps is not None:
            sample["smpl_prior_feature_maps"] = smpl_prior_feature_maps
        if human_prior_completion_masks is not None:
            sample["human_prior_completion_masks"] = human_prior_completion_masks
        if human_prior_completion_depths is not None:
            sample["human_prior_completion_depths"] = human_prior_completion_depths
        if human_prior_completion_world_points is not None:
            sample["human_prior_completion_world_points"] = human_prior_completion_world_points
        if human_prior_completion_point_masks is not None:
            sample["human_prior_completion_point_masks"] = human_prior_completion_point_masks
        if head_hair_region_masks is not None:
            sample["head_hair_region_masks"] = head_hair_region_masks
        if head_hair_detail_masks is not None:
            sample["head_hair_detail_masks"] = head_hair_detail_masks
        if "selection_anchor_camera" in batch:
            anchor_camera = batch["selection_anchor_camera"]
            sample["selection_anchor_camera"] = "" if anchor_camera is None else str(anchor_camera)
            anchor_view_index = -1
            camera_names = batch.get("camera_names")
            if anchor_camera is not None and camera_names is not None:
                try:
                    anchor_view_index = int(list(camera_names).index(str(anchor_camera)))
                except ValueError:
                    anchor_view_index = -1
            sample["selection_anchor_view_index"] = torch.tensor(
                int(anchor_view_index), dtype=torch.int64
            )
        if "selection_anchor_camera" in batch and "supervised_view_quality_scores" in batch:
            anchor_camera = batch["selection_anchor_camera"]
            anchor_quality_score = None
            if anchor_camera is not None:
                anchor_quality_score = batch["supervised_view_quality_scores"].get(str(anchor_camera))
            if anchor_quality_score is None:
                anchor_quality_score = float("nan")
            sample["selection_anchor_quality_score"] = torch.tensor(
                float(anchor_quality_score), dtype=torch.float32
            )
        if "selection_sample_manifest_applied" in batch:
            sample["selection_sample_manifest_applied"] = torch.tensor(
                bool(batch["selection_sample_manifest_applied"]), dtype=torch.bool
            )
        if "selection_sample_manifest_label" in batch:
            sample["selection_sample_manifest_label"] = str(
                batch.get("selection_sample_manifest_label") or ""
            )

        # --- Track Processing (if enabled) ---
        if self.load_track:
            if batch["tracks"] is not None:
                # Use pre-computed tracks from the dataset
                tracks = torch.from_numpy(np.stack(batch["tracks"]).astype(np.float32))
                track_vis_mask = torch.from_numpy(np.stack(batch["track_masks"]).astype(bool))

                # Sample a subset of tracks randomly
                valid_indices = torch.where(track_vis_mask[0])[0]
                if len(valid_indices) >= self.track_num:
                    # If we have enough tracks, sample without replacement
                    sampled_indices = valid_indices[torch.randperm(len(valid_indices))][:self.track_num]
                else:
                    # If not enough tracks, sample with replacement (allow duplicates)
                    sampled_indices = valid_indices[torch.randint(0, len(valid_indices),
                                                    (self.track_num,),
                                                    dtype=torch.int64,
                                                    device=valid_indices.device)]

                # Extract the sampled tracks and their masks
                tracks = tracks[:, sampled_indices, :]
                track_vis_mask = track_vis_mask[:, sampled_indices]
                track_positive_mask = torch.ones(track_vis_mask.shape[1]).bool()

            else:
                # Generate tracks on-the-fly using depth information
                # This creates synthetic tracks based on the 3D information available
                tracks, track_vis_mask, track_positive_mask = build_tracks_by_depth(
                    extrinsics, intrinsics, world_points, depths, point_masks, images,
                    target_track_num=self.track_num, seq_name=seq_name
                )

            # Add track information to the sample dictionary
            sample["tracks"] = tracks
            sample["track_vis_mask"] = track_vis_mask
            sample["track_positive_mask"] = track_positive_mask

        return sample


class TupleConcatDataset(ConcatDataset):
    """
    A custom ConcatDataset that supports indexing with a tuple.

    Standard PyTorch ConcatDataset only accepts an integer index. This class extends
    that functionality to allow passing a tuple like (sample_idx, num_images, aspect_ratio),
    where the first element is used to determine which sample to fetch, and the full
    tuple is passed down to the selected dataset's __getitem__ method.

    It also supports an option to randomly sample across all datasets, ignoring the
    provided index. This is useful during training when shuffling the entire dataset
    might cause memory issues due to duplicating dictionaries. If doing this, you can
    set pytorch's dataloader shuffle to False.
    """
    def __init__(self, datasets, common_config):
        """
        Initialize the TupleConcatDataset.

        Args:
            datasets (iterable): An iterable of PyTorch Dataset objects to concatenate.
            common_config (dict): Common configuration dict, used to check for random sampling.
        """
        super().__init__(datasets)
        # If True, ignores the input index and samples randomly across all datasets
        # This provides an alternative to dataloader shuffling for large datasets
        self.inside_random = common_config.inside_random

    def __getitem__(self, idx):
        """
        Retrieves an item using either an integer index or a tuple index.

        Args:
            idx (int or tuple): The index. If tuple, the first element is the sequence
                               index across the concatenated datasets, and the rest are
                               passed down. If int, it's treated as the sequence index.

        Returns:
            The item returned by the underlying dataset's __getitem__ method.

        Raises:
            ValueError: If the index is out of range or the tuple doesn't have exactly 3 elements.
        """
        idx_tuple = None
        if isinstance(idx, tuple):
            idx_tuple = idx
            idx = idx_tuple[0]  # Extract the sequence index

        # Override index with random value if inside_random is enabled
        if self.inside_random:
            total_len = self.cumulative_sizes[-1]
            idx = random.randint(0, total_len - 1)

        # Handle negative indices
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        # Find which dataset the index belongs to
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # Create the tuple to pass to the underlying dataset
        if len(idx_tuple) == 3:
            idx_tuple = (sample_idx,) + idx_tuple[1:]
        else:
            raise ValueError("Tuple index must have exactly three elements")

        # Pass the modified tuple to the appropriate dataset
        return self.datasets[dataset_idx][idx_tuple]
