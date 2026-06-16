from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import torch

from loss import MultitaskLoss


DEFAULT_NATIVE_HUMAN_PRIOR = {
    "weight": 0.10,
    "depth": {
        "gamma": 1.0,
        "alpha": 0.0,
        "gradient_loss_fn": None,
        "valid_range": 0.96,
        "supervise_conf": False,
    },
    "point": {
        "gamma": 1.0,
        "alpha": 0.0,
        "gradient_loss_fn": None,
        "valid_range": 0.96,
        "supervise_conf": False,
    },
    "normal": {
        "gamma": 1.0,
        "alpha": 0.0,
        "valid_range": 0.96,
        "supervise_conf": False,
    },
    "depth_point": {
        "consistency_weight": 0.05,
        "valid_range": 0.96,
    },
    "smplx_weak_anchor": {
        "body_weight": 0.25,
        "hand_weight": 0.50,
        "depth_loss_weight": 0.25,
        "point_loss_weight": 0.50,
        "gamma": 1.0,
        "alpha": 0.0,
        "valid_range": 0.96,
        "min_pixels": 64,
        "supervise_conf": False,
        "use_separate_hand_masks": True,
        "exclude_head_roi": True,
        "exclude_face_roi": True,
        "exclude_hairline_roi": True,
        "exclude_ear_band_roi": True,
    },
}


def _to_plain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    if overlay is None:
        return merged
    for key, value in overlay.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


class SMPLXNativePriorLoss(MultitaskLoss):
    """Compatibility wrapper for V15 SMPL-X-native prior experiments.

    The wrapper deliberately reuses the existing MultitaskLoss and
    compute_human_prior_loss functions. It only supplies native-prior defaults
    and optional key validation so the training config can point to a separate
    loss target without modifying the shared loss module.
    """

    def __init__(
        self,
        camera=None,
        depth=None,
        point=None,
        track=None,
        human_prior=None,
        native_prior=None,
        **kwargs,
    ):
        native_prior = _to_plain(native_prior) or {}
        human_prior = _to_plain(human_prior)

        use_default_human_prior = bool(native_prior.get("use_default_human_prior", False))
        if human_prior is None and use_default_human_prior:
            human_prior = deepcopy(DEFAULT_NATIVE_HUMAN_PRIOR)
        elif human_prior is not None and use_default_human_prior:
            human_prior = _deep_merge(DEFAULT_NATIVE_HUMAN_PRIOR, human_prior)

        super().__init__(
            camera=camera,
            depth=depth,
            point=point,
            track=track,
            human_prior=human_prior,
            **kwargs,
        )
        self.native_prior = native_prior
        self.strict_required_keys = bool(native_prior.get("strict_required_keys", False))
        self.required_batch_keys = tuple(str(key) for key in native_prior.get("required_batch_keys", ()))
        self.required_prediction_keys = tuple(
            str(key) for key in native_prior.get("required_prediction_keys", ())
        )

    @staticmethod
    def _zero_metric(loss_dict: Mapping[str, Any], predictions: Mapping[str, Any]) -> torch.Tensor:
        objective = loss_dict.get("objective")
        if torch.is_tensor(objective):
            return objective.detach() * 0.0
        for value in predictions.values():
            if torch.is_tensor(value):
                return value.sum() * 0.0
        return torch.zeros(())

    def _find_missing_keys(self, predictions: Mapping[str, Any], batch: Mapping[str, Any]) -> list[str]:
        missing = [f"batch.{key}" for key in self.required_batch_keys if key not in batch]
        missing.extend(f"predictions.{key}" for key in self.required_prediction_keys if key not in predictions)
        return missing

    def forward(self, predictions, batch) -> torch.Tensor:
        missing = self._find_missing_keys(predictions, batch)
        if missing and self.strict_required_keys:
            raise KeyError("SMPLXNativePriorLoss missing required keys: " + ", ".join(missing))

        loss_dict = super().forward(predictions, batch)
        zero = self._zero_metric(loss_dict, predictions)
        loss_dict["loss_smplx_native_missing_key_count"] = zero + float(len(missing))
        return loss_dict
