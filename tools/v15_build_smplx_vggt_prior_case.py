from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.prepare_4k4d_prior_training_case import _build_channel_group_meta  # noqa: E402


CHANNEL_POLICIES = ("smplx_native", "smplx_only", "adapter_compatible")
PRIOR_MASK_SOURCES = ("native_visible", "existing")
ANCHOR_MASK_KEYS = (
    "smplx_bodyhand_anchor_mask",
    "smplx_body_anchor_mask",
    "smplx_hand_anchor_mask",
    "smplx_left_hand_anchor_mask",
    "smplx_right_hand_anchor_mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a V15 SMPL-X-native VGGT prior training case from an existing "
            "case bundle. The output keeps the current DNA4K4DPseudoDataset and "
            "HumanPriorAdapter key contract: prior_maps, prior_summary_tokens, "
            "prior_depths, prior_points, teacher_mask, and SMPL-X anchor masks."
        )
    )
    parser.add_argument("--template-case-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case-id", help="Override case_id in the output manifest.")
    parser.add_argument(
        "--channel-policy",
        choices=CHANNEL_POLICIES,
        default="smplx_native",
        help=(
            "smplx_native keeps silhouette plus smplx_* dense channels; smplx_only "
            "keeps only smplx_* dense channels; adapter_compatible preserves all "
            "template channels."
        ),
    )
    parser.add_argument(
        "--prior-mask-source",
        choices=PRIOR_MASK_SOURCES,
        default="native_visible",
        help="Use the SMPL-X visible channel as prior_mask, or preserve the template prior_mask.",
    )
    parser.add_argument(
        "--native-visible-threshold",
        type=float,
        default=0.5,
        help="Threshold for the smplx_visible_mask channel when present.",
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=0,
        help="Optional diagnostic slice of the first N views. Default 0 keeps all views.",
    )
    parser.add_argument(
        "--preserve-teacher-mask",
        action="store_true",
        help="Keep an existing teacher_mask instead of replacing it with the native SMPL-X mask.",
    )
    parser.add_argument(
        "--overwrite-region-masks",
        action="store_true",
        help="Replace existing smplx_*_anchor_mask arrays with the native fallback masks.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the output summary without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing non-empty output directory.")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **payload)


def as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    array = np.asarray(value)
    if array.ndim == 0:
        return [str(array.item())]
    return [str(item) for item in array.tolist()]


def load_template_case(case_root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], Path, Path]:
    manifest_path = case_root / "case_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"case_manifest.json not found under {case_root}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs_path = case_root / manifest.get("inputs_npz", "inputs.npz")
    targets_path = case_root / manifest.get("targets_npz", "targets.npz")
    if not inputs_path.is_file():
        raise FileNotFoundError(f"inputs.npz not found: {inputs_path}")
    if not targets_path.is_file():
        raise FileNotFoundError(f"targets.npz not found: {targets_path}")

    inputs = load_npz(inputs_path)
    targets = load_npz(targets_path)
    return manifest, inputs, targets, inputs_path, targets_path


def infer_channel_names(manifest: dict[str, Any], prior_maps: np.ndarray) -> list[str]:
    channel_names = as_string_list(
        (manifest.get("prior_input_meta") or {}).get("channel_names")
        or manifest.get("prior_channels")
    )
    if not channel_names:
        channel_names = [f"prior_channel_{idx:03d}" for idx in range(int(prior_maps.shape[1]))]
    if len(channel_names) != int(prior_maps.shape[1]):
        raise ValueError(
            f"Manifest channel count {len(channel_names)} does not match prior_maps channels {prior_maps.shape[1]}"
        )
    return channel_names


def infer_summary_channel_names(manifest: dict[str, Any], prior_summary_tokens: np.ndarray | None) -> list[str]:
    if prior_summary_tokens is None:
        return []
    channel_names = as_string_list(
        (manifest.get("prior_input_meta") or {}).get("summary_channel_names")
        or manifest.get("prior_summary_channels")
    )
    if not channel_names:
        channel_names = [f"prior_summary_channel_{idx:03d}" for idx in range(int(prior_summary_tokens.shape[-1]))]
    if len(channel_names) != int(prior_summary_tokens.shape[-1]):
        raise ValueError(
            "Manifest summary channel count "
            f"{len(channel_names)} does not match prior_summary_tokens channels {prior_summary_tokens.shape[-1]}"
        )
    return channel_names


def select_dense_channels(channel_names: list[str], policy: str) -> list[int]:
    if policy == "adapter_compatible":
        keep = list(range(len(channel_names)))
    elif policy == "smplx_only":
        keep = [idx for idx, name in enumerate(channel_names) if name.startswith("smplx_")]
    elif policy == "smplx_native":
        keep = [
            idx
            for idx, name in enumerate(channel_names)
            if name == "silhouette" or name.startswith("smplx_")
        ]
    else:
        raise ValueError(f"Unsupported channel policy: {policy}")
    if not keep:
        raise ValueError(f"Channel policy {policy!r} selected no dense prior channels.")
    return keep


def select_summary_channels(summary_channel_names: list[str], policy: str) -> list[int]:
    if not summary_channel_names:
        return []
    if policy == "adapter_compatible":
        return list(range(len(summary_channel_names)))
    keep = [idx for idx, name in enumerate(summary_channel_names) if name.startswith("smplx_summary_")]
    if not keep:
        raise ValueError(f"Channel policy {policy!r} selected no summary channels.")
    return keep


def maybe_slice_views(payload: dict[str, np.ndarray], max_views: int, view_count: int) -> dict[str, np.ndarray]:
    if max_views <= 0 or max_views >= view_count:
        return dict(payload)
    sliced: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (view_count,):
            sliced[key] = value[:max_views].copy()
        else:
            sliced[key] = value
    return sliced


def make_native_mask(
    prior_maps: np.ndarray,
    channel_names: list[str],
    inputs: dict[str, np.ndarray],
    *,
    threshold: float,
) -> tuple[np.ndarray, str]:
    source = "prior_mask"
    if "prior_mask" in inputs:
        base_mask = np.asarray(inputs["prior_mask"], dtype=bool)
    elif "silhouette" in channel_names:
        base_mask = prior_maps[:, channel_names.index("silhouette")] > 0.5
        source = "silhouette"
    else:
        base_mask = np.ones(prior_maps.shape[0:1] + prior_maps.shape[2:4], dtype=bool)
        source = "all_pixels_fallback"

    if "smplx_visible_mask" in channel_names:
        visible = prior_maps[:, channel_names.index("smplx_visible_mask")] > float(threshold)
        return (base_mask & visible).astype(bool), "prior_mask_and_smplx_visible_mask"
    return base_mask.astype(bool), source


def ensure_anchor_masks(
    targets: dict[str, np.ndarray],
    native_mask: np.ndarray,
    *,
    overwrite: bool,
) -> dict[str, np.ndarray]:
    out = dict(targets)
    false_mask = np.zeros_like(native_mask, dtype=bool)
    fallback = {
        "smplx_bodyhand_anchor_mask": native_mask.astype(bool),
        "smplx_body_anchor_mask": native_mask.astype(bool),
        "smplx_hand_anchor_mask": false_mask,
        "smplx_left_hand_anchor_mask": false_mask,
        "smplx_right_hand_anchor_mask": false_mask,
    }
    for key, value in fallback.items():
        if overwrite or key not in out:
            out[key] = value
        else:
            out[key] = np.asarray(out[key], dtype=bool)
    return out


def shape_list(value: np.ndarray | None) -> list[int] | None:
    return None if value is None else [int(dim) for dim in value.shape]


def build_case(args: argparse.Namespace) -> dict[str, Any]:
    template_root = args.template_case_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest, inputs, targets, inputs_path, targets_path = load_template_case(template_root)

    if "prior_maps" not in inputs:
        raise KeyError("Template inputs.npz does not contain prior_maps.")
    prior_maps = np.asarray(inputs["prior_maps"], dtype=np.float32)
    if prior_maps.ndim != 4:
        raise ValueError(f"Expected prior_maps [V, C, H, W], got {prior_maps.shape}")

    view_count = int(prior_maps.shape[0])
    if int(args.max_views) > 0:
        keep_views = min(int(args.max_views), view_count)
        inputs = maybe_slice_views(inputs, keep_views, view_count)
        targets = maybe_slice_views(targets, keep_views, view_count)
        prior_maps = np.asarray(inputs["prior_maps"], dtype=np.float32)
        view_count = keep_views

    prior_summary_tokens = None
    if "prior_summary_tokens" in inputs:
        prior_summary_tokens = np.asarray(inputs["prior_summary_tokens"], dtype=np.float32)
        if prior_summary_tokens.ndim != 3 or prior_summary_tokens.shape[0] != view_count:
            raise ValueError(
                f"Expected prior_summary_tokens [V, T, C] aligned with views, got {prior_summary_tokens.shape}"
            )

    channel_names = infer_channel_names(manifest, prior_maps)
    summary_channel_names = infer_summary_channel_names(manifest, prior_summary_tokens)
    native_mask, native_mask_source = make_native_mask(
        prior_maps,
        channel_names,
        inputs,
        threshold=float(args.native_visible_threshold),
    )

    dense_keep = select_dense_channels(channel_names, args.channel_policy)
    summary_keep = select_summary_channels(summary_channel_names, args.channel_policy)
    output_channel_names = [channel_names[idx] for idx in dense_keep]
    output_summary_channel_names = [summary_channel_names[idx] for idx in summary_keep]
    output_prior_maps = prior_maps[:, dense_keep].astype(np.float16)
    output_summary_tokens = None
    if prior_summary_tokens is not None and summary_keep:
        output_summary_tokens = prior_summary_tokens[:, :, summary_keep].astype(np.float16)

    output_inputs = dict(inputs)
    output_inputs["prior_maps"] = output_prior_maps
    if output_summary_tokens is not None:
        output_inputs["prior_summary_tokens"] = output_summary_tokens
    else:
        output_inputs.pop("prior_summary_tokens", None)
    if args.prior_mask_source == "native_visible":
        output_inputs["prior_mask"] = native_mask.astype(bool)
    elif "prior_mask" in output_inputs:
        output_inputs["prior_mask"] = np.asarray(output_inputs["prior_mask"], dtype=bool)
    else:
        output_inputs["prior_mask"] = native_mask.astype(bool)

    output_targets = ensure_anchor_masks(
        targets,
        native_mask,
        overwrite=bool(args.overwrite_region_masks),
    )
    if not args.preserve_teacher_mask or "teacher_mask" not in output_targets:
        output_targets["teacher_mask"] = native_mask.astype(bool)
    else:
        output_targets["teacher_mask"] = np.asarray(output_targets["teacher_mask"], dtype=bool)
    output_targets["smplx_native_visible_mask"] = native_mask.astype(bool)

    prior_input_meta = dict(manifest.get("prior_input_meta") or {})
    prior_input_meta.update(
        {
            "channel_names": output_channel_names,
            "summary_channel_names": output_summary_channel_names,
            "channel_groups": _build_channel_group_meta(output_channel_names, output_summary_channel_names),
            "v15_native_smplx": {
                "source_template_case": str(template_root),
                "source_inputs_npz": str(inputs_path),
                "source_targets_npz": str(targets_path),
                "channel_policy": str(args.channel_policy),
                "dropped_dense_channels": [
                    name for idx, name in enumerate(channel_names) if idx not in set(dense_keep)
                ],
                "dropped_summary_channels": [
                    name for idx, name in enumerate(summary_channel_names) if idx not in set(summary_keep)
                ],
                "prior_mask_source": str(args.prior_mask_source),
                "native_mask_source": native_mask_source,
            },
        }
    )

    case_id = args.case_id or output_dir.name
    output_manifest = dict(manifest)
    output_manifest.update(
        {
            "case_id": case_id,
            "template_case_root": str(template_root),
            "num_views": view_count,
            "camera_ids": as_string_list(inputs.get("camera_ids"))[:view_count],
            "view_roles": as_string_list(inputs.get("view_roles"))[:view_count],
            "inputs_npz": "inputs.npz",
            "targets_npz": "targets.npz",
            "prior_channels": output_channel_names,
            "prior_summary_channels": output_summary_channel_names,
            "prior_summary_token_count": 0 if output_summary_tokens is None else int(output_summary_tokens.shape[1]),
            "prior_input_meta": prior_input_meta,
            "prior_geometry_source": "v15_smplx_native_prior_from_template",
            "prior_geometry_meta": {
                "source": "template_case_geometry_with_native_smplx_masks",
                "template_prior_geometry_source": manifest.get("prior_geometry_source"),
                "has_prior_depths": "prior_depths" in output_targets,
                "has_prior_points": "prior_points" in output_targets,
                "has_prior_normals": "prior_normals" in output_targets or "teacher_normals" in output_targets,
                "native_visible_pixels": int(native_mask.sum()),
                "native_visible_pixels_per_view": [int(v) for v in native_mask.reshape(view_count, -1).sum(axis=1)],
            },
            "v15_smplx_native_prior_case": {
                "format_version": 1,
                "adapter_dense_channels": int(output_prior_maps.shape[1]),
                "adapter_summary_channels": 0 if output_summary_tokens is None else int(output_summary_tokens.shape[-1]),
                "channel_policy": str(args.channel_policy),
                "prior_mask_source": str(args.prior_mask_source),
                "teacher_mask_source": "template_teacher_mask" if args.preserve_teacher_mask else "native_smplx_visible_mask",
                "loss_compatible_batch_keys": [
                    "prior_maps",
                    "prior_summary_tokens",
                    "prior_mask",
                    "prior_depths",
                    "prior_points",
                    "prior_normals",
                    "teacher_mask",
                    *ANCHOR_MASK_KEYS,
                ],
            },
        }
    )

    summary = {
        "task": "v15_build_smplx_vggt_prior_case",
        "dry_run": bool(args.dry_run),
        "template_case_root": str(template_root),
        "output_dir": str(output_dir),
        "case_id": case_id,
        "num_views": view_count,
        "dense_channels": output_channel_names,
        "summary_channels": output_summary_channel_names,
        "inputs_shape": {
            "prior_maps": shape_list(output_prior_maps),
            "prior_summary_tokens": shape_list(output_summary_tokens),
            "prior_mask": shape_list(output_inputs.get("prior_mask")),
        },
        "targets_available": {
            key: key in output_targets
            for key in (
                "prior_depths",
                "prior_points",
                "prior_normals",
                "teacher_normals",
                "teacher_mask",
                *ANCHOR_MASK_KEYS,
            )
        },
        "native_visible_pixels": int(native_mask.sum()),
        "blockers": [],
    }
    if "prior_depths" not in output_targets:
        summary["blockers"].append("template target bundle has no prior_depths; native depth prior loss will be dummy")
    if "prior_points" not in output_targets:
        summary["blockers"].append("template target bundle has no prior_points; native point prior loss will be dummy")
    if "prior_normals" not in output_targets and "teacher_normals" not in output_targets:
        summary["blockers"].append("template target bundle has no prior_normals/teacher_normals; native normal prior loss will be dummy")

    if args.dry_run:
        print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
        return summary

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} is not empty; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_npz(output_dir / "inputs.npz", output_inputs)
    save_npz(output_dir / "targets.npz", output_targets)
    (output_dir / "case_manifest.json").write_text(
        json.dumps(json_ready(output_manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "v15_smplx_native_prior_summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    build_case(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
