from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.models.detail_normal_refiner import DetailNormalRefiner  # noqa: E402
from vggt.utils.normal_refiner import normal_to_rgb  # noqa: E402


COARSE_NORMAL_CHANNELS = ("smplx_cam_nx", "smplx_cam_ny", "smplx_cam_nz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a detail_normal_refiner checkpoint to an ROI dataset export, stitch the refined normals "
            "back into scene-level prior_maps.npz, and emit a patched scene directory for one-case inference."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint produced by train_detail_normal_refiner.py")
    parser.add_argument("--dataset-npz", required=True, help="ROI dataset export with roi_box_xyxy/view_index metadata")
    parser.add_argument("--scene-dir", required=True, help="Scene directory containing prior_maps.npz")
    parser.add_argument("--output-scene-dir", required=True, help="Directory for the patched copied scene")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-mask-source", choices=("coarse_valid", "human_mask", "coarse_or_human"), default="coarse_valid")
    parser.add_argument("--visualize-count", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output scene dir if it already exists")
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resize_float_hw(array: np.ndarray, out_h: int, out_w: int, *, is_mask: bool = False) -> np.ndarray:
    arr = np.asarray(array)
    mode = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR

    if arr.ndim == 2:
        pil = Image.fromarray(arr.astype(np.float32), mode="F")
        pil = pil.resize((int(out_w), int(out_h)), mode)
        resized = np.asarray(pil, dtype=np.float32)
        return (resized > 0.5) if is_mask else resized

    if arr.ndim == 3 and arr.shape[-1] >= 1:
        channels = []
        for channel_idx in range(arr.shape[-1]):
            pil = Image.fromarray(arr[..., channel_idx].astype(np.float32), mode="F")
            pil = pil.resize((int(out_w), int(out_h)), mode)
            channel = np.asarray(pil, dtype=np.float32)
            if is_mask:
                channel = (channel > 0.5).astype(np.float32)
            channels.append(channel)
        stacked = np.stack(channels, axis=-1)
        if is_mask:
            return stacked > 0.5
        return stacked.astype(np.float32)

    raise ValueError(f"Unsupported shape for resize: {arr.shape}")


def _normalize_normals(normals: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float32)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normalized = normals / np.clip(norms, 1e-6, None)
    if mask is not None:
        normalized = normalized.copy()
        normalized[~np.asarray(mask, dtype=bool)] = 0.0
    return normalized.astype(np.float32)


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "rgb",
            "human_mask",
            "coarse_prior_normal",
            "coarse_prior_valid_mask",
            "roi_box_xyxy",
            "view_index",
            "view_name",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise KeyError(f"Dataset NPZ is missing required keys: {missing}")
        return {key: np.array(payload[key]) for key in payload.files}


def _predict_refined_normals(
    *,
    model: DetailNormalRefiner,
    device: torch.device,
    dataset_payload: dict[str, np.ndarray],
    batch_size: int,
) -> np.ndarray:
    rgb = torch.from_numpy(dataset_payload["rgb"]).permute(0, 3, 1, 2).float() / 255.0
    human_mask = torch.from_numpy(dataset_payload["human_mask"].astype(np.float32))[:, None]
    coarse = torch.from_numpy(dataset_payload["coarse_prior_normal"]).permute(0, 3, 1, 2).float()

    refined_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(rgb.shape[0]), int(batch_size)):
            end = min(int(rgb.shape[0]), start + int(batch_size))
            preds = model(
                rgb=rgb[start:end].to(device),
                coarse_normal=coarse[start:end].to(device),
                human_mask=human_mask[start:end].to(device),
            )
            refined = preds["refined_normal"].detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
            refined_batches.append(refined)
    return np.concatenate(refined_batches, axis=0)


def _load_prior_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.array(payload[key]) for key in payload.files}


def _lookup_normal_indices(prior_channels: np.ndarray) -> list[int]:
    names = [str(name) for name in np.asarray(prior_channels).tolist()]
    lookup = {name: idx for idx, name in enumerate(names)}
    return [int(lookup[name]) for name in COARSE_NORMAL_CHANNELS]


def _copy_scene_tree(scene_dir: Path, output_scene_dir: Path, *, overwrite: bool) -> None:
    if output_scene_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output scene dir already exists: {output_scene_dir}")
        shutil.rmtree(output_scene_dir)
    shutil.copytree(scene_dir, output_scene_dir)


def _patch_prior_maps(
    *,
    prior_payload: dict[str, np.ndarray],
    dataset_payload: dict[str, np.ndarray],
    refined_normals: np.ndarray,
    patch_mask_source: str,
    output_scene_dir: Path,
    visualize_count: int,
) -> dict[str, object]:
    prior_maps = np.array(prior_payload["prior_maps"], copy=True)
    prior_channels = np.array(prior_payload["prior_channels"])
    normal_indices = _lookup_normal_indices(prior_channels)

    roi_boxes = np.asarray(dataset_payload["roi_box_xyxy"], dtype=np.int32)
    view_indices = np.asarray(dataset_payload["view_index"], dtype=np.int32)
    view_names = np.asarray(dataset_payload["view_name"])
    coarse = np.asarray(dataset_payload["coarse_prior_normal"], dtype=np.float32)
    coarse_valid = np.asarray(dataset_payload["coarse_prior_valid_mask"], dtype=bool)
    human_mask = np.asarray(dataset_payload["human_mask"], dtype=bool)

    preview_dir = output_scene_dir / "refined_prior_patch_visuals"
    preview_dir.mkdir(parents=True, exist_ok=True)

    patch_records: list[dict[str, object]] = []
    for sample_idx in range(refined_normals.shape[0]):
        view_idx = int(view_indices[sample_idx])
        x0, y0, x1, y1 = [int(value) for value in roi_boxes[sample_idx].tolist()]
        roi_h = max(1, y1 - y0)
        roi_w = max(1, x1 - x0)

        if patch_mask_source == "coarse_valid":
            patch_mask = coarse_valid[sample_idx]
        elif patch_mask_source == "human_mask":
            patch_mask = human_mask[sample_idx]
        else:
            patch_mask = coarse_valid[sample_idx] | human_mask[sample_idx]

        refined_roi = _resize_float_hw(refined_normals[sample_idx], roi_h, roi_w, is_mask=False).astype(np.float32)
        refined_mask = _resize_float_hw(patch_mask.astype(np.float32), roi_h, roi_w, is_mask=True)
        if refined_mask.ndim == 3:
            refined_mask = refined_mask[..., 0]
        refined_roi = _normalize_normals(refined_roi, refined_mask)

        original_patch = prior_maps[view_idx, normal_indices, y0:y1, x0:x1].transpose(1, 2, 0).astype(np.float32)
        updated_patch = original_patch.copy()
        updated_patch[refined_mask] = refined_roi[refined_mask]
        prior_maps[view_idx, normal_indices, y0:y1, x0:x1] = updated_patch.transpose(2, 0, 1)

        changed_pixels = int(np.asarray(refined_mask, dtype=bool).sum())
        delta = np.linalg.norm(updated_patch - original_patch, axis=-1)
        patch_records.append(
            {
                "sample_index": sample_idx,
                "view_index": view_idx,
                "view_name": str(view_names[sample_idx]),
                "roi_box_xyxy": [x0, y0, x1, y1],
                "changed_pixels": changed_pixels,
                "delta_mean": float(delta[refined_mask].mean()) if changed_pixels > 0 else 0.0,
                "delta_max": float(delta[refined_mask].max()) if changed_pixels > 0 else 0.0,
            }
        )

        if sample_idx < max(1, int(visualize_count)):
            coarse_rgb = normal_to_rgb(coarse[sample_idx], coarse_valid[sample_idx])
            refined_rgb = normal_to_rgb(refined_normals[sample_idx], patch_mask)
            stitched_rgb = normal_to_rgb(updated_patch, refined_mask)
            summary = np.concatenate([coarse_rgb, refined_rgb, stitched_rgb], axis=1)
            stem = f"{sample_idx:02d}_view{view_idx:02d}_{str(view_names[sample_idx])}"
            Image.fromarray(coarse_rgb).save(preview_dir / f"{stem}_coarse_roi.png")
            Image.fromarray(refined_rgb).save(preview_dir / f"{stem}_refined_roi.png")
            Image.fromarray(stitched_rgb).save(preview_dir / f"{stem}_stitched_roi.png")
            Image.fromarray(summary).save(preview_dir / f"{stem}_summary_strip.png")

    prior_payload["prior_maps"] = prior_maps.astype(np.float32)
    np.savez_compressed(output_scene_dir / "prior_maps.npz", **prior_payload)
    return {
        "patch_records": patch_records,
        "normal_channel_indices": normal_indices,
    }


def _update_scene_manifest(scene_manifest_path: Path, output_scene_dir: Path) -> None:
    if not scene_manifest_path.is_file():
        return
    manifest = json.loads(scene_manifest_path.read_text(encoding="utf-8"))
    manifest["prior_maps_file"] = str((output_scene_dir / "prior_maps.npz").resolve())
    scene_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    dataset_npz = Path(args.dataset_npz).expanduser().resolve()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    output_scene_dir = Path(args.output_scene_dir).expanduser().resolve()
    device = _resolve_device(args.device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    model = DetailNormalRefiner(
        base_dim=int(ckpt_args.get("base_dim", 32)),
        residual_scale=float(ckpt_args.get("residual_scale", 0.35)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset_payload = _load_dataset(dataset_npz)
    refined_normals = _predict_refined_normals(
        model=model,
        device=device,
        dataset_payload=dataset_payload,
        batch_size=int(args.batch_size),
    )

    _copy_scene_tree(scene_dir, output_scene_dir, overwrite=bool(args.overwrite))
    prior_payload = _load_prior_bundle(scene_dir / "prior_maps.npz")
    patch_meta = _patch_prior_maps(
        prior_payload=prior_payload,
        dataset_payload=dataset_payload,
        refined_normals=refined_normals,
        patch_mask_source=str(args.patch_mask_source),
        output_scene_dir=output_scene_dir,
        visualize_count=int(args.visualize_count),
    )
    _update_scene_manifest(output_scene_dir / "scene_manifest.json", output_scene_dir)

    summary = {
        "message": "scene prior patched with detail_normal_refiner outputs",
        "checkpoint": str(checkpoint_path),
        "dataset_npz": str(dataset_npz),
        "scene_dir": str(scene_dir),
        "output_scene_dir": str(output_scene_dir),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "patch_mask_source": str(args.patch_mask_source),
        "num_samples": int(refined_normals.shape[0]),
        "patch_records": patch_meta["patch_records"],
        "normal_channel_indices": patch_meta["normal_channel_indices"],
        "notes": [
            "Only dense prior normal channels are patched; summary tokens remain unchanged.",
            "This is an offline one-case prior-conditioning experiment, not a full training-target integration.",
        ],
    }
    (output_scene_dir / "refined_prior_patch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
