from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_SIZE = 518
BACKGROUND_DARK = np.array([16, 19, 26], dtype=np.uint8)
PRIOR_NORMAL_CHANNELS = ("smplx_cam_nx", "smplx_cam_ny", "smplx_cam_nz")
PRIOR_NORMAL_INDEXES = (26, 27, 28)


@dataclass(frozen=True)
class CaseSpec:
    slug: str
    case_id: str
    view_tag: str
    scene_dir: Path
    predictions_npz: Path
    prior_maps_npz: Path
    variant: str
    source_label: str
    point_source: str = "depth_unprojection"


@dataclass(frozen=True)
class ProbeSpec:
    case_id: str
    view_tag: str
    inputs_npz: Path
    predictions_npz: Path
    summary_json: Path
    source_label: str = "pred_normal_frozen_probe"


def default_output_dir() -> Path:
    return REPO_ROOT / "output" / f"normal_advisor_pack_{date.today().strftime('%Y%m%d')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mentor-facing normal/geometry advisor pack.")
    parser.add_argument("--output-dir", default=str(default_output_dir()), help="Output directory root")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for point subsampling")
    parser.add_argument(
        "--conf-percentile",
        type=float,
        default=40.0,
        help="Point confidence percentile used before point-cloud rendering",
    )
    parser.add_argument(
        "--max-full-points",
        type=int,
        default=150000,
        help="Maximum points per full-body point-cloud render",
    )
    parser.add_argument(
        "--max-roi-points",
        type=int,
        default=65000,
        help="Maximum points per head/face ROI point-cloud render",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def open_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def open_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0


def preprocess_to_square(img: Image.Image, target_size: int, resample: Image.Resampling, fill: tuple[int, ...]) -> np.ndarray:
    width, height = img.size
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14
    resized = img.resize((new_width, new_height), resample)
    arr = np.asarray(resized)
    if arr.ndim == 2:
        canvas = np.full((target_size, target_size), fill[0], dtype=arr.dtype)
        top = (target_size - new_height) // 2
        left = (target_size - new_width) // 2
        canvas[top : top + new_height, left : left + new_width] = arr
        return canvas
    channels = arr.shape[2]
    canvas = np.full((target_size, target_size, channels), fill[:channels], dtype=arr.dtype)
    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = arr
    return canvas


def sorted_view_paths(scene_dir: Path, subdir: str) -> list[Path]:
    folder = scene_dir / subdir
    return sorted(path for path in folder.iterdir() if path.is_file())


def load_preprocessed_rgb_stack(scene_dir: Path) -> np.ndarray:
    image_paths = sorted_view_paths(scene_dir, "images")
    images = [
        preprocess_to_square(Image.open(path).convert("RGB"), TARGET_SIZE, Image.Resampling.BILINEAR, (255, 255, 255))
        for path in image_paths
    ]
    return np.stack(images, axis=0).astype(np.uint8)


def load_preprocessed_mask_stack(scene_dir: Path) -> np.ndarray:
    mask_paths = sorted_view_paths(scene_dir, "masks")
    masks = [
        preprocess_to_square(Image.open(path).convert("L"), TARGET_SIZE, Image.Resampling.NEAREST, (0,)).astype(bool)
        for path in mask_paths
    ]
    return np.stack(masks, axis=0)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_box(
    box: tuple[int, int, int, int] | None,
    image_shape: tuple[int, int],
    pad_x_ratio: float,
    pad_y_ratio: float,
) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    height, width = image_shape
    x0, y0, x1, y1 = box
    box_w = x1 - x0
    box_h = y1 - y0
    pad_x = int(round(box_w * pad_x_ratio))
    pad_y = int(round(box_h * pad_y_ratio))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    )


def head_and_face_boxes(mask: np.ndarray) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
    bbox = mask_bbox(mask)
    if bbox is None:
        return None, None
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0

    head_height = max(24, int(round(height * 0.44)))
    head_box = (x0, y0, x1, min(y1, y0 + head_height))
    head_box = expand_box(head_box, mask.shape, pad_x_ratio=0.10, pad_y_ratio=0.06)

    face_width = max(24, int(round(width * 0.56)))
    face_height = max(24, int(round(head_height * 0.66)))
    face_cx = (x0 + x1) // 2
    face_x0 = max(x0, face_cx - face_width // 2)
    face_x1 = min(x1, face_x0 + face_width)
    face_y0 = y0 + int(round(head_height * 0.11))
    face_y1 = min(y1, face_y0 + face_height)
    face_box = (face_x0, face_y0, face_x1, face_y1)
    face_box = expand_box(face_box, mask.shape, pad_x_ratio=0.22, pad_y_ratio=0.18)
    return head_box, face_box


def crop_box(arr: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray:
    if box is None:
        return arr
    x0, y0, x1, y1 = box
    return arr[y0:y1, x0:x1]


def map_box_between_resolutions(
    box: tuple[int, int, int, int] | None,
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    x_scale = dst_w / float(src_w)
    y_scale = dst_h / float(src_h)
    x0, y0, x1, y1 = box
    return (
        int(round(x0 * x_scale)),
        int(round(y0 * y_scale)),
        int(round(x1 * x_scale)),
        int(round(y1 * y_scale)),
    )


def normal_to_rgb(normals: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    rgb = np.clip((normals.astype(np.float32) + 1.0) * 0.5, 0.0, 1.0)
    rgb = (rgb * 255.0).astype(np.uint8)
    if mask is not None:
        rgb = rgb.copy()
        rgb[~mask] = 255
    return rgb


def select_overview_indices(num_views: int) -> list[int]:
    if num_views <= 6:
        return list(range(num_views))
    count = 6 if num_views >= 18 else 4
    indices = {0, num_views - 1}
    for idx in range(count):
        indices.add(int(round(idx * (num_views - 1) / max(1, count - 1))))
    return sorted(indices)


def save_panel_grid(
    images: list[np.ndarray],
    titles: list[str],
    output_path: Path,
    suptitle: str,
    ncols: int = 3,
    figsize_per_panel: tuple[float, float] = (4.2, 4.8),
) -> None:
    n = len(images)
    ncols = min(ncols, n)
    nrows = int(math.ceil(n / float(ncols)))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    for idx, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if idx >= n:
            continue
        ax.imshow(images[idx])
        ax.set_title(titles[idx], fontsize=11)
    fig.suptitle(suptitle, fontsize=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_roi_triptych(
    rgb_crop: np.ndarray,
    normal_crop: np.ndarray,
    output_path: Path,
    suptitle: str,
    normal_title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.2))
    fig.patch.set_facecolor("white")
    for ax, image, title in zip(axes, [rgb_crop, normal_crop], ["RGB ROI", normal_title]):
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
        ax.set_axis_off()
    fig.suptitle(suptitle, fontsize=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_prior_bundle(prior_maps_npz: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    prior_payload = np.load(prior_maps_npz, allow_pickle=False)
    channels = prior_payload["prior_channels"].tolist()
    if list(channels[idx] for idx in PRIOR_NORMAL_INDEXES) != list(PRIOR_NORMAL_CHANNELS):
        raise ValueError(f"Unexpected prior normal channels in {prior_maps_npz}")
    prior_normals = np.stack([prior_payload["prior_maps"][:, idx] for idx in PRIOR_NORMAL_INDEXES], axis=-1)
    prior_mask = prior_payload["prior_mask"].astype(bool)
    return prior_normals.astype(np.float32), prior_mask, channels


def save_prior_normal_outputs(case: CaseSpec, output_dir: Path, inventory: list[dict[str, Any]]) -> dict[str, Path]:
    normals_dir = output_dir / "normals"
    manifest = load_json(case.scene_dir / "scene_manifest.json")
    image_paths = sorted_view_paths(case.scene_dir, "images")
    mask_paths = sorted_view_paths(case.scene_dir, "masks")
    target_rgb = open_rgb(image_paths[0])
    target_mask = open_mask(mask_paths[0])
    prior_normals, prior_mask, _ = load_prior_bundle(case.prior_maps_npz)
    prior_rgb = normal_to_rgb(prior_normals, mask=prior_mask)

    selected = select_overview_indices(prior_rgb.shape[0])
    overview_images = [prior_rgb[idx] for idx in selected]
    overview_titles = [
        f"view {idx:02d} cam {manifest['exported_views'][idx]['camera_id']}"
        for idx in selected
    ]
    overview_path = normals_dir / f"{case.case_id}_prior_normal_overview_{case.view_tag}.png"
    save_panel_grid(
        overview_images,
        overview_titles,
        overview_path,
        suptitle=f"{case.case_id} {case.view_tag} prior normal overview",
        ncols=3,
    )

    head_box_small, face_box_small = head_and_face_boxes(prior_mask[0])
    head_box_rgb = map_box_between_resolutions(head_box_small, prior_mask[0].shape, target_mask.shape)
    face_box_rgb = map_box_between_resolutions(face_box_small, prior_mask[0].shape, target_mask.shape)

    head_rgb_crop = crop_box(target_rgb, head_box_rgb)
    face_rgb_crop = crop_box(target_rgb, face_box_rgb)
    head_normal_crop = crop_box(prior_rgb[0], head_box_small)
    face_normal_crop = crop_box(prior_rgb[0], face_box_small)

    head_path = normals_dir / f"{case.case_id}_targetcam00_head_roi_rgb_vs_prior_normal_{case.view_tag}.png"
    face_path = normals_dir / f"{case.case_id}_targetcam00_face_roi_rgb_vs_prior_normal_{case.view_tag}.png"
    save_roi_triptych(
        head_rgb_crop,
        head_normal_crop,
        head_path,
        suptitle=f"{case.case_id} head ROI RGB vs prior normal ({case.view_tag})",
        normal_title="SMPL-X prior normal ROI",
    )
    save_roi_triptych(
        face_rgb_crop,
        face_normal_crop,
        face_path,
        suptitle=f"{case.case_id} face ROI RGB vs prior normal ({case.view_tag})",
        normal_title="SMPL-X prior normal ROI",
    )

    outputs = {
        "overview": overview_path,
        "head_roi": head_path,
        "face_roi": face_path,
    }
    inventory.extend(
        [
            {
                "category": "normals",
                "path": str(overview_path.relative_to(output_dir)),
                "description": f"{case.view_tag} prior-normal overview contact sheet",
            },
            {
                "category": "normals",
                "path": str(head_path.relative_to(output_dir)),
                "description": f"{case.view_tag} target head ROI RGB vs prior normal",
            },
            {
                "category": "normals",
                "path": str(face_path.relative_to(output_dir)),
                "description": f"{case.view_tag} target face ROI RGB vs prior normal",
            },
        ]
    )
    return outputs


def save_probe_outputs(probe: ProbeSpec, output_dir: Path, inventory: list[dict[str, Any]]) -> dict[str, Path]:
    normals_dir = output_dir / "normals"
    inputs = np.load(probe.inputs_npz, allow_pickle=False)
    preds = np.load(probe.predictions_npz, allow_pickle=False)
    pred_normals = preds["normal"].astype(np.float32)
    pred_mask = inputs["prior_mask"].astype(bool)
    pred_rgb = normal_to_rgb(pred_normals, mask=pred_mask)
    rgb_inputs = inputs["images"].astype(np.uint8)

    selected = select_overview_indices(pred_rgb.shape[0])
    overview_images = [pred_rgb[idx] for idx in selected]
    overview_titles = [f"probe view {idx:02d}" for idx in selected]
    overview_path = normals_dir / f"{probe.case_id}_pred_normal_overview_{probe.view_tag}.png"
    save_panel_grid(
        overview_images,
        overview_titles,
        overview_path,
        suptitle=f"{probe.case_id} frozen-probe predicted normal overview ({probe.view_tag})",
        ncols=2,
        figsize_per_panel=(4.4, 4.6),
    )

    head_box_small, face_box_small = head_and_face_boxes(pred_mask[0])
    head_path = normals_dir / f"{probe.case_id}_head_roi_rgb_vs_pred_normal_{probe.view_tag}.png"
    face_path = normals_dir / f"{probe.case_id}_face_roi_rgb_vs_pred_normal_{probe.view_tag}.png"
    save_roi_triptych(
        crop_box(rgb_inputs[0], head_box_small),
        crop_box(pred_rgb[0], head_box_small),
        head_path,
        suptitle=f"{probe.case_id} head ROI RGB vs predicted normal ({probe.view_tag})",
        normal_title="Frozen-probe predicted normal ROI",
    )
    save_roi_triptych(
        crop_box(rgb_inputs[0], face_box_small),
        crop_box(pred_rgb[0], face_box_small),
        face_path,
        suptitle=f"{probe.case_id} face ROI RGB vs predicted normal ({probe.view_tag})",
        normal_title="Frozen-probe predicted normal ROI",
    )

    summary = load_json(probe.summary_json)
    outputs = {
        "overview": overview_path,
        "head_roi": head_path,
        "face_roi": face_path,
    }
    inventory.extend(
        [
            {
                "category": "normals",
                "path": str(overview_path.relative_to(output_dir)),
                "description": "4-view frozen-probe predicted-normal overview",
            },
            {
                "category": "normals",
                "path": str(head_path.relative_to(output_dir)),
                "description": "4-view frozen-probe head ROI RGB vs predicted normal",
            },
            {
                "category": "normals",
                "path": str(face_path.relative_to(output_dir)),
                "description": "4-view frozen-probe face ROI RGB vs predicted normal",
            },
            {
                "category": "docs",
                "path": str(probe.summary_json),
                "description": f"Probe summary reference with final loss {summary.get('final_loss_prior_normal')}",
            },
        ]
    )
    return outputs


def closed_form_inverse_se3_numpy(se3: np.ndarray) -> np.ndarray:
    rotation = se3[:, :3, :3]
    translation = se3[:, :3, 3:]
    rotation_t = np.transpose(rotation, (0, 2, 1))
    top_right = -np.matmul(rotation_t, translation)
    inverted = np.tile(np.eye(4, dtype=se3.dtype), (len(rotation), 1, 1))
    inverted[:, :3, :3] = rotation_t
    inverted[:, :3, 3:] = top_right
    return inverted


def unproject_depth_map_to_point_map_numpy(
    depth_map: np.ndarray,
    extrinsics_cam: np.ndarray,
    intrinsics_cam: np.ndarray,
) -> np.ndarray:
    cam_to_world = closed_form_inverse_se3_numpy(extrinsics_cam)
    world_points = []
    for frame_idx in range(depth_map.shape[0]):
        depth = depth_map[frame_idx].squeeze(-1).astype(np.float32)
        intrinsic = intrinsics_cam[frame_idx].astype(np.float32)
        height, width = depth.shape
        u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

        fu, fv = intrinsic[0, 0], intrinsic[1, 1]
        cu, cv = intrinsic[0, 2], intrinsic[1, 2]
        x_cam = (u - cu) * depth / fu
        y_cam = (v - cv) * depth / fv
        z_cam = depth
        cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1)

        rotation = cam_to_world[frame_idx, :3, :3]
        translation = cam_to_world[frame_idx, :3, 3]
        world = np.dot(cam_coords, rotation.T) + translation
        world_points.append(world.astype(np.float32))
    return np.stack(world_points, axis=0)


def resolve_point_source(payload: np.lib.npyio.NpzFile, point_source: str) -> tuple[np.ndarray, np.ndarray]:
    if point_source == "world_points":
        return payload["world_points"], payload["world_points_conf"]
    if point_source == "depth_unprojection":
        world_points = unproject_depth_map_to_point_map_numpy(payload["depth"], payload["extrinsic"], payload["intrinsic"])
        return world_points, payload["depth_conf"]
    raise ValueError(f"Unsupported point source: {point_source}")


def load_case_point_data(case: CaseSpec) -> dict[str, Any]:
    payload = np.load(case.predictions_npz, allow_pickle=False)
    world_points, point_conf = resolve_point_source(payload, case.point_source)
    rgb_stack = load_preprocessed_rgb_stack(case.scene_dir)
    mask_stack = load_preprocessed_mask_stack(case.scene_dir)
    prior_normals, prior_mask, _ = load_prior_bundle(case.prior_maps_npz)
    return {
        "world_points": world_points.astype(np.float32),
        "point_conf": point_conf.astype(np.float32),
        "rgb_stack": rgb_stack.astype(np.uint8),
        "mask_stack": mask_stack.astype(bool),
        "prior_normals": prior_normals,
        "prior_mask": prior_mask.astype(bool),
        "scene_manifest": load_json(case.scene_dir / "scene_manifest.json"),
    }


def flatten_masked_points(
    world_points: np.ndarray,
    point_conf: np.ndarray,
    rgb_stack: np.ndarray,
    mask_stack: np.ndarray,
    conf_percentile: float,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    points = world_points.reshape(-1, 3)
    conf = point_conf.reshape(-1)
    colors = rgb_stack.reshape(-1, 3)
    mask = mask_stack.reshape(-1)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(conf) & (conf > 0) & mask
    if not np.any(valid):
        raise RuntimeError("No valid masked points available")
    conf_threshold = float(np.percentile(conf[valid], conf_percentile))
    keep = valid & (conf >= conf_threshold)
    indices = np.flatnonzero(keep)
    if len(indices) > max_points:
        indices = rng.choice(indices, size=max_points, replace=False)
    return points[indices], colors[indices]


def sample_view_roi_points(
    world_points: np.ndarray,
    point_conf: np.ndarray,
    mask: np.ndarray,
    box: tuple[int, int, int, int] | None,
) -> np.ndarray:
    if box is None:
        return np.empty((0, 3), dtype=np.float32)
    crop_mask = np.zeros(mask.shape, dtype=bool)
    x0, y0, x1, y1 = box
    crop_mask[y0:y1, x0:x1] = True
    valid = crop_mask & mask & np.isfinite(point_conf) & (point_conf > 0) & np.isfinite(world_points).all(axis=-1)
    return world_points[valid].astype(np.float32)


def robust_aabb(points: np.ndarray, margin_ratio: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        raise RuntimeError("ROI point set is empty; cannot compute AABB")
    lower = np.percentile(points, 5.0, axis=0)
    upper = np.percentile(points, 95.0, axis=0)
    span = np.maximum(upper - lower, 1e-4)
    margin = span * margin_ratio
    return lower - margin, upper + margin


def make_box_mask(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.all(points >= lower[None, :], axis=1) & np.all(points <= upper[None, :], axis=1)


def build_reference_boxes(reference_case: CaseSpec) -> dict[str, Any]:
    data = load_case_point_data(reference_case)
    prior_mask = data["prior_mask"][0]
    head_box, face_box = head_and_face_boxes(prior_mask)
    body_points, _ = flatten_masked_points(
        data["world_points"],
        data["point_conf"],
        data["rgb_stack"],
        data["mask_stack"],
        conf_percentile=40.0,
        max_points=220000,
        rng=np.random.default_rng(0),
    )
    head_points = sample_view_roi_points(data["world_points"][0], data["point_conf"][0], data["mask_stack"][0], head_box)
    face_points = sample_view_roi_points(data["world_points"][0], data["point_conf"][0], data["mask_stack"][0], face_box)
    return {
        "head_box": head_box,
        "face_box": face_box,
        "head_aabb": robust_aabb(head_points, margin_ratio=0.16),
        "face_aabb": robust_aabb(face_points, margin_ratio=0.18),
        "body_aabb": robust_aabb(body_points, margin_ratio=0.08),
    }


def rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cx, sx = np.cos(pitch), np.sin(pitch)
    rot_y = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float32,
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ],
        dtype=np.float32,
    )
    return rot_x @ rot_y


def render_projected_points(
    ax: plt.Axes,
    points: np.ndarray,
    colors: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    title: str,
    fixed_aabb: tuple[np.ndarray, np.ndarray] | None,
    point_size: float,
) -> None:
    if len(points) == 0:
        ax.set_facecolor(BACKGROUND_DARK / 255.0)
        ax.set_title(title, fontsize=11, color="white")
        ax.set_axis_off()
        return
    rot = rotation_matrix(yaw_deg, pitch_deg)
    rotated = points @ rot.T
    order = np.argsort(rotated[:, 2])
    projected = rotated[order]
    projected_colors = colors[order].astype(np.float32) / 255.0
    ax.scatter(
        projected[:, 0],
        -projected[:, 1],
        c=projected_colors,
        s=point_size,
        linewidths=0,
        marker="o",
        alpha=0.95,
    )
    ax.set_title(title, fontsize=11, color="white")
    ax.set_axis_off()
    ax.set_facecolor(BACKGROUND_DARK / 255.0)
    ax.set_aspect("equal", adjustable="box")
    if fixed_aabb is not None:
        lower, upper = fixed_aabb
        corners = np.array(
            [
                [lower[0], lower[1], lower[2]],
                [lower[0], lower[1], upper[2]],
                [lower[0], upper[1], lower[2]],
                [lower[0], upper[1], upper[2]],
                [upper[0], lower[1], lower[2]],
                [upper[0], lower[1], upper[2]],
                [upper[0], upper[1], lower[2]],
                [upper[0], upper[1], upper[2]],
            ],
            dtype=np.float32,
        )
        rotated_corners = corners @ rot.T
        x_coords = rotated_corners[:, 0]
        y_coords = -rotated_corners[:, 1]
    else:
        x_coords = projected[:, 0]
        y_coords = -projected[:, 1]
    if len(x_coords) == 0:
        x_coords = np.array([-1.0, 1.0], dtype=np.float32)
        y_coords = np.array([-1.0, 1.0], dtype=np.float32)
    x_margin = max(1e-3, (x_coords.max() - x_coords.min()) * 0.08)
    y_margin = max(1e-3, (y_coords.max() - y_coords.min()) * 0.08)
    ax.set_xlim(float(x_coords.min() - x_margin), float(x_coords.max() + x_margin))
    ax.set_ylim(float(y_coords.min() - y_margin), float(y_coords.max() + y_margin))


def render_triptych(
    points: np.ndarray,
    colors: np.ndarray,
    output_path: Path,
    suptitle: str,
    fixed_aabb: tuple[np.ndarray, np.ndarray] | None,
    point_size: float,
    figure_size: tuple[float, float] = (13.6, 4.8),
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=figure_size)
    fig.patch.set_facecolor(BACKGROUND_DARK / 255.0)
    views = [
        ("Front", 0.0, 0.0),
        ("Three-quarter", 35.0, -10.0),
        ("Side", 90.0, 0.0),
    ]
    for ax, (title, yaw_deg, pitch_deg) in zip(axes, views):
        render_projected_points(ax, points, colors, yaw_deg, pitch_deg, title, fixed_aabb, point_size=point_size)
    fig.suptitle(suptitle, color="white", fontsize=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_pointcloud_outputs(
    case: CaseSpec,
    output_dir: Path,
    inventory: list[dict[str, Any]],
    reference_boxes: dict[str, Any],
    conf_percentile: float,
    max_full_points: int,
    max_roi_points: int,
    rng_seed: int,
) -> dict[str, Path]:
    point_dir = output_dir / "pointcloud_open3d"
    data = load_case_point_data(case)
    rng = np.random.default_rng(rng_seed)
    full_points, full_colors = flatten_masked_points(
        data["world_points"],
        data["point_conf"],
        data["rgb_stack"],
        data["mask_stack"],
        conf_percentile=conf_percentile,
        max_points=max_full_points,
        rng=rng,
    )
    head_mask = make_box_mask(full_points, *reference_boxes["head_aabb"])
    face_mask = make_box_mask(full_points, *reference_boxes["face_aabb"])
    head_points = full_points[head_mask]
    head_colors = full_colors[head_mask]
    face_points = full_points[face_mask]
    face_colors = full_colors[face_mask]

    if len(head_points) > max_roi_points:
        head_idx = rng.choice(len(head_points), size=max_roi_points, replace=False)
        head_points = head_points[head_idx]
        head_colors = head_colors[head_idx]
    if len(face_points) > max_roi_points:
        face_idx = rng.choice(len(face_points), size=max_roi_points, replace=False)
        face_points = face_points[face_idx]
        face_colors = face_colors[face_idx]

    body_path = point_dir / f"{case.case_id}_open3d_human_full_{case.view_tag}_{case.variant}.png"
    head_path = point_dir / f"{case.case_id}_open3d_head_closeup_{case.view_tag}_{case.variant}.png"
    face_path = point_dir / f"{case.case_id}_open3d_face_closeup_{case.view_tag}_{case.variant}.png"

    render_triptych(
        full_points,
        full_colors,
        body_path,
        suptitle=f"{case.case_id} human full point cloud ({case.view_tag}, {case.variant})",
        fixed_aabb=reference_boxes["body_aabb"],
        point_size=0.35,
        figure_size=(13.8, 4.9),
    )
    render_triptych(
        head_points,
        head_colors,
        head_path,
        suptitle=f"{case.case_id} head close-up ({case.view_tag}, {case.variant})",
        fixed_aabb=reference_boxes["head_aabb"],
        point_size=1.8,
        figure_size=(13.8, 4.8),
    )
    render_triptych(
        face_points,
        face_colors,
        face_path,
        suptitle=f"{case.case_id} face close-up ({case.view_tag}, {case.variant})",
        fixed_aabb=reference_boxes["face_aabb"],
        point_size=2.1,
        figure_size=(13.8, 4.8),
    )

    outputs = {"body": body_path, "head": head_path, "face": face_path}
    inventory.extend(
        [
            {
                "category": "pointcloud_open3d",
                "path": str(body_path.relative_to(output_dir)),
                "description": f"{case.view_tag} {case.variant} full-body point cloud triptych",
            },
            {
                "category": "pointcloud_open3d",
                "path": str(head_path.relative_to(output_dir)),
                "description": f"{case.view_tag} {case.variant} head close-up point cloud triptych",
            },
            {
                "category": "pointcloud_open3d",
                "path": str(face_path.relative_to(output_dir)),
                "description": f"{case.view_tag} {case.variant} face close-up point cloud triptych",
            },
        ]
    )
    return outputs


def compose_image_strip(
    image_paths: list[Path],
    titles: list[str],
    output_path: Path,
    suptitle: str,
    figsize_per_panel: tuple[float, float] = (4.8, 4.6),
) -> None:
    images = [open_rgb(path) for path in image_paths]
    fig, axes = plt.subplots(1, len(images), figsize=(figsize_per_panel[0] * len(images), figsize_per_panel[1]))
    if len(images) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
    fig.suptitle(suptitle, fontsize=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_docs(
    output_dir: Path,
    inventory: list[dict[str, Any]],
    files: dict[str, dict[str, Path]],
) -> tuple[Path, Path]:
    readme_path = output_dir / "README.md"
    assessment_path = output_dir / "docs" / "normal_pack_assessment.md"

    inventory_lines = [
        f"- `{item['category']}`: `{item['path']}` - {item['description']}"
        for item in inventory
    ]

    readme_text = "\n".join(
        [
            "# Normal Advisor Pack",
            "",
            f"- Generated on: {date.today().isoformat()}",
            "- Case: `0012_11`, frame `0000`",
            "- Available sparse-view ladders in the current workspace: `7v`, `13v`, `60v`",
            "- Extra predicted-normal evidence: `4v` frozen-backbone normal-head probe from `2026-04-20`",
            "",
            "## Directory Layout",
            "",
            "- `normals/`: prior-normal overviews and large target-view head/face ROI panels",
            "- `pointcloud_open3d/`: point-cloud triptychs for full body, head, and face",
            "- `comparisons/`: baseline vs surfacepose and normal-to-point summary figures",
            "- `sparse_views/`: fixed-region 7/13/60 sparse-view comparisons",
            "- `docs/`: assessment note",
            "",
            "## File Inventory",
            "",
            *inventory_lines,
            "",
            "## Key Notes",
            "",
            "- `7v` and `13v` currently have baseline geometry plus prior-normal observation only.",
            "- Direct baseline vs `+surfacepose` geometry comparison is available for `60v` in this pack.",
            "- The `4v` frozen probe figures show predicted normals, but they are not end-to-end geometry training curves.",
        ]
    )
    write_text(readme_path, readme_text)

    assessment_text = "\n".join(
        [
            "# normal_pack_assessment",
            "",
            "## What This Pack Can Prove",
            "",
            "- The normal visualization chain is stable: the pack now contains readable full-body overviews and enlarged head/face ROI figures instead of tiny crops.",
            "- The workspace does support a direct geometry comparison for `60v`: baseline VGGT versus the `60v surfacepose` run.",
            "- The sparse-view geometry trend is now easier to inspect because the `7v / 13v / 60v` head and face comparisons use a fixed 3D ROI window.",
            "",
            "## What It Still Cannot Fully Prove",
            "",
            "- The current workspace does not contain end-to-end `+normal` geometry outputs for `7v` or `13v` as of 2026-04-21, so the sparse-view figures are still baseline-only at the geometry level.",
            "- The `4v` frozen normal-head probe demonstrates predicted normal output, but it is still a frozen-backbone probe rather than a full end-to-end training result.",
            "- Because the sparse-view ladder available locally is `7v / 13v / 60v`, this pack does not claim a true `6v / 12v / 60v` result.",
            "",
            "## Current Bottleneck",
            "",
            "- The main missing evidence is not visualization any more; it is the absence of matching `+normal` end-to-end geometry runs for sparse-view settings.",
            "- That means the highest-value next step is to produce `7v` or `13v` end-to-end surfacepose/normal-conditioned inference so the new figure templates can be filled with a true baseline-vs-normal sparse-view ablation.",
            "",
            "## Priority Recommendation",
            "",
            "- First priority: run sparse-view end-to-end `+normal` inference or short training for `7v` and `13v`.",
            "- Second priority: if those runs are still unstable, strengthen the normal supervision branch before spending more time on additional visualization variants.",
            "- Third priority: only after sparse-view `+normal` geometry is available should we refine the pack toward a smaller mentor-facing subset.",
        ]
    )
    write_text(assessment_path, assessment_text)
    return readme_path, assessment_path


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        CaseSpec(
            slug="baseline_7v",
            case_id="0012_11",
            view_tag="7v",
            scene_dir=REPO_ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_7views",
            predictions_npz=REPO_ROOT / "output" / "modal_results" / "0012_11_frame0000_7views" / "predictions.npz",
            prior_maps_npz=REPO_ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_7views" / "prior_maps.npz",
            variant="baseline",
            source_label="vggt_baseline",
        ),
        CaseSpec(
            slug="baseline_13v",
            case_id="0012_11",
            view_tag="13v",
            scene_dir=REPO_ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_13views",
            predictions_npz=REPO_ROOT / "output" / "modal_results" / "0012_11_frame0000_13views" / "predictions.npz",
            prior_maps_npz=REPO_ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_13views" / "prior_maps.npz",
            variant="baseline",
            source_label="vggt_baseline",
        ),
        CaseSpec(
            slug="baseline_60v",
            case_id="0012_11",
            view_tag="60v",
            scene_dir=REPO_ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_60views",
            predictions_npz=REPO_ROOT / "output" / "modal_results" / "0012_11_frame0000_60views" / "predictions.npz",
            prior_maps_npz=REPO_ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_60views" / "prior_maps.npz",
            variant="baseline",
            source_label="vggt_baseline",
        ),
        CaseSpec(
            slug="surfacepose_60v",
            case_id="0012_11",
            view_tag="60v",
            scene_dir=REPO_ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_60views",
            predictions_npz=REPO_ROOT
            / "output"
            / "modal_results"
            / "0012_11_frame0000_60views_smplxsurfacepose_a10080_e2_r2"
            / "predictions.npz",
            prior_maps_npz=REPO_ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_60views" / "prior_maps.npz",
            variant="surfacepose",
            source_label="surfacepose_run",
        ),
    ]
    probe = ProbeSpec(
        case_id="0012_11",
        view_tag="4v_probe",
        inputs_npz=Path(r"D:\vggt\runs\normal_head\20260420_normal_probe\frozen_probe_surfacepose4v\probe_inputs.npz"),
        predictions_npz=Path(r"D:\vggt\runs\normal_head\20260420_normal_probe\frozen_probe_surfacepose4v\probe_predictions.npz"),
        summary_json=Path(r"D:\vggt\runs\normal_head\20260420_normal_probe\frozen_probe_surfacepose4v\probe_summary.json"),
    )

    inventory: list[dict[str, Any]] = []
    outputs: dict[str, dict[str, Path]] = {}

    reference_boxes = build_reference_boxes(next(case for case in cases if case.slug == "baseline_60v"))

    for case in cases:
        outputs[f"normals_{case.slug}"] = save_prior_normal_outputs(case, output_dir, inventory)

    outputs["normals_probe"] = save_probe_outputs(probe, output_dir, inventory)

    for idx, case in enumerate(cases):
        outputs[f"pointcloud_{case.slug}"] = save_pointcloud_outputs(
            case,
            output_dir,
            inventory,
            reference_boxes,
            conf_percentile=args.conf_percentile,
            max_full_points=args.max_full_points,
            max_roi_points=args.max_roi_points,
            rng_seed=args.seed + idx,
        )

    comparisons_dir = output_dir / "comparisons"
    sparse_dir = output_dir / "sparse_views"

    baseline_60 = outputs["pointcloud_baseline_60v"]
    surfacepose_60 = outputs["pointcloud_surfacepose_60v"]
    prior_60 = outputs["normals_baseline_60v"]
    sparse_7 = outputs["pointcloud_baseline_7v"]
    sparse_13 = outputs["pointcloud_baseline_13v"]

    compare_head_path = comparisons_dir / "0012_11_compare_baseline_vs_surfacepose_head_60v.png"
    compare_face_path = comparisons_dir / "0012_11_compare_baseline_vs_surfacepose_face_60v.png"
    demo_head_path = comparisons_dir / "0012_11_normal_to_point_head_demo_60v.png"
    demo_face_path = comparisons_dir / "0012_11_normal_to_point_face_demo_60v.png"
    sparse_head_path = sparse_dir / "0012_11_compare_views_head_7_13_60.png"
    sparse_face_path = sparse_dir / "0012_11_compare_views_face_7_13_60.png"

    compose_image_strip(
        [baseline_60["head"], surfacepose_60["head"], prior_60["head_roi"]],
        ["60v baseline head", "60v +surfacepose head", "Target RGB vs prior normal"],
        compare_head_path,
        suptitle="0012_11 60v baseline vs surfacepose head comparison",
        figsize_per_panel=(4.8, 4.8),
    )
    compose_image_strip(
        [baseline_60["face"], surfacepose_60["face"], prior_60["face_roi"]],
        ["60v baseline face", "60v +surfacepose face", "Target RGB vs prior normal"],
        compare_face_path,
        suptitle="0012_11 60v baseline vs surfacepose face comparison",
        figsize_per_panel=(4.8, 4.8),
    )
    compose_image_strip(
        [prior_60["head_roi"], baseline_60["head"], surfacepose_60["head"]],
        ["Target RGB vs prior normal", "60v baseline head cloud", "60v +surfacepose head cloud"],
        demo_head_path,
        suptitle="0012_11 head normal-to-point context demo",
        figsize_per_panel=(4.8, 4.8),
    )
    compose_image_strip(
        [prior_60["face_roi"], baseline_60["face"], surfacepose_60["face"]],
        ["Target RGB vs prior normal", "60v baseline face cloud", "60v +surfacepose face cloud"],
        demo_face_path,
        suptitle="0012_11 face normal-to-point context demo",
        figsize_per_panel=(4.8, 4.8),
    )
    compose_image_strip(
        [sparse_7["head"], sparse_13["head"], baseline_60["head"]],
        ["7v baseline head", "13v baseline head", "60v baseline head"],
        sparse_head_path,
        suptitle="0012_11 sparse-view head comparison (7v / 13v / 60v baseline)",
        figsize_per_panel=(4.8, 4.8),
    )
    compose_image_strip(
        [sparse_7["face"], sparse_13["face"], baseline_60["face"]],
        ["7v baseline face", "13v baseline face", "60v baseline face"],
        sparse_face_path,
        suptitle="0012_11 sparse-view face comparison (7v / 13v / 60v baseline)",
        figsize_per_panel=(4.8, 4.8),
    )

    inventory.extend(
        [
            {
                "category": "comparisons",
                "path": str(compare_head_path.relative_to(output_dir)),
                "description": "60v baseline vs surfacepose head comparison",
            },
            {
                "category": "comparisons",
                "path": str(compare_face_path.relative_to(output_dir)),
                "description": "60v baseline vs surfacepose face comparison",
            },
            {
                "category": "comparisons",
                "path": str(demo_head_path.relative_to(output_dir)),
                "description": "Head normal-to-point context demo",
            },
            {
                "category": "comparisons",
                "path": str(demo_face_path.relative_to(output_dir)),
                "description": "Face normal-to-point context demo",
            },
            {
                "category": "sparse_views",
                "path": str(sparse_head_path.relative_to(output_dir)),
                "description": "7v/13v/60v head sparse-view comparison",
            },
            {
                "category": "sparse_views",
                "path": str(sparse_face_path.relative_to(output_dir)),
                "description": "7v/13v/60v face sparse-view comparison",
            },
        ]
    )

    readme_path, assessment_path = build_docs(output_dir, inventory, outputs)
    manifest_path = output_dir / "docs" / "image_inventory.json"
    save_json(manifest_path, inventory)

    summary = {
        "output_dir": str(output_dir),
        "readme": str(readme_path),
        "assessment": str(assessment_path),
        "inventory_json": str(manifest_path),
        "generated_files": [item["path"] for item in inventory if not str(item["path"]).startswith(str(REPO_ROOT))],
        "mentor_minimal_pack": [
            str(compare_face_path.relative_to(output_dir)),
            str(surfacepose_60["head"].relative_to(output_dir)),
            str(surfacepose_60["face"].relative_to(output_dir)),
            str(sparse_face_path.relative_to(output_dir)),
        ],
        "notes": [
            "Sparse-view geometry comparison uses 7v/13v/60v because those are the actual available cases in the workspace.",
            "Only 60v has both baseline and surfacepose geometry outputs in the current workspace.",
            "4v frozen probe outputs are included as predicted-normal evidence, not as end-to-end geometry training evidence.",
        ],
    }
    save_json(output_dir / "docs" / "pack_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
