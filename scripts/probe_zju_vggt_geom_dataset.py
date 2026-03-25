import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"

for root in (str(REPO_ROOT), str(TRAINING_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from training.data.datasets.zju_vggt_geom import ZjuVggtGeomDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Probe the ZJU VGGT-geom pseudo-supervision dataset.")
    parser.add_argument(
        "--zju_dir",
        type=str,
        default="",
        help="Optional local ZJU root. Auto-detected when omitted.",
    )
    parser.add_argument("--seq_names", nargs="+", default=["CoreView_390"])
    parser.add_argument("--geom_subdir", type=str, default="vggt_geom")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--camera_source", type=str, default="gt", choices=["gt", "geom"])
    parser.add_argument("--mask_source", type=str, default="mask", choices=["none", "mask", "mask_cihp"])
    parser.add_argument("--source_policy", type=str, default="random", choices=["random", "nearest_ring", "uniform_ring"])
    parser.add_argument("--source_view_pool", type=str, default="cached_only", choices=["cached_only", "geom_plus_raw"])
    parser.add_argument("--source_view_pool_train_probability", type=float, default=1.0)
    parser.add_argument("--source_anchor_policy", type=str, default="random", choices=["random", "max_depth_conf"])
    parser.add_argument("--min_supervised_views", type=int, default=1)
    parser.add_argument(
        "--supervised_view_quality_filter",
        type=str,
        default="none",
        choices=["none", "drop_worst_by_depth_conf_if_multi_supervised"],
    )
    parser.add_argument(
        "--conf_depth_view_quality_filter",
        type=str,
        default="none",
        choices=["none", "drop_worst_by_depth_conf_if_multi_supervised"],
    )
    parser.add_argument("--allow_duplicate_img", action="store_true", help="Allow duplicate view sampling in the probe.")
    parser.add_argument("--min_depth_conf", type=float, default=0.0)
    parser.add_argument("--holdout_stride", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="output/zju_vggt_geom_probe")
    return parser.parse_args()


def resolve_zju_dir(requested, seq_names, geom_subdir):
    candidates = []
    requested = str(requested).strip()
    if requested:
        candidates.append(Path(requested))
    g_datasets = "G:\\" + chr(0x6570) + chr(0x636E) + chr(0x96C6)
    candidates.extend(
        [
            Path(r"F:\datasets\ZJU_MoCap\data\zju_mocap"),
            Path(g_datasets) / "datasets" / "ZJU_MoCap" / "data" / "zju_mocap",
        ]
    )

    geom_subdirs = [item.strip() for item in str(geom_subdir).split(",") if item.strip()]
    seen = set()
    best_candidate = None
    best_score = None
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            candidate = candidate.absolute()
        candidate_key = str(candidate).lower()
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if not candidate.is_dir():
            continue
        valid_subdir_count = 0
        total_frame_count = 0
        for seq_name in seq_names:
            for geom_subdir_name in geom_subdirs:
                geom_dir = candidate / str(seq_name) / geom_subdir_name
                if not geom_dir.is_dir():
                    continue
                frame_count = sum(1 for _ in geom_dir.glob("frame_*.npz"))
                if frame_count > 0:
                    valid_subdir_count += 1
                    total_frame_count += frame_count
        if valid_subdir_count <= 0:
            continue
        score = (int(valid_subdir_count), int(total_frame_count))
        if best_candidate is None or score > best_score:
            best_candidate = candidate
            best_score = score
    if best_candidate is not None:
        return best_candidate
    raise FileNotFoundError(
        f"Unable to resolve a ZJU root with {seq_names[0]}/{geom_subdir}/frame_*.npz from candidates: {candidates}"
    )


def write_binary_ply(path, points, colors):
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    vertex_data = np.empty(
        points.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex_data["x"] = points[:, 0]
    vertex_data["y"] = points[:, 1]
    vertex_data["z"] = points[:, 2]
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        vertex_data.tofile(handle)


def save_mosaic(path, images_hwc):
    labeled = [Image.fromarray(np.asarray(img, dtype=np.uint8)) for img in images_hwc]
    width = max(im.width for im in labeled)
    height = max(im.height for im in labeled)
    cols = min(4, len(labeled))
    rows = (len(labeled) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * width, rows * height), color=(16, 16, 16))
    for idx, image in enumerate(labeled):
        row = idx // cols
        col = idx % cols
        canvas.paste(image, (col * width, row * height))
    canvas.save(path)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zju_dir = resolve_zju_dir(args.zju_dir, args.seq_names, args.geom_subdir)

    common_conf = OmegaConf.create(
        {
            "debug": False,
            "training": args.split == "train",
            "inside_random": False,
            "allow_duplicate_img": bool(args.allow_duplicate_img),
            "load_depth": True,
            "img_size": 518,
            "patch_size": 14,
            "rescale": True,
            "rescale_aug": False,
            "landscape_check": False,
            "augs": {
                "scales": None,
            },
        }
    )

    dataset = ZjuVggtGeomDataset(
        common_conf=common_conf,
        split=args.split,
        ZJU_DIR=str(zju_dir),
        seq_names=args.seq_names,
        geom_subdir=args.geom_subdir,
        holdout_stride=args.holdout_stride,
        camera_source=args.camera_source,
        mask_source=args.mask_source,
        source_policy=args.source_policy,
        source_view_pool=args.source_view_pool,
        source_view_pool_train_probability=args.source_view_pool_train_probability,
        source_anchor_policy=args.source_anchor_policy,
        min_supervised_views=args.min_supervised_views,
        supervised_view_quality_filter=args.supervised_view_quality_filter,
        conf_depth_view_quality_filter=args.conf_depth_view_quality_filter,
        min_depth_conf=args.min_depth_conf,
        len_train=-1,
        len_test=-1,
    )

    sample = dataset.get_data(seq_index=args.sample_index, img_per_seq=args.num_images)

    images = np.stack(sample["images"]).astype(np.uint8)
    depths = np.stack(sample["depths"]).astype(np.float32)
    world_points = np.stack(sample["world_points"]).astype(np.float32)
    masks = np.stack(sample["point_masks"]).astype(bool)

    flat_points = world_points.reshape(-1, 3)[masks.reshape(-1)]
    flat_colors = images.reshape(-1, 3)[masks.reshape(-1)]
    if flat_points.shape[0] > 250000:
        keep = np.linspace(0, flat_points.shape[0] - 1, 250000, dtype=np.int64)
        flat_points = flat_points[keep]
        flat_colors = flat_colors[keep]

    write_binary_ply(output_dir / "sample_world_points.ply", flat_points, flat_colors)
    save_mosaic(output_dir / "sample_images.png", images)

    depth_vis = []
    for depth in depths:
        valid = depth > 0
        if valid.any():
            p95 = np.percentile(depth[valid], 95.0)
            scaled = np.clip(depth / max(p95, 1e-6), 0.0, 1.0)
        else:
            scaled = np.zeros_like(depth)
        depth_vis.append((scaled * 255.0).astype(np.uint8))
    save_mosaic(output_dir / "sample_depths.png", [np.stack([d, d, d], axis=-1) for d in depth_vis])

    payload = {
        "dataset_len": len(dataset),
        "sequence_list_len": dataset.sequence_list_len,
        "sample_seq_name": sample["seq_name"],
        "ids": sample["ids"].tolist(),
        "num_images": len(sample["images"]),
        "available_view_count": int(sample.get("available_view_count", len(sample["images"]))),
        "image_shape": list(images.shape[1:]),
        "depth_shape": list(depths.shape[1:]),
        "valid_ratio": float(masks.mean()),
        "valid_points": int(masks.sum()),
        "camera_source": args.camera_source,
        "mask_source": args.mask_source,
        "source_policy": args.source_policy,
        "requested_source_view_pool": str(sample.get("selection_requested_source_view_pool", args.source_view_pool)),
        "effective_source_view_pool": str(sample.get("selection_source_view_pool", args.source_view_pool)),
        "source_view_pool_train_probability": float(
            sample.get("selection_source_view_pool_train_probability", args.source_view_pool_train_probability)
        ),
        "rawpool_candidate_pool_used": bool(
            sample.get(
                "selection_rawpool_candidate_pool_used",
                sample.get("selection_source_view_pool", args.source_view_pool) == "geom_plus_raw",
            )
        ),
        "source_anchor_policy": str(sample.get("selection_source_anchor_policy", args.source_anchor_policy)),
        "min_supervised_views": int(sample.get("selection_min_supervised_views", args.min_supervised_views)),
        "supervised_view_quality_filter": str(
            sample.get("selection_supervised_view_quality_filter", args.supervised_view_quality_filter)
        ),
        "camera_names": list(sample.get("camera_names", [])),
        "candidate_supervised_camera_names": list(sample.get("candidate_supervised_camera_names", [])),
        "supervised_camera_names": list(sample.get("supervised_camera_names", [])),
        "source_only_camera_names": list(sample.get("source_only_camera_names", [])),
        "source_only_view_count": int(len(sample.get("source_only_camera_names", []))),
        "supervised_view_count": int(len(sample.get("supervised_camera_names", []))),
        "dropped_supervised_camera_names": list(sample.get("dropped_supervised_camera_names", [])),
        "conf_depth_camera_names": list(sample.get("conf_depth_camera_names", [])),
        "conf_depth_dropped_camera_names": list(sample.get("conf_depth_dropped_camera_names", [])),
        "supervised_view_quality_scores": dict(sample.get("supervised_view_quality_scores", {})),
        "geom_subdirs_present": list(sample.get("geom_subdirs_present", [])),
        "selection_anchor_camera": sample.get("selection_anchor_camera"),
        "available_candidate_view_count": int(sample.get("available_candidate_view_count", sample.get("available_view_count", len(sample["images"])))),
        "available_candidate_camera_names": list(sample.get("available_candidate_camera_names", [])),
        "zju_dir": str(zju_dir),
        "geom_subdir": args.geom_subdir,
        "seq_names": args.seq_names,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# ZJU VGGT-Geom Dataset Probe",
                "",
                f"- dataset_len: `{payload['dataset_len']}`",
                f"- sequence_list_len: `{payload['sequence_list_len']}`",
                f"- sample_seq_name: `{payload['sample_seq_name']}`",
                f"- ids: `{payload['ids']}`",
                f"- num_images: `{payload['num_images']}`",
                f"- available_view_count: `{payload['available_view_count']}`",
                f"- image_shape: `{payload['image_shape']}`",
                f"- depth_shape: `{payload['depth_shape']}`",
                f"- valid_ratio: `{payload['valid_ratio']:.4f}`",
                f"- valid_points: `{payload['valid_points']}`",
                f"- camera_source: `{payload['camera_source']}`",
                f"- mask_source: `{payload['mask_source']}`",
                f"- source_policy: `{payload['source_policy']}`",
                f"- requested_source_view_pool: `{payload['requested_source_view_pool']}`",
                f"- effective_source_view_pool: `{payload['effective_source_view_pool']}`",
                f"- source_view_pool_train_probability: `{payload['source_view_pool_train_probability']}`",
                f"- rawpool_candidate_pool_used: `{payload['rawpool_candidate_pool_used']}`",
                f"- source_anchor_policy: `{payload['source_anchor_policy']}`",
                f"- min_supervised_views: `{payload['min_supervised_views']}`",
                f"- supervised_view_quality_filter: `{payload['supervised_view_quality_filter']}`",
                f"- camera_names: `{payload['camera_names']}`",
                f"- candidate_supervised_camera_names: `{payload['candidate_supervised_camera_names']}`",
                f"- supervised_camera_names: `{payload['supervised_camera_names']}`",
                f"- source_only_camera_names: `{payload['source_only_camera_names']}`",
                f"- supervised_view_count: `{payload['supervised_view_count']}`",
                f"- source_only_view_count: `{payload['source_only_view_count']}`",
                f"- dropped_supervised_camera_names: `{payload['dropped_supervised_camera_names']}`",
                f"- conf_depth_camera_names: `{payload['conf_depth_camera_names']}`",
                f"- conf_depth_dropped_camera_names: `{payload['conf_depth_dropped_camera_names']}`",
                f"- supervised_view_quality_scores: `{payload['supervised_view_quality_scores']}`",
                f"- geom_subdirs_present: `{payload['geom_subdirs_present']}`",
                f"- selection_anchor_camera: `{payload['selection_anchor_camera']}`",
                f"- available_candidate_view_count: `{payload['available_candidate_view_count']}`",
                f"- available_candidate_camera_names: `{payload['available_candidate_camera_names']}`",
                "",
                "## Files",
                "",
                "- `sample_images.png`",
                "- `sample_depths.png`",
                "- `sample_world_points.ply`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_dir / "summary.md")


if __name__ == "__main__":
    main()
