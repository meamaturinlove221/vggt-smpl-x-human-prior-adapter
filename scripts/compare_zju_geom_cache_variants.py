import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two ZJU geometry-cache variants and measure whether their camera pools are complementary."
    )
    parser.add_argument("--seq_name", type=str, default="CoreView_390")
    parser.add_argument("--left_root", type=str, required=True)
    parser.add_argument("--left_geom_subdir", type=str, required=True)
    parser.add_argument("--right_root", type=str, required=True)
    parser.add_argument("--right_geom_subdir", type=str, required=True)
    parser.add_argument("--sample_frames", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_dir(seq_name: str, left_subdir: str, right_subdir: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = f"{seq_name}_{left_subdir}_vs_{right_subdir}_{stamp}".replace(",", "_")
    return ensure_dir(REPO_ROOT / "output" / f"zju_geom_cache_compare_{safe}")


def pick_sample_names(frame_names, sample_frames):
    frame_names = list(frame_names)
    if len(frame_names) <= sample_frames:
        return frame_names
    keep = np.linspace(0, len(frame_names) - 1, sample_frames, dtype=np.int64)
    return [frame_names[int(idx)] for idx in keep]


def load_camera_set(npz_path: Path) -> set[str]:
    payload = np.load(npz_path, allow_pickle=True)
    return {str(value) for value in payload["cam_names"].tolist()}


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict) -> None:
    left = payload["left"]
    right = payload["right"]
    overlap = payload["overlap"]
    lines = [
        "# ZJU Geom Cache Variant Compare",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- seq_name: `{payload['seq_name']}`",
        f"- left: root=`{left['root']}` geom_subdir=`{left['geom_subdir']}` frame_count=`{left['frame_count']}`",
        f"- right: root=`{right['root']}` geom_subdir=`{right['geom_subdir']}` frame_count=`{right['frame_count']}`",
        f"- common_frame_count: `{overlap['common_frame_count']}`",
        f"- sampled_common_frames: `{overlap['sampled_common_frames']}`",
        f"- sampled_union_size_min: `{overlap['sampled_union_size_min']}`",
        f"- sampled_union_size_p50: `{overlap['sampled_union_size_p50']}`",
        f"- sampled_union_size_max: `{overlap['sampled_union_size_max']}`",
        f"- adds_headroom_over_left_at_num_images_4: `{overlap['adds_headroom_over_left_at_num_images_4']}`",
        f"- adds_new_cameras_in_any_sample: `{overlap['adds_new_cameras_in_any_sample']}`",
        "",
        "## Sampled Frames",
        "",
    ]
    for row in overlap["sample_rows"]:
        lines.append(
            "- `{frame}`: left=`{left_count}` right=`{right_count}` union=`{union_count}` "
            "left_only=`{left_only}` right_only=`{right_only}`".format(
                frame=row["frame"],
                left_count=row["left_count"],
                right_count=row["right_count"],
                union_count=row["union_count"],
                left_only=row["left_only"],
                right_only=row["right_only"],
            )
        )
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(
        args.seq_name,
        args.left_geom_subdir,
        args.right_geom_subdir,
    )
    ensure_dir(output_dir)

    left_dir = Path(args.left_root) / args.seq_name / args.left_geom_subdir
    right_dir = Path(args.right_root) / args.seq_name / args.right_geom_subdir
    left_names = sorted(p.name for p in left_dir.glob("frame_*.npz")) if left_dir.exists() else []
    right_names = sorted(p.name for p in right_dir.glob("frame_*.npz")) if right_dir.exists() else []
    common = sorted(set(left_names) & set(right_names))

    sample_rows = []
    union_counts = []
    adds_new = False
    for frame_name in pick_sample_names(common, args.sample_frames):
        left_set = load_camera_set(left_dir / frame_name)
        right_set = load_camera_set(right_dir / frame_name)
        union_set = left_set | right_set
        left_only = sorted(left_set - right_set)
        right_only = sorted(right_set - left_set)
        if left_only or right_only:
            adds_new = True
        sample_rows.append(
            {
                "frame": frame_name,
                "left_count": len(left_set),
                "right_count": len(right_set),
                "union_count": len(union_set),
                "left_only": left_only,
                "right_only": right_only,
                "union_cameras": sorted(union_set),
            }
        )
        union_counts.append(len(union_set))

    union_counts_arr = np.asarray(union_counts, dtype=np.int32) if union_counts else None
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seq_name": args.seq_name,
        "left": {
            "root": str(Path(args.left_root)),
            "geom_subdir": args.left_geom_subdir,
            "frame_count": len(left_names),
        },
        "right": {
            "root": str(Path(args.right_root)),
            "geom_subdir": args.right_geom_subdir,
            "frame_count": len(right_names),
        },
        "overlap": {
            "common_frame_count": len(common),
            "sampled_common_frames": len(sample_rows),
            "sampled_union_size_min": int(union_counts_arr.min()) if union_counts else None,
            "sampled_union_size_p50": float(np.median(union_counts_arr)) if union_counts else None,
            "sampled_union_size_max": int(union_counts_arr.max()) if union_counts else None,
            "adds_headroom_over_left_at_num_images_4": bool(union_counts and np.max(union_counts_arr) > 4),
            "adds_new_cameras_in_any_sample": bool(adds_new),
            "sample_rows": sample_rows,
        },
    }
    write_json(output_dir / "summary.json", payload)
    write_markdown(output_dir / "summary.md", payload)
    print(output_dir / "summary.md")


if __name__ == "__main__":
    main()
