import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inventory local ZJU pseudo-geometry cache roots / subdirs / sampled view counts."
    )
    parser.add_argument("--zju_roots", nargs="*", default=[], help="Optional ZJU roots to scan before built-in defaults.")
    parser.add_argument("--seq_names", nargs="+", default=["CoreView_390"], help="Sequences to inspect.")
    parser.add_argument(
        "--geom_subdirs",
        nargs="+",
        default=["vggt_geom", "vggt_geom_4v_backup", "vggt_geom_test6", "vggt_geom_mvdebug_local"],
        help="Geometry cache subdirs to inspect under each sequence.",
    )
    parser.add_argument(
        "--sample_frames",
        type=int,
        default=8,
        help="Number of frame_*.npz files to sample when estimating per-frame view counts.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Optional output directory. Defaults to output/zju_geom_cache_inventory_<timestamp>/",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(REPO_ROOT / "output" / f"zju_geom_cache_inventory_{stamp}")


def build_root_candidates(requested_roots):
    roots = [Path(str(value)) for value in requested_roots]
    g_datasets = "G:\\" + chr(0x6570) + chr(0x636E) + chr(0x96C6)
    roots.extend(
        [
            Path(r"F:\datasets\ZJU_MoCap\data\zju_mocap"),
            Path(g_datasets) / "datasets" / "ZJU_MoCap" / "data" / "zju_mocap",
        ]
    )
    deduped = []
    seen = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(resolved)
    return deduped


def pick_sample_files(frame_paths, sample_frames):
    frame_paths = list(frame_paths)
    if len(frame_paths) <= sample_frames:
        return frame_paths
    keep = np.linspace(0, len(frame_paths) - 1, sample_frames, dtype=np.int64)
    return [frame_paths[int(idx)] for idx in keep]


def summarize_view_counts(frame_paths, sample_frames):
    if not frame_paths:
        return {
            "sampled_frame_count": 0,
            "sampled_view_count_min": None,
            "sampled_view_count_p50": None,
            "sampled_view_count_max": None,
            "sample_examples": [],
        }

    sampled = []
    for frame_path in pick_sample_files(frame_paths, sample_frames):
        payload = np.load(frame_path, allow_pickle=True)
        sampled.append(
            {
                "frame": frame_path.stem,
                "view_count": int(len(payload["cam_names"])),
            }
        )

    counts = np.asarray([row["view_count"] for row in sampled], dtype=np.int32)
    return {
        "sampled_frame_count": int(len(sampled)),
        "sampled_view_count_min": int(counts.min()),
        "sampled_view_count_p50": float(np.median(counts)),
        "sampled_view_count_max": int(counts.max()),
        "sample_examples": sampled,
    }


def inspect_geom_dir(geom_dir: Path, sample_frames: int) -> dict:
    if not geom_dir.exists():
        return {
            "status": "missing",
            "frame_count": 0,
            "selection_headroom_num_images_4": False,
            "selection_headroom_num_images_3": False,
            "path": str(geom_dir),
        }

    frame_paths = sorted(geom_dir.glob("frame_*.npz"))
    if not frame_paths:
        return {
            "status": "empty",
            "frame_count": 0,
            "selection_headroom_num_images_4": False,
            "selection_headroom_num_images_3": False,
            "path": str(geom_dir),
        }

    stats = summarize_view_counts(frame_paths, sample_frames)
    max_views = int(stats["sampled_view_count_max"])
    return {
        "status": "ok",
        "frame_count": int(len(frame_paths)),
        "selection_headroom_num_images_4": bool(max_views > 4),
        "selection_headroom_num_images_3": bool(max_views > 3),
        "path": str(geom_dir),
        **stats,
    }


def pick_recommended_root(rows):
    viable = [row for row in rows if row["status"] == "ok"]
    if not viable:
        return None
    viable.sort(
        key=lambda row: (
            int(row["frame_count"]),
            int(row.get("sampled_view_count_max") or 0),
            int(row.get("sampled_view_count_min") or 0),
        ),
        reverse=True,
    )
    best = viable[0]
    return {
        "root": best["root"],
        "seq_name": best["seq_name"],
        "geom_subdir": best["geom_subdir"],
        "frame_count": best["frame_count"],
        "sampled_view_count_min": best.get("sampled_view_count_min"),
        "sampled_view_count_max": best.get("sampled_view_count_max"),
        "selection_headroom_num_images_4": best["selection_headroom_num_images_4"],
        "selection_headroom_num_images_3": best["selection_headroom_num_images_3"],
    }


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# ZJU Geom Cache Inventory",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- sample_frames: `{payload['sample_frames']}`",
        f"- seq_names: `{payload['seq_names']}`",
        f"- geom_subdirs: `{payload['geom_subdirs']}`",
        "",
        "## Recommended Roots",
        "",
    ]
    for geom_subdir, summary in sorted(payload["recommended_roots"].items()):
        if summary is None:
            lines.append(f"- `{geom_subdir}`: no viable root")
            continue
        lines.append(
            "- `{geom_subdir}`: root=`{root}` seq=`{seq}` frame_count=`{frames}` sampled_views=`{vmin}..{vmax}` "
            "headroom@4=`{h4}` headroom@3=`{h3}`".format(
                geom_subdir=geom_subdir,
                root=summary["root"],
                seq=summary["seq_name"],
                frames=summary["frame_count"],
                vmin=summary["sampled_view_count_min"],
                vmax=summary["sampled_view_count_max"],
                h4=summary["selection_headroom_num_images_4"],
                h3=summary["selection_headroom_num_images_3"],
            )
        )

    lines.extend(["", "## Details", ""])
    for seq_name, seq_payload in sorted(payload["sequences"].items()):
        lines.append(f"### {seq_name}")
        lines.append("")
        for geom_subdir, rows in sorted(seq_payload.items()):
            lines.append(f"- `{geom_subdir}`")
            for row in rows:
                if row["status"] != "ok":
                    lines.append(
                        "  - root=`{root}` status=`{status}` frame_count=`{frame_count}`".format(
                            root=row["root"],
                            status=row["status"],
                            frame_count=row["frame_count"],
                        )
                    )
                    continue
                lines.append(
                    "  - root=`{root}` frame_count=`{frame_count}` sampled_views=`{vmin}/{vp50}/{vmax}` "
                    "headroom@4=`{h4}` headroom@3=`{h3}`".format(
                        root=row["root"],
                        frame_count=row["frame_count"],
                        vmin=row["sampled_view_count_min"],
                        vp50=row["sampled_view_count_p50"],
                        vmax=row["sampled_view_count_max"],
                        h4=row["selection_headroom_num_images_4"],
                        h3=row["selection_headroom_num_images_3"],
                    )
                )
        lines.append("")

    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    ensure_dir(output_dir)
    roots = build_root_candidates(args.zju_roots)

    sequence_payload = {}
    geom_rows = {geom_subdir: [] for geom_subdir in args.geom_subdirs}
    for seq_name in args.seq_names:
        seq_result = {}
        for geom_subdir in args.geom_subdirs:
            rows = []
            for root in roots:
                geom_dir = root / seq_name / geom_subdir
                entry = inspect_geom_dir(geom_dir, args.sample_frames)
                entry.update(
                    {
                        "root": str(root),
                        "seq_name": seq_name,
                        "geom_subdir": geom_subdir,
                    }
                )
                rows.append(entry)
                geom_rows[geom_subdir].append(entry)
            seq_result[geom_subdir] = rows
        sequence_payload[seq_name] = seq_result

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sample_frames": int(args.sample_frames),
        "zju_roots": [str(root) for root in roots],
        "seq_names": list(args.seq_names),
        "geom_subdirs": list(args.geom_subdirs),
        "recommended_roots": {
            geom_subdir: pick_recommended_root(rows)
            for geom_subdir, rows in geom_rows.items()
        },
        "sequences": sequence_payload,
    }
    write_json(output_dir / "summary.json", payload)
    write_markdown(output_dir / "summary.md", payload)
    print(output_dir / "summary.md")


if __name__ == "__main__":
    main()
