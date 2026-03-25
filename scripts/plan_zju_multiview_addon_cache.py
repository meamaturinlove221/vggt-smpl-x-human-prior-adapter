import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plan full-length multiview add-on cache coverage relative to a baseline ZJU geom cache."
    )
    parser.add_argument("--zju_root", type=str, required=True)
    parser.add_argument("--seq_name", type=str, default="CoreView_390")
    parser.add_argument("--base_geom_subdir", type=str, required=True)
    parser.add_argument("--addon_geom_subdirs", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_dir(seq_name: str, base_subdir: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(REPO_ROOT / "output" / f"zju_multiview_addon_plan_{seq_name}_{base_subdir}_{stamp}")


def list_valid_frames(geom_dir: Path):
    frame_map = {}
    invalid_names = []
    if not geom_dir.is_dir():
        return frame_map, invalid_names
    for path in sorted(geom_dir.iterdir()):
        if not path.is_file():
            continue
        if not path.name.startswith("frame_"):
            continue
        if path.suffix != ".npz":
            invalid_names.append(path.name)
            continue
        try:
            frame_id = int(path.stem.split("_")[-1])
        except ValueError:
            invalid_names.append(path.name)
            continue
        frame_map[frame_id] = path
    return frame_map, invalid_names


def load_camera_set(npz_path: Path) -> set[str]:
    payload = np.load(npz_path, allow_pickle=True)
    return {str(value) for value in payload["cam_names"].tolist()}


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict) -> None:
    base = payload["base"]
    lines = [
        "# ZJU Multiview Add-on Cache Plan",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- zju_root: `{payload['zju_root']}`",
        f"- seq_name: `{payload['seq_name']}`",
        f"- base_geom_subdir: `{base['geom_subdir']}`",
        f"- base_frame_count: `{base['frame_count']}`",
        f"- base_camera_set: `{base['stable_camera_set']}`",
        f"- addon_dirs_considered: `{payload['addon_geom_subdirs']}`",
        f"- addon_frame_union_count: `{payload['addon_frame_union_count']}`",
        f"- missing_addon_frame_count: `{payload['missing_addon_frame_count']}`",
        f"- recommended_extra_cameras: `{payload['recommended_extra_cameras']}`",
        f"- recommended_target_union_cameras: `{payload['recommended_target_union_cameras']}`",
        "",
        "## Add-on Coverage",
        "",
    ]

    for row in payload["addon_details"]:
        lines.append(
            "- `{geom_subdir}`: frame_count=`{frame_count}` invalid_entries=`{invalid_entries}` "
            "stable_new_cameras_vs_base=`{stable_new_cameras_vs_base}`".format(
                geom_subdir=row["geom_subdir"],
                frame_count=row["frame_count"],
                invalid_entries=row["invalid_entries"],
                stable_new_cameras_vs_base=row["stable_new_cameras_vs_base"],
            )
        )

    lines.extend(
        [
            "",
            "## Missing Frames",
            "",
            f"- first_missing_frame_ids: `{payload['missing_addon_frame_ids'][:16]}`",
            f"- last_missing_frame_ids: `{payload['missing_addon_frame_ids'][-16:]}`",
            "",
            "## Manifest",
            "",
            f"- [full_length_addon_manifest.json]({payload['full_length_addon_manifest_json']})",
        ]
    )

    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.seq_name, args.base_geom_subdir)
    ensure_dir(output_dir)

    zju_root = Path(args.zju_root)
    seq_dir = zju_root / args.seq_name
    base_dir = seq_dir / args.base_geom_subdir
    base_frames, base_invalid = list_valid_frames(base_dir)
    if not base_frames:
        raise FileNotFoundError(f"No valid frame_*.npz files found under base geom dir: {base_dir}")

    base_camera_rows = [load_camera_set(path) for _, path in sorted(base_frames.items())[: min(8, len(base_frames))]]
    stable_base_cameras = sorted(set.intersection(*base_camera_rows)) if base_camera_rows else []

    addon_details = []
    addon_frame_union = set()
    stable_extra_sets = []
    for geom_subdir in args.addon_geom_subdirs:
        addon_dir = seq_dir / geom_subdir
        addon_frames, invalid_names = list_valid_frames(addon_dir)
        addon_frame_union.update(addon_frames.keys())
        sampled_rows = []
        for frame_id, addon_path in sorted(addon_frames.items())[: min(8, len(addon_frames))]:
            if frame_id not in base_frames:
                continue
            base_cameras = load_camera_set(base_frames[frame_id])
            addon_cameras = load_camera_set(addon_path)
            sampled_rows.append(
                {
                    "frame_id": int(frame_id),
                    "new_cameras_vs_base": sorted(addon_cameras - base_cameras),
                    "addon_cameras": sorted(addon_cameras),
                }
            )
        if sampled_rows:
            stable_new_cameras = sorted(set.intersection(*[set(row["new_cameras_vs_base"]) for row in sampled_rows]))
            stable_extra_sets.append(set(stable_new_cameras))
        else:
            stable_new_cameras = []
        addon_details.append(
            {
                "geom_subdir": geom_subdir,
                "frame_count": len(addon_frames),
                "invalid_entries": invalid_names,
                "sampled_rows": sampled_rows,
                "stable_new_cameras_vs_base": stable_new_cameras,
            }
        )

    recommended_extra_cameras = sorted(set.union(*stable_extra_sets)) if stable_extra_sets else []
    target_union_cameras = sorted(set(stable_base_cameras) | set(recommended_extra_cameras))
    missing_frame_ids = sorted(frame_id for frame_id in base_frames if frame_id not in addon_frame_union)

    full_length_manifest = {
        "schema_version": 1,
        "status": "needs_full_length_multiview_addon",
        "zju_root": str(zju_root),
        "seq_name": args.seq_name,
        "base_geom_subdir": args.base_geom_subdir,
        "addon_geom_subdirs": list(args.addon_geom_subdirs),
        "base_frame_count": len(base_frames),
        "addon_frame_union_count": len(addon_frame_union),
        "missing_addon_frame_count": len(missing_frame_ids),
        "missing_addon_frame_ids": missing_frame_ids,
        "recommended_extra_cameras": recommended_extra_cameras,
        "recommended_target_union_cameras": target_union_cameras,
        "note": "Existing partial add-on caches prove the local union code path, but they do not yet provide full-length multiview coverage.",
    }

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "zju_root": str(zju_root),
        "seq_name": args.seq_name,
        "base": {
            "geom_subdir": args.base_geom_subdir,
            "frame_count": len(base_frames),
            "invalid_entries": base_invalid,
            "stable_camera_set": stable_base_cameras,
        },
        "addon_geom_subdirs": list(args.addon_geom_subdirs),
        "addon_frame_union_count": len(addon_frame_union),
        "missing_addon_frame_count": len(missing_frame_ids),
        "missing_addon_frame_ids": missing_frame_ids,
        "recommended_extra_cameras": recommended_extra_cameras,
        "recommended_target_union_cameras": target_union_cameras,
        "addon_details": addon_details,
        "full_length_addon_manifest_json": str((output_dir / "full_length_addon_manifest.json").resolve()),
    }
    write_json(output_dir / "summary.json", payload)
    write_json(output_dir / "full_length_addon_manifest.json", full_length_manifest)
    write_markdown(output_dir / "summary.md", payload)
    print(output_dir / "summary.md")


if __name__ == "__main__":
    main()
