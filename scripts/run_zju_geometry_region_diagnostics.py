import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_geometry_branches_zju_report import detect_local_zju_root, load_zju_cameras, scale_intrinsic
from scripts.zju_geometry_region_utils import compute_region_diagnostics, save_json, write_region_markdown_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retroactively compute target-mask region diagnostics from existing ZJU geometry compare case directories."
    )
    parser.add_argument("--case_dir", nargs="+", required=True, help="Existing compare case directories that contain summary.json.")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--local_zju_root", type=str, default="")
    parser.add_argument("--target_mask_source", type=str, default="", choices=["", "none", "mask", "mask_cihp"])
    parser.add_argument("--region_edge_px", type=int, default=-1)
    parser.add_argument("--bottom_band_ratio", type=float, default=-1.0)
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def resolve_output_dir(output_dir):
    if output_dir:
        return ensure_dir(output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path("output") / f"geometry_region_diagnostics_{stamp}")


def load_rgb01(path, render_hw):
    image = Image.open(path).convert("RGB")
    image = image.resize((int(render_hw[1]), int(render_hw[0])), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_gray01(path, render_hw):
    image = Image.open(path).convert("L")
    image = image.resize((int(render_hw[1]), int(render_hw[0])), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.float32) / 255.0


def read_image_hw(path):
    image = Image.open(path)
    width, height = image.size
    image.close()
    return (height, width)


def resolve_local_zju_root(args_local_root, summary_local_root):
    if args_local_root:
        return Path(args_local_root).resolve()
    if summary_local_root:
        candidate = Path(summary_local_root)
        if candidate.is_dir():
            return candidate.resolve()
    return detect_local_zju_root().resolve()


def load_source_colors(source_image_paths, render_hw):
    colors = []
    for image_path in source_image_paths:
        colors.append(load_rgb01(Path(image_path), render_hw))
    return np.stack(colors, axis=0).astype(np.float32)


def process_case(case_dir, case_output_dir, args):
    case_dir = Path(case_dir).resolve()
    summary_path = case_dir / "summary.json"
    predictions_path = case_dir / "predictions.npz"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary.json: {summary_path}")
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Missing predictions.npz: {predictions_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    predictions = np.load(predictions_path, allow_pickle=True)

    render_hw = tuple(int(v) for v in summary["run_config"]["render_size"])
    local_zju_root = resolve_local_zju_root(args.local_zju_root, summary["run_config"].get("local_zju_root", ""))
    seq_dir = local_zju_root / summary["case"]["seq_name"]
    frame_id = int(summary["case"]["frame_id"])
    target_camera = str(summary["case"]["target_camera"])

    target_image = load_rgb01(Path(summary["case"]["target_image_path"]), render_hw)
    point_render = {
        "image": load_rgb01(case_dir / summary["files"]["point_map_render_png"], render_hw),
        "weight": load_gray01(case_dir / summary["files"]["point_map_weight_png"], render_hw),
    }
    depth_render = {
        "image": load_rgb01(case_dir / summary["files"]["depth_unproject_render_png"], render_hw),
        "weight": load_gray01(case_dir / summary["files"]["depth_unproject_weight_png"], render_hw),
    }
    source_colors = load_source_colors(summary["case"]["source_image_paths"], render_hw)

    gt_cameras = load_zju_cameras(seq_dir, [target_camera])
    original_hw = read_image_hw(Path(summary["case"]["target_image_path"]))
    target_intrinsic = scale_intrinsic(
        gt_cameras[target_camera]["intrinsic"],
        original_hw,
        render_hw,
    )
    target_extrinsic = gt_cameras[target_camera]["extrinsic"]

    target_mask_source = args.target_mask_source or summary["run_config"].get("target_mask_source", "mask")
    region_edge_px = args.region_edge_px if args.region_edge_px >= 0 else int(summary["run_config"].get("region_edge_px", 5))
    bottom_band_ratio = (
        args.bottom_band_ratio
        if args.bottom_band_ratio >= 0.0
        else float(summary["run_config"].get("bottom_band_ratio", 0.2))
    )

    region_payload = compute_region_diagnostics(
        output_dir=case_output_dir,
        seq_dir=seq_dir,
        frame_id=frame_id,
        target_camera=target_camera,
        target_image=target_image,
        target_extrinsic=target_extrinsic,
        target_intrinsic=target_intrinsic,
        source_colors=source_colors,
        point_map_aligned=np.asarray(predictions["point_map_aligned"], dtype=np.float32),
        point_conf=np.asarray(predictions["point_conf"], dtype=np.float32),
        depth_points_aligned=np.asarray(predictions["depth_points_aligned"], dtype=np.float32),
        depth_conf=np.asarray(predictions["depth_conf"], dtype=np.float32),
        point_render=point_render,
        depth_render=depth_render,
        target_mask_source=target_mask_source,
        region_edge_px=region_edge_px,
        bottom_band_ratio=bottom_band_ratio,
        min_conf=float(summary["run_config"].get("min_conf", 1e-6)),
        export_max_points=int(summary["run_config"].get("export_max_points", 250000)),
        case_meta={
            "seq_name": summary["case"]["seq_name"],
            "frame_id": frame_id,
            "target_camera": target_camera,
            "view_profile": summary["case"]["view_profile"],
            "source_count": int(summary["case"]["source_count"]),
        },
        legacy_reference_metrics=summary.get("legacy_report_metrics", {}),
    )
    save_json(case_output_dir / "region_metrics.json", region_payload)
    write_region_markdown_report(case_output_dir / "region_metrics.md", region_payload)

    bg_far = region_payload["comparison"]["bg_far"]
    bg_bottom = region_payload["comparison"]["bg_bottom_band"]
    return {
        "case_name": case_dir.name,
        "source_case_dir": str(case_dir),
        "output_dir": str(case_output_dir),
        "seq_name": summary["case"]["seq_name"],
        "frame_id": frame_id,
        "target_camera": target_camera,
        "view_profile": summary["case"]["view_profile"],
        "bg_far_mae_winner": bg_far["mae_winner"],
        "bg_far_coverage_winner": bg_far["coverage_winner"],
        "bg_bottom_mae_winner": bg_bottom["mae_winner"],
        "bg_bottom_coverage_winner": bg_bottom["coverage_winner"],
        "region_metrics_json": str((case_output_dir / "region_metrics.json").resolve()),
        "region_metrics_md": str((case_output_dir / "region_metrics.md").resolve()),
    }


def main():
    args = parse_args()
    output_root = resolve_output_dir(args.output_dir)

    case_records = []
    for raw_case_dir in args.case_dir:
        case_dir = Path(raw_case_dir).resolve()
        case_output_dir = ensure_dir(output_root / case_dir.name)
        case_records.append(process_case(case_dir, case_output_dir, args))

    payload = {
        "output_root": str(output_root.resolve()),
        "case_count": len(case_records),
        "cases": case_records,
    }
    save_json(output_root / "summary.json", payload)

    lines = [
        "# Retroactive Geometry Region Diagnostics",
        "",
        f"- case_count: `{len(case_records)}`",
        "",
        "| Case | Profile | Target | bg_far MAE Winner | bg_bottom MAE Winner | Output |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in case_records:
        lines.append(
            "| `{case_name}` | `{view_profile}` | `{target_camera}` | `{bg_far_mae_winner}` | `{bg_bottom_mae_winner}` | `{output_dir}` |".format(
                **row
            )
        )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] Wrote {output_root / 'summary.md'}")


if __name__ == "__main__":
    main()
