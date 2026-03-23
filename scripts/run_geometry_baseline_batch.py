import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare_geometry_branches.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Run geometry-branch baseline across multiple example scenes.")
    parser.add_argument("--examples_root", type=str, default="examples", help="Root folder that contains scene directories.")
    parser.add_argument(
        "--scenes",
        type=str,
        default="auto",
        help='Comma-separated scene names, or "auto" to discover all multi-image scenes.',
    )
    parser.add_argument("--checkpoint", type=str, default="", help="Optional local checkpoint path.")
    parser.add_argument("--max_images", type=int, default=8, help="Per-scene maximum number of images.")
    parser.add_argument("--target_frames", type=str, default="0,4", help="Per-scene target frames passed to the single-scene runner.")
    parser.add_argument("--preprocess_mode", type=str, default="crop", choices=["crop", "pad"])
    parser.add_argument("--conf_percentile", type=float, default=25.0)
    parser.add_argument(
        "--output_root",
        type=str,
        default="",
        help="Root output directory. Defaults to output/geometry_baseline_batch/<timestamp>.",
    )
    parser.add_argument("--keep_predictions", action="store_true", help="Keep per-scene predictions.npz files.")
    return parser.parse_args()


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_output_root(output_root):
    if output_root:
        return ensure_dir(Path(output_root))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(REPO_ROOT / "output" / "geometry_baseline_batch" / timestamp)


def discover_scenes(examples_root, explicit_scenes):
    examples_root = Path(examples_root)
    if explicit_scenes.strip().lower() != "auto":
        names = [name.strip() for name in explicit_scenes.split(",") if name.strip()]
        if not names:
            raise ValueError("No valid scene names were provided.")
        return names

    names = []
    for scene_dir in sorted(examples_root.iterdir()):
        image_dir = scene_dir / "images"
        if not scene_dir.is_dir() or not image_dir.is_dir():
            continue
        image_count = len([path for path in image_dir.iterdir() if path.is_file()])
        if image_count >= 2:
            names.append(scene_dir.name)
    if not names:
        raise ValueError(f"No multi-image scenes found under {examples_root}")
    return names


def run_scene(scene_name, args, output_root):
    scene_output_dir = ensure_dir(output_root / scene_name)
    image_folder = Path(args.examples_root) / scene_name / "images"

    cmd = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--image_folder",
        str(image_folder),
        "--output_dir",
        str(scene_output_dir),
        "--max_images",
        str(args.max_images),
        "--target_frames",
        str(args.target_frames),
        "--preprocess_mode",
        str(args.preprocess_mode),
        "--conf_percentile",
        str(args.conf_percentile),
    ]
    if args.checkpoint:
        cmd += ["--checkpoint", str(args.checkpoint)]
    if not args.keep_predictions:
        cmd += ["--skip_save_predictions"]

    print(f"[batch] scene={scene_name} cmd={' '.join(cmd)}", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    log_path = scene_output_dir / "batch_run.log"
    with open(log_path, "w", encoding="utf-8") as handle:
        if proc.stdout:
            handle.write(proc.stdout)
            if not proc.stdout.endswith("\n"):
                handle.write("\n")
        if proc.stderr:
            handle.write(proc.stderr)
            if not proc.stderr.endswith("\n"):
                handle.write("\n")

    result = {
        "scene": scene_name,
        "scene_output_dir": str(scene_output_dir),
        "log_path": str(log_path),
        "returncode": int(proc.returncode),
        "status": "ok" if proc.returncode == 0 else "failed",
        "error": "",
        "summary": None,
    }

    if proc.returncode != 0:
        result["error"] = (proc.stderr or proc.stdout or "").strip().splitlines()[-1] if (proc.stderr or proc.stdout) else ""
        return result

    summary_path = scene_output_dir / "summary.json"
    if not summary_path.exists():
        result["status"] = "failed"
        result["error"] = "summary.json missing"
        return result

    with open(summary_path, "r", encoding="utf-8") as handle:
        result["summary"] = json.load(handle)
    return result


def safe_mean(values):
    valid = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not valid:
        return float("nan")
    return float(sum(valid) / len(valid))


def summarize_scene(result):
    summary = result["summary"]
    renders = summary["renders"]
    point_mae = safe_mean(render["point_map"]["mae_to_target"] for render in renders)
    depth_mae = safe_mean(render["depth_unproject"]["mae_to_target"] for render in renders)
    point_cov = safe_mean(render["point_map"]["coverage_ratio"] for render in renders)
    depth_cov = safe_mean(render["depth_unproject"]["coverage_ratio"] for render in renders)
    point_rendered = safe_mean(render["point_map"]["rendered_points"] for render in renders)
    depth_rendered = safe_mean(render["depth_unproject"]["rendered_points"] for render in renders)

    mae_winner = "tie"
    coverage_winner = "tie"
    decision = "tie"
    if not math.isnan(point_mae) and not math.isnan(depth_mae):
        if depth_mae < point_mae:
            mae_winner = "depth_unproject"
        elif point_mae < depth_mae:
            mae_winner = "point_map"

        if depth_cov > point_cov:
            coverage_winner = "depth_unproject"
        elif point_cov > depth_cov:
            coverage_winner = "point_map"

        if (depth_mae < point_mae) and (depth_cov >= point_cov):
            decision = "depth_unproject"
        elif (point_mae < depth_mae) and (point_cov >= depth_cov):
            decision = "point_map"

    return {
        "scene": result["scene"],
        "status": result["status"],
        "decision": decision,
        "mae_winner": mae_winner,
        "coverage_winner": coverage_winner,
        "point_map_mean_mae": point_mae,
        "depth_unproject_mean_mae": depth_mae,
        "point_map_mean_coverage": point_cov,
        "depth_unproject_mean_coverage": depth_cov,
        "point_map_mean_rendered_points": point_rendered,
        "depth_unproject_mean_rendered_points": depth_rendered,
        "summary_md": str(Path(result["scene_output_dir"]) / "summary.md"),
        "scene_output_dir": result["scene_output_dir"],
    }


def write_csv(path, rows):
    fieldnames = [
        "scene",
        "status",
        "decision",
        "mae_winner",
        "coverage_winner",
        "point_map_mean_mae",
        "depth_unproject_mean_mae",
        "point_map_mean_coverage",
        "depth_unproject_mean_coverage",
        "point_map_mean_rendered_points",
        "depth_unproject_mean_rendered_points",
        "summary_md",
        "scene_output_dir",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path, args, rows, failures):
    depth_wins = [row for row in rows if row["decision"] == "depth_unproject"]
    point_wins = [row for row in rows if row["decision"] == "point_map"]
    ties = [row for row in rows if row["decision"] == "tie"]
    depth_mae_wins = [row for row in rows if row["mae_winner"] == "depth_unproject"]
    point_mae_wins = [row for row in rows if row["mae_winner"] == "point_map"]
    depth_cov_wins = [row for row in rows if row["coverage_winner"] == "depth_unproject"]
    point_cov_wins = [row for row in rows if row["coverage_winner"] == "point_map"]

    lines = [
        "# Geometry Baseline Batch",
        "",
        f"- scenes_requested: `{args.scenes}`",
        f"- max_images: `{args.max_images}`",
        f"- target_frames: `{args.target_frames}`",
        f"- preprocess_mode: `{args.preprocess_mode}`",
        f"- checkpoint: `{args.checkpoint or 'official default'}`",
        "",
        f"- depth_unproject_wins: `{len(depth_wins)}`",
        f"- point_map_wins: `{len(point_wins)}`",
        f"- ties_or_inconclusive: `{len(ties)}`",
        f"- depth_unproject_lower_mae: `{len(depth_mae_wins)}`",
        f"- point_map_lower_mae: `{len(point_mae_wins)}`",
        f"- depth_unproject_higher_coverage: `{len(depth_cov_wins)}`",
        f"- point_map_higher_coverage: `{len(point_cov_wins)}`",
        "",
        "## Scene Summary",
        "",
        "| Scene | Decision | MAE Winner | Cov Winner | Point MAE | Depth MAE | Point Cov | Depth Cov | Summary |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in rows:
        lines.append(
            "| {scene} | {decision} | {mae_winner} | {coverage_winner} | {pmae:.4f} | {dmae:.4f} | {pcov:.4f} | {dcov:.4f} | `{summary}` |".format(
                scene=row["scene"],
                decision=row["decision"],
                mae_winner=row["mae_winner"],
                coverage_winner=row["coverage_winner"],
                pmae=row["point_map_mean_mae"],
                dmae=row["depth_unproject_mean_mae"],
                pcov=row["point_map_mean_coverage"],
                dcov=row["depth_unproject_mean_coverage"],
                summary=row["summary_md"],
            )
        )

    if failures:
        lines.extend(["", "## Failures", ""])
        for item in failures:
            lines.append(f"- `{item['scene']}`: {item['error'] or 'unknown error'}")

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- If `depth_unproject` wins on most scenes, keep the next step on the `depth + camera` geometry chain.",
            "- If a scene is inconclusive, inspect its per-scene `renders/` PNGs before changing training.",
            "- Do not reintroduce the legacy ghost stack based on image-only heuristics before these geometry baselines are understood.",
        ]
    )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    output_root = resolve_output_root(args.output_root)
    scenes = discover_scenes(args.examples_root, args.scenes)

    print(f"[batch] output_root={output_root}", flush=True)
    print(f"[batch] scenes={scenes}", flush=True)

    scene_results = []
    for scene_name in scenes:
        scene_results.append(run_scene(scene_name, args, output_root))

    failures = [result for result in scene_results if result["status"] != "ok"]
    rows = [summarize_scene(result) for result in scene_results if result["status"] == "ok"]

    batch_summary = {
        "args": vars(args),
        "output_root": str(output_root),
        "rows": rows,
        "failures": failures,
    }

    with open(output_root / "batch_summary.json", "w", encoding="utf-8") as handle:
        json.dump(batch_summary, handle, indent=2, ensure_ascii=False)

    write_csv(output_root / "batch_summary.csv", rows)
    write_markdown(output_root / "batch_summary.md", args, rows, failures)

    print(f"[batch] done: {output_root / 'batch_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
