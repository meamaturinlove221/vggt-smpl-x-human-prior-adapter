from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render per-input-view Open3D provenance sheets for a candidate. "
            "This is a diagnostic: it does not patch, train, or change thresholds."
        )
    )
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--headshoulder-npz", required=True)
    parser.add_argument("--headshoulder-scene", required=True)
    parser.add_argument("--fullbody-npz", required=True)
    parser.add_argument("--fullbody-scene", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--views", default="0,1,2,3,4,5")
    parser.add_argument("--point-sources", default="world_points,depth_unprojection")
    parser.add_argument("--max-points", type=int, default=80000)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_csv_ints(value: str) -> list[int]:
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_render(
    *,
    npz: Path,
    scene: Path,
    out: Path,
    point_source: str,
    roi: str,
    view: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sentinel = out / "open3d_summary.json"
    if sentinel.is_file() and not args.force:
        return load_render_summary(sentinel, exit_code=0, skipped=True)

    out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(TOOLS_DIR / "render_open3d_pointcloud.py"),
        "--predictions-npz",
        str(npz),
        "--scene-dir",
        str(scene),
        "--output-dir",
        str(out),
        "--point-source",
        point_source,
        "--roi",
        roi,
        "--roi-source",
        "2d",
        "--human-only",
        "--input-view-indices",
        str(view),
        "--camera-view-indices",
        str(view),
        "--max-points",
        str(int(args.max_points)),
        "--conf-percentile",
        "40",
        "--width",
        str(int(args.width)),
        "--height",
        str(int(args.height)),
        "--point-size",
        str(float(args.point_size)),
    ]
    log_path = out / "render.log"
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND: " + " ".join(command) + "\n\n")
        completed = subprocess.run(command, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        log.write(f"\nEXIT_CODE: {completed.returncode}\n")
    if sentinel.is_file():
        return load_render_summary(sentinel, exit_code=int(completed.returncode), skipped=False)
    return {
        "exit_code": int(completed.returncode),
        "summary_error": "missing open3d_summary.json",
        "output_dir": str(out),
    }


def load_render_summary(path: Path, *, exit_code: int, skipped: bool) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "exit_code": int(exit_code),
            "skipped_existing": bool(skipped),
            "summary_error": str(exc),
            "summary_path": str(path),
        }
    roi_summary = payload.get("roi_summary", {}) if isinstance(payload.get("roi_summary"), dict) else {}
    post = payload.get("postprocess_summary", {}) if isinstance(payload.get("postprocess_summary"), dict) else {}
    pre = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    return {
        "exit_code": int(exit_code),
        "skipped_existing": bool(skipped),
        "render_backend": payload.get("render_backend"),
        "points_before_conf": pre.get("valid_points_before_conf"),
        "points_after_conf": pre.get("points_after_conf"),
        "points_after_roi": roi_summary.get("points_after_roi"),
        "points_rendered": post.get("points_after_postprocess"),
        "screenshots": payload.get("screenshots", []),
        "output_dir": payload.get("output_dir"),
    }


def preferred_image(render_dir: Path, roi: str, view: int) -> Path | None:
    names = [
        f"camera_view_{view:02d}_crop.png",
        f"camera_view_{view:02d}.png",
        "face_close.png" if roi == "face" else "head_close.png" if roi == "head" else "front.png",
        "side.png",
    ]
    for name in names:
        path = render_dir / name
        if path.is_file():
            return path
    return None


def make_sheet(
    *,
    title: str,
    rows: list[dict[str, Any]],
    output_path: Path,
    thumb_size: tuple[int, int] = (220, 176),
) -> None:
    if not rows:
        return
    label_w = 260
    pad = 8
    width = label_w + len(rows[0]["cells"]) * (thumb_size[0] + pad) + pad
    header_h = 42
    row_h = thumb_size[1] + 44
    height = header_h + len(rows) * row_h + pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), title, fill=(0, 0, 0))
    y = header_h
    for row in rows:
        draw.text((pad, y + 8), str(row["label"]), fill=(0, 0, 0))
        x = label_w
        for cell in row["cells"]:
            image_path = cell.get("image_path")
            label = str(cell.get("label", ""))
            points = cell.get("points_rendered")
            if image_path and Path(image_path).is_file():
                image = Image.open(image_path).convert("RGB")
                image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                thumb = Image.new("RGB", thumb_size, "white")
                left = (thumb_size[0] - image.size[0]) // 2
                top = (thumb_size[1] - image.size[1]) // 2
                thumb.paste(image, (left, top))
            else:
                thumb = Image.new("RGB", thumb_size, (245, 245, 245))
                ImageDraw.Draw(thumb).text((12, 72), "missing", fill=(120, 0, 0))
            canvas.paste(thumb, (x, y))
            caption = label if points is None else f"{label} pts={points}"
            draw.text((x, y + thumb_size[1] + 4), caption[:42], fill=(0, 0, 0))
            x += thumb_size[0] + pad
        y += row_h
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_report(summary: dict[str, Any], output_dir: Path) -> None:
    lines = [
        f"# Per-View Open3D Provenance: {summary['candidate_name']}",
        "",
        "## Purpose",
        "",
        "This is a local diagnostic only. It isolates one input view at a time before Open3D rendering, using the same point sources and 2D human/ROI masks. It does not train, patch, tune thresholds, introduce a teacher, or claim mentor-final.",
        "",
        "## Render Scope",
        "",
        f"- views: `{','.join(str(v) for v in summary['views'])}`",
        f"- point sources: `{','.join(summary['point_sources'])}`",
        "- headshoulder ROIs: `head, face`",
        "- fullbody ROIs: `full, hands`",
        "- gate: `p40 confidence percentile`",
        "",
        "## Output Sheets",
        "",
    ]
    for key, path in summary.get("sheets", {}).items():
        lines.append(f"- {key}: `{path}`")
    lines.extend(
        [
            "",
            "## Automatic Counts",
            "",
            "| Dataset | ROI | Source | View | Rendered Points | Backend |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in summary["renders"]:
        lines.append(
            "| {dataset} | {roi} | {source} | {view} | {points} | {backend} |".format(
                dataset=row["dataset"],
                roi=row["roi"],
                source=row["point_source"],
                view=row["view"],
                points=row.get("points_rendered"),
                backend=row.get("render_backend"),
            )
        )
    lines.extend(
        [
            "",
            "## Review Policy",
            "",
            "A candidate still fails if per-view or fused Open3D shows shell-like head/face, broken hairline, slab/ghost body volume, missing/amputated hands, or scattered hand sheets. This diagnostic only decides where the failure is introduced.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    views = parse_csv_ints(args.views)
    point_sources = parse_csv_strings(args.point_sources)

    datasets = [
        {
            "name": "headshoulder",
            "npz": Path(args.headshoulder_npz).resolve(),
            "scene": Path(args.headshoulder_scene).resolve(),
            "rois": ["head", "face"],
        },
        {
            "name": "fullbody",
            "npz": Path(args.fullbody_npz).resolve(),
            "scene": Path(args.fullbody_scene).resolve(),
            "rois": ["full", "hands"],
        },
    ]

    render_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for roi in dataset["rois"]:
            for point_source in point_sources:
                for view in views:
                    render_dir = output_dir / "renders" / dataset["name"] / roi / point_source / f"view_{view:02d}"
                    result = run_render(
                        npz=dataset["npz"],
                        scene=dataset["scene"],
                        out=render_dir,
                        point_source=point_source,
                        roi=roi,
                        view=view,
                        args=args,
                    )
                    result.update(
                        {
                            "dataset": dataset["name"],
                            "roi": roi,
                            "point_source": point_source,
                            "view": int(view),
                            "render_dir": str(render_dir),
                            "image_path": str(preferred_image(render_dir, roi=roi, view=view) or ""),
                        }
                    )
                    render_rows.append(result)

    sheets: dict[str, str] = {}
    for dataset in datasets:
        for roi in dataset["rois"]:
            rows: list[dict[str, Any]] = []
            for view in views:
                cells = []
                for point_source in point_sources:
                    row = next(
                        item
                        for item in render_rows
                        if item["dataset"] == dataset["name"]
                        and item["roi"] == roi
                        and item["point_source"] == point_source
                        and item["view"] == view
                    )
                    cells.append(
                        {
                            "label": point_source,
                            "image_path": row.get("image_path"),
                            "points_rendered": row.get("points_rendered"),
                        }
                    )
                rows.append({"label": f"view {view:02d}", "cells": cells})
            sheet_path = output_dir / f"sheet_{dataset['name']}_{roi}.png"
            make_sheet(
                title=f"{args.candidate_name} / {dataset['name']} / {roi} / per-view p40",
                rows=rows,
                output_path=sheet_path,
            )
            sheets[f"{dataset['name']}_{roi}"] = str(sheet_path)

    summary = {
        "candidate_name": args.candidate_name,
        "views": views,
        "point_sources": point_sources,
        "renders": render_rows,
        "sheets": sheets,
        "diagnostic_only": True,
        "pass_claim": False,
    }
    (output_dir / "per_view_provenance_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(summary, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "sheets": sheets}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
