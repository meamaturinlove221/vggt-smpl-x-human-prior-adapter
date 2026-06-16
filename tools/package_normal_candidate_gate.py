from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from diagnose_face_roi_failure import apply_3d_roi  # noqa: E402
from normal_line_multiview_eval import build_roi_masks, load_scene_view  # noqa: E402
from render_open3d_pointcloud import unproject_depth_map_to_point_map_numpy  # noqa: E402
from vggt.utils.normal_refiner import face_box_from_mask, head_box_from_mask  # noqa: E402


POINT_SOURCES = ("world_points", "depth_unprojection")
GATES = ("p40", "fixed")
HEADSHOULDER_ROIS = ("full", "head", "face")
FULLBODY_ROIS = ("full", "head", "face", "hands")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package a normal-line candidate under the mentor gate: same-protocol "
            "head/face metrics, full-body/hands bottom-line audit, Open3D renders, "
            "normal consistency, shape metrics, and a truthful pass/fail report."
        )
    )
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--candidate-headshoulder-npz", required=True)
    parser.add_argument("--headshoulder-scene", required=True)
    parser.add_argument("--reference-name", default="signfix_ckpt4")
    parser.add_argument("--reference-headshoulder-npz", required=True)
    parser.add_argument("--candidate-fullbody-npz", default=None)
    parser.add_argument("--fullbody-scene", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--fixed-threshold", type=float, default=38.5067)
    parser.add_argument("--face-margin", type=int, default=500)
    parser.add_argument("--min-fullbody-largest-component", type=float, default=0.86)
    parser.add_argument("--min-fullbody-vertical-bin-ratio", type=float, default=0.018)
    parser.add_argument("--min-hand-views", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=400000)
    parser.add_argument("--render-width", type=int, default=1200)
    parser.add_argument("--render-height", type=int, default=1000)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--camera-view-indices", default="0,2,3")
    parser.add_argument("--skip-open3d", action="store_true")
    parser.add_argument("--skip-normal-consistency", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run commands even when summaries already exist.")
    return parser.parse_args()


def load_mask_stack(scene_dir: Path, view_count: int, hw: tuple[int, int]) -> np.ndarray:
    masks: list[np.ndarray] = []
    for view_idx in range(view_count):
        masks.append(load_scene_view(scene_dir, view_idx, hw).mask.astype(bool))
    return np.stack(masks, axis=0)


def resolve_source(data: np.lib.npyio.NpzFile, source: str) -> tuple[np.ndarray, np.ndarray]:
    if source == "world_points":
        return np.asarray(data["world_points"], dtype=np.float32), np.asarray(data["world_points_conf"], dtype=np.float32)
    if source == "depth_unprojection":
        return (
            unproject_depth_map_to_point_map_numpy(
                np.asarray(data["depth"], dtype=np.float32),
                np.asarray(data["extrinsic"], dtype=np.float32),
                np.asarray(data["intrinsic"], dtype=np.float32),
            ),
            np.asarray(data["depth_conf"], dtype=np.float32),
        )
    raise ValueError(f"unknown point source: {source}")


def threshold_for(conf: np.ndarray, support: np.ndarray, gate: str, percentile: float, fixed_threshold: float) -> float:
    if gate == "fixed":
        return float(fixed_threshold)
    values = conf[support & np.isfinite(conf) & (conf > 0.0)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, float(percentile)))


def filter_points(
    points_map: np.ndarray,
    conf: np.ndarray,
    masks: np.ndarray,
    gate: str,
    percentile: float,
    fixed_threshold: float,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    finite = np.isfinite(points_map).all(axis=-1)
    support = masks & finite & np.isfinite(conf) & (conf > 0.0)
    threshold = threshold_for(conf, support, gate, percentile, fixed_threshold)
    keep = support & (conf >= threshold)
    flat_points = points_map.reshape(-1, 3)
    flat_keep = keep.reshape(-1)
    return flat_points[flat_keep], {
        "gate": gate,
        "threshold": threshold,
        "threshold_source": "fixed" if gate == "fixed" else f"p{percentile:g}",
        "valid_points_before_conf": int(support.sum()),
        "points_after_conf": int(flat_keep.sum()),
    }, keep


def box_mask(box: tuple[int, int, int, int] | None, shape: tuple[int, int], support: np.ndarray) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    if box is None:
        return out
    x0, y0, x1, y1 = box
    x0 = max(0, min(shape[1], int(x0)))
    x1 = max(0, min(shape[1], int(x1)))
    y0 = max(0, min(shape[0], int(y0)))
    y1 = max(0, min(shape[0], int(y1)))
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = support[y0:y1, x0:x1]
    return out


def hairline_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    head_box = head_box_from_mask(mask)
    face_box = face_box_from_mask(mask)
    if head_box is None:
        return np.zeros_like(mask, dtype=bool)
    x0, y0, x1, y1 = head_box
    head_h = max(1, y1 - y0)
    top_band = (x0, y0, x1, min(y1, y0 + int(round(0.46 * head_h))))
    out = box_mask(top_band, mask.shape, mask)
    if face_box is not None:
        fx0, fy0, fx1, fy1 = face_box
        out[max(0, fy0) : min(mask.shape[0], fy1), max(0, fx0) : min(mask.shape[1], fx1)] = False
    return out & mask


def summarize_hairline_kept(keep: np.ndarray, masks: np.ndarray) -> dict[str, Any]:
    pixels = 0
    kept = 0
    per_view: dict[str, Any] = {}
    for view_idx in range(masks.shape[0]):
        roi = hairline_mask(masks[view_idx])
        roi_pixels = int(roi.sum())
        roi_kept = int((keep[view_idx] & roi).sum())
        pixels += roi_pixels
        kept += roi_kept
        per_view[str(view_idx)] = {
            "pixels": roi_pixels,
            "kept_pixels": roi_kept,
            "kept_ratio": float(roi_kept / max(roi_pixels, 1)) if roi_pixels else 0.0,
        }
    return {
        "pixels": pixels,
        "kept_pixels": kept,
        "kept_ratio": float(kept / max(pixels, 1)) if pixels else 0.0,
        "per_view": per_view,
    }


def roi_summary_for_entry(
    *,
    name: str,
    npz_path: Path,
    scene_dir: Path,
    percentile: float,
    fixed_threshold: float,
) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=False)
    view_count = int(np.asarray(data["world_points"]).shape[0])
    hw = tuple(int(v) for v in np.asarray(data["world_points"]).shape[1:3])
    masks = load_mask_stack(scene_dir, view_count, hw)
    summary: dict[str, Any] = {
        "name": name,
        "predictions_npz": str(npz_path.resolve()),
        "scene_dir": str(scene_dir.resolve()),
        "conf_percentile": float(percentile),
        "fixed_threshold": float(fixed_threshold),
        "sources": {},
    }
    for source in POINT_SOURCES:
        points_map, conf = resolve_source(data, source)
        source_rows: dict[str, Any] = {}
        for gate in GATES:
            points, filter_summary, keep = filter_points(points_map, conf, masks, gate, percentile, fixed_threshold)
            roi_rows = {roi: apply_3d_roi(points, roi) for roi in HEADSHOULDER_ROIS}
            source_rows[gate] = {
                "filter": filter_summary,
                "roi_3d": roi_rows,
                "hairline_2d": summarize_hairline_kept(keep, masks),
            }
        summary["sources"][source] = source_rows
    return summary


def get_count(summary: dict[str, Any], source: str, gate: str, roi: str) -> int:
    return int(summary["sources"][source][gate]["roi_3d"][roi]["points_after_roi"])


def evaluate_numeric_gate(
    ref: dict[str, Any],
    cand: dict[str, Any],
    face_margin: int,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for source in POINT_SOURCES:
        ref_full = get_count(ref, source, "p40", "full")
        ref_head = get_count(ref, source, "p40", "head")
        ref_face = get_count(ref, source, "p40", "face")
        cand_full = get_count(cand, source, "p40", "full")
        cand_head = get_count(cand, source, "p40", "head")
        cand_face = get_count(cand, source, "p40", "face")
        cand_fixed_face = get_count(cand, source, "fixed", "face")
        checks[source] = {
            "p40_full_no_regression": bool(cand_full >= ref_full),
            "p40_head_no_regression": bool(cand_head >= ref_head),
            "p40_face_meaningful_margin": bool(cand_face >= ref_face + int(face_margin)),
            "fixed_face_no_collapse": bool(cand_fixed_face >= ref_face),
            "ref_p40": {"full": ref_full, "head": ref_head, "face": ref_face},
            "candidate_p40": {"full": cand_full, "head": cand_head, "face": cand_face},
            "candidate_fixed_face": cand_fixed_face,
            "face_delta": int(cand_face - ref_face),
        }
    checks["pass"] = bool(
        all(
            row["p40_full_no_regression"]
            and row["p40_head_no_regression"]
            and row["p40_face_meaningful_margin"]
            and row["fixed_face_no_collapse"]
            for row in checks.values()
            if isinstance(row, dict) and "candidate_p40" in row
        )
    )
    return checks


def run_command(command: list[str], cwd: Path, log_path: Path, force: bool, sentinel: Path | None = None) -> int:
    if sentinel is not None and sentinel.exists() and not force:
        return 0
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND: " + " ".join(command) + "\n\n")
        completed = subprocess.run(command, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        log.write(f"\nEXIT_CODE: {completed.returncode}\n")
    return int(completed.returncode)


def render_one(
    *,
    npz_path: Path,
    scene_dir: Path,
    output_dir: Path,
    source: str,
    roi: str,
    gate: str,
    fixed_threshold: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out = output_dir / f"{roi}_{source}_{gate}"
    command = [
        sys.executable,
        str(TOOLS_DIR / "render_open3d_pointcloud.py"),
        "--predictions-npz",
        str(npz_path),
        "--scene-dir",
        str(scene_dir),
        "--output-dir",
        str(out),
        "--point-source",
        source,
        "--roi",
        roi,
        "--roi-source",
        "2d" if roi == "hands" else "3d",
        "--human-only",
        "--max-points",
        str(int(args.max_points)),
        "--conf-percentile",
        str(float(args.conf_percentile)),
        "--width",
        str(int(args.render_width)),
        "--height",
        str(int(args.render_height)),
        "--point-size",
        str(float(args.point_size)),
        "--camera-view-indices",
        str(args.camera_view_indices),
    ]
    if gate == "fixed":
        command.extend(["--conf-threshold", str(float(fixed_threshold))])
    code = run_command(
        command,
        REPO_ROOT,
        output_dir / "logs" / f"render_{roi}_{source}_{gate}.log",
        bool(args.force),
        sentinel=out / "open3d_summary.json",
    )
    return {"roi": roi, "source": source, "gate": gate, "output_dir": str(out), "exit_code": code}


def audit_fullbody_one(
    *,
    npz_path: Path,
    scene_dir: Path,
    output_dir: Path,
    source: str,
    gate: str,
    fixed_threshold: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out = output_dir / f"{source}_{gate}"
    command = [
        sys.executable,
        str(TOOLS_DIR / "audit_fullbody_hand_integrity.py"),
        "--predictions-npz",
        str(npz_path),
        "--scene-dir",
        str(scene_dir),
        "--output-dir",
        str(out),
        "--point-source",
        source,
        "--conf-percentile",
        str(float(args.conf_percentile)),
        "--min-largest-component-ratio",
        str(float(args.min_fullbody_largest_component)),
        "--min-vertical-bin-ratio",
        str(float(args.min_fullbody_vertical_bin_ratio)),
        "--min-hand-components",
        str(int(args.min_hand_views)),
    ]
    if gate == "fixed":
        command.extend(["--conf-threshold", str(float(fixed_threshold))])
    code = run_command(
        command,
        REPO_ROOT,
        output_dir / "logs" / f"audit_fullbody_{source}_{gate}.log",
        bool(args.force),
        sentinel=out / "fullbody_hand_integrity_summary.json",
    )
    payload: dict[str, Any] = {"source": source, "gate": gate, "output_dir": str(out), "exit_code": code}
    summary_path = out / "fullbody_hand_integrity_summary.json"
    if summary_path.exists():
        try:
            payload["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            payload["summary_error"] = str(exc)
    return payload


def run_auxiliary_metrics(args: argparse.Namespace, out_root: Path) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    reference_entry = (
        f"{args.reference_name}:{Path(args.reference_headshoulder_npz).resolve()}:"
        f"{Path(args.headshoulder_scene).resolve()}"
    )
    candidate_entry = (
        f"{args.candidate_name}:{Path(args.candidate_headshoulder_npz).resolve()}:"
        f"{Path(args.headshoulder_scene).resolve()}"
    )
    shape_dir = out_root / "metrics" / "shape"
    code = run_command(
        [
            sys.executable,
            str(TOOLS_DIR / "measure_face_shape_metrics.py"),
            "--entry",
            reference_entry,
            "--entry",
            candidate_entry,
            "--output-dir",
            str(shape_dir),
            "--conf-percentile",
            str(float(args.conf_percentile)),
            "--fixed-threshold",
            str(float(args.fixed_threshold)),
        ],
        REPO_ROOT,
        out_root / "logs" / "measure_face_shape_metrics.log",
        bool(args.force),
        sentinel=shape_dir / "face_shape_metrics.json",
    )
    outputs["shape_metrics"] = {"output_dir": str(shape_dir), "exit_code": code}

    diag_dir = out_root / "metrics" / "face_roi_failure"
    code = run_command(
        [
            sys.executable,
            str(TOOLS_DIR / "diagnose_face_roi_failure.py"),
            "--baseline",
            reference_entry,
            "--candidate",
            candidate_entry,
            "--output-dir",
            str(diag_dir),
            "--conf-percentile",
            str(float(args.conf_percentile)),
            "--fixed-threshold",
            str(float(args.fixed_threshold)),
        ],
        REPO_ROOT,
        out_root / "logs" / "diagnose_face_roi_failure.log",
        bool(args.force),
        sentinel=diag_dir / "face_roi_failure_diagnosis.json",
    )
    outputs["face_roi_failure"] = {"output_dir": str(diag_dir), "exit_code": code}

    if not args.skip_normal_consistency:
        normal_dir = out_root / "metrics" / "normal_consistency"
        code = run_command(
            [
                sys.executable,
                str(TOOLS_DIR / "normal_line_multiview_eval.py"),
                "--entry",
                reference_entry,
                "--entry",
                candidate_entry,
                "--output-dir",
                str(normal_dir),
            ],
            REPO_ROOT,
            out_root / "logs" / "normal_line_multiview_eval.log",
            bool(args.force),
            sentinel=normal_dir / "multiview_normal_consistency_summary.json",
        )
        outputs["normal_consistency"] = {"output_dir": str(normal_dir), "exit_code": code}
    return outputs


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def summarize_fullbody_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    pass_all = True
    for row in audits:
        summary = row.get("summary")
        if not isinstance(summary, dict):
            pass_all = False
            details.append({**row, "pass": False, "reason": "missing summary"})
            continue
        full_gate = summary.get("full_body_gate", {})
        hand_gate = summary.get("hand_gate", {})
        row_pass = bool(full_gate.get("pass")) and bool(hand_gate.get("pass"))
        pass_all = pass_all and row_pass
        details.append(
            {
                "source": row["source"],
                "gate": row["gate"],
                "pass": row_pass,
                "points_after_conf": summary.get("points_after_conf"),
                "full_body_gate": full_gate,
                "hand_gate": hand_gate,
                "cluster": summary.get("cluster"),
                "vertical_3d": summary.get("vertical_3d"),
                "output_dir": row.get("output_dir"),
            }
        )
    return {"pass": bool(pass_all and bool(audits)), "details": details}


def add_label(image: Image.Image, label: str) -> Image.Image:
    image = image.convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 34), "white")
    canvas.paste(image, (0, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 9), label[:120], fill=(0, 0, 0))
    return canvas


def make_contact_sheet(out_root: Path) -> str | None:
    picks = [
        ("headshoulder face world p40 side", out_root / "open3d_headshoulder" / "face_world_points_p40" / "side.png"),
        ("headshoulder face depth p40 side", out_root / "open3d_headshoulder" / "face_depth_unprojection_p40" / "side.png"),
        ("headshoulder head world p40 close", out_root / "open3d_headshoulder" / "head_world_points_p40" / "head_close.png"),
        ("headshoulder full world p40 front", out_root / "open3d_headshoulder" / "full_world_points_p40" / "front.png"),
        ("fullbody full world p40 front", out_root / "open3d_fullbody" / "full_world_points_p40" / "front.png"),
        ("fullbody full depth p40 front", out_root / "open3d_fullbody" / "full_depth_unprojection_p40" / "front.png"),
        ("fullbody hands world p40 side", out_root / "open3d_fullbody" / "hands_world_points_p40" / "side.png"),
        ("fullbody hands depth p40 side", out_root / "open3d_fullbody" / "hands_depth_unprojection_p40" / "side.png"),
    ]
    cells: list[Image.Image] = []
    for label, path in picks:
        if path.exists():
            img = Image.open(path).convert("RGB")
            img.thumbnail((360, 300), Image.Resampling.LANCZOS)
            cells.append(add_label(img, label))
    if not cells:
        return None
    cell_w = max(cell.width for cell in cells)
    cell_h = max(cell.height for cell in cells)
    cols = 2
    rows = int(np.ceil(len(cells) / cols))
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for idx, cell in enumerate(cells):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        sheet.paste(cell, (x, y))
    path = out_root / "candidate_gate_visual_sheet.png"
    sheet.save(path)
    return str(path)


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    numeric_gate: dict[str, Any],
    fullbody_gate: dict[str, Any] | None,
    aux_outputs: dict[str, Any],
    render_outputs: list[dict[str, Any]],
    contact_sheet: str | None,
) -> None:
    failed: list[str] = []
    if not numeric_gate.get("pass"):
        failed.append("same-protocol 6v headshoulder numeric gate")
    if fullbody_gate is None:
        failed.append("full-body/hands gate not run")
    elif not fullbody_gate.get("pass"):
        failed.append("full-body/hands bottom-line gate")
    bad_commands = [row for row in render_outputs if int(row.get("exit_code", 0)) != 0]
    if bad_commands:
        failed.append(f"{len(bad_commands)} Open3D render command(s)")
    aux_bad = [name for name, row in aux_outputs.items() if int(row.get("exit_code", 0)) != 0]
    if aux_bad:
        failed.append("auxiliary metrics: " + ", ".join(aux_bad))

    lines = [
        f"# Candidate gate report: {args.candidate_name}",
        "",
        "## Pass/Fail",
        "",
        "**Status: FAIL / not mentor-final.**" if failed else "**Status: NUMERIC PACKAGE PASS, VISUAL REVIEW STILL REQUIRED.**",
        "",
        "This report is a local gate package. It cannot by itself claim mentor-final; Open3D visual review must still verify modeled face/head/hairline geometry and no full-body/hand regression.",
        "",
        "## Numeric Gate",
        "",
        f"- Reference: `{args.reference_name}`",
        f"- Face margin required: `{int(args.face_margin)}` points over reference for both world_points and depth_unprojection.",
        "",
        "| Source | Ref face p40 | Cand face p40 | Delta | Cand fixed face | Full ok | Head ok | Face margin ok | Fixed ok |",
        "|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for source in POINT_SOURCES:
        row = numeric_gate[source]
        lines.append(
            "| "
            + " | ".join(
                [
                    source,
                    str(row["ref_p40"]["face"]),
                    str(row["candidate_p40"]["face"]),
                    str(row["face_delta"]),
                    str(row["candidate_fixed_face"]),
                    str(row["p40_full_no_regression"]),
                    str(row["p40_head_no_regression"]),
                    str(row["p40_face_meaningful_margin"]),
                    str(row["fixed_face_no_collapse"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Full-Body / Hands", ""])
    if fullbody_gate is None:
        lines.append("- Not run: no `--candidate-fullbody-npz` and `--fullbody-scene` were provided.")
    else:
        lines.append(f"- Bottom-line pass: `{fullbody_gate.get('pass')}`")
        lines.append("")
        lines.append("| Source | Gate | Points | Full pass | Hand pass | Hand views passing | Output |")
        lines.append("|---|---|---:|---|---|---:|---|")
        for row in fullbody_gate.get("details", []):
            hand = row.get("hand_gate", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("source")),
                        str(row.get("gate")),
                        str(row.get("points_after_conf")),
                        str(row.get("full_body_gate", {}).get("pass")),
                        str(hand.get("pass")),
                        str(hand.get("views_passing_hand_kept_ratio")),
                        f"`{row.get('output_dir')}`",
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Visual / Metric Artifacts", ""])
    if contact_sheet:
        lines.append(f"- Contact sheet: `{contact_sheet}`")
    for key, value in aux_outputs.items():
        lines.append(f"- {key}: `{value.get('output_dir')}` exit={value.get('exit_code')}")
    lines.append("- Open3D renders:")
    for row in render_outputs:
        lines.append(
            f"  - `{row['roi']}` / `{row['source']}` / `{row['gate']}`: `{row['output_dir']}` exit={row['exit_code']}"
        )
    lines.extend(["", "## Failure Reasons", ""])
    if failed:
        for item in failed:
            lines.append(f"- {item}")
    else:
        lines.append("- No automatic numeric/package failure was found; human visual gate is still mandatory.")
    lines.extend(
        [
            "",
            "## Truthful Next Action",
            "",
            "If the Open3D face/head views remain a shell or the hands are fragmented/amputated, this candidate remains negative even when point counts are high. Do not upload to cloud unless this local package and visual review both pass.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "metrics").mkdir(exist_ok=True)
    (out_root / "logs").mkdir(exist_ok=True)

    reference_npz = Path(args.reference_headshoulder_npz).resolve()
    candidate_npz = Path(args.candidate_headshoulder_npz).resolve()
    head_scene = Path(args.headshoulder_scene).resolve()
    ref_summary = roi_summary_for_entry(
        name=args.reference_name,
        npz_path=reference_npz,
        scene_dir=head_scene,
        percentile=float(args.conf_percentile),
        fixed_threshold=float(args.fixed_threshold),
    )
    cand_summary = roi_summary_for_entry(
        name=args.candidate_name,
        npz_path=candidate_npz,
        scene_dir=head_scene,
        percentile=float(args.conf_percentile),
        fixed_threshold=float(args.fixed_threshold),
    )
    roi_payload = {"reference": ref_summary, "candidate": cand_summary}
    roi_path = out_root / "metrics" / "roi_summary_headshoulder.json"
    roi_path.write_text(json.dumps(roi_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    numeric_gate = evaluate_numeric_gate(ref_summary, cand_summary, int(args.face_margin))
    numeric_path = out_root / "metrics" / "numeric_gate.json"
    numeric_path.write_text(json.dumps(numeric_gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    aux_outputs = run_auxiliary_metrics(args, out_root)

    render_outputs: list[dict[str, Any]] = []
    if not args.skip_open3d:
        head_out = out_root / "open3d_headshoulder"
        for roi in HEADSHOULDER_ROIS:
            for source in POINT_SOURCES:
                for gate in GATES:
                    render_outputs.append(
                        render_one(
                            npz_path=candidate_npz,
                            scene_dir=head_scene,
                            output_dir=head_out,
                            source=source,
                            roi=roi,
                            gate=gate,
                            fixed_threshold=float(args.fixed_threshold),
                            args=args,
                        )
                    )

    fullbody_gate: dict[str, Any] | None = None
    if args.candidate_fullbody_npz and args.fullbody_scene:
        full_npz = Path(args.candidate_fullbody_npz).resolve()
        full_scene = Path(args.fullbody_scene).resolve()
        full_audits: list[dict[str, Any]] = []
        for source in POINT_SOURCES:
            for gate in GATES:
                full_audits.append(
                    audit_fullbody_one(
                        npz_path=full_npz,
                        scene_dir=full_scene,
                        output_dir=out_root / "fullbody_hand_audit",
                        source=source,
                        gate=gate,
                        fixed_threshold=float(args.fixed_threshold),
                        args=args,
                    )
                )
        fullbody_gate = summarize_fullbody_audits(full_audits)
        (out_root / "metrics" / "fullbody_hand_gate.json").write_text(
            json.dumps(fullbody_gate, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if not args.skip_open3d:
            full_out = out_root / "open3d_fullbody"
            for roi in FULLBODY_ROIS:
                for source in POINT_SOURCES:
                    for gate in GATES:
                        render_outputs.append(
                            render_one(
                                npz_path=full_npz,
                                scene_dir=full_scene,
                                output_dir=full_out,
                                source=source,
                                roi=roi,
                                gate=gate,
                                fixed_threshold=float(args.fixed_threshold),
                                args=args,
                            )
                        )

    (out_root / "metrics" / "render_outputs.json").write_text(
        json.dumps(render_outputs, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    contact_sheet = None if args.skip_open3d else make_contact_sheet(out_root)
    write_report(
        out_root / "report.md",
        args=args,
        numeric_gate=numeric_gate,
        fullbody_gate=fullbody_gate,
        aux_outputs=aux_outputs,
        render_outputs=render_outputs,
        contact_sheet=contact_sheet,
    )
    package_summary = {
        "candidate_name": args.candidate_name,
        "output_dir": str(out_root),
        "numeric_gate": numeric_gate,
        "fullbody_gate": fullbody_gate,
        "aux_outputs": aux_outputs,
        "render_outputs": render_outputs,
        "contact_sheet": contact_sheet,
        "report": str(out_root / "report.md"),
    }
    (out_root / "candidate_gate_summary.json").write_text(
        json.dumps(package_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(package_summary, indent=2, ensure_ascii=False))
    # Do not make Open3D visual quality a silent pass. Exit non-zero only for
    # automatic gate failures so pipelines can stop before cloud upload.
    return 0 if numeric_gate.get("pass") and (fullbody_gate is not None and fullbody_gate.get("pass")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
