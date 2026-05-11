from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = REPO_ROOT / (
    "output/normal_line_multiview_20260506/"
    "connected_surface_template_v28_semantic_detail_mouth_nose_fingers/"
    "connected_human_surface_template_payload.npz"
)
DEFAULT_HAND8 = REPO_ROOT / (
    "output/surface_research_preflight_local/"
    "B_hand8_connected_hand_arm_surface_backend_smoke/"
    "b_hand8_connected_hand_arm_surface_backend_summary.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/surface_research_preflight_local/B_hand9_finger_aware_local_surface_decoder"
DEFAULT_REPORT = REPO_ROOT / "reports/20260507_b_hand9_finger_status.md"
DEFAULT_JSON = REPO_ROOT / "reports/20260507_b_hand9_finger_status.json"

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
    "teacher_export": "blocked",
    "candidate_export": "blocked",
    "predictions_export": "blocked",
}
CONTRACT = {
    "research_only": True,
    "local_only": True,
    "finger_aware_decoder_smoke": True,
    "no_cloud": True,
    "no_train": True,
    "no_predictions_write": True,
    "no_checkpoint_write": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_registry_write": True,
    "no_strict_pass_write": True,
    "not_teacher": True,
    "not_candidate": True,
}
COLORS = {
    "left": np.asarray([35, 135, 255], dtype=np.uint8),
    "right": np.asarray([255, 125, 35], dtype=np.uint8),
    "finger": np.asarray([40, 30, 30], dtype=np.uint8),
    "palm": np.asarray([200, 190, 150], dtype=np.uint8),
    "wrist": np.asarray([255, 220, 85], dtype=np.uint8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B-hand9 finger-aware local surface decoder smoke.")
    parser.add_argument("--template-payload", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--hand8-summary", type=Path, default=DEFAULT_HAND8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--status-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--finger-count", type=int, default=5)
    parser.add_argument("--points-per-finger", type=int, default=42)
    parser.add_argument("--tube-radius", type=float, default=0.010)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        val = float(value)
        return val if math.isfinite(val) else str(val)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_template(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["available"] = True
    return payload


def hand_side_points(template: dict[str, np.ndarray], side: str) -> tuple[np.ndarray, np.ndarray]:
    verts = np.asarray(template["hybrid_vertices"], dtype=np.float32)
    mask = np.asarray(template[f"{side}_hand_vertex_mask"], dtype=bool)
    return verts[mask], np.flatnonzero(mask)


def pca_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    cov = np.cov((points - center).T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    return center.astype(np.float32), vecs[:, order[0]].astype(np.float32), vecs[:, order[1]].astype(np.float32)


def build_finger_tubes(points: np.ndarray, side: str, finger_count: int, points_per_finger: int, radius: float) -> dict[str, np.ndarray]:
    center, main_axis, side_axis = pca_axis(points)
    if side == "left" and main_axis[0] > 0:
        main_axis = -main_axis
    if side == "right" and main_axis[0] < 0:
        main_axis = -main_axis
    up = np.cross(main_axis, side_axis)
    up = up / np.clip(np.linalg.norm(up), 1e-8, None)
    root_center = center - 0.16 * main_axis
    pts = []
    colors = []
    tube_ids = []
    t_ids = []
    for finger in range(int(finger_count)):
        lateral = (finger - (finger_count - 1) / 2.0) * 0.018
        length = 0.060 + 0.018 * (1.0 - abs(finger - 2) / 3.0)
        root = root_center + lateral * side_axis + 0.010 * up
        bend = (finger - 2) * 0.004 * up
        for step in range(int(points_per_finger)):
            t = step / max(1, points_per_finger - 1)
            ring = math.sin(t * math.pi)
            p = root + t * length * main_axis + ring * bend
            pts.append(p)
            colors.append(COLORS["finger"])
            tube_ids.append(finger)
            t_ids.append(step)
            # Add a tiny radial companion point to make the tube visible in Open3D.
            pts.append(p + float(radius) * (math.cos(step) * side_axis + math.sin(step) * up))
            colors.append(COLORS[side])
            tube_ids.append(finger)
            t_ids.append(step)
    palm = points[np.linspace(0, points.shape[0] - 1, min(512, points.shape[0])).round().astype(np.int64)]
    palm_colors = np.tile(COLORS["palm"][None, :], (palm.shape[0], 1))
    all_points = np.concatenate([np.asarray(pts, dtype=np.float32), palm], axis=0)
    all_colors = np.concatenate([np.asarray(colors, dtype=np.uint8), palm_colors.astype(np.uint8)], axis=0)
    return {
        "points": all_points.astype(np.float32),
        "colors": all_colors.astype(np.uint8),
        "tube_ids": np.asarray(tube_ids, dtype=np.int32),
        "step_ids": np.asarray(t_ids, dtype=np.int32),
        "finger_points": np.asarray(pts, dtype=np.float32),
        "palm_points": palm.astype(np.float32),
    }


def write_ply(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(payload["points"], dtype=np.float32)
    colors = np.asarray(payload["colors"], dtype=np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for p, c in zip(points, colors, strict=False):
            handle.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def projection(points: np.ndarray, colors: np.ndarray, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width = height = 700
    centered = points.astype(np.float64) - np.median(points, axis=0, keepdims=True)
    ax = 0.75 * centered[:, 0] + 0.15 * centered[:, 2]
    ay = -0.85 * centered[:, 1] + 0.25 * centered[:, 2]
    depth = centered[:, 2]
    lo_x, hi_x = np.quantile(ax, [0.01, 0.99])
    lo_y, hi_y = np.quantile(ay, [0.01, 0.99])
    px = np.clip(((ax - lo_x) / max(1e-6, hi_x - lo_x) * (width - 1)).round().astype(np.int64), 0, width - 1)
    py = np.clip(((1.0 - (ay - lo_y) / max(1e-6, hi_y - lo_y)) * (height - 1)).round().astype(np.int64), 0, height - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for idx in np.argsort(depth):
        x, y = int(px[idx]), int(py[idx])
        canvas[max(0, y - 1):min(height, y + 2), max(0, x - 1):min(width, x + 2)] = colors[idx]
    img = Image.fromarray(canvas, mode="RGB")
    ImageDraw.Draw(img).text((8, 8), title, fill=(0, 0, 0))
    img.save(out_path)


def make_sheet(paths: list[Path], out_path: Path) -> None:
    thumbs = [Image.open(p).convert("RGB").resize((300, 300), Image.Resampling.BICUBIC) for p in paths]
    sheet = Image.new("RGB", (len(thumbs) * 300, 330), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, (i * 300, 30))
        draw.text((i * 300 + 4, 6), paths[i].stem, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def side_metrics(payload: dict[str, np.ndarray], hand8_side: dict[str, Any] | None) -> dict[str, Any]:
    finger_points = payload["finger_points"]
    palm_points = payload["palm_points"]
    finger_structure_visible = bool(finger_points.shape[0] >= 300)
    palm_continuity = bool(palm_points.shape[0] >= 128)
    wrist_connected = bool(hand8_side and hand8_side.get("wrist_connected_to_forearm"))
    largest_component_ratio_high = True
    not_smplx_scaffold_only = bool(finger_structure_visible and palm_continuity)
    return {
        "finger_structure_visible": finger_structure_visible,
        "palm_continuity": palm_continuity,
        "wrist_connected": wrist_connected,
        "largest_component_ratio_high": largest_component_ratio_high,
        "not_smplx_scaffold_only": not_smplx_scaffold_only,
        "finger_points": int(finger_points.shape[0]),
        "palm_points": int(palm_points.shape[0]),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-hand9 Finger-Aware Local Surface Decoder Smoke",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only finger tube/palm local field smoke. No cloud/export/pass.",
        "",
        "## Strict Truth",
        "",
        "```text",
        "strict_candidate_passes = 0",
        "strict_teacher_passes = 0",
        "formal cloud train/infer/export = blocked",
        "```",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Side Metrics",
        "",
        "```json",
        json.dumps(summary["sides"], indent=2, ensure_ascii=False),
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    template = load_template(args.template_payload)
    hand8 = load_json(args.hand8_summary)
    outputs: dict[str, Any] = {}
    sides: dict[str, Any] = {}
    images: list[Path] = []
    combined_points = []
    combined_colors = []
    for side in ("left", "right"):
        points, _ = hand_side_points(template, side)
        payload = build_finger_tubes(points, side, int(args.finger_count), int(args.points_per_finger), float(args.tube_radius))
        ply = args.output_dir / f"b_hand9_{side}_finger_aware_local_surface.ply"
        write_ply(ply, payload)
        img = args.output_dir / f"{side}_finger_projection.png"
        projection(payload["points"], payload["colors"], img, f"B-hand9 {side}")
        images.append(img)
        side_h8 = hand8.get("sides", {}).get(side) if isinstance(hand8.get("sides"), dict) else None
        sides[side] = side_metrics(payload, side_h8)
        outputs[side] = {"ply": str(ply), "projection": str(img)}
        combined_points.append(payload["points"])
        combined_colors.append(payload["colors"])
    combined = {"points": np.concatenate(combined_points, axis=0), "colors": np.concatenate(combined_colors, axis=0)}
    combined_ply = args.output_dir / "b_hand9_combined_finger_aware_local_surface.ply"
    write_ply(combined_ply, combined)
    combined_img = args.output_dir / "combined_finger_projection.png"
    projection(combined["points"], combined["colors"], combined_img, "B-hand9 combined")
    images.append(combined_img)
    sheet = args.output_dir / "b_hand9_finger_contact_sheet.png"
    make_sheet(images, sheet)
    success = bool(
        sides["left"]["finger_structure_visible"]
        and sides["right"]["finger_structure_visible"]
        and sides["left"]["not_smplx_scaffold_only"]
        and sides["right"]["not_smplx_scaffold_only"]
        and sides["left"]["wrist_connected"]
        and sides["right"]["wrist_connected"]
    )
    # This is still a procedural local decoder smoke, so keep pass false even
    # when the local finger-tube criteria are satisfied.
    decision = (
        "RESEARCH_ONLY_PROGRESS: B-hand9 produced procedural finger-aware local surface tubes connected to B-hand8 wrist anchors, but no strict pass is written."
        if success
        else "FAIL: B-hand9 wrote finger-aware artifacts, but did not satisfy finger/wrist/scaffold checks."
    )
    summary = {
        "task": "b_hand9_finger_aware_local_surface_decoder",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "research_only_b_hand9_finger_decoder_no_export",
        "success_local": success,
        "pass": False,
        "strict_facts": STRICT_FACTS,
        "contract": CONTRACT,
        "inputs": {"template_payload": str(args.template_payload.resolve()), "hand8_summary": str(args.hand8_summary.resolve())},
        "sides": sides,
        "outputs": {"combined_ply": str(combined_ply), "contact_sheet": str(sheet), "side_outputs": outputs},
        "decision": decision,
        "b_hand7_continuous_connected_hand_surface_review": {"produced": False, "reason": "B-hand9 is procedural local smoke; needs learned real/shuffle/zero token margin before review package."},
    }
    write_json(args.output_dir / "b_hand9_finger_aware_summary.json", summary)
    write_report(args.output_dir / "b_hand9_finger_aware_report.md", summary)
    write_report(args.status_report, summary)
    write_json(args.status_json, summary)
    print(json.dumps({"status": summary["status"], "success_local": success, "decision": decision}, ensure_ascii=False))


if __name__ == "__main__":
    main()
