from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from v15_common import LOCAL_ROOT, REPORTS, json_ready, scalar_stats, write_json  # noqa: E402


DEFAULT_CASE_ROOT = REPO_ROOT / "output/training_cases/0012_11_frame0000_6views_smplx_native_prior_v15"
DEFAULT_RASTER_ROOT = LOCAL_ROOT / "V15_SMPLX_native_camera_raster_export"
DEFAULT_OUTPUT_DIR = LOCAL_ROOT / "V16_prior_metric_autopsy"
DEFAULT_OUTPUT_JSON = REPORTS / "20260508_v16_prior_metric_autopsy.json"
DEFAULT_OUTPUT_MD = REPORTS / "20260508_v16_prior_metric_autopsy.md"
DEFAULT_OUTPUT_CSV = REPORTS / "20260508_v16_prior_metric_autopsy_per_view.csv"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_v16_output_dir(path: Path, token: str) -> Path:
    resolved = path.expanduser().resolve()
    lower = resolved.as_posix().lower()
    if "surface_research_preflight_local" not in lower or token.lower() not in lower:
        raise ValueError(f"Refusing non-V16 research output path: {resolved}")
    for forbidden in ("predictions", "teacher_export", "candidate_export", "strict_gate_registry", "strict_pass", "package"):
        if forbidden in lower:
            raise ValueError(f"Refusing forbidden output path token {forbidden!r}: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def bool_arr(value: np.ndarray) -> np.ndarray:
    return np.asarray(value).astype(bool)


def ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = bool_arr(a)
    bb = bool_arr(b)
    union = int(np.logical_or(aa, bb).sum())
    return float(np.logical_and(aa, bb).sum() / union) if union else 0.0


def mask_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    aa = bool_arr(a)
    bb = bool_arr(b)
    intersection = int(np.logical_and(aa, bb).sum())
    union = int(np.logical_or(aa, bb).sum())
    a_pixels = int(aa.sum())
    b_pixels = int(bb.sum())
    return {
        "a_pixels": a_pixels,
        "b_pixels": b_pixels,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": float(intersection / union) if union else 0.0,
        "a_recall_by_b": ratio(intersection, b_pixels),
        "b_recall_by_a": ratio(intersection, a_pixels),
        "a_only_pixels": int(np.logical_and(aa, ~bb).sum()),
        "b_only_pixels": int(np.logical_and(~aa, bb).sum()),
    }


def as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    arr = np.asarray(value)
    if arr.ndim == 0:
        return [str(arr.item())]
    return [str(item) for item in arr.tolist()]


def channel_names(manifest: dict[str, Any], prior_maps: np.ndarray) -> list[str]:
    names = as_string_list(
        (manifest.get("prior_input_meta") or {}).get("channel_names")
        or manifest.get("prior_channels")
    )
    if len(names) != int(prior_maps.shape[1]):
        names = [f"prior_channel_{idx:03d}" for idx in range(int(prior_maps.shape[1]))]
    return names


def pick_channel(prior_maps: np.ndarray, names: list[str], name: str, fallback: np.ndarray | None = None) -> np.ndarray:
    if name in names:
        return np.asarray(prior_maps[:, names.index(name)])
    if fallback is not None:
        return np.asarray(fallback)
    raise KeyError(f"Required channel not found: {name}")


def summarize_pixels(values: list[float]) -> dict[str, Any]:
    return scalar_stats(np.asarray(values, dtype=np.float64))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def make_markdown(summary: dict[str, Any], path: Path) -> None:
    metrics = summary.get("metrics", {})
    lines = [
        "# V16 Prior Metric Autopsy",
        "",
        f"Status: `{summary.get('status')}`",
        "",
        "Research-only audit. No predictions, teacher package, candidate package, registry, or strict pass state is written.",
        "",
        "## Key Metrics",
        "",
        f"- available_views: `{metrics.get('available_views')}`",
        f"- full_60_view_support_available: `{metrics.get('full_60_view_support_available')}`",
        f"- true_smplx_vs_raw_mask_iou_mean: `{metrics.get('true_smplx_vs_raw_mask_iou_mean')}`",
        f"- native_prior_vs_raw_mask_iou_mean: `{metrics.get('native_prior_vs_raw_mask_iou_mean')}`",
        f"- visible_subset_vs_raw_mask_iou_mean: `{metrics.get('visible_subset_vs_raw_mask_iou_mean')}`",
        f"- native_prior_over_visible_subset_mean: `{metrics.get('native_prior_over_visible_subset_mean')}`",
        f"- visible_subset_discrepancy_pixels_total: `{metrics.get('visible_subset_discrepancy_pixels_total')}`",
        f"- hand_visible_pixels_total: `{metrics.get('hand_visible_pixels_total')}`",
        f"- head_visible_pixels_total: `{metrics.get('head_visible_pixels_total')}`",
        f"- face_front_visible_pixels_total: `{metrics.get('face_front_visible_pixels_total')}`",
        "",
        "## Per View",
        "",
        "| View | Raw | True SMPL-X | Prior | True/Raw IoU | Prior/Raw IoU | Prior/True | Hands | Head | Face |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("per_view", []):
        lines.append(
            "| {camera_id} | {raw_mask_pixels} | {true_smplx_pixels} | {native_prior_pixels} | "
            "{true_smplx_vs_raw_iou:.6f} | {native_prior_vs_raw_iou:.6f} | {native_prior_over_true_smplx_ratio:.6f} | "
            "{hand_pixels} | {head_pixels} | {face_front_pixels} |".format(**row)
        )
    lines.extend(["", "## Coverage Notes", ""])
    for note in summary.get("coverage_notes", []):
        lines.append(f"- {note}")
    lines.extend(["", "## Outputs", ""])
    for key, value in summary.get("outputs", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Blockers", ""])
    blockers = summary.get("blockers") or []
    lines.extend([f"- {item}" for item in blockers] if blockers else ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    case_root = Path(args.case_root).expanduser().resolve()
    raster_root = Path(args.raster_root).expanduser().resolve()
    out = safe_v16_output_dir(args.output_dir, "V16_prior_metric_autopsy")
    blockers: list[str] = []

    manifest_path = case_root / "case_manifest.json"
    inputs_path = case_root / "inputs.npz"
    targets_path = case_root / "targets.npz"
    raster_summary_path = raster_root / "summary.json"
    raster_mask_path = raster_root / "prior_mask.npz"
    raster_part_path = raster_root / "prior_part_maps.npz"
    required = [manifest_path, inputs_path, targets_path, raster_summary_path, raster_mask_path, raster_part_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {
            "task": "v16_prior_metric_autopsy",
            "created_utc": utc_now(),
            "status": "v16_prior_metric_autopsy_blocked_missing_inputs",
            "research_only": True,
            "no_predictions_write": True,
            "no_teacher_export": True,
            "no_candidate_export": True,
            "no_registry_write": True,
            "no_strict_pass_claim": True,
            "inputs": {"case_root": str(case_root), "raster_root": str(raster_root)},
            "blockers": [f"Missing required input: {path}" for path in missing],
        }

    manifest = read_json(manifest_path)
    raster_summary = read_json(raster_summary_path)
    inputs = load_npz(inputs_path)
    targets = load_npz(targets_path)
    raster_masks = load_npz(raster_mask_path)
    part_maps = load_npz(raster_part_path)

    prior_maps = np.asarray(inputs["prior_maps"])
    names = channel_names(manifest, prior_maps)
    raw_mask = bool_arr(inputs.get("point_masks", targets.get("teacher_mask")))
    silhouette = bool_arr(pick_channel(prior_maps, names, "silhouette", raster_masks.get("silhouette", raw_mask)) > 0.5)
    visible_subset = bool_arr(pick_channel(prior_maps, names, "smplx_visible_mask", raster_masks.get("prior_visibility", silhouette)) > 0.5)
    native_prior = bool_arr(inputs.get("prior_mask", targets.get("smplx_native_visible_mask", visible_subset)))
    target_native_visible = bool_arr(targets.get("smplx_native_visible_mask", native_prior))
    raster_prior = bool_arr(raster_masks.get("prior_mask", native_prior))
    raster_raw = bool_arr(raster_masks.get("raw_mask", raw_mask))
    raster_silhouette = bool_arr(raster_masks.get("silhouette", silhouette))

    body_map = bool_arr(part_maps.get("body_map", targets.get("smplx_body_anchor_mask", native_prior)))
    left_hand = bool_arr(part_maps.get("left_hand_map", targets.get("smplx_left_hand_anchor_mask", np.zeros_like(native_prior))))
    right_hand = bool_arr(part_maps.get("right_hand_map", targets.get("smplx_right_hand_anchor_mask", np.zeros_like(native_prior))))
    hand_map = bool_arr(part_maps.get("hand_map", left_hand | right_hand))
    head_map = bool_arr(part_maps.get("head_map", np.zeros_like(native_prior)))
    face_front_map = bool_arr(part_maps.get("face_front_map", np.zeros_like(native_prior)))

    camera_ids = as_string_list(inputs.get("camera_ids", manifest.get("camera_ids")))
    view_roles = as_string_list(inputs.get("view_roles", manifest.get("view_roles")))
    view_count = int(native_prior.shape[0])
    if len(camera_ids) != view_count:
        camera_ids = [str(idx).zfill(2) for idx in range(view_count)]
    if len(view_roles) != view_count:
        view_roles = ["unknown"] * view_count

    full_60_available = view_count >= 60 or int(manifest.get("num_views", view_count) or view_count) >= 60
    if not full_60_available:
        blockers.append(
            f"Only {view_count} V15 native prior views are available here; full 60-view support needs reraster/export."
        )

    if not np.array_equal(native_prior, target_native_visible):
        blockers.append("inputs.prior_mask and targets.smplx_native_visible_mask are not identical.")
    if not np.array_equal(native_prior, raster_prior):
        blockers.append("V15 raster prior_mask export differs from native case prior_mask.")
    if not np.array_equal(raw_mask, raster_raw):
        blockers.append("V15 raster raw_mask export differs from native case point_masks.")
    if not np.array_equal(silhouette, raster_silhouette):
        blockers.append("V15 raster silhouette export differs from native case silhouette channel.")

    per_view: list[dict[str, Any]] = []
    for idx in range(view_count):
        raw = raw_mask[idx]
        true = silhouette[idx]
        visible = visible_subset[idx]
        prior = native_prior[idx]
        body = body_map[idx]
        hands = hand_map[idx]
        head = head_map[idx]
        face = face_front_map[idx]
        prior_vs_visible = mask_metrics(prior, visible)
        true_vs_raw = mask_metrics(true, raw)
        prior_vs_raw = mask_metrics(prior, raw)
        visible_vs_raw = mask_metrics(visible, raw)
        row = {
            "view_index": idx,
            "camera_id": camera_ids[idx],
            "view_role": view_roles[idx],
            "raw_mask_pixels": int(raw.sum()),
            "true_smplx_pixels": int(true.sum()),
            "visible_subset_pixels": int(visible.sum()),
            "native_prior_pixels": int(prior.sum()),
            "body_pixels": int(body.sum()),
            "left_hand_pixels": int(left_hand[idx].sum()),
            "right_hand_pixels": int(right_hand[idx].sum()),
            "hand_pixels": int(hands.sum()),
            "head_pixels": int(head.sum()),
            "face_front_pixels": int(face.sum()),
            "true_smplx_vs_raw_iou": true_vs_raw["iou"],
            "visible_subset_vs_raw_iou": visible_vs_raw["iou"],
            "native_prior_vs_raw_iou": prior_vs_raw["iou"],
            "native_prior_vs_true_smplx_iou": mask_iou(prior, true),
            "native_prior_vs_visible_subset_iou": prior_vs_visible["iou"],
            "native_prior_over_true_smplx_ratio": ratio(int(prior.sum()), int(true.sum())),
            "native_prior_over_raw_ratio": ratio(int(prior.sum()), int(raw.sum())),
            "visible_subset_over_raw_ratio": ratio(int(visible.sum()), int(raw.sum())),
            "visible_subset_over_true_smplx_ratio": ratio(int(visible.sum()), int(true.sum())),
            "native_prior_over_visible_subset_ratio": ratio(int(prior.sum()), int(visible.sum())),
            "visible_subset_discrepancy_pixels": int(np.logical_xor(prior, visible).sum()),
            "visible_subset_missing_prior_pixels": int(np.logical_and(visible, ~prior).sum()),
            "prior_extra_over_visible_pixels": int(np.logical_and(prior, ~visible).sum()),
            "head_raw_coverage_ratio": ratio(int(head.sum()), int(raw.sum())),
            "hand_raw_coverage_ratio": ratio(int(hands.sum()), int(raw.sum())),
            "face_raw_coverage_ratio": ratio(int(face.sum()), int(raw.sum())),
            "head_prior_coverage_ratio": ratio(int(head.sum()), int(prior.sum())),
            "hand_prior_coverage_ratio": ratio(int(hands.sum()), int(prior.sum())),
            "face_prior_coverage_ratio": ratio(int(face.sum()), int(prior.sum())),
        }
        per_view.append(row)

    metrics = {
        "available_views": view_count,
        "full_60_view_support_available": full_60_available,
        "image_hw": manifest.get("image_hw", list(native_prior.shape[1:])),
        "raw_mask_pixels_total": int(raw_mask.sum()),
        "true_smplx_pixels_total": int(silhouette.sum()),
        "visible_subset_pixels_total": int(visible_subset.sum()),
        "native_prior_pixels_total": int(native_prior.sum()),
        "body_pixels_total": int(body_map.sum()),
        "left_hand_visible_pixels_total": int(left_hand.sum()),
        "right_hand_visible_pixels_total": int(right_hand.sum()),
        "hand_visible_pixels_total": int(hand_map.sum()),
        "head_visible_pixels_total": int(head_map.sum()),
        "face_front_visible_pixels_total": int(face_front_map.sum()),
        "true_smplx_vs_raw_mask_iou_total": mask_iou(silhouette, raw_mask),
        "visible_subset_vs_raw_mask_iou_total": mask_iou(visible_subset, raw_mask),
        "native_prior_vs_raw_mask_iou_total": mask_iou(native_prior, raw_mask),
        "native_prior_vs_true_smplx_iou_total": mask_iou(native_prior, silhouette),
        "native_prior_vs_visible_subset_iou_total": mask_iou(native_prior, visible_subset),
        "native_prior_over_true_smplx_ratio_total": ratio(int(native_prior.sum()), int(silhouette.sum())),
        "native_prior_over_visible_subset_ratio_total": ratio(int(native_prior.sum()), int(visible_subset.sum())),
        "visible_subset_discrepancy_pixels_total": int(np.logical_xor(native_prior, visible_subset).sum()),
        "visible_subset_missing_prior_pixels_total": int(np.logical_and(visible_subset, ~native_prior).sum()),
        "prior_extra_over_visible_pixels_total": int(np.logical_and(native_prior, ~visible_subset).sum()),
        "true_smplx_vs_raw_mask_iou_mean": float(np.mean([row["true_smplx_vs_raw_iou"] for row in per_view])),
        "visible_subset_vs_raw_mask_iou_mean": float(np.mean([row["visible_subset_vs_raw_iou"] for row in per_view])),
        "native_prior_vs_raw_mask_iou_mean": float(np.mean([row["native_prior_vs_raw_iou"] for row in per_view])),
        "native_prior_over_visible_subset_mean": float(np.mean([row["native_prior_over_visible_subset_ratio"] for row in per_view])),
        "hand_view_count": int(sum(1 for row in per_view if int(row["hand_pixels"]) > 0)),
        "head_view_count": int(sum(1 for row in per_view if int(row["head_pixels"]) > 0)),
        "face_front_view_count": int(sum(1 for row in per_view if int(row["face_front_pixels"]) > 0)),
        "per_view_true_smplx_vs_raw_iou": summarize_pixels([row["true_smplx_vs_raw_iou"] for row in per_view]),
        "per_view_native_prior_over_visible_subset": summarize_pixels([row["native_prior_over_visible_subset_ratio"] for row in per_view]),
    }

    coverage_notes = [
        "true_smplx is the dense silhouette channel from prior_maps/V15 raster export.",
        "visible_subset is the smplx_visible_mask channel; native_prior is inputs.prior_mask / targets.smplx_native_visible_mask.",
        "The visible subset and native prior differ strongly because native_prior keeps the body/anchor subset used by the V15 case.",
    ]
    if not full_60_available:
        coverage_notes.append(
            f"Only sparse {view_count}-view prior support exists in this artifact; hand/head/body ranking is useful for S0 but not a replacement for 60-view reraster."
        )

    per_view_npz = out / "per_view_metrics.npz"
    np.savez_compressed(
        per_view_npz,
        camera_ids=np.asarray(camera_ids),
        raw_mask_pixels=np.asarray([row["raw_mask_pixels"] for row in per_view], dtype=np.int64),
        true_smplx_pixels=np.asarray([row["true_smplx_pixels"] for row in per_view], dtype=np.int64),
        visible_subset_pixels=np.asarray([row["visible_subset_pixels"] for row in per_view], dtype=np.int64),
        native_prior_pixels=np.asarray([row["native_prior_pixels"] for row in per_view], dtype=np.int64),
        hand_pixels=np.asarray([row["hand_pixels"] for row in per_view], dtype=np.int64),
        head_pixels=np.asarray([row["head_pixels"] for row in per_view], dtype=np.int64),
        face_front_pixels=np.asarray([row["face_front_pixels"] for row in per_view], dtype=np.int64),
        true_smplx_vs_raw_iou=np.asarray([row["true_smplx_vs_raw_iou"] for row in per_view], dtype=np.float32),
        native_prior_vs_raw_iou=np.asarray([row["native_prior_vs_raw_iou"] for row in per_view], dtype=np.float32),
        native_prior_over_visible_subset_ratio=np.asarray([row["native_prior_over_visible_subset_ratio"] for row in per_view], dtype=np.float32),
    )
    write_csv(
        args.output_csv,
        per_view,
        [
            "view_index",
            "camera_id",
            "view_role",
            "raw_mask_pixels",
            "true_smplx_pixels",
            "visible_subset_pixels",
            "native_prior_pixels",
            "true_smplx_vs_raw_iou",
            "visible_subset_vs_raw_iou",
            "native_prior_vs_raw_iou",
            "native_prior_over_visible_subset_ratio",
            "visible_subset_discrepancy_pixels",
            "hand_pixels",
            "head_pixels",
            "face_front_pixels",
        ],
    )

    summary = {
        "task": "v16_prior_metric_autopsy",
        "created_utc": utc_now(),
        "status": "v16_prior_metric_autopsy_ready_sparse6_needs_60view_reraster"
        if not full_60_available
        else "v16_prior_metric_autopsy_ready",
        "research_only": True,
        "smplx_native_only": True,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_strict_pass_claim": True,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "inputs": {
            "case_root": str(case_root),
            "raster_root": str(raster_root),
            "raster_summary": str(raster_summary_path),
            "v15_raster_status": raster_summary.get("status"),
        },
        "metrics": metrics,
        "per_view": per_view,
        "coverage_notes": coverage_notes,
        "outputs": {
            "summary_json": str(Path(args.output_json).expanduser().resolve()),
            "summary_md": str(Path(args.output_md).expanduser().resolve()),
            "per_view_csv": str(Path(args.output_csv).expanduser().resolve()),
            "per_view_npz": str(per_view_npz),
        },
        "blockers": blockers,
        "decision": (
            "V16 prior metric autopsy is usable for sparse 6-view S0/view selection, but full 60-view support needs reraster."
            if not full_60_available
            else "V16 prior metric autopsy has full-view support available."
        ),
    }
    write_json(out / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="V16-DLINE routed SMPL-X prior metric autopsy.")
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--raster-root", type=Path, default=DEFAULT_RASTER_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    summary = build_summary(args)
    write_json(args.output_json, summary)
    make_markdown(summary, args.output_md)
    print(
        json.dumps(
            json_ready(
                {
                    "status": summary["status"],
                    "metrics": summary["metrics"],
                    "output_json": str(Path(args.output_json).resolve()),
                }
            ),
            ensure_ascii=False,
        )
    )
    return 0 if summary["status"].startswith("v16_prior_metric_autopsy_ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
