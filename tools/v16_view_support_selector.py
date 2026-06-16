from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from v15_common import LOCAL_ROOT, REPORTS, json_ready, write_json  # noqa: E402


DEFAULT_CASE_ROOT = REPO_ROOT / "output/training_cases/0012_11_frame0000_6views_smplx_native_prior_v15"
DEFAULT_RASTER_ROOT = LOCAL_ROOT / "V15_SMPLX_native_camera_raster_export"
DEFAULT_AUTOPSY_JSON = REPORTS / "20260508_v16_prior_metric_autopsy.json"
DEFAULT_OUTPUT_DIR = LOCAL_ROOT / "V16_view_support_selector"
DEFAULT_OUTPUT_JSON = REPORTS / "20260508_v16_view_support_selector.json"
DEFAULT_OUTPUT_MD = REPORTS / "20260508_v16_view_support_selector.md"
DEFAULT_OUTPUT_CSV = REPORTS / "20260508_v16_view_support_selector_sets.csv"


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


def as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    arr = np.asarray(value)
    if arr.ndim == 0:
        return [str(arr.item())]
    return [str(item) for item in arr.tolist()]


def pixel_union(mask: np.ndarray, indices: list[int]) -> int:
    if not indices:
        return 0
    return int(np.any(mask[indices], axis=0).sum())


def weighted_counts(values: dict[str, np.ndarray], indices: list[int], weights: dict[str, float]) -> float:
    total = 0.0
    for key, weight in weights.items():
        total += float(weight) * float(np.asarray(values[key])[indices].sum())
    return total


def best_combo(
    view_indices: list[int],
    *,
    max_size: int,
    values: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    weights: dict[str, float],
) -> list[int]:
    if not view_indices:
        return []
    max_size = max(1, min(int(max_size), len(view_indices)))
    best: tuple[float, tuple[int, ...]] | None = None
    for size in range(1, max_size + 1):
        for combo in combinations(view_indices, size):
            combo_list = list(combo)
            score = weighted_counts(values, combo_list, weights)
            score += 0.15 * pixel_union(masks["hand"], combo_list)
            score += 0.12 * pixel_union(masks["head"], combo_list)
            score += 0.10 * pixel_union(masks["face"], combo_list)
            score += 0.05 * pixel_union(masks["body"], combo_list)
            score += 10.0 * len(set(combo_list))
            key = (score, tuple(-idx for idx in combo))
            if best is None or key > best:
                best = key
    if best is None:
        return view_indices[:max_size]
    return [-idx for idx in best[1]]


def set_metrics(name: str, indices: list[int], camera_ids: list[str], values: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> dict[str, Any]:
    raw = int(values["raw"][indices].sum()) if indices else 0
    true = int(values["true"][indices].sum()) if indices else 0
    prior = int(values["prior"][indices].sum()) if indices else 0
    visible = int(values["visible"][indices].sum()) if indices else 0
    hand = int(values["hand"][indices].sum()) if indices else 0
    left_hand = int(values["left_hand"][indices].sum()) if indices else 0
    right_hand = int(values["right_hand"][indices].sum()) if indices else 0
    head = int(values["head"][indices].sum()) if indices else 0
    face = int(values["face"][indices].sum()) if indices else 0
    body = int(values["body"][indices].sum()) if indices else 0
    return {
        "set_name": name,
        "view_count": len(indices),
        "view_indices": [int(idx) for idx in indices],
        "camera_ids": [camera_ids[idx] for idx in indices],
        "raw_pixels_sum": raw,
        "true_smplx_pixels_sum": true,
        "visible_subset_pixels_sum": visible,
        "native_prior_pixels_sum": prior,
        "body_pixels_sum": body,
        "hand_pixels_sum": hand,
        "left_hand_pixels_sum": left_hand,
        "right_hand_pixels_sum": right_hand,
        "head_pixels_sum": head,
        "face_front_pixels_sum": face,
        "hand_union_pixels": pixel_union(masks["hand"], indices),
        "head_union_pixels": pixel_union(masks["head"], indices),
        "face_front_union_pixels": pixel_union(masks["face"], indices),
        "body_union_pixels": pixel_union(masks["body"], indices),
        "hand_per_raw_ratio": ratio(hand, raw),
        "head_per_raw_ratio": ratio(head, raw),
        "face_per_raw_ratio": ratio(face, raw),
        "native_prior_per_true_smplx_ratio": ratio(prior, true),
        "native_prior_per_visible_subset_ratio": ratio(prior, visible),
        "notes": "",
    }


def greedy_existing_order(view_count: int, target_index: int) -> list[int]:
    if view_count <= 0:
        return []
    order = [max(0, min(int(target_index), view_count - 1))]
    order.extend(idx for idx in range(view_count) if idx not in order)
    return order


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "set_name",
        "view_count",
        "camera_ids",
        "hand_pixels_sum",
        "head_pixels_sum",
        "face_front_pixels_sum",
        "body_pixels_sum",
        "hand_union_pixels",
        "head_union_pixels",
        "face_front_union_pixels",
        "native_prior_per_visible_subset_ratio",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["camera_ids"] = " ".join(str(item) for item in row.get("camera_ids", []))
            writer.writerow({field: out.get(field, "") for field in fields})


def make_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# V16 View Support Selector",
        "",
        f"Status: `{summary.get('status')}`",
        "",
        "Research-only view selection. No predictions, teacher package, candidate package, registry, or strict pass state is written.",
        "",
        "## Recommended Sets",
        "",
        "| Set | Views | Cameras | Hands | Head | Face | Body | Notes |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in summary.get("recommended_sets", []):
        lines.append(
            f"| {row['set_name']} | {row['view_count']} | {' '.join(row['camera_ids'])} | "
            f"{row['hand_pixels_sum']} | {row['head_pixels_sum']} | {row['face_front_pixels_sum']} | "
            f"{row['body_pixels_sum']} | {row.get('notes', '')} |"
        )
    lines.extend(["", "## View Support", ""])
    lines.append("| View | Camera | Raw | Prior | Hands | Head | Face | Support Class |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for row in summary.get("per_view_support", []):
        lines.append(
            f"| {row['view_index']} | {row['camera_id']} | {row['raw_mask_pixels']} | {row['native_prior_pixels']} | "
            f"{row['hand_pixels']} | {row['head_pixels']} | {row['face_front_pixels']} | {row['support_class']} |"
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
    out = safe_v16_output_dir(args.output_dir, "V16_view_support_selector")
    manifest_path = case_root / "case_manifest.json"
    inputs_path = case_root / "inputs.npz"
    targets_path = case_root / "targets.npz"
    part_path = raster_root / "prior_part_maps.npz"
    mask_path = raster_root / "prior_mask.npz"
    required = [manifest_path, inputs_path, targets_path, part_path, mask_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {
            "task": "v16_view_support_selector",
            "created_utc": utc_now(),
            "status": "v16_view_support_selector_blocked_missing_inputs",
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
    autopsy = read_json(Path(args.autopsy_json).expanduser().resolve())
    inputs = load_npz(inputs_path)
    targets = load_npz(targets_path)
    parts = load_npz(part_path)
    masks_payload = load_npz(mask_path)
    camera_ids = as_string_list(inputs.get("camera_ids", manifest.get("camera_ids")))
    view_roles = as_string_list(inputs.get("view_roles", manifest.get("view_roles")))
    raw_mask = bool_arr(inputs.get("point_masks", targets.get("teacher_mask")))
    prior_mask = bool_arr(inputs.get("prior_mask", targets.get("smplx_native_visible_mask", masks_payload.get("prior_mask"))))
    visible_mask = bool_arr(masks_payload.get("prior_visibility", targets.get("smplx_native_visible_mask", prior_mask)))
    true_smplx = bool_arr(masks_payload.get("silhouette", raw_mask))
    body = bool_arr(parts.get("body_map", targets.get("smplx_body_anchor_mask", prior_mask)))
    left_hand = bool_arr(parts.get("left_hand_map", targets.get("smplx_left_hand_anchor_mask", np.zeros_like(prior_mask))))
    right_hand = bool_arr(parts.get("right_hand_map", targets.get("smplx_right_hand_anchor_mask", np.zeros_like(prior_mask))))
    hand = bool_arr(parts.get("hand_map", left_hand | right_hand))
    head = bool_arr(parts.get("head_map", np.zeros_like(prior_mask)))
    face = bool_arr(parts.get("face_front_map", np.zeros_like(prior_mask)))
    view_count = int(prior_mask.shape[0])
    if len(camera_ids) != view_count:
        camera_ids = [str(idx).zfill(2) for idx in range(view_count)]
    if len(view_roles) != view_count:
        view_roles = ["unknown"] * view_count

    values = {
        "raw": raw_mask.reshape(view_count, -1).sum(axis=1).astype(np.int64),
        "true": true_smplx.reshape(view_count, -1).sum(axis=1).astype(np.int64),
        "visible": visible_mask.reshape(view_count, -1).sum(axis=1).astype(np.int64),
        "prior": prior_mask.reshape(view_count, -1).sum(axis=1).astype(np.int64),
        "body": body.reshape(view_count, -1).sum(axis=1).astype(np.int64),
        "left_hand": left_hand.reshape(view_count, -1).sum(axis=1).astype(np.int64),
        "right_hand": right_hand.reshape(view_count, -1).sum(axis=1).astype(np.int64),
        "hand": hand.reshape(view_count, -1).sum(axis=1).astype(np.int64),
        "head": head.reshape(view_count, -1).sum(axis=1).astype(np.int64),
        "face": face.reshape(view_count, -1).sum(axis=1).astype(np.int64),
    }
    masks = {"body": body, "hand": hand, "head": head, "face": face}

    per_view_support: list[dict[str, Any]] = []
    for idx in range(view_count):
        classes = []
        if int(values["hand"][idx]) > 0:
            classes.append("hand")
        if int(values["head"][idx]) > 0:
            classes.append("head")
        if int(values["face"][idx]) > 0:
            classes.append("face")
        if int(values["body"][idx]) > 0:
            classes.append("body")
        if not classes:
            classes.append("none")
        per_view_support.append(
            {
                "view_index": idx,
                "camera_id": camera_ids[idx],
                "view_role": view_roles[idx],
                "raw_mask_pixels": int(values["raw"][idx]),
                "true_smplx_pixels": int(values["true"][idx]),
                "visible_subset_pixels": int(values["visible"][idx]),
                "native_prior_pixels": int(values["prior"][idx]),
                "body_pixels": int(values["body"][idx]),
                "left_hand_pixels": int(values["left_hand"][idx]),
                "right_hand_pixels": int(values["right_hand"][idx]),
                "hand_pixels": int(values["hand"][idx]),
                "head_pixels": int(values["head"][idx]),
                "face_front_pixels": int(values["face"][idx]),
                "hand_rank": int((-values["hand"]).argsort().tolist().index(idx) + 1),
                "head_rank": int((-values["head"]).argsort().tolist().index(idx) + 1),
                "body_rank": int((-values["body"]).argsort().tolist().index(idx) + 1),
                "support_class": "+".join(classes),
            }
        )

    view_indices = list(range(view_count))
    target_index = int(manifest.get("target_view_index", 0) or 0)
    existing_sparse = greedy_existing_order(view_count, target_index)
    max_set_size = min(int(args.max_set_size), view_count)
    hand_balanced = best_combo(
        view_indices,
        max_size=max_set_size,
        values=values,
        masks=masks,
        weights={"hand": 8.0, "left_hand": 4.0, "right_hand": 4.0, "head": 1.0, "body": 0.2, "prior": 0.05},
    )
    head_set = best_combo(
        view_indices,
        max_size=max_set_size,
        values=values,
        masks=masks,
        weights={"head": 6.0, "face": 4.0, "hand": 0.5, "body": 0.2, "prior": 0.05},
    )
    body_set = best_combo(
        view_indices,
        max_size=max_set_size,
        values=values,
        masks=masks,
        weights={"body": 3.0, "prior": 0.4, "hand": 0.5, "head": 0.5},
    )
    balanced_set = best_combo(
        view_indices,
        max_size=max_set_size,
        values=values,
        masks=masks,
        weights={"body": 1.0, "hand": 4.0, "left_hand": 2.0, "right_hand": 2.0, "head": 3.0, "face": 2.5, "prior": 0.15},
    )

    recommended_sets = [
        set_metrics("S0_existing_sparse_6v", existing_sparse, camera_ids, values, masks),
        set_metrics("hand_balanced_sparse", hand_balanced, camera_ids, values, masks),
        set_metrics("head_face_sparse", head_set, camera_ids, values, masks),
        set_metrics("body_sparse", body_set, camera_ids, values, masks),
        set_metrics("balanced_sparse", balanced_set, camera_ids, values, masks),
    ]
    for row in recommended_sets:
        if row["set_name"] == "S0_existing_sparse_6v":
            row["notes"] = "Existing V15 sparse view order; keeps target view first."
        elif row["set_name"] == "hand_balanced_sparse":
            row["notes"] = "Sparse support is left-hand heavy; right-hand support exists only where available."
        elif row["set_name"] == "head_face_sparse":
            row["notes"] = "Uses V15 heuristic head/face maps from the real-camera raster export."
        elif row["set_name"] == "body_sparse":
            row["notes"] = "Optimizes native prior/body-anchor pixels within available six views."
        elif row["set_name"] == "balanced_sparse":
            row["notes"] = "Balances hand, head/face, and body support under the sparse artifact constraint."

    full_60_available = bool((autopsy.get("metrics") or {}).get("full_60_view_support_available")) or view_count >= 60
    blockers: list[str] = []
    if not full_60_available:
        blockers.append(
            f"Only {view_count} V15 native prior views are available; full 60-view support requires reraster/export before final selection."
        )
    if int(values["right_hand"].sum()) <= 0:
        blockers.append("No right-hand pixels are available, so a two-hand balanced set cannot be selected.")
    if int(values["head"].sum()) <= 0:
        blockers.append("No head pixels are available, so head/face support cannot be ranked.")

    full60_recommendation = {
        "status": "needs_reraster" if not full_60_available else "available",
        "requested_view_count": 60,
        "available_view_count": view_count,
        "reason": (
            "Reraster SMPL-X native masks/parts across the full 60 camera contract, then rerun this selector."
            if not full_60_available
            else "Full 60-view support appears available in the provided artifact."
        ),
    }
    coverage_notes = [
        "View support is computed from the V15 native case plus V15 real-camera raster part maps.",
        "S0 existing sparse 6v is emitted even when full 60-view support is unavailable.",
        "The selector does not write predictions, packages, registries, or strict pass state.",
    ]
    if not full_60_available:
        coverage_notes.append("Full 60-view support needs reraster; sparse recommendations are a useful table, not a final 60-view policy.")

    npz_path = out / "view_support_scores.npz"
    np.savez_compressed(
        npz_path,
        camera_ids=np.asarray(camera_ids),
        raw_pixels=values["raw"],
        true_smplx_pixels=values["true"],
        visible_subset_pixels=values["visible"],
        native_prior_pixels=values["prior"],
        body_pixels=values["body"],
        left_hand_pixels=values["left_hand"],
        right_hand_pixels=values["right_hand"],
        hand_pixels=values["hand"],
        head_pixels=values["head"],
        face_front_pixels=values["face"],
    )
    write_csv(Path(args.output_csv), recommended_sets)
    summary = {
        "task": "v16_view_support_selector",
        "created_utc": utc_now(),
        "status": "v16_view_support_selector_ready_sparse6_needs_60view_reraster"
        if not full_60_available
        else "v16_view_support_selector_ready",
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
            "autopsy_json": str(Path(args.autopsy_json).expanduser().resolve()),
            "autopsy_status": autopsy.get("status"),
        },
        "metrics": {
            "available_views": view_count,
            "max_set_size": max_set_size,
            "full_60_view_support_available": full_60_available,
            "hand_view_count": int(sum(1 for value in values["hand"] if int(value) > 0)),
            "head_view_count": int(sum(1 for value in values["head"] if int(value) > 0)),
            "face_front_view_count": int(sum(1 for value in values["face"] if int(value) > 0)),
            "body_view_count": int(sum(1 for value in values["body"] if int(value) > 0)),
            "left_hand_pixels_total": int(values["left_hand"].sum()),
            "right_hand_pixels_total": int(values["right_hand"].sum()),
            "head_pixels_total": int(values["head"].sum()),
            "face_front_pixels_total": int(values["face"].sum()),
            "body_pixels_total": int(values["body"].sum()),
        },
        "per_view_support": per_view_support,
        "recommended_sets": recommended_sets,
        "full60_recommendation": full60_recommendation,
        "coverage_notes": coverage_notes,
        "outputs": {
            "summary_json": str(Path(args.output_json).expanduser().resolve()),
            "summary_md": str(Path(args.output_md).expanduser().resolve()),
            "sets_csv": str(Path(args.output_csv).expanduser().resolve()),
            "scores_npz": str(npz_path),
        },
        "blockers": blockers,
        "decision": (
            "Sparse six-view recommendations are available; full 60-view support needs reraster before final routing."
            if not full_60_available
            else "View support selector produced recommendations from the available full-view artifact."
        ),
    }
    write_json(out / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="V16-DLINE SMPL-X view support selector.")
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--raster-root", type=Path, default=DEFAULT_RASTER_ROOT)
    parser.add_argument("--autopsy-json", type=Path, default=DEFAULT_AUTOPSY_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--max-set-size", type=int, default=6)
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
    return 0 if summary["status"].startswith("v16_view_support_selector_ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
