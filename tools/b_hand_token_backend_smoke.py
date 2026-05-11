from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
    "teacher_export": "blocked",
    "candidate_export": "blocked",
}
CONTRACT = {
    "research_only": True,
    "local_only": True,
    "no_train": True,
    "no_cloud": True,
    "no_candidate_export": True,
    "no_teacher_export": True,
    "no_strict_pass_write": True,
    "not_mediapipe_patch_teacher": True,
    "not_smplx_hand_residual_success_claim": True,
    "writes_predictions_npz": False,
    "writes_strict_registry": False,
    "writes_candidate": False,
    "writes_teacher": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only HGGT-inspired B-hand token backend smoke. It consumes a "
            "b_hand_evidence_cache.json and writes hand-token metadata plus blockers. "
            "It does not train, infer, write predictions, export teachers/candidates, "
            "write strict pass state, or call cloud."
        )
    )
    parser.add_argument("--hand-evidence-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch-start-idx", type=int, default=5)
    parser.add_argument("--min-side-views", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_cache(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_dir():
        resolved = resolved / "b_hand_evidence_cache.json"
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {resolved}")
    payload["_resolved_path"] = str(resolved)
    return payload


def token_ids_from_hook(hook: dict[str, Any], patch_start_idx: int) -> dict[str, Any]:
    patch_range = hook.get("patch_range_xyxy")
    grid = hook.get("patch_grid_hw")
    if not isinstance(patch_range, list) or len(patch_range) != 4 or not isinstance(grid, list) or len(grid) != 2:
        patch_ids = [int(v) for v in hook.get("token_ids_preview", [])]
        return {
            "status": "preview_only",
            "patch_token_ids": patch_ids,
            "aggregator_token_ids": [int(patch_start_idx + v) for v in patch_ids],
            "patch_range_xyxy": patch_range,
            "patch_grid_hw": grid,
        }
    px0, py0, px1, py1 = [int(v) for v in patch_range]
    grid_h, grid_w = [int(v) for v in grid]
    patch_ids: list[int] = []
    for y in range(max(0, py0), min(grid_h, py1)):
        for x in range(max(0, px0), min(grid_w, px1)):
            patch_ids.append(int(y * grid_w + x))
    return {
        "status": "from_patch_range",
        "patch_token_ids": patch_ids,
        "aggregator_token_ids": [int(patch_start_idx + v) for v in patch_ids],
        "patch_range_xyxy": [px0, py0, px1, py1],
        "patch_grid_hw": [grid_h, grid_w],
    }


def roi_quality(row: dict[str, Any]) -> dict[str, Any]:
    crop = row.get("crop_metadata") if isinstance(row.get("crop_metadata"), dict) else {}
    prior = row.get("smplx_prior") if isinstance(row.get("smplx_prior"), dict) else {}
    rays = row.get("camera_rays") if isinstance(row.get("camera_rays"), dict) else {}
    support = row.get("prediction_support") if isinstance(row.get("prediction_support"), dict) else {}
    roi_pixels = int(crop.get("roi_pixels", 0) or 0)
    visible_pixels = int(prior.get("visible_pixels", 0) or 0)
    support_pixels = int(support.get("support_pixels", 0) or 0)
    return {
        "roi_pixels": roi_pixels,
        "smplx_visible_pixels": visible_pixels,
        "smplx_visible_ratio_in_roi": float(visible_pixels / max(roi_pixels, 1)),
        "camera_rays_available": bool(rays.get("available")),
        "prediction_support_available": bool(support.get("available")),
        "prediction_support_pixels": support_pixels,
    }


def collect_side(cache: dict[str, Any], side: str, patch_start_idx: int) -> dict[str, Any]:
    per_view = cache.get("per_view") if isinstance(cache.get("per_view"), dict) else {}
    view_rows: dict[str, Any] = {}
    all_patch_ids: set[int] = set()
    all_agg_ids: set[int] = set()
    total_roi_pixels = 0
    total_smplx_visible = 0
    camera_view_count = 0
    prediction_view_count = 0
    roi_count = 0
    for view_key, view_payload in per_view.items():
        if not isinstance(view_payload, dict):
            continue
        rois = view_payload.get("hand_rois")
        if not isinstance(rois, list):
            continue
        side_rois: list[dict[str, Any]] = []
        for roi in rois:
            if not isinstance(roi, dict) or str(roi.get("side")) != side:
                continue
            hook = token_ids_from_hook(roi.get("vggt_token_hook", {}), patch_start_idx)
            quality = roi_quality(roi)
            all_patch_ids.update(int(v) for v in hook["patch_token_ids"])
            all_agg_ids.update(int(v) for v in hook["aggregator_token_ids"])
            total_roi_pixels += int(quality["roi_pixels"])
            total_smplx_visible += int(quality["smplx_visible_pixels"])
            roi_count += 1
            side_rois.append(
                {
                    "bbox_xyxy": roi.get("bbox_xyxy"),
                    "roi_source": roi.get("roi_source"),
                    "crop_metadata": roi.get("crop_metadata"),
                    "quality": quality,
                    "patch_tokens": {
                        key: value
                        for key, value in hook.items()
                        if key not in {"patch_token_ids", "aggregator_token_ids"}
                    }
                    | {
                        "patch_token_count": int(len(hook["patch_token_ids"])),
                        "aggregator_token_count": int(len(hook["aggregator_token_ids"])),
                        "aggregator_token_ids_preview": hook["aggregator_token_ids"][:48],
                    },
                }
            )
        if side_rois:
            camera_available = any(bool(row["quality"]["camera_rays_available"]) for row in side_rois)
            pred_available = any(bool(row["quality"]["prediction_support_available"]) for row in side_rois)
            camera_view_count += int(camera_available)
            prediction_view_count += int(pred_available)
            view_rows[str(view_key)] = {
                "roi_count": len(side_rois),
                "camera_rays_available": camera_available,
                "prediction_support_available": pred_available,
                "roi_pixels": int(sum(int(row["quality"]["roi_pixels"]) for row in side_rois)),
                "smplx_visible_pixels": int(sum(int(row["quality"]["smplx_visible_pixels"]) for row in side_rois)),
                "rois": side_rois,
            }
    support_views = int(len(view_rows))
    return {
        "side": side,
        "status": "ok" if support_views > 0 else "missing_roi",
        "support_views": support_views,
        "roi_count": roi_count,
        "total_roi_pixels": int(total_roi_pixels),
        "total_smplx_visible_pixels": int(total_smplx_visible),
        "camera_ray_views": int(camera_view_count),
        "prediction_support_views": int(prediction_view_count),
        "unique_patch_tokens": int(len(all_patch_ids)),
        "unique_aggregator_tokens": int(len(all_agg_ids)),
        "patch_token_ids_preview": sorted(all_patch_ids)[:64],
        "aggregator_token_ids_preview": sorted(all_agg_ids)[:64],
        "view_rows": view_rows,
        "token_plan": {
            "side_identity_token": f"{side}_hand_identity",
            "view_support_token": f"{side}_hand_view_support_histogram",
            "roi_patch_tokens": f"{side}_hand_roi_patch_token_set",
            "wrist_arm_anchor_token": f"{side}_wrist_arm_anchor_required_before_success",
            "finger_local_tokens": f"{side}_finger_tokens_require_backend_decoder",
            "confidence_token": f"{side}_hand_confidence_from_views_smplx_overlap_rays",
        },
    }


def build_stop_conditions(sides: dict[str, Any], min_side_views: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "name": "strict_gate_remains_zero",
            "must_stop": True,
            "reason": "B-hand smoke is research-only and cannot unblock cloud or write pass.",
        },
        {
            "name": "open3d_hand_must_connect_arm",
            "must_stop": True,
            "reason": "Any future visual review must reject detached hand sheets, sticks, or floating fragments.",
        },
        {
            "name": "no_hand_teacher_or_candidate_export",
            "must_stop": True,
            "reason": "Token metadata is not a hand teacher, MANO output, or candidate prediction.",
        },
    ]
    for side, row in sides.items():
        support_views = int(row.get("support_views", 0) or 0)
        camera_views = int(row.get("camera_ray_views", 0) or 0)
        rows.append(
            {
                "name": f"{side}_weak_view_support",
                "must_stop": support_views < int(min_side_views),
                "support_views": support_views,
                "min_side_views": int(min_side_views),
                "reason": f"{side} hand needs enough visible views before a decoder smoke can be trusted.",
            }
        )
        rows.append(
            {
                "name": f"{side}_missing_camera_rays",
                "must_stop": camera_views <= 0,
                "camera_ray_views": camera_views,
                "reason": f"{side} hand token backend needs camera-ray evidence for cross-view geometry.",
            }
        )
    return rows


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Hand Token Backend Smoke",
        "",
        "Status: `research_only_hggt_style_token_smoke`",
        "",
        "This is not a hand success claim, teacher, candidate, cloud run, or strict pass.",
        "",
        "## Gate Truth",
        "",
        "```json",
        json.dumps(STRICT_FACTS, indent=2),
        "```",
        "",
        "## Side Readout",
        "",
    ]
    for side, row in summary["sides"].items():
        lines.extend(
            [
                f"### {side}",
                "",
                f"- support_views: `{row['support_views']}`",
                f"- roi_count: `{row['roi_count']}`",
                f"- total_roi_pixels: `{row['total_roi_pixels']}`",
                f"- camera_ray_views: `{row['camera_ray_views']}`",
                f"- unique_aggregator_tokens: `{row['unique_aggregator_tokens']}`",
                "",
            ]
        )
    lines.extend(["## Stop Conditions", "", "```json", json.dumps(summary["stop_conditions"], indent=2), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = load_cache(args.hand_evidence_cache)
    sides = {
        "left": collect_side(cache, "left", int(args.patch_start_idx)),
        "right": collect_side(cache, "right", int(args.patch_start_idx)),
    }
    stop_conditions = build_stop_conditions(sides, int(args.min_side_views))
    summary = {
        "task": "b_hand_token_backend_smoke",
        "schema_version": 1,
        "status": "research_only_hggt_style_token_smoke",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "strict_facts": STRICT_FACTS,
        "contract": CONTRACT,
        "inputs": {
            "hand_evidence_cache": cache.get("_resolved_path"),
            "patch_start_idx": int(args.patch_start_idx),
            "min_side_views": int(args.min_side_views),
        },
        "source_cache": {
            "scene_dir": cache.get("scene_dir"),
            "target_size": cache.get("target_size"),
            "view_indices": cache.get("view_indices"),
            "visible_view_summary": cache.get("visible_view_summary"),
        },
        "sides": sides,
        "stop_conditions": stop_conditions,
        "outputs": {
            "summary_json": str(output_dir / "b_hand_token_backend_smoke_summary.json"),
            "report_md": str(output_dir / "b_hand_token_backend_smoke_report.md"),
        },
        "next_allowed_action": (
            "If both sides have sufficient evidence, build a local hand-token decoder skeleton. "
            "Do not export hand teacher/candidate or claim Open3D success without connected-arm visual pass."
        ),
    }
    write_json(output_dir / "b_hand_token_backend_smoke_summary.json", summary)
    write_markdown(output_dir / "b_hand_token_backend_smoke_report.md", summary)
    print(
        json.dumps(
            {
                "summary": str(output_dir / "b_hand_token_backend_smoke_summary.json"),
                "status": summary["status"],
                "must_stop": any(bool(item.get("must_stop")) for item in stop_conditions),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
