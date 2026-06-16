from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from normal_line_multiview_eval import (  # noqa: E402
    ROI_ORDER,
    build_roi_masks,
    conf_valid_mask,
    depth_to_camera_points,
    load_prediction_bundle,
    load_scene_view,
    normalize_target_view,
    normalize_vectors,
    parse_entry_spec,
)
from vggt.utils.normal_refiner import (  # noqa: E402
    extract_coarse_prior_normal,
    point_map_to_normal_numpy,
    points_world_to_camera,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit predicted normal sign conventions against depth-derived, "
            "point-derived, and optional SMPL-X coarse prior normals."
        )
    )
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Entry in name:predictions.npz:scene_dir form.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for audit outputs.")
    parser.add_argument(
        "--target-views",
        default="0",
        help="Comma-separated view indices, or 'all'. Defaults to 0.",
    )
    parser.add_argument(
        "--normal-format",
        choices=("auto", "vector", "rgb01", "rgb255"),
        default="auto",
        help="How to decode predicted normal arrays.",
    )
    return parser.parse_args()


def resolve_target_views(text: str, view_count: int) -> list[int]:
    if text.strip().lower() == "all":
        return list(range(view_count))
    views: list[int] = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        views.append(normalize_target_view(int(piece), view_count))
    if not views:
        raise ValueError("--target-views did not contain any valid view index")
    return views


def signed_stats(
    first: np.ndarray,
    first_valid: np.ndarray,
    second: np.ndarray,
    second_valid: np.ndarray,
    roi: np.ndarray,
) -> dict[str, Any]:
    valid = (
        np.asarray(first_valid, dtype=bool)
        & np.asarray(second_valid, dtype=bool)
        & np.asarray(roi, dtype=bool)
        & np.isfinite(first).all(axis=-1)
        & np.isfinite(second).all(axis=-1)
    )
    count = int(valid.sum())
    if count == 0:
        return {
            "valid_pixels": 0,
            "signed_cos_mean": None,
            "signed_cos_median": None,
            "signed_positive_frac": None,
            "signed_negative_frac": None,
            "abs_angle_mean_deg": None,
            "signed_angle_mean_deg": None,
        }
    cos = np.sum(first[valid] * second[valid], axis=-1).astype(np.float32)
    cos = np.clip(cos, -1.0, 1.0)
    abs_angle = np.degrees(np.arccos(np.abs(cos)))
    signed_angle = np.degrees(np.arccos(cos))
    return {
        "valid_pixels": count,
        "signed_cos_mean": float(np.mean(cos)),
        "signed_cos_median": float(np.median(cos)),
        "signed_positive_frac": float(np.mean(cos > 0.0)),
        "signed_negative_frac": float(np.mean(cos < 0.0)),
        "abs_angle_mean_deg": float(np.mean(abs_angle)),
        "signed_angle_mean_deg": float(np.mean(signed_angle)),
    }


def camera_ray_stats(
    normal: np.ndarray,
    normal_valid: np.ndarray,
    camera_points: np.ndarray,
    roi: np.ndarray,
) -> dict[str, Any]:
    rays, ray_valid = normalize_vectors(camera_points)
    stats = signed_stats(normal, normal_valid, rays, ray_valid, roi)
    stats["interpretation"] = "positive means normal points roughly along camera ray; negative means view-facing"
    return stats


def load_prior_normals(scene_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    prior_path = scene_dir / "prior_maps.npz"
    if not prior_path.is_file():
        return None, None, None
    with np.load(prior_path, allow_pickle=True) as payload:
        if "prior_maps" not in payload.files or "prior_channels" not in payload.files:
            return None, None, str(prior_path)
        normals, visible = extract_coarse_prior_normal(payload["prior_maps"], payload["prior_channels"])
    return normals.astype(np.float32, copy=False), visible.astype(bool, copy=False), str(prior_path)


def audit_entry(entry_text: str, output_dir: Path, target_views_arg: str, normal_format: str) -> list[dict[str, Any]]:
    spec = parse_entry_spec(entry_text)
    bundle = load_prediction_bundle(spec.predictions_npz, normal_format)
    view_count = int(bundle.normal.shape[0])
    target_views = resolve_target_views(target_views_arg, view_count)
    prior_normals, prior_visible, prior_path = load_prior_normals(spec.scene_dir)
    rows: list[dict[str, Any]] = []

    for view_idx in target_views:
        target_hw = tuple(int(value) for value in bundle.normal.shape[1:3])
        scene = load_scene_view(spec.scene_dir, view_idx, target_hw)
        support = scene.mask.astype(bool)
        roi_masks = build_roi_masks(support)

        pred_normal = bundle.normal[view_idx]
        pred_valid = bundle.normal_valid[view_idx] & support & conf_valid_mask(bundle.normal_conf[view_idx])

        depth_view = bundle.depth[view_idx]
        intrinsic = bundle.intrinsic[view_idx] if bundle.intrinsic is not None else np.eye(3, dtype=np.float32)
        depth_input_valid = (
            support
            & conf_valid_mask(bundle.depth_conf[view_idx])
            & np.isfinite(depth_view)
            & (depth_view > 0.0)
        )
        depth_camera_points = depth_to_camera_points(depth_view, intrinsic)
        depth_normal, depth_surface_valid = point_map_to_normal_numpy(depth_camera_points, depth_input_valid)
        depth_normal, depth_vec_valid = normalize_vectors(depth_normal)
        depth_valid = depth_surface_valid & depth_vec_valid & depth_input_valid

        world_points = bundle.world_points[view_idx]
        if bundle.extrinsic is None:
            point_camera = world_points.astype(np.float32, copy=False)
            point_camera_source = "assumed_camera"
        else:
            point_camera = points_world_to_camera(world_points, bundle.extrinsic[view_idx])
            point_camera_source = "world_to_camera"
        point_input_valid = (
            support
            & conf_valid_mask(bundle.world_points_conf[view_idx])
            & np.isfinite(point_camera).all(axis=-1)
        )
        point_normal, point_surface_valid = point_map_to_normal_numpy(point_camera, point_input_valid)
        point_normal, point_vec_valid = normalize_vectors(point_normal)
        point_valid = point_surface_valid & point_vec_valid & point_input_valid

        targets: list[tuple[str, np.ndarray, np.ndarray]] = [
            ("depth_normal_raw", depth_normal, depth_valid),
            ("depth_normal_flipped", -depth_normal, depth_valid),
            ("point_normal_raw", point_normal, point_valid),
            ("point_normal_flipped", -point_normal, point_valid),
        ]
        if prior_normals is not None and prior_visible is not None and view_idx < prior_normals.shape[0]:
            prior_normal, prior_vec_valid = normalize_vectors(prior_normals[view_idx])
            prior_valid = prior_visible[view_idx] & prior_vec_valid & support
            targets.extend(
                [
                    ("smplx_prior_raw", prior_normal, prior_valid),
                    ("smplx_prior_flipped", -prior_normal, prior_valid),
                ]
            )

        for roi_name in ROI_ORDER:
            roi = roi_masks[roi_name]
            for target_name, target_normal, target_valid in targets:
                stats = signed_stats(pred_normal, pred_valid, target_normal, target_valid, roi)
                rows.append(
                    {
                        "entry": spec.name,
                        "view": view_idx,
                        "roi": roi_name,
                        "comparison": f"pred_vs_{target_name}",
                        "pred_variant": "pred",
                        "target": target_name,
                        "point_camera_source": point_camera_source,
                        "prior_path": prior_path,
                        **stats,
                    }
                )
                flipped_stats = signed_stats(-pred_normal, pred_valid, target_normal, target_valid, roi)
                rows.append(
                    {
                        "entry": spec.name,
                        "view": view_idx,
                        "roi": roi_name,
                        "comparison": f"negpred_vs_{target_name}",
                        "pred_variant": "-pred",
                        "target": target_name,
                        "point_camera_source": point_camera_source,
                        "prior_path": prior_path,
                        **flipped_stats,
                    }
                )

            for normal_name, normal, valid in (
                ("pred", pred_normal, pred_valid),
                ("depth_normal_raw", depth_normal, depth_valid),
                ("point_normal_raw", point_normal, point_valid),
            ):
                stats = camera_ray_stats(normal, valid, depth_camera_points, roi)
                rows.append(
                    {
                        "entry": spec.name,
                        "view": view_idx,
                        "roi": roi_name,
                        "comparison": f"{normal_name}_vs_camera_ray",
                        "pred_variant": normal_name,
                        "target": "camera_ray",
                        "point_camera_source": point_camera_source,
                        "prior_path": prior_path,
                        **stats,
                    }
                )

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "entry",
        "view",
        "roi",
        "comparison",
        "pred_variant",
        "target",
        "valid_pixels",
        "signed_cos_mean",
        "signed_cos_median",
        "signed_positive_frac",
        "signed_negative_frac",
        "abs_angle_mean_deg",
        "signed_angle_mean_deg",
        "point_camera_source",
        "prior_path",
        "interpretation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    focus_rows = [
        row
        for row in rows
        if row["target"] in {"depth_normal_raw", "point_normal_raw", "smplx_prior_raw"}
        and row["pred_variant"] in {"pred", "-pred"}
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Normal sign convention audit\n\n")
        handle.write(
            "| Entry | View | ROI | Comparison | Valid | Mean signed cos | Negative frac | Mean abs angle deg |\n"
        )
        handle.write("|---|---:|---|---|---:|---:|---:|---:|\n")
        for row in focus_rows:
            handle.write(
                "| "
                + " | ".join(
                    [
                        str(row["entry"]),
                        str(row["view"]),
                        str(row["roi"]),
                        str(row["comparison"]),
                        str(row["valid_pixels"]),
                        format_value(row["signed_cos_mean"]),
                        format_value(row["signed_negative_frac"]),
                        format_value(row["abs_angle_mean_deg"]),
                    ]
                )
                + " |\n"
            )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for entry_text in args.entry:
        rows.extend(audit_entry(entry_text, output_dir, args.target_views, args.normal_format))

    json_path = output_dir / "normal_sign_convention_audit.json"
    csv_path = output_dir / "normal_sign_convention_audit.csv"
    markdown_path = output_dir / "normal_sign_convention_audit.md"
    summary = {
        "target_views": args.target_views,
        "normal_format": args.normal_format,
        "rows": rows,
        "outputs": {
            "json": str(json_path),
            "csv": str(csv_path),
            "markdown": str(markdown_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    write_markdown(markdown_path, rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
