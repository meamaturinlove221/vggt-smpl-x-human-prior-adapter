from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from audit_fullbody_hand_integrity import (  # noqa: E402
    connected_component_stats_2d,
    create_hand_landmarker,
    hand_risk_mask,
    mediapipe_hand_mask,
    point_box_spatial_stats,
)
from normal_line_multiview_eval import build_roi_masks, load_scene_view  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit scene-local SMPL-X prior maps as a weak full-body / hands anchor. "
            "This is a preflight gate only; it does not patch predictions or create a candidate."
        )
    )
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-body-views", type=int, default=4)
    parser.add_argument("--min-body-visible-ratio", type=float, default=0.40)
    parser.add_argument("--min-body-band-visible-ratio", type=float, default=0.03)
    parser.add_argument("--max-body-components", type=int, default=4)
    parser.add_argument("--min-body-largest-component-ratio", type=float, default=0.82)
    parser.add_argument("--min-hand-components", type=int, default=3)
    parser.add_argument("--min-hand-visible-ratio", type=float, default=0.25)
    parser.add_argument("--min-hand-visible-pixels", type=int, default=48)
    parser.add_argument("--min-hand-largest-component-ratio", type=float, default=0.30)
    parser.add_argument("--max-hand-kept-components", type=int, default=4)
    parser.add_argument("--max-hand-support-body-ratio", type=float, default=0.22)
    parser.add_argument("--max-hand-box-3d-extent", type=float, default=0.30)
    parser.add_argument("--max-hand-box-depth-range", type=float, default=0.18)
    parser.add_argument(
        "--hand-landmarker-model",
        default=str(REPO_ROOT / "external_models" / "hand_landmarker.task"),
    )
    parser.add_argument("--disable-mediapipe-hands", action="store_true")
    parser.add_argument("--hand-box-pad", type=int, default=24)
    parser.add_argument("--vertical-bins", type=int, default=10)
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_smplx_prior(scene_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    prior_path = scene_dir / "prior_maps.npz"
    if not prior_path.is_file():
        raise FileNotFoundError(prior_path)
    prior = np.load(prior_path, allow_pickle=True)
    prior_maps = np.asarray(prior["prior_maps"], dtype=np.float32)
    prior_mask = np.asarray(prior["prior_mask"], dtype=bool)
    channels = [str(item) for item in prior["prior_channels"]]
    channel_index = {name: idx for idx, name in enumerate(channels)}
    required = [
        "smplx_posed_cam_x",
        "smplx_posed_cam_y",
        "smplx_posed_cam_z",
        "smplx_visible_mask",
    ]
    missing = [name for name in required if name not in channel_index]
    if missing:
        raise KeyError(f"Missing SMPL-X prior channels: {missing}")
    smplx_cam = np.stack(
        [
            prior_maps[:, channel_index["smplx_posed_cam_x"]],
            prior_maps[:, channel_index["smplx_posed_cam_y"]],
            prior_maps[:, channel_index["smplx_posed_cam_z"]],
        ],
        axis=-1,
    ).astype(np.float32)
    visible = (
        prior_mask
        & (prior_maps[:, channel_index["smplx_visible_mask"]] > 0.5)
        & np.isfinite(smplx_cam).all(axis=-1)
        & (smplx_cam[..., 2] > 0.0)
    )
    return smplx_cam, visible, channels


def save_overlay(path: Path, rgb: np.ndarray, body: np.ndarray, visible: np.ndarray, hand: np.ndarray) -> None:
    out = rgb.astype(np.float32).copy()
    body_only = body & ~visible
    out[body_only] = out[body_only] * 0.68 + np.array([40, 90, 255], dtype=np.float32) * 0.32
    out[visible] = out[visible] * 0.45 + np.array([0, 220, 80], dtype=np.float32) * 0.55
    out[hand] = out[hand] * 0.35 + np.array([255, 120, 20], dtype=np.float32) * 0.65
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(path)


def vertical_band_stats(mask: np.ndarray, visible: np.ndarray, bins: int) -> tuple[list[dict[str, Any]], float]:
    y_values = np.nonzero(mask)[0]
    if y_values.size == 0:
        return [], 0.0
    y0, y1 = int(y_values.min()), int(y_values.max())
    edges = np.linspace(y0, y1 + 1, int(bins) + 1).astype(int)
    rows: list[dict[str, Any]] = []
    ratios: list[float] = []
    for band_idx in range(int(bins)):
        band = mask.copy()
        band[: int(edges[band_idx]), :] = False
        band[int(edges[band_idx + 1]) :, :] = False
        pixels = int(band.sum())
        visible_pixels = int((visible & band).sum())
        ratio = float(visible_pixels / max(pixels, 1)) if pixels else 0.0
        if pixels:
            ratios.append(ratio)
        rows.append(
            {
                "band": int(band_idx),
                "pixels": pixels,
                "visible_pixels": visible_pixels,
                "visible_ratio": ratio,
            }
        )
    return rows, float(min(ratios)) if ratios else 0.0


def main() -> int:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    smplx_cam, smplx_visible, channels = load_smplx_prior(scene_dir)
    view_count, height, width, _ = smplx_cam.shape
    detector = create_hand_landmarker(Path(args.hand_landmarker_model), bool(args.disable_mediapipe_hands))

    per_view: dict[str, Any] = {}
    body_views_ok = 0
    hand_views_ok = 0
    hand_candidate_views = 0
    compact_hand_box_views = 0
    implausible_hand_boxes = 0
    for view_idx in range(view_count):
        scene = load_scene_view(scene_dir, view_idx, (height, width))
        body = scene.mask.astype(bool)
        visible = smplx_visible[view_idx] & body
        rois = build_roi_masks(body)
        mp_hand = mediapipe_hand_mask(
            scene.rgb,
            body,
            detector,
            pad=int(args.hand_box_pad),
        )
        if mp_hand is None:
            hand_mask, hand_summary = hand_risk_mask(scene.rgb, body)
        else:
            hand_mask, hand_summary = mp_hand
        hand_visible = visible & hand_mask
        hand_support_pixels = int(hand_mask.sum())
        hand_visible_pixels = int(hand_visible.sum())
        hand_visible_ratio = float(hand_visible_pixels / max(hand_support_pixels, 1)) if hand_support_pixels else 0.0
        hand_support_body_ratio = float(hand_support_pixels / max(int(body.sum()), 1)) if hand_support_pixels else 0.0
        hand_components = connected_component_stats_2d(hand_visible)
        blocked = rois["head"] | rois["face"]
        hand_box_stats: list[dict[str, Any]] = []
        for box in hand_summary.get("boxes_xyxy", []) or []:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            x0, y0, x1, y1 = [int(value) for value in box]
            x0 = max(0, min(width, x0))
            x1 = max(0, min(width, x1))
            y0 = max(0, min(height, y0))
            y1 = max(0, min(height, y1))
            box_mask = np.zeros_like(body, dtype=bool)
            if x1 > x0 and y1 > y0:
                box_mask[y0:y1, x0:x1] = True
            box_visible = visible & box_mask & body & ~blocked
            spatial = point_box_spatial_stats(smplx_cam[view_idx][box_visible])
            box_ok = bool(
                int(spatial["points"]) >= int(args.min_hand_visible_pixels)
                and float(spatial["max_extent"]) <= float(args.max_hand_box_3d_extent)
                and float(spatial["depth_range"]) <= float(args.max_hand_box_depth_range)
            )
            hand_box_stats.append(
                {
                    "box_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                    "visible_pixels": int(box_visible.sum()),
                    "spatial": spatial,
                    "gate_ok": box_ok,
                }
            )
        hand_box_gate_ok = bool(
            hand_summary.get("roi_source") == "mediapipe_hand_landmarker"
            and hand_box_stats
            and all(row.get("gate_ok") for row in hand_box_stats)
        )
        hand_ok = bool(
            hand_support_pixels > 0
            and hand_summary.get("roi_source") == "mediapipe_hand_landmarker"
            and hand_visible_ratio >= float(args.min_hand_visible_ratio)
            and hand_visible_pixels >= int(args.min_hand_visible_pixels)
            and float(hand_components.get("largest_component_ratio", 0.0))
            >= float(args.min_hand_largest_component_ratio)
            and int(hand_components.get("components", 0)) <= int(args.max_hand_kept_components)
            and hand_support_body_ratio <= float(args.max_hand_support_body_ratio)
            and hand_box_gate_ok
        )
        hand_candidate_views += int(hand_support_pixels > 0)
        hand_views_ok += int(hand_ok)
        compact_hand_box_views += int(hand_box_gate_ok)
        implausible_hand_boxes += int(sum(1 for row in hand_box_stats if not row.get("gate_ok")))

        body_components = connected_component_stats_2d(visible)
        bands, min_band_visible = vertical_band_stats(body, visible, int(args.vertical_bins))
        body_visible_ratio = float(visible.sum() / max(int(body.sum()), 1))
        body_ok = bool(
            body_visible_ratio >= float(args.min_body_visible_ratio)
            and min_band_visible >= float(args.min_body_band_visible_ratio)
            and int(body_components.get("components", 0)) <= int(args.max_body_components)
            and float(body_components.get("largest_component_ratio", 0.0))
            >= float(args.min_body_largest_component_ratio)
        )
        body_views_ok += int(body_ok)

        save_overlay(output_dir / f"view_{view_idx:02d}_smplx_weak_anchor_overlay.png", scene.rgb, body, visible, hand_mask)
        per_view[str(view_idx)] = {
            "body_pixels": int(body.sum()),
            "smplx_visible_body_pixels": int(visible.sum()),
            "body_visible_ratio": body_visible_ratio,
            "body_components": body_components,
            "body_min_vertical_band_visible_ratio_2d": min_band_visible,
            "body_gate_ok": body_ok,
            "head_visible_ratio": float((visible & rois["head"]).sum() / max(int(rois["head"].sum()), 1)),
            "face_visible_ratio": float((visible & rois["face"]).sum() / max(int(rois["face"].sum()), 1)),
            "vertical_bands": bands,
            "hand_risk": {
                **hand_summary,
                "visible_pixels": hand_visible_pixels,
                "visible_ratio": hand_visible_ratio,
                "support_body_ratio": hand_support_body_ratio,
                "visible_components": hand_components,
                "hand_box_3d": {
                    "boxes": hand_box_stats,
                    "boxes_ok": hand_box_gate_ok,
                    "implausible_boxes": int(sum(1 for row in hand_box_stats if not row.get("gate_ok"))),
                },
                "gate_ok": hand_ok,
            },
            "overlay": str(output_dir / f"view_{view_idx:02d}_smplx_weak_anchor_overlay.png"),
        }

    if detector is not None:
        detector.close()

    body_gate = {
        "views_passing_body_anchor": int(body_views_ok),
        "min_body_views": int(args.min_body_views),
        "min_body_visible_ratio": float(args.min_body_visible_ratio),
        "min_body_band_visible_ratio": float(args.min_body_band_visible_ratio),
        "max_body_components": int(args.max_body_components),
        "min_body_largest_component_ratio": float(args.min_body_largest_component_ratio),
        "pass": bool(body_views_ok >= int(args.min_body_views)),
    }
    hand_gate = {
        "eligible_views_with_hand_candidates": int(hand_candidate_views),
        "views_passing_hand_anchor": int(hand_views_ok),
        "views_with_compact_3d_hand_boxes": int(compact_hand_box_views),
        "implausible_hand_boxes": int(implausible_hand_boxes),
        "min_hand_components": int(args.min_hand_components),
        "min_hand_visible_ratio": float(args.min_hand_visible_ratio),
        "min_hand_visible_pixels": int(args.min_hand_visible_pixels),
        "max_hand_box_3d_extent": float(args.max_hand_box_3d_extent),
        "max_hand_box_depth_range": float(args.max_hand_box_depth_range),
        "pass": bool(hand_views_ok >= int(args.min_hand_components)),
    }
    summary = {
        "task": "smplx_weak_anchor_preflight",
        "truthful_status": "preflight_only_not_candidate_not_face_teacher",
        "scene_dir": str(scene_dir),
        "prior_maps": str((scene_dir / "prior_maps.npz").resolve()),
        "output_dir": str(output_dir),
        "view_count": int(view_count),
        "channels": channels,
        "body_gate": body_gate,
        "hand_gate": hand_gate,
        "pass": bool(body_gate["pass"] and hand_gate["pass"]),
        "per_view": per_view,
        "notes": [
            "SMPL-X is allowed only as weak full-body / hands topology anchor.",
            "This preflight does not approve SMPL-X as face, hair, clothing, or skirt teacher.",
            "A pass here is necessary but not sufficient for training; final candidates still need strict Open3D full/head/face/hands gate.",
        ],
    }
    (output_dir / "smplx_weak_anchor_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
