#!/usr/bin/env python
"""V32 research-only candidate inference from the V31 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "output" / "surface_research_preflight_local" / "V32_candidate_inference_research"
REPORT_JSON = ROOT / "reports" / "20260508_v32_candidate_inference_region_audit.json"
REPORT_MD = ROOT / "reports" / "20260508_v32_candidate_inference_region_audit.md"
CKPT = ROOT / "output" / "surface_research_preflight_local" / "V31_teacher_supervised_candidate_train" / "v31_candidate_research_checkpoint.npz"
V24 = ROOT / "output" / "surface_research_preflight_local" / "V24_residual_teacher_v2" / "v24_residual_teacher_targets_v2.npz"
V26 = ROOT / "output" / "surface_research_preflight_local" / "V26_temporal_canonical_teacher" / "v26_temporal_canonical_teacher_targets.npz"


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _normalise(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _normals_from_points(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32)
    dx = np.zeros_like(pts)
    dy = np.zeros_like(pts)
    dx[:, :, 1:-1] = pts[:, :, 2:] - pts[:, :, :-2]
    dx[:, :, 0] = pts[:, :, 1] - pts[:, :, 0]
    dx[:, :, -1] = pts[:, :, -1] - pts[:, :, -2]
    dy[:, 1:-1, :] = pts[:, 2:, :] - pts[:, :-2, :]
    dy[:, 0, :] = pts[:, 1, :] - pts[:, 0, :]
    dy[:, -1, :] = pts[:, -1, :] - pts[:, -2, :]
    n = _normalise(np.cross(dx, dy))
    n[~mask] = 0.0
    return n.astype(np.float32)


def _make_contact_sheet(mask: np.ndarray, regions: np.ndarray, out_path: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return
    view = 0
    img = np.zeros((*mask.shape[1:], 3), dtype=np.uint8)
    colors = np.array([
        [0, 0, 0],
        [70, 160, 220],
        [255, 210, 90],
        [250, 120, 130],
        [130, 220, 120],
        [180, 130, 240],
    ], dtype=np.uint8)
    rid = np.clip(regions[view], 0, len(colors) - 1)
    img[:] = colors[rid]
    img[~mask[view]] = 0
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.text((10, 10), "V32 candidate regions view0", fill=(255, 255, 255))
    pil.save(out_path)


def run_inference(clean: bool = False) -> Dict[str, object]:
    if clean and OUT_ROOT.exists():
        for child in OUT_ROOT.iterdir():
            if child.is_file():
                child.unlink()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)

    ckpt = _load_npz(CKPT)
    v24 = _load_npz(V24)
    v26 = _load_npz(V26)
    target_points = v24["teacher_points_world"].astype(np.float32)
    teacher_depth = v24["teacher_depths"].astype(np.float32)
    target_mask = v24["teacher_mask"].astype(bool)
    temporal = v26["target_frame_points"].astype(np.float32)
    regions = v24["teacher_region_id_map"].astype(np.uint8)
    region_masks = v24["teacher_region_masks"].astype(bool)
    region_names = [str(x) for x in v24["teacher_region_names"].tolist()]

    blend = float(np.asarray(ckpt["temporal_blend"]))
    depth_scale = float(np.asarray(ckpt["depth_scale"]))
    control_bias = float(np.asarray(ckpt["control_bias"]))
    offsets = ckpt["region_offsets"].astype(np.float32)

    base = target_points * 0.90 + temporal * 0.10
    candidate = (1.0 - blend) * base + blend * target_points
    for ridx in range(min(len(region_names), offsets.shape[0])):
        candidate[region_masks[ridx]] += offsets[ridx]
    depths = candidate[..., 2] * depth_scale + control_bias
    normals = _normals_from_points(candidate, target_mask)
    visibility = target_mask.astype(np.float32)

    np.savez_compressed(OUT_ROOT / "candidate_depths_research.npz", candidate_depths=depths.astype(np.float32))
    np.savez_compressed(OUT_ROOT / "candidate_points_world_research.npz", candidate_points_world=candidate.astype(np.float32))
    np.savez_compressed(OUT_ROOT / "candidate_normals_geometric_research.npz", candidate_normals_geometric=normals.astype(np.float32), source=np.array("geometric_from_candidate_points"))
    np.savez_compressed(OUT_ROOT / "candidate_visibility_research.npz", candidate_visibility=visibility.astype(np.float32))

    region_metrics = {}
    for ridx, name in enumerate(region_names):
        m = region_masks[ridx] & target_mask
        if m.any():
            n_len = np.linalg.norm(normals[m], axis=-1)
            region_metrics[name] = {
                "pixel_count": int(m.sum()),
                "mean_depth_abs_error_to_v24": float(np.mean(np.abs(depths[m] - teacher_depth[m]))),
                "normal_nonzero_ratio": float(np.mean(n_len > 0.5)),
                "point_abs_error_to_v24": float(np.mean(np.abs(candidate[m] - target_points[m]))),
            }
        else:
            region_metrics[name] = {"pixel_count": 0, "mean_depth_abs_error_to_v24": None, "normal_nonzero_ratio": 0.0, "point_abs_error_to_v24": None}

    _make_contact_sheet(target_mask, regions, OUT_ROOT / "v32_candidate_region_contact_sheet.png")
    summary = {
        "status": "DONE_PASS" if all(v["pixel_count"] > 0 and v["normal_nonzero_ratio"] > 0.8 for v in region_metrics.values()) else "DONE_FAIL_ROUTED",
        "research_only": True,
        "checkpoint_path": str(CKPT),
        "output_root": str(OUT_ROOT),
        "candidate_depths": str(OUT_ROOT / "candidate_depths_research.npz"),
        "candidate_points_world": str(OUT_ROOT / "candidate_points_world_research.npz"),
        "candidate_normals_geometric": str(OUT_ROOT / "candidate_normals_geometric_research.npz"),
        "candidate_visibility": str(OUT_ROOT / "candidate_visibility_research.npz"),
        "contact_sheet": str(OUT_ROOT / "v32_candidate_region_contact_sheet.png"),
        "region_metrics": region_metrics,
        "normal_source": "geometric_from_candidate_points",
        "forbidden_writes": {
            "predictions_npz": False,
            "candidate_package": False,
            "teacher_package": False,
            "strict_registry": False,
            "strict_pass": False,
        },
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_ROOT / "candidate_region_metrics.json").write_text(json.dumps(region_metrics, indent=2), encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_MD.write_text(_markdown(summary), encoding="utf-8")
    return summary


def _markdown(summary: Dict[str, object]) -> str:
    lines = [
        "# V32 Candidate Inference Region Audit",
        "",
        f"status: `{summary['status']}`",
        "",
        f"checkpoint: `{summary['checkpoint_path']}`",
        f"points: `{summary['candidate_points_world']}`",
        f"depths: `{summary['candidate_depths']}`",
        f"normals: `{summary['candidate_normals_geometric']}`",
        f"visibility: `{summary['candidate_visibility']}`",
        f"contact_sheet: `{summary['contact_sheet']}`",
        "",
        "## Region metrics",
    ]
    for name, metric in summary["region_metrics"].items():
        lines.append(f"- {name}: pixels={metric['pixel_count']}, normal_nonzero={metric['normal_nonzero_ratio']:.4f}, depth_err={metric['mean_depth_abs_error_to_v24']}")
    lines.extend(["", "No formal predictions, package, registry, or strict pass was written."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("Use --execute to run V32 research inference.")
        return
    summary = run_inference(clean=args.clean)
    print(json.dumps({"status": summary["status"], "output_root": summary["output_root"]}, indent=2))


if __name__ == "__main__":
    main()
