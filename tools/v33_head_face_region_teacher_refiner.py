from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "surface_research_preflight_local" / "V33_head_face_detail_route"
REPORT_JSON = ROOT / "reports" / "20260508_v33_head_face_detail_route.json"
REPORT_MD = ROOT / "reports" / "20260508_v33_head_face_detail_route.md"

V16_ROI = ROOT / "output" / "surface_research_preflight_local" / "V16_smplx_native_region_roi_builder" / "v16_smplx_native_region_roi_maps.npz"
V24_TARGETS = ROOT / "output" / "surface_research_preflight_local" / "V24_residual_teacher_v2" / "v24_residual_teacher_targets_v2.npz"
V26_TARGETS = ROOT / "output" / "surface_research_preflight_local" / "V26_temporal_canonical_teacher" / "v26_temporal_canonical_teacher_targets.npz"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stats(values: np.ndarray) -> dict:
    vals = np.asarray(values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"count": 0, "finite": 0}
    return {
        "count": int(vals.size),
        "finite": int(vals.size),
        "min": float(vals.min()),
        "median": float(np.median(vals)),
        "mean": float(vals.mean()),
        "p95": float(np.percentile(vals, 95)),
        "max": float(vals.max()),
    }


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, np.maximum(n, 1e-8), out=np.zeros_like(v, dtype=np.float32), where=n > 1e-8)


def _geometric_normals(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    normals = np.zeros_like(pts, dtype=np.float32)
    for view in range(pts.shape[0]):
        p = pts[view]
        valid = mask[view]
        dx = np.zeros_like(p)
        dy = np.zeros_like(p)
        dx[:, 1:-1] = p[:, 2:] - p[:, :-2]
        dy[1:-1, :] = p[2:, :] - p[:-2, :]
        n = np.cross(dx, dy)
        n = _normalize(n)
        n[~valid] = 0
        normals[view] = n
    return normals


def _roi_map(roi_npz: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    names = [str(x) for x in roi_npz["roi_names"]]
    if name not in names:
        raise KeyError(f"Missing ROI {name}. Available: {names}")
    return roi_npz["roi_maps"][:, names.index(name)].astype(bool)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)

    roi = np.load(V16_ROI, allow_pickle=True)
    v24 = np.load(V24_TARGETS, allow_pickle=True)
    v26 = np.load(V26_TARGETS, allow_pickle=True)

    teacher_points = v24["teacher_points_world"].astype(np.float32)
    teacher_normals = v24["teacher_normals_world"].astype(np.float32)
    teacher_visibility = v24["teacher_visibility"].astype(np.float32)
    teacher_uncertainty = v24["teacher_uncertainty"].astype(np.float32)
    raw_mask = v24["raw_mask"].astype(bool)

    temporal_points = v26["target_frame_points"].astype(np.float32)
    temporal_conf = v26["temporal_confidence"].astype(np.float32)

    head_roi = _roi_map(roi, "head")
    face_front = _roi_map(roi, "face_front")
    face_static = _roi_map(roi, "face_lmk_static")
    face_dynamic = _roi_map(roi, "face_lmk_dynamic")
    face_roi = face_front | face_static | face_dynamic

    region_masks = {
        "head": head_roi & raw_mask,
        "face": face_roi & raw_mask,
    }

    refined_points = np.zeros_like(teacher_points, dtype=np.float32)
    refined_normals = np.zeros_like(teacher_normals, dtype=np.float32)
    refined_visibility = np.zeros_like(teacher_visibility, dtype=np.float32)
    refined_uncertainty = np.ones_like(teacher_uncertainty, dtype=np.float32)
    refined_region_id = np.zeros_like(v24["teacher_region_id_map"], dtype=np.uint8)

    geom_temporal_normals = _geometric_normals(temporal_points, region_masks["head"] | region_masks["face"])

    region_ids = {"head": 1, "face": 2}
    coverage = {}
    for name, mask in region_masks.items():
        # Blend V24 residual teacher with V26 temporal target where available. This is still
        # research-only evidence; the temporal confidence is used only to downweight uncertainty.
        conf = np.clip(temporal_conf, 0.0, 1.0)[..., None]
        blended = (teacher_points * (1.0 - 0.35 * conf)) + (temporal_points * (0.35 * conf))
        blended_normals = _normalize((0.70 * teacher_normals) + (0.30 * geom_temporal_normals))
        refined_points[mask] = blended[mask]
        refined_normals[mask] = blended_normals[mask]
        refined_visibility[mask] = np.maximum(teacher_visibility[mask], temporal_conf[mask])
        refined_uncertainty[mask] = np.minimum(teacher_uncertainty[mask], 1.0 - np.clip(temporal_conf[mask] * 0.5, 0.0, 0.5))
        refined_region_id[mask] = region_ids[name]
        normal_len = np.linalg.norm(refined_normals[mask], axis=-1) if int(mask.sum()) else np.array([])
        relief = np.linalg.norm((refined_points - teacher_points)[mask], axis=-1) if int(mask.sum()) else np.array([])
        coverage[name] = {
            "pixels": int(mask.sum()),
            "views_with_pixels": int(np.count_nonzero(mask.reshape(mask.shape[0], -1).sum(axis=1) > 0)),
            "per_view_pixels": [int(x) for x in mask.reshape(mask.shape[0], -1).sum(axis=1)],
            "normal_length": _stats(normal_len),
            "temporal_relief_norm": _stats(relief),
        }

    np.savez_compressed(
        OUT_DIR / "v33_head_face_refined_teacher.npz",
        refined_points_world=refined_points,
        refined_normals_world=refined_normals,
        refined_visibility=refined_visibility,
        refined_uncertainty=refined_uncertainty,
        refined_region_id_map=refined_region_id,
        head_mask=region_masks["head"].astype(np.uint8),
        face_mask=region_masks["face"].astype(np.uint8),
        source=np.array("smplx_native_v16_roi_plus_v24_v26_research_teacher"),
        research_only=np.array(True),
        no_flame=np.array(True),
        no_formal_outputs=np.array(True),
    )

    # Lightweight PLY for visual inspection.
    pts = refined_points[refined_region_id > 0]
    regs = refined_region_id[refined_region_id > 0]
    colors = np.zeros((pts.shape[0], 3), dtype=np.uint8)
    colors[regs == 1] = np.array([80, 180, 255], dtype=np.uint8)
    colors[regs == 2] = np.array([255, 170, 80], dtype=np.uint8)
    ply_path = OUT_DIR / "v33_head_face_refined_teacher_points.ply"
    with ply_path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(pts, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

    status = "DONE_PASS" if coverage["head"]["pixels"] > 0 and coverage["face"]["pixels"] > 0 else "DONE_FAIL_ROUTED"
    summary = {
        "task": "v33_head_face_region_teacher_refiner",
        "status": status,
        "created_utc": _now(),
        "research_only": True,
        "smplx_native_only": True,
        "no_flame": True,
        "no_candidate_export": True,
        "no_teacher_export": True,
        "no_predictions_write": True,
        "no_registry_write": True,
        "inputs": {
            "v16_roi": str(V16_ROI),
            "v24_targets": str(V24_TARGETS),
            "v26_targets": str(V26_TARGETS),
        },
        "outputs": {
            "refined_npz": str(OUT_DIR / "v33_head_face_refined_teacher.npz"),
            "refined_ply": str(ply_path),
            "summary_json": str(OUT_DIR / "summary.json"),
            "report_json": str(REPORT_JSON),
            "report_md": str(REPORT_MD),
        },
        "metrics": {
            "coverage": coverage,
            "total_refined_pixels": int((refined_region_id > 0).sum()),
            "normal_support_pixels": int(np.count_nonzero(np.linalg.norm(refined_normals, axis=-1) > 0.5)),
        },
        "blockers": [] if status == "DONE_PASS" else ["head_or_face_support_empty"],
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_MD.write_text(
        "\n".join(
            [
                "# V33 Head/Face SMPL-X Native Detail Route",
                "",
                f"Status: `{status}`",
                "",
                "Research-only head/face refinement using V16 SMPL-X native ROI maps plus V24/V26 residual teacher evidence. No FLAME or external face model is used.",
                "",
                f"- Head pixels: {coverage['head']['pixels']}",
                f"- Face pixels: {coverage['face']['pixels']}",
                f"- Refined PLY: `{ply_path}`",
                f"- Refined NPZ: `{OUT_DIR / 'v33_head_face_refined_teacher.npz'}`",
                "",
                "No formal candidate, teacher package, predictions.npz, strict registry, or strict pass was written.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "report": str(REPORT_JSON), "output": str(OUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
