from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "surface_research_preflight_local" / "V34_smplx_native_hand_route"
REPORT_JSON = ROOT / "reports" / "20260508_v34_smplx_native_hand_route.json"
REPORT_MD = ROOT / "reports" / "20260508_v34_smplx_native_hand_route.md"

V16_ROI = ROOT / "output" / "surface_research_preflight_local" / "V16_smplx_native_region_roi_builder" / "v16_smplx_native_region_roi_maps.npz"
V24_TARGETS = ROOT / "output" / "surface_research_preflight_local" / "V24_residual_teacher_v2" / "v24_residual_teacher_targets_v2.npz"
V26_TARGETS = ROOT / "output" / "surface_research_preflight_local" / "V26_temporal_canonical_teacher" / "v26_temporal_canonical_teacher_targets.npz"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _roi(roi_npz: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    names = [str(x) for x in roi_npz["roi_names"]]
    return roi_npz["roi_maps"][:, names.index(name)].astype(bool)


def _stats(vals: np.ndarray) -> dict:
    vals = np.asarray(vals)
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


def _geom_normals(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(points, dtype=np.float32)
    for view in range(points.shape[0]):
        p = points[view]
        dx = np.zeros_like(p)
        dy = np.zeros_like(p)
        dx[:, 1:-1] = p[:, 2:] - p[:, :-2]
        dy[1:-1, :] = p[2:, :] - p[:-2, :]
        n = _normalize(np.cross(dx, dy))
        n[~mask[view]] = 0
        normals[view] = n
    return normals


def _region_metrics(mask: np.ndarray, points: np.ndarray, bridge_mask: np.ndarray, finger_masks: dict[str, np.ndarray]) -> dict:
    pts = points[mask]
    bridge_pts = points[bridge_mask]
    if pts.size and bridge_pts.size:
        # Cheap connectivity proxy: nearest centroid distance between hand and wrist bridge per view.
        distances = []
        for v in range(mask.shape[0]):
            hm = mask[v]
            bm = bridge_mask[v]
            if hm.any() and bm.any():
                distances.append(float(np.linalg.norm(points[v][hm].mean(axis=0) - points[v][bm].mean(axis=0))))
        wrist_distance = _stats(np.asarray(distances, dtype=np.float32))
    else:
        wrist_distance = {"count": 0, "finite": 0}
    finger_pixels = {name: int(m.sum()) for name, m in finger_masks.items()}
    return {
        "pixels": int(mask.sum()),
        "views_with_pixels": int(np.count_nonzero(mask.reshape(mask.shape[0], -1).sum(axis=1) > 0)),
        "per_view_pixels": [int(x) for x in mask.reshape(mask.shape[0], -1).sum(axis=1)],
        "wrist_bridge_pixels": int(bridge_mask.sum()),
        "finger_pixels": finger_pixels,
        "finger_bands_nonempty": int(sum(1 for x in finger_pixels.values() if x > 0)),
        "hand_to_wrist_centroid_distance": wrist_distance,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    roi = np.load(V16_ROI, allow_pickle=True)
    v24 = np.load(V24_TARGETS, allow_pickle=True)
    v26 = np.load(V26_TARGETS, allow_pickle=True)

    raw_mask = v24["raw_mask"].astype(bool)
    teacher_points = v24["teacher_points_world"].astype(np.float32)
    teacher_normals = v24["teacher_normals_world"].astype(np.float32)
    temporal_points = v26["target_frame_points"].astype(np.float32)
    temporal_conf = v26["temporal_confidence"].astype(np.float32)

    side_specs = {
        "left": {
            "hand": "left_hand",
            "bridge": "wrist_bridge_left",
            "fingers": ["thumb_left", "index_left", "middle_left", "ring_left", "pinky_left"],
            "rid": 1,
        },
        "right": {
            "hand": "right_hand",
            "bridge": "wrist_bridge_right",
            "fingers": ["thumb_right", "index_right", "middle_right", "ring_right", "pinky_right"],
            "rid": 2,
        },
    }

    output_points = np.zeros_like(teacher_points, dtype=np.float32)
    output_normals = np.zeros_like(teacher_normals, dtype=np.float32)
    output_region = np.zeros_like(v24["teacher_region_id_map"], dtype=np.uint8)
    output_visibility = np.zeros_like(v24["teacher_visibility"], dtype=np.float32)
    metrics = {}

    for side, spec in side_specs.items():
        hand = _roi(roi, spec["hand"]) & raw_mask
        bridge = _roi(roi, spec["bridge"]) & raw_mask
        fingers = {name: (_roi(roi, name) & raw_mask) for name in spec["fingers"]}
        support = hand | bridge
        conf = np.clip(temporal_conf, 0.0, 1.0)[..., None]
        blended = teacher_points * (1.0 - 0.30 * conf) + temporal_points * (0.30 * conf)
        geom_normals = _geom_normals(blended, support)
        normals = _normalize(0.65 * teacher_normals + 0.35 * geom_normals)
        output_points[support] = blended[support]
        output_normals[support] = normals[support]
        output_region[support] = spec["rid"]
        output_visibility[support] = np.maximum(v24["teacher_visibility"][support], temporal_conf[support])
        metrics[side] = _region_metrics(hand, output_points, bridge, fingers)

    np.savez_compressed(
        OUT_DIR / "v34_smplx_native_hand_continuity_patch.npz",
        hand_points_world=output_points,
        hand_normals_world=output_normals,
        hand_visibility=output_visibility,
        hand_region_id_map=output_region,
        research_only=np.array(True),
        smplx_native_only=np.array(True),
        no_mano=np.array(True),
    )

    pts = output_points[output_region > 0]
    regs = output_region[output_region > 0]
    colors = np.zeros((pts.shape[0], 3), dtype=np.uint8)
    colors[regs == 1] = np.array([80, 255, 130], dtype=np.uint8)
    colors[regs == 2] = np.array([255, 80, 130], dtype=np.uint8)
    ply = OUT_DIR / "v34_smplx_native_hand_continuity_patch.ply"
    with ply.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(pts, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

    pass_flags = {
        side: bool(m["pixels"] > 0 and m["wrist_bridge_pixels"] > 0 and m["finger_bands_nonempty"] >= 3)
        for side, m in metrics.items()
    }
    status = "DONE_PASS" if all(pass_flags.values()) else "DONE_FAIL_ROUTED"
    summary = {
        "task": "v34_smplx_native_hand_continuity_refiner",
        "status": status,
        "created_utc": _now(),
        "research_only": True,
        "smplx_native_only": True,
        "no_mano": True,
        "no_candidate_export": True,
        "no_teacher_export": True,
        "no_predictions_write": True,
        "no_registry_write": True,
        "metrics": metrics,
        "pass_flags": pass_flags,
        "outputs": {
            "hand_patch_npz": str(OUT_DIR / "v34_smplx_native_hand_continuity_patch.npz"),
            "hand_patch_ply": str(ply),
            "summary_json": str(OUT_DIR / "summary.json"),
            "report_json": str(REPORT_JSON),
            "report_md": str(REPORT_MD),
        },
        "blockers": [] if status == "DONE_PASS" else ["left_or_right_hand_continuity_support_failed"],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_MD.write_text(
        "\n".join(
            [
                "# V34 SMPL-X Native Hand Continuity Route",
                "",
                f"Status: `{status}`",
                "",
                f"- Left hand pixels: {metrics['left']['pixels']}, wrist bridge pixels: {metrics['left']['wrist_bridge_pixels']}, finger bands nonempty: {metrics['left']['finger_bands_nonempty']}",
                f"- Right hand pixels: {metrics['right']['pixels']}, wrist bridge pixels: {metrics['right']['wrist_bridge_pixels']}, finger bands nonempty: {metrics['right']['finger_bands_nonempty']}",
                f"- Hand patch PLY: `{ply}`",
                "",
                "No MANO/HaMeR/WiLoR or formal output was used.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "report": str(REPORT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
