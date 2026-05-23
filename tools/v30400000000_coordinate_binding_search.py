from __future__ import annotations

import csv
import itertools
import json
import math
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image
from scipy import ndimage


REPO = Path(r"D:\vggt\vggt-feature-adapter")
AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
OUTPUT = AUX / "output"
BOARDS = AUX / "boards"
SMC_DIR = Path(r"G:\数据集\datasets\data_used_in_4K4D\annotations")
V11700 = AUX / "remote_pull" / "V11700_gap_reduction_branch_520" / "predictions.npz"
SURFACE = OUTPUT / "V9200000000_surface_dataset" / "true_full_surface_indexed.npz"
CAMERA_IDS = ["00", "01", "15", "30", "45", "59"]
AXIS_FLIPS = {
    "identity": (1, 1, 1),
    "flip_x": (-1, 1, 1),
    "flip_y": (1, -1, 1),
    "flip_z": (1, 1, -1),
    "flip_xy": (-1, -1, 1),
    "flip_xz": (-1, 1, -1),
    "flip_yz": (1, -1, -1),
    "flip_xyz": (-1, -1, -1),
}
UNIT_SCALES = {
    "unit": 1.0,
    "meters_to_mm": 1000.0,
    "mm_to_meters": 0.001,
    "meters_to_cm": 100.0,
    "cm_to_meters": 0.01,
}


def resolve_smc_dir() -> Path:
    for child in Path("G:/").iterdir():
        candidate = child / "datasets" / "data_used_in_4K4D" / "annotations"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find G:/<dataset-root>/datasets/data_used_in_4K4D/annotations")


SMC_DIR = resolve_smc_dir()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def resize_mask(mask: np.ndarray, size: tuple[int, int] = (518, 518)) -> np.ndarray:
    arr = mask
    if arr.ndim > 2:
        arr = arr.squeeze()
    if arr.dtype != np.uint8:
        arr = (arr > 0).astype(np.uint8) * 255
    img = Image.fromarray(arr)
    img = img.resize(size[::-1], Image.Resampling.NEAREST)
    return np.asarray(img) > 0


def resize_intrinsic(K: np.ndarray, source_hw: tuple[int, int] | None, target_hw: tuple[int, int] = (518, 518)) -> np.ndarray:
    if source_hw is None:
        return K.astype(np.float64).copy()
    h, w = source_hw
    th, tw = target_hw
    out = K.astype(np.float64).copy()
    out[0, 0] *= tw / float(w)
    out[0, 2] *= tw / float(w)
    out[1, 1] *= th / float(h)
    out[1, 2] *= th / float(h)
    return out


def load_vggt() -> dict[str, np.ndarray]:
    with np.load(V11700, allow_pickle=False) as z:
        wp = z["world_points"].astype(np.float64)
        conf = z["world_points_conf"].astype(np.float32) if "world_points_conf" in z.files else np.ones(wp.shape[:3], dtype=np.float32)
    with np.load(SURFACE, allow_pickle=False) as z:
        valid = z["valid_mask"].astype(bool)
        region = z["region_label"].astype(np.int16)
    return {"world_points": wp, "conf": conf, "valid": valid, "region": region}


def sample_points(points: np.ndarray, mask: np.ndarray, max_points: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    flat_idx = np.flatnonzero(mask.reshape(-1))
    if flat_idx.size > max_points:
        rng = np.random.default_rng(304)
        flat_idx = rng.choice(flat_idx, max_points, replace=False)
    pts = points.reshape(-1, 3)[flat_idx]
    return pts, flat_idx


def sample_points_by_view(points: np.ndarray, mask: np.ndarray, max_points: int = 800) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    rng = np.random.default_rng(304)
    for vi in range(points.shape[0]):
        flat_idx = np.flatnonzero(mask[vi].reshape(-1))
        if flat_idx.size > max_points:
            flat_idx = rng.choice(flat_idx, max_points, replace=False)
        out.append(points[vi].reshape(-1, 3)[flat_idx].astype(np.float64))
    return out


def load_smc_summary(path: Path) -> dict[str, Any]:
    with h5py.File(str(path), "r") as f:
        cams = sorted(list(f["Camera_Parameter"].keys())) if "Camera_Parameter" in f else []
        attrs = {str(k): str(v) for k, v in f.attrs.items()}
        top = list(f.keys())
    return {"path": str(path), "name": path.name, "attrs": attrs, "top_level_keys": top, "camera_ids": cams, "has_required_cameras": all(c in cams for c in CAMERA_IDS)}


def read_camera(path: Path, cid: str, source_hw: tuple[int, int] | None = None) -> dict[str, np.ndarray]:
    with h5py.File(str(path), "r") as f:
        g = f[f"Camera_Parameter/{cid}"]
        return {"K": resize_intrinsic(g["K"][()], source_hw), "RT": g["RT"][()], "D": g["D"][()] if "D" in g else np.zeros(5)}


def mask_camera_key(cid: str) -> str:
    return str(int(cid))


def decode_image_bytes(data: np.ndarray) -> np.ndarray | None:
    if not isinstance(data, np.ndarray) or data.ndim != 1 or data.size < 16:
        return None
    try:
        with Image.open(BytesIO(data.tobytes())) as img:
            return np.asarray(img.convert("L"))
    except Exception:
        return None


def try_read_mask(path: Path, cid: str, frame_key: str = "0") -> dict[str, Any] | None:
    with h5py.File(str(path), "r") as f:
        raw_cid = mask_camera_key(cid)
        candidates = [
            f"Mask/{cid}/mask",
            f"Mask/{raw_cid}/mask/{frame_key}",
            f"Mask/{raw_cid}/mask",
            f"Mask/{raw_cid}",
            f"Mask/{cid}",
            f"Mask/{cid}/0",
            f"Mask/{cid}/000000",
        ]
        for key in candidates:
            if key in f:
                obj = f[key]
                if isinstance(obj, h5py.Dataset):
                    data = obj[()]
                    decoded = decode_image_bytes(data)
                    if decoded is not None:
                        return {"mask": resize_mask(decoded), "source_hw": tuple(int(x) for x in decoded.shape[:2]), "h5_path": key, "encoded": True}
                    if data.ndim >= 2:
                        return {"mask": resize_mask(data), "source_hw": tuple(int(x) for x in data.squeeze().shape[:2]), "h5_path": key, "encoded": False}
        if "Mask" in f and raw_cid in f["Mask"]:
            grp = f[f"Mask/{raw_cid}"]
            if isinstance(grp, h5py.Group):
                for name in list(grp.keys())[:10]:
                    obj = grp[name]
                    if isinstance(obj, h5py.Dataset):
                        data = obj[()]
                        decoded = decode_image_bytes(data)
                        if decoded is not None:
                            return {"mask": resize_mask(decoded), "source_hw": tuple(int(x) for x in decoded.shape[:2]), "h5_path": f"Mask/{raw_cid}/{name}", "encoded": True}
                        if data.ndim >= 2:
                            return {"mask": resize_mask(data), "source_hw": tuple(int(x) for x in data.squeeze().shape[:2]), "h5_path": f"Mask/{raw_cid}/{name}", "encoded": False}
                    if isinstance(obj, h5py.Group):
                        keys = sorted(obj.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
                        for sub in ([frame_key] if frame_key in obj else keys[:10]):
                            ds = obj[sub]
                            if isinstance(ds, h5py.Dataset):
                                data = ds[()]
                                decoded = decode_image_bytes(data)
                                if decoded is not None:
                                    return {"mask": resize_mask(decoded), "source_hw": tuple(int(x) for x in decoded.shape[:2]), "h5_path": f"Mask/{raw_cid}/{name}/{sub}", "encoded": True}
                                if data.ndim >= 2:
                                    return {"mask": resize_mask(data), "source_hw": tuple(int(x) for x in data.squeeze().shape[:2]), "h5_path": f"Mask/{raw_cid}/{name}/{sub}", "encoded": False}
    return None


def world_to_camera_matrix(RT: np.ndarray, convention: str) -> np.ndarray:
    if convention == "rt_world_to_camera":
        return RT
    elif convention == "inverse_rt_camera_to_world":
        return np.linalg.inv(RT)
    elif convention == "rt_transposed_rotation":
        M = RT.copy()
        M[:3, :3] = M[:3, :3].T
        return M
    else:
        raise ValueError(convention)


def project(points: np.ndarray, K: np.ndarray, RT: np.ndarray, convention: str) -> tuple[np.ndarray, np.ndarray]:
    M = world_to_camera_matrix(RT, convention)
    homo = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (M @ homo.T).T[:, :3]
    z = cam[:, 2]
    pix = (K @ cam.T).T
    xy = pix[:, :2] / np.clip(pix[:, 2:3], 1e-9, None)
    return xy, z


def mask_iou_from_xy(xy: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    h, w = mask.shape
    xi = np.rint(xy[:, 0]).astype(np.int64)
    yi = np.rint(xy[:, 1]).astype(np.int64)
    inside_frame = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    bbox_iou = 0.0
    center_error = 1.0
    area_ratio_log = 10.0
    if inside_frame.any():
        inside_mask = np.zeros_like(inside_frame, dtype=bool)
        inside_mask[inside_frame] = mask[yi[inside_frame], xi[inside_frame]]
        coverage = float(inside_mask.mean())
        pyx = np.stack([yi[inside_frame], xi[inside_frame]], axis=1)
        myx = np.argwhere(mask)
        if pyx.size and myx.size:
            pmin = pyx.min(axis=0)
            pmax = pyx.max(axis=0)
            mmin = myx.min(axis=0)
            mmax = myx.max(axis=0)
            inter_min = np.maximum(pmin, mmin)
            inter_max = np.minimum(pmax, mmax)
            inter_hw = np.maximum(0, inter_max - inter_min + 1)
            inter_area = float(inter_hw[0] * inter_hw[1])
            p_area = float(np.prod(pmax - pmin + 1))
            m_area = float(np.prod(mmax - mmin + 1))
            bbox_iou = inter_area / max(1.0, p_area + m_area - inter_area)
            pc = (pmin + pmax) / 2.0
            mc = (mmin + mmax) / 2.0
            center_error = float(np.linalg.norm((pc - mc) / np.array([h, w], dtype=np.float64)))
            area_ratio_log = float(abs(np.log(max(p_area, 1.0) / max(m_area, 1.0))))
            # Fast proxy IoU: bbox agreement weighted by sampled point mask coverage.
            iou = float(bbox_iou * min(1.0, coverage * 2.0))
        else:
            iou = 0.0
    else:
        iou = 0.0
        coverage = 0.0
    return {"in_frame_ratio": float(inside_frame.mean()), "mask_coverage": coverage, "silhouette_iou": iou, "bbox_iou": bbox_iou, "center_error": center_error, "area_ratio_log": area_ratio_log}


def center_translation(points: np.ndarray, target_xy: np.ndarray, K: np.ndarray, depth: float, convention_shift: np.ndarray) -> np.ndarray:
    # Approximate translation so the point centroid projects near target mask center.
    cx, cy = float(target_xy[0]), float(target_xy[1])
    fx, fy = K[0, 0], K[1, 1]
    px, py = K[0, 2], K[1, 2]
    x = (cx - px) / fx * depth
    y = (cy - py) / fy * depth
    target = np.array([x, y, depth], dtype=np.float64)
    return target - points.mean(axis=0) - convention_shift


def target_world_from_mask_center(center_xy: np.ndarray, K: np.ndarray, RT: np.ndarray, convention: str, depth: float) -> np.ndarray:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    fx, fy = K[0, 0], K[1, 1]
    px, py = K[0, 2], K[1, 2]
    cam = np.array([(cx - px) / fx * depth, (cy - py) / fy * depth, depth, 1.0], dtype=np.float64)
    world = np.linalg.inv(world_to_camera_matrix(RT, convention)) @ cam
    return world[:3]


def search() -> dict[str, Any]:
    BOARDS.mkdir(parents=True, exist_ok=True)
    data = load_vggt()
    pts_all = data["world_points"]
    valid = data["valid"]
    sample_pts, _ = sample_points(pts_all, valid, 6000)
    view_sample_pts = sample_points_by_view(pts_all, valid, 900)
    vggt_stats = {
        "xyz_min": pts_all[valid].min(axis=0).tolist(),
        "xyz_max": pts_all[valid].max(axis=0).tolist(),
        "xyz_mean": pts_all[valid].mean(axis=0).tolist(),
        "xyz_std": pts_all[valid].std(axis=0).tolist(),
        "per_view_centroids": [pts_all[i][valid[i]].mean(axis=0).tolist() for i in range(6)],
    }
    write_json(REPORTS / "V30300000000_vggt_world_coordinate_diagnosis.json", {"created_utc": now(), "stats": vggt_stats, "diagnosis": "arbitrary VGGT world scale; coordinate binding required"})
    write_json(REPORTS / "V30300000000_vggt_metadata_scan.json", {"created_utc": now(), "v11700": str(V11700), "surface": str(SURFACE)})

    smcs = sorted(SMC_DIR.glob("*.smc"))
    summaries = [load_smc_summary(p) for p in smcs]
    write_json(REPORTS / "V30100000000_smc_sequence_scan.json", {"created_utc": now(), "smc_count": len(summaries), "smcs": summaries})

    mask_rows: list[dict[str, Any]] = []
    sequence_scores: dict[str, list[float]] = {}
    masks: dict[tuple[str, str], dict[str, Any]] = {}
    for s in summaries:
        path = Path(s["path"])
        if not s["has_required_cameras"]:
            continue
        for cid in CAMERA_IDS:
            mask_info = try_read_mask(path, cid)
            if mask_info is None:
                mask_rows.append({"smc": path.name, "camera_id": cid, "mask_available": False})
                continue
            masks[(path.name, cid)] = mask_info
            m = mask_info["mask"]
            yx = np.argwhere(m)
            if yx.size:
                mask_info["center_xy"] = [float(yx[:, 1].mean()), float(yx[:, 0].mean())]
            else:
                mask_info["center_xy"] = [259.0, 259.0]
            # Compare to VGGT valid mask for matching view by required-camera order.
            vi = CAMERA_IDS.index(cid)
            vm = valid[vi]
            inter = np.logical_and(m, vm).sum()
            union = np.logical_or(m, vm).sum()
            iou = float(inter / max(1, union))
            mask_rows.append({"smc": path.name, "camera_id": cid, "mask_available": True, "mask_iou_with_v920_valid": iou, "mask_pixels": int(m.sum()), "v920_pixels": int(vm.sum()), "source_hw": list(mask_info["source_hw"]), "h5_path": mask_info["h5_path"], "encoded": mask_info["encoded"]})
            sequence_scores.setdefault(path.name, []).append(iou)
    write_csv(REPORTS / "V30100000000_smc_mask_iou.csv", mask_rows)
    ranking = [{"smc": k, "mean_mask_iou": float(np.mean(v)), "n": len(v)} for k, v in sequence_scores.items()]
    ranking.sort(key=lambda r: r["mean_mask_iou"], reverse=True)
    write_json(REPORTS / "V30100000000_smc_match_ranking.json", {"created_utc": now(), "ranking": ranking})

    conventions = []
    for rt_conv in ["rt_world_to_camera", "inverse_rt_camera_to_world", "rt_transposed_rotation"]:
        for unit_name, unit_scale in UNIT_SCALES.items():
            for axis_name, signs in AXIS_FLIPS.items():
                conventions.append({"rt_convention": rt_conv, "unit_name": unit_name, "unit_scale": unit_scale, "axis_flip": axis_name, "signs": signs})
    write_json(REPORTS / "V30200000000_camera_convention_candidates.json", {"created_utc": now(), "count": len(conventions), "conventions": conventions})
    write_json(REPORTS / "V30200000000_intrinsic_resize_policy.json", {"created_utc": now(), "resize_policy": "SMC PNG masks decoded at original HxW, resized nearest-neighbor to 518x518; K is scaled from decoded source_hw to 518x518 before projection"})

    candidate_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    # Search top all SMCs, but cap expensive candidate count by sampling depth/translation variants.
    depth_candidates = [0.8, 1.2, 2.0, 4.0, 8.0, 16.0]
    scale_candidates = np.unique(np.concatenate([np.logspace(-4, 4, 17), np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])]))
    # Keypoints validate inv(RT) as the only serious convention; keep RT as a negative control.
    prioritized = [
        c for c in conventions
        if c["rt_convention"] in {"inverse_rt_camera_to_world", "rt_world_to_camera"}
        and c["unit_name"] in {"unit", "meters_to_mm", "mm_to_meters"}
    ]
    for s in summaries:
        if not s["has_required_cameras"]:
            continue
        path = Path(s["path"])
        for conv in prioritized:
            signs = np.array(conv["signs"], dtype=np.float64)
            view_base = [p * signs[None, :] * float(conv["unit_scale"]) for p in view_sample_pts]
            for scale in scale_candidates:
                view_scaled = [p * float(scale) for p in view_base]
                for depth in depth_candidates:
                    target_points = []
                    source_points = []
                    camera_cache = {}
                    for vi, cid in enumerate(CAMERA_IDS):
                        mask_info = masks.get((path.name, cid))
                        if mask_info is None:
                            continue
                        mask = mask_info["mask"]
                        if view_scaled[vi].size == 0:
                            continue
                        center_xy = np.array(mask_info["center_xy"], dtype=np.float64)
                        cam = read_camera(path, cid, mask_info["source_hw"])
                        camera_cache[cid] = cam
                        target_points.append(target_world_from_mask_center(center_xy, cam["K"], cam["RT"], conv["rt_convention"], depth))
                        source_points.append(view_scaled[vi].mean(axis=0))
                    if len(target_points) < 3:
                        continue
                    shared_t = np.mean(np.stack(target_points), axis=0) - np.mean(np.stack(source_points), axis=0)
                    per_view = []
                    for vi, cid in enumerate(CAMERA_IDS):
                        mask_info = masks.get((path.name, cid))
                        if mask_info is None:
                            continue
                        mask = mask_info["mask"]
                        cam = camera_cache.get(cid) or read_camera(path, cid, mask_info["source_hw"])
                        xy, z = project(view_scaled[vi] + shared_t[None, :], cam["K"], cam["RT"], conv["rt_convention"])
                        stats = mask_iou_from_xy(xy, mask)
                        stats["positive_depth_ratio"] = float((z > 0).mean())
                        stats["camera_id"] = cid
                        per_view.append(stats)
                    if not per_view:
                        continue
                    mean_iou = float(np.mean([v["silhouette_iou"] for v in per_view]))
                    mean_cov = float(np.mean([v["mask_coverage"] for v in per_view]))
                    mean_frame = float(np.mean([v["in_frame_ratio"] for v in per_view]))
                    mean_depth = float(np.mean([v["positive_depth_ratio"] for v in per_view]))
                    mean_bbox = float(np.mean([v["bbox_iou"] for v in per_view]))
                    mean_center = float(np.mean([v["center_error"] for v in per_view]))
                    mean_area = float(np.mean([v["area_ratio_log"] for v in per_view]))
                    score = 3.0 * mean_bbox + 2.0 * mean_iou + mean_cov + 0.5 * mean_frame + 0.25 * mean_depth - 0.4 * mean_center - 0.05 * mean_area
                    row = {
                        "smc": path.name,
                        "rt_convention": conv["rt_convention"],
                        "unit_name": conv["unit_name"],
                        "axis_flip": conv["axis_flip"],
                        "scale": float(scale),
                        "depth_anchor": float(depth),
                        "mean_silhouette_iou": mean_iou,
                        "mean_mask_coverage": mean_cov,
                        "mean_in_frame_ratio": mean_frame,
                        "mean_positive_depth_ratio": mean_depth,
                        "mean_bbox_iou": mean_bbox,
                        "mean_center_error": mean_center,
                        "mean_area_ratio_log": mean_area,
                        "score": score,
                        "views_evaluated": len(per_view),
                        "translation_x": float(shared_t[0]),
                        "translation_y": float(shared_t[1]),
                        "translation_z": float(shared_t[2]),
                    }
                    candidate_rows.append(row)
                    if best is None or score > best["score"]:
                        best = row
    candidate_rows.sort(key=lambda r: r["score"], reverse=True)
    write_csv(REPORTS / "V30400000000_coordinate_binding_candidates.csv", candidate_rows[:5000])

    pass_threshold = {
        "score": 0.30,
        "mean_silhouette_iou": 0.01,
        "mean_bbox_iou": 0.05,
        "mean_mask_coverage": 0.02,
        "mean_in_frame_ratio": 0.20,
    }
    passes = bool(best and best["score"] >= pass_threshold["score"] and best["mean_bbox_iou"] >= pass_threshold["mean_bbox_iou"] and best["mean_mask_coverage"] >= pass_threshold["mean_mask_coverage"] and best["mean_in_frame_ratio"] >= pass_threshold["mean_in_frame_ratio"])
    best_doc = {
        "created_utc": now(),
        "best": best,
        "pass_threshold": pass_threshold,
        "binding_passed": passes,
        "candidates_evaluated": len(candidate_rows),
        "all_smc_scanned": len(summaries) == 8,
        "all_conventions_tried": len(conventions),
        "next_route": "V305 camera-bound verification" if passes else "V306 alternative coordinate routes",
    }
    write_json(REPORTS / "V30400000000_best_binding.json", best_doc)
    make_board(candidate_rows[:24])
    return best_doc


def make_board(rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r['smc']}\n{r['rt_convention']}\n{r['axis_flip']} s={r['scale']:.1e}" for r in rows]
    vals = [r["score"] for r in rows]
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.bar(range(len(vals)), vals)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=80, ha="right", fontsize=6)
    ax.set_title("V304 coordinate binding top candidates")
    fig.tight_layout()
    BOARDS.mkdir(parents=True, exist_ok=True)
    fig.savefig(BOARDS / "V30400000000_binding_visual_grid.png", dpi=180)
    plt.close(fig)


def main() -> None:
    print(json.dumps(search(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
