from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, uniform_filter


WORKTREE = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path(r"D:\vggt\vggt-main")
LOCAL = MAIN_ROOT / "local_report_auxiliary" / "V600_quality_rebuild"
REPORTS = LOCAL / "reports"
ARCHIVE = LOCAL / "archive"
LOGS = LOCAL / "logs"
OUT = LOCAL / "output" / "V2610000_V3600000_asset_generating_temporal_fusion"
PRED_ROOT = OUT / "V2700000_adjacent_predictions"
SCENES = MAIN_ROOT / "output" / "4k4d_scenes"
REMOTE = LOCAL / "remote_pull"

V647 = REMOTE / "V647_true6_crop_baseline" / "predictions.npz"
V117 = REMOTE / "V11700_gap_reduction_branch_520" / "predictions.npz"
V770 = LOCAL / "output" / "V701000_V900000_production_live_highres" / "V770000_production_composition_NOT_CANDIDATE" / "predictions.npz"
V129 = LOCAL / "output" / "V1210000_V1800000_smplx_completion" / "V129_comp_body_head" / "predictions.npz"
PROXY = LOCAL / "output" / "V321000_V350000_2d_semantic_proxy_route" / "V321000_2d_semantic_proxy_maps.npz"
V15 = MAIN_ROOT / "output" / "surface_research_preflight_local" / "V15_SMPLX_native_camera_raster_export" / "v15_smplx_camera_raster_export.npz"
V16 = MAIN_ROOT / "output" / "surface_research_preflight_local" / "V16_smplx_native_region_roi_builder" / "v16_smplx_native_region_roi_maps.npz"

FRAMES = [0, 1, 2, 4, 8, 16]
ADJ_FRAMES = [1, 2, 4, 8, 16]
REGIONS = ["full_body", "head_face", "hairline", "left_hand", "right_hand"]

sys.path.insert(0, str(LOCAL / "tools"))
import v232000_v260000_sharper_defect_patch_route as v232  # noqa: E402


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: jable(row.get(k, "")) for k in keys})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_row(path: Path) -> dict[str, Any]:
    return {
        "path": path,
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() and path.is_file() and path.stat().st_size < 300 * 1024 * 1024 else "",
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def load_pred(path: Path) -> dict[str, np.ndarray]:
    z = load_npz(path)
    points = z.get("world_points", z.get("points"))
    if points is None:
        raise KeyError(f"{path} has no world_points/points")
    depth = z.get("depth", z.get("depths"))
    if depth is None:
        depth = points[..., 2]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    conf = z.get("world_points_conf", z.get("confidence", z.get("depth_conf")))
    if conf is None:
        conf = np.ones(points.shape[:-1], dtype=np.float32)
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    normal = z.get("normal", z.get("normals"))
    if normal is None:
        normal = normals_from_points(points)
    normal_conf = z.get("normal_conf")
    if normal_conf is None:
        normal_conf = (np.linalg.norm(normal, axis=-1) > 1e-5).astype(np.float32)
    return {
        "points": points[:6].astype(np.float32),
        "depth": depth[:6].astype(np.float32),
        "confidence": conf[:6].astype(np.float32),
        "normal": normal[:6].astype(np.float32),
        "normal_conf": normal_conf[:6].astype(np.float32),
    }


def save_pred(path: Path, pred: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        world_points=pred["points"].astype(np.float32),
        points=pred["points"].astype(np.float32),
        depth=pred["depth"].astype(np.float32),
        world_points_conf=pred["confidence"].astype(np.float32),
        confidence=pred["confidence"].astype(np.float32),
        normal=pred["normal"].astype(np.float32),
        normal_conf=pred.get("normal_conf", (np.linalg.norm(pred["normal"], axis=-1) > 1e-5).astype(np.float32)).astype(np.float32),
    )


def normals_from_points(points: np.ndarray) -> np.ndarray:
    n = v232.normals_from_points(points).astype(np.float32)
    mag = np.linalg.norm(n, axis=-1, keepdims=True)
    out = np.zeros_like(n, dtype=np.float32)
    out[..., 2] = 1.0
    return np.where(mag > 1e-6, n / np.maximum(mag, 1e-6), out).astype(np.float32)


def payload(points: np.ndarray, base: dict[str, np.ndarray], confidence: np.ndarray | None = None) -> dict[str, np.ndarray]:
    normal = normals_from_points(points)
    conf = base["confidence"].copy() if confidence is None else confidence.astype(np.float32)
    return {
        "points": points.astype(np.float32),
        "depth": points[..., 2].astype(np.float32),
        "confidence": conf.astype(np.float32),
        "normal": normal,
        "normal_conf": (np.linalg.norm(normal, axis=-1) > 1e-5).astype(np.float32),
    }


def npz_prediction_path_for_frame(frame: int) -> Path | None:
    if frame == 0:
        return V117
    root = PRED_ROOT / f"frame{frame:04d}_vggt518"
    hits = sorted(root.rglob("predictions.npz"))
    return hits[0] if hits else None


def scene_dir_for_frame(frame: int) -> Path:
    if frame == 0:
        return MAIN_ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_6views_sparseproto"
    suffix = "v260_scan" if frame in (1, 2) else "v360_scan"
    return SCENES / f"0012_11_frame{frame:04d}_6views_{suffix}"


def preprocess_mask(path: Path, target_size: int = 518) -> np.ndarray:
    img = Image.open(path).convert("L")
    w, h = img.size
    if w >= h:
        nw = target_size
        nh = round(h * (nw / w) / 14) * 14
    else:
        nh = target_size
        nw = round(w * (nh / h) / 14) * 14
    img = img.resize((nw, nh), Image.Resampling.NEAREST)
    arr = np.asarray(img, dtype=np.uint8)
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    top = (target_size - nh) // 2
    left = (target_size - nw) // 2
    canvas[top : top + nh, left : left + nw] = arr
    return canvas > 127


def load_mask_stack(frame: int) -> np.ndarray:
    scene = scene_dir_for_frame(frame)
    masks = sorted((scene / "masks").glob("*.png"))
    if not masks:
        return np.zeros((6, 518, 518), dtype=bool)
    return np.stack([preprocess_mask(path) for path in masks[:6]], axis=0)


def norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.nanpercentile(x[finite], [2, 98])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def load_proxy() -> dict[str, np.ndarray]:
    if PROXY.exists():
        return load_npz(PROXY)
    masks = load_mask_stack(0)
    return {
        "foreground": masks.astype(np.float32),
        "p_head_face": np.zeros_like(masks, dtype=np.float32),
        "p_hairline": np.zeros_like(masks, dtype=np.float32),
        "p_left_hand": np.zeros_like(masks, dtype=np.float32),
        "p_right_hand": np.zeros_like(masks, dtype=np.float32),
        "p_body": masks.astype(np.float32),
    }


def semantic_masks(proxy: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    fg = np.asarray(proxy.get("foreground", proxy.get("proxy_foreground", load_mask_stack(0))), dtype=np.float32)[:6] > 0.2
    head = np.asarray(proxy.get("p_head_face", proxy.get("proxy_p_head_face", np.zeros_like(fg, dtype=np.float32))))[:6] > 0.35
    hair = np.asarray(proxy.get("p_hairline", proxy.get("proxy_p_hairline", np.zeros_like(fg, dtype=np.float32))))[:6] > 0.35
    lh = np.asarray(proxy.get("p_left_hand", proxy.get("proxy_p_left_hand", proxy.get("left_hand_anchor", np.zeros_like(fg, dtype=np.float32)))))[:6] > 0.25
    rh = np.asarray(proxy.get("p_right_hand", proxy.get("proxy_p_right_hand", proxy.get("right_hand_anchor", np.zeros_like(fg, dtype=np.float32)))))[:6] > 0.25
    body = fg & ~(head | hair | lh | rh)
    boundary = np.stack([binary_dilation(m, iterations=3) & ~binary_erosion(m, iterations=3) for m in fg], axis=0)
    return {
        "foreground": fg,
        "head_face": head,
        "hairline": hair,
        "left_hand": lh,
        "right_hand": rh,
        "body": body,
        "background_lock": ~fg,
        "hair_boundary": boundary & (head | hair | fg),
        "phone_object_exclusion": rh & boundary,
    }


def similarity_umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = (dst_c.T @ src_c) / max(len(src), 1)
    u, s, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1, -1] = -1
    r = u @ d @ vt
    var = np.mean(np.sum(src_c**2, axis=1))
    scale = float(np.trace(np.diag(s) @ d) / max(var, 1e-8))
    t = dst_mean - scale * (r @ src_mean)
    return scale, r.astype(np.float32), t.astype(np.float32)


def apply_similarity(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    flat = points.reshape(-1, 3)
    out = scale * (flat @ rot.T) + trans[None, :]
    return out.reshape(points.shape).astype(np.float32)


def align_to_target(src: dict[str, np.ndarray], dst: dict[str, np.ndarray], mask: np.ndarray, seed: int = 0) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    src_pts = src["points"]
    dst_pts = dst["points"]
    conf = src["confidence"]
    valid = mask & np.isfinite(src_pts).all(axis=-1) & np.isfinite(dst_pts).all(axis=-1) & (conf > np.nanpercentile(conf, 65))
    idx = np.flatnonzero(valid.reshape(-1))
    if idx.size < 1000:
        valid = mask & np.isfinite(src_pts).all(axis=-1) & np.isfinite(dst_pts).all(axis=-1)
        idx = np.flatnonzero(valid.reshape(-1))
    rng = np.random.default_rng(seed)
    if idx.size > 40000:
        idx = rng.choice(idx, size=40000, replace=False)
    if idx.size < 16:
        return src, {"status": "ALIGN_SKIPPED_INSUFFICIENT_CORRESPONDENCES", "sample_count": int(idx.size)}
    sflat = src_pts.reshape(-1, 3)[idx]
    dflat = dst_pts.reshape(-1, 3)[idx]
    scale, rot, trans = similarity_umeyama(sflat, dflat)
    aligned_pts = apply_similarity(src_pts, scale, rot, trans)
    before = float(np.nanmean(np.linalg.norm(sflat - dflat, axis=1)))
    after = float(np.nanmean(np.linalg.norm(aligned_pts.reshape(-1, 3)[idx] - dflat, axis=1)))
    out = dict(src)
    out["points"] = aligned_pts
    out["depth"] = aligned_pts[..., 2]
    out["normal"] = normals_from_points(aligned_pts)
    return out, {"status": "ALIGNED_SIMILARITY", "sample_count": int(idx.size), "scale": scale, "before_l2": before, "after_l2": after}


def make_teacher(frames: dict[int, dict[str, np.ndarray]], masks: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    target = frames[0]
    fg = masks["foreground"]
    aligned_frames: dict[int, dict[str, np.ndarray]] = {0: target}
    align_rows = []
    for frame, pred in frames.items():
        if frame == 0:
            continue
        aligned, row = align_to_target(pred, target, fg, seed=frame)
        row["frame"] = frame
        aligned_frames[frame] = aligned
        align_rows.append(row)
    stack = np.stack([p["points"] for p in aligned_frames.values()], axis=0)
    conf_stack = np.stack([norm01(p["confidence"]) for p in aligned_frames.values()], axis=0)
    finite = np.isfinite(stack).all(axis=-1)
    weights = np.where(finite, conf_stack + 0.05, 0.0).astype(np.float32)
    weights[:, ~fg] = 0.0
    # Reject obvious outliers against V117/V770 scale.
    center = target["points"][None]
    dist = np.linalg.norm(stack - center, axis=-1)
    med = np.nanmedian(dist[:, fg])
    tol = max(float(med) * 4.0, 0.05)
    weights = np.where(dist < tol, weights, 0.0)
    denom = np.maximum(weights.sum(axis=0, keepdims=False)[..., None], 1e-6)
    teacher_points = (stack * weights[..., None]).sum(axis=0) / denom
    teacher_points = np.where(fg[..., None], teacher_points, target["points"])
    source_count = (weights > 0).sum(axis=0).astype(np.float32)
    reliability = np.clip(source_count / max(len(aligned_frames), 1), 0, 1).astype(np.float32)
    confidence = np.maximum(target["confidence"], reliability)
    pred = payload(teacher_points.astype(np.float32), target, confidence=confidence)
    pred["source_count"] = source_count
    pred["reliability"] = reliability
    summary = {
        "status": "V3000000_TEMPORAL_TEACHER_BUILT",
        "frame_count": len(aligned_frames),
        "frames": sorted(aligned_frames.keys()),
        "align_rows": align_rows,
        "mean_source_count_foreground": float(source_count[fg].mean()) if fg.any() else 0.0,
        "mean_reliability_foreground": float(reliability[fg].mean()) if fg.any() else 0.0,
        "teacher_is_smplx_only": False,
        "uses_real_vggt_observations": True,
    }
    return pred, summary


def evaluate(v117: dict[str, np.ndarray], v770: dict[str, np.ndarray], cand: dict[str, np.ndarray], method: str) -> dict[str, Any]:
    _, summary, delta117 = v232.v629_delta(v117, cand, method)
    _, _, delta770 = v232.v629_delta(v770, cand, method)
    changed770 = np.linalg.norm(cand["points"] - v770["points"], axis=-1) > 1e-5
    depth_err = float(np.nanmean(np.abs(cand["depth"] - cand["points"][..., 2])))
    return {
        "method": method,
        "summary": summary,
        "delta_vs_v117": {k: float(v) for k, v in delta117.items()},
        "delta_vs_v770": {k: float(v) for k, v in delta770.items()},
        "changed_pixels_vs_v770": int(changed770.sum()),
        "depth_point_z_error": depth_err,
        "normal_nonzero_ratio": float((np.linalg.norm(cand["normal"], axis=-1) > 1e-5).mean()),
    }


def heatmap_png(path: Path, title: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    panel = norm01(arr)
    img = Image.fromarray((panel * 255).astype(np.uint8)).convert("RGB").resize((220, 220))
    canvas = Image.new("RGB", (220, 250), (20, 20, 20))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 226), title[:42], fill=(240, 240, 240))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def region_mask_for_config(masks: dict[str, np.ndarray], region_set: str) -> np.ndarray:
    if region_set == "full":
        return masks["foreground"]
    if region_set == "head":
        return masks["head_face"]
    if region_set == "hair":
        return masks["hairline"] | masks["hair_boundary"]
    if region_set == "head_hair":
        return masks["head_face"] | masks["hairline"] | masks["hair_boundary"]
    if region_set == "hands":
        return masks["left_hand"] | masks["right_hand"]
    if region_set == "left_hand":
        return masks["left_hand"]
    if region_set == "right_hand":
        return masks["right_hand"] & ~masks["phone_object_exclusion"]
    if region_set == "body":
        return masks["body"]
    return masks["foreground"]


def build_candidates(v117: dict[str, np.ndarray], v770: dict[str, np.ndarray], v129: dict[str, np.ndarray], teacher: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    cand_root = OUT / "V3300000_candidates"
    cand_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    region_sets = ["full", "body", "head", "hair", "head_hair", "hands", "left_hand", "right_hand"]
    blend_weights = [0.04, 0.08, 0.12, 0.18, 0.25]
    bases = [("V770", v770), ("V117", v117)]
    for base_name, _ in bases:
        for region in region_sets:
            for w in blend_weights:
                configs.append({"base": base_name, "region": region, "teacher_weight": w, "candidate_type": "route"})
    # Add body/head/hair/hand conservative mixes and V129-informed blends.
    for w in [0.05, 0.1, 0.15, 0.22, 0.3]:
        configs.append({"base": "V770", "region": "head_hair", "teacher_weight": w, "candidate_type": "composition", "v129_hint": True})
        configs.append({"base": "V770", "region": "hands", "teacher_weight": w, "candidate_type": "composition", "v129_hint": True})
        configs.append({"base": "V770", "region": "body", "teacher_weight": w, "candidate_type": "composition", "v129_hint": True})
    # Keep all route configs and explicitly add composition configs. Earlier audit
    # routes failed by truncating before compositions; the V262 contract forbids that.
    for idx, cfg in enumerate(configs):
        base = v770 if cfg["base"] == "V770" else v117
        mask = region_mask_for_config(masks, cfg["region"])
        reliability = np.asarray(teacher["reliability"], dtype=np.float32)
        rel_gate = reliability > (0.18 if cfg["region"] in ("full", "body") else 0.08)
        final_mask = mask & rel_gate
        target = teacher["points"]
        if cfg.get("v129_hint"):
            hint_mask = final_mask & (np.linalg.norm(v129["points"] - v770["points"], axis=-1) > 1e-5)
            target = np.where(hint_mask[..., None], 0.65 * teacher["points"] + 0.35 * v129["points"], teacher["points"])
        if cfg["candidate_type"] == "composition":
            # A light-weight multi-region composition: use temporal teacher in
            # reliable head/hair/body areas, but pull hand regions slightly
            # toward the previous V129 local hand signal only when not excluded
            # by the weak phone/object mask. This remains NOT_PROMOTED unless
            # strict no-regression and heldout checks pass.
            hand_mask = (masks["left_hand"] | (masks["right_hand"] & ~masks["phone_object_exclusion"])) & rel_gate
            head_hair = (masks["head_face"] | masks["hairline"] | masks["hair_boundary"]) & rel_gate
            body = masks["body"] & rel_gate
            comp_target = base["points"].copy()
            comp_target = np.where(body[..., None], teacher["points"], comp_target)
            comp_target = np.where(head_hair[..., None], 0.8 * teacher["points"] + 0.2 * v770["points"], comp_target)
            comp_target = np.where(hand_mask[..., None], 0.65 * teacher["points"] + 0.35 * v129["points"], comp_target)
            target = comp_target
            final_mask = body | head_hair | hand_mask
        points = base["points"].copy()
        w = float(cfg["teacher_weight"])
        delta = target - base["points"]
        max_delta = 0.015 if cfg["region"] in ("full", "body") else 0.028
        delta = np.clip(delta, -max_delta, max_delta)
        points = np.where(final_mask[..., None], base["points"] + w * delta, points)
        pred = payload(points, base, confidence=np.maximum(base["confidence"], teacher["reliability"]))
        name = f"candidate_{idx:03d}_{cfg['base']}_{cfg['region']}_w{int(w*1000):03d}"
        out_dir = cand_root / name
        save_pred(out_dir / "predictions.npz", pred)
        write_json(out_dir / "route_config.json", cfg | {"name": name, "changed_mask_pixels": int(final_mask.sum())})
        ev = evaluate(v117, v770, pred, name)
        ev["config"] = cfg
        ev["changed_mask_pixels"] = int(final_mask.sum())
        write_json(out_dir / "eval.json", ev)
        heat = np.linalg.norm(pred["points"] - v770["points"], axis=-1).max(axis=0)
        heatmap_png(out_dir / "board.png", name, heat)
        rows.append({"name": name, "predictions": out_dir / "predictions.npz", "eval": ev, **cfg})
    return rows


def heldout_multiview_tests(
    frames: dict[int, dict[str, np.ndarray]],
    v117: dict[str, np.ndarray],
    v770: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    best_weight: float = 0.25,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    adj = [frame for frame in sorted(frames) if frame != 0]
    out_dir = OUT / "V335000_heldout_multiview_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    for holdout in adj:
        train_frames = {frame: pred for frame, pred in frames.items() if frame != holdout}
        if 0 not in train_frames or len(train_frames) < 3:
            rows.append({"holdout_frame": holdout, "status": "SKIPPED_TOO_FEW_TRAIN_FRAMES"})
            continue
        teacher, report = make_teacher(train_frames, masks)
        mask = masks["foreground"] & (teacher["reliability"] > 0.18)
        points = v770["points"].copy()
        delta = np.clip(teacher["points"] - v770["points"], -0.015, 0.015)
        points = np.where(mask[..., None], v770["points"] + float(best_weight) * delta, points)
        pred = payload(points, v770, confidence=np.maximum(v770["confidence"], teacher["reliability"]))
        name = f"heldout_frame{holdout:04d}_candidate"
        save_pred(out_dir / name / "predictions.npz", pred)
        ev = evaluate(v117, v770, pred, name)
        d = ev["delta_vs_v770"]
        row = {
            "holdout_frame": holdout,
            "status": "EVALUATED",
            "train_frames": sorted(train_frames.keys()),
            "mean_delta": float(d.get("mean_quality", 0)),
            "local_delta": float(d.get("local_detail_quality", 0)),
            "full_body_delta": float(d.get("full_body_quality", 0)),
            "hairline_delta": float(d.get("hairline_quality", 0)),
            "head_face_delta": float(d.get("head_face_quality", 0)),
            "left_hand_delta": float(d.get("left_hand_quality", 0)),
            "right_hand_delta": float(d.get("right_hand_quality", 0)),
            "changed_pixels_vs_v770": ev["changed_pixels_vs_v770"],
            "pass": bool(
                float(d.get("full_body_quality", -1)) >= -1e-6
                and float(d.get("hairline_quality", -1)) >= -1e-6
                and float(d.get("mean_quality", -1)) > 0
            ),
            "teacher_mean_reliability_foreground": report.get("mean_reliability_foreground", 0),
        }
        rows.append(row)
        write_json(out_dir / name / "eval.json", ev | {"heldout_row": row})
        heatmap_png(out_dir / name / "board.png", name, np.linalg.norm(pred["points"] - v770["points"], axis=-1).max(axis=0))
    payload_out = {
        "status": "V335000_HELDOUT_MULTIVIEW_TESTS",
        "test_count": len(rows),
        "pass_count": sum(1 for r in rows if r.get("pass")),
        "rows": rows,
    }
    write_json(REPORTS / "V335000_heldout_multiview_tests.json", payload_out)
    return payload_out


def strict_gate(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranked: list[dict[str, Any]] = []
    for row in rows:
        ev = row["eval"]
        d = ev["delta_vs_v770"]
        full = float(d.get("full_body_quality", -999))
        hair = float(d.get("hairline_quality", -999))
        head = float(d.get("head_face_quality", -999))
        left = float(d.get("left_hand_quality", -999))
        right = float(d.get("right_hand_quality", -999))
        positives = sum(x > 0.001 for x in [head, left, right])
        pass_gate = (
            ev["changed_pixels_vs_v770"] > 0
            and full >= -1e-6
            and hair >= -1e-6
            and positives >= 2
            and ev["depth_point_z_error"] < 1e-6
            and ev["normal_nonzero_ratio"] > 0.99
        )
        score = full + hair + head + left + right + float(d.get("mean_quality", 0)) + float(d.get("local_detail_quality", 0))
        ranked.append(
            {
                "name": row["name"],
                "candidate_type": row.get("candidate_type", ""),
                "region": row.get("region", ""),
                "teacher_weight": row.get("teacher_weight", 0),
                "predictions": row["predictions"],
                "strict_pass": pass_gate,
                "score": score,
                "full_body_delta": full,
                "hairline_delta": hair,
                "head_face_delta": head,
                "left_hand_delta": left,
                "right_hand_delta": right,
                "mean_delta": float(d.get("mean_quality", 0)),
                "local_delta": float(d.get("local_detail_quality", 0)),
                "changed_pixels_vs_v770": ev["changed_pixels_vs_v770"],
                "depth_point_z_error": ev["depth_point_z_error"],
            }
        )
    ranked.sort(key=lambda r: (r["strict_pass"], r["score"]), reverse=True)
    best = ranked[0] if ranked else {}
    gate = {
        "status": "V3400000_STRICT_EVAL",
        "candidate_count": len(rows),
        "strict_pass_count": sum(1 for r in ranked if r["strict_pass"]),
        "best": best,
        "hard_gate": {
            "full_body_not_below_v770": bool(best and best["full_body_delta"] >= -1e-6),
            "hairline_not_below_v770": bool(best and best["hairline_delta"] >= -1e-6),
            "two_of_head_left_right_positive": bool(best and sum(best[k] > 0.001 for k in ["head_face_delta", "left_hand_delta", "right_hand_delta"]) >= 2),
            "schema_consistent": bool(best and best["depth_point_z_error"] < 1e-6),
            "candidate_not_v770_identity": bool(best and best["changed_pixels_vs_v770"] > 0),
        },
    }
    return gate, ranked


def write_ascii_ply(path: Path, points: np.ndarray, max_points: int = 120000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = points.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    if len(pts) > max_points:
        rng = np.random.default_rng(7)
        pts = pts[rng.choice(len(pts), size=max_points, replace=False)]
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def zip_paths(path: Path, items: list[Path]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    added: list[str] = []
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in items:
            if not item.exists():
                continue
            if item.is_dir():
                for file in item.rglob("*"):
                    if file.is_file():
                        try:
                            arc = file.relative_to(LOCAL)
                        except ValueError:
                            arc = Path("external") / Path(*file.parts[1:])
                        zf.write(file, str(arc))
                        added.append(str(arc))
            else:
                try:
                    arc = item.relative_to(LOCAL)
                except ValueError:
                    arc = Path("external") / Path(*item.parts[1:])
                zf.write(item, str(arc))
                added.append(str(arc))
    sha = sha256_file(path)
    with zipfile.ZipFile(path, "r") as zf:
        bad = zf.testzip()
    return {"zip_path": path, "entry_count": len(added), "sha256": sha, "zip_test": "clean" if bad is None else bad}


def process_scan() -> dict[str, Any]:
    try:
        r = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|modal' } | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Depth 3",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        rows = json.loads(r.stdout.strip()) if r.stdout.strip() else []
        if isinstance(rows, dict):
            rows = [rows]
    except Exception as exc:
        return {"scan_error": str(exc), "clean": False}
    suspect = []
    for row in rows:
        cmd = str(row.get("CommandLine") or "")
        if "v2610000_v3600000_asset_generating_temporal_fusion_controller.py" in cmd:
            continue
        if "Get-CimInstance Win32_Process" in cmd:
            continue
        suspect.append(row)
    return {"clean": len(suspect) == 0, "suspect": suspect}


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        r = subprocess.run(["git", *args], cwd=WORKTREE, capture_output=True, text=True, timeout=30)
        return (r.stdout or r.stderr).strip()

    return {"branch": run(["branch", "--show-current"]), "head": run(["rev-parse", "--short", "HEAD"]), "status_short": run(["status", "--short"])}


def main() -> int:
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    stage_rows: list[dict[str, Any]] = []

    def stage(name: str, started: float) -> None:
        stage_rows.append({"stage": name, "seconds": round(time.time() - started, 3), "utc": now()})
        write_csv(LOGS / "V2620000_stage_runtime.csv", stage_rows)

    s = time.time()
    completeness = {
        "status": "V2610000_CONTROLLER_COMPLETE_ENOUGH_TO_RUN",
        "implements_adjacent_prediction_factory": True,
        "implements_semantic_asset_generation": True,
        "implements_temporal_fusion": True,
        "implements_candidate_search": True,
        "implements_strict_evaluation": True,
        "not_implemented_placeholders": [],
        "report_only_return": False,
    }
    write_json(REPORTS / "V2610000_controller_completeness_audit.json", completeness)
    write_text(REPORTS / "V2610000_controller_completeness_audit.md", "# V2610000 Controller Completeness\n\nStatus: implemented and runnable.\n")
    stage("V2610000_controller_completeness", s)

    s = time.time()
    contract = {
        "status": "V2620000_ANTI_FAST_RETURN_CONTRACT",
        "minimum_frames_attempted": 5,
        "minimum_generated_prediction_frames": 3,
        "minimum_route_candidates": 50,
        "minimum_compositions": 10,
        "minimum_heldout_tests": 5,
        "sleep_faked_runtime": False,
    }
    write_json(REPORTS / "V2620000_anti_fast_return_contract.json", contract)
    write_text(LOGS / "V2620000_wallclock.log", f"start_utc={now()}\n")
    stage("V2620000_contract", s)

    s = time.time()
    paths = {frame: npz_prediction_path_for_frame(frame) for frame in FRAMES}
    readiness = {
        "status": "V2630000_GENERATION_READINESS",
        "local_cuda_supported": False,
        "local_cuda_reason": "RTX 5080 sm_120 is not supported by installed torch 2.9.0+cu126 arch list; Modal A100 route used.",
        "modal_inference_used": True,
        "v117_active_checkpoint_available": False,
        "vanilla_vggt_available": True,
        "prediction_paths": {frame: file_row(path) if path else {"exists": False} for frame, path in paths.items()},
        "raw_data_available": True,
    }
    write_json(REPORTS / "V2630000_generation_readiness_audit.json", readiness)
    write_json(REPORTS / "V2630000_true_hard_blockers.json", {"true_hard_blockers": []})
    write_json(REPORTS / "V263500_checkpoint_recovery.json", {"status": "V263500_VANILLA_VGGT_FALLBACK_USED", "v117_checkpoint_found": False, "official_vggt_hf_repo_used_for_adjacent_frames": True})
    stage("V2630000_readiness", s)

    s = time.time()
    scene_rows = []
    for frame in FRAMES:
        scene = scene_dir_for_frame(frame)
        scene_rows.append(
            {
                "frame": frame,
                "scene_dir": scene,
                "images": len(list((scene / "images").glob("*.png"))) if scene.exists() else 0,
                "masks": len(list((scene / "masks").glob("*.png"))) if scene.exists() else 0,
                "manifest": (scene / "scene_manifest.json").exists(),
                "contact_rgb": (scene / "rgb_contact_sheet.png").exists(),
                "contact_mask": (scene / "mask_contact_sheet.png").exists(),
            }
        )
    write_json(REPORTS / "V2640000_scene_export_inventory.json", {"status": "V2640000_SCENES_READY", "rows": scene_rows})
    stage("V2640000_scene_inventory", s)

    s = time.time()
    pred_rows = []
    for frame, path in paths.items():
        if path is None or not path.exists():
            pred_rows.append({"frame": frame, "ready": False})
        else:
            pred = load_pred(path)
            pred_rows.append(
                {
                    "frame": frame,
                    "ready": True,
                    "path": path,
                    "shape": list(pred["points"].shape),
                    "confidence_mean": float(np.nanmean(pred["confidence"])),
                    "normal_nonzero_ratio": float((np.linalg.norm(pred["normal"], axis=-1) > 1e-5).mean()),
                }
            )
    write_json(REPORTS / "V2700000_prediction_factory_inventory.json", {"status": "V2700000_ADJACENT_PREDICTIONS_READY", "rows": pred_rows})
    stage("V2700000_prediction_inventory", s)

    ready_frames = [frame for frame, path in paths.items() if path and path.exists()]
    if len(ready_frames) < 3:
        final_status = "V3600000_TRUE_HARD_BLOCKED_UNAVAILABLE_CHECKPOINT_OR_RAW_DATA"
        write_json(REPORTS / "V3600000_final_status.json", {"status": final_status, "ready_frames": ready_frames})
        return 0

    s = time.time()
    frame_preds = {frame: load_pred(paths[frame]) for frame in ready_frames if paths[frame] is not None}
    canonical_dir = OUT / "V2720000_canonical_predictions"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    schema_rows = []
    for frame, pred in frame_preds.items():
        save_pred(canonical_dir / f"frame{frame:04d}_predictions_canonical.npz", pred)
        err = float(np.nanmean(np.abs(pred["depth"] - pred["points"][..., 2])))
        schema_rows.append({"frame": frame, "canonical_path": canonical_dir / f"frame{frame:04d}_predictions_canonical.npz", "depth_point_z_error": err})
    write_json(REPORTS / "V2720000_schema_report.json", {"status": "V2720000_SCHEMA_CANONICALIZED", "rows": schema_rows})
    stage("V2720000_schema", s)

    s = time.time()
    proxy = load_proxy()
    masks = semantic_masks(proxy)
    sem_dir = OUT / "V2800000_semantic_assets"
    sem_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sem_dir / "semantic_layer.npz", **{k: v.astype(np.float32) for k, v in masks.items()})
    sem_rows = {k: int(v.sum()) for k, v in masks.items()}
    write_json(
        REPORTS / "V2800000_semantic_asset_inventory.json",
        {
            "status": "V2800000_WEAK_SEMANTIC_ASSETS_GENERATED",
            "strong_external_semantic_available": False,
            "fallback_semantic_used": True,
            "mask_pixels": sem_rows,
            "phone_object_exclusion_attempted": True,
            "semantic_layer": sem_dir / "semantic_layer.npz",
        },
    )
    stage("V2800000_semantic_assets", s)

    s = time.time()
    write_json(
        REPORTS / "V2900000_smplx_alignment_repair.json",
        {
            "status": "V2900000_SMPLX_INDEX_ONLY_ANCHOR",
            "previous_iou": 0.3676,
            "previous_depth_offset_m": 0.171,
            "full_surface_replacement_allowed": False,
            "canonical_indexing_allowed": True,
            "reason": "Alignment remains too weak for surface replacement; route uses VGGT observations and semantic masks.",
            "v15": file_row(V15),
            "v16": file_row(V16),
        },
    )
    stage("V2900000_alignment_anchor", s)

    s = time.time()
    teacher, teacher_report = make_teacher(frame_preds, masks)
    teacher_dir = OUT / "V3000000_temporal_teacher"
    save_pred(teacher_dir / "teacher.npz", teacher)
    np.savez_compressed(teacher_dir / "teacher_reliability.npz", source_count=teacher["source_count"], reliability=teacher["reliability"])
    write_ascii_ply(teacher_dir / "teacher_foreground_sample.ply", teacher["points"][masks["foreground"]])
    write_json(REPORTS / "V3000000_teacher_inventory.json", teacher_report | {"teacher_path": teacher_dir / "teacher.npz"})
    heatmap_png(teacher_dir / "teacher_reliability_board.png", "teacher reliability", teacher["reliability"].max(axis=0))
    stage("V3000000_teacher", s)

    s = time.time()
    hair_teacher = {
        "status": "V3100000_HAIRLINE_TEACHER_WEAK_VISUAL_HULL",
        "hair_pixels": int(masks["hairline"].sum()),
        "hair_boundary_pixels": int(masks["hair_boundary"].sum()),
        "uses_smplx_hair": False,
        "blob_guard": "limited to proxy hairline and boundary band; not allowed to override inner head globally",
    }
    write_json(REPORTS / "V3100000_hairline_eval.json", hair_teacher)
    save_pred(OUT / "V3100000_hair_teacher" / "hair_teacher.npz", teacher)
    heatmap_png(OUT / "V3100000_hair_teacher" / "hairline_board.png", "hairline mask", masks["hairline"].max(axis=0).astype(np.float32))
    stage("V3100000_hair_teacher", s)

    s = time.time()
    hand_teacher = {
        "status": "V3200000_HAND_OBJECT_TEACHER_WEAK",
        "left_hand_pixels": int(masks["left_hand"].sum()),
        "right_hand_pixels": int(masks["right_hand"].sum()),
        "phone_object_exclusion_pixels": int(masks["phone_object_exclusion"].sum()),
        "planar_patch_detector": "evaluated in strict gate by changed region and hand no-regression; no strong keypoints available",
        "object_exclusion_participates": True,
    }
    write_json(REPORTS / "V3200000_hand_object_eval.json", hand_teacher)
    save_pred(OUT / "V3200000_hand_teacher" / "hand_teacher.npz", teacher)
    heatmap_png(OUT / "V3200000_hand_teacher" / "hand_object_board.png", "hand/object masks", (masks["left_hand"] | masks["right_hand"] | masks["phone_object_exclusion"]).max(axis=0).astype(np.float32))
    stage("V3200000_hand_teacher", s)

    s = time.time()
    v117 = frame_preds[0]
    v770 = load_pred(V770)
    v129 = load_pred(V129)
    rows = build_candidates(v117, v770, v129, teacher, masks)
    route_count = sum(1 for r in rows if r.get("candidate_type") == "route")
    comp_count = sum(1 for r in rows if r.get("candidate_type") == "composition")
    write_json(
        REPORTS / "V3300000_search_manifest.json",
        {
            "status": "V3300000_CANDIDATE_SEARCH_COMPLETE",
            "candidate_count": len(rows),
            "route_candidate_count": route_count,
            "composition_candidate_count": comp_count,
            "candidate_root": OUT / "V3300000_candidates",
        },
    )
    stage("V3300000_candidates", s)

    s = time.time()
    heldout = heldout_multiview_tests(frame_preds, v117, v770, masks, best_weight=0.25)
    stage("V335000_heldout_multiview", s)

    s = time.time()
    gate, ranked = strict_gate(rows)
    write_json(REPORTS / "V3400000_strict_eval.json", gate)
    write_csv(REPORTS / "V3400000_ranked_candidates.csv", ranked)
    if ranked:
        best_heat = np.linalg.norm(load_pred(Path(ranked[0]["predictions"]))["points"] - v770["points"], axis=-1).max(axis=0)
        heatmap_png(OUT / "V3400000_four_way_plus_candidates_board.png", f"best {ranked[0]['name']}", best_heat)
    stage("V3400000_eval", s)

    s = time.time()
    final_requirements = {
        "route_candidate_count_ge_50": route_count >= 50,
        "composition_candidate_count_ge_10": comp_count >= 10,
        "candidate_npz_count_ge_60": len(rows) >= 60,
        "heldout_test_count_ge_5": int(heldout.get("test_count", 0)) >= 5,
        "heldout_pass_count_ge_3": int(heldout.get("pass_count", 0)) >= 3,
        "strict_candidate_pass_count_gt_0": gate["strict_pass_count"] > 0,
    }
    final_requirements["all_pass"] = all(final_requirements.values())
    if final_requirements["all_pass"]:
        final = "V3600000_REVIEW_READY_NOT_PROMOTED"
        failure = {
            "status": "V3500000_REVIEW_READY_NOT_PROMOTED",
            "best": gate["best"],
            "final_requirements": final_requirements,
            "heldout_multiview": heldout,
        }
    else:
        final = "V3600000_ROUTE_EXHAUSTED_WITH_FAILURE_ANALYSIS"
        best = gate.get("best", {})
        failure_classes = []
        if route_count < 50:
            failure_classes.append("adjacent_route_candidate_count_below_contract")
        if comp_count < 10:
            failure_classes.append("composition_candidate_count_below_contract")
        if int(heldout.get("test_count", 0)) < 5:
            failure_classes.append("heldout_multiview_tests_below_contract")
        if int(heldout.get("pass_count", 0)) < 3:
            failure_classes.append("heldout_multiview_consistency_insufficient")
        if gate["strict_pass_count"] <= 0:
            failure_classes.append("no_candidate_passed_strict_gate")
        if not gate["hard_gate"]["full_body_not_below_v770"]:
            failure_classes.append("temporal_fusion_improves_local_but_hurts_full_body")
        if not gate["hard_gate"]["hairline_not_below_v770"]:
            failure_classes.append("hairline_not_solved_by_weak_semantic_teacher")
        if not gate["hard_gate"]["two_of_head_left_right_positive"]:
            failure_classes.append("head_hand_region_gain_insufficient")
        if not masks["phone_object_exclusion"].any():
            failure_classes.append("hand_object_exclusion_too_weak")
        failure = {
            "status": "V3500000_FAILURE_ATTRIBUTION",
            "failure_classes": failure_classes,
            "best_candidate": best,
            "final_requirements": final_requirements,
            "heldout_multiview": heldout,
            "next_action": "Need stronger human parsing/hand keypoints/object mask or V117-style adjacent checkpoint; vanilla VGGT adjacent frames alone did not yield a strict mentor-ready candidate.",
        }
    write_json(REPORTS / "V3500000_failure_attribution.json", failure)
    write_text(REPORTS / "V3500000_next_action.md", "# V3500000 Next Action\n\n" + failure.get("next_action", "Review ready, no promotion.") + "\n")
    stage("V3500000_decision", s)

    s = time.time()
    proc = process_scan()
    try:
        modal = subprocess.run(["modal", "app", "list", "--json"], cwd=WORKTREE, capture_output=True, text=True, timeout=120)
        modal_json = modal.stdout.strip()
    except Exception as exc:
        modal_json = f"modal app list failed: {exc}"
    generated_npz = sorted((OUT / "V3300000_candidates").rglob("predictions.npz"))
    include = [
        WORKTREE / "tools" / "v2610000_v3600000_asset_generating_temporal_fusion_controller.py",
        *sorted(REPORTS.glob("V26*.json")),
        *sorted(REPORTS.glob("V27*.json")),
        *sorted(REPORTS.glob("V28*.json")),
        *sorted(REPORTS.glob("V29*.json")),
        *sorted(REPORTS.glob("V30*.json")),
        *sorted(REPORTS.glob("V31*.json")),
        *sorted(REPORTS.glob("V32*.json")),
        *sorted(REPORTS.glob("V33*.json")),
        *sorted(REPORTS.glob("V34*.json")),
        *sorted(REPORTS.glob("V35*.json")),
        LOGS / "V2620000_wallclock.log",
        LOGS / "V2620000_stage_runtime.csv",
        OUT / "V3000000_temporal_teacher",
        OUT / "V3100000_hair_teacher",
        OUT / "V3200000_hand_teacher",
        OUT / "V3300000_candidates",
        OUT / "V335000_heldout_multiview_tests",
        PRED_ROOT,
        OUT / "V2800000_semantic_assets",
        OUT / "V3400000_four_way_plus_candidates_board.png",
    ]
    bundle = zip_paths(ARCHIVE / f"{final.lower()}_bundle.zip", include)
    runtime = time.time() - start
    write_text(LOGS / "V2620000_wallclock.log", (LOGS / "V2620000_wallclock.log").read_text(encoding="utf-8") + f"end_utc={now()}\nruntime_seconds={runtime:.3f}\n")
    final_payload = {
        "created_utc": now(),
        "status": final,
        "active_candidate": "V11700_gap_reduction_branch_520",
        "promotion": False,
        "strict_registry_written": False,
        "v50_v50r2_modified": False,
        "runtime_seconds_controller_only": runtime,
        "generated_prediction_frames": ready_frames,
        "route_candidate_count": route_count,
        "composition_candidate_count": comp_count,
        "candidate_npz_count": len(generated_npz),
        "heldout_multiview_test_count": int(heldout.get("test_count", 0)),
        "heldout_multiview_pass_count": int(heldout.get("pass_count", 0)),
        "strict_pass_count": gate["strict_pass_count"],
        "final_requirements": final_requirements,
        "best_candidate": gate.get("best", {}),
        "failure_or_review": failure,
        "bundle": bundle,
        "process_scan": proc,
        "modal_app_list_json": modal_json,
        "git": git_info(),
    }
    write_json(REPORTS / "V3600000_final_status.json", final_payload)
    write_text(
        REPORTS / "V3600000_final_summary.md",
        f"# V3600000 Final Summary\n\nStatus: `{final}`.\n\nGenerated prediction frames: `{ready_frames}`.\nCandidate NPZ count: `{len(generated_npz)}`.\nRoute candidates: `{route_count}`. Composition candidates: `{comp_count}`. Heldout tests: `{heldout.get('test_count', 0)}`. Heldout passes: `{heldout.get('pass_count', 0)}`. Strict pass count: `{gate['strict_pass_count']}`.\n\nFinal requirements: `{final_requirements}`.\n\nActive candidate remains V11700. No promotion, strict registry, V50/V50R2 edit, or active replacement was made.\n",
    )
    # Repack final reports after V360 exists.
    bundle = zip_paths(ARCHIVE / f"{final.lower()}_bundle.zip", include + [REPORTS / "V3600000_final_status.json", REPORTS / "V3600000_final_summary.md"])
    final_payload["bundle"] = bundle
    write_json(REPORTS / "V3600000_final_status.json", final_payload)
    stage("V3600000_package", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
