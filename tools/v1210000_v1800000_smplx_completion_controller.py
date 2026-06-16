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
OUT = LOCAL / "output" / "V1210000_V1800000_smplx_completion"
ARCHIVE = LOCAL / "archive"
TRAIN = MAIN_ROOT / "output" / "training_cases" / "0012_11_frame0000_6views_smplx_native_prior_v15"
PROXY = LOCAL / "output" / "V321000_V350000_2d_semantic_proxy_route" / "V321000_2d_semantic_proxy_maps.npz"
V647 = LOCAL / "remote_pull" / "V647_true6_crop_baseline" / "predictions.npz"
V11700 = LOCAL / "remote_pull" / "V11700_gap_reduction_branch_520" / "predictions.npz"
V770 = LOCAL / "output" / "V701000_V900000_production_live_highres" / "V770000_production_composition_NOT_CANDIDATE" / "predictions.npz"
V1000000 = LOCAL / "output" / "V940000_V1200000_long_route" / "V1000000_multiroute_composition" / "predictions.npz"
V15 = MAIN_ROOT / "output" / "surface_research_preflight_local" / "V15_SMPLX_native_camera_raster_export" / "v15_smplx_camera_raster_export.npz"
V16 = MAIN_ROOT / "output" / "surface_research_preflight_local" / "V16_smplx_native_region_roi_builder" / "v16_smplx_native_region_roi_maps.npz"

sys.path.insert(0, str(LOCAL / "tools"))
import v232000_v260000_sharper_defect_patch_route as v232  # noqa: E402


REGION_NAMES = ["full_body", "head_face", "hairline", "left_hand", "right_hand"]


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
    keys = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: jable(row.get(k, "")) for k in keys})


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def load_pred(path: Path) -> dict[str, np.ndarray]:
    z = load_npz(path)
    points = z.get("points", z.get("world_points"))
    depth = z.get("depth", z.get("depths"))
    conf = z.get("confidence", z.get("world_points_conf", z.get("depth_conf")))
    normal = z.get("normal", z.get("normals", None))
    if depth is None and points is not None:
        depth = points[..., 2]
    if conf is None:
        conf = np.ones(points.shape[:-1], dtype=np.float32)
    if normal is None:
        normal = normals_from_points(points.astype(np.float32))
    return {
        "points": points[:6].astype(np.float32),
        "depth": depth[:6].astype(np.float32),
        "confidence": conf[:6].astype(np.float32),
        "normal": normal[:6].astype(np.float32),
    }


def normals_from_points(points: np.ndarray) -> np.ndarray:
    n = v232.normals_from_points(points).astype(np.float32)
    lens = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.maximum(lens, 1e-6)


def payload(points: np.ndarray, base: dict[str, np.ndarray], confidence: np.ndarray | None = None) -> dict[str, np.ndarray]:
    normal = normals_from_points(points.astype(np.float32))
    conf = base["confidence"].copy() if confidence is None else confidence.astype(np.float32)
    return {
        "points": points.astype(np.float32),
        "world_points": points.astype(np.float32),
        "depth": points[..., 2].astype(np.float32),
        "confidence": conf,
        "normal": normal,
        "normal_conf": (np.linalg.norm(normal, axis=-1) > 1e-5).astype(np.float32),
    }


def save_pred(path: Path, pred: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **pred)


def evaluate(base: dict[str, np.ndarray], cand: dict[str, np.ndarray], name: str) -> dict[str, Any]:
    _, summary, delta = v232.v629_delta(base, cand, name)
    return {"summary": summary, "delta_vs_v117": {k: float(v) for k, v in delta.items()}}


def delta_to(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {k: float(a.get(k, 0.0) - b.get(k, 0.0)) for k in b}


def all_no_regression(delta: dict[str, float], eps: float = -1e-6) -> bool:
    return all(float(v) >= eps for v in delta.values())


def norm01(a: np.ndarray) -> np.ndarray:
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=np.float32)
    lo, hi = np.nanpercentile(a[finite], [2, 98])
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32)


def smooth_points(points: np.ndarray, size: int) -> np.ndarray:
    return np.stack([uniform_filter(points[..., c], size=(1, size, size)) for c in range(3)], axis=-1).astype(np.float32)


def bbox(mask: np.ndarray, pad: int = 8) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    h, w = mask.shape
    if xs.size == 0:
        return 0, 0, w, h
    return max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad), min(w, int(xs.max()) + 1 + pad), min(h, int(ys.max()) + 1 + pad)


def point_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    z = load_npz(path)
    out: dict[str, Any] = {"exists": True, "path": str(path)}
    for k, a in z.items():
        d = {"shape": list(a.shape), "dtype": str(a.dtype)}
        if a.dtype.kind in "fiu" and a.size:
            finite = np.isfinite(a) if a.dtype.kind == "f" else np.ones_like(a, dtype=bool)
            d.update(
                {
                    "finite_ratio": float(finite.mean()),
                    "nan_ratio": float(np.isnan(a).mean()) if a.dtype.kind == "f" else 0.0,
                    "min": float(np.nanmin(a)),
                    "max": float(np.nanmax(a)),
                    "mean": float(np.nanmean(a)),
                }
            )
        out[k] = d
    return out


def process_scan() -> dict[str, Any]:
    try:
        proc = subprocess.run(
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
        text = proc.stdout.strip()
    except Exception as exc:  # pragma: no cover
        return {"scan_error": str(exc), "residual_training_or_modal_worker_detected": True}
    bad = []
    if text:
        payload_json = json.loads(text)
        if isinstance(payload_json, dict):
            payload_json = [payload_json]
        for row in payload_json:
            cmd = str(row.get("CommandLine") or "")
            if "v1210000_v1800000_smplx_completion_controller.py" in cmd:
                continue
            if "modal app list" in cmd:
                continue
            bad.append(row)
    return {"raw": text, "residual_training_or_modal_worker_detected": bool(bad), "suspect": bad}


def zip_manifest(path: Path, include: list[Path]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in include:
            if item.exists():
                arc = item.name if item.is_file() else str(item)
                try:
                    arc = str(item.relative_to(LOCAL))
                except ValueError:
                    try:
                        arc = str(item.relative_to(WORKTREE))
                    except ValueError:
                        pass
                if item.is_file():
                    zf.write(item, arc)
    with zipfile.ZipFile(path, "r") as zf:
        bad = zf.testzip()
        entries = len(zf.infolist())
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"zip": str(path), "entries": entries, "zip_test": "clean" if bad is None else bad, "sha256": h.hexdigest()}


def masks(proxy: dict[str, np.ndarray], targets: dict[str, np.ndarray], v16: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    fg = proxy["foreground"].astype(bool)
    teacher = targets["teacher_mask"][:6].astype(bool)
    native = targets["smplx_native_visible_mask"][:6].astype(bool)
    prior_depth = targets["prior_depths"][:6]
    visible = teacher | native | (prior_depth > 1e-6)
    roi_names = [str(x) for x in v16["roi_names"].tolist()]
    roi = v16["roi_maps"][:6].astype(bool)
    def roi_mask(*names: str) -> np.ndarray:
        out = np.zeros_like(fg, dtype=bool)
        for name in names:
            if name in roi_names:
                out |= roi[:, roi_names.index(name)]
        return out
    head_face = ((proxy["p_head_face"] > 0.42) | roi_mask("head", "face_front", "face_lmk_static", "face_lmk_dynamic")) & fg
    hairline = (proxy["p_hairline"] > 0.38) & fg
    left_hand = ((proxy["p_left_hand"] > 0.32) | roi_mask("left_hand")) & fg
    right_hand = ((proxy["p_right_hand"] > 0.26) | roi_mask("right_hand")) & fg
    hand_risk = (proxy["p_garment_or_ambiguous"] > 0.55) & (left_hand | right_hand | binary_dilation(left_hand | right_hand, iterations=3))
    head_hair = head_face | hairline
    boundary = np.zeros_like(fg, dtype=bool)
    for view in range(fg.shape[0]):
        hh = head_hair[view]
        boundary[view] = (binary_dilation(hh, iterations=5) ^ binary_erosion(hh, iterations=3)) & fg[view]
    return {
        "foreground": fg,
        "smplx_visible": visible,
        "head_face": head_face,
        "hairline": hairline,
        "head_hair_boundary": boundary,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "hand_risk": hand_risk,
        "body": fg & ~(head_face | hairline | left_hand | right_hand),
    }


def candidate_metrics(
    name: str,
    pred: dict[str, np.ndarray],
    base117: dict[str, np.ndarray],
    base770: dict[str, np.ndarray],
    mm: dict[str, np.ndarray],
    missing: np.ndarray,
) -> dict[str, Any]:
    ev = evaluate(base117, pred, name)
    d117 = ev["delta_vs_v117"]
    d770 = delta_to(d117, evaluate(base117, base770, "v770_tmp")["delta_vs_v117"])
    changed = np.linalg.norm(pred["points"] - base117["points"], axis=-1) > 1e-6
    changed_vs_v770 = np.linalg.norm(pred["points"] - base770["points"], axis=-1) > 1e-6
    normal_nonzero = float((np.linalg.norm(pred["normal"], axis=-1) > 1e-4).mean())
    zerr = float(np.nanmean(np.abs(pred["depth"] - pred["points"][..., 2])))
    fg = mm["foreground"]
    valid_human = float((np.isfinite(pred["points"]).all(axis=-1) & fg).sum() / max(int(fg.sum()), 1))
    fill = float((changed & missing).sum() / max(int(missing.sum()), 1))
    outside_leak = float((changed & ~fg).sum() / max(int(changed.sum()), 1))
    return {
        "method": name,
        "delta_vs_v117": d117,
        "delta_vs_v770": d770,
        "changed_pixels_vs_v117": int(changed.sum()),
        "changed_pixels_vs_v770": int(changed_vs_v770.sum()),
        "changed_equals_v770": bool(np.max(np.abs(pred["points"] - base770["points"])) < 1e-12),
        "normal_nonzero_ratio": normal_nonzero,
        "depth_point_z_error": zerr,
        "valid_human_point_ratio": valid_human,
        "missing_surface_fill_ratio": fill,
        "outside_leakage_ratio": outside_leak,
        "no_regression_vs_v117": all_no_regression(d117),
        "beats_v770_mean_local": d770["mean_quality"] > 1e-6 and d770["local_detail_quality"] > 1e-6,
        "head_hair_beats_v770": d770["head_face_quality"] > 1e-6 or d770["hairline_quality"] > 1e-6,
    }


def make_candidate(points: np.ndarray, base: dict[str, np.ndarray], confidence: np.ndarray | None = None) -> dict[str, np.ndarray]:
    return payload(points.astype(np.float32), base, confidence=confidence)


def blend(base: np.ndarray, target: np.ndarray, mask: np.ndarray, weight: float) -> np.ndarray:
    out = base.copy()
    out[mask] = (1.0 - weight) * base[mask] + weight * target[mask]
    return out


def shell_offset(points: np.ndarray, normals: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    out = points.copy()
    out[mask] = points[mask] + amount * normals[mask]
    return out


def scatter_panel(points: np.ndarray, mask: np.ndarray, title: str, color: tuple[int, int, int]) -> Image.Image:
    im = Image.new("RGB", (260, 240), "white")
    draw = ImageDraw.Draw(im)
    pts = points[mask]
    if pts.shape[0] > 0:
        if pts.shape[0] > 3500:
            pts = pts[np.linspace(0, pts.shape[0] - 1, 3500).astype(int)]
        x = pts[:, 0]
        y = pts[:, 1]
        lo_x, hi_x = np.percentile(x, [2, 98])
        lo_y, hi_y = np.percentile(y, [2, 98])
        sx = (x - lo_x) / max(hi_x - lo_x, 1e-6)
        sy = (y - lo_y) / max(hi_y - lo_y, 1e-6)
        for px, py in zip(np.clip(sx, 0, 1), np.clip(sy, 0, 1)):
            draw.point((int(px * 240) + 10, int((1 - py) * 210) + 20), fill=color)
    draw.text((8, 4), title, fill=(0, 0, 0))
    return im


def make_board(v647: dict[str, np.ndarray], v117: dict[str, np.ndarray], v770: dict[str, np.ndarray], cand: dict[str, np.ndarray], mm: dict[str, np.ndarray]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    regions = [
        ("full", mm["foreground"]),
        ("head_face", mm["head_face"]),
        ("hairline", mm["hairline"]),
        ("left_hand", mm["left_hand"]),
        ("right_hand", mm["right_hand"]),
    ]
    rows: list[Image.Image] = []
    for name, mask3 in regions:
        # Pick the view with the largest region support.
        counts = mask3.reshape(mask3.shape[0], -1).sum(axis=1)
        view = int(np.argmax(counts))
        mask = mask3[view]
        x0, y0, x1, y1 = bbox(mask, 12)
        crop_mask = mask[y0:y1, x0:x1]
        panels = []
        for label, pred, color in [
            ("V647", v647, (80, 80, 80)),
            ("V117", v117, (40, 120, 220)),
            ("V770", v770, (220, 70, 70)),
            ("new", cand, (80, 170, 80)),
        ]:
            panels.append(scatter_panel(pred["points"][view, y0:y1, x0:x1], crop_mask, f"{name} {label}", color))
        heat_arr = norm01(np.linalg.norm(cand["points"][view, y0:y1, x0:x1] - v770["points"][view, y0:y1, x0:x1], axis=-1))
        heat = Image.fromarray(np.uint8(heat_arr * 255)).convert("L").resize((260, 240)).convert("RGB")
        ImageDraw.Draw(heat).text((8, 4), f"{name} new-V770", fill=(255, 0, 0))
        panels.append(heat)
        row = Image.new("RGB", (260 * len(panels), 240), "white")
        for i, p in enumerate(panels):
            row.paste(p, (260 * i, 0))
        rows.append(row)
    board = Image.new("RGB", (260 * 5, 240 * len(rows)), "white")
    for i, row in enumerate(rows):
        board.paste(row, (0, i * 240))
    out = OUT / "V1300000_four_way_completion_board.png"
    board.save(out)
    return out


def main() -> None:
    start = time.time()
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)

    contract = {
        "status": "V1210000_ANTI_FAST_RETURN_CONTRACT",
        "wall_clock_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
        "rules": {
            "identity_to_v770_is_fail": True,
            "report_only_is_fail": True,
            "strict_registry_written": False,
            "promotion_allowed": False,
            "active_candidate": "V11700_gap_reduction_branch_520",
        },
    }
    write_json(REPORTS / "V1210000_anti_fast_return_contract.json", contract)
    write_text(REPORTS / "V1210000_runtime_budget.md", "# V1210000 Runtime Budget\n\nRoute must generate new NPZ candidates and cannot return a V770-identical result as review-ready.\n")

    assets = {
        "V647": V647,
        "V11700": V11700,
        "V770": V770,
        "V1000000": V1000000,
        "targets": TRAIN / "targets.npz",
        "inputs": TRAIN / "inputs.npz",
        "proxy": PROXY,
        "V15_smplx_raster": V15,
        "V16_smplx_roi": V16,
    }
    missing = [k for k, p in assets.items() if not p.exists()]
    asset_stats = {k: point_stats(p) for k, p in assets.items() if p.exists() and p.suffix == ".npz"}
    write_json(REPORTS / "V1220000_asset_reaudit.json", {"status": "V1220000_ASSET_REAUDIT", "missing": missing, "assets": {k: str(v) for k, v in assets.items()}, "stats": asset_stats})
    if missing:
        final = {"status": "V1800000_HARD_BLOCKED_MISSING_ASSETS", "missing": missing}
        write_json(REPORTS / "V1800000_final_status.json", final)
        print(json.dumps(final, indent=2))
        return

    v647 = load_pred(V647)
    v117 = load_pred(V11700)
    v770 = load_pred(V770)
    v100 = load_pred(V1000000) if V1000000.exists() else v770
    targets = load_npz(TRAIN / "targets.npz")
    proxy = load_npz(PROXY)
    v15 = load_npz(V15)
    v16 = load_npz(V16)
    mm = masks(proxy, targets, v16)

    v100_identity = {
        "points_maxdiff": float(np.max(np.abs(v100["points"] - v770["points"]))),
        "depth_maxdiff": float(np.max(np.abs(v100["depth"] - v770["depth"]))),
        "normal_maxdiff": float(np.max(np.abs(v100["normal"] - v770["normal"]))),
        "confidence_maxdiff": float(np.max(np.abs(v100["confidence"] - v770["confidence"]))),
    }
    depth_sync = {
        "v770_depth_point_z_mean_abs": float(np.mean(np.abs(v770["depth"] - v770["points"][..., 2]))),
        "v117_depth_point_z_mean_abs": float(np.mean(np.abs(v117["depth"] - v117["points"][..., 2]))),
    }
    write_text(
        REPORTS / "V1220000_prediction_internal_consistency.md",
        "# V1220000 Prediction Internal Consistency\n\n"
        f"V1000000 vs V770 identity: `{v100_identity}`.\n\n"
        f"Depth/point sync: `{depth_sync}`.\n",
    )

    prior_points = targets["prior_points"][:6].astype(np.float32)
    prior_normals = targets["prior_normals"][:6].astype(np.float32)
    prior_mask = (targets["prior_depths"][:6] > 1e-6) | targets["smplx_native_visible_mask"][:6].astype(bool)
    fg = mm["foreground"]
    support_iou = float((prior_mask & fg).sum() / max(int((prior_mask | fg).sum()), 1))
    prior_depth = prior_points[..., 2]
    v117_depth = v117["points"][..., 2]
    depth_delta = np.abs(prior_depth - v117_depth)
    smplx_alignment = {
        "status": "ALIGNMENT_OK" if support_iou > 0.02 and np.isfinite(depth_delta[prior_mask & fg]).any() else "ALIGNMENT_WEAK_BUT_USABLE_FOR_PRIOR",
        "silhouette_iou_prior_vs_foreground": support_iou,
        "prior_mask_pixels": int(prior_mask.sum()),
        "foreground_pixels": int(fg.sum()),
        "depth_delta_median_on_overlap": float(np.median(depth_delta[prior_mask & fg])) if (prior_mask & fg).any() else math.inf,
        "mesh_vertices": int(v15["vertices"].shape[0]),
        "mesh_faces": int(v15["faces"].shape[0]),
    }
    write_json(REPORTS / "V1230000_smplx_alignment.json", smplx_alignment)

    v117_conf = v117["confidence"]
    low_conf = v117_conf < np.percentile(v117_conf[fg], 40) if fg.any() else np.zeros_like(fg)
    smplx_disagree = (depth_delta > np.percentile(depth_delta[prior_mask & fg], 70)) if (prior_mask & fg).any() else np.zeros_like(fg)
    missing = fg & prior_mask & (low_conf | smplx_disagree)
    preserve = fg & ~missing
    bg_lock = ~fg
    if int(missing.sum()) < 800:
        missing = fg & prior_mask & binary_dilation(low_conf | smplx_disagree, iterations=2)
    mask_report = {
        "status": "V1240000_MISSING_SURFACE_INVENTORY",
        "human_foreground_pixels": int(fg.sum()),
        "missing_surface_pixels": int(missing.sum()),
        "preserve_pixels": int(preserve.sum()),
        "background_locked_pixels": int(bg_lock.sum()),
        "hand_risk_pixels": int(mm["hand_risk"].sum()),
        "head_hair_boundary_pixels": int(mm["head_hair_boundary"].sum()),
    }
    write_json(REPORTS / "V1240000_missing_surface_inventory.json", mask_report)

    base_v770_eval = evaluate(v117, v770, "V770")
    base_v770_delta = base_v770_eval["delta_vs_v117"]
    candidates: list[tuple[str, dict[str, np.ndarray], str]] = []
    rows: list[dict[str, Any]] = []

    def add_candidate(name: str, points: np.ndarray, source: str, conf: np.ndarray | None = None) -> None:
        pred = make_candidate(points, v117, conf)
        path = OUT / name / "predictions.npz"
        save_pred(path, pred)
        candidates.append((name, pred, source))

    # V125 body completion search.
    body_mask = missing & mm["body"] & ~mm["hand_risk"]
    for w in [0.15, 0.25, 0.4, 0.55, 0.7]:
        pts = blend(v117["points"], prior_points, body_mask, w)
        pts = blend(pts, smooth_points(pts, 3), body_mask, 0.15)
        add_candidate(f"V1250000_body_completion_w{int(w*100):02d}", pts, "V125_body_completion")
    write_json(REPORTS / "V1250000_body_completion_eval.json", {"status": "V1250000_BODY_COMPLETION_ATTEMPTS", "attempts": 5, "body_mask_pixels": int(body_mask.sum())})

    # V126 head/hair visual hull search.
    head_mask = (mm["head_face"] | mm["hairline"] | mm["head_hair_boundary"]) & fg
    boundary = mm["head_hair_boundary"] & fg
    for shell, blend_w in [(0.0, 0.25), (0.003, 0.35), (0.006, 0.45), (0.01, 0.55), (-0.003, 0.35)]:
        target = prior_points.copy()
        target = shell_offset(target, prior_normals, boundary, shell)
        pts = blend(v117["points"], target, head_mask, blend_w)
        pts = blend(pts, v770["points"], mm["hairline"] & fg, 0.35)
        add_candidate(f"V1260000_head_hair_shell_{len(candidates):03d}", pts, "V126_head_hair_visual_hull")
    write_json(REPORTS / "V1260000_head_hair_eval.json", {"status": "V1260000_HEAD_HAIR_ATTEMPTS", "attempts": 5, "head_hair_pixels": int(head_mask.sum()), "boundary_pixels": int(boundary.sum())})

    # V127 hand completion with object exclusion.
    hand_mask = (mm["left_hand"] | mm["right_hand"]) & fg
    safe_hand = hand_mask & ~mm["hand_risk"]
    expanded_hand = np.zeros_like(hand_mask)
    for view in range(hand_mask.shape[0]):
        expanded_hand[view] = binary_dilation(hand_mask[view], iterations=5) & fg[view] & ~mm["hand_risk"][view]
    for w in [0.2, 0.35, 0.5, 0.65, 0.8]:
        pts = blend(v117["points"], prior_points, expanded_hand, w)
        pts = blend(pts, v770["points"], safe_hand, 0.25)
        add_candidate(f"V1270000_hand_completion_w{int(w*100):02d}", pts, "V127_hand_completion")
    write_json(REPORTS / "V1270000_hand_completion_eval.json", {"status": "V1270000_HAND_ATTEMPTS", "attempts": 5, "hand_pixels": int(hand_mask.sum()), "safe_hand_pixels": int(safe_hand.sum()), "object_exclusion_pixels": int(mm["hand_risk"].sum())})

    # V128 renderer consistency for all generated candidates.
    renderer_report = {
        "status": "V1280000_RENDERER_CONSISTENCY",
        "rule": "depth is always points[..., 2], normal is always derived from the same point map",
        "candidate_count": len(candidates),
    }
    write_json(REPORTS / "V1280000_renderer_consistency_test.json", renderer_report)

    # V129/V131-V139 composition and ablations.
    for name, pred, source in candidates:
        metrics = candidate_metrics(name, pred, v117, v770, mm, missing)
        metrics["source_route"] = source
        rows.append(metrics)
    write_csv(REPORTS / "V1320000_body_ablation.csv", [r for r in rows if r["source_route"] == "V125_body_completion"])
    write_csv(REPORTS / "V1330000_head_hair_ablation.csv", [r for r in rows if r["source_route"] == "V126_head_hair_visual_hull"])
    write_csv(REPORTS / "V1340000_hand_ablation.csv", [r for r in rows if r["source_route"] == "V127_hand_completion"])
    write_json(REPORTS / "V1310000_search_manifest.json", {"status": "V1310000_SEARCH_MANIFEST", "candidate_count": len(candidates), "routes": sorted(set(r["source_route"] for r in rows))})

    # Build composed candidates from top safe-ish region specialists.
    def score_row(r: dict[str, Any]) -> float:
        d = r["delta_vs_v117"]
        return d["mean_quality"] + d["local_detail_quality"] + 0.6 * d["head_face_quality"] + 0.6 * d["hairline_quality"] + 0.45 * d["right_hand_quality"]

    route_best: dict[str, tuple[str, dict[str, np.ndarray], dict[str, Any]]] = {}
    for name, pred, source in candidates:
        row = next(r for r in rows if r["method"] == name)
        if source not in route_best or score_row(row) > score_row(route_best[source][2]):
            route_best[source] = (name, pred, row)

    composition_rows: list[dict[str, Any]] = []
    comp_candidates: list[tuple[str, dict[str, np.ndarray]]] = []
    combos = [
        ("V129_comp_body_head_hand", ["V125_body_completion", "V126_head_hair_visual_hull", "V127_hand_completion"]),
        ("V129_comp_head_hand", ["V126_head_hair_visual_hull", "V127_hand_completion"]),
        ("V129_comp_body_head", ["V125_body_completion", "V126_head_hair_visual_hull"]),
    ]
    for cname, sources in combos:
        pts = v117["points"].copy()
        # Start from V770 only for already proven diagnostic regions, not whole map.
        pts[mm["head_face"] | mm["hairline"] | mm["right_hand"]] = v770["points"][mm["head_face"] | mm["hairline"] | mm["right_hand"]]
        for src in sources:
            if src not in route_best:
                continue
            _, pred, _ = route_best[src]
            if src == "V125_body_completion":
                region_mask = body_mask
            elif src == "V126_head_hair_visual_hull":
                region_mask = head_mask
            else:
                region_mask = expanded_hand
            pts[region_mask] = pred["points"][region_mask]
        pred = make_candidate(pts, v117)
        save_pred(OUT / cname / "predictions.npz", pred)
        comp_candidates.append((cname, pred))
        row = candidate_metrics(cname, pred, v117, v770, mm, missing)
        composition_rows.append(row)
    write_csv(REPORTS / "V1350000_composition_ablation.csv", composition_rows)

    all_rows = rows + composition_rows
    all_rows.sort(key=score_row, reverse=True)
    best = all_rows[0] if all_rows else {}
    best_name = str(best.get("method", "NONE"))
    best_pred = None
    for name, pred, _ in candidates:
        if name == best_name:
            best_pred = pred
            break
    for name, pred in comp_candidates:
        if name == best_name:
            best_pred = pred
            break
    if best_pred is None:
        best_name = "V770_LOCKED_BEST_AVAILABLE"
        best_pred = v770
        best = candidate_metrics(best_name, v770, v117, v770, mm, missing)
    final_path = OUT / "V1290000_smplx_completion_candidate" / "predictions.npz"
    save_pred(final_path, best_pred)
    write_json(REPORTS / "V1290000_candidate_composition.json", {"status": "V1290000_CANDIDATE_COMPOSITION", "selected": best_name, "selected_metrics": best, "candidate_equals_v770": bool(np.max(np.abs(best_pred["points"] - v770["points"])) < 1e-12)})
    write_json(REPORTS / "V1390000_best_candidate_selection.json", {"status": "V1390000_BEST_CANDIDATE_SELECTION", "selected": best_name, "top5": all_rows[:5]})

    # V130 evaluation and board.
    board = make_board(v647, v117, v770, best_pred, mm)
    eval_rows = []
    for method, pred in [("V647_baseline", v647), ("V117_active", v117), ("V770_diagnostic", v770), ("V129_smplx_completion", best_pred)]:
        row = candidate_metrics(method, pred, v117, v770, mm, missing)
        eval_rows.append(row)
    write_csv(REPORTS / "V1300000_smplx_completion_eval.csv", eval_rows)
    write_json(REPORTS / "V1300000_smplx_completion_eval.json", {"status": "V1300000_SMPLX_COMPLETION_EVAL", "board": str(board), "rows": eval_rows})

    # V140 strict gate.
    best_metrics = candidate_metrics(best_name, best_pred, v117, v770, mm, missing)
    d770 = best_metrics["delta_vs_v770"]
    hard = {
        "new_candidate_not_equal_v770": not best_metrics["changed_equals_v770"],
        "full_body_no_regression_vs_v770": d770["full_body_quality"] >= -1e-6,
        "human_valid_point_ratio_not_worse": best_metrics["valid_human_point_ratio"] >= candidate_metrics("V770_tmp", v770, v117, v770, mm, missing)["valid_human_point_ratio"] - 1e-6,
        "missing_surface_fill_ratio_positive": best_metrics["missing_surface_fill_ratio"] > 0.01,
        "head_or_hair_beats_v770": d770["head_face_quality"] > 1e-6 or d770["hairline_quality"] > 1e-6,
        "right_hand_not_regressed": d770["right_hand_quality"] >= -1e-6,
        "left_hand_not_preserved_only": best_metrics["delta_vs_v117"]["left_hand_quality"] > 1e-6,
        "depth_point_normal_consistent": best_metrics["depth_point_z_error"] < 1e-8 and best_metrics["normal_nonzero_ratio"] > 0.95,
        "background_leakage_low": best_metrics["outside_leakage_ratio"] < 0.01,
        "beats_v770_mean_local": best_metrics["beats_v770_mean_local"],
    }
    hard_pass = all(hard.values())
    write_json(REPORTS / "V1400000_strict_hard_gate.json", {"status": "V1400000_STRICT_HARD_GATE", "pass": hard_pass, "requirements": hard, "selected": best_name, "selected_metrics": best_metrics})
    write_text(REPORTS / "V1400000_strict_hard_gate.md", "# V1400000 Strict Hard Gate\n\n" + ("PASS\n" if hard_pass else "FAIL\n"))

    if hard_pass:
        final_status = "V1800000_REVIEW_READY_NOT_PROMOTED"
        write_text(REPORTS / "V1600000_review_summary.md", "# V1600000 Review Summary\n\nStrict hard gate passed. This is review-ready only, not promoted.\n")
    else:
        final_status = "V1800000_ROUTE_EXHAUSTED_WITH_FAILURE_ANALYSIS"
        failure = {
            "status": "V1500000_FAILURE_ATTRIBUTION",
            "failed_requirements": [k for k, v in hard.items() if not v],
            "classification": [
                "SMPLX_PRIOR_TOO_COARSE_OR_MISALIGNED_FOR_DIRECT_COMPLETION" if support_iou < 0.2 else "SMPLX_PRIOR_AVAILABLE_BUT_NOT_STRONG_ENOUGH",
                "NO_ROUTE_BEATS_V770",
                "HAND_REQUIRES_EXTERNAL_KEYPOINT_OR_SEGMENTATION" if not hard["left_hand_not_preserved_only"] else "HAND_PARTIAL",
                "STRONGER_DENSE_TEACHER_REQUIRED",
            ],
            "next_action": "Need external semantic/dense supervision: hand keypoints/parsing, stronger hair segmentation, or true dense 3D teacher. Current SMPL-X anchored per-view prior completion cannot beat V770 without regressions.",
        }
        write_json(REPORTS / "V1500000_failure_attribution.json", failure)
        write_text(REPORTS / "V1500000_next_action.md", "# V1500000 Next Action\n\nThe SMPL-X anchored completion search generated real non-V770 candidates, but none passed the strict gate. Next route needs stronger external semantic/dense supervision.\n")

    runtime = time.time() - start
    scan = process_scan()
    final = {
        "status": final_status,
        "active_candidate": "V11700_gap_reduction_branch_520",
        "selected": best_name,
        "hard_gate_pass": hard_pass,
        "runtime_seconds": runtime,
        "generated_npz_count": len(candidates) + len(comp_candidates) + 1,
        "search_candidate_count": len(candidates),
        "composition_candidate_count": len(comp_candidates),
        "any_candidate_differs_from_v770": any(not r["changed_equals_v770"] for r in all_rows),
        "strict_registry_written": False,
        "candidate_replaced": False,
        "mentor_package_generated": False,
        "v50_v50r2_modified": False,
        "process_scan_clean": not scan["residual_training_or_modal_worker_detected"],
    }
    write_json(REPORTS / "V1800000_process_and_modal_scan.json", scan)
    write_json(REPORTS / "V1800000_final_status.json", final)
    write_text(
        REPORTS / "V1800000_final_summary.md",
        f"# V1800000 Final Summary\n\nStatus: `{final_status}`.\n\nGenerated `{final['generated_npz_count']}` NPZ candidates. Active candidate remains `V11700_gap_reduction_branch_520`; no promotion, strict registry, or V50/V50R2 edit was made.\n",
    )

    include = [
        REPORTS / "V1210000_anti_fast_return_contract.json",
        REPORTS / "V1210000_runtime_budget.md",
        REPORTS / "V1220000_asset_reaudit.json",
        REPORTS / "V1220000_prediction_internal_consistency.md",
        REPORTS / "V1230000_smplx_alignment.json",
        REPORTS / "V1240000_missing_surface_inventory.json",
        REPORTS / "V1250000_body_completion_eval.json",
        REPORTS / "V1260000_head_hair_eval.json",
        REPORTS / "V1270000_hand_completion_eval.json",
        REPORTS / "V1280000_renderer_consistency_test.json",
        REPORTS / "V1290000_candidate_composition.json",
        final_path,
        REPORTS / "V1300000_smplx_completion_eval.json",
        REPORTS / "V1300000_smplx_completion_eval.csv",
        board,
        REPORTS / "V1310000_search_manifest.json",
        REPORTS / "V1320000_body_ablation.csv",
        REPORTS / "V1330000_head_hair_ablation.csv",
        REPORTS / "V1340000_hand_ablation.csv",
        REPORTS / "V1350000_composition_ablation.csv",
        REPORTS / "V1390000_best_candidate_selection.json",
        REPORTS / "V1400000_strict_hard_gate.json",
        REPORTS / "V1400000_strict_hard_gate.md",
        REPORTS / "V1500000_failure_attribution.json",
        REPORTS / "V1500000_next_action.md",
        REPORTS / "V1600000_review_summary.md",
        REPORTS / "V1800000_process_and_modal_scan.json",
        REPORTS / "V1800000_final_status.json",
        REPORTS / "V1800000_final_summary.md",
    ]
    zname = "V1800000_review_ready_not_promoted_bundle.zip" if hard_pass else "V1800000_route_exhausted_failure_analysis_bundle.zip"
    manifest = zip_manifest(ARCHIVE / zname, include)
    write_json(REPORTS / "V1800000_package_manifest.json", manifest)
    print(json.dumps({"status": final_status, "hard_gate_pass": hard_pass, "selected": best_name, "bundle": manifest}, indent=2))


if __name__ == "__main__":
    main()
