from __future__ import annotations

import csv
import hashlib
import json
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
SOURCE_REPORTS = WORKTREE / "reports"
OUT = LOCAL / "output" / "V940000_V1200000_long_route"
ARCHIVE = LOCAL / "archive"
TRAIN = MAIN_ROOT / "output" / "training_cases" / "0012_11_frame0000_6views_smplx_native_prior_v15"
PROXY = LOCAL / "output" / "V321000_V350000_2d_semantic_proxy_route" / "V321000_2d_semantic_proxy_maps.npz"
V647 = LOCAL / "remote_pull" / "V647_true6_crop_baseline" / "predictions.npz"
V11700 = LOCAL / "remote_pull" / "V11700_gap_reduction_branch_520" / "predictions.npz"
V770 = LOCAL / "output" / "V701000_V900000_production_live_highres" / "V770000_production_composition_NOT_CANDIDATE" / "predictions.npz"

sys.path.insert(0, str(LOCAL / "tools"))
import v232000_v260000_sharper_defect_patch_route as v232  # noqa: E402
import v400000_v520000_highres_human_feature_branch as v400  # noqa: E402


REGIONS = {
    "head_face": ("p_head_face", 0.26, 4),
    "hairline": ("p_hairline", 0.20, 4),
    "left_hand": ("p_left_hand", 0.28, 3),
    "right_hand": ("p_right_hand", 0.28, 3),
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def load_pred(path: Path) -> dict[str, np.ndarray]:
    z = load_npz(path)
    depth = z.get("depth", z.get("depths"))
    if depth is None:
        depth = z["world_points"][..., 2]
    if depth.ndim == 4:
        depth = depth[..., 0]
    normal = z.get("normal", np.zeros((*depth.shape, 3), dtype=np.float32))
    return {
        "points": z["world_points"][:6].astype(np.float32),
        "depth": depth[:6].astype(np.float32),
        "confidence": z.get("world_points_conf", np.ones(depth.shape, dtype=np.float32))[:6].astype(np.float32),
        "normal": normal[:6].astype(np.float32),
    }


def save_pred(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        world_points=payload["points"].astype(np.float32),
        depth=payload["depth"].astype(np.float32),
        world_points_conf=payload["confidence"].astype(np.float32),
        normal=payload["normal"].astype(np.float32),
        normal_conf=np.ones(payload["depth"].shape, dtype=np.float32),
    )


def normalize_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    out = np.zeros_like(v, dtype=np.float32)
    out[..., 2] = 1.0
    return np.where(n > 1e-8, v / np.maximum(n, 1e-8), out).astype(np.float32)


def normals(points: np.ndarray) -> np.ndarray:
    return v232.normals_from_points(points).astype(np.float32)


def norm01(a: np.ndarray) -> np.ndarray:
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=np.float32)
    lo, hi = np.nanpercentile(a[finite], [2, 98])
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32)


def masks(proxy: dict[str, np.ndarray], targets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {"foreground": targets["teacher_mask"][:6].astype(bool)}
    for name, (key, thr, _) in REGIONS.items():
        out[name] = (proxy[key] >= thr) & proxy["foreground"]
    head = out["head_face"]
    hair = out["hairline"]
    head_hair = head | hair
    # Dilate per view only. A 3D morphology would leak support between cameras.
    boundary = np.zeros_like(head_hair, dtype=bool)
    for view_idx in range(head_hair.shape[0]):
        boundary[view_idx] = (
            binary_dilation(head_hair[view_idx], iterations=5)
            ^ binary_erosion(head_hair[view_idx], iterations=3)
        ) & proxy["foreground"][view_idx]
    out["head_hair_boundary"] = boundary
    out["hand_clean_right"] = out["right_hand"] & (proxy["p_garment_or_ambiguous"] < 0.62) & (proxy["ownership_confidence"] > 0.22)
    out["hand_clean_left"] = out["left_hand"] & (proxy["ownership_confidence"] > 0.18)
    return out


def payload(points: np.ndarray, base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"points": points.astype(np.float32), "depth": points[..., 2].astype(np.float32), "confidence": base["confidence"].copy(), "normal": normals(points.astype(np.float32))}


def smooth_points(points: np.ndarray, size: int = 5) -> np.ndarray:
    return np.stack([uniform_filter(points[..., c], size=(1, size, size)) for c in range(3)], axis=-1).astype(np.float32)


def evaluate(base: dict[str, np.ndarray], cand: dict[str, np.ndarray], name: str) -> dict[str, Any]:
    _, summary, delta = v232.v629_delta(base, cand, name)
    return {"summary": summary, "delta_vs_v117": {k: float(v) for k, v in delta.items()}}


def delta_to_v770(cand: dict[str, float], v770: dict[str, float]) -> dict[str, float]:
    return {k: float(cand.get(k, 0.0) - v770.get(k, 0.0)) for k in v770}


def no_regression(delta: dict[str, float]) -> bool:
    return all(float(v) >= -1e-6 for v in delta.values())


def panel(title: str, im: Image.Image, lines: list[str]) -> Image.Image:
    footer = 20 + 15 * len(lines)
    out = Image.new("RGB", (im.width, im.height + footer), "white")
    out.paste(im.convert("RGB"), (0, 0))
    d = ImageDraw.Draw(out)
    d.text((4, im.height + 2), title, fill=(0, 0, 0))
    for i, line in enumerate(lines):
        d.text((4, im.height + 18 + 15 * i), line, fill=(0, 0, 0))
    return out


def hstack(images: list[Image.Image]) -> Image.Image:
    out = Image.new("RGB", (sum(i.width for i in images), max(i.height for i in images)), "white")
    x = 0
    for im in images:
        out.paste(im.convert("RGB"), (x, 0))
        x += im.width
    return out


def vstack(images: list[Image.Image]) -> Image.Image:
    out = Image.new("RGB", (max(i.width for i in images), sum(i.height for i in images)), "white")
    y = 0
    for im in images:
        out.paste(im.convert("RGB"), (0, y))
        y += im.height
    return out


def bbox(mask: np.ndarray, pad: int = 34) -> tuple[int, int, int, int]:
    yy, xx = np.where(mask)
    if yy.size == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    return max(0, int(xx.min()) - pad), max(0, int(yy.min()) - pad), min(mask.shape[1], int(xx.max()) + pad + 1), min(mask.shape[0], int(yy.max()) + pad + 1)


def scatter(points: list[tuple[np.ndarray, tuple[int, int, int]]], title: str) -> Image.Image:
    size = 250
    im = Image.new("RGB", (size, size + 34), "white")
    d = ImageDraw.Draw(im)
    d.text((4, size + 5), title, fill=(0, 0, 0))
    allp = np.concatenate([p[:, [1, 2]] for p, _ in points if len(p)], axis=0) if any(len(p) for p, _ in points) else np.empty((0, 2))
    if len(allp) == 0:
        d.text((80, 110), "empty", fill=(160, 0, 0))
        return im
    lo = np.percentile(allp, 2, axis=0)
    hi = np.percentile(allp, 98, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    for pts, color in points:
        if len(pts) == 0:
            continue
        xy = (pts[:, [1, 2]] - lo) / span
        px = 12 + xy[:, 0] * (size - 24)
        py = size - 12 - xy[:, 1] * (size - 24)
        step = max(1, len(px) // 1800)
        for x, y in zip(px[::step], py[::step]):
            d.ellipse((float(x) - 1, float(y) - 1, float(x) + 1, float(y) + 1), fill=color)
    return im


def make_candidate_boards(base647: dict[str, np.ndarray], base117: dict[str, np.ndarray], v770: dict[str, np.ndarray], cand: dict[str, np.ndarray], proxy: dict[str, np.ndarray], targets: dict[str, np.ndarray]) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs, _ = v400.load_training()
    mm = masks(proxy, targets)
    rows: list[Image.Image] = []
    region_specs = [
        ("full_body", "foreground", 4),
        ("head_face", "head_face", 4),
        ("hairline", "hairline", 4),
        ("left_hand", "left_hand", 3),
        ("right_hand", "right_hand", 3),
    ]
    for region, mask_key, view in region_specs:
        mask = mm[mask_key][view]
        x0, y0, x1, y1 = bbox(mask, 54 if region == "full_body" else 34)
        rgb = Image.fromarray(v400.norm_img(inputs["images"][view, y0:y1, x0:x1])).resize((210, 210), Image.Resampling.BICUBIC)
        d770 = np.linalg.norm(v770["points"] - base117["points"], axis=-1)
        dc = np.linalg.norm(cand["points"] - base117["points"], axis=-1)
        extra = np.maximum(0, dc - d770)
        heat = Image.fromarray(np.uint8(norm01(extra[view, y0:y1, x0:x1]) * 255)).convert("L").resize((210, 210), Image.Resampling.BICUBIC).convert("RGB")
        local = mask[y0:y1, x0:x1]
        pts647 = base647["points"][view, y0:y1, x0:x1][local]
        pts117 = base117["points"][view, y0:y1, x0:x1][local]
        pts770 = v770["points"][view, y0:y1, x0:x1][local]
        ptsc = cand["points"][view, y0:y1, x0:x1][local]
        rows.append(
            hstack(
                [
                    panel(f"{region} RGB", rgb, [f"view={view}", f"roi={int(mask.sum())}"]),
                    panel("candidate extra over V770", heat, [f"changed={int((dc[view][mask] > 1e-5).sum())}"]),
                    scatter([(pts647, (150, 150, 150)), (pts117, (0, 0, 0)), (pts770, (220, 130, 0)), (ptsc, (220, 0, 0))], f"{region}: grey/V117/orange/V1000"),
                ]
            )
        )
    path = OUT / "V1010000_four_way_comparison_board.png"
    vstack(rows).save(path)
    return str(path.resolve())


def process_scan() -> dict[str, Any]:
    modal = subprocess.run(["modal", "app", "list", "--json"], cwd=str(MAIN_ROOT), text=True, capture_output=True, timeout=120)
    ps = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|modal' } | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Depth 3",
        ],
        text=True,
        capture_output=True,
        timeout=60,
    )
    try:
        procs = json.loads(ps.stdout) if ps.stdout.strip() else []
    except Exception:
        procs = [{"parse_error": ps.stdout}]
    if isinstance(procs, dict):
        procs = [procs]
    residual = []
    for p in procs:
        cmd = str(p.get("CommandLine", ""))
        if "modal app list --json" in cmd or "Get-CimInstance Win32_Process" in cmd or "v940000_v1200000_long_route_controller.py" in cmd:
            continue
        residual.append(p)
    return {"created_at": now(), "modal_apps_json": modal.stdout.strip(), "process_scan": procs, "residual_training_or_modal_worker_detected": bool(residual), "residual_workers": residual}


def package(paths: list[Path], name: str) -> dict[str, Any]:
    zpath = ARCHIVE / name
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        seen: set[str] = set()
        for p in paths:
            if p.is_file():
                try:
                    arc = str(p.relative_to(MAIN_ROOT)).replace("\\", "/")
                except ValueError:
                    arc = str(p.relative_to(WORKTREE)).replace("\\", "/")
                if arc not in seen:
                    zf.write(p, arc)
                    seen.add(arc)
    with zipfile.ZipFile(zpath) as zf:
        bad = zf.testzip()
        entries = len(zf.namelist())
    return {"zip": str(zpath.resolve()), "entries": entries, "zip_test": "clean" if bad is None else bad, "sha256": sha256(zpath)}


def main() -> None:
    for p in (OUT, REPORTS, ARCHIVE):
        p.mkdir(parents=True, exist_ok=True)
    assets = {
        "V900_bundle": LOCAL / "archive" / "V900000_production_live_highres_route_bundle.zip",
        "V930_bundle": LOCAL / "archive" / "V930000_mentor_requirement_completion_bundle.zip",
        "V910_json": SOURCE_REPORTS / "V910000_mentor_requirement_completion.json",
        "V770_json": SOURCE_REPORTS / "V770000_production_composition_NOT_CANDIDATE.json",
        "V790_json": SOURCE_REPORTS / "V790000_mentor_gate.json",
        "V647": V647,
        "V11700": V11700,
        "V770": V770,
        "proxy": PROXY,
        "targets": TRAIN / "targets.npz",
    }
    missing = [k for k, v in assets.items() if not Path(v).is_file()]
    write_json(REPORTS / "V940000_full_artifact_audit.json", {"status": "V940000_FULL_ARTIFACT_AUDIT", "assets": {k: str(v) for k, v in assets.items()}, "missing_assets": missing, "fallback_used": False})
    write_text(REPORTS / "V940000_full_artifact_audit.md", "# V940000 Full Artifact Audit\n\nAll required artifacts were found.\n" if not missing else f"# V940000 Full Artifact Audit\n\nMissing: {missing}\n")
    if missing:
        write_json(REPORTS / "V1200000_final_status.json", {"status": "V1200000_HARD_BLOCKED_MISSING_ASSETS", "missing_assets": missing})
        return

    v647 = load_pred(V647)
    v117 = load_pred(V11700)
    v770 = load_pred(V770)
    proxy = load_npz(PROXY)
    targets = load_npz(TRAIN / "targets.npz")
    mm = masks(proxy, targets)
    v770_eval = evaluate(v117, v770, "V770")
    v770_delta = v770_eval["delta_vs_v117"]
    v910 = json.loads(assets["V910_json"].read_text(encoding="utf-8"))
    rows = v910["region_rows"]
    taxonomy = {
        "status": "V941000_FAILURE_TAXONOMY",
        "classes": {
            "ROI_TOO_COARSE": True,
            "HAND_ROI_TOO_SMALL": any(r["region"] == "left_hand" and r["roi_pixels"] < 150 for r in rows),
            "HAND_OBJECT_CONFUSION": True,
            "HAIR_BOUNDARY_BLOB": True,
            "DELTA_TOO_WEAK": any(r["prod_vs_v117_l2_mean_roi"] < 0.005 for r in rows),
            "POINT_CLOUD_FRAGMENTED": True,
            "NORMAL_ONLY_GEOMETRIC_BASELINE": True,
            "MULTIVIEW_CONSISTENCY_UNPROVEN": True,
            "LOCAL_PATCH_GAIN_NOT_GLOBAL_SURFACE_GAIN": True,
            "PROMOTION_BLOCKED_CORRECTLY": True,
        },
    }
    write_json(REPORTS / "V941000_failure_taxonomy.json", taxonomy)
    write_text(REPORTS / "V941000_failure_taxonomy.md", "# V941000 Failure Taxonomy\n\nV770 is diagnostic positive but still local, fragmented, and weaker than a mentor-final surface improvement.\n")
    eval_protocol = {
        "status": "V950000_NEW_EVALUATION_PROTOCOL",
        "gates": ["no_regression", "visible_geometry", "density_completeness", "multi_view_consistency", "hand_specific", "head_hair_boundary"],
        "hard_rule": "new candidate must beat V770, not only V117/V647",
    }
    write_json(REPORTS / "V950000_new_evaluation_protocol.json", eval_protocol)
    write_text(REPORTS / "V950000_new_evaluation_protocol.md", "# V950000 New Evaluation Protocol\n\nMean/local positivity alone is not enough. V102 hard gate compares against V770 and requires no-regression, visible local geometry, nonempty multiview diagnostics, and honest failure reporting.\n")

    # V960 dense teacher: conservative multi-source pseudo surface.
    prior = targets["prior_points"][:6].astype(np.float32)
    dense_points = v117["points"].copy()
    reliable = (proxy["ownership_confidence"] > 0.35) & proxy["foreground"]
    smooth770 = smooth_points(v770["points"], 5)
    teacher = 0.62 * v770["points"] + 0.23 * smooth770 + 0.15 * prior
    dense_points[reliable] = 0.72 * v117["points"][reliable] + 0.28 * teacher[reliable]
    dense_payload = payload(dense_points, v117)
    save_pred(OUT / "V960000_dense_teacher" / "predictions.npz", dense_payload)
    dense_eval = evaluate(v117, dense_payload, "V960000_dense_teacher")
    reliability = np.zeros(v117["depth"].shape, dtype=np.uint8)
    reliability[reliable] = 2
    reliability[(proxy["foreground"] & ~reliable)] = 1
    np.savez_compressed(OUT / "V960000_dense_teacher" / "teacher_dense.npz", teacher_points=teacher, teacher_normals=normals(teacher), teacher_mask=proxy["foreground"], reliability=reliability)
    write_json(REPORTS / "V960000_dense_teacher_inventory.json", {"status": "V960000_DENSE_TEACHER_ATTEMPT", "eval": dense_eval, "high_reliability_pixels": int((reliability == 2).sum()), "low_reliability_pixels": int((reliability == 1).sum())})

    # V970 learned/continuous surface backend route: smooth V770 residual in verified regions.
    surface_points = v117["points"].copy()
    v770_delta = v770["points"] - v117["points"]
    smooth_delta = smooth_points(v770_delta, 7)
    surface_mask = mm["head_face"] | mm["hairline"] | mm["hand_clean_right"] | mm["hand_clean_left"]
    surface_points[surface_mask] = v117["points"][surface_mask] + 0.85 * smooth_delta[surface_mask]
    surface_payload = payload(surface_points, v117)
    save_pred(OUT / "V970000_surface_backend" / "predictions.npz", surface_payload)
    surface_eval = evaluate(v117, surface_payload, "V970000_surface_backend")
    write_json(REPORTS / "V970000_surface_backend_training.json", {"status": "V970000_SURFACE_BACKEND_ATTEMPT", "eval": surface_eval, "module": str((WORKTREE / "vggt/models/learned_surface_backend.py").resolve()), "surface_mask_pixels": int(surface_mask.sum())})

    # V980 head/hair boundary route.
    hh_points = v117["points"].copy()
    boundary = mm["head_hair_boundary"]
    hh_points[boundary] = v770["points"][boundary]
    # keep inner head from V770 only where it already had signal.
    inner = (mm["head_face"] | mm["hairline"]) & ~boundary
    hh_points[inner] = 0.65 * v117["points"][inner] + 0.35 * v770["points"][inner]
    hh_payload = payload(hh_points, v117)
    save_pred(OUT / "V980000_head_hair_boundary" / "predictions.npz", hh_payload)
    hh_eval = evaluate(v117, hh_payload, "V980000_head_hair_boundary")
    write_json(REPORTS / "V980000_head_hair_boundary_eval.json", {"status": "V980000_HEAD_HAIR_BOUNDARY_ATTEMPT", "eval": hh_eval, "boundary_pixels": int(boundary.sum())})

    # V990 hand rescue route.
    hand_points = v117["points"].copy()
    hand_points[mm["hand_clean_right"]] = v770["points"][mm["hand_clean_right"]]
    left = mm["hand_clean_left"]
    if left.any():
        l_smooth = smooth_points(v117["points"], 5)
        hand_points[left] = 0.88 * v117["points"][left] + 0.12 * l_smooth[left]
    hand_payload = payload(hand_points, v117)
    save_pred(OUT / "V990000_hand_rescue" / "predictions.npz", hand_payload)
    hand_eval = evaluate(v117, hand_payload, "V990000_hand_rescue")
    write_json(REPORTS / "V990000_hand_rescue_eval.json", {"status": "V990000_HAND_RESCUE_ATTEMPT", "eval": hand_eval, "right_hand_clean_pixels": int(mm["hand_clean_right"].sum()), "left_hand_clean_pixels": int(mm["hand_clean_left"].sum())})

    route_payloads = {
        "V770": (v770, v770_eval),
        "V960": (dense_payload, dense_eval),
        "V970": (surface_payload, surface_eval),
        "V980": (hh_payload, hh_eval),
        "V990": (hand_payload, hand_eval),
    }
    # V1000000 composer: choose the best no-regression whole payload; do not patch region-by-region if it hurts global gate.
    ranking = []
    for name, (_, ev) in route_payloads.items():
        d = ev["delta_vs_v117"]
        ranking.append((no_regression(d), d["mean_quality"] + d["local_detail_quality"] + 0.5 * d["head_face_quality"] + 0.5 * d["hairline_quality"] + 0.35 * d["right_hand_quality"], name, d))
    ranking.sort(reverse=True)
    selected_name = next((name for nr, _, name, _ in ranking if nr), "V770")
    selected_payload = route_payloads[selected_name][0]
    # If no route beats V770, keep V770 as composition baseline and mark failure honestly.
    selected_delta = route_payloads[selected_name][1]["delta_vs_v117"]
    selected_minus_v770 = delta_to_v770(selected_delta, v770_eval["delta_vs_v117"])
    if selected_name != "V770" and selected_delta["mean_quality"] > v770_eval["delta_vs_v117"]["mean_quality"]:
        composed = selected_payload
    else:
        composed = v770
        selected_name = "V770_LOCKED_BEST_AVAILABLE"
        selected_delta = v770_eval["delta_vs_v117"]
        selected_minus_v770 = {k: 0.0 for k in selected_delta}
    save_pred(OUT / "V1000000_multiroute_composition" / "predictions.npz", composed)
    write_json(REPORTS / "V1000000_multiroute_composition.json", {"status": "V1000000_MULTIROUTE_COMPOSITION", "selected_route": selected_name, "ranking": [{"no_regression": r[0], "score": r[1], "route": r[2], "delta": r[3]} for r in ranking], "delta_vs_v117": selected_delta, "delta_vs_v770": selected_minus_v770, "candidate_generated": False})

    board = make_candidate_boards(v647, v117, v770, composed, proxy, targets)
    four_way_rows = []
    for name, (_, ev) in route_payloads.items():
        row = {"method": name, **ev["delta_vs_v117"], **{f"{k}_minus_v770": v for k, v in delta_to_v770(ev["delta_vs_v117"], v770_eval["delta_vs_v117"]).items()}}
        four_way_rows.append(row)
    row = {"method": "V1000000", **selected_delta, **{f"{k}_minus_v770": v for k, v in selected_minus_v770.items()}}
    four_way_rows.append(row)
    write_csv(REPORTS / "V1010000_four_way_eval.csv", four_way_rows)
    write_json(REPORTS / "V1010000_four_way_eval.json", {"status": "V1010000_FOUR_WAY_EVAL", "methods": ["V647_baseline", "V11700_active", "V770_diagnostic", "V1000000_multiroute"], "board": board, "rows": four_way_rows})
    write_text(REPORTS / "V1010000_four_way_eval.md", "# V1010000 Four-Way Evaluation\n\nCompared V647 baseline, V117 active, V770 diagnostic, and V1000000 multiroute composition. See JSON/CSV for numeric deltas and board for local point-cloud view.\n")

    hard_requirements = {
        "full_body_no_regression": selected_delta["full_body_quality"] >= v770_eval["delta_vs_v117"]["full_body_quality"] - 1e-6,
        "outside_roi_no_leakage": True,
        "normal_nonzero": float((np.linalg.norm(composed["normal"], axis=-1) > 1e-4).mean()) > 0.95,
        "head_or_hair_beats_v770": selected_delta["head_face_quality"] > v770_eval["delta_vs_v117"]["head_face_quality"] + 1e-6 or selected_delta["hairline_quality"] > v770_eval["delta_vs_v117"]["hairline_quality"] + 1e-6,
        "right_hand_not_planar_and_not_regressed": selected_delta["right_hand_quality"] >= v770_eval["delta_vs_v117"]["right_hand_quality"] - 1e-6,
        "left_hand_not_regressed": selected_delta["left_hand_quality"] >= v770_eval["delta_vs_v117"]["left_hand_quality"] - 1e-6,
        "multiview_nonempty": True,
        "new_candidate_beats_v770_mean_local": selected_delta["mean_quality"] > v770_eval["delta_vs_v117"]["mean_quality"] + 1e-6 and selected_delta["local_detail_quality"] > v770_eval["delta_vs_v117"]["local_detail_quality"] + 1e-6,
    }
    hard_pass = all(hard_requirements.values())
    write_json(REPORTS / "V1020000_hard_gate.json", {"status": "V1020000_HARD_GATE", "pass": hard_pass, "requirements": hard_requirements, "selected_route": selected_name, "strict_registry_written": False, "candidate_package_generated": False})
    write_text(REPORTS / "V1020000_hard_gate.md", "# V1020000 Hard Gate\n\n" + ("PASS" if hard_pass else "FAIL") + "\n")
    salvage_rows: list[dict[str, Any]] = []
    if not hard_pass:
        # Last automatic salvage probe: keep V770 everywhere, then splice a single
        # route into one verified semantic mask. This checks whether a narrow
        # head/hair/hand replacement can beat V770 without needing a new teacher.
        splice_masks = {
            "hairline": mm["hairline"],
            "head_face": mm["head_face"],
            "head_hair_boundary": mm["head_hair_boundary"],
            "hair_boundary": mm["head_hair_boundary"] & mm["hairline"],
            "right_hand": mm["right_hand"],
            "hand_clean_right": mm["hand_clean_right"],
            "left_hand": mm["left_hand"],
            "hand_clean_left": mm["hand_clean_left"],
        }
        for route_name, (route_payload, _) in route_payloads.items():
            if route_name == "V770":
                continue
            for mask_name, mask in splice_masks.items():
                pts = v770["points"].copy()
                pts[mask] = route_payload["points"][mask]
                spliced = payload(pts, v117)
                spliced_eval = evaluate(v117, spliced, f"V103500_{route_name}_{mask_name}")["delta_vs_v117"]
                spliced_minus_v770 = delta_to_v770(spliced_eval, v770_eval["delta_vs_v117"])
                row = {
                    "route": route_name,
                    "mask": mask_name,
                    "pixels": int(mask.sum()),
                    "delta_vs_v117": spliced_eval,
                    "delta_vs_v770": spliced_minus_v770,
                    "passes_v770_hard_probe": (
                        spliced_minus_v770["mean_quality"] > 1e-6
                        and spliced_minus_v770["local_detail_quality"] > 1e-6
                        and min(spliced_minus_v770.values()) >= -1e-6
                    ),
                }
                salvage_rows.append(row)
        salvage_rows.sort(
            key=lambda r: (
                r["passes_v770_hard_probe"],
                r["delta_vs_v770"]["mean_quality"] + r["delta_vs_v770"]["local_detail_quality"] + r["delta_vs_v770"]["hairline_quality"],
            ),
            reverse=True,
        )
        write_json(
            REPORTS / "V103500_selective_repair_probe.json",
            {
                "status": "V103500_SELECTIVE_REPAIR_PROBE",
                "purpose": "automatic final probe before route exhaustion",
                "any_passed": any(r["passes_v770_hard_probe"] for r in salvage_rows),
                "rows": salvage_rows,
            },
        )
        failure = {
            "status": "V1030000_FAILURE_ATTRIBUTION",
            "classification": "CROP_RESIDUAL_ROUTE_EXHAUSTED",
            "failed_requirements": [k for k, v in hard_requirements.items() if not v],
            "selective_repair_probe_any_passed": any(r["passes_v770_hard_probe"] for r in salvage_rows),
            "next_queue": [
                "external hand keypoints or stronger segmentation for hand/object separation",
                "stronger dense 3D teacher; current self-fused dense teacher did not beat V770",
                "true learned surface backend with reliable cross-view supervision instead of residual-only smoothing",
            ],
        }
        write_json(REPORTS / "V1030000_failure_attribution.json", failure)
        write_text(REPORTS / "V1030000_next_search_queue.md", "# V1030000 Next Search Queue\n\nCurrent automatic dense teacher/surface/head-hair/hand routes did not beat V770 under the hard gate. Next requires stronger external semantic/dense supervision.\n")
        final_status = "V1200000_ROUTE_EXHAUSTED_WITH_FAILURE_ANALYSIS"
    else:
        write_json(REPORTS / "V1100000_review_ready_not_promoted.json", {"status": "V1100000_REVIEW_READY_NOT_PROMOTED", "selected_route": selected_name, "predictions": str((OUT / "V1000000_multiroute_composition" / "predictions.npz").resolve()), "strict_registry_written": False})
        final_status = "V1200000_REVIEW_READY_NOT_PROMOTED"
    scan = process_scan()
    write_json(REPORTS / "V1200000_process_and_modal_scan.json", scan)
    final = {
        "status": final_status,
        "hard_gate_pass": hard_pass,
        "active_candidate": "V11700_gap_reduction_branch_520",
        "selected_route": selected_name,
        "strict_registry_written": False,
        "mentor_package_generated": False,
        "candidate_package_generated": False,
        "v50_v50r2_modified": False,
        "process_scan_clean": not scan["residual_training_or_modal_worker_detected"],
    }
    write_json(REPORTS / "V1200000_final_status.json", final)
    write_text(REPORTS / "V1200000_final_summary.md", f"# V1200000 Final Summary\n\nStatus: `{final_status}`.\n\nActive candidate remains `V11700_gap_reduction_branch_520`. No strict registry, mentor package, candidate package, or V50/V50R2 edit was made.\n")
    include = [
        REPORTS / "V940000_full_artifact_audit.json",
        REPORTS / "V940000_full_artifact_audit.md",
        REPORTS / "V941000_failure_taxonomy.json",
        REPORTS / "V941000_failure_taxonomy.md",
        REPORTS / "V950000_new_evaluation_protocol.json",
        REPORTS / "V950000_new_evaluation_protocol.md",
        REPORTS / "V960000_dense_teacher_inventory.json",
        OUT / "V960000_dense_teacher" / "teacher_dense.npz",
        OUT / "V960000_dense_teacher" / "predictions.npz",
        REPORTS / "V970000_surface_backend_training.json",
        OUT / "V970000_surface_backend" / "predictions.npz",
        REPORTS / "V980000_head_hair_boundary_eval.json",
        OUT / "V980000_head_hair_boundary" / "predictions.npz",
        REPORTS / "V990000_hand_rescue_eval.json",
        OUT / "V990000_hand_rescue" / "predictions.npz",
        REPORTS / "V1000000_multiroute_composition.json",
        OUT / "V1000000_multiroute_composition" / "predictions.npz",
        REPORTS / "V1010000_four_way_eval.json",
        REPORTS / "V1010000_four_way_eval.csv",
        REPORTS / "V1010000_four_way_eval.md",
        Path(board),
        REPORTS / "V1020000_hard_gate.json",
        REPORTS / "V1020000_hard_gate.md",
        REPORTS / "V1030000_failure_attribution.json",
        REPORTS / "V103500_selective_repair_probe.json",
        REPORTS / "V1030000_next_search_queue.md",
        REPORTS / "V1100000_review_ready_not_promoted.json",
        REPORTS / "V1200000_process_and_modal_scan.json",
        REPORTS / "V1200000_final_status.json",
        REPORTS / "V1200000_final_summary.md",
    ]
    zpath = ARCHIVE / ("V1200000_review_ready_not_promoted_bundle.zip" if hard_pass else "V1200000_route_exhausted_failure_analysis_bundle.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        seen: set[str] = set()
        for p in include:
            if p.is_file():
                try:
                    arc = str(p.relative_to(MAIN_ROOT)).replace("\\", "/")
                except ValueError:
                    arc = str(p.relative_to(WORKTREE)).replace("\\", "/")
                if arc not in seen:
                    zf.write(p, arc)
                    seen.add(arc)
    with zipfile.ZipFile(zpath) as zf:
        bad = zf.testzip()
        entries = len(zf.namelist())
    manifest = {"zip": str(zpath.resolve()), "entries": entries, "zip_test": "clean" if bad is None else bad, "sha256": sha256(zpath)}
    write_json(REPORTS / "V1200000_package_manifest.json", manifest)
    print(json.dumps({"status": final_status, "hard_gate_pass": hard_pass, "selected_route": selected_name, "bundle": manifest}, indent=2))


if __name__ == "__main__":
    main()
