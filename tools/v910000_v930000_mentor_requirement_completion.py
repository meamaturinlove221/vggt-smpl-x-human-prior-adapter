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


WORKTREE = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path(r"D:\vggt\vggt-main")
LOCAL = MAIN_ROOT / "local_report_auxiliary" / "V600_quality_rebuild"
REPORTS = WORKTREE / "reports"
OUT = LOCAL / "output" / "V910000_V930000_mentor_requirement_completion"
ARCHIVE = LOCAL / "archive"
TRAIN = MAIN_ROOT / "output" / "training_cases" / "0012_11_frame0000_6views_smplx_native_prior_v15"
PROXY = LOCAL / "output" / "V321000_V350000_2d_semantic_proxy_route" / "V321000_2d_semantic_proxy_maps.npz"
V647 = LOCAL / "remote_pull" / "V647_true6_crop_baseline" / "predictions.npz"
V11700 = LOCAL / "remote_pull" / "V11700_gap_reduction_branch_520" / "predictions.npz"
V770 = LOCAL / "output" / "V701000_V900000_production_live_highres" / "V770000_production_composition_NOT_CANDIDATE" / "predictions.npz"

sys.path.insert(0, str(LOCAL / "tools"))
import v400000_v520000_highres_human_feature_branch as v400  # noqa: E402
import v232000_v260000_sharper_defect_patch_route as v232  # noqa: E402


REGIONS = {
    "full_body": {"prob": None, "thr": 0.0, "view": 4},
    "head_face": {"prob": "p_head_face", "thr": 0.26, "view": 4},
    "hairline": {"prob": "p_hairline", "thr": 0.20, "view": 4},
    "left_hand": {"prob": "p_left_hand", "thr": 0.28, "view": 3},
    "right_hand": {"prob": "p_right_hand", "thr": 0.28, "view": 3},
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
        "normal": normal[:6].astype(np.float32),
        "confidence": z.get("world_points_conf", np.ones(depth.shape, dtype=np.float32))[:6].astype(np.float32),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm01(a: np.ndarray) -> np.ndarray:
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=np.float32)
    lo, hi = np.nanpercentile(a[finite], [2, 98])
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32)


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


def bbox(mask: np.ndarray, pad: int) -> tuple[int, int, int, int]:
    yy, xx = np.where(mask)
    if yy.size == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    return max(0, int(xx.min()) - pad), max(0, int(yy.min()) - pad), min(mask.shape[1], int(xx.max()) + pad + 1), min(mask.shape[0], int(yy.max()) + pad + 1)


def scatter(points: list[tuple[np.ndarray, tuple[int, int, int]]], title: str) -> Image.Image:
    size = 250
    out = Image.new("RGB", (size, size + 34), "white")
    d = ImageDraw.Draw(out)
    d.text((4, size + 5), title, fill=(0, 0, 0))
    all_pts = np.concatenate([p[:, [1, 2]] for p, _ in points if len(p)], axis=0) if any(len(p) for p, _ in points) else np.empty((0, 2))
    if len(all_pts) == 0:
        d.text((80, 110), "empty", fill=(160, 0, 0))
        return out
    lo = np.percentile(all_pts, 2, axis=0)
    hi = np.percentile(all_pts, 98, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    for p, color in points:
        if len(p) == 0:
            continue
        xy = (p[:, [1, 2]] - lo) / span
        px = 12 + xy[:, 0] * (size - 24)
        py = size - 12 - xy[:, 1] * (size - 24)
        step = max(1, len(px) // 2000)
        for x, y in zip(px[::step], py[::step]):
            d.ellipse((float(x) - 1, float(y) - 1, float(x) + 1, float(y) + 1), fill=color)
    return out


def region_mask(region: str, proxy: dict[str, np.ndarray], targets: dict[str, np.ndarray]) -> tuple[int, np.ndarray]:
    spec = REGIONS[region]
    view = int(spec["view"])
    fg = targets["teacher_mask"][:6].astype(bool)[view]
    if spec["prob"] is None:
        return view, fg
    prob = proxy[spec["prob"]][view]
    return view, (prob >= float(spec["thr"])) & proxy["foreground"][view]


def make_board_and_rows() -> tuple[str, list[dict[str, Any]]]:
    inputs, targets = v400.load_training()
    proxy = load_npz(PROXY)
    v647 = load_pred(V647)
    v117 = load_pred(V11700)
    v770 = load_pred(V770)
    rows: list[dict[str, Any]] = []
    panels: list[Image.Image] = []
    for region in REGIONS:
        view, mask = region_mask(region, proxy, targets)
        if not mask.any():
            continue
        pad = 70 if region == "full_body" else 36
        x0, y0, x1, y1 = bbox(mask, pad)
        rgb = Image.fromarray(v400.norm_img(inputs["images"][view, y0:y1, x0:x1])).resize((220, 220), Image.Resampling.BICUBIC)
        overlay = v400.overlay_points(inputs["images"][view], [(mask, (255, 0, 0, 170))]).crop((x0, y0, x1, y1)).resize((220, 220), Image.Resampling.BICUBIC)
        d117 = np.linalg.norm(v117["points"] - v647["points"], axis=-1)
        d770 = np.linalg.norm(v770["points"] - v117["points"], axis=-1)
        heat117 = Image.fromarray(np.uint8(norm01(d117[view, y0:y1, x0:x1]) * 255)).convert("L").resize((220, 220), Image.Resampling.BICUBIC).convert("RGB")
        heat770 = Image.fromarray(np.uint8(norm01(d770[view, y0:y1, x0:x1]) * 255)).convert("L").resize((220, 220), Image.Resampling.BICUBIC).convert("RGB")
        local_mask = mask[y0:y1, x0:x1]
        pts647 = v647["points"][view, y0:y1, x0:x1][local_mask]
        pts117 = v117["points"][view, y0:y1, x0:x1][local_mask]
        pts770 = v770["points"][view, y0:y1, x0:x1][local_mask]
        sc = scatter([(pts647, (120, 120, 120)), (pts117, (0, 0, 0)), (pts770, (220, 35, 35))], f"{region}: grey baseline / black V11700 / red prod")
        changed = int((d770[view] > 1e-5).sum())
        local_changed = int(((d770[view] > 1e-5) & mask).sum())
        rows.append(
            {
                "region": region,
                "view": view,
                "roi_pixels": int(mask.sum()),
                "production_changed_pixels_view": changed,
                "production_changed_pixels_in_roi": local_changed,
                "v117_vs_baseline_l2_mean_roi": float(d117[view][mask].mean()),
                "prod_vs_v117_l2_mean_roi": float(d770[view][mask].mean()),
                "normal_nonzero_ratio_prod": float((np.linalg.norm(v770["normal"], axis=-1) > 1e-4).mean()),
            }
        )
        panels.append(
            hstack(
                [
                    panel(f"{region} RGB", rgb, [f"view={view}", f"roi={int(mask.sum())}"]),
                    panel("semantic ROI", overlay, [f"prod changed ROI={local_changed}"]),
                    panel("V11700 vs VGGT baseline", heat117, [f"mean={rows[-1]['v117_vs_baseline_l2_mean_roi']:.2e}"]),
                    panel("production vs V11700", heat770, [f"mean={rows[-1]['prod_vs_v117_l2_mean_roi']:.2e}"]),
                    sc,
                ]
            )
        )
    board = vstack(panels)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "V910000_mentor_requirement_comparison_board.png"
    board.save(path)
    return str(path.resolve()), rows


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
        if "modal app list --json" in cmd or "Get-CimInstance Win32_Process" in cmd or "mentor_requirement_completion.py" in cmd or 'python.exe" -' in cmd:
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
    board, rows = make_board_and_rows()
    write_csv(REPORTS / "V910000_mentor_requirement_region_table.csv", rows)
    v770_report = json.loads((REPORTS / "V770000_production_composition_NOT_CANDIDATE.json").read_text(encoding="utf-8"))
    deltas = v770_report["v629_delta"]
    required_numeric = {
        "full_no_regression": deltas["full_body_quality"] >= 0,
        "head_face_positive": deltas["head_face_quality"] > 0,
        "hairline_positive": deltas["hairline_quality"] > 0,
        "right_hand_positive": deltas["right_hand_quality"] > 0,
        "left_hand_preserved": deltas["left_hand_quality"] >= 0,
        "mean_local_positive": deltas["mean_quality"] > 0 and deltas["local_detail_quality"] > 0,
    }
    visual_ready = bool(Path(board).is_file() and Path(board).stat().st_size > 100_000 and all(r["roi_pixels"] > 0 for r in rows))
    completion = {
        "created_at": now(),
        "status": "V910000_MENTOR_REQUIREMENT_EVIDENCE_READY",
        "active_candidate": "V11700_gap_reduction_branch_520",
        "baseline_used": "V647_true6_crop_baseline",
        "production_result": "V770000_PRODUCTION_COMPOSITION_NOT_CANDIDATE",
        "numeric_requirements": required_numeric,
        "numeric_requirements_pass": all(required_numeric.values()),
        "visual_board": board,
        "visual_board_ready": visual_ready,
        "mentor_requirements_met_for_review": bool(all(required_numeric.values()) and visual_ready),
        "promotion_allowed": False,
        "strict_registry_written": False,
        "mentor_package_generated": False,
        "candidate_package_generated": False,
        "region_rows": rows,
    }
    write_json(REPORTS / "V910000_mentor_requirement_completion.json", completion)
    verdict = "met for mentor review" if completion["mentor_requirements_met_for_review"] else "not met"
    write_text(
        REPORTS / "V910000_mentor_requirement_completion.md",
        "# V910000 Mentor Requirement Completion\n\n"
        f"Verdict: `{verdict}`.\n\n"
        "This completes the requested baseline comparison evidence loop. It compares VGGT true-6 crop baseline, V11700 active, and the production high-res branch diagnostic composition across full body, head-face, hairline, left hand, and right hand.\n\n"
        "It is still not a strict-registry promotion. Active candidate remains `V11700_gap_reduction_branch_520` unless the user explicitly approves promotion after review.\n\n"
        f"Board: `{board}`\n",
    )
    lock = {
        "status": "V920000_FINAL_PROMOTION_LOCK",
        "current_plan_completed": completion["mentor_requirements_met_for_review"],
        "active_candidate": "V11700_gap_reduction_branch_520",
        "not_candidate_npz": str(V770.resolve()),
        "allow_registry": False,
        "allow_candidate_package": False,
        "allow_mentor_package": False,
        "requires_explicit_user_approval_for_any_promotion": True,
    }
    write_json(REPORTS / "V920000_final_promotion_lock.json", lock)
    scan = process_scan()
    write_json(REPORTS / "V930000_final_process_and_modal_scan.json", scan)
    final = {
        "status": "V930000_MENTOR_REQUIREMENT_PLAN_COMPLETE_NOT_PROMOTED",
        "global_terminal_for_requested_plan": completion["mentor_requirements_met_for_review"],
        "active_candidate": "V11700_gap_reduction_branch_520",
        "mentor_requirement_evidence_ready": completion["mentor_requirements_met_for_review"],
        "candidate_promoted": False,
        "strict_registry_written": False,
        "process_scan_clean": not scan["residual_training_or_modal_worker_detected"],
        "next_requires": "review/approval before any candidate or registry promotion",
    }
    write_json(REPORTS / "V930000_final_plan_status.json", final)
    manifest = package(
        [
            REPORTS / "V910000_mentor_requirement_completion.json",
            REPORTS / "V910000_mentor_requirement_completion.md",
            REPORTS / "V910000_mentor_requirement_region_table.csv",
            REPORTS / "V920000_final_promotion_lock.json",
            REPORTS / "V930000_final_process_and_modal_scan.json",
            REPORTS / "V930000_final_plan_status.json",
            REPORTS / "V770000_production_composition_NOT_CANDIDATE.json",
            REPORTS / "V790000_mentor_gate.json",
            Path(board),
            V770,
        ],
        "V930000_mentor_requirement_completion_bundle.zip",
    )
    write_json(REPORTS / "V930000_package_manifest.json", manifest)
    print(json.dumps({"status": final["status"], "evidence_ready": completion["mentor_requirements_met_for_review"], "bundle": manifest}, indent=2))


if __name__ == "__main__":
    main()
