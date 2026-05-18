from __future__ import annotations

import json
import subprocess
import time
import zipfile
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


WORKTREE = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path(r"D:\vggt\vggt-main")
LOCAL = MAIN_ROOT / "local_report_auxiliary" / "V600_quality_rebuild"
REPORTS = WORKTREE / "reports"
OUT = LOCAL / "output" / "V820000_V900000_readiness_review"
ARCHIVE = LOCAL / "archive"
PROD_OUT = LOCAL / "output" / "V701000_V900000_production_live_highres"
V11700 = LOCAL / "remote_pull" / "V11700_gap_reduction_branch_520" / "predictions.npz"
V770 = PROD_OUT / "V770000_production_composition_NOT_CANDIDATE" / "predictions.npz"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def norm01(a: np.ndarray) -> np.ndarray:
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=np.float32)
    lo, hi = np.nanpercentile(a[finite], [2, 98])
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32)


def panel(label: str, im: Image.Image, lines: list[str]) -> Image.Image:
    footer = 20 + 15 * len(lines)
    out = Image.new("RGB", (im.width, im.height + footer), "white")
    out.paste(im.convert("RGB"), (0, 0))
    d = ImageDraw.Draw(out)
    d.text((4, im.height + 2), label, fill=(0, 0, 0))
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


def scatter_yz(a: np.ndarray, b: np.ndarray, title: str) -> Image.Image:
    size = 260
    im = Image.new("RGB", (size, size + 34), "white")
    d = ImageDraw.Draw(im)
    d.text((4, size + 5), title, fill=(0, 0, 0))
    pts = np.concatenate([a[:, [1, 2]], b[:, [1, 2]]], axis=0) if len(a) and len(b) else np.empty((0, 2))
    if len(pts) == 0:
        d.text((80, 110), "empty", fill=(160, 0, 0))
        return im
    lo = np.percentile(pts, 2, axis=0)
    hi = np.percentile(pts, 98, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    def draw(points: np.ndarray, color: tuple[int, int, int]) -> None:
        xy = (points[:, [1, 2]] - lo) / span
        x = 12 + xy[:, 0] * (size - 24)
        y = size - 12 - xy[:, 1] * (size - 24)
        step = max(1, len(x) // 2500)
        for px, py in zip(x[::step], y[::step]):
            d.ellipse((float(px)-1, float(py)-1, float(px)+1, float(py)+1), fill=color)
    draw(a, (0, 0, 0))
    draw(b, (220, 35, 35))
    return im


def make_readiness_board() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_pred(V11700)
    comp = load_pred(V770)
    delta = np.linalg.norm(comp["points"] - base["points"], axis=-1)
    changed = delta > 1e-5
    regions = [
        ("full", 4, np.ones(delta.shape[1:], dtype=bool)),
        ("head_face", 4, changed[4]),
        ("hairline", 4, changed[4]),
        ("right_hand", 3, changed[3]),
    ]
    rows: list[Image.Image] = []
    for name, view, mask in regions:
        if not mask.any():
            mask = delta[view] >= np.quantile(delta[view], 0.995)
        yy, xx = np.where(mask)
        if yy.size == 0:
            continue
        y0, y1 = max(0, int(yy.min()) - 32), min(delta.shape[1], int(yy.max()) + 33)
        x0, x1 = max(0, int(xx.min()) - 32), min(delta.shape[2], int(xx.max()) + 33)
        heat = Image.fromarray(np.uint8(norm01(delta[view, y0:y1, x0:x1]) * 255)).convert("L").resize((240, 240), Image.Resampling.BICUBIC).convert("RGB")
        normal_delta = np.linalg.norm(comp["normal"] - base["normal"], axis=-1)
        nheat = Image.fromarray(np.uint8(norm01(normal_delta[view, y0:y1, x0:x1]) * 255)).convert("L").resize((240, 240), Image.Resampling.BICUBIC).convert("RGB")
        pts_a = base["points"][view, y0:y1, x0:x1].reshape(-1, 3)
        pts_b = comp["points"][view, y0:y1, x0:x1].reshape(-1, 3)
        rows.append(
            hstack(
                [
                    panel(f"{name} delta", heat, [f"view={view}", f"changed={int(changed[view, y0:y1, x0:x1].sum())}"]),
                    panel(f"{name} normal", nheat, [f"normal nonzero={float((np.linalg.norm(comp['normal'], axis=-1)>1e-4).mean()):.3f}"]),
                    scatter_yz(pts_a, pts_b, f"{name} V11700 black / V770 red"),
                ]
            )
        )
    board = vstack(rows)
    path = OUT / "V820000_candidate_readiness_visual_review_board.png"
    board.save(path)
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
        if "modal app list --json" in cmd or "Get-CimInstance Win32_Process" in cmd or 'python.exe" -' in cmd:
            continue
        residual.append(p)
    return {
        "created_at": now(),
        "modal_returncode": modal.returncode,
        "modal_apps_json": modal.stdout.strip(),
        "process_scan": procs,
        "residual_training_or_modal_worker_detected": bool(residual),
        "residual_workers": residual,
    }


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
    OUT.mkdir(parents=True, exist_ok=True)
    v770 = load_json(REPORTS / "V770000_production_composition_NOT_CANDIDATE.json")
    v790 = load_json(REPORTS / "V790000_mentor_gate.json")
    v800 = load_json(REPORTS / "V800000_global_route_decision.json")
    v900 = load_json(REPORTS / "V900000_final_return_condition.json")
    board = make_readiness_board()
    readiness = {
        "created_at": now(),
        "status": "V820000_CANDIDATE_READINESS_REVIEW_READY_NOT_PROMOTION",
        "active_candidate": "V11700_gap_reduction_branch_520",
        "source_diagnostic": v770,
        "mentor_gate": v790,
        "route_decision": v800,
        "previous_return_condition": v900,
        "readiness_scope": "review bundle only; not active-candidate promotion",
        "candidate_readiness_package_generated": True,
        "mentor_package_generated": False,
        "strict_registry_written": False,
        "v50_v50r2_modified": False,
        "requires_human_approval_before_registry_or_candidate_promotion": True,
        "visual_review_board": board,
    }
    write_json(REPORTS / "V820000_candidate_readiness_review.json", readiness)
    write_text(
        REPORTS / "V820000_candidate_readiness_review.md",
        "# V820000 Candidate-Readiness Review\n\n"
        "This bundle is a review artifact, not a promotion. Active candidate remains `V11700_gap_reduction_branch_520`.\n\n"
        "Production composition is `NOT_CANDIDATE`, but it has nonnegative diagnostic V629 deltas across mean/local/full/head-face/hairline/right-hand with left-hand preserved. "
        "No strict registry, mentor package, or active candidate update is written.\n\n"
        f"Visual board: `{board}`\n",
    )
    lock = {
        "status": "V821000_PROMOTION_LOCK",
        "active_candidate": "V11700_gap_reduction_branch_520",
        "V770_predictions": str(V770.resolve()),
        "allow_as_candidate": False,
        "allow_strict_registry": False,
        "allow_mentor_package": False,
        "reason": "User/mentor must explicitly approve after reviewing V820 readiness evidence; current output remains NOT_CANDIDATE diagnostic.",
    }
    write_json(REPORTS / "V821000_promotion_lock.json", lock)
    git = subprocess.run(["git", "log", "-1", "--oneline"], cwd=str(WORKTREE), text=True, capture_output=True, timeout=30)
    handoff = {
        "status": "V822000_BRANCH_HANDOFF",
        "worktree": str(WORKTREE.resolve()),
        "branch": "codex/live-highres-crop",
        "last_commit": git.stdout.strip(),
        "production_controller": str((WORKTREE / "tools/v701000_v900000_production_highres_controller.py").resolve()),
        "readiness_controller": str((WORKTREE / "tools/v820000_v900000_candidate_readiness_review.py").resolve()),
        "next_allowed_actions": [
            "human/mentor review of V820 readiness board",
            "user-approved candidate-readiness packaging continuation",
            "new supervision/architecture route if review rejects visual gain",
        ],
    }
    write_json(REPORTS / "V822000_branch_handoff.json", handoff)
    scan = process_scan()
    write_json(REPORTS / "V900000_final_process_and_modal_scan_v2.json", scan)
    final = {
        "status": "V900000_PLAN_COMPLETED_TO_READINESS_REVIEW_NOT_PROMOTION",
        "global_terminal_for_current_plan": True,
        "active_candidate": "V11700_gap_reduction_branch_520",
        "readiness_review_ready": True,
        "candidate_promoted": False,
        "mentor_package_generated": False,
        "strict_registry_written": False,
        "v50_v50r2_modified": False,
        "process_scan_clean": not scan["residual_training_or_modal_worker_detected"],
        "next_requires": "human/mentor review or explicit approval before any promotion/registry/package step",
    }
    write_json(REPORTS / "V900000_plan_completed_to_readiness_review.json", final)
    include = [
        REPORTS / "V820000_candidate_readiness_review.json",
        REPORTS / "V820000_candidate_readiness_review.md",
        REPORTS / "V821000_promotion_lock.json",
        REPORTS / "V822000_branch_handoff.json",
        REPORTS / "V900000_final_process_and_modal_scan_v2.json",
        REPORTS / "V900000_plan_completed_to_readiness_review.json",
        REPORTS / "V704000_production_canary_hard_gate.json",
        REPORTS / "V710000_normal_branch_repair.json",
        REPORTS / "V760000_real_multiview_crop_training.json",
        REPORTS / "V770000_production_composition_NOT_CANDIDATE.json",
        REPORTS / "V780000_truthful_report.md",
        REPORTS / "V790000_mentor_gate.json",
        REPORTS / "V800000_global_route_decision.json",
        Path(board),
        V770,
    ]
    manifest = package(include, "V900000_candidate_readiness_review_bundle.zip")
    write_json(REPORTS / "V900000_candidate_readiness_review_manifest.json", manifest)
    print(json.dumps({"status": final["status"], "bundle": manifest, "process_clean": final["process_scan_clean"]}, indent=2))


if __name__ == "__main__":
    main()
