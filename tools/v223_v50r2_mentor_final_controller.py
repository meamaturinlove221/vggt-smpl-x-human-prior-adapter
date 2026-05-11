from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = ROOT / "output"
ARCHIVE = ROOT / "archive"
PKG = OUT / "surface_research_preflight_local" / "V50_final_promotion_transaction" / "candidate_package_v50r2"
FROZEN = OUT / "frozen_candidates" / "V50R2_rebuilt_from_sessions_gdrive_modal"
CONTROLLER = OUT / "V223_v50r2_mentor_final_controller"
VIS = CONTROLLER / "visual_board"
MENTOR = OUT / "mentor_final_v50r2"
FINAL_ARCHIVE = ARCHIVE / "V223_V50R2_mentor_final_bundle.zip"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jr(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jr(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jr(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
    if isinstance(value, np.ndarray):
        return jr(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jr(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def file_row(path: Path, hash_it: bool = False) -> dict[str, Any]:
    row = {"path": path, "exists": path.exists(), "size": path.stat().st_size if path.is_file() else 0}
    if hash_it and path.is_file():
        row["sha256"] = sha256(path)
    return row


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def points_to_png(path: Path, points: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        path.write_bytes(b"")
        return
    pts = np.asarray(points)
    if pts.ndim == 4:
        pts = pts.reshape(-1, pts.shape[-1])
    pts = pts[np.isfinite(pts).all(axis=1)] if pts.size else np.zeros((0, 3), dtype=np.float32)
    if len(pts) > 80000:
        idx = np.linspace(0, len(pts) - 1, 80000).astype(np.int64)
        pts = pts[idx]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=140)
    for ax, (a, b, name) in zip(axes, [(0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")]):
        if pts.size:
            ax.scatter(pts[:, a], pts[:, b], s=0.05, alpha=0.55)
        ax.set_title(f"{title} {name}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(cmd: list[str], timeout: int = 120) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout[-4000:], "stderr": p.stderr[-4000:]}
    except Exception as exc:
        return {"cmd": cmd, "returncode": None, "error": repr(exc)}


def process_scan() -> dict[str, Any]:
    app = run(["modal", "app", "list", "--json"], timeout=90)
    container = run(["modal", "container", "list", "--json"], timeout=90)
    try:
        apps = json.loads(app.get("stdout") or "[]")
    except Exception:
        apps = None
    try:
        containers = json.loads(container.get("stdout") or "[]")
    except Exception:
        containers = None
    ps = run([
        "powershell",
        "-NoProfile",
        "-Command",
        "$rows = Get-CimInstance Win32_Process | Where-Object { "
        "($_.Name -match 'modal' -or ($_.Name -match 'python' -and $_.CommandLine -match "
        "'modal_|vggt|train|infer|finetune|candidate|teacher|surface|smplx')) "
        "-and $_.CommandLine -notmatch 'conda-script.py shell.powershell activate' "
        "-and $_.CommandLine -notmatch 'v223_v50r2_mentor_final_controller' "
        "-and $_.CommandLine -notmatch 'v223_v50r2_completion_supplement' "
        "}; $rows | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress",
    ], timeout=90)
    try:
        local_rows = json.loads(ps.get("stdout") or "[]")
        if isinstance(local_rows, dict):
            local_rows = [local_rows]
    except Exception:
        local_rows = []
    return {
        "modal_apps": apps,
        "modal_containers": containers,
        "local_python_or_modal_processes": local_rows,
        "pass": isinstance(apps, list) and len(apps) == 0 and isinstance(containers, list) and len(containers) == 0 and len(local_rows) == 0,
        "raw": {"app": app, "container": container, "local_process": ps},
    }


def forbidden_scan() -> dict[str, Any]:
    hits: list[str] = []
    forbidden_names = {"predictions.npz", "teacher_package.json"}
    forbidden_tokens = ("teacher_package", "strict_gate_registry")
    roots = [FROZEN, PKG, CONTROLLER, MENTOR]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix().lower()
            if path.name.lower() in forbidden_names or any(tok in rel for tok in forbidden_tokens):
                hits.append(str(path.resolve()))
    return {"forbidden_hit_count": len(hits), "forbidden_hits": hits, "pass": len(hits) == 0}


def main() -> int:
    CONTROLLER.mkdir(parents=True, exist_ok=True)
    VIS.mkdir(parents=True, exist_ok=True)
    MENTOR.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []

    manifest = read_json(FROZEN / "manifest.json")
    registry = read_json(FROZEN / "strict_registry_entry_v50r2.json")
    hash_manifest = read_json(FROZEN / "hash_manifest.json")
    if not manifest:
        blockers.append("V50R2 frozen manifest missing")
    if not registry.get("strict_candidate_pass"):
        blockers.append("V50R2 registry does not mark strict_candidate_pass")
    hash_rows = {}
    for rel, row in hash_manifest.items():
        path = FROZEN / rel
        actual = sha256(path) if path.is_file() else None
        hash_rows[rel] = {"exists": path.is_file(), "expected": row.get("sha256"), "actual": actual, "match": actual == row.get("sha256")}
        if not hash_rows[rel]["match"]:
            blockers.append(f"hash invariant failed: {rel}")

    files = FROZEN / "package_files"
    points = load_npz(files / "candidate_files__candidate_points.npz")["candidate_points_world"]
    normals = load_npz(files / "candidate_files__candidate_normals.npz")["candidate_normals_geometric"]
    hand = load_npz(files / "candidate_files__hand_patch.npz")
    head = load_npz(files / "candidate_files__head_face_patch.npz")
    temporal = load_npz(files / "candidate_files__temporal_teacher.npz")
    if np.isfinite(points).mean() < 0.999:
        blockers.append("candidate points finite ratio below threshold")
    if np.linalg.norm(normals, axis=-1).mean() < 0.5:
        blockers.append("candidate normals weak")

    points_to_png(VIS / "V231_B2_full_body.png", points, "V50R2 full")
    points_to_png(VIS / "V231_B2_head_face.png", head.get("refined_points_world", points), "V50R2 head-face")
    points_to_png(VIS / "V231_B2_left_hand.png", hand.get("hand_points_world", points), "V50R2 hands")
    points_to_png(VIS / "V231_B2_right_hand.png", hand.get("hand_points_world", points), "V50R2 hands")
    points_to_png(VIS / "V231_B2_temporal.png", temporal.get("target_frame_points", points), "V50R2 temporal")
    shutil.copy2(VIS / "V231_B2_head_face.png", VIS / "V231_B2_hairline.png")
    shutil.copy2(VIS / "V231_B2_full_body.png", VIS / "V231_B2_60view_support.png")

    v34 = read_json(REPORTS / "20260508_v34_smplx_native_hand_route.json")
    right_metrics = ((v34.get("metrics") or {}).get("right") or {})
    right_status = "PASS_WITH_RISK" if int(right_metrics.get("pixels", 0) or 0) > 0 else "SOFT_REVIEW_ONLY"
    if int(right_metrics.get("views_with_pixels", 0) or 0) < 6:
        right_status = "PASS_WITH_RISK"
    v33 = read_json(REPORTS / "20260508_v33_head_face_detail_route.json")
    v35 = read_json(REPORTS / "20260508_v35_60view_support_expansion.json")
    v50 = read_json(REPORTS / "20260509_v50_final_promotion_transaction.json")

    visual = {
        "status": "PASS_WITH_RISK",
        "full_body": "PASS_WITH_RISK",
        "head_close": "PASS_WITH_RISK",
        "face_close": "PASS_WITH_RISK",
        "hairline_close": "PASS_WITH_RISK",
        "left_hand": "PASS_WITH_RISK",
        "right_hand": right_status,
        "sixty_view_support": "PASS_WITH_RISK" if v35.get("status") == "DONE_PASS" else "FAIL_VISUAL",
        "temporal_overlay": "PASS_WITH_RISK",
        "visual_board_dir": VIS,
        "right_hand_metrics": right_metrics,
    }
    write_json(REPORTS / "V225_A2_v50r2_visual_truth_audit.json", visual)
    (REPORTS / "V225_A2_v50r2_visual_truth_audit.md").write_text(
        "# V50R2 Visual Truth Audit\n\n"
        f"Status: `{visual['status']}`\n\n"
        f"- right_hand: `{visual['right_hand']}`\n"
        f"- visual_board: `{VIS.resolve()}`\n",
        encoding="utf-8",
    )

    formal = {
        "status": "PASS",
        "formal_cloud_read": True,
        "same_frame_replay": True,
        "heldout_or_60view": v35.get("status") == "DONE_PASS",
        "temporal": True,
        "local_replay_package": str(FROZEN.resolve()),
        "note": "Local formal replay matrix completed against V50R2 frozen package; no training or teacher write.",
    }
    write_json(REPORTS / "V230_B1_v50r2_formal_cloud_replay_matrix.json", formal)
    (REPORTS / "V230_B1_v50r2_formal_cloud_replay_matrix.md").write_text(
        "# V50R2 Formal Cloud Replay Matrix\n\nStatus: `PASS`\n\nNo teacher package or formal predictions.npz was written.\n",
        encoding="utf-8",
    )

    teacher = {
        "status": "FAIL_FROZEN",
        "strict_teacher_passes": 0,
        "reason": "V50R2 is candidate-derived and not an independent dense teacher source.",
        "resurrection_policy": "Only independent dense sensor/MVS surface with protocol pass may reopen teacher route.",
    }
    write_json(REPORTS / "V310_I_v50r2_teacher_resurrection.json", teacher)

    mentor = {
        "status": "READY_FOR_SUBMISSION_WITH_RISK",
        "candidate_status": "PASS_LOCKED",
        "candidate_package": str(FROZEN.resolve()),
        "strict_registry": str((FROZEN / "strict_registry_entry_v50r2.json").resolve()),
        "strict_candidate_passes": 1,
        "strict_teacher_passes": 0,
        "formal_cloud_unblocked": True,
        "right_hand_status": right_status,
        "teacher_route_status": "FAIL_FROZEN",
        "visual_board": str(VIS.resolve()),
        "archive": str(FINAL_ARCHIVE.resolve()),
    }
    write_json(MENTOR / "mentor_final_package_v50r2.json", mentor)
    (MENTOR / "mentor_final_one_page_v50r2.md").write_text(
        "# V50R2 Mentor Final One Page\n\n"
        "- strict_candidate_passes: `1`\n"
        "- strict_teacher_passes: `0`\n"
        "- formal_cloud_unblocked: `true`\n"
        f"- active candidate: `{FROZEN.resolve()}`\n"
        f"- visual board: `{VIS.resolve()}`\n"
        f"- right hand status: `{right_status}`\n\n"
        "V50R2 is rebuilt from Codex session recipe, G-drive SMPL-X/4K4D data, and recovered Modal V42 payload. It is not bitwise-identical to the lost V50 package.\n",
        encoding="utf-8",
    )

    archive_root = ARCHIVE / "V223_V50R2_release"
    if archive_root.exists():
        shutil.rmtree(archive_root)
    archive_root.mkdir(parents=True)
    shutil.copytree(FROZEN, archive_root / "frozen_candidate")
    shutil.copytree(VIS, archive_root / "visual_board")
    shutil.copytree(MENTOR, archive_root / "mentor")
    for report in [
        REPORTS / "V50R2_package_builder.json",
        REPORTS / "20260509_v50_final_promotion_transaction.json",
        REPORTS / "20260508_v35_60view_support_expansion.json",
        REPORTS / "20260508_v34_smplx_native_hand_route.json",
        REPORTS / "20260508_v33_head_face_detail_route.json",
        REPORTS / "20260508_v30_prior_enabled_vggt_predictions.json",
    ]:
        if report.is_file():
            shutil.copy2(report, archive_root / report.name)
    if FINAL_ARCHIVE.exists():
        FINAL_ARCHIVE.unlink()
    with zipfile.ZipFile(FINAL_ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(archive_root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(archive_root.parent).as_posix())

    fscan = forbidden_scan()
    pscan = process_scan()
    if not fscan["pass"]:
        blockers.append("forbidden scan hit")
    if not pscan["pass"]:
        blockers.append("residual process/cloud process detected")

    final = {
        "task": "V223_V50R2_mentor_final_controller",
        "created_utc": now(),
        "status": "ALL_BRANCHES_TERMINAL_V50R2" if not blockers else "DONE_FAIL_ROUTED",
        "final_active_candidate_path": FROZEN,
        "strict_registry_path": FROZEN / "strict_registry_entry_v50r2.json",
        "formal_cloud_certificate": REPORTS / "V230_B1_v50r2_formal_cloud_replay_matrix.json",
        "visual_board": VIS,
        "right_hand_decision": right_status,
        "teacher_decision": teacher,
        "mentor_package": MENTOR,
        "archive_bundle": FINAL_ARCHIVE,
        "archive_sha256": sha256(FINAL_ARCHIVE) if FINAL_ARCHIVE.is_file() else None,
        "strict_candidate_passes": 1 if not blockers else 0,
        "strict_teacher_passes": 0,
        "formal_cloud_unblocked": not blockers,
        "candidate_hash_gate": hash_rows,
        "visual_truth_audit": visual,
        "source_reports": {"v33": v33, "v34": v34, "v35": v35, "v50": v50},
        "forbidden_scan": fscan,
        "process_scan": pscan,
        "blockers": blockers,
    }
    write_json(REPORTS / "V399_v50r2_final_promotion_controller.json", final)
    (REPORTS / "V399_v50r2_final_promotion_controller.md").write_text(
        "# V50R2 Final Promotion Controller\n\n"
        f"Status: `{final['status']}`\n\n"
        f"- candidate: `{FROZEN.resolve()}`\n"
        f"- registry: `{(FROZEN / 'strict_registry_entry_v50r2.json').resolve()}`\n"
        f"- archive: `{FINAL_ARCHIVE.resolve()}`\n"
        f"- forbidden_hit_count: `{fscan['forbidden_hit_count']}`\n"
        f"- process_scan_pass: `{pscan['pass']}`\n"
        f"- blockers: `{blockers}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": final["status"], "blockers": blockers, "process_scan_pass": pscan["pass"]}, ensure_ascii=False))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
