from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(r"D:\vggt\vggt-feature-adapter")
AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
ARCHIVE = AUX / "archive"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str], check: bool = False) -> dict[str, Any]:
    p = subprocess.run(cmd, cwd=str(REPO), text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and p.returncode:
        raise RuntimeError(f"{cmd} failed: {p.stderr}")
    return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def classify_dirty(lines: list[str]) -> dict[str, list[str]]:
    out = {
        "current_research_files_to_commit": [],
        "historical_leftovers_keep": [],
        "temporary_cleaned_or_cleanable": [],
        "do_not_delete_evidence": [],
    }
    current_markers = ["V90010000000", "V900100", "V120000000000"]
    temp_markers = ["__pycache__", "__tmp", "__v380_modal_pull"]
    evidence_markers = ["V415", "V800", "V900", "V304", "V910", "V920"]
    for line in lines:
        path = line[3:] if len(line) > 3 else line
        if any(m in path for m in current_markers):
            out["current_research_files_to_commit"].append(line)
        elif any(m in path for m in temp_markers):
            out["temporary_cleaned_or_cleanable"].append(line)
        elif any(m in path for m in evidence_markers):
            out["do_not_delete_evidence"].append(line)
        else:
            out["historical_leftovers_keep"].append(line)
    return out


def main() -> None:
    branch = run(["git", "branch", "--show-current"])
    head = run(["git", "rev-parse", "HEAD"])
    remote = run(["git", "ls-remote", "origin", "refs/heads/codex/feature-adapter"])
    status = run(["git", "status", "--porcelain=v1"])
    dirty_lines = status["stdout"].splitlines() if status["stdout"] else []
    push_ok = head["stdout"] and head["stdout"] in remote["stdout"]
    patch_path = ARCHIVE / "V90010000000_push_recovery.patch"
    if not push_ok:
        patch = run(["git", "format-patch", "-1", "HEAD", "--stdout"])
        patch_path.write_text(patch["stdout"], encoding="utf-8")
    push_report = {
        "created_utc": now(),
        "branch": branch["stdout"],
        "head": head["stdout"],
        "remote_ref": remote["stdout"],
        "remote_contains_head": bool(push_ok),
        "patch_bundle": str(patch_path) if not push_ok else None,
        "patch_sha256": sha256(patch_path) if patch_path.exists() else None,
    }
    write_json(REPORTS / "V90010000000_push_recovery.json", push_report)

    dirty = classify_dirty(dirty_lines)
    md = ["# V900100 Dirty Worktree Classification", "", f"- created_utc: {now()}", f"- git_status_clean: {not dirty_lines}", ""]
    for key, vals in dirty.items():
        md.append(f"## {key}")
        if vals:
            md.extend(f"- `{v}`" for v in vals)
        else:
            md.append("- none")
        md.append("")
    (REPORTS / "V90010000000_dirty_worktree_classification.md").write_text("\n".join(md), encoding="utf-8")

    manifest = read_json(REPORTS / "V80000000000_upload_manifest_sidecar.json")
    bundles: dict[str, Any] = {}
    for name, item in manifest["bundles"].items():
        p = Path(item["path"])
        with zipfile.ZipFile(p) as zf:
            testzip = zf.testzip()
            entries = zf.namelist()
        actual = sha256(p)
        bundles[name] = {
            "path": str(p),
            "exists": p.exists(),
            "bytes": p.stat().st_size,
            "manifest_sha256": item["sha256"],
            "actual_sha256": actual,
            "hash_match": actual == item["sha256"],
            "testzip": testzip,
            "entries": len(entries),
        }
    selected = AUX / "output" / "V41500000000_modal_camera_mask_fullview_core_controls" / "predictions.npz"
    controls = [
        AUX / "output" / "V41100000000_modal_fullview_core_controls" / "predictions.npz",
        AUX / "output" / "V41300000000_modal_repaired_fullview_core_controls" / "predictions.npz",
    ]
    npz_checks: list[dict[str, Any]] = []
    for p in [selected, *controls]:
        with zipfile.ZipFile(p) as zf:
            bad = zf.testzip()
        with np.load(p, allow_pickle=False) as z:
            keys = sorted(z.files)
            shapes = {k: list(z[k].shape) for k in keys if k == "confidence" or k.endswith("_world_points") or k.endswith("_normal")}
            normal_keys = [k for k in keys if k.endswith("_normal")]
            normal_nonzero = {
                k: float((np.linalg.norm(z[k].astype(np.float32), axis=-1) > 0.1).mean())
                for k in normal_keys[:10]
            }
        npz_checks.append({"path": str(p), "testzip": bad, "keys": keys, "shapes": shapes, "normal_nonzero": normal_nonzero})
    projection = read_json(REPORTS / "V41500000000_camera_bound_projection_decision.json")
    rows = read_csv(REPORTS / "V41500000000_region_metrics.csv")
    artifact_audit = {
        "created_utc": now(),
        "bundles": bundles,
        "npz_checks": npz_checks,
        "visuals_new": [str(p) for p in sorted((AUX / "boards").glob("V415*.png"))],
        "projection_margin": projection["true_camera_bound_margin"],
        "projection_rank": projection["true_rank"],
        "region_status_counts": {
            "ok": sum(1 for r in rows if r.get("status") == "ok"),
            "non_ok": sum(1 for r in rows if r.get("status") != "ok"),
        },
    }
    write_json(REPORTS / "V90010000000_artifact_audit.json", artifact_audit)

    margin = float(projection["true_camera_bound_margin"])
    margin_doc = {
        "created_utc": now(),
        "true_camera_bound_margin": margin,
        "risk_level": "high" if margin < 0.005 else "medium" if margin < 0.02 else "low",
        "reason": "Margin is positive but small; advisor-defense requires bootstrap/view-ablation robustness and stronger visuals.",
        "next_route": "V910 binding/bootstrap robustness and V930 visual hardening",
    }
    write_json(REPORTS / "V90100000000_margin_risk_report.json", margin_doc)
    weakness = """# V901 Visual Weakness Table

| Risk | Evidence | Required hardening |
|---|---|---|
| Projection margin small | true margin = %.6f | bootstrap/view ablation and CI |
| Ranking chart visually tight | V415 projection bars close | add CI and per-view decomposition |
| Point clouds similar | V415 fullbody board | same-scale overlays, delta maps, projection overlays |
| Closeups sparse | V415 head/hair/hand board | region-local crops and confidence-gated sampling |
| Residual favors low-motion controls | V415 region metrics | separate residual stability from camera-bound score |
""" % margin
    (REPORTS / "V90100000000_visual_weakness_table.md").write_text(weakness, encoding="utf-8")
    print(json.dumps({"push": push_report, "artifact_audit": artifact_audit, "margin": margin_doc}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
