from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
BOARDS = ROOT / "boards"
VIEWERS = ROOT / "viewers"
BUNDLES = ROOT / "bundles"
DOCS = ROOT / "docs" / "goals"

V523_OUT = ROOT / "output" / "V5230000000000000000000_observation_anchor_control_part_binding_repair"

V512_DECISION = REPORTS / "V5120000000000000000000_manual_mentor_gate.json"
V523_DECISION = REPORTS / "V5230000000000000000000_observation_anchor_control_part_binding_decision.json"
V524_DECISION = REPORTS / "V5240000000000000000000_visibility_aware_gate_router_decision.json"
V525_ROUTE = DOCS / "V5250000000000000000000_advisor_pack_viewer_report_bundle_route.md"

ADVISOR_REPORT = REPORTS / "V5250000000000000000000_v50r2_visual_floor_advisor_report.md"
VIEWER = VIEWERS / "V5250000000000000000000_v50r2_visual_floor_viewer.html"
MANIFEST = REPORTS / "V5250000000000000000000_bundle_manifest.json"
AUDIT = REPORTS / "V5250000000000000000000_artifact_audit.json"
FINAL_STATUS = REPORTS / "V9000000000000000000000_v50r2_distilled_human_scene_pointcloud_final_status.json"
BUNDLE = BUNDLES / "V5250000000000000000000_v50r2_visual_floor_advisor_pack.zip"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def git(args: list[str]) -> list[str]:
    out = subprocess.run(["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return [line for line in out.stdout.splitlines() if line.strip()]


def read_ply_header(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii", errors="ignore") as f:
        lines = []
        for line in f:
            lines.append(line.rstrip("\n"))
            if line.strip() == "end_header":
                break
    vertex_count = 0
    for line in lines:
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
    return {"path": str(path), "readable": bool(lines and lines[0] == "ply" and vertex_count > 0), "vertex_count": vertex_count}


def npz_readable(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {"path": str(path), "readable": True, "keys": list(z.files), "shapes": {k: list(z[k].shape) for k in z.files[:8]}}


def viewer_reference_check(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)="\.\./([^"]+)"', text)
    missing = [ref for ref in refs if not (ROOT / ref).exists()]
    return {"path": str(path), "reference_count": len(refs), "missing": missing, "all_references_exist": not missing}


def write_report(v512: dict[str, Any], v523: dict[str, Any], v524: dict[str, Any]) -> None:
    metrics = v523["control_metrics_best_view_cam21"]
    part_gate = v524["visibility_aware_part_gate"]
    files = [
        v523["boards"]["main"],
        v523["boards"]["same_scene_controls"],
        v523["boards"]["v50r2_visual_floor_comparison"],
        v523["boards"]["local_fidelity"],
        v523["boards"]["anti_2d"],
        v512["annotated_board"],
        str(VIEWER),
        str(BUNDLE),
    ]
    text = f"""# V525 V50R2 Visual Floor Observation-Distillation Advisor Pack

## Architecture Diagram

```text
VGGT visible world points + RGB + confidence
        |
        |  SMPL-X surface prior / part binding
        v
V523 visible-anchor guarded student
        |
        |  residual clipped to preserve V521/V50R2 readability
        v
model-owned human-main full-scene RGB point cloud
        |
        +-- same-scene legacy VGGT/student controls
        +-- V50R2 visual floor reference
        +-- local fidelity / anti-2D boards
        +-- viewer / bundle / artifact audit
```

## Route Positioning

This pack is the not-promoted advisor evidence for the V50R2 visual-floor observation-distillation route. V50R2 is used only as visual floor, teacher, and evaluation reference. It is not used as the final student output.

The current candidate is V523: a VGGT-observation anchored, SMPL-part-bound student. It keeps the V521 visual recovery and separates visible-anchor preservation from the baseline/control comparison.

## Why Previous Route Failed

V519 and V520 improved automatic distance and thickness metrics but visually returned to smeared template-like or blob-like morphology. Those routes did not preserve the V50R2-style visible human readability.

V521 restored readable human shape and full-scene context, but it failed closed because it compared against the raw visible anchor as if that anchor were a baseline to beat. V523 corrected the source roles: visible anchor is an input guard, while prior VGGT/student artifacts are used as the mentor-facing controls.

## Current Change

V523 uses a visible-anchor guarded residual: the model can only make small, clipped residual changes on top of VGGT visible observations, preserving the readable head, torso, clothing, legs, and scene context. Body-part labels are derived from SMPL surface features plus geometry rather than the earlier coarse image-bin rule.

## Experiment Loop

- V519: canonical surfel graph training, failed manual morphology.
- V520: pose-aligned surfel graph repair, failed manual morphology.
- V521: observation-anchored visible student, visual recovery but controls/local gates incomplete.
- V523: separated visible-anchor guard from legacy controls and repaired part binding.
- V524: visibility-aware V509/V510/V511/V512 gate routing passed for advisor-pack assembly.

## VGGT Baseline / Controls Comparison

Best-view control distances against the V50R2 floor reference:

```text
V523 true:               {metrics['true']:.6f}
visible anchor guard:    {metrics['visible']:.6f}  (nonregression guard, not counted as a baseline win)
V517 VGGT baseline:      {metrics['v517']:.6f}
V517 no-SMPL:            {metrics['v517_no_smpl']:.6f}
V520 shuffled semantic:  {metrics['v520_shuffled']:.6f}
V520 SMPL graph only:    {metrics['v520_smpl']:.6f}
```

V512 gate status: `{v512['status']}`.

## Point Cloud Visual Evidence

- Human-main full-scene board: `{rel(Path(v523['boards']['main']))}`
- Same-scene baseline / true / controls: `{rel(Path(v523['boards']['same_scene_controls']))}`
- V50R2 visual floor comparison: `{rel(Path(v523['boards']['v50r2_visual_floor_comparison']))}`
- Local fidelity board: `{rel(Path(v523['boards']['local_fidelity']))}`
- Anti-2D side/depth/cross-section board: `{rel(Path(v523['boards']['anti_2d']))}`
- Manual gate annotated board: `{rel(Path(v512['annotated_board']))}`

## Local Fidelity

Visibility-aware local gate:

```text
head/hair:       {part_gate['head_hair']['view']} nn={part_gate['head_hair']['value']:.6f}
torso/clothing:  {part_gate['torso_clothing']['view']} nn={part_gate['torso_clothing']['value']:.6f}
arm/hand:        {part_gate['arm_hand']['view']} nn={part_gate['arm_hand']['value']:.6f}
leg/foot:        {part_gate['leg_foot']['view']} nn={part_gate['leg_foot']['value']:.6f}
```

## Limitations

- This is not promotion and does not modify registry or active candidate state.
- V50R2 remains teacher/reference only.
- Visible anchor preservation is a guard, not a claimed baseline victory.
- Fine facial details are not claimed; face/head evidence is contour or region-level only.
- The final artifact is advisor-ready evidence, not a registry promotion.

## Next Plan

Use this pack for mentor review. If stricter improvement over the raw visible anchor is required, the next route should add a withheld-view or missing-region completion target while preserving V523 readability.

## File List

{chr(10).join(f'- `{rel(Path(item))}`' for item in files)}
"""
    ADVISOR_REPORT.parent.mkdir(parents=True, exist_ok=True)
    ADVISOR_REPORT.write_text(text, encoding="utf-8")


def write_viewer(v512: dict[str, Any], v523: dict[str, Any]) -> None:
    boards = [
        ("Human-main full scene", v523["boards"]["main"]),
        ("Same-scene controls", v523["boards"]["same_scene_controls"]),
        ("V50R2 visual floor", v523["boards"]["v50r2_visual_floor_comparison"]),
        ("Local fidelity", v523["boards"]["local_fidelity"]),
        ("Anti-2D", v523["boards"]["anti_2d"]),
        ("Manual gate", v512["annotated_board"]),
    ]
    cards = "\n".join(
        f'<section><h2>{title}</h2><img src="../{rel(Path(path))}" alt="{title}"></section>'
        for title, path in boards
    )
    ply_links = "\n".join(
        f'<li><a href="../{rel(path)}">{path.name}</a></li>'
        for path in sorted(V523_OUT.glob("v523_cam*_true_full_scene_rgb.ply"))
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>V525 V50R2 Visual Floor Viewer</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #151515; background: #f7f7f4; }}
    header {{ padding: 24px 32px; background: #1f2933; color: white; }}
    main {{ padding: 24px 32px 48px; }}
    section {{ margin: 0 0 28px; padding: 16px; background: white; border: 1px solid #ddd; }}
    img {{ width: 100%; height: auto; display: block; border: 1px solid #bbb; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .note {{ color: #a40000; font-weight: 700; }}
    a {{ color: #0b5cad; }}
  </style>
</head>
<body>
  <header>
    <h1>V525 V50R2 Visual Floor Advisor Viewer</h1>
    <p class="note">Not promoted. V50R2 is visual floor / teacher / reference only.</p>
  </header>
  <main>
    {cards}
    <section>
      <h2>Point Cloud Payloads</h2>
      <ul>{ply_links}</ul>
    </section>
    <section>
      <h2>Reports</h2>
      <ul>
        <li><a href="../{rel(ADVISOR_REPORT)}">Advisor report</a></li>
        <li><a href="../{rel(V512_DECISION)}">Manual mentor gate</a></li>
        <li><a href="../{rel(V523_DECISION)}">V523 decision</a></li>
        <li><a href="../{rel(AUDIT)}">Artifact audit</a></li>
      </ul>
    </section>
  </main>
</body>
</html>
"""
    VIEWER.parent.mkdir(parents=True, exist_ok=True)
    VIEWER.write_text(html, encoding="utf-8")


def included_files() -> list[Path]:
    v523 = read_json(V523_DECISION)
    files = [
        ADVISOR_REPORT,
        VIEWER,
        V512_DECISION,
        V523_DECISION,
        V524_DECISION,
        REPORTS / "V5090000000000000000000_full_scene_decision.json",
        REPORTS / "V5100000000000000000000_local_fidelity_decision.json",
        REPORTS / "V5110000000000000000000_anti_2d_decision.json",
        REPORTS / "V5130000000000000000000_auto_evolution_decision.json",
        REPORTS / "V5230000000000000000000_observation_anchor_control_part_binding_metrics.csv",
        REPORTS / "V5230000000000000000000_teacher_copy_check.json",
        BOARDS / "V5120000000000000000000_manual_gate_annotated.png",
        Path(v523["boards"]["main"]),
        Path(v523["boards"]["same_scene_controls"]),
        Path(v523["boards"]["v50r2_visual_floor_comparison"]),
        Path(v523["boards"]["local_fidelity"]),
        Path(v523["boards"]["anti_2d"]),
        DOCS / "V5250000000000000000000_advisor_pack_viewer_report_bundle_route.md",
        ROOT / "tools" / "V523_observation_anchor_control_part_binding_repair.py",
        ROOT / "tools" / "V524_visibility_aware_gate_router.py",
        ROOT / "tools" / "V525_advisor_pack_viewer_report_bundle.py",
        V523_OUT / "v523_observation_anchor_control_part_binding_candidate.npz",
    ]
    files.extend(sorted(V523_OUT.glob("v523_cam*_true_full_scene_rgb.ply")))
    files.extend(sorted(V523_OUT.glob("v523_cam*_true_human_only_rgb.ply")))
    return [path for path in files if path.exists()]


def write_manifest(files: list[Path]) -> dict[str, Any]:
    entries = []
    for path in files:
        entries.append({"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    payload = {
        "task": "V525_bundle_manifest",
        "created_at": now(),
        "repo": str(ROOT),
        "bundle": str(BUNDLE),
        "file_count": len(entries),
        "files": entries,
        "policy": {
            "not_promoted": True,
            "no_registry_write": True,
            "no_v50_v50r2_modification": True,
            "v50r2_reference_only": True,
        },
    }
    write_json(MANIFEST, payload)
    return payload


def write_bundle(files: list[Path]) -> None:
    BUNDLES.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BUNDLE, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in files + [MANIFEST]:
            z.write(path, rel(path))


def write_final_status(v512: dict[str, Any], audit_pass: bool) -> dict[str, Any]:
    status = (
        "V9000000000000000000000_V50R2_DISTILLED_HUMAN_SCENE_POINTCLOUD_READY_NOT_PROMOTED"
        if audit_pass
        else "V9000000000000000000000_ARTIFACT_AUDIT_FAIL_CLOSED_NOT_PROMOTED"
    )
    payload = {
        "task": "V900_final_status",
        "status": status,
        "created_at": now(),
        "repo": str(ROOT),
        "manual_gate_status": v512["status"],
        "advisor_report": str(ADVISOR_REPORT),
        "viewer": str(VIEWER),
        "bundle": str(BUNDLE),
        "manifest": str(MANIFEST),
        "audit": str(AUDIT),
        "policy": {
            "not_promoted": True,
            "registry_modified": False,
            "v50_v50r2_modified": False,
            "active_candidate_replaced": False,
            "v50r2_reference_only": True,
        },
    }
    write_json(FINAL_STATUS, payload)
    return payload


def main() -> int:
    v512 = read_json(V512_DECISION)
    v523 = read_json(V523_DECISION)
    v524 = read_json(V524_DECISION)
    write_report(v512, v523, v524)
    write_viewer(v512, v523)
    files = included_files()
    manifest = write_manifest(files)
    write_bundle(files)
    zip_ok = False
    with zipfile.ZipFile(BUNDLE, "r") as z:
        zip_ok = z.testzip() is None
    ply_probe = read_ply_header(V523_OUT / "v523_cam21_true_full_scene_rgb.ply")
    npz_probe = npz_readable(V523_OUT / "v523_observation_anchor_control_part_binding_candidate.npz")
    viewer_probe = viewer_reference_check(VIEWER)
    v50_diff = git(["diff", "--name-only", "--", "V50", "V50R2", "reports/V50", "reports/V50R2"])
    audit_pass = bool(
        ADVISOR_REPORT.exists()
        and VIEWER.exists()
        and BUNDLE.exists()
        and zip_ok
        and viewer_probe["all_references_exist"]
        and ply_probe["readable"]
        and npz_probe["readable"]
        and read_json(REPORTS / "V5230000000000000000000_teacher_copy_check.json")["leak_detected"] is False
        and v512["gates"]["manual_mentor_gate_pass"] is True
        and not v50_diff
    )
    final_status = write_final_status(v512, audit_pass)
    audit = {
        "task": "V525_artifact_audit",
        "status": "V525_ARTIFACT_AUDIT_PASS_FINAL_READY_NOT_PROMOTED" if audit_pass else "V525_ARTIFACT_AUDIT_FAIL_CLOSED_NOT_PROMOTED",
        "created_at": now(),
        "repo": str(ROOT),
        "advisor_report": str(ADVISOR_REPORT),
        "viewer": str(VIEWER),
        "bundle": str(BUNDLE),
        "bundle_sha256": sha256(BUNDLE),
        "manifest": str(MANIFEST),
        "manifest_file_count": manifest["file_count"],
        "zip_test_ok": zip_ok,
        "viewer_probe": viewer_probe,
        "ply_probe": ply_probe,
        "npz_probe": npz_probe,
        "teacher_copy_leak_detected": False,
        "v50_v50r2_diff": v50_diff,
        "manual_gate_status": v512["status"],
        "final_status": final_status["status"],
        "gates": {
            "report_exists": ADVISOR_REPORT.exists(),
            "viewer_exists": VIEWER.exists(),
            "bundle_exists": BUNDLE.exists(),
            "bundle_zip_clean": zip_ok,
            "viewer_references_exist": viewer_probe["all_references_exist"],
            "ply_readable": ply_probe["readable"],
            "npz_readable": npz_probe["readable"],
            "manual_gate_pass": v512["gates"]["manual_mentor_gate_pass"],
            "no_v50_v50r2_diff": not v50_diff,
            "not_promoted": True,
        },
    }
    write_json(AUDIT, audit)
    # Re-write final status with audit hash now that audit exists.
    final_status["artifact_hashes"] = {
        "advisor_report": sha256(ADVISOR_REPORT),
        "viewer": sha256(VIEWER),
        "bundle": sha256(BUNDLE),
        "manifest": sha256(MANIFEST),
        "audit": sha256(AUDIT),
    }
    write_json(FINAL_STATUS, final_status)
    print(json.dumps({"status": audit["status"], "final_status": final_status["status"], "bundle": str(BUNDLE)}, indent=2, ensure_ascii=False))
    return 0 if audit_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
