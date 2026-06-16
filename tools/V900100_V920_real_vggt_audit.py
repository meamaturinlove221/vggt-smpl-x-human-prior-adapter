from __future__ import annotations

import csv
import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
FEATURE_REPO = Path(r"D:\vggt\vggt-feature-adapter")
MAIN_REPO = Path(r"D:\vggt\vggt-main")
SCENE_REPO = Path(r"D:\vggt\vggt-scene-context-evidence")
EVIDENCE_ROOT = MAIN_REPO / "local_report_auxiliary" / "V600_quality_rebuild"
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
OUTPUT = REPO / "output"
ARCHIVE = REPO / "archive"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def index_file(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": rel(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        "ext": path.suffix.lower(),
        "sha256": sha256(path) if path.exists() and path.is_file() else "",
        "role": "current" if path.exists() else "missing",
        "readable": False,
        "notes": "",
    }
    if not path.exists() or not path.is_file():
        return row
    try:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                bad = zf.testzip()
                row["readable"] = bad is None
                row["notes"] = f"members={len(zf.infolist())};bad={bad}"
        elif path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as z:
                row["readable"] = bool(z.files)
                row["notes"] = "keys=" + ",".join(z.files[:12])
        elif path.suffix.lower() == ".ply":
            with path.open("rb") as f:
                header = f.read(96)
            row["readable"] = header.startswith(b"ply\nformat ascii")
            row["notes"] = header[:80].decode("ascii", errors="replace").replace("\n", "|")
        elif path.suffix.lower() == ".png":
            with Image.open(path) as im:
                row["readable"] = True
                row["notes"] = f"{im.width}x{im.height}"
        elif path.suffix.lower() in {".html", ".md", ".json", ".csv", ".py"}:
            row["readable"] = path.stat().st_size > 0
        else:
            row["readable"] = path.stat().st_size > 0
    except Exception as exc:
        row["readable"] = False
        row["notes"] = repr(exc)
    return row


def current_artifact_paths() -> list[Path]:
    paths: list[Path] = []
    paths.extend(sorted(ARCHIVE.glob("V550000000000000_*bundle.zip")))
    for name in [
        "V900000000000000_final_status.json",
        "V410000000000000_mentor_main_gate.json",
        "V500000000000000_smpl_feature_bound_advisor_report.md",
        "V310000000000000_smpl_feature_binding_audit.json",
        "V310000000000000_posthoc_vs_vggt_path_decision.json",
        "V340000000000000_seed_metrics.csv",
        "V370000000000000_smpl_feature_control_firewall.csv",
        "V550000000000000_bundle_integrity.json",
        "V550000000000000_upload_manifest_sidecar.json",
    ]:
        paths.append(REPORTS / name)
    for name in [
        "V350000000000000_advisor_main_rgb_pointcloud.png",
        "V350000000000000_same_scene_baseline_true_controls.png",
        "V360000000000000_head_face_hair_detail_board.png",
        "V360000000000000_hand_arm_detail_board.png",
        "V360000000000000_clothing_boundary_board.png",
        "V400000000000000_multisequence_main_summary.png",
    ]:
        paths.append(BOARDS / name)
    paths.extend(sorted((OUTPUT / "V340000000000000_smpl_feature_bound_runs").rglob("predictions.npz"))[:16])
    paths.extend(sorted((OUTPUT / "V340000000000000_smpl_feature_bound_runs").rglob("student_active_rgb.ply"))[:16])
    paths.extend(sorted((OUTPUT / "V350000000000000_smpl_feature_bound_scene").glob("*.npz"))[:12])
    paths.extend(sorted((OUTPUT / "V350000000000000_smpl_feature_bound_scene").glob("*.ply"))[:12])
    return paths


def scan_vggt_repo(root: Path) -> dict[str, Any]:
    vggt_py = root / "vggt" / "models" / "vggt.py"
    agg_py = root / "vggt" / "models" / "aggregator.py"
    vggt_text = vggt_py.read_text(encoding="utf-8", errors="replace") if vggt_py.exists() else ""
    agg_text = agg_py.read_text(encoding="utf-8", errors="replace") if agg_py.exists() else ""
    checkpoints = []
    for pattern in ("*.pt", "*.pth", "*.ckpt", "*.safetensors"):
        checkpoints.extend(root.rglob(pattern))
    # Avoid indexing enormous trees in the report; keep paths and sizes only.
    ckpt_rows = [
        {"path": str(p), "size_bytes": p.stat().st_size}
        for p in sorted(checkpoints, key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)[:25]
        if p.is_file()
    ]
    tools = [
        root / "tools" / "v153_smoke_real_forward.py",
        root / "tools" / "v390_train_vggt_sparse_token_adapter.py",
        root / "tools" / "v540_train_vggt_sparse_token_adapter.py",
        root / "tools" / "v16_human_prior_adapter_probe.py",
    ]
    return {
        "root": str(root),
        "exists": root.exists(),
        "vggt_py": str(vggt_py),
        "vggt_py_exists": vggt_py.exists(),
        "aggregator_py": str(agg_py),
        "aggregator_py_exists": agg_py.exists(),
        "vggt_forward_exists": "class VGGT" in vggt_text and "def forward" in vggt_text,
        "aggregator_exists": "class Aggregator" in agg_text and "def forward" in agg_text,
        "vggt_forward_accepts_sparse_prior_tokens": "sparse_prior_tokens" in vggt_text,
        "aggregator_accepts_sparse_prior_tokens": "sparse_prior_tokens" in agg_text and "_build_sparse_prior_tokens" in agg_text,
        "sparse_prior_adapter_exists": "sparse_prior_adapter" in agg_text and "TokenPriorAdapter" in agg_text,
        "checkpoint_candidates": ckpt_rows,
        "checkpoint_candidate_count_limited": len(ckpt_rows),
        "reference_tools": [
            {"path": str(p), "exists": p.exists(), "mentions_real_forward": ("VGGT(" in p.read_text(encoding="utf-8", errors="replace") if p.exists() else False)}
            for p in tools
        ],
    }


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    final = read_json(REPORTS / "V900000000000000_final_status.json")
    gate = read_json(REPORTS / "V410000000000000_mentor_main_gate.json")
    v340_rows = []
    v340_path = REPORTS / "V340000000000000_seed_metrics.csv"
    if v340_path.exists():
        with v340_path.open("r", encoding="utf-8", newline="") as f:
            v340_rows = list(csv.DictReader(f))
    tiny_or_synthetic = any("smpl_feature_bound" in row.get("config", "") for row in v340_rows)
    freeze = {
        "created_at": utc_now(),
        "repo": str(REPO),
        "previous_status": final.get("final_status", "missing"),
        "previous_hard_gate_pass": final.get("hard_gate_pass", False),
        "downgraded_to_checkpoint": True,
        "downgrade_reasons": [
            "V340 final run used TinyV330-style local model and scene_tokens generated from anchor/rgb/mask random projection.",
            "V340 did not execute full real VGGT.forward or real Aggregator tokens from current images as final evidence.",
            "V350 boards are preserved, but local face/head/hair/hand/clothing detail remains a risk and must be defended against real VGGT baseline.",
            "Strong topology/posthoc/template controls require a stricter real-token control firewall.",
        ],
        "v410_gate_preserved": gate,
        "v340_rows": len(v340_rows),
        "tiny_v330_or_synthetic_token_risk": tiny_or_synthetic,
        "no_agent_rule": "No agents/subagents launched.",
    }
    write_json(REPORTS / "V900100000000000_v900_checkpoint_freeze.json", freeze)
    write_text(
        REPORTS / "V900100000000000_why_v900_is_not_final.md",
        """# Why V900 Is Not Final

V900 is preserved as a useful checkpoint, but it is not the final mentor deliverable for the new route.

- The V330 model file describes SMPL feature binding, but the V340 Modal evidence used a TinyV330-style model.
- V340 scene tokens were generated from anchor/rgb/mask random projections, not real VGGT.forward or real Aggregator tokens.
- The mentor main board is full-scene and human-main, but local detail remains too template-like to claim final real-VGGT detail improvement.
- Source-label and visible-delta evidence remains auxiliary.

The next hard gate is real VGGT.forward or real Aggregator token extraction, followed by SMPL feature binding on those real tokens.
""",
    )
    write_text(
        REPORTS / "V900100000000000_target_drift_risk.md",
        """# V900100 Target Drift Risk

Risk: accepting TinyV330/synthetic token evidence as real VGGT token binding.

Fail-closed rule:

- synthetic scene tokens cannot close V180;
- posthoc point composition cannot close V180;
- source-label or visible-delta boards cannot close V180;
- a metric/control score cannot replace a full-scene human-main RGB point cloud;
- if real VGGT code/checkpoint/token extraction is impossible, write TRUE_EXTERNAL_HARD_BLOCK with precise user actions.
""",
    )

    rows = [index_file(p) for p in current_artifact_paths()]
    write_csv(REPORTS / "V910000000000000_current_artifact_index.csv", rows)
    expired_patterns = [
        "Pre-V550 bundles and V260/V300/V850/V895 boards are historical unless explicitly re-indexed in V910.",
        "V900 boards remain checkpoint evidence only; final V180 report must use V970/V980/V990/V110 outputs.",
        "Diagnostic/source-label/projection images cannot be mentor main evidence.",
    ]
    write_text(
        REPORTS / "V910000000000000_expired_or_obsolete_evidence.md",
        "# Expired Or Obsolete Evidence\n\n" + "\n".join(f"- {item}" for item in expired_patterns),
    )
    write_json(
        REPORTS / "V910000000000000_current_evidence_decision.json",
        {
            "created_at": utc_now(),
            "current_file_count": len(rows),
            "all_current_readable": all(r["readable"] for r in rows if r["exists"]),
            "zip_clean_count": sum(1 for r in rows if r["ext"] == ".zip" and r["readable"]),
            "npz_readable_count": sum(1 for r in rows if r["ext"] == ".npz" and r["readable"]),
            "ply_readable_count": sum(1 for r in rows if r["ext"] == ".ply" and r["readable"]),
            "png_readable_count": sum(1 for r in rows if r["ext"] == ".png" and r["readable"]),
            "diagnostic_not_main_rule": True,
            "v900_current_evidence_preserved_but_downgraded": True,
        },
    )

    inventory = {
        "created_at": utc_now(),
        "repos": [scan_vggt_repo(p) for p in [REPO, FEATURE_REPO, MAIN_REPO]],
        "evidence_root": str(EVIDENCE_ROOT),
    }
    real_path_pass = any(
        r["vggt_forward_exists"]
        and r["aggregator_exists"]
        and r["vggt_forward_accepts_sparse_prior_tokens"]
        and r["aggregator_accepts_sparse_prior_tokens"]
        for r in inventory["repos"]
    )
    checkpoint_candidates = [ck for repo in inventory["repos"] for ck in repo["checkpoint_candidates"]]
    inventory["real_vggt_code_path_pass"] = real_path_pass
    inventory["checkpoint_candidate_count"] = len(checkpoint_candidates)
    inventory["checkpoint_required_for_full_pretrained_forward"] = True
    inventory["modal_gpu_feasible"] = True
    inventory["enter_v930_allowed"] = bool(real_path_pass)
    write_json(REPORTS / "V920000000000000_real_vggt_path_inventory.json", inventory)
    write_text(
        REPORTS / "V920000000000000_real_vggt_adapter_feasibility.md",
        f"""# V920 Real VGGT Adapter Feasibility

Decision: {"enter V930 token extraction" if real_path_pass else "hard block: real VGGT code path missing"}

Findings:

- Current repo has VGGT.forward and Aggregator: {inventory['repos'][0]['vggt_forward_exists']} / {inventory['repos'][0]['aggregator_exists']}
- Current repo VGGT.forward accepts sparse_prior_tokens: {inventory['repos'][0]['vggt_forward_accepts_sparse_prior_tokens']}
- Current repo Aggregator accepts sparse_prior_tokens: {inventory['repos'][0]['aggregator_accepts_sparse_prior_tokens']}
- Feature-adapter and vggt-main were also scanned.
- Checkpoint candidates found in scanned roots: {len(checkpoint_candidates)}

V930 must execute real VGGT.forward or real Aggregator.forward and save tokens with provenance. TinyV330 synthetic tokens are forbidden from final evidence.
""",
    )
    patch_items = []
    if not real_path_pass:
        patch_items.append("Provide or restore real VGGT code with VGGT.forward and Aggregator.forward.")
    if not checkpoint_candidates:
        patch_items.append("Provide VGGT pretrained checkpoint if full pretrained forward is required.")
    patch_items.extend(
        [
            "Add V930 token extraction script that records real Aggregator/VGGT provenance.",
            "Add V950 real-token adapter smoke that proves gradient/effect on real VGGT tokens.",
            "Keep TinyV330 only as a control, never final evidence.",
        ]
    )
    write_text(REPORTS / "V920000000000000_required_patch_list.md", "# V920 Required Patch List\n\n" + "\n".join(f"- {x}" for x in patch_items))
    print(json.dumps({"V900100": "written", "V910_files": len(rows), "V920_real_path_pass": real_path_pass, "checkpoint_candidates": len(checkpoint_candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
