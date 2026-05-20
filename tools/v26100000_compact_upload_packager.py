from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = ROOT / "reports"
BOARDS = ROOT / "boards"
ARCHIVE = ROOT / "archive"
RUN_ROOT = ROOT / "output" / "V10000000_V12000000_modal_sparseconv"

MB = 1024 * 1024
TOTAL_LIMIT = 500 * MB
TARGET_TOTAL = 380 * MB
PER_PACK_LIMIT = 220 * MB


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def file_row(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else 0,
    }
    if path.is_file():
        row["sha256"] = sha256(path)
    return row


def make_zip(path: Path, files: list[tuple[Path, str]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    seen: set[str] = set()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
        for source, arc in files:
            if not source.is_file():
                continue
            arc = arc.replace("\\", "/")
            if arc in seen:
                arc = f"duplicates/{len(seen):03d}_{Path(arc).name}"
            seen.add(arc)
            zf.write(source, arc)
    row = file_row(path)
    with zipfile.ZipFile(path, "r") as zf:
        row["zip_test"] = zf.testzip() or "clean"
        row["entry_count"] = len(zf.infolist())
    row["under_per_pack_limit"] = row["size"] <= PER_PACK_LIMIT
    return row


def rel_report(name: str) -> tuple[Path, str]:
    return REPORTS / name, f"reports/{name}"


def rel_board(name: str) -> tuple[Path, str]:
    return BOARDS / name, f"boards/{name}"


def prediction(path: Path, label: str) -> tuple[Path, str]:
    return path, f"predictions/{label}/predictions.npz"


def main() -> None:
    core_files = [
        rel_report("V26000000_final_status.json"),
        rel_report("V25000000_upload_manifest.json"),
        rel_report("V24000000_advisor_report.md"),
        rel_report("V24000000_advisor_one_page.md"),
        rel_report("V24000000_limitations.md"),
        rel_report("V23000000_causal_robustness_summary.json"),
        rel_report("V23000000_causal_robustness_v3.csv"),
        rel_report("V23100000_geometry_quality.csv"),
        rel_report("V23200000_hand_eval.json"),
        rel_report("V23200000_hairline_eval.json"),
        rel_report("V23400000_heldout_eval.json"),
        rel_report("V23500000_composition_search.csv"),
        rel_report("V23600000_strict_final_eval.json"),
        rel_report("V22000000_final_status.json"),
        rel_report("V22100000_mentor_goal_closure.json"),
        rel_report("V22100000_mentor_goal_closure.md"),
        rel_report("V19500000_cleanup_report.json"),
        rel_report("V22300000_v220_v221_evidence_repair.json"),
        rel_report("V22400000_hash_reconciliation.json"),
        rel_report("V22400000_hash_reconciliation.csv"),
        rel_report("V25500000_post_push_cleanup_report.json"),
    ]
    visual_files = [
        rel_board("V23300000_final_fullbody.png"),
        rel_board("V23300000_final_head_hair_hand.png"),
        rel_board("V23300000_causal_controls.png"),
        rel_board("V23300000_failure_cases.png"),
        rel_board("V23100000_quality_visualization.png"),
        rel_board("V23200000_hand_hair_specialist.png"),
        rel_board("V23400000_heldout_board.png"),
    ]
    main_predictions = [
        prediction(
            RUN_ROOT / "V126_no_v129_highscale_fast_20260520" / "candidates" / "cand_032_spconv_humanram_mix_s2p00" / "predictions.npz",
            "main_no_v129_highscale",
        ),
        prediction(
            RUN_ROOT / "V125_smpl_only_20260520" / "candidates" / "cand_032_spconv_humanram_mix_s1p25" / "predictions.npz",
            "smpl_only_control",
        ),
    ]
    control_predictions = [
        prediction(
            RUN_ROOT / "V157_observation_only_20260520" / "candidates" / "cand_029_spconv_humanram_mix_s2p00" / "predictions.npz",
            "observation_only_control",
        ),
        prediction(
            RUN_ROOT / "V230_smoke_no_sparseconv_mlp_seed0" / "candidates" / "cand_007_spconv_s0p83" / "predictions.npz",
            "no_sparseconv_mlp_control",
        ),
        prediction(
            RUN_ROOT / "V230_smoke_quiet_seed0" / "candidates" / "cand_003_spconv_s0p33" / "predictions.npz",
            "shuffled_smpl_full_control",
        ),
    ]

    bundles = {
        "core_evidence": make_zip(ARCHIVE / "V26100000_core_evidence_bundle.zip", core_files),
        "visuals": make_zip(ARCHIVE / "V26100000_visuals_bundle.zip", visual_files),
        "predictions_main": make_zip(ARCHIVE / "V26100000_predictions_main_bundle.zip", main_predictions),
        "predictions_controls": make_zip(ARCHIVE / "V26100000_predictions_controls_bundle.zip", control_predictions),
    }
    total_size = sum(b["size"] for b in bundles.values())
    omitted = []
    for p in sorted(ARCHIVE.glob("V25000000_candidate_shard_*.zip")):
        row = file_row(p)
        row["reason"] = "omitted_from_compact_upload_to_keep_total_under_500mb"
        omitted.append(row)
    manifest = {
        "created_utc": now(),
        "status": "V26100000_COMPACT_UPLOAD_SET_READY",
        "policy": {
            "total_upload_limit_bytes": TOTAL_LIMIT,
            "target_total_bytes": TARGET_TOTAL,
            "per_pack_limit_bytes": PER_PACK_LIMIT,
            "future_default": "Prefer compact evidence + visuals + key predictions only; omit bulk candidate shards unless specifically requested.",
        },
        "bundles": bundles,
        "total_size": total_size,
        "under_total_500mb": total_size <= TOTAL_LIMIT,
        "under_target_total": total_size <= TARGET_TOTAL,
        "omitted_large_files": omitted,
    }
    write_json(REPORTS / "V26100000_compact_upload_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
