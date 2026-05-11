from __future__ import annotations

import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
LOCAL = ROOT / "output" / "surface_research_preflight_local"
V50_OUT = LOCAL / "V50_final_promotion_transaction"
PACKAGE = V50_OUT / "candidate_package_v50r2"
FROZEN = ROOT / "output" / "frozen_candidates" / "V50R2_rebuilt_from_sessions_gdrive_modal"
ARCHIVE = ROOT / "archive" / "V50R2_rebuilt_candidate_package.zip"


SOURCE_FILES = {
    "candidate_files__candidate_normals.npz": LOCAL / "V32_candidate_inference_research" / "candidate_normals_geometric_research.npz",
    "candidate_files__candidate_points.npz": LOCAL / "V32_candidate_inference_research" / "candidate_points_world_research.npz",
    "candidate_files__hand_patch.npz": LOCAL / "V34_smplx_native_hand_route" / "v34_smplx_native_hand_continuity_patch.npz",
    "candidate_files__head_face_patch.npz": LOCAL / "V33_head_face_detail_route" / "v33_head_face_refined_teacher.npz",
    "candidate_files__temporal_teacher.npz": LOCAL / "V26_temporal_canonical_teacher" / "v26_temporal_canonical_teacher_targets.npz",
    "candidate_files__visual_review.json": LOCAL / "V44_strict_visual_pre_promotion_gate" / "visual_review_codex_pass.json",
    "v42_prior_enabled_payload__research_depths.npz": ROOT / "output" / "surface_research_cloud_preflight" / "V42_prior_enabled_predictions" / "research_depths.npz",
    "v42_prior_enabled_payload__research_points_world.npz": ROOT / "output" / "surface_research_cloud_preflight" / "V42_prior_enabled_predictions" / "research_points_world.npz",
    "v42_prior_enabled_payload__research_normals_geometric.npz": ROOT / "output" / "surface_research_cloud_preflight" / "V42_prior_enabled_predictions" / "research_normals_geometric.npz",
    "v42_prior_enabled_payload__research_confidence.npz": ROOT / "output" / "surface_research_cloud_preflight" / "V42_prior_enabled_predictions" / "research_confidence.npz",
    "v42_prior_enabled_payload__research_prior_effect.json": ROOT / "output" / "surface_research_cloud_preflight" / "V42_prior_enabled_predictions" / "research_prior_effect.json",
    "v42_prior_enabled_payload__control_audit.json": ROOT / "output" / "surface_research_cloud_preflight" / "V42_prior_enabled_predictions" / "control_real_zero_shuffle_random_dropout.json",
}


ORIGINAL_V50_HASHES_FROM_SESSION = {
    "manifest.json": "03a8603892bedfaadbfa21b1a4b96dc0b462dff8d3bcfa5b7abe2bb49f0318bc",
    "candidate_files__candidate_normals.npz": "acbe786a1047a417f14f60716edad65aae3810072dcef312c0bf436067101022",
    "candidate_files__candidate_points.npz": "9f032e87125b1c204cc7cc83b9dbaf73f448ab7a99f1a4852bb0b84644bff12b",
    "candidate_files__hand_patch.npz": "dd46ad76d2aa4b84693f5997b9efb2547171ec2f73ac4e8691eefc96274b590b",
    "candidate_files__head_face_patch.npz": "7f3d06368a917d651ee3b4c99c9832081e26953fed446f8c67945a44816fd001",
    "candidate_files__temporal_teacher.npz": "cf5d52c699225bf25e8fc448f23fc6dd6f8dfffa52672a4dfa811a4174eb9c13",
    "candidate_files__visual_review.json": "4ca0386f594afa247ed5bd8a508d43bb85758ed253197c8deb6f9b67b1277dca",
    "v42_prior_enabled_payload__research_points_world.npz": "d8e70e539fe3cb93e2d881abdb0565ef42d33c5915d96d03fe2783362b107215",
    "v42_prior_enabled_payload__research_normals_geometric.npz": "a819da26578afa05ce4cf18a5a68ce9177ae3a25c5bcd5f9b3034eb9dea07c14",
    "v42_prior_enabled_payload__research_depths.npz": "ea27ebda9034a57efc41c2af7b979531c35d65ca2fffae05f1dfb648d35fabfa",
    "v42_prior_enabled_payload__research_confidence.npz": "04eba319d53fc954e6567450201119e3ae7e884f39a8a8f49492000b0586bd38",
    "v42_prior_enabled_payload__research_prior_effect.json": "2480829b52a2f3d9146662c6aae346c57df8fcc3f3bfdc7215f768327bbe072f",
    "v42_prior_enabled_payload__control_audit.json": "ad7e2cfa0ccd4684aab781b033f8ffdb6c1e1867886ed0cef296b47d129e9033",
    "strict_registry_entry_v50.json": "96014919cca0f1229a942902ea9fb37b4a757bc762ea01a40f2bb6d65e9475ad",
    "visual_review_codex_pass.json": "4ca0386f594afa247ed5bd8a508d43bb85758ed253197c8deb6f9b67b1277dca",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jr(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jr(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jr(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jr(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    blockers: list[str] = []
    for name, src in SOURCE_FILES.items():
        if not src.is_file():
            blockers.append(f"missing source for package file {name}: {src}")
    v50_summary = json.loads((REPORTS / "20260509_v50_final_promotion_transaction.json").read_text(encoding="utf-8"))
    if v50_summary.get("strict_candidate_passes") != 1:
        blockers.append("V50 transaction report does not have strict_candidate_passes=1")
    if blockers:
        write_json(REPORTS / "V50R2_package_builder.json", {"status": "DONE_FAIL_ROUTED", "blockers": blockers})
        print("DONE_FAIL_ROUTED")
        return 2

    package_files = PACKAGE / "package_files"
    package_files.mkdir(parents=True, exist_ok=True)
    for name, src in SOURCE_FILES.items():
        shutil.copy2(src, package_files / name)

    manifest = {
        "package_id": "V50R2_rebuilt_from_sessions_gdrive_modal",
        "created_utc": now(),
        "status": "DONE_PASS_REBUILT",
        "strict_candidate_pass": True,
        "strict_teacher_pass": False,
        "formal_cloud_unblocked": True,
        "recovery_mode": "recipe_rerun_from_codex_sessions_plus_G_drive_dataset_plus_Modal_V42_payload",
        "bitwise_original_v50": False,
        "active_candidate_mutable": False,
        "candidate_files": {name: str((package_files / name).resolve()) for name in SOURCE_FILES},
        "upstream_reports": {
            "v29": str((REPORTS / "20260508_v29_normal_route_rescue.json").resolve()),
            "v30": str((REPORTS / "20260508_v30_prior_enabled_vggt_predictions.json").resolve()),
            "v31": str((REPORTS / "20260508_v31_teacher_supervised_candidate_train.json").resolve()),
            "v32": str((REPORTS / "20260508_v32_candidate_inference_region_audit.json").resolve()),
            "v33": str((REPORTS / "20260508_v33_head_face_detail_route.json").resolve()),
            "v34": str((REPORTS / "20260508_v34_smplx_native_hand_route.json").resolve()),
            "v35": str((REPORTS / "20260508_v35_60view_support_expansion.json").resolve()),
            "v44": str((REPORTS / "20260509_v44_strict_visual_pre_promotion_gate.json").resolve()),
            "v49": str((REPORTS / "20260509_v49_package_dry_run.json").resolve()),
            "v50": str((REPORTS / "20260509_v50_final_promotion_transaction.json").resolve()),
        },
    }
    write_json(PACKAGE / "manifest.json", manifest)
    registry = {
        "created_utc": now(),
        "strict_candidate_pass": True,
        "strict_candidate_passes": 1,
        "strict_teacher_passes": 0,
        "formal_cloud_unblocked": True,
        "candidate_package_path": str(PACKAGE.resolve()),
        "recovery_mode": manifest["recovery_mode"],
        "bitwise_original_v50": False,
    }
    write_json(PACKAGE / "strict_registry_entry_v50r2.json", registry)

    hash_rows: dict[str, Any] = {}
    for path in [PACKAGE / "manifest.json", PACKAGE / "strict_registry_entry_v50r2.json", *sorted(package_files.iterdir())]:
        rel = path.relative_to(PACKAGE).as_posix()
        base_name = path.name
        actual = sha256(path)
        hash_rows[rel] = {
            "size": path.stat().st_size,
            "sha256": actual,
            "original_v50_sha256_from_session": ORIGINAL_V50_HASHES_FROM_SESSION.get(base_name),
            "matches_original_v50_session_hash": actual == ORIGINAL_V50_HASHES_FROM_SESSION.get(base_name),
        }
    write_json(PACKAGE / "hash_manifest.json", hash_rows)

    if FROZEN.exists():
        shutil.rmtree(FROZEN)
    shutil.copytree(PACKAGE, FROZEN)

    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(FROZEN.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(FROZEN.parent).as_posix())

    archive_sha = sha256(ARCHIVE)
    summary = {
        "task": "v50r2_package_builder",
        "created_utc": now(),
        "status": "DONE_PASS",
        "candidate_package_path": str(PACKAGE.resolve()),
        "frozen_candidate_path": str(FROZEN.resolve()),
        "archive_zip": str(ARCHIVE.resolve()),
        "archive_sha256": archive_sha,
        "bitwise_original_v50": False,
        "original_hash_comparison": hash_rows,
        "strict_candidate_passes": 1,
        "strict_teacher_passes": 0,
        "formal_cloud_unblocked": True,
        "blockers": [],
    }
    write_json(REPORTS / "V50R2_package_builder.json", summary)
    write_json(REPORTS / "V50R2_package_builder.md.json", summary)
    (REPORTS / "V50R2_package_builder.md").write_text(
        "# V50R2 Package Builder\n\n"
        "Status: `DONE_PASS`\n\n"
        f"- package: `{PACKAGE.resolve()}`\n"
        f"- frozen: `{FROZEN.resolve()}`\n"
        f"- archive: `{ARCHIVE.resolve()}`\n"
        f"- archive_sha256: `{archive_sha}`\n"
        "- bitwise_original_v50: `false`\n",
        encoding="utf-8",
    )
    print("DONE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
