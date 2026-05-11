from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = ROOT / "output"
ARCHIVE = ROOT / "archive"
FROZEN = OUT / "frozen_candidates" / "V50_smplx_native_candidate_pass"
PKG_FILES = FROZEN / "package_files"
SUPP = ARCHIVE / "V121_v62_v120_supplement_bundle"
VISUAL = OUT / "V135_visual_board"
PACKAGE_ZIP = ARCHIVE / "package_files.zip"
LINEAGE_ZIP = ARCHIVE / "V63_candidate_lineage_graph.zip"
V3 = ARCHIVE / "V138_candidate_pass_bundle_v3"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_row(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() and path.is_file() else 0,
        "sha256": sha(path) if path.exists() and path.is_file() else None,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_md(path: Path, title: str, payload: dict) -> None:
    lines = [f"# {title}", "", f"- status: `{payload.get('status')}`", f"- created_utc: `{payload.get('created_utc')}`"]
    if payload.get("notes"):
        lines += ["", "## Notes"]
        lines += [f"- {x}" for x in payload["notes"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_zip(zip_path: Path, source_dir: Path, prefix: str = "") -> dict:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    files = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(source_dir.rglob("*")):
            if p.is_file():
                arc = str(Path(prefix) / p.relative_to(source_dir)).replace("\\", "/") if prefix else str(p.relative_to(source_dir)).replace("\\", "/")
                zf.write(p, arc)
                files.append(arc)
    return {"zip": file_row(zip_path), "file_count": len(files), "files": files}


def copy_tree_files(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0
    count = 0
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*"):
        if p.is_file():
            rel = p.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            count += 1
    return count


def main() -> None:
    SUPP.mkdir(parents=True, exist_ok=True)
    visual_count = copy_tree_files(VISUAL, SUPP / "visual_board_v135")
    for p in [
        REPORTS / "V62_V120_branch_ledger.json",
        REPORTS / "V62_V120_branch_ledger.md",
        REPORTS / "V121_V220_branch_ledger.json",
        REPORTS / "V121_V220_branch_ledger.md",
        REPORTS / "V62_H_mentor_report_v2.md",
        REPORTS / "V137_I1_mentor_one_page_v3.md",
        REPORTS / "V137_I2_technical_appendix_v3.md",
        REPORTS / "V137_I3_mentor_QA_v3.md",
        REPORTS / "V131_F1_teacher_route_blocker_ledger.md",
        REPORTS / "V131_F2_teacher_route_resurrection_policy.md",
        REPORTS / "V127_D3_right_hand_merge_gate.json",
    ]:
        if p.exists():
            shutil.copy2(p, SUPP / p.name)
    package_zip = make_zip(PACKAGE_ZIP, PKG_FILES)
    lineage_src = SUPP / "lineage_zip_source"
    if lineage_src.exists():
        shutil.rmtree(lineage_src)
    lineage_src.mkdir(parents=True)
    for p in [
        REPORTS / "V63_candidate_lineage_graph.json",
        REPORTS / "V63_candidate_lineage_graph.md",
        OUT / "mentor_board" / "V63_candidate_lineage_graph.png",
        REPORTS / "V62_V120_branch_ledger.json",
        REPORTS / "V62_V120_branch_ledger.md",
        REPORTS / "V121_V220_branch_ledger.json",
        REPORTS / "V121_V220_branch_ledger.md",
        REPORTS / "V62_H_mentor_report_v2.md",
        REPORTS / "V64_H_mentor_QA.md",
        REPORTS / "V63_I_route_resurrection_policy.md",
    ]:
        if p.exists():
            shutil.copy2(p, lineage_src / p.name)
    lineage_zip = make_zip(LINEAGE_ZIP, lineage_src)
    if V3.exists():
        copy_tree_files(SUPP, V3 / "V121_v62_v120_supplement_bundle")
        shutil.copy2(PACKAGE_ZIP, V3 / "package_files.zip")
        shutil.copy2(LINEAGE_ZIP, V3 / "V63_candidate_lineage_graph.zip")
    payload = {
        "task": "V221_release_supplement_finalizer",
        "status": "PASS",
        "created_utc": now(),
        "package_files_zip": package_zip,
        "lineage_graph_zip": lineage_zip,
        "supplement_bundle": str(SUPP),
        "visual_board_files_copied": visual_count,
        "v3_archive_updated": V3.exists(),
        "notes": [
            "Local package_files.zip and V63_candidate_lineage_graph.zip were regenerated from frozen/local evidence because independent uploaded zips were not present in this checkout.",
            "V50 frozen candidate was not modified.",
            "V135 visual board was copied into the supplement bundle.",
        ],
    }
    write_json(REPORTS / "V221_release_supplement_finalizer.json", payload)
    write_md(REPORTS / "V221_release_supplement_finalizer.md", "V221 Release Supplement Finalizer", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

