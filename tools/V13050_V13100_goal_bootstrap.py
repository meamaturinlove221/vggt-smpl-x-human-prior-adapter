from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
OUTPUT = REPO / "output"
DOCS = REPO / "docs" / "goals"

KEY_PATHS = [
    REPORTS / "V13040000000000000000_anti_billboard_decision.json",
    REPORTS / "V13040000000000000000_anti_billboard_audit.csv",
    REPORTS / "V13050000000000000000_topology_volume_occupancy_decision.json",
    REPORTS / "V13050000000000000000_topology_volume_occupancy_metrics.csv",
    REPORTS / "V13050000000000000000_topology_volume_occupancy_manifest.csv",
    REPORTS / "V13030000000000000000_current_volume_route_state.md",
    DOCS / "V13000000000000000000_auto_evolved_volume_morphology_route.md",
    BOARDS / "V13040000000000000000_anti_billboard_cross_section.png",
    BOARDS / "V13040000000000000000_anti_billboard_turntable.png",
    BOARDS / "V13050000000000000000_topology_occupancy_cross_section.png",
    BOARDS / "V13050000000000000000_topology_occupancy_turntable.png",
]

AUDIT_ROOTS = [
    REPORTS,
    BOARDS,
    REPO / "viewer",
    OUTPUT / "V10400000000000000000_weak_volume_regions",
    OUTPUT / "V10700000000000000000_volume_aware_training_matrix",
    OUTPUT / "V13010000000000000000_volume_shell_repair_candidate",
    OUTPUT / "V13020000000000000000_topology_coherent_volume_candidate",
    OUTPUT / "V13050000000000000000_topology_volume_occupancy_candidate",
]


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure(path.parent)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["path"])
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return "npz"
    if suffix == ".ply":
        return "ply"
    if suffix == ".png":
        return "png"
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix in {".md", ".txt"}:
        return "text"
    if suffix == ".html":
        return "html"
    if suffix == ".zip":
        return "zip"
    if suffix == ".py":
        return "py"
    return suffix.lstrip(".") or "unknown"


def quick_readable(path: Path, kind: str) -> tuple[bool, str]:
    try:
        if kind == "npz":
            with np.load(path, allow_pickle=False) as z:
                _ = z.files
            return True, "npz_readable"
        if kind == "png":
            with Image.open(path) as im:
                im.verify()
            return True, "png_open"
        if kind == "json":
            json.loads(path.read_text(encoding="utf-8-sig"))
            return True, "json_readable"
        if kind in {"csv", "text", "html", "py"}:
            _ = path.read_text(encoding="utf-8", errors="replace")[:4096]
            return True, f"{kind}_readable"
        if kind == "ply":
            with path.open("rb") as f:
                head = f.read(64)
            return head.startswith(b"ply"), "ply_header"
        if kind == "zip":
            with zipfile.ZipFile(path) as z:
                bad = z.testzip()
            return bad is None, "zip_clean" if bad is None else f"zip_bad:{bad}"
    except Exception as exc:
        return False, repr(exc)
    return True, "unchecked"


def freeze_v13050(created_at: str) -> None:
    decision = read_json(REPORTS / "V13050000000000000000_topology_volume_occupancy_decision.json")
    v13040 = read_json(REPORTS / "V13040000000000000000_anti_billboard_decision.json")
    freeze = {
        "created_at": created_at,
        "status": "V13050_PROCEDURAL_OCCUPANCY_CHECKPOINT_INTERNAL_VISUAL_HARD_BLOCK",
        "mentor_ready": False,
        "external_hard_block": False,
        "route_exhausted": False,
        "limitation_disclosed_final": False,
        "source_status": decision.get("status"),
        "v13040_true_billboard_fail_cases": v13040.get("true_billboard_fail_cases", []),
        "v13050_failures": decision.get("failures", []),
        "why": [
            "V13050 is procedural occupancy, not a trained anti-billboard topology-volume student.",
            "V13050 visuals still read as torn multi-layer textured sheets.",
            "Anti-billboard score improvement is not equivalent to mentor visual pass.",
            "same-topology/shuffled controls remain close or stronger in multiple cases.",
            "Next route must enter trained anti-billboard topology-volume student.",
        ],
        "face_detail_claim_allowed": False,
        "allowed_face_claim": "head/face contour and hair region only",
    }
    write_json(REPORTS / "V13050000000000000000_v13050_checkpoint_freeze.json", freeze)
    (REPORTS / "V13050000000000000000_why_v13050_is_not_final.md").write_text(
        "# Why V13050 Is Not Final\n\n"
        "V13050 is a checkpoint and internal visual hard block. It is not mentor-ready, not an external hard block, and not route-exhausted.\n\n"
        "- The candidate is still procedural occupancy rather than a trained topology-volume student.\n"
        "- The turntable and cross-section boards still read as torn multi-layer textured sheets.\n"
        "- A higher anti-billboard score does not prove mentor visual success.\n"
        "- Same-topology and shuffled controls remain close or stronger in the anti-billboard gate.\n"
        "- Face detail is not applicable; only head/face contour and hair region may be claimed.\n\n"
        "The next route must train an anti-billboard topology-volume student with cross-section occupancy, part continuity, and hard-control separation.\n",
        encoding="utf-8",
    )
    write_json(
        REPORTS / "V13050000000000000000_billboard_failure_register.json",
        {
            "created_at": created_at,
            "failures": [
                {"gate": "mentor_visual", "reason": "human still reads as billboard/textured sheet"},
                {"gate": "representation", "reason": "procedural occupancy is not trained topology-volume representation"},
                {"gate": "controls", "reason": "same-topology/shuffled controls remain close or stronger"},
                {"gate": "local_detail", "reason": "local morphology remains contour/sheet-level"},
                {"gate": "claim_boundary", "reason": "face invisible; facial detail claims forbidden"},
            ],
        },
    )


def audit_artifacts(created_at: str) -> None:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in AUDIT_ROOTS:
        if not root.exists():
            rows.append({"path": str(root), "exists": False, "kind": "root", "readable": False, "note": "missing_root"})
            continue
        files = list(root.rglob("*")) if root.is_dir() else [root]
        for path in files:
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            if len(rows) > 2500:
                break
            kind = file_kind(path)
            readable, note = quick_readable(path, kind) if kind in {"npz", "ply", "png", "json", "csv", "text", "html", "zip", "py"} else (True, "not_checked")
            rows.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
                    "exists": True,
                    "kind": kind,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path) if path.stat().st_size < 64 * 1024 * 1024 else "",
                    "readable": readable,
                    "note": note,
                    "is_key_v13050_input": path in KEY_PATHS,
                }
            )
    write_csv(REPORTS / "V13100000000000000000_current_artifact_index.csv", rows)
    kind_counts: dict[str, int] = {}
    unreadable = []
    for row in rows:
        kind_counts[row["kind"]] = kind_counts.get(row["kind"], 0) + 1
        if row.get("exists") and not row.get("readable"):
            unreadable.append(row)
    missing_keys = [str(p) for p in KEY_PATHS if not p.exists()]
    quality = {
        "created_at": created_at,
        "repo": str(REPO),
        "checked_file_count": sum(1 for r in rows if r.get("exists")),
        "kind_counts": kind_counts,
        "missing_key_files": missing_keys,
        "unreadable_count": len(unreadable),
        "unreadable_examples": unreadable[:20],
        "procedural_candidate_final_allowed": False,
        "face_detail_claim_allowed": False,
        "mentor_ready": False,
        "external_hard_block": False,
    }
    write_json(REPORTS / "V13100000000000000000_current_artifact_quality_audit.json", quality)
    (REPORTS / "V13100000000000000000_current_route_decision.md").write_text(
        "# V13100 Current Route Decision\n\n"
        "Current files are sufficient to continue the route, but they do not prove mentor readiness.\n\n"
        f"- Checked files: {quality['checked_file_count']}\n"
        f"- Missing key V13050 files: {len(missing_keys)}\n"
        f"- Unreadable files: {len(unreadable)}\n"
        "- V13050 remains a procedural occupancy checkpoint.\n"
        "- Final success requires trained anti-billboard topology-volume output and mentor visual gates.\n"
        "- Visual failure is not an external hard block.\n",
        encoding="utf-8",
    )


def main() -> int:
    created_at = now()
    freeze_v13050(created_at)
    audit_artifacts(created_at)
    print(json.dumps({"created_at": created_at, "status": "V13050_V13100_BOOTSTRAP_DONE", "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
