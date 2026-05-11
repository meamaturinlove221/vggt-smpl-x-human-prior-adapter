from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = ROOT / "output"
ACTIVE = OUT / "frozen_candidates" / "V50R_rebuilt_after_artifact_loss"
ACTIVE_FILES = ACTIVE / "package_files"
PROMO = OUT / "V400_v50r_strict_promotion_transaction"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, payload: dict[str, Any]) -> None:
    lines = [f"# {title}", "", f"- Status: `{payload.get('status')}`"]
    for key in [
        "strict_candidate_passes_written",
        "strict_teacher_passes_written",
        "formal_cloud_unblocked",
        "decision",
    ]:
        if key in payload:
            lines.append(f"- {key}: `{payload[key]}`")
    blockers = payload.get("blockers") or []
    if blockers:
        lines += ["", "## Blockers"]
        lines += [f"- {b}" for b in blockers]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def npz_basic_gate(path: Path, keys: list[str]) -> tuple[bool, list[str], dict[str, Any]]:
    blockers: list[str] = []
    stats: dict[str, Any] = {"path": rel(path), "exists": path.exists(), "keys": {}}
    if not path.exists():
        return False, [f"missing {rel(path)}"], stats
    data = np.load(path, allow_pickle=True)
    for key in keys:
        if key not in data.files:
            blockers.append(f"missing array {key} in {rel(path)}")
            continue
        arr = data[key]
        row: dict[str, Any] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        if np.issubdtype(arr.dtype, np.number):
            finite = np.isfinite(arr)
            row["finite_ratio"] = float(finite.mean()) if finite.size else 0.0
            row["nonzero_ratio"] = float((arr != 0).mean()) if arr.size else 0.0
            if finite.size and finite.any():
                row["min"] = float(arr[finite].min())
                row["max"] = float(arr[finite].max())
                row["mean"] = float(arr[finite].mean())
            if row["finite_ratio"] < 0.999:
                blockers.append(f"{key} finite_ratio below strict threshold: {row['finite_ratio']}")
        stats["keys"][key] = row
    return not blockers, blockers, stats


def active_hash_gate() -> tuple[bool, list[str], dict[str, Any]]:
    manifest = read_json(ACTIVE / "hash_manifest.json", {})
    blockers: list[str] = []
    rows = []
    for item in manifest.get("files", []):
        path = ROOT / item["path"]
        exists = path.exists()
        actual = sha256_file(path) if exists and path.is_file() else None
        match = exists and actual == item.get("sha256")
        rows.append({"path": item["path"], "exists": exists, "expected_sha256": item.get("sha256"), "actual_sha256": actual, "match": match})
        if not match:
            blockers.append(f"hash mismatch or missing: {item['path']}")
    return not blockers and bool(rows), blockers, {"checked": rows, "file_count": len(rows)}


def process_gate() -> dict[str, Any]:
    def run(cmd: list[str]) -> dict[str, Any]:
        try:
            p = subprocess.run(cmd, text=True, capture_output=True, timeout=90)
            return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
        except Exception as exc:
            return {"cmd": cmd, "returncode": -1, "stdout": "", "stderr": repr(exc)}

    apps = run(["modal", "app", "list", "--json"])
    containers = run(["modal", "container", "list", "--json"])
    try:
        app_rows = json.loads(apps["stdout"] or "[]")
    except Exception:
        app_rows = None
    try:
        container_rows = json.loads(containers["stdout"] or "[]")
    except Exception:
        container_rows = None
    return {
        "modal_app_count": len(app_rows) if isinstance(app_rows, list) else None,
        "modal_container_count": len(container_rows) if isinstance(container_rows, list) else None,
        "modal_apps": app_rows,
        "modal_containers": container_rows,
        "pass": isinstance(app_rows, list) and not app_rows and isinstance(container_rows, list) and not container_rows,
    }


def main() -> int:
    PROMO.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    evidence: dict[str, Any] = {}

    manifest = read_json(ACTIVE / "manifest.json", {})
    evidence["manifest"] = {
        "exists": bool(manifest),
        "package_kind": manifest.get("package_kind"),
        "not_original_v50": manifest.get("not_original_v50"),
        "strict_candidate_passes_written": manifest.get("strict_candidate_passes_written"),
        "strict_teacher_passes_written": manifest.get("strict_teacher_passes_written"),
    }
    if not manifest:
        blockers.append("active V50R manifest missing")
    if manifest.get("not_original_v50") is True:
        blockers.append("active candidate is V50R rebuild, not original V50; requires new mentor/D-line acceptance")

    ok_hash, hash_blockers, hash_evidence = active_hash_gate()
    evidence["hash_gate"] = hash_evidence
    if not ok_hash:
        blockers.extend(hash_blockers)

    for filename, keys in [
        ("candidate_points_from_v42.npz", ["frame0000", "frame0001", "frame0002"]),
        ("candidate_normals_from_v42.npz", ["frame0000", "frame0001", "frame0002"]),
        ("candidate_depths_from_v42.npz", ["frame0000", "frame0001", "frame0002"]),
        ("candidate_confidence_from_v42.npz", ["frame0000_depth_conf", "frame0000_world_points_conf", "frame0000_normal_conf"]),
    ]:
        ok, b, stats = npz_basic_gate(ACTIVE_FILES / filename, keys)
        evidence[filename] = stats
        if not ok:
            blockers.extend(b)

    visual = read_json(REPORTS / "V225_A2_visual_truth_audit.json", {})
    evidence["visual_truth_audit"] = visual
    strict_visual_required = {
        "full_body": "PASS_VISUAL",
        "head_close": "PASS_VISUAL",
        "face_close": "PASS_VISUAL",
        "hairline_close": "PASS_VISUAL",
        "left_hand": "PASS_VISUAL",
        "right_hand": "PASS_VISUAL",
        "sixty_view_support": "PASS_VISUAL",
        "temporal_overlay": "PASS_VISUAL",
    }
    for key, required in strict_visual_required.items():
        got = visual.get(key)
        if got != required:
            blockers.append(f"{key} strict visual not pass: {got}")

    right = read_json(REPORTS / "V253_C4_right_hand_hard_merge_gate.json", {})
    evidence["right_hand_hard_merge_gate"] = right
    if right.get("result") != "MERGE_PASS_CREATE_V50_PLUS_HAND":
        blockers.append(f"right hand hard merge not pass: {right.get('result')}")

    teacher = read_json(REPORTS / "V310_I_teacher_resurrection.json", {})
    evidence["teacher_route"] = teacher
    if teacher.get("strict_teacher_passes", 0) != 0:
        blockers.append("unexpected teacher pass write detected")

    proc = process_gate()
    evidence["process_gate"] = proc
    if not proc["pass"]:
        blockers.append("residual Modal app/container detected")

    strict_pass = not blockers
    payload = {
        "task": "V400_v50r_strict_promotion_transaction",
        "created_utc": now(),
        "status": "DONE_PASS" if strict_pass else "DONE_HARD_IMPOSSIBLE_WITH_EVIDENCE",
        "active_candidate": "V50R_rebuilt_after_artifact_loss",
        "active_candidate_path": rel(ACTIVE),
        "strict_candidate_passes_written": 1 if strict_pass else 0,
        "strict_teacher_passes_written": 0,
        "formal_cloud_unblocked": strict_pass,
        "writes_registry": strict_pass,
        "writes_candidate_package": strict_pass,
        "writes_teacher_package": False,
        "blockers": blockers,
        "evidence": evidence,
        "decision": (
            "V50R strict candidate promotion passed and may write a new registry."
            if strict_pass
            else "V50R cannot be promoted under strict mentor gate. No registry/pass/package was written."
        ),
    }
    write_json(REPORTS / "V400_v50r_strict_promotion_transaction.json", payload)
    write_json(PROMO / "summary.json", payload)
    write_md(REPORTS / "V400_v50r_strict_promotion_transaction.md", "V400 V50R Strict Promotion Transaction", payload)
    print(json.dumps({"status": payload["status"], "strict_candidate_passes_written": payload["strict_candidate_passes_written"], "blocker_count": len(blockers)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
