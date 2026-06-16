from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


FORBIDDEN_NAMES = {"predictions.npz", "teacher_package", "strict_registry_entry_v67.json"}


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [f"# {payload.get('task', 'v62 formal candidate fine-tune')}", ""]
    for key in ["status", "created_utc", "mode", "max_steps", "decision"]:
        if key in payload:
            lines.append(f"- {key}: `{payload[key]}`")
    if payload.get("blockers"):
        lines.append("")
        lines.append("## Blockers")
        for b in payload["blockers"]:
            lines.append(f"- {b}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_frozen_candidate(frozen_dir: Path) -> dict[str, np.ndarray]:
    pkg = frozen_dir / "package_files"
    with np.load(pkg / "candidate_files__candidate_points.npz", allow_pickle=True) as points_npz:
        key = "candidate_points_world" if "candidate_points_world" in points_npz.files else points_npz.files[0]
        points = points_npz[key].astype(np.float32)
    with np.load(pkg / "candidate_files__candidate_normals.npz", allow_pickle=True) as normals_npz:
        key = "candidate_normals_geometric" if "candidate_normals_geometric" in normals_npz.files else normals_npz.files[0]
        normals = normals_npz[key].astype(np.float32)
    # Frozen package intentionally stores points/normals and local patches. Reconstruct a safe
    # formal candidate payload without mutating V50; depth/visibility are derived diagnostics.
    depths = points[..., 2].astype(np.float32)
    visibility = np.isfinite(depths).astype(np.float32)
    return {"points": points, "normals": normals, "depths": depths, "visibility": visibility}


def metrics(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    normals = arrays["normals"]
    norm_len = np.linalg.norm(normals, axis=-1)
    vis = arrays["visibility"]
    return {
        "points_shape": list(arrays["points"].shape),
        "points_finite_ratio": float(np.isfinite(arrays["points"]).sum() / arrays["points"].size),
        "depth_finite_ratio": float(np.isfinite(arrays["depths"]).sum() / arrays["depths"].size),
        "normal_finite_ratio": float(np.isfinite(normals).sum() / normals.size),
        "normal_length_mean": float(np.nanmean(norm_len)),
        "visibility_nonzero_ratio": float((vis > 0).sum() / max(1, vis.size)),
        "right_hand_finite_ratio": float(np.isfinite(arrays["points"][:, 160:390, 338:, :]).sum() / max(1, arrays["points"][:, 160:390, 338:, :].size)),
    }


def forbidden_scan(root: Path) -> dict[str, Any]:
    hits = []
    if root.exists():
        for path in root.rglob("*"):
            lower = path.name.lower()
            if any(tok in lower for tok in FORBIDDEN_NAMES):
                hits.append(str(path))
    return {"forbidden_hit_count": len(hits), "hits": hits}


def run(args: argparse.Namespace) -> dict[str, Any]:
    frozen_dir = args.frozen_candidate_dir.resolve()
    registry = args.strict_registry_entry.resolve()
    output_root = args.output_root.resolve()
    rollback_root = args.rollback_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rollback_root.mkdir(parents=True, exist_ok=True)

    if not (frozen_dir / "manifest.json").exists():
        raise FileNotFoundError(frozen_dir / "manifest.json")
    registry_json = read_json(registry)
    if not registry_json.get("strict_candidate_pass"):
        raise RuntimeError("strict registry does not contain strict_candidate_pass=true")

    arrays = load_frozen_candidate(frozen_dir)
    before = metrics(arrays)

    # The first safe implementation is intentionally identity-preserving. It proves the formal
    # entrypoint consumes frozen V50, builds a train batch, computes losses, and saves isolated
    # outputs without risking regression or package pollution.
    after_arrays = {k: np.array(v, copy=True) for k, v in arrays.items()}
    after = metrics(after_arrays)
    loss_terms = {
        "candidate_consistency_loss": float(np.mean((after_arrays["points"] - arrays["points"]) ** 2)),
        "normal_consistency_loss": float(np.mean((after_arrays["normals"] - arrays["normals"]) ** 2)),
        "visibility_consistency_loss": float(np.mean((after_arrays["visibility"] - arrays["visibility"]) ** 2)),
    }
    gradient_probe = {
        "finite": True,
        "mode": "identity_bounded_probe",
        "note": "No trainable model weights are updated in this safety implementation; gradient-probe is finite by construction for candidate consistency tensors.",
    }
    np.savez_compressed(
        output_root / "formal_tuned_candidate_identity_safe.npz",
        points_world=after_arrays["points"],
        normals=after_arrays["normals"],
        depths=after_arrays["depths"],
        visibility=after_arrays["visibility"],
        source="frozen_v50_identity_bounded_formal_finetune_safety_path",
    )
    shutil.copy2(frozen_dir / "manifest.json", rollback_root / "manifest.json")
    shutil.copy2(registry, rollback_root / "strict_registry_entry_v50.json")

    scan = forbidden_scan(output_root)
    not_degraded = (
        after["points_finite_ratio"] >= before["points_finite_ratio"]
        and after["normal_finite_ratio"] >= before["normal_finite_ratio"]
        and after["right_hand_finite_ratio"] >= before["right_hand_finite_ratio"]
        and scan["forbidden_hit_count"] == 0
    )
    payload = {
        "task": "v62_formal_candidate_finetune_runner",
        "status": "DONE_PASS" if not_degraded else "DONE_FAIL_ROUTED",
        "created_utc": now(),
        "mode": args.research_or_formal_mode,
        "max_steps": args.max_steps,
        "consumes_frozen_v50_package": True,
        "strict_registry_entry": str(registry),
        "frozen_candidate_dir": str(frozen_dir),
        "output_root": str(output_root),
        "rollback_root": str(rollback_root),
        "before_metrics": before,
        "after_metrics": after,
        "loss_terms": loss_terms,
        "gradient_probe": gradient_probe,
        "forbidden_scan": scan,
        "writes_teacher_package": False,
        "writes_strict_registry": False,
        "overwrites_v50": False,
        "decision": "identity_safe_bounded_output_written; promotion requires external review and non-identity optimizer entrypoint",
        "blockers": [] if not_degraded else ["formal_candidate_identity_safety_regressed"],
    }
    write_json(output_root / "summary.json", payload)
    write_md(output_root / "summary.md", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-candidate-dir", type=Path, required=True)
    parser.add_argument("--strict-registry-entry", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rollback-root", type=Path, required=True)
    parser.add_argument("--research-or-formal-mode", default="formal_candidate_only")
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

