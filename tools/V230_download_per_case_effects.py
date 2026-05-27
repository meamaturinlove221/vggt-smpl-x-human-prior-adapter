from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
REPORTS = REPO / "reports"
OUTPUT = REPO / "output"
VOLUME = "vggt-v230-per-case-full-forward-effect-output"
ROOT = "out/V23000000000000000_per_case_full_forward_effect"
LOCAL_ROOT = OUTPUT / "V23000000000000000_per_case_full_forward_effect"
CASES = ["current_v895_0021_03", "0021_03_frame001", "0012_11_frame001", "0013_01_frame001"]


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run(args: list[str]) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(args, cwd=REPO, text=True, capture_output=True, encoding="utf-8", errors="replace", env=env, timeout=600)
    return {"cmd": args, "returncode": proc.returncode, "stdout": proc.stdout[-1000:], "stderr": proc.stderr[-1000:]}


def modal_get(remote: str, local: Path) -> dict[str, Any]:
    local.parent.mkdir(parents=True, exist_ok=True)
    return run(["modal", "volume", "get", VOLUME, remote, str(local), "--force"])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    rows = []
    downloads = []
    failures = []
    for case in CASES:
        trace_path = LOCAL_ROOT / case / "trace.json"
        npz_path = LOCAL_ROOT / case / "full_forward_outputs.npz"
        for filename, local in [("trace.json", trace_path), ("full_forward_outputs.npz", npz_path)]:
            result = modal_get(f"{ROOT}/{case}/{filename}", local)
            downloads.append(result)
            if result["returncode"] != 0:
                failures.append({"case_id": case, "file": filename, "result": result})
        if trace_path.exists():
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            npz_ok = False
            if npz_path.exists():
                try:
                    with np.load(npz_path, allow_pickle=False) as z:
                        npz_ok = bool(z.files)
                except Exception:
                    npz_ok = False
            rows.append(
                {
                    "case_id": case,
                    "trace": str(trace_path),
                    "npz": str(npz_path),
                    "full_vggt_forward_executed": trace.get("full_vggt_forward_executed"),
                    "camera_output_present": trace.get("camera_output_present"),
                    "depth_output_present": trace.get("depth_output_present"),
                    "point_output_present": trace.get("point_output_present"),
                    "smpl_prior_token_injection_attempted": trace.get("smpl_prior_token_injection_attempted"),
                    "sparse_prior_grad_mean": trace.get("sparse_prior_grad_mean"),
                    "point_effect_l1": trace.get("point_effect_l1"),
                    "depth_effect_l1": trace.get("depth_effect_l1"),
                    "confidence_effect_l1": trace.get("confidence_effect_l1"),
                    "output_effect_l1": trace.get("output_effect_l1"),
                    "projection_seed": trace.get("projection_seed"),
                    "teacher_points_used_at_inference": trace.get("teacher_points_used_at_inference"),
                    "raw_kinect_depth_used_at_inference": trace.get("raw_kinect_depth_used_at_inference"),
                    "npz_readable": npz_ok,
                }
            )
    effects = [float(r["output_effect_l1"]) for r in rows if r.get("output_effect_l1") is not None]
    grads = [float(r["sparse_prior_grad_mean"]) for r in rows if r.get("sparse_prior_grad_mean") is not None]
    pass_gate = (
        len(rows) == len(CASES)
        and not failures
        and all(r.get("full_vggt_forward_executed") and r.get("npz_readable") for r in rows)
        and all(float(r.get("sparse_prior_grad_mean", 0.0)) > 0 for r in rows)
        and all(float(r.get("output_effect_l1", 0.0)) > 0 for r in rows)
        and len(set(round(v, 6) for v in effects)) > 1
    )
    write_csv(REPORTS / "V22000000000000000_full_forward_effect_per_case.csv", rows)
    write_json(
        REPORTS / "V22000000000000000_full_forward_effect_decision.json",
        {
            "created_at": now(),
            "per_case_full_forward_effect_pass": pass_gate,
            "case_count": len(rows),
            "effect_values": effects,
            "grad_values": grads,
            "effects_case_specific": len(set(round(v, 6) for v in effects)) > 1,
            "v930_single_smoke_reuse_rejected": True,
            "failures": failures,
        },
    )
    write_json(
        REPORTS / "V23000000000000000_per_case_full_forward_manifest.json",
        {
            "created_at": now(),
            "downloads": len(downloads),
            "failures": failures,
            "rows": rows,
        },
    )
    write_json(
        REPORTS / "V23000000000000000_per_case_full_forward_decision.json",
        {
            "created_at": now(),
            "per_case_full_forward_effect_pass": pass_gate,
            "no_teacher_points_at_inference": all(not r.get("teacher_points_used_at_inference") for r in rows),
            "no_raw_kinect_depth_at_inference": all(not r.get("raw_kinect_depth_used_at_inference") for r in rows),
        },
    )
    print(json.dumps({"per_case_full_forward_effect_pass": pass_gate, "case_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
