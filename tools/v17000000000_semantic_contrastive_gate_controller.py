from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(r"D:\vggt\vggt-feature-adapter")
AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
OUTPUT = AUX / "output"
BOARDS = AUX / "boards"
V151_PAYLOAD = OUTPUT / "V15100000000_point_transformer_payload" / "surface_samples.npz"
OUT = OUTPUT / "V17000000000_semantic_contrastive_gate"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run_cmd(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(args, cwd=str(REPO), text=True, capture_output=True, encoding="utf-8", errors="replace")
    return {"args": args, "returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}


def route_doc() -> Path:
    path = REPO / "docs" / "goals" / "V17000000000_semantic_contrastive_gate_route.md"
    text = """# V17000000000 Semantic Contrastive Gate Route

## Failure Attribution

V151/V160 point-transformer reduced routes still let random_semantic beat true. The likely cause is that the training target is still too synthetic and can be satisfied by random feature energy instead of verified SMPL-X semantic coherence.

## New Hypothesis

Before further long Modal training, add a semantic coherence gate that explicitly scores true SMPL-X semantic consistency: canonical/posed correspondence, barycentric validity, skinning/joint consistency, part/local-frame agreement, curvature/surface-distance consistency. Random/shuffled semantic should be rejected by this gate before geometry transport.

## Hard Gate

true semantic gate score must exceed random and shuffled by a positive margin across 5 seeds. If this cannot be achieved from current fields, the route requires new supervision or a differentiable SMPL surface renderer.
"""
    path.write_text(text, encoding="utf-8")
    write_json(REPORTS / "V17000000000_route_generation.json", {"created_utc": now(), "route_file": str(path), "execute_immediately": True})
    return path


def run_gate(seeds: list[int]) -> dict[str, Any]:
    route_doc()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(V151_PAYLOAD, allow_pickle=False) as z:
        canonical = z["canonical"].astype(np.float32)
        posed = z["posed"].astype(np.float32)
        bary = z["barycentric"].astype(np.float32)
        skinning = z["skinning"].astype(np.float32)
        joint = z["joint_distance"].astype(np.float32)
        bone = z["bone_relative"].astype(np.float32)
        local = z["local_frame"].astype(np.float32)
        curv = z["curvature"].astype(np.float32)
        surf = z["surface_distance"].astype(np.float32)
        part = z["part_id"].astype(np.float32)
        wp = z["world_points"].astype(np.float32)
        normal = z["normal"].astype(np.float32)
    base = np.concatenate([canonical, posed, bary, skinning, joint, bone, local, curv, surf, part[:, None]], axis=1)
    # Coherence features: lower residuals and valid bary/skinning sums indicate stronger true semantic consistency.
    bary_err = np.abs(bary.sum(axis=1) - 1.0)
    skin_err = np.abs(skinning.sum(axis=1) - skinning.sum(axis=1).mean())
    canon_pose = np.linalg.norm(posed - canonical, axis=1)
    local_norm = np.linalg.norm(local.reshape(-1, 3, 3), axis=2).mean(axis=1)
    normal_norm = np.linalg.norm(normal, axis=1)
    true_signal = np.exp(-bary_err) + 0.15 * np.tanh(canon_pose) + 0.1 * np.tanh(joint[:, :3].mean(axis=1)) + 0.1 * np.tanh(curv[:, 0]) - 0.05 * np.abs(local_norm - 1.0) - 0.05 * np.abs(normal_norm - 1.0)
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(17000000000 + seed)
        shuffled = true_signal[rng.permutation(true_signal.shape[0])]
        random_sem = rng.normal(float(true_signal.mean()), float(true_signal.std()), size=true_signal.shape[0])
        # The current fields should allow true to win. If random still wins, semantic fields lack discriminative coherence.
        for group, vals in {
            "true_semantic_gate": true_signal,
            "random_semantic_gate": random_sem,
            "shuffled_semantic_gate": shuffled,
        }.items():
            rows.append({
                "group": group,
                "seed": seed,
                "mean_score": float(np.mean(vals)),
                "std_score": float(np.std(vals)),
                "n": int(vals.size),
            })
    write_csv(REPORTS / "V17000000000_semantic_gate_metrics.csv", rows)
    grouped: dict[str, list[float]] = {}
    for r in rows:
        grouped.setdefault(r["group"], []).append(r["mean_score"])
    ranking = [{"group": g, "mean_score": float(np.mean(v)), "std_score": float(np.std(v)), "n": len(v)} for g, v in grouped.items()]
    ranking.sort(key=lambda r: r["mean_score"], reverse=True)
    passed = ranking[0]["group"] == "true_semantic_gate" and ranking[0]["mean_score"] > ranking[1]["mean_score"]
    payload = {
        "created_utc": now(),
        "semantic_gate_passed": passed,
        "ranking": ranking,
        "interpretation": "This gate tests field-level semantic coherence only; it is not geometry transport success.",
    }
    write_json(REPORTS / "V17000000000_semantic_gate_decision.json", payload)
    np.savez_compressed(OUT / "semantic_gate_scores.npz", true_signal=true_signal.astype(np.float32))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x != ""]
    print(json.dumps(run_gate(seeds), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
