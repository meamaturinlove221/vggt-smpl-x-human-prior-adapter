from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from PIL import Image, ImageDraw


REPO = Path(os.environ.get("VGGT_REPO_ROOT", r"D:\vggt\vggt-canonical-surfel-adapter"))
sys.path.insert(0, str(REPO))
OUTPUT = REPO / "output"
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
OUT_ROOT = OUTPUT / "V20500000000000000000_learned_part_proposals"
TARGET_ROOT = OUTPUT / "V20410000000000000000_part_local_targets"
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"

from tools.V17300_multishell_topology_decoder_training import as_rgb, compose, load_npz, read_manifest  # noqa: E402
from tools.V20420_part_local_target_student import nearest_distance, repo_path  # noqa: E402


PART_NAMES = {
    -1: "unknown",
    0: "torso/clothing",
    1: "shoulder/neck",
    2: "left_arm/hand",
    3: "right_arm/hand",
    4: "left_leg/foot",
    5: "right_leg/foot",
    6: "head/hair",
    7: "head/hair",
}


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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
        writer = csv.DictWriter(f, fieldnames=fields or ["case"])
        writer.writeheader()
        writer.writerows(rows)


def interp(values: np.ndarray, n: int) -> np.ndarray:
    if len(values) == n:
        return values
    return np.interp(np.linspace(0, len(values) - 1, n), np.arange(len(values)), values.astype(np.float32))


def normalized(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def render_proposal(points: np.ndarray, rgb: np.ndarray, score: np.ndarray, lock: np.ndarray, title: str) -> Image.Image:
    size = (460, 340)
    im = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(im)
    pts = points.astype(np.float32)
    centered = pts - np.median(pts, axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, len(pts) - 1)
    vals, vecs = np.linalg.eigh(cov)
    up = vecs[:, np.argsort(vals)[-1]]
    if up[1] < 0:
        up = -up
    right = np.array([1.0, 0.0, 0.0]) - up * float(np.dot(up, np.array([1.0, 0.0, 0.0])))
    right = right / max(np.linalg.norm(right), 1e-9)
    depth = np.cross(right, up)
    rot = np.stack([right, up, depth], axis=0)
    proj = centered @ rot.T
    lo = np.percentile(proj[:, :2], 1, axis=0)
    hi = np.percentile(proj[:, :2], 99, axis=0)
    pad = (hi - lo) * np.array([0.16, 0.12]) + 1e-6
    q = (proj[:, :2] - (lo - pad)[None]) / ((hi + pad - (lo - pad))[None] + 1e-9)
    q[:, 1] = 1.0 - q[:, 1]
    xy = np.clip(q * np.array([size[0] - 52, size[1] - 74]) + np.array([26, 44]), 0, [size[0] - 1, size[1] - 1]).astype(int)
    colors = as_rgb(rgb).copy()
    colors[lock] = np.clip(colors[lock].astype(np.float32) * 0.52, 0, 255).astype(np.uint8)
    hot = score >= np.quantile(score, 0.94)
    warm = (score >= np.quantile(score, 0.86)) & (~hot)
    colors[warm] = np.array([245, 160, 32], dtype=np.uint8)
    colors[hot] = np.array([220, 36, 24], dtype=np.uint8)
    order = np.argsort(proj[:, 2])
    step = max(1, len(order) // 52000)
    for i in order[::step]:
        x, y = xy[i]
        c = tuple(colors[i].tolist())
        im.putpixel((int(x), int(y)), c)
        if hot[i] and 1 <= x < size[0] - 1 and 1 <= y < size[1] - 1:
            im.putpixel((int(x + 1), int(y)), c)
            im.putpixel((int(x), int(y + 1)), c)
    draw.text((8, 8), title[:80], fill=(10, 10, 10))
    draw.text((8, size[1] - 24), "red=top proposal; orange=secondary; dim=locked visible surface", fill=(55, 55, 55))
    return im


def case_proposal(row: dict[str, str]) -> dict[str, Any]:
    case = row["case"]
    baseline = load_npz(repo_path(row["baseline_path"]))
    graph = load_npz(repo_path(row["graph_path"]))
    target = load_npz(TARGET_ROOT / case / "part_local_targets.npz")
    pts = np.asarray(baseline["human_points"], dtype=np.float32)
    rgb = as_rgb(baseline["human_rgb"])
    body = np.asarray(baseline["body_part_id"], dtype=np.int16)
    n = len(pts)
    conf = interp(np.asarray(graph["mentor_smpl_confidence"], dtype=np.float32), n)
    weak = interp(np.asarray(graph["mentor_weak_region_score"], dtype=np.float32), n)
    lock = interp(np.asarray(target["visible_lock_mask"], dtype=np.float32), n) >= 0.5
    seed = interp(np.asarray(target["part_local_target_mask"], dtype=np.float32), n) >= 0.5
    seed_dist = nearest_distance(pts, pts[seed] if bool(np.any(seed)) else pts[~lock])
    edit_band = np.exp(-seed_dist / max(float(np.quantile(seed_dist[seed | (~lock)], 0.18)), 1e-6))
    conf_drop = 1.0 - normalized(conf)
    weak_n = normalized(weak)
    # Expand beyond fixed V20410 masks, but keep it tied to nearby visible weak
    # regions. This is a proposal diagnostic, not a success claim.
    score = 0.42 * weak_n + 0.24 * conf_drop + 0.28 * edit_band + 0.06 * seed.astype(np.float32)
    score[lock] *= 0.06
    # Penalize fully unsupported head/arm/leg claims if the current target data
    # did not expose them. They need a future stronger source, not hallucination.
    for part in np.unique(body):
        part_ids = body == part
        if int(np.sum(seed & part_ids)) == 0 and int(part) != 0:
            score[part_ids] *= 0.45
    top = score >= np.quantile(score, 0.94)
    out_dir = ensure(OUT_ROOT / case)
    np.savez_compressed(
        out_dir / "proposal_scores.npz",
        human_points=pts,
        human_rgb=rgb,
        body_part_id=body,
        proposal_score=score.astype(np.float32),
        proposal_top_mask=top.astype(bool),
        visible_lock_mask=lock.astype(bool),
        seed_target_mask=seed.astype(bool),
        facial_detail_target_applicable=np.array(False),
        face_detail_claim_allowed=np.array(False),
        allowed_face_claim=np.array(ALLOWED_FACE_CLAIM),
    )
    part_rows = {}
    for part in sorted(set(int(x) for x in body.tolist())):
        ids = body == part
        part_rows[f"part_{part}_top_points"] = int(np.sum(top & ids))
        part_rows[f"part_{part}_mean_score"] = float(np.mean(score[ids])) if bool(np.any(ids)) else 0.0
    return {
        "case": case,
        "proposal_npz": str(out_dir / "proposal_scores.npz"),
        "top_proposal_points": int(np.sum(top)),
        "top_proposal_ratio": float(np.mean(top)),
        "top_locked_ratio": float(np.mean(lock[top])) if bool(np.any(top)) else 0.0,
        "seed_target_points": int(np.sum(seed)),
        "locked_ratio": float(np.mean(lock)),
        **part_rows,
    }


def main() -> int:
    created_at = now()
    rows = read_manifest()
    manifest = [case_proposal(row) for row in rows]
    write_csv(REPORTS / "V20500000000000000000_learned_part_proposal_manifest.csv", manifest)
    panels = []
    for row in rows:
        prop = load_npz(OUT_ROOT / row["case"] / "proposal_scores.npz")
        panels.append(
            render_proposal(
                np.asarray(prop["human_points"], dtype=np.float32),
                as_rgb(prop["human_rgb"]),
                np.asarray(prop["proposal_score"], dtype=np.float32),
                np.asarray(prop["visible_lock_mask"], dtype=bool),
                f"{row['case']} learned proposal diagnostic",
            )
        )
    if panels:
        compose(panels, 2, BOARDS / "V20500000000000000000_learned_part_proposal_preview.png")
    failures: list[dict[str, Any]] = []
    for row in manifest:
        if float(row["top_locked_ratio"]) > 0.08:
            failures.append({"case": row["case"], "reason": "proposal_enters_locked_visible_surface", "top_locked_ratio": row["top_locked_ratio"]})
        non_torso = sum(int(v) for k, v in row.items() if k.endswith("_top_points") and not k.startswith("part_0_"))
        if non_torso == 0:
            failures.append({"case": row["case"], "reason": "proposal_collapses_to_torso_only"})
    decision = {
        "created_at": created_at,
        "status": "V205_LEARNED_PART_PROPOSAL_DIAGNOSTIC_FAIL_CLOSED" if failures else "V205_LEARNED_PART_PROPOSAL_READY_FOR_STUDENT_TRAINING",
        "mentor_ready": False,
        "external_hard_block": False,
        "failures": failures,
        "manifest": str(REPORTS / "V20500000000000000000_learned_part_proposal_manifest.csv"),
        "preview": str(BOARDS / "V20500000000000000000_learned_part_proposal_preview.png"),
        "summary": "V205 builds proposal rankings around V20410 safety anchors. It is diagnostic-only until a student trained on these proposals beats baseline and controls in mentor visuals.",
    }
    write_json(REPORTS / "V20500000000000000000_learned_part_proposal_decision.json", decision)
    print(json.dumps({"created_at": created_at, "status": decision["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
