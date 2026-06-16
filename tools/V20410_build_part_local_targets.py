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
OUT_ROOT = OUTPUT / "V20410000000000000000_part_local_targets"

from tools.V17300_multishell_topology_decoder_training import as_rgb, compose, load_npz, read_manifest  # noqa: E402


PART_MASKS = {
    "head_hair": ("head_hair_contour_mask", "mask_head_hair"),
    "shoulder_neck": ("shoulder_neck_mask", "mask_shoulder_neck"),
    "clothing_torso": ("clothing_torso_boundary_mask", "mask_torso_clothing_boundary"),
    "arm_hand": ("hand_arm_endpoint_mask", "mask_arms_hands"),
    "leg_foot": ("leg_foot_morphology_mask", "mask_feet_leg_boundary"),
}
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"


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


def repo_path(value: str | Path) -> Path:
    p = Path(value)
    if p.exists():
        return p
    text = str(value).replace("\\", "/")
    marker = "vggt-canonical-surfel-adapter/"
    if marker in text:
        mapped = REPO / text.split(marker, 1)[1]
        if mapped.exists():
            return mapped
    return p


def interpolate_bool(mask: np.ndarray, n: int) -> np.ndarray:
    if len(mask) == n:
        return mask.astype(bool)
    values = np.interp(np.linspace(0, len(mask) - 1, n), np.arange(len(mask)), mask.astype(np.float32))
    return values >= 0.5


def render_target(points: np.ndarray, rgb: np.ndarray, target: np.ndarray, lock: np.ndarray, title: str) -> Image.Image:
    size = (430, 320)
    im = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(im)
    pts = points.astype(np.float32)
    centered = pts - np.median(pts, axis=0, keepdims=True)
    # Make body long axis vertical for readable source-view diagnostics.
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
    pad = (hi - lo) * np.array([0.18, 0.14]) + 1e-6
    lo -= pad
    hi += pad
    q = (proj[:, :2] - lo[None]) / (hi[None] - lo[None] + 1e-9)
    q[:, 1] = 1.0 - q[:, 1]
    xy = np.clip(q * np.array([size[0] - 48, size[1] - 70]) + np.array([24, 44]), 0, [size[0] - 1, size[1] - 1]).astype(int)
    colors = as_rgb(rgb).copy()
    colors[lock] = np.clip(colors[lock].astype(np.float32) * 0.55, 0, 255).astype(np.uint8)
    colors[target] = np.array([220, 60, 28], dtype=np.uint8)
    order = np.argsort(proj[:, 2])
    step = max(1, len(order) // 52000)
    for i in order[::step]:
        x, y = xy[i]
        c = tuple(colors[i].tolist())
        im.putpixel((int(x), int(y)), c)
        if target[i] and 1 <= x < size[0] - 1 and 1 <= y < size[1] - 1:
            im.putpixel((int(x + 1), int(y)), c)
            im.putpixel((int(x), int(y + 1)), c)
    draw.text((8, 8), title[:76], fill=(10, 10, 10))
    draw.text((8, size[1] - 24), "red=allowed part-local target; dim=locked visible baseline", fill=(55, 55, 55))
    return im


def build_case(row: dict[str, str]) -> dict[str, Any]:
    case = row["case"]
    baseline = load_npz(repo_path(row["baseline_path"]))
    graph = load_npz(repo_path(row["graph_path"]))
    visible = load_npz(repo_path(row["visible_target_path"]))
    pts = np.asarray(baseline["human_points"], dtype=np.float32)
    rgb = as_rgb(baseline["human_rgb"])
    body = np.asarray(baseline["body_part_id"], dtype=np.int16)
    n = len(pts)
    conf = np.asarray(graph["mentor_smpl_confidence"], dtype=np.float32)
    weak = np.asarray(graph["mentor_weak_region_score"], dtype=np.float32)
    no_change = np.asarray(graph["no_change_mask"], dtype=bool)
    if len(conf) != n:
        conf = np.interp(np.linspace(0, len(conf) - 1, n), np.arange(len(conf)), conf).astype(np.float32)
        weak = np.interp(np.linspace(0, len(weak) - 1, n), np.arange(len(weak)), weak).astype(np.float32)
        no_change = np.zeros(n, dtype=bool)
    lock = ((conf >= np.quantile(conf, 0.68)) & (weak <= np.quantile(weak, 0.52))) | no_change
    target_any = np.zeros(n, dtype=bool)
    part_counts: dict[str, int] = {}
    visible_body = np.asarray(visible["body_part_id"], dtype=np.int16)
    for name, (graph_key, visible_key) in PART_MASKS.items():
        gmask = interpolate_bool(np.asarray(graph[graph_key], dtype=bool), n)
        # Visible masks are in V161 feature-bank order. Use body-part support as
        # a semantic prior, then intersect graph region and weak non-locked area.
        if visible_key in visible:
            v_part_ids = visible_body[np.asarray(visible[visible_key], dtype=bool)]
            allowed_parts = set(int(x) for x in np.unique(v_part_ids).tolist())
            if allowed_parts:
                vmask = np.isin(body, list(allowed_parts))
            else:
                vmask = np.ones(n, dtype=bool)
        else:
            vmask = np.ones(n, dtype=bool)
        local = gmask & vmask & (~lock) & (weak >= np.quantile(weak, 0.50))
        # Keep targets small and explicit. If a mask is huge, retain the weakest
        # part only so V203's global contamination does not repeat.
        cap = max(256, int(0.055 * n))
        ids = np.flatnonzero(local)
        if len(ids) > cap:
            ids = ids[np.argsort(-weak[ids])[:cap]]
            local = np.zeros(n, dtype=bool)
            local[ids] = True
        target_any |= local
        part_counts[f"{name}_target_points"] = int(local.sum())
    out_dir = ensure(OUT_ROOT / case)
    np.savez_compressed(
        out_dir / "part_local_targets.npz",
        human_points=pts,
        human_rgb=rgb,
        body_part_id=body,
        visible_lock_mask=lock,
        part_local_target_mask=target_any,
        weak_score=weak,
        confidence=conf,
        facial_detail_target_applicable=np.array(False),
        face_detail_claim_allowed=np.array(False),
        allowed_face_claim=np.array(ALLOWED_FACE_CLAIM),
    )
    return {
        "case": case,
        "target_npz": str(out_dir / "part_local_targets.npz"),
        "visible_lock_ratio": float(lock.mean()),
        "part_local_target_ratio": float(target_any.mean()),
        "part_local_target_points": int(target_any.sum()),
        **part_counts,
    }


def main() -> int:
    created_at = now()
    rows = read_manifest()
    manifest = [build_case(row) for row in rows]
    write_csv(REPORTS / "V20410000000000000000_part_local_target_manifest.csv", manifest)
    panels = []
    for row in rows:
        target = load_npz(OUT_ROOT / row["case"] / "part_local_targets.npz")
        panels.append(
            render_target(
                np.asarray(target["human_points"], dtype=np.float32),
                as_rgb(target["human_rgb"]),
                np.asarray(target["part_local_target_mask"], dtype=bool),
                np.asarray(target["visible_lock_mask"], dtype=bool),
                f"{row['case']} part-local target",
            )
        )
    if panels:
        compose(panels, 2, BOARDS / "V20410000000000000000_part_local_target_preview.png")
    failures = [row for row in manifest if int(row["part_local_target_points"]) <= 0]
    decision = {
        "created_at": created_at,
        "status": "V20410_PART_LOCAL_TARGETS_READY_FOR_TRAINING" if not failures else "V20410_PART_LOCAL_TARGETS_FAIL_CLOSED",
        "mentor_ready": False,
        "external_hard_block": False,
        "failures": failures,
        "manifest": str(REPORTS / "V20410000000000000000_part_local_target_manifest.csv"),
        "preview": str(BOARDS / "V20410000000000000000_part_local_target_preview.png"),
        "summary": "V20410 constructs explicit per-part weak target masks so the next student can improve only allowed local regions while locking visible baseline surface.",
    }
    write_json(REPORTS / "V20410000000000000000_part_local_target_decision.json", decision)
    print(json.dumps({"created_at": created_at, "status": decision["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
