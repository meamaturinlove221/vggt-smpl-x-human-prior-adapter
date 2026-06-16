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
OUT_ROOT = OUTPUT / "V21000000000000000000_patch_geometry_sources"
PROPOSAL_ROOT = OUTPUT / "V20500000000000000000_learned_part_proposals"

from tools.V17300_multishell_topology_decoder_training import (  # noqa: E402
    as_rgb,
    compose,
    load_npz,
    read_manifest,
)
from tools.V20420_part_local_target_student import nearest_distance, repo_path  # noqa: E402


ALLOWED_FACE_CLAIM = "head/face contour and hair region only"
PART_NAMES = {
    0: "torso_clothing",
    1: "shoulder_neck",
    2: "left_arm_endpoint",
    3: "right_arm_endpoint",
    4: "left_leg_foot",
    5: "right_leg_foot",
    6: "head_hair_contour",
    7: "head_hair_contour",
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


def pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    center = np.median(pts, axis=0).astype(np.float32)
    centered = pts - center[None]
    if len(pts) < 4:
        return center, np.eye(3, dtype=np.float32)[0], np.eye(3, dtype=np.float32)[1], np.eye(3, dtype=np.float32)[2]
    cov = centered.T @ centered / max(1, len(pts) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)
    normal = vecs[:, order[0]].astype(np.float32)
    tangent = vecs[:, order[-1]].astype(np.float32)
    binormal = np.cross(normal, tangent).astype(np.float32)
    binormal /= max(float(np.linalg.norm(binormal)), 1e-8)
    tangent = np.cross(binormal, normal).astype(np.float32)
    tangent /= max(float(np.linalg.norm(tangent)), 1e-8)
    normal /= max(float(np.linalg.norm(normal)), 1e-8)
    return center, tangent, binormal, normal


def sample_boundary_anchors(points: np.ndarray, center: np.ndarray, tangent: np.ndarray, binormal: np.ndarray, count: int = 16) -> np.ndarray:
    local = np.stack([(points - center) @ tangent, (points - center) @ binormal], axis=1)
    if len(local) == 0:
        return np.zeros((count, 3), dtype=np.float32)
    angles = np.linspace(-np.pi, np.pi, count, endpoint=False)
    anchors = []
    theta = np.arctan2(local[:, 1], local[:, 0])
    radius = np.linalg.norm(local, axis=1)
    for a in angles:
        delta = np.abs(np.angle(np.exp(1j * (theta - a))))
        ids = np.argsort(delta + 0.12 / np.maximum(radius, 1e-5))[: max(4, len(points) // 80)]
        anchors.append(np.mean(points[ids], axis=0))
    return np.asarray(anchors, dtype=np.float32)


def build_patch_samples(points: np.ndarray, rgb: np.ndarray, body: np.ndarray, proposal_score: np.ndarray, proposal_mask: np.ndarray, lock_mask: np.ndarray) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    patch_rows: list[dict[str, Any]] = []
    patch_centers: list[np.ndarray] = []
    patch_frames: list[np.ndarray] = []
    patch_radii: list[float] = []
    patch_thickness: list[float] = []
    patch_parts: list[int] = []
    patch_scores: list[float] = []
    boundary_anchors: list[np.ndarray] = []
    for part in sorted(set(int(x) for x in body.tolist())):
        if part < 0:
            continue
        ids = np.flatnonzero((body == part) & proposal_mask & (~lock_mask))
        if len(ids) < 24:
            continue
        pts = points[ids]
        scores = proposal_score[ids]
        # Split each body part into 1-3 score-weighted local patches. This is
        # a source/target builder only; it does not claim mentor success.
        patch_count = int(np.clip(np.ceil(len(ids) / 1400), 1, 3))
        order = ids[np.argsort(-scores)]
        seeds = order[:patch_count]
        assigned = np.argmin(((points[ids, None, :] - points[seeds][None, :, :]) ** 2).sum(axis=2), axis=1)
        for patch_id in range(patch_count):
            local_ids = ids[assigned == patch_id]
            if len(local_ids) < 16:
                continue
            local_pts = points[local_ids]
            local_rgb = rgb[local_ids]
            center, tangent, binormal, normal = pca_frame(local_pts)
            proj_t = (local_pts - center) @ tangent
            proj_b = (local_pts - center) @ binormal
            proj_n = (local_pts - center) @ normal
            radius = float(np.percentile(np.sqrt(proj_t**2 + proj_b**2), 92))
            thickness = float(max(np.percentile(proj_n, 92) - np.percentile(proj_n, 8), 0.012))
            anchors = sample_boundary_anchors(local_pts, center, tangent, binormal)
            patch_centers.append(center)
            patch_frames.append(np.stack([tangent, binormal, normal], axis=0))
            patch_radii.append(radius)
            patch_thickness.append(thickness)
            patch_parts.append(part)
            patch_scores.append(float(np.mean(proposal_score[local_ids])))
            boundary_anchors.append(anchors)
            patch_rows.append(
                {
                    "body_part": int(part),
                    "part_name": PART_NAMES.get(int(part), "unknown"),
                    "patch_index": int(len(patch_rows)),
                    "source_points": int(len(local_ids)),
                    "proposal_score_mean": float(np.mean(proposal_score[local_ids])),
                    "radius": radius,
                    "thickness": thickness,
                    "rgb_mean": json.dumps(np.mean(as_rgb(local_rgb), axis=0).round(2).tolist()),
                }
            )
    if patch_centers:
        arrays = {
            "patch_centers": np.asarray(patch_centers, dtype=np.float32),
            "patch_frames": np.asarray(patch_frames, dtype=np.float32),
            "patch_radius": np.asarray(patch_radii, dtype=np.float32),
            "patch_thickness": np.asarray(patch_thickness, dtype=np.float32),
            "patch_body_part_id": np.asarray(patch_parts, dtype=np.int16),
            "patch_proposal_score": np.asarray(patch_scores, dtype=np.float32),
            "patch_boundary_anchors": np.asarray(boundary_anchors, dtype=np.float32),
        }
    else:
        arrays = {
            "patch_centers": np.zeros((0, 3), dtype=np.float32),
            "patch_frames": np.zeros((0, 3, 3), dtype=np.float32),
            "patch_radius": np.zeros((0,), dtype=np.float32),
            "patch_thickness": np.zeros((0,), dtype=np.float32),
            "patch_body_part_id": np.zeros((0,), dtype=np.int16),
            "patch_proposal_score": np.zeros((0,), dtype=np.float32),
            "patch_boundary_anchors": np.zeros((0, 16, 3), dtype=np.float32),
        }
    return arrays, patch_rows


def render_patch_preview(points: np.ndarray, rgb: np.ndarray, proposal_mask: np.ndarray, lock_mask: np.ndarray, patches: dict[str, np.ndarray], title: str) -> Image.Image:
    size = (480, 360)
    im = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(im)
    pts = points.astype(np.float32)
    center = np.median(pts, axis=0, keepdims=True)
    cov = (pts - center).T @ (pts - center) / max(1, len(pts) - 1)
    vals, vecs = np.linalg.eigh(cov)
    up = vecs[:, np.argsort(vals)[-1]]
    right = np.array([1.0, 0.0, 0.0]) - up * float(np.dot(up, np.array([1.0, 0.0, 0.0])))
    right /= max(float(np.linalg.norm(right)), 1e-8)
    depth = np.cross(right, up)
    rot = np.stack([right, up, depth], axis=0)
    proj = (pts - center) @ rot.T
    lo = np.percentile(proj[:, :2], 1, axis=0)
    hi = np.percentile(proj[:, :2], 99, axis=0)
    pad = (hi - lo) * np.array([0.16, 0.12]) + 1e-6
    q = (proj[:, :2] - (lo - pad)[None]) / ((hi + pad - (lo - pad))[None] + 1e-9)
    q[:, 1] = 1.0 - q[:, 1]
    xy = np.clip(q * np.array([size[0] - 54, size[1] - 76]) + np.array([27, 44]), 0, [size[0] - 1, size[1] - 1]).astype(int)
    colors = as_rgb(rgb).copy()
    colors[lock_mask] = np.clip(colors[lock_mask].astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
    colors[proposal_mask] = np.array([220, 48, 32], dtype=np.uint8)
    order = np.argsort(proj[:, 2])
    step = max(1, len(order) // 52000)
    for i in order[::step]:
        x, y = xy[i]
        c = tuple(colors[i].tolist())
        im.putpixel((int(x), int(y)), c)
        if proposal_mask[i] and 1 <= x < size[0] - 1 and 1 <= y < size[1] - 1:
            im.putpixel((int(x + 1), int(y)), c)
    # Project patch centers and anchors.
    centers = patches["patch_centers"]
    if len(centers):
        cproj = (centers - center.reshape(3)) @ rot.T
        cq = (cproj[:, :2] - (lo - pad)[None]) / ((hi + pad - (lo - pad))[None] + 1e-9)
        cq[:, 1] = 1.0 - cq[:, 1]
        cxy = np.clip(cq * np.array([size[0] - 54, size[1] - 76]) + np.array([27, 44]), 0, [size[0] - 1, size[1] - 1]).astype(int)
        for j, (x, y) in enumerate(cxy):
            r = 4
            draw.ellipse((int(x - r), int(y - r), int(x + r), int(y + r)), outline=(20, 90, 230), width=2)
            draw.text((int(x + 5), int(y - 5)), str(j), fill=(20, 90, 230))
    draw.text((8, 8), title[:82], fill=(10, 10, 10))
    draw.text((8, size[1] - 24), "red=proposal weak region; blue=candidate patch source center; dim=locked visible", fill=(55, 55, 55))
    return im


def case_patch_source(row: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]], Image.Image]:
    case = row["case"]
    baseline = load_npz(repo_path(row["baseline_path"]))
    prop = load_npz(PROPOSAL_ROOT / case / "proposal_scores.npz")
    pts = np.asarray(baseline["human_points"], dtype=np.float32)
    rgb = as_rgb(baseline["human_rgb"])
    body = np.asarray(baseline["body_part_id"], dtype=np.int16)
    proposal_score = np.asarray(prop["proposal_score"], dtype=np.float32)
    proposal_mask = np.asarray(prop["proposal_top_mask"], dtype=bool)
    lock_mask = np.asarray(prop["visible_lock_mask"], dtype=bool)
    if len(proposal_score) != len(pts):
        proposal_score = np.interp(np.linspace(0, len(proposal_score) - 1, len(pts)), np.arange(len(proposal_score)), proposal_score).astype(np.float32)
        proposal_mask = proposal_score >= np.quantile(proposal_score, 0.94)
        lock_mask = np.interp(np.linspace(0, len(lock_mask) - 1, len(pts)), np.arange(len(lock_mask)), lock_mask.astype(np.float32)) >= 0.5
    patches, patch_rows = build_patch_samples(pts, rgb, body, proposal_score, proposal_mask, lock_mask)
    out_dir = ensure(OUT_ROOT / case)
    np.savez_compressed(
        out_dir / "patch_geometry_sources.npz",
        human_points=pts,
        human_rgb=rgb,
        body_part_id=body,
        proposal_score=proposal_score,
        proposal_mask=proposal_mask,
        visible_lock_mask=lock_mask,
        **patches,
        facial_detail_target_applicable=np.array(False),
        face_detail_claim_allowed=np.array(False),
        allowed_face_claim=np.array(ALLOWED_FACE_CLAIM),
    )
    preview = render_patch_preview(pts, rgb, proposal_mask, lock_mask, patches, f"{case} patch geometry source")
    manifest = {
        "case": case,
        "patch_npz": str(out_dir / "patch_geometry_sources.npz"),
        "patch_count": int(len(patches["patch_centers"])),
        "proposal_points": int(np.sum(proposal_mask)),
        "visible_lock_ratio": float(np.mean(lock_mask)),
        "mean_patch_radius": float(np.mean(patches["patch_radius"])) if len(patches["patch_radius"]) else 0.0,
        "mean_patch_thickness": float(np.mean(patches["patch_thickness"])) if len(patches["patch_thickness"]) else 0.0,
        "face_detail_claim_allowed": False,
        "allowed_face_claim": ALLOWED_FACE_CLAIM,
    }
    for patch in patch_rows:
        patch["case"] = case
    return manifest, patch_rows, preview


def main() -> int:
    created_at = now()
    rows = read_manifest()
    manifests: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    previews: list[Image.Image] = []
    failures: list[dict[str, Any]] = []
    for row in rows:
        manifest, patches, preview = case_patch_source(row)
        manifests.append(manifest)
        patch_rows.extend(patches)
        previews.append(preview)
        if int(manifest["patch_count"]) < 2:
            failures.append({"case": manifest["case"], "reason": "too_few_patch_sources", "patch_count": manifest["patch_count"]})
        if float(manifest["mean_patch_thickness"]) <= 0.012:
            failures.append({"case": manifest["case"], "reason": "patch_sources_too_thin", "mean_patch_thickness": manifest["mean_patch_thickness"]})
    write_csv(REPORTS / "V21000000000000000000_patch_geometry_source_manifest.csv", manifests)
    write_csv(REPORTS / "V21000000000000000000_patch_geometry_source_patches.csv", patch_rows)
    if previews:
        compose(previews, 2, BOARDS / "V21000000000000000000_patch_geometry_source_preview.png")
    decision = {
        "created_at": created_at,
        "status": "V210_PATCH_GEOMETRY_SOURCE_READY_FOR_PATCH_DECODER" if not failures else "V210_PATCH_GEOMETRY_SOURCE_DIAGNOSTIC_FAIL_CLOSED",
        "mentor_ready": False,
        "external_hard_block": False,
        "failures": failures,
        "manifest": str(REPORTS / "V21000000000000000000_patch_geometry_source_manifest.csv"),
        "patches": str(REPORTS / "V21000000000000000000_patch_geometry_source_patches.csv"),
        "preview": str(BOARDS / "V21000000000000000000_patch_geometry_source_preview.png"),
        "summary": "V210 builds explicit local patch geometry sources from V205 proposals and baseline/SMPL body parts. It is a route input, not mentor-ready evidence.",
    }
    write_json(REPORTS / "V21000000000000000000_patch_geometry_source_decision.json", decision)
    print(json.dumps({"created_at": created_at, "status": decision["status"], "patch_cases": len(manifests)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
