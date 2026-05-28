from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from PIL import Image, ImageDraw


REPO = Path(os.environ.get("VGGT_REPO_ROOT", r"D:\vggt\vggt-canonical-surfel-adapter"))
OUTPUT = REPO / "output"
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
MANIFEST = REPORTS / "V10210000000000000000_training_asset_manifest.csv"
WEAK_ROOT = OUTPUT / "V13400000000000000000_billboard_weak_regions"
OUT_ROOT = OUTPUT / "V17100000000000000000_part_graph_cross_section_targets"
BASE_MATRIX = OUTPUT / "V10700000000000000000_volume_aware_training_matrix"
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"


PART_NAMES = {
    "head_hair": "head_hair_mask",
    "shoulder_neck": "shoulder_neck_mask",
    "hand_arm": "hand_arm_mask",
    "clothing": "clothing_mask",
    "leg_foot": "leg_foot_mask",
}

CONTROL_CONFIGS = [
    "same_topology_no_semantic",
    "shuffled_smpl_feature",
    "thickness_only_control",
]


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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("eligible_for_training_payload") == "True"]


def pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    x = points - center[None]
    _u, _s, vh = np.linalg.svd(x, full_matrices=False)
    return center.astype(np.float32), vh.astype(np.float32), (x @ vh.T).astype(np.float32)


def control_score(case: str, config: str) -> float:
    path = BASE_MATRIX / case / config / "predictions.npz"
    if not path.exists():
        return 0.0
    pred = load_npz(path)
    pts = np.asarray(pred["human_points"], dtype=np.float32)
    center, axes, proj = pca_frame(pts)
    ranges = np.ptp(proj, axis=0)
    return float(max(ranges[2] / max(ranges[0], 1e-9), 0.0))


def build_case(row: dict[str, str]) -> dict[str, Any]:
    case = row["case"]
    base = load_npz(Path(row["baseline_path"]))
    graph = load_npz(Path(row["graph_path"]))
    weak = load_npz(WEAK_ROOT / case / "billboard_weak_regions.npz")
    pts = np.asarray(base["human_points"], dtype=np.float32)
    center, axes, proj = pca_frame(pts)
    thin = proj[:, 2]
    lo, hi = np.percentile(thin, [1, 99])
    q = np.clip((thin - lo) / max(float(hi - lo), 1e-9), 0.0, 1.0)
    cross_bin = np.clip(np.floor(q * 8).astype(np.int16), 0, 7)
    repair = np.asarray(weak["billboard_repair_region_mask"], dtype=bool)
    no_change = np.asarray(weak["no_change_mask"], dtype=bool)
    weak_score = np.asarray(graph["mentor_weak_region_score"], dtype=np.float32)
    conf = np.asarray(graph["mentor_smpl_confidence"], dtype=np.float32)
    body = np.asarray(graph["geometry_body_part_id"], dtype=np.int16)

    normal = axes[2]
    tangent = axes[1]
    sign = np.sign(thin).astype(np.float32)
    sign[sign == 0] = 1.0
    gate = np.maximum(repair.astype(np.float32), weak_score * 0.30)
    front_target = pts + normal[None] * sign[:, None] * (0.020 + 0.040 * gate[:, None])
    back_target = pts - normal[None] * sign[:, None] * (0.018 + 0.035 * gate[:, None])
    side_target = pts + tangent[None] * np.sign((body.astype(np.int32) % 5) - 2).astype(np.float32)[:, None] * (0.014 + 0.020 * repair.astype(np.float32)[:, None])
    shell_conf = np.clip(gate * 0.80 + conf * 0.20, 0, 1).astype(np.float32)

    part_continuity = np.zeros((len(pts), 8), dtype=np.float32)
    for part in range(8):
        part_continuity[:, part] = (body == part).astype(np.float32)
    repair_masks = {name: np.asarray(weak[key], dtype=bool) for name, key in PART_NAMES.items()}
    target_path = ensure(OUT_ROOT / case) / "part_graph_cross_section_targets.npz"
    np.savez_compressed(
        target_path,
        human_points=pts,
        body_part_id=body,
        cross_section_bin_target=cross_bin,
        cross_section_occupancy_target=np.eye(8, dtype=np.float32)[cross_bin],
        front_shell_target=front_target.astype(np.float32),
        back_shell_target=back_target.astype(np.float32),
        side_shell_target=side_target.astype(np.float32),
        shell_confidence=shell_conf,
        billboard_repair_region_mask=repair,
        no_change_mask=no_change,
        part_continuity_target=part_continuity,
        pca_center=center,
        pca_axes=axes,
        facial_detail_target_applicable=np.array(False),
        face_detail_claim_allowed=np.array(False),
        allowed_face_claim=np.array(ALLOWED_FACE_CLAIM),
        **{f"{name}_repair_mask": mask for name, mask in repair_masks.items()},
    )
    c_scores = {cfg: control_score(case, cfg) for cfg in CONTROL_CONFIGS}
    return {
        "case": case,
        "target_path": str(target_path),
        "repair_ratio": float(np.mean(repair)),
        "no_change_ratio": float(np.mean(no_change)),
        "mean_shell_confidence": float(np.mean(shell_conf[repair])) if bool(repair.any()) else 0.0,
        "cross_section_bins_present": int(len(np.unique(cross_bin[repair]))) if bool(repair.any()) else 0,
        **{f"{name}_repair_ratio": float(np.mean(mask & repair)) for name, mask in repair_masks.items()},
        **{f"{cfg}_thickness_proxy": val for cfg, val in c_scores.items()},
        "face_detail_claim_allowed": False,
    }


def preview(rows: list[dict[str, Any]]) -> None:
    panels: list[Image.Image] = []
    for row in rows:
        data = load_npz(Path(row["target_path"]))
        pts = np.asarray(data["human_points"], dtype=np.float32)
        repair = np.asarray(data["billboard_repair_region_mask"], dtype=bool)
        bins = np.asarray(data["cross_section_bin_target"], dtype=np.int16)
        size = (360, 260)
        im = Image.new("RGB", size, (248, 248, 244))
        draw = ImageDraw.Draw(im)
        xy = pts[:, :2]
        lo = np.percentile(xy, 1, axis=0)
        hi = np.percentile(xy, 99, axis=0)
        pad = (hi - lo) * 0.15 + 1e-6
        lo -= pad
        hi += pad
        q = (xy - lo[None]) / (hi[None] - lo[None] + 1e-9)
        q[:, 1] = 1.0 - q[:, 1]
        pix = np.clip(q * np.array([size[0] - 48, size[1] - 68]) + np.array([24, 44]), 0, [size[0] - 1, size[1] - 1]).astype(int)
        palette = np.array(
            [
                [64, 80, 70],
                [74, 117, 88],
                [97, 139, 92],
                [153, 151, 75],
                [194, 132, 67],
                [190, 91, 61],
                [155, 67, 75],
                [111, 59, 80],
            ],
            dtype=np.uint8,
        )
        step = max(1, len(pix) // 50000)
        for i in range(0, len(pix), step):
            x, y = pix[i]
            color = tuple(palette[int(bins[i])].tolist()) if repair[i] else (55, 72, 61)
            im.putpixel((int(x), int(y)), color)
        draw.text((8, 8), row["case"], fill=(10, 10, 10))
        panels.append(im)
    if not panels:
        return
    canvas = Image.new("RGB", (720, 520), (255, 255, 255))
    for i, panel in enumerate(panels[:4]):
        canvas.paste(panel, ((i % 2) * 360, (i // 2) * 260))
    ensure(BOARDS)
    canvas.save(BOARDS / "V17100000000000000000_part_graph_cross_section_target_preview.png")


def main() -> int:
    created_at = now()
    rows = [build_case(row) for row in read_manifest()]
    write_csv(REPORTS / "V17100000000000000000_part_graph_cross_section_target_manifest.csv", rows)
    preview(rows)
    write_json(
        REPORTS / "V17100000000000000000_part_graph_cross_section_target_decision.json",
        {
            "created_at": created_at,
            "status": "V17100_PART_GRAPH_TARGETS_READY_FOR_V172_TRAINING",
            "mentor_ready": False,
            "external_hard_block": False,
            "case_count": len(rows),
            "face_detail_claim_allowed": False,
            "allowed_face_claim": ALLOWED_FACE_CLAIM,
            "note": "Targets are training supervision only; they are not mentor visual evidence.",
        },
    )
    print(json.dumps({"created_at": created_at, "status": "V17100_DONE", "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
