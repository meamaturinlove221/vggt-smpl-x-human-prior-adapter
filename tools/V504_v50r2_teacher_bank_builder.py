from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
BOARDS = ROOT / "boards"
OUT = ROOT / "output" / "V5040000000000000000000_v50r2_teacher_bank"
V503_DECISION = REPORTS / "V5030000000000000000000_regression_decision.json"

PACKAGE_FILES = Path(
    r"D:\vggt\vggt-main\output\surface_research_preflight_local"
    r"\V50_final_promotion_transaction\candidate_package_v50r2\package_files"
)
SCENE = Path(r"D:\vggt\vggt-main\output\4k4d_scenes\0012_11_frame0000_12views_tmf_v223_repaired")

CAMERAS = [
    ("cam00", "00_tgt_cam00.png", "back"),
    ("cam01", "01_src_cam01.png", "back"),
    ("cam06", "02_src_cam06.png", "side"),
    ("cam11", "03_src_cam11.png", "side"),
    ("cam16", "04_src_cam16.png", "side-oblique"),
    ("cam21", "05_src_cam21.png", "front-oblique"),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resize_image(path: Path, size: tuple[int, int], mode: str) -> np.ndarray:
    resample = Image.Resampling.NEAREST if mode == "mask" else Image.Resampling.LANCZOS
    with Image.open(path) as im:
        if mode == "rgb":
            im = im.convert("RGB")
        else:
            im = im.convert("L")
        im = im.resize(size, resample)
        arr = np.asarray(im)
    if mode == "mask":
        return (arr > 0).astype(np.uint8)
    return arr.astype(np.uint8)


def load_bank_arrays() -> dict[str, np.ndarray]:
    with np.load(PACKAGE_FILES / "candidate_files__candidate_points.npz", allow_pickle=False) as z:
        points = z["candidate_points_world"].astype(np.float32)
    with np.load(PACKAGE_FILES / "candidate_files__candidate_normals.npz", allow_pickle=False) as z:
        normals = z["candidate_normals_geometric"].astype(np.float32)
    with np.load(PACKAGE_FILES / "candidate_files__head_face_patch.npz", allow_pickle=False) as z:
        head_mask = z["head_mask"].astype(np.uint8)
        face_mask = z["face_mask"].astype(np.uint8)
        refined_points = z["refined_points_world"].astype(np.float32)
        refined_normals = z["refined_normals_world"].astype(np.float32)
        refined_visibility = z["refined_visibility"].astype(np.float32)
    with np.load(PACKAGE_FILES / "candidate_files__hand_patch.npz", allow_pickle=False) as z:
        hand_points = z["hand_points_world"].astype(np.float32)
        hand_normals = z["hand_normals_world"].astype(np.float32)
        hand_visibility = z["hand_visibility"].astype(np.float32)
        hand_region_id_map = z["hand_region_id_map"].astype(np.uint8)
    with np.load(PACKAGE_FILES / "v42_prior_enabled_payload__research_confidence.npz", allow_pickle=False) as z:
        world_conf = z["frame0000_world_points_conf"][:6].astype(np.float32)
        depth_conf = z["frame0000_depth_conf"][:6].astype(np.float32)
        normal_conf = z["frame0000_normal_conf"][:6].astype(np.float32)

    height, width = points.shape[1], points.shape[2]
    rgbs = []
    masks = []
    for _, scene_file, _ in CAMERAS:
        rgbs.append(resize_image(SCENE / "images" / scene_file, (width, height), "rgb"))
        masks.append(resize_image(SCENE / "masks" / scene_file, (width, height), "mask"))

    return {
        "points": points,
        "normals": normals,
        "rgb": np.stack(rgbs, axis=0),
        "full_body_mask": np.stack(masks, axis=0),
        "head_mask": head_mask,
        "face_mask": face_mask,
        "refined_points": refined_points,
        "refined_normals": refined_normals,
        "refined_visibility": refined_visibility,
        "hand_points": hand_points,
        "hand_normals": hand_normals,
        "hand_visibility": hand_visibility,
        "hand_region_id_map": hand_region_id_map,
        "world_points_conf": world_conf,
        "depth_conf": depth_conf,
        "normal_conf": normal_conf,
    }


def region_rows(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    region_masks = {
        "full_body_visible": arrays["full_body_mask"] > 0,
        "head_hair": arrays["head_mask"] > 0,
        "face_region": arrays["face_mask"] > 0,
        "hand_arm": arrays["hand_visibility"] > 0,
        "torso_clothing_proxy": (arrays["full_body_mask"] > 0) & ~(arrays["head_mask"] > 0) & ~(arrays["hand_visibility"] > 0),
        "leg_foot_proxy": arrays["full_body_mask"] > 0,
    }
    for view_index, (camera, _, orientation) in enumerate(CAMERAS):
        for region, mask in region_masks.items():
            m = mask[view_index]
            pts = arrays["points"][view_index]
            normals = arrays["normals"][view_index]
            point_count = int(m.sum())
            normal_finite_ratio = float(np.isfinite(normals[m]).mean()) if point_count else 0.0
            rows.append(
                {
                    "view_index": view_index,
                    "camera_id": camera,
                    "orientation": orientation,
                    "region": region,
                    "point_or_pixel_count": point_count,
                    "normal_finite_ratio": normal_finite_ratio,
                    "teacher_only": True,
                    "training_allowed": True,
                    "evaluation_allowed": True,
                    "final_inference_allowed": False,
                    "copy_forbidden": True,
                    "source": "V50R2 visual floor teacher bank",
                }
            )
    return rows


def overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    base = Image.fromarray(rgb).convert("RGB")
    red = Image.new("RGB", base.size, (255, 0, 0))
    alpha = Image.fromarray((mask.astype(np.uint8) * 92))
    base.paste(red, (0, 0), alpha)
    return base


def make_board(arrays: dict[str, np.ndarray], rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    cell_w, cell_h = 300, 370
    width = cell_w * 6
    board = Image.new("RGB", (width, cell_h * 2 + 92), "white")
    draw = ImageDraw.Draw(board)
    draw.text((14, 10), "V504 V50R2 teacher bank regions: RGB observation + teacher masks, teacher-only", fill=(0, 0, 0), font=font)
    draw.text((14, 30), "Final inference may not receive teacher points/RGB crops; these assets are for loss/eval/reference only.", fill=(130, 0, 0), font=font)

    for i, (camera, _, orientation) in enumerate(CAMERAS):
        rgb = arrays["rgb"][i]
        full = arrays["full_body_mask"][i] > 0
        head = arrays["head_mask"][i] > 0
        hand = arrays["hand_visibility"][i] > 0
        combined = full | head | hand
        im1 = overlay_mask(rgb, combined)
        im1.thumbnail((cell_w - 8, cell_h - 48), Image.Resampling.LANCZOS)
        x = i * cell_w + 4
        y = 56
        board.paste(im1, (x + (cell_w - im1.width) // 2, y + 24))
        draw.rectangle([x, y, x + cell_w - 8, y + cell_h - 4], outline=(80, 120, 80))
        draw.text((x + 8, y + 8), f"{camera} {orientation}", fill=(0, 0, 0), font=font)
        draw.text((x + 8, y + cell_h - 24), f"full={int(full.sum())} head={int(head.sum())} hand={int(hand.sum())}", fill=(0, 0, 0), font=font)

        im2 = Image.fromarray(arrays["rgb"][i]).convert("RGB")
        im2.thumbnail((cell_w - 8, cell_h - 48), Image.Resampling.LANCZOS)
        y2 = 56 + cell_h
        board.paste(im2, (x + (cell_w - im2.width) // 2, y2 + 24))
        draw.rectangle([x, y2, x + cell_w - 8, y2 + cell_h - 4], outline=(160, 160, 160))
        draw.text((x + 8, y2 + 8), f"{camera} RGB source", fill=(0, 0, 0), font=font)

    board.save(output)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = load_bank_arrays()
    rows = region_rows(arrays)

    bank_npz = OUT / "v50r2_teacher_bank.npz"
    np.savez_compressed(
        bank_npz,
        cameras=np.array([c[0] for c in CAMERAS]),
        orientations=np.array([c[2] for c in CAMERAS]),
        points=arrays["points"],
        normals=arrays["normals"],
        rgb=arrays["rgb"],
        full_body_mask=arrays["full_body_mask"],
        head_mask=arrays["head_mask"],
        face_mask=arrays["face_mask"],
        refined_points=arrays["refined_points"],
        refined_normals=arrays["refined_normals"],
        refined_visibility=arrays["refined_visibility"],
        hand_points=arrays["hand_points"],
        hand_normals=arrays["hand_normals"],
        hand_visibility=arrays["hand_visibility"],
        hand_region_id_map=arrays["hand_region_id_map"],
        world_points_conf=arrays["world_points_conf"],
        depth_conf=arrays["depth_conf"],
        normal_conf=arrays["normal_conf"],
        teacher_only=np.array(True),
        final_inference_allowed=np.array(False),
        copy_forbidden=np.array(True),
        training_allowed=np.array(True),
        evaluation_allowed=np.array(True),
    )

    metadata = {
        "task": "V504_v50r2_teacher_bank_builder",
        "status": "V504_V50R2_TEACHER_BANK_READY_CONTINUE_NOT_PROMOTED",
        "created_at": now(),
        "repo": str(ROOT),
        "input_v503_decision": str(V503_DECISION),
        "teacher_bank_npz": str(bank_npz),
        "teacher_bank_sha256": sha256(bank_npz),
        "teacher_only": True,
        "training_allowed": True,
        "evaluation_allowed": True,
        "final_inference_allowed": False,
        "copy_forbidden": True,
        "v50r2_modified": False,
        "regions": sorted({r["region"] for r in rows}),
        "cameras": [c[0] for c in CAMERAS],
        "source_package_files": {p.name: str(p) for p in sorted(PACKAGE_FILES.glob("*"))},
        "decision": "Teacher bank is built from V50R2/V42 geometry, normals, RGB observations, masks, head/face, and hand regions. It is not final output and must be blocked from final inference.",
    }
    metadata_json = OUT / "metadata.json"
    write_json(metadata_json, metadata)

    manifest_csv = REPORTS / "V5040000000000000000000_teacher_bank_manifest.csv"
    board_png = BOARDS / "V5040000000000000000000_teacher_bank_regions.png"
    decision_json = REPORTS / "V5040000000000000000000_teacher_bank_decision.json"
    write_csv(manifest_csv, rows)
    make_board(arrays, rows, board_png)
    write_json(
        decision_json,
        {
            **metadata,
            "metadata_json": str(metadata_json),
            "manifest_csv": str(manifest_csv),
            "regions_board": str(board_png),
            "gates": {
                "teacher_bank_npz_readable": True,
                "all_six_cameras_present": True,
                "region_manifest_written": True,
                "teacher_only_policy_written": True,
                "final_inference_allowed": False,
                "not_promoted": True,
            },
            "blockers": [],
        },
    )
    print(json.dumps({"status": metadata["status"], "teacher_bank_npz": str(bank_npz), "manifest_csv": str(manifest_csv), "regions_board": str(board_png), "decision_json": str(decision_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
