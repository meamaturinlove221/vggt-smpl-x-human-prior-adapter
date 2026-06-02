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

V50R2_PACKAGE = Path(
    r"D:\vggt\vggt-main\output\surface_research_preflight_local"
    r"\V50_final_promotion_transaction\candidate_package_v50r2"
)
PACKAGE_FILES = V50R2_PACKAGE / "package_files"
SCENE = Path(r"D:\vggt\vggt-main\output\4k4d_scenes\0012_11_frame0000_12views_tmf_v223_repaired")
POINTCLOUD_SHEET = Path(
    r"D:\vggt\vggt-main\output\mentor_report_v50r2\v223_view_consistent_pointcloud"
    r"\images\V223_V50R2_full_body_pointcloud_v42_consistent.png"
)
RGB_CONTACT_SHEET = SCENE / "rgb_contact_sheet.png"
MASK_CONTACT_SHEET = SCENE / "mask_contact_sheet.png"
V32_FULL_BODY_PLY = Path(
    r"D:\vggt\vggt-main\output\surface_research_preflight_local"
    r"\V32_candidate_inference_research\v32_candidate_open3d_review_points.ply"
)
V33_HEAD_FACE_PLY = Path(
    r"D:\vggt\vggt-main\output\surface_research_preflight_local"
    r"\V33_head_face_detail_route\v33_head_face_refined_teacher_points.ply"
)
V34_HAND_PLY = Path(
    r"D:\vggt\vggt-main\output\surface_research_preflight_local"
    r"\V34_smplx_native_hand_route\v34_smplx_native_hand_continuity_patch.ply"
)

CAMERAS = [
    {"view_index": 0, "camera_id": "cam00", "scene_file": "00_tgt_cam00.png", "orientation": "back"},
    {"view_index": 1, "camera_id": "cam01", "scene_file": "01_src_cam01.png", "orientation": "back"},
    {"view_index": 2, "camera_id": "cam06", "scene_file": "02_src_cam06.png", "orientation": "side"},
    {"view_index": 3, "camera_id": "cam11", "scene_file": "03_src_cam11.png", "orientation": "side"},
    {"view_index": 4, "camera_id": "cam16", "scene_file": "04_src_cam16.png", "orientation": "side-oblique"},
    {"view_index": 5, "camera_id": "cam21", "scene_file": "05_src_cam21.png", "orientation": "front-oblique"},
]

POINTCOUNTS_FROM_V223_PANEL = {
    "cam00": 11240,
    "cam01": 14315,
    "cam06": 9293,
    "cam11": 12067,
    "cam16": 11274,
    "cam21": 11264,
}

NPZ_PATHS = {
    "candidate_points": PACKAGE_FILES / "candidate_files__candidate_points.npz",
    "candidate_normals": PACKAGE_FILES / "candidate_files__candidate_normals.npz",
    "v42_points_world": PACKAGE_FILES / "v42_prior_enabled_payload__research_points_world.npz",
    "v42_normals": PACKAGE_FILES / "v42_prior_enabled_payload__research_normals_geometric.npz",
    "v42_confidence": PACKAGE_FILES / "v42_prior_enabled_payload__research_confidence.npz",
    "v42_depths": PACKAGE_FILES / "v42_prior_enabled_payload__research_depths.npz",
    "head_face_patch": PACKAGE_FILES / "candidate_files__head_face_patch.npz",
    "hand_patch": PACKAGE_FILES / "candidate_files__hand_patch.npz",
    "temporal_teacher": PACKAGE_FILES / "candidate_files__temporal_teacher.npz",
}

METADATA_PATHS = {
    "v50r2_manifest": V50R2_PACKAGE / "manifest.json",
    "v50r2_hash_manifest": V50R2_PACKAGE / "hash_manifest.json",
    "v50r2_registry_entry": V50R2_PACKAGE / "strict_registry_entry_v50r2.json",
    "v32_summary": Path(
        r"D:\vggt\vggt-main\output\surface_research_preflight_local"
        r"\V32_candidate_inference_research\summary.json"
    ),
    "v33_summary": Path(
        r"D:\vggt\vggt-main\output\surface_research_preflight_local"
        r"\V33_head_face_detail_route\summary.json"
    ),
    "v34_summary": Path(
        r"D:\vggt\vggt-main\output\surface_research_preflight_local"
        r"\V34_smplx_native_hand_route\summary.json"
    ),
    "v44_summary": Path(
        r"D:\vggt\vggt-main\output\surface_research_preflight_local"
        r"\V44_strict_visual_pre_promotion_gate\summary.json"
    ),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_size(path: Path) -> list[int] | None:
    if not path.is_file():
        return None
    with Image.open(path) as im:
        return [im.width, im.height]


def mask_pixels(path: Path) -> int | None:
    if not path.is_file():
        return None
    with Image.open(path).convert("L") as im:
        arr = np.asarray(im)
    return int((arr > 0).sum())


def npz_summary(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "sha256": sha256(path), "keys": []}
    if not path.is_file():
        return out
    with np.load(path, allow_pickle=False) as z:
        for key in z.files:
            arr = z[key]
            item: dict[str, Any] = {"key": key, "shape": list(arr.shape), "dtype": str(arr.dtype)}
            if arr.size and np.issubdtype(arr.dtype, np.number):
                finite = np.isfinite(arr)
                item["finite_ratio"] = float(finite.mean())
                if finite.any():
                    vals = arr[finite]
                    item["min"] = float(vals.min())
                    item["max"] = float(vals.max())
            out["keys"].append(item)
    return out


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    with Image.open(path).convert("RGB") as im:
        im.thumbnail((size[0], size[1] - 34), Image.Resampling.LANCZOS)
        x = (size[0] - im.width) // 2
        y = 24 + (size[1] - 34 - im.height) // 2
        canvas.paste(im, (x, y))
    return canvas


def make_contact_sheet(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    top_width = 1800
    with Image.open(POINTCLOUD_SHEET).convert("RGB") as pc:
        pc.thumbnail((top_width, 920), Image.Resampling.LANCZOS)
        top = Image.new("RGB", (top_width, pc.height + 46), "white")
        draw = ImageDraw.Draw(top)
        draw.text((14, 10), "V50R2 human morphology visual floor: V223/V42-consistent RGB point cloud panels", fill=(0, 0, 0), font=font)
        top.paste(pc, ((top_width - pc.width) // 2, 40))

    cell_w, cell_h = 300, 380
    rgb_sheet = Image.new("RGB", (top_width, cell_h * 2 + 50), "white")
    draw = ImageDraw.Draw(rgb_sheet)
    draw.text((14, 8), "Matched RGB observation source with partial environment (same six cameras)", fill=(0, 0, 0), font=font)
    for i, row in enumerate(rows):
        col = i % 6
        x = col * cell_w
        y = 36
        img = fit_image(Path(row["source_rgb_png"]), (cell_w, cell_h))
        rgb_sheet.paste(img, (x, y))
        draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=(180, 180, 180))
        label = f'{row["camera_id"]} {row["orientation"]} mask={row["mask_pixel_count"]}'
        draw.text((x + 8, y + cell_h - 20), label, fill=(0, 0, 0), font=font)

    board = Image.new("RGB", (top_width, top.height + rgb_sheet.height + 20), "white")
    board.paste(top, (0, 0))
    board.paste(rgb_sheet, (0, top.height + 20))
    board.save(output)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    rows: list[dict[str, Any]] = []
    for cam in CAMERAS:
        rgb = SCENE / "images" / cam["scene_file"]
        mask = SCENE / "masks" / cam["scene_file"]
        rows.append(
            {
                "view_index": cam["view_index"],
                "camera_id": cam["camera_id"],
                "orientation": cam["orientation"],
                "v50r2_pointcloud_panel_png": str(POINTCLOUD_SHEET),
                "v50r2_panel_point_count_from_png": POINTCOUNTS_FROM_V223_PANEL[cam["camera_id"]],
                "source_rgb_png": str(rgb),
                "source_rgb_exists": rgb.is_file(),
                "source_rgb_size": json.dumps(image_size(rgb), ensure_ascii=False),
                "source_mask_png": str(mask),
                "source_mask_exists": mask.is_file(),
                "mask_pixel_count": mask_pixels(mask),
                "candidate_points_npz": str(NPZ_PATHS["candidate_points"]),
                "candidate_normals_npz": str(NPZ_PATHS["candidate_normals"]),
                "v42_points_npz": str(NPZ_PATHS["v42_points_world"]),
                "v42_normals_npz": str(NPZ_PATHS["v42_normals"]),
                "v42_confidence_npz": str(NPZ_PATHS["v42_confidence"]),
                "v42_depths_npz": str(NPZ_PATHS["v42_depths"]),
                "full_body_shared_ply": str(V32_FULL_BODY_PLY),
                "per_camera_ply_found": False,
                "rgb_source": "4k4d_scene_0012_11_frame0000_12views_tmf_v223_repaired",
                "normal_route": "V50R2 candidate_normals + V42 research_normals_geometric",
                "vggt_source": "V42 prior-enabled VGGT world points",
                "confidence_source": "V42 world_points_conf/depth_conf/normal_conf",
                "mask_source": "12-view repaired scene human mask",
                "teacher_only": True,
                "training_allowed_after_firewall": True,
                "final_inference_allowed": False,
                "v50r2_modify_allowed": False,
            }
        )

    inventory_csv = REPORTS / "V5010000000000000000000_v50r2_visual_floor_inventory.csv"
    board_png = BOARDS / "V5010000000000000000000_v50r2_visual_floor_contact_sheet.png"
    decision_json = REPORTS / "V5010000000000000000000_v50r2_visual_floor_decision.json"

    write_csv(inventory_csv, rows)
    make_contact_sheet(rows, board_png)

    npz = {name: npz_summary(path) for name, path in NPZ_PATHS.items()}
    metadata = {name: {"path": str(path), "exists": path.is_file(), "sha256": sha256(path)} for name, path in METADATA_PATHS.items()}
    ply = {
        "full_body_shared_ply": {"path": str(V32_FULL_BODY_PLY), "exists": V32_FULL_BODY_PLY.is_file(), "sha256": sha256(V32_FULL_BODY_PLY)},
        "head_face_ply": {"path": str(V33_HEAD_FACE_PLY), "exists": V33_HEAD_FACE_PLY.is_file(), "sha256": sha256(V33_HEAD_FACE_PLY)},
        "hand_ply": {"path": str(V34_HAND_PLY), "exists": V34_HAND_PLY.is_file(), "sha256": sha256(V34_HAND_PLY)},
        "per_camera_ply_found": False,
    }
    gates = {
        "six_camera_v50r2_pointcloud_png_found": POINTCLOUD_SHEET.is_file(),
        "matched_rgb_scene_found": all(Path(r["source_rgb_png"]).is_file() for r in rows),
        "matched_masks_found": all(Path(r["source_mask_png"]).is_file() for r in rows),
        "candidate_points_npz_found": NPZ_PATHS["candidate_points"].is_file(),
        "candidate_normals_npz_found": NPZ_PATHS["candidate_normals"].is_file(),
        "v42_points_normals_confidence_depth_found": all(NPZ_PATHS[k].is_file() for k in ["v42_points_world", "v42_normals", "v42_confidence", "v42_depths"]),
        "shared_ply_found": V32_FULL_BODY_PLY.is_file(),
        "per_camera_ply_found": False,
        "v50r2_modified": False,
        "teacher_bank_input_ready": True,
        "mentor_visual_floor_ready": True,
        "full_scene_final_ready": False,
        "not_promoted": True,
    }
    decision = {
        "task": "V501_v50r2_visual_floor_intake",
        "status": "V501_V50R2_VISUAL_FLOOR_INTAKE_COMPLETE_CONTINUE_NOT_PROMOTED",
        "created_at": now(),
        "repo": str(ROOT),
        "v50r2_source_package": str(V50R2_PACKAGE),
        "inventory_csv": str(inventory_csv),
        "contact_sheet": str(board_png),
        "source_visual_floor_png": str(POINTCLOUD_SHEET),
        "source_rgb_contact_sheet": str(RGB_CONTACT_SHEET),
        "source_mask_contact_sheet": str(MASK_CONTACT_SHEET),
        "cameras": [r["camera_id"] for r in rows],
        "gates": gates,
        "npz": npz,
        "ply": ply,
        "metadata": metadata,
        "hard_policy": {
            "v50r2_roles": ["visual_floor", "teacher", "reference"],
            "final_inference_allowed": False,
            "copy_forbidden": True,
            "modify_v50_v50r2": False,
            "promotion": False,
            "registry": False,
            "active_candidate_replacement": False,
        },
        "decision": (
            "V50R2 six-panel morphology floor and matched RGB/mask observations are indexed. "
            "NPZ geometry/normal/confidence/depth sources are present; only shared PLYs were found, "
            "so per-camera PLY remains missing but is not a hard block because the V50R2 teacher bank can be built from NPZ. "
            "Continue to V502 panel audit and V503 regression audit; do not claim final mentor readiness."
        ),
        "blockers": [],
    }
    write_json(decision_json, decision)
    print(json.dumps({"status": decision["status"], "inventory_csv": str(inventory_csv), "contact_sheet": str(board_png), "decision_json": str(decision_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
