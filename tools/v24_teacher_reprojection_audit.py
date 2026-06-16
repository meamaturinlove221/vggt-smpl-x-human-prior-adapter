from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
DEFAULT_V24 = REPO_ROOT / "output" / "surface_research_preflight_local" / "V24_residual_teacher_v2"
DEFAULT_TARGETS = DEFAULT_V24 / "v24_residual_teacher_targets_v2.npz"
DEFAULT_CASE = REPO_ROOT / "output" / "training_cases" / "0012_11_frame0000_6views_smplx_native_prior_v15"
DEFAULT_JSON = REPORTS / "20260508_v24_teacher_reprojection_audit.json"
DEFAULT_MD = REPORTS / "20260508_v24_teacher_reprojection_audit.md"


REGION_NAMES = ("body", "head", "face", "left_hand", "right_hand")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jr(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jr(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jr(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
    if isinstance(value, np.ndarray):
        return jr(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jr(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def finite_stats(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool)]
    arr = arr.reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": int(arr.size), "finite": 0}
    return {
        "count": int(arr.size),
        "finite": int(finite.size),
        "min": float(finite.min()),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(finite.max()),
    }


def scene_support(scene_dir: Path) -> dict[str, Any]:
    image_count = len(list((scene_dir / "images").glob("*.png"))) if (scene_dir / "images").is_dir() else 0
    mask_count = len(list((scene_dir / "masks").glob("*.png"))) if (scene_dir / "masks").is_dir() else 0
    return {
        "path": scene_dir,
        "exists": scene_dir.is_dir(),
        "image_count": image_count,
        "mask_count": mask_count,
        "manifest_exists": (scene_dir / "scene_manifest.json").is_file(),
        "prior_maps_exists": (scene_dir / "prior_maps.npz").is_file(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit V24 residual teacher reprojection/support safety.")
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    teacher = load_npz(args.targets)
    inputs = load_npz(args.case_root / "inputs.npz")
    targets = load_npz(args.case_root / "targets.npz")
    teacher_mask = np.asarray(teacher["teacher_mask"], dtype=bool)
    raw_mask = np.asarray(inputs["point_masks"], dtype=bool)
    teacher_depth = np.asarray(teacher["teacher_depths"], dtype=np.float32)
    raw_depth = np.asarray(targets["depths"], dtype=np.float32)
    prior_depth = np.asarray(targets["prior_depths"], dtype=np.float32)
    if teacher_depth.ndim == 4:
        teacher_depth_plane = teacher_depth[..., 0]
    else:
        teacher_depth_plane = teacher_depth

    intersection = teacher_mask & raw_mask
    union = teacher_mask | raw_mask
    mean_iou = float(intersection.sum() / max(1, union.sum()))
    per_view_iou = []
    for view in range(teacher_mask.shape[0]):
        inter = int((teacher_mask[view] & raw_mask[view]).sum())
        uni = int((teacher_mask[view] | raw_mask[view]).sum())
        per_view_iou.append(float(inter / max(1, uni)))

    depth_delta_raw = np.abs(teacher_depth_plane - raw_depth)
    depth_delta_prior = np.abs(teacher_depth_plane - prior_depth)
    region_id_map = np.asarray(teacher["teacher_region_id_map"], dtype=np.uint8)
    region_rows: dict[str, Any] = {}
    blockers: list[str] = []
    for idx, name in enumerate(REGION_NAMES, start=1):
        mask = region_id_map == idx
        pixels = int(mask.sum())
        per_view = mask.reshape(mask.shape[0], -1).sum(axis=1).astype(int).tolist()
        region_rows[name] = {
            "pixels": pixels,
            "views_with_pixels": int(np.sum(np.asarray(per_view) > 0)),
            "per_view_pixels": per_view,
            "teacher_vs_raw_depth_abs": finite_stats(depth_delta_raw, mask),
            "teacher_vs_prior_depth_abs": finite_stats(depth_delta_prior, mask),
        }
        if pixels <= 0:
            blockers.append(f"{name} region empty in V24 teacher")

    support_6v = {
        "case_root": args.case_root,
        "view_count": int(teacher_mask.shape[0]),
        "mask_iou_mean": mean_iou,
        "mask_iou_per_view": per_view_iou,
        "teacher_pixels": int(teacher_mask.sum()),
        "raw_pixels": int(raw_mask.sum()),
        "teacher_vs_raw_depth_abs": finite_stats(depth_delta_raw, teacher_mask),
        "teacher_vs_prior_depth_abs": finite_stats(depth_delta_prior, teacher_mask),
    }
    support_12v = scene_support(REPO_ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_12views_tmf")
    support_60v = {
        "available_scene_dirs": [p for p in (REPO_ROOT / "output").glob("*60*") if p.is_dir()][:20],
        "note": "V24 dense teacher arrays are 6-view only; 60-view support is audited as scene availability, not promoted as teacher coverage.",
    }

    if support_6v["teacher_pixels"] <= 1000:
        blockers.append("6v teacher pixels below minimum")
    if support_12v["image_count"] < 12 or support_12v["mask_count"] < 12:
        blockers.append("12v scene support incomplete")

    status = "DONE_PASS" if not blockers else "DONE_FAIL_ROUTED"
    audit = {
        "task": "v24_teacher_reprojection_audit",
        "created_utc": utc_now(),
        "status": status,
        "research_only": True,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "inputs": {"targets_npz": args.targets, "case_root": args.case_root},
        "support_audits": {"6v": support_6v, "12v": support_12v, "60v": support_60v},
        "region_coverage": region_rows,
        "blockers": blockers,
    }
    write_json(args.output_json, audit)
    lines = [
        "# V24 Teacher Reprojection Audit",
        "",
        f"Status: `{status}`",
        "",
        "## 6V Audit",
        "",
        f"- teacher_pixels: `{support_6v['teacher_pixels']}`",
        f"- raw_pixels: `{support_6v['raw_pixels']}`",
        f"- mean_mask_iou: `{support_6v['mask_iou_mean']}`",
        "",
        "## Region Coverage",
        "",
    ]
    for name, row in region_rows.items():
        lines.append(f"- {name}: pixels=`{row['pixels']}`, views=`{row['views_with_pixels']}`")
    lines.extend(["", "## 12V/60V Support", "", f"- 12v: `{jr(support_12v)}`", f"- 60v: `{jr(support_60v)}`", "", "## Blockers", ""])
    lines.extend([f"- {item}" for item in blockers] or ["- none"])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(jr({"status": status, "json": args.output_json}), ensure_ascii=False))
    return 0 if status in {"DONE_PASS", "DONE_FAIL_ROUTED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
