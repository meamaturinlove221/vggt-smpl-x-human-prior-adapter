from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
DEFAULT_TARGETS = REPO_ROOT / "output" / "surface_research_preflight_local" / "V24_residual_teacher_v2" / "v24_residual_teacher_targets_v2.npz"
DEFAULT_OUT = REPO_ROOT / "output" / "surface_research_preflight_local" / "V24_residual_teacher_v2"
DEFAULT_JSON = REPORTS / "20260508_v24_teacher_visual_review_pack.json"
DEFAULT_MD = REPORTS / "20260508_v24_teacher_visual_review_pack.md"


REGION_NAMES = ("body", "head", "face", "left_hand", "right_hand")
COLORS = {
    1: (120, 210, 255),
    2: (255, 200, 90),
    3: (255, 110, 170),
    4: (100, 240, 140),
    5: (170, 120, 255),
}


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
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jr(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def normalize_image(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.percentile(arr[finite], 2))
    hi = float(np.percentile(arr[finite], 98))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return (out * 255).astype(np.uint8)


def make_contact_sheet(images: list[Image.Image], labels: list[str], cols: int, cell_pad: int = 8) -> Image.Image:
    if not images:
        return Image.new("RGB", (640, 360), (20, 20, 20))
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    label_h = 22
    sheet = Image.new("RGB", (cols * (w + cell_pad) + cell_pad, rows * (h + label_h + cell_pad) + cell_pad), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for i, img in enumerate(images):
        x = cell_pad + (i % cols) * (w + cell_pad)
        y = cell_pad + (i // cols) * (h + label_h + cell_pad)
        draw.text((x, y), labels[i], fill=(235, 235, 235))
        sheet.paste(img.convert("RGB"), (x, y + label_h))
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description="Create V24 visual review contact sheets.")
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.targets, allow_pickle=False) as z:
        teacher_depths = np.asarray(z["teacher_depths"], dtype=np.float32)
        visibility = np.asarray(z["teacher_visibility"], dtype=np.float32) > 0
        region_map = np.asarray(z["teacher_region_id_map"], dtype=np.uint8)
        uncertainty = np.asarray(z["teacher_uncertainty"], dtype=np.float32)
    depth = teacher_depths[..., 0] if teacher_depths.ndim == 4 else teacher_depths

    depth_imgs: list[Image.Image] = []
    region_imgs: list[Image.Image] = []
    unc_imgs: list[Image.Image] = []
    labels: list[str] = []
    for v in range(depth.shape[0]):
        labels.append(f"view {v}")
        depth_gray = normalize_image(depth[v], visibility[v])
        depth_rgb = np.stack([depth_gray, depth_gray, depth_gray], axis=-1)
        depth_rgb[~visibility[v]] = 0
        depth_imgs.append(Image.fromarray(depth_rgb, "RGB").resize((259, 259), Image.Resampling.NEAREST))

        region_rgb = np.zeros((*region_map.shape[1:], 3), dtype=np.uint8)
        for rid, color in COLORS.items():
            region_rgb[region_map[v] == rid] = color
        region_imgs.append(Image.fromarray(region_rgb, "RGB").resize((259, 259), Image.Resampling.NEAREST))

        unc = normalize_image(uncertainty[v], visibility[v])
        unc_rgb = np.stack([unc, 255 - unc, np.zeros_like(unc)], axis=-1)
        unc_rgb[~visibility[v]] = 0
        unc_imgs.append(Image.fromarray(unc_rgb, "RGB").resize((259, 259), Image.Resampling.NEAREST))

    depth_path = args.output_dir / "v24_teacher_depth_contact_sheet.png"
    region_path = args.output_dir / "v24_teacher_region_contact_sheet.png"
    unc_path = args.output_dir / "v24_teacher_uncertainty_contact_sheet.png"
    make_contact_sheet(depth_imgs, labels, cols=3).save(depth_path)
    make_contact_sheet(region_imgs, labels, cols=3).save(region_path)
    make_contact_sheet(unc_imgs, labels, cols=3).save(unc_path)

    per_region = {}
    for idx, name in enumerate(REGION_NAMES, start=1):
        per_region[name] = {
            "pixels": int((region_map == idx).sum()),
            "views_with_pixels": int(np.sum((region_map == idx).reshape(region_map.shape[0], -1).sum(axis=1) > 0)),
        }
    blockers = [f"{name} visual region empty" for name, row in per_region.items() if row["pixels"] <= 0]
    status = "DONE_PASS" if not blockers else "DONE_FAIL_ROUTED"
    summary = {
        "task": "v24_teacher_visual_review_pack",
        "created_utc": utc_now(),
        "status": status,
        "research_only": True,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "outputs": {
            "depth_contact_sheet": depth_path,
            "region_contact_sheet": region_path,
            "uncertainty_contact_sheet": unc_path,
        },
        "region_coverage": per_region,
        "blockers": blockers,
    }
    write_json(args.output_json, summary)
    lines = [
        "# V24 Teacher Visual Review Pack",
        "",
        f"Status: `{status}`",
        "",
        f"- depth_contact_sheet: `{depth_path}`",
        f"- region_contact_sheet: `{region_path}`",
        f"- uncertainty_contact_sheet: `{unc_path}`",
        "",
        "## Regions",
        "",
    ]
    for name, row in per_region.items():
        lines.append(f"- {name}: pixels=`{row['pixels']}`, views=`{row['views_with_pixels']}`")
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- {item}" for item in blockers] or ["- none"])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(jr({"status": status, "json": args.output_json}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
