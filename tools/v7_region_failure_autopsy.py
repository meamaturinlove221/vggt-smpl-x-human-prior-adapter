from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/surface_research_preflight_local/V7_region_failure_autopsy"
DEFAULT_REPORT = REPO_ROOT / "reports/20260507_v7_region_failure_autopsy.md"

ARTIFACTS = {
    "B-Fus3D0-v2": {
        "summary": REPO_ROOT / "output/surface_research_preflight_local/B_Fus3D0_v2_latent_grid_sdf_backend_smoke/b_fus3d0_v2_latent_grid_sdf_backend_summary.json",
        "real": REPO_ROOT / "output/surface_research_preflight_local/B_Fus3D0_v2_latent_grid_sdf_backend_smoke/b_fus3d0_v2_real_latent_grid_sdf_occupied_points.ply",
        "shuffle": REPO_ROOT / "output/surface_research_preflight_local/B_Fus3D0_v2_latent_grid_sdf_backend_smoke/b_fus3d0_v2_shuffle_latent_grid_sdf_occupied_points.ply",
        "zero": REPO_ROOT / "output/surface_research_preflight_local/B_Fus3D0_v2_latent_grid_sdf_backend_smoke/b_fus3d0_v2_zero_latent_grid_sdf_occupied_points.ply",
        "random_view": REPO_ROOT / "output/surface_research_preflight_local/B_Fus3D0_v2_latent_grid_sdf_backend_smoke/b_fus3d0_v2_random_view_latent_grid_sdf_occupied_points.ply",
        "contact": REPO_ROOT / "output/surface_research_preflight_local/B_Fus3D0_v2_latent_grid_sdf_backend_smoke/b_fus3d0_v2_open3d_contact_sheet.png",
        "genealogy": {"uses_vggt_tokens": True, "learned_decoder": False, "scaffold_only": False, "uses_smplx_anchor": False},
    },
    "B-GS1": {
        "summary": REPO_ROOT / "output/surface_research_preflight_local/B_GS1_visibility_aware_free_gaussian_backend/b_gs1_summary.json",
        "real": REPO_ROOT / "output/surface_research_preflight_local/B_GS1_visibility_aware_free_gaussian_backend/b_gs1_visibility_aware_combined.ply",
        "constrained": REPO_ROOT / "output/surface_research_preflight_local/B_GS1_visibility_aware_free_gaussian_backend/b_gs1_constrained_baseline.ply",
        "random": REPO_ROOT / "output/surface_research_preflight_local/B_GS1_visibility_aware_free_gaussian_backend/b_gs1_random_control_combined.ply",
        "raw": REPO_ROOT / "output/surface_research_preflight_local/B_GS1_visibility_aware_free_gaussian_backend/b_gs1_raw_free_candidates.ply",
        "contact": REPO_ROOT / "output/surface_research_preflight_local/B_GS1_visibility_aware_free_gaussian_backend/open3d_contact_sheet/b_gs1_visibility_aware_combined_contact_sheet.png",
        "genealogy": {"uses_vggt_tokens": False, "learned_decoder": False, "scaffold_only": False, "uses_smplx_anchor": True},
    },
    "B-hair1": {
        "summary": REPO_ROOT / "reports/20260507_b_hair1_backend_status.json",
        "real": REPO_ROOT / "output/surface_research_preflight_local/B_hair1_backend_smoke_v6/b_hair1_backend_real_strand_gaussian_chain.ply",
        "shuffle": REPO_ROOT / "output/surface_research_preflight_local/B_hair1_backend_smoke_v6/b_hair1_backend_shuffle_strand_gaussian_chain.ply",
        "zero": REPO_ROOT / "output/surface_research_preflight_local/B_hair1_backend_smoke_v6/b_hair1_backend_zero_strand_gaussian_chain.ply",
        "mask_only": REPO_ROOT / "output/surface_research_preflight_local/B_hair1_backend_smoke_v6/b_hair1_backend_mask_only_strand_gaussian_chain.ply",
        "contact": REPO_ROOT / "output/surface_research_preflight_local/B_hair1_backend_smoke_v6/b_hair1_backend_head_hairline_headtop_contact_sheet.png",
        "genealogy": {"uses_vggt_tokens": True, "learned_decoder": False, "scaffold_only": False, "uses_smplx_anchor": True},
    },
    "B-hand8": {
        "summary": REPO_ROOT / "output/surface_research_preflight_local/B_hand8_connected_hand_arm_surface_backend_smoke/b_hand8_connected_hand_arm_surface_backend_summary.json",
        "real": REPO_ROOT / "output/surface_research_preflight_local/B_hand8_connected_hand_arm_surface_backend_smoke/b_hand8_combined_connected_hand_arm_surface_pointcloud.ply",
        "contact": REPO_ROOT / "output/surface_research_preflight_local/B_hand8_connected_hand_arm_surface_backend_smoke/b_hand8_open3d_contact_sheet.png",
        "genealogy": {"uses_vggt_tokens": False, "learned_decoder": False, "scaffold_only": True, "uses_smplx_anchor": True, "uses_colmap_depth": True},
    },
}

REGIONS = ("full_body", "head", "face_core", "hairline", "head_top", "left_hand", "right_hand", "hands")
STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V7 region-wise autopsy for v6 backend artifacts.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=760)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        val = float(value)
        return val if math.isfinite(val) else str(val)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "path": str(path), "error": "missing"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"available": False, "path": str(path), "error": "not_dict"}


def read_ascii_ply_vertices(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {"points": np.zeros((0, 3), dtype=np.float32), "colors": np.zeros((0, 3), dtype=np.uint8)}
    with path.open("r", encoding="ascii", errors="ignore") as handle:
        header: list[str] = []
        vertex_count = 0
        properties: list[str] = []
        in_vertex = False
        for line in handle:
            line = line.strip()
            header.append(line)
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                in_vertex = True
                continue
            if line.startswith("element ") and not line.startswith("element vertex"):
                in_vertex = False
            if in_vertex and line.startswith("property"):
                properties.append(line.split()[-1])
            if line == "end_header":
                break
        rows: list[list[float]] = []
        for _ in range(vertex_count):
            parts = handle.readline().strip().split()
            if len(parts) < 3:
                continue
            rows.append([float(v) for v in parts[: min(len(parts), max(6, len(properties))) ]])
    if not rows:
        return {"points": np.zeros((0, 3), dtype=np.float32), "colors": np.zeros((0, 3), dtype=np.uint8)}
    arr = np.asarray(rows, dtype=np.float32)
    points = arr[:, :3].astype(np.float32)
    if arr.shape[1] >= 6:
        colors = np.clip(arr[:, 3:6], 0, 255).astype(np.uint8)
    else:
        colors = np.full((points.shape[0], 3), 160, dtype=np.uint8)
    return {"points": points, "colors": colors}


def region_mask(points: np.ndarray, colors: np.ndarray, region: str) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
    mask = np.ones((points.shape[0],), dtype=bool)
    if region == "full_body":
        return mask
    if region == "head":
        return y <= np.quantile(y, 0.26)
    if region == "face_core":
        return (y <= np.quantile(y, 0.35)) & (np.abs(x - np.median(x)) <= max(0.08, np.std(x) * 0.75))
    if region == "hairline":
        purple = (r > 120) & (b > 150) & (g < 150)
        return purple | (y <= np.quantile(y, 0.18))
    if region == "head_top":
        return y <= np.quantile(y, 0.12)
    if region == "left_hand":
        blue = (b > 180) & (r < 90)
        return blue | (x <= np.quantile(x, 0.18))
    if region == "right_hand":
        orange = (r > 200) & (g < 170) & (b < 120)
        return orange | (x >= np.quantile(x, 0.82))
    if region == "hands":
        return region_mask(points, colors, "left_hand") | region_mask(points, colors, "right_hand")
    return mask


def projection(points: np.ndarray, colors: np.ndarray, out_path: Path, *, width: int, height: int, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if points.size == 0:
        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), title + " empty", fill=(0, 0, 0))
        img.save(out_path)
        return
    centered = points.astype(np.float64) - np.median(points, axis=0, keepdims=True)
    ax = 0.70 * centered[:, 0] + 0.25 * centered[:, 2]
    ay = -0.85 * centered[:, 1] + 0.16 * centered[:, 2]
    depth = centered[:, 2] + 0.2 * centered[:, 0]
    lo_x, hi_x = np.quantile(ax, [0.01, 0.99])
    lo_y, hi_y = np.quantile(ay, [0.01, 0.99])
    pad_x = max(1e-6, float(hi_x - lo_x) * 0.10)
    pad_y = max(1e-6, float(hi_y - lo_y) * 0.10)
    px = np.clip(((ax - (lo_x - pad_x)) / max(1e-6, hi_x - lo_x + 2 * pad_x) * (width - 1)).round().astype(np.int64), 0, width - 1)
    py = np.clip(((1.0 - (ay - (lo_y - pad_y)) / max(1e-6, hi_y - lo_y + 2 * pad_y)) * (height - 1)).round().astype(np.int64), 0, height - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for idx in np.argsort(depth):
        x, y = int(px[idx]), int(py[idx])
        canvas[max(0, y - 1) : min(height, y + 2), max(0, x - 1) : min(width, x + 2)] = colors[idx]
    img = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.text((8, 8), title, fill=(0, 0, 0))
    img.save(out_path)


def make_sheet(paths: list[Path], out_path: Path, title: str) -> None:
    thumbs = [Image.open(p).convert("RGB").resize((260, 220), Image.Resampling.BICUBIC) for p in paths if p.is_file()]
    labels = [p.stem for p in paths if p.is_file()]
    if not thumbs:
        return
    cols = 4
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 260, rows * 246 + 34), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=(0, 0, 0))
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * 260
        y = 34 + (idx // cols) * 246
        sheet.paste(thumb, (x, y))
        draw.text((x + 5, y + 224), labels[idx][:36], fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def score_region(points: np.ndarray, colors: np.ndarray, region: str) -> dict[str, Any]:
    mask = region_mask(points, colors, region)
    count = int(np.count_nonzero(mask))
    total = int(points.shape[0])
    if count == 0:
        return {"point_count": 0, "point_ratio": 0.0, "spread": 0.0, "region_score": 0.0}
    sub = points[mask]
    extent = sub.max(axis=0) - sub.min(axis=0)
    spread = float(np.linalg.norm(extent))
    density = float(count / max(total, 1))
    score = float(np.clip(0.55 * min(1.0, density * 8.0) + 0.45 * min(1.0, spread / 0.35), 0.0, 1.0))
    return {"point_count": count, "point_ratio": density, "spread": spread, "region_score": score}


def compare_regions(real: dict[str, np.ndarray], controls: dict[str, dict[str, np.ndarray]], region: str) -> dict[str, Any]:
    real_score = score_region(real["points"], real["colors"], region)["region_score"]
    out = {"real_region_score": real_score}
    for name, payload in controls.items():
        if payload["points"].shape[0] == 0:
            out[f"real_vs_{name}_by_region"] = None
            continue
        ctrl = score_region(payload["points"], payload["colors"], region)["region_score"]
        out[f"{name}_region_score"] = ctrl
        out[f"real_vs_{name}_by_region"] = real_score - ctrl
    return out


def derive_line_scores(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    if name == "B-GS1":
        return {
            "overfill_by_region": "global visibility_minus_constrained_overfill is positive; region-specific render metrics unavailable in v6",
            "template_dependency_score": 0.65,
            "freeze": True,
        }
    if name == "B-hair1":
        comp = summary.get("comparison", {})
        return {
            "real_vs_shuffle_by_region": comp.get("real_minus_shuffle_root_score"),
            "real_vs_zero_by_region": comp.get("real_minus_zero_root_score"),
            "real_vs_mask_only_by_region": comp.get("real_minus_mask_only_root_score"),
            "template_dependency_score": 0.40,
            "freeze": True,
        }
    if name == "B-hand8":
        hard = summary.get("hard_checks", {})
        return {
            "wrist_connection_score": float(bool(hard.get("left_hand_connected_to_wrist_forearm")) and bool(hard.get("right_hand_connected_to_wrist_forearm"))),
            "finger_structure_score": 0.0,
            "template_dependency_score": 1.0,
            "freeze": True,
        }
    if name == "B-Fus3D0-v2":
        comp = summary.get("comparison", {})
        return {
            "real_vs_shuffle_by_region": comp.get("real_minus_shuffle_query_occupied"),
            "real_vs_zero_by_region": comp.get("real_minus_zero_query_occupied"),
            "template_dependency_score": 0.20,
            "freeze": False,
        }
    return {}


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# V7 Region Failure Autopsy",
        "",
        "Status: `research_only_region_autopsy_strict_gate_red`",
        "",
        "This report decomposes v6 artifacts into hard-gate regions before any v7 model escalation.",
        "",
        "## Strict Truth",
        "",
        "```text",
        "strict_candidate_passes = 0",
        "strict_teacher_passes = 0",
        "formal cloud train/infer/export = blocked",
        "```",
        "",
        "## Best Current Artifact By Hard Gate",
        "",
        "| gate | best source | score | failure reason |",
        "| --- | --- | ---: | --- |",
    ]
    for gate, row in payload["best_sources"].items():
        lines.append(f"| `{gate}` | `{row['source']}` | {row['score']:.4f} | {row['failure_reason']} |")
    lines += ["", "## Per-Line Region Scores", "", "| line | region | score | real-control notes |", "| --- | --- | ---: | --- |"]
    for line, item in payload["lines"].items():
        for region, score in item["regions"].items():
            notes = []
            for key in ("real_vs_shuffle_by_region", "real_vs_zero_by_region", "real_vs_mask_only_by_region"):
                if key in score and score[key] is not None:
                    notes.append(f"{key}={score[key]:.4f}")
            lines.append(f"| `{line}` | `{region}` | {score['region_score']:.4f} | {'; '.join(notes)} |")
    lines += ["", "## Freeze / Keep", ""]
    for item in payload["freeze_list"]:
        lines.append(f"- freeze: {item}")
    for item in payload["keep_list"]:
        lines.append(f"- keep: {item}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    lines: dict[str, Any] = {}
    all_region_best: dict[str, tuple[str, float]] = {region: ("none", 0.0) for region in REGIONS}
    for line_name, cfg in ARTIFACTS.items():
        summary = load_json(cfg["summary"])
        real = read_ascii_ply_vertices(cfg["real"])
        controls = {
            key: read_ascii_ply_vertices(path)
            for key, path in cfg.items()
            if key in {"shuffle", "zero", "mask_only", "random", "random_view", "constrained", "raw"} and isinstance(path, Path)
        }
        line_dir = output_dir / line_name.replace("-", "_").replace(" ", "_")
        image_paths: list[Path] = []
        region_rows: dict[str, Any] = {}
        for region in REGIONS:
            mask = region_mask(real["points"], real["colors"], region)
            sub_points = real["points"][mask]
            sub_colors = real["colors"][mask]
            img_path = line_dir / f"{region}.png"
            projection(sub_points, sub_colors, img_path, width=args.width, height=args.height, title=f"{line_name} {region}")
            image_paths.append(img_path)
            row = score_region(real["points"], real["colors"], region)
            row.update(compare_regions(real, controls, region))
            region_rows[region] = row
            if row["region_score"] > all_region_best[region][1]:
                all_region_best[region] = (line_name, row["region_score"])
        sheet = line_dir / f"{line_name.replace('-', '_')}_region_contact_sheet.png"
        make_sheet(image_paths, sheet, f"{line_name} region contact sheet")
        if cfg.get("contact") and Path(cfg["contact"]).is_file():
            shutil.copy2(Path(cfg["contact"]), line_dir / Path(cfg["contact"]).name)
        extra = derive_line_scores(line_name, summary)
        lines[line_name] = {
            "summary_path": str(cfg["summary"]),
            "real_artifact": str(cfg["real"]),
            "region_contact_sheet": str(sheet),
            "regions": region_rows,
            "genealogy": cfg["genealogy"],
            **extra,
        }

    best_sources: dict[str, Any] = {}
    hard_gate_to_region = {
        "full_body": "full_body",
        "head": "head",
        "face_core": "face_core",
        "hairline": "hairline",
        "head_top": "head_top",
        "left_hand": "left_hand",
        "right_hand": "right_hand",
        "wrist_connection": "hands",
        "finger_structure": "hands",
    }
    for gate, region in hard_gate_to_region.items():
        source, score = all_region_best.get(region, ("none", 0.0))
        reason = "requires region-specific learned geometry and visual review"
        if gate == "finger_structure":
            source = "B-hand8"
            score = 0.0
            reason = "B-hand8 has wrist connection but finger_structure_visible=false and scaffold_only=true"
        if gate in {"hairline", "head_top"} and source == "B-hair1":
            reason = "B-hair1 is geometric but real token loses to zero/mask-only controls"
        best_sources[gate] = {"source": source, "score": float(score), "failure_reason": reason}

    payload = {
        "task": "v7_region_failure_autopsy",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "research_only_region_autopsy_strict_gate_red",
        **STRICT_FACTS,
        "output_dir": str(output_dir),
        "lines": lines,
        "best_sources": best_sources,
        "freeze_list": [
            "B-GS1 current scoring recipe",
            "B-hair1 current root_score / strand-chain evidence recipe",
            "B-hand8 scaffold-only connected hand-arm recipe",
            "A5 COLMAP view/threshold/source-pair/fusion loop",
            "B19/B16/B18/B2 residual shell loop",
        ],
        "keep_list": [
            "B-Fus3D0-v2 latent-grid SDF backend",
            "B-GS1 artifact family and renderer",
            "B-hand8 wrist/forearm connection scaffold as weak anchor",
            "B-hair1 rooted strand primitive as geometry primitive",
            "D-line referee",
        ],
    }
    write_json(output_dir / "v7_region_failure_autopsy_summary.json", payload)
    write_markdown(output_dir / "v7_region_failure_autopsy_report.md", payload)
    write_markdown(args.report.resolve(), payload)
    print(json.dumps({"status": payload["status"], "output_dir": str(output_dir), "lines": len(lines)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
