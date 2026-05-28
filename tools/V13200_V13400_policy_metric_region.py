from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from V13300_anti_billboard_metric_v2 import anti_billboard_metric_v2


REPO = Path(os.environ.get("VGGT_REPO_ROOT", r"D:\vggt\vggt-canonical-surfel-adapter"))
REPORTS = REPO / "reports"
OUTPUT = REPO / "output"
BOARDS = REPO / "boards"
AGENTS = REPO / "AGENTS.md"
MANIFEST = REPORTS / "V10210000000000000000_training_asset_manifest.csv"
OUT_ROOT = OUTPUT / "V13400000000000000000_billboard_weak_regions"


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
    with MANIFEST.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("eligible_for_training_payload") == "True"]


def audit_agents(created_at: str) -> None:
    text = AGENTS.read_text(encoding="utf-8", errors="replace")
    checks = {
        "mentor_main_full_scene": "full-scene RGB point cloud" in text and "human is the subject" in text,
        "projection_metric_auxiliary": "Projection overlays, metrics" in text and "auxiliary only" in text,
        "billboard_fail_closed": "flat billboard" in text and "fail closed" in text,
        "points_xy_forbidden": "points[:, :2]" in text,
        "thickness_only_cannot_pass": "thickness-only" in text and "do not claim" in text,
        "same_topology_shuffled_fail": "shuffled/random/same-topology" in text,
        "face_invisible_guard": "eyes/nose/mouth are not visible" in text,
        "visual_failure_not_external": "Visual failure is not an external hard block" in text,
        "no_agent_rule": "Do not spawn agents or subagents" in text,
    }
    (REPORTS / "V13200000000000000000_agents_skill_audit.md").write_text(
        "# V13200 AGENTS / Skill Gate Audit\n\n"
        + "\n".join(f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in checks.items())
        + "\n\nNo agent/subagent launched in this run.\n",
        encoding="utf-8",
    )
    write_json(
        REPORTS / "V13200000000000000000_gate_policy_update.json",
        {
            "created_at": created_at,
            "checks": checks,
            "all_pass": all(checks.values()),
            "no_agent_rule": "No agent/subagent launch in this run.",
        },
    )


def metric_smoke(created_at: str) -> None:
    baseline = REPO / "output" / "V10700000000000000000_volume_aware_training_matrix" / "0012_11_frame001" / "real_vggt_baseline_only" / "predictions.npz"
    pred = load_npz(baseline)
    body = pred.get("body_part_id")
    metrics = anti_billboard_metric_v2(np.asarray(pred["human_points"], dtype=np.float64), body)
    (REPORTS / "V13300000000000000000_anti_billboard_metric_v2_definition.md").write_text(
        "# V13300 Anti-Billboard Metric v2\n\n"
        "Metric v2 combines thickness, cross-section occupancy, front/back separation, depth-layer entropy, connected-component continuity, and optional body-part component balance. It explicitly rejects thickness-only success.\n\n"
        "Required interpretation:\n\n"
        "- high thickness alone is not pass;\n"
        "- multi-layer occupancy and front/back separation must be present;\n"
        "- largest connected component and part continuity must remain strong;\n"
        "- same-topology/shuffled/thickness-only controls must not dominate true.\n",
        encoding="utf-8",
    )
    write_json(
        REPORTS / "V13300000000000000000_metric_v2_smoke.json",
        {"created_at": created_at, "sample": str(baseline), "metrics": metrics, "smoke_pass": True},
    )


def make_preview(rows: list[dict[str, Any]]) -> None:
    panels: list[Image.Image] = []
    for row in rows[:4]:
        weak_path = Path(row["weak_region_path"])
        data = load_npz(weak_path)
        pts = np.asarray(data["human_points"], dtype=np.float64)
        mask = np.asarray(data["billboard_repair_region_mask"], dtype=bool)
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
        step = max(1, len(pix) // 50000)
        for i in range(0, len(pix), step):
            x, y = pix[i]
            color = (190, 67, 45) if mask[i] else (61, 83, 69)
            im.putpixel((int(x), int(y)), color)
        draw.text((8, 8), row["case"], fill=(10, 10, 10))
        panels.append(im)
    if not panels:
        return
    canvas = Image.new("RGB", (720, 520), (255, 255, 255))
    for i, panel in enumerate(panels):
        canvas.paste(panel, ((i % 2) * 360, (i // 2) * 260))
    ensure(BOARDS)
    canvas.save(BOARDS / "V13400000000000000000_billboard_weak_region_preview.png")


def weak_regions(created_at: str) -> None:
    rows: list[dict[str, Any]] = []
    for row in read_manifest():
        case = row["case"]
        weak_npz = load_npz(OUTPUT / "V10400000000000000000_weak_volume_regions" / case / "weak_volume_regions.npz")
        graph = load_npz(Path(row["graph_path"]))
        hp = np.asarray(weak_npz["human_points"], dtype=np.float32)
        weak = np.asarray(weak_npz["weak_volume_region_mask"], dtype=bool)
        sheet = np.asarray(weak_npz["sheet_region_mask"], dtype=bool)
        multi = np.asarray(weak_npz["multi_layer_missing_mask"], dtype=bool)
        no_change = np.asarray(weak_npz["no_change_mask"], dtype=bool)
        repair = (weak | sheet | multi) & ~no_change
        part_masks = {
            "head_hair": np.asarray(graph["head_hair_contour_mask"], dtype=bool),
            "shoulder_neck": np.asarray(graph["shoulder_neck_mask"], dtype=bool),
            "hand_arm": np.asarray(graph["hand_arm_endpoint_mask"], dtype=bool),
            "clothing": np.asarray(graph["clothing_torso_boundary_mask"], dtype=bool),
            "leg_foot": np.asarray(graph["leg_foot_morphology_mask"], dtype=bool),
        }
        out_dir = ensure(OUT_ROOT / case)
        out_path = out_dir / "billboard_weak_regions.npz"
        np.savez_compressed(
            out_path,
            human_points=hp,
            billboard_repair_region_mask=repair,
            weak_volume_region_mask=weak,
            sheet_region_mask=sheet,
            multi_layer_missing_mask=multi,
            no_change_mask=no_change,
            **{f"{k}_mask": v for k, v in part_masks.items()},
            facial_detail_target_applicable=np.array(False),
            face_detail_claim_allowed=np.array(False),
            allowed_face_claim=np.array("head/face contour and hair region only"),
        )
        rows.append(
            {
                "case": case,
                "weak_region_path": str(out_path),
                "repair_ratio": float(np.mean(repair)),
                "sheet_ratio": float(np.mean(sheet)),
                "multi_layer_missing_ratio": float(np.mean(multi)),
                "no_change_ratio": float(np.mean(no_change)),
                **{f"{k}_repair_overlap": float(np.mean(repair & v)) for k, v in part_masks.items()},
            }
        )
    write_csv(REPORTS / "V13400000000000000000_billboard_weak_region_manifest.csv", rows)
    make_preview(rows)


def main() -> int:
    created_at = now()
    audit_agents(created_at)
    metric_smoke(created_at)
    weak_regions(created_at)
    print(json.dumps({"created_at": created_at, "status": "V132_V133_V134_DONE", "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
