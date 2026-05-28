from __future__ import annotations

import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


REPO = Path(os.environ.get("VGGT_REPO_ROOT", r"D:\vggt\vggt-canonical-surfel-adapter"))
OUTPUT = REPO / "output"
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
MANIFEST = REPORTS / "V10210000000000000000_training_asset_manifest.csv"
OUT_ROOT = OUTPUT / "V13700000000000000000_anti_billboard_training_matrix"
BASE_MATRIX = OUTPUT / "V10700000000000000000_volume_aware_training_matrix"
WEAK_ROOT = OUTPUT / "V13400000000000000000_billboard_weak_regions"
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"

sys.path.insert(0, str(REPO))
from models.v135_anti_billboard_topology_volume_student import (  # noqa: E402
    AntiBillboardTopologyVolumeConfig,
    AntiBillboardTopologyVolumeStudent,
)
from tools.V13300_anti_billboard_metric_v2 import anti_billboard_metric_v2  # noqa: E402


CONTROL_CONFIGS = [
    "real_vggt_baseline_only",
    "posthoc_surfel_only",
    "same_topology_no_semantic",
    "tiny_synthetic_token_control",
    "shuffled_smpl_feature",
    "baseline_preservation_only",
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


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("eligible_for_training_payload") == "True"]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    text = str(value).replace("\\", "/")
    marker = "vggt-canonical-surfel-adapter/"
    if marker in text:
        rel = text.split(marker, 1)[1]
        mapped = REPO / rel
        if mapped.exists():
            return mapped
    return path


def as_rgb(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype != np.uint8:
        if out.size and np.issubdtype(out.dtype, np.number) and float(np.nanmax(out)) <= 1.5:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out[:, :3]


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    ensure(path.parent)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, as_rgb(colors), strict=False):
            f.write(f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def pca_numpy(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    x = pts - center[None]
    cov = (x.T @ x) / max(1, len(x) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    proj = x @ vecs
    return center, vals, vecs, proj


def build_batch(row: dict[str, str], max_points: int, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, np.ndarray]]:
    cfg = AntiBillboardTopologyVolumeConfig()
    base = load_npz(repo_path(row["baseline_path"]))
    graph = load_npz(repo_path(row["graph_path"]))
    weak = load_npz(WEAK_ROOT / row["case"] / "billboard_weak_regions.npz")
    hp = np.asarray(base["human_points"], dtype=np.float32)
    rgb = as_rgb(base["human_rgb"]).astype(np.float32) / 255.0
    repair = np.asarray(weak["billboard_repair_region_mask"], dtype=bool)
    no_change = np.asarray(weak["no_change_mask"], dtype=bool)
    weak_score = np.asarray(graph["mentor_weak_region_score"], dtype=np.float32)
    conf = np.asarray(graph["mentor_smpl_confidence"], dtype=np.float32)
    body = np.asarray(graph["geometry_body_part_id"], dtype=np.int64)
    priority = repair.astype(np.float32) * 5.0 + weak_score + conf * 0.03
    idx = np.argsort(-priority)[: min(max_points, len(priority))]
    smpl = np.zeros((len(idx), cfg.smpl_feature_dim), dtype=np.float32)
    smpl[:, : cfg.part_count] = np.eye(cfg.part_count, dtype=np.float32)[np.clip(body[idx], 0, cfg.part_count - 1)]
    smpl[:, cfg.part_count] = weak_score[idx]
    smpl[:, cfg.part_count + 1] = conf[idx]
    smpl[:, cfg.part_count + 2] = repair[idx].astype(np.float32)
    batch = {
        "anchor_xyz": torch.from_numpy(hp[idx][None]).to(device),
        "anchor_rgb": torch.from_numpy(rgb[idx][None]).to(device),
        "confidence": torch.from_numpy(conf[idx][None]).to(device),
        "weak_region": torch.from_numpy(np.maximum(repair[idx].astype(np.float32), weak_score[idx] * 0.20)[None]).to(device),
        "billboard_region": torch.from_numpy(repair[idx].astype(np.float32)[None]).to(device),
        "anchor_features": torch.zeros(1, len(idx), cfg.anchor_feature_dim, device=device),
        "smpl_features": torch.from_numpy(smpl[None]).to(device),
        "vggt_token_context": torch.zeros(1, cfg.token_dim, device=device),
        "body_part_id": torch.from_numpy(body[idx][None]).to(device),
    }
    center = hp[idx].mean(axis=0, keepdims=True)
    centered = hp[idx] - center
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1].astype(np.float32)
    tangent = vh[1].astype(np.float32)
    repair_f = repair[idx].astype(np.float32)
    score_f = np.clip(weak_score[idx], 0, 1)
    sign = np.sign(centered @ normal).astype(np.float32)
    sign[sign == 0] = 1.0
    side_sign = np.sign((body[idx] % 5) - 2).astype(np.float32)
    side_sign[side_sign == 0] = 1.0
    gate = np.maximum(repair_f, score_f * 0.30)
    target = hp[idx] + normal[None] * sign[:, None] * (0.015 + 0.035 * gate[:, None])
    target += tangent[None] * side_sign[:, None] * (0.010 + 0.018 * repair_f[:, None])
    targets = {
        "target_xyz": torch.from_numpy(target[None].astype(np.float32)).to(device),
        "weak": batch["weak_region"].unsqueeze(-1),
        "billboard": batch["billboard_region"].unsqueeze(-1),
        "no_change": torch.from_numpy(no_change[idx][None]).to(device),
    }
    aux = {
        "idx": idx,
        "human_points": hp,
        "human_rgb": as_rgb(base["human_rgb"]),
        "environment_points": np.asarray(base["environment_points"], dtype=np.float32),
        "environment_rgb": as_rgb(base["environment_rgb"]),
        "no_change": no_change,
        "body_part_id": body,
    }
    return batch, targets, aux


def topology_volume_losses(out: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    weak = targets["weak"].float()
    billboard = targets["billboard"].float()
    no_change = targets["no_change"].bool()
    residual = out["residual_xyz"]
    loss_preserve = residual[no_change].square().mean() if bool(no_change.any()) else residual.square().mean() * 0.0
    loss_repair = ((out["student_xyz"] - targets["target_xyz"]) ** 2 * weak).mean()
    shell = out["shell_offsets"]
    shell_sep = F.relu(0.018 - shell[..., 0:2].mean()) + F.relu(0.014 - shell[..., 2:4].mean())
    occ_target = billboard.expand_as(out["cross_section_occupancy"])
    loss_cross = F.binary_cross_entropy(out["cross_section_occupancy"], occ_target.clamp(0, 1))
    loss_occ = F.binary_cross_entropy(out["occupancy"], billboard.clamp(0, 1))
    loss_part = F.binary_cross_entropy_with_logits(out["part_continuity_logits"], torch.ones_like(out["part_continuity_logits"]) * 0.65)
    # Encourage non-zero weak-region spread without using teacher/Kinect points.
    pts = out["student_xyz"][0]
    centered = pts - pts.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(1, pts.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov)
    loss_volume = F.relu(0.024 - eig[0] / eig[-1].clamp_min(1e-8))
    total = loss_repair + 1.6 * loss_preserve + 0.24 * loss_cross + 0.12 * loss_occ + 0.18 * shell_sep + 0.08 * loss_part + 0.45 * loss_volume
    return {
        "total": total,
        "baseline_preservation_loss": loss_preserve,
        "weak_region_repair_loss": loss_repair,
        "cross_section_occupancy_loss": loss_cross,
        "shell_separation_loss": shell_sep,
        "part_continuity_loss": loss_part,
        "topology_volume_loss": loss_volume,
    }


def train_case(row: dict[str, str], steps: int, max_points: int, device: torch.device) -> dict[str, Any]:
    model = AntiBillboardTopologyVolumeStudent().to(device)
    batch, targets, aux = build_batch(row, max_points=max_points, device=device)
    optim = torch.optim.AdamW(model.parameters(), lr=1.4e-3, weight_decay=1e-4)
    history: list[dict[str, Any]] = []
    for step in range(steps):
        optim.zero_grad(set_to_none=True)
        out = model(batch)
        losses = topology_volume_losses(out, targets)
        losses["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if step in {0, steps - 1} or (step + 1) % 50 == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(losses["total"].detach().cpu()),
                    "cross_section_loss": float(losses["cross_section_occupancy_loss"].detach().cpu()),
                    "topology_volume_loss": float(losses["topology_volume_loss"].detach().cpu()),
                    "grad_norm": float(grad_norm),
                }
            )
    with torch.no_grad():
        out = model(batch)
    human = aux["human_points"].copy()
    idx = aux["idx"]
    pred = out["student_xyz"][0].detach().cpu().numpy().astype(np.float32)
    replace = ~aux["no_change"][idx]
    human[idx[replace]] = pred[replace]
    rgb = aux["human_rgb"]
    full = np.concatenate([human, aux["environment_points"]], axis=0)
    full_rgb = np.concatenate([rgb, aux["environment_rgb"]], axis=0)
    out_dir = ensure(OUT_ROOT / row["case"] / "anti_billboard_topology_volume_true")
    np.savez_compressed(
        out_dir / "predictions.npz",
        human_points=human,
        human_rgb=rgb,
        environment_points=aux["environment_points"],
        environment_rgb=aux["environment_rgb"],
        full_scene_points=full,
        full_scene_rgb=full_rgb,
        body_part_id=aux["body_part_id"],
        model_owned_student_output=np.array(True),
        teacher_points_used_at_inference=np.array(False),
        raw_kinect_depth_used_at_inference=np.array(False),
        facial_detail_target_applicable=np.array(False),
        face_detail_claim_allowed=np.array(False),
        allowed_face_claim=np.array(ALLOWED_FACE_CLAIM),
        config=np.array("anti_billboard_topology_volume_true"),
        case_id=np.array(row["case"]),
    )
    write_ply(out_dir / "full_scene_rgb_pointcloud.ply", full, full_rgb)
    metrics = anti_billboard_metric_v2(human, aux["body_part_id"])
    return {
        "case": row["case"],
        "config": "anti_billboard_topology_volume_true",
        "prediction": str(out_dir / "predictions.npz"),
        "ply": str(out_dir / "full_scene_rgb_pointcloud.ply"),
        "steps": steps,
        "max_points": max_points,
        "device": str(device),
        "history_json": json.dumps(history),
        "model_owned_student_output": True,
        "teacher_points_used_at_inference": False,
        "raw_kinect_depth_used_at_inference": False,
        "facial_detail_target_applicable": False,
        "face_detail_claim_allowed": False,
        **metrics,
    }


def load_config(case: str, config: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if config == "anti_billboard_topology_volume_true":
        path = OUT_ROOT / case / config / "predictions.npz"
    else:
        path = BASE_MATRIX / case / config / "predictions.npz"
    pred = load_npz(path)
    body = np.asarray(pred["body_part_id"]) if "body_part_id" in pred else None
    return np.asarray(pred["human_points"], dtype=np.float32), as_rgb(pred["human_rgb"]), body


def rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return rz @ rx


def render_panel(points: np.ndarray, colors: np.ndarray, title: str, rot: np.ndarray) -> Image.Image:
    size = (390, 282)
    im = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(im)
    pts = (points - points.mean(axis=0, keepdims=True)) @ rot.T
    lo = np.percentile(pts[:, :2], 1, axis=0)
    hi = np.percentile(pts[:, :2], 99, axis=0)
    pad = (hi - lo) * 0.17 + 1e-6
    lo -= pad
    hi += pad
    q = (pts[:, :2] - lo[None]) / (hi[None] - lo[None] + 1e-9)
    q[:, 1] = 1.0 - q[:, 1]
    xy = np.clip(q * np.array([size[0] - 48, size[1] - 68]) + np.array([24, 44]), 0, [size[0] - 1, size[1] - 1]).astype(int)
    depth = pts[:, 2]
    d = (depth - np.quantile(depth, 0.03)) / max(np.quantile(depth, 0.97) - np.quantile(depth, 0.03), 1e-9)
    d = np.clip(d, 0.0, 1.0)
    rgb = np.clip(colors.astype(np.float32) * (0.63 + 0.44 * d[:, None]), 0, 255).astype(np.uint8)
    order = np.argsort(depth)
    step = max(1, len(order) // 52000)
    for i in order[::step]:
        x, y = xy[i]
        c = tuple(rgb[i].tolist())
        im.putpixel((int(x), int(y)), c)
        if 1 <= x < size[0] - 1 and 1 <= y < size[1] - 1:
            im.putpixel((int(x + 1), int(y)), c)
            im.putpixel((int(x), int(y + 1)), c)
    draw.text((8, 8), title, fill=(10, 10, 10))
    return im


def cross_panel(points: np.ndarray, title: str) -> Image.Image:
    size = (390, 282)
    im = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(im)
    _center, _vals, _axes, proj = pca_numpy(points)
    xy_src = proj[:, [1, 2]]
    lo = np.percentile(xy_src, 1, axis=0)
    hi = np.percentile(xy_src, 99, axis=0)
    pad = (hi - lo) * np.array([0.18, 0.55]) + 1e-6
    lo -= pad
    hi += pad
    q = (xy_src - lo[None]) / (hi[None] - lo[None] + 1e-9)
    q[:, 1] = 1.0 - q[:, 1]
    xy = np.clip(q * np.array([size[0] - 48, size[1] - 68]) + np.array([24, 44]), 0, [size[0] - 1, size[1] - 1]).astype(int)
    order = np.argsort(proj[:, 0])
    step = max(1, len(order) // 52000)
    for i in order[::step]:
        x, y = xy[i]
        im.putpixel((int(x), int(y)), (46, 71, 58))
        if 1 <= x < size[0] - 1 and 1 <= y < size[1] - 1:
            im.putpixel((int(x + 1), int(y)), (81, 106, 89))
    draw.text((8, 8), title, fill=(10, 10, 10))
    draw.text((8, size[1] - 23), "mid vs thin axis", fill=(35, 35, 35))
    return im


def compose(panels: list[Image.Image], cols: int, path: Path) -> None:
    ensure(path.parent)
    w, h = panels[0].size
    canvas = Image.new("RGB", (cols * w, int(math.ceil(len(panels) / cols)) * h), (255, 255, 255))
    for i, panel in enumerate(panels):
        canvas.paste(panel, ((i % cols) * w, (i // cols) * h))
    canvas.save(path)


def compare_and_render(rows: list[dict[str, str]], created_at: str) -> None:
    metric_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    configs = ["anti_billboard_topology_volume_true", *CONTROL_CONFIGS]
    for row in rows:
        case = row["case"]
        case_metrics: dict[str, dict[str, Any]] = {}
        for config in configs:
            cfg_path = (OUT_ROOT / case / config / "predictions.npz") if config == "anti_billboard_topology_volume_true" else (BASE_MATRIX / case / config / "predictions.npz")
            if not cfg_path.exists():
                metric_rows.append({"case": case, "config": config, "missing": True})
                continue
            pts, _rgb, body = load_config(case, config)
            metrics = anti_billboard_metric_v2(pts, body)
            case_metrics[config] = metrics
            metric_rows.append({"case": case, "config": config, "missing": False, **metrics})
        true_metrics = case_metrics.get("anti_billboard_topology_volume_true")
        if not true_metrics:
            failures.append({"case": case, "reason": "missing_true_prediction"})
            continue
        true_score = float(true_metrics["anti_billboard_score_v2"])
        if bool(true_metrics["billboard_fail_v2"]):
            failures.append({"case": case, "reason": "true_billboard_fail_v2", "true_score": true_score})
        for control, metrics in case_metrics.items():
            if control == "anti_billboard_topology_volume_true":
                continue
            control_score = float(metrics["anti_billboard_score_v2"])
            if control_score >= true_score * 0.96:
                failures.append({"case": case, "reason": "control_close_or_better_v2", "control": control, "true_score": true_score, "control_score": control_score})
    write_csv(REPORTS / "V13700000000000000000_seed_metrics.csv", metric_rows)
    first_case = rows[0]["case"]
    board_configs = ["anti_billboard_topology_volume_true", "real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control", "tiny_synthetic_token_control"]
    rot = rotation_matrix(-30, 61)
    panels: list[Image.Image] = []
    cross: list[Image.Image] = []
    for config in board_configs:
        pts, rgb, _body = load_config(first_case, config)
        panels.append(render_panel(pts, rgb, config.replace("_", " "), rot))
        cross.append(cross_panel(pts, config.replace("_", " ")))
    compose(panels, 3, BOARDS / "V13700000000000000000_training_matrix_turntable.png")
    compose(cross, 3, BOARDS / "V13700000000000000000_training_matrix_cross_section.png")
    write_json(
        REPORTS / "V13700000000000000000_training_decision.json",
        {
            "created_at": created_at,
            "status": "V13700_LOCAL_OR_MODAL_TRAINING_FAIL_CLOSED_CONTINUE" if failures else "V13700_TRAINING_PRECHECK_PASS_REQUIRES_FULL_VISUAL_GATE",
            "mentor_ready": False,
            "external_hard_block": False,
            "failures": failures,
            "face_detail_claim_allowed": False,
            "allowed_face_claim": ALLOWED_FACE_CLAIM,
            "boards": {
                "turntable": str(BOARDS / "V13700000000000000000_training_matrix_turntable.png"),
                "cross_section": str(BOARDS / "V13700000000000000000_training_matrix_cross_section.png"),
            },
            "note": "Training matrix evidence is still subordinate to V138/V140 mentor visual and causality gates.",
        },
    )


def select_device() -> tuple[torch.device, dict[str, Any]]:
    info: dict[str, Any] = {
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "selected_device": "cpu",
        "cuda_error": "",
        "modal_required_for_final": True,
    }
    if torch.cuda.is_available():
        try:
            probe = torch.zeros(8, device="cuda")
            _ = (probe + 1).sum().item()
            info["selected_device"] = "cuda"
            info["cuda_device_name"] = torch.cuda.get_device_name()
            return torch.device("cuda"), info
        except Exception as exc:
            info["cuda_error"] = str(exc)
    return torch.device("cpu"), info


def main() -> int:
    created_at = now()
    rows = read_manifest()
    device, device_info = select_device()
    steps = int(os.environ.get("V13700_STEPS", "35"))
    max_points = int(os.environ.get("V13700_MAX_POINTS", "4096"))
    manifest_rows = [train_case(row, steps=steps, max_points=max_points, device=device) for row in rows]
    write_csv(REPORTS / "V13700000000000000000_training_manifest.csv", manifest_rows)
    compare_and_render(rows, created_at)
    write_json(
        REPORTS / "V13700000000000000000_runtime_environment.json",
        {
            "created_at": created_at,
            **device_info,
            "steps": steps,
            "max_points": max_points,
            "note": "Local CPU smoke is not final. Modal A10/A100 long training remains required before mentor-ready claims.",
        },
    )
    failed_jobs = [] if manifest_rows else [{"reason": "no_training_rows"}]
    write_json(REPORTS / "V13700000000000000000_failed_jobs.json", {"created_at": created_at, "failed_jobs": failed_jobs})
    print(json.dumps({"created_at": created_at, "status": "V13700_DONE", "device": str(device), "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
