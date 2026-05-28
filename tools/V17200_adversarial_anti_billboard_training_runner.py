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
TARGET_ROOT = OUTPUT / "V17100000000000000000_part_graph_cross_section_targets"
OUT_ROOT = OUTPUT / "V17200000000000000000_adversarial_anti_billboard_training"
BASE_MATRIX = OUTPUT / "V10700000000000000000_volume_aware_training_matrix"
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"

sys.path.insert(0, str(REPO))
from models.v135_anti_billboard_topology_volume_student import (  # noqa: E402
    AntiBillboardTopologyVolumeConfig,
    AntiBillboardTopologyVolumeStudent,
)
from tools.V13300_anti_billboard_metric_v2 import anti_billboard_metric_v2  # noqa: E402


CONTROL_CONFIGS = [
    "real_vggt_baseline_only",
    "same_topology_no_semantic",
    "shuffled_smpl_feature",
    "thickness_only_control",
    "posthoc_surfel_only",
    "tiny_synthetic_token_control",
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


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    text = str(value).replace("\\", "/")
    marker = "vggt-canonical-surfel-adapter/"
    if marker in text:
        mapped = REPO / text.split(marker, 1)[1]
        if mapped.exists():
            return mapped
    return path


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


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
    target = load_npz(TARGET_ROOT / row["case"] / "part_graph_cross_section_targets.npz")
    pts = np.asarray(base["human_points"], dtype=np.float32)
    rgb = as_rgb(base["human_rgb"]).astype(np.float32) / 255.0
    repair = np.asarray(target["billboard_repair_region_mask"], dtype=bool)
    no_change = np.asarray(target["no_change_mask"], dtype=bool)
    weak_score = np.asarray(graph["mentor_weak_region_score"], dtype=np.float32)
    conf = np.asarray(graph["mentor_smpl_confidence"], dtype=np.float32)
    body = np.asarray(target["body_part_id"], dtype=np.int64)
    shell_conf = np.asarray(target["shell_confidence"], dtype=np.float32)
    priority = repair.astype(np.float32) * 5.5 + shell_conf * 1.0 + weak_score * 0.4 + conf * 0.03
    idx = np.argsort(-priority)[: min(max_points, len(priority))]
    smpl = np.zeros((len(idx), cfg.smpl_feature_dim), dtype=np.float32)
    smpl[:, : cfg.part_count] = np.eye(cfg.part_count, dtype=np.float32)[np.clip(body[idx], 0, cfg.part_count - 1)]
    smpl[:, cfg.part_count] = weak_score[idx]
    smpl[:, cfg.part_count + 1] = conf[idx]
    smpl[:, cfg.part_count + 2] = repair[idx].astype(np.float32)
    batch = {
        "anchor_xyz": torch.from_numpy(pts[idx][None]).to(device),
        "anchor_rgb": torch.from_numpy(rgb[idx][None]).to(device),
        "confidence": torch.from_numpy(conf[idx][None]).to(device),
        "weak_region": torch.from_numpy(np.maximum(repair[idx].astype(np.float32), weak_score[idx] * 0.20)[None]).to(device),
        "billboard_region": torch.from_numpy(repair[idx].astype(np.float32)[None]).to(device),
        "anchor_features": torch.zeros(1, len(idx), cfg.anchor_feature_dim, device=device),
        "smpl_features": torch.from_numpy(smpl[None]).to(device),
        "vggt_token_context": torch.zeros(1, cfg.token_dim, device=device),
        "body_part_id": torch.from_numpy(body[idx][None]).to(device),
    }
    targets = {
        "front_shell_target": torch.from_numpy(np.asarray(target["front_shell_target"], dtype=np.float32)[idx][None]).to(device),
        "back_shell_target": torch.from_numpy(np.asarray(target["back_shell_target"], dtype=np.float32)[idx][None]).to(device),
        "side_shell_target": torch.from_numpy(np.asarray(target["side_shell_target"], dtype=np.float32)[idx][None]).to(device),
        "cross_section_occupancy_target": torch.from_numpy(np.asarray(target["cross_section_occupancy_target"], dtype=np.float32)[idx][None]).to(device),
        "part_continuity_target": torch.from_numpy(np.asarray(target["part_continuity_target"], dtype=np.float32)[idx][None]).to(device),
        "weak": batch["weak_region"].unsqueeze(-1),
        "billboard": batch["billboard_region"].unsqueeze(-1),
        "no_change": torch.from_numpy(no_change[idx][None]).to(device),
    }
    aux = {
        "idx": idx,
        "human_points": pts,
        "human_rgb": as_rgb(base["human_rgb"]),
        "environment_points": np.asarray(base["environment_points"], dtype=np.float32),
        "environment_rgb": as_rgb(base["environment_rgb"]),
        "no_change": no_change,
        "body_part_id": body,
        "repair": repair,
    }
    return batch, targets, aux


def adversarial_control_stats(case: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for config in CONTROL_CONFIGS:
        path = BASE_MATRIX / case / config / "predictions.npz"
        if not path.exists():
            continue
        pred = load_npz(path)
        body = np.asarray(pred["body_part_id"]) if "body_part_id" in pred else None
        values[f"{config}_score_v2"] = float(anti_billboard_metric_v2(np.asarray(pred["human_points"], dtype=np.float32), body)["anti_billboard_score_v2"])
    return values


def losses(out: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], case_adv: dict[str, float]) -> dict[str, torch.Tensor]:
    weak = targets["weak"].float()
    no_change = targets["no_change"].bool()
    residual = out["residual_xyz"]
    preserve = residual[no_change].square().mean() if bool(no_change.any()) else residual.square().mean() * 0.0
    front = ((out["front_shell"] - targets["front_shell_target"]) ** 2 * weak).mean()
    back = ((out["back_shell"] - targets["back_shell_target"]) ** 2 * weak).mean()
    side = ((out["side_shell"] - targets["side_shell_target"]) ** 2 * weak).mean()
    cross = F.binary_cross_entropy(out["cross_section_occupancy"], targets["cross_section_occupancy_target"].clamp(0, 1))
    part = F.binary_cross_entropy_with_logits(out["part_continuity_logits"], targets["part_continuity_target"].clamp(0, 1))
    shell_width = (out["front_shell"] - out["back_shell"]).norm(dim=-1).mean() + (out["left_shell"] - out["right_shell"]).norm(dim=-1).mean()
    shell_width_loss = F.relu(0.055 - shell_width)
    pts = out["student_xyz"][0]
    centered = pts - pts.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(1, pts.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov)
    volume_loss = F.relu(0.030 - eig[0] / eig[-1].clamp_min(1e-8))
    # Differentiable proxy: if hard controls are strong, demand a larger shell
    # margin. This is not the final metric, only a training pressure.
    strongest_control = max(case_adv.values()) if case_adv else 0.0
    adversarial_margin = torch.relu(torch.tensor(strongest_control + 0.08, device=pts.device) - shell_width * 5.0)
    total = (
        front
        + back
        + side
        + 1.8 * preserve
        + 0.35 * cross
        + 0.16 * part
        + 0.22 * shell_width_loss
        + 0.45 * volume_loss
        + 0.10 * adversarial_margin
    )
    return {
        "total": total,
        "preserve": preserve,
        "front": front,
        "back": back,
        "side": side,
        "cross": cross,
        "part": part,
        "shell_width": shell_width_loss,
        "volume": volume_loss,
        "adversarial_margin": adversarial_margin,
    }


def train_case(row: dict[str, str], steps: int, max_points: int, device: torch.device) -> dict[str, Any]:
    model = AntiBillboardTopologyVolumeStudent().to(device)
    batch, targets, aux = build_batch(row, max_points=max_points, device=device)
    adv = adversarial_control_stats(row["case"])
    optim = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1e-4)
    history = []
    for step in range(steps):
        optim.zero_grad(set_to_none=True)
        out = model(batch)
        ls = losses(out, targets, adv)
        ls["total"].backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if step in {0, steps - 1} or (step + 1) % 50 == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(ls["total"].detach().cpu()),
                    "cross": float(ls["cross"].detach().cpu()),
                    "adversarial_margin": float(ls["adversarial_margin"].detach().cpu()),
                    "volume": float(ls["volume"].detach().cpu()),
                    "grad_norm": float(grad),
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
    out_dir = ensure(OUT_ROOT / row["case"] / "adversarial_anti_billboard_true")
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
        config=np.array("adversarial_anti_billboard_true"),
        case_id=np.array(row["case"]),
    )
    write_ply(out_dir / "full_scene_rgb_pointcloud.ply", full, full_rgb)
    metric = anti_billboard_metric_v2(human, aux["body_part_id"])
    return {
        "case": row["case"],
        "config": "adversarial_anti_billboard_true",
        "prediction": str(out_dir / "predictions.npz"),
        "ply": str(out_dir / "full_scene_rgb_pointcloud.ply"),
        "steps": steps,
        "max_points": max_points,
        "device": str(device),
        "history_json": json.dumps(history),
        **{f"adv_{k}": v for k, v in adv.items()},
        "model_owned_student_output": True,
        "teacher_points_used_at_inference": False,
        "raw_kinect_depth_used_at_inference": False,
        "facial_detail_target_applicable": False,
        "face_detail_claim_allowed": False,
        **metric,
    }


def load_config(case: str, config: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if config == "adversarial_anti_billboard_true":
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


def render_panel(points: np.ndarray, colors: np.ndarray, title: str) -> Image.Image:
    size = (390, 282)
    im = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(im)
    pts = (points - points.mean(axis=0, keepdims=True)) @ rotation_matrix(-30, 61).T
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
    d = np.clip(d, 0, 1)
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


def compose(panels: list[Image.Image], cols: int, path: Path) -> None:
    ensure(path.parent)
    w, h = panels[0].size
    canvas = Image.new("RGB", (cols * w, int(math.ceil(len(panels) / cols)) * h), (255, 255, 255))
    for i, panel in enumerate(panels):
        canvas.paste(panel, ((i % cols) * w, (i // cols) * h))
    canvas.save(path)


def compare(rows: list[dict[str, str]], created_at: str) -> None:
    metric_rows = []
    failures = []
    configs = ["adversarial_anti_billboard_true", *CONTROL_CONFIGS]
    for row in rows:
        case = row["case"]
        case_metrics: dict[str, dict[str, Any]] = {}
        for config in configs:
            pts, rgb, body = load_config(case, config)
            met = anti_billboard_metric_v2(pts, body)
            case_metrics[config] = met
            metric_rows.append({"case": case, "config": config, **met})
        true = case_metrics["adversarial_anti_billboard_true"]
        true_score = float(true["anti_billboard_score_v2"])
        if bool(true["billboard_fail_v2"]):
            failures.append({"case": case, "reason": "true_billboard_fail_v2", "true_score": true_score})
        for config in CONTROL_CONFIGS:
            c_score = float(case_metrics[config]["anti_billboard_score_v2"])
            if c_score >= true_score * 0.96:
                failures.append({"case": case, "reason": "control_close_or_better_v2", "control": config, "true_score": true_score, "control_score": c_score})
    write_csv(REPORTS / "V17200000000000000000_seed_metrics.csv", metric_rows)
    first = rows[0]["case"]
    panels = []
    for config in ["adversarial_anti_billboard_true", "real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control", "tiny_synthetic_token_control"]:
        pts, rgb, _body = load_config(first, config)
        panels.append(render_panel(pts, rgb, f"{first} {config.replace('_', ' ')}"))
    compose(panels, 3, BOARDS / "V17200000000000000000_adversarial_training_board.png")
    write_json(
        REPORTS / "V17200000000000000000_training_decision.json",
        {
            "created_at": created_at,
            "status": "V17200_ADVERSARIAL_TRAINING_FAIL_CLOSED_CONTINUE" if failures else "V17200_ADVERSARIAL_TRAINING_PRECHECK_PASS_REQUIRES_VISUAL_GATE",
            "mentor_ready": False,
            "external_hard_block": False,
            "failures": failures,
            "board": str(BOARDS / "V17200000000000000000_adversarial_training_board.png"),
            "face_detail_claim_allowed": False,
            "allowed_face_claim": ALLOWED_FACE_CLAIM,
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
    steps = int(os.environ.get("V17200_STEPS", "30"))
    max_points = int(os.environ.get("V17200_MAX_POINTS", "4096"))
    manifest = [train_case(row, steps=steps, max_points=max_points, device=device) for row in rows]
    write_csv(REPORTS / "V17200000000000000000_training_manifest.csv", manifest)
    compare(rows, created_at)
    write_json(REPORTS / "V17200000000000000000_runtime_environment.json", {"created_at": created_at, **device_info, "steps": steps, "max_points": max_points})
    print(json.dumps({"created_at": created_at, "status": "V17200_DONE", "device": str(device), "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
