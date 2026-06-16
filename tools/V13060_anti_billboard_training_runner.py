from __future__ import annotations

import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO = Path(r"D:\vggt\vggt-canonical-surfel-adapter")
OUTPUT = REPO / "output"
REPORTS = REPO / "reports"
BOARDS = REPO / "boards"
MANIFEST = REPORTS / "V10210000000000000000_training_asset_manifest.csv"
OUT_ROOT = OUTPUT / "V13060000000000000000_anti_billboard_training"
MATRIX_ROOT = OUTPUT / "V10700000000000000000_volume_aware_training_matrix"
ALLOWED_FACE_CLAIM = "head/face contour and hair region only"

sys.path.insert(0, str(REPO))
from models.v105_volume_aware_visible_morphology_student import (  # noqa: E402
    VolumeAwareVisibleMorphologyConfig,
    VolumeAwareVisibleMorphologyStudent,
)


CONTROL_CONFIGS = [
    "real_vggt_baseline_only",
    "posthoc_surfel_only",
    "same_topology_no_semantic",
    "tiny_synthetic_token_control",
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


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("eligible_for_training_payload") == "True"]


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


def anti_billboard_metrics(points: np.ndarray) -> dict[str, float | int | bool]:
    _center, vals, axes, proj = pca_numpy(points)
    ranges = np.ptp(proj, axis=0)
    long_range = max(float(ranges[0]), 1e-9)
    mid_range = max(float(ranges[1]), 1e-9)
    thin_range = max(float(ranges[2]), 1e-9)
    bins_a = np.clip(np.floor((proj[:, 0] - proj[:, 0].min()) / long_range * 16).astype(int), 0, 15)
    bins_b = np.clip(np.floor((proj[:, 1] - proj[:, 1].min()) / mid_range * 12).astype(int), 0, 11)
    bins_t = np.clip(np.floor((proj[:, 2] - proj[:, 2].min()) / thin_range * 8).astype(int), 0, 7)
    sections = 0
    multi = 0
    dense = 0
    layers: list[int] = []
    for a in range(16):
        for b in range(12):
            m = (bins_a == a) & (bins_b == b)
            if int(m.sum()) < 12:
                continue
            sections += 1
            occupied = len(np.unique(bins_t[m]))
            layers.append(occupied)
            multi += int(occupied >= 3)
            dense += int(occupied >= 4)
    section_ratio = multi / max(sections, 1)
    dense_ratio = dense / max(sections, 1)
    mean_layers = float(np.mean(layers)) if layers else 0.0
    anti = min(1.0, 0.62 * section_ratio + 0.25 * dense_ratio + 0.13 * min(mean_layers / 4.0, 1.0))
    billboard = 1.0 - anti
    return {
        "pca_thickness_ratio": float(thin_range / long_range),
        "eigen_ratio_small_large": float(vals[2] / max(vals[0], 1e-12)),
        "section_count": int(sections),
        "multi_layer_section_ratio": float(section_ratio),
        "dense_section_ratio": float(dense_ratio),
        "mean_thin_axis_layers": float(mean_layers),
        "anti_billboard_score": float(anti),
        "billboard_score": float(billboard),
        "billboard_fail": bool(billboard > 0.54 or section_ratio < 0.28 or mean_layers < 2.15),
    }


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


def build_batch(row: dict[str, str], max_points: int, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, np.ndarray]]:
    base = load_npz(Path(row["baseline_path"]))
    graph = load_npz(Path(row["graph_path"]))
    weak_npz = load_npz(OUTPUT / "V10400000000000000000_weak_volume_regions" / row["case"] / "weak_volume_regions.npz")
    hp = np.asarray(base["human_points"], dtype=np.float32)
    rgb = as_rgb(base["human_rgb"]).astype(np.float32) / 255.0
    score = np.asarray(graph["mentor_weak_region_score"], dtype=np.float32)
    conf = np.asarray(graph["mentor_smpl_confidence"], dtype=np.float32)
    weak = np.asarray(weak_npz["weak_volume_region_mask"], dtype=bool).astype(np.float32)
    sheet = np.asarray(weak_npz["sheet_region_mask"], dtype=bool).astype(np.float32)
    multi = np.asarray(weak_npz["multi_layer_missing_mask"], dtype=bool).astype(np.float32)
    no_change = np.asarray(graph["no_change_mask"], dtype=bool)
    body = np.asarray(graph["geometry_body_part_id"], dtype=np.int64)
    priority = weak + sheet * 0.65 + multi * 0.8 + score * 0.2 + conf * 0.02
    order = np.argsort(-priority)
    idx = order[: min(max_points, len(order))]
    cfg = VolumeAwareVisibleMorphologyConfig()
    smpl = np.zeros((len(idx), cfg.smpl_feature_dim), dtype=np.float32)
    smpl[:, :8] = np.eye(8, dtype=np.float32)[np.clip(body[idx], 0, 7)]
    smpl[:, 8] = score[idx]
    smpl[:, 9] = conf[idx]
    smpl[:, 10] = sheet[idx]
    smpl[:, 11] = multi[idx]
    batch = {
        "anchor_xyz": torch.from_numpy(hp[idx][None]).to(device),
        "anchor_rgb": torch.from_numpy(rgb[idx][None]).to(device),
        "confidence": torch.from_numpy(conf[idx][None]).to(device),
        "weak_region": torch.from_numpy(np.maximum.reduce([weak[idx], sheet[idx], multi[idx], score[idx] * 0.25])[None]).to(device),
        "anchor_features": torch.zeros(1, len(idx), cfg.anchor_feature_dim, device=device),
        "smpl_features": torch.from_numpy(smpl[None]).to(device),
        "vggt_token_context": torch.zeros(1, cfg.token_dim, device=device),
    }
    axes = np.asarray(weak_npz["pca_axes"], dtype=np.float32)
    normal = axes[:, 2]
    side = axes[:, 1]
    anchor = hp[idx]
    center = anchor.mean(axis=0, keepdims=True)
    sign = np.sign((anchor - center) @ normal)
    sign[sign == 0] = 1.0
    side_sign = np.sign(((body[idx].astype(np.int32) * 37) % 5) - 2).astype(np.float32)
    side_sign[side_sign == 0] = 1.0
    target = anchor + normal[None] * sign[:, None] * (0.018 + 0.030 * np.maximum.reduce([weak[idx], sheet[idx], multi[idx], score[idx]])[:, None])
    target += side[None] * side_sign[:, None] * (0.010 + 0.012 * np.maximum(sheet[idx], score[idx])[:, None])
    targets = {
        "target_xyz": torch.from_numpy(target[None].astype(np.float32)).to(device),
        "weak": batch["weak_region"],
        "no_change": torch.from_numpy(no_change[idx][None]).to(device),
    }
    aux = {
        "idx": idx,
        "human_points": hp,
        "human_rgb": as_rgb(base["human_rgb"]),
        "environment_points": np.asarray(base["environment_points"], dtype=np.float32),
        "environment_rgb": as_rgb(base["environment_rgb"]),
        "no_change": no_change,
    }
    return batch, targets, aux


def anti_billboard_loss(points: torch.Tensor, weak: torch.Tensor) -> torch.Tensor:
    pts = points[0]
    centered = pts - pts.mean(dim=0, keepdim=True)
    # Differentiable covariance eigenvalue ratio encourages a non-degenerate
    # third axis, while the pairwise spread term pushes weak regions into layers.
    cov = centered.T @ centered / max(1, pts.shape[0] - 1)
    vals = torch.linalg.eigvalsh(cov)
    ratio_loss = F.relu(0.020 - vals[0] / vals[-1].clamp_min(1e-8))
    w = weak[0].squeeze(-1).clamp(0, 1)
    if int((w > 0.15).sum()) < 16:
        return ratio_loss
    q = centered[w > 0.15]
    sample = q[: min(q.shape[0], 768)]
    # Use the smallest-variance direction as the current thin axis.
    eigvals, eigvecs = torch.linalg.eigh(cov.detach())
    thin_axis = eigvecs[:, 0]
    spread = torch.std(sample @ thin_axis)
    spread_loss = F.relu(0.030 - spread)
    return ratio_loss + 0.35 * spread_loss


def train_case(row: dict[str, str], steps: int, max_points: int, device: torch.device) -> dict[str, Any]:
    cfg = VolumeAwareVisibleMorphologyConfig(max_residual=0.050, max_shell_offset=0.045, max_rgb_delta=0.025)
    model = VolumeAwareVisibleMorphologyStudent(cfg).to(device)
    batch, targets, aux = build_batch(row, max_points=max_points, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=1e-4)
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        weak = targets["weak"].unsqueeze(-1) if targets["weak"].ndim == 2 else targets["weak"]
        no_change = targets["no_change"].bool()
        residual = out["residual_xyz"]
        loss_target = ((out["student_xyz"] - targets["target_xyz"]) ** 2 * weak).mean()
        loss_preserve = (residual[no_change].square().mean() if bool(no_change.any()) else residual.square().mean() * 0)
        loss_occ = F.binary_cross_entropy(out["occupancy"], weak.clamp(0, 1))
        loss_ab = anti_billboard_loss(out["student_xyz"], weak)
        loss_shell = F.relu(0.012 - out["thickness_field"].mean())
        loss = loss_target + 1.8 * loss_preserve + 0.18 * loss_occ + 0.65 * loss_ab + 0.12 * loss_shell
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in {0, steps - 1} or (step + 1) % 25 == 0:
            history.append({"step": step + 1, "loss": float(loss.detach().cpu()), "anti_billboard_loss": float(loss_ab.detach().cpu()), "grad_norm": float(grad_norm)})
    with torch.no_grad():
        out = model(batch)
    human = aux["human_points"].copy()
    idx = aux["idx"]
    pred = out["student_xyz"][0].detach().cpu().numpy().astype(np.float32)
    # Preserve no-change zones, replace only sampled weak/sheet points.
    replace_mask = ~aux["no_change"][idx]
    human[idx[replace_mask]] = pred[replace_mask]
    rgb = aux["human_rgb"]
    full = np.concatenate([human, aux["environment_points"]], axis=0)
    full_rgb = np.concatenate([rgb, aux["environment_rgb"]], axis=0)
    out_dir = ensure(OUT_ROOT / row["case"] / "anti_billboard_trained_true")
    np.savez_compressed(
        out_dir / "predictions.npz",
        human_points=human,
        human_rgb=rgb,
        environment_points=aux["environment_points"],
        environment_rgb=aux["environment_rgb"],
        full_scene_points=full,
        full_scene_rgb=full_rgb,
        model_owned_student_output=np.array(True),
        teacher_points_used_at_inference=np.array(False),
        raw_kinect_depth_used_at_inference=np.array(False),
        facial_detail_target_applicable=np.array(False),
        face_detail_claim_allowed=np.array(False),
        allowed_face_claim=np.array(ALLOWED_FACE_CLAIM),
    )
    write_ply(out_dir / "full_scene_rgb_pointcloud.ply", full, full_rgb)
    metrics = anti_billboard_metrics(human)
    return {
        "case": row["case"],
        "prediction": str(out_dir / "predictions.npz"),
        "ply": str(out_dir / "full_scene_rgb_pointcloud.ply"),
        "steps": steps,
        "max_points": max_points,
        "device": str(device),
        "history_json": json.dumps(history),
        "model_owned_student_output": True,
        "teacher_points_used_at_inference": False,
        "raw_kinect_depth_used_at_inference": False,
        **metrics,
    }


def load_config(case: str, config: str) -> tuple[np.ndarray, np.ndarray]:
    if config == "anti_billboard_trained_true":
        path = OUT_ROOT / case / "anti_billboard_trained_true" / "predictions.npz"
    else:
        path = MATRIX_ROOT / case / config / "predictions.npz"
    pred = load_npz(path)
    return np.asarray(pred["human_points"], dtype=np.float32), as_rgb(pred["human_rgb"])


def rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return rz @ rx


def render_panel(points: np.ndarray, colors: np.ndarray, title: str, rot: np.ndarray) -> Image.Image:
    size = (380, 275)
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
    rgb = np.clip(colors.astype(np.float32) * (0.64 + 0.42 * d[:, None]), 0, 255).astype(np.uint8)
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
    size = (380, 275)
    im = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(im)
    _center, _vals, axes, proj = pca_numpy(points)
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


def compare(rows: list[dict[str, str]], created_at: str) -> None:
    metric_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    configs = ["anti_billboard_trained_true", *CONTROL_CONFIGS]
    for row in rows:
        case = row["case"]
        case_metrics: dict[str, dict[str, Any]] = {}
        for config in configs:
            pts, _rgb = load_config(case, config)
            met = anti_billboard_metrics(pts)
            case_metrics[config] = met
            metric_rows.append({"case": case, "config": config, **met})
        true_score = float(case_metrics["anti_billboard_trained_true"]["anti_billboard_score"])
        if case_metrics["anti_billboard_trained_true"]["billboard_fail"]:
            failures.append({"case": case, "reason": "true_billboard_fail", "true_score": true_score})
        for config in CONTROL_CONFIGS:
            control_score = float(case_metrics[config]["anti_billboard_score"])
            if control_score >= true_score * 0.96:
                failures.append({"case": case, "reason": "control_close_or_better", "control": config, "true_score": true_score, "control_score": control_score})
    write_csv(REPORTS / "V13060000000000000000_anti_billboard_training_metrics.csv", metric_rows)
    case = rows[0]["case"]
    board_configs = ["anti_billboard_trained_true", "real_vggt_baseline_only", "same_topology_no_semantic", "shuffled_smpl_feature", "thickness_only_control", "tiny_synthetic_token_control"]
    rot = rotation_matrix(-30, 61)
    panels = []
    cross = []
    for config in board_configs:
        pts, rgb = load_config(case, config)
        panels.append(render_panel(pts, rgb, config.replace("_", " "), rot))
        cross.append(cross_panel(pts, config.replace("_", " ")))
    compose(panels, 3, BOARDS / "V13060000000000000000_anti_billboard_training_turntable.png")
    compose(cross, 3, BOARDS / "V13060000000000000000_anti_billboard_training_cross_section.png")
    write_json(
        REPORTS / "V13060000000000000000_anti_billboard_training_decision.json",
        {
            "created_at": created_at,
            "status": "V13060_ANTI_BILLBOARD_TRAINING_FAIL_CLOSED_CONTINUE" if failures else "V13060_ANTI_BILLBOARD_TRAINING_PRECHECK_PASS_REQUIRES_MODAL_VISUAL",
            "mentor_ready": False,
            "external_hard_block": False,
            "failures": failures,
            "face_detail_claim_allowed": False,
            "allowed_face_claim": ALLOWED_FACE_CLAIM,
            "modal_required_for_final": True,
            "boards": {
                "turntable": str(BOARDS / "V13060000000000000000_anti_billboard_training_turntable.png"),
                "cross_section": str(BOARDS / "V13060000000000000000_anti_billboard_training_cross_section.png"),
            },
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
            if torch.cuda.get_device_name():
                info["cuda_device_name"] = torch.cuda.get_device_name()
            return torch.device("cuda"), info
        except Exception as exc:  # pragma: no cover - hardware dependent
            info["cuda_error"] = str(exc)
    return torch.device("cpu"), info


def main() -> int:
    created_at = now()
    rows = read_manifest()
    device, device_info = select_device()
    steps = int(os.environ.get("V13060_STEPS", "45"))
    max_points = int(os.environ.get("V13060_MAX_POINTS", "4096"))
    manifest = [train_case(row, steps=steps, max_points=max_points, device=device) for row in rows]
    write_csv(REPORTS / "V13060000000000000000_anti_billboard_training_manifest.csv", manifest)
    compare(rows, created_at)
    write_json(
        REPORTS / "V13060000000000000000_runtime_environment.json",
        {
            "created_at": created_at,
            **device_info,
            "note": "Local CPU smoke is allowed for wiring. Final mentor evidence still requires Modal/A10/A100 training per goal.",
        },
    )
    print(json.dumps({"created_at": created_at, "status": "V13060_DONE", "device": str(device), "mentor_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
