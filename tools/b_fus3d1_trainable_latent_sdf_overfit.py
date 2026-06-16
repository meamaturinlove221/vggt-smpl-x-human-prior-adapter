from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import b_gs0_smplx_anchored_free_gaussian_smoke as bgs0  # noqa: E402


DEFAULT_LATENT_GRID = REPO_ROOT / (
    "output/surface_research_preflight_local/"
    "B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23/"
    "b_fus3d_latent_grid_evidence_arrays.npz"
)
DEFAULT_QUERY_EVIDENCE = REPO_ROOT / (
    "output/surface_research_preflight_local/"
    "B_Fus3D6_query_evidence_cache_hybrid6_layer23/"
    "b_fus3d_query_evidence_cache.npz"
)
DEFAULT_TOKEN_CACHE = REPO_ROOT / (
    "output/surface_research_preflight_local/"
    "B_Fus3D0_token_cache_extract_hybrid6_518_roi_withhands_arrays_v2/"
    "token_cache/aggregator_layer_23.npz"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/surface_research_preflight_local/B_Fus3D1_trainable_latent_sdf_overfit"
DEFAULT_REPORT = REPO_ROOT / "reports/20260507_b_fus3d1_trainable_latent_sdf_overfit_status.md"

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
    "teacher_export": "blocked",
    "candidate_export": "blocked",
    "predictions_export": "blocked",
    "registry_write": "blocked",
}
CONTRACT = {
    "research_only": True,
    "local_only": True,
    "trainable_single_frame_overfit": True,
    "no_cloud": True,
    "no_teacher": True,
    "no_candidate": True,
    "no_predictions": True,
    "no_checkpoint_write": True,
    "no_strict_pass_write": True,
    "no_registry_write": True,
    "writes_predictions_npz": False,
    "writes_checkpoint": False,
}
CONTROL_NAMES = ("real", "shuffle", "zero", "random_view")
REGION_NAMES = ("full_body", "face_core", "hairline", "left_hand", "right_hand")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B-Fus3D1 trainable latent-grid SDF overfit, research-only.")
    parser.add_argument("--scene-dir", type=Path, default=bgs0.DEFAULT_SCENE_DIR)
    parser.add_argument("--latent-grid", type=Path, default=DEFAULT_LATENT_GRID)
    parser.add_argument("--query-evidence", type=Path, default=DEFAULT_QUERY_EVIDENCE)
    parser.add_argument("--token-cache", type=Path, default=DEFAULT_TOKEN_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--view-indices", default="0,10,24,36,45,57")
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    if isinstance(value, torch.Tensor):
        return json_ready(value.detach().cpu().numpy())
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def stat_row(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": int(values.size), "finite": 0}
    return {
        "count": int(values.size),
        "finite": int(finite.size),
        "min": float(finite.min()),
        "p10": float(np.percentile(finite, 10)),
        "median": float(np.median(finite)),
        "mean": float(finite.mean()),
        "p90": float(np.percentile(finite, 90)),
        "max": float(finite.max()),
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.percentile(finite, [5.0, 95.0])
    if hi <= lo + 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - float(lo)) / float(hi - lo), 0.0, 1.0).astype(np.float32)


def control_features(latent: dict[str, np.ndarray], control: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    out = {key: np.asarray(value).copy() for key, value in latent.items()}
    if control == "real":
        return out
    if control == "shuffle":
        perm = rng.permutation(out["points"].shape[0])
        for key in ("evidence_score", "token_cosine", "rgb_range", "rgb_variance"):
            out[key] = out[key][perm]
        return out
    if control == "zero":
        for key in ("evidence_score", "token_cosine", "rgb_range", "rgb_variance"):
            out[key] = np.zeros_like(out[key])
        return out
    if control == "random_view":
        out["evidence_score"] = np.roll(out["evidence_score"], 137) + rng.normal(0.0, 0.10, out["evidence_score"].shape).astype(np.float32)
        out["token_cosine"] = np.roll(out["token_cosine"], -83)
        return out
    raise ValueError(control)


def init_logits(features: dict[str, np.ndarray]) -> np.ndarray:
    occ = np.asarray(features["occupancy_ratio"], dtype=np.float32)
    evidence = normalize01(features["evidence_score"])
    token = normalize01(features["token_cosine"])
    visible = np.asarray(features["visible_count"], dtype=np.float32) / 6.0
    mask = np.asarray(features["mask_count"], dtype=np.float32) / 6.0
    rgb_range = normalize01(features["rgb_range"])
    score = 1.45 * occ + 0.85 * evidence + 0.45 * token + 0.35 * visible + 0.40 * mask - 0.30 * rgb_range - 1.20
    return score.astype(np.float32)


def project_points_torch(points: torch.Tensor, world_to_cam: np.ndarray, intrinsic: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    w2c = torch.as_tensor(world_to_cam, dtype=points.dtype, device=points.device)
    k = torch.as_tensor(intrinsic, dtype=points.dtype, device=points.device)
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    uvw = cam @ k.T
    uv = uvw[:, :2] / torch.clamp(uvw[:, 2:3], min=1e-6)
    return uv, z


def splat_mask(points: torch.Tensor, weights: torch.Tensor, view: dict[str, Any], camera: dict[str, np.ndarray], target_size: int, radius: int, temperature: float) -> torch.Tensor:
    intrinsic = bgs0.align_intrinsics_for_loaded_scene_view(np.asarray(camera["intrinsic"], dtype=np.float32), view, target_size=target_size)
    uv, depth = project_points_torch(points, np.asarray(camera["world_to_cam"], dtype=np.float32), intrinsic)
    valid = (
        torch.isfinite(uv).all(dim=1)
        & torch.isfinite(depth)
        & (depth > 1e-6)
        & (uv[:, 0] >= -radius)
        & (uv[:, 0] < target_size + radius)
        & (uv[:, 1] >= -radius)
        & (uv[:, 1] < target_size + radius)
    )
    uv = uv[valid]
    w = weights[valid]
    if uv.numel() == 0:
        return torch.zeros((target_size, target_size), dtype=points.dtype, device=points.device)
    xi = torch.round(uv[:, 0]).long()
    yi = torch.round(uv[:, 1]).long()
    mask = torch.zeros((target_size, target_size), dtype=points.dtype, device=points.device)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            x = xi + dx
            y = yi + dy
            inside = (x >= 0) & (x < target_size) & (y >= 0) & (y < target_size)
            if inside.any():
                falloff = math.exp(-(dx * dx + dy * dy) / max(temperature * 24.0, 1e-6))
                mask.index_put_((y[inside], x[inside]), w[inside] * float(falloff), accumulate=True)
    return 1.0 - torch.exp(-torch.clamp(mask, min=0.0, max=16.0))


def mask_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred_b = np.asarray(pred) >= 0.5
    target_b = np.asarray(target, dtype=bool)
    inter = int(np.count_nonzero(pred_b & target_b))
    union = int(np.count_nonzero(pred_b | target_b))
    pred_count = int(np.count_nonzero(pred_b))
    target_count = int(np.count_nonzero(target_b))
    overfill = int(np.count_nonzero(pred_b & ~target_b))
    return {
        "iou": float(inter / union) if union else 0.0,
        "target_recall": float(inter / target_count) if target_count else 0.0,
        "overfill_ratio": float(overfill / max(pred_count, 1)),
        "pred_pixels": pred_count,
        "target_pixels": target_count,
        "intersection": inter,
    }


def render_numpy(points: np.ndarray, occ: np.ndarray, views: list[dict[str, Any]], cameras: dict[str, dict[str, np.ndarray]], target_size: int, radius: int, output_dir: Path, name: str) -> dict[str, Any]:
    point_t = torch.as_tensor(points, dtype=torch.float32, device="cpu")
    weight_t = torch.as_tensor(occ, dtype=torch.float32, device="cpu")
    rows = []
    for view in views:
        camera = cameras[bgs0.normalize_camera_id(view["camera_id"])]
        pred = splat_mask(point_t, weight_t, view, camera, target_size, radius, temperature=0.10).detach().cpu().numpy()
        target = np.asarray(view["mask"], dtype=bool)
        metrics = mask_metrics(pred, target)
        metrics["view_index"] = int(view["view_index"])
        metrics["camera_id"] = str(view["camera_id"])
        rows.append(metrics)
        out = output_dir / "renders" / name / f"view_{int(view['view_index']):02d}_mask.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((np.clip(pred, 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(out)
    return {
        "views": rows,
        "mean_iou": float(np.mean([r["iou"] for r in rows])) if rows else 0.0,
        "mean_target_recall": float(np.mean([r["target_recall"] for r in rows])) if rows else 0.0,
        "mean_overfill_ratio": float(np.mean([r["overfill_ratio"] for r in rows])) if rows else 0.0,
    }


def query_region_metrics(points: np.ndarray, occ: np.ndarray, query: dict[str, np.ndarray]) -> dict[str, Any]:
    qpos = np.asarray(query["query_positions"], dtype=np.float32)
    fam = np.asarray(query["query_families"]).astype(str)
    if qpos.size == 0:
        return {}
    nearest = np.zeros((qpos.shape[0],), dtype=np.int64)
    for start in range(0, qpos.shape[0], 256):
        stop = min(start + 256, qpos.shape[0])
        d2 = ((qpos[start:stop, None, :] - points[None, :, :]) ** 2).sum(axis=2)
        nearest[start:stop] = np.argmin(d2, axis=1)
    out = {}
    for family in REGION_NAMES:
        m = fam == family
        if not np.any(m):
            continue
        vals = occ[nearest[m]]
        out[family] = {
            "query_count": int(m.sum()),
            "mean_occupancy": float(vals.mean()),
            "occupied_query_ratio": float((vals >= 0.56).mean()),
        }
    return out


def write_ply(path: Path, points: np.ndarray, occ: np.ndarray, conf: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = occ >= 0.56
    pts = points[keep]
    occ_k = occ[keep]
    conf_k = conf[keep]
    colors = np.zeros((pts.shape[0], 3), dtype=np.uint8)
    colors[:, 0] = np.clip(60 + 180 * occ_k, 0, 255).astype(np.uint8)
    colors[:, 1] = np.clip(70 + 160 * conf_k, 0, 255).astype(np.uint8)
    colors[:, 2] = 190
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {pts.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty blue\n".replace("property blue", "property uchar blue"))
        handle.write("property float occupancy\nproperty float confidence\n")
        handle.write("end_header\n")
        for p, c, o, cf in zip(pts, colors, occ_k, conf_k, strict=False):
            handle.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {int(c[0])} {int(c[1])} {int(c[2])} {float(o):.7f} {float(cf):.7f}\n")


def render_contact_sheet(render_dir: Path, out_path: Path, title: str) -> None:
    paths = sorted(render_dir.rglob("*.png"))
    thumbs = []
    labels = []
    for p in paths[:32]:
        thumbs.append(Image.open(p).convert("RGB").resize((220, 220), Image.Resampling.BICUBIC))
        labels.append(p.parent.name + "/" + p.stem)
    if not thumbs:
        return
    cols = 4
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 220, rows * 246 + 34), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=(0, 0, 0))
    for i, thumb in enumerate(thumbs):
        x = (i % cols) * 220
        y = 34 + (i // cols) * 246
        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + 224), labels[i][:30], fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def train_control(
    name: str,
    latent: dict[str, np.ndarray],
    query: dict[str, np.ndarray],
    views: list[dict[str, Any]],
    cameras: dict[str, dict[str, np.ndarray]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    points_np = np.asarray(latent["points"], dtype=np.float32)
    init = init_logits(latent)
    evidence = normalize01(latent["evidence_score"])
    conf_np = np.clip(0.35 * (np.asarray(latent["visible_count"], dtype=np.float32) / 6.0) + 0.35 * evidence + 0.20 * normalize01(latent["token_cosine"]), 0.0, 1.0)
    points = torch.as_tensor(points_np, dtype=torch.float32, device=device)
    logits = torch.nn.Parameter(torch.as_tensor(init, dtype=torch.float32, device=device))
    base_logits = torch.as_tensor(init, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([logits], lr=float(args.lr))
    target_masks = [torch.as_tensor(np.asarray(view["mask"], dtype=np.float32), device=device) for view in views]
    trace = []
    for step in range(int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        occ = torch.sigmoid(logits)
        losses = []
        ious = []
        overfills = []
        recalls = []
        for vi, view in enumerate(views):
            camera = cameras[bgs0.normalize_camera_id(view["camera_id"])]
            pred = splat_mask(points, occ, view, camera, int(args.target_size), int(args.point_radius), float(args.temperature))
            target = target_masks[vi]
            inter_soft = (pred * target).sum()
            union_soft = pred.sum() + target.sum() - inter_soft + 1e-6
            iou_loss = 1.0 - inter_soft / union_soft
            overfill = (pred * (1.0 - target)).sum() / (pred.sum() + 1e-6)
            recall = inter_soft / (target.sum() + 1e-6)
            recall_loss = F.relu(0.82 - recall)
            losses.append(iou_loss + 0.42 * overfill + 0.25 * recall_loss)
            ious.append((inter_soft / union_soft).detach())
            overfills.append(overfill.detach())
            recalls.append(recall.detach())
        sparsity = occ.mean()
        regularize = ((logits - base_logits) ** 2).mean()
        eikonal_proxy = torch.mean(torch.abs(occ[1:] - occ[:-1]))
        loss = torch.stack(losses).mean() + 0.020 * sparsity + 0.010 * regularize + 0.006 * eikonal_proxy
        if step > 0:
            loss.backward()
            optimizer.step()
        if step % int(args.log_every) == 0 or step == int(args.steps):
            trace.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "mean_soft_iou": float(torch.stack(ious).mean().cpu()),
                    "mean_soft_overfill": float(torch.stack(overfills).mean().cpu()),
                    "mean_soft_recall": float(torch.stack(recalls).mean().cpu()),
                    "mean_occupancy": float(occ.mean().detach().cpu()),
                }
            )
    final_occ = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    metrics = render_numpy(points_np, final_occ, views, cameras, int(args.target_size), int(args.point_radius), args.output_dir, name)
    metrics["training_trace"] = trace
    metrics["query_region_metrics"] = query_region_metrics(points_np, final_occ, query)
    metrics["occupancy_stats"] = stat_row(final_occ)
    metrics["confidence_stats"] = stat_row(conf_np)
    write_ply(args.output_dir / f"b_fus3d1_{name}_trainable_latent_sdf_occupied_points.ply", points_np, final_occ, conf_np)
    return metrics, final_occ, conf_np


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Fus3D1 Trainable Latent SDF Overfit",
        "",
        "Status: `research_only_trainable_overfit_no_export`",
        "",
        "## Strict Truth",
        "",
        "```text",
        "strict_candidate_passes = 0",
        "strict_teacher_passes = 0",
        "formal cloud train/infer/export = blocked",
        "```",
        "",
        "## Control Metrics",
        "",
        "| control | IoU | overfill | recall | occupied mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in CONTROL_NAMES:
        row = summary["controls"][name]
        lines.append(f"| `{name}` | {row['mean_iou']:.4f} | {row['mean_overfill_ratio']:.4f} | {row['mean_target_recall']:.4f} | {row['occupancy_stats']['mean']:.4f} |")
    lines += ["", "## Comparison", "", "```json", json.dumps(summary["comparison"], indent=2, ensure_ascii=False), "```", "", "## Decision", "", summary["decision"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latent_base = load_npz(args.latent_grid)
    query = load_npz(args.query_evidence)
    token_cache = load_npz(args.token_cache)
    views, cameras, camera_source = bgs0.load_views(args.scene_dir, args.dataset_root, args.subset_name, args.view_indices, int(args.target_size))
    rng = np.random.default_rng(20260507)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    controls: dict[str, Any] = {}
    occ_arrays: dict[str, np.ndarray] = {}
    conf_arrays: dict[str, np.ndarray] = {}
    for name in CONTROL_NAMES:
        features = control_features(latent_base, name, rng)
        metrics, occ, conf = train_control(name, features, query, views, cameras, args, device)
        controls[name] = metrics
        occ_arrays[name] = occ
        conf_arrays[name] = conf
    real = controls["real"]
    comparison = {
        "real_minus_shuffle_iou": float(real["mean_iou"] - controls["shuffle"]["mean_iou"]),
        "real_minus_zero_iou": float(real["mean_iou"] - controls["zero"]["mean_iou"]),
        "real_minus_random_view_iou": float(real["mean_iou"] - controls["random_view"]["mean_iou"]),
        "real_minus_shuffle_overfill": float(real["mean_overfill_ratio"] - controls["shuffle"]["mean_overfill_ratio"]),
        "real_minus_zero_overfill": float(real["mean_overfill_ratio"] - controls["zero"]["mean_overfill_ratio"]),
        "real_minus_random_view_overfill": float(real["mean_overfill_ratio"] - controls["random_view"]["mean_overfill_ratio"]),
        "real_minus_shuffle_recall": float(real["mean_target_recall"] - controls["shuffle"]["mean_target_recall"]),
        "real_minus_zero_recall": float(real["mean_target_recall"] - controls["zero"]["mean_target_recall"]),
        "real_minus_random_view_recall": float(real["mean_target_recall"] - controls["random_view"]["mean_target_recall"]),
    }
    region_compare: dict[str, Any] = {}
    for region in REGION_NAMES:
        real_region = real["query_region_metrics"].get(region, {})
        row = {"real": real_region}
        for ctrl in ("shuffle", "zero", "random_view"):
            ctrl_region = controls[ctrl]["query_region_metrics"].get(region, {})
            row[ctrl] = ctrl_region
            if real_region and ctrl_region:
                row[f"real_minus_{ctrl}_occupied_query_ratio"] = float(real_region["occupied_query_ratio"] - ctrl_region["occupied_query_ratio"])
                row[f"real_minus_{ctrl}_mean_occupancy"] = float(real_region["mean_occupancy"] - ctrl_region["mean_occupancy"])
        region_compare[region] = row
    real_wins_render = bool(comparison["real_minus_shuffle_iou"] > 0.002 and comparison["real_minus_zero_iou"] > 0.002)
    hard_region_positive = any(
        region_compare.get(region, {}).get("real_minus_shuffle_occupied_query_ratio", 0.0) > 0.05
        and region_compare.get(region, {}).get("real_minus_zero_occupied_query_ratio", 0.0) > 0.05
        for region in ("face_core", "hairline", "left_hand", "right_hand")
    )
    decision = (
        "RESEARCH_ONLY_PROGRESS: B-Fus3D1 produced trainable overfit curves and real beats shuffle/zero in rendered IoU and at least one hard region."
        if real_wins_render and hard_region_positive
        else "FAIL: B-Fus3D1 trained locally and wrote artifacts, but did not meet v7 rendered/control/hard-region improvement criteria."
    )
    np.savez_compressed(
        args.output_dir / "b_fus3d1_trainable_latent_sdf_fields.npz",
        points=np.asarray(latent_base["points"], dtype=np.float32),
        **{f"{name}_occupancy": occ for name, occ in occ_arrays.items()},
        **{f"{name}_confidence": conf for name, conf in conf_arrays.items()},
    )
    render_contact_sheet(args.output_dir / "renders", args.output_dir / "b_fus3d1_render_contact_sheet.png", "B-Fus3D1 rendered mask controls")
    summary = {
        "task": "b_fus3d1_trainable_latent_sdf_overfit",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "research_only_trainable_overfit_no_export",
        "pass": False,
        "strict_facts": STRICT_FACTS,
        **STRICT_FACTS,
        "contract": CONTRACT,
        "inputs": {
            "latent_grid": str(args.latent_grid.resolve()),
            "query_evidence": str(args.query_evidence.resolve()),
            "token_cache": str(args.token_cache.resolve()),
            "scene_dir": str(args.scene_dir.resolve()),
            "camera_source": camera_source,
            "view_indices": [int(v["view_index"]) for v in views],
            "target_size": int(args.target_size),
            "steps": int(args.steps),
            "device": str(device),
            "tokens_shape": list(token_cache["tokens"].shape) if "tokens" in token_cache else None,
        },
        "controls": controls,
        "comparison": comparison,
        "region_compare": region_compare,
        "real_wins_render": real_wins_render,
        "hard_region_positive": hard_region_positive,
        "outputs": {
            "output_dir": str(args.output_dir),
            "summary_json": str((args.output_dir / "b_fus3d1_trainable_latent_sdf_overfit_summary.json").resolve()),
            "report_md": str((args.output_dir / "b_fus3d1_trainable_latent_sdf_overfit_report.md").resolve()),
            "field_npz": str((args.output_dir / "b_fus3d1_trainable_latent_sdf_fields.npz").resolve()),
            "contact_sheet": str((args.output_dir / "b_fus3d1_render_contact_sheet.png").resolve()),
        },
        "decision": decision,
    }
    write_json(args.output_dir / "b_fus3d1_trainable_latent_sdf_overfit_summary.json", summary)
    write_markdown(args.output_dir / "b_fus3d1_trainable_latent_sdf_overfit_report.md", summary)
    write_markdown(args.status_report.resolve(), summary)
    print(json.dumps({"status": summary["status"], "real_wins_render": real_wins_render, "hard_region_positive": hard_region_positive, "decision": decision}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
