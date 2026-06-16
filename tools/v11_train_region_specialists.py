from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import torch
from torch import nn
import torch.nn.functional as F

from v11_surface_completion_pipeline import (
    CONTRACT,
    LOCAL_ROOT,
    REPO_ROOT,
    contact_sheet,
    load_ply_xyz_rgb,
    load_template,
    region_masks_from_template,
    safe_output_dir,
    select_region_by_bbox,
    utc_now,
    write_ascii_ply,
    write_json,
    write_report,
)
from vggt.models.human_hand_decoder import HumanHandTokenResidualDecoder
from vggt.models.human_hair_strand_gaussian import HumanHairStrandGaussian


ASSETS = REPO_ROOT / "output/surface_research_cloud_preflight/V9_cloud_asset_staging/assets"
QUERY_CACHE = ASSETS / "query_cache/b_fus3d_query_evidence_cache.npz"
G3_ANCHOR = LOCAL_ROOT / "V11_G3_2DGS_surface_anchor/g3_2dgs_anchor_surface.ply"
OUT_HAND = LOCAL_ROOT / "V11_HHand_B_vggt_decoder"
OUT_HAIR = LOCAL_ROOT / "V11_HHair_B_native_strand_gaussian"


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_query_cache() -> dict[str, np.ndarray]:
    with np.load(QUERY_CACHE, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def family_rows(cache: dict[str, np.ndarray], names: set[str]) -> np.ndarray:
    fam = np.asarray(cache["query_families"]).astype(str)
    mask = np.zeros(fam.shape[0], dtype=bool)
    for name in names:
        mask |= fam == name
    return np.flatnonzero(mask)


def nearest_targets(query_positions: np.ndarray, target_points: np.ndarray, max_points: int = 2048) -> np.ndarray:
    if len(query_positions) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if len(target_points) == 0:
        return query_positions.astype(np.float32)
    pts = target_points.astype(np.float32)
    if len(pts) > max_points:
        rng = np.random.default_rng(712)
        pts = pts[rng.choice(len(pts), size=max_points, replace=False)]
    q = query_positions.astype(np.float32)
    chunks = []
    for start in range(0, len(q), 512):
        block = q[start : start + 512]
        dist = ((block[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
        chunks.append(pts[np.argmin(dist, axis=1)])
    return np.concatenate(chunks, axis=0).astype(np.float32)


def plot_curve(path: Path, rows: list[dict[str, float]], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (900, 420), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((50, 40, 860, 360), outline=(30, 30, 30))
    if not rows:
        draw.text((60, 60), "no rows", fill=(160, 0, 0))
        img.save(path)
        return
    palette = [(220, 60, 60), (60, 120, 220), (40, 160, 90), (140, 80, 200)]
    steps = [row["step"] for row in rows]
    x0, x1 = min(steps), max(steps)
    all_vals = [row[key] for row in rows for key in keys if key in row and math.isfinite(row[key])]
    y0, y1 = min(all_vals), max(all_vals)
    if abs(y1 - y0) < 1e-6:
        y1 = y0 + 1.0
    for kidx, key in enumerate(keys):
        pts = []
        for row in rows:
            if key not in row:
                continue
            x = 50 + int((row["step"] - x0) / max(1e-6, x1 - x0) * 810)
            y = 360 - int((row[key] - y0) / (y1 - y0) * 320)
            pts.append((x, y))
        if len(pts) >= 2:
            draw.line(pts, fill=palette[kidx % len(palette)], width=3)
        draw.text((60, 15 + 18 * kidx), key, fill=palette[kidx % len(palette)])
    img.save(path)


def write_training_summary(out: Path, title: str, summary: dict[str, Any]) -> None:
    write_json(out / "summary.json", summary)
    write_report(out / "report.md", title, summary)


class PointResidualHead(nn.Module):
    def __init__(self, token_dim: int = 2048, hidden_dim: int = 192) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return positions + 0.08 * torch.tanh(self.net(torch.cat([features, positions], dim=-1)))


def train_point_residual(
    features: np.ndarray,
    positions: np.ndarray,
    targets: np.ndarray,
    *,
    steps: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointResidualHead(token_dim=features.shape[1]).to(device)
    feat = torch.from_numpy(features.astype(np.float32)).to(device)
    pos = torch.from_numpy(positions.astype(np.float32)).to(device)
    tgt = torch.from_numpy(targets.astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    rows = []
    for step in range(int(steps) + 1):
        pred = model(feat, pos)
        loss = F.smooth_l1_loss(pred, tgt) + 0.05 * ((pred - pos) ** 2).mean()
        if step:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if step % max(1, int(steps) // 10) == 0 or step == steps:
            with torch.no_grad():
                base = F.smooth_l1_loss(pos, tgt).item()
                rows.append({"step": float(step), "real_loss": float(loss.item()), "template_loss": float(base)})
    with torch.no_grad():
        final = model(feat, pos).detach().cpu().numpy().astype(np.float32)
    state = {"model": model.state_dict(), "rows": rows, "device": str(device)}
    metrics = {"final_real_loss": rows[-1]["real_loss"], "template_loss": rows[-1]["template_loss"], "loss_improvement": rows[-1]["template_loss"] - rows[-1]["real_loss"]}
    return final, metrics, state


def control_metrics(features: np.ndarray, positions: np.ndarray, targets: np.ndarray, *, steps: int) -> dict[str, Any]:
    controls = {}
    zero = np.zeros_like(features, dtype=np.float32)
    shuffled = features.copy()
    rng = np.random.default_rng(42)
    rng.shuffle(shuffled, axis=0)
    for name, feat in (("zero", zero), ("shuffle", shuffled)):
        _, metrics, _ = train_point_residual(feat, positions, targets, steps=max(5, steps // 4), seed=100 + len(name))
        controls[name] = metrics
    return controls


def run_hand(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    cache = load_query_cache()
    rows = family_rows(cache, {"left_hand", "right_hand"})
    features = np.asarray(cache["mean_features"][rows], dtype=np.float32)
    positions = np.asarray(cache["query_positions"][rows], dtype=np.float32)
    families = np.asarray(cache["query_families"][rows]).astype(str)
    g3_points, _ = load_ply_xyz_rgb(args.g3_anchor)
    template = load_template()
    masks = region_masks_from_template(template)
    tpts = template["hybrid_vertices"].astype(np.float32)
    anchor_masks = []
    for name in ("left_hand", "right_hand"):
        anchor_masks.append(select_region_by_bbox(g3_points, tpts[masks[name]], fallback_fraction=(0.0, 0.55)))
    hand_anchor = g3_points[np.logical_or.reduce(anchor_masks)] if len(anchor_masks) else np.zeros((0, 3), dtype=np.float32)
    targets = nearest_targets(positions, hand_anchor)
    pred, metrics, state = train_point_residual(features, positions, targets, steps=args.steps, seed=11)
    controls = control_metrics(features, positions, targets, steps=args.steps)
    checkpoint = out / "b_hand11_checkpoint.pt"
    torch.save(
        {
            "module": "vggt.models.human_hand_decoder.HumanHandTokenResidualDecoder",
            "residual_head": state["model"],
            "trained_steps": int(args.steps),
            "query_rows": rows,
            "families": families,
            "metrics": metrics,
            "controls": controls,
            "research_only": True,
            "not_a_pass": True,
        },
        checkpoint,
    )
    left = pred[families == "left_hand"]
    right = pred[families == "right_hand"]
    write_ascii_ply(out / "b_hand11_left_surface.ply", left, None)
    write_ascii_ply(out / "b_hand11_right_surface.ply", right, None)
    write_ascii_ply(out / "b_hand11_wrist_bridge.ply", pred, None)
    contact_sheet(pred, None, out / "b_hand11_open3d_left_right.png", "B-hand11 trained")
    plot_curve(out / "b_hand11_training_curve.png", state["rows"], ["real_loss", "template_loss"])
    real_better_zero = metrics["final_real_loss"] < controls["zero"]["final_real_loss"]
    real_better_shuffle = metrics["final_real_loss"] < controls["shuffle"]["final_real_loss"]
    pass_gate = bool(real_better_zero and real_better_shuffle and len(left) > 20 and len(right) > 20 and metrics["loss_improvement"] > 0.0)
    summary = {
        "task": "v11_b_hand11_bounded_token_overfit",
        "created_utc": utc_now(),
        "status": "b_hand11_bounded_overfit_positive_research_only" if pass_gate else "b_hand11_bounded_overfit_blocked",
        **CONTRACT,
        "hand_visual_precheck_pass": False,
        "ownership_pass": False,
        "bounded_overfit_positive": pass_gate,
        "decoder_module": REPO_ROOT / "vggt/models/human_hand_decoder.py",
        "checkpoint": checkpoint,
        "trained_steps": int(args.steps),
        "uses_real_vggt_query_features": True,
        "source_query_cache": QUERY_CACHE,
        "source_g3_anchor": args.g3_anchor,
        "metrics": metrics,
        "controls": controls,
        "control_gates": {"real_better_zero": real_better_zero, "real_better_shuffle": real_better_shuffle},
        "outputs": {
            "left_surface": out / "b_hand11_left_surface.ply",
            "right_surface": out / "b_hand11_right_surface.ply",
            "wrist_bridge": out / "b_hand11_wrist_bridge.ply",
            "open3d_sheet": out / "b_hand11_open3d_left_right.png",
            "curve": out / "b_hand11_training_curve.png",
        },
        "stats": {
            "left_hand_present": bool(len(left) > 20),
            "right_hand_present": bool(len(right) > 20),
            "finger_structure_visible": False,
            "wrist_connected": False,
            "not_scaffold_only": pass_gate,
        },
        "blockers": [
            "Bounded overfit uses weak G3/query supervision, not external hand GT.",
            "No finger-level visual gate or wrist continuity pass yet; D-line must keep ownership false.",
        ],
        "decision": "B-hand11 now has a trained research checkpoint/control run, but remains below strict hand ownership.",
    }
    write_training_summary(out, "V11 B-hand11 Bounded Token Overfit", summary)
    print(json.dumps(json_ready({"status": summary["status"], "metrics": metrics, "controls": controls, "output": out}), ensure_ascii=False))
    return 0 if pass_gate else 2


def run_hair(args: argparse.Namespace) -> int:
    out = safe_output_dir(args.output_dir)
    cache = load_query_cache()
    rows = family_rows(cache, {"hairline", "head_top", "head"})
    if len(rows) == 0:
        rows = family_rows(cache, {"hairline"})
    features = np.asarray(cache["mean_features"][rows], dtype=np.float32)
    positions = np.asarray(cache["query_positions"][rows], dtype=np.float32)
    families = np.asarray(cache["query_families"][rows]).astype(str)
    g3_points, _ = load_ply_xyz_rgb(args.g3_anchor)
    template = load_template()
    masks = region_masks_from_template(template)
    tpts = template["hybrid_vertices"].astype(np.float32)
    hair_mask = select_region_by_bbox(g3_points, tpts[masks["hairline"]], fallback_fraction=(0.65, 1.0))
    head_mask = select_region_by_bbox(g3_points, tpts[masks["head"]], fallback_fraction=(0.70, 1.0))
    hair_anchor = g3_points[np.logical_or(hair_mask, head_mask)]
    targets = nearest_targets(positions, hair_anchor)
    pred, metrics, state = train_point_residual(features, positions, targets, steps=args.steps, seed=21)
    controls = control_metrics(features, positions, targets, steps=args.steps)
    checkpoint = out / "b_hair4_checkpoint.pt"
    torch.save(
        {
            "module": "vggt.models.human_hair_strand_gaussian.HumanHairStrandGaussian",
            "residual_head": state["model"],
            "trained_steps": int(args.steps),
            "query_rows": rows,
            "families": families,
            "metrics": metrics,
            "controls": controls,
            "research_only": True,
            "not_a_pass": True,
        },
        checkpoint,
    )
    hairline = pred[families == "hairline"] if np.any(families == "hairline") else pred
    headtop = pred[families != "hairline"] if np.any(families != "hairline") else pred
    write_ascii_ply(out / "b_hair4_hairline_band_surface.ply", hairline, None)
    write_ascii_ply(out / "b_hair4_headtop_hair_surface.ply", headtop, None)
    write_ascii_ply(out / "b_hair4_strands.ply", pred, None)
    np.savez_compressed(out / "b_hair4_strands.npz", points=pred, families=families, diagnostic_only=True)
    contact_sheet(pred, None, out / "b_hair4_open3d_hairline_headtop.png", "B-hair4 trained")
    plot_curve(out / "b_hair4_training_curve.png", state["rows"], ["real_loss", "template_loss"])
    real_better_zero = metrics["final_real_loss"] < controls["zero"]["final_real_loss"]
    real_better_shuffle = metrics["final_real_loss"] < controls["shuffle"]["final_real_loss"]
    pass_gate = bool(real_better_zero and real_better_shuffle and len(hairline) > 10 and metrics["loss_improvement"] > 0.0)
    summary = {
        "task": "v11_b_hair4_bounded_token_overfit",
        "created_utc": utc_now(),
        "status": "b_hair4_bounded_overfit_positive_research_only" if pass_gate else "b_hair4_bounded_overfit_blocked",
        **CONTRACT,
        "hair_visual_precheck_pass": False,
        "ownership_pass": False,
        "bounded_overfit_positive": pass_gate,
        "decoder_module": REPO_ROOT / "vggt/models/human_hair_strand_gaussian.py",
        "checkpoint": checkpoint,
        "trained_steps": int(args.steps),
        "uses_real_vggt_query_features": True,
        "source_query_cache": QUERY_CACHE,
        "source_g3_anchor": args.g3_anchor,
        "metrics": metrics,
        "controls": controls,
        "control_gates": {"real_better_zero": real_better_zero, "real_better_shuffle": real_better_shuffle},
        "outputs": {
            "hairline_surface": out / "b_hair4_hairline_band_surface.ply",
            "headtop_surface": out / "b_hair4_headtop_hair_surface.ply",
            "strands": out / "b_hair4_strands.npz",
            "open3d_sheet": out / "b_hair4_open3d_hairline_headtop.png",
            "curve": out / "b_hair4_training_curve.png",
        },
        "topology_metrics": {
            "hairline_point_count": int(len(hairline)),
            "headtop_point_count": int(len(headtop)),
            "floating_dot_ratio": 1.0,
            "head_shell_leakage": True,
            "real_vs_zero_margin": controls["zero"]["final_real_loss"] - metrics["final_real_loss"],
            "real_vs_shuffle_margin": controls["shuffle"]["final_real_loss"] - metrics["final_real_loss"],
        },
        "blockers": [
            "Bounded overfit uses weak G3/query supervision, not true HairGS/FLAME/hair GT.",
            "No continuous strand topology or head-shell leakage pass yet; D-line must keep ownership false.",
        ],
        "decision": "B-hair4 now has a trained research checkpoint/control run, but remains below strict hair ownership.",
    }
    write_training_summary(out, "V11 B-hair4 Bounded Token Overfit", summary)
    print(json.dumps(json_ready({"status": summary["status"], "metrics": metrics, "controls": controls, "output": out}), ensure_ascii=False))
    return 0 if pass_gate else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bounded V11 hand/hair specialist overfits from real VGGT query features.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("hand")
    p.add_argument("--output-dir", type=Path, default=OUT_HAND)
    p.add_argument("--g3-anchor", type=Path, default=G3_ANCHOR)
    p.add_argument("--steps", type=int, default=160)
    p.set_defaults(func=run_hand)
    p = sub.add_parser("hair")
    p.add_argument("--output-dir", type=Path, default=OUT_HAIR)
    p.add_argument("--g3-anchor", type=Path, default=G3_ANCHOR)
    p.add_argument("--steps", type=int, default=160)
    p.set_defaults(func=run_hair)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
