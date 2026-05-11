#!/usr/bin/env python
"""V39 adapter-only prior microfit.

Research-only route. This trains a compact, reloadable prior adapter on the
existing SMPL-X native prior maps and V24/V26/V29 teacher tensors. It does not
write formal predictions, candidate packages, teacher packages, registry files,
or strict pass state.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "surface_research_preflight_local" / "V39_adapter_microfit"
REPORT_JSON = ROOT / "reports" / "20260509_v39_adapter_microfit.json"
REPORT_MD = ROOT / "reports" / "20260509_v39_adapter_microfit.md"
CASE6 = ROOT / "output" / "training_cases" / "0012_11_frame0000_6views_smplx_native_prior_v15" / "inputs.npz"
CASE12 = ROOT / "output" / "training_cases" / "0012_11_frame0000_12views_sparseproto_smplxsurfacepose_v2" / "inputs.npz"
V24 = ROOT / "output" / "surface_research_preflight_local" / "V24_residual_teacher_v2" / "v24_residual_teacher_targets_v2.npz"
V26 = ROOT / "output" / "surface_research_preflight_local" / "V26_temporal_canonical_teacher" / "v26_temporal_canonical_teacher_targets.npz"
V29 = ROOT / "output" / "surface_research_preflight_local" / "V29_normal_route_rescue" / "v29_teacher_normals_world.npz"
V38 = ROOT / "output" / "surface_research_preflight_local" / "V38_prior_enabled_checkpoint_scaffold"

REGIONS = ("body", "head", "face", "left_hand", "right_hand")
CONTROLS = ("real", "zero", "shuffle", "random-region", "prior-dropout")
FORBIDDEN_NAMES = {"predictions.npz"}
FORBIDDEN_TOKENS = (
    "candidate_package",
    "teacher_package",
    "strict_registry",
    "strict_pass",
    "formal_candidate",
    "formal_teacher",
)


class PriorAdapterMicrofit(nn.Module):
    def __init__(self, channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 7, kernel_size=1),
        )
        self.region_bias = nn.Parameter(torch.zeros(len(REGIONS), 7))
        self.gate_logit = nn.Parameter(torch.tensor(-1.25))

    def forward(self, prior_maps: torch.Tensor, region_masks: torch.Tensor) -> dict[str, torch.Tensor]:
        # prior_maps: [V,C,H,W], region_masks: [R,V,H,W]
        raw = self.net(prior_maps)
        bias = torch.einsum("rvhw,ro->vohw", region_masks.float(), self.region_bias)
        gate = torch.sigmoid(self.gate_logit)
        out = gate * (raw + bias)
        return {
            "point_delta": out[:, 0:3].permute(0, 2, 3, 1).contiguous(),
            "depth_delta": out[:, 3],
            "normal_raw": out[:, 4:7].permute(0, 2, 3, 1).contiguous(),
            "gate": gate,
        }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def np_load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def normalise_np(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(n, eps)).astype(np.float32)


def choose_case() -> tuple[Path, dict[str, np.ndarray]]:
    if CASE6.is_file():
        return CASE6, np_load(CASE6)
    if CASE12.is_file():
        return CASE12, np_load(CASE12)
    raise FileNotFoundError(f"Missing prior cases: {CASE6} and {CASE12}")


def load_payload(max_views: int | None, target_size: int, device: torch.device) -> dict[str, Any]:
    case_path, case = choose_case()
    v24 = np_load(V24)
    v26 = np_load(V26)
    v29 = np_load(V29)

    view_count = int(v24["teacher_points_world"].shape[0])
    if max_views:
        view_count = min(view_count, max_views)
    prior_maps = np.asarray(case["prior_maps"][:view_count], dtype=np.float32)
    if prior_maps.shape[0] < view_count:
        raise ValueError(f"Prior case {case_path} has {prior_maps.shape[0]} views, need {view_count}")

    target_points = np.asarray(v24["teacher_points_world"][:view_count], dtype=np.float32)
    target_depths = np.asarray(v24["teacher_depths"][:view_count], dtype=np.float32)
    region_masks = np.asarray(v24["teacher_region_masks"][:, :view_count], dtype=bool)
    teacher_mask = np.asarray(v24["teacher_mask"][:view_count], dtype=bool)
    temporal_points = np.asarray(v26["target_frame_points"][:view_count], dtype=np.float32)
    if "v29_teacher_normals_world" in v29:
        target_normals = np.asarray(v29["v29_teacher_normals_world"][:view_count], dtype=np.float32)
    else:
        target_normals = np.asarray(v24["teacher_normals_world"][:view_count], dtype=np.float32)
    target_normals = normalise_np(target_normals)
    target_normals[~teacher_mask] = 0.0

    base_points = (0.78 * temporal_points + 0.22 * np.asarray(v24["teacher_points_world"][:view_count], dtype=np.float32)).astype(np.float32)
    base_depths = base_points[..., 2].astype(np.float32)
    base_normals = normalise_np(np.asarray(v24["teacher_normals_world"][:view_count], dtype=np.float32))
    base_normals[~teacher_mask] = 0.0

    def tchw(a: np.ndarray, mode: str = "bilinear") -> torch.Tensor:
        ten = torch.from_numpy(a)
        if ten.ndim == 4 and ten.shape[-1] in (3, 7):
            ten = ten.permute(0, 3, 1, 2).contiguous()
        ten = ten.to(torch.float32)
        if ten.shape[-2:] != (target_size, target_size):
            ten = F.interpolate(ten, size=(target_size, target_size), mode=mode, align_corners=False if mode == "bilinear" else None)
        return ten

    prior_t = tchw(prior_maps).to(device)
    tp = tchw(target_points).permute(0, 2, 3, 1).contiguous().to(device)
    bp = tchw(base_points).permute(0, 2, 3, 1).contiguous().to(device)
    tn = tchw(target_normals).permute(0, 2, 3, 1).contiguous().to(device)
    bn = tchw(base_normals).permute(0, 2, 3, 1).contiguous().to(device)
    td = tchw(target_depths[:, None]).squeeze(1).to(device)
    bd = tchw(base_depths[:, None]).squeeze(1).to(device)
    rm = torch.from_numpy(region_masks.astype(np.float32)).to(device)
    if rm.shape[-2:] != (target_size, target_size):
        rm = F.interpolate(rm.reshape(-1, 1, *rm.shape[-2:]), size=(target_size, target_size), mode="nearest").reshape(len(REGIONS), view_count, target_size, target_size)
    mask = torch.from_numpy(teacher_mask.astype(np.float32)).to(device)
    if mask.shape[-2:] != (target_size, target_size):
        mask = F.interpolate(mask[:, None], size=(target_size, target_size), mode="nearest").squeeze(1)
    return {
        "case_path": case_path,
        "prior_maps": prior_t,
        "target_points": tp,
        "target_depths": td,
        "target_normals": tn,
        "base_points": bp,
        "base_depths": bd,
        "base_normals": bn,
        "region_masks": rm,
        "mask": mask.bool(),
        "prior_channels": int(prior_t.shape[1]),
        "view_count": int(view_count),
        "target_size": int(target_size),
    }


def make_control(prior: torch.Tensor, name: str) -> torch.Tensor:
    if name == "real":
        return prior
    if name == "zero":
        return torch.zeros_like(prior)
    if name == "shuffle":
        return torch.roll(prior, shifts=1, dims=0)
    if name == "random-region":
        return torch.roll(prior, shifts=max(1, prior.shape[1] // 3), dims=1)
    if name == "prior-dropout":
        out = prior.clone()
        out[:, ::2] = 0
        return out
    raise ValueError(name)


def compute_outputs(model: PriorAdapterMicrofit, prior_maps: torch.Tensor, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    adapter = model(prior_maps, batch["region_masks"])
    points = batch["base_points"] + adapter["point_delta"]
    depth = batch["base_depths"] + adapter["depth_delta"]
    normals = F.normalize(batch["base_normals"] + 0.1 * adapter["normal_raw"], dim=-1)
    normals = torch.where(batch["mask"][..., None], normals, torch.zeros_like(normals))
    return {"points": points, "depth": depth, "normals": normals, "gate": adapter["gate"]}


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() <= 0:
        return x.new_tensor(0.0)
    if x.ndim == mask.ndim + 1:
        mask = mask[..., None]
    return (x * mask.float()).sum() / mask.float().sum().clamp_min(1.0) / (x.shape[-1] if x.ndim == mask.ndim else 1.0)


def loss_terms(outputs: dict[str, torch.Tensor], batch: dict[str, Any], weights: dict[str, float] | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    weights = weights or {r: 1.0 for r in REGIONS}
    mask = batch["mask"]
    point_l1 = torch.abs(outputs["points"] - batch["target_points"])
    depth_l1 = torch.abs(outputs["depth"] - batch["target_depths"])
    ncos = 1.0 - torch.clamp((outputs["normals"] * batch["target_normals"]).sum(dim=-1), -1.0, 1.0)
    region_losses: dict[str, torch.Tensor] = {}
    total = outputs["points"].new_tensor(0.0)
    for ridx, region in enumerate(REGIONS):
        rm = mask & (batch["region_masks"][ridx] > 0.5)
        p = masked_mean(point_l1, rm)
        d = masked_mean(depth_l1, rm)
        n = masked_mean(ncos, rm)
        reg_total = p + 0.35 * d + 0.05 * n
        region_losses[region] = reg_total
        total = total + float(weights.get(region, 1.0)) * reg_total
    total = total / max(1e-6, sum(float(weights.get(r, 1.0)) for r in REGIONS))
    total = total + 0.0005 * sum((p ** 2).mean() for p in outputs.values() if torch.is_tensor(p) and p.ndim > 0)
    details = {
        "total": total,
        "point_l1": masked_mean(point_l1, mask),
        "depth_l1": masked_mean(depth_l1, mask),
        "normal_cosine": masked_mean(ncos, mask),
        "region_loss": region_losses,
    }
    return total, details


def tensor_to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def evaluate(model: PriorAdapterMicrofit, batch: dict[str, Any], weights: dict[str, float] | None = None) -> dict[str, Any]:
    model.eval()
    metrics: dict[str, Any] = {}
    with torch.no_grad():
        for control in CONTROLS:
            outputs = compute_outputs(model, make_control(batch["prior_maps"], control), batch)
            _, details = loss_terms(outputs, batch, weights)
            metrics[control] = {
                "total": tensor_to_float(details["total"]),
                "point_l1": tensor_to_float(details["point_l1"]),
                "depth_l1": tensor_to_float(details["depth_l1"]),
                "normal_cosine": tensor_to_float(details["normal_cosine"]),
                "region_loss": {name: tensor_to_float(value) for name, value in details["region_loss"].items()},
                "gate": tensor_to_float(outputs["gate"]),
            }
    return metrics


def save_checkpoint(path: Path, model: PriorAdapterMicrofit, batch: dict[str, Any], kind: str, metrics: dict[str, Any], command: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_kind": kind,
            "state_dict": model.state_dict(),
            "model_class": "PriorAdapterMicrofit",
            "prior_channels": batch["prior_channels"],
            "regions": list(REGIONS),
            "source_case": str(batch["case_path"]),
            "teacher_sources": {"v24": str(V24), "v26": str(V26), "v29": str(V29)},
            "metrics": metrics,
            "command": command,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        path,
    )


def scan_forbidden(root: Path) -> list[dict[str, str]]:
    findings = []
    if not root.exists():
        return findings
    for p in root.rglob("*"):
        rel = p.relative_to(root).as_posix().lower()
        if p.is_file() and p.name.lower() in FORBIDDEN_NAMES:
            findings.append({"path": str(p), "reason": "forbidden predictions.npz"})
        hits = [t for t in FORBIDDEN_TOKENS if t in rel]
        if hits:
            findings.append({"path": str(p), "reason": "formal token in path: " + ",".join(hits)})
    return findings


def wins(metrics: dict[str, Any]) -> dict[str, bool]:
    real = metrics["real"]["total"]
    return {control: bool(real < metrics[control]["total"]) for control in CONTROLS if control != "real"}


def region_win_count(metrics: dict[str, Any]) -> dict[str, int]:
    out = {}
    for region in REGIONS:
        real = metrics["real"]["region_loss"][region]
        out[region] = int(sum(real < metrics[c]["region_loss"][region] for c in CONTROLS if c != "real"))
    return out


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V39 Adapter Microfit",
        "",
        f"status: `{summary['status']}`",
        f"v38_available: `{summary['v38_available']}`",
        f"case: `{summary['input_case']}`",
        f"command: `{summary['command']}`",
        "",
        "## Checkpoints",
        "",
        f"- adapter_only: `{summary['checkpoints']['adapter_only']}`",
        f"- delta_only: `{summary['checkpoints']['delta_only']}`",
        "",
        "## Control Metrics",
        "",
    ]
    for control, row in summary["metrics"].items():
        lines.append(f"- `{control}` total=`{row['total']:.6f}` point=`{row['point_l1']:.6f}` depth=`{row['depth_l1']:.6f}` normal=`{row['normal_cosine']:.6f}`")
        reg = ", ".join(f"{k}={v:.6f}" for k, v in row["region_loss"].items())
        lines.append(f"  - regions: {reg}")
    lines.extend(["", "## Decision", "", f"- real_wins_controls: `{summary['real_wins_controls']}`", f"- region_win_count: `{summary['region_win_count']}`", f"- next_route: `{summary['next_route']}`", f"- forbidden_hits: `{len(summary['forbidden_hits'])}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    batch = load_payload(args.max_views, args.target_size, device)
    model = PriorAdapterMicrofit(batch["prior_channels"], hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    weights = {r: 1.0 for r in REGIONS}
    curve = []
    for step in range(args.steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        outputs = compute_outputs(model, batch["prior_maps"], batch)
        loss, details = loss_terms(outputs, batch, weights)
        loss.backward()
        grad_norm = float(sum((p.grad.detach().norm().item() for p in model.parameters() if p.grad is not None)))
        opt.step()
        if step % max(1, args.steps // 12) == 0 or step == args.steps - 1:
            curve.append(
                {
                    "step": step,
                    "loss": tensor_to_float(loss),
                    "grad_norm": grad_norm,
                    "gate": tensor_to_float(outputs["gate"]),
                    "region_loss": {k: tensor_to_float(v) for k, v in details["region_loss"].items()},
                }
            )
    metrics = evaluate(model, batch, weights)
    adapter_path = OUT / "v39_adapter_only_checkpoint.pt"
    delta_path = OUT / "v39_delta_only_checkpoint.pt"
    command = "python tools\\v39_adapter_microfit.py " + " ".join(args.raw_args)
    save_checkpoint(adapter_path, model, batch, "v39_adapter_only_prior_microfit", metrics, command)
    save_checkpoint(delta_path, model, batch, "v39_delta_only_prior_effect", metrics, command)
    (OUT / "v39_training_curve.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=True) for x in curve) + "\n", encoding="utf-8")
    win_map = wins(metrics)
    reg_wins = region_win_count(metrics)
    critical_regions = ("head", "face", "left_hand", "right_hand")
    status = "DONE_PASS" if all(win_map.values()) and sum(reg_wins[r] >= 3 for r in critical_regions) >= 3 else "DONE_FAIL_ROUTED"
    summary = {
        "status": status,
        "research_only": True,
        "v38_available": V38.is_dir(),
        "v38_path": str(V38),
        "input_case": str(batch["case_path"]),
        "teacher_sources": {"v24": str(V24), "v26": str(V26), "v29": str(V29)},
        "prior_channels": batch["prior_channels"],
        "view_count": batch["view_count"],
        "target_size": batch["target_size"],
        "steps": args.steps,
        "hidden": args.hidden,
        "metrics": metrics,
        "real_wins_controls": win_map,
        "region_win_count": reg_wins,
        "checkpoints": {"adapter_only": str(adapter_path), "delta_only": str(delta_path)},
        "curve_path": str(OUT / "v39_training_curve.jsonl"),
        "forbidden_hits": scan_forbidden(OUT),
        "next_route": "V40_REGION_BALANCED_RESCUE" if status != "DONE_PASS" else "V42_READY_FROM_V39",
        "command": command,
    }
    write_json(OUT / "summary.json", summary)
    write_json(REPORT_JSON, summary)
    write_markdown(REPORT_MD, summary)
    print(json.dumps({"status": status, "next_route": summary["next_route"], "adapter": str(adapter_path)}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V39 adapter-only prior microfit.")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--steps", type=int, default=140)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--max-views", type=int, default=6)
    parser.add_argument("--seed", type=int, default=39039)
    args, raw = parser.parse_known_args()
    args.raw_args = raw
    run(args)


if __name__ == "__main__":
    main()
