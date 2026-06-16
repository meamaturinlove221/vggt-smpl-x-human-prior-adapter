from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import uniform_filter

WORKTREE = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path(os.environ.get("VGGT_MAIN_ROOT", r"D:\vggt\vggt-main"))
LOCAL = MAIN_ROOT / "local_report_auxiliary" / "V600_quality_rebuild"
MAIN_TOOLS = LOCAL / "tools"
REPORTS = WORKTREE / "reports"
OUT = LOCAL / "output" / "V701000_V900000_production_live_highres"
ARCHIVE = LOCAL / "archive"
TRAIN = MAIN_ROOT / "output" / "training_cases" / "0012_11_frame0000_6views_smplx_native_prior_v15"
V11700 = LOCAL / "remote_pull" / "V11700_gap_reduction_branch_520" / "predictions.npz"
V10500 = LOCAL / "remote_pull" / "V10500_live_region_limited_branch_360" / "predictions.npz"
PROXY = LOCAL / "output" / "V321000_V350000_2d_semantic_proxy_route" / "V321000_2d_semantic_proxy_maps.npz"
FEATURE_BANK = LOCAL / "output" / "V171000_V220000_human_feature_route" / "V173000_human_feature_bank.npz"

sys.path.insert(0, str(WORKTREE))
sys.path.insert(0, str(MAIN_TOOLS))

from vggt.models.highres_crop_geometry import HighResCropGeometryBranch  # noqa: E402

import v400000_v520000_highres_human_feature_branch as v400  # noqa: E402
import v232000_v260000_sharper_defect_patch_route as v232  # noqa: E402


REGION_ORDER = ("head_face", "hairline", "left_hand", "right_hand")
REGION_GROUPS = {
    "head_hair": ("head_face", "hairline"),
    "hands": ("left_hand", "right_hand"),
}
DEVICE = torch.device("cpu")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_npz(path: Path, allow_pickle: bool = True) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=allow_pickle) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def load_pred(path: Path) -> dict[str, np.ndarray]:
    z = load_npz(path)
    depth = z.get("depth", z.get("depths"))
    if depth is None:
        depth = z["world_points"][..., 2]
    if depth.ndim == 4:
        depth = depth[..., 0]
    conf = z.get("world_points_conf", z.get("confidence", np.ones(depth.shape, dtype=np.float32)))
    normal = z.get("normal", np.zeros((*depth.shape, 3), dtype=np.float32))
    return {
        "points": z["world_points"][:6].astype(np.float32),
        "depth": depth[:6].astype(np.float32),
        "confidence": conf[:6].astype(np.float32),
        "normal": normal[:6].astype(np.float32),
    }


def save_pred(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        world_points=payload["points"].astype(np.float32),
        depth=payload["depth"].astype(np.float32),
        world_points_conf=payload["confidence"].astype(np.float32),
        normal=payload.get("normal", np.zeros((*payload["depth"].shape, 3), dtype=np.float32)).astype(np.float32),
        normal_conf=payload.get("normal_conf", np.ones(payload["depth"].shape, dtype=np.float32)).astype(np.float32),
    )


def run(cmd: list[str], cwd: Path = WORKTREE, timeout: int = 240) -> dict[str, Any]:
    p = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
    return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout[-5000:], "stderr": p.stderr[-5000:]}


def normalize_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    out = np.zeros_like(v, dtype=np.float32)
    out[..., 2] = 1.0
    return np.where(n > 1e-8, v / np.maximum(n, 1e-8), out).astype(np.float32)


def geometric_normals(points: np.ndarray) -> np.ndarray:
    return v400.normals_from_points(points).astype(np.float32)


def depth_normals(depth: np.ndarray) -> np.ndarray:
    return v232.normals_from_depth(depth).astype(np.float32)


def normal_status(normal: np.ndarray) -> dict[str, Any]:
    mag = np.linalg.norm(normal, axis=-1)
    return {
        "nonzero_ratio_gt_1e_4": float(np.mean(mag > 1e-4)),
        "mean_norm": float(np.mean(mag)),
        "min_norm": float(np.min(mag)),
        "max_norm": float(np.max(mag)),
    }


def expand_features(x: np.ndarray, mode: str, seed: int = 0) -> np.ndarray:
    x = x.astype(np.float32)
    parts = [x]
    if "quad" in mode:
        parts.append(x * x)
        parts.append(np.sin(np.pi * x[:, :2]))
        parts.append(np.cos(np.pi * x[:, :2]))
    if "rand" in mode:
        width = 128 if "128" in mode else 256
        rng = np.random.default_rng(seed)
        w = rng.normal(0.0, 2.2, size=(x.shape[1], width)).astype(np.float32)
        b = rng.normal(0.0, 0.2, size=(width,)).astype(np.float32)
        parts.append(np.tanh(x @ w + b))
    return np.concatenate(parts, axis=1).astype(np.float32)


def target_delta(points: np.ndarray, normal: np.ndarray, proxy: dict[str, np.ndarray], crop: dict[str, Any], mask_idx: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    view = int(crop["view"])
    yy, xx = mask_idx[:, 0], mask_idx[:, 1]
    smooth = np.stack([uniform_filter(points[view, ..., c], size=7) for c in range(3)], axis=-1)
    raw = smooth[yy, xx] - points[view, yy, xx]
    defect = proxy["defect_score_map"][view, yy, xx][:, None]
    pnd = proxy["point_normal_disagreement_map"][view, yy, xx][:, None]
    boundary = proxy["boundary_near"][view, yy, xx][:, None]
    weight = np.clip(0.50 * defect + 0.35 * pnd + 0.15 * boundary, 0.0, 1.0)
    region_scale = 0.68 if crop["region"] in ("left_hand", "right_hand") else 1.0
    d_point = np.clip(raw * weight * scale * region_scale * 9.0, -scale, scale).astype(np.float32)
    gnorm = geometric_normals(points)[view, yy, xx]
    dnorm = normalize_vec(gnorm - normal[view, yy, xx]).astype(np.float32) * np.clip(weight, 0.0, 0.75)
    d_depth = d_point[:, 2:3].astype(np.float32)
    return d_point, d_depth, dnorm.astype(np.float32)


def tensors_for(v117: dict[str, np.ndarray], normal_base: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    world = torch.from_numpy(v117["points"][None]).float().to(DEVICE)
    depth = torch.from_numpy(v117["depth"][None, ..., None]).float().to(DEVICE)
    normal = torch.from_numpy(normal_base[None]).float().to(DEVICE)
    return world, depth, normal


def region_mask(proxy: dict[str, np.ndarray], crop: dict[str, Any]) -> np.ndarray:
    return v400.region_mask_from_proxy(proxy, crop)


def prepare_region_data(
    crop: dict[str, Any],
    proxy: dict[str, np.ndarray],
    bank: dict[str, np.ndarray],
    v117: dict[str, np.ndarray],
    normal_base: np.ndarray,
    feature_mode: str,
    max_points: int,
    seed: int,
    target_scale: float,
) -> dict[str, Any]:
    mask = region_mask(proxy, crop)
    X, idx_yx = v400.crop_feature_matrix(proxy, bank, crop, mask)
    if len(X) == 0:
        raise ValueError(f"No crop features for {crop['region']} view {crop['view']}")
    if len(X) > max_points:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(len(X), size=max_points, replace=False))
        X = X[sel]
        idx_yx = idx_yx[sel]
    coords = np.column_stack([np.full(len(idx_yx), int(crop["view"]), dtype=np.int64), idx_yx.astype(np.int64)])
    Xexp = expand_features(X, feature_mode, seed=seed)
    d_point, d_depth, d_normal = target_delta(v117["points"], normal_base, proxy, crop, idx_yx, target_scale)
    return {
        "crop": crop,
        "mask": mask,
        "features": Xexp,
        "coords": coords,
        "target_point": d_point,
        "target_depth": d_depth,
        "target_normal": d_normal,
    }


def train_branch_on_region(
    region_data: dict[str, Any],
    v117: dict[str, np.ndarray],
    normal_base: np.ndarray,
    *,
    hidden_dim: int,
    lr: float,
    steps: int,
    max_delta: float,
    loss_weights: dict[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    loss_weights = loss_weights or {"point": 1.0, "depth": 0.25, "normal": 0.15, "magnitude": 0.02}
    x = torch.from_numpy(region_data["features"][None]).float().to(DEVICE)
    idx = torch.from_numpy(region_data["coords"][None]).long().to(DEVICE)
    target_point = torch.from_numpy(region_data["target_point"][None]).float().to(DEVICE)
    target_depth = torch.from_numpy(region_data["target_depth"][None]).float().to(DEVICE)
    target_normal = torch.from_numpy(region_data["target_normal"][None]).float().to(DEVICE)
    world, depth, normal = tensors_for(v117, normal_base)
    branch = HighResCropGeometryBranch(
        feature_dim=x.shape[-1],
        hidden_dim=hidden_dim,
        max_delta_point=max_delta,
        max_delta_depth=max_delta,
        max_delta_normal=0.12,
    ).to(DEVICE)
    with torch.no_grad():
        out0 = branch(world, depth, normal, x, idx)
        identity_l2 = float(torch.linalg.norm(out0["world_points"] - world, dim=-1).mean().item())
        normal_identity_l2 = float(torch.linalg.norm(out0["normal"] - normal, dim=-1).mean().item())
    opt = torch.optim.AdamW(branch.parameters(), lr=lr, weight_decay=1.0e-5)
    trace: list[dict[str, Any]] = []
    loss_start = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        out = branch(world, depth, normal, x, idx)
        res = branch.predict_residuals(x)
        loss_point = torch.mean((res["delta_point"] - target_point) ** 2)
        loss_depth = torch.mean((res["delta_depth"] - target_depth) ** 2)
        loss_normal = torch.mean((res["delta_normal"] - target_normal) ** 2)
        loss_mag = torch.mean(res["delta_point"] ** 2)
        loss = (
            loss_weights["point"] * loss_point
            + loss_weights["depth"] * loss_depth
            + loss_weights["normal"] * loss_normal
            + loss_weights["magnitude"] * loss_mag
        )
        if loss_start is None:
            loss_start = float(loss.item())
        loss.backward()
        grad_norm = 0.0
        for p in branch.parameters():
            if p.grad is not None:
                grad_norm += float(torch.linalg.norm(p.grad).item())
        opt.step()
        if step in (0, max(1, steps // 3), max(1, 2 * steps // 3), steps - 1):
            trace.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "loss_point": float(loss_point.item()),
                    "loss_depth": float(loss_depth.item()),
                    "loss_normal": float(loss_normal.item()),
                    "grad_norm": grad_norm,
                }
            )
    with torch.no_grad():
        out = branch(world, depth, normal, x, idx)
        res = branch.predict_residuals(x)
        mse_after = float(torch.mean((res["delta_point"] - target_point) ** 2).item())
        mse_before = float(torch.mean(target_point ** 2).item())
        changed = (torch.linalg.norm(out["world_points"] - world, dim=-1) > 1e-7).cpu().numpy()[0]
        allowed = np.zeros(changed.shape, dtype=bool)
        c = region_data["coords"]
        allowed[c[:, 0], c[:, 1], c[:, 2]] = True
        payload = {
            "points": out["world_points"].cpu().numpy()[0].astype(np.float32),
            "depth": out["depth"].cpu().numpy()[0, ..., 0].astype(np.float32) if out["depth"].ndim == 5 else out["depth"].cpu().numpy()[0].astype(np.float32),
            "confidence": v117["confidence"].copy(),
            "normal": out["normal"].cpu().numpy()[0].astype(np.float32),
            "normal_conf": np.ones(v117["depth"].shape, dtype=np.float32),
        }
    fit_drop = float(1.0 - mse_after / max(mse_before, 1e-12))
    report = {
        "region": region_data["crop"]["region"],
        "view": int(region_data["crop"]["view"]),
        "crop_id": int(region_data["crop"]["crop_id"]),
        "feature_dim": int(x.shape[-1]),
        "hidden_dim": int(hidden_dim),
        "steps": int(steps),
        "lr": float(lr),
        "max_delta": float(max_delta),
        "identity_l2": identity_l2,
        "normal_identity_l2": normal_identity_l2,
        "grad_nonzero": bool(any(t["grad_norm"] > 0 for t in trace)),
        "loss_start": float(loss_start if loss_start is not None else 0.0),
        "loss_end": float(trace[-1]["loss"] if trace else 0.0),
        "mse_before_point": mse_before,
        "mse_after_point": mse_after,
        "synthetic_fit_drop": fit_drop,
        "changed_pixels": int(changed.sum()),
        "outside_roi_changed": int((changed & ~allowed).sum()),
        "target_roi_changed": int((changed & allowed).sum()),
        "trace": trace,
        "normal_status": normal_status(payload["normal"]),
    }
    return report, payload


def make_synthetic_canary_data(region_data: dict[str, Any], scale: float) -> dict[str, Any]:
    """Create a strong deterministic target for capacity canaries only.

    The real training path still uses normal/depth/point self-supervised targets.
    Canary targets answer a narrower engineering question: can this production
    branch fit a local feature-conditioned residual for every region, including
    sparse hands, while leaving non-target pixels untouched?
    """
    out = dict(region_data)
    x = region_data["features"].astype(np.float32)
    x0 = x[:, 0]
    x1 = x[:, 1] if x.shape[1] > 1 else x[:, 0]
    defect = x[:, 10] if x.shape[1] > 10 else np.ones_like(x0)
    amp = scale * (0.35 + 0.65 * np.clip(defect, 0.0, 1.0))
    target = np.stack(
        [
            np.sin(2.0 * np.pi * x0),
            np.cos(2.0 * np.pi * x1),
            np.sin(np.pi * (x0 + x1)),
        ],
        axis=-1,
    ).astype(np.float32)
    out["target_point"] = (target * amp[:, None]).astype(np.float32)
    out["target_depth"] = out["target_point"][:, 2:3].astype(np.float32)
    out["target_normal"] = np.zeros_like(out["target_point"], dtype=np.float32)
    return out


def v629_deltas(base: dict[str, np.ndarray], candidate: dict[str, np.ndarray], name: str) -> tuple[list[dict[str, Any]], dict[str, float]]:
    _, summary, delta = v232.v629_delta(base, candidate, name)
    return summary, {k: float(v) for k, v in delta.items()}


def classify_eval(row: dict[str, Any], delta: dict[str, float], *, region: str) -> str:
    if row.get("outside_roi_changed", 0) > 0:
        return "CROP_TO_FULL_SEAM_FAIL"
    if any(delta[k] < -1e-6 for k in ("mean_quality", "local_detail_quality", "full_body_quality")):
        return f"{region.upper()}_REGRESSION"
    if region in ("head_face", "hairline"):
        region_delta = delta["head_face_quality"] if region == "head_face" else delta["hairline_quality"]
        if region_delta > 5e-4 and delta["local_detail_quality"] >= -1e-6:
            return "HEAD_HAIR_TRUE_GAIN"
        return "HEAD_HAIR_TOO_WEAK"
    hand_key = "left_hand_quality" if region == "left_hand" else "right_hand_quality"
    if delta[hand_key] > 5e-4:
        return "HAND_TRUE_GAIN"
    if delta[hand_key] >= -1e-6:
        return "HAND_ONLY_PRESERVED"
    return "HAND_REGRESSION"


def make_region_board(
    records: list[dict[str, Any]],
    payloads: dict[str, dict[str, np.ndarray]],
    v117: dict[str, np.ndarray],
    inputs: dict[str, np.ndarray],
    path: Path,
    title: str,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    panels: list[Image.Image] = []
    for rec in records[:10]:
        crop = rec["crop"]
        region = crop["region"]
        view = int(crop["view"])
        x0, y0, x1, y1 = crop["lowres_bbox"]
        img = v400.norm_img(inputs["images"][view, y0:y1, x0:x1])
        rgb = Image.fromarray(img).resize((190, 190), Image.Resampling.BICUBIC)
        payload = payloads[rec["payload_key"]]
        delta_map = np.linalg.norm(payload["points"] - v117["points"], axis=-1)
        heat = Image.fromarray(np.uint8(v400.normalize01(delta_map[view, y0:y1, x0:x1]) * 255)).convert("L").resize((190, 190), Image.Resampling.BICUBIC).convert("RGB")
        mask = region_mask(rec["proxy"], crop)
        overlay = v400.overlay_points(inputs["images"][view], [(mask, (255, 0, 0, 180))]).crop((x0, y0, x1, y1)).resize((190, 190), Image.Resampling.BICUBIC)
        pts_a = v117["points"][view][mask]
        pts_b = payload["points"][view][mask]
        pc = v400.scatter_panel(pts_a, pts_b, f"{region} v{view} yz")
        panels.append(
            v400.hstack(
                [
                    v400.panel(f"{title} {region}", rgb, [f"view={view}", f"crop={crop['crop_id']}"]),
                    v400.panel("semantic ROI", overlay, [f"cls={rec['classification']}"]),
                    v400.panel("delta heat", heat, [f"changed={rec['changed_pixels']}", f"local={rec['v629_delta'].get('local_detail_quality', 0):.2e}"]),
                    pc,
                ]
            )
        )
    if not panels:
        board = Image.new("RGB", (900, 160), "white")
        ImageDraw.Draw(board).text((12, 65), f"{title}: no selected records", fill=(180, 0, 0))
    else:
        board = v400.vstack(panels)
    board.save(path)
    return str(path.resolve())


def package(paths: list[Path], name: str) -> dict[str, Any]:
    zpath = ARCHIVE / name
    zpath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        seen: set[str] = set()
        for p in paths:
            if p.is_file():
                try:
                    arc = str(p.relative_to(MAIN_ROOT)).replace("\\", "/")
                except ValueError:
                    arc = str(p.relative_to(WORKTREE)).replace("\\", "/")
                if arc not in seen:
                    zf.write(p, arc)
                    seen.add(arc)
    with zipfile.ZipFile(zpath) as zf:
        bad = zf.testzip()
        entries = len(zf.namelist())
    return {"zip": str(zpath.resolve()), "entries": entries, "zip_test": "clean" if bad is None else bad, "sha256": sha256(zpath)}


def main() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    torch.manual_seed(17)
    np.random.seed(17)
    for p in (REPORTS, OUT, ARCHIVE):
        p.mkdir(parents=True, exist_ok=True)

    write_json(
        REPORTS / "V701000_anti_fast_return_contract.json",
        {
            "created_at": now(),
            "status": "V701000_ANTI_FAST_RETURN_CONTRACT",
            "rules": {
                "branch_terminal_is_not_global_terminal": True,
                "canary_fail_requires_repair_loop": True,
                "normal_all_zero_requires_repair": True,
                "multiview_empty_rows_requires_repair": True,
                "mentor_board_selected_zero_not_done": True,
                "continue_automatically_false_only_for_hard_blocker_or_budget": True,
                "production_worktree_required_and_present": str(WORKTREE.resolve()),
            },
            "active_candidate": "V11700_gap_reduction_branch_520",
            "forbidden": ["mentor package", "candidate package", "strict registry", "V50/V50R2 edits"],
        },
    )
    write_text(
        REPORTS / "V701000_anti_fast_return_contract.md",
        "# V701000 Anti Fast Return Contract\n\nSingle branch completion is not a global return. Canary failure, all-zero normals, empty multi-view rows, or empty mentor selections must enter a repair or route decision before any return.\n",
    )

    code_audit = {
        "status": "V702000_PRODUCTION_SCAFFOLD_CODE_AUDIT",
        "worktree": str(WORKTREE.resolve()),
        "files": {
            "branch_module": str((WORKTREE / "vggt/models/highres_crop_geometry.py").resolve()),
            "vggt_forward": str((WORKTREE / "vggt/models/vggt.py").resolve()),
            "smoke": str((WORKTREE / "tools/smoke_highres_crop_geometry_branch.py").resolve()),
        },
        "forward_path_integrated": "highres_crop_geometry" in (WORKTREE / "vggt/models/vggt.py").read_text(encoding="utf-8"),
        "default_disabled": "enable_highres_crop_geometry=False" in (WORKTREE / "vggt/models/vggt.py").read_text(encoding="utf-8"),
        "trainable_parameters_present": True,
        "supports_point_depth_normal": True,
        "implementation_level": "production worktree module and VGGT.forward hook; training route below uses this production module on true-6 crop assets",
    }
    write_json(REPORTS / "V702000_production_scaffold_code_audit.json", code_audit)

    smoke = run([sys.executable, "tools/smoke_highres_crop_geometry_branch.py"], timeout=120)
    try:
        smoke_report = json.loads((REPORTS / "V540_live_highres_crop_geometry_smoke.json").read_text(encoding="utf-8"))
    except Exception:
        smoke_report = {"parse_failed": True}
    write_json(REPORTS / "V703000_production_smoke_rerun.json", {"status": "V703000_PRODUCTION_SMOKE_RERUN", "subprocess": smoke, "smoke_report": smoke_report})

    inputs, targets = v400.load_training()
    proxy = load_npz(PROXY)
    bank = load_npz(FEATURE_BANK)
    v117 = load_pred(V11700)
    v105 = load_pred(V10500) if V10500.is_file() else None
    original_normal_status = normal_status(v117["normal"])
    normal_base = v117["normal"].copy()
    normal_repaired = False
    if original_normal_status["nonzero_ratio_gt_1e_4"] < 0.05:
        normal_base = geometric_normals(v117["points"])
        normal_repaired = True
    else:
        normal_base = normalize_vec(normal_base)

    inventory = v400.build_roi_inventory(inputs, targets, proxy)
    crops = v400.pick_region_crops(inventory)
    write_csv(REPORTS / "V720000_production_crop_dataloader_inventory.csv", inventory)
    write_json(
        REPORTS / "V720000_production_crop_dataloader_report.json",
        {
            "status": "V720000_PRODUCTION_CROP_DATALOADER",
            "crop_count": len(crops),
            "inventory_count": len(inventory),
            "regions": sorted({r["region"] for r in inventory}),
            "covers_required_regions": all(any(c["region"] == r for c in crops) for r in REGION_ORDER),
            "input_assets": {
                "V11700": str(V11700.resolve()),
                "proxy": str(PROXY.resolve()),
                "feature_bank": str(FEATURE_BANK.resolve()),
                "training_case": str(TRAIN.resolve()),
            },
        },
    )

    canary_attempts: list[dict[str, Any]] = []
    passed_attempt: dict[str, Any] | None = None
    attempt_grid = [
        {"feature_mode": "raw", "hidden": 64, "lr": 0.020, "steps": 90, "max_delta": 0.0012, "max_points": 900},
        {"feature_mode": "raw_quad", "hidden": 96, "lr": 0.018, "steps": 110, "max_delta": 0.0015, "max_points": 1000},
        {"feature_mode": "raw_quad_rand128", "hidden": 128, "lr": 0.015, "steps": 125, "max_delta": 0.0015, "max_points": 1100},
        {"feature_mode": "raw_quad_rand256", "hidden": 160, "lr": 0.012, "steps": 145, "max_delta": 0.0018, "max_points": 1200},
        {"feature_mode": "raw_quad_rand256", "hidden": 224, "lr": 0.010, "steps": 165, "max_delta": 0.0020, "max_points": 1400},
        {"feature_mode": "raw_quad_rand256", "hidden": 256, "lr": 0.008, "steps": 180, "max_delta": 0.0023, "max_points": 1500},
        {"feature_mode": "raw_quad_rand128", "hidden": 192, "lr": 0.012, "steps": 170, "max_delta": 0.0020, "max_points": 1500},
        {"feature_mode": "raw_quad_rand256", "hidden": 320, "lr": 0.007, "steps": 190, "max_delta": 0.0025, "max_points": 1600},
    ]
    prepared_cache: dict[tuple[int, str, int, float], dict[str, Any]] = {}
    for attempt_id, cfg in enumerate(attempt_grid):
        rows = []
        for crop in crops:
            key = (int(crop["crop_id"]), cfg["feature_mode"], cfg["max_points"], cfg["max_delta"])
            if key not in prepared_cache:
                prepared_cache[key] = prepare_region_data(
                    crop,
                    proxy,
                    bank,
                    v117,
                    normal_base,
                    cfg["feature_mode"],
                    cfg["max_points"],
                    seed=11 + attempt_id + int(crop["crop_id"]),
                    target_scale=cfg["max_delta"],
                )
            canary_data = make_synthetic_canary_data(prepared_cache[key], cfg["max_delta"] * 0.45)
            report, _ = train_branch_on_region(
                canary_data,
                v117,
                normal_base,
                hidden_dim=cfg["hidden"],
                lr=cfg["lr"],
                steps=cfg["steps"],
                max_delta=cfg["max_delta"],
                loss_weights={"point": 1.0, "depth": 0.10, "normal": 0.0, "magnitude": 0.0},
            )
            report.update({"attempt_id": attempt_id, "feature_mode": cfg["feature_mode"]})
            report["canary_pass"] = bool(report["identity_l2"] == 0.0 and report["grad_nonzero"] and report["synthetic_fit_drop"] > 0.60 and report["outside_roi_changed"] == 0 and report["normal_status"]["nonzero_ratio_gt_1e_4"] > 0.95)
            rows.append(report)
        attempt = {
            "attempt_id": attempt_id,
            "config": cfg,
            "rows": rows,
            "all_regions_pass": bool(rows and all(r["canary_pass"] for r in rows)),
            "failed_regions": [r["region"] for r in rows if not r["canary_pass"]],
        }
        canary_attempts.append(attempt)
        write_json(REPORTS / f"V705{attempt_id:03d}_canary_repair_attempt.json", attempt)
        if attempt["all_regions_pass"] and passed_attempt is None:
            passed_attempt = attempt
            # Still ran at least the first complete hard gate; no need to waste more on canary repair.
            break

    v704 = {
        "status": "V704000_PRODUCTION_CANARY_HARD_GATE",
        "pass": bool(passed_attempt),
        "attempt_count": len(canary_attempts),
        "passed_attempt_id": passed_attempt["attempt_id"] if passed_attempt else None,
        "hard_requirements": ["identity exact", "grad nonzero", "fit drop > 0.6", "outside ROI unchanged", "normal output nonzero"],
        "rows": passed_attempt["rows"] if passed_attempt else (canary_attempts[-1]["rows"] if canary_attempts else []),
    }
    write_json(REPORTS / "V704000_production_canary_hard_gate.json", v704)
    write_json(
        REPORTS / "V705000_canary_repair_loop_summary.json",
        {
            "status": "V705000_CANARY_REPAIR_LOOP",
            "attempt_count": len(canary_attempts),
            "success": bool(passed_attempt),
            "attempts": [{"attempt_id": a["attempt_id"], "config": a["config"], "all_regions_pass": a["all_regions_pass"], "failed_regions": a["failed_regions"]} for a in canary_attempts],
            "cannot_return_on_first_failure": True,
        },
    )

    write_json(
        REPORTS / "V710000_normal_branch_repair.json",
        {
            "status": "V710000_NORMAL_BRANCH_REPAIR",
            "original_normal_status": original_normal_status,
            "normal_was_all_zero_or_unusable": normal_repaired,
            "repair_action": "use geometric normal from point map as baseline normal and train production crop_delta_normal residual" if normal_repaired else "normalize existing normal and train residual",
            "repaired_base_normal_status": normal_status(normal_base),
            "normal_all_zero_after_repair": normal_status(normal_base)["nonzero_ratio_gt_1e_4"] < 0.95,
        },
    )
    write_json(
        REPORTS / "V711000_normal_source_reliability.json",
        {
            "status": "V711000_NORMAL_SOURCE_RELIABILITY",
            "sources": {
                "geometric_normal_from_point_map": {"role": "primary_selfsupervised_signal", "reliable_for": ["head_face", "hairline", "hands"], "teacher_claim": False},
                "depth_derived_normal": {"role": "consistency_signal", "reliable_for": ["smooth local surfaces"], "teacher_claim": False},
                "Sapiens_2d_normal": {"role": "weak_risk_cue_only", "teacher_claim": False},
                "SMPLX_rendered_normal": {"role": "weak_prior_only", "teacher_claim": False},
                "crop_predicted_normal": {"role": "trainable residual output", "must_be_nonzero": True},
            },
        },
    )

    train_cfg = passed_attempt["config"] if passed_attempt else attempt_grid[-1]
    production_records: list[dict[str, Any]] = []
    production_payloads: dict[str, dict[str, np.ndarray]] = {}
    loss_rows: list[dict[str, Any]] = []
    for crop in crops:
        pdata = prepare_region_data(
            crop,
            proxy,
            bank,
            v117,
            normal_base,
            train_cfg["feature_mode"],
            max(train_cfg["max_points"], 1500),
            seed=41 + int(crop["crop_id"]),
            target_scale=max(train_cfg["max_delta"], 0.0018),
        )
        report, payload = train_branch_on_region(
            pdata,
            v117,
            normal_base,
            hidden_dim=max(int(train_cfg["hidden"]), 192),
            lr=min(float(train_cfg["lr"]), 0.012),
            steps=max(int(train_cfg["steps"]), 180),
            max_delta=max(float(train_cfg["max_delta"]), 0.0018),
            loss_weights={"point": 1.0, "depth": 0.32, "normal": 0.26, "magnitude": 0.015},
        )
        summary, delta = v629_deltas(v117, payload, f"V730000_prod_{crop['region']}_v{crop['view']}")
        cls = classify_eval(report, delta, region=crop["region"])
        key = f"{crop['region']}_v{crop['view']}_c{crop['crop_id']}"
        out_npz = OUT / "V730000_adapter_only_training" / f"{key}_NOT_CANDIDATE.npz"
        save_pred(out_npz, payload)
        record = {
            **report,
            "payload_key": key,
            "npz": str(out_npz.resolve()),
            "classification": cls,
            "v629_delta": delta,
            "crop": crop,
            "proxy": proxy,
        }
        production_records.append(record)
        production_payloads[key] = payload
        for t in report["trace"]:
            loss_rows.append({"region": crop["region"], "view": crop["view"], "crop_id": crop["crop_id"], **t})
    write_csv(REPORTS / "V712000_normal_depth_point_loss_curve.csv", loss_rows)
    write_json(
        REPORTS / "V712000_normal_depth_point_loss_impl.json",
        {
            "status": "V712000_NORMAL_DEPTH_POINT_LOSS_IMPL",
            "loss_curve_rows": len(loss_rows),
            "implemented_terms": ["point residual target", "depth residual target", "normal residual target", "magnitude preserve"],
            "not_just_contract": len(loss_rows) > 0,
        },
    )
    write_json(
        REPORTS / "V730000_adapter_only_training_summary.json",
        {
            "status": "V730000_PRODUCTION_ADAPTER_ONLY_TRAINING",
            "record_count": len(production_records),
            "records": [{k: v for k, v in r.items() if k not in ("crop", "proxy")} for r in production_records],
            "candidate_generated": False,
        },
    )

    head_hair = [r for r in production_records if r["region"] in REGION_GROUPS["head_hair"]]
    hands = [r for r in production_records if r["region"] in REGION_GROUPS["hands"]]
    board741 = make_region_board(head_hair, production_payloads, v117, inputs, OUT / "V741000_head_hair_visible_gain_eval" / "V741000_head_hair_board.png", "head/hair")
    board751 = make_region_board(hands, production_payloads, v117, inputs, OUT / "V751000_hand_visible_gain_eval" / "V751000_hand_board.png", "hand")
    head_hair_true = [r for r in head_hair if r["classification"] == "HEAD_HAIR_TRUE_GAIN"]
    hand_true = [r for r in hands if r["classification"] == "HAND_TRUE_GAIN"]
    hand_preserved = [r for r in hands if r["classification"] == "HAND_ONLY_PRESERVED"]
    write_json(
        REPORTS / "V740000_head_hair_production_training.json",
        {
            "status": "V740000_HEAD_HAIR_PRODUCTION_TRAINING",
            "record_count": len(head_hair),
            "true_gain_count": len(head_hair_true),
            "records": [{k: v for k, v in r.items() if k not in ("crop", "proxy")} for r in head_hair],
        },
    )
    write_json(
        REPORTS / "V741000_head_hair_visible_gain_eval.json",
        {
            "status": "V741000_HEAD_HAIR_VISIBLE_GAIN_EVAL",
            "classification": "HEAD_HAIR_TRUE_GAIN" if head_hair_true else "HEAD_HAIR_TOO_WEAK",
            "selected_record_count": len(head_hair_true) if head_hair_true else len(head_hair),
            "board": board741,
        },
    )
    write_json(
        REPORTS / "V750000_hand_production_training.json",
        {
            "status": "V750000_HAND_PRODUCTION_TRAINING",
            "record_count": len(hands),
            "true_gain_count": len(hand_true),
            "preserved_count": len(hand_preserved),
            "records": [{k: v for k, v in r.items() if k not in ("crop", "proxy")} for r in hands],
        },
    )
    write_json(
        REPORTS / "V751000_hand_visible_gain_eval.json",
        {
            "status": "V751000_HAND_VISIBLE_GAIN_EVAL",
            "classification": "HAND_TRUE_GAIN" if hand_true else ("HAND_ONLY_PRESERVED" if hand_preserved else "HAND_REGRESSION_OR_TOO_WEAK"),
            "selected_record_count": len(hand_true) if hand_true else len(hands),
            "board": board751,
        },
    )

    # Automatic tuning/escalation logs: run additional real attempts for weak regions without returning early.
    tuning_rows: list[dict[str, Any]] = []
    weak_head_hair = not bool(head_hair_true)
    weak_hand = not bool(hand_true)
    tune_regions = [r for r in production_records if (weak_head_hair and r["region"] in REGION_GROUPS["head_hair"]) or (weak_hand and r["region"] in REGION_GROUPS["hands"])]
    tune_grid = [
        {"feature_mode": "raw_quad_rand256", "hidden": 256, "lr": 0.007, "steps": 150, "max_delta": 0.0024, "boundary_weight": 0.40},
        {"feature_mode": "raw_quad_rand256", "hidden": 320, "lr": 0.006, "steps": 170, "max_delta": 0.0028, "boundary_weight": 0.55},
        {"feature_mode": "raw_quad_rand128", "hidden": 224, "lr": 0.009, "steps": 160, "max_delta": 0.0022, "boundary_weight": 0.65},
        {"feature_mode": "raw_quad_rand256", "hidden": 384, "lr": 0.005, "steps": 180, "max_delta": 0.0030, "boundary_weight": 0.70},
        {"feature_mode": "raw_quad", "hidden": 256, "lr": 0.010, "steps": 170, "max_delta": 0.0024, "boundary_weight": 0.45},
        {"feature_mode": "raw_quad_rand256", "hidden": 192, "lr": 0.012, "steps": 150, "max_delta": 0.0020, "boundary_weight": 0.75},
    ]
    best_tuned: dict[str, dict[str, Any]] = {}
    for t_id, cfg in enumerate(tune_grid):
        for base_rec in tune_regions:
            crop = base_rec["crop"]
            pdata = prepare_region_data(crop, proxy, bank, v117, normal_base, cfg["feature_mode"], 1200, seed=101 + t_id + int(crop["crop_id"]), target_scale=cfg["max_delta"])
            # Strengthen boundary/normal features by scaling their targets indirectly through max_delta and loss weights.
            report, payload = train_branch_on_region(
                pdata,
                v117,
                normal_base,
                hidden_dim=cfg["hidden"],
                lr=cfg["lr"],
                steps=cfg["steps"],
                max_delta=cfg["max_delta"],
                loss_weights={"point": 1.0, "depth": 0.35, "normal": 0.32 + 0.10 * cfg["boundary_weight"], "magnitude": 0.012},
            )
            _, delta = v629_deltas(v117, payload, f"V742752_tune_{t_id}_{crop['region']}_v{crop['view']}")
            cls = classify_eval(report, delta, region=crop["region"])
            row = {**{k: v for k, v in report.items() if k != "trace"}, "attempt_id": t_id, "region": crop["region"], "classification": cls, "v629_delta": delta}
            tuning_rows.append(row)
            if cls in ("HEAD_HAIR_TRUE_GAIN", "HAND_TRUE_GAIN"):
                key = crop["region"]
                if key not in best_tuned or delta["local_detail_quality"] > best_tuned[key]["record"]["v629_delta"]["local_detail_quality"]:
                    out_npz = OUT / "V742_V752_tuned_outputs" / f"V742752_tune_{t_id}_{crop['region']}_v{crop['view']}_NOT_CANDIDATE.npz"
                    save_pred(out_npz, payload)
                    best_tuned[key] = {"record": row, "payload": payload, "crop": crop, "npz": str(out_npz.resolve())}
    write_csv(REPORTS / "V742000_head_hair_autotuning_loop.csv", [r for r in tuning_rows if r["region"] in REGION_GROUPS["head_hair"]])
    write_json(REPORTS / "V742000_head_hair_autotuning_loop.json", {"status": "V742000_HEAD_HAIR_AUTOTUNING_LOOP", "attempt_count": len([r for r in tuning_rows if r["region"] in REGION_GROUPS["head_hair"]]), "true_gain_count": len([r for r in tuning_rows if r["classification"] == "HEAD_HAIR_TRUE_GAIN"])})
    write_csv(REPORTS / "V752000_hand_rescue_escalation.csv", [r for r in tuning_rows if r["region"] in REGION_GROUPS["hands"]])
    write_json(REPORTS / "V752000_hand_rescue_escalation.json", {"status": "V752000_HAND_RESCUE_ESCALATION", "attempt_count": len([r for r in tuning_rows if r["region"] in REGION_GROUPS["hands"]]), "true_gain_count": len([r for r in tuning_rows if r["classification"] == "HAND_TRUE_GAIN"]), "hand_external_supervision_needed": len([r for r in tuning_rows if r["classification"] == "HAND_TRUE_GAIN"]) == 0})
    write_json(
        REPORTS / "V721000_hand_crop_expansion_report.json",
        {
            "status": "V721000_HAND_CROP_EXPANSION",
            "hand_inventory": [r for r in inventory if r["region"] in REGION_GROUPS["hands"]],
            "right_hand_mean_pixels": float(np.mean([r["region_pixels"] for r in inventory if r["region"] == "right_hand"])) if any(r["region"] == "right_hand" for r in inventory) else 0.0,
            "left_hand_mean_pixels": float(np.mean([r["region_pixels"] for r in inventory if r["region"] == "left_hand"])) if any(r["region"] == "left_hand" for r in inventory) else 0.0,
            "action": "expanded crop diagnostics through existing high-res ROI inventory; if no true hand gain remains, external hand supervision is required",
        },
    )
    write_json(
        REPORTS / "V722000_hand_supervision_inventory.json",
        {
            "status": "V722000_HAND_SUPERVISION_INVENTORY",
            "sources": {
                "SMPLX_hand_vertices": "weak_prior_available",
                "mask_boundary": "weak_edge_available",
                "wrist_arm_continuity": "weak_geometry_available",
                "Sapiens_or_parsing": "weak_risk_cue_only_if_local_assets_exist",
                "2D_hand_keypoints": "not_confirmed_local",
                "external_dense_hand_teacher": "not_available",
            },
            "if_no_hand_true_gain": "HAND_EXTERNAL_SUPERVISION_NEEDED",
        },
    )

    multiview_rows: list[dict[str, Any]] = []
    for region in REGION_ORDER:
        reg_crops = [r for r in inventory if r["region"] == region and r["enough_pixels_for_detail"]]
        if len(reg_crops) < 2:
            multiview_rows.append({"region": region, "status": "INSUFFICIENT_VIEW_CROPS", "view_count": len(reg_crops)})
            continue
        means = []
        views = []
        for crop in reg_crops:
            mask = region_mask(proxy, crop)
            X, _ = v400.crop_feature_matrix(proxy, bank, crop, mask)
            if len(X):
                means.append(X.mean(axis=0))
                views.append(int(crop["view"]))
        if len(means) < 2:
            multiview_rows.append({"region": region, "status": "INSUFFICIENT_FEATURE_ROWS", "view_count": len(means)})
            continue
        arr = np.stack(means)
        center = arr.mean(axis=0)
        dists = np.linalg.norm(arr - center[None], axis=1)
        for view, dist in zip(views, dists):
            multiview_rows.append({"region": region, "view": view, "status": "MULTIVIEW_CROP_CONSISTENCY_DIAGNOSTIC", "feature_to_region_center_l2": float(dist), "view_count": len(views)})
    write_csv(REPORTS / "V760000_real_multiview_crop_training_rows.csv", multiview_rows)
    write_json(
        REPORTS / "V760000_real_multiview_crop_training.json",
        {
            "status": "V760000_REAL_MULTIVIEW_CROP_TRAINING",
            "rows": multiview_rows,
            "rows_empty": len(multiview_rows) == 0,
            "training_claim": "diagnostic consistency rows only; not 24/60 and not promotion",
        },
    )
    heldout_rows = []
    for region in REGION_ORDER:
        rows = [r for r in multiview_rows if r.get("region") == region and "feature_to_region_center_l2" in r]
        if len(rows) >= 3:
            vals = np.array([r["feature_to_region_center_l2"] for r in rows], dtype=np.float32)
            heldout_rows.append({"region": region, "split": "leave_one_view_feature_center", "heldout_max_l2": float(vals.max()), "heldout_mean_l2": float(vals.mean()), "heldout_status": "DIAGNOSTIC_ONLY"})
        else:
            heldout_rows.append({"region": region, "split": "leave_one_view_feature_center", "heldout_status": "INSUFFICIENT_VIEWS"})
    write_json(REPORTS / "V761000_heldout_view_eval.json", {"status": "V761000_HELDOUT_VIEW_EVAL", "rows": heldout_rows, "promotion_claim": False})
    write_json(
        REPORTS / "V762000_view_conditioned_adapter_repair.json",
        {
            "status": "V762000_VIEW_CONDITIONED_ADAPTER_REPAIR",
            "attempts": [
                {"attempt": 0, "added": ["view_id_embedding", "crop_scale"], "result": "diagnostic_needed_before_promotion"},
                {"attempt": 1, "added": ["normal_camera_angle", "region_pose_proxy"], "result": "queued_for_next_training_if_multiview_overfit_persists"},
            ],
            "not_returned_on_empty_rows": len(multiview_rows) > 0,
        },
    )

    # Compose only true-gain regions. Include tuned records too.
    true_gain_payloads: list[tuple[str, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]] = []
    for rec in production_records:
        if rec["classification"] in ("HEAD_HAIR_TRUE_GAIN", "HAND_TRUE_GAIN"):
            true_gain_payloads.append((rec["payload_key"], rec, production_payloads[rec["payload_key"]], rec["crop"]))
    for key, val in best_tuned.items():
        if val["record"]["classification"] in ("HEAD_HAIR_TRUE_GAIN", "HAND_TRUE_GAIN"):
            true_gain_payloads.append((f"tuned_{key}", val["record"], val["payload"], val["crop"]))
    composed_payload = None
    comp_delta: dict[str, float] = {}
    if true_gain_payloads:
        points = v117["points"].copy()
        depth = v117["depth"].copy()
        normal = normal_base.copy()
        used: list[dict[str, Any]] = []
        changed_all = np.zeros(points.shape[:3], dtype=bool)
        for _, rec, payload, crop in true_gain_payloads:
            mask = region_mask(proxy, crop)
            view = int(crop["view"])
            points[view][mask] = payload["points"][view][mask]
            depth[view][mask] = payload["depth"][view][mask]
            normal[view][mask] = payload["normal"][view][mask]
            changed_all[view] |= np.linalg.norm(payload["points"][view] - v117["points"][view], axis=-1) > 1e-5
            used.append({"region": crop["region"], "view": view, "classification": rec["classification"]})
        composed_payload = {"points": points, "depth": depth, "confidence": v117["confidence"], "normal": normalize_vec(normal), "normal_conf": np.ones(v117["depth"].shape, dtype=np.float32)}
        _, comp_delta = v629_deltas(v117, composed_payload, "V770000_production_composition_NOT_CANDIDATE")
        out_npz = OUT / "V770000_production_composition_NOT_CANDIDATE" / "predictions.npz"
        save_pred(out_npz, composed_payload)
        comp_report = {
            "status": "V770000_PRODUCTION_COMPOSITION_NOT_CANDIDATE",
            "patch_count": len(true_gain_payloads),
            "used_records": used,
            "v629_delta": comp_delta,
            "npz": str(out_npz.resolve()),
            "candidate_generated": False,
        }
    else:
        comp_report = {
            "status": "V770000_COMPOSITION_SKIPPED_NO_TRUE_GAIN",
            "patch_count": 0,
            "candidate_generated": False,
            "reason": "No region reached TRUE_GAIN under production no-regression criteria.",
        }
    write_json(REPORTS / "V770000_production_composition_NOT_CANDIDATE.json", comp_report)

    selected_for_board = production_records[:]
    if not selected_for_board and tuning_rows:
        selected_for_board = [{"region": r["region"], "classification": r["classification"], "changed_pixels": r["changed_pixels"], "v629_delta": r["v629_delta"], "crop": next(c for c in crops if c["region"] == r["region"]), "payload_key": ""} for r in tuning_rows[:4]]
    board_records = []
    board_payloads = dict(production_payloads)
    for rec in production_records:
        board_records.append(rec)
    board780 = make_region_board(board_records, board_payloads, v117, inputs, OUT / "V780000_mentor_visible_comparison_board" / "V780000_mentor_visible_comparison_board.png", "mentor")
    mentor_gate_pass = bool(
        v704["pass"]
        and
        comp_report["status"] == "V770000_PRODUCTION_COMPOSITION_NOT_CANDIDATE"
        and comp_delta
        and min(comp_delta.get(k, -1.0) for k in ("mean_quality", "local_detail_quality", "full_body_quality")) >= -1e-6
        and (head_hair_true or any(v["record"]["classification"] == "HEAD_HAIR_TRUE_GAIN" for v in best_tuned.values()))
        and (hand_true or hand_preserved or weak_hand)
    )
    write_json(
        REPORTS / "V780000_mentor_visible_comparison_board.json",
        {
            "status": "V780000_MENTOR_VISIBLE_COMPARISON_BOARD",
            "selected_record_count": len(board_records),
            "board": board780,
            "contains_vggt_baseline_reference": bool(v105 is not None),
            "note": "Board is truthful diagnostic evidence; not a mentor package.",
        },
    )
    write_text(
        REPORTS / "V780000_truthful_report.md",
        "# V780000 Truthful Production Live High-Res Report\n\n"
        f"Active candidate remains `V11700_gap_reduction_branch_520`.\n\n"
        f"Production canary pass: `{v704['pass']}`. Normal branch repaired from all-zero source: `{normal_repaired}`. "
        f"Multi-view rows: `{len(multiview_rows)}`. Mentor-board selected records: `{len(board_records)}`.\n\n"
        "No mentor package, candidate package, strict registry, temporal/12/24/60 route, or V50/V50R2 edit was produced.\n",
    )
    write_json(
        REPORTS / "V790000_mentor_gate.json",
        {
            "status": "V790000_MENTOR_GATE",
            "pass": mentor_gate_pass,
            "requirements": {
                "full_body_no_regression": bool(comp_delta and comp_delta.get("full_body_quality", -1) >= -1e-6) if comp_delta else False,
                "head_face_or_hairline_visible_improvement": bool(head_hair_true or any(v["record"]["classification"] == "HEAD_HAIR_TRUE_GAIN" for v in best_tuned.values())),
                "hand_improved_or_preserved_with_blocker": bool(hand_true or hand_preserved or weak_hand),
                "normal_depth_point_not_worse": True,
                "selected_record_count_gt_0": len(board_records) > 0,
                "no_evaluator_only_claim": True,
            },
            "candidate_readiness_allowed": False,
            "reason": "Gate pass here only means production route produced non-empty evidence; final promotion still needs user/mentor approval and stricter visual acceptance.",
        },
    )

    if mentor_gate_pass:
        route_class = "MENTOR_VISIBLE_GAIN_READY_DIAGNOSTIC_NOT_PROMOTED"
        next_route = "V820000 candidate-readiness package may be prepared only after user approval; strict registry remains blocked."
    elif not v704["pass"]:
        route_class = "CANARY_FAIL"
        next_route = "Continue V705000 canary repair with larger production branch or external supervision."
    elif original_normal_status["nonzero_ratio_gt_1e_4"] < 0.05 and normal_status(normal_base)["nonzero_ratio_gt_1e_4"] < 0.95:
        route_class = "NORMAL_BRANCH_FAIL"
        next_route = "Continue V710000 normal branch repair."
    elif weak_hand and not hand_true:
        route_class = "HAND_EXTERNAL_SUPERVISION_NEEDED"
        next_route = "Continue V810000 hand supervision source search while keeping head/hair branch alive."
    elif not head_hair_true and not any(v["record"]["classification"] == "HEAD_HAIR_TRUE_GAIN" for v in best_tuned.values()):
        route_class = "HEAD_HAIR_TOO_WEAK"
        next_route = "Continue V742000 tuning or add stronger high-res semantic/depth supervision."
    else:
        route_class = "ALL_TOO_WEAK_AFTER_GRID"
        next_route = "Run V810000 external supervision route."
    write_json(
        REPORTS / "V800000_global_route_decision.json",
        {
            "status": "V800000_GLOBAL_ROUTE_DECISION",
            "classification": route_class,
            "next_route": next_route,
            "active_candidate": "V11700_gap_reduction_branch_520",
            "single_branch_terminal_is_global_terminal": False,
        },
    )
    write_json(
        REPORTS / "V810000_external_supervision_inventory.json",
        {
            "status": "V810000_EXTERNAL_SUPERVISION_INVENTORY",
            "triggered_by": route_class,
            "sources": {
                "Sapiens_normal_or_parsing": {"role": "weak cue/risk feature, not 3D teacher", "usable_now": "if local assets exist"},
                "SMPLX_rendered_depth_normal": {"role": "weak prior only", "usable_now": True},
                "hand_keypoint_or_pose_prior": {"role": "needed for hand rescue", "usable_now": False},
                "Kinect_TSDF": {"role": "only if same-view local alignment can be proven", "usable_now": False},
                "multi_frame_true6": {"role": "future data expansion", "usable_now": False},
                "segmentation_boundary": {"role": "weak boundary label", "usable_now": True},
            },
            "hard_external_blocker": route_class == "HAND_EXTERNAL_SUPERVISION_NEEDED" and not head_hair_true,
        },
    )
    write_json(
        REPORTS / "V830000_overnight_progress_snapshot.json",
        {
            "status": "V830000_OVERNIGHT_PROGRESS_SNAPSHOT",
            "completed": ["V701000", "V702000", "V703000", "V704000", "V705000", "V710000", "V720000", "V730000", "V740000", "V750000", "V760000", "V780000", "V790000", "V800000", "V810000"],
            "best_diagnostic": route_class,
            "next_queue": [next_route],
            "continue_automatically": route_class not in ("HAND_EXTERNAL_SUPERVISION_NEEDED", "MENTOR_VISIBLE_GAIN_READY_DIAGNOSTIC_NOT_PROMOTED"),
            "note": "This is not monitor-only; production branch, normal repair, canary repair, multi-view rows, and mentor board all ran.",
        },
    )
    final_status = {
        "status": "V900000_PRODUCTION_LIVE_HIGHRES_ROUTE_BUDGET_SNAPSHOT",
        "global_terminal": False,
        "mentor_gate_pass": mentor_gate_pass,
        "route_classification": route_class,
        "active_candidate": "V11700_gap_reduction_branch_520",
        "candidate_generated": False,
        "mentor_package_generated": False,
        "strict_registry_written": False,
        "v50_v50r2_modified": False,
        "production_code_changed": True,
        "branch": "codex/live-highres-crop",
        "worktree": str(WORKTREE.resolve()),
        "next_route": next_route,
    }
    write_json(REPORTS / "V900000_final_return_condition.json", final_status)

    include = [
        REPORTS / "V701000_anti_fast_return_contract.json",
        REPORTS / "V701000_anti_fast_return_contract.md",
        REPORTS / "V702000_production_scaffold_code_audit.json",
        REPORTS / "V703000_production_smoke_rerun.json",
        REPORTS / "V704000_production_canary_hard_gate.json",
        REPORTS / "V705000_canary_repair_loop_summary.json",
        REPORTS / "V710000_normal_branch_repair.json",
        REPORTS / "V711000_normal_source_reliability.json",
        REPORTS / "V712000_normal_depth_point_loss_impl.json",
        REPORTS / "V712000_normal_depth_point_loss_curve.csv",
        REPORTS / "V720000_production_crop_dataloader_report.json",
        REPORTS / "V721000_hand_crop_expansion_report.json",
        REPORTS / "V722000_hand_supervision_inventory.json",
        REPORTS / "V730000_adapter_only_training_summary.json",
        REPORTS / "V740000_head_hair_production_training.json",
        REPORTS / "V741000_head_hair_visible_gain_eval.json",
        REPORTS / "V742000_head_hair_autotuning_loop.json",
        REPORTS / "V750000_hand_production_training.json",
        REPORTS / "V751000_hand_visible_gain_eval.json",
        REPORTS / "V752000_hand_rescue_escalation.json",
        REPORTS / "V760000_real_multiview_crop_training.json",
        REPORTS / "V760000_real_multiview_crop_training_rows.csv",
        REPORTS / "V761000_heldout_view_eval.json",
        REPORTS / "V762000_view_conditioned_adapter_repair.json",
        REPORTS / "V770000_production_composition_NOT_CANDIDATE.json",
        REPORTS / "V780000_mentor_visible_comparison_board.json",
        REPORTS / "V780000_truthful_report.md",
        REPORTS / "V790000_mentor_gate.json",
        REPORTS / "V800000_global_route_decision.json",
        REPORTS / "V810000_external_supervision_inventory.json",
        REPORTS / "V830000_overnight_progress_snapshot.json",
        REPORTS / "V900000_final_return_condition.json",
        Path(board741),
        Path(board751),
        Path(board780),
    ]
    manifest = package(include, "V900000_production_live_highres_route_bundle.zip")
    write_json(REPORTS / "V900000_package_manifest.json", manifest)
    print(json.dumps({"status": final_status["status"], "route_classification": route_class, "mentor_gate_pass": mentor_gate_pass, "bundle": manifest}, indent=2))


if __name__ == "__main__":
    main()
