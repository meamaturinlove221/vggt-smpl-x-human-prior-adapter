from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.v506_v50r2_observation_distilled_student import (  # noqa: E402
    V506ObservationDistilledStudentConfig,
    V506V50R2ObservationDistilledStudent,
)
from tools.V505_teacher_copy_detector import detect as detect_teacher_copy  # noqa: E402


REPORTS = ROOT / "reports"
BOARDS = ROOT / "boards"
OUT = ROOT / "output" / "V5140000000000000000000_v50r2_checkpoint_adjudicator"

SCENE = Path(r"D:\vggt\vggt-main\output\4k4d_scenes\0012_11_frame0000_12views_tmf_v223_repaired")
PACKAGE = Path(r"D:\vggt\vggt-main\output\surface_research_preflight_local\V50_final_promotion_transaction\candidate_package_v50r2\package_files")
V42_POINTS = PACKAGE / "v42_prior_enabled_payload__research_points_world.npz"
V42_NORMALS = PACKAGE / "v42_prior_enabled_payload__research_normals_geometric.npz"
V42_CONF = PACKAGE / "v42_prior_enabled_payload__research_confidence.npz"
SMPL_PRIOR = SCENE / "human_prior" / "smplx_vertex_feature_maps.npz"
TEACHER_BANK = ROOT / "output" / "V5040000000000000000000_v50r2_teacher_bank" / "v50r2_teacher_bank.npz"
CHECKPOINT = ROOT / "output" / "V5080000000000000000000_v50r2_distillation_matrix" / "v50r2_distilled_true" / "checkpoint_4000.pt"

V514_DECISION = REPORTS / "V5140000000000000000000_checkpoint_adjudication_decision.json"
V514_METRICS = REPORTS / "V5140000000000000000000_checkpoint_adjudication_metrics.csv"
V514_COPY = REPORTS / "V5140000000000000000000_teacher_copy_check.json"
V514_BOARD = BOARDS / "V5140000000000000000000_checkpoint_adjudication_board.png"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resize_rgb(path: Path, size: int = 518) -> np.ndarray:
    with Image.open(path).convert("RGB") as im:
        im = im.resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def resize_mask(path: Path, size: int = 518) -> np.ndarray:
    with Image.open(path).convert("L") as im:
        im = im.resize((size, size), Image.Resampling.NEAREST)
        return (np.asarray(im) > 0).astype(np.float32)


def scene_images_and_masks() -> tuple[list[Path], list[Path]]:
    image_paths = sorted((SCENE / "images").glob("*.png"))
    mask_paths = sorted((SCENE / "masks").glob("*.png"))
    if len(image_paths) != 12 or len(mask_paths) != 12:
        raise FileNotFoundError(f"expected 12 RGB/mask views, got {len(image_paths)} images and {len(mask_paths)} masks")
    return image_paths, mask_paths


def load_observation_inputs(max_human_per_view: int, max_env_per_view: int, seed: int) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    image_paths, mask_paths = scene_images_and_masks()
    with np.load(V42_POINTS, allow_pickle=False) as z:
        points = z["frame0000"].astype(np.float32)
        pose = z["frame0000_pose_enc"].astype(np.float32)
    with np.load(V42_NORMALS, allow_pickle=False) as z:
        normals = z["frame0000"].astype(np.float32)
    with np.load(V42_CONF, allow_pickle=False) as z:
        conf = z["frame0000_world_points_conf"].astype(np.float32)
    with np.load(SMPL_PRIOR, allow_pickle=False) as z:
        smpl = z["smpl_surface_feature_maps"].astype(np.float32)
        smpl_mask = z["smpl_prior_masks"].astype(bool)

    rgbs = np.stack([resize_rgb(path) for path in image_paths], axis=0)
    masks = np.stack([resize_mask(path) for path in mask_paths], axis=0).astype(bool)
    valid = np.isfinite(points).all(axis=-1) & np.isfinite(normals).all(axis=-1)
    human_select = masks & valid
    env_select = (~masks) & valid & (conf > np.percentile(conf, 55))

    human_points: list[np.ndarray] = []
    human_rgb: list[np.ndarray] = []
    human_conf: list[np.ndarray] = []
    human_normals: list[np.ndarray] = []
    human_smpl: list[np.ndarray] = []
    human_local: list[np.ndarray] = []
    human_cam: list[np.ndarray] = []
    human_mask: list[np.ndarray] = []
    human_parts: list[np.ndarray] = []
    selected_indices: list[np.ndarray] = []
    env_points: list[np.ndarray] = []
    env_rgb: list[np.ndarray] = []

    view_count, h, w, _ = points.shape
    for view in range(view_count):
        idx = np.flatnonzero(human_select[view].reshape(-1))
        if idx.size == 0:
            raise RuntimeError(f"no human mask pixels for view {view}")
        take = min(max_human_per_view, idx.size)
        choice = rng.choice(idx, size=take, replace=False)
        yy, xx = np.divmod(choice, w)
        selected_indices.append(np.stack([np.full(take, view), yy, xx], axis=-1).astype(np.int32))

        pts = points[view, yy, xx]
        rgb = rgbs[view, yy, xx]
        cnf = conf[view, yy, xx]
        cnf = (cnf / max(float(np.max(conf[view])), 1.0)).clip(0.0, 1.0)
        nrm = normals[view, yy, xx]
        nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1.0e-6)
        smpl_feat = np.moveaxis(smpl[view], 0, -1)[yy, xx]
        prior = smpl_mask[view, yy, xx].astype(np.float32)

        local = np.zeros((take, 9), dtype=np.float32)
        local[:, :3] = nrm
        local[:, 3:6] = smpl_feat[:, 2:5] if smpl_feat.shape[-1] >= 5 else 0.0
        local[:, 6:9] = pose[view, :3]

        cam_id = np.full((take, 1), view / max(view_count - 1, 1), dtype=np.float32)
        xy = np.stack([xx / max(w - 1, 1), yy / max(h - 1, 1)], axis=-1).astype(np.float32)
        cam = np.concatenate([cam_id, xy, rgb[:, :2], cnf[:, None]], axis=-1).astype(np.float32)

        parts = np.zeros(take, dtype=np.int64)
        parts[yy < int(h * 0.28)] = 1
        parts[(yy >= int(h * 0.28)) & (yy < int(h * 0.62))] = 3
        parts[yy >= int(h * 0.62)] = 6
        parts[(xx < int(w * 0.28)) | (xx > int(w * 0.72))] = np.maximum(parts[(xx < int(w * 0.28)) | (xx > int(w * 0.72))], 4)
        parts[prior <= 0] = 0

        human_points.append(pts)
        human_rgb.append(rgb)
        human_conf.append(cnf[:, None])
        human_normals.append(nrm)
        smpl_padded = np.zeros((take, 16), dtype=np.float32)
        smpl_padded[:, : min(16, smpl_feat.shape[-1])] = smpl_feat[:, :16]
        human_smpl.append(smpl_padded)
        human_local.append(local)
        human_cam.append(cam[:, :6])
        human_mask.append(masks[view, yy, xx].astype(np.float32)[:, None])
        human_parts.append(parts)

        env_idx = np.flatnonzero(env_select[view].reshape(-1))
        if env_idx.size:
            env_take = min(max_env_per_view, env_idx.size)
            env_choice = rng.choice(env_idx, size=env_take, replace=False)
            eyy, exx = np.divmod(env_choice, w)
            env_points.append(points[view, eyy, exx])
            env_rgb.append(rgbs[view, eyy, exx])

    hp = torch.from_numpy(np.stack(human_points)).float()
    hr = torch.from_numpy(np.stack(human_rgb)).float()
    hc = torch.from_numpy(np.stack(human_conf).squeeze(-1)).float()
    hn = F.normalize(torch.from_numpy(np.stack(human_normals)).float(), dim=-1)
    hs = torch.from_numpy(np.stack(human_smpl)).float()
    hl = torch.from_numpy(np.stack(human_local)).float()
    hcam = torch.from_numpy(np.stack(human_cam)).float()
    hm = torch.from_numpy(np.stack(human_mask).squeeze(-1)).float()
    parts_t = torch.from_numpy(np.stack(human_parts)).long()

    vggt_features = torch.cat([hp, hr, hn, hc.unsqueeze(-1), hm.unsqueeze(-1), (parts_t > 0).float().unsqueeze(-1)], dim=-1)
    if vggt_features.shape[-1] < 12:
        vggt_features = torch.cat([vggt_features, torch.zeros(*vggt_features.shape[:-1], 12 - vggt_features.shape[-1])], dim=-1)
    else:
        vggt_features = vggt_features[..., :12]

    env_p = torch.from_numpy(np.concatenate(env_points, axis=0)[None, ...]).float()
    env_r = torch.from_numpy(np.concatenate(env_rgb, axis=0)[None, ...]).float()
    batch = {
        "vggt_world_points": hp,
        "vggt_rgb": hr,
        "vggt_confidence": hc,
        "vggt_features": vggt_features,
        "smplx_graph_features": hs,
        "smplx_normal": hn,
        "smplx_local_frame": hl,
        "camera_features": hcam,
        "human_mask": hm,
        "body_part_id": parts_t,
        "environment_points": env_p,
        "environment_rgb": env_r,
    }
    meta = {
        "selected_indices": np.concatenate(selected_indices, axis=0),
        "environment_points": env_p.numpy()[0],
        "environment_rgb": env_r.numpy()[0],
        "view_count": np.array(view_count, dtype=np.int32),
        "human_points_per_view": np.array([arr.shape[0] for arr in human_points], dtype=np.int32),
        "environment_point_count": np.array(env_p.shape[1], dtype=np.int32),
    }
    return batch, meta


def load_model(checkpoint: Path) -> tuple[V506V50R2ObservationDistilledStudent, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = V506ObservationDistilledStudentConfig(**ckpt.get("config", {}))
    model = V506V50R2ObservationDistilledStudent(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def flatten_output(out: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        "points": out["student_human_points"].detach().cpu().numpy().reshape(-1, 3).astype(np.float32),
        "rgb": out["student_human_rgb"].detach().cpu().numpy().reshape(-1, 3).astype(np.float32),
        "confidence": out["student_confidence"].detach().cpu().numpy().reshape(-1).astype(np.float32),
        "visibility": out["surfel_visibility"].detach().cpu().numpy().reshape(-1).astype(np.float32),
        "occupancy": out["surfel_occupancy"].detach().cpu().numpy().reshape(-1).astype(np.float32),
        "residual": out["residual"].detach().cpu().numpy().reshape(-1, 3).astype(np.float32),
    }


def nearest_mean(a: np.ndarray, b: np.ndarray, max_a: int = 6000, max_b: int = 6000, seed: int = 514) -> float:
    rng = np.random.default_rng(seed)
    aa = a[np.isfinite(a).all(axis=-1)]
    bb = b[np.isfinite(b).all(axis=-1)]
    if aa.shape[0] > max_a:
        aa = aa[rng.choice(aa.shape[0], size=max_a, replace=False)]
    if bb.shape[0] > max_b:
        bb = bb[rng.choice(bb.shape[0], size=max_b, replace=False)]
    at = torch.from_numpy(aa).float()
    bt = torch.from_numpy(bb).float()
    best = []
    for start in range(0, at.shape[0], 512):
        d = torch.cdist(at[start:start + 512], bt)
        best.append(d.min(dim=1).values)
    return float(torch.cat(best).mean().item())


def load_teacher_points() -> np.ndarray:
    with np.load(TEACHER_BANK, allow_pickle=False) as z:
        pts = z["points"].astype(np.float32)
        mask = z["full_body_mask"].astype(bool)
    return pts[mask]


def write_ply(path: Path, points: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, rgb_u8):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def render_points(points: np.ndarray, rgb: np.ndarray, size: tuple[int, int], axes: tuple[int, int] = (0, 2)) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    if points.size == 0:
        return canvas
    pts = points[np.isfinite(points).all(axis=-1)]
    cols = rgb[np.isfinite(points).all(axis=-1)]
    if pts.shape[0] > 45000:
        idx = np.linspace(0, pts.shape[0] - 1, 45000).astype(np.int64)
        pts = pts[idx]
        cols = cols[idx]
    xy = pts[:, axes]
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    span = np.maximum(hi - lo, 1.0e-6)
    norm = (xy - lo) / span
    px = np.clip((norm[:, 0] * (size[0] - 20) + 10).astype(np.int32), 0, size[0] - 1)
    py = np.clip(((1.0 - norm[:, 1]) * (size[1] - 20) + 10).astype(np.int32), 0, size[1] - 1)
    pix = canvas.load()
    rgb_u8 = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
    for x, y, c in zip(px, py, rgb_u8):
        pix[int(x), int(y)] = (int(c[0]), int(c[1]), int(c[2]))
    return canvas


def make_board(true_points: np.ndarray, true_rgb: np.ndarray, base_points: np.ndarray, base_rgb: np.ndarray, teacher_points: np.ndarray, metrics: dict[str, Any]) -> None:
    V514_BOARD.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    board = Image.new("RGB", (1800, 1220), "white")
    draw = ImageDraw.Draw(board)
    draw.text((18, 14), "V514 checkpoint adjudication: VGGT observation inference, not V50R2 final copy", fill=(0, 0, 0), font=font)
    draw.text((18, 36), "Pass requires no teacher copy and true > baseline/no-SMPL under the V50R2 floor. Current decision is written in JSON.", fill=(120, 0, 0), font=font)
    panels = [
        ("true full-scene XZ", render_points(true_points, true_rgb, (580, 360), (0, 2))),
        ("true full-scene XY", render_points(true_points, true_rgb, (580, 360), (0, 1))),
        ("VGGT baseline XZ", render_points(base_points, base_rgb, (580, 360), (0, 2))),
        ("V50R2 floor reference XZ", render_points(teacher_points, np.tile(np.array([[0.2, 0.2, 0.2]], dtype=np.float32), (teacher_points.shape[0], 1)), (580, 360), (0, 2))),
    ]
    for i, (label, img) in enumerate(panels):
        x = 18 + (i % 3) * 595
        y = 70 + (i // 3) * 430
        board.paste(img, (x, y + 26))
        draw.rectangle([x, y + 26, x + 580, y + 386], outline=(80, 80, 80), width=1)
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
    y = 935
    for line in [
        f"status: {metrics['status']}",
        f"teacher_copy_leak_detected: {metrics['teacher_copy_leak_detected']}",
        f"true_to_v50r2_nn_mean: {metrics['true_to_v50r2_nn_mean']:.6f}",
        f"baseline_to_v50r2_nn_mean: {metrics['baseline_to_v50r2_nn_mean']:.6f}",
        f"no_smpl_to_v50r2_nn_mean: {metrics['no_smpl_to_v50r2_nn_mean']:.6f}",
        f"true_improves_baseline: {metrics['true_improves_baseline']}",
        f"accepted_model_owned_student: {metrics['accepted_model_owned_student']}",
    ]:
        draw.text((18, y), line, fill=(0, 0, 0), font=font)
        y += 28
    board.save(V514_BOARD)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    seed = 514
    batch, meta = load_observation_inputs(max_human_per_view=4096, max_env_per_view=2048, seed=seed)
    model, ckpt = load_model(CHECKPOINT)
    controls = {}
    with torch.no_grad():
        for control in ["true", "vggt_baseline", "no_smpl", "shuffled_semantic"]:
            controls[control] = flatten_output(model(batch, control=control))

    env_points = meta["environment_points"].astype(np.float32)
    env_rgb = meta["environment_rgb"].astype(np.float32)
    true_full_points = np.concatenate([env_points, controls["true"]["points"]], axis=0)
    true_full_rgb = np.concatenate([env_rgb, controls["true"]["rgb"]], axis=0)
    base_full_points = np.concatenate([env_points, controls["vggt_baseline"]["points"]], axis=0)
    base_full_rgb = np.concatenate([env_rgb, controls["vggt_baseline"]["rgb"]], axis=0)
    no_smpl_full_points = np.concatenate([env_points, controls["no_smpl"]["points"]], axis=0)
    no_smpl_full_rgb = np.concatenate([env_rgb, controls["no_smpl"]["rgb"]], axis=0)

    candidate_npz = OUT / "v514_model_owned_student_candidate.npz"
    np.savez_compressed(
        candidate_npz,
        predicted_points=controls["true"]["points"],
        predicted_rgb=controls["true"]["rgb"],
        full_scene_points=true_full_points,
        full_scene_rgb=true_full_rgb,
        vggt_baseline_points=controls["vggt_baseline"]["points"],
        no_smpl_points=controls["no_smpl"]["points"],
        selected_indices=meta["selected_indices"],
        model_owned_student_output=np.array(True),
        no_teacher_points_inference=np.array(True),
        no_v50r2_inference=np.array(True),
        no_kinect_depth_inference=np.array(True),
        final_inference_allowed=np.array(True),
        source=np.array("model_owned_vggt_smplx_checkpoint_v514"),
    )
    copy_result = detect_teacher_copy(TEACHER_BANK, candidate_npz)
    write_json(V514_COPY, {"task": "V514_teacher_copy_check", "created_at": now(), **copy_result})

    teacher_points = load_teacher_points()
    true_nn = nearest_mean(controls["true"]["points"], teacher_points, seed=seed)
    base_nn = nearest_mean(controls["vggt_baseline"]["points"], teacher_points, seed=seed + 1)
    no_smpl_nn = nearest_mean(controls["no_smpl"]["points"], teacher_points, seed=seed + 2)
    shuffled_nn = nearest_mean(controls["shuffled_semantic"]["points"], teacher_points, seed=seed + 3)
    true_improves_baseline = bool(true_nn < base_nn - 5.0e-4)
    true_improves_no_smpl = bool(true_nn < no_smpl_nn - 5.0e-4)
    true_improves_shuffled = bool(true_nn < shuffled_nn - 5.0e-4)
    residual_mean = float(np.linalg.norm(controls["true"]["residual"], axis=-1).mean())
    residual_p95 = float(np.percentile(np.linalg.norm(controls["true"]["residual"], axis=-1), 95))
    human_count = int(controls["true"]["points"].shape[0])
    environment_count = int(env_points.shape[0])
    env_visible = environment_count >= 12000
    accepted = bool(
        not copy_result["leak_detected"]
        and true_improves_baseline
        and true_improves_no_smpl
        and true_improves_shuffled
        and env_visible
        and human_count >= 30000
    )
    status = (
        "V514_ACCEPTED_MODEL_OWNED_STUDENT_READY_FOR_V509_NOT_PROMOTED"
        if accepted
        else "V514_FAIL_CLOSED_NO_ACCEPTED_MODEL_OWNED_STUDENT_CONTINUE_REPAIR_NOT_PROMOTED"
    )

    write_ply(OUT / "v514_true_full_scene_rgb.ply", true_full_points, true_full_rgb)
    write_ply(OUT / "v514_vggt_baseline_full_scene_rgb.ply", base_full_points, base_full_rgb)
    write_ply(OUT / "v514_no_smpl_full_scene_rgb.ply", no_smpl_full_points, no_smpl_full_rgb)
    write_ply(OUT / "v514_true_human_only_rgb.ply", controls["true"]["points"], controls["true"]["rgb"])

    row = {
        "status": status,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_steps": int(ckpt.get("steps", 0)),
        "human_point_count": human_count,
        "environment_point_count": environment_count,
        "teacher_copy_leak_detected": copy_result["leak_detected"],
        "true_to_v50r2_nn_mean": true_nn,
        "baseline_to_v50r2_nn_mean": base_nn,
        "no_smpl_to_v50r2_nn_mean": no_smpl_nn,
        "shuffled_to_v50r2_nn_mean": shuffled_nn,
        "true_improves_baseline": true_improves_baseline,
        "true_improves_no_smpl": true_improves_no_smpl,
        "true_improves_shuffled": true_improves_shuffled,
        "residual_mean": residual_mean,
        "residual_p95": residual_p95,
        "accepted_model_owned_student": accepted,
    }
    write_csv(V514_METRICS, [row])
    make_board(true_full_points, true_full_rgb, base_full_points, base_full_rgb, teacher_points, row)

    payload = {
        "task": "V514_v50r2_checkpoint_adjudicator",
        "status": status,
        "created_at": now(),
        "repo": str(ROOT),
        "checkpoint": str(CHECKPOINT),
        "candidate_npz": str(candidate_npz),
        "true_full_scene_ply": str(OUT / "v514_true_full_scene_rgb.ply"),
        "baseline_full_scene_ply": str(OUT / "v514_vggt_baseline_full_scene_rgb.ply"),
        "no_smpl_full_scene_ply": str(OUT / "v514_no_smpl_full_scene_rgb.ply"),
        "board": str(V514_BOARD),
        "metrics_csv": str(V514_METRICS),
        "teacher_copy_check": str(V514_COPY),
        "input_policy": {
            "vggt_points_source": str(V42_POINTS),
            "rgb_source": str(SCENE / "images"),
            "mask_source": str(SCENE / "masks"),
            "smplx_prior_source": str(SMPL_PRIOR),
            "v50r2_teacher_points_used_at_inference": False,
            "candidate_points_used_at_inference": False,
            "kinect_depth_used_at_inference": False,
            "teacher_used_for_evaluation_only": True,
        },
        "gates": {
            "model_owned_forward_ran": True,
            "teacher_copy_leak_detected": copy_result["leak_detected"],
            "no_teacher_copy": not copy_result["leak_detected"],
            "full_scene_environment_visible": env_visible,
            "human_point_count_ok": human_count >= 30000,
            "true_improves_vggt_baseline": true_improves_baseline,
            "true_improves_no_smpl": true_improves_no_smpl,
            "true_improves_shuffled_semantic": true_improves_shuffled,
            "accepted_model_owned_student": accepted,
            "mentor_ready": False,
            "not_promoted": True,
        },
        "metrics": row,
        "decision": (
            "Accepted for V509 insertion candidate handoff, but not promoted; V509/V512 must still verify full-scene visual evidence."
            if accepted
            else "Fail closed: the checkpoint can run on VGGT observation inputs and passes direct-copy detection, but it does not clearly beat baseline/no-SMPL/shuffled controls under the V50R2 floor. Continue model repair rather than promoting this candidate."
        ),
        "artifact_hashes": {
            "candidate_npz": sha256(candidate_npz),
            "true_full_scene_ply": sha256(OUT / "v514_true_full_scene_rgb.ply"),
            "baseline_full_scene_ply": sha256(OUT / "v514_vggt_baseline_full_scene_rgb.ply"),
            "board": sha256(V514_BOARD),
        },
    }
    write_json(V514_DECISION, payload)
    print(json.dumps({"status": status, "decision": str(V514_DECISION), "board": str(V514_BOARD)}, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
