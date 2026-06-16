from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from preflight_differentiable_renderer_backend import (  # noqa: E402
    align_intrinsics_for_loaded_scene_view,
    describe_cuda_device,
    import_nvdiffrast,
    load_view_rgb_mask,
    parse_view_indices,
    render_nvdiffrast_view,
    save_image,
)
from prepare_4k4d_prior_training_case import (  # noqa: E402
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
)
from tools.smplx_numpy import compute_vertex_normals  # noqa: E402


DEFAULT_SCENE_DIR = Path("output/4k4d_preprocessed_scene_variants/0012_11_frame0000_60views_human_crop")
DEFAULT_TEMPLATE_PAYLOAD = Path(
    "output/surface_research_preflight_local/connected_payload_self_describing/"
    "connected_human_surface_template_payload_self_describing.npz"
)
DEFAULT_QUERY_EVIDENCE = Path(
    "output/surface_research_preflight_local/B_Fus3D6_query_evidence_cache_hybrid6_layer23/"
    "b_fus3d_query_evidence_cache.npz"
)
DEFAULT_LATENT_REAL = Path(
    "output/surface_research_preflight_local/B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23/"
    "b_fus3d_latent_grid_evidence_arrays.npz"
)
DEFAULT_LATENT_SHUFFLE = Path(
    "output/surface_research_preflight_local/B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23_shuffle/"
    "b_fus3d_latent_grid_evidence_arrays.npz"
)
DEFAULT_LATENT_ZERO = Path(
    "output/surface_research_preflight_local/B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23_zero/"
    "b_fus3d_latent_grid_evidence_arrays.npz"
)
DEFAULT_B17_CONTRACT = Path(
    "output/surface_research_preflight_local/B_Fus3D17_surface_sdf_contract_preflight_hybrid6_layer23/"
    "b_fus3d_surface_sdf_contract_summary.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/B_Fus3D19_bounded_surface_sdf_render_smoke_hybrid6_layer23"
)
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_fus3d_b19_bounded_surface_sdf_render_status.md")

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
}
CONTRACT = {
    "research_only": True,
    "local_only": True,
    "bounded_smoke_only": True,
    "no_cloud": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_predictions_write": True,
    "no_strict_pass_write": True,
    "no_registry_write": True,
    "writes_checkpoint": False,
    "not_teacher": True,
    "not_candidate": True,
}
FORBIDDEN_PATH_TOKENS = ("strict_pass", "teacher_export", "candidate_export")
CONTROL_NAMES = ("real", "shuffle", "zero")
FAMILY_TO_PARTS = {
    "full_body": (0, 5),
    "left_hand": (1,),
    "right_hand": (2,),
    "face_core": (3,),
    "hairline": (4,),
}
PART_LIMITS = {
    0: 0.0015,
    1: 0.0040,
    2: 0.0040,
    3: 0.0030,
    4: 0.0050,
    5: 0.0030,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "B-Fus3D19 bounded surface-token/query-SDF rendered smoke. It builds "
            "a tiny fail-closed query-to-carrier displacement field for real, "
            "shuffle, and zero controls, renders all controls through the same "
            "raw cameras/masks/RGB, and compares diagnostics. It is not formal "
            "training and never exports a teacher/candidate or strict pass."
        )
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--template-payload", type=Path, default=DEFAULT_TEMPLATE_PAYLOAD)
    parser.add_argument("--query-evidence", type=Path, default=DEFAULT_QUERY_EVIDENCE)
    parser.add_argument("--latent-grid-real", type=Path, default=DEFAULT_LATENT_REAL)
    parser.add_argument("--latent-grid-shuffle", type=Path, default=DEFAULT_LATENT_SHUFFLE)
    parser.add_argument("--latent-grid-zero", type=Path, default=DEFAULT_LATENT_ZERO)
    parser.add_argument("--b17-contract", type=Path, default=DEFAULT_B17_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--view-indices", default="0,10,24,36,45,57")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.35)
    parser.add_argument("--z-sign", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_safe_path(path: Path) -> None:
    text = str(path).replace("\\", "/").lower()
    for token in FORBIDDEN_PATH_TOKENS:
        if token in text:
            raise ValueError(f"Refusing path containing forbidden token {token!r}: {path}")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_npz(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as payload:
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise KeyError(f"{resolved} missing arrays: {missing}")
        return {key: np.asarray(payload[key]) for key in payload.files}


def load_template(path: Path) -> dict[str, np.ndarray]:
    payload = load_npz(path, ("hybrid_vertices", "hybrid_faces", "part_ids"))
    vertices = np.asarray(payload["hybrid_vertices"], dtype=np.float32)
    faces = np.asarray(payload["hybrid_faces"], dtype=np.int32)
    part_ids = np.asarray(payload["part_ids"], dtype=np.int64)
    if part_ids.shape[0] != vertices.shape[0]:
        raise ValueError(f"part_ids length {part_ids.shape[0]} != vertices {vertices.shape[0]}")
    normals = compute_vertex_normals(vertices, faces).astype(np.float32)
    return {"vertices": vertices, "faces": faces, "part_ids": part_ids, "normals": normals}


def load_query(path: Path) -> dict[str, np.ndarray]:
    return load_npz(
        path,
        (
            "query_positions",
            "query_families",
            "support",
            "mean_features",
            "variance_features",
            "selected_view_indices",
        ),
    )


def load_latent(path: Path) -> dict[str, np.ndarray]:
    return load_npz(
        path,
        (
            "points",
            "evidence_score",
            "token_cosine",
            "visible_count",
            "mask_count",
            "boundary_like",
            "selected_view_indices",
        ),
    )


def load_views(scene_dir: Path, dataset_root: Path | None, subset_name: str, view_spec: str, target_size: int):
    manifest = recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir))
    exported = manifest["exported_views"]
    view_indices = parse_view_indices(view_spec, len(exported))
    views: list[dict[str, Any]] = []
    for idx in view_indices:
        row = dict(exported[idx])
        row["view_index"] = int(idx)
        views.append(row)
    resolved_dataset_root = dataset_root or Path(str(manifest.get("dataset_root", "")))
    cameras, camera_source = resolve_scene_camera_params(manifest, resolved_dataset_root, subset_name)
    return views, cameras, camera_source, view_indices


def family_vertex_masks(part_ids: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for family, parts in FAMILY_TO_PARTS.items():
        mask = np.zeros(part_ids.shape, dtype=bool)
        for part in parts:
            mask |= np.asarray(part_ids) == int(part)
        out[family] = mask
    return out


def nearest_query_delta(
    vertices: np.ndarray,
    normals: np.ndarray,
    part_ids: np.ndarray,
    query: dict[str, np.ndarray],
    latent: dict[str, np.ndarray],
    *,
    control: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    query_pos = np.asarray(query["query_positions"], dtype=np.float32)
    query_families = np.asarray(query["query_families"]).astype(str).reshape(-1)
    query_support = np.asarray(query["support"], dtype=np.float32).reshape(-1)
    mean_features = np.asarray(query["mean_features"], dtype=np.float32)
    variance_features = np.asarray(query["variance_features"], dtype=np.float32)
    evidence = np.asarray(latent["evidence_score"], dtype=np.float32).reshape(-1)
    token_cosine = np.asarray(latent["token_cosine"], dtype=np.float32).reshape(-1)
    token_cosine = np.where(np.isfinite(token_cosine), token_cosine, 0.0)
    latent_points = np.asarray(latent["points"], dtype=np.float32)

    # Map each query to its nearest latent-grid evidence. This is a bounded
    # evidence readout, not a teacher.
    q2l = np.zeros((query_pos.shape[0],), dtype=np.int64)
    chunk = 128
    for start in range(0, query_pos.shape[0], chunk):
        q = query_pos[start : start + chunk]
        dist2 = ((q[:, None, :] - latent_points[None, :, :]) ** 2).sum(axis=-1)
        q2l[start : start + chunk] = np.argmin(dist2, axis=1)
    q_evidence = evidence[q2l]
    q_token = token_cosine[q2l]
    feat_energy = np.tanh(np.mean(mean_features[:, :64], axis=1) * 0.03)
    feat_uncert = np.tanh(np.mean(variance_features[:, :64], axis=1) * 0.01)
    support_gate = np.clip(query_support / 6.0, 0.0, 1.0)
    control_bias = {"real": 1.0, "shuffle": 1.0, "zero": 1.0}[control]
    q_score = control_bias * (0.55 * (q_evidence - 0.40) + 0.25 * (q_token - 0.75) + 0.15 * feat_energy - 0.05 * feat_uncert)
    q_score = np.tanh(q_score) * support_gate

    family_masks = family_vertex_masks(part_ids)
    vertex_delta = np.zeros_like(vertices, dtype=np.float32)
    family_rows: dict[str, Any] = {}
    for family, vmask in family_masks.items():
        qmask = query_families == family
        if not np.any(vmask) or not np.any(qmask):
            family_rows[family] = {"vertex_count": int(vmask.sum()), "query_count": int(qmask.sum()), "status": "missing"}
            continue
        qpos = query_pos[qmask]
        qscore = q_score[qmask]
        family_vertices = vertices[vmask]
        nearest = np.zeros((family_vertices.shape[0],), dtype=np.int64)
        for start in range(0, family_vertices.shape[0], 1024):
            fv = family_vertices[start : start + 1024]
            dist2 = ((fv[:, None, :] - qpos[None, :, :]) ** 2).sum(axis=-1)
            nearest[start : start + 1024] = np.argmin(dist2, axis=1)
        score = qscore[nearest]
        # Keep the carrier connected and bounded. This is not enough for mentor
        # success by design; it only tests whether real controls move rendered
        # diagnostics better than shuffle/zero.
        part_limit = np.asarray([PART_LIMITS.get(int(p), 0.0015) for p in part_ids[vmask]], dtype=np.float32)
        delta = normals[vmask] * (score[:, None] * part_limit[:, None])
        vertex_delta[vmask] = delta.astype(np.float32)
        family_rows[family] = {
            "vertex_count": int(vmask.sum()),
            "query_count": int(qmask.sum()),
            "mean_query_support": float(query_support[qmask].mean()),
            "mean_query_score": float(qscore.mean()),
            "mean_abs_vertex_delta": float(np.linalg.norm(delta, axis=1).mean()),
            "max_abs_vertex_delta": float(np.linalg.norm(delta, axis=1).max()),
        }
    return vertex_delta, family_rows


def mask_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred_b = np.asarray(pred, dtype=bool)
    tgt_b = np.asarray(target, dtype=bool)
    inter = int((pred_b & tgt_b).sum())
    union = int((pred_b | tgt_b).sum())
    pred_count = int(pred_b.sum())
    target_count = int(tgt_b.sum())
    return {
        "pred_pixels": pred_count,
        "target_pixels": target_count,
        "intersection": inter,
        "union": union,
        "iou": float(inter / union) if union else 0.0,
        "target_recall": float(inter / target_count) if target_count else 0.0,
        "overfill_ratio": float(max(pred_count - inter, 0) / max(pred_count, 1)),
    }


def finite_mean(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


def render_control(
    *,
    dr: Any,
    ctx: Any,
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    views: list[dict[str, Any]],
    cameras: dict[str, dict[str, Any]],
    target_size: int,
    output_dir: Path,
    z_sign: float,
) -> dict[str, Any]:
    device = torch.device("cuda")
    vertices_t = torch.as_tensor(vertices, dtype=torch.float32, device=device).contiguous()
    faces_t = torch.as_tensor(faces, dtype=torch.int32, device=device).contiguous()
    normals_t = torch.as_tensor(normals, dtype=torch.float32, device=device).contiguous()
    colors_np = np.clip(normals * 0.20 + 0.55, 0.15, 0.95).astype(np.float32)
    colors_t = torch.as_tensor(colors_np, dtype=torch.float32, device=device).contiguous()
    control_dir = output_dir / name
    control_dir.mkdir(parents=True, exist_ok=True)

    view_rows: list[dict[str, Any]] = []
    ious: list[float] = []
    recalls: list[float] = []
    overfills: list[float] = []
    rgb_residuals: list[float] = []
    for view in views:
        view_index = int(view["view_index"])
        params = cameras[str(view["camera_id"])]
        intrinsic_np = align_intrinsics_for_loaded_scene_view(np.asarray(params["intrinsic"], dtype=np.float32), view, target_size)
        world_to_cam_np = np.asarray(params["world_to_cam"], dtype=np.float32)
        rgb_np, mask_np = load_view_rgb_mask(view, target_size)
        rgb_f = rgb_np.astype(np.float32) / 255.0
        render = render_nvdiffrast_view(
            dr,
            ctx,
            vertices_t,
            faces_t,
            normals_t,
            colors_t,
            torch.as_tensor(world_to_cam_np, dtype=torch.float32, device=device),
            torch.as_tensor(intrinsic_np, dtype=torch.float32, device=device),
            target_size,
            target_size,
            z_sign=float(z_sign),
        )
        pred_mask = render["mask"].detach().cpu().numpy() > 0.5
        rendered_rgb = render["color"].detach().cpu().numpy().astype(np.float32)
        rendered_normal = render["normal"].detach().cpu().numpy().astype(np.float32)
        metrics = mask_metrics(pred_mask, mask_np)
        valid = pred_mask & mask_np
        rgb_res = float(np.mean(np.sqrt(np.sum((rendered_rgb[valid] - rgb_f[valid]) ** 2, axis=1) + 1e-6))) if np.any(valid) else None
        save_image(control_dir / f"view_{view_index:02d}_render_mask.png", pred_mask.astype(np.float32))
        save_image(control_dir / f"view_{view_index:02d}_render_rgb.png", rendered_rgb)
        save_image(control_dir / f"view_{view_index:02d}_render_normal.png", rendered_normal * 0.5 + 0.5)
        save_image(control_dir / f"view_{view_index:02d}_target_mask.png", mask_np.astype(np.float32))
        delta = np.zeros((*pred_mask.shape, 3), dtype=np.float32)
        delta[..., 0] = np.logical_and(pred_mask, ~mask_np).astype(np.float32)
        delta[..., 1] = np.logical_and(pred_mask, mask_np).astype(np.float32)
        delta[..., 2] = np.logical_and(~pred_mask, mask_np).astype(np.float32)
        save_image(control_dir / f"view_{view_index:02d}_mask_delta_rgb.png", delta)
        ious.append(float(metrics["iou"]))
        recalls.append(float(metrics["target_recall"]))
        overfills.append(float(metrics["overfill_ratio"]))
        if rgb_res is not None:
            rgb_residuals.append(rgb_res)
        view_rows.append({"view_index": view_index, "camera_id": str(view["camera_id"]), **metrics, "rgb_residual": rgb_res})
    return {
        "control": name,
        "views": view_rows,
        "mean_iou": finite_mean(ious),
        "mean_target_recall": finite_mean(recalls),
        "mean_overfill_ratio": finite_mean(overfills),
        "mean_rgb_residual": finite_mean(rgb_residuals),
    }


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {vertices.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(f"element face {faces.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for row in vertices:
            handle.write(f"{float(row[0])} {float(row[1])} {float(row[2])}\n")
        for tri in faces:
            handle.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Fus3D19 Bounded Surface/SDF Render Smoke",
        "",
        "Status: `research_only_bounded_smoke_no_export`",
        "",
        "This fixed-budget smoke tests a bounded query-to-carrier surface/SDF-like",
        "readout with real/shuffle/zero controls. It is not a teacher, candidate,",
        "strict pass, or cloud unblock.",
        "",
        "## Strict Truth",
        "",
        "```text",
        "strict_candidate_passes = 0",
        "strict_teacher_passes = 0",
        "formal_cloud_train_infer_export = blocked",
        "```",
        "",
        "## Control Metrics",
        "",
    ]
    for name, row in summary["controls"].items():
        lines.extend(
            [
                f"### `{name}`",
                "",
                "```json",
                json.dumps(row["metrics"], indent=2, ensure_ascii=False, sort_keys=True),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Comparison",
            "",
            "```json",
            json.dumps(summary["comparison"], indent=2, ensure_ascii=False, sort_keys=True),
            "```",
            "",
            "## Decision",
            "",
            "```text",
            summary["decision"],
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_safe_path(args.output_dir)
    ensure_safe_path(args.status_report)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} already exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    b17 = json.loads(args.b17_contract.read_text(encoding="utf-8"))
    if not bool(b17.get("contract_preflight_complete")):
        raise RuntimeError("B17 contract is not compatible; refusing B19 smoke")

    dr, import_error = import_nvdiffrast()
    if dr is None:
        raise RuntimeError(f"nvdiffrast unavailable: {import_error}")
    cuda_info = describe_cuda_device()
    if not bool(cuda_info.get("available")):
        raise RuntimeError("CUDA unavailable")

    template = load_template(args.template_payload)
    query = load_query(args.query_evidence)
    latent_payloads = {
        "real": load_latent(args.latent_grid_real),
        "shuffle": load_latent(args.latent_grid_shuffle),
        "zero": load_latent(args.latent_grid_zero),
    }
    selected_ref = np.asarray(latent_payloads["real"]["selected_view_indices"]).reshape(-1).astype(int).tolist()
    for name, payload in latent_payloads.items():
        if payload["points"].shape != latent_payloads["real"]["points"].shape:
            raise RuntimeError(f"{name} latent grid shape mismatch")
        if np.asarray(payload["selected_view_indices"]).reshape(-1).astype(int).tolist() != selected_ref:
            raise RuntimeError(f"{name} selected view mismatch")
    views, cameras, camera_source, view_indices = load_views(
        args.scene_dir, args.dataset_root, args.subset_name, args.view_indices, int(args.target_size)
    )
    ctx = dr.RasterizeCudaContext(device=torch.device("cuda"))

    controls: dict[str, Any] = {}
    for name in CONTROL_NAMES:
        delta, family_rows = nearest_query_delta(
            template["vertices"],
            template["normals"],
            template["part_ids"],
            query,
            latent_payloads[name],
            control=name,
        )
        vertices = template["vertices"] + delta
        normals = compute_vertex_normals(vertices.astype(np.float32), template["faces"]).astype(np.float32)
        mesh_path = args.output_dir / f"{name}_bounded_surface_sdf_proxy_mesh.ply"
        write_ply(mesh_path, vertices.astype(np.float32), template["faces"].astype(np.int32))
        metrics = render_control(
            dr=dr,
            ctx=ctx,
            name=name,
            vertices=vertices.astype(np.float32),
            faces=template["faces"].astype(np.int32),
            normals=normals,
            views=views,
            cameras=cameras,
            target_size=int(args.target_size),
            output_dir=args.output_dir,
            z_sign=float(args.z_sign),
        )
        controls[name] = {
            "mesh_path": str(mesh_path.resolve()),
            "family_rows": family_rows,
            "metrics": metrics,
            "max_delta": float(np.linalg.norm(delta, axis=1).max()),
            "mean_delta": float(np.linalg.norm(delta, axis=1).mean()),
        }

    real = controls["real"]["metrics"]
    shuffle = controls["shuffle"]["metrics"]
    zero = controls["zero"]["metrics"]
    comparison = {
        "real_minus_shuffle_iou": float((real["mean_iou"] or 0.0) - (shuffle["mean_iou"] or 0.0)),
        "real_minus_zero_iou": float((real["mean_iou"] or 0.0) - (zero["mean_iou"] or 0.0)),
        "real_minus_shuffle_recall": float((real["mean_target_recall"] or 0.0) - (shuffle["mean_target_recall"] or 0.0)),
        "real_minus_zero_recall": float((real["mean_target_recall"] or 0.0) - (zero["mean_target_recall"] or 0.0)),
        "real_rgb_better_than_shuffle": bool((real["mean_rgb_residual"] or 999.0) < (shuffle["mean_rgb_residual"] or 999.0)),
        "real_rgb_better_than_zero": bool((real["mean_rgb_residual"] or 999.0) < (zero["mean_rgb_residual"] or 999.0)),
    }
    real_beats_controls = (
        comparison["real_minus_shuffle_iou"] > 0.005
        and comparison["real_minus_zero_iou"] > 0.005
        and comparison["real_rgb_better_than_shuffle"]
        and comparison["real_rgb_better_than_zero"]
    )
    decision = (
        "B19 real beats controls on this tiny rendered smoke, but this is still not a pass; Open3D strict review would be required next."
        if real_beats_controls
        else "B19 real does not beat shuffle/zero in rendered diagnostics; freeze this bounded query-to-carrier implementation."
    )
    summary = {
        "status": "research_only_bounded_smoke_no_export",
        "strict_facts": STRICT_FACTS,
        "contract": CONTRACT,
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda": cuda_info,
            "nvdiffrast_import_error": import_error,
            "camera_source": camera_source,
            "view_indices": view_indices,
            "target_size": int(args.target_size),
            "max_steps": int(args.max_steps),
        },
        "controls": controls,
        "comparison": comparison,
        "real_beats_controls": bool(real_beats_controls),
        "decision": decision,
        "blocked_actions": [
            "no_checkpoint_write",
            "no_predictions_write",
            "no_teacher_export",
            "no_candidate_export",
            "no_strict_registry_write",
            "no_formal_cloud_train_infer_export",
        ],
    }
    write_json(args.output_dir / "b_fus3d_b19_summary.json", summary)
    write_json(args.output_dir / "b_fus3d_b19_control_comparison.json", comparison)
    write_report(args.output_dir / "b_fus3d_b19_report.md", summary)
    write_report(args.status_report, summary)
    print(json.dumps(json_ready({"status": summary["status"], "real_beats_controls": real_beats_controls, "decision": decision}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
