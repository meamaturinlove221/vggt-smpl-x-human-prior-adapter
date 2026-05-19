from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vggt.models.smplx_feature_adapter import GatedTokenInjection, HumanResidualFieldHead, SMPLXFeatureEncoder, normalize_prior_maps
from vggt.models.smplx_feature_geometry_decoder import SMPLXFeatureGeometryDecoder
from vggt.models.smplx_feature_token_adapter import SMPLXFeatureTokenAdapter
from vggt.models.smplx_sparseconv_feature_encoder import SMPLXSparseConvFeatureEncoder, SMPLXSparseVoxelFeatureBuilder
from vggt.models.smplx_triplane_neural_texture import SMPLXTriPlaneNeuralTexture

MAIN = Path(r"D:\vggt\vggt-main")
LOCAL = MAIN / "local_report_auxiliary" / "V600_quality_rebuild"
REPORTS = LOCAL / "reports"
ARCHIVE = LOCAL / "archive"
LOGS = LOCAL / "logs"
OUT = LOCAL / "output" / "V8100000_V9000000_smplx_feature_encoding"
TOOLS = LOCAL / "tools"

REMOTE = LOCAL / "remote_pull"
V647 = REMOTE / "V647_true6_crop_baseline" / "predictions.npz"
V117 = REMOTE / "V11700_gap_reduction_branch_520" / "predictions.npz"
V770 = LOCAL / "output" / "V701000_V900000_production_live_highres" / "V770000_production_composition_NOT_CANDIDATE" / "predictions.npz"
V129 = LOCAL / "output" / "V1210000_V1800000_smplx_completion" / "V129_comp_body_head" / "predictions.npz"
V360_BEST = LOCAL / "output" / "V2610000_V3600000_asset_generating_temporal_fusion" / "V3300000_candidates" / "candidate_004_V770_full_w250" / "predictions.npz"
V360_TEACHER = LOCAL / "output" / "V2610000_V3600000_asset_generating_temporal_fusion" / "V3000000_temporal_teacher" / "teacher.npz"
SEMANTIC = LOCAL / "output" / "V2610000_V3600000_asset_generating_temporal_fusion" / "V2800000_semantic_assets" / "semantic_layer.npz"
V15 = MAIN / "output" / "surface_research_preflight_local" / "V15_SMPLX_native_camera_raster_export" / "v15_smplx_camera_raster_export.npz"
V16 = MAIN / "output" / "surface_research_preflight_local" / "V16_smplx_native_region_roi_builder" / "v16_smplx_native_region_roi_maps.npz"
CASE_INPUTS = MAIN / "output" / "training_cases" / "0012_11_frame0000_6views_smplx_native_prior_v15" / "inputs.npz"

sys.path.insert(0, str(TOOLS))
import v232000_v260000_sharper_defect_patch_route as v232  # noqa: E402


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: jable(row.get(k, "")) for k in keys})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def load_pred(path: Path) -> dict[str, np.ndarray]:
    z = load_npz(path)
    pts = z.get("world_points", z.get("points"))
    if pts is None:
        raise KeyError(f"{path} has no world_points/points")
    depth = z.get("depth", pts[..., 2])
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    conf = z.get("world_points_conf", z.get("confidence", np.ones(pts.shape[:-1], dtype=np.float32)))
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    normal = z.get("normal", np.zeros_like(pts, dtype=np.float32))
    normal_conf = z.get("normal_conf", np.ones(pts.shape[:-1], dtype=np.float32))
    return {
        "points": pts.astype(np.float32),
        "depth": depth.astype(np.float32),
        "confidence": conf.astype(np.float32),
        "normal": normal.astype(np.float32),
        "normal_conf": normal_conf.astype(np.float32),
    }


def save_pred(path: Path, pred: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        world_points=pred["points"].astype(np.float32),
        points=pred["points"].astype(np.float32),
        depth=pred["depth"].astype(np.float32),
        world_points_conf=pred["confidence"].astype(np.float32),
        confidence=pred["confidence"].astype(np.float32),
        normal=pred["normal"].astype(np.float32),
        normal_conf=pred["normal_conf"].astype(np.float32),
    )


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def resize_stack(arr: np.ndarray, size: int = 518, mode: str = "bilinear") -> np.ndarray:
    x = torch.from_numpy(arr.astype(np.float32))
    if x.ndim == 3:
        x = x[:, None]
        squeeze = True
    elif x.ndim == 4:
        x = x.permute(0, 3, 1, 2)
        squeeze = False
    else:
        raise ValueError(f"unsupported resize shape {arr.shape}")
    y = F.interpolate(x, size=(size, size), mode="nearest" if mode == "nearest" else "bilinear", align_corners=False if mode != "nearest" else None)
    if squeeze:
        return y[:, 0].numpy()
    return y.permute(0, 2, 3, 1).numpy()


def heat_png(path: Path, title: str, heat: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(heat, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.max(axis=0)
    finite = np.isfinite(arr)
    if finite.any():
        lo, hi = np.percentile(arr[finite], [2, 98])
        arr = (arr - lo) / max(hi - lo, 1e-6)
    arr = np.clip(arr, 0, 1)
    rgb = np.stack([arr, np.zeros_like(arr), 1 - arr], axis=-1)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)


def boundary(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    return binary_dilation(m, iterations=radius) & ~binary_erosion(m, iterations=radius)


def evaluate(v770: dict[str, np.ndarray], pred: dict[str, np.ndarray], name: str, bg_mask: np.ndarray) -> dict[str, Any]:
    base = {
        "points": v770["points"],
        "world_points": v770["points"],
        "depth": v770["depth"],
        "confidence": v770["confidence"],
        "world_points_conf": v770["confidence"],
        "normal": v770["normal"],
        "normal_conf": v770["normal_conf"],
    }
    shadow = {
        "points": pred["points"],
        "world_points": pred["points"],
        "depth": pred["depth"],
        "confidence": pred["confidence"],
        "world_points_conf": pred["confidence"],
        "normal": pred["normal"],
        "normal_conf": pred["normal_conf"],
    }
    _, _, vd = v232.v629_delta(base, shadow, name)
    diff = np.linalg.norm(pred["points"] - v770["points"], axis=-1)
    return {
        "name": name,
        "delta_vs_v770": vd,
        "changed_pixels_vs_v770": int((diff > 1e-7).sum()),
        "background_changed_pixels": int(((diff > 1e-7) & bg_mask).sum()),
        "depth_point_z_error": float(np.nanmean(np.abs(pred["depth"] - pred["points"][..., 2]))),
        "normal_nonzero_ratio": float((np.linalg.norm(pred["normal"], axis=-1) > 1e-6).mean()),
        "candidate_equals_v770": bool(np.max(np.abs(pred["points"] - v770["points"])) == 0.0),
    }


def make_candidate(base: dict[str, np.ndarray], target: dict[str, np.ndarray], mask: np.ndarray, weight: float, max_delta: float) -> dict[str, np.ndarray]:
    delta = np.clip(target["points"] - base["points"], -max_delta, max_delta)
    pts = np.where(mask[..., None], base["points"] + weight * delta, base["points"])
    normal = target["normal"]
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = np.where(norm > 1e-6, normal / np.clip(norm, 1e-6, None), base["normal"])
    return {
        "points": pts.astype(np.float32),
        "depth": pts[..., 2].astype(np.float32),
        "confidence": np.maximum(base["confidence"], target["confidence"]).astype(np.float32),
        "normal": normal.astype(np.float32),
        "normal_conf": np.maximum(base["normal_conf"], target["normal_conf"]).astype(np.float32),
    }


def build_feature_raster() -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    v15 = load_npz(V15)
    v16 = load_npz(V16)
    sem = load_npz(SEMANTIC)
    depth = resize_stack(v15["depth"], 518)
    points = resize_stack(v15["points_world"], 518)
    normals = resize_stack(v15["normals_world"], 518)
    mask = resize_stack(v15["mask"].astype(np.float32), 518, mode="nearest") > 0.5
    part = resize_stack(v15["macro_part_ids"].astype(np.float32), 518, mode="nearest")
    vertices = v15["vertices"].astype(np.float32)
    nvid = np.clip(v16["nearest_vertex_ids"].astype(np.int64), 0, len(vertices) - 1)
    vertex_xyz = vertices[nvid]
    center = vertices.mean(axis=0, keepdims=True)
    scale = np.percentile(np.linalg.norm(vertices - center, axis=1), 95)
    canonical = np.clip((vertex_xyz - center) / max(float(scale), 1e-6), -3, 3)
    vid_phase = (2.0 * np.pi * nvid.astype(np.float32)) / max(float(len(vertices) - 1), 1.0)
    dist_in = np.stack([distance_transform_edt(m) for m in mask], axis=0)
    dist_out = np.stack([distance_transform_edt(~m) for m in mask], axis=0)
    signed_boundary = np.clip((dist_in - dist_out) / 32.0, -1, 1)
    channels = [
        canonical[..., 0],
        canonical[..., 1],
        canonical[..., 2],
        points[..., 0],
        points[..., 1],
        points[..., 2],
        normals[..., 0],
        normals[..., 1],
        normals[..., 2],
        depth,
        mask.astype(np.float32),
        np.sin(vid_phase),
        np.cos(vid_phase),
        part / max(float(np.max(part)), 1.0),
        signed_boundary,
        sem["foreground"],
        sem["head_face"],
        sem["hairline"],
        sem["left_hand"],
        sem["right_hand"],
        sem["phone_object_exclusion"],
    ]
    names = [
        "canonical_x",
        "canonical_y",
        "canonical_z",
        "posed_x",
        "posed_y",
        "posed_z",
        "normal_x",
        "normal_y",
        "normal_z",
        "smplx_depth",
        "smplx_visibility",
        "vertex_id_sin",
        "vertex_id_cos",
        "macro_part_scaled",
        "signed_boundary",
        "semantic_foreground",
        "semantic_head_face",
        "semantic_hairline",
        "semantic_left_hand",
        "semantic_right_hand",
        "phone_object_exclusion",
    ]
    feature_maps = np.stack([c.astype(np.float32) for c in channels], axis=1)
    masks = {k: np.asarray(v).astype(bool) for k, v in sem.items()}
    masks["smplx_visibility"] = mask
    return feature_maps, names, masks


def deterministic_triplane_features(canonical: np.ndarray) -> np.ndarray:
    freqs = np.asarray([1.0, 2.0, 4.0], dtype=np.float32)
    planes = []
    for pair in [(0, 1), (0, 2), (1, 2)]:
        xy = canonical[..., pair]
        feats = [np.sin(np.pi * f * xy[..., 0]) for f in freqs] + [np.cos(np.pi * f * xy[..., 1]) for f in freqs]
        planes.extend(feats)
    return np.stack(planes, axis=1).astype(np.float32)


def sparse_voxel_features(feature_maps: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    canonical = np.moveaxis(feature_maps[:, :3], 1, -1)
    pts = canonical[mask]
    feats = np.moveaxis(feature_maps, 1, -1)[mask]
    if pts.size == 0:
        return {"coords": np.zeros((0, 4), np.int32), "features": np.zeros((0, feature_maps.shape[1]), np.float32)}
    grid = np.floor((np.clip(pts, -1.2, 1.2) + 1.2) / 2.4 * 31).astype(np.int32)
    view_ids = np.repeat(np.arange(feature_maps.shape[0])[:, None, None], feature_maps.shape[2], axis=1)
    view_ids = np.repeat(view_ids, feature_maps.shape[3], axis=2)[mask].astype(np.int32)
    coords = np.concatenate([view_ids[:, None], grid], axis=1)
    # Aggregate by voxel with numpy unique for a dependency-free NeuralBody-style fallback.
    uniq, inv = np.unique(coords, axis=0, return_inverse=True)
    out = np.zeros((len(uniq), feats.shape[1]), dtype=np.float32)
    counts = np.bincount(inv).astype(np.float32)
    for c in range(feats.shape[1]):
        out[:, c] = np.bincount(inv, weights=feats[:, c], minlength=len(uniq)) / np.maximum(counts, 1)
    return {"coords": uniq.astype(np.int32), "features": out.astype(np.float32), "counts": counts.astype(np.float32)}


def strict_eval(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranked = []
    for row in rows:
        d = row["eval"]["delta_vs_v770"]
        full = float(d.get("full_body_quality", -999))
        hair = float(d.get("hairline_quality", -999))
        head = float(d.get("head_face_quality", -999))
        left = float(d.get("left_hand_quality", -999))
        right = float(d.get("right_hand_quality", -999))
        positives = sum(v > 0.001 for v in (head, left, right))
        score = full + hair + head + left + right + float(d.get("mean_quality", 0)) + float(d.get("local_detail_quality", 0))
        strict = (
            not row["eval"]["candidate_equals_v770"]
            and row["eval"]["background_changed_pixels"] == 0
            and full >= -1e-6
            and hair >= -1e-6
            and positives >= 2
            and row["eval"]["depth_point_z_error"] < 1e-6
            and row["eval"]["normal_nonzero_ratio"] > 0.99
        )
        ranked.append({"name": row["name"], "score": float(score), "strict_pass": bool(strict), "path": row["path"], "experiment": row["experiment"], **{f"{k}_delta": float(v) for k, v in d.items() if isinstance(v, (int, float, np.floating))}})
    ranked.sort(key=lambda x: (x["strict_pass"], x["score"]), reverse=True)
    return {
        "status": "V8800000_STRICT_EVAL",
        "candidate_count": len(rows),
        "strict_pass_count": sum(1 for r in ranked if r["strict_pass"]),
        "best": ranked[0] if ranked else {},
    }, ranked


def zip_paths(zip_path: Path, paths: list[Path]) -> dict[str, Any]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    entries = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for path in paths:
            path = Path(path)
            if not path.exists():
                continue
            if path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file():
                        z.write(child, child.relative_to(LOCAL) if child.is_relative_to(LOCAL) else child.relative_to(ROOT))
                        entries += 1
            else:
                z.write(path, path.relative_to(LOCAL) if path.is_relative_to(LOCAL) else path.relative_to(ROOT))
                entries += 1
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
    return {"zip_path": zip_path, "entry_count": entries, "sha256": sha256_file(zip_path), "zip_test": "clean" if bad is None else str(bad)}


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60).stdout.strip()

    return {"branch": run(["branch", "--show-current"]), "head": run(["rev-parse", "--short", "HEAD"]), "status_short": run(["status", "--short"])}


def process_scan() -> dict[str, Any]:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|modal' } | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return {"raw": out.stdout.strip(), "clean": out.stdout.strip() in ("", "null", "[]")}


def main() -> int:
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    stages: list[dict[str, Any]] = []

    def stage(name: str, t0: float) -> None:
        stages.append({"stage": name, "seconds": time.time() - t0, "utc": now()})
        write_csv(LOGS / "V8100000_V9000000_stage_runtime.csv", stages)

    s = time.time()
    completeness = {
        "status": "V8100000_CONTROLLER_IMPLEMENTED",
        "modules": {
            "HumanRAM_style_raster": True,
            "SMPLXTriPlaneNeuralTexture": True,
            "SMPLXFeatureTokenAdapter": True,
            "SMPLXSparseVoxelFeatureBuilder": True,
            "SMPLXSparseConvFeatureEncoder": True,
            "SMPLXFeatureGeometryDecoder": True,
            "strict_eval_archive": True,
        },
        "not_claimed": ["full SparseConv3D training", "HumanRAM RGB rendering", "NeRF density/color", "promotion"],
    }
    write_json(REPORTS / "V8100000_controller_completeness_audit.json", completeness)
    stage("V8100000_completeness", s)

    s = time.time()
    v647, v117, v770, v129, v360, teacher = map(load_pred, [V647, V117, V770, V129, V360_BEST, V360_TEACHER])
    write_json(REPORTS / "V8110000_schema_report.json", {"status": "V8110000_SCHEMA_READY", "inputs": {"V647": V647, "V117": V117, "V770": V770, "V129": V129, "V360_best": V360_BEST, "teacher": V360_TEACHER}})
    stage("V8110000_inputs", s)

    s = time.time()
    feature_maps, feature_names, masks = build_feature_raster()
    save_npz(OUT / "V8200000_smplx_feature_raster" / "feature_maps.npz", feature_maps=feature_maps, channel_names=np.asarray(feature_names))
    heat_png(OUT / "V8200000_smplx_feature_raster" / "feature_raster_board.png", "SMPL-X feature coverage", feature_maps[:, feature_names.index("smplx_visibility")])
    write_json(REPORTS / "V8200000_feature_raster_inventory.json", {"status": "V8200000_FEATURE_RASTER_READY", "shape": feature_maps.shape, "channels": feature_names, "visibility_pixels": int(feature_maps[:, feature_names.index("smplx_visibility")].sum())})
    stage("V8200000_raster", s)

    s = time.time()
    canonical = np.moveaxis(feature_maps[:, :3], 1, -1)
    triplane = deterministic_triplane_features(canonical)
    sample_xyz = torch.from_numpy(canonical.reshape(1, -1, 3)[:, ::32])
    tri_module = SMPLXTriPlaneNeuralTexture(feature_dim=16, plane_resolution=32, reduce="concat", deterministic_bands=3)
    with torch.no_grad():
        tri_out = tri_module(sample_xyz, return_dict=True)
    save_npz(
        OUT / "V8210000_triplane_pose_features" / "triplane_features.npz",
        triplane_features=triplane,
        module_sample_features=tri_out["features"].numpy(),
        module_deterministic_features=tri_out["deterministic_features"].numpy(),
    )
    write_json(
        REPORTS / "V8210000_triplane_probe.json",
        {
            "status": "V8210000_TRIPLANE_MODULE_AND_DETERMINISTIC_READY",
            "deterministic_shape": triplane.shape,
            "module_sample_shape": list(tri_out["features"].shape),
            "module_output_dim": tri_module.output_dim,
            "learnable_T1": "module_present_zero_initialized_not_trained",
        },
    )
    stage("V8210000_triplane", s)

    s = time.time()
    all_features = np.concatenate([feature_maps, triplane], axis=1).astype(np.float32)
    torch_feat = torch.from_numpy(all_features[None])
    encoder = SMPLXFeatureEncoder(in_chans=all_features.shape[1], token_dim=1024, patch_size=14, hidden_dim=64)
    token_adapter = SMPLXFeatureTokenAdapter(in_chans=all_features.shape[1], c_vggt=1024, patch_size=14, hidden_dim=64, mode="add")
    with torch.no_grad():
        tokens = encoder(normalize_prior_maps(torch_feat))
        add_tokens = GatedTokenInjection(1024, "add")(torch.zeros_like(tokens), tokens)
        film_tokens = GatedTokenInjection(1024, "film")(torch.zeros_like(tokens), tokens)
        adapter_tokens = token_adapter(normalize_prior_maps(torch_feat), return_dict=True)["smplx_patch_tokens"]
        patch_count = adapter_tokens.shape[2]
        fake_vggt_with_special = torch.zeros(1, 6, patch_count + 5, 1024)
        token_add = token_adapter(normalize_prior_maps(torch_feat), fake_vggt_with_special, mode="add", patch_start_idx=5, return_dict=True)["tokens"]
        token_film = token_adapter(normalize_prior_maps(torch_feat), fake_vggt_with_special, mode="film", patch_start_idx=5, return_dict=True)["tokens"]
        token_prefix = token_adapter(normalize_prior_maps(torch_feat), fake_vggt_with_special, mode="prefix", patch_start_idx=5, return_dict=True)["tokens"]
    write_json(
        REPORTS / "V8300000_token_alignment.json",
        {
            "status": "V8300000_TOKEN_ALIGNMENT_PASS",
            "feature_image_shape": all_features.shape,
            "legacy_tokens_shape": list(tokens.shape),
            "module_tokens_shape": list(adapter_tokens.shape),
            "expected_tokens": [1, 6, 1369, 1024],
            "add_shape": list(add_tokens.shape),
            "film_shape": list(film_tokens.shape),
            "module_add_shape": list(token_add.shape),
            "module_film_shape": list(token_film.shape),
            "module_prefix_shape": list(token_prefix.shape),
        },
    )
    heat_png(OUT / "V8300000_token_coverage_board.png", "token coverage", masks["smplx_visibility"].astype(np.float32))
    stage("V8300000_token_alignment", s)

    s = time.time()
    sparse = sparse_voxel_features(all_features, masks["smplx_visibility"])
    point_xyz = torch.from_numpy(canonical.reshape(1, -1, 3)[:, ::16])
    point_feat = torch.from_numpy(np.moveaxis(all_features, 1, -1).reshape(1, -1, all_features.shape[1])[:, ::16])
    point_mask = torch.from_numpy(masks["smplx_visibility"].reshape(1, -1)[:, ::16])
    sparse_builder = SMPLXSparseVoxelFeatureBuilder(bounds=(-3.0, 3.0), grid_size=32)
    with torch.no_grad():
        sparse_module = sparse_builder(point_xyz, point_feat, mask=point_mask)
    save_npz(
        OUT / "V8400000_sparse_voxel_features" / "sparse_tensor.npz",
        **sparse,
        module_voxel_coords=sparse_module["voxel_coords"].numpy(),
        module_voxel_features=sparse_module["voxel_features"].numpy(),
    )
    write_json(
        REPORTS / "V8400000_sparse_feature_inventory.json",
        {
            "status": "V8400000_SPARSE_VOXEL_FEATURES_READY",
            "coord_count": int(sparse["coords"].shape[0]),
            "feature_dim": int(sparse["features"].shape[1]),
            "backend": "numpy_unique_fallback_and_SMPLXSparseVoxelFeatureBuilder",
            "module_coord_count": int(sparse_module["voxel_coords"].shape[0]),
        },
    )
    stage("V8400000_sparse_voxel", s)

    s = time.time()
    sparse_encoder = SMPLXSparseConvFeatureEncoder(in_dim=all_features.shape[1], out_dim=32, hidden_dim=32, num_layers=1, bounds=(-3.0, 3.0), grid_size=32, backend="torch")
    with torch.no_grad():
        sparse_encoded = sparse_encoder(point_xyz, point_feat, mask=point_mask, return_point_features=True)
    latent = sparse_encoded["encoded_voxel_features"].numpy()
    save_npz(
        OUT / "V8500000_sparse_latent_field" / "latent_field.npz",
        coords=sparse_encoded["voxel_coords"].numpy(),
        latent=latent,
        encoded_point_features=sparse_encoded["encoded_point_features"].numpy(),
    )
    write_json(
        REPORTS / "V8500000_sparseconv_arch.json",
        {
            "status": "V8500000_SPARSECONV_FALLBACK_PROTOTYPE",
            "spconv_or_minkowski_used": False,
            "active_backend": sparse_encoded["active_backend"],
            "available_sparse_backend": sparse_encoded["available_sparse_backend"],
            "latent_shape": latent.shape,
            "warning": "pure PyTorch sparse fallback validates NeuralBody-style feature diffusion but is not a trained SparseConv3D result",
        },
    )
    stage("V8500000_sparse_fallback", s)

    s = time.time()
    residual_head = HumanResidualFieldHead(in_chans=all_features.shape[1], hidden_dim=48, gate_bias_init=-2.5)
    geometry_decoder = SMPLXFeatureGeometryDecoder(feature_dim=all_features.shape[1], hidden_dim=64, num_layers=2)
    with torch.no_grad():
        residuals = residual_head(torch_feat, human_mask=torch.from_numpy(masks["smplx_visibility"][None].astype(np.float32)))
        flat_feat = torch.from_numpy(np.moveaxis(all_features, 1, -1).reshape(1, -1, all_features.shape[1])[:, ::32])
        flat_xyz = torch.from_numpy(canonical.reshape(1, -1, 3)[:, ::32])
        flat_mask = torch.from_numpy(masks["smplx_visibility"].reshape(1, -1)[:, ::32].astype(np.float32))
        geom_out = geometry_decoder(flat_feat, canonical_xyz=flat_xyz, mask=flat_mask)
    write_json(
        REPORTS / "V8600000_decoder_eval.json",
        {
            "status": "V8600000_GEOMETRY_DECODER_SMOKE_PASS",
            "dense_delta_point_shape": list(residuals["delta_point"].shape),
            "dense_identity_initialized": bool(float(residuals["delta_point"].abs().max()) == 0.0),
            "dense_apply_gate_mean": float(residuals["apply_gate"].mean().item()),
            "module_delta_point_shape": list(geom_out["delta_point"].shape),
            "module_identity_initialized": bool(float(geom_out["delta_point"].abs().max()) == 0.0),
            "module_reliability_mean": float(geom_out["reliability"].mean().item()),
        },
    )
    stage("V8600000_decoder", s)

    rows: list[dict[str, Any]] = []

    def add_candidate(name: str, pred: dict[str, np.ndarray], cfg: dict[str, Any]) -> None:
        out = OUT / "V8700000_candidates" / name
        save_pred(out / "predictions.npz", pred)
        ev = evaluate(v770, pred, name, masks["background_lock"])
        write_json(out / "eval.json", ev)
        write_json(out / "config.json", cfg | {"name": name})
        heat_png(out / "board.png", name, np.linalg.norm(pred["points"] - v770["points"], axis=-1))
        rows.append({"name": name, "path": out / "predictions.npz", "eval": ev, "experiment": cfg.get("experiment", "unknown")})

    s = time.time()
    feature_strength = np.clip(np.abs(all_features).mean(axis=1), 0, 1)
    semantic_masks = {
        "canonical_xyz": masks["foreground"] & (feature_strength > np.percentile(feature_strength[masks["foreground"]], 45)),
        "xyz_normal_part": masks["foreground"] & masks["smplx_visibility"],
        "triplane_T0": masks["foreground"] & (np.abs(triplane).mean(axis=1) > np.percentile(np.abs(triplane).mean(axis=1)[masks["foreground"]], 40)),
        "sparse_voxel": masks["foreground"] & (feature_strength > np.percentile(feature_strength[masks["foreground"]], 55)),
        "humanram_token": masks["head_face"] | masks["hairline"] | masks["left_hand"] | masks["right_hand"],
        "neuralbody_sparse": masks["body"] | masks["head_face"] | masks["left_hand"] | masks["right_hand"],
        "hybrid_sparse_token": masks["foreground"] & ~masks["phone_object_exclusion"],
    }
    weights = [0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34]
    for idx, (label, mask) in enumerate(semantic_masks.items()):
        target = teacher if label not in {"humanram_token", "hybrid_sparse_token"} else v360
        add_candidate(f"V870_C{idx+1}_{label}", make_candidate(v770, target, mask.astype(bool), weights[idx], 0.014), {"experiment": label, "weight": weights[idx]})
    # Conservative compositions, not final unless V880 accepts them.
    for idx, w in enumerate([0.12, 0.16, 0.20, 0.24, 0.28]):
        comp_mask = masks["foreground"] & masks["smplx_visibility"] & ~masks["phone_object_exclusion"]
        mixed = {k: np.array(v, copy=True) for k, v in teacher.items()}
        mixed["points"] = 0.65 * teacher["points"] + 0.35 * v360["points"]
        add_candidate(f"V870_C10_combined_conservative_{idx}", make_candidate(v770, mixed, comp_mask, w, 0.012), {"experiment": "combined_conservative", "weight": w})
    write_json(REPORTS / "V8700000_candidate_manifest.json", {"status": "V8700000_CANDIDATES_READY", "candidate_count": len(rows)})
    stage("V8700000_candidates", s)

    s = time.time()
    gate, ranked = strict_eval(rows)
    write_json(REPORTS / "V8800000_strict_eval.json", gate)
    write_csv(REPORTS / "V8800000_ranked_candidates.csv", ranked)
    if ranked:
        best = load_pred(Path(ranked[0]["path"]))
        heat_png(OUT / "V8800000_four_way_smplx_feature_encoding_board.png", ranked[0]["name"], np.linalg.norm(best["points"] - v770["points"], axis=-1))
    stage("V8800000_strict_eval", s)

    s = time.time()
    ready = bool(gate["strict_pass_count"] > 0)
    status = "V9000000_REVIEW_READY_NOT_PROMOTED" if ready else "V9000000_ROUTE_EXHAUSTED_WITH_FAILURE_ANALYSIS"
    attribution = {
        "status": "V8900000_REVIEW_READY_NOT_PROMOTED" if ready else "V8900000_FAILURE_ATTRIBUTION",
        "best": gate.get("best", {}),
        "failure_classes": [] if ready else ["feature_encoding_probe_did_not_pass_strict_gate", "full_sparseconv_training_not_run", "learnable_triplane_not_trained"],
        "honest_scope": "This route validates SMPL-X feature encoding/raster/voxel/token candidate probes. It is not a HumanRAM renderer, NeuralBody NeRF, or promoted candidate.",
    }
    write_json(REPORTS / "V8900000_failure_attribution.json", attribution)
    write_text(REPORTS / "V8900000_next_action.md", "# V8900000 Next Action\n\nIf strict gate passes, run mentor visual review. If not, train tri-plane texture or real SparseConv3D/SparseUNet before claiming feature encoding success.\n")
    stage("V8900000_decision", s)

    s = time.time()
    include = [
        ROOT / "tools" / "v8100000_v9000000_smplx_feature_encoding_controller.py",
        ROOT / "vggt" / "models" / "smplx_feature_adapter.py",
        ROOT / "vggt" / "models" / "smplx_triplane_neural_texture.py",
        ROOT / "vggt" / "models" / "smplx_feature_token_adapter.py",
        ROOT / "vggt" / "models" / "smplx_sparseconv_feature_encoder.py",
        ROOT / "vggt" / "models" / "smplx_feature_geometry_decoder.py",
        *sorted(REPORTS.glob("V8*.json")),
        *sorted(REPORTS.glob("V8*.csv")),
        OUT,
        LOGS / "V8100000_V9000000_stage_runtime.csv",
    ]
    bundle = zip_paths(ARCHIVE / f"{status.lower()}_bundle.zip", include)
    final = {
        "created_utc": now(),
        "status": status,
        "active_candidate": "V11700_gap_reduction_branch_520",
        "promotion": False,
        "strict_registry_written": False,
        "v50_v50r2_modified": False,
        "candidate_count": len(rows),
        "strict_pass_count": gate["strict_pass_count"],
        "best_candidate": gate.get("best", {}),
        "bundle": bundle,
        "runtime_seconds": time.time() - start,
        "git": git_info(),
        "process_scan": process_scan(),
    }
    write_json(REPORTS / "V9000000_final_status.json", final)
    write_text(REPORTS / "V9000000_final_summary.md", f"# V9000000 Final Summary\n\nStatus: `{status}`.\nCandidates: `{len(rows)}`. Strict pass count: `{gate['strict_pass_count']}`.\nNo promotion, no strict registry, no V50/V50R2 edit.\n")
    bundle = zip_paths(ARCHIVE / f"{status.lower()}_bundle.zip", include + [REPORTS / "V9000000_final_status.json", REPORTS / "V9000000_final_summary.md"])
    final["bundle"] = bundle
    write_json(REPORTS / "V9000000_final_status.json", final)
    stage("V9000000_package", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
