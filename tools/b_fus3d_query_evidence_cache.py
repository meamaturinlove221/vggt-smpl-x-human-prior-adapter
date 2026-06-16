from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from preflight_differentiable_renderer_backend import (  # noqa: E402
    align_intrinsics_for_loaded_scene_view,
    load_connected_mesh,
    load_view_rgb_mask,
    parse_view_indices,
)
from prepare_4k4d_prior_training_case import (  # noqa: E402
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
)


DEFAULT_SCENE_DIR = Path("output/4k4d_preprocessed_scene_variants/0012_11_frame0000_60views_human_crop")
DEFAULT_TOKEN_CACHE = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D0_token_cache_extract_hybrid6_518_roi_withhands_arrays_v2/"
    "token_cache/aggregator_layer_23.npz"
)
DEFAULT_TEMPLATE_PAYLOAD = Path(
    "output/surface_research_preflight_local/connected_payload_self_describing/"
    "connected_human_surface_template_payload_self_describing.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D6_query_evidence_cache_hybrid6_layer23"
)

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
}

CONTRACT = {
    "research_only": True,
    "local_only": True,
    "no_train": True,
    "no_cloud": True,
    "no_candidate_export": True,
    "no_teacher_export": True,
    "no_strict_pass_write": True,
    "uses_vggt_depth_point_normal_as_hard_teacher": False,
    "writes_predictions_npz": False,
    "writes_prediction_arrays": False,
    "writes_strict_registry": False,
    "writes_candidate": False,
    "writes_teacher": False,
    "writes_checkpoint": False,
}

FAMILY_TO_PARTS = {
    "full_body": [0, 5],
    "left_hand": [1],
    "right_hand": [2],
    "face_core": [3],
    "hairline": [4],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only B-Fus3D 3D query evidence cache. It projects connected "
            "surface query points into selected VGGT aggregator patch-token maps and "
            "stores per-query multi-view token mean/variance/support. It never trains, "
            "writes predictions, exports teachers/candidates, writes strict pass state, "
            "or calls cloud."
        )
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--token-cache", type=Path, default=DEFAULT_TOKEN_CACHE)
    parser.add_argument("--template-payload", type=Path, default=DEFAULT_TEMPLATE_PAYLOAD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument(
        "--query-spec",
        default="full_body:128,face_core:128,hairline:128,left_hand:96,right_hand:96",
        help="Comma-separated family:count surface query sampling spec.",
    )
    parser.add_argument("--use-mask-gate", action="store_true", default=True)
    parser.add_argument("--no-mask-gate", dest="use_mask_gate", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def parse_query_spec(text: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        family, count = raw.split(":", 1)
        family = family.strip()
        if family not in FAMILY_TO_PARTS:
            raise ValueError(f"Unknown query family {family!r}; known={sorted(FAMILY_TO_PARTS)}")
        out.append((family, int(count)))
    return out


def even_sample_indices(indices: np.ndarray, count: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indices.size == 0 or count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if indices.size <= count:
        return indices
    positions = np.linspace(0, indices.size - 1, int(count)).round().astype(np.int64)
    return indices[positions]


def sample_queries(vertices: np.ndarray, part_ids: np.ndarray, spec: list[tuple[str, int]]) -> dict[str, Any]:
    query_indices: list[np.ndarray] = []
    query_families: list[str] = []
    for family, count in spec:
        parts = np.asarray(FAMILY_TO_PARTS[family], dtype=np.int64)
        candidates = np.flatnonzero(np.isin(part_ids, parts))
        sampled = even_sample_indices(candidates, count)
        query_indices.append(sampled)
        query_families.extend([family] * int(sampled.size))
    if query_indices:
        all_indices = np.concatenate(query_indices, axis=0).astype(np.int64)
    else:
        all_indices = np.zeros((0,), dtype=np.int64)
    return {
        "indices": all_indices,
        "positions": vertices[all_indices].astype(np.float32),
        "part_ids": part_ids[all_indices].astype(np.int64),
        "families": np.asarray(query_families, dtype="<U32"),
    }


def load_tokens(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as payload:
        tokens = np.asarray(payload["tokens"])
        patch_start_idx = int(np.asarray(payload["patch_start_idx"]).reshape(-1)[0])
        selected_view_indices = [int(v) for v in np.asarray(payload["selected_view_indices"]).reshape(-1)]
    if tokens.ndim != 4 or tokens.shape[0] != 1:
        raise ValueError(f"Expected tokens shape [1,S,P,C], got {tokens.shape}")
    return {
        "path": str(resolved),
        "tokens": tokens,
        "patch_start_idx": patch_start_idx,
        "selected_view_indices": selected_view_indices,
        "view_count": int(tokens.shape[1]),
        "token_count": int(tokens.shape[2]),
        "feature_dim": int(tokens.shape[3]),
    }


def load_scene_views(
    scene_dir: Path,
    dataset_root: Path | None,
    subset_name: str,
    selected_view_indices: list[int],
    target_size: int,
) -> tuple[list[dict[str, Any]], str]:
    manifest = recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir))
    exported = manifest["exported_views"]
    view_indices = parse_view_indices(",".join(str(v) for v in selected_view_indices), len(exported), len(exported))
    resolved_dataset_root = dataset_root or Path(str(manifest.get("dataset_root", "")))
    cameras, camera_source = resolve_scene_camera_params(manifest, resolved_dataset_root, subset_name)
    rows: list[dict[str, Any]] = []
    for view_index in view_indices:
        view = exported[view_index]
        camera_id = str(view["camera_id"])
        params = cameras[camera_id]
        intrinsic = align_intrinsics_for_loaded_scene_view(np.asarray(params["intrinsic"], dtype=np.float32), view, target_size)
        world_to_cam = np.asarray(params["world_to_cam"], dtype=np.float32)
        _rgb, mask = load_view_rgb_mask(view, target_size)
        rows.append(
            {
                "view_index": int(view_index),
                "camera_id": camera_id,
                "intrinsic": intrinsic.astype(np.float32),
                "world_to_cam": world_to_cam.astype(np.float32),
                "mask": mask.astype(bool),
            }
        )
    return rows, camera_source


def project_queries(
    positions: np.ndarray,
    world_to_cam: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(positions, dtype=np.float32)
    rotation = world_to_cam[:3, :3].astype(np.float32)
    translation = world_to_cam[:3, 3].astype(np.float32)
    cam = vertices @ rotation.T + translation[None, :]
    z = cam[:, 2]
    uvw = (intrinsic @ cam.T).T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
    return uv.astype(np.float32), z.astype(np.float32), cam.astype(np.float32)


def patch_ids_from_uv(
    uv: np.ndarray,
    z: np.ndarray,
    mask: np.ndarray,
    target_size: int,
    patch_start_idx: int,
    patch_grid: int,
    use_mask_gate: bool,
) -> tuple[np.ndarray, np.ndarray]:
    u = uv[:, 0]
    v = uv[:, 1]
    inside = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(z)
        & (z > 1e-6)
        & (u >= 0.0)
        & (u < target_size)
        & (v >= 0.0)
        & (v < target_size)
    )
    xi = np.clip(np.floor(u).astype(np.int64), 0, target_size - 1)
    yi = np.clip(np.floor(v).astype(np.int64), 0, target_size - 1)
    if use_mask_gate:
        inside &= mask[yi, xi]
    patch_size = float(target_size) / float(patch_grid)
    patch_x = np.clip(np.floor(u / patch_size).astype(np.int64), 0, patch_grid - 1)
    patch_y = np.clip(np.floor(v / patch_size).astype(np.int64), 0, patch_grid - 1)
    patch_ids = patch_start_idx + patch_y * patch_grid + patch_x
    patch_ids[~inside] = -1
    return patch_ids.astype(np.int64), inside.astype(bool)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Fus3D Query Evidence Cache",
        "",
        "This is a research-only 3D query to VGGT token evidence cache. It is not a",
        "decoder, not a teacher, not a candidate, and not a strict pass.",
        "",
        "## Strict Truth",
        "",
        "```text",
        f"strict_candidate_passes = {summary['strict_candidate_passes']}",
        f"strict_teacher_passes = {summary['strict_teacher_passes']}",
        f"formal_cloud_train_infer_export = {summary['formal_cloud_train_infer_export']}",
        "```",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary["summary"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Family Support",
        "",
        "```json",
        json.dumps(summary["family_support"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Outputs",
        "",
    ]
    for value in summary["outputs"].values():
        lines.append(f"- `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    token_payload = load_tokens(args.token_cache)
    tokens = token_payload["tokens"].astype(np.float32)
    patch_start_idx = int(token_payload["patch_start_idx"])
    patch_count = int(tokens.shape[2] - patch_start_idx)
    patch_grid_float = float(np.sqrt(patch_count))
    patch_grid = int(round(patch_grid_float))
    if patch_grid * patch_grid != patch_count:
        raise ValueError(f"Patch count {patch_count} is not a square grid after patch_start_idx={patch_start_idx}")
    mesh = load_connected_mesh(args.template_payload)
    vertices = np.asarray(mesh["vertices"], dtype=np.float32)
    part_ids = np.asarray(mesh["part_ids"], dtype=np.int64)
    query_spec = parse_query_spec(args.query_spec)
    queries = sample_queries(vertices, part_ids, query_spec)
    views, camera_source = load_scene_views(
        args.scene_dir.resolve(),
        args.dataset_root,
        args.subset_name,
        token_payload["selected_view_indices"],
        int(args.target_size),
    )
    if len(views) != int(tokens.shape[1]):
        raise ValueError(f"Scene views {len(views)} do not match token views {tokens.shape[1]}")

    query_count = int(queries["positions"].shape[0])
    feature_dim = int(tokens.shape[3])
    feature_sum = np.zeros((query_count, feature_dim), dtype=np.float64)
    feature_sq_sum = np.zeros((query_count, feature_dim), dtype=np.float64)
    support = np.zeros((query_count,), dtype=np.int32)
    token_ids = np.full((query_count, len(views)), -1, dtype=np.int32)
    uv_all = np.full((query_count, len(views), 2), np.nan, dtype=np.float32)
    z_all = np.full((query_count, len(views)), np.nan, dtype=np.float32)

    for slot, view in enumerate(views):
        uv, z, _cam = project_queries(queries["positions"], view["world_to_cam"], view["intrinsic"])
        patch_ids, inside = patch_ids_from_uv(
            uv,
            z,
            view["mask"],
            int(args.target_size),
            patch_start_idx,
            patch_grid,
            bool(args.use_mask_gate),
        )
        token_ids[:, slot] = patch_ids.astype(np.int32)
        uv_all[:, slot, :] = uv
        z_all[:, slot] = z
        valid_rows = np.flatnonzero(inside & (patch_ids >= patch_start_idx) & (patch_ids < tokens.shape[2]))
        if valid_rows.size:
            gathered = tokens[0, slot, patch_ids[valid_rows], :].astype(np.float32)
            feature_sum[valid_rows] += gathered.astype(np.float64)
            feature_sq_sum[valid_rows] += (gathered.astype(np.float64) ** 2)
            support[valid_rows] += 1

    mean_features = np.zeros((query_count, feature_dim), dtype=np.float32)
    var_features = np.zeros((query_count, feature_dim), dtype=np.float32)
    valid = support > 0
    mean_features[valid] = (feature_sum[valid] / support[valid, None]).astype(np.float32)
    var_features[valid] = np.maximum(
        feature_sq_sum[valid] / support[valid, None] - feature_sum[valid] ** 2 / (support[valid, None] ** 2),
        0.0,
    ).astype(np.float32)

    family_support: dict[str, Any] = {}
    families = queries["families"]
    for family, _count in query_spec:
        mask = families == family
        values = support[mask]
        family_support[family] = {
            "query_count": int(mask.sum()),
            "support_ge_1": int((values >= 1).sum()),
            "support_ge_2": int((values >= 2).sum()),
            "support_ge_3": int((values >= 3).sum()),
            "mean_support": float(values.mean()) if values.size else 0.0,
            "max_support": int(values.max()) if values.size else 0,
        }

    output_npz = output_dir / "b_fus3d_query_evidence_cache.npz"
    np.savez_compressed(
        output_npz,
        query_indices=queries["indices"].astype(np.int64),
        query_positions=queries["positions"].astype(np.float32),
        query_part_ids=queries["part_ids"].astype(np.int64),
        query_families=families,
        support=support.astype(np.int32),
        token_ids=token_ids.astype(np.int32),
        uv=uv_all.astype(np.float32),
        depth=z_all.astype(np.float32),
        mean_features=mean_features.astype(np.float16),
        variance_features=var_features.astype(np.float16),
        selected_view_indices=np.asarray(token_payload["selected_view_indices"], dtype=np.int32),
        patch_start_idx=np.asarray([patch_start_idx], dtype=np.int32),
        patch_grid=np.asarray([patch_grid], dtype=np.int32),
    )

    summary = {
        **STRICT_FACTS,
        "task": "b_fus3d_query_evidence_cache",
        "truthful_status": "research_query_evidence_only_not_decoder_not_candidate_not_teacher",
        "contract": CONTRACT,
        "summary": {
            "scene_dir": str(args.scene_dir.resolve()),
            "token_cache": token_payload["path"],
            "template_payload": str(args.template_payload.resolve()),
            "camera_source": camera_source,
            "selected_view_indices": token_payload["selected_view_indices"],
            "token_shape": list(tokens.shape),
            "patch_start_idx": patch_start_idx,
            "patch_grid": patch_grid,
            "target_size": int(args.target_size),
            "query_count": query_count,
            "feature_dim": feature_dim,
            "mask_gate": bool(args.use_mask_gate),
            "queries_support_ge_1": int((support >= 1).sum()),
            "queries_support_ge_2": int((support >= 2).sum()),
            "queries_support_ge_3": int((support >= 3).sum()),
            "mean_support": float(support.mean()) if support.size else 0.0,
            "max_support": int(support.max()) if support.size else 0,
        },
        "query_spec": [{"family": family, "count": count} for family, count in query_spec],
        "family_support": family_support,
        "outputs": {
            "query_evidence_npz": str(output_npz),
            "summary_json": str(output_dir / "b_fus3d_query_evidence_cache_summary.json"),
            "summary_md": str(output_dir / "b_fus3d_query_evidence_cache_summary.md"),
        },
        "decision": (
            "3D surface queries now have view-aware VGGT patch-token evidence. "
            "This is a necessary cache for a real 3D latent/SDF decoder, but it is "
            "not a decoder, not a learned surface, not a visual pass, and not a "
            "teacher/candidate export."
        ),
    }
    summary = json_ready(summary)
    (output_dir / "b_fus3d_query_evidence_cache_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "b_fus3d_query_evidence_cache_summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
