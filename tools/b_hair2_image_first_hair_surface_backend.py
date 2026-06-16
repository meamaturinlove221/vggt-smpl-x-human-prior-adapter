from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for root in (REPO_ROOT, TOOLS_DIR):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import b_hair0_backend_smoke as hair1  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("output/surface_research_preflight_local/B_hair2_image_first_hair_surface_backend")
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_hair2_image_first_status.md")
DEFAULT_STATUS_JSON = Path("reports/20260507_b_hair2_image_first_status.json")

CONTROL_NAMES = ("real_image_real_token", "real_image_zero_token", "mask_only", "shuffle_token", "zero_token", "random_root")
STRICT_FACTS = hair1.STRICT_FACTS
CONTRACT = {
    "research_only": True,
    "local_only": True,
    "image_first_hair_backend": True,
    "no_cloud": True,
    "no_train": True,
    "no_predictions_write": True,
    "no_checkpoint_write": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_registry_write": True,
    "no_strict_pass_write": True,
    "not_teacher": True,
    "not_candidate": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B-hair2 image-first hairline/head-top surface backend smoke.")
    parser.add_argument("--scene-dir", type=Path, default=hair1.DEFAULT_SCENE_DIR)
    parser.add_argument("--template-payload", type=Path, default=hair1.DEFAULT_TEMPLATE_PAYLOAD)
    parser.add_argument("--hair0-arrays", type=Path, default=hair1.DEFAULT_HAIR0_ARRAYS)
    parser.add_argument("--query-evidence", type=Path, default=hair1.DEFAULT_QUERY_EVIDENCE)
    parser.add_argument("--latent-real", type=Path, default=hair1.DEFAULT_LATENT_REAL)
    parser.add_argument("--latent-shuffle", type=Path, default=hair1.DEFAULT_LATENT_SHUFFLE)
    parser.add_argument("--latent-zero", type=Path, default=hair1.DEFAULT_LATENT_ZERO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--status-json", type=Path, default=DEFAULT_STATUS_JSON)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--target-size", type=int, default=160)
    parser.add_argument("--max-roots", type=int, default=360)
    parser.add_argument("--chain-steps", type=int, default=7)
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    return hair1.json_ready(value)


def write_json(path: Path, payload: Any) -> None:
    hair1.write_json(path, payload)


def safe_path(path: Path) -> None:
    text = str(path).replace("\\", "/").lower()
    forbidden = ("strict_pass", "teacher_export", "candidate_export", "predictions", "checkpoint")
    bad = [item for item in forbidden if item in text]
    if bad:
        raise ValueError(f"Refusing forbidden output path tokens {bad}: {path}")


def image_boundary_scores(roots: dict[str, np.ndarray], views: list[dict[str, Any]], cameras: dict[str, dict[str, np.ndarray]], target_size: int) -> dict[str, np.ndarray]:
    points = np.asarray(roots["root_points"], dtype=np.float32)
    support = np.zeros((points.shape[0],), dtype=np.float32)
    edge = np.zeros((points.shape[0],), dtype=np.float32)
    contrast = np.zeros((points.shape[0],), dtype=np.float32)
    for view in views:
        camera = cameras[hair1.normalize_camera_id(view["camera_id"])]
        intrinsic = hair1.align_intrinsics_for_loaded_scene_view(np.asarray(camera["intrinsic"], dtype=np.float32), view, target_size)
        uv, depth = hair1.project_points(points, np.asarray(camera["world_to_cam"], dtype=np.float32), intrinsic)
        xi = np.rint(uv[:, 0]).astype(np.int64)
        yi = np.rint(uv[:, 1]).astype(np.int64)
        inside = (
            np.isfinite(uv).all(axis=1)
            & np.isfinite(depth)
            & (depth > 1e-6)
            & (xi >= 1)
            & (xi < target_size - 1)
            & (yi >= 1)
            & (yi < target_size - 1)
        )
        if not np.any(inside):
            continue
        mask = np.asarray(view["mask"], dtype=bool)
        rgb = np.asarray(view["rgb"], dtype=np.float32) / 255.0
        gx = np.abs(mask[:, 2:].astype(np.int16) - mask[:, :-2].astype(np.int16))
        gy = np.abs(mask[2:, :].astype(np.int16) - mask[:-2, :].astype(np.int16))
        edge_map = np.zeros(mask.shape, dtype=np.float32)
        edge_map[:, 1:-1] += gx
        edge_map[1:-1, :] += gy
        gray = rgb.mean(axis=2)
        cx = np.abs(gray[:, 2:] - gray[:, :-2])
        cy = np.abs(gray[2:, :] - gray[:-2, :])
        contrast_map = np.zeros(mask.shape, dtype=np.float32)
        contrast_map[:, 1:-1] += cx
        contrast_map[1:-1, :] += cy
        idx = np.flatnonzero(inside)
        support[idx] += mask[yi[idx], xi[idx]].astype(np.float32)
        edge[idx] += edge_map[yi[idx], xi[idx]]
        contrast[idx] += contrast_map[yi[idx], xi[idx]]
    denom = np.maximum(1.0, np.asarray(roots["root_support"], dtype=np.float32))
    return {
        "image_mask_support": np.clip(support / max(1, len(views)), 0.0, 1.0),
        "image_edge_score": hair1.robust_normalize(edge / denom),
        "image_contrast_score": hair1.robust_normalize(contrast / denom),
    }


def select_image_first_roots(hair0: dict[str, np.ndarray], template: dict[str, np.ndarray], views: list[dict[str, Any]], cameras: dict[str, dict[str, np.ndarray]], args: argparse.Namespace) -> dict[str, np.ndarray]:
    base = hair1.select_roots(hair0, template, max_roots=max(int(args.max_roots) * 3, 512), min_support=int(args.min_support), seed=int(args.seed))
    img = image_boundary_scores(base, views, cameras, int(args.target_size))
    support_norm = np.clip(np.asarray(base["root_support"], dtype=np.float32) / 6.0, 0.0, 1.0)
    score = 0.45 * img["image_edge_score"] + 0.28 * img["image_contrast_score"] + 0.20 * img["image_mask_support"] + 0.07 * support_norm
    rng = np.random.default_rng(int(args.seed))
    order = np.argsort(score + 1e-4 * rng.random(score.shape[0]))[::-1]
    order = order[: int(args.max_roots)]
    out = {key: np.asarray(value)[order] for key, value in base.items()}
    out["image_edge_score"] = img["image_edge_score"][order]
    out["image_contrast_score"] = img["image_contrast_score"][order]
    out["image_mask_support"] = img["image_mask_support"][order]
    out["image_first_root_score"] = score[order].astype(np.float32)
    return out


def make_scores(roots: dict[str, np.ndarray], query: dict[str, np.ndarray], latents: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    root_points = roots["root_points"]
    image = np.clip(0.50 * roots["image_edge_score"] + 0.30 * roots["image_contrast_score"] + 0.20 * roots["image_mask_support"], 0.0, 1.0).astype(np.float32)
    support = np.clip(roots["root_support"] / 6.0, 0.0, 1.0).astype(np.float32)
    query_readout = hair1.query_hairline_readout(query, root_points)
    query_norm = np.clip(query_readout["query_support"] / 6.0, 0.0, 1.0).astype(np.float32)
    real_readout = hair1.latent_root_readout(root_points, latents["real"])
    shuffle_readout = hair1.latent_root_readout(root_points, latents["shuffle"])
    zero_readout = hair1.latent_root_readout(root_points, latents["zero"])
    def token_residual(readout: dict[str, np.ndarray]) -> np.ndarray:
        evidence = hair1.robust_normalize(readout["evidence_score"])
        token = hair1.robust_normalize(np.nan_to_num(readout["token_cosine"], nan=0.0))
        occ = np.clip(readout["occupancy_ratio"], 0.0, 1.0)
        return np.clip(0.45 * evidence + 0.35 * token + 0.20 * occ, 0.0, 1.0).astype(np.float32)
    residual_real = token_residual(real_readout)
    residual_shuffle = token_residual(shuffle_readout)
    residual_zero = token_residual(zero_readout)
    out = {
        "real_image_real_token": 0.72 * image + 0.14 * query_norm + 0.14 * residual_real,
        "real_image_zero_token": 0.78 * image + 0.14 * query_norm + 0.08 * residual_zero,
        "mask_only": 0.64 * roots["image_mask_support"] + 0.26 * support + 0.10 * query_norm,
        "shuffle_token": 0.72 * image + 0.14 * query_norm + 0.14 * residual_shuffle,
        "zero_token": 0.72 * image + 0.14 * query_norm + 0.14 * residual_zero,
    }
    rng = np.random.default_rng(20260507)
    out["random_root"] = np.clip(rng.random(root_points.shape[0]).astype(np.float32), 0.0, 1.0)
    rows: dict[str, dict[str, np.ndarray]] = {}
    for name, score in out.items():
        rows[name] = {
            "root_score": np.clip(score, 0.0, 1.0).astype(np.float32),
            "image_score": image,
            "query_support_norm": query_norm,
            "support_norm": support,
            "token_residual_real": residual_real,
            "token_residual_shuffle": residual_shuffle,
            "token_residual_zero": residual_zero,
        }
    return rows


def write_report(path: Path, summary: dict[str, Any]) -> None:
    comp = summary["comparison"]
    lines = [
        "# B-hair2 Image-First Hair Surface Backend",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only image-first hairline/head-top backend smoke. No cloud/export/pass.",
        "",
        "## Strict Truth",
        "",
        "```text",
        "strict_candidate_passes = 0",
        "strict_teacher_passes = 0",
        "formal cloud train/infer/export = blocked",
        "```",
        "",
        "## Comparison",
        "",
        "```json",
        json.dumps(comp, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Decision",
        "",
        summary["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    for p in (args.output_dir, args.status_report, args.status_json):
        safe_path(p)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    template = hair1.load_template(args.template_payload)
    hair0 = hair1.load_hair0(args.hair0_arrays)
    query = hair1.load_query(args.query_evidence)
    latents = {
        "real": hair1.load_latent(args.latent_real, "real"),
        "shuffle": hair1.load_latent(args.latent_shuffle, "shuffle"),
        "zero": hair1.load_latent(args.latent_zero, "zero"),
    }
    views, cameras, camera_source = hair1.load_views(args.scene_dir, np.asarray(hair0["selected_view_indices"], dtype=np.int32), args.dataset_root, args.subset_name, int(args.target_size))
    roots = select_image_first_roots(hair0, template, views, cameras, args)
    scores = make_scores(roots, query, latents)
    chains = {}
    metrics = {}
    outputs = {}
    for name in CONTROL_NAMES:
        chain = hair1.build_strand_chain(roots, scores[name], control="real" if name == "real_image_real_token" else name, chain_steps=int(args.chain_steps))
        chains[name] = chain
        ply = args.output_dir / f"b_hair2_{name}_image_first_strand_chain.ply"
        hair1.write_chain_ply(ply, chain, int(args.chain_steps))
        metrics[name] = hair1.render_control(name, chain, views, cameras, target_size=int(args.target_size), point_radius=int(args.point_radius), output_dir=args.output_dir)
        metrics[name]["root_score"] = hair1.stats(scores[name]["root_score"])
        outputs[name] = {"ply": str(ply)}
    hair1.make_contact_sheet(args.output_dir / "b_hair2_image_first_contact_sheet.png", views, chains, cameras, target_size=int(args.target_size), point_radius=int(args.point_radius))
    comp = {
        "real_minus_real_image_zero_token_iou": float(metrics["real_image_real_token"]["mean_iou"] - metrics["real_image_zero_token"]["mean_iou"]),
        "real_minus_mask_only_iou": float(metrics["real_image_real_token"]["mean_iou"] - metrics["mask_only"]["mean_iou"]),
        "real_minus_shuffle_token_iou": float(metrics["real_image_real_token"]["mean_iou"] - metrics["shuffle_token"]["mean_iou"]),
        "real_minus_zero_token_iou": float(metrics["real_image_real_token"]["mean_iou"] - metrics["zero_token"]["mean_iou"]),
        "real_minus_mask_only_root_score": float(metrics["real_image_real_token"]["root_score"]["mean"] - metrics["mask_only"]["root_score"]["mean"]),
        "real_minus_real_image_zero_token_root_score": float(metrics["real_image_real_token"]["root_score"]["mean"] - metrics["real_image_zero_token"]["root_score"]["mean"]),
        "real_overfill_minus_hair1": float(metrics["real_image_real_token"]["mean_overfill_ratio"] - 0.7604490371806167),
    }
    success = bool(comp["real_minus_mask_only_iou"] > 0.0 and comp["real_minus_real_image_zero_token_iou"] > 0.0 and comp["real_overfill_minus_hair1"] <= 0.0)
    decision = (
        "RESEARCH_ONLY_PROGRESS: image-first hair real+token beats mask-only/zero and does not increase B-hair1 overfill."
        if success
        else "FAIL: B-hair2 wrote image-first hair artifacts, but real+token did not satisfy mask-only/zero/overfill gates."
    )
    summary = {
        "task": "b_hair2_image_first_hair_surface_backend",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "research_only_b_hair2_image_first_no_export",
        "success": success,
        "pass": False,
        "strict_facts": STRICT_FACTS,
        "contract": CONTRACT,
        "inputs": {
            "scene_dir": str(args.scene_dir.resolve()),
            "hair0_arrays": str(args.hair0_arrays.resolve()),
            "query_evidence": str(args.query_evidence.resolve()),
            "camera_source": camera_source,
        },
        "primitive": {
            "root_count": int(roots["root_points"].shape[0]),
            "chain_steps": int(args.chain_steps),
            "chain_point_count": int(roots["root_points"].shape[0] * int(args.chain_steps)),
            "root_source": "image/mask boundary ranked hair0 hairline roots, token residual refinement only",
        },
        "controls": metrics,
        "comparison": comp,
        "outputs": {
            "output_dir": str(args.output_dir.resolve()),
            "summary_json": str((args.output_dir / "b_hair2_image_first_summary.json").resolve()),
            "report_md": str((args.output_dir / "b_hair2_image_first_report.md").resolve()),
            "contact_sheet": str((args.output_dir / "b_hair2_image_first_contact_sheet.png").resolve()),
            "control_outputs": outputs,
        },
        "decision": decision,
    }
    write_json(args.output_dir / "b_hair2_image_first_summary.json", summary)
    write_report(args.output_dir / "b_hair2_image_first_report.md", summary)
    write_report(args.status_report, summary)
    write_json(args.status_json, summary)
    print(json.dumps({"status": summary["status"], "success": success, "decision": decision}, ensure_ascii=False))


if __name__ == "__main__":
    main()
