from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
LOCAL_ROOT = REPO_ROOT / "output" / "surface_research_preflight_local"
DEFAULT_CASE_ROOT = REPO_ROOT / "output" / "training_cases" / "0012_11_frame0000_6views_smplx_native_prior_v15"
DEFAULT_OUTPUT_DIR = LOCAL_ROOT / "V17_smplx_residual_surface_optimizer"
DEFAULT_JSON = REPORTS / "20260508_v17_smplx_residual_surface_optimizer.json"
DEFAULT_MD = REPORTS / "20260508_v17_smplx_residual_surface_optimizer.md"

FORBIDDEN_OUTPUT_TOKENS = (
    "predictions",
    "teacher_export",
    "candidate_export",
    "formal_candidate",
    "strict_gate_registry",
    "strict_pass",
    "candidate_gate",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve() if value.exists() else value)
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


def safe_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    lower = resolved.as_posix().lower()
    if "surface_research_preflight_local" not in lower or "v17_smplx_residual_surface_optimizer" not in lower:
        raise ValueError(f"Refusing non-V17 research output path: {resolved}")
    for token in FORBIDDEN_OUTPUT_TOKENS:
        if token in lower:
            raise ValueError(f"Refusing forbidden output token {token!r}: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def finite_stats(arr: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    values = np.asarray(arr, dtype=np.float32)
    if mask is not None:
        values = values[np.asarray(mask).astype(bool)]
    values = values.reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": int(values.size), "finite": 0}
    return {
        "count": int(values.size),
        "finite": int(finite.size),
        "min": float(finite.min()),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "max": float(finite.max()),
    }


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if colors is None:
        cols = np.full((pts.shape[0], 3), 210, dtype=np.uint8)
    else:
        cols = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)[finite]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {pts.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(pts, cols):
            handle.write(
                f"{float(point[0]):.6f} {float(point[1]):.6f} {float(point[2]):.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def build_residual_surface(case_root: Path, output_dir: Path, sample_limit: int) -> dict[str, Any]:
    inputs = load_npz(case_root / "inputs.npz")
    targets = load_npz(case_root / "targets.npz")
    prior_points = np.asarray(targets["prior_points"], dtype=np.float32)
    world_points = np.asarray(targets["world_points"], dtype=np.float32)
    prior_normals = np.asarray(targets["prior_normals"], dtype=np.float32)
    prior_mask = np.asarray(targets["smplx_native_visible_mask"], dtype=bool)
    raw_mask = np.asarray(inputs["point_masks"], dtype=bool)
    body_mask = np.asarray(targets.get("smplx_body_anchor_mask", prior_mask), dtype=bool)
    hand_mask = np.asarray(targets.get("smplx_hand_anchor_mask", np.zeros_like(prior_mask)), dtype=bool)

    evidence_mask = prior_mask & raw_mask & np.isfinite(prior_points).all(axis=-1) & np.isfinite(world_points).all(axis=-1)
    residual = np.zeros_like(prior_points, dtype=np.float32)
    residual[evidence_mask] = world_points[evidence_mask] - prior_points[evidence_mask]
    residual_norm = np.linalg.norm(residual, axis=-1)

    weak_residual = np.zeros_like(residual, dtype=np.float32)
    weak_residual[body_mask & evidence_mask] = 0.10 * residual[body_mask & evidence_mask]
    weak_residual[hand_mask & evidence_mask] = 0.25 * residual[hand_mask & evidence_mask]

    normal_len = np.linalg.norm(prior_normals, axis=-1, keepdims=True)
    normal_unit = np.divide(prior_normals, np.maximum(normal_len, 1e-6), out=np.zeros_like(prior_normals), where=normal_len > 1e-6)
    normal_relief = 0.003 * normal_unit
    residual_surface = prior_points + weak_residual + normal_relief * evidence_mask[..., None]

    flat_mask = evidence_mask.reshape(-1)
    flat_points = residual_surface.reshape(-1, 3)
    flat_prior = prior_points.reshape(-1, 3)
    flat_residual_norm = residual_norm.reshape(-1)
    valid_indices = np.flatnonzero(flat_mask)
    if valid_indices.size > sample_limit:
        rng = np.random.default_rng(20260508)
        valid_indices = np.sort(rng.choice(valid_indices, size=sample_limit, replace=False))

    colors = np.zeros((valid_indices.size, 3), dtype=np.uint8)
    sampled_res = flat_residual_norm[valid_indices]
    if sampled_res.size:
        scale = np.percentile(sampled_res[np.isfinite(sampled_res)], 95) if np.isfinite(sampled_res).any() else 1.0
        scaled = np.clip(sampled_res / max(float(scale), 1e-6), 0.0, 1.0)
        colors[:, 0] = (255 * scaled).astype(np.uint8)
        colors[:, 1] = (180 * (1.0 - scaled)).astype(np.uint8)
        colors[:, 2] = 80

    residual_ply = output_dir / "v17_smplx_residual_surface_points.ply"
    prior_ply = output_dir / "v17_smplx_prior_sample_points.ply"
    write_ply(residual_ply, flat_points[valid_indices], colors)
    write_ply(prior_ply, flat_prior[valid_indices], np.full_like(colors, 180, dtype=np.uint8))

    return {
        "evidence_pixels": int(evidence_mask.sum()),
        "body_pixels": int((body_mask & evidence_mask).sum()),
        "hand_pixels": int((hand_mask & evidence_mask).sum()),
        "sampled_points": int(valid_indices.size),
        "residual_norm_stats": finite_stats(residual_norm, evidence_mask),
        "body_residual_norm_stats": finite_stats(residual_norm, body_mask & evidence_mask),
        "hand_residual_norm_stats": finite_stats(residual_norm, hand_mask & evidence_mask),
        "outputs": {
            "residual_surface_ply": residual_ply,
            "prior_sample_ply": prior_ply,
        },
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V17 SMPL-X Residual Surface Optimizer",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Research-only minimal V17 stub. It does not write predictions, teacher package, candidate package, registry, or strict pass state.",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Metrics",
        "",
    ]
    for key, value in summary["metrics"].items():
        lines.append(f"- {key}: `{json_ready(value)}`")
    lines.extend(["", "## Outputs", ""])
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{json_ready(value)}`")
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- {item}" for item in summary.get("blockers", [])] or ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal V17 SMPL-X anchored residual surface research stub.")
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--sample-limit", type=int, default=25000)
    args = parser.parse_args()

    output_dir = safe_output_dir(args.output_dir)
    if not (args.case_root / "inputs.npz").is_file() or not (args.case_root / "targets.npz").is_file():
        raise FileNotFoundError(f"Missing V15 native prior case inputs/targets under {args.case_root}")

    metrics = build_residual_surface(args.case_root, output_dir, int(args.sample_limit))
    summary = {
        "task": "v17_smplx_residual_surface_optimizer",
        "created_utc": utc_now(),
        "status": "v17_smplx_residual_surface_research_stub_ready",
        "research_only": True,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "no_package_write": True,
        "decision": "V17 minimal residual surface artifact is available for research review after V16 negative/inconclusive microfit. It is not a strict teacher or candidate.",
        "metrics": {k: v for k, v in metrics.items() if k != "outputs"},
        "outputs": metrics["outputs"],
        "blockers": [
            "This is a residual research stub, not a differentiable nvdiffrast optimizer yet.",
            "It cannot be promoted without strict visual/region/6-view audits.",
        ],
    }
    write_json(args.output_json, summary)
    write_json(output_dir / "summary.json", summary)
    write_markdown(args.output_md, summary)
    print(json.dumps(json_ready({"status": summary["status"], "json": args.output_json, "md": args.output_md}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
