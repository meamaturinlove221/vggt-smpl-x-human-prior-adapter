from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


DEFAULT_REAL = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23/"
    "b_fus3d_latent_grid_evidence_arrays.npz"
)
DEFAULT_SHUFFLE = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23_shuffle/"
    "b_fus3d_latent_grid_evidence_arrays.npz"
)
DEFAULT_ZERO = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D15_latent_grid_evidence_preflight_hybrid6_layer23_zero/"
    "b_fus3d_latent_grid_evidence_arrays.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/surface_research_preflight_local/"
    "B_Fus3D16_latent_field_smoke_fixed_hybrid6_layer23"
)
DEFAULT_STATUS_REPORT = Path("reports/20260507_b_fus3d_latent_field_smoke_status.md")

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
}

CONTRACT = {
    "research_only": True,
    "local_only": True,
    "single_fixed_smoke": True,
    "not_teacher": True,
    "not_candidate": True,
    "no_cloud": True,
    "no_candidate_export": True,
    "no_teacher_export": True,
    "no_registry_write": True,
    "no_strict_pass_write": True,
    "writes_predictions_npz": False,
    "writes_formal_prediction_arrays": False,
    "writes_research_diagnostic_arrays": True,
    "writes_checkpoint": False,
}


class TinyField(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only fixed B-Fus3D16 latent-field smoke. It consumes B15 "
            "latent-grid evidence real/shuffle/zero controls, fits one tiny fixed "
            "field per control, extracts a diagnostic mesh via marching cubes, "
            "and writes visual-review artifacts. It never exports a teacher or "
            "candidate, writes predictions, writes strict pass state, or calls cloud."
        )
    )
    parser.add_argument("--real-arrays", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--shuffle-arrays", type=Path, default=DEFAULT_SHUFFLE)
    parser.add_argument("--zero-arrays", type=Path, default=DEFAULT_ZERO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-report", type=Path, default=DEFAULT_STATUS_REPORT)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--level", type=float, default=0.55)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
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
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def stat_array(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values).reshape(-1)
    finite = np.isfinite(arr)
    if not finite.any():
        return {"count": int(arr.size), "finite": 0}
    data = arr[finite].astype(np.float64)
    return {
        "count": int(arr.size),
        "finite": int(data.size),
        "min": float(data.min()),
        "p10": float(np.quantile(data, 0.10)),
        "median": float(np.quantile(data, 0.50)),
        "mean": float(data.mean()),
        "p90": float(np.quantile(data, 0.90)),
        "max": float(data.max()),
    }


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def infer_grid(points: np.ndarray) -> int:
    n = int(points.shape[0])
    res = int(round(n ** (1.0 / 3.0)))
    if res**3 != n:
        raise ValueError(f"point count {n} is not a dense cube grid")
    return res


def normalize(values: np.ndarray, default: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    out = np.full(arr.shape, float(default), dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return out
    lo, hi = np.percentile(arr[finite], [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(arr[finite].min())
        hi = float(arr[finite].max())
    if hi > lo:
        out[finite] = np.clip((arr[finite] - lo) / (hi - lo), 0.0, 1.0)
    return out


def build_features_and_target(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    points = np.asarray(arrays["points"], dtype=np.float32)
    center = points.mean(axis=0, keepdims=True)
    scale = np.percentile(np.linalg.norm(points - center, axis=1), 95)
    if not np.isfinite(scale) or float(scale) <= 1e-6:
        scale = 1.0
    xyz = (points - center) / float(scale)
    visible = normalize(np.asarray(arrays["visible_count"], dtype=np.float32), default=0.0)
    masks = normalize(np.asarray(arrays["mask_count"], dtype=np.float32), default=0.0)
    token_count = normalize(np.asarray(arrays["token_count"], dtype=np.float32), default=0.0)
    occ = np.asarray(arrays["occupancy_ratio"], dtype=np.float32).reshape(-1)
    rgb_var = normalize(np.nan_to_num(arrays["rgb_variance"], nan=np.nan), default=1.0)
    rgb_good = 1.0 - rgb_var
    token_cos = np.asarray(arrays["token_cosine"], dtype=np.float32).reshape(-1)
    token_good = np.nan_to_num((token_cos + 1.0) * 0.5, nan=0.0, posinf=0.0, neginf=0.0)
    evidence = normalize(np.asarray(arrays["evidence_score"], dtype=np.float32), default=0.0)
    features = np.concatenate(
        [
            xyz.astype(np.float32),
            visible[:, None],
            masks[:, None],
            token_count[:, None],
            rgb_good[:, None],
            token_good[:, None],
            evidence[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    # Fixed diagnostic target: a weak evidence field, not an SDF truth.
    target = np.clip(0.58 * occ + 0.22 * rgb_good + 0.14 * token_good + 0.06 * token_count, 0.0, 1.0).astype(np.float32)
    meta = {
        "feature_dim": int(features.shape[1]),
        "target_stats": stat_array(target),
        "token_good_stats": stat_array(token_good),
        "rgb_good_stats": stat_array(rgb_good),
        "occupancy_stats": stat_array(occ),
    }
    return features, target, meta


def train_fixed_field(features: np.ndarray, target: np.ndarray, *, hidden_dim: int, steps: int, seed: int) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(target, dtype=torch.float32)
    model = TinyField(x.shape[1], int(hidden_dim))
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    losses: list[float] = []
    for _step in range(int(steps)):
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        pred = model(x).detach().cpu().numpy().astype(np.float32)
    return {
        "pred": pred,
        "losses": losses,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
    }


def mesh_components(faces: np.ndarray) -> dict[str, Any]:
    faces = np.asarray(faces, dtype=np.int64)
    parent = np.arange(int(faces.shape[0]), dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    vertex_owner: dict[int, int] = {}
    for face_idx, face in enumerate(faces):
        for vertex in face.tolist():
            prev = vertex_owner.get(int(vertex))
            if prev is None:
                vertex_owner[int(vertex)] = int(face_idx)
            else:
                union(int(face_idx), int(prev))
    roots = np.asarray([find(i) for i in range(faces.shape[0])], dtype=np.int64)
    unique, counts = np.unique(roots, return_counts=True)
    largest = int(counts.max()) if counts.size else 0
    return {
        "component_count": int(unique.size),
        "largest_face_count": largest,
        "largest_component_ratio": float(largest / max(int(faces.shape[0]), 1)),
    }


def write_mesh_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {vertices.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {faces.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex in vertices:
            handle.write(
                f"{float(vertex[0]):.7f} {float(vertex[1]):.7f} {float(vertex[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def extract_mesh(points: np.ndarray, field: np.ndarray, level: float, out_path: Path, color: tuple[int, int, int]) -> dict[str, Any]:
    try:
        from skimage import measure
    except Exception as exc:  # noqa: BLE001
        return {"status": "blocked_no_skimage", "error": repr(exc), "mesh_path": ""}
    res = infer_grid(points)
    grid = np.asarray(field, dtype=np.float32).reshape(res, res, res)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    spacing = (hi - lo) / max(res - 1, 1)
    if not (np.nanmin(grid) <= level <= np.nanmax(grid)):
        return {
            "status": "blocked_level_outside_field_range",
            "level": float(level),
            "field_min": float(np.nanmin(grid)),
            "field_max": float(np.nanmax(grid)),
            "mesh_path": "",
        }
    vertices, faces, _normals, _values = measure.marching_cubes(grid, level=float(level), spacing=tuple(float(v) for v in spacing))
    vertices = vertices.astype(np.float32) + lo[None, :].astype(np.float32)
    faces = faces.astype(np.int64)
    write_mesh_ply(out_path, vertices, faces, color)
    return {
        "status": "mesh_extracted",
        "mesh_path": str(out_path),
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
        "field_min": float(np.nanmin(grid)),
        "field_max": float(np.nanmax(grid)),
        "field_mean": float(np.nanmean(grid)),
        **mesh_components(faces),
    }


def run_control(name: str, path: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    arrays = load_arrays(path)
    features, target, feature_meta = build_features_and_target(arrays)
    fit = train_fixed_field(features, target, hidden_dim=args.hidden_dim, steps=args.steps, seed=args.seed)
    mesh_info = extract_mesh(
        np.asarray(arrays["points"], dtype=np.float32),
        np.asarray(fit["pred"], dtype=np.float32),
        float(args.level),
        out_dir / f"{name}_latent_field_mesh.ply",
        {"real": (190, 190, 190), "shuffle": (230, 170, 80), "zero": (120, 160, 230)}.get(name, (190, 190, 190)),
    )
    pred_path = out_dir / f"{name}_field_values.npz"
    np.savez_compressed(
        pred_path,
        points=np.asarray(arrays["points"], dtype=np.float32),
        target=target.astype(np.float32),
        pred=np.asarray(fit["pred"], dtype=np.float32),
        features=features.astype(np.float32),
    )
    return {
        "name": name,
        "arrays": str(path),
        "field_values": str(pred_path),
        "feature_meta": feature_meta,
        "fit": {
            "steps": int(args.steps),
            "hidden_dim": int(args.hidden_dim),
            "seed": int(args.seed),
            "initial_loss": fit["initial_loss"],
            "final_loss": fit["final_loss"],
            "loss_ratio": float(fit["final_loss"] / max(fit["initial_loss"], 1e-12)) if fit["initial_loss"] else None,
            "pred_stats": stat_array(fit["pred"]),
        },
        "mesh": mesh_info,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Fus3D16 Latent Field Smoke",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is a fixed research-only learned-field smoke over B15 latent-grid evidence.",
        "It is not a teacher, candidate, strict pass, or cloud unblock.",
        "",
        "## Strict Truth",
        "",
        "```text",
        f"strict_candidate_passes = {STRICT_FACTS['strict_candidate_passes']}",
        f"strict_teacher_passes = {STRICT_FACTS['strict_teacher_passes']}",
        f"formal_cloud_train_infer_export = {STRICT_FACTS['formal_cloud_train_infer_export']}",
        "```",
        "",
        "## Controls",
        "",
    ]
    for row in summary["controls"]:
        lines.extend(
            [
                f"### {row['name']}",
                "",
                "```json",
                json.dumps(json_ready({"fit": row["fit"], "mesh": row["mesh"]}), indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            "```json",
            json.dumps(json_ready(summary["decision"]), indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory exists; pass --overwrite: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    controls = [
        run_control("real", args.real_arrays, out_dir, args),
        run_control("shuffle", args.shuffle_arrays, out_dir, args),
        run_control("zero", args.zero_arrays, out_dir, args),
    ]
    real = next(row for row in controls if row["name"] == "real")
    decision = {
        "status": "research_latent_field_smoke_no_pass",
        "mesh_extracted": bool(real["mesh"].get("status") == "mesh_extracted"),
        "sufficient_for_teacher_or_candidate": False,
        "sufficient_for_cloud_unblock": False,
        "interpretation": (
            "This fixed smoke checks whether B15 latent-grid evidence can form a "
            "bounded learned field and mesh-like diagnostic artifact. It is still "
            "a weak visual-hull/evidence field, not a mentor-level human surface."
        ),
        "next_allowed_action": (
            "Render Open3D reviews for the extracted meshes, compare real vs controls, "
            "and freeze if the real mesh remains a shell/slab/template-like surface."
        ),
        "blocked_actions": [
            "do_not_tune_hidden_steps_level_after_this_smoke",
            "do_not_export_teacher_or_candidate",
            "do_not_write_strict_registry",
            "do_not_unblock_cloud",
        ],
    }
    summary = {
        **STRICT_FACTS,
        "status": "research_only_latent_field_smoke_not_teacher_not_candidate",
        "contract": CONTRACT,
        "inputs": {
            "real_arrays": str(args.real_arrays),
            "shuffle_arrays": str(args.shuffle_arrays),
            "zero_arrays": str(args.zero_arrays),
            "steps": int(args.steps),
            "hidden_dim": int(args.hidden_dim),
            "seed": int(args.seed),
            "level": float(args.level),
        },
        "controls": controls,
        "decision": decision,
        "outputs": {
            "summary_json": str(out_dir / "b_fus3d_latent_field_smoke_summary.json"),
            "summary_md": str(out_dir / "b_fus3d_latent_field_smoke_summary.md"),
            "status_report": str(args.status_report),
        },
    }
    (out_dir / "b_fus3d_latent_field_smoke_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir / "b_fus3d_latent_field_smoke_summary.md", summary)
    write_report(args.status_report, summary)
    print(json.dumps(json_ready(decision), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
