from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from data.datasets.human_surface_sdf_dataset import (  # noqa: E402
    CONTROL_NAMES,
    FAMILY_ORDER,
    HumanSurfaceSDFDataset,
    apply_control_features,
    json_ready,
    scalar_stats,
)


DEFAULT_CONFIG = REPO_ROOT / "training/config/b_fus3d2_human_dataset.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/surface_research_cloud_preflight/Cloud_A_B_Fus3D2_human_dataset_train_smoke"
DEFAULT_STATUS_MD = REPO_ROOT / "reports/20260507_v8_cloud_a_b_fus3d2_status.md"
DEFAULT_STATUS_JSON = REPO_ROOT / "reports/20260507_v8_cloud_a_b_fus3d2_status.json"

STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
    "teacher_export": "blocked",
    "candidate_export": "blocked",
    "predictions_export": "blocked",
    "registry_write": "blocked",
}
CONTRACT = {
    "research_only": True,
    "dataset_level_train_smoke": True,
    "synthetic_mesh_supervision_allowed": True,
    "no_predictions_write": True,
    "no_teacher_export": True,
    "no_candidate_export": True,
    "no_registry_write": True,
    "no_strict_pass_write": True,
    "no_cloud_guard_write": True,
    "no_referee_write": True,
    "writes_checkpoint": False,
    "not_teacher": True,
    "not_candidate": True,
}
FORBIDDEN_OUTPUT_TOKENS = (
    "strict_pass",
    "teacher_export",
    "candidate_export",
    "predictions.npz",
    "strict_gate_registry",
)


class TinySDFClassifier(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(int(input_dim), int(hidden_dim)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V8-A research-only B-Fus3D2 human dataset-level train smoke. "
            "Uses available query/template mesh supervision, or external "
            "surface_sdf cases if supplied, and writes bounded diagnostics only."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--query-cache", type=Path)
    parser.add_argument("--template-payload", type=Path)
    parser.add_argument("--external-case-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--status-report", type=Path)
    parser.add_argument("--status-json", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}


def cfg_get(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def resolve_path(value: Any, default: Path) -> Path:
    path = default if value in (None, "") else Path(str(value))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.expanduser().resolve()


def ensure_safe_output(path: Path) -> None:
    text = str(path).replace("\\", "/").lower()
    for token in FORBIDDEN_OUTPUT_TOKENS:
        if token in text:
            raise ValueError(f"Refusing output path containing forbidden token {token!r}: {path}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def normalize_train_eval(features: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    features = np.asarray(features, dtype=np.float32)
    mean = features[train_idx].mean(axis=0, keepdims=True)
    std = features[train_idx].std(axis=0, keepdims=True)
    normed = (features - mean) / np.clip(std, 1e-6, None)
    normed = np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return normed, {
        "feature_dim": int(features.shape[1]),
        "mean_stats": scalar_stats(mean.reshape(-1)),
        "std_stats": scalar_stats(std.reshape(-1)),
    }


def split_indices(labels: np.ndarray, families: np.ndarray, case_ids: np.ndarray, seed: int, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    train_rows: list[int] = []
    test_rows: list[int] = []
    labels_i = np.asarray(labels).astype(np.int64)
    families_s = np.asarray(families).astype(str)
    case_s = np.asarray(case_ids).astype(str)
    for case in sorted(set(case_s.tolist())):
        for family in sorted(set(families_s.tolist())):
            for label in sorted(set(labels_i.tolist())):
                rows = np.flatnonzero((case_s == case) & (families_s == family) & (labels_i == label))
                if rows.size == 0:
                    continue
                rng.shuffle(rows)
                if rows.size == 1:
                    train_rows.extend(rows.tolist())
                    continue
                n_train = int(round(rows.size * float(train_fraction)))
                n_train = min(max(1, n_train), rows.size - 1)
                train_rows.extend(rows[:n_train].tolist())
                test_rows.extend(rows[n_train:].tolist())
    if not test_rows:
        all_rows = np.arange(labels_i.shape[0], dtype=np.int64)
        rng.shuffle(all_rows)
        cut = max(1, int(round(all_rows.size * float(train_fraction))))
        train_rows = all_rows[:cut].tolist()
        test_rows = all_rows[cut:].tolist()
    return np.asarray(sorted(train_rows), dtype=np.int64), np.asarray(sorted(test_rows), dtype=np.int64)


def binary_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    prob = np.asarray(prob, dtype=np.float32).reshape(-1)
    pred = prob >= 0.5
    truth = y_true >= 0.5
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    tn = int(np.count_nonzero(~pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    acc = float((tp + tn) / max(y_true.size, 1))
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    iou = float(tp / max(tp + fp + fn, 1))
    eps = 1e-7
    bce = -np.mean(y_true * np.log(np.clip(prob, eps, 1.0)) + (1.0 - y_true) * np.log(np.clip(1.0 - prob, eps, 1.0)))
    return {
        "count": int(y_true.size),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "bce": float(bce),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "positive_ratio": float(np.mean(truth)) if truth.size else 0.0,
        "predicted_positive_ratio": float(np.mean(pred)) if pred.size else 0.0,
    }


def grouped_metrics(y_true: np.ndarray, prob: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    groups = np.asarray(groups).astype(str)
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        if mask.any():
            out[group] = binary_metrics(y_true[mask], prob[mask])
    return out


def train_control(
    *,
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    families: np.ndarray,
    case_ids: np.ndarray,
    max_steps: int,
    max_hours: float,
    batch_size: int,
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    x_norm, norm_stats = normalize_train_eval(features, train_idx)
    x = torch.as_tensor(x_norm, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels.astype(np.float32), dtype=torch.float32, device=device)
    train_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    model = TinySDFClassifier(x.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    rng = np.random.default_rng(int(seed))
    trace: list[dict[str, Any]] = []
    start = time.monotonic()
    max_seconds = max(1.0, float(max_hours) * 3600.0)
    steps_done = 0
    batch_size = max(1, int(batch_size))
    for step in range(int(max_steps)):
        if time.monotonic() - start > max_seconds:
            break
        local = rng.choice(train_idx, size=min(batch_size, train_idx.size), replace=train_idx.size < batch_size)
        batch = torch.as_tensor(local, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x[batch])
        loss = F.binary_cross_entropy_with_logits(logits, y[batch])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        steps_done = step + 1
        if step == 0 or (step + 1) % max(1, int(max_steps) // 5) == 0 or step + 1 == int(max_steps):
            with torch.no_grad():
                train_prob = torch.sigmoid(model(x[train_t])).detach().cpu().numpy()
            train_m = binary_metrics(labels[train_idx], train_prob)
            trace.append(
                {
                    "step": int(step + 1),
                    "loss": float(loss.detach().cpu()),
                    "train_accuracy": train_m["accuracy"],
                    "train_iou": train_m["iou"],
                    "elapsed_seconds": float(time.monotonic() - start),
                }
            )
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).detach().cpu().numpy().astype(np.float32)
    elapsed = float(time.monotonic() - start)
    metrics = {
        "control": name,
        "steps_done": int(steps_done),
        "elapsed_seconds": elapsed,
        "stopped_by_max_hours": bool(elapsed >= max_seconds and steps_done < int(max_steps)),
        "normalization": norm_stats,
        "trace": trace,
        "train": binary_metrics(labels[train_idx], prob[train_idx]),
        "eval": binary_metrics(labels[test_idx], prob[test_idx]),
        "family_eval": grouped_metrics(labels[test_idx], prob[test_idx], families[test_idx]),
        "case_eval": grouped_metrics(labels[test_idx], prob[test_idx], case_ids[test_idx]),
        "prob_stats": scalar_stats(prob),
    }
    return metrics, prob


def write_pointcloud_ply(path: Path, points: np.ndarray, prob: np.ndarray, families: np.ndarray, *, max_points: int = 3000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    prob = np.asarray(prob, dtype=np.float32).reshape(-1)
    families = np.asarray(families).astype(str).reshape(-1)
    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points).round().astype(np.int64)
        points = points[idx]
        prob = prob[idx]
        families = families[idx]
    colors_by_family = {
        "full_body": np.asarray([150, 150, 150], dtype=np.uint8),
        "face_core": np.asarray([245, 185, 65], dtype=np.uint8),
        "hairline": np.asarray([155, 80, 220], dtype=np.uint8),
        "left_hand": np.asarray([35, 135, 245], dtype=np.uint8),
        "right_hand": np.asarray([250, 95, 55], dtype=np.uint8),
    }
    colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    for idx, family in enumerate(families):
        base = colors_by_family.get(str(family), np.asarray([180, 180, 180], dtype=np.uint8)).astype(np.float32)
        colors[idx] = np.clip(base * (0.40 + 0.60 * float(prob[idx])), 0, 255).astype(np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("property float occupancy_probability\n")
        handle.write("end_header\n")
        for point, color, p in zip(points, colors, prob, strict=False):
            handle.write(
                f"{float(point[0]):.7f} {float(point[1]):.7f} {float(point[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {float(p):.7f}\n"
            )


def render_projection(points: np.ndarray, values: np.ndarray, families: np.ndarray, path: Path, title: str) -> None:
    points = np.asarray(points, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    families = np.asarray(families).astype(str)
    if points.shape[0] == 0:
        return
    if points.shape[0] > 2500:
        idx = np.linspace(0, points.shape[0] - 1, 2500).round().astype(np.int64)
        points = points[idx]
        values = values[idx]
        families = families[idx]
    centered = points - np.median(points, axis=0, keepdims=True)
    xy = centered[:, [0, 1]]
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    norm = np.clip((xy - lo[None, :]) / span[None, :], 0.0, 1.0)
    width, height = 440, 440
    canvas = Image.new("RGB", (width, height + 34), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title[:70], fill=(0, 0, 0))
    family_color = {
        "full_body": (125, 125, 125),
        "face_core": (230, 150, 35),
        "hairline": (135, 70, 205),
        "left_hand": (40, 120, 230),
        "right_hand": (230, 80, 45),
    }
    px = np.clip((norm[:, 0] * (width - 1)).round().astype(np.int64), 0, width - 1)
    py = np.clip(((1.0 - norm[:, 1]) * (height - 1)).round().astype(np.int64), 0, height - 1) + 34
    depth = centered[:, 2]
    for idx in np.argsort(depth):
        base = np.asarray(family_color.get(str(families[idx]), (160, 160, 160)), dtype=np.float32)
        color = tuple(np.clip(base * (0.35 + 0.65 * float(values[idx])), 0, 255).astype(np.uint8).tolist())
        x, y = int(px[idx]), int(py[idx])
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def make_contact_sheet(image_paths: list[Path], out_path: Path, title: str) -> None:
    thumbs: list[Image.Image] = []
    labels: list[str] = []
    for path in image_paths:
        if path.is_file():
            thumbs.append(Image.open(path).convert("RGB").resize((260, 280), Image.Resampling.BICUBIC))
            labels.append(path.stem)
    if not thumbs:
        return
    cols = min(4, len(thumbs))
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 260, rows * 308 + 34), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=(0, 0, 0))
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * 260
        y = 34 + (idx // cols) * 308
        sheet.paste(thumb, (x, y))
        draw.text((x + 6, y + 284), labels[idx][:34], fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V8-A Cloud A B-Fus3D2 Human Dataset Train Smoke",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is a research-only dataset-level train smoke. It is not a formal cloud run, not a teacher, not a candidate, and not a predictions export.",
        "",
        "## Strict Truth",
        "",
        "```text",
        f"strict_candidate_passes = {summary['strict_candidate_passes']}",
        f"strict_teacher_passes = {summary['strict_teacher_passes']}",
        "formal cloud train/infer/export = blocked",
        "predictions/teacher/candidate/registry writes = none",
        "```",
        "",
        "## Bounds",
        "",
        "```json",
        json.dumps(summary["run_limits"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Control Metrics",
        "",
        "| control | steps | eval acc | eval IoU | eval BCE | recall | predicted pos |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in summary["control_order"]:
        row = summary["controls"][name]
        ev = row["eval"]
        lines.append(
            f"| `{name}` | {row['steps_done']} | {ev['accuracy']:.4f} | {ev['iou']:.4f} | "
            f"{ev['bce']:.4f} | {ev['recall']:.4f} | {ev['predicted_positive_ratio']:.4f} |"
        )
    lines += [
        "",
        "## Comparison",
        "",
        "```json",
        json.dumps(summary["comparison"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Outputs",
        "",
        f"- summary: `{summary['outputs']['summary_json']}`",
        f"- report: `{summary['outputs']['report_md']}`",
        f"- diagnostics_npz: `{summary['outputs']['diagnostics_npz']}`",
        f"- contact_sheet: `{summary['outputs']['contact_sheet']}`",
        f"- real_ply: `{summary['outputs']['real_ply']}`",
        "",
        "## Decision",
        "",
        summary["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    run_limits = dict(cfg_get(config, "run_limits", {}) or {})
    model_cfg = dict(cfg_get(config, "model", {}) or {})
    dataset_cfg = dict(cfg_get(config, "dataset", {}) or {})
    outputs_cfg = dict(cfg_get(config, "outputs", {}) or {})
    inputs_cfg = dict(cfg_get(config, "inputs", {}) or {})

    max_steps = int(args.max_steps if args.max_steps is not None else run_limits.get("max_steps", 80))
    max_cases = int(args.max_cases if args.max_cases is not None else run_limits.get("max_cases", 3))
    max_hours = float(args.max_hours if args.max_hours is not None else run_limits.get("max_hours", 0.15))
    seed = int(args.seed if args.seed is not None else run_limits.get("seed", 20260507))
    batch_size = int(args.batch_size if args.batch_size is not None else model_cfg.get("batch_size", 512))
    hidden_dim = int(args.hidden_dim if args.hidden_dim is not None else model_cfg.get("hidden_dim", 64))
    lr = float(args.lr if args.lr is not None else model_cfg.get("learning_rate", 0.01))
    weight_decay = float(model_cfg.get("weight_decay", 1e-4))

    query_cache = resolve_path(args.query_cache or inputs_cfg.get("query_cache"), REPO_ROOT / "missing_query_cache.npz")
    template_payload = resolve_path(args.template_payload or inputs_cfg.get("template_payload"), REPO_ROOT / "missing_template_payload.npz")
    external_roots = [Path(v) for v in args.external_case_root]
    external_roots += [resolve_path(v, REPO_ROOT) for v in inputs_cfg.get("external_case_roots", []) or []]
    output_dir = resolve_path(args.output_dir or outputs_cfg.get("output_dir"), DEFAULT_OUTPUT_DIR)
    status_md = resolve_path(args.status_report or outputs_cfg.get("status_report_md"), DEFAULT_STATUS_MD)
    status_json = resolve_path(args.status_json or outputs_cfg.get("status_report_json"), DEFAULT_STATUS_JSON)
    ensure_safe_output(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)
    shell_offsets = tuple(float(v) for v in dataset_cfg.get("shell_offsets", [-0.014, -0.007, 0.0, 0.007, 0.014]))
    dataset = HumanSurfaceSDFDataset(
        query_cache=query_cache,
        template_payload=template_payload,
        external_case_roots=external_roots,
        max_cases=max_cases,
        shell_offsets=shell_offsets,
        feature_bins=int(dataset_cfg.get("feature_bins", 32)),
        seed=seed,
    )
    arrays = dataset.as_arrays()
    labels = arrays["labels"].astype(np.float32)
    families = arrays["families"].astype(str)
    case_ids = arrays["case_ids"].astype(str)
    train_idx, test_idx = split_indices(
        labels,
        families,
        case_ids,
        seed=seed,
        train_fraction=float(dataset_cfg.get("train_fraction", 0.72)),
    )
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    controls_cfg = tuple(str(v) for v in (cfg_get(config, "controls", CONTROL_NAMES) or CONTROL_NAMES))
    control_order = tuple(v for v in controls_cfg if v in CONTROL_NAMES)
    controls: dict[str, Any] = {}
    probabilities: dict[str, np.ndarray] = {}
    image_paths: list[Path] = []
    for idx, control in enumerate(control_order):
        controlled_features = apply_control_features(arrays["features"], control, seed + 1000 + idx)
        metrics, prob = train_control(
            name=control,
            features=controlled_features,
            labels=labels,
            train_idx=train_idx,
            test_idx=test_idx,
            families=families,
            case_ids=case_ids,
            max_steps=max_steps,
            max_hours=max_hours,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed + idx,
            device=device,
        )
        controls[control] = metrics
        probabilities[control] = prob
        png_path = output_dir / f"b_fus3d2_{control}_projection.png"
        render_projection(arrays["positions"], prob, families, png_path, f"B-Fus3D2 {control} occupancy prob")
        image_paths.append(png_path)
        ply_path = output_dir / f"b_fus3d2_{control}_surface_sdf_points.ply"
        write_pointcloud_ply(ply_path, arrays["positions"], prob, families)
        controls[control]["ply"] = str(ply_path.resolve())
        controls[control]["projection_png"] = str(png_path.resolve())

    contact_sheet = output_dir / "b_fus3d2_control_contact_sheet.png"
    make_contact_sheet(image_paths, contact_sheet, "B-Fus3D2 human dataset smoke controls")
    diagnostics_npz = output_dir / "b_fus3d2_human_dataset_train_diagnostics.npz"
    np.savez_compressed(
        diagnostics_npz,
        positions=arrays["positions"].astype(np.float32),
        labels=labels.astype(np.float32),
        sdf=arrays["sdf"].astype(np.float32),
        families=families.astype("<U32"),
        case_ids=case_ids.astype("<U96"),
        sample_offsets=arrays["sample_offsets"].astype(np.float32),
        train_idx=train_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        **{f"{name}_probability": probabilities[name].astype(np.float32) for name in control_order},
    )

    real = controls.get("real", {})
    comparison: dict[str, Any] = {}
    for control in control_order:
        if control == "real" or "real" not in controls:
            continue
        comparison[f"real_minus_{control}_eval_accuracy"] = float(real["eval"]["accuracy"] - controls[control]["eval"]["accuracy"])
        comparison[f"real_minus_{control}_eval_iou"] = float(real["eval"]["iou"] - controls[control]["eval"]["iou"])
        comparison[f"real_minus_{control}_eval_bce"] = float(real["eval"]["bce"] - controls[control]["eval"]["bce"])
    real_beats_zero = comparison.get("real_minus_zero_eval_iou", 0.0) >= -0.02 and comparison.get("real_minus_zero_eval_bce", 0.0) <= 0.05
    real_beats_shuffle = comparison.get("real_minus_shuffle_eval_iou", 0.0) > 0.02 and comparison.get("real_minus_shuffle_eval_bce", 0.0) < 0.0
    real_beats_random = comparison.get("real_minus_random_eval_iou", 0.0) > 0.02 and comparison.get("real_minus_random_eval_bce", 0.0) < 0.0
    research_progress = bool(real_beats_shuffle and real_beats_random)
    decision = (
        "RESEARCH_ONLY_PROGRESS: dataset-level B-Fus3D2 train smoke completed under bounds and real features beat shuffle/random controls. Strict gates remain red."
        if research_progress
        else "RESEARCH_ONLY_SMOKE_COMPLETE_FAIL_CLOSED: dataset-level B-Fus3D2 train smoke ran and wrote diagnostics, but control separation is not strong enough for any formal claim. Strict gates remain red."
    )

    summary = {
        "task": "b_fus3d2_human_dataset_train",
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "research_only_dataset_train_smoke_no_export",
        "truthful_status": "research_only_not_teacher_not_candidate_not_predictions",
        "success": bool(research_progress),
        "pass": False,
        **STRICT_FACTS,
        "strict_facts": STRICT_FACTS,
        "contract": CONTRACT,
        "run_limits": {
            "max_steps": int(max_steps),
            "max_cases": int(max_cases),
            "max_hours": float(max_hours),
            "batch_size": int(batch_size),
            "hidden_dim": int(hidden_dim),
            "learning_rate": float(lr),
            "weight_decay": float(weight_decay),
            "seed": int(seed),
            "device": str(device),
        },
        "inputs": {
            "config": str(args.config.expanduser().resolve()),
            "query_cache": str(query_cache),
            "template_payload": str(template_payload),
            "external_case_roots": [str(Path(v).expanduser().resolve()) for v in external_roots],
        },
        "dataset": {
            "case_count": len(dataset),
            "sample_count": int(labels.shape[0]),
            "train_count": int(train_idx.shape[0]),
            "eval_count": int(test_idx.shape[0]),
            "feature_dim": int(arrays["features"].shape[1]),
            "label_stats": scalar_stats(labels),
            "sdf_stats": scalar_stats(arrays["sdf"]),
            "families": {family: int(np.count_nonzero(families == family)) for family in sorted(set(families.tolist()))},
        },
        "genealogy": dataset.genealogy(),
        "control_order": list(control_order),
        "controls": controls,
        "comparison": comparison,
        "research_progress": research_progress,
        "real_beats_zero": real_beats_zero,
        "real_beats_shuffle": real_beats_shuffle,
        "real_beats_random": real_beats_random,
        "outputs": {
            "output_dir": str(output_dir),
            "summary_json": str((output_dir / "summary.json").resolve()),
            "report_md": str((output_dir / "report.md").resolve()),
            "diagnostics_npz": str(diagnostics_npz.resolve()),
            "contact_sheet": str(contact_sheet.resolve()),
            "real_ply": str((output_dir / "b_fus3d2_real_surface_sdf_points.ply").resolve()),
            "status_report_md": str(status_md),
            "status_report_json": str(status_json),
        },
        "decision": decision,
        "blockers": [
            "strict_candidate_passes remains 0",
            "strict_teacher_passes remains 0",
            "research smoke writes diagnostics only, no predictions.npz",
            "no teacher/candidate/cloud/registry/export side effects",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    write_markdown(output_dir / "report.md", summary)
    write_markdown(status_md, summary)
    write_json(status_json, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "research_progress": research_progress,
                "real_eval_iou": controls.get("real", {}).get("eval", {}).get("iou"),
                "output_dir": str(output_dir),
                "decision": decision,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
