from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


STRICT_FACTS = {
    "strict_candidate_passes": 0,
    "strict_teacher_passes": 0,
    "formal_cloud_train_infer_export": "blocked",
    "teacher_export": "blocked",
    "candidate_export": "blocked",
}
CONTRACT = {
    "research_only": True,
    "local_only": True,
    "no_train": True,
    "no_cloud": True,
    "no_candidate_export": True,
    "no_teacher_export": True,
    "no_strict_pass_write": True,
    "not_mano_success_claim": True,
    "not_smplx_hand_residual_success_claim": True,
    "writes_predictions_npz": False,
    "writes_strict_registry": False,
    "writes_candidate": False,
    "writes_teacher": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only B-hand decoder skeleton smoke. It consumes B-hand token smoke metadata "
            "and emits a deterministic toy latent contract for left/right hand token decoder wiring. "
            "It never trains, never writes predictions, never exports teacher/candidate, never writes "
            "strict pass state, and never calls cloud."
        )
    )
    parser.add_argument("--hand-token-smoke", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--local-token-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_summary(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_dir():
        resolved = resolved / "b_hand_token_backend_smoke_summary.json"
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON in {resolved}")
    payload["_resolved_path"] = str(resolved)
    return payload


def side_decoder_contract(side: str, row: dict[str, Any], rng: np.random.Generator, latent_dim: int, local_token_count: int) -> tuple[dict[str, Any], np.ndarray]:
    support_views = int(row.get("support_views", 0) or 0)
    token_count = int(row.get("unique_aggregator_tokens", 0) or 0)
    roi_pixels = int(row.get("total_roi_pixels", 0) or 0)
    smplx_pixels = int(row.get("total_smplx_visible_pixels", 0) or 0)
    camera_views = int(row.get("camera_ray_views", 0) or 0)
    scale = np.asarray(
        [
            support_views / 8.0,
            min(token_count, 128) / 128.0,
            min(roi_pixels, 10000) / 10000.0,
            smplx_pixels / max(roi_pixels, 1),
            camera_views / max(support_views, 1),
        ],
        dtype=np.float32,
    )
    base = rng.normal(0.0, 0.02, size=(local_token_count, latent_dim)).astype(np.float32)
    side_bias = -0.05 if side == "left" else 0.05
    base[:, : min(5, latent_dim)] += scale[: min(5, latent_dim)] + side_bias
    latent = base.astype(np.float16)
    blockers: list[str] = []
    if support_views <= 0:
        blockers.append("missing visible hand views")
    if camera_views <= 0:
        blockers.append("missing camera-ray views")
    if token_count <= 0:
        blockers.append("missing ROI patch-token support")
    blockers.extend(
        [
            "no MANO/local hand mesh decoder implemented",
            "no wrist-arm connected Open3D visual pass",
            "no depth/normal/mask rendering",
            "no strict fullbody-hand gate pass",
        ]
    )
    return (
        {
            "side": side,
            "status": "decoder_contract_only",
            "support_views": support_views,
            "unique_aggregator_tokens": token_count,
            "total_roi_pixels": roi_pixels,
            "total_smplx_visible_pixels": smplx_pixels,
            "camera_ray_views": camera_views,
            "latent_shape": [int(v) for v in latent.shape],
            "latent_stats": {
                "mean": float(latent.astype(np.float32).mean()),
                "std": float(latent.astype(np.float32).std()),
                "abs_p95": float(np.quantile(np.abs(latent.astype(np.float32)), 0.95)),
            },
            "decoder_wiring": {
                "input_tokens": [
                    f"{side}_hand_identity",
                    f"{side}_hand_view_support_histogram",
                    f"{side}_hand_roi_patch_token_set",
                    f"{side}_wrist_arm_anchor_required_before_success",
                    f"{side}_finger_tokens_require_backend_decoder",
                ],
                "required_outputs_before_gate": [
                    f"{side}_hand_mesh_or_surface",
                    f"{side}_hand_visibility",
                    f"{side}_hand_depth_normal_mask_render",
                    f"{side}_hand_wrist_arm_connection_visual",
                ],
            },
            "blockers": blockers,
            "hard_gate_allowed": False,
        },
        latent,
    )


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# B-Hand Decoder Skeleton Smoke",
        "",
        "Status: `research_only_decoder_contract_only`",
        "",
        "This is a deterministic wiring smoke, not a decoder result and not a hand pass.",
        "",
        "## Gate Truth",
        "",
        "```json",
        json.dumps(STRICT_FACTS, indent=2),
        "```",
        "",
        "## Side Contracts",
        "",
    ]
    for side, row in summary["sides"].items():
        lines.extend(
            [
                f"### {side}",
                "",
                f"- support_views: `{row['support_views']}`",
                f"- unique_aggregator_tokens: `{row['unique_aggregator_tokens']}`",
                f"- latent_shape: `{row['latent_shape']}`",
                f"- hard_gate_allowed: `{row['hard_gate_allowed']}`",
                "",
            ]
        )
    lines.extend(["## Blockers", ""])
    for blocker in summary["blockers"]:
        lines.append(f"- {blocker}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    source = load_summary(args.hand_token_smoke)
    sides_payload = source.get("sides") if isinstance(source.get("sides"), dict) else {}
    rng = np.random.default_rng(int(args.seed))
    latent_arrays: dict[str, np.ndarray] = {}
    side_rows: dict[str, Any] = {}
    blockers: list[str] = []
    for side in ("left", "right"):
        row = sides_payload.get(side) if isinstance(sides_payload.get(side), dict) else {}
        side_row, latent = side_decoder_contract(side, row, rng, int(args.latent_dim), int(args.local_token_count))
        side_rows[side] = side_row
        latent_arrays[f"{side}_hand_latent"] = latent
        blockers.extend([f"{side}: {item}" for item in side_row["blockers"]])

    latent_path = output_dir / "b_hand_decoder_skeleton_latents.npz"
    np.savez_compressed(
        latent_path,
        **latent_arrays,
        side_names=np.asarray(["left", "right"]),
        seed=np.asarray([int(args.seed)], dtype=np.int64),
    )
    summary = {
        "task": "b_hand_decoder_skeleton_smoke",
        "schema_version": 1,
        "status": "research_only_decoder_contract_only",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "strict_facts": STRICT_FACTS,
        "contract": CONTRACT,
        "inputs": {
            "hand_token_smoke": source.get("_resolved_path"),
            "latent_dim": int(args.latent_dim),
            "local_token_count": int(args.local_token_count),
            "seed": int(args.seed),
        },
        "sides": side_rows,
        "outputs": {
            "summary_json": str(output_dir / "b_hand_decoder_skeleton_smoke_summary.json"),
            "report_md": str(output_dir / "b_hand_decoder_skeleton_smoke_report.md"),
            "latent_npz": str(latent_path),
        },
        "blockers": blockers,
        "next_allowed_action": (
            "Replace this deterministic contract with a local MANO/local-hand surface decoder and render "
            "hand mask/depth/normal. Do not export a candidate or claim hand pass without Open3D connected-arm review."
        ),
    }
    write_json(output_dir / "b_hand_decoder_skeleton_smoke_summary.json", summary)
    write_markdown(output_dir / "b_hand_decoder_skeleton_smoke_report.md", summary)
    print(json.dumps({"summary": summary["outputs"]["summary_json"], "status": summary["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
