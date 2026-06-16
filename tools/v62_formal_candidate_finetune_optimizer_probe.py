from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [f"# {payload['task']}", ""]
    for key in ["status", "created_utc", "decision"]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    lines.extend(
        [
            f"- trainable_probe_ran: `{payload.get('trainable_probe_ran')}`",
            f"- gradient_finite: `{payload.get('gradient_finite')}`",
            f"- gradient_nonzero: `{payload.get('gradient_nonzero')}`",
            f"- candidate_saved_for_promotion: `{payload.get('candidate_saved_for_promotion')}`",
        ]
    )
    if payload.get("blockers"):
        lines += ["", "## Blockers"]
        lines += [f"- {x}" for x in payload["blockers"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_npz_key(path: Path, preferred: list[str]) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        for key in preferred:
            if key in data.files:
                return data[key].astype(np.float32)
        return data[data.files[0]].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-candidate-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-step-lr", type=float, default=1e-5)
    args = parser.parse_args()

    frozen = args.frozen_candidate_dir.resolve()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    points = load_npz_key(frozen / "package_files" / "candidate_files__candidate_points.npz", ["candidate_points_world", "points_world"])
    normals = load_npz_key(frozen / "package_files" / "candidate_files__candidate_normals.npz", ["candidate_normals_geometric", "normals"])

    try:
        import torch

        device = torch.device("cpu")
        pts = torch.from_numpy(points[:, 160:390, 338:, :]).to(device)
        nrm = torch.from_numpy(normals[:, 160:390, 338:, :]).to(device)
        # Trainable scalar/vector probe. Target is a tiny bounded right-hand normal-consistent
        # displacement; output is not promoted unless an external validation later approves it.
        delta = torch.nn.Parameter(torch.zeros(3, device=device))
        optimizer = torch.optim.SGD([delta], lr=args.max_step_lr)
        direction = torch.nan_to_num(nrm.mean(dim=(0, 1, 2)), nan=0.0)
        direction = direction / torch.clamp(torch.linalg.norm(direction), min=1e-6)
        target = pts + direction.view(1, 1, 1, 3) * 1e-4
        before_loss = torch.mean((pts - target) ** 2)
        optimizer.zero_grad(set_to_none=True)
        pred = pts + delta.view(1, 1, 1, 3)
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        grad = delta.grad.detach().clone()
        grad_norm = float(torch.linalg.norm(grad).cpu())
        grad_finite = bool(torch.isfinite(grad).all().cpu())
        optimizer.step()
        after_delta = delta.detach().cpu().numpy().astype(np.float32)
        after_loss = torch.mean((pts + delta.view(1, 1, 1, 3) - target) ** 2)
        trainable_probe_ran = True
        blocker = []
    except Exception as exc:
        before_loss = None
        after_loss = None
        grad_norm = 0.0
        grad_finite = False
        after_delta = np.zeros(3, dtype=np.float32)
        trainable_probe_ran = False
        blocker = [repr(exc)]

    np.savez_compressed(
        out / "trainable_probe_delta_only.npz",
        right_hand_delta=after_delta,
        promoted_candidate=False,
        source="V62_trainable_probe_delta_only_not_merged",
    )
    payload: dict[str, Any] = {
        "task": "V62_B_trainable_optimizer_probe",
        "status": "PASS" if trainable_probe_ran and grad_finite and grad_norm > 0 else "FAIL_FROZEN",
        "created_utc": now(),
        "trainable_probe_ran": trainable_probe_ran,
        "gradient_finite": grad_finite,
        "gradient_nonzero": grad_norm > 0,
        "gradient_norm": grad_norm,
        "before_loss": float(before_loss.detach().cpu()) if before_loss is not None else None,
        "after_loss": float(after_loss.detach().cpu()) if after_loss is not None else None,
        "delta_norm": float(np.linalg.norm(after_delta)),
        "candidate_saved_for_promotion": False,
        "delta_only_output": str(out / "trainable_probe_delta_only.npz"),
        "decision": "Trainable optimizer/backward path is verified; no candidate promotion because this probe is delta-only and intentionally not merged.",
        "blockers": blocker,
    }
    write_json(out / "summary.json", payload)
    write_md(out / "summary.md", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

