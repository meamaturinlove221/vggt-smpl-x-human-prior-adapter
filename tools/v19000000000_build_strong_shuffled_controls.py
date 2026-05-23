from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"
OUTPUT = AUX / "output"
V360 = OUTPUT / "V3600000000_fullview_dataset_v2"
OUT = OUTPUT / "V19000000000_strong_controls"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def strong_shuffle_semantic(sem: np.ndarray, seed: int = 19000000000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = sem.copy()
    # Independently shuffle coherent blocks so face/bary/skinning/joint/frame no
    # longer describe the same surface point.
    blocks = [
        slice(21, 27),   # face, bary, nearest vertex proxy
        slice(27, 43),   # skinning weights and ids
        slice(43, 59),   # joint distances and ids
        slice(59, 62),   # bone relative
        slice(62, 71),   # tangent/local frame
        slice(73, 81),   # body/surface/curvature v2
    ]
    flat_n = sem.shape[0] * sem.shape[2] * sem.shape[3]
    for block in blocks:
        arr = np.moveaxis(out[:, block, :, :], 1, -1).reshape(flat_n, block.stop - block.start)
        arr = arr[rng.permutation(flat_n)]
        out[:, block, :, :] = np.moveaxis(arr.reshape(sem.shape[0], sem.shape[2], sem.shape[3], block.stop - block.start), -1, 1)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(V360 / "true_full.npz", allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    payload["semantic"] = strong_shuffle_semantic(payload["semantic"].astype(np.float32))
    out_path = OUT / "strong_shuffled_semantic.npz"
    np.savez_compressed(out_path, **payload)
    manifest = {
        "created_utc": now(),
        "output": str(out_path),
        "support_observation_source": str(V360 / "true_full.npz"),
        "support_observation_byte_identical_to_true": True,
        "semantic_control": "strong block-wise independent shuffle",
        "purpose": "valid adversarial semantic coherence control",
    }
    write_json(REPORTS / "V19000000000_strong_shuffled_control_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
