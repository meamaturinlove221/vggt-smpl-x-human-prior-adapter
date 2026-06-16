from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path(r"D:\vggt\vggt-main")
LOCAL = MAIN_ROOT / "local_report_auxiliary" / "V600_quality_rebuild"
sys.path.insert(0, str(LOCAL / "tools"))

import v232000_v260000_sharper_defect_patch_route as v232  # noqa: E402


def load_pred(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        depth = z.get("depth", z.get("depths"))
        if depth.ndim == 4:
            depth = depth[..., 0]
        return {
            "points": z["world_points"][:6].astype(np.float32),
            "depth": depth[:6].astype(np.float32),
            "confidence": z.get("world_points_conf", np.ones(depth.shape, dtype=np.float32))[:6].astype(np.float32),
            "normal": z.get("normal", np.zeros((*depth.shape, 3), dtype=np.float32))[:6].astype(np.float32),
        }


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit("usage: python v950000_evaluate_geometry_candidate.py BASELINE_NPZ CANDIDATE_NPZ NAME")
    base = load_pred(Path(sys.argv[1]))
    cand = load_pred(Path(sys.argv[2]))
    _, summary, delta = v232.v629_delta(base, cand, sys.argv[3])
    print(json.dumps({"summary": summary, "delta": delta}, indent=2))


if __name__ == "__main__":
    main()
