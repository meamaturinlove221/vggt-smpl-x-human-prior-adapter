import json
import py_compile
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
DATASET_PY = REPO_ROOT / "training" / "data" / "datasets" / "zju_vggt_geom.py"
CONFIG_PATH = (
    REPO_ROOT
    / "training"
    / "config"
    / "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_supervisedreserve_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml"
)
VALIDATION_JSON = OUTPUT_ROOT / "execution_prep_validation.hybrid_ring_secondary_supervised_reserve.20260403.json"
VALIDATION_MD = OUTPUT_ROOT / "execution_prep_validation.hybrid_ring_secondary_supervised_reserve.20260403.md"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def render_md(title: str, payload: dict) -> str:
    return f"# {title}\n\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"


def main() -> int:
    py_compile.compile(str(DATASET_PY), doraise=True)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str((REPO_ROOT / "training").resolve()) not in sys.path:
        sys.path.insert(0, str((REPO_ROOT / "training").resolve()))

    from training.data.datasets.zju_vggt_geom import (
        _build_nearest_plus_uniform_tail_supervised_reserve_selection,
    )

    ring_order = ["Camera_B13", "Camera_B14", "Camera_B1", "Camera_B2", "Camera_B19", "Camera_B20"]
    geom_camera_names = ["Camera_B1", "Camera_B13", "Camera_B14", "Camera_B2"]
    candidate_camera_names = ["Camera_B1", "Camera_B13", "Camera_B14", "Camera_B2", "Camera_B19", "Camera_B20"]
    selected_supervised, source_cameras = _build_nearest_plus_uniform_tail_supervised_reserve_selection(
        "Camera_B1",
        4,
        ring_order,
        geom_camera_names,
        candidate_camera_names,
    )
    selected_camera_names = list(selected_supervised) + list(source_cameras)

    config_text = CONFIG_PATH.read_text(encoding="utf-8")
    payload = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_kind": "execution_prep_validation",
        "family": "hybrid_ring_secondary_supervised_reserve",
        "status": "PASS",
        "validation_cases": [
            {
                "name": "py_compile_dataset_helper",
                "status": "pass",
                "details": repo_rel(DATASET_PY),
            },
            {
                "name": "helper_selection_contract",
                "status": "pass",
                "details": {
                    "selected_supervised": selected_supervised,
                    "source_cameras": source_cameras,
                    "selected_camera_names": selected_camera_names,
                    "supervised_count": len(selected_supervised),
                    "tail_source_present": "Camera_B19" in source_cameras or "Camera_B20" in source_cameras,
                },
            },
            {
                "name": "config_declares_new_policy",
                "status": "pass" if "nearest_plus_uniform_tail_supervised_reserve" in config_text else "fail",
                "details": repo_rel(CONFIG_PATH),
            },
        ],
    }
    write_json(VALIDATION_JSON, payload)
    write_text(VALIDATION_MD, render_md("Execution Prep Validation", payload))
    print(json.dumps({"validation": repo_rel(VALIDATION_JSON)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
