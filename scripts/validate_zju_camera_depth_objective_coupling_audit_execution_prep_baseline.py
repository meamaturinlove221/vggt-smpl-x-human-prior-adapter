import json
import py_compile
import sys
import types
from datetime import datetime
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
LOSS_PY = REPO_ROOT / "training" / "loss.py"
BASE_CONFIG = (
    REPO_ROOT
    / "training"
    / "config"
    / "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_minimal.yaml"
)
CANDIDATE_CONFIG = (
    REPO_ROOT
    / "training"
    / "config"
    / "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_cameraweight095_depthhold100_minimal.yaml"
)

VALIDATION_JSON = OUTPUT_ROOT / "execution_prep_baseline_validation.camera_depth_objective_coupling_audit.20260403.json"
VALIDATION_MD = OUTPUT_ROOT / "execution_prep_baseline_validation.camera_depth_objective_coupling_audit.20260403.md"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def ensure_iopath_stub() -> bool:
    try:
        import iopath.common.file_io  # noqa: F401
        return False
    except ModuleNotFoundError:
        iopath = types.ModuleType("iopath")
        common = types.ModuleType("iopath.common")
        file_io = types.ModuleType("iopath.common.file_io")

        class _DummyPathMgr:
            def isdir(self, path): return False
            def isfile(self, path): return False
            def open(self, *args, **kwargs): raise FileNotFoundError

        file_io.g_pathmgr = _DummyPathMgr()
        common.file_io = file_io
        sys.modules["iopath"] = iopath
        sys.modules["iopath.common"] = common
        sys.modules["iopath.common.file_io"] = file_io
        return True


def load_loss_module():
    stubbed = ensure_iopath_stub()
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    training_root = str(REPO_ROOT / "training")
    if training_root not in sys.path:
        sys.path.insert(0, training_root)
    import loss  # noqa: F401
    return loss, stubbed


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].rstrip()


def parse_defaults(path: Path) -> list[str]:
    defaults: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    in_defaults = False
    defaults_indent = 0
    for raw_line in lines:
        line = _strip_comment(raw_line)
        if not line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = line.strip()
        if not in_defaults:
            if stripped == "defaults:":
                in_defaults = True
                defaults_indent = indent
            continue
        if indent <= defaults_indent:
            break
        if stripped.startswith("- "):
            value = stripped[2:].strip()
            if value:
                defaults.append(value)
    return defaults


def parse_loss_weight_overrides(path: Path) -> dict[str, float]:
    overrides: dict[str, float] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    in_loss = False
    loss_indent = 0
    current_section = ""
    section_indent = 0
    for raw_line in lines:
        line = _strip_comment(raw_line)
        if not line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = line.strip()
        if not in_loss:
            if stripped == "loss:":
                in_loss = True
                loss_indent = indent
            continue
        if indent <= loss_indent:
            in_loss = False
            current_section = ""
            continue
        if stripped in {"camera:", "depth:"}:
            current_section = stripped[:-1]
            section_indent = indent
            continue
        if current_section and indent <= section_indent:
            current_section = ""
        if current_section and stripped.startswith("weight:"):
            overrides[current_section] = float(stripped.split(":", 1)[1].strip())
    return overrides


def load_effective_loss_weights(path: Path, _seen: set[Path] | None = None) -> dict[str, float]:
    seen = set() if _seen is None else _seen
    path = path.resolve()
    if path in seen:
        return {}
    seen.add(path)
    weights: dict[str, float] = {}
    for item in parse_defaults(path):
        if item == "_self_" or ":" in item:
            continue
        parent_path = path.parent / f"{item}.yaml"
        if parent_path.exists():
            weights.update(load_effective_loss_weights(parent_path, seen))
    weights.update(parse_loss_weight_overrides(path))
    return weights


def assert_close(actual: torch.Tensor, expected: torch.Tensor, *, message: str, atol: float = 1e-6, rtol: float = 1e-6) -> None:
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(message)


def validate_py_compile_and_import(loss_module, used_iopath_stub: bool) -> dict:
    py_compile.compile(str(LOSS_PY), doraise=True)
    return {
        "name": "py_compile_and_import_smoke",
        "status": "pass",
        "details": {
            "loss_py": repo_rel(LOSS_PY),
            "used_iopath_stub_for_import": used_iopath_stub,
            "has_multitask_loss": hasattr(loss_module, "MultitaskLoss"),
            "has_resolve_two_stage_scalar": hasattr(loss_module, "resolve_two_stage_scalar"),
        },
    }


def validate_config_compose_weights() -> dict:
    base_cfg = load_effective_loss_weights(BASE_CONFIG)
    candidate_cfg = load_effective_loss_weights(CANDIDATE_CONFIG)
    candidate_camera_weight = float(candidate_cfg["camera"])
    candidate_depth_weight = float(candidate_cfg["depth"])
    if float(base_cfg["camera"]) != 1.0:
        raise AssertionError("baseline camera weight drifted from 1.0")
    if float(base_cfg["depth"]) != 1.0:
        raise AssertionError("baseline depth weight drifted from 1.0")
    if candidate_camera_weight != 0.95:
        raise AssertionError("candidate camera weight is not 0.95")
    if candidate_depth_weight != 1.0:
        raise AssertionError("candidate depth weight is not 1.0")
    return {
        "name": "config_compose_camera_relief_depth_hold",
        "status": "pass",
        "details": {
            "candidate_config": repo_rel(CANDIDATE_CONFIG),
            "base_camera_weight": float(base_cfg["camera"]),
            "candidate_camera_weight": candidate_camera_weight,
            "base_depth_weight": float(base_cfg["depth"]),
            "candidate_depth_weight": candidate_depth_weight,
        },
    }


def validate_multitask_objective_weighting(loss_module) -> dict:
    candidate_weights = load_effective_loss_weights(CANDIDATE_CONFIG)
    camera_cfg = {"weight": float(candidate_weights["camera"]), "loss_type": "l1"}
    depth_cfg = {"weight": float(candidate_weights["depth"]), "gradient_loss_fn": "grad"}
    model = loss_module.MultitaskLoss(
        camera=camera_cfg,
        depth=depth_cfg,
        point=None,
        track=None,
        unproject_geometry=None,
    )

    original_camera_loss = loss_module.compute_camera_loss
    original_depth_loss = loss_module.compute_depth_loss
    try:
        loss_module.compute_camera_loss = lambda predictions, batch, **kwargs: {
            "loss_camera": torch.tensor(1.2, dtype=torch.float32),
            "loss_T": torch.tensor(0.3, dtype=torch.float32),
            "loss_R": torch.tensor(0.4, dtype=torch.float32),
            "loss_FL": torch.tensor(0.5, dtype=torch.float32),
        }
        loss_module.compute_depth_loss = lambda predictions, batch, **kwargs: {
            "loss_conf_depth": torch.tensor(0.7, dtype=torch.float32),
            "loss_reg_depth": torch.tensor(0.8, dtype=torch.float32),
            "loss_grad_depth": torch.tensor(0.9, dtype=torch.float32),
        }

        result = model({"pose_enc_list": [torch.zeros(1)], "depth": torch.zeros(1)}, {})
    finally:
        loss_module.compute_camera_loss = original_camera_loss
        loss_module.compute_depth_loss = original_depth_loss

    expected_objective = torch.tensor(1.2 * 0.95 + 0.7 + 0.8 + 0.9, dtype=torch.float32)
    assert_close(result["objective"], expected_objective, message="objective weight aggregation drifted")
    assert_close(result["loss_camera"], torch.tensor(1.2), message="loss_camera readout changed unexpectedly")
    assert_close(result["loss_conf_depth"], torch.tensor(0.7), message="loss_conf_depth readout changed unexpectedly")
    assert_close(result["loss_reg_depth"], torch.tensor(0.8), message="loss_reg_depth readout changed unexpectedly")
    assert_close(result["loss_grad_depth"], torch.tensor(0.9), message="loss_grad_depth readout changed unexpectedly")
    return {
        "name": "multitask_objective_weighting_contract",
        "status": "pass",
        "details": {
            "camera_weight": 0.95,
            "depth_weight": 1.0,
            "objective": float(result["objective"]),
            "expected_objective": float(expected_objective),
        },
    }


def render_md(payload: dict) -> str:
    lines = [
        "# Execution-Prep Baseline Validation: Camera-Depth Objective Coupling Audit (2026-04-03)",
        "",
        f"- overall_status: `{payload['overall_status']}`",
        f"- family: `{payload['family']}`",
        f"- ready_for_execution: `{payload['ready_for_execution']}`",
        "",
        "## Validation Cases",
        "",
    ]
    for case in payload["validation_cases"]:
        lines.extend(
            [
                f"### {case['name']}",
                "",
                f"- status: `{case['status']}`",
                f"- details: `{json.dumps(case['details'], ensure_ascii=False)}`",
                "",
            ]
        )
    lines.extend(["## Summary", "", f"- {payload['summary']}", ""])
    return "\n".join(lines)


def main() -> int:
    loss_module, used_iopath_stub = load_loss_module()
    validation_cases = [
        validate_py_compile_and_import(loss_module, used_iopath_stub),
        validate_config_compose_weights(),
        validate_multitask_objective_weighting(loss_module),
    ]

    payload = {
        "checked_at": now_iso(),
        "artifact_kind": "execution_prep_baseline_validation",
        "family": "camera_depth_objective_coupling_audit",
        "target_files": [
            repo_rel(LOSS_PY),
            repo_rel(CANDIDATE_CONFIG),
        ],
        "ready_for_execution": False,
        "overall_status": "PASS_STRONGER_LOCAL_VALIDATION",
        "validation_cases": validation_cases,
        "summary": (
            "The higher-level camera-depth coupling audit passed stronger local validation: the candidate config composes "
            "cleanly, camera weight is reduced from 1.0 to 0.95 while depth stays at 1.0, and the current MultitaskLoss "
            "objective aggregation already honors that camera-relief/depth-hold contract without widening the code surface."
        ),
        "environment_note": {
            "used_iopath_stub_for_import": bool(used_iopath_stub),
            "training_executed": False,
            "cloud_used": False,
        },
        "next_requirement": "decide_validated_camera_depth_objective_coupling_audit_can_enter_execution_ready_packaging",
    }

    write_json(VALIDATION_JSON, payload)
    write_text(VALIDATION_MD, render_md(payload))
    print(json.dumps({"execution_prep_baseline_validation": repo_rel(VALIDATION_JSON)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
