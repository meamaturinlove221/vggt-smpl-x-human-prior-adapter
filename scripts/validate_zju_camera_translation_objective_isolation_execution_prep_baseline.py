import inspect
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

VALIDATION_JSON = OUTPUT_ROOT / "execution_prep_baseline_validation.camera_translation_objective_isolation.20260401.json"
VALIDATION_MD = OUTPUT_ROOT / "execution_prep_baseline_validation.camera_translation_objective_isolation.20260401.md"


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
            def isdir(self, path):
                return False

            def isfile(self, path):
                return False

            def open(self, *args, **kwargs):
                raise FileNotFoundError

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


def make_batch(batch_size=2, sequence_len=1, height=16, width=16):
    extrinsics = torch.zeros(batch_size, sequence_len, 3, 4, dtype=torch.float32)
    extrinsics[..., :3, :3] = torch.eye(3, dtype=torch.float32)
    intrinsics = torch.zeros(batch_size, sequence_len, 3, 3, dtype=torch.float32)
    intrinsics[..., 0, 0] = 100.0
    intrinsics[..., 1, 1] = 120.0
    intrinsics[..., 0, 2] = width / 2
    intrinsics[..., 1, 2] = height / 2
    intrinsics[..., 2, 2] = 1.0
    images = torch.zeros(batch_size, sequence_len, 3, height, width, dtype=torch.float32)
    point_masks = torch.ones(batch_size, sequence_len, height, width, dtype=torch.bool)
    return {
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "images": images,
        "point_masks": point_masks,
    }


def make_predicted_pose(gt_pose, *, translation_offsets, rotation_offsets, focal_offsets):
    pred_pose = gt_pose.clone()
    for sample_idx, offset in enumerate(translation_offsets):
        pred_pose[sample_idx, :, 0] += float(offset)
    for sample_idx, offset in enumerate(rotation_offsets):
        pred_pose[sample_idx, :, 3] += float(offset)
    for sample_idx, offset in enumerate(focal_offsets):
        pred_pose[sample_idx, :, 7] += float(offset)
    return pred_pose


def assert_close(a, b, *, message, atol=1e-6, rtol=1e-6):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(message)


def validate_signature_and_return_contract(loss_module, gt_pose):
    sig = inspect.signature(loss_module.compute_camera_loss)
    if "loss_t_isolation_scale" not in sig.parameters:
        raise AssertionError("loss_t_isolation_scale missing from compute_camera_loss signature")
    if sig.parameters["loss_t_isolation_scale"].default != 1.0:
        raise AssertionError("loss_t_isolation_scale default is not identity")
    batch = make_batch(batch_size=2)
    pred_pose = make_predicted_pose(gt_pose, translation_offsets=[0.9, 0.3], rotation_offsets=[0.4, 0.2], focal_offsets=[0.2, 0.1])
    result = loss_module.compute_camera_loss({"pose_enc_list": [pred_pose]}, batch, weight_trans=1.0, weight_rot=1.0, weight_focal=0.5)
    required_keys = {"loss_camera", "loss_T", "loss_R", "loss_FL"}
    if set(result.keys()) != required_keys:
        raise AssertionError(f"unexpected return keys: {sorted(result.keys())}")
    return {
        "name": "signature_and_return_contract",
        "status": "pass",
        "details": {
            "loss_t_isolation_scale_default": sig.parameters["loss_t_isolation_scale"].default,
            "return_keys": sorted(result.keys()),
        },
    }


def validate_identity_formula(loss_module, gt_pose):
    batch = make_batch(batch_size=2)
    pred_pose = make_predicted_pose(gt_pose, translation_offsets=[0.9, 0.3], rotation_offsets=[0.4, 0.2], focal_offsets=[0.2, 0.1])
    result = loss_module.compute_camera_loss({"pose_enc_list": [pred_pose]}, batch, weight_trans=1.3, weight_rot=0.7, weight_focal=0.5)
    expected = result["loss_T"] * 1.3 + result["loss_R"] * 0.7 + result["loss_FL"] * 0.5
    assert_close(result["loss_camera"], expected, message="default loss_camera formula changed")
    return {
        "name": "identity_formula",
        "status": "pass",
        "details": {
            "loss_camera": float(result["loss_camera"]),
            "expected_formula_value": float(expected),
        },
    }


def validate_scale_only_affects_total_camera(loss_module, gt_pose):
    batch = make_batch(batch_size=2)
    pred_pose = make_predicted_pose(gt_pose, translation_offsets=[0.7, 0.4], rotation_offsets=[0.5, 0.1], focal_offsets=[0.3, 0.05])
    base = loss_module.compute_camera_loss(
        {"pose_enc_list": [pred_pose]},
        batch,
        weight_trans=1.0,
        weight_rot=1.0,
        weight_focal=0.5,
        loss_t_isolation_scale=1.0,
    )
    scaled = loss_module.compute_camera_loss(
        {"pose_enc_list": [pred_pose]},
        batch,
        weight_trans=1.0,
        weight_rot=1.0,
        weight_focal=0.5,
        loss_t_isolation_scale=0.0,
    )
    assert_close(base["loss_T"], scaled["loss_T"], message="loss_T readout changed under isolation scale")
    assert_close(base["loss_R"], scaled["loss_R"], message="loss_R changed under isolation scale")
    assert_close(base["loss_FL"], scaled["loss_FL"], message="loss_FL changed under isolation scale")
    expected_scaled = base["loss_R"] + base["loss_FL"] * 0.5
    assert_close(scaled["loss_camera"], expected_scaled, message="scaled loss_camera formula mismatch")
    return {
        "name": "scale_only_affects_loss_camera_total",
        "status": "pass",
        "details": {
            "base_loss_camera": float(base["loss_camera"]),
            "scaled_loss_camera": float(scaled["loss_camera"]),
            "returned_loss_T": float(base["loss_T"]),
        },
    }


def validate_multistage_manual_accumulation(loss_module, gt_pose):
    batch = make_batch(batch_size=2)
    stage1 = make_predicted_pose(gt_pose, translation_offsets=[0.3, 0.1], rotation_offsets=[0.2, 0.05], focal_offsets=[0.4, 0.1])
    stage2 = make_predicted_pose(gt_pose, translation_offsets=[0.9, 0.3], rotation_offsets=[0.5, 0.2], focal_offsets=[0.25, 0.15])
    gamma = 0.6
    result = loss_module.compute_camera_loss(
        {"pose_enc_list": [stage1, stage2]},
        batch,
        gamma=gamma,
        weight_trans=1.2,
        weight_rot=0.8,
        weight_focal=0.5,
        loss_t_isolation_scale=0.9,
    )
    valid_frame_mask = batch["point_masks"].sum(dim=[-1, -2]) > 100
    gt_pose_valid = gt_pose[valid_frame_mask].clone()
    total_t = torch.tensor(0.0)
    total_r = torch.tensor(0.0)
    total_fl = torch.tensor(0.0)
    for idx, pred_stage in enumerate([stage1, stage2]):
        stage_weight = gamma ** (2 - idx - 1)
        loss_t, loss_r, loss_fl = loss_module.camera_loss_single(pred_stage[valid_frame_mask].clone(), gt_pose_valid, loss_type="l1")
        total_t = total_t + loss_t * stage_weight
        total_r = total_r + loss_r * stage_weight
        total_fl = total_fl + loss_fl * stage_weight
    avg_t = total_t / 2
    avg_r = total_r / 2
    avg_fl = total_fl / 2
    expected_camera = avg_t * 1.2 * 0.9 + avg_r * 0.8 + avg_fl * 0.5
    assert_close(result["loss_T"], avg_t, message="multistage avg T mismatch")
    assert_close(result["loss_R"], avg_r, message="multistage avg R mismatch")
    assert_close(result["loss_FL"], avg_fl, message="multistage avg FL mismatch")
    assert_close(result["loss_camera"], expected_camera, message="multistage camera total mismatch")
    return {
        "name": "multistage_manual_accumulation",
        "status": "pass",
        "details": {
            "avg_loss_T": float(avg_t),
            "avg_loss_R": float(avg_r),
            "avg_loss_FL": float(avg_fl),
            "loss_camera": float(result["loss_camera"]),
        },
    }


def render_md(payload: dict) -> str:
    lines = [
        "# Execution-Prep Baseline Validation: Camera Translation Objective Isolation (2026-04-01)",
        "",
        f"- overall_status: `{payload['overall_status']}`",
        f"- target_file: `{payload['target_file']}`",
        f"- ready_for_execution: `{payload['ready_for_execution']}`",
        "",
        "## Validation Cases",
        "",
    ]
    for case in payload["validation_cases"]:
        lines.extend([f"### {case['name']}", "", f"- status: `{case['status']}`", f"- details: `{json.dumps(case['details'], ensure_ascii=False)}`", ""])
    lines.extend(["## Summary", "", f"- {payload['summary']}", ""])
    return "\n".join(lines)


def main() -> int:
    checked_at = now_iso()
    py_compile.compile(str(LOSS_PY), doraise=True)
    loss_module, stubbed_iopath = load_loss_module()
    from vggt.utils.pose_enc import extri_intri_to_pose_encoding

    batch = make_batch(batch_size=2)
    gt_pose = extri_intri_to_pose_encoding(
        batch["extrinsics"],
        batch["intrinsics"],
        batch["images"].shape[-2:],
        pose_encoding_type="absT_quaR_FoV",
    )

    validation_cases = [
        validate_signature_and_return_contract(loss_module, gt_pose),
        validate_identity_formula(loss_module, gt_pose),
        validate_scale_only_affects_total_camera(loss_module, gt_pose),
        validate_multistage_manual_accumulation(loss_module, gt_pose),
    ]

    payload = {
        "checked_at": checked_at,
        "artifact_kind": "execution_prep_baseline_validation",
        "family": "camera_translation_objective_isolation",
        "target_file": "training/loss.py",
        "ready_for_execution": False,
        "overall_status": "PASS_STRONGER_LOCAL_VALIDATION",
        "validation_cases": validation_cases,
        "summary": (
            "The reviewed loss.py-only translation hook passed stronger local validation: the default formula is preserved, "
            "the isolation scale only changes total loss_camera aggregation, and the multistage accumulation matches manual expectations "
            "without training execution."
        ),
        "environment_note": {
            "used_iopath_stub_for_import": bool(stubbed_iopath),
            "training_executed": False,
            "cloud_used": False,
        },
        "next_requirement": "decide_validated_loss_py_translation_hook_can_enter_execution_ready_packaging",
    }

    write_json(VALIDATION_JSON, payload)
    write_text(VALIDATION_MD, render_md(payload))
    print(json.dumps({"execution_prep_baseline_validation": repo_rel(VALIDATION_JSON)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
