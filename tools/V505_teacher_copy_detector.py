from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER = ROOT / "output" / "V5040000000000000000000_v50r2_teacher_bank" / "v50r2_teacher_bank.npz"
SMOKE_ROOT = ROOT / "output" / "V5050000000000000000000_teacher_student_firewall_smoke"
SMOKE_JSON = ROOT / "reports" / "V5050000000000000000000_firewall_smoke.json"

FORBIDDEN_METADATA_TOKENS = [
    "v50r2_final",
    "teacher_as_final",
    "teacher_points_inference",
    "kinect_inference",
    "direct_teacher_rgb",
    "copy_teacher",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def exact_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


def near_equal(a: np.ndarray, b: np.ndarray, atol: float = 1e-8) -> bool:
    return a.shape == b.shape and np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number) and bool(np.allclose(a, b, atol=atol, rtol=0.0))


def metadata_flags(candidate: dict[str, np.ndarray]) -> list[str]:
    flags: list[str] = []
    for key, arr in candidate.items():
        if arr.shape == () and arr.dtype.kind in {"U", "S"}:
            value = str(arr.item()).lower()
            for token in FORBIDDEN_METADATA_TOKENS:
                if token in value:
                    flags.append(f"metadata_forbidden_token:{key}:{token}")
        if key.lower() in {"teacher_only", "final_inference_allowed", "copy_forbidden"}:
            try:
                val = bool(arr.item())
            except Exception:
                continue
            if key.lower() == "teacher_only" and val:
                flags.append("candidate_marks_teacher_only_true")
            if key.lower() == "final_inference_allowed" and not val:
                flags.append("candidate_marks_final_inference_allowed_false")
    return flags


def detect(teacher_path: Path, candidate_path: Path) -> dict[str, Any]:
    teacher = load_npz(teacher_path)
    candidate = load_npz(candidate_path)
    hits: list[dict[str, Any]] = []

    teacher_point_keys = ["points", "refined_points", "hand_points"]
    teacher_rgb_keys = ["rgb"]
    teacher_mask_keys = ["full_body_mask", "head_mask", "face_mask", "hand_visibility"]

    for ckey, carr in candidate.items():
        for tkey in teacher_point_keys:
            if tkey in teacher and (exact_equal(carr, teacher[tkey]) or near_equal(carr, teacher[tkey])):
                hits.append({"candidate_key": ckey, "teacher_key": tkey, "type": "teacher_point_exact_or_near_copy", "shape": list(carr.shape)})
        for tkey in teacher_rgb_keys:
            if tkey in teacher and exact_equal(carr, teacher[tkey]):
                hits.append({"candidate_key": ckey, "teacher_key": tkey, "type": "teacher_rgb_exact_copy", "shape": list(carr.shape)})
        for tkey in teacher_mask_keys:
            if tkey in teacher and exact_equal(carr, teacher[tkey]):
                hits.append({"candidate_key": ckey, "teacher_key": tkey, "type": "teacher_mask_exact_copy", "shape": list(carr.shape)})

    flags = metadata_flags(candidate)
    leak_detected = bool(hits or flags)
    return {
        "teacher_bank": str(teacher_path),
        "candidate_npz": str(candidate_path),
        "leak_detected": leak_detected,
        "hit_count": len(hits),
        "hits": hits,
        "metadata_flags": flags,
        "policy": {
            "teacher_points_in_final_inference_allowed": False,
            "teacher_rgb_in_final_inference_allowed": False,
            "teacher_mask_as_student_ownership_allowed": False,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_smoke(teacher_path: Path, output_json: Path) -> dict[str, Any]:
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    teacher = load_npz(teacher_path)
    leak_candidate = SMOKE_ROOT / "known_teacher_copy_candidate.npz"
    safe_candidate = SMOKE_ROOT / "safe_noncopy_candidate.npz"

    rng = np.random.default_rng(505)
    safe_points = teacher["points"].astype(np.float32).copy()
    safe_points = safe_points[:, ::32, ::32, :] + rng.normal(0.0, 0.003, size=safe_points[:, ::32, ::32, :].shape).astype(np.float32)
    safe_rgb = teacher["rgb"][:, ::32, ::32, :].copy()

    np.savez_compressed(
        leak_candidate,
        predicted_points=teacher["points"],
        predicted_rgb=teacher["rgb"],
        source=np.array("teacher_points_inference_copy_teacher"),
    )
    np.savez_compressed(
        safe_candidate,
        predicted_points=safe_points,
        predicted_rgb=safe_rgb,
        source=np.array("model_owned_synthetic_noncopy_smoke"),
    )

    leak_result = detect(teacher_path, leak_candidate)
    safe_result = detect(teacher_path, safe_candidate)
    passed = leak_result["leak_detected"] and not safe_result["leak_detected"]
    payload = {
        "task": "V505_teacher_student_firewall_smoke",
        "status": "V505_FIREWALL_SMOKE_PASS_CONTINUE_NOT_PROMOTED" if passed else "V505_FIREWALL_SMOKE_FAIL_NOT_PROMOTED",
        "created_at": now(),
        "teacher_bank": str(teacher_path),
        "known_leak_candidate": str(leak_candidate),
        "safe_noncopy_candidate": str(safe_candidate),
        "known_leak_result": leak_result,
        "safe_noncopy_result": safe_result,
        "gates": {
            "known_teacher_copy_detected": leak_result["leak_detected"],
            "safe_noncopy_not_flagged": not safe_result["leak_detected"],
            "final_inference_allowed": False,
            "not_promoted": True,
        },
        "decision": "Firewall smoke proves direct teacher point/RGB copy detection only. Continue to V506; do not treat this as mentor visual success.",
    }
    write_json(output_json, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-bank", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--candidate-npz", type=Path)
    parser.add_argument("--output-json", type=Path, default=SMOKE_JSON)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        payload = run_smoke(args.teacher_bank, args.output_json)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if payload["status"].endswith("PASS_CONTINUE_NOT_PROMOTED") else 2

    if not args.candidate_npz:
        raise SystemExit("--candidate-npz is required unless --smoke is used")
    payload = {
        "task": "V505_teacher_copy_detector",
        "created_at": now(),
        **detect(args.teacher_bank, args.candidate_npz),
    }
    write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if not payload["leak_detected"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
