import json
import subprocess
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"

RESULT_JSON = OUTPUT_ROOT / "cloud_validation_result.source_policy_hybrid_ring_regularization.20260403.json"
RESULT_MD = OUTPUT_ROOT / "cloud_validation_result.source_policy_hybrid_ring_regularization.20260403.md"
RUNTIME_JSON = OUTPUT_ROOT / "cloud_runtime_state.20260403.json"
RUNTIME_MD = OUTPUT_ROOT / "cloud_runtime_state.20260403.md"
DECISION_JSON = OUTPUT_ROOT / "cloud_promotion_decision.source_policy_hybrid_ring_regularization.20260403.json"
CLOUD_PAIR_JSON = OUTPUT_ROOT / "zju_next_cloud_pair.source_policy_hybrid_ring_regularization.20260403.json"

APP_ID = "ap-vgOk2fs2BsVwGb9f28VVt6"
APP_DESCRIPTION = "vggt-zju-geometry-minimal-finetune"
STARTED_AT = "2026-04-03 03:04:17 +08:00"
STOPPED_AT = "2026-04-03 03:12:28 +08:00"
OUTPUT_SUBDIR = "zju_source_policy_research_loop/cloud_runs/20260403_promoted_hybrid_ring_latest_lead_v1"

VAL_METRICS = {
    "loss_camera": 0.0046,
    "loss_T": 0.0001,
    "loss_conf_depth": -0.2167,
    "loss_reg_depth": 0.0171,
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def list_modal_apps() -> list[dict]:
    result = run_cmd(["modal", "app", "list", "--json"])
    if result.returncode != 0:
        raise RuntimeError(f"modal app list failed: {result.stderr.strip()}")
    return json.loads(result.stdout.strip() or "[]")


def volume_ls(remote_path: str) -> list[str]:
    result = run_cmd(["modal", "volume", "ls", "vggt-out", remote_path])
    if result.returncode != 0:
        raise RuntimeError(f"modal volume ls failed for {remote_path}: {result.stderr.strip()}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def render_md(title: str, payload: dict) -> str:
    return f"# {title}\n\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"


def main() -> int:
    decision = load_json(DECISION_JSON)
    cloud_pair = load_json(CLOUD_PAIR_JSON)
    apps = list_modal_apps()
    active_apps = [row for row in apps if str(row.get("State", "")).lower() != "stopped"]
    matching_app = next((row for row in apps if str(row.get("App ID", "")).strip() == APP_ID), {})
    ckpt_entries = volume_ls(f"/{OUTPUT_SUBDIR}/ckpts")

    result = {
        "checked_at": now_iso(),
        "artifact_kind": "cloud_validation_result",
        "family": "source_policy_hybrid_ring_regularization",
        "shape": "stablelead_nearest_plus_uniform_tail",
        "mode": "promoted_lead_cloud_validation",
        "modal_app_id": APP_ID,
        "modal_app_description": APP_DESCRIPTION,
        "started_at": STARTED_AT,
        "stopped_at": STOPPED_AT,
        "output_subdir": OUTPUT_SUBDIR,
        "remote_output_root": f"vggt-out:/{OUTPUT_SUBDIR}",
        "val": dict(VAL_METRICS),
        "checkpoint_written": any("checkpoint.pt" in entry for entry in ckpt_entries),
        "checkpoint_files": ckpt_entries,
        "active_modal_app_count_after_finish": len(active_apps),
        "modal_app_list_literal_empty_after_finish": len(apps) == 0,
        "stopped_app_record_present_after_finish": bool(matching_app),
        "cloud_decision_artifact": str(DECISION_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        "cloud_pair_artifact": str(CLOUD_PAIR_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        "launch_contract": {
            "cloud_gate": bool(decision.get("cloud_gate")),
            "launch_cloud_now": bool(decision.get("launch_cloud_now")),
            "pair_cloud_gate": bool(cloud_pair.get("cloud_gate")),
            "pair_launch_cloud_now": bool(cloud_pair.get("launch_cloud_now")),
        },
    }

    runtime = {
        "checked_at": now_iso(),
        "artifact_kind": "cloud_runtime_state",
        "modal_app_id": APP_ID,
        "family": "source_policy_hybrid_ring_regularization",
        "mode": "promoted_lead_cloud_validation",
        "output_subdir": OUTPUT_SUBDIR,
        "active_modal_app_count": len(active_apps),
        "stopped_app_record_present": bool(matching_app),
        "modal_app_state": str(matching_app.get("State", "")),
        "modal_app_created_at": str(matching_app.get("Created at", "")),
        "modal_app_stopped_at": str(matching_app.get("Stopped at", "")),
        "cleanup_ok": len(active_apps) == 0,
        "no_redundant_cloud_process": len(active_apps) == 0,
        "modal_app_list_literal_empty_after_finish": len(apps) == 0,
    }

    write_json(RESULT_JSON, result)
    write_text(RESULT_MD, render_md("Cloud Validation Result", result))
    write_json(RUNTIME_JSON, runtime)
    write_text(RUNTIME_MD, render_md("Cloud Runtime State", runtime))
    print(
        json.dumps(
            {
                "cloud_validation_result": str(RESULT_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
                "cloud_runtime_state": str(RUNTIME_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
