from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "vggt-v53-formal-cloud-unlock-smoke"
ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "output" / "surface_research_preflight_local" / "V50_final_promotion_transaction" / "strict_registry_entry_v50.json"
MANIFEST_PATH = ROOT / "output" / "surface_research_preflight_local" / "V50_final_promotion_transaction" / "candidate_package_v50" / "manifest.json"
OUT_DIR = ROOT / "output" / "formal_cloud_smoke" / "V53_candidate_formal_smoke"


app = modal.App(APP_NAME)
image = modal.Image.debian_slim()


@app.function(image=image, timeout=120)
def formal_registry_package_read_smoke(registry_json: str, manifest_json: str) -> dict:
    registry = json.loads(registry_json)
    manifest = json.loads(manifest_json)
    candidate_files = manifest.get("candidate_files", {})
    v42_payload = manifest.get("v42_prior_enabled_payload", {})
    strict_candidate_pass = bool(registry.get("strict_candidate_pass"))
    manifest_unblocked = bool(manifest.get("formal_cloud_unblocked"))
    no_teacher_write = bool(manifest.get("no_teacher_package_written", True))
    return {
        "task": "v53_formal_cloud_unlock_smoke_remote",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "modal_app_name": APP_NAME,
        "formal_cloud_started": True,
        "formal_entrypoint_detects_strict_candidate_passes": 1 if strict_candidate_pass else 0,
        "formal_cloud_reads_candidate_package": bool(candidate_files) and bool(v42_payload),
        "formal_cloud_exits_cleanly": strict_candidate_pass and manifest_unblocked and no_teacher_write,
        "no_teacher_package_write": no_teacher_write,
        "candidate_file_count": len(candidate_files),
        "v42_payload_file_count": len(v42_payload),
        "strict_teacher_passes": 0,
        "status": "DONE_PASS" if strict_candidate_pass and manifest_unblocked and no_teacher_write else "DONE_FAIL_ROUTED",
    }


@app.local_entrypoint()
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry_json = REGISTRY_PATH.read_text(encoding="utf-8")
    manifest_json = MANIFEST_PATH.read_text(encoding="utf-8")
    result = formal_registry_package_read_smoke.remote(registry_json, manifest_json)
    result["local_registry_path"] = str(REGISTRY_PATH)
    result["local_manifest_path"] = str(MANIFEST_PATH)
    result_path = OUT_DIR / "modal_smoke_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = [
        "# V53 Formal Cloud Unlock Smoke",
        "",
        f"- status: `{result['status']}`",
        f"- modal_app_name: `{APP_NAME}`",
        f"- formal_cloud_started: `{result['formal_cloud_started']}`",
        f"- strict_candidate_passes_detected: `{result['formal_entrypoint_detects_strict_candidate_passes']}`",
        f"- reads_candidate_package: `{result['formal_cloud_reads_candidate_package']}`",
        f"- exits_cleanly: `{result['formal_cloud_exits_cleanly']}`",
        f"- no_teacher_package_write: `{result['no_teacher_package_write']}`",
    ]
    (OUT_DIR / "modal_smoke_result.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

