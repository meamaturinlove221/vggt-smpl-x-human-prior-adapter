from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def run_text(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace")
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def powershell_json(script: str) -> list[dict[str, object]]:
    code, out = run_text(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script])
    if code != 0:
        return [{"error": out.strip(), "returncode": code}]
    text = out.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [{"raw": text}]
    if isinstance(data, list):
        return data
    return [data]


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    self_pid = os.getpid()
    process_script = (
        "$names=@('python','python3','modal'); "
        f"$self={self_pid}; "
        "Get-Process -ErrorAction SilentlyContinue | "
        "Where-Object { $names -contains $_.ProcessName -and $_.Id -ne $self } | "
        "Select-Object Id,ProcessName,CPU,StartTime,Path | ConvertTo-Json -Depth 4"
    )
    before = powershell_json(process_script)
    app_code, app_text = run_text(["modal", "app", "list"])
    container_code, container_text = run_text(["modal", "container", "list"])
    time.sleep(3)
    after = powershell_json(process_script)

    data = {
        "task": "v50r2_no_residual_process_proof",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "excluded_self_pid": self_pid,
        "local_processes_before_modal_checks": before,
        "local_processes_after_modal_checks": after,
        "modal_app_list_returncode": app_code,
        "modal_app_list": app_text,
        "modal_container_list_returncode": container_code,
        "modal_container_list": container_text,
        "modal_apps_empty": app_code == 0 and "Apps" in app_text and "Running" not in app_text and "app-" not in app_text,
        "modal_containers_empty": container_code == 0 and "Active Containers" in container_text and "None" in container_text,
        "residual_local_python_or_modal_count_after": len(after),
    }
    json_path = REPORTS / "20260509_v50r2_no_residual_process_proof.json"
    md_path = REPORTS / "20260509_v50r2_no_residual_process_proof.md"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# V50R2 无残留进程证明",
                "",
                f"- generated: {data['created_utc']}",
                f"- modal apps empty: {data['modal_apps_empty']}",
                f"- modal containers empty: {data['modal_containers_empty']}",
                f"- residual local python/modal process count after checks: {data['residual_local_python_or_modal_count_after']}",
                "",
                "## Modal app list",
                "```text",
                app_text.strip(),
                "```",
                "",
                "## Modal container list",
                "```text",
                container_text.strip(),
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
