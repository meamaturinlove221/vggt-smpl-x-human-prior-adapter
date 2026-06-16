from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "tools" / "v9_backend_cloud_logs"
ACTIVE_PATH = LOG_DIR / "active_backend_smokes.json"


def _load_records() -> list[dict]:
    if not ACTIVE_PATH.is_file():
        return []
    return json.loads(ACTIVE_PATH.read_text(encoding="utf-8"))


def _pid_alive(pid: int) -> bool:
    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"if (Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue) {{ 'alive' }}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return "alive" in proc.stdout


def _tail(path: str, chars: int = 1200) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        return ""
    data = file_path.read_text(encoding="utf-8", errors="replace")
    return data[-chars:]


def main() -> int:
    records = _load_records()
    rows = []
    for item in records:
        summary_path = Path(item["Summary"])
        summary = None
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - diagnostic helper
                summary = {"status": f"unreadable_summary:{exc!r}"}
        rows.append(
            {
                "backend": item["Backend"],
                "pid": item["Pid"],
                "alive": _pid_alive(int(item["Pid"])),
                "summary_exists": summary_path.is_file(),
                "status": summary.get("status") if summary else None,
                "decision": summary.get("decision") if summary else None,
                "stdout_tail": _tail(item["Stdout"]),
                "stderr_tail": _tail(item["Stderr"]),
            }
        )
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    if rows and all((row["summary_exists"] or not row["alive"]) for row in rows):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
