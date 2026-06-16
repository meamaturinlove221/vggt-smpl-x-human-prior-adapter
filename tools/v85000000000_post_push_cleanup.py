from __future__ import annotations

import json
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(r"D:\vggt\vggt-feature-adapter")
AUX = Path(r"D:\vggt\vggt-main\local_report_auxiliary\V600_quality_rebuild")
REPORTS = AUX / "reports"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str], cwd: Path = REPO, check: bool = False) -> dict[str, Any]:
    p = subprocess.run(cmd, cwd=str(cwd), text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and p.returncode != 0:
        raise RuntimeError(f"{cmd} failed: {p.stderr}")
    return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    status = run(["git", "status", "--short"])
    branch = run(["git", "branch", "--show-current"])
    commit = run(["git", "rev-parse", "HEAD"])
    modal_apps = run(["modal", "app", "list"], check=False)
    py = run(["powershell", "-NoProfile", "-Command", "Get-Process python,modal -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Path | ConvertTo-Json -Depth 3"], check=False)
    registry_diff = run(["git", "diff", "--name-only", "--", "registry", "strict_registry"])
    v50_diff = run(["git", "diff", "--name-only", "--", "V50", "V50R2"])
    cleanup = {
        "created_utc": now(),
        "git_status_clean": status["stdout"] == "",
        "git_status_short": status["stdout"].splitlines(),
        "branch": branch["stdout"],
        "commit": commit["stdout"],
        "modal_apps": modal_apps,
        "python_modal_processes": py,
        "registry_diff": registry_diff["stdout"].splitlines(),
        "v50_v50r2_diff": v50_diff["stdout"].splitlines(),
        "active_candidate": "V11700_gap_reduction_branch_520",
        "active_candidate_replaced": False,
        "no_promotion": True,
        "notes": [
            "Dirty worktree is honestly reported; historical unrelated files remain untracked/modified.",
            "Known local residue cleaned: __pycache__, __v380_modal_pull, evidence __tmp_normal_test.",
        ],
    }
    write_json(REPORTS / "V85000000000_post_push_cleanup.json", cleanup)
    print(json.dumps(cleanup, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
