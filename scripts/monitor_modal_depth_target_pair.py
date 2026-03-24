import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


STEP_PATTERN = re.compile(
    r"(?P<phase>Train|Val) Epoch:\s+\[\d+\]\[\s*(?P<step>\d+)/(?P<total>\d+)\]"
)
METRIC_PATTERN = re.compile(
    r"Loss/(?P<phase>train|val)_(?P<name>[A-Za-z0-9_]+):\s+"
    r"(?P<current>-?\d+(?:\.\d+)?)\s+\((?P<average>-?\d+(?:\.\d+)?)\)"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lightweight local monitor for a detached Modal depth-target reliability pair run."
    )
    parser.add_argument("--modal-exe", default=r"D:\anaconda\Scripts\modal.exe")
    parser.add_argument("--app-id", required=True)
    parser.add_argument("--volume-name", default="vggt-out")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--max-polls", type=int, default=0, help="0 means unlimited until completion.")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def run_modal_capture(modal_exe, args):
    cmd = [modal_exe, *args]
    result = subprocess.run(cmd, capture_output=True)
    stdout = result.stdout.decode("utf-8", errors="ignore")
    stderr = result.stderr.decode("utf-8", errors="ignore")
    return {
        "returncode": int(result.returncode),
        "stdout": stdout,
        "stderr": stderr,
        "command": cmd,
    }


def get_app_state(modal_exe, app_id):
    payload = run_modal_capture(modal_exe, ["app", "list", "--json"])
    if payload["returncode"] != 0:
        return {"error": payload["stderr"].strip() or payload["stdout"].strip()}
    app_rows = json.loads(payload["stdout"])
    for row in app_rows:
        if row.get("App ID") == app_id:
            return {
                "app_id": app_id,
                "description": row.get("Description"),
                "state": row.get("State"),
                "tasks": row.get("Tasks"),
                "created_at": row.get("Created at"),
                "stopped_at": row.get("Stopped at"),
            }
    return {"app_id": app_id, "state": "not_found"}


def get_volume_file_text(modal_exe, volume_name, remote_path):
    payload = run_modal_capture(modal_exe, ["volume", "get", volume_name, remote_path, "-"])
    text = payload["stdout"].strip()
    if text:
        return text
    return ""


def list_volume_path(modal_exe, volume_name, remote_path):
    payload = run_modal_capture(modal_exe, ["volume", "ls", volume_name, remote_path])
    if payload["returncode"] != 0:
        return []
    return [line.strip() for line in payload["stdout"].splitlines() if line.strip()]


def volume_file_exists(modal_exe, volume_name, remote_path):
    parent = str(Path(remote_path).parent).replace("\\", "/")
    entries = list_volume_path(modal_exe, volume_name, parent)
    target_name = Path(remote_path).name
    for entry in entries:
        normalized = entry.replace("\\", "/").rstrip("/")
        if normalized.endswith("/" + target_name) or normalized == target_name:
            return True
    return False


def parse_latest_log_summary(log_text):
    phase_summaries = {"train": None, "val": None}
    for raw_line in log_text.splitlines():
        step_match = STEP_PATTERN.search(raw_line)
        metric_matches = list(METRIC_PATTERN.finditer(raw_line))
        if not step_match or not metric_matches:
            continue
        phase = step_match.group("phase").lower()
        summary = {
            "step": int(step_match.group("step")),
            "total": int(step_match.group("total")),
            "metrics": {},
        }
        for metric_match in metric_matches:
            metric_phase = metric_match.group("phase")
            if metric_phase != phase:
                continue
            summary["metrics"][metric_match.group("name")] = {
                "current": float(metric_match.group("current")),
                "average": float(metric_match.group("average")),
            }
        phase_summaries[phase] = summary
    return phase_summaries


def load_pair_status(modal_exe, volume_name, remote_root):
    text = get_volume_file_text(modal_exe, volume_name, f"{remote_root}/pair_status.json")
    if not text:
        return {"error": "pair_status.json not readable yet"}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "pair_status.json decode failed", "raw": text[:1000]}


def get_stage_log_text(modal_exe, volume_name, remote_root, stage_name, stage_status):
    preferred_paths = []
    driver_live_log = (stage_status or {}).get("driver_live_log")
    if driver_live_log:
        preferred_paths.append(driver_live_log)
    preferred_paths.append(f"{remote_root}/{stage_name}/driver_live.log")
    preferred_paths.append(f"{remote_root}/{stage_name}/logs/log.txt")

    checked = []
    for remote_path in preferred_paths:
        normalized = remote_path.replace("\\", "/")
        if normalized in checked:
            continue
        checked.append(normalized)
        if not volume_file_exists(modal_exe, volume_name, normalized):
            continue
        text = get_volume_file_text(modal_exe, volume_name, normalized)
        if text:
            return normalized, text
    return None, ""


def build_stage_log_status(modal_exe, volume_name, remote_root, stage_name, stage_status=None):
    remote_log_dir = f"{remote_root}/{stage_name}/logs"
    entries = list_volume_path(modal_exe, volume_name, remote_log_dir)
    result = {
        "stage_name": stage_name,
        "remote_log_dir": remote_log_dir,
        "entries": entries,
        "log_exists": False,
        "log_source": None,
        "summary": None,
    }
    log_remote_path, text = get_stage_log_text(
        modal_exe,
        volume_name,
        remote_root,
        stage_name,
        stage_status,
    )
    if text:
        result["log_exists"] = True
        result["log_source"] = log_remote_path
        result["summary"] = parse_latest_log_summary(text)
    return result


def write_snapshot(output_dir, snapshot):
    ensure_dir(output_dir)
    snapshot_path = Path(output_dir) / "latest_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Modal Pair Monitor",
        "",
        f"- timestamp: `{snapshot['timestamp']}`",
        f"- app_id: `{snapshot['app_state'].get('app_id', '')}`",
        f"- app_state: `{snapshot['app_state'].get('state', '')}`",
        f"- remote_root: `{snapshot['remote_root']}`",
    ]
    pair_status = snapshot.get("pair_status", {})
    if pair_status:
        md_lines.extend(
            [
                f"- pair_state: `{pair_status.get('state', '')}`",
                "",
                "## Stages",
                "",
                "| Stage | Pair State | Train Step | Val Step | loss_objective(train avg) | loss_objective(val avg) |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        stages = pair_status.get("stages", {})
        stage_names = snapshot.get("stage_names") or list(stages.keys()) or ["baseline", "unproject"]
        for stage_name in stage_names:
            stage_status = stages.get(stage_name, {})
            log_status = snapshot["stage_logs"].get(stage_name, {})
            train_summary = (log_status.get("summary") or {}).get("train")
            val_summary = (log_status.get("summary") or {}).get("val")
            train_obj = None
            val_obj = None
            if train_summary and "objective" in train_summary["metrics"]:
                train_obj = train_summary["metrics"]["objective"]["average"]
            if val_summary and "objective" in val_summary["metrics"]:
                val_obj = val_summary["metrics"]["objective"]["average"]
            md_lines.append(
                "| `{stage}` | `{pair_state}` | {train_step} | {val_step} | {train_obj} | {val_obj} |".format(
                    stage=stage_name,
                    pair_state=stage_status.get("state", "n/a"),
                    train_step="n/a" if not train_summary else f"{train_summary['step']}/{train_summary['total']}",
                    val_step="n/a" if not val_summary else f"{val_summary['step']}/{val_summary['total']}",
                    train_obj="n/a" if train_obj is None else f"{train_obj:.4f}",
                    val_obj="n/a" if val_obj is None else f"{val_obj:.4f}",
                )
            )
    (Path(output_dir) / "latest_snapshot.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def should_stop(snapshot):
    pair_state = (snapshot.get("pair_status") or {}).get("state")
    app_state = (snapshot.get("app_state") or {}).get("state", "")
    if pair_state in {"completed", "failed"}:
        return True
    if isinstance(app_state, str) and ("stopped" in app_state or app_state == "not_found"):
        return True
    return False


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    poll_index = 0
    while True:
        poll_index += 1
        pair_status = load_pair_status(args.modal_exe, args.volume_name, args.remote_root)
        stage_names = list((pair_status.get("stages") or {}).keys()) or ["baseline", "unproject"]
        snapshot = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "app_state": get_app_state(args.modal_exe, args.app_id),
            "pair_status": pair_status,
            "remote_root": args.remote_root,
            "stage_names": stage_names,
            "stage_logs": {
                stage_name: build_stage_log_status(
                    args.modal_exe,
                    args.volume_name,
                    args.remote_root,
                    stage_name,
                    (pair_status.get("stages") or {}).get(stage_name, {}),
                )
                for stage_name in stage_names
            },
            "poll_index": poll_index,
        }
        write_snapshot(output_dir, snapshot)
        if args.once or should_stop(snapshot):
            break
        if args.max_polls > 0 and poll_index >= args.max_polls:
            break
        time.sleep(max(args.interval_sec, 10))


if __name__ == "__main__":
    main()
