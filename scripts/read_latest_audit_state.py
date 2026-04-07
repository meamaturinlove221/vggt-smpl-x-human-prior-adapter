import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "output" / "zju_source_policy_research_loop"
WATCH_JSON = REPO_ROOT / "output" / "zju_source_policy_research_watch" / "latest_watch_snapshot.json"
TASK_PLAN_JSON = REPO_ROOT / "output" / "zju_source_policy_rawpool_status_20260326_current" / "task_plan.json"
ALLOWLIST_JSON = RESEARCH_ROOT / "repo_process_allowlist.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _latest_matching(pattern: str) -> Path | None:
    matches = sorted(RESEARCH_ROOT.glob(pattern))
    return matches[-1] if matches else None


def collect_latest_audit_state() -> dict:
    panel_dir = REPO_ROOT / "output" / "teacher_geometry_multiview_correspondence_audit" / "task.20260407" / "advisor_panels"
    panels = sorted(str(path.relative_to(REPO_ROOT)).replace("\\", "/") for path in panel_dir.glob("*.png"))
    payload = {
        "checked_at": _load_json(RESEARCH_ROOT / "research_loop_status.json").get("checked_at", ""),
        "research_status": _load_json(RESEARCH_ROOT / "research_loop_status.json"),
        "watch": _load_json(WATCH_JSON),
        "task_plan": _load_json(TASK_PLAN_JSON),
        "allowlist": _load_json(ALLOWLIST_JSON),
        "correspondence_result": _load_json(RESEARCH_ROOT / "teacher_geometry_multiview_correspondence_audit_result.json"),
        "correspondence_postmortem": _load_json(RESEARCH_ROOT / "teacher_geometry_multiview_correspondence_audit_postmortem.json"),
        "correspondence_loop_state": _load_json(RESEARCH_ROOT / "teacher_geometry_multiview_correspondence_audit_loop_state.json"),
        "proxy_best": _load_json(_latest_matching("teacher_geometry_correspondence_proxy_best.iter*.json")),
        "proxy_ranking": _load_json(_latest_matching("teacher_geometry_correspondence_proxy_ranking.iter*.json")),
        "proxy_sweep": _load_json(_latest_matching("teacher_geometry_correspondence_proxy_sweep.iter*.json")),
        "iteration_report": _load_json(_latest_matching("teacher_geometry_multiview_correspondence_audit_iteration_report.iter*.json")),
        "iteration_decision": _load_json(_latest_matching("teacher_geometry_multiview_correspondence_audit_iteration_decision.iter*.json")),
        "panels": panels,
    }
    return payload


def main() -> int:
    print(json.dumps(collect_latest_audit_state(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
