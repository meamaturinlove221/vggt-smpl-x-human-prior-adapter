import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_STATUS = REPO_ROOT / "output" / "zju_source_policy_research_loop" / "research_loop_status.json"
WATCH_JSON = REPO_ROOT / "output" / "zju_source_policy_research_watch" / "latest_watch_snapshot.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    checked_at = datetime.now().astimezone().isoformat()
    research = _load(RESEARCH_STATUS)
    research["checked_at"] = checked_at
    research["current_priority_reason"] = "teacher_fixed_visual_lift_benchmark completed clean locally and in cloud, so no active family remains open."
    research["preferred_first_family_reason"] = "No auto-next ticket is currently selected. The current cloud-deliverable line is complete; wait for a fresh manual problem."
    research["current_frontier_priority"] = "teacher_fixed_visual_lift_benchmark completed clean and no further cloud action is active."
    _write(RESEARCH_STATUS, research)

    watch = _load(WATCH_JSON)
    watch["checked_at"] = checked_at
    if "research" in watch and isinstance(watch["research"], dict):
        watch["research"]["research_status"] = research
        if "summary" in watch["research"] and isinstance(watch["research"]["summary"], dict):
            watch["research"]["summary"]["state"] = "IDLE_GUARD"
            watch["research"]["summary"]["approved_problem_present"] = False
            watch["research"]["summary"]["approved_problem_ready"] = False
            watch["research"]["summary"]["manual_action_required"] = False
            watch["research"]["summary"]["manual_action_kind"] = ""
            watch["research"]["summary"]["ready_for_execution"] = False
            watch["research"]["summary"]["current_review_packet"] = "output\\zju_source_policy_research_loop\\teacher_fixed_visual_lift_cloud_deliverable_completion_result.20260404.json"
    watch["watch_conclusion"] = "teacher_fixed_visual_lift benchmark completed clean locally and in cloud, no active cloud app remains, and no active local family run is open"
    _write(WATCH_JSON, watch)
    print(RESEARCH_STATUS)
    print(WATCH_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
