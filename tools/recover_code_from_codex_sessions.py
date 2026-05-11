from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SESSIONS = Path(r"C:\Users\WINDOWS\.codex\sessions")
RECOVERY = ROOT / "recovery_from_sessions"
RESTORED = RECOVERY / "restored_files"
PATCHES = RECOVERY / "patches"
INDEX_JSON = RECOVERY / "session_recovery_index.json"
INDEX_MD = RECOVERY / "session_recovery_index.md"


KEYWORDS = [
    r"D:\\vggt\\vggt-main",
    "V50",
    "V62",
    "V121",
    "V223",
    "strict_registry_entry_v50",
    "candidate_package_v50",
    "v223_mentor_final_controller",
    "v121_v220_release_controller",
    "v62_v120_multibranch_controller",
]


@dataclass
class PatchRecord:
    session: str
    timestamp: str
    call_id: str
    kind: str
    file: str
    patch_index: int
    recovered_path: str | None = None


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iter_session_files() -> list[Path]:
    if not SESSIONS.exists():
        return []
    return sorted(SESSIONS.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except Exception:
                continue


def extract_apply_patch_input(item: dict[str, Any]) -> tuple[str, str] | None:
    payload = item.get("payload") or {}
    if payload.get("type") != "custom_tool_call":
        return None
    if payload.get("name") != "apply_patch":
        return None
    return str(payload.get("call_id", "")), str(payload.get("input", ""))


def split_hunks(patch: str) -> list[tuple[str, str, str]]:
    """Return (kind, filename, body) for simple Add/Update/Delete hunks."""
    pattern = re.compile(
        r"^\*\*\* (Add File|Update File|Delete File): ([^\n]+)\n",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(patch))
    hunks: list[tuple[str, str, str]] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else patch.find("*** End Patch", start)
        if end == -1:
            end = len(patch)
        hunks.append((match.group(1), match.group(2).strip(), patch[start:end]))
    return hunks


def recover_add_file(filename: str, body: str) -> str:
    lines = []
    for line in body.splitlines():
        if line.startswith("+"):
            lines.append(line[1:])
    return "\n".join(lines) + "\n"


def safe_output_path(filename: str) -> Path:
    normalized = filename.replace("\\", "/").lstrip("/")
    while normalized.startswith("../"):
        normalized = normalized[3:]
    return RESTORED / normalized


def main() -> None:
    RECOVERY.mkdir(parents=True, exist_ok=True)
    RESTORED.mkdir(parents=True, exist_ok=True)
    PATCHES.mkdir(parents=True, exist_ok=True)

    session_rows = []
    records: list[PatchRecord] = []
    keyword_hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    add_file_latest: dict[str, PatchRecord] = {}

    patch_counter = 0
    for session in iter_session_files():
        text_hit_counts = {k: 0 for k in KEYWORDS}
        session_patch_count = 0
        session_add_count = 0
        session_update_count = 0

        for line_no, item in load_jsonl(session):
            raw = json.dumps(item, ensure_ascii=False)
            for kw in KEYWORDS:
                if kw in raw:
                    text_hit_counts[kw] += 1
                    if len(keyword_hits[kw]) < 40:
                        keyword_hits[kw].append(
                            {
                                "session": str(session),
                                "line": line_no,
                                "timestamp": item.get("timestamp"),
                            }
                        )

            extracted = extract_apply_patch_input(item)
            if not extracted:
                continue
            call_id, patch = extracted
            patch_counter += 1
            session_patch_count += 1
            patch_path = PATCHES / f"patch_{patch_counter:05d}_{call_id or 'no_call_id'}.patch"
            patch_path.write_text(patch, encoding="utf-8")
            timestamp = str(item.get("timestamp", ""))
            for kind, filename, body in split_hunks(patch):
                rec = PatchRecord(
                    session=str(session),
                    timestamp=timestamp,
                    call_id=call_id,
                    kind=kind,
                    file=filename,
                    patch_index=patch_counter,
                )
                if kind == "Add File":
                    session_add_count += 1
                    content = recover_add_file(filename, body)
                    out_path = safe_output_path(filename)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(content, encoding="utf-8")
                    rec.recovered_path = str(out_path)
                    add_file_latest[filename] = rec
                elif kind == "Update File":
                    session_update_count += 1
                records.append(rec)

        hit_total = sum(text_hit_counts.values())
        if hit_total or session_patch_count:
            session_rows.append(
                {
                    "session": str(session),
                    "mtime_utc": datetime.fromtimestamp(session.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "size": session.stat().st_size,
                    "keyword_hit_counts": {k: v for k, v in text_hit_counts.items() if v},
                    "apply_patch_count": session_patch_count,
                    "add_file_count": session_add_count,
                    "update_file_count": session_update_count,
                }
            )

    payload = {
        "task": "recover_code_from_codex_sessions",
        "created_utc": now(),
        "sessions_root": str(SESSIONS),
        "recovery_root": str(RECOVERY),
        "session_count_indexed": len(session_rows),
        "patch_count": patch_counter,
        "recovered_add_file_count": len(add_file_latest),
        "record_count": len(records),
        "sessions": session_rows,
        "keyword_hits": keyword_hits,
        "records": [r.__dict__ for r in records],
        "latest_add_files": {k: v.__dict__ for k, v in sorted(add_file_latest.items())},
    }
    INDEX_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Session Code Recovery Index",
        "",
        f"- created_utc: `{payload['created_utc']}`",
        f"- sessions_indexed: `{payload['session_count_indexed']}`",
        f"- patch_count: `{payload['patch_count']}`",
        f"- recovered_add_file_count: `{payload['recovered_add_file_count']}`",
        "",
        "## Latest Recovered Files",
    ]
    for file, rec in sorted(add_file_latest.items()):
        lines.append(f"- `{file}` -> `{rec.recovered_path}`")
    INDEX_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"index": str(INDEX_JSON), "recovered_add_file_count": len(add_file_latest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
