from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-candidate-dir", type=Path, required=True)
    parser.add_argument("--strict-registry-entry", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rollback-root", type=Path, required=True)
    parser.add_argument("--research-or-formal-mode", default="formal_candidate_only")
    args = parser.parse_args()

    # This wrapper intentionally delegates to the local runner. It is the formal entrypoint
    # contract used by V62; cloud execution can wrap the same CLI once artifact volumes are
    # mounted. The important property is that all inputs are explicit frozen-package paths.
    cmd = [
        sys.executable,
        "tools/v62_formal_candidate_finetune_runner.py",
        "--frozen-candidate-dir",
        str(args.frozen_candidate_dir),
        "--strict-registry-entry",
        str(args.strict_registry_entry),
        "--max-steps",
        str(args.max_steps),
        "--output-root",
        str(args.output_root),
        "--rollback-root",
        str(args.rollback_root),
        "--research-or-formal-mode",
        args.research_or_formal_mode,
    ]
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parent, text=True, capture_output=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()

