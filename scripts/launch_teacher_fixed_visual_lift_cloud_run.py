import argparse
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import modal_zju_visual_lift_benchmark as visual_benchmark  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a visual-lift cloud run through the Modal Python SDK.")
    parser.add_argument("--cfg-json-path", required=True)
    parser.add_argument("--app-name", default=visual_benchmark.APP_NAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg_json = Path(args.cfg_json_path).read_text(encoding="utf-8").strip()
    fn = modal.Function.from_name(args.app_name, "run_remote_visual_lift_benchmark")
    call = fn.spawn(cfg_json)
    print(call.object_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
