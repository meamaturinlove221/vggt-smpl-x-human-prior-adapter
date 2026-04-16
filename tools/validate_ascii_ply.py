from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a simple ASCII PLY file by header and vertex line count.")
    parser.add_argument("ply_path", help="Path to the ASCII .ply file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.ply_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PLY file not found: {path}")

    with path.open("r", encoding="utf-8", errors="strict") as handle:
        header_lines = 0
        vertex_count = None
        for raw_line in handle:
            header_lines += 1
            line = raw_line.strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            if line == "end_header":
                break

        if vertex_count is None:
            raise ValueError(f"Missing 'element vertex' header in {path}")

        data_lines = 0
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 6:
                raise ValueError(
                    f"Unexpected vertex field count in {path}: expected 6 fields, got {len(parts)} in line '{line}'"
                )
            data_lines += 1

    if data_lines != vertex_count:
        raise ValueError(
            f"PLY vertex count mismatch for {path}: header={vertex_count}, data_lines={data_lines}, total_expected_lines={header_lines + vertex_count}"
        )

    print(f"valid ascii ply: {path}")
    print(f"header_lines={header_lines}")
    print(f"vertex_count={vertex_count}")
    print(f"data_lines={data_lines}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
