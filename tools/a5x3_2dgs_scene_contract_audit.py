from __future__ import annotations

import sys

from v10_surface_completion_pipeline import main


if __name__ == "__main__":
    sys.argv.insert(1, "audit-2dgs")
    raise SystemExit(main())
