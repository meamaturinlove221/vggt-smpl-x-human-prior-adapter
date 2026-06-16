from __future__ import annotations

import sys

from v11_surface_completion_pipeline import main


if __name__ == "__main__":
    sys.argv.insert(1, "g3")
    raise SystemExit(main())
