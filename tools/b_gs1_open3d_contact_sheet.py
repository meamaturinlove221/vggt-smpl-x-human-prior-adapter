from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import b_gs0_open3d_contact_sheet as contact  # noqa: E402


def main() -> int:
    # Reuse the B-GS0 contact-sheet renderer, but point it at the B-GS1
    # artifacts and family of PLY filenames. This remains visual review only.
    contact.PLY_FILES = (
        "b_gs1_constrained_baseline.ply",
        "b_gs1_raw_free_candidates.ply",
        "b_gs1_visibility_selected_free.ply",
        "b_gs1_visibility_aware_combined.ply",
        "b_gs1_random_control_combined.ply",
    )
    contact.DEFAULT_INPUT_DIR = Path("output/surface_research_preflight_local/B_GS1_visibility_aware_free_gaussian_backend")
    contact.DEFAULT_OUTPUT_DIR = contact.DEFAULT_INPUT_DIR / "open3d_contact_sheet"
    return contact.main()


if __name__ == "__main__":
    raise SystemExit(main())
