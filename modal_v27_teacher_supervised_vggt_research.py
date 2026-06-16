from __future__ import annotations

"""Research-only Modal entrypoint placeholder for V27.

The local V27 audit writes and validates the research training contract. This Modal
entrypoint intentionally refuses formal outputs and delegates to the same auditor
inside the mounted workspace when invoked.
"""

from pathlib import Path

import modal


APP_NAME = "vggt-v27-teacher-supervised-research"
WORKDIR = Path("/root/vggt-main")

app = modal.App(APP_NAME)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyyaml")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    volumes={"/root/vggt-main": modal.Volume.from_name("vggt-workspace", create_if_missing=True)},
)
def run_v27_research_training() -> dict:
    import json
    import subprocess

    cmd = ["python", "tools/v27_teacher_supervised_audit.py"]
    proc = subprocess.run(cmd, cwd=str(WORKDIR), text=True, capture_output=True, check=False)
    return {
        "research_only": True,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "no_predictions_write": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_registry_write": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


if __name__ == "__main__":
    # Local execution keeps this file non-destructive and aligned with the research-only contract.
    print(
        json.dumps(
            {
                "app": APP_NAME,
                "research_only": True,
                "entrypoint": "modal_v27_teacher_supervised_vggt_research.py::run_v27_research_training",
                "formal_outputs": "forbidden",
            },
            indent=2,
        )
    )
