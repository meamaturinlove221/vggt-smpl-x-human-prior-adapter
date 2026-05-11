from __future__ import annotations

import json
import os
from pathlib import PurePosixPath

import modal


APP_NAME = os.environ.get("VGGT_MODAL_PSHUMAN_SPACE_APP_NAME", "vggt-pshuman-space-probe")
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
GPU_SPEC = os.environ.get("VGGT_MODAL_PSHUMAN_SPACE_GPU", "A10G")
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_PSHUMAN_SPACE_TIMEOUT_SEC", str(30 * 60)))

image = modal.Image.debian_slim(python_version="3.10").pip_install("gradio_client==2.5.0")

app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU_SPEC,
    timeout=TIMEOUT_SEC,
    volumes={REMOTE_OUTPUT_DIR.as_posix(): output_volume},
)
def probe_space(space_id: str = "fffiloni/PSHuman", output_subdir: str = "pshuman_space_probe") -> dict:
    from gradio_client import Client

    out_root = REMOTE_OUTPUT_DIR / output_subdir.strip("/").replace("\\", "/")
    out_path = str(out_root)
    os.makedirs(out_path, exist_ok=True)
    summary = {"ok": False, "space_id": space_id, "output_subdir": output_subdir}
    try:
        client = Client(space_id)
        api = client.view_api(return_format="dict")
        summary["ok"] = True
        summary["api"] = api
    except Exception as exc:
        summary["error"] = repr(exc)
    with open(os.path.join(out_path, "pshuman_space_probe_summary.json"), "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)
    output_volume.commit()
    return summary


@app.local_entrypoint()
def main(space_id: str = "fffiloni/PSHuman", output_subdir: str = "pshuman_space_probe"):
    summary = probe_space.remote(space_id=space_id, output_subdir=output_subdir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
