import modal

app = modal.App("vggt-modal-minimal-cpu-smoke")


@app.function(cpu=1, memory=1024, timeout=300)
def ping(text: str) -> str:
    return f"pong:{text}"


@app.local_entrypoint()
def main(text: str = "ok") -> None:
    print(ping.remote(text))
