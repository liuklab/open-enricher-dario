"""Kernel entrypoint. Extract the checkout, start the API, publish a tunnel URL.

Kaggle kernels have no inbound SSH. This script keeps the same FastAPI worker
used on Colab and prints ``OPEN_ENRICHER_TUNNEL_URL`` so the laptop client can
reach port 8000. The process then sleeps so the kernel stays up for stage calls.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

CODE_INPUT = Path("/kaggle/input/open-enricher-code")
VIDEO_INPUT = Path("/kaggle/input/open-enricher-dataset")
HOME = Path("/kaggle/working")
DATA_ROOT = HOME / "Open-Enricher-Database"
TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-linux-amd64"
)
HOLD_SECONDS = 8 * 60 * 60


def _export_runtime() -> None:
    os.environ["OPEN_ENRICHER_HOME"] = str(HOME)
    os.environ["OPEN_ENRICHER_DATA_ROOT"] = str(DATA_ROOT)
    os.environ["OPEN_ENRICHER_VIDEO_INPUTS"] = str(VIDEO_INPUT)
    os.environ["OPEN_ENRICHER_ENV"] = str(CODE_INPUT / ".env")
    os.environ["OPEN_ENRICHER_TARBALL"] = str(CODE_INPUT / "open-enricher.tgz")
    os.environ["PYTHONUNBUFFERED"] = "1"


def _run_launch() -> None:
    script = CODE_INPUT / "launch.sh"
    if not script.is_file():
        raise SystemExit(f"Missing {script}. The code dataset was not uploaded.")
    subprocess.run(["bash", str(script)], check=True)


def _cloudflared() -> Path:
    binary = Path("/tmp/cloudflared")
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary
    print("[INFO] Downloading cloudflared for the outbound tunnel...")
    urllib.request.urlretrieve(CLOUDFLARED_URL, binary)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    return binary


def _drain(stream) -> None:
    for line in stream:
        print(line, end="", flush=True)


def _open_tunnel() -> None:
    process = subprocess.Popen(
        [
            str(_cloudflared()),
            "tunnel",
            "--url",
            "http://127.0.0.1:8000",
            "--no-autoupdate",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        match = TUNNEL_RE.search(line)
        if match:
            print(f"OPEN_ENRICHER_TUNNEL_URL={match.group(0)}", flush=True)
            threading.Thread(target=_drain, args=(process.stdout,), daemon=True).start()
            return
    raise SystemExit("cloudflared exited before publishing a tunnel URL.")


def main() -> None:
    _export_runtime()
    _run_launch()
    _open_tunnel()
    print(f"[INFO] Holding the Kaggle kernel open for {HOLD_SECONDS} seconds.")
    time.sleep(HOLD_SECONDS)


if __name__ == "__main__":
    main()
