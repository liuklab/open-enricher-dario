"""Network client and automated tunneling (Infrastructure Layer)."""

from __future__ import annotations

import shlex
import socket
import subprocess
import time
from contextlib import contextmanager
from typing import Generator

import requests


def _wait_for_local_port(port: int, timeout: float) -> bool:
    """Return True once something accepts TCP connections on localhost."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _start_colab_tunnel(session: str, port: int) -> subprocess.Popen[str]:
    """Forward a local port to the Colab VM through `colab ssh --proxy-mode`."""
    proxy = "colab ssh --proxy-mode -s " + shlex.quote(session)
    command = [
        "ssh",
        "-N",
        "-L",
        f"{port}:127.0.0.1:{port}",
        "-o",
        f"ProxyCommand={proxy}",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "LogLevel=ERROR",
        "root@colab-runtime",
    ]
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _stop_process(process: subprocess.Popen[str]) -> str:
    """Terminate a tunnel process and return any stderr it produced."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if process.stderr is None:
        return ""
    return process.stderr.read().strip()


@contextmanager
def setup_server_connection(mode: str, target: str, port: int = 8000) -> Generator[str, None, None]:
    """Abstracts network complexity by auto-managing SSH tunnels if needed."""
    if mode == "tailscale":
        yield f"http://{target}:{port}"

    elif mode == "kaggle":
        from deployment.kaggle import read_tunnel_url

        print(f"[INFO] Looking up the Kaggle tunnel for '{target}'...")
        yield read_tunnel_url(target)

    elif mode == "colab-cli":
        print(f"[INFO] Automatically establishing SSH tunnel to '{target}'...")
        process = _start_colab_tunnel(target, port)
        try:
            if not _wait_for_local_port(port, timeout=30) or process.poll() is not None:
                detail = _stop_process(process)
                message = f"Colab SSH tunnel did not open localhost:{port}."
                if detail:
                    message = f"{message} {detail}"
                raise RuntimeError(message)
            yield f"http://127.0.0.1:{port}"
        finally:
            print("[INFO] Tearing down SSH tunnel...")
            _stop_process(process)

    else:
        raise ValueError(f"Unknown network mode: {mode}")

def post_stage(base_url: str, stage_name: str, video_name: str, timeout: int) -> dict[str, object]:
    """Underlying HTTP POST request to the FastAPI worker."""
    url = f"{base_url}/stage/{stage_name}"
    payload = {"video_name": video_name}
    
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()

class PipelineClient:
    """Handles network communication for pipeline stages."""
    
    def __init__(self, base_url: str, timeout: int = 3600):
        self.base_url = base_url
        self.timeout = timeout

    def execute_stage(self, stage_name: str, video_name: str) -> dict[str, object]:
        print(f"[INFO] Initiating stage over network: {stage_name}")
        return post_stage(
            base_url=self.base_url,
            stage_name=stage_name,
            video_name=video_name,
            timeout=self.timeout
        )