"""Network client and automated tunneling (Infrastructure Layer)."""

from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager
from typing import Generator

import requests

@contextmanager
def setup_server_connection(mode: str, target: str, port: int = 8000) -> Generator[str, None, None]:
    """Abstracts network complexity by auto-managing SSH tunnels if needed."""
    if mode == "tailscale":
        yield f"http://{target}:{port}"
        
    elif mode == "colab-cli":
        print(f"[INFO] Automatically establishing SSH tunnel to '{target}'...")
        # Apriamo il tunnel come processo in background
        process = subprocess.Popen(
            ["colab", "tunnel", "--port", str(port), "--session", target],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(3)  # Diamo 3 secondi a SSH per stabilire la connessione
        
        try:
            yield f"http://localhost:{port}"
        finally:
            print("[INFO] Tearing down SSH tunnel...")
            process.terminate()
            process.wait()
            
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