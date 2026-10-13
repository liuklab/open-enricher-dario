"""Minimal HTTP client for POST /stages/{name}. No Tailscale logic."""

from __future__ import annotations

from typing import Any

import requests


def post_stage(
    base_url: str,
    name: str,
    video: str,
    *,
    timeout: int = 3600,
) -> dict[str, Any]:
    """POST one stage by name. `video` is a file name resolved on the worker."""

    url = f"{base_url.rstrip('/')}/stages/{name}"
    response = requests.post(url, json={"video": video}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object from {url}, got {type(payload).__name__}")
    return payload
