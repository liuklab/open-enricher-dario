"""Load KEY=value pairs from a local .env without extra dependencies."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> Path | None:
    """Populate os.environ from .env. Existing variables are left unchanged."""

    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        candidates.append(Path.cwd() / ".env")
        # src/open_enricher/env.py -> repository root
        candidates.append(Path(__file__).resolve().parents[2] / ".env")

    env_path = next((p for p in candidates if p.is_file()), None)
    if env_path is None:
        return None

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.lower().startswith("export "):
            key = key[7:].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
    return env_path


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)
