"""Load committed project settings from config.toml."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_path() -> Path:
    return repo_root() / "config.toml"


def parse_simple_toml(text: str) -> dict[str, dict[str, Any]]:
    """Parse the small TOML subset used by config.toml (tables, strings, ints)."""

    data: dict[str, dict[str, Any]] = {}
    section: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            data[section] = {}
            continue
        if section is None or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            parsed: Any = value[1:-1]
        elif value.isdigit():
            parsed = int(value)
        else:
            parsed = value
        data[section][key] = parsed
    return data


def load_project_config(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    file_path = Path(path) if path is not None else config_path()
    if not file_path.is_file():
        return {}
    return parse_simple_toml(file_path.read_text(encoding="utf-8"))
