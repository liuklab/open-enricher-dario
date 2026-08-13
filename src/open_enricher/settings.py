"""Load committed project settings from config.toml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Maps config.toml paths to the env names used on the Colab kernel.
_ENV_MAP = {
    ("git", "url"): "OPEN_ENRICHER_GIT_URL",
    ("git", "ref"): "OPEN_ENRICHER_GIT_REF",
    ("git", "repo_dir"): "OPEN_ENRICHER_REPO",
    ("drive", "data"): "OPEN_ENRICHER_DATA",
    ("drive", "output"): "OPEN_ENRICHER_OUTPUT",
    ("drive", "video"): "OPEN_ENRICHER_VIDEO",
    ("model", "id"): "OPEN_ENRICHER_MODEL",
    ("model", "num_frames"): "OPEN_ENRICHER_NUM_FRAMES",
    ("model", "max_new_tokens"): "OPEN_ENRICHER_MAX_NEW_TOKENS",
    ("colab", "session"): "COLAB_SESSION",
    ("colab", "gpu"): "COLAB_GPU",
    ("colab", "timeout"): "COLAB_TIMEOUT",
    ("colab", "install_timeout"): "COLAB_INSTALL_TIMEOUT",
    ("colab", "auth"): "COLAB_AUTH",
}


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


def config_environ(path: str | Path | None = None) -> dict[str, str]:
    """Flatten config.toml into OPEN_ENRICHER_* / COLAB_* environment keys."""

    loaded = load_project_config(path)
    out: dict[str, str] = {}
    for (section, key), env_name in _ENV_MAP.items():
        if section not in loaded or key not in loaded[section]:
            continue
        value = loaded[section][key]
        if value is None:
            continue
        text = str(value)
        if text == "":
            continue
        out[env_name] = text
    return out


def apply_config_to_environ() -> None:
    """Fill missing os.environ keys from config.toml. Does not override existing values."""

    import os

    for key, value in config_environ().items():
        if key not in os.environ or os.environ[key].strip() == "":
            os.environ[key] = value
