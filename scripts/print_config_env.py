"""Print config.toml as KEY=value lines for local bootstrap scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / "src" / "open_enricher" / "settings.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("open_enricher_settings", SETTINGS)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot load {SETTINGS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key, value in module.config_environ().items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
