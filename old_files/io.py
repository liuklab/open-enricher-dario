"""Save and load text and JSON artifacts."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any


def build_output_tarball(src: Path, dest: Path) -> Path:
    """Pack an output directory into a gzip tarball at dest."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as archive:
        archive.add(src, arcname=src.name)
    return dest


def save_text(path: str | Path, text: str) -> Path:
    """Write a text file, creating parent directories as needed."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def load_text(path: str | Path) -> str:
    """Read a UTF-8 text file."""

    return Path(path).read_text(encoding="utf-8")


def save_json(path: str | Path, payload: Any) -> Path:
    """Write a JSON file, creating parent directories as needed."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return out


def load_json(path: str | Path) -> Any:
    """Read a JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))
