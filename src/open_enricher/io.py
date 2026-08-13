"""Save and load JSON metadata and numpy embedding artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_json(path: str | Path, payload: Any) -> Path:
    """Write a JSON file, creating parent directories as needed."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return out


def load_json(path: str | Path) -> Any:
    """Read a JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_embedding(
    path: str | Path,
    array: np.ndarray,
    sidecar: dict[str, Any] | None = None,
) -> Path:
    """Write a `.npy` embedding and an optional JSON sidecar next to it."""

    out = Path(path)
    if out.suffix.lower() != ".npy":
        out = out.with_suffix(".npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, array)
    if sidecar is not None:
        save_json(out.with_suffix(".json"), sidecar)
    return out


def load_embedding(path: str | Path) -> np.ndarray:
    """Load a `.npy` embedding array."""

    return np.load(Path(path))
