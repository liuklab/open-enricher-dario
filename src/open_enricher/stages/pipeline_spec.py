"""Experiment pipeline knobs for the single local metadata run.

A JSON file under the package (or the path in OPEN_ENRICHER_PIPELINE_SPEC)
sets the experiment id and the fixed interval used as a scene. The local
experiment loop points this env var at the packaged variant.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_PATH = Path(__file__).resolve().parent / "pipeline_spec.json"


@dataclass(frozen=True)
class PipelineSpec:
    """
    Validated knobs for the scene-context pipeline.

    Fields mirror ``experiments/variants/01-scene-context.json``.
    """

    experiment_id: str
    title: str
    interval_seconds: int


def _require_positive_int(name: str, value: Any) -> int:
    """
    Require a positive integer field.

    Args:
        name: Field name for error messages.
        value: Raw JSON value.

    Returns:
        The validated positive int.

    Raises:
        ValueError: When the value is not a positive int.
    """
    if type(value) is not int or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def load_pipeline_spec(path: Path | None = None) -> PipelineSpec:
    """
    Load and validate the active experiment pipeline specification.

    Preference order for the file path:
    1. ``OPEN_ENRICHER_PIPELINE_SPEC`` when set
    2. Explicit ``path`` argument
    3. Packaged ``pipeline_spec.json`` next to this module

    Args:
        path: Optional explicit path used when the env override is empty.

    Returns:
        Validated ``PipelineSpec`` dataclass.

    Raises:
        FileNotFoundError: When the resolved path does not exist.
        ValueError: When the JSON shape or field types are invalid.
    """
    override = os.environ.get("OPEN_ENRICHER_PIPELINE_SPEC", "").strip()
    spec_path = Path(override) if override else (path or _DEFAULT_PATH)
    if not spec_path.is_file():
        raise FileNotFoundError(f"Pipeline spec not found: {spec_path}")
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Pipeline spec must be a JSON object")

    return PipelineSpec(
        experiment_id=str(data.get("experiment_id") or "default").strip(),
        title=str(data.get("title") or "default").strip(),
        interval_seconds=_require_positive_int(
            "interval_seconds", int(data.get("interval_seconds", 300))
        ),
    )
