"""Public API for open-enricher video understanding experiments.

Heavy modules (torch / transformers) load only when their symbols are accessed,
so the laptop orchestrator can import pipeline order and the HTTP client alone.
"""

from __future__ import annotations

from typing import Any

from open_enricher.config import (
    DRIVE_CONFIG,
    METADATA_CONFIG,
    METADATA_PARAMS_PATH,
    MODEL_CONFIG,
    DriveConfig,
    MetadataConfig,
    ModelConfig,
    load_metadata_params,
    validate_metadata_params,
)
from open_enricher.io import build_output_tarball, load_json, load_text, save_json, save_text
from open_enricher.video import list_videos, resolve_video

__all__ = [
    "DRIVE_CONFIG",
    "METADATA_CONFIG",
    "METADATA_PARAMS_PATH",
    "MODEL_CONFIG",
    "DriveConfig",
    "MetadataConfig",
    "ModelConfig",
    "extract_metadata",
    "list_videos",
    "load_json",
    "load_metadata_params",
    "load_model",
    "load_text",
    "resolve_video",
    "save_json",
    "save_text",
    "unload_model",
    "build_output_tarball",
    "validate_metadata_params",
]

_LAZY_ATTRS = {
    "extract_metadata": ("open_enricher.metadata", "extract_metadata"),
    "load_model": ("open_enricher.model", "load_model"),
    "unload_model": ("open_enricher.model", "unload_model"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    import importlib

    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value
