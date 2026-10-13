"""Textual metadata stage: Qwen3-VL generation via the existing library."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from open_enricher.config import METADATA_CONFIG, MODEL_CONFIG
from open_enricher.metadata import extract_metadata
from open_enricher.model import load_model, unload_model


class MetadataStage:
    """Load the configured VLM once per process and return generated text."""

    name = "metadata"

    def __init__(self) -> None:
        self._model: Any | None = None
        self._processor: Any | None = None

    def _ensure_model(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            self._model, self._processor = load_model(MODEL_CONFIG)
        return self._model, self._processor

    def run(self, video_path: Path) -> dict[str, Any]:
        """Return the decoded generation for one resolved video file."""

        model, processor = self._ensure_model()
        text = extract_metadata(model, processor, video_path, METADATA_CONFIG)
        return {
            "video": video_path.name,
            "model_id": MODEL_CONFIG.model_id,
            "text": text,
        }

    def unload(self) -> None:
        """Drop a previously loaded model so GPU memory can be reclaimed."""

        if self._model is not None:
            unload_model(self._model)
        self._model = None
        self._processor = None
