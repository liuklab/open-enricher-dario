"""Experiment pipeline knobs read by VLM and LLM stages.

A JSON file under the package (or the path in OPEN_ENRICHER_PIPELINE_SPEC)
selects segmentation mode, where dialogue and color evidence is injected,
and optional continuity aids such as VLM memory or claim checking. The
local experiment loop points this env var at each variant in sequence.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Routing = Literal["vlm", "llm"]
Segmentation = Literal["interval", "scenes"]

_DEFAULT_PATH = Path(__file__).resolve().parent / "pipeline_spec.json"


@dataclass(frozen=True)
class PipelineSpec:
    """
    Validated experiment knobs for one pipeline run.

    Fields mirror the variant JSON schema under
    ``open_enricher.experiments.variants``.
    """

    experiment_id: str
    title: str
    segmentation: Segmentation
    interval_seconds: int
    scene_min_seconds: int
    scene_max_seconds: int
    scene_max_count: int
    dialogue_color_routing: Routing
    vlm_memory: bool
    dialogue_summary_to_vlm: bool
    rolling_plot: bool
    claim_check: bool

    @property
    def inject_dialogue_color_into_vlm(self) -> bool:
        """True when per-window dialogue and color evidence feed the VLM."""
        return self.dialogue_color_routing == "vlm"

    @property
    def inject_dialogue_color_into_llm(self) -> bool:
        """True when dialogue and color evidence are reserved for the LLM."""
        return self.dialogue_color_routing == "llm"


def _require_bool(name: str, value: Any) -> bool:
    """
    Require a JSON boolean (not 0/1 integers).

    Args:
        name: Field name for error messages.
        value: Raw JSON value.

    Returns:
        The boolean value.

    Raises:
        ValueError: When the value is not a Python bool.
    """
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return value


def _require_positive_int(name: str, value: Any) -> int:
    """
    Require a positive integer field.

    Args:
        name: Field name for error messages.
        value: Raw JSON value (already coerced with ``int(...)`` by the caller
            when needed).

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

    segmentation = data.get("segmentation", "interval")
    if segmentation not in ("interval", "scenes"):
        raise ValueError(f"segmentation must be interval or scenes, got {segmentation!r}")
    routing = data.get("dialogue_color_routing", "vlm")
    if routing not in ("vlm", "llm"):
        raise ValueError(f"dialogue_color_routing must be vlm or llm, got {routing!r}")

    return PipelineSpec(
        experiment_id=str(data.get("experiment_id") or "default").strip(),
        title=str(data.get("title") or "default").strip(),
        segmentation=segmentation,
        interval_seconds=_require_positive_int(
            "interval_seconds", int(data.get("interval_seconds", 300))
        ),
        scene_min_seconds=_require_positive_int(
            "scene_min_seconds", int(data.get("scene_min_seconds", 45))
        ),
        scene_max_seconds=_require_positive_int(
            "scene_max_seconds", int(data.get("scene_max_seconds", 150))
        ),
        scene_max_count=_require_positive_int(
            "scene_max_count", int(data.get("scene_max_count", 24))
        ),
        dialogue_color_routing=routing,
        vlm_memory=_require_bool("vlm_memory", data.get("vlm_memory", False)),
        dialogue_summary_to_vlm=_require_bool(
            "dialogue_summary_to_vlm", data.get("dialogue_summary_to_vlm", False)
        ),
        rolling_plot=_require_bool("rolling_plot", data.get("rolling_plot", False)),
        claim_check=_require_bool("claim_check", data.get("claim_check", False)),
    )
