"""Experiment pipeline knobs read by VLM and LLM stages.

A JSON file under the package (or OPEN_ENRICHER_PIPELINE_SPEC) selects
segmentation, where dialogue/color evidence is injected, and optional
continuity aids. Defaults match the current segmented-5min worker.
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
        return self.dialogue_color_routing == "vlm"

    @property
    def inject_dialogue_color_into_llm(self) -> bool:
        return self.dialogue_color_routing == "llm"


def _require_bool(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return value


def _require_positive_int(name: str, value: Any) -> int:
    if type(value) is not int or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def load_pipeline_spec(path: Path | None = None) -> PipelineSpec:
    """Load and validate the active experiment pipeline specification."""
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
