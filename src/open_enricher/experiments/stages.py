"""Stage order for experimental variants.

The default CLI list stays ``audio dialogue vlm llm``. When a variant
turns a feature on and the caller asked for that full list, the optional
stages are inserted. An explicit subset is left unchanged.
"""

from __future__ import annotations

from typing import Sequence

from open_enricher.experiments.config import PIPELINE_STAGES
from open_enricher.stages.pipeline_spec import PipelineSpec

OPTIONAL_STAGES: tuple[str, ...] = (
    "timeline",
    "bassl",
    "character_graph",
    "visual_memory",
)

STAGE_CHOICES: tuple[str, ...] = PIPELINE_STAGES + OPTIONAL_STAGES


def stages_for_spec(
    requested: Sequence[str],
    spec: PipelineSpec | None,
) -> list[str]:
    """
    Return the stages to execute for this spec.

    Args:
        requested: Names from the CLI. The default is the four core stages.
        spec: Active pipeline spec. ``None`` keeps ``requested``.

    Returns:
        Stage names in execution order. ``evidence`` is not a model stage;
        the runner publishes it after the list returns.
    """
    requested_list = list(requested)
    if spec is None or not spec.features.any_enabled:
        return requested_list
    if any(name not in PIPELINE_STAGES for name in requested_list):
        return requested_list
    if tuple(requested_list) != PIPELINE_STAGES:
        return requested_list
    ordered = ["timeline", "audio"]
    if spec.features.bassl:
        ordered.append("bassl")
    ordered.append("dialogue")
    if spec.features.character_graph:
        ordered.append("character_graph")
    if spec.features.visual_memory:
        ordered.append("visual_memory")
    ordered.extend(["vlm", "llm"])
    return ordered
