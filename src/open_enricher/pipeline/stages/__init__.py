"""Stage registry. Every name in STAGE_ORDER must have an entry here."""

from __future__ import annotations

from open_enricher.pipeline.order import STAGE_ORDER
from open_enricher.pipeline.stages.audio import AudioStage
from open_enricher.pipeline.stages.color import ColorStage
from open_enricher.pipeline.stages.metadata import MetadataStage

STAGES = {
    AudioStage.name: AudioStage(),
    ColorStage.name: ColorStage(),
    MetadataStage.name: MetadataStage(),
}


def require_registered_stages() -> None:
    """Raise if STAGE_ORDER names a stage that is not registered."""

    missing = [name for name in STAGE_ORDER if name not in STAGES]
    if missing:
        raise RuntimeError(f"STAGE_ORDER names unregistered stages: {missing}")
