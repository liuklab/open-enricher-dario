"""Frozen EVA-CLIP and BLIP-2 Q-Former tokens.

The worker loads the pinned LAVIS encoders and local checkpoints.
A missing weight or a key mismatch keeps an empty bank, and the history
budget returns to the current clip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from open_enricher import storage
from open_enricher.adapters.isolate import call_in_child
from open_enricher.stages.pipeline_spec import PipelineSpec


def eva_path(spec: PipelineSpec) -> Path:
    if spec.checkpoints.eva_clip:
        return Path(spec.checkpoints.eva_clip)
    return storage.models_cache() / "external" / "lavis" / "eva_vit_g.pth"


def qformer_path(spec: PipelineSpec) -> Path:
    if spec.checkpoints.qformer:
        return Path(spec.checkpoints.qformer)
    return storage.models_cache() / "external" / "lavis" / "blip2_pretrained_flant5xxl.pth"


def encode_video(
    video: Path,
    spec: PipelineSpec,
    *,
    sample_fps: float,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Encode candidate frames. The parent does not import the encoders.

    ``max_frames`` caps how many sampled frames are encoded. ``None`` reads
    the whole file at ``sample_fps``. The pipeline leaves it unset.
    """
    eva = eva_path(spec)
    qformer = qformer_path(spec)
    if not eva.is_file() or not qformer.is_file():
        return {
            "status": "fallback",
            "reason": "EVA-CLIP or Q-Former checkpoint is missing",
            "frames": [],
        }
    payload: dict[str, Any] = {
        "video": str(video),
        "eva": str(eva),
        "qformer": str(qformer),
        "sample_fps": sample_fps,
    }
    if max_frames is not None:
        payload["max_frames"] = max_frames
    return call_in_child(
        "open_enricher.adapters.visual_encode",
        "run_encode_payload",
        payload,
    )


def run_encode_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Child entry. Loads ViT-G and the Q-Former and returns query tokens."""
    try:
        import torch as _torch
    except Exception as exc:  # noqa: BLE001 - optional dependency
        return {"status": "fallback", "reason": f"torch import failed: {exc}", "frames": []}
    del _torch
    try:
        from open_enricher.adapters.eva_qformer import encode_payload
    except Exception as exc:  # noqa: BLE001 - recorded fallback
        return {"status": "fallback", "reason": str(exc), "frames": []}
    return encode_payload(payload)
