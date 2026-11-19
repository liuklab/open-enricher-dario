"""Frozen EVA-CLIP and BLIP-2 Q-Former tokens.

LAVIS is not imported. The checkpoints are opened in a child process.
When the forward pass is not available the parent keeps an empty bank
and the history budget returns to the current clip.
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


def encode_video(video: Path, spec: PipelineSpec, *, sample_fps: float) -> dict[str, Any]:
    """Encode candidate frames. The parent does not import the encoders."""
    eva = eva_path(spec)
    qformer = qformer_path(spec)
    if not eva.is_file() or not qformer.is_file():
        return {
            "status": "fallback",
            "reason": "EVA-CLIP or Q-Former checkpoint is missing",
            "frames": [],
        }
    return call_in_child(
        "open_enricher.adapters.visual_encode",
        "run_encode_payload",
        {
            "video": str(video),
            "eva": str(eva),
            "qformer": str(qformer),
            "sample_fps": sample_fps,
        },
    )


def run_encode_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Child entry.

    The released files are checked and their top-level tensor names are
    recorded. Building 32 query tokens requires the vendored EVA ViT-G
    and Q-Former forward. Until that forward is bound, the bank stays
    empty and the caller returns the history budget to the current clip.
    """
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - optional dependency
        return {"status": "fallback", "reason": f"torch import failed: {exc}", "frames": []}
    found: dict[str, int] = {}
    try:
        for label, path in ("eva", payload["eva"]), ("qformer", payload["qformer"]):
            blob = torch.load(path, map_location="cpu")
            if isinstance(blob, dict):
                found[label] = len(blob)
            else:
                found[label] = 1
    except Exception as exc:  # noqa: BLE001 - recorded fallback
        return {"status": "fallback", "reason": str(exc), "frames": []}
    return {
        "status": "fallback",
        "reason": "EVA/Q-Former forward is not bound; history budget returns to the current clip",
        "checkpoint_present": True,
        "tensors": found,
        "frames": [],
    }
