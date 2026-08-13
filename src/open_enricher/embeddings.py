"""Experimental video embeddings via mean-pooled last hidden states."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from open_enricher.config import EmbeddingConfig
from open_enricher.model import model_device, prepare_video_inputs


def extract_embedding(
    model: Any,
    processor: Any,
    video_path: str | Path,
    config: EmbeddingConfig | None = None,
) -> np.ndarray:
    """Return a 1-D vector for one video.

    This is an experimental pooled representation from the loaded VLM, not a
    dedicated embedding checkpoint. Mean-pooling uses the last hidden state
    and the attention mask when it is available.
    """

    config = config or EmbeddingConfig()
    inputs = prepare_video_inputs(
        processor,
        video_path,
        config.prompt,
        num_frames=config.num_frames,
        fps=config.fps,
        device=model_device(model),
        add_generation_prompt=False,
    )

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[-1]
    mask = inputs.get("attention_mask")
    if mask is not None:
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
    else:
        pooled = hidden.mean(dim=1)

    vector = pooled.squeeze(0).detach().float().cpu().numpy()
    return np.asarray(vector, dtype=np.float32)
