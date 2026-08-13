"""Load and unload vision-language models, plus shared video input prep."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from open_enricher.config import ModelConfig, sampling_kwargs


def load_model(config: ModelConfig | None = None) -> tuple[Any, Any]:
    """Load a Qwen3-VL-compatible model and its processor.

    Uses AutoModelForImageTextToText so dense and MoE checkpoints can share
    the same call site. Switch models by changing `config.model_id`.
    """

    config = config or ModelConfig()
    load_kwargs: dict[str, Any] = {
        "dtype": config.dtype,
        "device_map": config.device_map,
    }
    if config.attn_implementation:
        load_kwargs["attn_implementation"] = config.attn_implementation

    model = AutoModelForImageTextToText.from_pretrained(config.model_id, **load_kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(config.model_id)
    return model, processor


def unload_model(model: Any | None = None) -> None:
    """Drop a loaded model and free GPU cache when possible."""

    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_device(model: Any) -> torch.device:
    """Best-effort device for moving processor outputs onto the model."""

    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def prepare_video_inputs(
    processor: Any,
    video_path: str | Path,
    text: str,
    *,
    num_frames: int | None = None,
    fps: float | None = None,
    device: torch.device | str | None = None,
    add_generation_prompt: bool = True,
) -> Any:
    """Turn a local video path and a text prompt into model-ready tensors."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": text},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
        **sampling_kwargs(num_frames, fps),
    )
    if device is not None:
        inputs = inputs.to(device)
    return inputs
