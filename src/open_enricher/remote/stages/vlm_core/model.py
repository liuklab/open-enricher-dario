"""Load Qwen3-VL in 4-bit and run constrained-memory video inference."""

from __future__ import annotations

import gc
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from src.open_enricher.remote.stages.vlm_core.config import ModelConfig



def load_model(config: ModelConfig) -> tuple[Any, Any]:
    """Load a Qwen3-VL model (with optional 4-bit quantization) and its processor."""
    quantization_config = None
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForImageTextToText.from_pretrained(
        config.model_id,
        quantization_config=quantization_config,
        dtype=torch.float16 if config.dtype == "float16" else config.dtype,
        device_map=config.device_map,
        attn_implementation=config.attn_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True)
    return model, processor


def unload_model(model: Any, processor: Any = None) -> None:
    """Drop loaded VLM objects and free GPU cache."""
    del model
    if processor is not None:
        del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_device(model: Any) -> torch.device:
    """Device used to move processor outputs onto the model."""
    try:
        return model.device
    except AttributeError as exc:
        raise AttributeError(
            f"{type(model).__name__} has no `.device`; expected a Hugging Face PreTrainedModel"
        ) from exc


def apply_long_video_limits(
    processor: Any,
    *,
    min_frames: int,
    max_frames: int,
    max_video_tokens: int,
    shortest_edge: int,
    longest_edge: int,
) -> None:
    """Set the frame cap and pixel budget on the video processor."""
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is None:
        raise RuntimeError("Processor has no video_processor; expected a Qwen3-VL processor")
    if min_frames > max_frames:
        raise ValueError(f"min_frames ({min_frames}) must be <= max_frames ({max_frames})")
    if shortest_edge > longest_edge:
        raise ValueError(
            f"shortest_edge ({shortest_edge}) must be <= longest_edge ({longest_edge})"
        )

    video_processor.min_frames = min_frames
    video_processor.max_frames = max_frames
    video_processor.max_video_tokens = max_video_tokens
    video_processor.size = {"shortest_edge": shortest_edge, "longest_edge": longest_edge}
    video_processor.min_pixels = shortest_edge
    video_processor.max_pixels = longest_edge