"""Load vision-language models and turn a local video into model inputs."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from old_files.config import ModelConfig

# Qwen2/3-VL may pack video under either name depending on Transformers version.
_VIDEO_TENSOR_KEYS = ("pixel_values_videos", "pixel_values")


def load_model(config: ModelConfig) -> tuple[Any, Any]:
    """Load a Qwen3-VL-compatible model and its processor."""

    # ImageTextToText covers both dense and MoE VL checkpoints with one call site.
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_id,
        dtype=config.dtype,
        device_map=config.device_map,
        attn_implementation=config.attn_implementation,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(config.model_id)
    return model, processor


def unload_model(model: Any) -> None:
    """Drop a loaded model and free GPU cache when possible."""

    del model
    gc.collect()
    # `del` drops the Python ref; CUDA memory stays reserved until the caching allocator is cleared.
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
    """Set the frame cap and pixel budget on the video processor.

    sample_frames reads max_frames and min_frames from the processor instance.
    A max_frames value passed only through processor_kwargs is ignored.
    """

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


def prepare_video_inputs(
    processor: Any,
    video_path: str | Path,
    text: str,
    *,
    fps: float,
    device: torch.device,
    add_generation_prompt: bool,
) -> Any:
    """Turn a local video path and a text prompt into model-ready tensors."""

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {path}")
    if not text.strip():
        raise ValueError("Prompt text is empty")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(path)},
                {"type": "text", "text": text},
            ],
        }
    ]
    # fps must live in processor_kwargs; a top-level fps= is ignored by Transformers.
    # add_generation_prompt appends the assistant prefix (needed for generate, not for pooling).
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"fps": fps, "cap_pixels_per_frame": True},
    )
    if "input_ids" not in inputs:
        raise RuntimeError(f"Processor did not return input_ids. Keys: {list(inputs.keys())}")
    if inputs["input_ids"].shape[0] != 1:
        raise RuntimeError(f"Expected batch size 1, got {tuple(inputs['input_ids'].shape)}")
    if not any(key in inputs and inputs[key] is not None for key in _VIDEO_TENSOR_KEYS):
        raise RuntimeError(
            "Processor produced no video tensors "
            f"(looked for {_VIDEO_TENSOR_KEYS}). Keys: {list(inputs.keys())}"
        )

    return inputs.to(device)  # token ids stay on CPU otherwise; generate/forward would fail or copy later.
