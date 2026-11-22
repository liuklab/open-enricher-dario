"""Load Qwen3-VL in 4-bit and run constrained-memory video inference."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from open_enricher.logging_setup import get_logger
from open_enricher.stages.vlm_core.config import MetadataConfig, ModelConfig
from open_enricher.stages.vlm_core.video import video_input_path

logger = get_logger(__name__)

_VIDEO_TENSOR_KEYS = ("pixel_values_videos", "pixel_values")


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


def prepare_video_inputs(
    processor: Any,
    video_path: str | Path,
    text: str,
    *,
    fps: float,
    max_frames: int,
    shortest_edge: int,
    longest_edge: int,
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
                {
                    "type": "video",
                    "video": str(path),
                    "fps": fps,
                    "max_frames": max_frames,
                    "min_pixels": shortest_edge,
                    "max_pixels": longest_edge,
                },
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

    total_tokens = inputs["input_ids"].shape[1]
    logger.info(f"VLM prepared input sequence length: {total_tokens} tokens")
    return inputs.to(device)


def extract_metadata(
    model: Any,
    processor: Any,
    video_path: str | Path,
    config: MetadataConfig,
) -> str:
    """Return the decoded generation for one video. Raises if nothing was generated."""
    apply_long_video_limits(
        processor,
        min_frames=config.min_frames,
        max_frames=config.max_frames,
        max_video_tokens=config.max_video_tokens,
        shortest_edge=config.video_shortest_edge,
        longest_edge=config.video_longest_edge,
    )

    with video_input_path(video_path, config.video_seconds) as model_video:
        inputs = prepare_video_inputs(
            processor,
            model_video,
            config.prompt,
            fps=config.fps,
            max_frames=config.max_frames,
            shortest_edge=config.video_shortest_edge,
            longest_edge=config.video_longest_edge,
            device=model_device(model),
            add_generation_prompt=True,
        )

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": config.max_new_tokens,
                "do_sample": config.do_sample,
                "repetition_penalty": config.repetition_penalty,
            }
            if config.do_sample:
                gen_kwargs["temperature"] = config.temperature
                gen_kwargs["top_p"] = config.top_p
                gen_kwargs["top_k"] = config.top_k

            with torch.inference_mode():
                generated = model.generate(**inputs, **gen_kwargs)

            prompt_len = inputs.input_ids.shape[1]
            if generated.shape[0] != 1:
                raise RuntimeError(f"Expected batch size 1 from generate, got {tuple(generated.shape)}")
            if generated.shape[1] <= prompt_len:
                raise RuntimeError("Model produced no new tokens")

            text = processor.batch_decode(
                generated[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        finally:
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not text.strip():
        raise RuntimeError("Decoded generation is empty")
    return text


def extract_metadata_from_images(
    model: Any,
    processor: Any,
    image_paths: list[str | Path],
    config: MetadataConfig,
) -> str:
    """Run one generation on an explicit list of images instead of a clip."""
    paths = [Path(path) for path in image_paths]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Explicit frame is missing: {missing[0]}")
    if not config.prompt.strip():
        raise ValueError("Prompt text is empty")
    content: list[dict[str, str]] = [
        {"type": "image", "image": str(path)} for path in paths
    ]
    content.append({"type": "text", "text": config.prompt})
    messages = [{"role": "user", "content": content}]
    image_processor = processor.image_processor
    token_edge = image_processor.patch_size * image_processor.merge_size
    pixel_limit = config.max_video_tokens * token_edge**2
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "images_kwargs": {
                "size": {
                    "shortest_edge": min(config.video_shortest_edge, pixel_limit),
                    "longest_edge": pixel_limit,
                }
            }
        },
    )
    if "input_ids" not in inputs:
        raise RuntimeError(f"Processor did not return input_ids. Keys: {list(inputs.keys())}")
    inputs = inputs.to(model_device(model))
    try:
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "repetition_penalty": config.repetition_penalty,
        }
        if config.do_sample:
            gen_kwargs["temperature"] = config.temperature
            gen_kwargs["top_p"] = config.top_p
            gen_kwargs["top_k"] = config.top_k
        with torch.inference_mode():
            generated = model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs.input_ids.shape[1]
        if generated.shape[1] <= prompt_len:
            raise RuntimeError("Model produced no new tokens")
        text = processor.batch_decode(
            generated[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    finally:
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not text.strip():
        raise RuntimeError("Decoded generation is empty")
    return text
