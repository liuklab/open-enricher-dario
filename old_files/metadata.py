"""Text generation from a prompted vision-language model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from old.config import MetadataConfig
from old.model import apply_long_video_limits, model_device, prepare_video_inputs
from old.video import video_input_path


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
    # The processor samples the whole file it is given. A prefix clip is a
    # separate file so the unread tail never enters the frame index.
    with video_input_path(video_path, config.video_seconds) as model_video:
        inputs = prepare_video_inputs(
            processor,
            model_video,
            config.prompt,
            fps=config.fps,
            device=model_device(model),
            add_generation_prompt=True,
        )

        with torch.inference_mode():
            # generate() returns prompt tokens concatenated with the continuation.
            generated = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
            )

    prompt_len = inputs.input_ids.shape[1]
    if generated.shape[0] != 1:
        raise RuntimeError(f"Expected batch size 1 from generate, got {tuple(generated.shape)}")
    if generated.shape[1] <= prompt_len:
        raise RuntimeError("Model produced no new tokens")

    text = processor.batch_decode(
        generated[:, prompt_len:],
        skip_special_tokens=True,  # drop eos / chat markers; leave whitespace as decoded
        clean_up_tokenization_spaces=False,
    )[0]
    if not text.strip():
        raise RuntimeError("Decoded generation is empty")
    return text
