"""Textual metadata extraction via prompted VLM generation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

from open_enricher.config import MetadataConfig
from open_enricher.model import model_device, prepare_video_inputs


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_from_text(text: str) -> dict[str, Any]:
    """Best-effort JSON object parse. Falls back to `{"text": ...}`."""

    stripped = _FENCE_RE.sub("", text.strip()).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(stripped[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {"text": text}


def extract_metadata(
    model: Any,
    processor: Any,
    video_path: str | Path,
    config: MetadataConfig | None = None,
    *,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Generate structured (or raw) textual metadata for one video."""

    config = config or MetadataConfig()
    inputs = prepare_video_inputs(
        processor,
        video_path,
        config.prompt,
        num_frames=config.num_frames,
        fps=config.fps,
        device=model_device(model),
    )

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=config.max_new_tokens, do_sample=False)

    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
    raw_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    payload: dict[str, Any] = {
        "video": str(video_path),
        "prompt": config.prompt,
        "metadata": parse_json_from_text(raw_text),
        "raw_text": raw_text,
    }
    if model_id:
        payload["model_id"] = model_id
    return payload
