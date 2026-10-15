"""Compact color digest and Whisper summary injected into the VLM prompt.

The text model runs and is unloaded before the VLM loads, so both fits stay
inside the T4 budget. Color aggregation is deterministic and does not use a model.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.open_enricher.remote.stages.pipeline_spec import load_pipeline_spec
from src.open_enricher.remote.storage import db
from src.open_enricher.remote.timings import TimingRecorder

try:
    EXPERIMENT_ID = load_pipeline_spec().experiment_id
except Exception:
    EXPERIMENT_ID = "segmented-5min"
SUMMARY_MODEL_ID = "Qwen/Qwen3-8B"
# A full episode transcript does not fit comfortably next to the VLM frames.
# The text model sees a capped sample; the VLM sees only the short summary.
TRANSCRIPT_CHAR_BUDGET = 12000
COLOR_BUCKET_SECONDS = 300
SUMMARY_MAX_NEW_TOKENS = 420

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def _clock(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required enrichment input is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Enrichment input is not valid JSON: {path}") from exc


def color_digest(samples: Any, bucket_seconds: int = COLOR_BUCKET_SECONDS) -> str:
    """One line per time bucket: the three most common hex colors in that window."""
    if not isinstance(samples, list) or not samples:
        raise ValueError("Color metadata must be a non-empty list of frame samples")
    if bucket_seconds <= 0:
        raise ValueError(f"bucket_seconds must be positive, got {bucket_seconds}")

    buckets: dict[int, list[str]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        try:
            timestamp = float(sample.get("timestamp", 0))
        except (TypeError, ValueError):
            continue
        palette = sample.get("color_palette") or []
        if not isinstance(palette, list):
            continue
        bucket = int(timestamp // bucket_seconds)
        colors = [str(color).lower() for color in palette if isinstance(color, str) and color.strip()]
        if colors:
            buckets.setdefault(bucket, []).extend(colors)

    if not buckets:
        raise ValueError("Color metadata contained no usable palettes")

    lines = []
    for bucket in sorted(buckets):
        start = bucket * bucket_seconds
        top = Counter(buckets[bucket]).most_common(3)
        pretty = ", ".join(f"{color} x{count}" for color, count in top)
        lines.append(f"{_clock(start)}-{_clock(start + bucket_seconds)}: {pretty}")
    return "\n".join(lines)


def condense_transcript(lines: Any, max_chars: int = TRANSCRIPT_CHAR_BUDGET) -> str:
    """Format dialogue lines and thin the middle when the transcript is long."""
    if not isinstance(lines, list) or not lines:
        raise ValueError("Audio metadata must be a non-empty list of dialogue lines")
    if max_chars < 500:
        raise ValueError(f"max_chars must be at least 500, got {max_chars}")

    formatted: list[str] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        speaker = str(line.get("speaker") or "UNKNOWN").strip() or "UNKNOWN"
        try:
            start = float(line.get("start_timestamp", 0))
        except (TypeError, ValueError):
            start = 0.0
        formatted.append(f"[{_clock(start)}] {speaker}: {text}")

    if not formatted:
        raise ValueError("Audio metadata contained no dialogue text")

    blob = "\n".join(formatted)
    if len(blob) <= max_chars:
        return blob

    head_n = min(25, len(formatted) // 4)
    tail_n = head_n
    head = formatted[:head_n]
    tail = formatted[-tail_n:]
    middle = formatted[head_n : len(formatted) - tail_n]
    marker_head = "[... evenly sampled lines from the middle of the episode ...]"
    marker_tail = "[... ]"
    reserved = len("\n".join(head + tail)) + len(marker_head) + len(marker_tail) + 4
    budget = max_chars - reserved
    if budget < 400 or not middle:
        clipped = "\n".join(head + [marker_head] + tail)
        return clipped[:max_chars]

    sampled = list(middle)
    while len(sampled) > 1 and len("\n".join(sampled)) > budget:
        sampled = sampled[::2]
    return "\n".join(head + [marker_head] + sampled + [marker_tail] + tail)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def summarize_transcript(
    transcript: str,
    *,
    timing: TimingRecorder | None = None,
) -> str:
    """Summarize dialogue with the local 8B text model, then free the GPU."""
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not transcript.strip():
        raise ValueError("Transcript passed to the summary model is empty")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    system_prompt = (
        "You summarize television dialogue for a cataloguer who will also see sparse video frames. "
        "Write plain prose. Do not use JSON, lists of timestamps, or markdown."
    )
    user_prompt = (
        "The lines below are a Whisper transcript. Some middle lines may have been sampled out.\n\n"
        f"{transcript}\n\n"
        "Write 150 to 220 words. Cover the opening, the middle, and the ending. "
        "Name people only when a speaker or the dialogue names them. "
        "Mention places, conflicts, and what the characters are trying to do. "
        "Do not invent events that the transcript does not support."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    print(f"[INFO] Loading {SUMMARY_MODEL_ID} to summarize the Whisper transcript...")
    quantization_config = None
    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = None
    tokenizer = None
    try:
        with (timing.span("summary_load", model_id=SUMMARY_MODEL_ID) if timing else _nullcontext()):
            tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL_ID, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                SUMMARY_MODEL_ID,
                quantization_config=quantization_config,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = tokenizer([prompt_text], return_tensors="pt").to(device)
        with (timing.span("summary_generate", model_id=SUMMARY_MODEL_ID) if timing else _nullcontext()):
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=SUMMARY_MAX_NEW_TOKENS,
                    do_sample=False,
                    repetition_penalty=1.05,
                )
        new_tokens = generated_ids[0][len(model_inputs.input_ids[0]) :]
        summary = _strip_think(tokenizer.decode(new_tokens, skip_special_tokens=True))
    finally:
        print("[INFO] Unloading transcript summary model and freeing VRAM...")
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(summary) < 40:
        raise RuntimeError(f"Transcript summary is too short to be useful: {summary!r}")
    return summary


def _nullcontext():
    from contextlib import nullcontext

    return nullcontext()


def build_vlm_context(
    video_filename: str,
    *,
    timing: TimingRecorder | None = None,
) -> tuple[str, str]:
    """Return (evidence block for non-segmented path, dialogue summary text)."""
    color_path = db.get_output_dir("color") / f"{video_filename}_color.json"
    audio_path = db.get_output_dir("audio") / f"{video_filename}_audio.json"
    print(f"[INFO] Building VLM context from {color_path.name} and {audio_path.name}")

    colors = color_digest(_load_json(color_path))
    transcript = condense_transcript(_load_json(audio_path))
    print(f"[INFO] Condensed transcript is {len(transcript)} characters")
    summary = summarize_transcript(transcript, timing=timing)

    context = (
        "Prior evidence extracted from this same file, before these frames:\n"
        "Color samples are k-means hex palettes from frames spread across the file. "
        "Use them in lighting_and_color and visual_style when they agree with what you see.\n"
        "The dialogue summary condenses the Whisper transcript. "
        "Use it for topics, spoken names, and how the story moves. "
        "Do not add dialogue lines that the summary does not support. "
        "If a frame contradicts the summary, trust the frame and say so in evidence_notes.\n\n"
        f"COLOR SAMPLES (about every {COLOR_BUCKET_SECONDS // 60} minutes):\n{colors}\n\n"
        f"DIALOGUE SUMMARY:\n{summary}"
    )
    return context, summary
