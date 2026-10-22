"""Compact color digest and Whisper summary injected into the VLM prompt.

Builds context consumed by the VLM stage before Qwen3-VL loads. The text
model (Qwen3-8B) runs and is unloaded first so both fits stay inside a
typical 15 GB GPU budget. Color aggregation is deterministic and does not
use a model.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from open_enricher import storage
from open_enricher.logging_setup import get_logger
from open_enricher.stages.pipeline_spec import load_pipeline_spec
from open_enricher.stages.prompt_load import fill_prompt, load_prompt
from open_enricher.timings import TimingRecorder

logger = get_logger(__name__)

try:
    EXPERIMENT_ID = load_pipeline_spec().experiment_id
except Exception:
    EXPERIMENT_ID = "segmented-5min"
SUMMARY_MODEL_ID = "Qwen/Qwen3-8B"
COLOR_BUCKET_SECONDS = 300
SUMMARY_MAX_NEW_TOKENS = 420
# Opening and closing lines stay intact. Only the middle is thinned, and only
# when the full prompt exceeds the model input budget.
_SAMPLE_EDGE_LINES = 25
_MIDDLE_MARKER = "[... evenly sampled lines from the middle of the episode ...]"

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def dialogue_summary_system_prompt() -> str:
    """Return the system message for the dialogue summary model."""
    return load_prompt("dialogue_summary_system_prompt.md")


def dialogue_summary_user_prompt(transcript: str, *, sampled: bool) -> str:
    """
    Return the user message with the Whisper transcript inserted.

    Args:
        transcript: Formatted dialogue lines, one utterance per row.
        sampled: True when middle lines were dropped to fit the model input.
    """
    note = (
        "Some middle lines were sampled out because the full transcript did not fit the model input."
        if sampled
        else "Every dialogue line is included, in order."
    )
    return fill_prompt(
        load_prompt("dialogue_summary_user_prompt.md"),
        {"TRANSCRIPT": transcript, "SAMPLING_NOTE": note},
    )


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


def format_dialogue_lines(lines: Any) -> list[str]:
    """
    Format Whisper utterances as one row per spoken line.

    Args:
        lines: Audio-stage JSON list of ``start_timestamp``, ``speaker``, ``text``.

    Returns:
        Rows shaped as ``[HH:MM:SS] SPEAKER: text``.

    Raises:
        ValueError: When the payload has no usable dialogue text.
    """
    if not isinstance(lines, list) or not lines:
        raise ValueError("Audio metadata must be a non-empty list of dialogue lines")

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
    return formatted


def fit_dialogue_lines(
    lines: list[str],
    *,
    input_budget: int,
    prompt_tokens: Callable[[str, bool], int],
) -> tuple[str, bool]:
    """
    Keep every dialogue line that fits in one model input.

    Sampling runs only when ``prompt_tokens`` of the full transcript exceeds
    ``input_budget``. The opening and closing lines stay; the middle is halved
    until the prompt fits. A warning starting with ``ALERT`` is written when
    any line is dropped.

    Args:
        lines: Formatted dialogue rows from ``format_dialogue_lines``.
        input_budget: Maximum prompt tokens for this inference, already
            excluding the tokens reserved for the generated summary.
        prompt_tokens: ``(transcript, sampled) -> token count`` of the chat
            prompt that would actually be sent.

    Returns:
        The transcript body and whether middle lines were sampled.

    Raises:
        ValueError: When ``lines`` is empty or ``input_budget`` is not positive.
        RuntimeError: When sampling cannot bring the prompt under the budget.
    """
    if not lines:
        raise ValueError("Dialogue lines are empty")
    if input_budget <= 0:
        raise ValueError(f"input_budget must be positive, got {input_budget}")

    full = "\n".join(lines)
    full_tokens = prompt_tokens(full, False)
    if full_tokens <= input_budget:
        logger.info(
            "Whisper dialogue fits the model input "
            "(prompt_tokens=%s input_budget=%s lines=%s); no sampling",
            full_tokens,
            input_budget,
            len(lines),
        )
        return full, False

    head_n = min(_SAMPLE_EDGE_LINES, len(lines) // 4)
    tail_n = head_n
    head = lines[:head_n]
    tail = lines[-tail_n:] if tail_n else []
    middle = lines[head_n : len(lines) - tail_n] if tail_n else lines[head_n:]

    def assemble(kept_head: list[str], kept_middle: list[str], kept_tail: list[str]) -> str:
        parts = list(kept_head)
        parts.append(_MIDDLE_MARKER)
        parts.extend(kept_middle)
        parts.append("[... ]")
        parts.extend(kept_tail)
        return "\n".join(parts)

    kept_head = head
    kept_middle = list(middle)
    kept_tail = tail
    seen: set[tuple[int, int, int]] = set()
    sampled_tokens = full_tokens
    while True:
        body = assemble(kept_head, kept_middle, kept_tail)
        sampled_tokens = prompt_tokens(body, True)
        if sampled_tokens <= input_budget:
            kept = len(kept_head) + len(kept_middle) + len(kept_tail)
            logger.warning(
                "ALERT: Whisper dialogue does not fit the model input; sampling lines. "
                "prompt_tokens=%s input_budget=%s lines=%s kept_lines=%s "
                "sampled_prompt_tokens=%s",
                full_tokens,
                input_budget,
                len(lines),
                kept,
                sampled_tokens,
            )
            return body, True

        state = (len(kept_head), len(kept_middle), len(kept_tail))
        if state in seen:
            break
        seen.add(state)
        if len(kept_middle) > 1:
            kept_middle = kept_middle[::2]
            continue
        if len(kept_middle) == 1:
            kept_middle = []
            continue
        if len(kept_head) > 1:
            # Keep the opening; drop lines nearest the middle.
            kept_head = kept_head[: max(1, len(kept_head) // 2)]
            continue
        if len(kept_tail) > 1:
            # Keep the ending; drop lines nearest the middle.
            kept_tail = kept_tail[max(1, len(kept_tail) // 2) :]
            continue
        break

    logger.warning(
        "ALERT: Whisper dialogue does not fit the model input after sampling. "
        "prompt_tokens=%s input_budget=%s lines=%s sampled_prompt_tokens=%s",
        full_tokens,
        input_budget,
        len(lines),
        sampled_tokens,
    )
    raise RuntimeError(
        "Whisper dialogue still exceeds the model input after sampling "
        f"(prompt_tokens={full_tokens} sampled_prompt_tokens={sampled_tokens} "
        f"input_budget={input_budget} lines={len(lines)})"
    )


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _count_prompt_tokens(tokenizer: Any, transcript: str, sampled: bool) -> int:
    """Token count of the chat prompt that ``summarize_transcript`` would send."""
    messages = [
        {"role": "system", "content": dialogue_summary_system_prompt()},
        {"role": "user", "content": dialogue_summary_user_prompt(transcript, sampled=sampled)},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer([prompt_text], return_tensors="pt")
    return int(encoded.input_ids.shape[-1])


def summarize_transcript(
    dialogue_lines: list[str],
    *,
    timing: TimingRecorder | None = None,
) -> str:
    """
    Summarize dialogue with the local 8B text model, then free the GPU.

    Every formatted line is sent when the chat prompt fits in
    ``max_position_embeddings - SUMMARY_MAX_NEW_TOKENS``. Otherwise middle
    lines are sampled and an ``ALERT`` warning is logged.

    Args:
        dialogue_lines: Rows from ``format_dialogue_lines``.
        timing: Optional recorder for load and generate spans.

    Returns:
        Plain-prose summary of the dialogue.
    """
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not dialogue_lines:
        raise ValueError("Transcript passed to the summary model is empty")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading {SUMMARY_MODEL_ID} to summarize the Whisper transcript...")
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
        max_position = int(getattr(model.config, "max_position_embeddings", 0) or 0)
        input_budget = max_position - SUMMARY_MAX_NEW_TOKENS
        if input_budget < 1:
            raise RuntimeError(
                f"{SUMMARY_MODEL_ID} max_position_embeddings={max_position} "
                f"does not leave room for {SUMMARY_MAX_NEW_TOKENS} summary tokens"
            )
        transcript, _sampled = fit_dialogue_lines(
            dialogue_lines,
            input_budget=input_budget,
            prompt_tokens=lambda body, sampled: _count_prompt_tokens(
                tokenizer, body, sampled
            ),
        )
        messages = [
            {"role": "system", "content": dialogue_summary_system_prompt()},
            {
                "role": "user",
                "content": dialogue_summary_user_prompt(transcript, sampled=_sampled),
            },
        ]
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
        logger.info("Unloading transcript summary model and freeing VRAM...")
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
    color_path = storage.get_output_dir("color") / f"{video_filename}_color.json"
    audio_path = storage.get_output_dir("audio") / f"{video_filename}_audio.json"
    logger.info(f"Building VLM context from {color_path.name} and {audio_path.name}")

    colors = color_digest(_load_json(color_path))
    dialogue_lines = format_dialogue_lines(_load_json(audio_path))
    logger.info("Formatted %s Whisper dialogue lines", len(dialogue_lines))
    summary = summarize_transcript(dialogue_lines, timing=timing)

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
