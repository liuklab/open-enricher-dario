"""Format and fit timestamped ASR evidence in dialogue model prompts."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from open_enricher.logging_setup import get_logger

logger = get_logger(__name__)
SUMMARY_MODEL_ID = "Qwen/Qwen3-8B"
_SAMPLE_EDGE_LINES = 25
_MIDDLE_MARKER = "[... evenly sampled lines from the middle of the episode ...]"
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def _clock(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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
