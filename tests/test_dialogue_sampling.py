"""Dialogue sampling stays off unless the summary prompt exceeds the model input."""

from __future__ import annotations

import logging

import pytest

from open_enricher.stages.vlm_core.enrichment import fit_dialogue_lines, format_dialogue_lines


def _tokens(body: str, sampled: bool) -> int:
    """One token per character, plus a small surcharge for the sampling note."""
    return len(body) + (10 if sampled else 0)


def test_format_dialogue_lines_keeps_every_utterance() -> None:
    rows = format_dialogue_lines(
        [
            {"start_timestamp": 1.2, "speaker": "UNKNOWN", "text": "Hello"},
            {"start_timestamp": 3, "speaker": "", "text": "  Again  "},
        ]
    )
    assert rows == ["[00:00:01] UNKNOWN: Hello", "[00:00:03] UNKNOWN: Again"]


def test_full_transcript_is_kept_when_it_fits(caplog: pytest.LogCaptureFixture) -> None:
    lines = [f"line {index}" for index in range(8)]
    with caplog.at_level(logging.INFO):
        body, sampled = fit_dialogue_lines(
            lines,
            input_budget=10_000,
            prompt_tokens=_tokens,
        )
    assert sampled is False
    assert body == "\n".join(lines)
    assert "ALERT" not in caplog.text
    assert "no sampling" in caplog.text


def test_samples_and_alerts_only_when_over_input_budget(
    caplog: pytest.LogCaptureFixture,
) -> None:
    lines = [f"utterance {index:03d} " + ("word " * 12) for index in range(80)]
    budget = 400
    with caplog.at_level(logging.WARNING):
        body, sampled = fit_dialogue_lines(
            lines,
            input_budget=budget,
            prompt_tokens=_tokens,
        )
    assert sampled is True
    assert _tokens(body, True) <= budget
    assert lines[0] in body
    assert lines[-1] in body
    assert "evenly sampled lines" in body
    assert "ALERT: Whisper dialogue does not fit the model input" in caplog.text


def test_alert_when_sampling_cannot_fit(caplog: pytest.LogCaptureFixture) -> None:
    lines = ["one long line"]
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="still exceeds the model input"):
            fit_dialogue_lines(lines, input_budget=5, prompt_tokens=lambda _body, _sampled: 50)
    assert "ALERT: Whisper dialogue does not fit the model input after sampling" in caplog.text
