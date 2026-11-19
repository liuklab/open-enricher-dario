"""Narrative cues used when adaptive frame allocation is requested."""

from __future__ import annotations

import json
import re
from typing import Any

_CUES = (
    "decision",
    "decide",
    "reveal",
    "revelation",
    "realize",
    "reversal",
    "secret",
    "confess",
)


def cue_from_lines(lines: list[dict[str, Any]]) -> tuple[float, list[str]]:
    """Score 1 when a line contains a decision, revelation or reversal cue."""
    hits: list[str] = []
    for index, line in enumerate(lines):
        text = str(line.get("text") or "")
        lowered = text.lower()
        if any(cue in lowered for cue in _CUES):
            hits.append(str(line.get("id") or index))
    return (1.0 if hits else 0.0), hits


def parse_label(text: str) -> tuple[float, list[str]] | None:
    """Parse a model JSON label. ``None`` means the caller should use the cue list."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    label = str(payload.get("label") or "").strip().lower()
    raw_ids = payload.get("line_ids") or []
    line_ids = [str(item) for item in raw_ids] if isinstance(raw_ids, list) else []
    if label in {"decision", "revelation", "reversal"}:
        return 1.0, line_ids
    if label == "none":
        return 0.0, line_ids
    return None
