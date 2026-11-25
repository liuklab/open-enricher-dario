"""Resolve window-local model selections to the ASR rows shown in its prompt."""

from __future__ import annotations

import json
import re
from typing import Any

from open_enricher.logging_setup import get_logger

logger = get_logger(__name__)
_FOOTER = re.compile(r"^\s*EXCHANGES[ \t]*:[ \t]*", re.MULTILINE | re.IGNORECASE)


def resolve_exchanges(
    reply: str,
    rows: list[dict[str, Any]],
    numbered_lines: list[str],
    transcript: str,
) -> tuple[str, list[list[dict[str, Any]]]]:
    """Keep the prose and resolve at most two non-overlapping, visible selections."""
    parts = _FOOTER.split(reply, maxsplit=1)
    summary = parts[0].strip()
    if not summary:
        raise ValueError("Dialogue model omitted the interval summary")
    if len(parts) == 1:
        logger.warning("Dialogue model omitted the EXCHANGES footer")
        return summary, []
    try:
        selections = json.loads(parts[1].strip())
    except ValueError:
        selections = None
    if not isinstance(selections, list):
        logger.warning("Dialogue model returned an invalid EXCHANGES footer")
        return summary, []

    shown = set(transcript.splitlines())
    visible = {
        index: row
        for index, (row, line) in enumerate(zip(rows, numbered_lines, strict=True), start=1)
        if line in shown
    }
    exchanges = []
    used: set[int] = set()
    for ids in selections:
        if (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= 3
            or any(type(i) is not int or i not in visible or i in used for i in ids)
            or ids != list(range(ids[0], ids[0] + len(ids)))
        ):
            logger.warning("Dropping invalid or overlapping dialogue exchange IDs: %r", ids)
            continue
        exchanges.append([
            {
                "line_id": i,
                "start_timestamp": visible[i]["start_timestamp"],
                "end_timestamp": visible[i].get("end_timestamp"),
                "speaker": visible[i].get("speaker") or "UNKNOWN",
                "text": visible[i]["text"],
            }
            for i in ids
        ])
        used.update(ids)
        if len(exchanges) == 2:
            break
    return summary, exchanges
