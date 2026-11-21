"""Clock for one face on one frame.

A detection is an instant. The readable form is HH:MM:SS. The numeric
seconds stay beside it because the pipeline aligns on seconds.

The fraction is the real frame clock, ``frame index / fps``, when that
offset is actually known. Whole seconds stay whole. A value such as
178.178 is not a sub-second offset: it is what you get by stuffing the
integer second count into the millisecond field of ``HH:MM:SS.mmm`` and
parsing that string back. Those fractions are dropped.
"""

from __future__ import annotations

import math
from typing import Any


def clock_hms(seconds: float) -> str:
    """HH:MM:SS for the second that contains ``seconds``. No milliseconds."""
    whole = int(math.floor(seconds))
    if whole < 0:
        whole = 0
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def numeric_seconds(seconds: float) -> int | float:
    """Seconds for alignment.

    Keeps a real frame fraction (``index / fps``). Collapses a stuffed
    millisecond copy of the integer, and leaves a whole second as an
    integer so no ``.000`` is invented.
    """
    if seconds <= 0:
        return 0
    rounded = round(float(seconds), 3)
    whole = int(math.floor(rounded + 1e-9))
    stuffed = round(whole + (whole % 1000) / 1000.0, 3)
    if rounded == stuffed:
        return whole
    nearest = round(seconds)
    if abs(seconds - nearest) <= 1e-6:
        return int(nearest)
    return round(seconds, 6)


def face_instant(seconds: float) -> dict[str, Any]:
    """One face at one frame. No end field."""
    value = numeric_seconds(seconds)
    return {"time": clock_hms(float(value)), "seconds": value}


def face_span(start: float, end: float) -> dict[str, Any]:
    """First and last instant a character is seen.

    ``end`` is omitted when it only repeats ``start``.
    """
    row = face_instant(start)
    finish = face_instant(end)
    if finish["seconds"] == row["seconds"]:
        return row
    row["end"] = finish["time"]
    row["end_seconds"] = finish["seconds"]
    return row
