"""Shared presentation timeline for the experimental variants.

Intervals are half-open ranges in seconds on the source clock. Fixed
windows are known from the media duration before any model runs. BaSSL
may replace them after its boundaries have been validated. Variant 06
keeps ``dialogue.interval_windows`` and ``lines_in_window`` and does not
call this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Interval:
    """One half-open span ``[start, end)`` on the source presentation clock."""

    id: str
    start: float
    end: float
    kind: str
    source: str
    parent_id: str | None = None
    scene_ids: tuple[str, ...] = ()
    repaired: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the interval for a timeline artifact."""
        return {
            "id": self.id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "kind": self.kind,
            "source": self.source,
            "parent_id": self.parent_id,
            "scene_ids": list(self.scene_ids),
            "repaired": self.repaired,
        }


def fixed_windows(
    duration: float,
    interval_seconds: int,
    *,
    prefix: str = "w",
    source: str = "fixed",
) -> list[Interval]:
    """
    Cover ``duration`` with contiguous windows, including a short tail.

    A remainder under one second stays in its own interval. That is tighter
    than ``interval_windows``, which stops once the cursor is within a
    second of the end.

    Args:
        duration: Media duration in seconds.
        interval_seconds: Nominal window length.
        prefix: Stable id prefix. Windows use ``w`` and repaired gaps use ``s``.
        source: Provenance stored on each interval.

    Returns:
        Half-open intervals in order. Empty when ``duration`` is not positive.
    """
    if interval_seconds <= 0:
        raise ValueError(f"interval_seconds must be positive, got {interval_seconds}")
    if duration <= 0:
        return []
    windows: list[Interval] = []
    start = 0.0
    index = 1
    while start < duration:
        end = min(duration, start + float(interval_seconds))
        if end <= start:
            break
        windows.append(
            Interval(
                id=f"{prefix}{index:04d}",
                start=start,
                end=end,
                kind="window",
                source=source,
                repaired=False,
            )
        )
        index += 1
        start = end
    return windows


def control_windows(duration: float, interval_seconds: int) -> list[tuple[float, float]]:
    """Return the windows variant 06 would actually build for this duration."""
    # Lazy import: the dialogue stage imports this module at load time.
    from open_enricher.stages.dialogue import interval_windows

    return interval_windows(duration, interval_seconds)


def utterance_bounds(line: dict[str, Any]) -> tuple[float, float] | None:
    """
    Read an utterance span from a WhisperX row.

    A missing or inverted end collapses the span to a point at ``start``.
    """
    try:
        start = float(line.get("start_timestamp"))
    except (TypeError, ValueError):
        return None
    end_raw = line.get("end_timestamp", None)
    if end_raw is None:
        return (start, start)
    try:
        end = float(end_raw)
    except (TypeError, ValueError):
        return (start, start)
    if end < start:
        return (start, start)
    return (start, end)


def _overlap(start: float, end: float, interval: Interval) -> float:
    return max(0.0, min(end, interval.end) - max(start, interval.start))


def assign_utterances(
    lines: list[Any],
    intervals: list[Interval],
) -> dict[str, list[dict[str, Any]]]:
    """
    Assign each utterance to one interval.

    A positive-length utterance goes to the interval with the greater
    overlap. Equal overlap keeps the interval that starts earlier. A
    zero-length utterance goes to the half-open interval that contains
    its start, so a timestamp equal to a boundary belongs to the later
    interval. Original timestamps are not rewritten. Unmatched rows are
    omitted. The last interval also keeps a point that lands exactly on
    the media end.
    """
    buckets: dict[str, list[dict[str, Any]]] = {item.id: [] for item in intervals}
    ordered = sorted(intervals, key=lambda item: (item.start, item.id))
    if not ordered:
        return buckets
    final = ordered[-1]
    for line in lines:
        if not isinstance(line, dict):
            continue
        bounds = utterance_bounds(line)
        if bounds is None:
            continue
        start, end = bounds
        if end > start:
            best: Interval | None = None
            best_amount = 0.0
            for item in ordered:
                amount = _overlap(start, end, item)
                if amount > best_amount:
                    best = item
                    best_amount = amount
            if best is not None and best_amount > 0:
                buckets[best.id].append(line)
            continue
        placed = False
        for item in ordered:
            if item.start <= start < item.end:
                buckets[item.id].append(line)
                placed = True
                break
        if not placed and start == final.end:
            buckets[final.id].append(line)
    return buckets


def _as_pair(raw: Any) -> tuple[float, float] | None:
    """Read one boundary object or pair into ``(start, end)`` seconds."""
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            return (float(raw[0]), float(raw[1]))
        except (TypeError, ValueError):
            return None
    if not isinstance(raw, dict):
        return None
    start = raw.get("start", raw.get("start_time", raw.get("start_seconds")))
    end = raw.get("end", raw.get("end_time", raw.get("end_seconds")))
    try:
        return (float(start), float(end))
    except (TypeError, ValueError):
        return None


def validate_scene_bounds(
    raw_bounds: list[Any] | None,
    duration: float,
    *,
    interval_seconds: int = 300,
    gap_tolerance: float = 1.0,
) -> tuple[list[Interval], list[str], bool]:
    """
    Check order, bounds and coverage of model boundaries.

    Verified scenes are kept. A gap larger than ``gap_tolerance`` is
    covered with fixed windows and marked repaired. A tiny gap is closed
    by extending the next scene back to the cursor. When the input is
    empty, inverted beyond repair, or leaves more than half the duration
    uncovered before repair, the whole timeline is replaced with fixed
    windows.

    Args:
        raw_bounds: Scene objects or ``(start, end)`` pairs from BaSSL.
        duration: Source duration in seconds.
        interval_seconds: Fixed-window length used for replacements.
        gap_tolerance: Gaps at or under this many seconds are closed.

    Returns:
        ``(intervals, notes, replaced_all)``. ``replaced_all`` is True
        when no BaSSL boundary survived.
    """
    notes: list[str] = []
    if duration <= 0:
        return [], ["duration is not positive"], True
    pairs: list[tuple[float, float]] = []
    for raw in raw_bounds or []:
        pair = _as_pair(raw)
        if pair is None:
            notes.append("dropped a boundary that had no numeric span")
            continue
        start, end = pair
        if end < start:
            notes.append(f"dropped inverted boundary {start:.3f}-{end:.3f}")
            continue
        start = max(0.0, min(duration, start))
        end = max(0.0, min(duration, end))
        if end - start <= 1e-3:
            notes.append(f"dropped empty boundary at {start:.3f}")
            continue
        pairs.append((start, end))
    pairs.sort()
    merged: list[tuple[float, float]] = []
    for start, end in pairs:
        if merged and start < merged[-1][1] - 1e-3:
            notes.append(f"merged overlapping boundary at {start:.3f}")
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    verified = sum(end - start for start, end in merged)
    if not merged or verified < duration * 0.5:
        notes.append("boundaries rejected; restored fixed windows")
        windows = fixed_windows(duration, interval_seconds, prefix="w", source="fixed")
        return windows, notes, True

    intervals: list[Interval] = []
    cursor = 0.0
    index = 1

    def _append_fixed(span_start: float, span_end: float) -> None:
        nonlocal index
        length = span_end - span_start
        for piece in fixed_windows(length, interval_seconds, prefix="r", source="fixed"):
            intervals.append(
                Interval(
                    id=f"s{index:04d}",
                    start=span_start + piece.start,
                    end=span_start + piece.end,
                    kind="window",
                    source="fixed",
                    repaired=True,
                )
            )
            index += 1

    for start, end in merged:
        if start - cursor > gap_tolerance:
            notes.append(f"covered gap {cursor:.3f}-{start:.3f} with fixed windows")
            _append_fixed(cursor, start)
        elif start - cursor > 1e-3:
            notes.append(f"closed gap {cursor:.3f}-{start:.3f}")
            start = cursor
        intervals.append(
            Interval(
                id=f"s{index:04d}",
                start=start,
                end=end,
                kind="scene",
                source="bassl",
                repaired=False,
            )
        )
        index += 1
        cursor = end
    if duration - cursor > gap_tolerance:
        notes.append(f"covered tail {cursor:.3f}-{duration:.3f} with fixed windows")
        _append_fixed(cursor, duration)
    elif duration - cursor > 1e-3 and intervals:
        notes.append(f"extended final interval to {duration:.3f}")
        last = intervals[-1]
        intervals[-1] = Interval(
            id=last.id,
            start=last.start,
            end=duration,
            kind=last.kind,
            source=last.source,
            parent_id=last.parent_id,
            scene_ids=last.scene_ids,
            repaired=True,
        )
    return intervals, notes, False
