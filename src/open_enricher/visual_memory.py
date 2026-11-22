"""Short-term and long-term visual memory over token matrices.

Each entry stores the Q-Former token matrix, the source frame id and the
source timestamp. Consolidation merges the closest adjacent pair by mean
cosine similarity over corresponding tokens, weighted by how many frames
the entry already represents. A scene may read only entries whose
timestamp is strictly earlier than the scene start.
"""

from __future__ import annotations

from heapq import heappop, heappush
from dataclasses import dataclass, field
from typing import Any

from open_enricher.graph import cosine


@dataclass
class MemoryEntry:
    """One buffer row. ``tokens`` is a list of query vectors."""

    frame_id: str
    timestamp: float
    tokens: list[list[float]]
    weight: int = 1
    sources: list[str] = field(default_factory=list)
    start: float | None = None
    end: float | None = None

    def __post_init__(self) -> None:
        if not self.sources:
            self.sources = [self.frame_id]
        if self.start is None:
            self.start = self.timestamp
        if self.end is None:
            self.end = self.timestamp

    def to_dict(self) -> dict[str, Any]:
        """Manifest row. Token matrices stay out of the portable archive."""
        return {
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 3),
            "weight": self.weight,
            "sources": list(self.sources),
            "start": round(float(self.start or 0), 3),
            "end": round(float(self.end or 0), 3),
            "queries": len(self.tokens),
        }


def mean_token_cosine(left: MemoryEntry, right: MemoryEntry) -> float:
    """Mean cosine over aligned query tokens. Missing pairs score 0."""
    count = min(len(left.tokens), len(right.tokens))
    if count == 0:
        return 0.0
    total = 0.0
    for index in range(count):
        total += cosine(left.tokens[index], right.tokens[index])
    return total / count


def _merge_pair(left: MemoryEntry, right: MemoryEntry) -> MemoryEntry:
    """Weight the merged tokens by the number of frames each side represents."""
    total = left.weight + right.weight
    queries = max(len(left.tokens), len(right.tokens))
    width = 0
    if left.tokens:
        width = len(left.tokens[0])
    elif right.tokens:
        width = len(right.tokens[0])
    merged: list[list[float]] = []
    for index in range(queries):
        left_row = left.tokens[index] if index < len(left.tokens) else [0.0] * width
        right_row = right.tokens[index] if index < len(right.tokens) else [0.0] * width
        merged.append(
            [
                (a * left.weight + b * right.weight) / total
                for a, b in zip(left_row, right_row)
            ]
        )
    # The representative frame is the side closer to the merged tokens.
    synthetic = MemoryEntry(
        frame_id="merged",
        timestamp=0.0,
        tokens=merged,
        weight=total,
    )
    left_score = mean_token_cosine(synthetic, left)
    right_score = mean_token_cosine(synthetic, right)
    representative = left if left_score >= right_score else right
    return MemoryEntry(
        frame_id=representative.frame_id,
        timestamp=representative.timestamp,
        tokens=merged,
        weight=total,
        sources=list(left.sources) + list(right.sources),
        start=min(float(left.start or left.timestamp), float(right.start or right.timestamp)),
        end=max(float(left.end or left.timestamp), float(right.end or right.timestamp)),
    )


def consolidate(entries: list[MemoryEntry], target: int) -> list[MemoryEntry]:
    """
    Merge the closest adjacent pair until ``target`` entries remain.

    ``target`` below 1 keeps a single entry when the buffer is not empty.
    The input order is chronological and is preserved.
    """
    if target < 1:
        target = 1
    pending = list(entries)
    while len(pending) > target:
        scores = [
            mean_token_cosine(pending[index], pending[index + 1])
            for index in range(len(pending) - 1)
        ]
        # Ties keep the earlier pair.
        pair_index = max(range(len(scores)), key=lambda index: (scores[index], -index))
        merged = _merge_pair(pending[pair_index], pending[pair_index + 1])
        pending = pending[:pair_index] + [merged] + pending[pair_index + 2 :]
    return pending


def stream_memory(
    frames: list[MemoryEntry],
    *,
    short_capacity: int,
    short_target: int,
    long_capacity: int,
) -> dict[str, Any]:
    """
    Push a chronological stream through the short-term buffer.

    At capacity the buffer is consolidated to ``short_target``. Those
    entries append to long-term memory. The last short-term entry is
    carried into the next buffer. The final partial buffer is flushed.
    Long-term memory is consolidated, without dropping the first or last
    entry, when it exceeds ``long_capacity``.
    """
    if short_capacity < 2:
        raise ValueError("short_capacity must be at least 2")
    if not 1 <= short_target < short_capacity:
        raise ValueError("short_target must be at least 1 and below capacity")
    short: list[MemoryEntry] = []
    long_term: list[MemoryEntry] = []
    spans: list[dict[str, Any]] = []

    def _flush(final: bool) -> None:
        if len(short) <= short_target and not final:
            return
        target = 1 if final and len(short) == 1 else min(short_target, len(short))
        if final and len(short) <= short_target:
            consolidated = list(short)
        else:
            consolidated = consolidate(short, target if not final else min(short_target, len(short)))
        if not final and consolidated:
            carry = consolidated[-1]
            stored = consolidated[:-1]
        else:
            carry = None
            stored = consolidated
        for entry in stored:
            long_term.append(entry)
        spans.append(
            {
                "final": final,
                "input_frames": sum(entry.weight for entry in short),
                "stored": [entry.frame_id for entry in stored],
                "carry": None if carry is None else carry.frame_id,
            }
        )
        short.clear()
        if carry is not None:
            short.append(carry)

    for frame in frames:
        short.append(frame)
        if len(short) >= short_capacity:
            _flush(final=False)
    if short:
        _flush(final=True)
    compacted = False
    if len(long_term) > long_capacity:
        long_term = _compact_long_term(long_term, long_capacity)
        compacted = True
    return {
        "long_term": long_term,
        "spans": spans,
        "long_term_compacted": compacted,
    }


def _compact_long_term(entries: list[MemoryEntry], capacity: int) -> list[MemoryEntry]:
    """Merge interior adjacent pairs so the first and last entries remain."""
    if len(entries) <= capacity or capacity < 2:
        return consolidate(entries, max(1, capacity))
    pending = list(entries)
    next_index = list(range(1, len(pending) + 1))
    previous_index = list(range(-1, len(pending) - 1))
    revisions = [0] * len(pending)
    heap: list[tuple[float, int, int, int, int]] = []
    last = len(pending) - 1

    def queue_pair(left: int) -> None:
        right = next_index[left]
        if 0 < left < right < last:
            score = mean_token_cosine(pending[left], pending[right])
            heappush(heap, (-score, left, right, revisions[left], revisions[right]))

    for left in range(1, last - 1):
        queue_pair(left)

    remaining = len(pending)
    while remaining > capacity and heap:
        _, left, right, left_revision, right_revision = heappop(heap)
        if (
            next_index[left] != right
            or revisions[left] != left_revision
            or revisions[right] != right_revision
        ):
            continue

        pending[left] = _merge_pair(pending[left], pending[right])
        revisions[left] += 1
        revisions[right] += 1
        successor = next_index[right]
        next_index[left] = successor
        previous_index[successor] = left
        remaining -= 1
        queue_pair(previous_index[left])
        queue_pair(left)

    compacted = []
    index = 0
    while index <= last:
        compacted.append(pending[index])
        index = next_index[index]
    return compacted


def visible_entries(entries: list[MemoryEntry], scene_start: float) -> list[MemoryEntry]:
    """Use entries only when their representative and all merged frames are earlier."""
    return [
        entry for entry in entries
        if max(entry.timestamp, entry.end if entry.end is not None else entry.timestamp) < scene_start
    ]


def retrieve(
    current: MemoryEntry,
    entries: list[MemoryEntry],
    *,
    limit: int,
    scene_start: float,
) -> list[dict[str, Any]]:
    """
    Select older entries for the current tokens.

    Recent context is the latest visible entry. The other slots are the
    strongest matches that are not adjacent in time to one already chosen,
    so a lookalike sitting between two views of the same object does not
    crowd out the true earlier frame when the current tokens match it.
    """
    visible = visible_entries(entries, scene_start)
    if limit <= 0 or not visible:
        return []
    ranked = sorted(
        (
            {
                "frame_id": entry.frame_id,
                "timestamp": entry.timestamp,
                "score": mean_token_cosine(current, entry),
                "sources": list(entry.sources),
            }
            for entry in visible
        ),
        key=lambda row: (row["score"], -row["timestamp"]),
        reverse=True,
    )
    chosen: list[dict[str, Any]] = []
    latest = max(visible, key=lambda entry: entry.timestamp)
    chosen.append(
        {
            "frame_id": latest.frame_id,
            "timestamp": latest.timestamp,
            "score": mean_token_cosine(current, latest),
            "sources": list(latest.sources),
            "reason": "recent",
        }
    )
    for row in ranked:
        if len(chosen) >= limit:
            break
        if any(item["frame_id"] == row["frame_id"] for item in chosen):
            continue
        if any(abs(item["timestamp"] - row["timestamp"]) < 1e-6 for item in chosen):
            continue
        row = dict(row)
        row["reason"] = "match"
        chosen.append(row)
    chosen.sort(key=lambda row: row["timestamp"])
    return chosen


def temporal_history(
    entries: list[dict[str, Any]],
    scene_start: float,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """
    Pick earlier entries without token scores.

    The most recent entry before ``scene_start`` is kept, then older
    entries spread across the prefix. Nothing at or after ``scene_start``
    is visible.
    """
    prior = [
        row
        for row in entries
        if isinstance(row, dict)
        and float(row.get("timestamp") or 0) < scene_start
        and float(row.get("end") if row.get("end") is not None else row.get("timestamp") or 0)
        < scene_start
    ]
    if not prior or limit <= 0:
        return []
    prior.sort(key=lambda row: float(row.get("timestamp") or 0))
    chosen = [prior[-1]]
    if limit > 1 and len(prior) > 1:
        step = max(1, (len(prior) - 1) // limit)
        for index in range(0, len(prior) - 1, step):
            if len(chosen) >= limit:
                break
            row = prior[index]
            if row.get("frame_id") not in {item.get("frame_id") for item in chosen}:
                chosen.append(row)
    chosen.sort(key=lambda row: float(row.get("timestamp") or 0))
    return [
        {
            "frame_id": row.get("frame_id"),
            "timestamp": row.get("timestamp"),
            "reason": "temporal",
        }
        for row in chosen[:limit]
    ]


def retrieve_scene_history(
    frames: list[MemoryEntry],
    bank: list[MemoryEntry],
    scene_starts: list[float],
    *,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Query the same memory bank at each scene boundary."""
    rows = [entry.to_dict() for entry in bank]
    history = []
    for start in scene_starts:
        query = next((frame for frame in frames if frame.timestamp >= start), None)
        picked = retrieve(query, bank, limit=limit, scene_start=start) if query else []
        if not picked:
            picked = temporal_history(rows, start, limit=limit)
        history.append({"start": start, "frames": picked})
    return history


def dedupe_frame_ids(short_ids: list[str], long_ids: list[str]) -> list[str]:
    """Keep short-term ids first, then long-term ids that are not already included."""
    seen: set[str] = set()
    ordered: list[str] = []
    for frame_id in list(short_ids) + list(long_ids):
        if frame_id in seen:
            continue
        seen.add(frame_id)
        ordered.append(frame_id)
    return ordered
