"""Consolidation and prefix-only retrieval."""

from __future__ import annotations

from open_enricher.visual_memory import (
    MemoryEntry,
    consolidate,
    retrieve,
    visible_entries,
)


def _entry(frame_id: str, timestamp: float, vector: list[float]) -> MemoryEntry:
    return MemoryEntry(
        frame_id=frame_id,
        timestamp=timestamp,
        tokens=[vector],
    )


def test_consolidation_merges_the_closest_adjacent_pair() -> None:
    entries = [
        _entry("a", 0.0, [1.0, 0.0]),
        _entry("a2", 1.0, [0.99, 0.01]),
        _entry("b", 2.0, [0.0, 1.0]),
    ]
    merged = consolidate(entries, target=2)
    assert len(merged) == 2
    assert merged[0].weight == 2
    assert "b" in {merged[0].frame_id, merged[1].frame_id}


def test_second_appearance_retrieves_the_original_and_not_the_future() -> None:
    first = _entry("lamp", 0.0, [1.0, 0.0, 0.0])
    lookalike = _entry("lookalike", 5.0, [0.0, 1.0, 0.0])
    again = _entry("lamp-late", 10.0, [1.0, 0.0, 0.0])
    bank = [first, lookalike, again]
    assert visible_entries(bank, 0.0) == []
    assert retrieve(first, bank, limit=2, scene_start=0.0) == []
    picked = retrieve(again, bank, limit=2, scene_start=10.0)
    ids = [row["frame_id"] for row in picked]
    assert "lamp" in ids
    assert "lamp-late" not in ids
    assert "lookalike" in ids


def test_retrieval_excludes_entries_with_future_constituents() -> None:
    from open_enricher.visual_memory import temporal_history

    query = MemoryEntry("query", 10.0, [[1.0, 0.0]])
    past = MemoryEntry("past", 6.0, [[0.0, 1.0]], end=9.0)
    straddling = MemoryEntry("straddling", 8.0, [[1.0, 0.0]], end=10.0)
    bank = [past, straddling]

    assert visible_entries(bank, 10.0) == [past]
    assert [row["frame_id"] for row in retrieve(query, bank, limit=4, scene_start=10.0)] == ["past"]
    assert [row["frame_id"] for row in temporal_history(
        [entry.to_dict() for entry in bank], 10.0, limit=4
    )] == ["past"]
