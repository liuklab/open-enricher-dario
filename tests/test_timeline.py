"""Shared timeline: partial tails, overlap assignment, and BaSSL repair."""

from __future__ import annotations

from open_enricher.stages.dialogue import interval_windows
from open_enricher.timeline import (
    assign_utterances,
    fixed_windows,
    validate_scene_bounds,
)


def test_fixed_windows_keep_a_short_tail() -> None:
    windows = fixed_windows(300.4, 300)
    assert [(item.start, item.end) for item in windows] == [(0.0, 300.0), (300.0, 300.4)]
    assert [item.id for item in windows] == ["w0001", "w0002"]
    assert interval_windows(300.4, 300) == [(0.0, 300.0)]


def test_overlap_rule_prefers_the_greater_share_and_breaks_ties_early() -> None:
    intervals = fixed_windows(600, 300)
    lines = [
        {"start_timestamp": 290.0, "end_timestamp": 310.0, "text": "mostly-early"},
        {"start_timestamp": 250.0, "end_timestamp": 350.0, "text": "tie"},
        {"start_timestamp": 300.0, "text": "point"},
    ]
    assigned = assign_utterances(lines, intervals)
    assert [line["text"] for line in assigned["w0001"]] == ["mostly-early", "tie"]
    assert [line["text"] for line in assigned["w0002"]] == ["point"]
    assert lines[0]["start_timestamp"] == 290.0


def test_invalid_boundaries_restore_fixed_windows() -> None:
    intervals, notes, replaced = validate_scene_bounds(
        [{"start": 10, "end": 5}],
        650,
        interval_seconds=300,
    )
    assert replaced is True
    assert [(item.start, item.end) for item in intervals] == [
        (0.0, 300.0),
        (300.0, 600.0),
        (600.0, 650.0),
    ]
    assert notes


def test_modelscope_clock_timestamps_cover_the_file() -> None:
    """Scene rows shaped like ModelScope output are not dropped as non-numeric."""
    raw = [
        {"timestamps": ["00:00:00.000", "00:04:41.573"], "frame": ["0", "6741"]},
        {"timestamps": ["00:04:41.573", "00:22:06.080"], "frame": ["6742", "31826"]},
    ]
    intervals, notes, replaced = validate_scene_bounds(
        raw,
        1326.08,
        interval_seconds=300,
    )
    assert replaced is False
    assert [item.source for item in intervals] == ["bassl", "bassl"]
    assert abs(intervals[0].end - 281.573) < 1e-3
    assert abs(intervals[-1].end - 1326.08) < 1e-3
    assert not any("no numeric span" in note for note in notes)


def test_valid_scenes_keep_ids_and_cover_a_gap() -> None:
    intervals, notes, replaced = validate_scene_bounds(
        [(0.0, 100.0), (120.0, 300.0)],
        300,
        interval_seconds=300,
    )
    assert replaced is False
    assert intervals[0].source == "bassl"
    assert intervals[0].id == "s0001"
    assert any(item.repaired for item in intervals)
    assert abs(intervals[-1].end - 300.0) < 1e-6
    assert notes
