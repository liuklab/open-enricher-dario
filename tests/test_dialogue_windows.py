"""Fixed 5-minute windows and the dialogue lines that fall inside them."""

from __future__ import annotations

from open_enricher.stages.dialogue import interval_windows, lines_in_window


def test_interval_windows_cover_the_file() -> None:
    windows = interval_windows(750, 300)
    assert windows == [(0.0, 300.0), (300.0, 600.0), (600.0, 750.0)]


def test_lines_in_window_keep_the_opening_boundary_only() -> None:
    lines = [
        {"start_timestamp": 0.0, "text": "open"},
        {"start_timestamp": 299.9, "text": "late"},
        {"start_timestamp": 300.0, "text": "next"},
    ]
    picked = lines_in_window(lines, 0.0, 300.0)
    assert [line["text"] for line in picked] == ["open", "late"]
