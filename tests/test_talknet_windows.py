"""TalkNet window selection for dialogue coverage."""

from open_enricher.adapters.talknet_asd import select_windows


def test_window_budget_covers_dialogue_across_the_clip() -> None:
    tracks = [
        {"track_id": str(index), "start": float(index), "bbox": [0, 0, 60, 60]}
        for index in range(30)
    ]
    windows = select_windows(tracks, 4, [[5, 9], [20, 24]])
    assert [window["start"] for window in windows] == [6, 8, 21, 23]
