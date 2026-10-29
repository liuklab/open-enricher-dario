"""Index rule for the VLM explainability frame cache.

These tests do not decode video and do not load the model. They check the
same count and index formula the Qwen processor applies to a window clip.
"""

from __future__ import annotations

from open_enricher.stages.vlm_core.frames import (
    frame_indices,
    sampled_frame_count,
)


def test_long_window_is_capped_at_eight_evenly_spaced_frames() -> None:
    """A 300 s, 24 fps clip at sample fps 2 keeps 8 frames, first and last."""
    total = 300 * 24
    count = sampled_frame_count(total, video_fps=24, sample_fps=2, min_frames=2, max_frames=8)
    indices = frame_indices(total, count)
    assert count == 8
    assert indices[0] == 0
    assert indices[-1] == total - 1
    assert indices == sorted(set(indices))


def test_short_clip_follows_sample_fps_instead_of_the_cap() -> None:
    """Two seconds at 24 fps and sample fps 2 yields 4 frames, not 8."""
    total = 48
    count = sampled_frame_count(total, video_fps=24, sample_fps=2, min_frames=2, max_frames=8)
    assert count == 4
    assert frame_indices(total, count)[-1] == total - 1


def test_max_frames_rounds_down_to_the_temporal_patch() -> None:
    """Qwen pairs frames, so a cap of 7 becomes 6."""
    count = sampled_frame_count(7200, video_fps=24, sample_fps=2, min_frames=2, max_frames=7)
    assert count == 6
