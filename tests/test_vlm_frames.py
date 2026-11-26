"""Frame budgets for the optional scene planner and evidence from the VLM clip sampler."""

from __future__ import annotations

from open_enricher.stages.vlm_core.frames import sampled_frame_count


def test_long_window_budget_is_capped_at_forty_frames() -> None:
    """A 300 s, 24 fps clip at sample fps 2 budgets 40 frames."""
    total = 300 * 24
    count = sampled_frame_count(total, video_fps=24, sample_fps=2, min_frames=2, max_frames=40)
    assert count == 40


def test_short_clip_follows_sample_fps_instead_of_the_cap() -> None:
    """Two seconds at 24 fps and sample fps 2 yields 4 frames, not 8."""
    total = 48
    count = sampled_frame_count(total, video_fps=24, sample_fps=2, min_frames=2, max_frames=8)
    assert count == 4


def test_max_frames_rounds_down_to_the_temporal_patch() -> None:
    """The optional scene planner allocates even budgets."""
    count = sampled_frame_count(7200, video_fps=24, sample_fps=2, min_frames=2, max_frames=7)
    assert count == 6


def test_frame_evidence_preserves_odd_processor_selection(tmp_path, monkeypatch) -> None:
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    from open_enricher.stages.vlm_core import frames

    processor = Qwen3VLVideoProcessor()
    monkeypatch.setattr(frames, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(frames, "probe_clip", lambda clip: (60, 24.0))
    monkeypatch.setattr(frames, "save_scene_frames", lambda clip, dest, indices: [
        dest / f"frame_{index}.jpg" for index in indices
    ])
    span = frames.sample_scene(
        tmp_path / "clip.mp4", video_processor=processor, video_filename="film.mkv",
        experiment_id="13", scene_index=1, start_seconds=120.0,
        sample_fps=2, min_frames=2, max_frames=40,
    )
    assert span["sampled_frames"] == 5
    assert [frame["index"] for frame in span["frames"]] == [0, 15, 30, 44, 59]
    assert span["frames"][-1]["timestamp_seconds"] == 122.458
