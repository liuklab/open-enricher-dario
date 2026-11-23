"""Title frame budget and importance allocation."""

from __future__ import annotations

from pathlib import Path

from open_enricher.stages.pipeline_spec import load_pipeline_spec
from open_enricher.stages.vlm_core import planned
from open_enricher.stages.vlm_core.allocation import allocate_frames, control_frame_budget
from open_enricher.timeline import fixed_windows


def test_control_budget_matches_one_capped_window() -> None:
    budget = control_frame_budget(
        300,
        300,
        video_fps=24,
        sample_fps=2,
        min_frames=2,
        max_frames=40,
    )
    assert budget == 40


def test_equal_scenes_with_different_importance_get_different_density() -> None:
    intervals = fixed_windows(600, 300)
    budget = control_frame_budget(
        600,
        300,
        video_fps=24,
        sample_fps=2,
        min_frames=2,
        max_frames=40,
    )
    plan = allocate_frames(
        intervals,
        budget=budget,
        history_fraction=0.0,
        mode="adaptive",
        narrative=[0.0, 1.0],
        cuts_per_minute=None,
        visual_change=None,
    )
    counts = [group["frames"] for group in plan["groups"]]
    assert sum(counts) == plan["current_frames"] == budget
    assert counts[1] > counts[0]


def test_scoring_failure_falls_back_to_duration() -> None:
    intervals = fixed_windows(600, 300)
    plan = allocate_frames(
        intervals,
        budget=40,
        history_fraction=0.0,
        mode="adaptive",
        narrative=[0.0, 1.0],
        scoring_failed=True,
    )
    assert plan["mode"] == "duration"
    assert plan["scoring_failed"] is True
    assert sum(group["frames"] for group in plan["groups"]) == 40

def test_partial_frame_decode_drops_explicit_frame_labels(monkeypatch) -> None:
    spec = load_pipeline_spec(
        Path(__file__).resolve().parents[1]
        / "src/open_enricher/experiments/variants/12-combined.json"
    )
    scene = {"scene_id": "s1", "start_seconds": 10, "end_seconds": 20}
    group = {"call_id": "c1", "scene_ids": ["s1"], "start": 10, "end": 20,
             "frames": 2, "timestamps": [11, 12]}
    monkeypatch.setattr(planned, "matching_artifact", lambda path: {"status": "success"})
    monkeypatch.setattr(planned, "allocate_frames", lambda *args, **kwargs: {"groups": [group]})
    monkeypatch.setattr(planned, "_history_stamps", lambda *args: [8, 9])
    monkeypatch.setattr(planned, "_grab", lambda *args: ["earlier.jpg", "current.jpg"])

    prepared, _ = planned.prepare_scenes(Path("clip.mkv"), "clip.mkv", [scene], spec, sample_fps=2)

    assert prepared[0]["image_paths"] == []
    assert prepared[0]["prompt_extra"] == ""
