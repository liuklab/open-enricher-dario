"""Narration context uses the two previous dialogue summaries and nothing else."""

from __future__ import annotations

from open_enricher.stages.llm_core.narrate import (
    narration_user_prompt,
    previous_scene_summaries,
)


def _scenes() -> list[dict]:
    return [
        {
            "start_time": "00:00:00",
            "end_time": "00:05:00",
            "dialogue_description": "Mike files the patent.",
            "subjects_and_objects": ["desk"],
            "actions": ["sitting"],
        },
        {
            "start_time": "00:05:00",
            "end_time": "00:10:00",
            "dialogue_description": "Harvey confronts Mike.",
            "subjects_and_objects": ["office"],
            "actions": ["standing"],
        },
        {
            "start_time": "00:10:00",
            "end_time": "00:15:00",
            "dialogue_description": "The judge denies the injunction.",
            "subjects_and_objects": ["courtroom"],
            "actions": ["speaking"],
        },
    ]


def test_opening_scene_has_no_prior_summary() -> None:
    scenes = _scenes()
    assert previous_scene_summaries(scenes, 0) == []


def test_third_scene_keeps_only_the_previous_two_summaries() -> None:
    scenes = _scenes()
    prior = previous_scene_summaries(scenes, 2)
    assert [item["dialogue_description"] for item in prior] == [
        "Mike files the patent.",
        "Harvey confronts Mike.",
    ]


def test_narration_prompt_grounds_the_current_scene() -> None:
    scenes = _scenes()
    text = narration_user_prompt(scenes[2], previous_scene_summaries(scenes, 2))
    assert "The judge denies the injunction." in text
    assert "courtroom" in text
    assert "speaking" in text
    assert "Mike files the patent." in text
    assert "00:10:00" in text
