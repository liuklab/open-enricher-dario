"""VLM memory keeps only the last n scene records when the spec says so."""

from __future__ import annotations

from open_enricher.stages.vlm_core.segments import (
    _scene_prompt,
    records_for_vlm_memory,
    select_prior_scenes,
)


def _scenes(count: int) -> list[dict[str, str]]:
    return [{"start_time": f"{index:02d}"} for index in range(count)]


def test_null_limit_keeps_every_prior_scene() -> None:
    prior = _scenes(6)
    assert select_prior_scenes(prior, None) == prior


def test_limit_two_keeps_the_latest_two() -> None:
    prior = _scenes(6)
    kept = select_prior_scenes(prior, 2)
    assert [item["start_time"] for item in kept] == ["04", "05"]


def test_limit_four_keeps_the_latest_four() -> None:
    prior = _scenes(6)
    kept = select_prior_scenes(prior, 4)
    assert [item["start_time"] for item in kept] == ["02", "03", "04", "05"]


def test_short_history_is_not_padded() -> None:
    prior = _scenes(2)
    assert select_prior_scenes(prior, 4) == prior


def test_visual_only_memory_drops_dialogue() -> None:
    prior = [{"description": "a room", "dialogue_description": "they argue about the patent"}]
    kept = records_for_vlm_memory(prior, visual_only=True)
    assert kept == [{"description": "a room"}]
    assert "patent" not in _scene_prompt(
        {"start_time": "00:00:00", "end_time": "00:05:00", "description": "they argue about the patent"},
        memory="",
        visual_only=True,
    )
