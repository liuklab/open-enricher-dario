"""Rewrite one scene from its dialogue summary plus visual extracts."""

from __future__ import annotations

from typing import Any

from open_enricher.stages.prompt_load import fill_prompt, load_prompt

# Sliding context for the narration pass. The VLM does not receive it.
NARRATION_CONTEXT_SCENES = 2


def previous_scene_summaries(
    scenes: list[dict[str, Any]],
    index: int,
    *,
    limit: int = NARRATION_CONTEXT_SCENES,
) -> list[dict[str, Any]]:
    """
    Return the dialogue summaries of the scenes immediately before ``index``.

    Args:
        scenes: Scene records in timeline order. Each may carry
            ``dialogue_description``.
        index: Index of the scene being narrated.
        limit: How many earlier scenes to keep. The oldest of those is first.

    Returns:
        Up to ``limit`` earlier scene dicts. Empty when ``index`` is 0.
    """
    if index <= 0 or limit <= 0:
        return []
    start = max(0, index - limit)
    return list(scenes[start:index])


def format_prior_summaries(scenes: list[dict[str, Any]]) -> str:
    """
    Format earlier dialogue summaries for the narration prompt.

    Args:
        scenes: Records from ``previous_scene_summaries``.

    Returns:
        A plain-text block, or a line saying this is the opening scene.
    """
    if not scenes:
        return "None. This is the opening scene."
    blocks: list[str] = []
    for scene in scenes:
        start = str(scene.get("start_time") or "")
        end = str(scene.get("end_time") or "")
        summary = str(scene.get("dialogue_description") or "").strip() or "(no summary)"
        blocks.append(f"{start}-{end}: {summary}")
    return "\n\n".join(blocks)


def narration_system_prompt() -> str:
    """Return the system message for one scene narration."""
    return load_prompt("scene_narration_system_prompt.md")


def narration_user_prompt(scene: dict[str, Any], prior: list[dict[str, Any]]) -> str:
    """
    Return the user message for one scene narration.

    Args:
        scene: Current scene, including ``dialogue_description``,
            ``subjects_and_objects``, and ``actions``.
        prior: Earlier scenes whose summaries are continuity only.
    """
    objects = scene.get("subjects_and_objects") or []
    actions = scene.get("actions") or []
    object_text = "\n".join(f"- {item}" for item in objects) or "- none listed"
    action_text = "\n".join(f"- {item}" for item in actions) or "- none listed"
    return fill_prompt(
        load_prompt("scene_narration_user_prompt.md"),
        {
            "START": str(scene.get("start_time") or ""),
            "END": str(scene.get("end_time") or ""),
            "PRIOR": format_prior_summaries(prior),
            "SCENE_SUMMARY": str(scene.get("dialogue_description") or "").strip()
            or "(no dialogue summary)",
            "OBJECTS": object_text,
            "ACTIONS": action_text,
        },
    )
