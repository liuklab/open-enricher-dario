"""Prompt builders for the empty-context LLM evaluator.

Loads system and user templates from ``stages/prompts``.
Placeholders are filled at runtime with score bounds, rubrics, weights,
title, and profile JSON. Synopsis text and pipeline evidence are not used.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from open_enricher.stages.evaluator_core import criteria as C
from open_enricher.stages.prompt_load import PROMPTS_DIR, fill_prompt, load_prompt

_SYSTEM_PATH = PROMPTS_DIR / "evaluator_system_prompt.md"
_USER_PATH = PROMPTS_DIR / "evaluator_user_prompt.md"


def _format_rubrics() -> str:
    """Render dimension rubrics as a bullet list for the system prompt."""
    return "\n".join(
        f"- {name}: {text}" for name, text in C.DIMENSION_RUBRICS.items()
    )


def _format_weights() -> str:
    """Render dimension weights as a compact comma-separated list."""
    return ", ".join(
        f"{name}={weight}" for name, weight in C.DIMENSION_WEIGHTS.items()
    )


def system_prompt() -> str:
    """
    Return the system message for a fresh evaluator conversation.

    Returns:
        Instruction string with rubrics and score bounds filled in.
    """
    return fill_prompt(
        load_prompt("evaluator_system_prompt.md"),
        {
            "SCORE_MIN": str(C.SCORE_MIN),
            "SCORE_MAX": str(C.SCORE_MAX),
            "DIMENSION_RUBRICS": _format_rubrics(),
            "DIMENSION_WEIGHTS": _format_weights(),
        },
    )


def user_prompt(*, title: str, profile: Dict[str, Any]) -> str:
    """
    Return the user message that carries title and profile only.

    Args:
        title: Human-readable title derived from the video file name.
        profile: Parsed ``VideoProfile`` dict from the LLM stage.

    Returns:
        User content with title and profile JSON substituted.
    """
    profile_json = json.dumps(profile, ensure_ascii=False, indent=2)
    return fill_prompt(
        load_prompt("evaluator_user_prompt.md"),
        {
            "TITLE": title,
            "PROFILE_JSON": profile_json,
        },
    )
