"""Prompt helpers for the external evaluators.

Loads system and user templates from ``evaluation/prompts``. Placeholders are
filled at runtime with score bounds, rubrics, weights, title, reference
synopsis, and profile JSON for the RecSys task.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from open_enricher.evaluation import criteria as C

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_SYSTEM_PATH = PROMPTS_DIR / "evaluator_system_prompt.md"
_USER_PATH = PROMPTS_DIR / "evaluator_user_prompt.md"


@lru_cache(maxsize=16)
def load_prompt(name: str) -> str:
    """Load a prompt template by file name from ``evaluation/prompts``."""
    prompt_path = PROMPTS_DIR / name
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def clean_json_candidate(raw_text: str) -> str:
    """Isolate the JSON object between the first '{' and last '}'."""
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _fill_prompt(template: str, mapping: dict[str, str]) -> str:
    """Replace ``{{KEY}}`` placeholders in a template."""
    text = template
    for key, value in mapping.items():
        text = text.replace("{{" + key + "}}", value)
    return text


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
    return _fill_prompt(
        load_prompt("evaluator_system_prompt.md"),
        {
            "SCORE_MIN": str(C.SCORE_MIN),
            "SCORE_MAX": str(C.SCORE_MAX),
            "DIMENSION_RUBRICS": _format_rubrics(),
            "DIMENSION_WEIGHTS": _format_weights(),
        },
    )


def user_prompt(
    *,
    title: str,
    profile: Dict[str, Any],
    synopsis: str,
) -> str:
    """
    Return the user message that carries the title, reference synopsis, and profile.

    Args:
        title: Human-readable title derived from the video file name.
        profile: Parsed ``VideoProfile`` dict from the LLM stage.
        synopsis: Nonempty reference synopsis for this film or episode.

    Returns:
        User content with placeholders substituted.
    """
    if not synopsis or not synopsis.strip():
        raise ValueError("The RecSys evaluation requires a nonempty reference synopsis.")
    profile_json = json.dumps(profile, ensure_ascii=False, indent=2)
    return _fill_prompt(
        load_prompt("evaluator_user_prompt.md"),
        {
            "TITLE": title,
            "SYNOPSIS": synopsis.strip(),
            "PROFILE_JSON": profile_json,
        },
    )
