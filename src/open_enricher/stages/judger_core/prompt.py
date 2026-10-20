"""Prompt builders for the empty-context LLM judger.

Loads system and user templates from ``judger_prompt.md`` next to this module.
The markdown file intentionally excludes synopsis text and upstream pipeline
artifacts. Placeholders are filled at runtime with score bounds, rubrics,
weights, title, and profile JSON.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from open_enricher.stages.judger_core import criteria as C

_PROMPT_PATH = Path(__file__).resolve().parent / "judger_prompt.md"


@lru_cache(maxsize=1)
def _load_sections() -> Dict[str, str]:
    """
    Parse ``judger_prompt.md`` into named sections.

    Expected headings are ``## System`` and ``## User``. Content under each
    heading is kept verbatim (placeholders still present).

    Returns:
        Mapping of lowercased section name to template body.

    Raises:
        FileNotFoundError: When the markdown file is missing.
        ValueError: When a required section is absent.
    """
    if not _PROMPT_PATH.is_file():
        raise FileNotFoundError(f"Judger prompt file not found: {_PROMPT_PATH}")

    sections: Dict[str, list[str]] = {}
    current: str | None = None
    for line in _PROMPT_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line[3:].strip().lower()
            sections[current] = []
            continue
        if current is None:
            continue
        sections[current].append(line)

    result = {
        name: "\n".join(lines).strip()
        for name, lines in sections.items()
        if name in ("system", "user")
    }
    if "system" not in result or "user" not in result:
        raise ValueError(
            f"judger_prompt.md must define ## System and ## User sections "
            f"(found: {sorted(sections)})"
        )
    return result


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


def _fill(template: str, mapping: Dict[str, str]) -> str:
    """
    Replace ``{{KEY}}`` placeholders in a template.

    Args:
        template: Markdown section body.
        mapping: Placeholder name to replacement string.

    Returns:
        Filled template with placeholders substituted.
    """
    text = template
    for key, value in mapping.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def system_prompt() -> str:
    """
    Return the system message for a fresh judger conversation.

    Returns:
        Instruction string loaded from ``judger_prompt.md`` with rubrics and
        score bounds filled in.
    """
    return _fill(
        _load_sections()["system"],
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
        User content string from ``judger_prompt.md`` with title and profile
        JSON substituted.
    """
    profile_json = json.dumps(profile, ensure_ascii=False, indent=2)
    return _fill(
        _load_sections()["user"],
        {
            "TITLE": title,
            "PROFILE_JSON": profile_json,
        },
    )
