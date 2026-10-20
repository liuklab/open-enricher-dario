"""Prompt builders for the LLM profiling stage.

Loads system and user templates from ``stages/prompts``.
"""

from __future__ import annotations

import json
from typing import Any

from open_enricher.stages.prompt_load import fill_prompt, load_prompt


def profile_system_prompt() -> str:
    """Return the system message for VideoProfile generation."""
    return load_prompt("profile_system_prompt.md")


def profile_user_prompt(*, script_text: str, example_structure: dict[str, Any]) -> str:
    """
    Return the user message that carries metadata and the target schema.

    Args:
        script_text: Compact JSON string of gathered upstream metadata.
        example_structure: Empty VideoProfile shape the model must populate.
    """
    return fill_prompt(
        load_prompt("profile_user_prompt.md"),
        {
            "SCRIPT_TEXT": script_text,
            "EXAMPLE_STRUCTURE": json.dumps(example_structure, indent=2),
        },
    )


def claim_check_system_prompt() -> str:
    """Return the system message for the claim-check rewrite pass."""
    return load_prompt("claim_check_system_prompt.md")


def claim_check_user_prompt(*, evidence: dict[str, Any]) -> str:
    """
    Return the user message with timeline and dialogue evidence.

    Args:
        evidence: Profile plus optional VLM JSON and dialogue summary.
    """
    return fill_prompt(
        load_prompt("claim_check_user_prompt.md"),
        {
            "EVIDENCE_JSON": json.dumps(
                evidence, separators=(",", ":"), ensure_ascii=False
            ),
        },
    )


def profile_repair_user_prompt(*, error: object) -> str:
    """
    Return the follow-up user message after a schema validation failure.

    Args:
        error: Validation error shown to the model.
    """
    return fill_prompt(
        load_prompt("profile_repair_user_prompt.md"),
        {"ERROR": str(error)},
    )
