"""Read markdown prompt templates from the shared ``stages/prompts`` directory."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=64)
def load_prompt(name: str) -> str:
    """
    Load a prompt template by file name from ``stages/prompts``.

    Args:
        name: File name only (for example ``evaluator_system_prompt.md``).

    Returns:
        File contents with leading and trailing whitespace stripped.

    Raises:
        FileNotFoundError: When the file does not exist.
    """
    prompt_path = PROMPTS_DIR / name
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def fill_prompt(template: str, mapping: dict[str, str]) -> str:
    """
    Replace ``{{KEY}}`` placeholders in a template.

    Args:
        template: Markdown body that may contain placeholders.
        mapping: Placeholder name (without braces) to replacement string.

    Returns:
        Template with all listed placeholders substituted.
    """
    text = template
    for key, value in mapping.items():
        text = text.replace("{{" + key + "}}", value)
    return text
