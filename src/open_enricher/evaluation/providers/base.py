"""Provider protocol for external metadata evaluators."""

from __future__ import annotations

from typing import Protocol


class EvaluatorProvider(Protocol):
    """
    One-shot text completion for an external evaluation task.

    Implementations must start a fresh conversation for every ``complete``
    call so each video and task is judged independently.
    """

    name: str
    model: str

    def complete(self, system: str, user: str) -> str:
        """
        Run a single evaluation turn and return the assistant text.

        Args:
            system: System instruction (rubrics, score bounds).
            user: User message with the inputs for the selected task.

        Returns:
            Raw assistant text, expected to be a JSON object.

        Raises:
            Exception: Provider-specific startup or run failures.
        """
        ...
