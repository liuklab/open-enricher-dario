"""Cursor SDK evaluator provider using Grok 4.7 with an empty toolset.

Each ``complete`` call uses ``Agent.prompt`` so the agent is created, run once,
and disposed. Tools are disabled and ``cwd`` is an empty temporary directory
so the agent cannot read the repository or pipeline outputs.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any


# Default model id for Grok 4.7 via Cursor. Override with OPEN_ENRICHER_EVAL_MODEL.
# Cursor.models.list() accepts "grok-4.7", not IDE slugs such as "grok-4.7-high-fast".
DEFAULT_MODEL = "grok-4.7"
BRIDGE_STARTUP_TIMEOUT_SECONDS = 120


class CursorGrokProvider:
    """
    One-shot Cursor agent that returns JSON scores as plain text.

    Requires the optional ``cursor-sdk`` extra and ``CURSOR_API_KEY``.
    """

    name = "cursor"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """
        Configure API key and model for Cursor agent runs.

        Args:
            api_key: Cursor API key. Defaults to ``CURSOR_API_KEY``.
            model: Model id. Defaults to ``OPEN_ENRICHER_EVAL_MODEL`` or
                ``grok-4.7``.

        Raises:
            ValueError: When no API key is available.
            ImportError: When ``cursor-sdk`` is not installed.
        """
        key = (api_key if api_key is not None else os.getenv("CURSOR_API_KEY", "")).strip()
        if not key:
            raise ValueError(
                "CURSOR_API_KEY is required for the cursor evaluator provider"
            )
        self._api_key = key
        env_model = os.getenv("OPEN_ENRICHER_EVAL_MODEL", "").strip()
        self.model = (model or env_model or DEFAULT_MODEL).strip()

    def complete(self, system: str, user: str) -> str:
        """
        Ask Grok for a JSON score object in a fresh agent run.

        Args:
            system: System instruction.
            user: User message with the inputs for the selected task.

        Returns:
            Assistant text from the finished run.

        Raises:
            CursorAgentError: When the run never starts (auth, config, network).
            RuntimeError: When the run starts but finishes with status ``error``,
                or when no assistant text is returned.
        """
        from cursor_sdk import (  # type: ignore[import-untyped]
            Agent,
            AgentOptions,
            Client,
            LocalAgentOptions,
        )

        # Combine into one user message: Cursor Agent.prompt is a single turn.
        message = f"{system.strip()}\n\n---\n\n{user.strip()}"

        with tempfile.TemporaryDirectory(prefix="open_enricher_eval_") as tmp:
            options = AgentOptions(
                api_key=self._api_key,
                model=self.model,
                tools=[],
                local=LocalAgentOptions(cwd=tmp),
            )
            # The bridge can take longer than the SDK default on a cold Slurm node.
            with Client.launch_bridge(
                workspace=tmp, timeout=BRIDGE_STARTUP_TIMEOUT_SECONDS,
            ) as client:
                result: Any = Agent.prompt(message, options, client=client)

        status = getattr(result, "status", None)
        run_id = getattr(result, "id", None)
        if status != "finished":
            raise RuntimeError(
                f"Cursor agent run ended with status={status!r}"
                + (f" (run id {run_id})" if run_id else "")
            )

        text = getattr(result, "result", None)
        if text is None:
            text = getattr(result, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Cursor agent returned empty assistant text")
        return text.strip()
