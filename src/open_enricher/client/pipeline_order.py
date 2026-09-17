"""Canonical stage order for the Open-Enricher pipeline."""

from __future__ import annotations

STAGE_ORDER: tuple[str, ...] = ("audio", "color", "vlm", "llm", "judger")