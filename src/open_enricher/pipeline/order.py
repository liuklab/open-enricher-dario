"""Canonical stage order. Import nothing heavy so the laptop can use this list."""

from __future__ import annotations

STAGE_ORDER: tuple[str, ...] = ("audio", "color", "metadata")
