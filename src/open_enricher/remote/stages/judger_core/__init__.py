"""Deterministic metadata quality scoring for the Judger pipeline stage.

Public entry point: ``score_profile``. Literature and design notes:
``docs/judger_references.md``.
"""

from .score import score_profile

__all__ = ["score_profile"]
