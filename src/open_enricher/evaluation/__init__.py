"""External metadata evaluator (separate from the local GPU pipeline).

To score scene understanding and the final RecSys profile, run::

    python -m open_enricher.evaluation --video <file>

Providers live under ``evaluation.providers``. The default is Cursor SDK with
Grok 4.7. The two tasks use independent prompts and reports.
"""

from open_enricher.evaluation.score import (
    build_report,
    failed_report,
    title_from_filename,
    weighted_overall,
)

__all__ = [
    "build_report",
    "failed_report",
    "title_from_filename",
    "weighted_overall",
]
