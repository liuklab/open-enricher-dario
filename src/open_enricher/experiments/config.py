"""Fixed paths and defaults for the local experiment round.

Centralizes repository-relative locations for variants, archived runs,
shared audio cache, and the default stage order. The local runner
reads these constants instead of Colab or Kaggle machine bindings.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root: experiments/config.py -> open_enricher -> src -> repo.
REPO_ROOT = Path(__file__).resolve().parents[3]

# Packaged experiment variant JSON files.
VARIANTS_DIR = Path(
    os.environ.get("OPEN_ENRICHER_VARIANTS_DIR") or Path(__file__).resolve().parent / "variants"
).resolve()

# Archived outputs for the local sequential round.
RUNS_ROOT = Path(os.environ.get("OPEN_ENRICHER_RUNS_ROOT") or REPO_ROOT / "runs" / "local").resolve()

# Shared audio cache reused across runs of the same video.
CACHE_ROOT = RUNS_ROOT / "_cache"

# Append-only resume log for the experiment loop.
PROGRESS_PATH = RUNS_ROOT / "progress.jsonl"

# Stages executed for each local experiment (metadata generation only).
# Kept here (not imported from runner) so CLI --list stays free of torch/whisperx.
# Scoring lives in ``open_enricher.evaluation`` and runs after each experiment.
PIPELINE_STAGES: tuple[str, ...] = ("audio", "dialogue", "vlm", "llm")

# Host-side evaluator weights (must match evaluation.criteria.DIMENSION_WEIGHTS).
EVALUATOR_SCORE_WEIGHTS = {
    "completeness": 0.15,
    "specificity": 0.25,
    "fidelity": 0.35,
    "consistency": 0.10,
    "recsys_usefulness": 0.15,
}
