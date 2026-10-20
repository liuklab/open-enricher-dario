"""Fixed paths and defaults for the local experiment round.

Centralizes repository-relative locations for variants, archived runs,
shared audio/color cache, and the default stage order. The local runner
reads these constants instead of Colab or Kaggle machine bindings.
"""

from __future__ import annotations

from pathlib import Path

# Repository root: experiments/config.py -> open_enricher -> src -> repo.
REPO_ROOT = Path(__file__).resolve().parents[3]

# Packaged experiment variant JSON files.
VARIANTS_DIR = Path(__file__).resolve().parent / "variants"

# Archived outputs for the local sequential round.
RUNS_ROOT = REPO_ROOT / "runs" / "local"

# Shared audio and color cache reused across variants for the same video.
CACHE_ROOT = RUNS_ROOT / "_cache"

# Append-only resume log for the experiment loop.
PROGRESS_PATH = RUNS_ROOT / "progress.jsonl"

# Round prefix written into the data lake under outputs/.
EXPERIMENT_PREFIX = "local"

# Stages executed for each local experiment (includes deterministic judger).
# Kept here (not imported from runner) so CLI --list stays free of torch/whisperx.
PIPELINE_STAGES: tuple[str, ...] = ("audio", "color", "vlm", "llm", "judger")
