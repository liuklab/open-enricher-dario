"""Fixed machine ↔ video mapping and paths for the exp2 round."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VARIANTS_DIR = Path(__file__).resolve().parent / "variants"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
RUNS_ROOT = REPO_ROOT / "runs" / "exp2"
CACHE_ROOT = REPO_ROOT / "runs" / "exp2" / "_cache"
SUMMARIES_DIR = REPO_ROOT / "test_videos_summary"
PROGRESS_PATH = RUNS_ROOT / "progress.jsonl"

EXPERIMENT_PREFIX = "exp2"
BASE_BRANCH = "exp2"
# refs/heads/exp2 already exists, so refs/heads/exp2/<id> cannot be created.
EXPERIMENT_BRANCH_PREFIX = "exp2-run"

COLAB_VIDEO = "Stranger_Things.S01E03.mkv"
KAGGLE_VIDEO = "Suits.S01E02.mkv"

# Stages executed in this round (deterministic judger stays out).
PIPELINE_STAGES = ("audio", "color", "vlm", "llm")

# Benchmark score weights (sum to 1.0). Applied by the harness, not the agent.
SCORE_WEIGHTS = {
    "completeness": 0.15,
    "specificity": 0.25,
    "fidelity": 0.35,
    "consistency": 0.10,
    "recsys_usefulness": 0.15,
}

BENCHMARK_MODEL = "composer-2.5"


@dataclass(frozen=True)
class MachineBinding:
    name: str  # colab | kaggle
    mode: str
    target: str
    video: str

    @property
    def summary_path(self) -> Path:
        return SUMMARIES_DIR / f"{self.video}_summary.txt"


def default_machines(
    *,
    colab_target: str = "enricher",
    kaggle_target: str = "open-enricher",
) -> tuple[MachineBinding, MachineBinding]:
    return (
        MachineBinding(
            name="colab",
            mode="colab-cli",
            target=colab_target,
            video=COLAB_VIDEO,
        ),
        MachineBinding(
            name="kaggle",
            mode="kaggle",
            target=kaggle_target,
            video=KAGGLE_VIDEO,
        ),
    )
