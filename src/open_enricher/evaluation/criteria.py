"""Dimension definitions and weights for the metadata evaluator.

The evaluator provider scores each dimension on a 1-5 integer scale. The host
process applies these weights to produce the final quality metric.
"""

from __future__ import annotations

from typing import Dict, Tuple

# Ordered dimensions the evaluator must return. Weights sum to 1.0.
DIMENSION_WEIGHTS: Dict[str, float] = {
    "completeness": 0.15,
    "specificity": 0.25,
    "fidelity": 0.35,
    "consistency": 0.10,
    "recsys_usefulness": 0.15,
}

DIMENSION_ORDER: Tuple[str, ...] = tuple(DIMENSION_WEIGHTS.keys())

# Inclusive integer score range expected from the provider.
SCORE_MIN: int = 1
SCORE_MAX: int = 5

# Short rubrics injected into the evaluator prompt.
DIMENSION_RUBRICS: Dict[str, str] = {
    "completeness": (
        "Are all RecSys fields present, non-empty, and informative "
        "(genre, pacing, mood, tags, topics, audience, visual_style, narrative_hook)?"
    ),
    "specificity": (
        "Are values concrete and episode-level rather than generic fillers "
        "(avoid vague words like video, scene, content, various)?"
    ),
    "fidelity": (
        "Do the plot, characters, setting and genre claims agree with the synopsis? "
        "Penalize contradictions. Treat details the synopsis leaves open as unverified."
    ),
    "consistency": (
        "Do fields agree with each other (genre vs mood vs tags vs narrative_hook) "
        "without contradictions?"
    ),
    "recsys_usefulness": (
        "Would these fields help a recommender retrieve or recommend this title "
        "to the right audience?"
    ),
}

# Profile is considered usable when the weighted overall (1-5) clears this bar
# and fidelity is not a hard failure.
USABLE_OVERALL_MIN: float = 3.0
USABLE_FIDELITY_MIN: int = 2
