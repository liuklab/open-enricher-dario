"""Dimension definitions and weights for the LLM metadata evaluator.

The evaluator model scores each dimension on a 1-5 integer scale. The host
process applies these weights to produce the final quality metric. No
synopsis or upstream pipeline evidence is used: fidelity is judged against
the model's training knowledge of the titled work.
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

# Inclusive integer score range expected from the LLM.
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
        "Given only the film or episode title, how well do the metadata claims "
        "match the real work as you know it from training? Penalize invented "
        "plot points, wrong genre, wrong characters, or wrong setting. "
        "If you do not recognize the title, score low and say so in the note."
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
