"""Named constants for the deterministic metadata quality judger.

Each constant is tied to a criterion from the quality frameworks catalogued in
``docs/judger_references.md`` (Bruce & Hillmann 2004; Ochoa & Duval 2009;
EU Metadata Quality Assessment). Thresholds are hypotheses, not human-calibrated
cutoffs: Ochoa & Duval explicitly recommend later calibration against expert review.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple

# ---------------------------------------------------------------------------
# Usability gate (Park 2009 functional perspective + Ochoa & Duval calibration note)
# ---------------------------------------------------------------------------
# A profile is marked ``usable`` only when accessibility is 1 and all three
# thresholds below are met. These numbers are starting hypotheses.
USABLE_COMPLETENESS_MIN: float = 0.70
USABLE_GROUNDING_MIN: float = 0.40
USABLE_OVERALL_MIN: float = 0.65

# ---------------------------------------------------------------------------
# Dimension weights for the overall score (EU MQA: mandatory > recommended)
# ---------------------------------------------------------------------------
# Accessibility is a hard gate (score 0 forces overall = 0) and does not enter
# the weighted mean. Timeliness is intentionally absent (one-shot profile).
DIMENSION_WEIGHTS: Dict[str, float] = {
    "completeness": 0.20,
    "conformance": 0.15,
    "consistency": 0.15,
    "specificity": 0.20,
    "grounding": 0.20,
    "provenance": 0.10,
    "accessibility": 0.0,
}

# ---------------------------------------------------------------------------
# Per-field importance weights (EU MQA layered importance)
# ---------------------------------------------------------------------------
# Retrieval fields support discovery hardest, so they weigh more.
FIELD_WEIGHTS: Dict[str, float] = {
    "recsys_metadata.primary_genre": 1.0,
    "recsys_metadata.sub_genres": 0.8,
    "recsys_metadata.pacing": 0.8,
    "recsys_metadata.mood": 0.8,
    "search_and_discovery.semantic_tags": 1.0,
    "search_and_discovery.main_topics": 1.0,
    "audience_profiling.target_age_group": 0.8,
    "audience_profiling.viewer_interests": 0.8,
    "content_summary.visual_style": 0.7,
    "content_summary.narrative_hook": 0.7,
}

# Ordered leaf paths matching VideoProfile in video_profile.py.
FIELD_PATHS: Tuple[str, ...] = tuple(FIELD_WEIGHTS.keys())

# ---------------------------------------------------------------------------
# Cardinality windows (from Pydantic field descriptions on VideoProfile)
# ---------------------------------------------------------------------------
# Strings are omitted: when present they always pass the cardinality check.
LIST_CARDINALITY: Dict[str, Tuple[int, int]] = {
    "recsys_metadata.sub_genres": (1, 4),
    "recsys_metadata.mood": (1, 4),
    "search_and_discovery.semantic_tags": (3, 6),
    "search_and_discovery.main_topics": (1, 4),
    "audience_profiling.viewer_interests": (1, 4),
}

# ---------------------------------------------------------------------------
# Closed vocabularies (Bruce & Hillmann "conformance to expectations")
# ---------------------------------------------------------------------------
PACING_VOCABULARY: FrozenSet[str] = frozenset({"Fast", "Moderate", "Slow"})

# Case-insensitive match against the normalized (title-case) form.
PRIMARY_GENRE_VOCABULARY: FrozenSet[str] = frozenset(
    {
        "Action",
        "Adventure",
        "Animation",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Education",
        "Fantasy",
        "Horror",
        "Music",
        "Mystery",
        "News",
        "Romance",
        "Sci-Fi",
        "Science Fiction",
        "Sports",
        "Thriller",
        "Vlog",
        "Western",
    }
)

TARGET_AGE_VOCABULARY: FrozenSet[str] = frozenset(
    {
        "All Ages",
        "Children",
        "Teens",
        "Young Adults",
        "Gen Z",
        "Millennials",
        "Adults",
        "Family",
    }
)

# Fields that apply a closed-vocabulary check. Free-text fields are omitted.
CONFORMANCE_FIELDS: Dict[str, FrozenSet[str]] = {
    "recsys_metadata.pacing": PACING_VOCABULARY,
    "recsys_metadata.primary_genre": PRIMARY_GENRE_VOCABULARY,
    "audience_profiling.target_age_group": TARGET_AGE_VOCABULARY,
}

# ---------------------------------------------------------------------------
# Placeholders rejected by the presence check (Ochoa completeness)
# ---------------------------------------------------------------------------
PLACEHOLDER_VALUES: FrozenSet[str] = frozenset(
    {
        "",
        "string",
        "n/a",
        "na",
        "none",
        "null",
        "unknown",
        "todo",
        "...",
        "tbd",
        "placeholder",
    }
)

# ---------------------------------------------------------------------------
# Specificity / text information content (Ochoa & Duval strongest human correlate)
# ---------------------------------------------------------------------------
# Fixed English stopword list: no NLTK dependency, deterministic across machines.
ENGLISH_STOPWORDS: FrozenSet[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
        "don",
        "should",
        "now",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "of",
        "as",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "he",
        "she",
        "they",
        "them",
        "their",
        "we",
        "you",
        "your",
        "i",
        "me",
        "my",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "why",
        "where",
    }
)

# Generic fillers that add almost no retrieval value for a recommender.
GENERIC_FILLERS: FrozenSet[str] = frozenset(
    {
        "video",
        "videos",
        "scene",
        "scenes",
        "people",
        "person",
        "something",
        "someone",
        "various",
        "stuff",
        "things",
        "thing",
        "content",
        "media",
        "clip",
        "clips",
        "footage",
        "general",
        "interesting",
        "nice",
        "good",
        "bad",
        "etc",
    }
)

# Minimum number of distinct content tokens for a full specificity score of 1.0.
SPECIFICITY_MIN_CONTENT_TOKENS: int = 2

# ---------------------------------------------------------------------------
# Consistency band for Jaccard(semantic_tags, main_topics)
# ---------------------------------------------------------------------------
# Some shared terms are useful; a full copy is not. Score 1.0 inside the band,
# linear falloff outside.
TAGS_TOPICS_JACCARD_LOW: float = 0.15
TAGS_TOPICS_JACCARD_HIGH: float = 0.75

# ---------------------------------------------------------------------------
# Color name → approximate RGB (for visual_style vs color stage palettes)
# ---------------------------------------------------------------------------
# Used only when visual_style mentions a known color word and color evidence
# is present. Distance is Euclidean in RGB; names within COLOR_MATCH_MAX_DISTANCE
# of any palette hex count as a match.
COLOR_NAME_TO_RGB: Dict[str, Tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (220, 40, 40),
    "green": (40, 160, 60),
    "blue": (40, 80, 200),
    "yellow": (230, 210, 40),
    "orange": (230, 130, 30),
    "purple": (140, 50, 180),
    "pink": (230, 120, 160),
    "brown": (120, 70, 40),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "cyan": (40, 200, 200),
    "teal": (30, 140, 140),
    "navy": (20, 40, 100),
    "gold": (210, 175, 40),
    "silver": (192, 192, 192),
    "beige": (210, 190, 150),
    "dark": (40, 40, 40),
    "bright": (240, 240, 200),
    "warm": (220, 150, 80),
    "cool": (80, 140, 200),
}

COLOR_MATCH_MAX_DISTANCE: float = 90.0

# ---------------------------------------------------------------------------
# Method metadata written into every report
# ---------------------------------------------------------------------------
METHOD_SOURCES: List[str] = [
    "Bruce and Hillmann 2004",
    "Ochoa and Duval 2009",
    "EU Metadata Quality Assessment",
    "Park 2009",
]

METHOD_LIMITATIONS: str = (
    "Grounding is token overlap with upstream JSON, not human factuality. "
    "Timeliness is not scored. Thresholds are hypotheses pending human calibration. "
    "See docs/judger_references.md."
)
