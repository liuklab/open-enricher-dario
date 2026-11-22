"""Pydantic schema for the final RecSys video profile.

Shared by the LLM stage (writer) and the evaluation package (reader) so the
scored shape cannot drift from the generated shape. Depends only on pydantic.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class RecSysMetadata(BaseModel):
    primary_genre: Literal[
        "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
        "Drama", "Education", "Fantasy", "Horror", "Music", "Mystery", "News",
        "Romance", "Sci-Fi", "Science Fiction", "Sports", "Thriller", "Vlog", "Western",
    ] = Field(..., description="Main genre, e.g. Action, Comedy, Education, Vlog")
    sub_genres: List[str] = Field(..., description="1 to 4 specific sub-genres")
    pacing: Literal["Fast", "Moderate", "Slow"] = Field(..., description="Fast, Moderate, or Slow")
    mood: List[str] = Field(..., description="1 to 4 mood adjectives")


class SearchAndDiscovery(BaseModel):
    semantic_tags: List[str] = Field(..., description="3 to 6 search keywords/tags")
    main_topics: List[str] = Field(..., description="1 to 4 main topics discussed or shown")


class AudienceProfiling(BaseModel):
    target_age_group: Literal[
        "All Ages", "Children", "Teens", "Young Adults", "Gen Z", "Millennials", "Adults", "Family",
    ] = Field(..., description="e.g. Gen Z, Millennials, Young Adults, All Ages")
    viewer_interests: List[str] = Field(..., description="1 to 4 target audience interests")


class ContentSummary(BaseModel):
    visual_style: str = Field(..., description="Concise description of aesthetic and color palette")
    narrative_hook: str = Field(..., description="One sentence describing what makes the video engaging")


class VideoProfile(BaseModel):
    """Final JSON schema validated for the Recommender System."""

    recsys_metadata: RecSysMetadata
    search_and_discovery: SearchAndDiscovery
    audience_profiling: AudienceProfiling
    content_summary: ContentSummary
