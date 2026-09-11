"""Large Language Model profiling stage for RecSys (Domain Layer)."""

from typing import List
from pydantic import BaseModel

# ==========================================
# 1. DEFINIZIONE DELLO SCHEMA (PYDANTIC)
# ==========================================
class RecSysMetadata(BaseModel):
    primary_genre: str
    sub_genres: List[str]
    pacing: str
    mood: List[str]

class SearchAndDiscovery(BaseModel):
    semantic_tags: List[str]
    main_topics: List[str]

class AudienceProfiling(BaseModel):
    target_age_group: str
    viewer_interests: List[str]

class ContentSummary(BaseModel):
    visual_style: str
    narrative_hook: str

class VideoProfile(BaseModel):
    """Il modello JSON finale perfetto e garantito."""
    recsys_metadata: RecSysMetadata
    search_and_discovery: SearchAndDiscovery
    audience_profiling: AudienceProfiling
    content_summary: ContentSummary