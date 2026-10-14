"""Configuration data structures for the pipeline (Data Layer)."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class PipelineConfig:
    """Strict configuration state for the pipeline execution."""
    mode: str
    target: str
    video: str
    timeout: int
    stages: list[str]

    @property
    def base_url(self) -> str:
        """Dynamically construct the target URL."""
        return f"http://{self.host}:{self.port}"