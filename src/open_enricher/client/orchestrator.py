"""Pipeline orchestration and job execution (Domain Layer)."""

from __future__ import annotations

import json

from src.open_enricher.client.pipeline_cmd import PipelineConfig
from src.open_enricher.client.network import PipelineClient

class PipelineOrchestrator:
    """Coordinates the sequential execution of pipeline stages."""
    
    def __init__(self, client: PipelineClient, config: PipelineConfig):
        self.client = client
        self.config = config

    def run_jobs(self) -> dict[str, object]:
        """Runs the requested stages in the exact order specified by the configuration."""
        results: dict[str, object] = {}
        
        for name in self.config.stages:
            result = self.client.execute_stage(name, self.config.video)
            results[name] = result
            
            # Print the structured JSON response from the server
            print(json.dumps(result, indent=2, ensure_ascii=True))
            
        return results