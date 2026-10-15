"""Pipeline orchestration and job execution (Domain Layer)."""

from __future__ import annotations

import json
from pathlib import Path

from src.open_enricher.client.pipeline_cmd import PipelineConfig
from src.open_enricher.client.network import PipelineClient

OUTPUT_DIR = Path("outputs")


def save_stage_result(video: str, stage: str, payload: object) -> Path:
    """Write one stage document under outputs/ on the machine that launched the client."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{video}_{stage}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


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

            if isinstance(result, dict) and result.get("status") == "failed":
                print(json.dumps(result, indent=2, ensure_ascii=False))
                error_msg = result.get("error", "Stage returned failed status.")
                raise RuntimeError(f"Stage '{name}' failed: {error_msg}")

            document = result.get("result", result) if isinstance(result, dict) else result
            print(json.dumps(document, indent=2, ensure_ascii=False))
            if isinstance(result, dict) and "result" in result:
                saved = save_stage_result(self.config.video, name, result["result"])
                print(f"[INFO] Saved {name} result to {saved.resolve()}")
            if isinstance(result, dict) and "timings" in result:
                timing_path = OUTPUT_DIR / f"{self.config.video}_{name}_timings.json"
                timing_path.write_text(
                    json.dumps(result["timings"], indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                print(f"[INFO] Saved {name} timings to {timing_path.resolve()}")

        return results