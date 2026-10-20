"""Deterministic metadata quality assurance stage.

Fifth pipeline stage. Scores the RecSys ``VideoProfile`` from the LLM stage
against upstream audio, color, and VLM evidence using explicit rules. No
second model call: the same inputs always produce the same report.

See ``docs/judger_references.md`` for the quality frameworks behind each
dimension (Bruce and Hillmann, Ochoa and Duval, EU MQA, Park).
"""

from __future__ import annotations

import json
from typing import Any, Dict

from open_enricher.stages.judger_core import score_profile
from open_enricher.storage import db


class JudgerStage:
    """Evaluates completeness, conformance, consistency, specificity, and grounding."""

    name = "judger"

    def __init__(self) -> None:
        # No model weights: this stage is CPU-only and deterministic.
        pass

    def _load_json(self, path) -> Any:
        """Load a JSON file; return None when missing or unreadable."""
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError) as err:
            print(f"[WARNING] Failed to parse {path.name}: {err}")
            return None

    def _gather_evidence(self, video_filename: str) -> Dict[str, Any]:
        """Collect upstream stage JSON the same way ``LLMStage._gather_metadata`` does."""
        evidence: Dict[str, Any] = {}
        expected_files = {
            "audio": db.get_output_dir("audio") / f"{video_filename}_audio.json",
            "color": db.get_output_dir("color") / f"{video_filename}_color.json",
            "vlm": db.get_output_dir("vlm") / f"{video_filename}_vlm.json",
        }
        for stage_name, file_path in expected_files.items():
            payload = self._load_json(file_path)
            if payload is not None:
                print(f"[INFO] Found evidence for stage: {stage_name}")
                evidence[stage_name] = payload
            else:
                print(f"[INFO] No evidence found for stage: {stage_name}. Skipping.")
        return evidence

    def run(self, video_filename: str) -> Dict[str, Any]:
        """Score ``<video>_llm.json`` and write ``<video>_judger.json``."""
        print(f"[INFO] Starting Judger validation stage for {video_filename}")

        profile_path = db.get_output_dir("llm") / f"{video_filename}_llm.json"
        if not profile_path.exists():
            return {
                "stage": self.name,
                "status": "failed",
                "error": f"Missing LLM profile: {profile_path.name}",
                "output_file": None,
            }

        profile_data = self._load_json(profile_path)
        if profile_data is None:
            return {
                "stage": self.name,
                "status": "failed",
                "error": f"Unreadable LLM profile: {profile_path.name}",
                "output_file": None,
            }

        evidence = self._gather_evidence(video_filename)
        report = score_profile(
            profile_data,
            evidence,
            video=video_filename,
            profile_file=profile_path.name,
        )

        output_path = db.get_output_dir(self.name) / f"{video_filename}_judger.json"
        print(f"[INFO] Saving judger report to {output_path}")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)

        dimension_scores = {
            name: payload["score"] for name, payload in report["dimensions"].items()
        }
        return {
            "stage": self.name,
            "status": "success",
            "overall": report["overall"],
            "usable": report["usable"],
            "dimensions": dimension_scores,
            "output_file": str(output_path.name),
        }
