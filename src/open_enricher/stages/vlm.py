"""Vision-Language Model stage orchestrator (Qwen3-VL).

Runs after the dialogue stage. For each fixed interval, Qwen3-VL sees the
clip, the dialogue description of that interval, and the metadata written
for earlier intervals. Those earlier records are passed in the prompt:
the model does not keep them between generate calls.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from open_enricher import storage
from open_enricher.logging_setup import get_logger
from open_enricher.stages.pipeline_spec import load_pipeline_spec
from open_enricher.stages.vlm_core.config import build_configs
from open_enricher.stages.vlm_core.segments import (
    SEGMENT_MAX_FRAMES,
    SEGMENT_MAX_NEW_TOKENS,
    run_segmented,
)
from open_enricher.timings import TimingRecorder

logger = get_logger(__name__)


class VLMStage:
    """Accumulates per-interval visual metadata with Qwen3-VL in 4-bit."""

    name = "vlm"

    def run(self, video_filename: str) -> Dict[str, Any]:
        """
        Run one VLM pass per dialogue scene and persist the accumulated JSON.

        Args:
            video_filename: Exact file name under ``video_inputs``.

        Returns:
            Status dict with output file names and parse success flags.

        Raises:
            FileNotFoundError: When the video or the dialogue JSON is missing.
            ValueError: When the dialogue file has no scenes.
        """
        video_path = storage.video_inputs() / video_filename
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found in dataset: {video_path}")
        dialogue_path = storage.get_output_dir("dialogue") / f"{video_filename}_dialogue.json"
        if not dialogue_path.is_file():
            raise FileNotFoundError(f"Dialogue stage output not found: {dialogue_path}")

        spec = load_pipeline_spec()
        timing = TimingRecorder(self.name)
        dialogue = json.loads(dialogue_path.read_text(encoding="utf-8"))
        scenes = dialogue.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            raise ValueError(f"Dialogue file has no scenes: {dialogue_path}")

        logger.info("Starting VLM stage for %s (%s)", video_filename, spec.experiment_id)
        model_cfg, meta_cfg = build_configs()
        raw_text, parsed_payload, is_valid_json, segment_count = run_segmented(
            video_path,
            video_filename,
            model_cfg,
            meta_cfg,
            scenes,
            timing=timing,
            experiment_id=spec.experiment_id,
            vlm_memory_scenes=spec.vlm_memory_scenes,
            visual_only=spec.visual_only,
        )

        output_dir = storage.get_output_dir(self.name)
        raw_path = output_dir / f"{video_filename}_vlm_raw.txt"
        raw_path.write_text(raw_text, encoding="utf-8")

        json_path = output_dir / f"{video_filename}_vlm.json"
        json_path.write_text(
            json.dumps(parsed_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        timings_path = output_dir / f"{video_filename}_vlm_timings.json"
        timing.save(timings_path)

        sidecar_path = output_dir / f"{video_filename}_vlm_sidecar.json"
        sidecar_payload = {
            "video": str(video_path),
            "model_id": model_cfg.model_id,
            "dtype": model_cfg.dtype,
            "load_in_4bit": model_cfg.load_in_4bit,
            "device_map": model_cfg.device_map,
            "attn_implementation": model_cfg.attn_implementation,
            "fps": meta_cfg.fps,
            "max_frames": SEGMENT_MAX_FRAMES,
            "max_new_tokens": SEGMENT_MAX_NEW_TOKENS,
            "do_sample": meta_cfg.do_sample,
            "repetition_penalty": meta_cfg.repetition_penalty,
            "json_valid": is_valid_json,
            "experiment_id": spec.experiment_id,
            "interval_seconds": spec.interval_seconds,
            "vlm_memory_scenes": spec.vlm_memory_scenes,
            "visual_only": spec.visual_only,
            "segment_count": segment_count,
            "memory_compacted": parsed_payload.get("memory_compacted"),
            "dialogue_file": dialogue_path.name,
            "timings_file": timings_path.name,
        }
        sidecar_path.write_text(
            json.dumps(sidecar_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info("VLM stage completed. Saved to %s (json_valid=%s)", json_path.name, is_valid_json)
        return {
            "stage": self.name,
            "status": "success",
            "json_valid": is_valid_json,
            "result": parsed_payload,
            "timings": timing.document(),
            "output_file": str(json_path.name),
            "raw_file": str(raw_path.name),
            "sidecar_file": str(sidecar_path.name),
            "timings_file": str(timings_path.name),
        }
