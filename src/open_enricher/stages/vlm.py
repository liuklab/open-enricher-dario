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
from open_enricher.fingerprint import current_stamp
from open_enricher.stages.pipeline_spec import load_pipeline_spec
from open_enricher.stages.vlm_core.config import build_configs
from open_enricher.stages.vlm_core.planned import prepare_scenes
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
            if not spec.features.any_enabled:
                raise ValueError(f"Dialogue file has no scenes: {dialogue_path}")
            scenes = []

        frame_plan: dict[str, Any] = {"mode": None}
        if spec.features.any_enabled and scenes:
            try:
                scenes, frame_plan = prepare_scenes(
                    video_path,
                    video_filename,
                    scenes,
                    spec,
                    sample_fps=2,
                )
            except Exception as exc:  # noqa: BLE001 - keep the dialogue windows
                logger.warning("Frame plan failed for %s: %s", video_filename, exc)
                frame_plan = {"mode": "duration", "scoring_failed": True, "reason": str(exc)}
        if not scenes:
            raise ValueError(f"Dialogue file has no scenes: {dialogue_path}")

        logger.info("Starting VLM stage for %s (%s)", video_filename, spec.experiment_id)
        model_cfg, meta_cfg = build_configs()
        try:
            raw_text, parsed_payload, is_valid_json, segment_count, frame_spans = run_segmented(
                video_path,
                video_filename,
                model_cfg,
                meta_cfg,
                scenes,
                timing=timing,
                experiment_id=spec.experiment_id,
                vlm_memory_scenes=spec.vlm_memory_scenes,
                visual_only=spec.visual_only,
                extract_visual=spec.narrate_from_visual,
                continue_on_error=spec.features.any_enabled,
            )
        except Exception as exc:
            if not spec.features.any_enabled:
                raise
            logger.warning("VLM failed for %s and will be skipped: %s", video_filename, exc)
            return _degraded_vlm(self.name, video_filename, spec.experiment_id, str(exc), timing)

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

        frames_path = output_dir / f"{video_filename}_vlm_frames.json"
        frames_wall = round(sum(float(span.get("seconds") or 0) for span in frame_spans), 3)
        frames_document = {
            "stage": "vlm_frames",
            "wall_seconds": frames_wall,
            "experiment_id": spec.experiment_id,
            "video": video_filename,
            "sample_fps": meta_cfg.fps,
            "min_frames": 2,
            "max_frames": SEGMENT_MAX_FRAMES,
            "frame_factor": 2,
            "spans": frame_spans,
        }
        frames_path.write_text(
            json.dumps(frames_document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        plan_path = output_dir / f"{video_filename}_frame_plan.json"
        plan_document = dict(frame_plan)
        plan_document.update(current_stamp())
        plan_document["video"] = video_filename
        plan_path.write_text(
            json.dumps(plan_document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

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
            "vlm_memory_scenes": 0 if spec.narrate_from_visual else spec.vlm_memory_scenes,
            "visual_only": spec.visual_only or spec.narrate_from_visual,
            "narrate_from_visual": spec.narrate_from_visual,
            "segment_count": segment_count,
            "memory_compacted": parsed_payload.get("memory_compacted"),
            "dialogue_file": dialogue_path.name,
            "timings_file": timings_path.name,
            "frames_file": frames_path.name,
        }
        sidecar_path.write_text(
            json.dumps(sidecar_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        stage_status = "success"
        if spec.features.any_enabled and (
            not is_valid_json
            or frame_plan.get("scoring_failed")
            or frame_plan.get("history_returned")
            or any(span.get("fallback") for span in frame_spans)
        ):
            stage_status = "degraded"
        logger.info("VLM stage completed. Saved to %s (json_valid=%s)", json_path.name, is_valid_json)
        return {
            "stage": self.name,
            "status": stage_status,
            "json_valid": is_valid_json,
            "result": parsed_payload,
            "timings": timing.document(),
            "output_file": str(json_path.name),
            "raw_file": str(raw_path.name),
            "sidecar_file": str(sidecar_path.name),
            "timings_file": str(timings_path.name),
            "frames_file": str(frames_path.name),
        }


def _degraded_vlm(
    stage: str,
    video_filename: str,
    experiment_id: str,
    reason: str,
    timing: TimingRecorder,
) -> dict[str, Any]:
    """Write an empty visual timeline so a stale file cannot reach the writer."""
    output_dir = storage.get_output_dir(stage)
    json_path = output_dir / f"{video_filename}_vlm.json"
    payload = {
        "experiment_id": experiment_id,
        "scene_timeline": [],
        "segments_parsed": 0,
        "segments_total": 0,
        "degraded_reason": reason,
    }
    payload.update(current_stamp())
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    timings_path = output_dir / f"{video_filename}_vlm_timings.json"
    timing.save(timings_path)
    return {
        "stage": stage,
        "status": "degraded",
        "json_valid": False,
        "error": reason,
        "output_file": json_path.name,
        "timings_file": timings_path.name,
    }
