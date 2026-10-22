"""One VLM pass per dialogue interval, with earlier metadata passed back in."""

from __future__ import annotations

import json
import re
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from open_enricher.logging_setup import get_logger
from open_enricher.stages.prompt_load import fill_prompt, load_prompt
from open_enricher.stages.vlm_core.config import MetadataConfig, ModelConfig
from open_enricher.stages.vlm_core.model import extract_metadata, load_model, unload_model
from open_enricher.stages.vlm_core.video import clip_video_range
from open_enricher.timings import TimingRecorder

logger = get_logger(__name__)

SEGMENT_MAX_FRAMES = 8
SEGMENT_MAX_NEW_TOKENS = 450
# Prior scene JSON is re-sent on every call. Past this size, keep a short
# description per interval so the current frames still fit.
MEMORY_CHAR_BUDGET = 8000


def _parse_object(raw_text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    candidate = re.sub(r",\s*([}\]])", r"\1", cleaned[start : end + 1])
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()][:6]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _memory_block(prior: list[dict[str, Any]]) -> tuple[str, bool]:
    """Return the prompt block of earlier metadata and whether it was compacted."""
    if not prior:
        return "", False
    body = json.dumps(prior, ensure_ascii=False, separators=(",", ":"))
    compacted = False
    if len(body) > MEMORY_CHAR_BUDGET:
        compacted = True
        compact = [
            {
                "start_time": segment.get("start_time"),
                "end_time": segment.get("end_time"),
                "description": str(segment.get("description") or "")[:180],
            }
            for segment in prior
        ]
        body = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        logger.warning(
            "ALERT: prior VLM metadata does not fit the memory budget; "
            "passing compact descriptions (chars=%s budget=%s scenes=%s)",
            len(body),
            MEMORY_CHAR_BUDGET,
            len(prior),
        )
    block = fill_prompt(load_prompt("vlm_scene_memory.md"), {"PRIOR_METADATA": body})
    return "\n" + block + "\n", compacted


def _scene_prompt(scene: dict[str, Any], memory: str) -> str:
    return fill_prompt(
        load_prompt("vlm_scene_prompt.md"),
        {
            "START": str(scene.get("start_time") or ""),
            "END": str(scene.get("end_time") or ""),
            "SCENE_DESCRIPTION": str(scene.get("description") or "").strip(),
            "MEMORY": memory,
        },
    )


def run_segmented(
    video_path: Path,
    video_filename: str,
    model_cfg: ModelConfig,
    meta_cfg: MetadataConfig,
    scenes: list[dict[str, Any]],
    *,
    timing: TimingRecorder | None = None,
    experiment_id: str = "",
) -> tuple[str, dict[str, Any], bool, int]:
    """
    Run the VLM once per dialogue interval and accumulate the metadata.

    Each call receives the dialogue description of that interval and the
    metadata written for earlier intervals. Qwen does not keep that record
    internally between ``generate`` calls.

    Args:
        video_path: Source video.
        video_filename: File name, used only in the saved payload.
        model_cfg: VLM load configuration.
        meta_cfg: Sampling and generation configuration.
        scenes: Dialogue-stage scene objects with start/end seconds and a
            prose ``description``.
        timing: Optional recorder for load and per-window generate spans.
        experiment_id: Active experiment id copied into the payload.

    Returns:
        Raw joined text, parsed timeline document, whether every window
        parsed as JSON, and the window count.
    """
    if not scenes:
        raise ValueError(f"No dialogue scenes to show the VLM for {video_filename}")

    logger.info(
        "VLM over %s scenes (experiment=%s)",
        len(scenes),
        experiment_id or "default",
    )
    logger.info("Loading VLM: %s (4-bit=%s)...", model_cfg.model_id, model_cfg.load_in_4bit)
    load_started = time.perf_counter()
    model, processor = load_model(model_cfg)
    if timing is not None:
        timing.entries.append(
            {
                "name": "vlm_load",
                "seconds": round(time.perf_counter() - load_started, 3),
                "model_id": model_cfg.model_id,
            }
        )
        logger.info("TIMING vlm.vlm_load=%ss", timing.entries[-1]["seconds"])

    segments: list[dict[str, Any]] = []
    raw_parts: list[str] = []
    parsed_count = 0
    memory_compacted = False
    try:
        for index, scene in enumerate(scenes, start=1):
            start = float(scene["start_seconds"])
            end = float(scene["end_seconds"])
            logger.info(
                "VLM scene %s/%s %s-%s",
                index,
                len(scenes),
                scene.get("start_time"),
                scene.get("end_time"),
            )
            memory, compacted = _memory_block(segments)
            memory_compacted = memory_compacted or compacted
            prompt = _scene_prompt(scene, memory)
            window_cfg = replace(
                meta_cfg,
                prompt=prompt,
                min_frames=2,
                max_frames=SEGMENT_MAX_FRAMES,
                max_new_tokens=SEGMENT_MAX_NEW_TOKENS,
                video_seconds=None,
            )
            infer_started = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix="open-enricher-segment-") as tmp:
                clip = clip_video_range(
                    video_path,
                    start,
                    max(1.0, end - start),
                    Path(tmp) / f"window_{index}.mp4",
                )
                raw_text = extract_metadata(model, processor, clip, window_cfg)
            if timing is not None:
                timing.entries.append(
                    {
                        "name": "vlm_generate",
                        "seconds": round(time.perf_counter() - infer_started, 3),
                        "window": index,
                        "start": scene.get("start_time"),
                        "end": scene.get("end_time"),
                        "model_id": model_cfg.model_id,
                    }
                )
                logger.info(
                    "TIMING vlm.vlm_generate.window%s=%ss",
                    index,
                    timing.entries[-1]["seconds"],
                )

            raw_parts.append(f"--- {scene.get('start_time')}-{scene.get('end_time')} ---\n{raw_text}")
            parsed = _parse_object(raw_text)
            if parsed is None:
                description = raw_text.strip()[:500] or "No description."
                subjects: list[str] = []
                actions: list[str] = []
                camera = ""
                lighting = ""
                texts: list[str] = []
            else:
                parsed_count += 1
                description = str(parsed.get("description") or "").strip() or "No description."
                subjects = _as_list(parsed.get("subjects_and_objects"))
                actions = _as_list(parsed.get("actions"))
                camera = str(parsed.get("camera_and_shot") or "").strip()
                lighting = str(parsed.get("lighting_and_color") or "").strip()
                texts = _as_list(parsed.get("on_screen_text"))
            segments.append(
                {
                    "start_time": scene.get("start_time"),
                    "end_time": scene.get("end_time"),
                    "dialogue_description": str(scene.get("description") or "").strip(),
                    "description": description,
                    "subjects_and_objects": subjects,
                    "actions": actions,
                    "camera_and_shot": camera,
                    "lighting_and_color": lighting,
                    "on_screen_text": texts,
                }
            )
    finally:
        logger.info("Unloading VLM and freeing VRAM...")
        unload_model(model, processor)

    payload = {
        "experiment_id": experiment_id,
        "scene_timeline": segments,
        "segments_parsed": parsed_count,
        "segments_total": len(scenes),
        "memory_compacted": memory_compacted,
    }
    raw_document = "\n\n".join(raw_parts)
    json_valid = parsed_count == len(scenes) and len(scenes) > 0
    return raw_document, payload, json_valid, len(scenes)
