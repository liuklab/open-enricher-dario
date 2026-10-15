"""One VLM pass per fixed time window, then a single timeline document."""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.open_enricher.remote.stages.vlm_core.config import MetadataConfig, ModelConfig
from src.open_enricher.remote.stages.vlm_core.model import extract_metadata, load_model, unload_model
from src.open_enricher.remote.stages.vlm_core.video import clip_video_range
from src.open_enricher.remote.storage import db

SEGMENT_SECONDS = 300
SEGMENT_MAX_FRAMES = 8
SEGMENT_MAX_NEW_TOKENS = 450


def _clock(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _windows(duration: float, segment_seconds: int) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration - 1:
        end = min(duration, start + segment_seconds)
        windows.append((start, end))
        start = end
    return windows


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


def _lines_in_window(video_filename: str, start: float, end: float) -> str:
    path = db.get_output_dir("audio") / f"{video_filename}_audio.json"
    if not path.is_file():
        return ""
    try:
        lines = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(lines, list):
        return ""
    picked: list[str] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        try:
            stamp = float(line.get("start_timestamp", -1))
        except (TypeError, ValueError):
            continue
        if stamp < start or stamp >= end:
            continue
        text = str(line.get("text") or "").strip()
        if text:
            picked.append(f"[{_clock(stamp)}] {text}")
        if len(picked) >= 12:
            break
    return "\n".join(picked)


def _colors_in_window(video_filename: str, start: float, end: float) -> str:
    path = db.get_output_dir("color") / f"{video_filename}_color.json"
    if not path.is_file():
        return ""
    try:
        samples = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(samples, list):
        return ""
    palettes: list[str] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        try:
            stamp = float(sample.get("timestamp", -1))
        except (TypeError, ValueError):
            continue
        if stamp < start or stamp >= end:
            continue
        palette = sample.get("color_palette") or []
        if isinstance(palette, list) and palette:
            palettes.append(", ".join(str(color) for color in palette[:3]))
        if len(palettes) >= 3:
            break
    return "; ".join(palettes)


def _segment_prompt(start: float, end: float, dialogue: str, colors: str) -> str:
    extra = ""
    if dialogue:
        extra += (
            "\nSpoken lines from this window only. Use a name only if it appears here. "
            f"Do not resolve contradictions:\n{dialogue}\n"
        )
    if colors:
        extra += (
            "\nSampled hex colors from this window. Mention one only when it matches "
            f"the light you see: {colors}\n"
        )
    return (
        f"These frames are only {_clock(start)} to {_clock(end)} of a longer file. "
        "Describe what is visible in this window. Do not describe other parts of the file. "
        "Return one JSON object and no other text, with these keys: "
        "description (string), subjects_and_objects (list of strings), actions (list of strings), "
        "camera_and_shot (string), lighting_and_color (string), on_screen_text (list of strings). "
        "lighting_and_color describes the image. on_screen_text is empty when nothing is readable."
        f"{extra}"
    )


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()][:6]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def run_segmented(
    video_path: Path,
    video_filename: str,
    model_cfg: ModelConfig,
    meta_cfg: MetadataConfig,
    duration: float,
) -> tuple[str, dict[str, Any], bool, int]:
    """Clip the file into fixed windows, run the VLM once per window, merge the timeline."""
    windows = _windows(duration, SEGMENT_SECONDS)
    print(f"[INFO] Segmenting {duration:.0f}s into {len(windows)} windows of {SEGMENT_SECONDS}s")
    print(f"[INFO] Loading VLM: {model_cfg.model_id} (4-bit={model_cfg.load_in_4bit})...")
    model, processor = load_model(model_cfg)
    segments: list[dict[str, Any]] = []
    raw_parts: list[str] = []
    parsed_count = 0
    try:
        for index, (start, end) in enumerate(windows, start=1):
            print(f"[INFO] VLM window {index}/{len(windows)} {_clock(start)}-{_clock(end)}")
            prompt = _segment_prompt(
                start,
                end,
                _lines_in_window(video_filename, start, end),
                _colors_in_window(video_filename, start, end),
            )
            window_cfg = replace(
                meta_cfg,
                prompt=prompt,
                min_frames=2,
                max_frames=SEGMENT_MAX_FRAMES,
                max_new_tokens=SEGMENT_MAX_NEW_TOKENS,
                video_seconds=None,
            )
            with tempfile.TemporaryDirectory(prefix="open-enricher-segment-") as tmp:
                clip = clip_video_range(
                    video_path,
                    start,
                    max(1.0, end - start),
                    Path(tmp) / f"window_{index}.mp4",
                )
                raw_text = extract_metadata(model, processor, clip, window_cfg)
            raw_parts.append(f"--- {_clock(start)}-{_clock(end)} ---\n{raw_text}")
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
                    "start_time": _clock(start),
                    "end_time": _clock(end),
                    "description": description,
                    "subjects_and_objects": subjects,
                    "actions": actions,
                    "camera_and_shot": camera,
                    "lighting_and_color": lighting,
                    "on_screen_text": texts,
                }
            )
    finally:
        print("[INFO] Unloading VLM and freeing VRAM...")
        unload_model(model, processor)

    on_screen: list[str] = []
    for segment in segments:
        for text in segment["on_screen_text"]:
            if text not in on_screen:
                on_screen.append(text)
    summary = " ".join(segment["description"] for segment in segments)
    payload = {
        "title_guess": "Segmented viewing of the file",
        "short_summary": summary[:1600],
        "content_type": "episode",
        "setting": "",
        "visual_style": "",
        "scene_timeline": segments,
        "on_screen_text": on_screen[:8],
        "segmentation_seconds": SEGMENT_SECONDS,
        "segments_parsed": parsed_count,
        "segments_total": len(windows),
    }
    # Drop the per-segment text lists from the timeline objects the prompt schema
    # did not ask the downstream LLM to treat as nested scene fields beyond description.
    for segment in payload["scene_timeline"]:
        segment.pop("on_screen_text", None)
    raw_document = "\n\n".join(raw_parts)
    json_valid = parsed_count == len(windows) and len(windows) > 0
    return raw_document, payload, json_valid, len(windows)
