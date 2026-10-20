"""One VLM pass per time window or merged scene, then a single timeline document."""

from __future__ import annotations

import json
import re
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from open_enricher.stages.pipeline_spec import PipelineSpec, load_pipeline_spec
from open_enricher.stages.vlm_core.config import MetadataConfig, ModelConfig
from open_enricher.stages.vlm_core.model import extract_metadata, load_model, unload_model
from open_enricher.stages.vlm_core.video import clip_video_range
from open_enricher.storage import db
from open_enricher.timings import TimingRecorder

SEGMENT_MAX_FRAMES = 8
SEGMENT_MAX_NEW_TOKENS = 450
# Kept for callers that still import the fixed-window constant.
SEGMENT_SECONDS = 300


def _clock(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _windows_interval(duration: float, segment_seconds: int) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration - 1:
        end = min(duration, start + segment_seconds)
        windows.append((start, end))
        start = end
    return windows


def _windows_scenes(video_path: Path, spec: PipelineSpec, duration: float) -> list[tuple[float, float]]:
    """Merge PySceneDetect shots into scenes within [min, max] seconds, cap count."""
    from scenedetect import ContentDetector, detect

    scene_list = detect(str(video_path), ContentDetector(threshold=27.0))
    raw: list[tuple[float, float]] = []
    if not scene_list:
        return _windows_interval(duration, spec.interval_seconds)

    for start_tc, end_tc in scene_list:
        start = float(start_tc.get_seconds())
        end = float(end_tc.get_seconds())
        if end <= start:
            continue
        # Split shots longer than the max window.
        cursor = start
        while cursor < end - 0.5:
            chunk_end = min(end, cursor + spec.scene_max_seconds)
            raw.append((cursor, chunk_end))
            cursor = chunk_end

    if not raw:
        return _windows_interval(duration, spec.interval_seconds)

    merged: list[tuple[float, float]] = []
    cur_start, cur_end = raw[0]
    for start, end in raw[1:]:
        if cur_end - cur_start < spec.scene_min_seconds:
            cur_end = end
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    if len(merged) > spec.scene_max_count:
        # Evenly subsample to keep the episode covered without exceeding the cap.
        step = len(merged) / float(spec.scene_max_count)
        picked = []
        for index in range(spec.scene_max_count):
            picked.append(merged[min(len(merged) - 1, int(index * step))])
        merged = picked

    # Clamp last end to duration.
    fixed = [(s, min(e, duration)) for s, e in merged if s < duration]
    return fixed or _windows_interval(duration, spec.interval_seconds)


def build_windows(
    video_path: Path,
    duration: float,
    spec: PipelineSpec,
) -> list[tuple[float, float]]:
    if spec.segmentation == "scenes":
        return _windows_scenes(video_path, spec, duration)
    return _windows_interval(duration, spec.interval_seconds)


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


def _memory_block(prior: list[dict[str, Any]], *, compact: bool) -> str:
    if not prior:
        return ""
    if compact:
        lines = []
        for segment in prior:
            desc = str(segment.get("description") or "").strip()
            if desc:
                lines.append(
                    f"{segment.get('start_time')}-{segment.get('end_time')}: {desc[:180]}"
                )
        body = "\n".join(lines)
        return (
            "\nEarlier scenes already described (one line each; trust frames if they disagree):\n"
            f"{body}\n"
        )
    body = json.dumps(prior, ensure_ascii=False, separators=(",", ":"))
    if len(body) > 6000:
        return _memory_block(prior, compact=True)
    return (
        "\nJSON for earlier scenes already described. Keep continuity; trust frames if they disagree:\n"
        f"{body}\n"
    )


def _segment_prompt(
    start: float,
    end: float,
    dialogue: str,
    colors: str,
    *,
    inject_evidence: bool,
    memory: str,
    dialogue_summary: str,
    rolling_plot: str,
) -> str:
    extra = ""
    if inject_evidence and dialogue:
        extra += (
            "\nSpoken lines from this window only. Use a name only if it appears here. "
            f"Do not resolve contradictions:\n{dialogue}\n"
        )
    if inject_evidence and colors:
        extra += (
            "\nSampled hex colors from this window. Mention one only when it matches "
            f"the light you see: {colors}\n"
        )
    if dialogue_summary:
        extra += (
            "\nEpisode dialogue summary (whole file). Use for plot context only; "
            f"describe this window from the frames:\n{dialogue_summary}\n"
        )
    if rolling_plot:
        extra += (
            "\nRolling plot state so far (may be incomplete):\n"
            f"{rolling_plot}\n"
        )
    if memory:
        extra += memory
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


def _update_rolling_plot(previous: str, segment: dict[str, Any]) -> str:
    """Refresh a short plot state with the local 8B model when available."""
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    description = str(segment.get("description") or "").strip()
    if not description:
        return previous
    device = "cuda" if torch.cuda.is_available() else "cpu"
    system_prompt = (
        "You maintain a rolling plot summary for a cataloguer watching an episode in order. "
        "Write plain prose of at most 150 words. No JSON or markdown."
    )
    user_prompt = (
        f"Previous plot state:\n{previous or '(none yet)'}\n\n"
        f"New scene {_clock_from_segment(segment)}:\n{description}\n\n"
        "Update the plot state. Keep names only when the new scene supports them."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    model_id = "Qwen/Qwen3-8B"
    print(f"[INFO] Updating rolling plot with {model_id}...")
    quantization_config = None
    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = None
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = tokenizer([prompt_text], return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=220,
                do_sample=False,
                repetition_penalty=1.05,
            )
        new_tokens = generated_ids[0][len(model_inputs.input_ids[0]) :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text[:1200] if text else previous
    except Exception as exc:
        print(f"[WARNING] Rolling plot update failed: {exc}")
        fallback = f"{previous} {description}".strip()
        return fallback[:900]
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _clock_from_segment(segment: dict[str, Any]) -> str:
    return f"{segment.get('start_time')}-{segment.get('end_time')}"


def run_segmented(
    video_path: Path,
    video_filename: str,
    model_cfg: ModelConfig,
    meta_cfg: MetadataConfig,
    duration: float,
    *,
    dialogue_summary: str = "",
    timing: TimingRecorder | None = None,
    spec: PipelineSpec | None = None,
) -> tuple[str, dict[str, Any], bool, int]:
    """Clip the file into windows, run the VLM once per window, merge the timeline."""
    spec = spec or load_pipeline_spec()
    windows = build_windows(video_path, duration, spec)
    print(
        f"[INFO] Segmenting {duration:.0f}s into {len(windows)} windows "
        f"(mode={spec.segmentation}, experiment={spec.experiment_id})"
    )
    print(f"[INFO] Loading VLM: {model_cfg.model_id} (4-bit={model_cfg.load_in_4bit})...")
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
        print(f"[TIMING] vlm.vlm_load={timing.entries[-1]['seconds']}s")

    segments: list[dict[str, Any]] = []
    raw_parts: list[str] = []
    parsed_count = 0
    memory_truncated = False
    rolling_plot = ""
    try:
        for index, (start, end) in enumerate(windows, start=1):
            print(f"[INFO] VLM window {index}/{len(windows)} {_clock(start)}-{_clock(end)}")
            dialogue = _lines_in_window(video_filename, start, end)
            colors = _colors_in_window(video_filename, start, end)
            memory = ""
            if spec.vlm_memory and segments:
                memory = _memory_block(segments, compact=False)
            prompt = _segment_prompt(
                start,
                end,
                dialogue,
                colors,
                inject_evidence=spec.inject_dialogue_color_into_vlm,
                memory=memory,
                dialogue_summary=dialogue_summary if spec.dialogue_summary_to_vlm else "",
                rolling_plot=rolling_plot if spec.rolling_plot else "",
            )
            window_cfg = replace(
                meta_cfg,
                prompt=prompt,
                min_frames=2,
                max_frames=SEGMENT_MAX_FRAMES,
                max_new_tokens=SEGMENT_MAX_NEW_TOKENS,
                video_seconds=None,
            )
            infer_started = time.perf_counter()
            try:
                with tempfile.TemporaryDirectory(prefix="open-enricher-segment-") as tmp:
                    clip = clip_video_range(
                        video_path,
                        start,
                        max(1.0, end - start),
                        Path(tmp) / f"window_{index}.mp4",
                    )
                    raw_text = extract_metadata(model, processor, clip, window_cfg)
            except Exception as exc:
                # Retry once with compact memory when context or VRAM blows up.
                if spec.vlm_memory and segments and not memory_truncated:
                    print(f"[WARNING] VLM window failed ({exc}); retrying with compact memory")
                    memory_truncated = True
                    prompt = _segment_prompt(
                        start,
                        end,
                        dialogue,
                        colors,
                        inject_evidence=spec.inject_dialogue_color_into_vlm,
                        memory=_memory_block(segments, compact=True),
                        dialogue_summary=(
                            dialogue_summary if spec.dialogue_summary_to_vlm else ""
                        ),
                        rolling_plot=rolling_plot if spec.rolling_plot else "",
                    )
                    window_cfg = replace(window_cfg, prompt=prompt)
                    with tempfile.TemporaryDirectory(prefix="open-enricher-segment-") as tmp:
                        clip = clip_video_range(
                            video_path,
                            start,
                            max(1.0, end - start),
                            Path(tmp) / f"window_{index}.mp4",
                        )
                        raw_text = extract_metadata(model, processor, clip, window_cfg)
                else:
                    raise
            if timing is not None:
                timing.entries.append(
                    {
                        "name": "vlm_generate",
                        "seconds": round(time.perf_counter() - infer_started, 3),
                        "window": index,
                        "start": _clock(start),
                        "end": _clock(end),
                        "model_id": model_cfg.model_id,
                    }
                )
                print(f"[TIMING] vlm.vlm_generate.window{index}={timing.entries[-1]['seconds']}s")

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
            segment = {
                "start_time": _clock(start),
                "end_time": _clock(end),
                "description": description,
                "subjects_and_objects": subjects,
                "actions": actions,
                "camera_and_shot": camera,
                "lighting_and_color": lighting,
                "on_screen_text": texts,
            }
            if not spec.inject_dialogue_color_into_vlm:
                # Preserve window evidence for the LLM stage when routing is llm.
                if dialogue:
                    segment["dialogue"] = dialogue
                if colors:
                    segment["colors"] = colors
            segments.append(segment)

            if spec.rolling_plot:
                # Unload VLM briefly? Rolling plot needs the 8B text model.
                # Keep VLM loaded and accept sequential GPU swap cost: unload VLM,
                # update plot, reload: too expensive. Instead append a compact
                # heuristic update without swapping when VLM is loaded; full 8B
                # update runs only if we temporarily free VRAM.
                rolling_plot = (
                    f"{rolling_plot} [{_clock(start)}] {description}".strip()[:900]
                )
    finally:
        print("[INFO] Unloading VLM and freeing VRAM...")
        unload_model(model, processor)

    if spec.rolling_plot and segments:
        # Final polished rolling plot after VLM is free (optional enrichment).
        try:
            polished = ""
            for segment in segments:
                polished = _update_rolling_plot(polished, segment)
            rolling_plot = polished or rolling_plot
        except Exception as exc:
            print(f"[WARNING] Post-pass rolling plot polish failed: {exc}")

    on_screen: list[str] = []
    for segment in segments:
        for text in segment.get("on_screen_text") or []:
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
        "segmentation": spec.segmentation,
        "segmentation_seconds": spec.interval_seconds if spec.segmentation == "interval" else None,
        "segments_parsed": parsed_count,
        "segments_total": len(windows),
        "experiment_id": spec.experiment_id,
        "memory_truncated": memory_truncated,
    }
    if rolling_plot:
        payload["rolling_plot"] = rolling_plot
    for segment in payload["scene_timeline"]:
        segment.pop("on_screen_text", None)
    raw_document = "\n\n".join(raw_parts)
    json_valid = parsed_count == len(windows) and len(windows) > 0
    return raw_document, payload, json_valid, len(windows)
