"""Turn dialogue scenes into the frame plan Qwen will consume.

History slots are taken from the memory artifact when that artifact
belongs to this attempt and is not a fallback. Otherwise the reserve
returns to the current scenes.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from open_enricher import storage
from open_enricher.experiments.config import CACHE_ROOT
from open_enricher.graph import relations_for_interval
from open_enricher.manifest import matching_artifact
from open_enricher.stages.pipeline_spec import PipelineSpec
from open_enricher.stages.vlm_core.allocation import allocate_frames, control_frame_budget
from open_enricher.stages.vlm_core.frames import FRAME_FACTOR
from open_enricher.timeline import Interval


def _clock(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _intervals(scenes: list[dict[str, Any]]) -> list[Interval]:
    intervals: list[Interval] = []
    for index, scene in enumerate(scenes, start=1):
        start = float(scene.get("start_seconds") or 0)
        end = float(scene.get("end_seconds") or start)
        scene_id = str(scene.get("scene_id") or f"w{index:04d}")
        intervals.append(
            Interval(
                id=scene_id,
                start=start,
                end=max(end, start),
                kind="scene",
                source="dialogue",
            )
        )
    return intervals


def _grab(video: Path, timestamps: list[float], dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for index, stamp in enumerate(timestamps, start=1):
        target = dest / f"explicit_{index:03d}.jpg"
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{float(stamp):.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(target),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError):
            continue
        if target.is_file() and target.stat().st_size > 0:
            written.append(str(target))
    return written


def _history_stamps(video_filename: str, start: float, limit: int) -> list[float]:
    if limit <= 0:
        return []
    memory = matching_artifact(
        storage.get_output_dir("visual_memory") / f"{video_filename}_memory.json"
    )
    if memory is None or memory.get("status") == "fallback":
        return []
    for row in memory.get("retrievals") or []:
        if not isinstance(row, dict):
            continue
        if abs(float(row.get("start") or -1) - start) > 0.05:
            continue
        stamps = []
        for frame in row.get("frames") or []:
            if isinstance(frame, dict):
                try:
                    stamps.append(float(frame.get("timestamp")))
                except (TypeError, ValueError):
                    continue
        return stamps[:limit]
    return []


def prepare_scenes(
    video: Path,
    video_filename: str,
    scenes: list[dict[str, Any]],
    spec: PipelineSpec,
    *,
    sample_fps: float,
    video_fps: float = 24.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Attach per-call frame caps and, when ffmpeg can, explicit JPEGs.

    The returned plan is archived beside the VLM output. Scene records
    keep the original dialogue description. Relations and times go in
    ``prompt_extra``.
    """
    mode = spec.allocation_mode()
    if mode is None:
        return scenes, {"mode": None, "groups": []}
    intervals = _intervals(scenes)
    try:
        duration = max((item.end for item in intervals), default=0.0)
        budget = control_frame_budget(
            duration,
            spec.interval_seconds,
            video_fps=video_fps,
            sample_fps=sample_fps,
            min_frames=2,
            max_frames=spec.budgets.max_qwen_frames,
        )
    except Exception as exc:  # noqa: BLE001 - duration sampling still runs
        return scenes, {"mode": "duration", "scoring_failed": True, "reason": str(exc), "groups": []}
    narrative = None
    scoring_failed = False
    if mode == "adaptive":
        if any("narrative_score" not in scene for scene in scenes):
            scoring_failed = True
        else:
            narrative = [float(scene.get("narrative_score") or 0) for scene in scenes]
    memory_ok = matching_artifact(
        storage.get_output_dir("visual_memory") / f"{video_filename}_memory.json"
    )
    history = spec.history_fraction()
    if memory_ok is None or memory_ok.get("status") == "fallback":
        history = 0.0
    try:
        plan = allocate_frames(
            intervals,
            budget=budget,
            history_fraction=history,
            mode="duration" if scoring_failed else mode,
            min_frames=spec.budgets.coverage_reserve_frames,
            max_frames_per_call=spec.budgets.max_qwen_frames,
            max_seconds=float(spec.budgets.max_scene_seconds),
            narrative=narrative,
            scoring_failed=scoring_failed,
            pacing_cap=spec.budgets.pacing_cap,
        )
    except Exception as exc:  # noqa: BLE001
        return scenes, {"mode": "duration", "scoring_failed": True, "reason": str(exc), "groups": []}
    graph = matching_artifact(
        storage.get_output_dir("character_graph") / f"{video_filename}_graph.json"
    )
    by_id = {str(scene.get("scene_id")): scene for scene in scenes if scene.get("scene_id")}
    prepared: list[dict[str, Any]] = []
    for group in plan.get("groups") or []:
        members = [by_id[scene_id] for scene_id in group.get("scene_ids") or [] if scene_id in by_id]
        if not members:
            members = [
                scene
                for scene in scenes
                if abs(float(scene.get("start_seconds") or -1) - float(group["start"])) < 0.05
            ]
        description = "\n".join(
            str(scene.get("description") or "").strip() for scene in members if scene.get("description")
        )
        start = float(group["start"])
        end = float(group["end"])
        current_count = int(group.get("frames") or 0)
        per_call_history = 0
        if history > 0 and current_count > 0:
            per_call_history = max(0, spec.budgets.max_qwen_frames - current_count)
            per_call_history -= per_call_history % FRAME_FACTOR
        stamps = list(group.get("timestamps") or [])
        history_stamps = [
            stamp
            for stamp in _history_stamps(video_filename, start, per_call_history)
            if stamp not in stamps
        ]
        image_paths: list[str] = []
        if stamps or history_stamps:
            dest = CACHE_ROOT / video_filename / "explicit" / spec.experiment_id / str(group["call_id"])
            image_paths = _grab(video, history_stamps + stamps, dest)
        extra_lines = []
        if len(image_paths) != len(history_stamps) + len(stamps):
            # The clip sampler can recover when explicit decoding loses an image.
            image_paths = []
        if image_paths and history_stamps:
            shown = ", ".join(f"{stamp:.1f}s" for stamp in history_stamps)
            extra_lines.append(f"Earlier frames from this title, in order: {shown}.")
            extra_lines.append(
                f"The first {len(history_stamps)} images are from earlier scenes. "
                f"The final {len(stamps)} images show the current scene. "
                "Every described action must be visible in the current images. "
                "Earlier images can help establish continuity."
            )
        if image_paths and stamps:
            shown = ", ".join(f"{stamp:.1f}s" for stamp in stamps)
            extra_lines.append(f"Current frames at source times: {shown}.")
        if graph:
            relations = relations_for_interval(graph.get("edges") or [], start, end)
            if relations:
                extra_lines.append(
                    "Scene-local graph observations: " + json.dumps(relations[:8])
                )
                extra_lines.append(
                    "Treat speaker labels as local references. Co-presence supports visibility "
                    "during speech. Leave the recipient and relationship unresolved when ambiguous."
                )
        row = {
            "start_time": _clock(start),
            "end_time": _clock(end),
            "start_seconds": start,
            "end_seconds": end,
            "description": description,
            "scene_id": (group.get("scene_ids") or [None])[0],
            "scene_ids": list(group.get("scene_ids") or []),
            "max_frames": max(FRAME_FACTOR, current_count) if current_count else FRAME_FACTOR,
            "no_frame": bool(group.get("no_frame")),
            "image_paths": image_paths,
            "prompt_extra": "\n".join(extra_lines),
        }
        prepared.append(row)
    plan["history_returned"] = history == 0.0 and spec.history_fraction() > 0
    if not prepared:
        return scenes, plan
    return prepared, plan
