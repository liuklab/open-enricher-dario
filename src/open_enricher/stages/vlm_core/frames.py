"""Explainability copies of the frames a Qwen3-VL window would sample.

The VLM reads a temporary clip and samples frames in memory. This module
repeats that index rule on the same clip and writes JPEGs plus timestamp
records. The model call does not read those files.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from open_enricher.experiments.config import CACHE_ROOT, REPO_ROOT
from open_enricher.logging_setup import get_logger

logger = get_logger(__name__)

# Qwen video processors pair frames (temporal patch size 2). Counts are
# rounded down to a multiple of this factor.
FRAME_FACTOR = 2


def floor_by_factor(value: float, factor: int) -> int:
    """Round ``value`` down to a multiple of ``factor``."""
    return int(value // factor * factor)


def ceil_by_factor(value: float, factor: int) -> int:
    """Round ``value`` up to a multiple of ``factor``."""
    return int(-(-int(value) // factor) * factor)


def sampled_frame_count(
    total_frames: int,
    video_fps: float,
    sample_fps: float,
    min_frames: int,
    max_frames: int,
    *,
    frame_factor: int = FRAME_FACTOR,
) -> int:
    """
    Return how many frames Qwen's ``smart_nframes`` keeps for one clip.

    Args:
        total_frames: Decoded frame count of the clip the model reads.
        video_fps: Frame rate of that clip.
        sample_fps: Requested sampling rate (``vlm_params.json`` ``fps``).
        min_frames: Lower bound before the factor rounding.
        max_frames: Upper bound before the factor rounding.
        frame_factor: Temporal patch size. Qwen uses 2.

    Returns:
        Frame count actually indexed. A 300 s window at ``sample_fps=2``
        asks for hundreds of frames and is then capped at ``max_frames``.
    """
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if video_fps <= 0:
        raise ValueError(f"video_fps must be positive, got {video_fps}")
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be positive, got {sample_fps}")

    lower = ceil_by_factor(min_frames, frame_factor)
    upper = floor_by_factor(min(max_frames, total_frames), frame_factor)
    if upper < frame_factor:
        upper = min(total_frames, frame_factor)
    estimated = total_frames / video_fps * sample_fps
    count = min(max(estimated, lower), upper, total_frames)
    count = floor_by_factor(count, frame_factor)
    if count < 1:
        count = 1
    return int(min(count, total_frames))


def frame_indices(total_frames: int, nframes: int) -> list[int]:
    """
    Evenly spaced indices from the first frame to the last, inclusive.

    Matches ``linspace(0, total_frames - 1, nframes)`` rounded half to even,
    which is what the Qwen video processor uses.

    Args:
        total_frames: Decoded frame count of the clip.
        nframes: Count returned by ``sampled_frame_count``.

    Returns:
        Increasing frame indices into the clip.
    """
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if nframes <= 0:
        raise ValueError(f"nframes must be positive, got {nframes}")
    if nframes == 1:
        return [0]
    span = total_frames - 1
    return [int(round(span * step / (nframes - 1))) for step in range(nframes)]


def probe_clip(path: Path) -> tuple[int, float]:
    """
    Read the clip frame count and average frame rate with ffprobe.

    Args:
        path: Re-encoded window the VLM is about to read.

    Returns:
        ``(total_frames, video_fps)``.

    Raises:
        RuntimeError: When ffprobe is missing or returns no video stream.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,avg_frame_rate,duration",
        "-of", "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required to count frames in the VLM clip") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        raise RuntimeError(f"ffprobe failed for {path}: {detail}") from exc

    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe returned no video stream for {path}")
    stream = streams[0]
    video_fps = _parse_rate(str(stream.get("avg_frame_rate") or ""))
    duration = float(stream.get("duration") or 0)
    raw_count = stream.get("nb_frames")
    total = int(raw_count) if isinstance(raw_count, str) and raw_count.isdigit() else 0
    if total <= 0 and video_fps > 0 and duration > 0:
        total = int(round(duration * video_fps))
    if total <= 0 or video_fps <= 0:
        raise RuntimeError(f"ffprobe could not measure frames for {path}")
    return total, video_fps


def scene_frame_dir(video_filename: str, experiment_id: str, scene_index: int) -> Path:
    """
    Return the cache directory for one scene of one experiment.

    Args:
        video_filename: Exact video file name.
        experiment_id: Active experiment id. Empty becomes ``default``.
        scene_index: 1-based window number.

    Returns:
        ``runs/local/_cache/<video>/frames/<experiment_id>/scene_XX``.
    """
    experiment = experiment_id.strip() or "default"
    return (
        CACHE_ROOT
        / video_filename
        / "frames"
        / experiment
        / f"scene_{scene_index:02d}"
    )


def save_scene_frames(
    clip: Path,
    dest_dir: Path,
    indices: list[int],
) -> list[Path]:
    """
    Write one JPEG per sampled index from the clip, without changing the clip.

    Args:
        clip: Temporary window file the VLM reads.
        dest_dir: Scene directory under the frame cache.
        indices: Frame indices into ``clip``, in sample order.

    Returns:
        Written JPEG paths, in sample order.

    Raises:
        RuntimeError: When ffmpeg is missing or writes no images.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for old in dest_dir.glob("frame_*.jpg"):
        old.unlink()
    if not indices:
        return []

    expression = "+".join(f"eq(n\\,{index})" for index in indices)
    pattern = dest_dir / "frame_%02d.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(clip),
        "-vf", f"select='{expression}'",
        "-vsync", "vfr",
        str(pattern),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to save VLM sample frames") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg failed to save frames from {clip}: {detail}") from exc

    written = sorted(dest_dir.glob("frame_*.jpg"))
    if len(written) != len(indices):
        raise RuntimeError(
            f"Expected {len(indices)} frames from {clip.name}, wrote {len(written)}"
        )
    return written


def sample_scene(
    clip: Path,
    *,
    video_filename: str,
    experiment_id: str,
    scene_index: int,
    start_seconds: float,
    sample_fps: float,
    min_frames: int,
    max_frames: int,
) -> dict[str, Any]:
    """
    Probe one clip, save the frames Qwen would index, and describe them.

    Args:
        clip: Temporary window file the VLM reads.
        video_filename: Exact video file name.
        experiment_id: Active experiment id.
        scene_index: 1-based window number.
        start_seconds: Window start on the original film clock.
        sample_fps: Requested sampling rate passed to the processor.
        min_frames: Lower bound passed to the processor for this window.
        max_frames: Upper bound passed to the processor for this window.

    Returns:
        One span record: indices, film timestamps, and cache-relative paths.
    """
    total_frames, video_fps = probe_clip(clip)
    count = sampled_frame_count(
        total_frames,
        video_fps,
        sample_fps,
        min_frames,
        max_frames,
    )
    indices = frame_indices(total_frames, count)
    dest = scene_frame_dir(video_filename, experiment_id, scene_index)
    written = save_scene_frames(clip, dest, indices)
    frames = []
    for sample_index, (frame_index, path) in enumerate(zip(indices, written), start=1):
        clip_seconds = frame_index / video_fps
        frames.append(
            {
                "sample": sample_index,
                "index": frame_index,
                "clip_seconds": round(clip_seconds, 3),
                "timestamp_seconds": round(start_seconds + clip_seconds, 3),
                "file": str(path.relative_to(REPO_ROOT)),
            }
        )
    return {
        "name": "vlm_frames",
        "window": scene_index,
        "clip_frames": total_frames,
        "clip_fps": round(video_fps, 6),
        "sampled_frames": count,
        "frames": frames,
    }


def _parse_rate(value: str) -> float:
    """Parse an ffprobe rational frame rate such as ``24000/1001``."""
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denom = float(denominator)
        if denom == 0:
            return 0.0
        return float(numerator) / denom
    return float(value)
