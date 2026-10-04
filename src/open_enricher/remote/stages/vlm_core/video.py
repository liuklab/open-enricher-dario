"""Video probing and prefix clipping utilities via ffprobe and ffmpeg."""

from __future__ import annotations

from pathlib import Path
import subprocess


def probe_duration_seconds(path: str | Path) -> float:
    """Return the container duration in seconds."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required to read video duration before clipping") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        raise RuntimeError(f"ffprobe failed for {path}: {detail}") from exc

    raw = result.stdout.strip()
    try:
        duration = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"ffprobe returned no duration for {path}: {raw!r}") from exc
    if duration <= 0:
        raise RuntimeError(f"ffprobe duration must be positive for {path}, got {duration}")
    return duration


def clip_video_prefix(src: str | Path, seconds: int, dest: str | Path) -> Path:
    """Re-encode the first `seconds` of `src` into `dest`, without audio."""
    if type(seconds) is not int or seconds <= 0:
        raise ValueError(f"seconds must be a positive int, got {seconds!r}")

    source = Path(src)
    if not source.is_file():
        raise FileNotFoundError(f"Video file does not exist: {source}")

    output = Path(dest)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(source),
        "-t", str(seconds),
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        str(output),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to clip the input video") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg failed to clip {source}: {detail}") from exc

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg wrote no clip for {source}")
    return output