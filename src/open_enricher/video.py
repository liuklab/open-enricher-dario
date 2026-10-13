"""Discover and resolve video files in a dataset directory."""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def list_videos(root: str | Path) -> list[Path]:
    """Return video files in `root` only, in sorted order.

    Does not walk subdirectories. `root` must be an existing directory.
    """

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Video root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Video root is not a directory: {root_path}")

    # Flat listing only: Drive layout puts clips in one folder, not nested trees.
    videos = [
        path
        for path in root_path.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos)


def resolve_video(name: str | Path, root: str | Path) -> Path:
    """Resolve a video file name inside the dataset directory.

    `name` is a file name with extension, not a path. The file must sit
    directly in `root`.
    """

    filename = Path(name).name
    # Path.name strips directories, so compare parts to reject `subdir/clip.mp4`.
    if Path(name).parts != (filename,):
        raise FileNotFoundError(f"Expected a file name, not a path: {name}")

    candidate = Path(root) / filename
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Video not found: {filename} in {root}")


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
    """Re-encode the first `seconds` of `src` into `dest`, without audio.

    Re-encoding stops on a frame boundary. Stream copy would stop at the next
    keyframe and can keep more than `seconds`.
    """

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


@contextmanager
def video_input_path(path: str | Path, seconds: int | None) -> Iterator[Path]:
    """Yield the file the model should read.

    `None` yields the original file. A positive int yields a temporary re-encode
    of the first that many seconds. If the source is already that short, the
    original file is yielded and nothing is re-encoded.
    """

    source = Path(path)
    if seconds is None:
        yield source
        return
    if type(seconds) is not int or seconds <= 0:
        raise ValueError(f"video seconds must be a positive int or None, got {seconds!r}")
    if probe_duration_seconds(source) <= seconds:
        yield source
        return

    with tempfile.TemporaryDirectory(prefix="open-enricher-clip-") as tmp:
        dest = Path(tmp) / f"{source.stem}.mp4"
        yield clip_video_prefix(source, seconds, dest)
