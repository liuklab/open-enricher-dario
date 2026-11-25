"""Resolve reference synopsis files for the external evaluator."""

from __future__ import annotations

from pathlib import Path

from open_enricher.experiments.config import REPO_ROOT

# Default location for human synopsis summaries next to the repository root.
DEFAULT_SYNOPSIS_DIR = REPO_ROOT / "synopsis"


def synopsis_path(video_filename: str, synopsis_dir: Path | None = None) -> Path:
    """
    Return the expected synopsis file path for a video.

    Args:
        video_filename: Exact video file name under ``video_inputs``.
        synopsis_dir: Directory holding ``<video>_summary.txt`` files.
            Defaults to ``<repo>/synopsis``.

    Returns:
        Path to ``<video>_summary.txt``.
    """
    root = synopsis_dir if synopsis_dir is not None else DEFAULT_SYNOPSIS_DIR
    return root / f"{video_filename}_summary.txt"


def load_synopsis(
    video_filename: str,
    *,
    synopsis_dir: Path | None = None,
) -> str:
    """
    Load and return synopsis text for a video.

    Args:
        video_filename: Exact video file name under ``video_inputs``.
        synopsis_dir: Optional override for the synopsis directory.

    Returns:
        Non-empty synopsis string.

    Raises:
        FileNotFoundError: When the synopsis file is missing.
        ValueError: When the synopsis file is empty.
    """
    path = synopsis_path(video_filename, synopsis_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Missing synopsis file: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty synopsis file: {path}")
    return text
