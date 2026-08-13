"""Discover and resolve video files under a generic dataset root."""

from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def list_videos(root: str | Path) -> list[Path]:
    """Return video files under `root`, recursively, in sorted order."""

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Video root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Video root is not a directory: {root_path}")

    videos = [
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos)


def resolve_video(path_or_name: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a video from an absolute path, a relative path, or a file name.

    Search order:
    1. The given path if it already exists.
    2. `root / path_or_name` if `root` is set.
    3. A unique filename match under `root`.
    """

    candidate = Path(path_or_name)
    if candidate.exists() and candidate.is_file():
        return candidate.resolve()

    if root is None:
        raise FileNotFoundError(f"Video not found: {path_or_name}")

    rooted = Path(root) / candidate
    if rooted.exists() and rooted.is_file():
        return rooted.resolve()

    name = candidate.name
    matches = [path for path in list_videos(root) if path.name == name]
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple videos named {name!r} under {root}. Pass a more specific path."
        )
    raise FileNotFoundError(f"Video not found: {path_or_name} (root={root})")
