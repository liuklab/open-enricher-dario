"""Local run archive under runs/local and progress resume helpers.

The experiment loop writes per-variant folders here, appends progress
events for resume, and merges per-stage timing documents after a run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from open_enricher.experiments.config import CACHE_ROOT, PROGRESS_PATH, RUNS_ROOT


def experiment_dir(experiment_id: str) -> Path:
    """
    Return (and create) the archive folder for one experiment id.

    Args:
        experiment_id: Stable id from the variant JSON ``experiment_id`` field.

    Returns:
        Path ``runs/local/<experiment_id>/``.
    """
    path = RUNS_ROOT / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def video_archive_dir(experiment_id: str, video_filename: str) -> Path:
    """
    Return (and create) the archive folder for one video inside an experiment.

    Args:
        experiment_id: Stable experiment id.
        video_filename: Exact video file name (used as a subfolder name).

    Returns:
        Path ``runs/local/<experiment_id>/<video_filename>/``.
    """
    path = experiment_dir(experiment_id) / video_filename
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir(video_filename: str) -> Path:
    """
    Return (and create) the shared cache folder for one video.

    Args:
        video_filename: Exact video file name.

    Returns:
        Path ``runs/local/_cache/<video_filename>/``.
    """
    path = CACHE_ROOT / video_filename
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, document: Any) -> Path:
    """
    Serialize a document as indented UTF-8 JSON.

    Args:
        path: Destination file.
        document: JSON-serializable value.

    Returns:
        The same ``path`` for chaining.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def append_progress(event: dict[str, Any]) -> None:
    """
    Append one progress event to ``runs/local/progress.jsonl``.

    Args:
        event: Dict describing experiment_id, step, status, and optional
            details. A UTC ``ts`` field is added when missing.
    """
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = dict(event)
    row.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with PROGRESS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_progress() -> list[dict[str, Any]]:
    """
    Load all progress events from the JSONL resume log.

    Returns:
        List of parsed event dicts. Malformed lines are skipped.
    """
    if not PROGRESS_PATH.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in PROGRESS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def is_done(experiment_id: str, step: str = "archived") -> bool:
    """
    Report whether an experiment already completed a named step successfully.

    Args:
        experiment_id: Stable experiment id.
        step: Progress step name (default ``archived``).

    Returns:
        True when a matching ``status=ok`` event exists in the progress log.
    """
    for row in load_progress():
        if (
            row.get("experiment_id") == experiment_id
            and row.get("step") == step
            and row.get("status") == "ok"
        ):
            return True
    return False


def _recorded_fingerprint(experiment_id: str, video: str) -> str:
    """Return the source fingerprint stored on an archived manifest, or ""."""
    path = RUNS_ROOT / experiment_id / video / f"{video}_manifest.json"
    if not path.is_file():
        return ""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(document, dict):
        return ""
    return str(document.get("source_fingerprint") or "")


def archive_matches_sources(
    experiment_id: str,
    videos: list[str],
    step: str,
) -> bool:
    """
    Report whether the latest successful step covers these source files.

    A shorter file that shares a video name is not the same source. The
    progress row must name every requested video, and each archived manifest
    must still match the file now under ``video_inputs``.
    """
    if not videos:
        return False
    latest: dict[str, Any] | None = None
    for row in load_progress():
        if (
            row.get("experiment_id") == experiment_id
            and row.get("step") == step
            and row.get("status") == "ok"
        ):
            latest = row
    if latest is None:
        return False
    covered = {
        str(item.get("video"))
        for item in (latest.get("results") or [])
        if isinstance(item, dict) and item.get("video")
    }
    if not set(videos) <= covered:
        return False
    from open_enricher.fingerprint import source_fingerprint
    from open_enricher import storage

    for video in videos:
        recorded = _recorded_fingerprint(experiment_id, video)
        source = storage.video_inputs() / video
        if not recorded or not source.is_file():
            return False
        if recorded != source_fingerprint(source):
            return False
    return True


def retain_different_source(experiment_id: str, video: str, source: Path) -> Path | None:
    """
    Move an archive aside when it was built from a different file.

    The previous folder keeps its profile. The caller can then archive the
    current source under the original video name.
    """
    from open_enricher.fingerprint import source_fingerprint

    folder = RUNS_ROOT / experiment_id / video
    recorded = _recorded_fingerprint(experiment_id, video)
    if not recorded or not source.is_file():
        return None
    if recorded == source_fingerprint(source):
        return None
    kept = folder.with_name(f"{video}.source-{recorded[:12]}")
    if kept.exists():
        return kept
    folder.rename(kept)
    return kept


def list_variants(variants_dir: Path) -> list[Path]:
    """
    List variant JSON files in sorted order.

    Args:
        variants_dir: Directory containing ``*.json`` variant specs.

    Returns:
        Sorted list of variant paths.
    """
    return sorted(variants_dir.glob("*.json"))


def merge_timings(paths: Iterable[Path]) -> dict[str, Any]:
    """
    Merge per-stage timing JSON files into one document.

    Args:
        paths: Candidate timing file paths. Missing or invalid files are skipped.

    Returns:
        Dict with a ``stages`` list of timing documents.
    """
    documents = []
    for path in paths:
        if path.is_file():
            try:
                documents.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
    return {"stages": documents}
