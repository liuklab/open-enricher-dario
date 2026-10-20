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
