"""Attempt manifest. Consumers accept only artifacts listed here.

Archives that have no manifest stay readable by filename, which is how
variants 01-08 are scored today.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from open_enricher import storage


def file_sha256(path: Path) -> str:
    """Return the hex sha256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_dir() -> Path:
    """Return the shared evidence output directory."""
    path = storage.get_output_dir("evidence")
    path.mkdir(parents=True, exist_ok=True)
    return path


def manifest_path(video_filename: str) -> Path:
    """Path of the live manifest for one video."""
    return evidence_dir() / f"{video_filename}_manifest.json"


def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON object, or return None when it is missing or invalid."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def matching_artifact(path: Path) -> dict[str, Any] | None:
    """
    Load an artifact only when it belongs to the current attempt.

    A legacy process with no attempt id never reads experimental files,
    so a stale graph cannot enter a control run.
    """
    from open_enricher.fingerprint import current_stamp

    stamp = current_stamp()
    if not stamp["attempt_id"]:
        return None
    document = load_json(path)
    if document is None or document.get("attempt_id") != stamp["attempt_id"]:
        return None
    return document


def profile_decision(manifest: dict[str, Any] | None, profile_name: str) -> str | None:
    """
    Decide whether a profile may be scored.

    Returns None when there is no manifest (legacy archive). Otherwise
    ``full``, ``degraded``, or ``rejected``.
    """
    if manifest is None:
        return None
    names = {
        str(item.get("name"))
        for item in manifest.get("artifacts") or []
        if isinstance(item, dict)
    }
    if profile_name not in names:
        return "rejected"
    kind = str(manifest.get("profile_kind") or "none")
    if kind == "full":
        return "full"
    if kind == "degraded":
        return "degraded"
    return "rejected"


def publish_manifest(
    video_filename: str,
    *,
    attempt_id: str,
    source_fingerprint: str,
    settings_hash: str,
    profile_kind: str,
    outcomes: list[dict[str, Any]],
    artifact_paths: list[Path],
    terminal_reason: str | None,
    features: dict[str, bool],
) -> dict[str, Any]:
    """Write the manifest atomically and return the document."""
    artifacts = []
    for path in artifact_paths:
        if path.is_file():
            stat = path.stat()
            artifacts.append(
                {
                    "name": path.name,
                    "sha256": file_sha256(path),
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    document = {
        "video": video_filename,
        "attempt_id": attempt_id,
        "source_fingerprint": source_fingerprint,
        "settings_hash": settings_hash,
        "profile_kind": profile_kind,
        "features": features,
        "outcomes": outcomes,
        "terminal_reason": terminal_reason,
        "artifacts": artifacts,
    }
    path = manifest_path(video_filename)
    storage.atomic_write_text(
        path,
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
    )
    return document
