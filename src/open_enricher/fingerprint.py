"""Source fingerprint and attempt identity for experimental resumes.

A later run reuses an artifact only when the source fingerprint, the
effective settings and the attempt id stored on that artifact match.
Legacy variants never set the attempt environment variables.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from open_enricher.adapters.upstream import LAVIS_REV, TALKNET_REV
from open_enricher.stages.pipeline_spec import PipelineSpec

ATTEMPT_ID = "OPEN_ENRICHER_ATTEMPT_ID"
SOURCE_FINGERPRINT = "OPEN_ENRICHER_SOURCE_FINGERPRINT"
SETTINGS_HASH = "OPEN_ENRICHER_SETTINGS_HASH"


def source_fingerprint(path: Path) -> str:
    """
    Hash the size, mtime and the first and last 64 KiB of a source file.

    The full file is not read. A replaced video with the same size still
    changes the digest when the sampled bytes differ.
    """
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("ascii"))
    digest.update(str(stat.st_mtime_ns).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(65536))
        if stat.st_size > 65536:
            handle.seek(max(0, stat.st_size - 65536))
            digest.update(handle.read(65536))
    return digest.hexdigest()


def settings_hash(spec: PipelineSpec) -> str:
    """Hash the knobs that change experimental artifacts."""
    payload = {
        "experiment_id": spec.experiment_id,
        "interval_seconds": spec.interval_seconds,
        "vlm_memory_scenes": spec.vlm_memory_scenes,
        "visual_only": spec.visual_only,
        "narrate_from_visual": spec.narrate_from_visual,
        "frame_allocation": spec.frame_allocation,
        "features": asdict(spec.features),
        "budgets": asdict(spec.budgets),
        "checkpoints": asdict(spec.checkpoints),
    }
    if spec.features.character_graph:
        payload["talknet_revision"] = TALKNET_REV
    if spec.features.visual_memory:
        payload["lavis_revision"] = LAVIS_REV
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def current_stamp() -> dict[str, str]:
    """Return the attempt identifiers for this process, possibly empty."""
    return {
        "attempt_id": os.environ.get(ATTEMPT_ID, "").strip(),
        "source_fingerprint": os.environ.get(SOURCE_FINGERPRINT, "").strip(),
        "settings_hash": os.environ.get(SETTINGS_HASH, "").strip(),
    }


def reusable(document: dict[str, Any] | None) -> bool:
    """True when ``document`` was produced for the current attempt inputs."""
    if not isinstance(document, dict):
        return False
    stamp = current_stamp()
    if not stamp["attempt_id"] and not stamp["source_fingerprint"]:
        return False
    if stamp["source_fingerprint"] and document.get("source_fingerprint") != stamp["source_fingerprint"]:
        return False
    if stamp["settings_hash"] and document.get("settings_hash") != stamp["settings_hash"]:
        return False
    status = str(document.get("status") or "")
    return status in {"success", "degraded", "fallback"}
