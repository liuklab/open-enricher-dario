"""Local run archive under runs/exp2 and progress resume helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.open_enricher.experiments.config import CACHE_ROOT, PROGRESS_PATH, RUNS_ROOT


def experiment_dir(experiment_id: str) -> Path:
    path = RUNS_ROOT / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def machine_dir(experiment_id: str, machine: str) -> Path:
    path = experiment_dir(experiment_id) / machine
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir(machine: str) -> Path:
    path = CACHE_ROOT / machine
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, document: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def append_progress(event: dict[str, Any]) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = dict(event)
    row.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with PROGRESS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_progress() -> list[dict[str, Any]]:
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


def is_done(experiment_id: str, machine: str, step: str) -> bool:
    for row in load_progress():
        if (
            row.get("experiment_id") == experiment_id
            and row.get("machine") == machine
            and row.get("step") == step
            and row.get("status") == "ok"
        ):
            return True
    return False


def list_variants(variants_dir: Path) -> list[Path]:
    return sorted(variants_dir.glob("*.json"))


def merge_timings(paths: Iterable[Path]) -> dict[str, Any]:
    documents = []
    for path in paths:
        if path.is_file():
            try:
                documents.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
    return {"stages": documents}
