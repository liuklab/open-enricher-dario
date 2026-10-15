"""Wall-clock timing helpers for remote pipeline stages."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List


class TimingRecorder:
    """Accumulate named timing spans for one stage run."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.entries: List[Dict[str, Any]] = []
        self._wall_start = time.perf_counter()

    @contextmanager
    def span(self, name: str, **extra: Any) -> Generator[None, None, None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            row: Dict[str, Any] = {
                "name": name,
                "seconds": round(elapsed, 3),
            }
            row.update(extra)
            self.entries.append(row)
            print(f"[TIMING] {self.stage}.{name}={elapsed:.3f}s")

    def document(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "wall_seconds": round(time.perf_counter() - self._wall_start, 3),
            "spans": list(self.entries),
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.document(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path
