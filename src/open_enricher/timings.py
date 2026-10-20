"""Wall-clock timing helpers for pipeline stages.

Each stage constructs a ``TimingRecorder``, wraps heavy work in named
``span`` contexts, and writes a JSON timing document next to its artifacts.
The experiment archive merges those documents into a single ``timings.json``.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List


class TimingRecorder:
    """
    Accumulate named timing spans for one stage run.

    The wall clock starts at construction. Each ``span`` records elapsed
    seconds and optional metadata keys that are useful in later analysis.
    """

    def __init__(self, stage: str) -> None:
        """
        Start recording for a pipeline stage.

        Args:
            stage: Stable stage name (audio, color, vlm, llm, judger).
        """
        self.stage = stage
        self.entries: List[Dict[str, Any]] = []
        self._wall_start = time.perf_counter()

    @contextmanager
    def span(self, name: str, **extra: Any) -> Generator[None, None, None]:
        """
        Time a named block and append it to ``entries``.

        Args:
            name: Short label for the work unit (model load, inference, ...).
            **extra: Optional fields stored alongside the duration.

        Yields:
            Control to the caller for the timed block. Timing is always
            recorded, even when the block raises.
        """
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
        """
        Build the serializable timing document for this stage run.

        Returns:
            Dict with stage name, total wall seconds, and the span list.
        """
        return {
            "stage": self.stage,
            "wall_seconds": round(time.perf_counter() - self._wall_start, 3),
            "spans": list(self.entries),
        }

    def save(self, path: Path) -> Path:
        """
        Write the timing document as indented JSON.

        Args:
            path: Destination file. Parent directories are created as needed.

        Returns:
            The same ``path`` for chaining.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.document(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path
