"""Minimal audio stage: stream properties from the container via PyAV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import av


class AudioStage:
    """Read duration, sample rate, and channel count from the first audio stream."""

    name = "audio"

    def run(self, video_path: Path) -> dict[str, Any]:
        """Return basic audio properties for one resolved video file."""

        with av.open(str(video_path)) as container:
            if not container.streams.audio:
                raise RuntimeError(f"No audio stream in {video_path}")
            
            stream = container.streams.audio[0]
            duration = float(stream.duration * stream.time_base) if stream.duration else None
            if duration is None and container.duration is not None:
                duration = float(container.duration / av.time_base)
            sample_rate = int(stream.rate) if stream.rate else None
            channels = int(stream.channels) if stream.channels else None
        if duration is None or duration <= 0:
            raise RuntimeError(f"Could not read audio duration from {video_path}")
        if sample_rate is None or sample_rate <= 0:
            raise RuntimeError(f"Could not read sample rate from {video_path}")
        if channels is None or channels <= 0:
            raise RuntimeError(f"Could not read channel count from {video_path}")
        return {
            "video": video_path.name,
            "duration_seconds": round(duration, 3),
            "sample_rate": sample_rate,
            "channels": channels,
        }
