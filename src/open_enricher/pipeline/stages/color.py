"""Minimal color stage: hex palette from a few frames via PyAV and numpy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import av
import numpy as np

# Sparse sample across the file. Enough for a rough palette without OpenCV.
_FRAME_INDICES = (0, 24, 48, 72, 96)
_BINS = 4


def _frame_hex_colors(frame: np.ndarray) -> list[str]:
    """Return the top histogram bins as hex colors for one RGB frame."""

    # Quantize each channel into a few buckets, then take the most common bins.
    quantized = (frame.astype(np.int32) * _BINS) // 256
    quantized = np.clip(quantized, 0, _BINS - 1)
    flat = quantized.reshape(-1, 3)
    # Pack RGB into one int so bincount can rank the palette.
    codes = flat[:, 0] * (_BINS * _BINS) + flat[:, 1] * _BINS + flat[:, 2]
    counts = np.bincount(codes, minlength=_BINS * _BINS * _BINS)
    top = np.argsort(counts)[::-1][:3]
    colors: list[str] = []
    for code in top:
        if counts[code] == 0:
            continue
        r = int(((code // (_BINS * _BINS)) % _BINS) * (256 // _BINS) + (128 // _BINS))
        g = int(((code // _BINS) % _BINS) * (256 // _BINS) + (128 // _BINS))
        b = int((code % _BINS) * (256 // _BINS) + (128 // _BINS))
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors


class ColorStage:
    """Sample a few decoded frames and return a short hex palette."""

    name = "color"

    def run(self, video_path: Path) -> dict[str, Any]:
        """Return dominant colors collected from sparsely sampled frames."""

        palettes: list[str] = []
        seen: set[str] = set()
        with av.open(str(video_path)) as container:
            if not container.streams.video:
                raise RuntimeError(f"No video stream in {video_path}")
            for index, frame in enumerate(container.decode(video=0)):
                if index not in _FRAME_INDICES:
                    if index > _FRAME_INDICES[-1]:
                        break
                    continue
                rgb = frame.to_ndarray(format="rgb24")
                for color in _frame_hex_colors(rgb):
                    if color not in seen:
                        seen.add(color)
                        palettes.append(color)
        if not palettes:
            raise RuntimeError(f"Could not sample colors from {video_path}")
        return {
            "video": video_path.name,
            "colors": palettes[:6],
        }
