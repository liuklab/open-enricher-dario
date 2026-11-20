"""Run the local BaSSL checkpoint and write scene boundaries only.

Uses the ModelScope movie-scene pipeline already wrapped in
``open_enricher.adapters.bassl``. Shot lists are dropped. The JSON file
and the stdout table contain one row per scene.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from open_enricher.adapters.bassl import segment_video  # noqa: E402
from open_enricher.stages.pipeline_spec import (  # noqa: E402
    FeatureFlags,
    PipelineSpec,
)
from open_enricher.stages.vlm_core.video import probe_duration_seconds  # noqa: E402

DEFAULT_VIDEO = REPO / "dataset" / "video_inputs" / "How_I_Met_Your_Mother.S01E02.mkv"


def _seconds(value: object) -> float | None:
    """Parse a ModelScope timecode or a plain number of seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if ":" not in text:
        try:
            return float(text)
        except ValueError:
            return None
    total = 0.0
    for part in text.split(":"):
        try:
            total = total * 60 + float(part)
        except ValueError:
            return None
    return total


def _clock(seconds: float) -> str:
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    if millis == 1000:
        whole += 1
        millis = 0
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _scene_row(index: int, raw: object) -> dict[str, object]:
    timestamps = raw.get("timestamps") if isinstance(raw, dict) else None
    start_raw = end_raw = None
    if isinstance(timestamps, (list, tuple)) and len(timestamps) >= 2:
        start_raw, end_raw = timestamps[0], timestamps[1]
    start = _seconds(start_raw) if start_raw is not None else None
    end = _seconds(end_raw) if end_raw is not None else None
    row: dict[str, object] = {"index": index}
    if start is not None and end is not None:
        row["start"] = _clock(start)
        row["end"] = _clock(end)
        row["start_seconds"] = round(start, 3)
        row["end_seconds"] = round(end, 3)
    else:
        row["start"] = start_raw
        row["end"] = end_raw
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="BaSSL scene boundaries for one video.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON path. Defaults to test_scripts/output/<video>.scenes.json",
    )
    args = parser.parse_args()
    video = args.video if args.video.is_absolute() else REPO / args.video
    if not video.is_file():
        print(f"video not found: {video}", file=sys.stderr)
        return 1
    output = args.output
    if output is None:
        output = REPO / "test_scripts" / "output" / f"{video.name}.scenes.json"
    elif not output.is_absolute():
        output = REPO / output

    duration = probe_duration_seconds(video)
    spec = PipelineSpec(
        experiment_id="bassl-scenes",
        title="bassl-scenes",
        interval_seconds=300,
        vlm_memory_scenes=None,
        visual_only=False,
        narrate_from_visual=False,
        features=FeatureFlags(bassl=True),
    )
    print(f"video {video.name} duration {_clock(duration)}", flush=True)
    document = segment_video(video, duration, spec)
    status = str(document.get("status") or "")
    if status != "success":
        print(f"BaSSL failed: {document.get('reason') or status}", file=sys.stderr)
        return 1

    scenes = [
        _scene_row(index, raw)
        for index, raw in enumerate(document.get("scenes") or [], start=1)
    ]
    payload = {
        "video": video.name,
        "duration_seconds": round(duration, 3),
        "scene_num": len(scenes),
        "scenes": scenes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"scenes {len(scenes)}", flush=True)
    print(f"{'#':>4}  {'start':>12}  {'end':>12}  {'seconds':>8}", flush=True)
    for scene in scenes:
        length = ""
        if "start_seconds" in scene and "end_seconds" in scene:
            length = f"{float(scene['end_seconds']) - float(scene['start_seconds']):8.1f}"
        print(
            f"{scene['index']:4}  {str(scene['start']):>12}  {str(scene['end']):>12}  {length}",
            flush=True,
        )
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
