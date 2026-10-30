"""One-off backfill of VLM sample frames for experiments 06, 07, and 08.

Not part of the project. Replays clip_video_range + sample_scene, which is
the path the VLM stage uses before it deletes the temporary window clip.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path("/SLURM_shared/st_mdp26/open-enricher")
sys.path.insert(0, str(REPO / "src"))

from open_enricher.stages.vlm_core.frames import sample_scene  # noqa: E402
from open_enricher.stages.vlm_core.video import clip_video_range  # noqa: E402

VIDEO_NAME = "Suits.S01E02.mkv"
SAMPLE_FPS = 2.0
MIN_FRAMES = 2
MAX_FRAMES = 40
EXPERIMENTS = (
    "06-memory-2",
    "07-window-600",
    "08-narrate-visual",
)


def video_path() -> Path:
    candidate = REPO / "dataset" / "video_inputs" / VIDEO_NAME
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Source video not found: {candidate}")


def windows_for(experiment_id: str) -> list[tuple[float, float]]:
    dialogue = (
        REPO
        / "runs"
        / "local"
        / experiment_id
        / VIDEO_NAME
        / f"{VIDEO_NAME}_dialogue.json"
    )
    payload = json.loads(dialogue.read_text(encoding="utf-8"))
    windows = []
    for scene in payload["scenes"]:
        start = float(scene["start_seconds"])
        end = float(scene["end_seconds"])
        windows.append((start, end))
    return windows


def main() -> None:
    source = video_path()
    print(f"video {source}", flush=True)
    grouped: dict[tuple[tuple[float, float], ...], list[str]] = {}
    for experiment_id in EXPERIMENTS:
        key = tuple(windows_for(experiment_id))
        grouped.setdefault(key, []).append(experiment_id)
        print(
            f"{experiment_id}: {len(key)} windows "
            f"{key[0][0]:.0f}-{key[-1][1]:.0f}s",
            flush=True,
        )

    for windows, experiment_ids in grouped.items():
        with tempfile.TemporaryDirectory(prefix="open-enricher-frames-") as tmp:
            tmp_dir = Path(tmp)
            for index, (start, end) in enumerate(windows, start=1):
                clip = tmp_dir / f"window_{index}.mp4"
                duration = max(1.0, end - start)
                print(
                    f"clip window {index}/{len(windows)} "
                    f"{start:.3f}+{duration:.3f}s -> {clip.name}",
                    flush=True,
                )
                clip_video_range(source, start, duration, clip)
                for experiment_id in experiment_ids:
                    span = sample_scene(
                        clip,
                        video_filename=VIDEO_NAME,
                        experiment_id=experiment_id,
                        scene_index=index,
                        start_seconds=start,
                        sample_fps=SAMPLE_FPS,
                        min_frames=MIN_FRAMES,
                        max_frames=MAX_FRAMES,
                    )
                    frames = span.get("frames") or []
                    first = frames[0]["file"] if frames else "(none)"
                    print(
                        f"  {experiment_id} scene {index}: "
                        f"{span.get('sampled_frames')} frames, first {first}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
