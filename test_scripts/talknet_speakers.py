"""Score a short TalkNet window and show which speakers it can name.

InsightFace runs only on the cut clip. TalkNet then scores a few
one-second face windows. Exit 0 when those windows produce scores.
Diarized lines from the episode transcript are overlapped afterwards so
the log shows the ``speaks`` edges that would reach the profile.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAIN_DATASET = Path("/SLURM_shared/st_mdp26/open-enricher/dataset")
os.environ.setdefault("OPEN_ENRICHER_DATASET_ROOT", str(MAIN_DATASET))
sys.path.insert(0, str(REPO / "src"))

from open_enricher.adapters.face_speech import active_speakers, detect_tracks  # noqa: E402
from open_enricher.graph import (  # noqa: E402
    TrackObservation,
    assign_character_ids,
    compact_account,
    speaker_edges,
)
from open_enricher.stages.pipeline_spec import PipelineSpec  # noqa: E402

DEFAULT_NAME = "How_I_Met_Your_Mother.S01E02.mkv"
AUDIO_JSON = (
    REPO
    / "runs"
    / "local"
    / "10-character-graph"
    / DEFAULT_NAME
    / f"{DEFAULT_NAME}_audio.json"
)


def _video_path(name: str) -> Path:
    local = REPO / "dataset" / "video_inputs" / name
    if local.is_file():
        return local
    shared = MAIN_DATASET / "video_inputs" / name
    return shared


def _ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidate = Path("/SLURM_shared/st_mdp26/ffmpeg-libs/bin/ffmpeg")
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError("ffmpeg is not on PATH")


def _cut_clip(video: Path, start: float, seconds: float, dest: Path) -> None:
    command = [
        _ffmpeg(),
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{seconds:.3f}",
        "-i",
        str(video),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dest),
    ]
    subprocess.run(command, check=True)


def _shift(scores: list[dict], offset: float) -> list[dict]:
    shifted = []
    for row in scores:
        item = dict(row)
        item["start"] = round(float(row.get("start") or 0) + offset, 3)
        item["end"] = round(float(row.get("end") or 0) + offset, 3)
        shifted.append(item)
    return shifted


def main() -> int:
    parser = argparse.ArgumentParser(description="Short TalkNet speaker probe.")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--start", type=float, default=20.0)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--max-tracks", type=int, default=4)
    parser.add_argument("--audio", type=Path, default=AUDIO_JSON)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.seconds <= 0 or args.max_frames <= 0 or args.max_tracks <= 0 or args.sample_fps <= 0:
        print("start window, fps, frames and tracks must be positive", file=sys.stderr)
        return 1
    video = args.video if args.video is not None else _video_path(DEFAULT_NAME)
    if not video.is_absolute():
        video = REPO / video
    if not video.is_file():
        print(f"video not found: {video}", file=sys.stderr)
        return 1
    output = args.output
    if output is None:
        output = REPO / "test_scripts" / "output" / f"{video.name}.talknet.json"
    elif not output.is_absolute():
        output = REPO / output

    spec = PipelineSpec(
        experiment_id="talknet-speakers",
        title="talknet-speakers",
        interval_seconds=300,
        vlm_memory_scenes=None,
        visual_only=False,
        narrate_from_visual=False,
    )
    print(
        f"video {video.name} clip {args.start:.1f}+{args.seconds:.1f}s "
        f"sample_fps {args.sample_fps} max_frames {args.max_frames} max_tracks {args.max_tracks}",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="talknet-clip-") as tmp:
        clip = Path(tmp) / "clip.mkv"
        _cut_clip(video, args.start, args.seconds, clip)
        detected = detect_tracks(
            clip,
            spec,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
        )
        if detected.get("status") != "success":
            print(
                f"InsightFace for TalkNet tracks failed: {detected.get('reason') or detected.get('status')}",
                file=sys.stderr,
            )
            return 1
        tracks = list(detected.get("tracks") or [])
        print(
            f"provider {detected.get('provider')} faces {len(tracks)}",
            flush=True,
        )
        if not tracks:
            print("no face tracks to score", file=sys.stderr)
            return 1
        result = active_speakers(clip, spec, tracks, max_tracks=args.max_tracks)

    status = str(result.get("status") or "")
    reason = str(result.get("reason") or "")
    scores = _shift(list(result.get("scores") or []), args.start)
    observations = []
    for row in tracks:
        embedding = row.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            continue
        observations.append(
            TrackObservation(
                track_id=str(row.get("track_id")),
                embedding=[float(value) for value in embedding],
                start=float(row.get("start") or 0) + args.start,
                end=float(row.get("end") or 0) + args.start,
                shot_id=str(row.get("shot_id") or ""),
                co_visible=tuple(str(item) for item in row.get("co_visible") or []),
            )
        )
    nodes, decisions = assign_character_ids(observations, threshold=0.45, margin=0.08)
    lines: list = []
    audio = args.audio
    if audio is not None and audio.is_file():
        loaded = json.loads(audio.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            lines = loaded
    edges = speaker_edges(scores, decisions, lines)
    account_nodes = [
        {
            "character_id": node.character_id,
            "alias": node.alias,
            "tracks": list(node.tracks),
        }
        for node in nodes
    ]
    account = compact_account(account_nodes, [edge.to_dict() for edge in edges])
    payload = {
        "video": video.name,
        "clip_start": args.start,
        "clip_seconds": args.seconds,
        "track_count": len(tracks),
        "status": status,
        "reason": reason,
        "device": result.get("device"),
        "checkpoint_present": bool(result.get("checkpoint_present")),
        "loaded": bool(result.get("loaded")),
        "scores": scores,
        "speaks": [edge.to_dict() for edge in edges],
        "account": account,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    speaking = sum(1 for row in scores if row.get("reliable"))
    print(f"status {status}", flush=True)
    print(f"reason {reason}", flush=True)
    print(f"device {result.get('device')}", flush=True)
    print(
        f"tracks {len(tracks)} scores {len(scores)} speaking {speaking} links {len(edges)}",
        flush=True,
    )
    for row in scores:
        print(
            f"score {row['track_id']} {row['start']:.3f}-{row['end']:.3f} "
            f"mean {row['score']} speaking {row['reliable']}",
            flush=True,
        )
    for edge in edges:
        print(
            f"speaks {edge.source} -> {edge.target} {edge.start:.3f}-{edge.end:.3f} "
            f"confidence {edge.confidence}",
            flush=True,
        )
    if account:
        print(f"account {account}", flush=True)
    print(f"wrote {output}", flush=True)
    if status == "success" and scores:
        return 0
    print(f"TalkNet smoke failed: {reason or status}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
