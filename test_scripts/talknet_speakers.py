"""Probe TalkNet via the face_speech adapter for one video.

Runs InsightFace ``detect_tracks`` for embeddings, then ``active_speakers``.
The adapter currently loads TalkNet weights only and returns fallback because
the forward pass is not bound — this script reports that clearly and does
not pretend speaker scoring succeeded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from open_enricher.adapters.face_speech import (  # noqa: E402
    active_speakers,
    detect_tracks,
)
from open_enricher.stages.pipeline_spec import PipelineSpec  # noqa: E402

DEFAULT_VIDEO = REPO / "dataset" / "video_inputs" / "How_I_Met_Your_Mother.S01E02.mkv"
UNBOUND = "TalkNet forward is not bound"


def main() -> int:
    parser = argparse.ArgumentParser(description="TalkNet adapter smoke for one video.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    video = args.video if args.video.is_absolute() else REPO / args.video
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
    print(f"video {video.name} sample_fps {args.sample_fps}", flush=True)
    detected = detect_tracks(video, spec, sample_fps=args.sample_fps)
    if detected.get("status") != "success":
        print(
            f"InsightFace for TalkNet tracks failed: {detected.get('reason') or detected.get('status')}",
            file=sys.stderr,
        )
        return 1
    tracks = list(detected.get("tracks") or [])
    if not tracks:
        print("no face tracks to score", file=sys.stderr)
        return 1

    result = active_speakers(video, spec, tracks)
    status = str(result.get("status") or "")
    reason = str(result.get("reason") or "")
    payload = {
        "video": video.name,
        "track_count": len(tracks),
        "status": status,
        "reason": reason,
        "checkpoint_present": bool(result.get("checkpoint_present")),
        "state_keys": result.get("state_keys") or [],
        "scores": result.get("scores") or [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"status {status}", flush=True)
    print(f"reason {reason}", flush=True)
    print(f"checkpoint_present {payload['checkpoint_present']}", flush=True)
    print(f"tracks {len(tracks)} scores {len(payload['scores'])}", flush=True)
    print(f"wrote {output}", flush=True)

    if status == "success":
        return 0
    if payload["checkpoint_present"] and UNBOUND in reason:
        print(
            "SMOKE NOTE: TalkNet weights load, but forward is unbound; "
            "speaker links stay unresolved. Not pretending success.",
            flush=True,
        )
        return 0
    print(f"TalkNet smoke failed: {reason or status}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
