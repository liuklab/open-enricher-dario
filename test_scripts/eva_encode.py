"""Probe EVA-CLIP / Q-Former encode via the visual_encode adapter.

Calls ``encode_video`` for one film. The adapter currently opens the
checkpoint files and returns fallback because the EVA/Q-Former forward
pass is not bound — this script reports that clearly and does not pretend
tokens were produced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from open_enricher.adapters.visual_encode import encode_video  # noqa: E402
from open_enricher.stages.pipeline_spec import PipelineSpec  # noqa: E402

DEFAULT_VIDEO = REPO / "dataset" / "video_inputs" / "How_I_Met_Your_Mother.S01E02.mkv"
UNBOUND = "EVA/Q-Former forward is not bound"


def main() -> int:
    parser = argparse.ArgumentParser(description="EVA/Q-Former adapter smoke for one video.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--sample-fps", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    video = args.video if args.video.is_absolute() else REPO / args.video
    if not video.is_file():
        print(f"video not found: {video}", file=sys.stderr)
        return 1
    output = args.output
    if output is None:
        output = REPO / "test_scripts" / "output" / f"{video.name}.eva.json"
    elif not output.is_absolute():
        output = REPO / output

    spec = PipelineSpec(
        experiment_id="eva-encode",
        title="eva-encode",
        interval_seconds=300,
        vlm_memory_scenes=None,
        visual_only=False,
        narrate_from_visual=False,
    )
    print(f"video {video.name} sample_fps {args.sample_fps}", flush=True)
    result = encode_video(video, spec, sample_fps=args.sample_fps)
    status = str(result.get("status") or "")
    reason = str(result.get("reason") or "")
    frames = result.get("frames") or []
    payload = {
        "video": video.name,
        "sample_fps": args.sample_fps,
        "status": status,
        "reason": reason,
        "checkpoint_present": bool(result.get("checkpoint_present")),
        "tensors": result.get("tensors") or {},
        "frame_count": len(frames),
        "frames": frames,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"status {status}", flush=True)
    print(f"reason {reason}", flush=True)
    print(f"checkpoint_present {payload['checkpoint_present']}", flush=True)
    print(f"tensors {payload['tensors']}", flush=True)
    print(f"frames {len(frames)}", flush=True)
    print(f"wrote {output}", flush=True)

    if status == "success" and frames:
        return 0
    if payload["checkpoint_present"] and UNBOUND in reason:
        print(
            "SMOKE NOTE: EVA/Q-Former checkpoints load, but forward is unbound; "
            "history budget returns to the current clip. Not pretending success.",
            flush=True,
        )
        return 0
    print(f"EVA encode smoke failed: {reason or status}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
