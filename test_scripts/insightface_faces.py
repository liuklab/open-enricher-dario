"""Detect faces in one video and group reappearances by embedding.

InsightFace ``buffalo_l`` samples the film. Embeddings are matched with the
same cosine rule as the character graph: a face joins an existing id only
when the best score clears the threshold and the margin over the next
candidate. Faces in the same frame stay separate. The written file and the
stdout table list when each id is visible. Raw embeddings are not stored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from open_enricher.adapters.face_speech import detect_tracks  # noqa: E402
from open_enricher.face_time import face_span  # noqa: E402
from open_enricher.graph import TrackObservation, assign_character_ids  # noqa: E402
from open_enricher.stages.pipeline_spec import PipelineSpec  # noqa: E402

DEFAULT_VIDEO = REPO / "dataset" / "video_inputs" / "How_I_Met_Your_Mother.S01E02.mkv"
MATCH_THRESHOLD = 0.45
MATCH_MARGIN = 0.08


def _track_bounds(row: dict) -> tuple[float, float]:
    """Instant from the track. A missing end repeats the start."""
    start = float(row.get("seconds", row.get("start") or 0))
    end = row.get("end")
    if end is None:
        return start, start
    return start, float(end)


def _spans(stamps: list[float], gap: float) -> list[tuple[float, float]]:
    ordered = sorted(stamps)
    if not ordered:
        return []
    start = previous = ordered[0]
    spans: list[tuple[float, float]] = []
    for stamp in ordered[1:]:
        if stamp - previous <= gap:
            previous = stamp
            continue
        spans.append((start, previous))
        start = previous = stamp
    spans.append((start, previous))
    return spans


def _cuda_provider_ready() -> str | None:
    """Error text when this process cannot see the CUDA execution provider."""
    try:
        import onnxruntime as ort
    except Exception as exc:  # noqa: BLE001 - reported to the job log
        return f"onnxruntime import failed: {exc}"
    providers = list(ort.get_available_providers())
    if "CUDAExecutionProvider" not in providers:
        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            try:
                preload()
            except Exception as exc:  # noqa: BLE001 - the provider list is the result
                print(f"onnxruntime preload_dlls failed: {exc}", file=sys.stderr)
            providers = list(ort.get_available_providers())
    location = getattr(ort, "__file__", "unknown")
    print(f"onnxruntime {ort.__version__} providers {providers} at {location}", flush=True)
    if "CUDAExecutionProvider" not in providers:
        return (
            "CUDAExecutionProvider is not available "
            f"(onnxruntime {ort.__version__}). "
            "The CPU wheel and the GPU wheel both own this import; "
            "only onnxruntime-gpu 1.26 is valid on this CUDA 12 stack."
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="InsightFace reappearance map for one video.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.sample_fps <= 0:
        print("sample fps must be positive", file=sys.stderr)
        return 1
    if args.max_frames is not None and args.max_frames <= 0:
        print("max frames must be positive", file=sys.stderr)
        return 1
    video = args.video if args.video.is_absolute() else REPO / args.video
    if not video.is_file():
        print(f"video not found: {video}", file=sys.stderr)
        return 1
    output = args.output
    if output is None:
        output = REPO / "test_scripts" / "output" / f"{video.name}.faces.json"
    elif not output.is_absolute():
        output = REPO / output

    spec = PipelineSpec(
        experiment_id="insightface-faces",
        title="insightface-faces",
        interval_seconds=300,
        vlm_memory_scenes=None,
        visual_only=False,
        narrate_from_visual=False,
    )
    print(f"video {video.name} sample_fps {args.sample_fps}", flush=True)
    cuda_error = _cuda_provider_ready()
    if cuda_error:
        print(cuda_error, file=sys.stderr)
        return 1
    detected = detect_tracks(
        video,
        spec,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
    )
    if detected.get("status") != "success":
        print(f"InsightFace failed: {detected.get('reason') or detected.get('status')}", file=sys.stderr)
        return 1
    if detected.get("provider") != "cuda":
        print(
            f"InsightFace stayed on {detected.get('provider')} {detected.get('providers')}",
            file=sys.stderr,
        )
        return 1

    observations: list[TrackObservation] = []
    for row in detected.get("tracks") or []:
        embedding = row.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            continue
        start, end = _track_bounds(row)
        observations.append(
            TrackObservation(
                track_id=str(row.get("track_id")),
                embedding=[float(value) for value in embedding],
                start=start,
                end=end,
                shot_id=str(row.get("shot_id") or ""),
                co_visible=tuple(str(item) for item in row.get("co_visible") or []),
            )
        )
    nodes, decisions = assign_character_ids(
        observations,
        threshold=MATCH_THRESHOLD,
        margin=MATCH_MARGIN,
    )
    stamps: dict[str, list[float]] = {node.character_id: [] for node in nodes}
    by_track = {item.track_id: item for item in observations}
    for decision in decisions:
        observation = by_track[str(decision["track_id"])]
        stamps[str(decision["character_id"])].append(observation.start)

    gap = 1.5 / args.sample_fps
    characters = []
    for node in nodes:
        ranges = _spans(stamps[node.character_id], gap)
        characters.append(
            {
                "character_id": node.character_id,
                "samples": len(node.tracks),
                "provisional": node.provisional,
                "spans": [face_span(start, end) for start, end in ranges],
            }
        )

    payload = {
        "video": video.name,
        "provider": detected.get("provider"),
        "providers": detected.get("providers"),
        "sample_fps": args.sample_fps,
        "match_threshold": MATCH_THRESHOLD,
        "match_margin": MATCH_MARGIN,
        "face_samples": len(observations),
        "characters": characters,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(
        f"provider {detected.get('provider')} faces {len(observations)} characters {len(characters)}",
        flush=True,
    )
    for character in characters:
        spans = character["spans"]
        print(
            f"{character['character_id']}  samples={character['samples']}  "
            f"spans={len(spans)}  provisional={character['provisional']}",
            flush=True,
        )
        for span in spans:
            if "end" in span:
                print(f"  {span['time']} - {span['end']}", flush=True)
            else:
                print(f"  {span['time']}", flush=True)
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
