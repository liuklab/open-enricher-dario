"""InsightFace smoke on one 30s slice from the middle of a film.

The character graph calls ``detect_tracks`` with
``budgets.face_sample_fps`` (5 in ``10-character-graph.json``). Sampling
inside that call keeps one frame every

    step = max(1, round(native_fps / sample_fps))

frames. This smoke cuts 30 seconds from the middle of the episode, then
uses that same step on the cut. It writes each sampled frame, an annotated
copy with detection boxes, and one crop per detection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "test_scripts"))

os.environ.setdefault(
    "OPEN_ENRICHER_DATASET_ROOT",
    "/SLURM_shared/st_mdp26/open-enricher/dataset",
)

from open_enricher.adapters.face_speech import insightface_root  # noqa: E402
from open_enricher.face_time import face_instant, face_span  # noqa: E402
from open_enricher.graph import TrackObservation, assign_character_ids  # noqa: E402
from open_enricher.stages.pipeline_spec import PipelineSpec  # noqa: E402
from insightface_faces import (  # noqa: E402
    MATCH_MARGIN,
    MATCH_THRESHOLD,
    _cuda_provider_ready,
)

DEFAULT_VIDEO = Path(
    "/SLURM_shared/st_mdp26/open-enricher/dataset/video_inputs/"
    "How_I_Met_Your_Mother.S01E02.mkv"
)
VARIANT = REPO / "src/open_enricher/experiments/variants/10-character-graph.json"
FFMPEG = Path("/SLURM_shared/st_mdp26/ffmpeg-libs/bin/ffmpeg")
FFMPEG_LIB = Path("/SLURM_shared/st_mdp26/ffmpeg-libs/lib")
SEGMENT_SECONDS = 30.0
OUTPUT_DIR = REPO / "test_scripts" / "output" / "himym-s01e02-mid30"

PALETTE = (
    (40, 180, 255),
    (80, 220, 80),
    (255, 120, 40),
    (200, 80, 255),
    (80, 80, 255),
    (255, 220, 40),
    (255, 80, 160),
    (40, 220, 220),
)


def sample_step(native_fps: float, sample_fps: float) -> int:
    """Frame stride used by ``_sample_tracks``."""
    return max(1, int(round(native_fps / max(sample_fps, 0.1))))


def middle_start(duration: float, length: float = SEGMENT_SECONDS) -> float:
    """Start time of a window centered on the film. The opening is not used."""
    if duration <= length:
        raise ValueError(f"film is {duration:.3f}s, shorter than {length:.3f}s")
    return (duration - length) / 2.0


def load_face_sample_fps(path: Path = VARIANT) -> float:
    """Read the character-graph face rate. Missing keys are an error."""
    document = json.loads(path.read_text(encoding="utf-8"))
    budgets = document.get("budgets")
    if not isinstance(budgets, dict) or "face_sample_fps" not in budgets:
        raise ValueError(f"{path} has no budgets.face_sample_fps")
    rate = float(budgets["face_sample_fps"])
    if rate <= 0:
        raise ValueError(f"face_sample_fps must be positive, got {rate}")
    return rate


def probe_video(path: Path) -> tuple[float, int]:
    """Return ``(cv2 fps, frame count)``, the values ``detect_tracks`` uses."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"no fps or frames for {path} (fps={fps}, frames={frames})")
    return fps, frames


def stamp_name(seconds: float) -> str:
    """Episode clock safe for a filename, millisecond precision."""
    whole = int(math.floor(seconds))
    millis = int(round((seconds - whole) * 1000))
    if millis == 1000:
        whole += 1
        millis = 0
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}-{minutes:02d}-{secs:02d}.{millis:03d}"


def _ffmpeg_env() -> dict[str, str]:
    env = os.environ.copy()
    lib = str(FFMPEG_LIB)
    current = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib}:{current}" if current else lib
    return env


def cut_segment(source: Path, dest: Path, start: float, length: float) -> None:
    """Cut ``length`` seconds at ``start``. Accurate seek, source frame rate kept."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not FFMPEG.is_file():
        raise RuntimeError(f"ffmpeg not found: {FFMPEG}")
    command = [
        str(FFMPEG),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(source),
        "-t",
        f"{length:.6f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "passthrough",
        str(dest),
    ]
    completed = subprocess.run(
        command,
        env=_ffmpeg_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not dest.is_file():
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"ffmpeg cut failed ({completed.returncode}): {detail}")


def _clear_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()


def _clip_box(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    if len(bbox) < 4:
        return None
    x1 = max(0, min(width - 1, int(math.floor(bbox[0]))))
    y1 = max(0, min(height - 1, int(math.floor(bbox[1]))))
    x2 = max(0, min(width, int(math.ceil(bbox[2]))))
    y2 = max(0, min(height, int(math.ceil(bbox[3]))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _color(character_id: str) -> tuple[int, int, int]:
    if not character_id.startswith("c"):
        return (180, 180, 180)
    try:
        number = int(character_id[1:])
    except ValueError:
        return (180, 180, 180)
    return PALETTE[(number - 1) % len(PALETTE)]


def _draw_box(
    image: Any,
    box: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    import cv2

    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    if not label:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    top = y1 - text_height - baseline - 6
    if top < 0:
        top = min(y1 + 4, image.shape[0] - text_height - baseline - 6)
    top = max(0, top)
    cv2.rectangle(
        image,
        (x1, top),
        (min(image.shape[1] - 1, x1 + text_width + 8), top + text_height + baseline + 6),
        color,
        -1,
    )
    cv2.putText(
        image,
        label,
        (x1 + 4, top + text_height + 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _write_jpeg(path: Path, image: Any) -> None:
    import cv2

    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"could not write {path}")


def _normalize(embedding: Any) -> list[float] | None:
    if embedding is None:
        return None
    values = [float(value) for value in embedding.tolist()]
    if not values:
        return None
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        return None
    return [value / norm for value in values]


def run_detection(
    segment: Path,
    *,
    start: float,
    sample_fps: float,
    native_fps: float,
    step: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Score sampled frames and write raw frames, annotations, and crops."""
    import cv2
    from insightface.app import FaceAnalysis

    root = insightface_root(
        PipelineSpec(
            experiment_id="insightface-mid30",
            title="insightface-mid30",
            interval_seconds=300,
            vlm_memory_scenes=None,
            visual_only=False,
            narrate_from_visual=False,
        )
    )
    pack = root / "models" / "buffalo_l"
    if not pack.is_dir():
        raise RuntimeError(f"buffalo_l missing under {root}")

    frames_dir = output_dir / "frames"
    crops_dir = output_dir / "crops"
    _clear_dir(frames_dir)
    _clear_dir(crops_dir)

    app = FaceAnalysis(name="buffalo_l", root=str(root))
    try:
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception:
        app.prepare(ctx_id=-1, det_size=(640, 640))
    from open_enricher.adapters.face_speech import _session_providers

    active = _session_providers(app)
    provider = "cuda" if "CUDAExecutionProvider" in active else "cpu"
    if provider != "cuda":
        raise RuntimeError(f"InsightFace stayed on {provider} {active}")

    capture = cv2.VideoCapture(str(segment))
    if not capture.isOpened():
        raise RuntimeError(f"could not open segment {segment}")

    pending: list[dict[str, Any]] = []
    observations: list[TrackObservation] = []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step:
                index += 1
                continue
            source_time = start + index / native_fps
            instant = face_instant(source_time)
            order = len(pending) + 1
            stem = f"f{order:04d}_t{stamp_name(source_time)}"
            raw_path = frames_dir / f"{stem}.jpg"
            _write_jpeg(raw_path, frame)
            height, width = frame.shape[:2]
            detections: list[dict[str, Any]] = []
            ids_here: list[str] = []
            for face_index, face in enumerate(app.get(frame) or []):
                bbox = [float(value) for value in getattr(face, "bbox", [])]
                box = _clip_box(bbox, width, height)
                crop = None
                if box is not None:
                    x1, y1, x2, y2 = box
                    crop = frame[y1:y2, x1:x2].copy()
                embedding = _normalize(getattr(face, "embedding", None))
                track_id = None
                if embedding is not None:
                    track_id = f"f{index:06d}_{face_index}"
                    ids_here.append(track_id)
                detections.append(
                    {
                        "det": face_index,
                        "bbox": bbox,
                        "box": box,
                        "crop": crop,
                        "track_id": track_id,
                        "embedding": embedding,
                    }
                )
            for track_id in ids_here:
                detection = next(item for item in detections if item["track_id"] == track_id)
                embedding = detection["embedding"]
                assert embedding is not None
                observations.append(
                    TrackObservation(
                        track_id=track_id,
                        embedding=embedding,
                        start=float(instant["seconds"]),
                        end=float(instant["seconds"]),
                        shot_id="frame",
                        co_visible=tuple(other for other in ids_here if other != track_id),
                    )
                )
            pending.append(
                {
                    "order": order,
                    "index": index,
                    "time": instant["time"],
                    "seconds": instant["seconds"],
                    "stem": stem,
                    "raw": raw_path,
                    "detections": detections,
                }
            )
            index += 1
    finally:
        capture.release()

    nodes, decisions = assign_character_ids(
        observations,
        threshold=MATCH_THRESHOLD,
        margin=MATCH_MARGIN,
    )
    character_of = {str(row["track_id"]): str(row["character_id"]) for row in decisions}

    frame_rows: list[dict[str, Any]] = []
    crop_count = 0
    for item in pending:
        annotated = cv2.imread(str(item["raw"]))
        if annotated is None:
            raise RuntimeError(f"could not reread {item['raw']}")
        det_rows: list[dict[str, Any]] = []
        for detection in item["detections"]:
            track_id = detection["track_id"]
            character_id = character_of.get(track_id) if track_id else None
            box = detection["box"]
            label = character_id or ""
            if box is not None:
                _draw_box(annotated, box, label, _color(character_id or ""))
            crop_name = None
            crop = detection["crop"]
            if crop is not None and crop.size:
                who = character_id or "face"
                crop_name = f"{item['stem']}_{who}_d{detection['det']:02d}.jpg"
                _write_jpeg(crops_dir / crop_name, crop)
                crop_count += 1
            det_rows.append(
                {
                    "det": detection["det"],
                    "track_id": track_id,
                    "character_id": character_id,
                    "time": item["time"],
                    "seconds": item["seconds"],
                    "bbox": [round(value, 1) for value in detection["bbox"]],
                    "crop": f"crops/{crop_name}" if crop_name else None,
                }
            )
        annotated_path = frames_dir / f"{item['stem']}_annotated.jpg"
        _write_jpeg(annotated_path, annotated)
        frame_rows.append(
            {
                "order": item["order"],
                "frame_index": item["index"],
                "time": item["time"],
                "seconds": item["seconds"],
                "raw": f"frames/{item['raw'].name}",
                "annotated": f"frames/{annotated_path.name}",
                "detections": det_rows,
            }
        )

    stamps: dict[str, list[float]] = {node.character_id: [] for node in nodes}
    by_track = {obs.track_id: obs for obs in observations}
    for decision in decisions:
        stamps[str(decision["character_id"])].append(by_track[str(decision["track_id"])].start)
    gap = 1.5 / sample_fps
    characters = []
    for node in nodes:
        ordered = sorted(stamps[node.character_id])
        spans: list[tuple[float, float]] = []
        if ordered:
            span_start = previous = ordered[0]
            for stamp in ordered[1:]:
                if stamp - previous <= gap:
                    previous = stamp
                    continue
                spans.append((span_start, previous))
                span_start = previous = stamp
            spans.append((span_start, previous))
        characters.append(
            {
                "character_id": node.character_id,
                "samples": len(node.tracks),
                "provisional": node.provisional,
                "spans": [face_span(begin, end) for begin, end in spans],
            }
        )

    return {
        "provider": provider,
        "providers": active,
        "sampled_frames": len(frame_rows),
        "face_samples": len(observations),
        "detection_count": sum(len(row["detections"]) for row in frame_rows),
        "crop_count": crop_count,
        "characters": characters,
        "frames": frame_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="30s middle-segment InsightFace smoke.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Cut the segment and print the sample plan. Do not load InsightFace.",
    )
    args = parser.parse_args()
    video = args.video if args.video.is_absolute() else REPO / args.video
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    if not video.is_file():
        print(f"video not found: {video}", file=sys.stderr)
        return 1

    sample_fps = load_face_sample_fps()
    source_fps, source_frames = probe_video(video)
    source_duration = source_frames / source_fps
    start = middle_start(source_duration)
    source_step = sample_step(source_fps, sample_fps)
    interval = source_step / source_fps
    print(
        f"source {video.name} duration {source_duration:.6f} fps {source_fps} "
        f"frames {source_frames}",
        flush=True,
    )
    print(
        f"segment_start {start:.6f} duration {SEGMENT_SECONDS:.6f} "
        f"sample_fps {sample_fps} step {source_step} interval {interval:.6f}",
        flush=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    segment = output_dir / "segment.mkv"
    plan_path = output_dir / "segment_plan.json"
    plan = {
        "source": str(video),
        "segment_start_seconds": round(start, 6),
        "segment_duration_seconds": SEGMENT_SECONDS,
        "source_fps": source_fps,
        "source_frames": source_frames,
        "sample_fps": sample_fps,
        "step_frames": source_step,
    }
    cached = None
    if plan_path.is_file() and segment.is_file():
        try:
            cached = json.loads(plan_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cached = None
    if cached != plan:
        print(f"cutting {segment}", flush=True)
        cut_segment(video, segment, start, SEGMENT_SECONDS)
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    else:
        print(f"reusing {segment}", flush=True)

    segment_fps, segment_frames = probe_video(segment)
    step = sample_step(segment_fps, sample_fps)
    if step != source_step:
        print(
            f"segment step {step} (fps {segment_fps}) does not match "
            f"source step {source_step}",
            file=sys.stderr,
        )
        return 1
    sampled = [index for index in range(segment_frames) if index % step == 0]
    segment_duration = segment_frames / segment_fps
    print(
        f"segment_fps {segment_fps} segment_frames {segment_frames} "
        f"segment_duration {segment_duration:.6f} sampled_frames {len(sampled)}",
        flush=True,
    )
    if sampled:
        first = start + sampled[0] / segment_fps
        last = start + sampled[-1] / segment_fps
        print(
            f"first_sample {first:.6f} {stamp_name(first)} "
            f"last_sample {last:.6f} {stamp_name(last)}",
            flush=True,
        )
    if args.prepare_only:
        return 0

    cuda_error = _cuda_provider_ready()
    if cuda_error:
        print(cuda_error, file=sys.stderr)
        return 1

    try:
        detected = run_detection(
            segment,
            start=start,
            sample_fps=sample_fps,
            native_fps=segment_fps,
            step=step,
            output_dir=output_dir,
        )
    except Exception as exc:  # noqa: BLE001 - the job log is the result
        print(f"InsightFace failed: {exc}", file=sys.stderr)
        return 1

    frames_dir = output_dir / "frames"
    crops_dir = output_dir / "crops"
    payload = {
        "video": video.name,
        "segment": str(segment),
        "segment_start_seconds": round(start, 6),
        "segment_duration_seconds": round(segment_duration, 6),
        "requested_duration_seconds": SEGMENT_SECONDS,
        "source_fps": source_fps,
        "segment_fps": segment_fps,
        "sample_fps": sample_fps,
        "step_frames": step,
        "sampling_interval_seconds": round(step / segment_fps, 6),
        "sampled_frames": detected["sampled_frames"],
        "provider": detected["provider"],
        "providers": detected["providers"],
        "match_threshold": MATCH_THRESHOLD,
        "match_margin": MATCH_MARGIN,
        "face_samples": detected["face_samples"],
        "detection_count": detected["detection_count"],
        "crop_count": detected["crop_count"],
        "character_count": len(detected["characters"]),
        "frames_dir": str(frames_dir),
        "crops_dir": str(crops_dir),
        "characters": detected["characters"],
        "frames": detected["frames"],
    }
    result_path = output_dir / "faces.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"provider {detected['provider']} faces {detected['face_samples']} "
        f"characters {len(detected['characters'])} "
        f"detections {detected['detection_count']} crops {detected['crop_count']}",
        flush=True,
    )
    print(f"frames {frames_dir} count {detected['sampled_frames']}", flush=True)
    print(f"crops {crops_dir} count {detected['crop_count']}", flush=True)
    print(f"wrote {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
