"""InsightFace tracks and TalkNet active-speaker scores.

Both adapters load weights only inside a child process. A missing
package or a failed constructor is a fallback document, not an exception
for the parent pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from open_enricher import storage
from open_enricher.adapters.isolate import call_in_child
from open_enricher.stages.pipeline_spec import PipelineSpec


def insightface_root(spec: PipelineSpec) -> Path:
    """InsightFace root. The pack lives in ``root/models/buffalo_l``."""
    if spec.checkpoints.insightface:
        return Path(spec.checkpoints.insightface)
    return storage.models_cache() / "external" / "insightface"


def talknet_weights(spec: PipelineSpec) -> Path:
    """Path of ``pretrain_TalkSet.model``."""
    if spec.checkpoints.talknet:
        return Path(spec.checkpoints.talknet)
    return storage.models_cache() / "external" / "talknet" / "pretrain_TalkSet.model"


def detect_tracks(
    video: Path,
    spec: PipelineSpec,
    *,
    sample_fps: float,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Detect faces. The parent does not import InsightFace.

    ``max_frames`` caps how many sampled frames are scored. ``None`` reads
    the whole file. The pipeline leaves it unset.
    """
    root = insightface_root(spec)
    pack = root / "models" / "buffalo_l"
    if not pack.is_dir():
        return {"status": "fallback", "reason": f"buffalo_l missing under {root}", "tracks": []}
    payload: dict[str, Any] = {
        "video": str(video),
        "root": str(root),
        "sample_fps": sample_fps,
    }
    if max_frames is not None:
        payload["max_frames"] = max_frames
    return call_in_child(
        "open_enricher.adapters.face_speech",
        "run_insightface_payload",
        payload,
    )


def active_speakers(
    video: Path,
    spec: PipelineSpec,
    tracks: list[dict[str, Any]],
    *,
    max_tracks: int | None = None,
) -> dict[str, Any]:
    """Score tracks with TalkNet. Without a loaded net this is a fallback.

    ``max_tracks`` caps how many one-second windows are forwarded. The
    pipeline leaves it unset, and the child then scores four windows so a
    character-graph pass does not hold the GPU for the whole episode.
    """
    weights = talknet_weights(spec)
    if not weights.is_file():
        return {"status": "fallback", "reason": f"TalkNet weights missing: {weights}", "scores": []}
    if not tracks:
        return {"status": "fallback", "reason": "no face tracks to score", "scores": []}
    slim = [
        {
            "track_id": row.get("track_id"),
            "start": row.get("start"),
            "end": row.get("end"),
            "bbox": row.get("bbox"),
        }
        for row in tracks
    ]
    payload: dict[str, Any] = {"video": str(video), "weights": str(weights), "tracks": slim}
    if max_tracks is not None:
        payload["max_tracks"] = max_tracks
    return call_in_child(
        "open_enricher.adapters.face_speech",
        "run_talknet_payload",
        payload,
    )


def _session_providers(app: Any) -> list[str]:
    """Providers actually bound on the loaded buffalo_l sessions."""
    names: list[str] = []
    for model in getattr(app, "models", {}).values():
        session = getattr(model, "session", None)
        getter = getattr(session, "get_providers", None)
        if not callable(getter):
            continue
        for name in getter():
            if name not in names:
                names.append(str(name))
    return names


def run_insightface_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Child entry for ``buffalo_l``. CUDA is preferred and CPU is the fallback."""
    try:
        import cv2
        from insightface.app import FaceAnalysis
    except Exception as exc:  # noqa: BLE001 - optional dependency
        return {"status": "fallback", "reason": f"insightface import failed: {exc}", "tracks": []}
    try:
        app = FaceAnalysis(name="buffalo_l", root=payload["root"])
        try:
            app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception:
            app.prepare(ctx_id=-1, det_size=(640, 640))
        # prepare(ctx_id=0) does not raise when ORT drops CUDA and keeps CPU.
        active = _session_providers(app)
        provider = "cuda" if "CUDAExecutionProvider" in active else "cpu"
        limit = payload.get("max_frames")
        tracks = _sample_tracks(
            app,
            payload["video"],
            sample_fps=float(payload["sample_fps"]),
            cv2=cv2,
            max_frames=None if limit is None else int(limit),
        )
    except Exception as exc:  # noqa: BLE001 - recorded fallback
        return {"status": "fallback", "reason": str(exc), "tracks": []}
    return {
        "status": "success",
        "provider": provider,
        "providers": active,
        "tracks": tracks,
    }


def run_talknet_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Child entry for TalkNet. A checkpoint mismatch stays a fallback."""
    try:
        from open_enricher.adapters.talknet_asd import score_talknet
    except Exception as exc:  # noqa: BLE001 - optional dependency
        return {"status": "fallback", "reason": f"talknet import failed: {exc}", "scores": []}
    try:
        return score_talknet(payload)
    except Exception as exc:  # noqa: BLE001 - recorded fallback
        return {"status": "fallback", "reason": str(exc), "scores": []}


def _sample_tracks(
    app: Any,
    video: str,
    *,
    sample_fps: float,
    cv2: Any,
    max_frames: int | None = None,
) -> list[dict[str, Any]]:
    """Pool one embedding per detected face across sampled frames."""
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise RuntimeError(f"could not open video {video}")
    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    step = max(1, int(round(native_fps / max(sample_fps, 0.1))))
    grouped: dict[str, dict[str, Any]] = {}
    index = 0
    sampled = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step:
                index += 1
                continue
            stamp = index / native_fps
            faces = app.get(frame) or []
            ids_here: list[str] = []
            for face_index, face in enumerate(faces):
                embedding = getattr(face, "embedding", None)
                if embedding is None:
                    continue
                norm = float((embedding ** 2).sum()) ** 0.5
                if norm <= 0:
                    continue
                vector = [float(value) / norm for value in embedding.tolist()]
                bbox = [float(value) for value in getattr(face, "bbox", [])]
                track_id = f"f{index:06d}_{face_index}"
                ids_here.append(track_id)
                grouped[track_id] = {
                    "track_id": track_id,
                    "embedding": vector,
                    "start": round(stamp, 3),
                    "end": round(stamp + 1.0 / max(sample_fps, 0.1), 3),
                    "shot_id": "frame",
                    "bbox": bbox,
                    "co_visible": [],
                }
            for track_id in ids_here:
                grouped[track_id]["co_visible"] = [
                    other for other in ids_here if other != track_id
                ]
            index += 1
            sampled += 1
            if max_frames is not None and sampled >= max_frames:
                break
    finally:
        capture.release()
    return list(grouped.values())
