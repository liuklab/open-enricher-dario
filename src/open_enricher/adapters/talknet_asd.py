"""Score Open-Enricher face windows with the authors' TalkNet-ASD model."""

from __future__ import annotations

import math
import shutil
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import numpy
import torch
from torch import nn
from scipy.fftpack import dct

from open_enricher.adapters.upstream import activate


class TalkNetASD(nn.Module):
    """Group the released encoder and heads under the TalkSet checkpoint keys."""

    def __init__(self) -> None:
        super().__init__()
        activate("talknet")
        from loss import lossA, lossAV, lossV
        from model.talkNetModel import talkNetModel

        self.model = talkNetModel()
        self.lossAV = lossAV()
        self.lossA = lossA()
        self.lossV = lossV()


def load_talknet(weights: str | Path, device: str | None = None) -> tuple[TalkNetASD, str]:
    """Load ``pretrain_TalkSet.model`` strictly. A key mismatch raises."""
    path = Path(weights)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    network = TalkNetASD()
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict):
        raise TypeError(f"checkpoint is {type(blob).__name__}, expected a state dict")
    cleaned = {}
    for key, value in blob.items():
        name = str(key)
        if name.startswith("module."):
            name = name[len("module.") :]
        cleaned[name] = value
    network.load_state_dict(cleaned, strict=True)
    network.eval()
    network.to(device)
    return network, device


def forward_scores(network: TalkNetASD, audio: numpy.ndarray, visual: numpy.ndarray) -> numpy.ndarray:
    """Run one window. ``audio`` is ``(100, 13)`` MFCC and ``visual`` is ``(25, 112, 112)``."""
    if audio.shape[0] < 100 or visual.shape[0] < 25:
        raise ValueError(f"window is short: audio {audio.shape} visual {visual.shape}")
    audio_tensor = torch.from_numpy(numpy.ascontiguousarray(audio[:100])).float().unsqueeze(0)
    visual_tensor = torch.from_numpy(numpy.ascontiguousarray(visual[:25])).float().unsqueeze(0)
    device = next(network.parameters()).device
    audio_tensor = audio_tensor.to(device)
    visual_tensor = visual_tensor.to(device)
    with torch.no_grad():
        audio_embed = network.model.forward_audio_frontend(audio_tensor)
        visual_embed = network.model.forward_visual_frontend(visual_tensor)
        audio_embed, visual_embed = network.model.forward_cross_attention(audio_embed, visual_embed)
        fused = network.model.forward_audio_visual_backend(audio_embed, visual_embed)
        return network.lossAV(fused)


def _hz_to_mel(hz: numpy.ndarray) -> numpy.ndarray:
    return 2595.0 * numpy.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: numpy.ndarray) -> numpy.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def _filterbank(nfilt: int, nfft: int, samplerate: int) -> numpy.ndarray:
    lowmel = _hz_to_mel(numpy.array(0.0))
    highmel = _hz_to_mel(numpy.array(samplerate / 2.0))
    melpoints = numpy.linspace(lowmel, highmel, nfilt + 2)
    bins = numpy.floor((nfft + 1) * _mel_to_hz(melpoints) / samplerate)
    bank = numpy.zeros((nfilt, nfft // 2 + 1), dtype=numpy.float64)
    for index in range(nfilt):
        left = int(bins[index])
        middle = int(bins[index + 1])
        right = int(bins[index + 2])
        span_up = middle - left
        span_down = right - middle
        for bin_index in range(left, middle):
            if span_up:
                bank[index, bin_index] = (bin_index - bins[index]) / span_up
        for bin_index in range(middle, right):
            if span_down:
                bank[index, bin_index] = (bins[index + 2] - bin_index) / span_down
    return bank


def _frame_signal(signal: numpy.ndarray, frame_len: float, frame_step: float) -> numpy.ndarray:
    length = int(Decimal(frame_len).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    step = int(Decimal(frame_step).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    size = len(signal)
    if size <= length:
        count = 1
    else:
        count = 1 + int(math.ceil((size - length) / float(step)))
    padded = numpy.zeros((count - 1) * step + length, dtype=numpy.float64)
    padded[:size] = signal
    frames = numpy.zeros((count, length), dtype=numpy.float64)
    for index in range(count):
        start = index * step
        frames[index] = padded[start : start + length]
    return frames


def mfcc(signal: numpy.ndarray, samplerate: int = 16000) -> numpy.ndarray:
    """13 MFCCs with the python_speech_features defaults TalkNet was trained on.

    Rectangular window, 25 ms / 10 ms, 26 mel filters, lifter 22, and the
    zeroth coefficient replaced by log energy. ``signal`` is the int16
    waveform promoted to float, which is what that library does.
    """
    samples = numpy.asarray(signal, dtype=numpy.float64)
    if samples.size == 0:
        return numpy.zeros((0, 13), dtype=numpy.float64)
    emphasized = numpy.empty_like(samples)
    emphasized[0] = samples[0]
    emphasized[1:] = samples[1:] - 0.97 * samples[:-1]
    frames = _frame_signal(emphasized, 0.025 * samplerate, 0.010 * samplerate)
    nfft = 512
    spectrum = numpy.fft.rfft(frames, nfft)
    power = (1.0 / nfft) * numpy.square(numpy.abs(spectrum))
    energy = numpy.sum(power, axis=1)
    energy = numpy.where(energy == 0, numpy.finfo(float).eps, energy)
    bank = _filterbank(26, nfft, samplerate)
    features = numpy.dot(power, bank.T)
    features = numpy.where(features == 0, numpy.finfo(float).eps, features)
    features = numpy.log(features)
    cepstra = dct(features, type=2, axis=1, norm="ortho")[:, :13]
    ramp = numpy.arange(13)
    cepstra *= 1.0 + 11.0 * numpy.sin(numpy.pi * ramp / 22.0)
    cepstra[:, 0] = numpy.log(energy)
    return cepstra


def _bbox(track: dict[str, Any]) -> list[float] | None:
    raw = track.get("bbox")
    if not isinstance(raw, list) or len(raw) < 4:
        return None
    box = [float(value) for value in raw[:4]]
    width = box[2] - box[0]
    height = box[3] - box[1]
    if width < 40 or height < 40:
        return None
    return box


def _iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    union = area_left + area_right - inter
    if union <= 0:
        return 0.0
    return inter / union


def _clusters(tracks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    ordered = sorted(tracks, key=lambda row: float(row.get("start") or 0))
    clusters: list[list[dict[str, Any]]] = []
    for track in ordered:
        box = _bbox(track)
        if box is None:
            continue
        start = float(track.get("start") or 0)
        placed = False
        for cluster in reversed(clusters):
            last = cluster[-1]
            last_box = _bbox(last)
            if last_box is None:
                continue
            if start - float(last.get("start") or 0) > 0.5:
                continue
            if _iou(box, last_box) < 0.3:
                continue
            cluster.append(track)
            placed = True
            break
        if not placed:
            clusters.append([track])
    return clusters


def _slice_cluster(cluster: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(cluster, key=lambda row: float(row.get("start") or 0))
    first = float(ordered[0].get("start") or 0)
    last = float(ordered[-1].get("start") or first)
    windows: list[dict[str, Any]] = []
    cursor = first
    while cursor <= last + 1e-6 and len(windows) < limit:
        anchor = min(ordered, key=lambda row: abs(float(row.get("start") or 0) - (cursor + 0.5)))
        box = _bbox(anchor)
        if box is not None:
            windows.append(
                {
                    "track_id": str(anchor.get("track_id")),
                    "start": round(cursor, 3),
                    "end": round(cursor + 1.0, 3),
                    "bbox": box,
                }
            )
        cursor += 1.0
    return windows


def select_windows(
    tracks: list[dict[str, Any]],
    max_tracks: int,
    dialogue_spans: list[list[float]] | None = None,
) -> list[dict[str, Any]]:
    """Turn per-frame boxes into a few one-second TalkNet windows.

    Faces that overlap in time stay in one cluster. Windows are taken
    round-robin so one long face does not crowd out the other people,
    and each cluster is capped at ``max_tracks`` slices.
    """
    if max_tracks < 1:
        return []
    clusters = _clusters(tracks)
    if dialogue_spans is not None:
        windows = sorted(
            (
                window
                for cluster in clusters
                for window in _slice_cluster(cluster, len(cluster))
                if any(
                    min(window["end"], end) > max(window["start"], start)
                    for start, end in dialogue_spans
                )
            ),
            key=lambda window: (window["start"], window["track_id"]),
        )
        if len(windows) <= max_tracks:
            return windows
        return [
            windows[(2 * index + 1) * len(windows) // (2 * max_tracks)]
            for index in range(max_tracks)
        ]

    queues = [_slice_cluster(cluster, max_tracks) for cluster in clusters]
    queues = [queue for queue in queues if queue]
    queues.sort(key=len, reverse=True)
    picked: list[dict[str, Any]] = []
    while len(picked) < max_tracks:
        moved = False
        for queue in queues:
            if not queue:
                continue
            picked.append(queue.pop(0))
            moved = True
            if len(picked) >= max_tracks:
                break
        if not moved:
            break
    return picked


def _ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise FileNotFoundError("ffmpeg is not on PATH")


def read_pcm(video: str, start: float, duration: float) -> numpy.ndarray:
    """Read one mono 16 kHz int16 slice from ``video``."""
    command = [
        _ffmpeg(),
        "-v",
        "error",
        "-i",
        video,
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "pipe:1",
    ]
    raw = subprocess.check_output(command)
    if len(raw) < 2:
        return numpy.zeros(0, dtype=numpy.int16)
    return numpy.frombuffer(raw, dtype=numpy.int16).copy()


def crop_gray(frame: numpy.ndarray, bbox: list[float], crop_scale: float = 0.40) -> numpy.ndarray | None:
    """Center a face the way the TalkNet demo does, then keep the middle 112 pixels."""
    x1, y1, x2, y2 = bbox
    side = max(y2 - y1, x2 - x1) / 2.0
    if side < 1:
        return None
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    pad = int(side * (1 + 2 * crop_scale))
    image = numpy.pad(frame, ((pad, pad), (pad, pad), (0, 0)), constant_values=110)
    my = center_y + pad
    mx = center_x + pad
    top = int(my - side)
    bottom = int(my + side * (1 + 2 * crop_scale))
    left = int(mx - side * (1 + crop_scale))
    right = int(mx + side * (1 + crop_scale))
    face = image[top:bottom, left:right]
    if face.size == 0 or face.shape[0] < 2 or face.shape[1] < 2:
        return None
    import cv2

    face = cv2.resize(face, (224, 224))
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return gray[56:168, 56:168]


def _visual_window(capture: Any, fps: float, start: float, bbox: list[float]) -> numpy.ndarray | None:
    native_count = max(8, int(round(fps)))
    origin = max(0, int(round(start * fps)))
    capture.set(cv2_pos_frames(), origin)
    frames: list[numpy.ndarray] = []
    for _ in range(native_count):
        ok, frame = capture.read()
        if not ok:
            break
        gray = crop_gray(frame, bbox)
        if gray is None:
            return None
        frames.append(gray)
    if len(frames) < 8:
        return None
    if len(frames) == 25:
        stacked = frames
    else:
        last = len(frames) - 1
        stacked = [frames[int(round(index * last / 24))] for index in range(25)]
    return numpy.stack(stacked).astype(numpy.float32)


def cv2_pos_frames() -> int:
    import cv2

    return int(cv2.CAP_PROP_POS_FRAMES)


def _score_window(
    network: TalkNetASD,
    capture: Any,
    fps: float,
    video: str,
    window: dict[str, Any],
) -> dict[str, Any] | None:
    visual = _visual_window(capture, fps, float(window["start"]), list(window["bbox"]))
    if visual is None:
        return None
    audio = read_pcm(video, float(window["start"]), 1.25)
    features = mfcc(audio)
    if features.shape[0] < 100:
        return None
    logits = forward_scores(network, features, visual)
    values = [round(float(value), 4) for value in logits.tolist()]
    mean = float(sum(values) / len(values))
    return {
        "track_id": window["track_id"],
        "bbox": window["bbox"],
        "start": window["start"],
        "end": window["end"],
        "score": round(mean, 4),
        "reliable": mean >= 0.0,
        "frames": len(values),
        "frame_scores": values,
    }


def score_talknet(payload: dict[str, Any]) -> dict[str, Any]:
    """Score a handful of face windows. The cap keeps a probe off the GPU."""
    weights = str(payload.get("weights") or "")
    video = str(payload.get("video") or "")
    tracks = list(payload.get("tracks") or [])
    max_tracks = int(payload.get("max_tracks") or 4)
    present = Path(weights).is_file()
    windows = select_windows(tracks, max_tracks, payload.get("dialogue_spans"))
    if not windows:
        return {
            "status": "fallback",
            "reason": "no face window to score",
            "checkpoint_present": present,
            "scores": [],
        }
    try:
        network, device = load_talknet(weights)
    except Exception as exc:  # noqa: BLE001 - recorded for the probe log
        return {
            "status": "fallback",
            "reason": f"TalkNet model could not be loaded: {exc}",
            "checkpoint_present": present,
            "scores": [],
        }
    import cv2

    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        return {
            "status": "fallback",
            "reason": f"could not open video {video}",
            "checkpoint_present": True,
            "loaded": True,
            "scores": [],
        }
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    scores: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        for window in windows:
            try:
                row = _score_window(network, capture, fps, video, window)
            except Exception as exc:  # noqa: BLE001 - one bad window does not hide the rest
                errors.append(str(exc))
                continue
            if row is not None:
                scores.append(row)
    finally:
        capture.release()
    if not scores:
        return {
            "status": "fallback",
            "reason": errors[0] if errors else "TalkNet forward produced no scores",
            "checkpoint_present": True,
            "loaded": True,
            "device": device,
            "scores": [],
        }
    return {
        "status": "success",
        "reason": "",
        "checkpoint_present": True,
        "loaded": True,
        "device": device,
        "scores": scores,
    }
