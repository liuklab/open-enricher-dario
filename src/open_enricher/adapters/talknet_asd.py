"""TalkNet-ASD forward for ``pretrain_TalkSet.model``.

The module names and tensor ranks follow that checkpoint: a 3D visual
front end, a depthwise temporal stack, an SE-ResNet audio encoder, two
cross-attention blocks, and the audio-visual classifier. Scores are the
speaking-class logits. A positive mean is the same speaking threshold the
TalkNet demo uses.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fftpack import dct


_FFMPEG_CANDIDATES = (
    "/SLURM_shared/st_mdp26/ffmpeg-libs/bin/ffmpeg",
)


class _VisualResBlock(nn.Module):
    """Two residual units. The 1x1 downsample exists even when stride is 1."""

    def __init__(self, inplanes: int, outplanes: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1a(self.conv1a(batch)))
        out = self.conv2a(out)
        residual = batch if self.stride == 1 else self.downsample(batch)
        out = out + residual
        intermediate = out
        out = F.relu(self.outbna(out))
        out = F.relu(self.bn1b(self.conv1b(out)))
        out = self.conv2b(out)
        return F.relu(self.outbnb(out + intermediate))


class _VisualResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = _VisualResBlock(64, 64, stride=1)
        self.layer2 = _VisualResBlock(64, 128, stride=2)
        self.layer3 = _VisualResBlock(128, 256, stride=2)
        self.layer4 = _VisualResBlock(256, 512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = self.layer1(batch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        return self.avgpool(batch)


class _VisualFrontend(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.resnet = _VisualResNet()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.transpose(0, 1).transpose(1, 2)
        batchsize = batch.shape[0]
        batch = self.frontend3D(batch)
        batch = batch.transpose(1, 2)
        batch = batch.reshape(batch.shape[0] * batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])
        batch = self.resnet(batch)
        batch = batch.reshape(batchsize, -1, 512)
        batch = batch.transpose(1, 2)
        return batch.transpose(1, 2).transpose(0, 1)


class _ChannelNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        mean = batch.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        var = (batch - mean).pow(2).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        return self.gamma * (batch - mean) / torch.pow(var + 1e-8, 0.5) + self.beta


class _DSConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, 3, stride=1, padding=1, dilation=1, groups=512, bias=False),
            nn.PReLU(),
            _ChannelNorm(512),
            nn.Conv1d(512, 512, 1, bias=False),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.net(batch) + batch


class _VisualTCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(*[_DSConv() for _ in range(5)])

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.net(batch)


class _VisualConv1D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(512, 256, 5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.net(batch)


class _SELayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 8) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        scale = self.avg_pool(batch).view(batch.shape[0], batch.shape[1])
        scale = self.fc(scale).view(batch.shape[0], batch.shape[1], 1, 1)
        return batch * scale


class _SEBasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int | tuple[int, int] = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = _SELayer(planes)
        self.downsample = downsample

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        residual = batch
        out = self.relu(self.conv1(batch))
        out = self.bn1(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(batch)
        return self.relu(out + residual)


class _AudioEncoder(nn.Module):
    def __init__(self, layers: list[int], num_filters: list[int]) -> None:
        super().__init__()
        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=7, stride=(2, 1), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(num_filters[0], layers[0])
        self.layer2 = self._make_layer(num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(num_filters[3], layers[3], stride=(1, 1))

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int | tuple[int, int] = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers: list[nn.Module] = [_SEBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_SEBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = self.relu(self.bn1(self.conv1(batch)))
        batch = self.layer1(batch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        batch = torch.mean(batch, dim=2, keepdim=True)
        batch = batch.view(batch.shape[0], batch.shape[1], -1)
        return batch.transpose(1, 2)


class _Attention(nn.Module):
    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.activation = F.relu

    def forward(self, src: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
        src = src.transpose(0, 1)
        tar = tar.transpose(0, 1)
        attended = self.self_attn(tar, src, src, attn_mask=None, key_padding_mask=None)[0]
        src = self.norm1(src + self.dropout1(attended))
        src = self.norm2(src + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(src))))))
        return src.transpose(0, 1)


class _TalkNetEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visualFrontend = _VisualFrontend()
        self.visualTCN = _VisualTCN()
        self.visualConv1D = _VisualConv1D()
        self.audioEncoder = _AudioEncoder([3, 4, 6, 3], [16, 32, 64, 128])
        self.crossA2V = _Attention(128, 8)
        self.crossV2A = _Attention(128, 8)
        self.selfAV = _Attention(256, 8)

    def forward_visual_frontend(self, batch: torch.Tensor) -> torch.Tensor:
        groups, frames, width, height = batch.shape
        batch = batch.view(groups * frames, 1, 1, width, height)
        batch = (batch / 255 - 0.4161) / 0.1688
        batch = self.visualFrontend(batch)
        batch = batch.view(groups, frames, 512)
        batch = batch.transpose(1, 2)
        batch = self.visualTCN(batch)
        batch = self.visualConv1D(batch)
        return batch.transpose(1, 2)

    def forward_audio_frontend(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.unsqueeze(1).transpose(2, 3)
        return self.audioEncoder(batch)

    def forward_cross_attention(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio_context = self.crossA2V(src=audio, tar=visual)
        visual_context = self.crossV2A(src=visual, tar=audio)
        return audio_context, visual_context

    def forward_audio_visual_backend(self, audio: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        fused = torch.cat((audio, visual), 2)
        fused = self.selfAV(src=fused, tar=fused)
        return torch.reshape(fused, (-1, 256))


class _Classifier(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.FC = nn.Linear(width, 2)


class TalkNetASD(nn.Module):
    """Checkpoint root. ``model``, ``lossAV``, ``lossA`` and ``lossV`` are the saved names."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _TalkNetEncoder()
        self.lossAV = _Classifier(256)
        self.lossA = _Classifier(128)
        self.lossV = _Classifier(128)


def load_talknet(weights: str | Path, device: str | None = None) -> tuple[TalkNetASD, str]:
    """Load ``pretrain_TalkSet.model`` strictly. A key mismatch raises."""
    path = Path(weights)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    network = TalkNetASD()
    blob = torch.load(path, map_location="cpu", weights_only=False)
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
        logits = network.lossAV.FC(fused)[:, 1]
    return logits.detach().float().cpu().numpy()


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
    for candidate in _FFMPEG_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
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
            "reason": f"TalkNet checkpoint does not match the vendored forward: {exc}",
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
