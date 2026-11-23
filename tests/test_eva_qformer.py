"""Checkpoint loading, BLIP-2 image preprocessing and the public sampler."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from open_enricher.adapters.eva_qformer import CLIP_MEAN, CLIP_STD, _prepare_frame, load_matching
from open_enricher.adapters.visual_encode import encode_video
from open_enricher.stages.pipeline_spec import Checkpoints, PipelineSpec


def test_checkpoint_requires_each_inference_tensor_but_allows_unused_heads() -> None:
    layer = torch.nn.Linear(5, 3)
    state = dict(layer.state_dict(), **{"t5_proj.weight": torch.empty(17, 3)})
    assert load_matching(layer, state) is None
    assert "weight" in load_matching(layer, dict(state, weight=torch.empty(4, 5)))
    del state["bias"]
    assert "bias" in load_matching(layer, state)


def test_frame_matches_lavis_rgb_bicubic_eval_transform() -> None:
    rng = np.random.default_rng(12)
    bgr = rng.integers(0, 256, (283, 493, 3), dtype=np.uint8)
    rgb = Image.fromarray(bgr[:, :, ::-1]).resize((224, 224), Image.Resampling.BICUBIC)
    expected = np.asarray(rgb).astype(np.float32) / 255
    expected = (expected - np.array(CLIP_MEAN, np.float32)) / np.array(CLIP_STD, np.float32)
    np.testing.assert_allclose(_prepare_frame(bgr).numpy(), expected.transpose(2, 0, 1), atol=1e-6)


def test_max_frames_is_sent_only_when_set(monkeypatch, tmp_path: Path) -> None:
    eva = tmp_path / "eva.pth"
    qformer = tmp_path / "qformer.pth"
    eva.write_bytes(b"x")
    qformer.write_bytes(b"x")
    captured: list[dict] = []

    def fake(module: str, function: str, payload: dict, **kwargs: object) -> dict:
        del module, function, kwargs
        captured.append(payload)
        return {"status": "success", "frames": []}

    monkeypatch.setattr("open_enricher.adapters.visual_encode.call_in_child", fake)
    spec = PipelineSpec(
        experiment_id="eva-limit",
        title="eva-limit",
        interval_seconds=300,
        vlm_memory_scenes=None,
        visual_only=False,
        narrate_from_visual=False,
        checkpoints=Checkpoints(eva_clip=str(eva), qformer=str(qformer)),
    )
    encode_video(tmp_path / "video.mkv", spec, sample_fps=0.25, max_frames=4)
    encode_video(tmp_path / "video.mkv", spec, sample_fps=0.25)
    assert captured[0]["max_frames"] == 4
    assert "max_frames" not in captured[1]


def test_released_qformer_on_two_video_frames() -> None:
    """Opt-in smoke test using the public, isolated adapter and local weights."""
    source = os.environ.get("OPEN_ENRICHER_SMOKE_VIDEO")
    if not source:
        pytest.skip("Set OPEN_ENRICHER_SMOKE_VIDEO to a clip with two distinct frames four seconds apart")
    spec = PipelineSpec(
        experiment_id="encoder-smoke",
        title="encoder smoke",
        interval_seconds=300,
        vlm_memory_scenes=None,
        visual_only=False,
        narrate_from_visual=False,
    )
    result = encode_video(Path(source), spec, sample_fps=0.25, max_frames=2)
    assert result["status"] == "success", result.get("reason")
    frames = result["frames"]
    assert len(frames) == 2 and frames[0]["timestamp"] < frames[1]["timestamp"]
    tokens = np.array([frame["tokens"] for frame in frames])
    assert tokens.shape == (2, 32, 768)
    assert np.isfinite(tokens).all()
    assert not np.allclose(tokens[0], tokens[1])
