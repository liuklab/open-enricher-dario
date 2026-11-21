"""Vendored EVA ViT-G and BLIP-2 Q-Former token forward."""

from __future__ import annotations

from pathlib import Path

import torch

from open_enricher.adapters.eva_qformer import (
    KEY_MISMATCH,
    build_blip2_qformer,
    build_eva_vit,
    build_eva_vit_g,
    build_qformer,
    encode_images,
    parameter_mismatch,
)
from open_enricher.adapters.visual_encode import encode_video
from open_enricher.stages.pipeline_spec import Checkpoints, PipelineSpec


def test_production_names_match_the_released_checkpoints() -> None:
    with torch.device("meta"):
        vision = build_eva_vit_g()
        bridge = build_blip2_qformer()
    vision_state = vision.state_dict()
    bridge_state = bridge.state_dict()
    # 39 blocks x 13 tensors, plus cls, pos, patch weight/bias, and the final norm.
    assert len(vision_state) == 513
    assert "blocks.39.norm1.weight" not in vision_state
    assert tuple(vision_state["pos_embed"].shape) == (1, 257, 1408)
    assert tuple(vision_state["blocks.0.attn.qkv.weight"].shape) == (4224, 1408)
    assert tuple(vision_state["blocks.0.mlp.fc1.weight"].shape) == (6144, 1408)
    # 6 cross-attention layers and 6 self-only layers, plus queries and two layer-norms.
    assert len(bridge_state) == 257
    assert tuple(bridge_state["query_tokens"].shape) == (1, 32, 768)
    key = "Qformer.bert.encoder.layer.0.crossattention.self.key.weight"
    assert tuple(bridge_state[key].shape) == (768, 1408)
    assert "Qformer.bert.encoder.layer.1.crossattention.self.key.weight" not in bridge_state
    assert "t5_proj.weight" not in bridge_state


def test_tiny_forward_emits_finite_query_tokens() -> None:
    vision = build_eva_vit(
        img_size=16,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )
    bridge = build_qformer(
        hidden=32,
        heads=4,
        layers=2,
        queries=4,
        vision_width=32,
        intermediate=64,
        cross_every=2,
    )
    images = torch.randn(2, 3, 16, 16)
    tokens = encode_images(vision, bridge, images)
    assert tuple(tokens.shape) == (2, 4, 32)
    assert torch.isfinite(tokens).all()


def test_extra_checkpoint_keys_are_ignored_and_a_missing_weight_is_not() -> None:
    vision = build_eva_vit(
        img_size=16,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )
    state = {name: value.clone() for name, value in vision.state_dict().items()}
    state["blocks.39.norm1.weight"] = torch.zeros(32)
    state["head.weight"] = torch.zeros(8, 32)
    assert parameter_mismatch(vision, state) is None
    del state["blocks.0.attn.qkv.weight"]
    reason = parameter_mismatch(vision, state)
    assert reason is not None
    assert KEY_MISMATCH in reason
    assert "blocks.0.attn.qkv.weight" in reason


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
