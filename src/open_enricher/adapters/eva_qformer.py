"""Extract BLIP-2 image query tokens with the original Salesforce LAVIS encoders."""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from typing import Any

import torch
from torch import nn

from open_enricher.adapters.upstream import activate

EVA_IMAGE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
KEY_MISMATCH = "checkpoint keys do not match"


def build_eva_vit_g() -> nn.Module:
    from lavis.models.eva_vit import VisionTransformer

    # The constructor from LAVIS create_eva_vit_g, with a local checkpoint.
    return VisionTransformer(
        img_size=EVA_IMAGE,
        patch_size=14,
        use_mean_pooling=False,
        embed_dim=1408,
        depth=39,
        num_heads=16,
        mlp_ratio=4.3637,
        qkv_bias=True,
        drop_path_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


class QueryEncoder(nn.Module):
    """Load the image path of Blip2T5 without constructing the text decoder."""

    def __init__(self) -> None:
        super().__init__()
        from lavis.models.blip2_models.Qformer import BertConfig, BertModel

        config = BertConfig()
        config.encoder_width = 1408
        config.add_cross_attention = True
        config.cross_attention_freq = 2
        config.query_length = 32
        self.ln_vision = nn.LayerNorm(config.encoder_width)
        self.query_tokens = nn.Parameter(torch.empty(1, config.query_length, config.hidden_size))
        self.bert = BertModel(config, add_pooling_layer=False)
        self.bert.embeddings.word_embeddings = None
        self.bert.embeddings.position_embeddings = None
        for layer in self.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        features = self.ln_vision(vision_tokens)
        mask = torch.ones(features.shape[:-1], dtype=torch.long, device=features.device)
        return self.bert(
            query_embeds=self.query_tokens.expand(features.shape[0], -1, -1),
            encoder_hidden_states=features,
            encoder_attention_mask=mask,
            return_dict=True,
        ).last_hidden_state


def load_matching(module: nn.Module, state: dict[str, Any]) -> str | None:
    """Require the entire inference path, ignoring weights for unused heads."""
    used = {name: state[name] for name in module.state_dict() if name in state}
    try:
        module.load_state_dict(used, strict=True)
    except RuntimeError as exc:
        return f"{KEY_MISMATCH}: {exc}"
    return None


def _tensor_count(blob: Any) -> int:
    if isinstance(blob, dict):
        return sum(1 for value in blob.values() if torch.is_tensor(value))
    return 0


def _qformer_state(blob: Any) -> dict[str, Any]:
    if isinstance(blob, dict) and isinstance(blob.get("model"), dict):
        return blob["model"]
    if isinstance(blob, dict):
        return blob
    raise TypeError("Q-Former checkpoint is not a state dict")


def _place(module: nn.Module, device: torch.device) -> tuple[nn.Module, torch.device]:
    """Move to ``device``. A CUDA out-of-memory error keeps the module on CPU."""
    if device.type != "cuda":
        return module.to(device), device
    try:
        return module.to(device), device
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        cpu = torch.device("cpu")
        return module.to(cpu), cpu


def _prepare_frame(frame: Any) -> torch.Tensor:
    """Apply the PIL bicubic resize and CLIP normalization used by LAVIS eval."""
    import cv2
    from PIL import Image
    from torchvision.transforms import InterpolationMode, functional as TF

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = TF.resize(image, [EVA_IMAGE, EVA_IMAGE], InterpolationMode.BICUBIC)
    return TF.normalize(TF.to_tensor(image), CLIP_MEAN, CLIP_STD)


def _sample_frames(video: str, sample_fps: float, max_frames: int | None) -> Iterator[tuple[int, float, Any]]:
    import cv2

    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise RuntimeError(f"could not open video {video}")
    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    step = max(1, int(round(native_fps / max(sample_fps, 0.01))))
    count = 0
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                yield index, index / native_fps, frame
                count += 1
                if max_frames is not None and count >= max_frames:
                    break
            index += 1
    finally:
        capture.release()


def _token_rows(tokens: torch.Tensor) -> list[list[float]]:
    array = tokens.detach().float().cpu()
    return [[round(float(value), 6) for value in row] for row in array.tolist()]


def encode_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Load both checkpoints and emit one query matrix per sampled frame.

    ``max_frames`` caps the sample. Unset, every frame at ``sample_fps`` is
    encoded. The pipeline leaves the cap unset.
    """
    try:
        activate("lavis")
        eva_blob = torch.load(payload["eva"], map_location="cpu", weights_only=True)
        qformer_blob = torch.load(payload["qformer"], map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - recorded fallback
        return {"status": "fallback", "reason": str(exc), "frames": []}
    eva_count = _tensor_count(eva_blob)
    try:
        qformer_state = _qformer_state(qformer_blob)
    except TypeError as exc:
        return {"status": "fallback", "reason": str(exc), "frames": []}
    qformer_count = _tensor_count(qformer_state)
    counts = {"eva": eva_count, "qformer": qformer_count}
    try:
        vision = build_eva_vit_g()
        bridge = QueryEncoder()
    except (ImportError, OSError, RuntimeError) as exc:
        return {"status": "fallback", "reason": str(exc), "frames": [], "tensors": counts}
    eva_reason = load_matching(vision, eva_blob if isinstance(eva_blob, dict) else {})
    if eva_reason:
        return _mismatch(eva_reason, counts)
    bridge_state = {name.removeprefix("Qformer."): value for name, value in qformer_state.items()}
    qformer_reason = load_matching(bridge, bridge_state)
    if qformer_reason:
        return _mismatch(qformer_reason, counts)
    del eva_blob, qformer_blob, qformer_state, bridge_state
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders, device = _place(nn.Sequential(vision, bridge), device)
    encoders.eval()
    limit = payload.get("max_frames")
    frames: list[dict[str, Any]] = []
    try:
        sampled = _sample_frames(
            str(payload["video"]), float(payload["sample_fps"]),
            None if limit is None else int(limit),
        )
        with torch.inference_mode():
            for index, stamp, frame in sampled:
                image = _prepare_frame(frame).unsqueeze(0).to(device)
                tokens = encoders(image)[0]
                if not torch.isfinite(tokens).all():
                    return {
                        "status": "fallback",
                        "reason": "EVA/Q-Former forward produced non-finite tokens",
                        "checkpoint_present": True,
                        "tensors": counts,
                        "frames": [],
                    }
                frames.append(
                    {
                        "frame_id": f"f{index:06d}",
                        "timestamp": round(float(stamp), 3),
                        "tokens": _token_rows(tokens),
                    }
                )
                del image, tokens
    except (RuntimeError, ValueError, OSError) as exc:
        return {
            "status": "fallback",
            "reason": str(exc),
            "checkpoint_present": True,
            "tensors": counts,
            "frames": [],
        }
    if not frames:
        return {
            "status": "fallback",
            "reason": "no frames decoded",
            "checkpoint_present": True,
            "tensors": counts,
            "frames": [],
        }
    return {
        "status": "success",
        "reason": "",
        "checkpoint_present": True,
        "tensors": counts,
        "device": device.type,
        "queries": len(frames[0]["tokens"]),
        "width": len(frames[0]["tokens"][0]),
        "frames": frames,
    }


def _mismatch(reason: str, counts: dict[str, int]) -> dict[str, Any]:
    return {
        "status": "fallback",
        "reason": f"{reason}; history budget returns to the current clip",
        "checkpoint_present": True,
        "tensors": counts,
        "frames": [],
    }
