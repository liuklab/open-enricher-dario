"""EVA-CLIP ViT-G trunk and BLIP-2 Q-Former query tokens.

The released ``eva_vit_g.pth`` is a 40-block ViT-G. BLIP-2 was trained on
the first 39 blocks, so the last block and the contrastive head are left
unloaded. ``blip2_pretrained_flant5xxl.pth`` stores the Q-Former under
``model``. The Flan-T5 projection and the unused position-id buffer are
not part of the visual token. LAVIS is not imported.

A missing or misshapen weight leaves the bank empty. The caller then
returns the history budget to the current clip.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

EVA_IMAGE = 224
EVA_PATCH = 14
EVA_EMBED = 1408
EVA_DEPTH = 39
EVA_HEADS = 16
EVA_MLP_RATIO = 4.3637
QUERY_TOKENS = 32
QFORMER_HIDDEN = 768
QFORMER_HEADS = 12
QFORMER_LAYERS = 12
QFORMER_INTERMEDIATE = 3072
QFORMER_CROSS_EVERY = 2
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
KEY_MISMATCH = "checkpoint keys do not match"


def build_eva_vit(
    *,
    img_size: int = EVA_IMAGE,
    patch_size: int = EVA_PATCH,
    embed_dim: int = EVA_EMBED,
    depth: int = EVA_DEPTH,
    num_heads: int = EVA_HEADS,
    mlp_ratio: float = EVA_MLP_RATIO,
) -> EvaVision:
    """ViT trunk whose parameter names match ``eva_vit_g.pth``."""
    return EvaVision(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )


def build_eva_vit_g() -> EvaVision:
    """BLIP-2 visual trunk: ViT-G/14 at 224, without the 40th block."""
    return build_eva_vit()


def build_qformer(
    *,
    hidden: int = QFORMER_HIDDEN,
    heads: int = QFORMER_HEADS,
    layers: int = QFORMER_LAYERS,
    queries: int = QUERY_TOKENS,
    vision_width: int = EVA_EMBED,
    intermediate: int = QFORMER_INTERMEDIATE,
    cross_every: int = QFORMER_CROSS_EVERY,
) -> BlipQueryBridge:
    """Q-Former whose parameter names match the BLIP-2 checkpoint."""
    return BlipQueryBridge(
        hidden=hidden,
        heads=heads,
        layers=layers,
        queries=queries,
        vision_width=vision_width,
        intermediate=intermediate,
        cross_every=cross_every,
    )


def build_blip2_qformer() -> BlipQueryBridge:
    """32 query tokens over a 1408-d vision sequence."""
    return build_qformer()


class PatchEmbed(nn.Module):
    """Stride-14 convolution used by EVA ViT-G."""

    def __init__(self, img_size: int, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(images)
        return tokens.flatten(2).transpose(1, 2)


class EvaAttention(nn.Module):
    """Fused qkv projection with separate q and v biases, and no k bias."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tokens.shape
        bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = nn.functional.linear(tokens, self.qkv.weight, bias)
        qkv = qkv.reshape(batch, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = (query * self.scale) @ key.transpose(-2, -1)
        context = torch.softmax(scores, dim=-1) @ value
        context = context.transpose(1, 2).reshape(batch, length, -1)
        return self.proj(context)


class EvaMlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.fc2(nn.functional.gelu(self.fc1(tokens)))


class EvaBlock(nn.Module):
    """Pre-norm residual block. Stochastic depth is off at eval."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = EvaAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = EvaMlp(dim, int(dim * mlp_ratio))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attn(self.norm1(tokens))
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class EvaVision(nn.Module):
    """Token sequence including the class token, after the final norm."""

    def __init__(
        self,
        *,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        positions = self.patch_embed.num_patches + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, positions, embed_dim))
        self.blocks = nn.ModuleList(
            EvaBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(images)
        batch = tokens.shape[0]
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        if tokens.shape[1] != self.pos_embed.shape[1]:
            raise RuntimeError(
                "EVA position embedding is "
                f"{self.pos_embed.shape[1]} tokens, frame produced {tokens.shape[1]}"
            )
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


class BertSelfAttention(nn.Module):
    """Bidirectional attention. Cross attention uses the vision width for k and v."""

    def __init__(self, hidden: int, heads: int, kv_dim: int) -> None:
        super().__init__()
        self.num_heads = heads
        self.head_dim = hidden // heads
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(kv_dim, hidden)
        self.value = nn.Linear(kv_dim, hidden)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        batch, query_len, _ = queries.shape
        key_len = keys.shape[1]
        query = self.query(queries).view(batch, query_len, self.num_heads, self.head_dim)
        key = self.key(keys).view(batch, key_len, self.num_heads, self.head_dim)
        value = self.value(keys).view(batch, key_len, self.num_heads, self.head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        context = torch.matmul(torch.softmax(scores, dim=-1), value)
        context = context.transpose(1, 2).contiguous().view(batch, query_len, -1)
        return context


class BertSelfOutput(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=1e-12)

    def forward(self, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(self.dense(hidden) + residual)


class BertAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_dim: int) -> None:
        super().__init__()
        self.self = BertSelfAttention(hidden, heads, kv_dim)
        self.output = BertSelfOutput(hidden)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        context = self.self(queries, keys)
        return self.output(context, queries)


class QueryIntermediate(nn.Module):
    """Query-only MLP input. The text-side feed-forward is not in the checkpoint."""

    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden, intermediate)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(self.dense(hidden))


class QueryOutput(nn.Module):
    def __init__(self, intermediate: int, hidden: int) -> None:
        super().__init__()
        self.dense = nn.Linear(intermediate, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=1e-12)

    def forward(self, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(self.dense(hidden) + residual)


class QFormerLayer(nn.Module):
    def __init__(
        self,
        hidden: int,
        heads: int,
        vision_width: int,
        intermediate: int,
        *,
        cross: bool,
    ) -> None:
        super().__init__()
        self.attention = BertAttention(hidden, heads, hidden)
        if cross:
            self.crossattention = BertAttention(hidden, heads, vision_width)
        self.intermediate_query = QueryIntermediate(hidden, intermediate)
        self.output_query = QueryOutput(intermediate, hidden)

    def forward(self, queries: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        hidden = self.attention(queries, queries)
        cross = getattr(self, "crossattention", None)
        if cross is not None:
            hidden = cross(hidden, vision)
        return self.output_query(self.intermediate_query(hidden), hidden)


class QFormerEncoder(nn.Module):
    def __init__(
        self,
        hidden: int,
        heads: int,
        layers: int,
        vision_width: int,
        intermediate: int,
        cross_every: int,
    ) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            QFormerLayer(
                hidden,
                heads,
                vision_width,
                intermediate,
                cross=index % cross_every == 0,
            )
            for index in range(layers)
        )

    def forward(self, queries: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        hidden = queries
        for layer in self.layer:
            hidden = layer(hidden, vision)
        return hidden


class BlipQueryBridge(nn.Module):
    """Layer-norm on the vision tokens, then the learned query tokens."""

    def __init__(
        self,
        *,
        hidden: int,
        heads: int,
        layers: int,
        queries: int,
        vision_width: int,
        intermediate: int,
        cross_every: int,
    ) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, queries, hidden))
        self.ln_vision = nn.LayerNorm(vision_width)
        self.Qformer = nn.Module()
        self.Qformer.bert = nn.Module()  # type: ignore[assignment]
        self.Qformer.bert.embeddings = nn.Module()  # type: ignore[assignment]
        self.Qformer.bert.embeddings.LayerNorm = nn.LayerNorm(hidden, eps=1e-12)
        self.Qformer.bert.encoder = QFormerEncoder(  # type: ignore[assignment]
            hidden,
            heads,
            layers,
            vision_width,
            intermediate,
            cross_every,
        )

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        features = self.ln_vision(vision_tokens)
        batch = features.shape[0]
        queries = self.query_tokens.expand(batch, -1, -1)
        hidden = self.Qformer.bert.embeddings.LayerNorm(queries)
        return self.Qformer.bert.encoder(hidden, features)


def parameter_mismatch(module: nn.Module, state: dict[str, Any]) -> str | None:
    """
    Return a short reason when a required weight is missing or the wrong shape.

    Extra checkpoint entries are ignored. The EVA file's 40th block, its
    contrastive head, and the Q-Former file's T5 projection are extras.
    """
    missing: list[str] = []
    shaped: list[str] = []
    for name, param in module.state_dict().items():
        value = state.get(name)
        if not torch.is_tensor(value):
            missing.append(name)
            continue
        if tuple(value.shape) != tuple(param.shape):
            shaped.append(name)
    if missing:
        shown = ", ".join(missing[:6])
        return f"{KEY_MISMATCH}; missing {shown}"
    if shaped:
        shown = ", ".join(shaped[:6])
        return f"{KEY_MISMATCH}; shape {shown}"
    return None


def load_matching(module: nn.Module, state: dict[str, Any]) -> str | None:
    """Copy matching tensors into ``module``. The reason is None on success."""
    reason = parameter_mismatch(module, state)
    if reason:
        return reason
    module_state = module.state_dict()
    filtered = {
        name: state[name].to(dtype=module_state[name].dtype) for name in module_state
    }
    module.load_state_dict(filtered, strict=True)
    return None


def encode_images(
    vision: EvaVision,
    bridge: BlipQueryBridge,
    images: torch.Tensor,
) -> torch.Tensor:
    """Return ``[batch, queries, hidden]`` query tokens for normalized images."""
    return bridge(vision(images))


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
    """BGR uint8 frame from OpenCV to a CLIP-normalized 224 CHW tensor."""
    import cv2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype("float32") / 255.0
    resized = cv2.resize(rgb, (EVA_IMAGE, EVA_IMAGE), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous()
    mean = torch.tensor(CLIP_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def _sample_frames(video: str, sample_fps: float, max_frames: int | None) -> list[tuple[int, float, Any]]:
    import cv2

    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise RuntimeError(f"could not open video {video}")
    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    step = max(1, int(round(native_fps / max(sample_fps, 0.01))))
    chosen: list[tuple[int, float, Any]] = []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                chosen.append((index, index / native_fps, frame))
                if max_frames is not None and len(chosen) >= max_frames:
                    break
            index += 1
    finally:
        capture.release()
    return chosen


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
        eva_blob = torch.load(payload["eva"], map_location="cpu", weights_only=False)
        qformer_blob = torch.load(payload["qformer"], map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - recorded fallback
        return {"status": "fallback", "reason": str(exc), "frames": []}
    eva_count = _tensor_count(eva_blob)
    try:
        qformer_state = _qformer_state(qformer_blob)
    except TypeError as exc:
        return {"status": "fallback", "reason": str(exc), "frames": []}
    qformer_count = _tensor_count(qformer_state)
    counts = {"eva": eva_count, "qformer": qformer_count}
    vision = build_eva_vit_g()
    bridge = build_blip2_qformer()
    eva_reason = load_matching(vision, eva_blob if isinstance(eva_blob, dict) else {})
    if eva_reason:
        return _mismatch(eva_reason, counts)
    qformer_reason = load_matching(bridge, qformer_state)
    if qformer_reason:
        return _mismatch(qformer_reason, counts)
    del eva_blob, qformer_blob, qformer_state
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision, device = _place(vision, device)
    bridge, device = _place(bridge, device)
    vision.eval()
    bridge.eval()
    limit = payload.get("max_frames")
    try:
        sampled = _sample_frames(
            str(payload["video"]),
            float(payload["sample_fps"]),
            None if limit is None else int(limit),
        )
    except Exception as exc:  # noqa: BLE001 - recorded fallback
        return {
            "status": "fallback",
            "reason": str(exc),
            "checkpoint_present": True,
            "tensors": counts,
            "frames": [],
        }
    frames: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for index, stamp, frame in sampled:
                image = _prepare_frame(frame).unsqueeze(0).to(device)
                tokens = encode_images(vision, bridge, image)[0]
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
    except RuntimeError as exc:
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
