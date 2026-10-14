"""Configuration and strict parameter validation for the VLM stage."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_DTYPE = "float16"
MODEL_DEVICE_MAP = "auto"
MODEL_LOAD_IN_4BIT = True

AttnImplementation = Literal["eager", "sdpa", "flash_attention_2"]
ATTN_IMPLEMENTATIONS: tuple[AttnImplementation, ...] = ("eager", "sdpa", "flash_attention_2")
MODEL_ATTN_IMPLEMENTATION: AttnImplementation = "sdpa"

# None: analizza l'intero file. Un intero positivo ricodifica solo i primi N secondi.
# Per un film di 50 minuti il prefisso nasconderebbe tutto tranne l'apertura.
METADATA_VIDEO_SECONDS: int | None = None

# Budget tarato per una T4 da 15 GB con Qwen3-VL-8B in 4-bit.
# 64 frame sul film intero (~3000 s) sono un campione ogni ~47 s, spaziati in modo
# uniforme, e restano intorno ai 2500 token visivi (~5.2M pixel). 768 frame stanno
# su una GPU da 22 GB e vanno in out-of-memory sulla T4.
# fps in vlm_params.json deve restare sopra 64/3000 ≈ 0.022: sotto quel valore il
# tetto non scatta e la copertura peggiora. Con fps=2 il tetto è quello che conta.
METADATA_MIN_FRAMES = 4
METADATA_MAX_FRAMES = 64
METADATA_MAX_VIDEO_TOKENS = 128
METADATA_VIDEO_SHORTEST_EDGE = 64 * 32 * 32
METADATA_VIDEO_LONGEST_EDGE = (80 * 32 * 32) * METADATA_MAX_FRAMES

VLM_CORE_DIR = Path(__file__).resolve().parent
VLM_PROMPTS_PATH = VLM_CORE_DIR / "vlm_prompts.md"
VLM_PARAMS_PATH = VLM_CORE_DIR / "vlm_params.json"

VLM_PARAMS_KEYS = (
    "fps",
    "max_new_tokens",
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
)


def _require_str(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_bool(name: str, value: bool) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return value


def _require_positive_int(name: str, value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_optional_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    return _require_positive_int(name, value)


def _require_int_range(name: str, value: int, minimum: int, maximum: int) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise ValueError(f"{name} must be an int")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}")
    return value


def _require_float_range(
    name: str,
    value: float,
    minimum: float,
    maximum: float,
    *,
    min_exclusive: bool = False,
) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if min_exclusive:
        if number <= minimum or number > maximum:
            raise ValueError(
                f"{name} must be greater than {minimum} and at most {maximum}, got {number}"
            )
    elif number < minimum or number > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {number}")
    return number


def _require_attn(value: str) -> AttnImplementation:
    if value not in ATTN_IMPLEMENTATIONS:
        raise ValueError(f"MODEL_ATTN_IMPLEMENTATION must be one of {ATTN_IMPLEMENTATIONS}, got {value!r}")
    return value  # type: ignore[return-value]


def validate_vlm_params(payload: Any) -> dict[str, Any]:
    """Validate a full vlm_params object. Returns the normalized values."""
    if not isinstance(payload, dict):
        raise ValueError("vlm_params must be a JSON object")
    missing = [key for key in VLM_PARAMS_KEYS if key not in payload]
    if missing:
        raise ValueError(f"vlm_params missing keys: {missing}")
    extra = [key for key in payload if key not in VLM_PARAMS_KEYS]
    if extra:
        raise ValueError(f"vlm_params has unknown keys: {extra}")

    return {
        "fps": _require_float_range("fps", payload["fps"], 0.0, 4.0, min_exclusive=True),
        "max_new_tokens": _require_int_range("max_new_tokens", payload["max_new_tokens"], 1, 8192),
        "do_sample": _require_bool("do_sample", payload["do_sample"]),
        "temperature": _require_float_range(
            "temperature", payload["temperature"], 0.0, 1.2, min_exclusive=True
        ),
        "top_p": _require_float_range("top_p", payload["top_p"], 0.0, 1.0, min_exclusive=True),
        "top_k": _require_int_range("top_k", payload["top_k"], 1, 100),
        "repetition_penalty": _require_float_range(
            "repetition_penalty", payload["repetition_penalty"], 1.0, 1.5
        ),
    }


def load_vlm_params(path: str | Path | None = None) -> dict[str, Any]:
    """Read and validate vlm_params.json. Raises if missing or invalid."""
    params_path = Path(path) if path is not None else VLM_PARAMS_PATH
    if not params_path.is_file():
        raise FileNotFoundError(f"vlm_params file does not exist: {params_path}")
    try:
        payload = json.loads(params_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"vlm_params is not valid JSON: {params_path}") from exc
    return validate_vlm_params(payload)


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    dtype: str
    device_map: str
    attn_implementation: AttnImplementation
    load_in_4bit: bool


@dataclass(frozen=True)
class MetadataConfig:
    prompt: str
    prompt_path: str
    max_new_tokens: int
    fps: float
    video_seconds: int | None
    min_frames: int
    max_frames: int
    max_video_tokens: int
    video_shortest_edge: int
    video_longest_edge: int
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float


def build_configs() -> tuple[ModelConfig, MetadataConfig]:
    """Reads vlm_prompts.md and vlm_params.json from disk and returns validated configs."""
    if not VLM_PROMPTS_PATH.is_file():
        raise FileNotFoundError(f"VLM prompt file does not exist: {VLM_PROMPTS_PATH}")
    prompt_text = VLM_PROMPTS_PATH.read_text(encoding="utf-8").strip()
    params = load_vlm_params(VLM_PARAMS_PATH)

    model_config = ModelConfig(
        model_id=_require_str("MODEL_ID", MODEL_ID),
        dtype=_require_str("MODEL_DTYPE", MODEL_DTYPE),
        device_map=_require_str("MODEL_DEVICE_MAP", MODEL_DEVICE_MAP),
        attn_implementation=_require_attn(MODEL_ATTN_IMPLEMENTATION),
        load_in_4bit=_require_bool("MODEL_LOAD_IN_4BIT", MODEL_LOAD_IN_4BIT),
    )

    metadata_config = MetadataConfig(
        prompt=_require_str("VLM_PROMPTS", prompt_text),
        prompt_path=_require_str("VLM_PROMPTS_PATH", str(VLM_PROMPTS_PATH)),
        max_new_tokens=params["max_new_tokens"],
        fps=params["fps"],
        video_seconds=_require_optional_positive_int("METADATA_VIDEO_SECONDS", METADATA_VIDEO_SECONDS),
        min_frames=_require_positive_int("METADATA_MIN_FRAMES", METADATA_MIN_FRAMES),
        max_frames=_require_positive_int("METADATA_MAX_FRAMES", METADATA_MAX_FRAMES),
        max_video_tokens=_require_positive_int("METADATA_MAX_VIDEO_TOKENS", METADATA_MAX_VIDEO_TOKENS),
        video_shortest_edge=_require_positive_int(
            "METADATA_VIDEO_SHORTEST_EDGE", METADATA_VIDEO_SHORTEST_EDGE
        ),
        video_longest_edge=_require_positive_int(
            "METADATA_VIDEO_LONGEST_EDGE", METADATA_VIDEO_LONGEST_EDGE
        ),
        do_sample=params["do_sample"],
        temperature=params["temperature"],
        top_p=params["top_p"],
        top_k=params["top_k"],
        repetition_penalty=params["repetition_penalty"],
    )

    return model_config, metadata_config