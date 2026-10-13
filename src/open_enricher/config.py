"""Experiment settings. Edit the constants below; invalid values raise at import.

Tokens come from the repo-root `.env` file (HF_TOKEN). Paths are not secrets.
Metadata generation params live in metadata_params.json next to this file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

# Always the checkout `.env`, not whatever cwd Colab/exec happens to have.
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH)

DRIVE_DATA = "/content/drive/MyDrive/open_enricher/data"
DRIVE_OUTPUT = "/content/drive/MyDrive/open_enricher/outputs"
DRIVE_VIDEO = "Stranger_Things.S01E03.mkv"

LOCAL_OUTPUT = "/content/open-enricher/outputs"

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_DTYPE = "float16"
MODEL_DEVICE_MAP = "cuda"
AttnImplementation = Literal["eager", "sdpa", "flash_attention_2"]
ATTN_IMPLEMENTATIONS: tuple[AttnImplementation, ...] = ("eager", "sdpa", "flash_attention_2")
MODEL_ATTN_IMPLEMENTATION: AttnImplementation = "sdpa"

# None: metadata sees the whole file. A positive int: only the first N seconds.
# The clip is re-encoded before the processor, which always samples the full input it is given.
METADATA_VIDEO_SECONDS: int | None = None

# Frame cap and pixel budget for a ~50 minute episode on a 22GB GPU.
# The Qwen long-video recipe (2048 frames, ~224K visual tokens) OOMs in the
# vision encoder: it tried to allocate another 13 GiB with ~20 GiB already in use.
# 768 frames still span the full duration. longest_edge matches the per-frame
# floor (1.05 * shortest_edge) times that cap, about 50K visual tokens.
METADATA_MIN_FRAMES = 4
METADATA_MAX_FRAMES = 768
METADATA_MAX_VIDEO_TOKENS = 768
METADATA_VIDEO_SHORTEST_EDGE = 128 * 32 * 32
METADATA_VIDEO_LONGEST_EDGE = (128 * 32 * 32 * 105 // 100) * 768

# Prompt body is the Markdown file. It ships with src/, which the Colab tarball already packs.
METADATA_PROMPT_PATH = Path(__file__).resolve().parent / "metadata_prompt.md"
METADATA_PROMPT = METADATA_PROMPT_PATH.read_text(encoding="utf-8").strip()

METADATA_PARAMS_PATH = Path(__file__).resolve().parent / "metadata_params.json"
METADATA_PARAMS_KEYS = (
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
    # `bool` is a subclass of `int`; reject it with an identity check.
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


def validate_metadata_params(payload: Any) -> dict[str, Any]:
    """Validate a full metadata_params object. Returns the normalized values."""

    if not isinstance(payload, dict):
        raise ValueError("metadata_params must be a JSON object")
    missing = [key for key in METADATA_PARAMS_KEYS if key not in payload]
    if missing:
        raise ValueError(f"metadata_params missing keys: {missing}")
    extra = [key for key in payload if key not in METADATA_PARAMS_KEYS]
    if extra:
        raise ValueError(f"metadata_params has unknown keys: {extra}")

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


def load_metadata_params(path: str | Path | None = None) -> dict[str, Any]:
    """Read and validate metadata_params.json. Raises if the file is missing or invalid."""

    params_path = Path(path) if path is not None else METADATA_PARAMS_PATH
    if not params_path.is_file():
        raise FileNotFoundError(f"metadata_params file does not exist: {params_path}")
    try:
        payload = json.loads(params_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"metadata_params is not valid JSON: {params_path}") from exc
    return validate_metadata_params(payload)


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    dtype: str
    device_map: str
    attn_implementation: AttnImplementation


@dataclass(frozen=True)
class DriveConfig:
    root: str
    output_dir: str
    video: str


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


# Frozen snapshots of the constants above. Built at import so a bad edit fails immediately.
MODEL_CONFIG = ModelConfig(
    model_id=_require_str("MODEL_ID", MODEL_ID),
    dtype=_require_str("MODEL_DTYPE", MODEL_DTYPE),
    device_map=_require_str("MODEL_DEVICE_MAP", MODEL_DEVICE_MAP),
    attn_implementation=_require_attn(MODEL_ATTN_IMPLEMENTATION),
)

DRIVE_CONFIG = DriveConfig(
    root=_require_str("DRIVE_DATA", DRIVE_DATA),
    output_dir=_require_str("LOCAL_OUPUT", LOCAL_OUTPUT), # files are not saved on the drive disk, but in a local folder in the Colab VM
    video=_require_str("DRIVE_VIDEO", DRIVE_VIDEO),
)

_METADATA_PARAMS = load_metadata_params()
METADATA_CONFIG = MetadataConfig(
    prompt=_require_str("METADATA_PROMPT", METADATA_PROMPT),
    prompt_path=_require_str("METADATA_PROMPT_PATH", str(METADATA_PROMPT_PATH)),
    max_new_tokens=_METADATA_PARAMS["max_new_tokens"],
    fps=_METADATA_PARAMS["fps"],
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
    do_sample=_METADATA_PARAMS["do_sample"],
    temperature=_METADATA_PARAMS["temperature"],
    top_p=_METADATA_PARAMS["top_p"],
    top_k=_METADATA_PARAMS["top_k"],
    repetition_penalty=_METADATA_PARAMS["repetition_penalty"],
)
