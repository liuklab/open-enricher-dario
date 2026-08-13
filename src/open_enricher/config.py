"""Dataclass configs for model loading, Drive paths, and inference modes.

Non-secret defaults come from config.toml. Tokens come from .env / process env.
"""

from dataclasses import dataclass
from typing import Any

from open_enricher.env import load_dotenv
from open_enricher.settings import load_project_config

load_dotenv()
_CFG = load_project_config()

_FALLBACKS: dict[tuple[str, str], Any] = {
    ("model", "id"): "Qwen/Qwen3-VL-2B-Instruct",
    ("drive", "data"): "/content/drive/MyDrive/open_enricher/data",
    ("drive", "output"): "/content/drive/MyDrive/open_enricher/outputs",
    ("drive", "video"): "",
    ("model", "num_frames"): 16,
    ("model", "max_new_tokens"): 512,
}


def _setting(section: str, key: str) -> Any:
    fallback = _FALLBACKS[(section, key)]
    value = _CFG.get(section, {}).get(key, fallback)
    if value is None or value == "":
        return fallback
    return value


DEFAULT_MODEL_ID = str(_setting("model", "id"))
DEFAULT_DRIVE_ROOT = str(_setting("drive", "data"))
DEFAULT_OUTPUT_DIR = str(_setting("drive", "output"))
DEFAULT_VIDEO = str(_setting("drive", "video"))
DEFAULT_NUM_FRAMES = int(_setting("model", "num_frames"))
DEFAULT_MAX_NEW_TOKENS = int(_setting("model", "max_new_tokens"))

DEFAULT_METADATA_PROMPT = (
    "Analyze this video and return JSON only with these keys:\n"
    '"title_guess": string,\n'
    '"short_summary": string,\n'
    '"tags": list of strings,\n'
    '"notable_objects": list of strings,\n'
    '"notable_scenes": list of strings\n'
    "Do not include any text outside the JSON object."
)

DEFAULT_EMBEDDING_PROMPT = "Represent this video."


@dataclass
class ModelConfig:
    """How to load a Qwen3-VL (or compatible) checkpoint.

    Change `model_id` to swap sizes, for example Qwen/Qwen3-VL-8B-Instruct.
    """

    model_id: str = DEFAULT_MODEL_ID
    dtype: str = "auto"
    device_map: str = "auto"
    attn_implementation: str | None = None


@dataclass
class DriveConfig:
    """Generic Google Drive locations for videos and inference outputs."""

    root: str = DEFAULT_DRIVE_ROOT
    output_dir: str = DEFAULT_OUTPUT_DIR
    video: str = DEFAULT_VIDEO


@dataclass
class MetadataConfig:
    """Settings for prompted textual metadata extraction."""

    prompt: str = DEFAULT_METADATA_PROMPT
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    # Fixed frame count is the default because it bounds GPU memory on Colab.
    num_frames: int | None = DEFAULT_NUM_FRAMES
    fps: float | None = None


@dataclass
class EmbeddingConfig:
    """Settings for pooled video embedding extraction."""

    prompt: str = DEFAULT_EMBEDDING_PROMPT
    num_frames: int | None = DEFAULT_NUM_FRAMES
    fps: float | None = None


def sampling_kwargs(num_frames: int | None, fps: float | None) -> dict:
    """Build processor kwargs. `num_frames` and `fps` are mutually exclusive."""

    if num_frames is not None:
        return {"num_frames": num_frames, "fps": None}
    if fps is not None:
        return {"fps": fps}
    return {}
