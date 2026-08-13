"""Dataclass configs for model loading, Drive paths, and inference modes.

Non-secret defaults come from config.toml. Tokens come from .env / process env.
"""

from dataclasses import dataclass

from open_enricher.env import env_int, env_str, load_dotenv
from open_enricher.settings import apply_config_to_environ

load_dotenv()
apply_config_to_environ()

DEFAULT_MODEL_ID = env_str("OPEN_ENRICHER_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
DEFAULT_DRIVE_ROOT = env_str("OPEN_ENRICHER_DATA", "/content/drive/MyDrive/open_enricher/data")
DEFAULT_OUTPUT_DIR = env_str("OPEN_ENRICHER_OUTPUT", "/content/drive/MyDrive/open_enricher/outputs")
DEFAULT_NUM_FRAMES = env_int("OPEN_ENRICHER_NUM_FRAMES", 16)
DEFAULT_MAX_NEW_TOKENS = env_int("OPEN_ENRICHER_MAX_NEW_TOKENS", 512)

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
