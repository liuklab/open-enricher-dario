"""Core modular components for the Vision-Language Model stage."""

from src.open_enricher.remote.stages.vlm_core.config import (
    VLM_PARAMS_PATH,
    VLM_PROMPTS_PATH,
    MetadataConfig,
    ModelConfig,
    build_configs,
    load_vlm_params,
    validate_vlm_params,
)
from src.open_enricher.remote.stages.vlm_core.model import (
    extract_metadata,
    load_model,
    unload_model,
)
from src.open_enricher.remote.stages.vlm_core.video import video_input_path

__all__ = [
    "VLM_PARAMS_PATH",
    "VLM_PROMPTS_PATH",
    "MetadataConfig",
    "ModelConfig",
    "build_configs",
    "extract_metadata",
    "load_vlm_params",
    "load_model",
    "unload_model",
    "validate_vlm_params",
    "video_input_path",
]