"""Shared local dataset paths for Open-Enricher.

One dataset serves every experiment: source videos, model caches, and live
stage outputs. Per-experiment archives live under ``runs/``, not here.

Default root is ``<repository>/dataset``. Override with
``OPEN_ENRICHER_DATASET_ROOT`` or point videos elsewhere with
``OPEN_ENRICHER_VIDEO_INPUTS``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


from open_enricher.logging_setup import get_logger

logger = get_logger(__name__)

_STAGE_DIRS: dict[str, str] = {
    "audio": "audio_results",
    "color": "color_results",
    "dialogue": "dialogue_results",
    "vlm": "vlms_results",
    "llm": "llms_results",
    "timeline": "timeline_results",
    "bassl": "bassl_results",
    "character_graph": "graph_results",
    "visual_memory": "memory_results",
    "evidence": "evidence_results",
}


def _repo_root() -> Path:
    """
    Resolve the repository root from this package file.

    ``storage.py`` lives at ``src/open_enricher/storage.py``, so two parents
    up from the package directory lands on the checkout root.
    """
    return Path(__file__).resolve().parents[2]


def dataset_root() -> Path:
    """
    Return the shared dataset root for this process.

    Preference order:
    1. ``OPEN_ENRICHER_DATASET_ROOT`` if set
    2. ``OPEN_ENRICHER_DATA_ROOT`` if set (legacy alias)
    3. ``<repo>/dataset``

    Returns:
        Path to the dataset directory.
    """
    override = (
        os.environ.get("OPEN_ENRICHER_DATASET_ROOT")
        or os.environ.get("OPEN_ENRICHER_DATA_ROOT")
    )
    if override:
        return Path(override)
    return _repo_root() / "dataset"


def video_inputs() -> Path:
    """
    Return the directory that holds source video files.

    Returns:
        ``OPEN_ENRICHER_VIDEO_INPUTS`` when set, otherwise
        ``dataset_root()/video_inputs``.
    """
    override = os.environ.get("OPEN_ENRICHER_VIDEO_INPUTS")
    if override:
        return Path(override)
    return dataset_root() / "video_inputs"


def outputs_dir() -> Path:
    """
    Return the shared live outputs directory.

    Returns:
        ``dataset_root()/outputs``.
    """
    override = os.environ.get("OPEN_ENRICHER_OUTPUT_ROOT")
    return Path(override) if override else dataset_root() / "outputs"


def models_cache() -> Path:
    """
    Return the model weight cache directory.

    Returns:
        ``dataset_root()/models_cache``.
    """
    return dataset_root() / "models_cache"


def experiment_log() -> Path:
    """
    Return the path of the shared worker log under outputs.

    Returns:
        ``outputs_dir()/worker.log``.
    """
    return outputs_dir() / "worker.log"


def get_output_dir(stage_name: str) -> Path:
    """
    Return the shared output directory for a named stage.

    Args:
        stage_name: One of audio, color, dialogue, vlm, llm.

    Returns:
        Path under ``dataset/outputs/<stage>_results``.

    Raises:
        ValueError: If ``stage_name`` is not a known stage key.
    """
    if stage_name not in _STAGE_DIRS:
        raise ValueError(f"Unknown stage '{stage_name}'.")
    return outputs_dir() / _STAGE_DIRS[stage_name]


def setup_dataset() -> None:
    """
    Create the shared folder structure if it does not exist.

    Ensures model caches, default ``video_inputs``, and every stage output
    folder exist. Raises ``FileNotFoundError`` when
    ``OPEN_ENRICHER_VIDEO_INPUTS`` points at a missing directory.
    """
    root = dataset_root()
    logger.info(f"Initializing shared dataset at {root}...")

    cache = models_cache()
    hf_cache = cache / "huggingface"
    torch_cache = cache / "torch"
    cache.mkdir(parents=True, exist_ok=True)
    hf_cache.mkdir(exist_ok=True)
    torch_cache.mkdir(exist_ok=True)

    out = outputs_dir()
    out.mkdir(parents=True, exist_ok=True)

    inputs = video_inputs()
    if inputs == root / "video_inputs":
        inputs.mkdir(parents=True, exist_ok=True)
    elif not inputs.is_dir():
        raise FileNotFoundError(f"Video input directory does not exist: {inputs}")

    for stage_name in _STAGE_DIRS:
        get_output_dir(stage_name).mkdir(parents=True, exist_ok=True)

    logger.info("Dataset structure verified.")


def atomic_write_text(path: Path, text: str) -> None:
    """
    Write ``text`` by replacing ``path`` only after the temp file is flushed.

    A crash while the temp file is open leaves the previous destination in
    place.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def configure_environment() -> None:
    """
    Route Hugging Face and PyTorch caches into the shared dataset.

    Sets ``HF_HOME``, ``TORCH_HOME``, disables hub symlinks, and enables
    expandable CUDA allocator segments.
    """
    cache = models_cache()
    os.environ["HF_HOME"] = str(cache / "huggingface")
    os.environ["TORCH_HOME"] = str(cache / "torch")
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
