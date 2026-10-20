"""Local data lake for Open-Enricher.

Owns the on-disk layout used by every pipeline stage: source videos,
downloaded model weights, and per-stage JSON or frame outputs.

Default root is ``<repository>/data``. Override with
``OPEN_ENRICHER_DATA_ROOT`` or point videos elsewhere with
``OPEN_ENRICHER_VIDEO_INPUTS``. When ``OPEN_ENRICHER_EXPERIMENT_ID`` is set,
stage outputs go under ``outputs/<prefix>/<experiment_id>/`` so sequential
experiment runs do not overwrite each other.
"""

from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    """
    Resolve the repository root from this package file.

    ``storage.py`` lives at ``src/open_enricher/storage.py``, so two parents
    up from the package directory lands on the checkout root. Used only when
    building the default data lake path.
    """
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """
    Return the persistent data lake root for this process.

    Preference order:
    1. ``OPEN_ENRICHER_DATA_ROOT`` if set
    2. ``<repo>/data`` next to the package source tree

    Returns:
        Absolute or relative Path to the data lake directory.
    """
    override = os.environ.get("OPEN_ENRICHER_DATA_ROOT")
    if override:
        return Path(override)
    return _repo_root() / "data"


def video_input_dir(root: Path) -> Path:
    """
    Return the directory that holds source video files.

    Args:
        root: Data lake root from ``data_root()``.

    Returns:
        ``OPEN_ENRICHER_VIDEO_INPUTS`` when set, otherwise ``root/video_inputs``.
    """
    override = os.environ.get("OPEN_ENRICHER_VIDEO_INPUTS")
    if override:
        return Path(override)
    return root / "video_inputs"


def experiment_prefix() -> str:
    """
    Return the round prefix used under ``outputs/``.

    Defaults to ``local`` so archived runs line up with ``runs/local/``.
    Override with ``OPEN_ENRICHER_EXPERIMENT_PREFIX``.
    """
    return os.environ.get("OPEN_ENRICHER_EXPERIMENT_PREFIX", "local").strip() or "local"


def experiment_id() -> str:
    """
    Return the active experiment id, or empty for the shared outputs tree.

    When empty, stage folders sit directly under ``outputs/``. When set, they
    sit under ``outputs/<prefix>/<id>/``.
    """
    return os.environ.get("OPEN_ENRICHER_EXPERIMENT_ID", "").strip()


class DatabaseManager:
    """
    Directory registry for videos, caches, and stage output folders.

    Constructed once as the module-level ``db`` singleton. Call
    ``setup_database()`` before running stages and ``configure_environment()``
    before importing heavy ML libraries so Hugging Face and Torch caches land
    inside the data lake.
    """

    def __init__(self) -> None:
        """
        Bind paths from the current environment.

        Does not create directories. Call ``setup_database()`` for that.
        Re-read environment variables by constructing a new instance or by
        calling ``refresh()`` after changing ``OPEN_ENRICHER_*`` vars.
        """
        self.refresh()

    def refresh(self) -> None:
        """
        Re-read environment-backed paths into this instance.

        Call after the experiment loop changes ``OPEN_ENRICHER_EXPERIMENT_ID``
        or ``OPEN_ENRICHER_PIPELINE_SPEC`` so stage folders point at the new
        experiment namespace.
        """
        self.root = data_root()
        self.models_cache = self.root / "models_cache"
        self.video_inputs = video_input_dir(self.root)
        self.outputs = self.root / "outputs"
        self.hf_cache = self.models_cache / "huggingface"
        self.torch_cache = self.models_cache / "torch"
        self._rebuild_stage_folders()

    def _rebuild_stage_folders(self) -> None:
        """
        Point stage folders at the namespaced experiment directory when set.

        Side effect: updates ``self.stage_folders`` and ``self.experiment_log``.
        """
        exp = experiment_id()
        if exp:
            base = self.outputs / experiment_prefix() / exp
        else:
            base = self.outputs
        self.stage_folders = {
            "audio": base / "audio_results",
            "color": base / "color_results",
            "vlm": base / "vlms_results",
            "llm": base / "llms_results",
            "judger": base / "judger_results",
        }
        self.experiment_log = base / "worker.log" if exp else self.outputs / "worker.log"

    def setup_database(self) -> None:
        """
        Create the required folder structure if it does not exist.

        Ensures model caches, video inputs (when using the default location),
        and every stage output folder exist. Raises ``FileNotFoundError`` when
        ``OPEN_ENRICHER_VIDEO_INPUTS`` points at a missing directory.
        """
        self.refresh()
        print(f"[INFO] Initializing data lake at {self.root}...")
        if experiment_id():
            print(
                f"[INFO] Experiment outputs under "
                f"{experiment_prefix()}/{experiment_id()}"
            )

        self.models_cache.mkdir(parents=True, exist_ok=True)
        self.hf_cache.mkdir(exist_ok=True)
        self.torch_cache.mkdir(exist_ok=True)
        self.outputs.mkdir(parents=True, exist_ok=True)
        if self.video_inputs == self.root / "video_inputs":
            self.video_inputs.mkdir(parents=True, exist_ok=True)
        elif not self.video_inputs.is_dir():
            raise FileNotFoundError(
                f"Video input directory does not exist: {self.video_inputs}"
            )

        for folder in self.stage_folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        if self.experiment_log.parent != self.outputs or experiment_id():
            self.experiment_log.parent.mkdir(parents=True, exist_ok=True)

        print("[INFO] Database structure verified.")

    def configure_environment(self) -> None:
        """
        Route Hugging Face and PyTorch caches into the data lake.

        Sets ``HF_HOME``, ``TORCH_HOME``, disables hub symlinks, and enables
        expandable CUDA allocator segments. Call before importing transformers
        or torch when possible.
        """
        os.environ["HF_HOME"] = str(self.hf_cache)
        os.environ["TORCH_HOME"] = str(self.torch_cache)
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def get_output_dir(self, stage_name: str) -> Path:
        """
        Return the output directory for a named stage.

        Args:
            stage_name: One of audio, color, vlm, llm, judger.

        Returns:
            Path to that stage folder under the active experiment namespace.

        Raises:
            ValueError: If ``stage_name`` is not a known stage key.
        """
        if stage_name not in self.stage_folders:
            raise ValueError(f"Unknown stage '{stage_name}'.")
        return self.stage_folders[stage_name]


# Process-wide singleton used by every stage module.
db = DatabaseManager()
