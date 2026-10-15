"""Manages the data lake used by the remote worker."""

import os
from pathlib import Path


def data_root() -> Path:
    """Persistent output and model-cache directory for this runtime."""
    override = os.environ.get("OPEN_ENRICHER_DATA_ROOT")
    if override:
        return Path(override)
    return Path("/content/drive/MyDrive/Open-Enricher-Database")


def video_input_dir(root: Path) -> Path:
    """Directory that holds the source videos."""
    override = os.environ.get("OPEN_ENRICHER_VIDEO_INPUTS")
    if override:
        return Path(override)
    return root / "Video-Inputs"


def experiment_prefix() -> str:
    """Round prefix isolating parallel agent worktrees (default exp2)."""
    return os.environ.get("OPEN_ENRICHER_EXPERIMENT_PREFIX", "exp2").strip() or "exp2"


def experiment_id() -> str:
    """Current experiment id, or empty when writing to the shared Outputs/ tree."""
    return os.environ.get("OPEN_ENRICHER_EXPERIMENT_ID", "").strip()


class DatabaseManager:
    """Singleton handling directory structures and caching paths."""

    def __init__(self) -> None:
        self.root = data_root()
        self.models_cache = self.root / "Models-Cache"
        self.video_inputs = video_input_dir(self.root)
        self.outputs = self.root / "Outputs"
        self.hf_cache = self.models_cache / "huggingface"
        self.torch_cache = self.models_cache / "torch"
        self._rebuild_stage_folders()

    def _rebuild_stage_folders(self) -> None:
        """Point stage folders at the namespaced experiment directory when set."""
        exp = experiment_id()
        if exp:
            base = self.outputs / experiment_prefix() / exp
        else:
            base = self.outputs
        self.stage_folders = {
            "audio": base / "Audio-Results",
            "color": base / "Color-Results",
            "vlm": base / "VLMs-Results",
            "llm": base / "LLMs-Results",
            "judger": base / "Judger-Results",
        }
        self.experiment_log = base / "worker.log" if exp else self.outputs / "worker.log"

    def setup_database(self) -> None:
        """Creates the required folder structure if it doesn't exist."""
        self._rebuild_stage_folders()
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
        if self.video_inputs == self.root / "Video-Inputs":
            self.video_inputs.mkdir(parents=True, exist_ok=True)
        elif not self.video_inputs.is_dir():
            raise FileNotFoundError(f"Video input directory does not exist: {self.video_inputs}")

        for folder in self.stage_folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        if self.experiment_log.parent != self.outputs or experiment_id():
            self.experiment_log.parent.mkdir(parents=True, exist_ok=True)

        print("[INFO] Database structure verified.")

    def configure_environment(self) -> None:
        """Route Hugging Face and PyTorch caches into the data lake."""
        os.environ["HF_HOME"] = str(self.hf_cache)
        os.environ["TORCH_HOME"] = str(self.torch_cache)
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def get_output_dir(self, stage_name: str) -> Path:
        """Retrieves the output directory for a specific stage."""
        if stage_name not in self.stage_folders:
            raise ValueError(f"Unknown stage '{stage_name}'.")
        return self.stage_folders[stage_name]


# Global singleton instance
db = DatabaseManager()
