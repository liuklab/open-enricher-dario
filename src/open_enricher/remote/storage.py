"""Manages the Data Lake structure on Google Drive (Data Layer)."""

import os
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive")
DATABASE_ROOT = DRIVE_ROOT / "Open-Enricher-Database"

class DatabaseManager:
    """Singleton handling directory structures and caching paths."""
    
    def __init__(self) -> None:
        self.root = DATABASE_ROOT
        self.models_cache = self.root / "Models-Cache"
        self.video_inputs = self.root / "Video-Inputs"
        self.outputs = self.root / "Outputs"

        self.hf_cache = self.models_cache / "huggingface"
        self.torch_cache = self.models_cache / "torch"

        self.stage_folders = {
            "audio": self.outputs / "Audio-Results",
            "color": self.outputs / "Color-Results",
            "vlm": self.outputs / "VLMs-Results",
            "llm": self.outputs / "LLMs-Results",
            "judger": self.outputs / "Judger-Results"
        }

    def setup_database(self) -> None:
        """Creates the required folder structure if it doesn't exist."""
        print("[INFO] Initializing Database on Google Drive...")
        
        self.models_cache.mkdir(parents=True, exist_ok=True)
        self.hf_cache.mkdir(exist_ok=True)
        self.torch_cache.mkdir(exist_ok=True)
        self.video_inputs.mkdir(parents=True, exist_ok=True)
        self.outputs.mkdir(parents=True, exist_ok=True)
        
        for folder in self.stage_folders.values():
            folder.mkdir(exist_ok=True)
            
        print("[INFO] Database structure verified.")

    def configure_environment(self) -> None:
        """Routes HuggingFace and PyTorch caches to Drive to save Colab space."""
        os.environ["HF_HOME"] = str(self.hf_cache)
        os.environ["TORCH_HOME"] = str(self.torch_cache)

    def get_output_dir(self, stage_name: str) -> Path:
        """Retrieves the output directory for a specific stage."""
        if stage_name not in self.stage_folders:
            raise ValueError(f"Unknown stage '{stage_name}'.")
        return self.stage_folders[stage_name]

# Global singleton instance
db = DatabaseManager()
