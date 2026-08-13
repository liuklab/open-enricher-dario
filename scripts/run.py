"""Thin runner: metadata or embeddings inference on a Colab VM.

MODE is patched by bootstrap_colab.py in a temp copy before colab exec.
Settings come from config.toml on the VM via the installed package.
"""

from __future__ import annotations

import sys
from pathlib import Path

MODE = "metadata"

# The Colab Jupyter kernel often does not see a just-finished `pip install -e`
# until restart. Prefer the uploaded checkout on disk.
_REPO_SRC = Path("/content/open-enricher/src")
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from open_enricher import (
    DriveConfig,
    EmbeddingConfig,
    MetadataConfig,
    ModelConfig,
    extract_embedding,
    extract_metadata,
    list_videos,
    load_model,
    resolve_video,
    save_embedding,
    save_json,
)


def pick_video(drive: DriveConfig) -> Path:
    videos = list_videos(drive.root)
    print(f"Found {len(videos)} video(s) under {drive.root}")
    if not videos:
        raise FileNotFoundError(f"No videos under {drive.root}")
    if drive.video:
        return resolve_video(drive.video, drive.root)
    return videos[0]


def run_metadata(drive: DriveConfig, video_path: Path, model_config: ModelConfig) -> None:
    model, processor = load_model(model_config)
    result = extract_metadata(
        model,
        processor,
        video_path,
        MetadataConfig(),
        model_id=model_config.model_id,
    )
    out_path = Path(drive.output_dir) / "metadata" / f"{video_path.stem}.json"
    save_json(out_path, result)
    print("Saved:", out_path)


def run_embeddings(drive: DriveConfig, video_path: Path, model_config: ModelConfig) -> None:
    embedding_config = EmbeddingConfig()
    model, processor = load_model(model_config)
    vector = extract_embedding(model, processor, video_path, embedding_config)
    sidecar = {
        "video": str(video_path),
        "model_id": model_config.model_id,
        "shape": list(vector.shape),
        "dtype": str(vector.dtype),
        "num_frames": embedding_config.num_frames,
        "fps": embedding_config.fps,
    }
    out_path = Path(drive.output_dir) / "embeddings" / f"{video_path.stem}.npy"
    save_embedding(out_path, vector, sidecar=sidecar)
    print("Saved:", out_path)
    print("Shape:", vector.shape)


def main() -> None:
    if MODE not in {"metadata", "embeddings"}:
        raise ValueError(f"Unknown MODE: {MODE}")

    drive = DriveConfig()
    video_path = pick_video(drive)
    print("Using:", video_path)

    model_config = ModelConfig()
    if MODE == "metadata":
        run_metadata(drive, video_path, model_config)
        return
    run_embeddings(drive, video_path, model_config)


if __name__ == "__main__":
    main()
