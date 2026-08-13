"""Thin runner: extract a pooled video embedding on a Colab VM."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_REPO_DIR = "/content/open-enricher"
DEFAULT_DATA_ROOT = "/content/drive/MyDrive/open_enricher/data"
DEFAULT_OUTPUT_DIR = "/content/drive/MyDrive/open_enricher/outputs"
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


def ensure_package(repo_dir: str) -> None:
    """Import open_enricher, using the VM git checkout if it is not installed."""

    try:
        import open_enricher  # noqa: F401
        return
    except ImportError:
        pass
    src = Path(repo_dir) / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import open_enricher  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a pooled embedding from one video.")
    parser.add_argument("--repo-dir", default=os.environ.get("OPEN_ENRICHER_REPO", DEFAULT_REPO_DIR))
    parser.add_argument("--data-root", default=os.environ.get("OPEN_ENRICHER_DATA", DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--output-dir", default=os.environ.get("OPEN_ENRICHER_OUTPUT", DEFAULT_OUTPUT_DIR)
    )
    parser.add_argument(
        "--video",
        default=os.environ.get("OPEN_ENRICHER_VIDEO", ""),
        help="Filename or path under data-root. Empty uses the first video found.",
    )
    parser.add_argument("--model-id", default=os.environ.get("OPEN_ENRICHER_MODEL", DEFAULT_MODEL_ID))
    parser.add_argument(
        "--num-frames",
        type=int,
        default=int(os.environ.get("OPEN_ENRICHER_NUM_FRAMES", "16") or "16"),
    )
    return parser.parse_args()


def main() -> None:
    # Hugging Face reads HF_TOKEN from the environment when it is set via colab exec --env.
    args = parse_args()
    ensure_package(args.repo_dir)

    from open_enricher import (
        DriveConfig,
        EmbeddingConfig,
        ModelConfig,
        extract_embedding,
        list_videos,
        load_model,
        resolve_video,
        save_embedding,
    )

    drive = DriveConfig(root=args.data_root, output_dir=args.output_dir)
    videos = list_videos(drive.root)
    print(f"Found {len(videos)} video(s) under {drive.root}")
    if not videos:
        raise FileNotFoundError(f"No videos under {drive.root}")

    video_path = resolve_video(args.video, drive.root) if args.video else videos[0]
    print("Using:", video_path)

    model_config = ModelConfig(model_id=args.model_id)
    embedding_config = EmbeddingConfig(num_frames=args.num_frames)
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


if __name__ == "__main__":
    main()
