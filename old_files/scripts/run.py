"""Thin runner: textual metadata inference on a Colab or Kaggle VM.

Settings come from open_enricher.config on the VM (metadata params from metadata_params.json).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# The Colab Jupyter kernel often does not see a just-finished `pip install -e`
# until restart. Prefer the uploaded checkout on disk.
_REPO_SRC = Path("/content/open-enricher/src")
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

# Colab exec reuses the same kernel. Drop a previous import so this run
# reads the tarball just extracted.
for _name in list(sys.modules):
    if _name == "open_enricher" or _name.startswith("open_enricher."):
        del sys.modules[_name]

from open_enricher import (
    DRIVE_CONFIG,
    METADATA_CONFIG,
    MODEL_CONFIG,
    extract_metadata,
    load_model,
    resolve_video,
    save_json,
    save_text,
    build_output_tarball,
)


def new_run_dir(output_root: str | Path) -> Path:
    # One directory per invocation so later runs do not overwrite earlier artifacts.
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = Path(output_root) / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_metadata(video_path: Path, run_dir: Path) -> None:
    print(f"Video seconds: {METADATA_CONFIG.video_seconds!r}")
    model, processor = load_model(MODEL_CONFIG)
    text = extract_metadata(model, processor, video_path, METADATA_CONFIG)
    out_path = run_dir / "metadata" / f"{video_path.stem}.txt"
    save_text(out_path, text)
    # Sidecar keeps the text file as raw model output.
    save_json(
        out_path.with_suffix(".json"),
        {
            "video": str(video_path),
            "model_id": MODEL_CONFIG.model_id,
            "dtype": MODEL_CONFIG.dtype,
            "device_map": MODEL_CONFIG.device_map,
            "attn_implementation": MODEL_CONFIG.attn_implementation,
            "fps": METADATA_CONFIG.fps,
            "video_seconds": METADATA_CONFIG.video_seconds,
            "min_frames": METADATA_CONFIG.min_frames,
            "max_frames": METADATA_CONFIG.max_frames,
            "max_video_tokens": METADATA_CONFIG.max_video_tokens,
            "video_shortest_edge": METADATA_CONFIG.video_shortest_edge,
            "video_longest_edge": METADATA_CONFIG.video_longest_edge,
            "cap_pixels_per_frame": True,
            "max_new_tokens": METADATA_CONFIG.max_new_tokens,
            "do_sample": METADATA_CONFIG.do_sample,
            "temperature": METADATA_CONFIG.temperature,
            "top_p": METADATA_CONFIG.top_p,
            "top_k": METADATA_CONFIG.top_k,
            "repetition_penalty": METADATA_CONFIG.repetition_penalty,
            "prompt_path": METADATA_CONFIG.prompt_path,
            "prompt": METADATA_CONFIG.prompt,
        },
    )
    print("Saved:", out_path)
    print("Log:", out_path.with_suffix(".json"))


def main() -> None:
    print(f"Configured video: {DRIVE_CONFIG.video!r}")
    video_path = resolve_video(DRIVE_CONFIG.video, DRIVE_CONFIG.root)
    print("Using:", video_path)
    run_dir = new_run_dir(DRIVE_CONFIG.output_dir)
    print("Run directory:", run_dir)

    run_metadata(video_path, run_dir)

    output_tarball = build_output_tarball(
        src=run_dir,
        dest=Path(DRIVE_CONFIG.output_dir).parent / "last_run.tgz",
    )
    print("Output tarball:", output_tarball)


if __name__ == "__main__":
    main()
