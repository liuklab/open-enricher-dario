# Local experiment loop

This round runs on branch `local`. Each experiment points the process at one variant JSON, runs the enricher stages in process on the local GPU, and archives outputs under `runs/local/`. Metadata generation stops at `llm`. Scoring is a separate step: see [evaluator.md](evaluator.md).

## Prerequisites

1. CUDA-capable GPU with enough VRAM for WhisperX, Qwen3-VL-8B (4-bit), and Qwen3-8B (4-bit).
2. Videos under `dataset/video_inputs/` (or `OPEN_ENRICHER_VIDEO_INPUTS`).
3. `.env` with `HF_TOKEN` when gated weights are required.

```bash
pip install -r requirements.txt
pip install -e .
python -m open_enricher --list
```

## Commands

```bash
# The packaged experiment
python -m open_enricher

# The same run by id
python -m open_enricher --only 01-scene-context

# Explicit videos
python -m open_enricher --video Stranger_Things.S01E03.mkv Suits.S01E02.mkv
```

## What happens per variant

1. Set `OPEN_ENRICHER_PIPELINE_SPEC` to the variant file. The dataset under
   `dataset/` is shared by every experiment.
2. For each video, run `audio -> dialogue -> vlm -> llm` via `open_enricher.runner`.
3. Reuse audio already present in the shared dataset (or restore from
   `runs/local/_cache/<video>/`); refresh the cache after a fresh run.
4. Copy artifacts into `runs/local/<experiment_id>/<video>/` and append
   `progress.jsonl`.
5. Score each archived profile with the Cursor SDK evaluator (`cursor-sdk`,
   Grok 4.7, `CURSOR_API_KEY`). The prompt includes
   `synopsis/<video>_summary.txt` together with the profile JSON. Fidelity is
   judged against that text. A missing synopsis fails that video. This writes
   `score.json` and refreshes `scores.json` before the next variant starts.
6. On stage or evaluation failure, record the error and continue to the next
   variant.

`pip install -e ".[evaluation]"` is required so `cursor-sdk` is importable.
To score an archive again:

```bash
python -m open_enricher.evaluation --experiment 01-scene-context --with-synopsis
```

## Archive layout

```text
runs/local/
├── progress.jsonl
├── loop_summary.json
├── _cache/<video>/audio/
├── _cache/<video>/frames/<experiment_id>/scene_XX/frame_YY.jpg
└── <experiment_id>/
    ├── variant.json
    ├── scores.json          # written at the end of the experiment
    ├── video_results.json
    ├── EXPERIMENT.md
    └── <video>/
        ├── *_audio.json, *_dialogue.json, *_vlm.json, *_vlm_frames.json, *_llm.json
        ├── score.json       # written at the end of the experiment
        ├── timings.json
        └── worker.log
```

## VLM frame cache

Each VLM window is a temporary clip. The model samples frames from that file in memory and the clip is deleted. Before that deletion the loop copies the sampled frames to `runs/local/_cache/<video>/frames/<experiment_id>/scene_XX/` and writes `<video>_vlm_frames.json` next to the other VLM artifacts. The JSON uses the same shape as the timing files (`stage`, `wall_seconds`, `spans`). Inference does not read the cache.

The index rule matches the Qwen video processor:

1. Let `N` be the decoded frame count of the window clip and `video_fps` its average frame rate.
2. The requested count is `N / video_fps * sample_fps` with `sample_fps` from `vlm_params.json` (2).
3. Clamp that count between `min_frames` (2 on each window) and `max_frames` (40), then round down to a multiple of 2. Qwen packs frames in pairs. A 300-second window asks for hundreds of frames, so the cap of 40 is what remains.
4. Take that many indices with `linspace` from `0` through `N - 1`, rounded half to even. The first and last frames of the clip are included.
5. The film timestamp is the window start plus `index / video_fps`.

The JPEG is the full-resolution frame at that index in the re-encoded clip. The model receives a resized copy of the same moment. The absolute time can sit a fraction of a second off the original file because ffmpeg seeks to a nearby keyframe before re-encoding the window.

## Variants

Specs live in [`src/open_enricher/experiments/variants/`](../src/open_enricher/experiments/variants/). `interval_seconds` is the fixed scene length (300). `vlm_memory_scenes` is null in `01-scene-context` (every earlier scene), 2 in `02-memory-2`, and 4 in `03-memory-4`. `05-visual-only` sets `visual_only` so the VLM does not read dialogue descriptions.
