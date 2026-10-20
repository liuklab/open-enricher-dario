# Local experiment loop

This round runs on branch `local`. Each experiment points the process at one variant JSON, runs the enricher stages in process on the local GPU, archives outputs under `runs/local/`, and scores each profile with the deterministic `judger` stage.

## Prerequisites

1. CUDA-capable GPU with enough VRAM for WhisperX, Qwen3-VL-8B (4-bit), and Qwen3-8B (4-bit).
2. Videos under `data/Video-Inputs/` (or `OPEN_ENRICHER_VIDEO_INPUTS`).
3. `.env` with `HF_TOKEN` when gated weights are required.

```bash
pip install -r requirements.txt
pip install -e .
python -m open_enricher --list
```

## Commands

```bash
# All variants in order (01 through 10)
python -m open_enricher

# One variant
python -m open_enricher --only 01-interval-llm

# Start at a variant and continue
python -m open_enricher --from 04-scenes-llm

# Explicit videos
python -m open_enricher --video Stranger_Things.S01E03.mkv Suits.S01E02.mkv
```

## What happens per variant

1. Set `OPEN_ENRICHER_PIPELINE_SPEC` to the variant file and namespace outputs under `Outputs/local/<experiment_id>/`.
2. For each video, run `audio -> color -> vlm -> llm -> judger` via `open_enricher.runner`.
3. Restore audio/color from `runs/local/_cache/<video>/` when present; refresh the cache after a fresh run.
4. Copy artifacts into `runs/local/<experiment_id>/<video>/`, write `score.json` from the judger report, and append `progress.jsonl`.
5. On stage failure, record the error and continue to the next variant.

## Archive layout

```text
runs/local/
├── progress.jsonl
├── loop_summary.json
├── _cache/<video>/{audio,color}/
└── <experiment_id>/
    ├── variant.json
    ├── scores.json
    ├── video_results.json
    ├── EXPERIMENT.md
    └── <video>/
        ├── *_audio.json, *_color.json, *_vlm.json, *_llm.json, *_judger.json
        ├── score.json
        ├── timings.json
        └── worker.log
```

## Variants

Specs live in [`src/open_enricher/experiments/variants/`](../src/open_enricher/experiments/variants/). Knobs are documented in `pipeline_spec.py` (segmentation, dialogue/color routing, VLM memory, dialogue summary, rolling plot, claim check).
