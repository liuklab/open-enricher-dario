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

1. Set `OPEN_ENRICHER_PIPELINE_SPEC` to the variant file. The dataset under
   `dataset/` is shared by every experiment.
2. For each video, run `audio -> color -> vlm -> llm` via `open_enricher.runner`.
3. Reuse audio/color already present in the shared dataset (or restore from
   `runs/local/_cache/<video>/`); refresh the cache after a fresh run.
4. Copy artifacts into `runs/local/<experiment_id>/<video>/` and append
   `progress.jsonl`.
5. On stage failure, record the error and continue to the next variant.

After the loop, score profiles separately:

```bash
pip install -e ".[evaluation]"
python -m open_enricher.evaluation --experiment 01-interval-llm
python -m open_enricher.evaluation --experiment 01-interval-llm --with-synopsis
```

## Archive layout

```text
runs/local/
├── progress.jsonl
├── loop_summary.json
├── _cache/<video>/{audio,color}/
└── <experiment_id>/
    ├── variant.json
    ├── scores.json          # written by the evaluation CLI
    ├── video_results.json
    ├── EXPERIMENT.md
    └── <video>/
        ├── *_audio.json, *_color.json, *_vlm.json, *_llm.json
        ├── score.json       # written by the evaluation CLI
        ├── timings.json
        └── worker.log
```

## Variants

Specs live in [`src/open_enricher/experiments/variants/`](../src/open_enricher/experiments/variants/). Knobs are documented in `pipeline_spec.py` (segmentation, dialogue/color routing, VLM memory, dialogue summary, rolling plot, claim check).
