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
5. On stage failure, record the error and continue to the next variant.

After the loop, score profiles separately:

```bash
pip install -e ".[evaluation]"
python -m open_enricher.evaluation --experiment 01-scene-context
python -m open_enricher.evaluation --experiment 01-scene-context --with-synopsis
```

## Archive layout

```text
runs/local/
├── progress.jsonl
├── loop_summary.json
├── _cache/<video>/audio/
└── <experiment_id>/
    ├── variant.json
    ├── scores.json          # written by the evaluation CLI
    ├── video_results.json
    ├── EXPERIMENT.md
    └── <video>/
        ├── *_audio.json, *_dialogue.json, *_vlm.json, *_llm.json
        ├── score.json       # written by the evaluation CLI
        ├── timings.json
        └── worker.log
```

## Variants

Specs live in [`src/open_enricher/experiments/variants/`](../src/open_enricher/experiments/variants/). `interval_seconds` is the fixed scene length (300). `vlm_memory_scenes` is null in `01-scene-context` (every earlier scene), 2 in `02-memory-2`, and 4 in `03-memory-4`. `05-visual-only` sets `visual_only` so the VLM does not read dialogue descriptions.
