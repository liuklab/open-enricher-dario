# Open-Enricher

Local video understanding pipeline. A GPU host runs four stages in process and writes structured metadata for a recommender:

`audio` (WhisperX) -> `color` (scene colors) -> `vlm` (Qwen3-VL) -> `llm` (Qwen3-8B profile)

Scoring is separate from the pipeline. After profiles are written, run the external evaluator (Cursor SDK / Grok by default):

```bash
pip install -e ".[evaluation]"
python -m open_enricher.evaluation --video Suits.S01E02.mkv
```

There is no client/server split, no Colab or Kaggle deploy, and no Google Drive. Experiment variants run in sequence on the same machine.

## Install

```bash
python -m venv .venv
source .venv/bin/activate

# Install PyTorch for your CUDA version first, then:
pip install -r requirements.txt
pip install -e .
# Optional: external evaluator
pip install -e ".[evaluation]"
```

Create `.env` in the repository root:

```env
HF_TOKEN=hf_your_token
CURSOR_API_KEY=cursor_your_key
# OPEN_ENRICHER_LOG_LEVEL=INFO
# OPEN_ENRICHER_EVAL_MODEL=grok-4.7-high-fast
```

The Hugging Face token is used only for gated model weights (WhisperX / pyannote). If diarization weights cannot be downloaded, the audio stage still saves the transcript and marks every speaker as `UNKNOWN`. `CURSOR_API_KEY` is used only by the external evaluator.

Logs use the standard library `logging` package. Every line is CSV with `;` as separator:

```text
YYYY-MM-DD HH:MM:SS.mmm;LEVEL;logger.name;message
```

Console and `dataset/outputs/worker.log` share the same format. Set `OPEN_ENRICHER_LOG_LEVEL` to change the level (default `INFO`).

## Dataset

Default layout under `dataset/` (override with `OPEN_ENRICHER_DATASET_ROOT`):

```text
dataset/                   # shared by every experiment
├── models_cache/
├── video_inputs/          # put source videos here
└── outputs/
    ├── audio_results/
    ├── color_results/
    ├── vlms_results/
    ├── llms_results/
    └── evaluation/        # external evaluator reports
```

Put episode files in `dataset/video_inputs/` before running. Per-experiment
copies of stage outputs are archived under `runs/local/<experiment_id>/`.

## Run one video through all stages

The experiment loop is the main entrypoint. To process every video once with the default packaged `pipeline_spec.json` knobs via a single variant:

```bash
python -m open_enricher --only 01-interval-llm --video Stranger_Things.S01E03.mkv
```

Then score:

```bash
python -m open_enricher.evaluation --experiment 01-interval-llm
python -m open_enricher.evaluation --experiment 01-interval-llm --with-synopsis
```

## Sequential experiments

Variants live in `src/open_enricher/experiments/variants/`. Each run sets `OPEN_ENRICHER_PIPELINE_SPEC` to that JSON, executes `audio -> color -> vlm -> llm` in process, archives under `runs/local/<experiment_id>/`, and continues to the next variant. Audio and color outputs are cached under `runs/local/_cache/` because they do not depend on the variant knobs.

```bash
python -m open_enricher --list
python -m open_enricher --only 01-interval-llm
python -m open_enricher --from 04-scenes-llm
python -m open_enricher
```

Resume uses `runs/local/progress.jsonl`. Pass `--force` to re-run experiments already marked done. Pass `--no-cache` to ignore the shared audio/color cache.

See [docs/local_loop.md](docs/local_loop.md) and [docs/evaluator.md](docs/evaluator.md).

## Stages

| Stage | What it writes |
| :--- | :--- |
| `audio` | Timestamped transcript |
| `color` | Keyframes and dominant hex colors |
| `vlm` | One description per window or scene, merged into one JSON. A local 8B model summarizes the transcript first when the variant enables that option. |
| `llm` | `VideoProfile` for search, audience, and a short summary. |

## External evaluator

| Piece | What it does |
| :--- | :--- |
| `python -m open_enricher.evaluation` | Scores a profile with a fresh external agent (title from the file name; optional synopsis). Host applies dimension weights. |

## Layout

```text
src/open_enricher/
  runner.py                 # in-process stage registry and pipeline
  storage.py                # local dataset
  stages/                   # audio, color, vlm, llm
  evaluation/               # external scoring (Cursor / future providers)
  experiments/
    loop.py                 # sequential variants
    variants/*.json         # experiment knobs
runs/local/                 # archived results and scores
dataset/                    # videos, model cache, live outputs
synopsis/                   # optional human synopsis files for evaluation
```

Historical `runs/exp2/` archives from the remote Colab/Kaggle round may still be present. New runs write only under `runs/local/`.
