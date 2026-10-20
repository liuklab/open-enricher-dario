# Open-Enricher

Local video understanding pipeline. A GPU host runs five stages in process and writes structured metadata for a recommender:

`audio` (WhisperX) -> `color` (scene colors) -> `vlm` (Qwen3-VL) -> `llm` (Qwen3-8B profile) -> `judger` (deterministic score).

There is no client/server split, no Colab or Kaggle deploy, and no Google Drive. Experiment variants run in sequence on the same machine.

## Install

```bash
python -m venv .venv
source .venv/bin/activate

# Install PyTorch for your CUDA version first, then:
pip install -r requirements.txt
pip install -e .
```

Create `.env` in the repository root:

```env
HF_TOKEN=hf_your_token
```

The token is used only for gated model weights (WhisperX / pyannote). If diarization weights cannot be downloaded, the audio stage still saves the transcript and marks every speaker as `UNKNOWN`.

## Data lake

Default layout under `data/` (override with `OPEN_ENRICHER_DATA_ROOT`):

```text
data/
├── models_cache/
├── video_inputs/          # put source videos here
└── outputs/
    └── local/<experiment_id>/
        ├── audio_results/
        ├── color_results/
        ├── vlms_results/
        ├── llms_results/
        └── judger_results/
```

Put episode files in `data/video_inputs/` before running.

## Run one video through all stages

The experiment loop is the main entrypoint. To process every video once with the default packaged `pipeline_spec.json` knobs via a single variant:

```bash
python -m open_enricher --only 01-interval-llm --video Stranger_Things.S01E03.mkv
```

## Sequential experiments

Variants live in `src/open_enricher/experiments/variants/`. Each run sets `OPEN_ENRICHER_PIPELINE_SPEC` to that JSON, executes `audio -> color -> vlm -> llm -> judger` in process, archives under `runs/local/<experiment_id>/`, and continues to the next variant. Audio and color outputs are cached under `runs/local/_cache/` because they do not depend on the variant knobs.

```bash
python -m open_enricher --list
python -m open_enricher --only 01-interval-llm
python -m open_enricher --from 04-scenes-llm
python -m open_enricher
```

Resume uses `runs/local/progress.jsonl`. Pass `--force` to re-run experiments already marked done. Pass `--no-cache` to ignore the shared audio/color cache.

See [docs/local_loop.md](docs/local_loop.md).

## Stages

| Stage | What it writes |
| :--- | :--- |
| `audio` | Timestamped transcript |
| `color` | Keyframes and dominant hex colors |
| `vlm` | One description per window or scene, merged into one JSON. A local 8B model summarizes the transcript first when the variant enables that option. |
| `llm` | `VideoProfile` for search, audience, and a short summary. |
| `judger` | Deterministic score of that profile. No second model. |

## Layout

```text
src/open_enricher/
  runner.py                 # in-process stage registry and pipeline
  storage.py                # local data lake
  stages/                   # audio, color, vlm, llm, judger
  experiments/
    loop.py                 # sequential variants
    variants/*.json         # experiment knobs
runs/local/                 # archived results and scores
data/                       # videos, model cache, live outputs
```

Historical `runs/exp2/` archives from the remote Colab/Kaggle round may still be present. New runs write only under `runs/local/`.
