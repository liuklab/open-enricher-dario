# Open-Enricher

Local video understanding pipeline. A GPU host runs four stages in process and writes structured metadata for a recommender:

`audio` (WhisperX) -> `dialogue` (Qwen3-8B plot, then one description per 5-minute interval) -> `vlm` (Qwen3-VL, with prior scene metadata passed back in the prompt) -> `llm` (Qwen3-8B profile)

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
# OPEN_ENRICHER_EVAL_MODEL=grok-4.7
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
    ├── dialogue_results/
    ├── vlms_results/
    ├── llms_results/
    └── evaluation/        # external evaluator reports
```

Put episode files in `dataset/video_inputs/` before running. Per-experiment
copies of stage outputs are archived under `runs/local/<experiment_id>/`.

## Run one video through all stages

The experiment loop is the main entrypoint. To process every video once with the default packaged `pipeline_spec.json` knobs via a single variant:

```bash
python -m open_enricher --only 01-scene-context --video Stranger_Things.S01E03.mkv
```

When the variant finishes, the loop scores each profile with the Cursor SDK
evaluator. The prompt includes `synopsis/<video>_summary.txt` and the profile
JSON, and writes `runs/local/<experiment_id>/<video>/score.json`. To score
the same archive again:

```bash
python -m open_enricher.evaluation --experiment 01-scene-context --with-synopsis
```

## Sequential experiments

Variants live in `src/open_enricher/experiments/variants/`. Each run sets `OPEN_ENRICHER_PIPELINE_SPEC` to that JSON, executes `audio -> dialogue -> vlm -> llm` in process, and archives under `runs/local/<experiment_id>/`. Audio output is cached under `runs/local/_cache/` because it does not depend on the variant knobs. `01-scene-context` passes every earlier scene to the VLM. `02-memory-2` and `06-memory-2` pass the last 2, and `03-memory-4` passes the last 4. `05-visual-only` keeps every earlier scene but hides dialogue text from the VLM; the final LLM joins speech and frames. `07-window-600` matches `06-memory-2` with 600-second windows. `08-narrate-visual` keeps 300-second windows: the VLM lists objects and actions from the frames alone, then the text model narrates each scene from that list, the current dialogue summary, and the two previous summaries.

```bash
python -m open_enricher --list
python -m open_enricher --only 01-scene-context
python -m open_enricher
```

Resume uses `runs/local/progress.jsonl`. Pass `--force` to re-run an experiment already marked done. Pass `--no-cache` to ignore the shared audio cache.

See [docs/local_loop.md](docs/local_loop.md), [docs/evaluator.md](docs/evaluator.md), and [docs/debug_sync.md](docs/debug_sync.md). The last one is the login-node process that publishes `jobs_logs/` and `runs/` to the `debug` branch. Planned runs are shell scripts in `next_run/`; a fetch of `debug` submits each new or changed script: see [docs/debug_batch.md](docs/debug_batch.md).

## Stages

| Stage | What it writes |
| :--- | :--- |
| `audio` | Timestamped transcript with speaker labels when diarization succeeds |
| `dialogue` | High-level plot from every line, then a detailed dialogue description per 5-minute interval |
| `vlm` | Visual metadata per interval. Each call also receives the dialogue description and the metadata of earlier intervals, limited by `vlm_memory_scenes`. Sampled frames are copied to `runs/local/_cache/<video>/frames/<experiment_id>/` and their timestamps to `*_vlm_frames.json`. |
| `llm` | `VideoProfile` enriched from the plot summary and the accumulated scene metadata |

## External evaluator

| Piece | What it does |
| :--- | :--- |
| Cursor SDK (`cursor-sdk`) | Runs automatically at the end of each experiment. Scores a profile with a fresh external agent. The prompt carries the title and `synopsis/<video>_summary.txt`. Host applies dimension weights. |

## Layout

```text
src/open_enricher/
  runner.py                 # in-process stage registry and pipeline
  storage.py                # local dataset
  stages/                   # audio, dialogue, vlm, llm
  evaluation/               # external scoring (Cursor / future providers)
  experiments/
    loop.py                 # sequential variants
    variants/*.json         # experiment knobs
runs/local/                 # archived results and scores
dataset/                    # videos, model cache, live outputs
synopsis/                   # optional human synopsis files for evaluation
```

Historical `runs/exp2/` archives from the remote Colab/Kaggle round may still be present. New runs write only under `runs/local/`.
