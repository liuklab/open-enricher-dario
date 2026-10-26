# External metadata evaluator

After `llm` writes a `VideoProfile`, the experiment loop scores that profile
before starting the next variant. The provider is Cursor SDK (`cursor-sdk`)
with Grok 4.7. The same scorer can still be run on its own:

```bash
pip install -e ".[evaluation]"
python -m open_enricher.evaluation --video Suits.S01E02.mkv
```

The default provider is Cursor SDK with Grok 4.7 (`CURSOR_API_KEY` in `.env`).
Each video gets a fresh one-shot `Agent.prompt` with `tools=[]` and an empty
temporary `cwd`, so the agent cannot read the repository, synopsis files, or
pipeline outputs unless you put that text in the prompt yourself.

## Inputs

1. Title derived from the video file name (for example
   `Stranger_Things.S01E03.mkv` -> `Stranger Things S01E03`)
2. The profile JSON (`*_llm.json`)
3. Optional: `synopsis/<video>_summary.txt` when `--with-synopsis` is set

No transcript, color frames, or VLM output is passed.

## Synopsis flag

| Flag | Fidelity judged against |
| :--- | :--- |
| (default) | Model training knowledge of the titled work |
| `--with-synopsis` | The synopsis file text |

If `--with-synopsis` is set and the synopsis file is missing, that video fails
(no silent fallback).

## Dimensions (1-5) and weights

| Dimension | Weight | Meaning |
| :--- | ---: | :--- |
| completeness | 0.15 | Fields present and informative |
| specificity | 0.25 | Concrete, not generic fillers |
| fidelity | 0.35 | Matches the real titled film/episode |
| consistency | 0.10 | Fields agree with each other |
| recsys_usefulness | 0.15 | Useful for retrieval / recommendation |

The host computes `overall` as the weighted mean on the 1-5 scale and
`overall_normalized` on 0-1. A profile is `usable` when
`overall >= 3.0` and `fidelity >= 2`.

## Providers

Modular registry under `src/open_enricher/evaluation/providers/`:

| Key | Backend | Notes |
| :--- | :--- | :--- |
| `cursor` | Cursor SDK | Default. Model `grok-4.7` (override with `--model` or `OPEN_ENRICHER_EVAL_MODEL`) |

Future providers (Codex, Gemini, ...) register the same way without changing
weights or report assembly.

## CLI examples

```bash
# Live dataset outputs
python -m open_enricher.evaluation --video Suits.S01E02.mkv

# With synopsis
python -m open_enricher.evaluation --video Suits.S01E02.mkv --with-synopsis

# Score an archived experiment (writes score.json under runs/local/<id>/<video>/)
python -m open_enricher.evaluation --experiment 01-scene-context --with-synopsis
```

## Outputs

- Live: `dataset/outputs/evaluation/<video>_evaluation.json`
- Experiment: `runs/local/<id>/<video>/score.json` and refreshed `scores.json`
- On JSON parse failure: sibling `*_evaluation_raw.txt` with the raw text

## Prompt templates

Editable markdown under
[`src/open_enricher/evaluation/prompts/`](../src/open_enricher/evaluation/prompts/).
Placeholders such as `{{TITLE}}`, `{{SYNOPSIS_BLOCK}}`, and `{{PROFILE_JSON}}`
are filled by `prompt.py`.
