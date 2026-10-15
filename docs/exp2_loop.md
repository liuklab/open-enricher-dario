# Exp2 unattended experiment loop

This round runs on branch `exp2`. Each experiment forks `exp2-run/<id>` from that tip in a temporary worktree, deploys the same commit to **two independent terminals**, archives results under `runs/exp2/`, and scores each profile with a clean-context Cursor agent. Git cannot store `exp2/<id>` while the branch `exp2` exists.

## Machine mapping (fixed)

| Terminal | Mode | Video |
| :--- | :--- | :--- |
| Colab | `colab-cli` | `Stranger_Things.S01E03.mkv` on Drive |
| Kaggle | `kaggle` | `Suits.S01E02.mkv` in the Kaggle dataset |

## Two agents

- **Orchestrator** — sees `runs/exp2/`, worker logs, timings, metadata JSON, and both scores. Writes a **unified** conclusion, then changes stage code / `pipeline_spec.json` for the next experiment. Writes `EXPERIMENT.md`.
- **Benchmark** — `cursor-sdk` `Agent.prompt` with a temporary directory containing only the profile, the episode summary, and [`benchmark.md`](../src/open_enricher/experiments/prompts/benchmark.md). Persists **only** `score.json`. Never edits the repo.

## Run the loop

```bash
pip install -r requirements-cli.txt
# .env needs HF_TOKEN and CURSOR_API_KEY

python -m src.open_enricher.experiments --list
python -m src.open_enricher.experiments --only 01-interval-llm
python -m src.open_enricher.experiments   # 01..07 by default
```

Resume is automatic via `runs/exp2/progress.jsonl`. Audio/color JSON are cached per machine under `runs/exp2/_cache/` (gitignored).

## Parallel Codex worktree

Use a different Colab session, Kaggle kernel slug, `OPEN_ENRICHER_CODE_SLUG`, and `OPEN_ENRICHER_EXPERIMENT_PREFIX` so the two loops do not overwrite each other.

## Variants 01–07

Specs live in [`src/open_enricher/experiments/variants/`](../src/open_enricher/experiments/variants/). The active worker knobs are copied to [`pipeline_spec.json`](../src/open_enricher/remote/stages/pipeline_spec.json) before deploy. Variants 08–10 are hypotheses; the orchestrator revises them from the unified conclusion after 01–07.
