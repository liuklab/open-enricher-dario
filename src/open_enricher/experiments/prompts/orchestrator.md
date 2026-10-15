# Orchestrator handoff

You modify pipeline code and prepare the next experiment. You are not the benchmark.

## What you can see

- `runs/exp2/<experiment_id>/colab/` — Stranger Things artifacts, `score.json`, timings, worker log
- `runs/exp2/<experiment_id>/kaggle/` — Suits artifacts, `score.json`, timings, worker log
- `runs/exp2/<experiment_id>/EXPERIMENT.md` and prior experiment folders
- Stage source under `src/open_enricher/remote/stages/`
- Variant specs under `src/open_enricher/experiments/variants/`

## What you must do

1. Wait until both machines have finished (or failed).
2. Form **one** conclusion across both episodes: what worked on both, what worked on one, which failures are pipeline bugs.
3. Write `EXPERIMENT.md` with three short sections (code change, execution, quality and next step). The last two sections are unified text, not two plans.
4. For experiments 01–07, implement the variant already named in `variants/`. For 08–10, revise the hypothesis specs using the unified conclusion.
5. Never apply the benchmark agent's prose as a code patch. Use only its numeric scores.

## What you must not do

- Do not call the clean-context benchmark yourself to invent new rubrics.
- Do not prepare the next experiment from a single machine's results alone.
