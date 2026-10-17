# Experiment 09-best-combo

## Code change

- Spec title: `best-combo`
- Segmentation: `interval`
- Dialogue/color routing: `vlm`
- VLM memory: `False`
- Dialogue summary to VLM: `True`
- Rolling plot: `False`
- Claim check: `False`

## Execution

- `colab` / `Stranger_Things.S01E03.mkv`: **ok**

## Quality and next step

_Orchestrator: replace this section with a conclusion from `runs/exp2/<id>/colab/score.json`, metadata JSON, timings, and worker logs. That conclusion is the only input for the next experiment's code changes._

# EXPERIMENT.md template (orchestrator)

Write three short sections after the colab run finishes:

1. **Code change** — what the pipeline_spec / stage code changed for this experiment
2. **Execution** — outcome of Colab (Stranger Things): errors, OOM, truncations
3. **Quality and next step** — one conclusion; what the next experiment should change

Scores stay in each machine's `score.json`.

