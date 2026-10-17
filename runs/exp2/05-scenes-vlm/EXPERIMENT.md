# Experiment 05-scenes-vlm

## Code change

- Spec title: `scenes-vlm`
- Segmentation: `scenes`
- Dialogue/color routing: `vlm`
- VLM memory: `False`
- Dialogue summary to VLM: `False`
- Rolling plot: `False`
- Claim check: `False`

## Execution

- `kaggle` / `Suits.S01E02.mkv`: **error** — Command failed (1): /home/darionapolitano/projects/open_enricher/open-enricher/.venv/bin/python deploy.py --mode kaggle --target open-enricher --gpu T4
- `colab` / `Stranger_Things.S01E03.mkv`: **error** — ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))

## Quality and next step

_Orchestrator: replace this section with a unified conclusion from both `runs/exp2/<id>/*/score.json`, metadata JSON, timings, and worker logs. That conclusion is the only input for the next experiment's code changes._

# EXPERIMENT.md template (orchestrator)

Write three short sections after BOTH machines finish:

1. **Code change** — what the pipeline_spec / stage code changed for this experiment
2. **Execution** — unified outcome of Colab (Stranger Things) and Kaggle (Suits): errors, OOM, truncations
3. **Quality and next step** — one conclusion across both episodes; what the next experiment should change

Do not treat the two machines as separate plans. Scores stay in each machine's `score.json`.

