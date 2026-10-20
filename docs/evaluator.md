# LLM evaluator

After `llm` writes a `VideoProfile`, the `evaluator` stage loads Qwen3-8B with a
fresh chat (system + user only). It receives:

1. The title derived from the video file name (for example
   `Stranger_Things.S01E03.mkv` -> `Stranger Things S01E03`)
2. The profile JSON

It does **not** receive a synopsis, transcript, color frames, or VLM output.
Fidelity is judged against the model's training knowledge of the titled work.

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

## Prompt

Editable templates live in
[`src/open_enricher/stages/evaluator_core/evaluator_prompt.md`](../src/open_enricher/stages/evaluator_core/evaluator_prompt.md)
(`## System` and `## User`). Placeholders such as `{{TITLE}}` and
`{{PROFILE_JSON}}` are filled by `prompt.py`.

## Outputs

- `dataset/outputs/local/<id>/evaluator_results/<video>_evaluator.json`
- Archived copy under `runs/local/<id>/<video>/score.json`
