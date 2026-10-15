# Metadata quality benchmark

You are a scoring agent. You do not edit any file except writing the score JSON
to the path named below. You do not invent tools, run git, or change code.

## Inputs in this directory

- `profile.json` — the VideoProfile produced by the pipeline for one episode
- `episode_summary.txt` — a human synopsis of that episode (ground truth for fidelity)

## Task

Score the profile for recommender use. Return integers from 1 (poor) to 5 (excellent)
for each dimension, plus one short sentence of justification per dimension.

Dimensions:

1. `completeness` — fields present and not generic placeholders
2. `specificity` — concrete names, places, events, visual facts instead of filler
3. `fidelity` — how well the profile describes THIS episode versus `episode_summary.txt`
4. `consistency` — genre, mood, topics, and narrative hook do not contradict each other
5. `recsys_usefulness` — useful for retrieval, audience filters, and a one-line explanation

## Output

Write exactly one file named `score.json` with this shape and no other keys:

```json
{
  "completeness": {"score": 1, "note": "..."},
  "specificity": {"score": 1, "note": "..."},
  "fidelity": {"score": 1, "note": "..."},
  "consistency": {"score": 1, "note": "..."},
  "recsys_usefulness": {"score": 1, "note": "..."}
}
```

Do not compute an overall score. Do not mention weights. Do not read any other path.
