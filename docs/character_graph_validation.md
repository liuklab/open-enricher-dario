# Validate 12-combined on Slurm

The branch starts from `debug` and includes the 16 intermediate commits on `cursor/12-combined-himym-aa84`. The historical HIMYM clip starts at 00:00:20 and lasts 30 seconds. Job 122717 wrote a degraded profile with 69 graph nodes and no `speaks` edges. Job 122809 was cancelled after about 6 hours and 25 minutes. It produced 6,750 nodes and no `speaks` edges for HIMYM, then ran out of GPU memory during profile generation. The run stopped while building the Silicon Valley graph, before The Kingdom started. The intermediate fixes reduce face sampling, pool observations, cap the graph summary, and accept BaSSL clock timestamps. Job 122853 produced 14 actual intervals. The timeline extended the final interval to 1326.08 seconds and recorded `degraded`.

## Experiment sequence

| Experiment | Input | Pass criteria |
| --- | --- | --- |
| InsightFace | A few HIMYM clip frames | Confirm the active CUDA provider, nonempty embeddings, sample count, and timing |
| Audio and TalkNet | HIMYM clip with dialogue | Timestamped diarization, real model scores, and two distinct reliable windows for a manually identified visible speaker |
| Character graph | Same clip with fresh outputs | Separate co-visible faces, consistent identities across appearances, timestamped `speaks` edges pointing to published nodes, and a bounded summary. Compare with manual annotations |
| BaSSL and timeline | Clip, then job 122853 artifacts or a full episode | Real scene boundaries on the full episode and ordered, valid intervals covering the input |
| EVA/Q-Former and visual memory | A clip with several scenes | Real 32×768 tokens per frame, populated memory, and retrieval limited to earlier frames |
| Profile composition | HIMYM clip, then short extracts from the other titles | Schema-valid profiles with traceable evidence and consistent timestamps. Check audio, dialogue, frame allocation, VLM, LLM, and evaluation. Inspect stage results even when the Slurm exit code is zero |
| Full pipeline | All inputs with the complete 12-combined stage sequence | A valid profile per title, manifests matching the source files, no unexpected fallbacks, and recorded time and peak memory. The shared input directory currently contains HIMYM, Silicon Valley, Suits, and The Kingdom |

## Fixes and follow-up checks

- The graph preserves the mapping from original face detection IDs to pooled tracks. Regression tests also verify that a speaker with three qualifying identities stays unresolved.
- TalkNet job 122566 forwarded four windows on CUDA and classified all four as non-speaking. The forward succeeded, but the probe supplied no evidence for a speaker link. Measure how many windows overlap diarized speech and verify the resulting edges on the clip. If coverage is insufficient, allocate a bounded set of windows across dialogue spans and count each window once per speaker and character.
- InsightFace job 122804 reported CUDA and 136 face detections on a midpoint clip of The Kingdom. EVA/Q-Former job 122629 produced four CUDA frames with 32 queries of width 768 each. These historical smokes verify the forwards under their recorded configurations.
- Graph job 122989 failed after InsightFace returned output that could not be decoded as JSON. The adapter runner now routes native and Python diagnostics to stderr and reserves stdout for the response. Job 122993 retries the CPU graph probe.
- Record face detection, pooling, assignment, TalkNet, and profile generation separately. If graph time grows too quickly, measure comparisons against exemplars. If profile generation still exhausts memory, record input tokens and peak GPU memory before changing the budget.
- The Qwen processor applies the visual budget to explicit frames. With historical HIMYM frames it produced 3,672 input tokens for all 30 capped frames and 4,096 tokens for two frames at the previous settings. Falling back to clip generation now marks the stage degraded. GPU generation remains to be validated.
- Keep clip and full-film outputs isolated. Verify that an archive for a shorter source with the same filename cannot skip a full-film run.

## Slurm operation

Submit from the intended checkout with `sbatch --export=ALL test_scripts/run_12_combined_himym_short.sh`. Record the commit, input, job ID, output path, and stage results for each probe. Stop the Slack-backed `oe-slurm-watcher` session before submission. Poll the queue every 10 minutes, then check state and exit code with `sacct` and inspect the artifacts. Stop after three failed attempts per issue. Submit `test_scripts/run_12_combined_full.sh` once the clip validations pass.

## Experiment 123005

CPU job 122993 recovered 80 face detections, 56 pooled tracks, and 10 published characters on the HIMYM clip. It produced no `speaks` edges. Historical job 122566 showed that the four-window TalkNet probe ran successfully but returned no positive scores.

Job 123005 holds the clip, diarization, and face sampling constant. It allows up to 96 TalkNet windows, selects windows that overlap diarized dialogue, and samples across the clip if the candidate count exceeds the budget. The graph saves the per-window score, time, and source track ID. Compare the number of reliable windows, the identity and dialogue overlap for each, and the resulting `speaks` edges. One window that intersects several utterances cannot satisfy the repeated-evidence requirement on its own.
