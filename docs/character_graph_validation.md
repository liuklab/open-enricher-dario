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

Job 123005 completed in 3 minutes 48 seconds. It retained 10 characters, scored 67 windows, and found 13 reliable windows. Two windows linked `c0004` to `SPEAKER_00`. The original writer stretched that edge from 6.006 to 26.025 seconds, including unsupported time and dialogue from another speaker. The next revision keeps each contiguous supported interval separate and saves the face crop bounds alongside the scores. Job 123009 reruns the clip to verify the published intervals and evidence.

## Interval and performance validation

Job 123009 completed successfully. It reproduced 80 face detections, 10 published characters, 67 scores, and 13 reliable windows. The graph attributed `SPEAKER_00` to `c0004` only at 6.006–7.006, 25.025–25.180, and 25.590–26.025 seconds. The two saved face crops were visually consistent. The graph excludes the gap and the intervening `SPEAKER_01` dialogue.

The historical full-film run spent hours in character matching. A local stress check with 650 distinct 512-dimensional tracks took 28.44 seconds with per-exemplar Python cosine scoring and 0.26 seconds with cached unit vectors. On 180 clustered tracks the updated matcher reproduced the reference decisions and node contents. Slurm job 123013 checks real clip output parity with 123009.

The in-progress integration job 122988 produced 13 timestamped dialogue lines from two diarized speakers and three BaSSL scenes with 21 shots on the clip. The BaSSL timeline covered the clip without repairs.

Job 123013 completed in 12 minutes 36 seconds. Its detections, pooled track counts, published nodes, match decisions, scores, and edges exactly matched job 123009. Reviewing the graph also found a false supported edge, `speaker addresses Hi`. Replaying the original diarization after excluding greetings yields an unresolved mention of Ted at that timestamp. The regression test preserves that distinction.

Job 123017 uses the updated checkout for an end-to-end run on the same clip. It is queued behind job 122988 and the successful graph parity job. The full-film job remains gated on the updated clip profile.

The baseline GPU job 122988 reached the VLM and parsed all three scenes as valid JSON from explicit-frame inputs. The three scene inputs used 2, 5, and 26 frames. Frame allocation and image inference reported no fallback. The memory bank held 15 entries and every retrieved frame preceded its query scene.

Cross-title graph probes use 30-second clips and the shared historical full-title audio artifacts. The script trims dialogue spans to the clip and rebases them to zero. The submitted jobs are 123021 for Silicon Valley at 180 seconds, 123022 for Suits at 60 seconds, and 123023 for The Kingdom at 120 seconds.

## Profile quality and remaining probes

Job 122988 completed in 1 hour 47 minutes. Its manifest records a full schema-valid profile and success for all eight stages. The Cursor evaluator failed during bridge discovery. Job 123032 verified bridge startup on a CPU node in 11.98 seconds. The evaluator now owns its bridge explicitly with a 120-second startup budget. Job 123033 successfully scored the archived profile without rerunning inference. Its score was 2.5/5 against the full-episode synopsis, below the usability threshold.

The first 30 seconds contain an opening recap. The prior profile treated that recap as the main episode plot and leaned on generic romance tags. Profile generation now uses bounded excerpts from the original transcript alongside the scene evidence and distinguishes narrated past events from present action. Job 123037 scores the old profile against a reference for the actual clip. Job 123038 regenerates the profile from the same upstream artifacts, then uses the same clip reference.

Suits job 123022 returned four published characters and four supported speech intervals. The Kingdom job 123023 returned three characters and three supported speech intervals. Reviewing Suits found sentence-initial words classified as names, including a supported edge targeting “Wait.” The name parser now keeps weak candidates out of supported metadata. Silicon Valley job 123021 had 159 InsightFace threads inside an allocation of four CPUs. The adapter now caps ONNX Runtime and OpenCV threads to the allocation. Job 123034 repeats the same clip on the same node to check the fix.

The clip-scoped baseline evaluation in job 123037 scored the same profile at 3.15/5 and marked it usable. This score is the comparison baseline for job 123038. The full-episode score measures a different source scope.

Silicon Valley job 123021 timed out after 30 minutes. With the same clip, node, and CPU allocation, job 123034 finished in 2 minutes 56 seconds. It reported four ONNX intra-op threads, 401 face detections, 65 published characters in a crowded scene, 96 scored windows, and two supported speech intervals. Across all four title clips, published speaker edges point to existing nodes, cite reliable windows, and overlap diarization for the same speaker. No two co-visible face samples share the same published character ID.

## Full-run submission

Job 123017 completed in 1 hour 7 minutes with a full profile and all eight stages successful. Its GPU graph published three supported speech spans, each within a matching diarized utterance and backed by a reliable TalkNet window. Its earlier evaluator configuration timed out at bridge startup.

Job 123038 completed in 32 minutes and validated the grounded profile with the updated evaluator. The clip-scoped score rose from 3.15/5 to 3.85/5, with fidelity 4/5 and consistency 5/5. The evaluator marked the profile usable and identified a few unsupported details. The standalone harness now sets the shared model cache before importing the model, avoiding the duplicate download observed in this job.

With both clip checks passed, Slurm accepted job 123065 for all four full-length titles at revision `208ec3a`. The run has its own dataset output directory and the local monitor checks it every ten minutes. Audit the per-title manifests, profile files, graph evidence, and evaluator results after completion.

Slurm estimated a multi-day wait for job 123065. A scheduling dry run with four CPUs and 24 GB of host RAM showed a same-day slot. Job 123117 replaced the original while it was still pending. It keeps one GPU, the 24-hour limit, all four full-length titles, and the same pipeline code. The host-memory request leaves room above the observed clip peak. Full-length resource use still needs verification.

## Character interaction experiment

The first full episode from job 123117 published 278 character nodes, ten supported speaker intervals, and zero edges between character nodes. All 77 mentions and 23 addresses were unresolved. This exposed a gap between speaker attribution and the interaction evidence available to profile generation.

| Probe | Input and control | Evidence to inspect |
| --- | --- | --- |
| Graph smoke, job 123143 | HIMYM at 1195 seconds for 60 seconds, using the first full run's diarization rebased to the clip | Both endpoints exist in the published graph, both detections belong to the same frame, and the frame timestamp falls inside the supported speaker interval |
| GPU integration, job 123146 | The same source span with fresh audio, graph, and visual outputs | Stage manifest, observed relations in scene context, profile fidelity, and unsupported claims about named addressees |
| Profile comparison | Replay the integrated clip artifacts with the new edges present and removed, keeping source, model, and text reference fixed | Compare source-grounded interaction facts and check that neither profile asserts addressee identity or a relationship from co-presence alone |

Job 123143 completed in 9 minutes 32 seconds. It published 34 character nodes, seven supported speaker intervals, and six speech co-presence observations. The six observations represent four ordered character pairs. This probe measures what the camera and audio jointly support. The identity of an intended addressee remains unresolved.

The integrated clip and both graph controls were scored against the same reviewed clip synopsis.

| Inference job | Evaluator job | Graph evidence supplied | Overall |
| --- | --- | --- | --- |
| 123146 | 123235 | Full graph at both stages | 2.80/5 |
| 123147 | 123291 | Eight speech co-presence edges removed from the writer input | 2.30/5 |
| 123179 | 123292 | Graph omitted from the VLM and writer inputs | 2.40/5 |

All three profiles were below the evaluator's usability threshold and framed the clip as an academic rivalry. The shared dialogue descriptions overemphasized the paper and speculated about peer rivalry. Scene replay 123233 and evaluator 123303 test whether revising those descriptions improves the profile while retaining the archived visual extraction and graph.

## Full-episode graph check

In the Silicon Valley integrated replay, job 123242 finished graph construction and advanced to visual memory. The graph stage reported success with 8,596 face observations, 4,184 pooled tracks, and 639 retained identities. None of the retained identities had an alias. The graph stored 119 unresolved mentions, 39 unresolved addresses, and two supported speaker intervals for the same visual identity. It produced no supported character-to-character edges.

The clip result therefore does not establish that speech co-presence survives full-episode processing. Compare the active-speaker window coverage and reliable detections with the supported speaker intervals before changing identity filters or adding interaction evidence.

Replaying alignment with the saved graph decisions and 656 diarized lines found 15 distinct reliable windows that aligned to dialogue. Only one speaker-to-character pair cleared the repeated-window and uniqueness checks. Multiple transcript rows from a single model window do not count as independent support.

Job 123332 runs the same graph stage on seconds 470–570 of the Silicon Valley episode, using the full-run transcript rebased to the clip. The full run scored 15 windows in this span, which contains one supported speaker interval. The probe tests whether a clip-local candidate pool improves the coverage and yields supported co-presence with the same model and attribution policy.
