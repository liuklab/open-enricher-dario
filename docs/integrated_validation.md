# Integrated metadata validation

The dataset currently contains How I Met Your Mother S01E02, Silicon Valley S06E01, Suits S01E02, and The Kingdom S01E02. The full run must archive a manifest, profile, and synopsis-based evaluation for each title. Compare the four profiles against job 123117 and review plot claims and distinctive metadata against the local synopses.

## Full-episode pilot

Job 123306 ran The Kingdom on the grounded-planning revision. All eight stages returned an artifact. BaSSL extended its final interval by approximately 0.45 seconds beyond the last shot and marked the result degraded. The profile was archived and evaluated separately. Job 123369 scored it 4.25/5, compared with 4.00/5 in baseline evaluator job 123221. Specificity rose from 3 to 4. Fidelity stayed at 4, and the remaining dimensions stayed level.

The grounded and baseline runs each returned descriptions for all 33 VLM calls. The grounded run returned action lists for 25 calls, compared with 26 in the baseline. Both profiles capture the religious and political fallout of a murder. The grounded profile adds a murder-investigation topic but still omits Emilio, Remigio, the confession, and the political succession from its hook. This is weak evidence of a final-profile improvement.

## Integration checks

- BaSSL frame-index recovery and chronological memory retrieval are present on the integration branch. Compare BaSSL scene placement against fixed windows before assigning a metadata quality gain to segmentation.
- Evaluate the paired Suits consolidation replays before changing the default visual-memory policy.
- Compare the raw and recurring co-presence writer prompts before promoting the 192-window TalkNet setting.
- Review the chronological dialogue and genre controls for factual accuracy before integrating prompt changes. Track each unresolved issue against the three-attempt limit.
- Run the selected configuration first on a short clip and then across the four titles using the existing Slurm launcher. Audit the actual manifests and manually compare profiles with their synopsis after the evaluator finishes.

## Segmentation control

Job 123370 compares BaSSL against fixed windows on the same 100-second Silicon Valley clip beginning at source second 470. Both variant specifications use the same per-call cap, pacing policy, and memory configuration. Scene boundaries can change the number of calls and realized frame counts. Compare the archived plans, scene claims, and profiles against the clip, and report any realized frame or call imbalance when attributing quality differences. The full-episode synopsis can provide context, but only events inside the clip count toward accuracy and coverage.

## Writer candidate

The third genre prompt returned Drama and Legal Drama for Suits (jobs 123360 and 123361, 3.80/5) and Comedy and Sitcom for HIMYM (jobs 123362 and 123363, 3.90/5). Suits improved from 3.40/5 on the second prompt while remaining below its 3.90/5 full-run baseline. HIMYM improved from 3.45/5 on the baseline. The Suits hook still obscures the source of pressure on Mike, and both visual styles need stronger support. This branch adopts the tested prompt and writer taxonomy. The existing segmentation experiment runs on a separate worktree.

## Character lookup controls

Job 123427 compares the combined pipeline with graph construction enabled and disabled on the same 100-second Silicon Valley clip. Both arms use the candidate writer, BaSSL, and visual memory. Score both archived profiles against the same clip reference and verify that the graph-free manifest skipped graph construction.

A candidate lookup would retain face-cluster IDs, timestamped appearance evidence, and supported links to diarized speakers. Names would require explicit transcript evidence and ambiguous names would stay unset. A person mentioned in dialogue may be absent from the frame. The lookup experiment should compare an unnamed table and evidence-backed names with the graph-free control before passing either table to the default writer.

The face-only export superseded the speaker and naming probe. Job 123431 finished in 7 minutes 14 seconds. It returned 146 unique clip-local clusters with appearance timestamps, including 18 provisional clusters. Sixty clusters had only two or three sampled appearances. This is a cluster inventory, and its size on a 100-second excerpt calls for caution when using it as a cast index. The export stays outside model prompts.
