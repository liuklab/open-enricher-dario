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

The clip-scoped evaluators 123423 and 123424 both scored 2.65/5 with fidelity 2/5. BaSSL retained three scene intervals and the fixed-window control used one. Both plans allocated 30 current frames. Both hooks invented a competition and omitted the misheard compliment and the toast, and both profiles added unsupported romance. This clip offers no evidence that BaSSL improves metadata quality over fixed windows.

The BaSSL clip cuts were at 0.083 and 1.710 seconds. Realized VLM inputs were 33 frames across three calls for BaSSL and 30 frames in one call for fixed windows. On full titles, a five-second minimum would flag 25 of 60 Silicon Valley intervals and five of 32 Kingdom intervals. A ten-second minimum would flag 32 and five, respectively. These counts include legitimate short scenes and do not justify merging them automatically. The longest archived scenes were 302.135 and 374 seconds. Keep the inference chunk limit distinct from the scene boundaries.

Use `test_scripts/score_scene_boundaries.py` to compare the archived timeline with independently annotated interior timestamps. Record the tolerance, inspect false splits and misses, and choose duration settings on a calibration title before scoring a held-out title. Keep scene-boundary scores separate from the downstream metadata evaluation.

A preliminary manual audit used contact sheets from seconds 60–300 of The Kingdom. The span shows one continuous interview followed by opening credits. Counting the credits montage as one block, the reference transition is approximately 262.5 seconds. The full-run BaSSL archive contains five cuts in that span. At a two-second tolerance, one matches the reference, four are extra, and none are missed (precision 0.20, recall 1.00, F1 0.33). The extra cuts include a camera change during the interview, a second cut around the transition into credits, and two cuts inside the credits. A minimum duration can target the close pair, but longer segments produced by the other cuts require semantic evidence. This is a single-span audit with one reference boundary and does not estimate episode-wide accuracy.

## MovieNet-SSeg held-out annotations

Downloaded the annotation archive from the project CDN listed by upstream BaSSL revision `0651dd6`. The extracted test split contains 105,976 shot records from 64 films, with 7,272 records labeled as scene boundaries and 838 carrying the sentinel label -1. The NDJSON identifies boundaries by shot index. Benchmark inference needs matching keyframes or a compatible shot-index alignment. The keyframe ZIP URL in the upstream download script returned HTTP 404. The annotations remain in the local audit directory. [Upstream download script](https://github.com/kakaobrain/bassl/blob/0651dd6/scripts/download_movienet_data.sh)
