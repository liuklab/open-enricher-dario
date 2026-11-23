# Visual memory and evidence experiments

## Current pipeline

BaSSL supplies timestamped scene intervals when it succeeds. Dialogue assigns utterances to those intervals and writes a plot summary and scene descriptions. Adaptive frame allocation sets the current-image budget for each VLM call. Qwen also receives up to two earlier text scene records.

EVA/Q-Former encodes sampled frames for visual retrieval. The short-term buffer consolidates 16 observations toward eight and carries one entry into the next buffer. The long-term bank retains up to 64 entries and selects up to four prior frames per scene. Qwen receives decoded images of the selections, with earlier and current timestamps in separate prompt lines. It re-encodes those images. The EVA embeddings stay in the retrieval pipeline.

The VLM gets a compact account of the whole character graph in each scene. The profile writer gets that account, the VLM scene timeline, the dialogue summary and selected original utterances. Its direct visual-memory field currently reports the size of the bank. The graph account can include provisional IDs and does not provide a verified mapping from visible faces to names spoken in dialogue.

## Controlled Slurm replay

All four jobs replay the same full HIMYM source, saved BaSSL scene windows, dialogue, graph, model settings, and per-scene number of images. They archive the exact frame timestamps, raw and parsed VLM results, and final profile in separate job directories. A common random seed reduces sampling variability. The comparison uses the local episode synopsis, relevant source dialogue, and the sampled frames.

| Job | Variant | Question |
| --- | --- | --- |
| 123187 | Current-only frames | How much does the same number of images help when every image comes from the current scene? |
| 123188 | Recent prior frames | Does immediate visual continuity improve the scene account or introduce details from the previous scene? This is a recent-history control. |
| 123189 | Production memory retrieval | Does similarity and recency retrieval from the consolidated bank outperform simple recent frames? |
| 123190 | Grounded evidence template | With the same retrieved frames as 123189, does explicitly marking their role and supplying scene-local graph relations improve fidelity or specificity? The profile writer also receives instructions about using those observations. |

Compare matching scenes by ID and verify frame counts before reading scores. At the VLM, check JSON coverage, current-scene factual accuracy, correct continuity references, and leaks from earlier frames. At the LLM, check distinctive, supported tags and hooks, unsupported identities or relationships, and repeated coverage of the same events. Record measured time and available token or VRAM counters. Inspect the claims manually and use the synopsis-based evaluator as a secondary check.

## Follow-up tests

1. Isolate the short-term consolidation itself. Reuse the same encoded candidates and queries, and compare the production bank with a bank that skips short-term merges but uses the same 64-entry cap and retrieval rule. Compare retained events, their distance from queries, and downstream metadata. The recent-frame replay tests proximity and does not isolate the internal consolidation algorithm.
2. Split the grounded template experiment into VLM-only and writer-only variants if its paired result improves. Prefer timestamped facts scoped to a scene, mark historical images as continuity evidence, and require the writer to prioritize recurring actions, objects, or conflicts. The existing clip controls additionally test removing new graph edges at the writer and removing graph evidence from both VLM and writer.
3. Test source-supported BaSSL intervals against fixed windows with the same title coverage and image budget once the Silicon Valley BaSSL replay passes. Check whether meaningful events remain intact and whether metadata mentions more distinctive, correct details.
4. Repeat the useful variants on a second title, integrate the validated changes, and run the complete pipeline. Check each title manually against its synopsis and preserve the artifacts for comparison.

## Chronology audit

Slot timestamps alone missed five retrieved entries whose representative frame was earlier than the query while their embedding included frames at or after the query. Three are in HIMYM and two are in Silicon Valley. The production retrieval and temporal fallback now require each entry's entire source interval to precede the query. The queued replay reads pre-fix archives, so all four arms exclude these entries before choosing matching frame counts. This historical replay cannot rerank candidates because the archive intentionally omits embedding matrices.

## Replay results

| Inference job | Evaluator job | Parsed scenes | Structured visual descriptions | Overall | Fidelity |
| --- | --- | --- | --- | --- | --- |
| 123187 | 123269 | 15/15 | 2/15 | 3.00/5 | 3/5 |
| 123188 | 123270 | 14/15 | 2/15 | 3.40/5 | 3/5 |
| 123189 | 123271 | 15/15 | 0/15 | 3.00/5 | 3/5 |
| 123190 | 123272 | 15/15 | 14/15 | 3.75/5 | 4/5 |

The retrieved-frame and grounded runs used identical frame manifests. The recent-frame run incorrectly imported arcade footage into the rooftop scene. The grounded run kept that description within the rooftop scene and added a party tag, but its hook still omitted Ted, Robin, and the repeated parties. Writer replay 123276 holds the grounded inputs fixed and applies the specificity prompt. Evaluator job 123277 depends on that replay. Suits memory probes 123194 and 123195 test a fresh visual-memory stage run and the effect of short-term consolidation on identical embeddings.

Job 123194 encoded the Suits episode and built a 64-entry bank, but the probe omitted the dialogue artifact that supplies scene starts. Its successful exit therefore verified bank construction only. The matching dialogue artifact contains 46 scenes and was linked into job 123195 while it was encoding frames. The corrected stage probe, job 123280, includes dialogue from startup and requires at least one non-empty retrieval before it passes.

Job 123195 failed because it could not load scene starts. Job 123281 retries with the dialogue artifact available from startup and validates scene queries before encoding frames.

Job 123280 completed in 12 minutes 59 seconds. The Suits artifact has 172 short-term spans, 64 compacted long-term entries, and 175 retrieved frames across 46 scene queries. Forty-five queries returned at least one frame. Every returned frame maps to a bank entry whose source interval ends strictly before the query. This verifies retrieval chronology on a second title. Job 123281 measures the effect of short-term consolidation on the selected history using identical embeddings.

Job 123281 compared both banks using 2,572 identical candidate embeddings. Both retained 64 entries and retrieved 175 frames, with the same count at every scene query. The selected frame set differed in 26 of 46 scenes. Short-then-long consolidation took 59.20 seconds and yielded a median frame age of 382.8 seconds. Direct-long consolidation took 34.37 seconds and yielded a median age of 432.2 seconds. These differences establish an effect on retrieval.

Grounded Suits replays 123346 (short-then-long) and 123348 (direct-long) hold the other inference inputs fixed. Evaluators 123347 and 123349 use the same synopsis. Compare the archived manifests for equal current frames and total image counts before attributing changes in the generated profiles to consolidation.

Short-then-long inference job 123346 completed in 1 hour 11 minutes. It parsed 45 of 46 VLM responses and returned structured visual descriptions for 44. Evaluator 123347 scored the resulting profile 3.90/5 with fidelity 4/5, level with the previous full Suits baseline. The patent-dispute tag is accurate. The hook omits both the satellite-phone prototype and Louis's pressure on Mike, and the Legal Thriller primary label is outside the current production taxonomy. The direct-long replay is required to judge the effect of consolidation.

The archived Suits manifests confirm 46 matching scene groups, identical current frames, and equal history counts per group. The earlier-frame timestamps differ in 26 groups. Direct-long job 123348 completed in 1 hour 8 minutes, parsed all 46 VLM responses, and supplied structured descriptions for 45. Evaluator 123349 scored the profile 3.40/5 with fidelity 3/5. Both profiles correctly tag the patent dispute. The direct-long profile adds corporate espionage to both the genre and hook, which the synopsis does not support. The short-then-long profile avoids that error and scored 3.90/5 with fidelity 4/5, matching the prior baseline. Keep the production short-to-long consolidation policy for integration. This comparison covers one episode and cannot establish a gain across the dataset.
