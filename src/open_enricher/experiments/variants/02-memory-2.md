# 02-memory-2

Same pipeline as [01-scene-context](01-scene-context.md). The only change is how much VLM memory is passed forward.

`vlm_memory_scenes` is 2. When the VLM describes a window, the prompt contains the dialogue description of that window and the metadata of the two windows immediately before it. Older windows are left out. The first window has no memory. The second window receives only the first.

Qwen still does not keep context between `generate` calls. The two previous records are re-sent in the prompt. If those two records exceed 8,000 characters, they are compacted and an `ALERT` is logged.

Audio, the global plot summary, the per-window dialogue descriptions, and the final `VideoProfile` are unchanged. Windows stay 300 seconds.
