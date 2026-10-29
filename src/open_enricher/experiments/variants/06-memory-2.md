# 06-memory-2

Relaunch of [02-memory-2](02-memory-2.md) under the current VLM settings: 40 frames per window, the scene summary used as a reading key, and synopsis-based scoring.

`interval_seconds` is 300. `vlm_memory_scenes` is 2. `visual_only` is false.

When the VLM describes a window, the prompt contains the dialogue description of that window and the metadata of the two windows immediately before it. Older windows are left out. The first window has no memory. The second window receives only the first.

Qwen still does not keep context between `generate` calls. The two previous records are re-sent in the prompt. If those two records exceed 8,000 characters, they are compacted and an `ALERT` is logged.

Audio stays cached. Dialogue windows, the VLM pass, and the final `VideoProfile` are produced again.
