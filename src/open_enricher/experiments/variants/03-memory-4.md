# 03-memory-4

Same pipeline as [01-scene-context](01-scene-context.md). The only change is how much VLM memory is passed forward.

`vlm_memory_scenes` is 4. When the VLM describes a window, the prompt contains the dialogue description of that window and the metadata of the four windows immediately before it. Older windows are left out. A window with fewer than four predecessors receives all of them.

Qwen still does not keep context between `generate` calls. The four previous records are re-sent in the prompt. If those four records exceed 8,000 characters, they are compacted and an `ALERT` is logged.

Audio, the global plot summary, the per-window dialogue descriptions, and the final `VideoProfile` are unchanged. Windows stay 300 seconds.
