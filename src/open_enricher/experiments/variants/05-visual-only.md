# 05-visual-only

Same pipeline as [01-scene-context](01-scene-context.md). The only change is what the VLM is allowed to read.

`visual_only` is true. Each VLM call sees the frames of the current 300-second window and the visual metadata of every earlier window. It does not see the dialogue description of the current window. `dialogue_description` is also removed from the earlier records before they are placed in the prompt, so speech from previous windows does not leak back in.

The dialogue description is still stored on each scene in `scene_timeline`. The final LLM is the first model that reads both: `dialogue_description` for what was said, and the other fields for what the frames showed.

`vlm_memory_scenes` stays null, so the visual memory is every earlier scene, as in `01-scene-context`. Audio, the global plot summary, and the per-window dialogue descriptions are unchanged.
