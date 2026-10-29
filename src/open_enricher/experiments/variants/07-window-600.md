# 07-window-600

Same pipeline as [06-memory-2](06-memory-2.md). The only change is the window length.

`interval_seconds` is 600. Each dialogue description and each VLM call covers ten minutes instead of five. The last window is shorter when the file is not an exact multiple. `vlm_memory_scenes` stays 2 and `visual_only` stays false, so the prompt still carries this window's dialogue description and the two previous visual records.

Forty frames are still the cap, so they are spread across the longer clip: about one frame every 15 seconds instead of every 7.5. Audio stays cached. Dialogue windows are rebuilt because they follow `interval_seconds`.
