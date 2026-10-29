# 08-narrate-visual

Same window length as [06-memory-2](06-memory-2.md): 300 seconds. The VLM and the narration step are split.

The VLM sees only the frames of the current window. It does not receive the dialogue summary and it does not receive earlier scenes. The prompt asks for visible objects, people, actions, and expressions.

Qwen3-8B then rewrites each scene. The prompt carries the dialogue summary of that window, the objects and actions from the VLM, and the dialogue summaries of the two previous windows. Those earlier summaries are continuity only. The narration must not add events that are not in the current summary or in the visual lists. The first window has no earlier summary. The second window receives only the first.

The rewritten prose is stored as `description` on the VLM timeline. The short visual sentence stays on `visual_description`. The profile LLM then reads that timeline as usual.
