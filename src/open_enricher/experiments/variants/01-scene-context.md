# 01-scene-context

This control uses 300-second windows and provides the VLM with earlier scene metadata.
WhisperX supplies timestamped speech. Qwen3-8B writes a transcript-based plot and
summarizes the dialogue available for each interval. Qwen3-VL generates the visual
scene timeline. The final text model writes a recommender profile.

The episode-level plot and interval descriptions are saved in `*_dialogue.json`.
The timed scene records are saved in `*_vlm.json` and the profile in `*_llm.json`.
Historical runs of this variant used earlier prompt and code revisions. Use the
archived run manifest, configuration and notes when interpreting their outputs.
