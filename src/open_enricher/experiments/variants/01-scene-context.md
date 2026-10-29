# 01-scene-context

One experiment. Stages run in process, in this order:

`audio` → `dialogue` → `vlm` → `llm`

Color is not computed. Audio stays cached under `runs/local/_cache` because it does not depend on `interval_seconds`. The executable spec is `01-scene-context.json`. The window length is 300 seconds. `vlm_memory_scenes` is null, so each VLM call receives every earlier scene.

## 1. Audio

WhisperX `large-v2` transcribes, aligns timestamps, and tries speaker diarization with pyannote. Each line has a start, an end, a speaker, and text.

If the diarization weights cannot be downloaded, the transcript is still saved and every speaker stays `UNKNOWN`. The dialogue stage then writes an `ALERT` to the log.

Output: `<video>_audio.json`.

## 2. Dialogue, global summary

First load of Qwen3-8B in 4-bit. Every line, formatted as `[HH:MM:SS] SPEAKER: text`, goes into one prompt. The model writes a high-level plot summary of about 200–320 words: opening, middle, ending, conflicts, and goals. A name appears only when it is spoken. The summary does not describe the image.

The model context is 40,960 tokens. 520 are reserved for the output. Middle lines are sampled only when the prompt does not fit, and the log then receives an `ALERT`. The model is then unloaded.

## 3. Dialogue, one description per window

Second load of the same Qwen3-8B, after the first unload. The video is split into contiguous windows of `interval_seconds` (300 s). The last window is shorter when the file is not an exact multiple.

For each window the model receives the global summary and only the lines whose start falls in that interval. It writes a detailed account of the speech, about 120–220 words: who is speaking, what they want, what is decided, and how the exchange ends. The global summary is used to resolve names and conflicts, not to import events from other windows.

A window with no lines does not call the model. It saves `No transcribed dialogue in this interval.` If the window's lines do not fit the input (380 tokens reserved for the output), the middle is sampled and an `ALERT` is logged. The model is then unloaded.

Output: `<video>_dialogue.json`, with `plot_summary` and the list of scenes.

## 4. VLM

Qwen3-VL-8B-Instruct in 4-bit, loaded once. For each window it clips the video and samples at most 40 frames.

The prompt contains the dialogue description of that window and the metadata already written for every earlier window. Qwen does not keep context between `generate` calls: the memory is only that JSON, passed again in every prompt. Past 8,000 characters it is compacted to a timestamp plus a short description, and an `ALERT` is logged.

The model uses the dialogue description as a reading key for who is involved and which facts the interval is about, then writes what the frames show of that moment: actions, objects, expressions, and place. It does not retell the spoken summary. When a frame contradicts the text, the frame wins. For each window it saves `description`, `subjects_and_objects`, `actions`, `camera_and_shot`, `lighting_and_color`, and `on_screen_text`, and it also keeps `dialogue_description`.

The accumulation lives in `scene_timeline` of `<video>_vlm.json`. The VLM is then unloaded.

## 5. Final LLM

Third load of Qwen3-8B. The input is `plot_summary` plus `scene_metadata`. The output is a validated `VideoProfile`: genre, sub-genres, pacing, mood, tags, topics, age group, interests, visual style, and narrative hook.

If the schema does not validate, there is one repair attempt. There is no claim-check.

Output: `<video>_llm.json`.
