Analyze the entire video supplied with this request and extract detailed, evidence-grounded metadata for training and evaluating video recommender systems.

Instructions:
- Review the full video from beginning to end. Cover the opening, middle, and ending; do not infer the whole video from only its first or last frames. If you can inspect only sampled frames, distribute inspection across the full duration and do not claim details that the samples cannot support.
- Describe visible content directly. Separate observable facts from reasonable high-level interpretations. Do not invent names, events, dialogue, settings, objects, or context. Use cautious wording for uncertain interpretations.
- Analyze temporal structure, important scene changes, subject actions, camera movement, continuity, lighting, and visual style where discernible. Include timestamps when available, formatted HH:MM:SS.mmm. Do not imply timestamp precision beyond what the video input supports.
- Make metadata useful for recommendation: capture topics, themes, genre, activities, mood, pacing, visual and narrative characteristics, and concrete similarity signals. Do not infer a viewer's identity, protected traits, or personal circumstances. Do not claim a target demographic; describe content suitability only from what is actually depicted or heard.
- Keep tags concise, specific, and useful for search or content similarity. Avoid redundant tags and generic filler. Use empty arrays for unsupported list fields and an empty string for unsupported string fields. Set confidence to a number from 0 to 1 based on the evidence available.
- Every JSON array other than scene lists holds at most 10 elements, and only elements that are relevant. This covers tags, objects, genres, topics, themes, activities, nested arrays, and every other non-scene list. After the last relevant element, close the array and write the next key. Do not repeat an element. Do not pad a list to reach 10.
- Scene lists ("notable_scenes" and "scene_timeline") have no fixed length. Include as many distinct, relevant scenes as the video supports. For "scene_timeline", space segments about evenly across the full video duration, from the opening through the ending. Do not repeat a scene.
- Return exactly one valid JSON object matching the structure below. Include every key, preserve the value types, and do not add keys. Do not include Markdown fences, comments, or any text outside the JSON. Escape quotation marks and line breaks as required for valid JSON.

Required JSON structure:
{
  "title_guess": "",
  "short_summary": "",
  "tags": [],
  "notable_objects": [],
  "notable_scenes": [],
  "content_type": "",
  "primary_category": "",
  "genres": [],
  "topics": [],
  "themes": [],
  "activities": [],
  "setting": "",
  "people_or_characters": [],
  "narrative_structure": "",
  "opening_scene": "",
  "ending_scene": "",
  "scene_timeline": [
    {
      "start_time": "",
      "end_time": "",
      "description": "",
      "subjects_and_objects": [],
      "actions": [],
      "camera_and_shot": "",
      "lighting_and_color": ""
    }
  ],
  "visual_style": "",
  "mood": [],
  "pacing": "",
  "on_screen_text": [],
  "recommendation_signals": {
    "similarity_keywords": [],
    "likely_viewing_interests": [],
    "distinctive_appeal_factors": [],
    "content_intensity": "",
    "content_suitability_notes": ""
  },
  "content_warnings": [],
  "evidence_notes": []
}

Field guidance:
- "title_guess": A concise, plausible descriptive title; do not present it as an official title unless the video itself establishes that.
- "short_summary": A compact summary of the video's main content and progression.
- "notable_scenes": The most important scenes across the whole video, each as a concise string, and only scenes that are relevant. Include a timestamp when available.
- "scene_timeline": Chronological segments spaced about evenly from the start of the video to the end. Include camera and lighting details only when discernible.
- "recommendation_signals": Describe content-based characteristics that could help match this video with similar content or likely viewing interests. These are content descriptors, not claims about particular viewers.
- "evidence_notes": Briefly cite timestamps for important or uncertain observations where possible. If a claim is interpretive, make that clear.